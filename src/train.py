"""
train.py
--------
Training code and the experiment matrix.

Two design decisions here are corrections of mistakes we made in the
first version, and both matter for whether the results mean anything:

1. The model is built INSIDE train_cnn, after set_seed(). Previously we
   called train_cnn(RetinaCNN(), ...), and Python builds that model
   before entering the function - so the seed was reset only after the
   weights had already been initialised. Every experiment therefore
   started from a different random initialisation, results were not
   reproducible, and part of the gap we measured between configurations
   was initialisation noise rather than the change we were studying.
   train_cnn now takes a *function* that builds the model.

2. Checkpoint selection is declared, not accidental. A run with
   early_stopping_patience=None trains the full budget and is evaluated
   on its final weights; a run with a patience value is evaluated on its
   best-validation weights. That difference IS early stopping, so it is
   listed as a factor in the matrix below rather than varying silently
   between configurations.

Every configuration is trained with several seeds and reported as
mean +/- std, because a single run is not evidence.
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.evaluate import evaluate_model
from src.models import RetinaCNN, count_params, match_width_for_params
from src.utils import (
    CHECKPOINTS_DIR,
    METRICS_DIR,
    EarlyStopping,
    get_device,
    load_metrics,
    metrics_exist,
    plot_learning_curves,
    plot_pred_vs_true,
    regression_metrics,
    save_metrics,
    set_seed,
)

SEEDS = [0, 1, 2]

# Huber is quadratic for errors below `delta` and linear above it. PyTorch
# defaults delta=1.0, which is wrong for this task: our errors are 8-15
# YEARS, so with delta=1 the loss is in its linear regime for essentially
# every sample and Huber becomes numerically identical to L1 - we verified
# that, the two configs produced the same metrics to three decimals. Setting
# delta on the scale of a typical error is what makes it the intended
# compromise: fine errors get MSE-like treatment, large ones stay robust.
HUBER_DELTA = 5.0   # years

LOSSES = {
    "mse": lambda: nn.MSELoss(),
    "l1": lambda: nn.L1Loss(),
    "huber": lambda: nn.HuberLoss(delta=HUBER_DELTA),
}


# ---------------------------------------------------------------------------
# The null model - the bar every other model has to clear
# ---------------------------------------------------------------------------
def run_mean_predictor(force_rerun=False):
    """
    Always predicts the mean age of the training set. Its RMSE equals the
    standard deviation of the ages, so it is the definition of "learned
    nothing". A model that does not beat this is worthless no matter how
    reasonable its MAE looks, which is why this number belongs at the top
    of the results table.
    """
    if not force_rerun and metrics_exist("mean_predictor"):
        return load_metrics("mean_predictor")

    from src.dataset import download_dataset, load_split_dataframe

    root = download_dataset()
    train_age = load_split_dataframe("train", root)["patient_age"].values
    test_age = load_split_dataframe("test", root)["patient_age"].values

    pred = np.full_like(test_age, train_age.mean(), dtype=float)
    metrics = regression_metrics(test_age, pred)
    metrics["train_mean_age"] = float(train_age.mean())
    save_metrics("mean_predictor", metrics)
    print(f"[mean_predictor] MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
          f"R2={metrics['R2']:.3f}   (predicts {train_age.mean():.1f} for everyone)")
    return metrics


def check_patient_leakage():
    """
    The single most important correctness check in the project. The
    dataset has ~1.8 images per patient, so if the splits were made per
    image instead of per patient, another photo of the same eye sits in
    training and every number we report is inflated.

    All three overlaps must be zero.
    """
    from src.dataset import download_dataset, load_split_dataframe

    root = download_dataset()
    ids = {s: set(load_split_dataframe(s, root)["patient_id"]) for s in ("train", "val", "test")}
    overlaps = {
        "train n val": len(ids["train"] & ids["val"]),
        "train n test": len(ids["train"] & ids["test"]),
        "val n test": len(ids["val"] & ids["test"]),
    }
    for k, v in overlaps.items():
        print(f"  {k:14s}: {v}")
    print("  patients per split:", {k: len(v) for k, v in ids.items()})

    if any(overlaps.values()):
        raise RuntimeError(
            "PATIENT LEAKAGE: the same patient appears in more than one split. "
            "Every metric in this project would be optimistic. Stop and re-split by patient."
        )
    print("  OK - no patient appears in more than one split.")
    return overlaps


# ---------------------------------------------------------------------------
# Deep models
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in tqdm(loader, desc="  train batches", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            total_loss += criterion(model(x), y).item() * x.size(0)
            n += x.size(0)
    return total_loss / n


def train_cnn(build_model, run_name, seed=0, epochs=20, lr=1e-3, weight_decay=0.0,
              early_stopping_patience=None, batch_size=32, img_size=224,
              augment_train=False, loss="mse", force_rerun=False):
    """
    Trains one configuration with one seed.

    build_model is a callable, not a model - see the note at the top of
    the file. Results are saved per seed as <run_name>_seed<seed>.json.
    """
    tag = f"{run_name}_seed{seed}"
    if not force_rerun and metrics_exist(tag):
        print(f"[{tag}] already done, skipping")
        return load_metrics(tag)

    set_seed(seed)              # first...
    model = build_model()       # ...then build, so the init is reproducible

    device = get_device()
    model = model.to(device)
    if device.type == "cuda":
        # NOTE, and it is a real limitation of our results: benchmark mode
        # lets cuDNN time several convolution algorithms and keep the
        # fastest. The winner depends on what else the GPU is doing, so two
        # runs with the SAME seed can pick different algorithms, sum in a
        # different order, and diverge. We measured this: cnn_aug seed 1
        # trained 20 epochs to MAE 7.62, while cnn_aug_long seed 1 - same
        # config, only a larger epoch ceiling - stopped at epoch 9 with MAE
        # 8.87. Setting benchmark=False and cudnn.deterministic=True would
        # remove it at roughly a 15-20% speed cost. We kept speed because
        # the whole project ran on a free GPU quota, and instead report
        # every configuration as a mean over 3 seeds and refuse to read
        # anything into differences below about half a year of MAE.
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, img_size=img_size, augment_train=augment_train,
        seed=seed,
    )

    criterion = LOSSES[loss]()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    stopper = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience else None

    print(f"[{tag}] {count_params(model):,} params | loss={loss} | wd={weight_decay} | "
          f"select={'best-val' if stopper else 'final-epoch'}")

    history = {"train_loss": [], "val_loss": []}
    start = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, criterion, device)
        va = eval_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        print(f"  epoch {epoch+1}/{epochs}  train={tr:.2f}  val={va:.2f}  ({time.time()-t0:.0f}s)")
        if stopper is not None:
            stopper.step(va, model)
            if stopper.should_stop:
                print(f"  early stop at epoch {epoch+1} (best val {stopper.best_loss:.2f})")
                break

    # Only runs configured with early stopping are evaluated on their best
    # weights; the others are evaluated as they finished. This is a declared
    # factor in the matrix, not an inconsistency.
    if stopper is not None:
        stopper.restore_best(model)

    train_time = time.time() - start
    y_true, y_pred = evaluate_model(model, test_loader, device)
    metrics = regression_metrics(y_true, y_pred)
    metrics.update({
        "seed": seed,
        "params": count_params(model),
        "loss_fn": loss,
        "weight_decay": weight_decay,
        "selection": "best-val" if stopper else "final-epoch",
        "epochs_run": len(history["train_loss"]),
        "train_time_sec": round(train_time, 1),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
    })

    save_metrics(tag, metrics)
    plot_learning_curves(history, f"Learning curves - {tag}",
                          save_name=f"{tag}_curves", ylabel=f"{loss.upper()} loss")
    plot_pred_vs_true(y_true, y_pred, f"Predicted vs true age - {tag}", save_name=f"{tag}_pred_vs_true")
    torch.save(model.state_dict(), CHECKPOINTS_DIR / f"{tag}.pt")
    print(f"[{tag}] MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  R2={metrics['R2']:.3f}")

    del model, optimizer, train_loader, val_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def train_linear_baseline(seed=0, img_size=224, batch_size=32, alpha=1.0, force_rerun=False):
    """Ridge regression on hand-crafted features. Deterministic, so one seed is enough."""
    tag = "ridge_linear"
    if not force_rerun and metrics_exist(tag):
        print(f"[{tag}] already done, skipping")
        return load_metrics(tag)

    set_seed(seed)
    train_loader, _, test_loader = get_dataloaders(
        batch_size=batch_size, img_size=img_size, return_features=True
    )

    def collect(loader, desc):
        xs, ys = [], []
        for x, y in tqdm(loader, desc=desc):
            xs.append(x.numpy())
            ys.append(y.numpy())
        return np.concatenate(xs), np.concatenate(ys)

    X_tr, y_tr = collect(train_loader, f"{tag}: train features")
    X_te, y_te = collect(test_loader, f"{tag}: test features")

    model = Ridge(alpha=alpha).fit(X_tr, y_tr)
    metrics = regression_metrics(y_te, model.predict(X_te))
    metrics["seed"] = seed
    save_metrics(tag, metrics)
    plot_pred_vs_true(y_te, model.predict(X_te), f"Predicted vs true age - {tag}",
                       save_name=f"{tag}_pred_vs_true")
    print(f"[{tag}] MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  R2={metrics['R2']:.3f}")
    return metrics


# ---------------------------------------------------------------------------
# The experiment matrix
#
# Each entry names the anchor it is compared against and the single factor
# that differs from that anchor. cnn_plain is the reference for the
# regularization block; cnn_all is the anchor for everything after it.
# ---------------------------------------------------------------------------
CONTROL_PARAMS = count_params(RetinaCNN())
WIDTH_7X7, PARAMS_7X7, ERR_7X7 = match_width_for_params(
    CONTROL_PARAMS, filter_size=7, num_blocks=3
)

EXPERIMENTS = {
    # name:            (build_model,                                    kwargs,                                              anchor,      factor)
    "cnn_plain":       (lambda: RetinaCNN(dropout_position="none"),     dict(epochs=15),                                     "ridge",     "CNN instead of linear"),
    "cnn_dropout":     (lambda: RetinaCNN(dropout_position="late"),     dict(epochs=15),                                     "cnn_plain", "dropout only"),
    "cnn_wd":          (lambda: RetinaCNN(dropout_position="none"),     dict(epochs=15, weight_decay=1e-4),                  "cnn_plain", "weight decay only"),
    "cnn_es":          (lambda: RetinaCNN(dropout_position="none"),     dict(epochs=15, early_stopping_patience=5),          "cnn_plain", "early stopping only"),
    "cnn_all":         (lambda: RetinaCNN(dropout_position="late"),     dict(epochs=20, weight_decay=1e-4,
                                                                             early_stopping_patience=5),                     "cnn_plain", "all three together"),
    "cnn_aug":         (lambda: RetinaCNN(dropout_position="late"),     dict(epochs=20, weight_decay=1e-4,
                                                                             early_stopping_patience=5, augment_train=True), "cnn_all",   "augmentation"),
    # cnn_aug reached the 20-epoch ceiling in all three seeds without early
    # stopping ever firing, so its validation loss was still improving when
    # the budget ran out and its reported MAE is a lower bound. This run
    # changes ONE thing - the epoch budget - to find out how much of the
    # improvement we were leaving on the table. Same patience, so it stops
    # by itself once validation genuinely stops improving.
    "cnn_aug_long":    (lambda: RetinaCNN(dropout_position="late"),     dict(epochs=40, weight_decay=1e-4,
                                                                             early_stopping_patience=5, augment_train=True), "cnn_aug",   "epoch budget 20 -> 40"),
    "arch_7x7":        (lambda: RetinaCNN(dropout_position="late", filter_size=7, num_blocks=3, width=WIDTH_7X7),
                                                                        dict(epochs=20, weight_decay=1e-4,
                                                                             early_stopping_patience=5),                     "cnn_all",   "7x7 filters, params matched"),
    "arch_valid":      (lambda: RetinaCNN(dropout_position="late", padding="valid"),
                                                                        dict(epochs=20, weight_decay=1e-4,
                                                                             early_stopping_patience=5),                     "cnn_all",   "valid padding, params identical"),
    "arch_drop_early": (lambda: RetinaCNN(dropout_position="early"),    dict(epochs=20, weight_decay=1e-4,
                                                                             early_stopping_patience=5),                     "cnn_all",   "dropout in conv blocks"),
    "loss_l1":         (lambda: RetinaCNN(dropout_position="late"),     dict(epochs=20, weight_decay=1e-4,
                                                                             early_stopping_patience=5, loss="l1"),          "cnn_all",   "L1 loss"),
    "loss_huber":      (lambda: RetinaCNN(dropout_position="late"),     dict(epochs=20, weight_decay=1e-4,
                                                                             early_stopping_patience=5, loss="huber"),       "cnn_all",   "Huber loss"),
}


def run_experiment(name, seeds=SEEDS):
    """Runs one configuration across all seeds."""
    build, kwargs, anchor, factor = EXPERIMENTS[name]
    print(f"\n{'='*70}\n{name}   (vs {anchor}: {factor})\n{'='*70}")
    return [train_cnn(build, name, seed=s, **kwargs) for s in seeds]


def aggregate_results():
    """
    Collects every saved metrics file into one table: mean +/- std across
    seeds for each configuration. This is what goes in the report.
    """
    rows = []
    for f in sorted(METRICS_DIR.glob("*.json")):
        stem = f.stem
        name = stem.rsplit("_seed", 1)[0] if "_seed" in stem else stem
        m = load_metrics(stem)
        rows.append({"config": name, **{k: v for k, v in m.items() if isinstance(v, (int, float))}})
    if not rows:
        print("no results yet")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    g = df.groupby("config")
    out = pd.DataFrame({
        "n_seeds": g.size(),
        "MAE": g["MAE"].mean().round(3),
        "MAE_std": g["MAE"].std().round(3),
        "RMSE": g["RMSE"].mean().round(3),
        "R2": g["R2"].mean().round(4),
        "params": g["params"].first() if "params" in df.columns else np.nan,
    }).sort_values("MAE")
    out.to_csv(METRICS_DIR / "summary.csv")
    return out


def main(seeds=SEEDS):
    """Session 0 checks, then the whole matrix."""
    check_patient_leakage()
    run_mean_predictor()
    train_linear_baseline()
    for name in EXPERIMENTS:
        run_experiment(name, seeds=seeds)
    print("\n", aggregate_results().to_string())


if __name__ == "__main__":
    main()
