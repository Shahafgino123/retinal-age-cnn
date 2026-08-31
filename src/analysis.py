"""
analysis.py
-----------
Post-hoc error analysis. No retraining happens here - we load the
checkpoints saved by train.py and only run inference on the test set.

This exists because of something we noticed in the predicted-vs-true
plots: the points bend away from the diagonal at both ends. Young
patients look older to the model and old patients look younger, which is
the classic "regression to the mean" behaviour of a regression model
trained on an unbalanced age distribution. A single overall MAE hides
this completely, so here we break the error down by age group and look
at the signed error ("retinal age gap") rather than just its magnitude.

Two different quantities, and the difference matters:
  MAE  = mean |predicted - true|  -> how wrong, ignoring direction
  bias = mean (predicted - true)  -> which direction the model leans
A group can have a large MAE with almost no bias (noisy but centred), or
a small MAE with a clear bias (consistently off in one direction).

Run from the Colab notebook:
    from src.analysis import run_age_analysis
    run_age_analysis("cnn_all")
"""

import numpy as np
import pandas as pd
import torch

from src.dataset import get_dataloaders, load_split_dataframe, download_dataset
from src.evaluate import evaluate_model
from src.train import EXPERIMENTS
from src.utils import CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR, get_device

# The model builders come straight from the experiment matrix in train.py,
# so there is only one definition of each configuration. Defining them
# again here is how the two files would silently drift apart and
# load_state_dict would start failing on tensor shapes.
MODEL_BUILDERS = {name: build for name, (build, _kw, _a, _f) in EXPERIMENTS.items()}

# Colours: one hue for plain magnitude, and a warm/cool pair for the
# signed plots where the sign is the whole point. Blue vs orange stays
# distinguishable for the common forms of colour blindness.
COLOR_MAIN = "#2b7bba"
COLOR_OVER = "#d1622b"   # model predicts older than reality
COLOR_UNDER = "#2b7bba"  # model predicts younger than reality
COLOR_NEUTRAL = "#9aa0a6"


def load_trained_model(run_name, seed=0):
    """
    Rebuilds the architecture and loads its saved weights.

    Checkpoints are saved per seed (<config>_seed<n>.pt), so the analysis
    always states which run it is describing rather than silently picking
    one of several.
    """
    if run_name not in MODEL_BUILDERS:
        raise ValueError(f"unknown run '{run_name}'. known: {list(MODEL_BUILDERS)}")
    ckpt = CHECKPOINTS_DIR / f"{run_name}_seed{seed}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt} - run that experiment first")

    device = get_device()
    model = MODEL_BUILDERS[run_name]().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, device


def predict_test_set(run_name, seed=0, batch_size=32):
    """
    Returns one row per test image: the true age, the predicted age, and
    the signed gap between them.

    The test loader is built with shuffle=False, so its output order
    matches test.csv row for row - that's what lets us glue the
    predictions back onto the metadata columns.
    """
    model, device = load_trained_model(run_name, seed=seed)
    _, _, test_loader = get_dataloaders(batch_size=batch_size)
    y_true, y_pred = evaluate_model(model, test_loader, device)

    root = download_dataset()
    test_df = load_split_dataframe("test", root).reset_index(drop=True)
    if len(test_df) != len(y_true):
        raise RuntimeError(f"test.csv has {len(test_df)} rows but got {len(y_true)} predictions")

    df = test_df.copy()
    df["true_age"] = y_true
    df["predicted_age"] = y_pred
    df["age_gap"] = df["predicted_age"] - df["true_age"]   # + = looks older than it is
    df["abs_error"] = df["age_gap"].abs()
    return df


def add_age_group(df, bin_width=10):
    """
    Bins patients by decade. The dataset ships its own age_group_broad
    column, but we bin ourselves so the groups are even and we control
    the width - the broad column is kept alongside if it exists, in case
    it's useful for the report.
    """
    lo = int(np.floor(df["true_age"].min() / bin_width) * bin_width)
    hi = int(np.ceil(df["true_age"].max() / bin_width) * bin_width)
    edges = list(range(lo, hi + bin_width, bin_width))
    labels = [f"{e}-{e + bin_width - 1}" for e in edges[:-1]]
    df = df.copy()
    df["age_group"] = pd.cut(df["true_age"], bins=edges, labels=labels,
                              right=False, include_lowest=True)
    return df


def error_by_age_group(df, min_count=20):
    """
    MAE and bias per age group. Groups with very few patients are kept in
    the table but flagged, since an average over a handful of images is
    not something we want to draw conclusions from.
    """
    g = df.groupby("age_group", observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "MAE": g["abs_error"].mean(),
        "bias": g["age_gap"].mean(),
        "std": g["age_gap"].std(),
    })
    out["reliable"] = out["n"] >= min_count
    return out.reset_index()


def plot_mae_by_age_group(table, run_name):
    """Plain magnitude - how big is the error in each decade."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(table["age_group"].astype(str), table["MAE"],
                   color=[COLOR_MAIN if r else COLOR_NEUTRAL for r in table["reliable"]])
    for b, v, n in zip(bars, table["MAE"], table["n"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(b.get_x() + b.get_width() / 2, 0.2, f"n={n}", ha="center", va="bottom",
                 fontsize=7, color="white")
    ax.axhline(table["MAE"].mean(), color="#444", linestyle="--", linewidth=1)
    ax.text(len(table) - 0.4, table["MAE"].mean(), " overall", va="center", fontsize=8, color="#444")
    ax.set_xlabel("True age group")
    ax.set_ylabel("MAE (years)")
    ax.set_title(f"Error by age group - {run_name}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{run_name}_mae_by_age_group.png", dpi=150)
    plt.close(fig)


def plot_bias_by_age_group(table, run_name):
    """
    Signed error, so the direction of the mistake is visible. Bars above
    zero mean the model called those patients older than they are; below
    zero, younger. This is the plot that shows regression to the mean.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [(COLOR_NEUTRAL if not r else (COLOR_OVER if v > 0 else COLOR_UNDER))
              for v, r in zip(table["bias"], table["reliable"])]
    bars = ax.bar(table["age_group"].astype(str), table["bias"], color=colors)
    for b, v in zip(bars, table["bias"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.1f}", ha="center",
                 va="bottom" if v > 0 else "top", fontsize=8)
    ax.axhline(0, color="#444", linewidth=1)
    ax.set_xlabel("True age group")
    ax.set_ylabel("Mean predicted - true (years)")
    ax.set_title(f"Prediction bias by age group - {run_name}\n"
                  f"above 0 = predicted too old, below 0 = predicted too young", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{run_name}_bias_by_age_group.png", dpi=150)
    plt.close(fig)


def plot_age_gap_distribution(df, run_name):
    """
    Distribution of the retinal age gap over the whole test set. A model
    with no systematic bias gives a symmetric histogram centred on zero.
    """
    import matplotlib.pyplot as plt

    gap = df["age_gap"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gap, bins=40, color=COLOR_MAIN, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#444", linewidth=1)
    ax.axvline(gap.mean(), color=COLOR_OVER, linestyle="--", linewidth=1.5)
    ax.text(gap.mean(), ax.get_ylim()[1] * 0.95, f" mean {gap.mean():+.2f}",
             color=COLOR_OVER, fontsize=9, va="top")
    ax.set_xlabel("Retinal age gap (predicted - true, years)")
    ax.set_ylabel("Number of test images")
    ax.set_title(f"Retinal age gap distribution - {run_name}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{run_name}_age_gap_distribution.png", dpi=150)
    plt.close(fig)


def run_age_analysis(run_name="cnn_all", seed=0, bin_width=10):
    """
    Full per-age-group analysis for one trained model. Saves a CSV and
    three figures, and prints the table so it can go straight into the
    report.
    """
    print(f"[{run_name}] running inference on the test set (no training)")
    df = predict_test_set(run_name, seed=seed)
    df = add_age_group(df, bin_width=bin_width)
    table = error_by_age_group(df)

    out_csv = METRICS_DIR / f"{run_name}_error_by_age_group.csv"
    table.to_csv(out_csv, index=False)
    df[["image_id", "true_age", "predicted_age", "age_gap"]].to_csv(
        METRICS_DIR / f"{run_name}_test_predictions.csv", index=False)

    plot_mae_by_age_group(table, run_name)
    plot_bias_by_age_group(table, run_name)
    plot_age_gap_distribution(df, run_name)

    overall_mae = df["abs_error"].mean()
    overall_bias = df["age_gap"].mean()
    print(f"\noverall MAE={overall_mae:.2f}  overall bias={overall_bias:+.2f}")
    print(table.to_string(index=False))
    print(f"\nsaved {out_csv.name} and 3 figures to results/")
    return df, table


if __name__ == "__main__":
    run_age_analysis()
