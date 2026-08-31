"""
calibration.py
--------------
Post-hoc bias correction, and the experiment it lets us run.

Background - what analysis.py found: the model's prediction bias falls
monotonically with age (correlation -0.96), from about +8 years in the
youngest reliable group to about -13 in the oldest. The predictions are
*compressed* toward the middle of the training age distribution instead
of spanning the real range.

That leaves an open question with two very different answers:

  (a) representation failure - the network genuinely cannot read age off
      an 85-year-old retina, so the information isn't there; or
  (b) calibration failure - the network ranks patients correctly but
      squashes the scale, because MSE on an age-imbalanced training set
      rewards predicting near the mean.

If (b) holds, undoing a single global linear squash should straighten
the predictions out. If (a) holds, no rescaling can help, because the
ordering itself is wrong.

Everything is fitted on the VALIDATION split and evaluated on TEST.
Fitting on test would be leakage - using the answers to build the
corrector we then grade on.

--------------------------------------------------------------------
Two corrections, because "fix the bias" and "reduce the error" turn out
to be different goals (we checked this on simulated data first):

  A. MSE-optimal   - ordinary least squares of true ~ a*pred + b.
     This is the best linear predictor of age given the model output, so
     it gives the lowest MAE. But because `pred` is itself noisy, the
     fitted slope is attenuated: it does NOT fully undo the compression,
     and a bias-vs-age trend survives.

  B. Bias-removing - fit the shrinkage in the forward direction
     (pred ~ s*true + i) and invert it: a = 1/s, b = -i/s.
     This stretches predictions back onto the identity line, so the
     bias becomes independent of age - at the cost of a slightly higher
     MAE, since stretching amplifies noise as well as signal.

Neither is "the" right answer; which one you want depends on the use.
For reporting accuracy, A. For a "retinal age gap" biomarker - where a
gap that still correlates with age is worthless - B, and only B.
--------------------------------------------------------------------

Run from the Colab notebook:
    from src.calibration import run_calibration_analysis
    run_calibration_analysis("cnn_all")
"""

import numpy as np
import pandas as pd

from src.analysis import add_age_group, error_by_age_group, load_trained_model
from src.dataset import get_dataloaders, load_split_dataframe, download_dataset
from src.evaluate import evaluate_model
from src.utils import FIGURES_DIR, METRICS_DIR, regression_metrics

COLOR_RAW = "#2b7bba"   # blue
COLOR_MSE = "#e8a33d"   # amber
COLOR_BIAS = "#d1622b"  # orange

VARIANTS = [
    ("predicted_age", "raw", COLOR_RAW),
    ("cal_mse", "A: MSE-optimal", COLOR_MSE),
    ("cal_bias", "B: bias-removing", COLOR_BIAS),
]


def predict_split(run_name, split, seed=0, batch_size=32):
    """
    Predictions for one split ('val' or 'test'), returned next to that
    split's metadata rows. Both loaders use shuffle=False, so the model
    output lines up with the CSV row order.
    """
    model, device = load_trained_model(run_name, seed=seed)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]

    y_true, y_pred = evaluate_model(model, loader, device)

    root = download_dataset()
    df = load_split_dataframe(split, root).reset_index(drop=True)
    if len(df) != len(y_true):
        raise RuntimeError(f"{split}.csv has {len(df)} rows but got {len(y_true)} predictions")

    df = df.copy()
    df["true_age"] = y_true
    df["predicted_age"] = y_pred
    return df


def fit_mse_optimal(val_true, val_pred):
    """Method A - OLS of true on pred. Lowest error, partial bias fix."""
    a, b = np.polyfit(val_pred, val_true, 1)
    return float(a), float(b)


def fit_bias_removing(val_true, val_pred):
    """
    Method B - measure the shrinkage (pred ~ s*true + i) and invert it.
    Returns the inverted coefficients plus s itself, which is the number
    that quantifies the compression.
    """
    s, i = np.polyfit(val_true, val_pred, 1)
    # A model that barely varies with age gives s near 0, and inverting
    # that produces an enormous slope and meaningless output. Better to
    # stop here than to report nonsense - if this fires, the model has
    # not learned age at all and calibration is the wrong tool.
    if s < 0.2:
        raise ValueError(
            f"shrinkage slope is {s:.3f}: predictions barely track age, so "
            f"inverting the compression is not meaningful. Check the model first."
        )
    return float(1.0 / s), float(-i / s), float(s)


def fit_isotonic(val_true, val_pred):
    """
    Non-parametric check: fits any monotonic mapping instead of a line.
    If this beats both linear fits by a lot, the compression is not a
    uniform squash and something more structured is going on.
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_pred, val_true)
    return iso.predict


def summarize(df, pred_col, label):
    """
    Overall metrics for one set of predictions.

    bias_vs_age is the important one here: the correlation between the
    signed error and the true age. It is the single number that says
    whether the age-dependent bias is still present (near -1) or has
    been removed (near 0).
    """
    err = df[pred_col] - df["true_age"]
    m = regression_metrics(df["true_age"], df[pred_col])
    m["bias"] = float(err.mean())
    m["bias_vs_age"] = float(np.corrcoef(df["true_age"], err)[0, 1])
    m["variant"] = label
    return m


def per_group_comparison(df, bin_width=10):
    """MAE and bias per age group for every variant."""
    out = []
    for col, label, _ in VARIANTS:
        if col not in df.columns:
            continue
        d = df.copy()
        d["age_gap"] = d[col] - d["true_age"]
        d["abs_error"] = d["age_gap"].abs()
        t = error_by_age_group(add_age_group(d, bin_width=bin_width))
        t["variant"] = label
        out.append(t)
    return pd.concat(out, ignore_index=True)


def _grouped_bars(table, value, ylabel, title, fname, zero_line=False):
    """Shared plotting for the before/after comparisons."""
    import matplotlib.pyplot as plt

    labels = [lab for _, lab, _ in VARIANTS]
    colors = [c for _, _, c in VARIANTS]
    sub = [table[table.variant == lab].reset_index(drop=True) for lab in labels]
    groups = sub[0]["age_group"].astype(str)
    x = np.arange(len(groups))
    w = 0.26

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, (s, lab, c) in enumerate(zip(sub, labels, colors)):
        ax.bar(x + (k - 1) * w, s[value], w, label=lab, color=c)
    if zero_line:
        ax.axhline(0, color="#444", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlabel("True age group")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


def run_calibration_analysis(run_name="cnn_all", seed=0, bin_width=10):
    """Fits both corrections on validation, applies them to test, reports what changed."""
    print(f"[{run_name}] predicting on validation (to fit the corrections)")
    val = predict_split(run_name, "val", seed=seed)
    print(f"[{run_name}] predicting on test (to evaluate them)")
    test = predict_split(run_name, "test", seed=seed)

    vt, vp = val["true_age"].values, val["predicted_age"].values
    a_mse, b_mse = fit_mse_optimal(vt, vp)
    a_bias, b_bias, shrink = fit_bias_removing(vt, vp)
    apply_iso = fit_isotonic(vt, vp)

    test["cal_mse"] = a_mse * test["predicted_age"] + b_mse
    test["cal_bias"] = a_bias * test["predicted_age"] + b_bias
    test["cal_isotonic"] = apply_iso(test["predicted_age"].values)

    print(f"\nmeasured compression on validation: pred = {shrink:.3f} * true + ...")
    print(f"  the model spans only {shrink*100:.0f}% of the real age range")
    print(f"A (MSE-optimal)   : true = {a_mse:.3f} * pred + {b_mse:.2f}")
    print(f"B (bias-removing) : true = {a_bias:.3f} * pred + {b_bias:.2f}")

    rows = [summarize(test, c, l) for c, l, _ in VARIANTS]
    rows.append(summarize(test, "cal_isotonic", "isotonic (check)"))
    overall = pd.DataFrame(rows)[["variant", "MAE", "RMSE", "R2", "bias", "bias_vs_age"]]
    print("\n=== overall on the test split ===")
    print(overall.to_string(index=False))

    groups = per_group_comparison(test, bin_width=bin_width)
    print("\n=== MAE per age group ===")
    print(groups.pivot(index="age_group", columns="variant", values="MAE").to_string())
    print("\n=== bias per age group ===")
    print(groups.pivot(index="age_group", columns="variant", values="bias").to_string())

    overall.to_csv(METRICS_DIR / f"{run_name}_calibration_overall.csv", index=False)
    groups.to_csv(METRICS_DIR / f"{run_name}_calibration_by_age_group.csv", index=False)
    _grouped_bars(groups, "bias", "Mean predicted - true (years)",
                   f"Prediction bias by age group - {run_name}",
                   f"{run_name}_bias_calibration.png", zero_line=True)
    _grouped_bars(groups, "MAE", "MAE (years)",
                   f"Error by age group - {run_name}",
                   f"{run_name}_mae_calibration.png")
    print("\nsaved 2 CSVs and 2 figures to results/")
    return test, overall, groups


if __name__ == "__main__":
    run_calibration_analysis()
