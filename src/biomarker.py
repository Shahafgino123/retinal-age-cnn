"""
biomarker.py
------------
Is the retinal age gap related to disease, or only to age?

The proposal frames the gap (predicted age minus chronological age) as a
possible marker of biological ageing. That claim usually stays
speculative in a course project, but this dataset ships clinical labels -
diabetes, diabetic retinopathy, hypertensive retinopathy, macular oedema -
so we can actually look.

Two things make this analysis easy to get wrong, and both are handled here.

1. THE RAW GAP IS NOT USABLE. analysis.py measured a correlation of -0.96
   between the raw gap and true age: the model compresses its predictions
   toward the training mean, so young patients get a positive gap and old
   patients a negative one no matter how healthy they are. Disease
   prevalence also rises with age. Comparing raw gaps between sick and
   healthy patients would therefore mostly compare their ages. We use the
   BIAS-CORRECTED gap from calibration.py (method B), which is built
   specifically to make the gap independent of age.

2. WE DO NOT HAVE THE SAMPLE SIZE FOR SIGNIFICANCE TESTING, so we do not
   do it. Zhu et al. (2023) needed 35,913 participants to find a hazard
   ratio of 1.02 with a confidence interval that still touched 1.00. Our
   test split is 1,462 images. Running p-values on that would produce
   numbers with no power behind them and invite exactly the wrong
   conclusion. This module reports group sizes, means, spreads and an
   effect size, and nothing else. The report should say "we observed an
   association of this size", never "we found that X causes Y".

Run:
    from src.biomarker import run_biomarker_analysis
    run_biomarker_analysis("cnn_all", seed=0)
"""

import numpy as np
import pandas as pd

from src.calibration import fit_bias_removing, predict_split
from src.utils import FIGURES_DIR, METRICS_DIR

COLOR_HEALTHY = "#2b7bba"
COLOR_AFFECTED = "#d1622b"

# Columns that are metadata rather than clinical findings.
# Anything derived from age belongs here: including such a column would
# just re-measure the age effect we spent this module removing.
NON_CLINICAL = {"image_id", "patient_id", "patient_age", "true_age",
                "predicted_age", "age_gap", "abs_error", "corrected_gap",
                "age_group", "age_group_broad", "age_group_numeric",
                "age_group_broad_numeric"}


def find_clinical_columns(df, max_levels=6):
    """
    Finds the label columns without hard-coding their names, since we
    should not assume the dataset's exact schema. A clinical column here
    is any non-metadata column with a small number of distinct values -
    that covers yes/no flags and graded severity alike.
    """
    out = []
    for c in df.columns:
        if c in NON_CLINICAL:
            continue
        vals = df[c].dropna().unique()
        if 2 <= len(vals) <= max_levels:
            out.append(c)
    return out


def corrected_gap(run_name, seed=0):
    """
    Test-set predictions with the age-dependent bias removed.

    The correction is fitted on VALIDATION and applied to test - fitting
    it on test would be using the answers to build the corrector we then
    draw conclusions from.
    """
    val = predict_split(run_name, "val", seed=seed)
    test = predict_split(run_name, "test", seed=seed)

    a, b, shrink = fit_bias_removing(val["true_age"].values, val["predicted_age"].values)
    test = test.copy()
    test["age_gap"] = test["predicted_age"] - test["true_age"]
    test["corrected_gap"] = (a * test["predicted_age"] + b) - test["true_age"]

    raw_corr = np.corrcoef(test["true_age"], test["age_gap"])[0, 1]
    cor_corr = np.corrcoef(test["true_age"], test["corrected_gap"])[0, 1]
    print(f"[{run_name}] compression slope {shrink:.3f}")
    print(f"  gap vs age, raw       : {raw_corr:+.3f}   <- confounded by age")
    print(f"  gap vs age, corrected : {cor_corr:+.3f}   <- what we analyse")
    # Threshold set from a controlled test, not by taste: with a residual
    # correlation of -0.21 and groups 28 years apart in age, an injected
    # +3-year disease effect came out as -1.3 - the sign flipped. Anything
    # above ~0.1 is enough to overwhelm effects of the size we could hope
    # to see, so that is where the warning sits.
    if abs(cor_corr) > 0.10:
        print(f"  WARNING: residual age dependence is {cor_corr:+.3f}. Group "
              f"comparisons below are unreliable wherever the groups differ in age.")
    return test, {"shrink": shrink, "gap_vs_age_raw": raw_corr, "gap_vs_age_corrected": cor_corr}


def cohens_d(a, b):
    """Standardised difference between two group means."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else np.nan


def compare_groups(df, column, min_group=30):
    """
    Corrected gap per level of one clinical column.

    Groups smaller than min_group are reported but flagged: a mean over a
    handful of patients is not something to build an argument on.
    """
    rows = []
    for level, g in df.groupby(column, observed=True):
        vals = g["corrected_gap"].values
        rows.append({
            "variable": column,
            "level": str(level),
            "n": len(vals),
            "mean_gap": round(float(np.mean(vals)), 2),
            "std_gap": round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else np.nan,
            "mean_age": round(float(g["true_age"].mean()), 1),
            "reliable": len(vals) >= min_group,
        })
    out = pd.DataFrame(rows).sort_values("level")

    # Effect size between the two largest levels, plus the thing that
    # actually decides whether that number means anything: how far apart
    # the groups are in AGE. Disease prevalence rises steeply with age, so
    # a sick group is also an old group. Any age dependence still left in
    # the corrected gap then shows up as a fake disease effect - we
    # verified this can flip the sign of a real effect entirely.
    if len(out) >= 2:
        big = out.nlargest(2, "n")["level"].tolist()
        sub = {lv: df[df[column].astype(str) == lv] for lv in big}
        a, b = (sub[lv]["corrected_gap"].values for lv in big)
        age_diff = float(sub[big[0]]["true_age"].mean() - sub[big[1]]["true_age"].mean())
        out["cohens_d_top2"] = round(cohens_d(a, b), 3)
        out["age_diff_between_groups"] = round(age_diff, 1)
        out["age_confounded"] = abs(age_diff) > 5.0
    return out


def plot_gap_by_condition(df, columns, run_name):
    """
    Distribution of the corrected gap per clinical group.

    A boxplot rather than bars: the spread is the whole story here. If the
    boxes overlap almost entirely, the group means are not telling us much
    however different they look in a table.
    """
    import matplotlib.pyplot as plt

    cols = [c for c in columns if df[c].nunique() <= 4][:4]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        levels = sorted(df[c].dropna().unique(), key=str)
        data = [df[df[c] == lv]["corrected_gap"].values for lv in levels]
        bp = ax.boxplot(data, labels=[str(l) for l in levels], patch_artist=True,
                         medianprops=dict(color="#222"), widths=0.55)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLOR_HEALTHY if i == 0 else COLOR_AFFECTED)
            patch.set_alpha(0.75)
        ax.axhline(0, color="#444", linewidth=1, linestyle="--")
        ax.set_title(f"{c}\n(n per group: {[len(d) for d in data]})", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("corrected retinal age gap (years)")
    fig.suptitle(f"Retinal age gap by clinical group - {run_name}\n"
                  f"descriptive only; no significance testing", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{run_name}_age_gap_by_condition.png", dpi=150)
    plt.close(fig)


def run_biomarker_analysis(run_name="cnn_all", seed=0):
    """Corrects the gap, then describes it across the clinical labels."""
    test, diag = corrected_gap(run_name, seed=seed)

    cols = find_clinical_columns(test)
    print(f"\nclinical columns found: {cols}")
    if not cols:
        print("no clinical label columns in the split CSVs - nothing to compare")
        return test, pd.DataFrame()

    tables = [compare_groups(test, c) for c in cols]
    summary = pd.concat(tables, ignore_index=True)

    plot_gap_by_condition(test, cols, run_name)
    summary.to_csv(METRICS_DIR / f"{run_name}_age_gap_by_condition.csv", index=False)
    pd.DataFrame([diag]).to_csv(METRICS_DIR / f"{run_name}_gap_correction_diag.csv", index=False)

    print("\n" + summary.to_string(index=False))
    conf = summary[summary.get("age_confounded", False) == True]["variable"].unique()
    if len(conf):
        print(f"\n!! AGE-CONFOUNDED, do not report as a disease effect: {list(conf)}")
        print("   These groups differ by more than 5 years in mean age. Any gap")
        print("   difference is at least partly an age difference, not a finding.")

    print("\nReminder for the write-up: these are group descriptions, not tests. "
          "Zhu et al. needed ~36,000 participants for a hazard ratio of 1.02; "
          "with 1,462 test images we can describe a pattern, not establish one.")
    return test, summary


if __name__ == "__main__":
    run_biomarker_analysis()
