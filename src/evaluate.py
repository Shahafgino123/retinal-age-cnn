"""
evaluate.py
-----------
Model evaluation helpers and the final comparison-table builder used for
the report's results section.

Run directly to regenerate the comparison table + bar chart from whatever
metrics JSON files already exist in results/metrics/:
    python -m src.evaluate
"""

from __future__ import annotations

import numpy as np
import torch

from src.utils import load_all_metrics, plot_comparison_bar


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Runs the model over a DataLoader and returns (y_true, y_pred) as
    flat numpy arrays, for use with regression_metrics() or plotting."""
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        y_pred.append(pred)
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def build_comparison_table() -> "pandas.DataFrame":
    """Builds the final MAE/RMSE/R2 comparison table across every
    experiment that has been run so far, for direct use in the report."""
    import pandas as pd

    all_metrics = load_all_metrics()
    if not all_metrics:
        print("No metrics found yet in results/metrics/ — run train.py first.")
        return pd.DataFrame()

    df = pd.DataFrame(all_metrics).T
    df = df[["MAE", "RMSE", "R2"] + [c for c in df.columns if c not in ("MAE", "RMSE", "R2")]]
    df = df.sort_values("MAE")
    return df


def main():
    df = build_comparison_table()
    if df.empty:
        return
    print(df.to_string())
    out_csv = df.reset_index().rename(columns={"index": "experiment"})
    from src.utils import METRICS_DIR

    out_path = METRICS_DIR / "comparison_table.csv"
    out_csv.to_csv(out_path, index=False)
    print(f"\nSaved comparison table to {out_path}")

    plot_comparison_bar(load_all_metrics(), metric="MAE", save_name="comparison_MAE")
    plot_comparison_bar(load_all_metrics(), metric="R2", save_name="comparison_R2")
    print("Saved comparison bar charts to results/figures/")


if __name__ == "__main__":
    main()
