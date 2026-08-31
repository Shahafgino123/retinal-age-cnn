"""
utils.py
--------
Shared utilities: reproducibility, device selection, early stopping,
regression metrics, and plotting helpers used across all experiments.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

def _results_root():
    """
    Where results go.

    Normally that is <repo>/results. On Kaggle the code may be attached as
    a read-only dataset, so fall back to /kaggle/working - which is the
    writable directory and also what Kaggle saves as the notebook output,
    so results survive the session either way.
    """
    repo = Path(__file__).resolve().parent.parent
    if os.access(repo, os.W_OK):
        return repo / "results"
    kaggle = Path("/kaggle/working")
    if kaggle.is_dir():
        return kaggle / "results"
    return Path.home() / "retina_results"


RESULTS_DIR = _results_root()
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

for d in (METRICS_DIR, FIGURES_DIR, CHECKPOINTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix all relevant random seeds for reproducibility, per the
    project's requirement that results be reproducible from README
    instructions alone."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EarlyStopping:
    """Stops training when validation loss hasn't improved for `patience`
    epochs, as specified in the proposal for the regularized CNN config."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False
        self.best_state_dict = None

    def step(self, val_loss: float, model: torch.nn.Module) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: torch.nn.Module) -> None:
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, R^2 — the metrics specified in the project plan for
    comparing all configurations on a common basis."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}


def save_metrics(name: str, metrics: dict) -> Path:
    path = METRICS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path


def metrics_exist(name: str) -> bool:
    """True if this experiment already has saved metrics from a previous
    run. Lets train.py skip re-running an experiment that already
    finished (e.g. after a Colab disconnect, or when re-running the
    notebook from the top), instead of wasting GPU time redoing it."""
    return (METRICS_DIR / f"{name}.json").exists()


def load_metrics(name: str) -> Optional[dict]:
    """Loads one experiment's saved metrics, or None if it hasn't run yet."""
    path = METRICS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_all_metrics() -> dict:
    """Loads every saved *.json metrics file into one dict, keyed by
    experiment name — used to build the final comparison table."""
    all_metrics = {}
    for f in sorted(METRICS_DIR.glob("*.json")):
        with open(f) as fh:
            all_metrics[f.stem] = json.load(fh)
    return all_metrics


def plot_learning_curves(history: dict, title: str, save_name: Optional[str] = None,
                          ylabel: str = "Loss"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / f"{save_name}.png", dpi=150)
    plt.close(fig)


def plot_pred_vs_true(y_true, y_pred, title: str, save_name: Optional[str] = None):
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.4, s=12)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("True age")
    ax.set_ylabel("Predicted age")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / f"{save_name}.png", dpi=150)
    plt.close(fig)


def plot_comparison_bar(all_metrics: dict, metric: str = "MAE", save_name: Optional[str] = None):
    import matplotlib.pyplot as plt

    names = list(all_metrics.keys())
    values = [all_metrics[n].get(metric, np.nan) for n in names]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    ax.bar(names, values)
    ax.set_ylabel(metric)
    ax.set_title(f"Model comparison — {metric} (lower is better)" if metric != "R2" else f"Model comparison — {metric} (higher is better)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / f"{save_name}.png", dpi=150)
    plt.close(fig)
