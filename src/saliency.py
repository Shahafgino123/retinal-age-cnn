"""
saliency.py
-----------
Which parts of the retina drive the age prediction?

Method: gradient-based visualisation of activated features, as taught in
lecture 10 (slides 53-57). We take the gradient of the model's output with
respect to the INPUT PIXELS, then reduce it to a grayscale map by taking
the maximum absolute value across the three RGB channels - exactly the
recipe in the slides. Bright pixels are the ones the prediction is most
sensitive to.

We deliberately use this rather than Grad-CAM. Grad-CAM answers the same
question and is the more modern refinement, but it was never covered in
the course, whereas this method was taught directly. Same scientific
question, no unexplained machinery.

Why it matters here: Poplin et al. (2018) had three ophthalmologists
score 100 attention maps and found blood vessels highlighted in 95% of
them for age prediction (optic disc 33%, non-specific 38%). That was a
large pretrained model on a healthy cohort. Our question is whether a
small CNN trained from scratch on a clinical population shows the same
pattern - and whether the receptive-field ablation changes WHERE the
model looks, which is the mechanism our 7x7-vs-3x3 experiment assumes.

We cannot repeat the ophthalmologist scoring, so the quantitative parts
here are deliberately modest and clearly labelled as proxies:
  - a radial profile (centre vs periphery)
  - correlation with local image contrast, a crude stand-in for vessels
  - agreement between the mean maps of two configurations

Run:
    from src.saliency import run_saliency_analysis
    run_saliency_analysis(["cnn_all", "arch_7x7"], seed=0)
"""

import numpy as np
import torch

from src.analysis import load_trained_model
from src.dataset import get_dataloaders
from src.utils import FIGURES_DIR, METRICS_DIR

# Sequential single-hue ramp for magnitude. Avoiding 'jet' on purpose:
# a rainbow map invents visual boundaries where the data is smooth.
SALIENCY_CMAP = "inferno"


def saliency_batch(model, x, device):
    """
    Gradient of the predicted age with respect to each input pixel.

    Summing the batch before calling backward() gives every sample its own
    gradient in a single pass: each output depends only on its own input,
    so the sum's gradient w.r.t. image i is exactly image i's gradient.

    Returns (B, H, W) after the max-over-channels reduction from the slides.
    """
    model.eval()
    x = x.to(device).clone().requires_grad_(True)
    pred = model(x)
    model.zero_grad(set_to_none=True)
    pred.sum().backward()
    sal = x.grad.detach().abs().amax(dim=1)   # max over R, G, B
    return sal.cpu().numpy(), pred.detach().cpu().numpy()


def collect_saliency(run_name, seed=0, n_images=256, batch_size=32):
    """
    Averages saliency over the first n_images of the test set.

    A single map is dominated by that one photo's anatomy; averaging over
    a few hundred shows where the model looks *in general*. We also keep
    a handful of individual examples for the report figure.
    """
    model, device = load_trained_model(run_name, seed=seed)
    _, _, test_loader = get_dataloaders(batch_size=batch_size)

    total = None
    n_seen = 0
    examples = []
    for x, y in test_loader:
        sal, pred = saliency_batch(model, x, device)
        total = sal.sum(axis=0) if total is None else total + sal.sum(axis=0)
        if len(examples) < 4:
            examples.append((x[0].cpu().numpy(), sal[0], float(y[0]), float(pred[0])))
        n_seen += len(sal)
        if n_seen >= n_images:
            break

    return total / n_seen, examples, n_seen


def radial_profile(mean_map, n_bins=20):
    """
    Mean saliency as a function of distance from the image centre.

    Fundus photographs are roughly centred on the macula with the optic
    disc off to one side, so a centre-heavy profile and a flat one mean
    genuinely different things about what the model uses.
    """
    h, w = mean_map.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    r = r / r.max()
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    vals = mean_map.ravel()
    prof = np.array([vals[idx == b].mean() if (idx == b).any() else np.nan
                     for b in range(n_bins)])
    centres = (bins[:-1] + bins[1:]) / 2
    return centres, prof


def contrast_proxy(img_chw):
    """
    Local contrast of the image, as a rough stand-in for vascular
    structure - vessels are the high-contrast edges in a fundus photo.

    This is a PROXY, not a vessel segmentation. A high correlation
    between saliency and this map is suggestive, not proof that the model
    looks at vessels, and the report should say so.
    """
    gray = img_chw.mean(axis=0)
    gy, gx = np.gradient(gray)
    return np.sqrt(gx ** 2 + gy ** 2)


def plot_examples(examples, run_name):
    """Original, saliency, and overlay for a few test images."""
    import matplotlib.pyplot as plt

    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(8, 2.7 * n))
    axes = np.atleast_2d(axes)
    for i, (img, sal, true_age, pred_age) in enumerate(examples):
        # undo the ImageNet normalisation just for display
        show = img.transpose(1, 2, 0)
        show = show * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        show = np.clip(show, 0, 1)
        s = sal / (sal.max() + 1e-12)

        axes[i, 0].imshow(show)
        axes[i, 0].set_ylabel(f"true {true_age:.0f}\npred {pred_age:.0f}", fontsize=9)
        axes[i, 1].imshow(s, cmap=SALIENCY_CMAP)
        axes[i, 2].imshow(show)
        axes[i, 2].imshow(s, cmap=SALIENCY_CMAP, alpha=0.55)
        for j, t in enumerate(["retina", "saliency", "overlay"]):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(t, fontsize=10)
    fig.suptitle(f"Gradient saliency - {run_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{run_name}_saliency_examples.png", dpi=150)
    plt.close(fig)


def plot_mean_maps(maps, run_names):
    """Mean saliency map per configuration, on a shared colour scale."""
    import matplotlib.pyplot as plt

    vmax = max(m.max() for m in maps)
    fig, axes = plt.subplots(1, len(maps), figsize=(4 * len(maps), 4.2))
    axes = np.atleast_1d(axes)
    for ax, m, name in zip(axes, maps, run_names):
        im = ax.imshow(m, cmap=SALIENCY_CMAP, vmin=0, vmax=vmax)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean |gradient|")
    fig.suptitle("Where the model looks, averaged over the test set", fontsize=11)
    fig.savefig(FIGURES_DIR / "saliency_mean_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radial(profiles, run_names):
    """Centre-to-edge profile, one line per configuration."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (c, p), name in zip(profiles, run_names):
        ax.plot(c, p / p.max(), label=name, linewidth=2)
    ax.set_xlabel("distance from image centre (0 = centre, 1 = corner)")
    ax.set_ylabel("mean saliency (normalised)")
    ax.set_title("Radial saliency profile")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "saliency_radial_profile.png", dpi=150)
    plt.close(fig)


def run_saliency_analysis(run_names=("cnn_all", "arch_7x7"), seed=0, n_images=256):
    """
    Compares where several trained configurations look.

    The comparison is the point: our receptive-field ablation assumes 7x7
    filters pick up coarse structures and 3x3 filters fine texture. If the
    two mean maps are nearly identical, that assumption does not hold and
    the ablation is measuring something else.
    """
    import pandas as pd

    maps, profiles, rows = [], [], []
    for name in run_names:
        print(f"[{name}] computing saliency over {n_images} test images")
        mean_map, examples, n_seen = collect_saliency(name, seed=seed, n_images=n_images)
        plot_examples(examples, name)
        maps.append(mean_map)
        profiles.append(radial_profile(mean_map))

        # how concentrated is the map, and does it track local contrast?
        flat = mean_map.ravel()
        share_top10 = np.sort(flat)[::-1][: max(1, len(flat) // 10)].sum() / flat.sum()
        contrasts = [contrast_proxy(img) for img, _, _, _ in examples]
        sals = [s for _, s, _, _ in examples]
        corr = float(np.mean([np.corrcoef(c.ravel(), s.ravel())[0, 1]
                              for c, s in zip(contrasts, sals)]))
        c, p = profiles[-1]
        rows.append({
            "config": name,
            "n_images": n_seen,
            "share_of_saliency_in_top10pct_pixels": round(float(share_top10), 3),
            "corr_with_local_contrast": round(corr, 3),
            "centre_over_edge": round(float(np.nanmean(p[:5]) / np.nanmean(p[-5:])), 3),
        })

    plot_mean_maps(maps, list(run_names))
    plot_radial(profiles, list(run_names))

    # how similar are two configurations' maps to each other?
    if len(maps) >= 2:
        for i in range(len(maps)):
            for j in range(i + 1, len(maps)):
                r = np.corrcoef(maps[i].ravel(), maps[j].ravel())[0, 1]
                print(f"  mean-map correlation {run_names[i]} vs {run_names[j]}: {r:+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "saliency_summary.csv", index=False)
    print("\n" + df.to_string(index=False))
    print("\nsaved saliency figures and saliency_summary.csv")
    return df, maps


if __name__ == "__main__":
    run_saliency_analysis()
