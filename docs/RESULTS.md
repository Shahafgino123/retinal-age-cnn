# Results

Every number the project produced, with what it means. Source files are named
so each table can be traced back to the run that produced it.

All CNN results are the mean over seeds 0, 1, 2 on the **test** split, which was
touched only for final evaluation. Model selection used validation.

---

## 1. The full comparison

`results/metrics/*.json`

| config | MAE | sd | RMSE | R² | params | epochs run (seeds 0, 1, 2) |
|---|---|---|---|---|---|---|
| `cnn_aug` | **7.638** | 0.132 | 10.161 | 0.682 | 360,321 | 20, 20, 20 |
| `cnn_wd` | 7.655 | 0.157 | 10.154 | 0.683 | 360,321 | 15, 15, 15 |
| `loss_huber` | 7.722 | 0.270 | 10.416 | 0.666 | 360,321 | 20, 19, 20 |
| `arch_valid` | 7.888 | 0.539 | 10.511 | 0.659 | 360,321 | 11, 20, 20 |
| `cnn_es` | 7.970 | 0.305 | 10.587 | 0.655 | 360,321 | 12, 15, 15 |
| `arch_7x7` | 8.018 | 0.079 | 10.632 | 0.652 | 359,446 | 20, 20, 20 |
| `cnn_aug_long` | 8.032 | 0.733 | 10.651 | 0.649 | 360,321 | 21, 9, 30 |
| `loss_l1` | 8.109 | 0.142 | 10.847 | 0.638 | 360,321 | 15, 19, 20 |
| `cnn_plain` | 8.129 | 0.490 | 10.664 | 0.649 | 360,321 | 15, 15, 15 |
| `cnn_dropout` | 8.600 | 0.606 | 11.187 | 0.614 | 360,321 | 15, 15, 15 |
| `cnn_all` | 8.672 | 0.085 | 11.457 | 0.596 | 360,321 | 10, 13, 14 |
| `arch_drop_early` | 9.051 | 0.103 | 12.056 | 0.553 | 360,321 | 20, 19, 20 |
| `ridge_linear` | 9.553 | — | 12.645 | 0.508 | — | — |
| `mean_predictor` | 14.316 | — | 18.025 | 0.000 | — | — |

`sd` is the sample standard deviation across the three seeds.

**Read the sd column before reading the ranking.** See limitation 1 in the
README: differences under about half a year are inside the noise floor. Every
configuration within half a year of the best (nine of the twelve, down to
`cnn_plain`) is, for practical purposes, tied with it.

### The null model

Predicting the training mean (57.42 years) for every image gives MAE 14.32 and
RMSE 18.03. The RMSE of a mean predictor equals the standard deviation of the
labels, which is an independent check that the split loaded correctly — and it
came out identical in all four sessions.

### Regularization, isolated

Against the `cnn_plain` anchor (8.129):

(`cnn_all` has a 20-epoch cap against `cnn_plain`'s 15, but early stopping
ended its three seeds at epochs 10, 13 and 14, so the cap never came into play.)

| factor | Δ MAE | reading |
|---|---|---|
| weight decay | **−0.47** | the only regularizer that helps |
| early stopping | −0.16 | inside the noise |
| dropout | +0.47 | hurts |
| all three | +0.54 | worse than any one alone |

The interaction is the interesting part: combining three regularizers is
**+0.71 years worse than additivity predicts** (8.672 measured against the
7.97 the individual deltas sum to) and **+1.02 years worse than the best single
one**, weight decay. The model has 360k parameters
and 6,902 training images; it is not in the regime where stacking regularization
pays. Note also that variance collapses as regularization is added — sd goes
from 0.49 (`cnn_plain`) to 0.085 (`cnn_all`) — which is the bias–variance
tradeoff appearing directly in the seed spread.

### Ablations against `cnn_all` (8.672)

| change | Δ MAE |
|---|---|
| augmentation | −1.03 |
| Huber loss | −0.95 |
| valid padding | −0.78 |
| 7×7 filters | −0.65 |
| L1 loss | −0.56 |
| dropout moved into conv blocks | +0.38 |

Five of six single changes improve the anchor, which is consistent with the
regularization finding: `cnn_all` is over-regularized, and almost anything that
loosens it helps.

**Huber beats L1** (7.72 vs 8.11) even though L1 is exactly the metric being
reported. L1's gradient has constant magnitude no matter how small the error,
which makes fine convergence harder; Huber is quadratic below δ = 5 years and
linear above it, so it converges smoothly while staying robust to outliers.
δ matters: at PyTorch's default of 1.0 our 8–15 year errors sit entirely in the
linear regime and Huber becomes numerically identical to L1 — we hit exactly
that and caught it because two configurations produced identical metrics.

### Does a bigger epoch budget help? No.

`cnn_aug` used all 20 epochs in every seed without early stopping firing, so its
result looked like a lower bound. Doubling the budget to 40:

| seed | `cnn_aug` (20) | `cnn_aug_long` (40) | epochs |
|---|---|---|---|
| 0 | 7.780 | 7.723 | 20 → 21 |
| 1 | 7.617 | 8.869 | 20 → **9** |
| 2 | 7.517 | 7.503 | 20 → 30 |

Seeds 0 and 2 improved by 0.06 and 0.01 years — nothing. **The ceiling was not
the limiting factor.** Seed 1 is what makes this run worth reporting: a larger
budget cannot cause an *earlier* stop, so the two runs must have diverged
despite sharing a seed. That is what exposed the cuDNN nondeterminism described
in the README.

---

## 2. Where the error lives

`results/metrics/cnn_all_error_by_age_group.csv`
Figures: `cnn_all_mae_by_age_group.png`, `cnn_all_bias_by_age_group.png`,
`cnn_all_age_gap_distribution.png`
The Hebrew report prints `cnn_all_mae_by_age_group_report.png` — same numbers,
re-rendered larger for print.

| age group | n | MAE | bias | reliable |
|---|---|---|---|---|
| 0–9 | 8 | 36.82 | +36.82 | no |
| 10–19 | 57 | 14.70 | +14.45 | yes |
| 20–29 | 71 | 14.10 | +13.43 | yes |
| 30–39 | 123 | 11.62 | +10.82 | yes |
| 40–49 | 153 | 8.44 | +7.55 | yes |
| 50–59 | 289 | 6.04 | +3.45 | yes |
| 60–69 | 362 | 6.22 | −0.46 | yes |
| 70–79 | 288 | 8.19 | −6.76 | yes |
| 80–89 | 103 | 13.85 | −13.74 | yes |
| 90–99 | 8 | 17.07 | −17.07 | no |

Textbook regression to the mean. The bias runs monotonically from +14.5 years
at 10–19 to −13.7 at 80–89 and crosses zero at 60–69, right beside the training
mean of 57.4. MAE is U-shaped with a minimum of 6.0 in the modal group.

The overall MAE of 8.74 is therefore an average over two very different
regimes, and quoting it alone would be misleading: the model is reasonable in
middle age and close to useless at the extremes.

---

## 3. Can the compression be corrected?

`results/metrics/cnn_all_calibration_overall.csv`, `..._by_age_group.csv`
Figures: `cnn_all_mae_calibration.png`, `cnn_all_bias_calibration.png`

Measured on validation: `pred = 0.569 × true + c`. The model spans only **57%**
of the real age range. Both corrections are fitted on validation and applied to
test.

| variant | MAE | RMSE | R² | bias | corr(gap, age) |
|---|---|---|---|---|---|
| raw | 8.738 | 11.703 | 0.578 | +1.29 | −0.677 |
| A: MSE-optimal | 8.748 | 11.681 | 0.580 | +0.74 | −0.704 |
| B: bias-removing | 11.508 | 15.074 | 0.301 | +0.95 | **−0.012** |
| isotonic (check) | 8.626 | 11.580 | 0.587 | +0.79 | −0.682 |

Three conclusions:

- **A changes almost nothing** (slope 0.967, near identity). That is not a
  failure — it confirms the raw predictions are already MSE-optimal. The
  compression is not a bug to be fixed; it is the correct behaviour for the
  loss we trained on.
- **B works, and the price is explicit.** It drives the age dependence to
  −0.012, essentially zero, and costs 2.8 years of MAE — a 32% increase. This
  is the bias–variance tradeoff made quantitative.
- **The isotonic check** improves MAE by only 0.11 years, so there is almost no
  non-linear structure left to exploit. Useful because it closes the question
  rather than leaving it open.

Practical consequence: use raw predictions when accuracy is the goal, and the
B-corrected gap when the gap itself is the object of study — which is exactly
what the biomarker analysis does.

---

## 4. Where does the model look?

`results/metrics/saliency_summary.csv`
Figures: `saliency_mean_maps.png`, `saliency_radial_profile.png`,
`cnn_all_saliency_examples.png`, `arch_7x7_saliency_examples.png`

Gradient of the predicted age w.r.t. input pixels, reduced by max over RGB —
the method from lecture 10, slides 53–57. Grad-CAM would answer the same
question but was never taught in the course, so we used what was.

| | `cnn_all` (3×3) | `arch_7x7` |
|---|---|---|
| share of saliency in top 10% of pixels | 0.218 | 0.187 |
| centre / edge ratio | 1.322 | **2.244** |
| correlation with local contrast | 0.158 | 0.128 |

Mean-map correlation between the two: **+0.411**.

That number is the point of the experiment. High enough that both models look
at broadly similar regions, low enough to confirm they are not doing the same
thing — which is the assumption the 7×7-vs-3×3 ablation rests on. The 7×7 model
is markedly more centre-weighted and less peaky, exactly what a larger receptive
field predicts.

**We could not reproduce Poplin's vessel finding.** They had three
ophthalmologists score 100 attention maps and found blood vessels highlighted
in 95% of them. Our correlation with local contrast — a crude proxy for
vasculature, and explicitly labelled as such in the code — is 0.13–0.16 for both
models. This is weak evidence, and the honest statement is that with this proxy
and a small from-scratch model we have no basis for claiming the network uses
vascular structure.

---

## 5. Is the age gap clinically related?

`results/metrics/cnn_all_age_gap_by_condition.csv`, `..._gap_correction_diag.csv`
Figure: `cnn_all_age_gap_by_condition.png`
The Hebrew report prints `cnn_all_age_gap_by_condition_report.png` — same
numbers, laid out one row per finding.

Uses the **B-corrected** gap. The raw gap correlates −0.677 with age, so
comparing raw gaps between patient groups would mostly compare their ages.
After correction the correlation is −0.012, so group comparisons are meaningful.

Five variables were flagged age-confounded (groups differing by more than 5
years in mean age) and are **not** reported as findings:
`age_group_broad_numeric`, `camera`, `macula`, `hemorrhage`,
`hypertensive_retinopathy`.

Of the rest, the largest effects:

| variable | Cohen's d | mean gap, level 0 | level 1 | age diff |
|---|---|---|---|---|
| diabetes | 0.285 | +2.05 (n=1084) | −2.21 (n=378) | −2.8 |
| optic disc finding | 0.281 | +1.93 (n=1122) | −2.27 (n=340) | −4.6 |
| vessel finding | 0.255 | +1.19 (n=1371) | −2.64 (n=91) | −2.2 |
| increased cup/disc | 0.249 | +1.82 (n=1121) | −1.91 (n=341) | −4.6 |
| diabetic retinopathy | −0.075 | +0.85 (n=1330) | +1.98 (n=132) | −1.2 |

**The direction is backwards.** Diabetic patients get a *negative* gap — the
model calls them younger than they are — and the same pattern repeats across
every pathology label. The literature predicts the opposite: disease should make
a retina look older.

This is too consistent to be chance, and it is not explained by age: with a
residual correlation of 0.012, a 3–5 year age difference between groups
contributes less than a tenth of a year. The more likely explanation is that
these labels track how patients entered the dataset — referral route, screening
programme, imaging equipment — rather than retinal biology.

**This is an observation with its direction stated, not a finding.**
No significance tests were run: this is a description of a
pattern in 1,462 images, and the honest conclusion is that it raises questions
about the label semantics rather than answering anything about biological
ageing.

---

## 6. Methodological problems we found and fixed

Most of these would have silently produced wrong conclusions.

| problem | how it was caught | consequence if missed |
|---|---|---|
| Model built before `set_seed()` | two identical runs gave different weights | every experiment started from a different init; part of every measured gap was init noise |
| Three "different" ablations were byte-identical models | hashing the constructed models | three wasted runs reported as three findings |
| 7×7 had 1.98× the parameters; valid padding 0.74× | counting parameters | the architecture ablations would have measured capacity, not architecture |
| Huber ≡ L1 at PyTorch's default δ=1.0 | two configs produced identical metrics to 3 decimals | a whole loss comparison that compared nothing |
| Mixed checkpoint selection (final vs best-val) | reading the training loop | a hidden extra variable in every comparison |
| Unseeded dataloader workers and shuffle order | reading `get_dataloaders` | reproducibility that looked real and was not |
| Biomarker sign flip under age confounding | injecting a synthetic +3-year effect and getting −1.32 back | a confidently reported disease effect with the wrong sign |
| Non-deep baseline read full-resolution images while CNNs read 224px | a crash on Kaggle | an unfair baseline comparison |
| cuDNN nondeterminism | `cnn_aug_long` seed 1 stopping at epoch 9 | over-reading differences of 0.1–0.3 years |

The last one was found after all runs were complete and is reported rather than
fixed, because fixing it would have required re-running all 36 models.
