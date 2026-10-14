# Retinal age prediction from fundus images

TAU Deep Learning (2026b) — final project.
**Shahaf Gino & Adi Dror**

A small convolutional network is trained from scratch to predict a patient's
chronological age from a single retinal fundus photograph, and the project then
asks what the network actually learned: where the error lives, whether its
systematic bias can be corrected, where in the image it looks, and whether the
gap between predicted and real age relates to anything clinical.

---

## Headline result

| model | test MAE (years) | RMSE | R² |
|---|---|---|---|
| predict the training mean (null) | 14.32 | 18.03 | 0.00 |
| ridge on hand-crafted features | 9.55 | 12.64 | 0.51 |
| **best CNN** (`cnn_aug`) | **7.64 ± 0.13** | 10.16 | 0.68 |

Every CNN configuration beats both baselines. The best is a 360k-parameter
network with weight decay, dropout, early stopping and light augmentation,
trained for at most 20 epochs on 6,902 images.

For context, Poplin et al. (2018) report MAE 3.26 on ~285,000 images with a
pretrained Inception-v3. We are two orders of magnitude smaller in data and
training from scratch, so the comparison is about method, not about beating them.

---

## What is in here

```
src/                 all project code (see below)
notebooks/
  kaggle_01_prepare_data.ipynb        one-time: build the 224px image cache
  kaggle_02_run_all_experiments.ipynb every experiment, in the order they ran
results/
  metrics/           one JSON per run, plus the analysis CSVs
  figures/           85 PNGs — learning curves, pred-vs-true, and the analyses
  checkpoints/       36 trained models (config x seed)
reports/
  Final_Report_Retinal_Age.pdf  the final report (Hebrew)
docs/
  RESULTS.md         every result with its interpretation
```

### The code

| file | what it does |
|---|---|
| `dataset.py` | splits, patient-level leakage guard, the 224px cache, dataloaders with seeded workers |
| `models.py` | `RetinaCNN` — one class covering every architecture variant; `match_width_for_params` for fair comparisons |
| `train.py` | training loop, the experiment matrix, the ridge and mean baselines |
| `evaluate.py` | inference on a split |
| `analysis.py` | per-age-group error and bias |
| `calibration.py` | two post-hoc corrections for the model's compressed predictions |
| `saliency.py` | gradient saliency (lecture 10, slides 53–57) |
| `biomarker.py` | the corrected age gap against the clinical labels |
| `utils.py` | seeding, device, early stopping, metrics, plots |

---

## Data

The images come from the public `ramankamran/retina-age-analysis` dataset on
Hugging Face: 9,857 fundus photographs with patient age and a set of clinical
labels (diabetes, diabetic retinopathy, macular oedema, hypertensive
retinopathy, optic disc / vessel / macula findings, camera model).

Splits are **by patient, not by image** — most patients contribute two images,
one per eye, and splitting by image would put the same retina on both sides of
the wall. `check_patient_leakage()` runs before every session and raises if any
patient appears in more than one split.

| split | images | patients |
|---|---|---|
| train | 6,902 | 3,775 |
| val | 1,493 | 809 |
| test | 1,462 | 809 |

Ages run from 1 to 99 with a mean of 57.4 and a heavy concentration in the
50–79 range, which turns out to explain most of the model's behaviour.

---

## Reproducing the results

Everything ran on Kaggle's free GPU (T4). Total GPU time: about 3.5 hours.

1. **Build the image cache** — run `notebooks/kaggle_01_prepare_data.ipynb`
   once, with internet on. It downloads the dataset and writes an `images_224`
   folder plus the split CSVs. Decoding the original ~1900×1900 JPEGs every
   epoch made an epoch take 600 seconds with the GPU idle; the cache brought
   that to 11 seconds.
2. **Publish `src/` as a Kaggle dataset** named `Retina-code`.
3. **Run `notebooks/kaggle_02_run_all_experiments.ipynb`** with both attached.
   `train_cnn` skips any run whose metrics file already exists, so the notebook
   is safe to re-run and safe to resume after a disconnect.

Locally, `pip install -r requirements.txt` and the same code runs on CPU or MPS,
just slowly.

---

## How the experiments are designed

Every configuration differs from its anchor in **exactly one factor**. An early
version of this project compared configurations that differed in six things at
once, which meant nothing could be attributed to anything.

| config | anchor | the one thing that changes |
|---|---|---|
| `cnn_plain` | ridge | a CNN instead of a linear model |
| `cnn_dropout` | `cnn_plain` | dropout |
| `cnn_wd` | `cnn_plain` | weight decay |
| `cnn_es` | `cnn_plain` | early stopping |
| `cnn_all` | `cnn_plain` | all three together (epoch cap 15 → 20, but early stopping ended every seed at epoch 10–14, so the cap never bound) |
| `cnn_aug` | `cnn_all` | augmentation |
| `cnn_aug_long` | `cnn_aug` | epoch budget 20 → 40 |
| `arch_7x7` | `cnn_all` | 7×7 filters, parameter-matched (3 blocks instead of 4, so receptive field and depth change together) |
| `arch_valid` | `cnn_all` | valid padding, parameter-identical |
| `arch_drop_early` | `cnn_all` | dropout inside the conv blocks |
| `loss_l1` | `cnn_all` | L1 loss |
| `loss_huber` | `cnn_all` | Huber loss (δ = 5 years) |

Two details make those comparisons honest rather than merely labelled:

- **Parameter matching.** An `AdaptiveAvgPool2d` before the classifier head
  makes the head size independent of the padding mode, so `arch_valid` has
  *identical* parameter count to the anchor (360,321). For `arch_7x7` a
  bisection on a channel-width multiplier lands at 359,446 — within 0.25%.
  Without this, 7×7 would have had 1.98× the parameters and the experiment
  would have measured capacity, not receptive field.
- **Checkpoint selection is a declared factor.** A run without early stopping
  is evaluated on its final weights; a run with it, on its best-validation
  weights. That difference *is* early stopping, so it appears in the matrix
  rather than varying silently.

Every configuration is run with 3 seeds and reported as mean ± std.

---

## Known limitations

Three, stated plainly because they change how the numbers should be read.

**1. Runs are not bit-reproducible on GPU.** `train.py` sets
`cudnn.benchmark = True`, which lets cuDNN pick convolution algorithms by
timing them. The winner depends on what else the GPU is doing, so two runs with
the same seed can diverge. We measured it: `cnn_aug` seed 1 trained 20 epochs to
MAE 7.62, while `cnn_aug_long` seed 1 — same config, only a larger epoch
ceiling — stopped at epoch 9 with MAE 8.87. **Differences below roughly half a
year of MAE should not be interpreted.** The large ones (any CNN vs the
baselines; the ablations vs `cnn_all`) are far outside this.

**2. The model compresses its predictions.** It spans only 57% of the true age
range, so young patients are predicted too old and old patients too young. This
is not a bug — it is what minimising MSE on an unbalanced age distribution
produces, and `calibration.py` shows that correcting it costs 2.8 years of MAE.

**3. The clinical analysis is descriptive only.** With 1,462 test images we
report group sizes, means and effect sizes, and run no significance tests. Zhu
et al. (2023) needed ~36,000 participants for a hazard ratio of 1.02.

---

## Licence and attribution

Course project, not for clinical use. The dataset is used under its own terms.
