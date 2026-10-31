# Segmentation results

Research prototype. Not clinically validated and not for diagnostic use.

## Setup

- **Data:** 281 contrast-enhanced CT cases from MSD Task07 Pancreas, preprocessed to
  1.0 x 1.0 x 2.5 mm with a fixed HU window of [-100, 240] (see `src/preprocessing/`).
- **Split:** fixed, stratified by tumor size, seed 42: 197 train / 42 val / 42 test
  (`docs/splits.json`). The test split was evaluated once, after the checkpoint was chosen.
- **Model:** MONAI 3D U-Net, Dice + cross-entropy loss, patch size 64 x 128 x 128,
  full-volume sliding-window inference.
- **Training:** two Kaggle sessions on one Tesla T4 (75.0 + 37.6 minutes, about 113 minutes
  in total), one 30,000-iteration cosine learning-rate schedule resumed from a checkpoint.
- **Selected checkpoint:** iteration 22,500, chosen by the best validation score
  (mean of pancreas and tumor Dice = 0.581). Test results were not used to choose it.
- **Evaluation:** `src/segmentation/evaluate.py` (merge commit `60be630`, PR #17).
  Exact parameters (seed, bootstrap samples, arguments) are in each `summary.json`.

## Definitions

- **Pancreas Dice:** labels 1 or 2 (whole organ including tumor). **Tumor Dice:** label 2.
- **Size group:** sphere-equivalent tumor diameter computed from the ground-truth tumor volume,
  small if < 20 mm, large otherwise. This is not the longest-axis diameter a radiologist measures.
- **Detected (overlap):** at least one predicted tumor voxel overlaps the ground-truth tumor.
  This is lenient. **Detected (Dice >= 0.10):** a stricter criterion.
- **Confidence intervals:** percentile bootstrap for mean Dice; Wilson score interval for detection rates.
- Every case contains a tumor, so specificity cannot be computed on this dataset.

## Test split (headline result, n = 42)

| Group | N | Pancreas Dice, mean [95% CI] | Tumor Dice, mean [95% CI] (median) | Detected (overlap) | Detected (Dice >= 0.10) |
|---|---|---|---|---|---|
| Overall | 42 | 0.762 [0.720, 0.797] | 0.323 [0.232, 0.419] (0.243) | 66.7% (28/42) [51.6%, 79.0%] | 59.5% (25/42) [44.5%, 73.0%] |
| Small (< 20 mm) | 14 | 0.794 [0.762, 0.824] | 0.128 [0.027, 0.252] (0.000) | 35.7% (5/14) [16.3%, 61.2%] | 35.7% (5/14) [16.3%, 61.2%] |
| Large (>= 20 mm) | 28 | 0.746 [0.685, 0.796] | 0.421 [0.309, 0.530] (0.537) | 82.1% (23/28) [64.4%, 92.1%] | 71.4% (20/28) [52.9%, 84.7%] |

## Validation split (for reference, n = 42)

The validation split was used to choose the checkpoint, so these numbers are optimistic.

| Group | N | Pancreas Dice, mean [95% CI] | Tumor Dice, mean [95% CI] (median) | Detected (overlap) | Detected (Dice >= 0.10) |
|---|---|---|---|---|---|
| Overall | 42 | 0.790 [0.752, 0.820] | 0.371 [0.286, 0.453] (0.399) | 81.0% (34/42) [66.7%, 90.0%] | 73.8% (31/42) [58.9%, 84.7%] |
| Small (< 20 mm) | 16 | 0.793 [0.725, 0.838] | 0.372 [0.239, 0.502] (0.403) | 81.2% (13/16) [57.0%, 93.4%] | 75.0% (12/16) [50.5%, 89.8%] |
| Large (>= 20 mm) | 26 | 0.788 [0.741, 0.824] | 0.371 [0.266, 0.478] (0.399) | 80.8% (21/26) [62.1%, 91.5%] | 73.1% (19/26) [53.9%, 86.3%] |

The validation numbers from this script match the training log (pancreas 0.790, tumor 0.371),
which checks the evaluation script on real data.

## Training curve (validation split, n = 42)

| Iteration | Pancreas Dice | Tumor Dice | Score |
|---|---|---|---|
| 2,500 | 0.489 | 0.125 | 0.307 |
| 5,000 | 0.590 | 0.167 | 0.378 |
| 7,500 | 0.699 | 0.252 | 0.475 |
| 10,000 | 0.681 | 0.232 | 0.457 |
| 12,500 | 0.749 | 0.307 | 0.528 |
| 15,000 | 0.773 | 0.360 | 0.566 |
| 17,500 | 0.782 | 0.361 | 0.571 |
| 20,000 | 0.778 | 0.350 | 0.564 |
| 21,056 (end of session 1) | 0.783 | 0.366 | 0.574 |
| **22,500 (selected)** | **0.790** | **0.371** | **0.581** |
| 25,000 | 0.789 | 0.354 | 0.571 |
| 27,500 | 0.790 | 0.358 | 0.574 |
| 30,000 | 0.790 | 0.356 | 0.573 |

## Observations

- Pancreas Dice on the test split (0.762) is above the 0.6 target in `requirements.md`.
  Tumor Dice (0.323) has no hard target, as agreed in the requirements.
- Small tumors are mostly missed: 5 of 14 were detected on the test split, and the median
  tumor Dice for that group is 0.000. Large tumors are detected far more often (23 of 28).
  This is the weak point of the current model.
- Validation and test results differ, most visibly for small-tumor detection (81% on validation,
  36% on test). Part of the gap is expected, because the checkpoint was selected on validation.
  The size groups also hold only 14 to 28 cases, so the confidence intervals are wide
  and these estimates are unstable.
- Training had converged by about iteration 22,500; the validation score stays within about
  0.01 afterwards.

## Limitations

- One public dataset, one fixed split, no external validation.
- No healthy cases: specificity and false-positive behavior on tumor-free scans are not measured.
- The dataset contains a mix of pancreatic tumor types, so results should not be read as
  specific to pancreatic ductal adenocarcinoma.
- About two GPU-hours of training with no hyperparameter search.
- The test split was used once for reporting; any later model change based on it would
  invalidate these test numbers.

## Files

- `docs/results/segmentation/test/` and `docs/results/segmentation/val/`: `summary.md`,
  `summary.json`, `per_case.csv` (the test folder also holds overlay figures).
- Data credit: MSD Task07 Pancreas (Antonelli et al., Nature Communications, 2022).
  Check the dataset license before redistributing figures.
