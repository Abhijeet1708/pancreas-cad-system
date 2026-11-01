# Slice-level tumor classifier results

Research prototype. Not clinically validated and not for diagnostic use.

## Task and data

- **Task:** classify axial slices of the pancreas as tumor-containing (1) or tumor-free (0).
- **Input:** a 160 x 160 px crop (1 mm in-plane) centered on the ground-truth pancreas centroid;
  the central 128 x 128 window is used at evaluation, a random 128 x 128 window in training.
  Ground-truth masks are used for cropping in this experiment.
- **Labels:** positive if at least 30 tumor pixels fall inside the central 96 x 96 window of the crop
  (so a positive slice always shows tumor in every crop); negative if the slice has no tumor
  pixels at all. Slices with tumor that fail the positive rule are skipped as ambiguous.
- **Splits:** the same case-level split as segmentation (`docs/splits.json`, seed 42).

| Split | Cases | Positive slices | Negative slices | Skipped |
|---|---|---|---|---|
| Train | 197 | 1,889 | 4,560 | 42 |
| Val | 42 | 367 | 999 | 1 |
| Test | 42 | 400 | 998 | 3 |

Every case in the dataset contains a tumor, so this is a slice-level task. Case-level
specificity cannot be measured.

## Model and training

- ResNet18 pretrained on ImageNet (weights downloaded successfully), one output logit.
- BCE loss with `pos_weight = n_neg / n_pos`, AdamW (lr 1e-4, weight decay 1e-4), cosine schedule,
  15 epochs, batch size 64, about 1 minute on a Tesla T4.
- Augmentation: random crop, horizontal flip, small intensity scale and shift.
- Checkpoint chosen by validation AUC (epoch 11). The decision threshold (0.0443) maximizes
  Youden's J on the validation split. The test split was evaluated once.
- Confidence intervals: case-cluster bootstrap (1,000 resamples, all valid).

## Results

### Test split (headline result)

ROC AUC: **0.666** [0.598, 0.732]

| Metric | At validation threshold (0.0443) | At fixed threshold 0.5 |
|---|---|---|
| Accuracy | 0.564 [0.517, 0.611] | 0.709 [0.668, 0.751] |
| Sensitivity | 0.683 [0.572, 0.775] | 0.410 [0.297, 0.519] |
| Specificity | 0.517 [0.444, 0.589] | 0.829 [0.782, 0.872] |
| Precision | 0.362 [0.306, 0.420] | 0.490 [0.401, 0.581] |
| F1 | 0.473 [0.407, 0.530] | 0.446 [0.346, 0.537] |
| TP / FP / TN / FN | 273 / 482 / 516 / 127 | 164 / 171 / 827 / 236 |

### Validation split (for reference)

ROC AUC: 0.681 [0.630, 0.730]

| Metric | At validation threshold (0.0443) | At fixed threshold 0.5 |
|---|---|---|
| Accuracy | 0.594 [0.543, 0.640] | 0.714 [0.679, 0.745] |
| Sensitivity | 0.722 [0.650, 0.790] | 0.384 [0.297, 0.481] |
| Specificity | 0.548 [0.471, 0.615] | 0.835 [0.775, 0.883] |
| Precision | 0.370 [0.318, 0.429] | 0.461 [0.383, 0.556] |
| F1 | 0.489 [0.438, 0.541] | 0.419 [0.349, 0.487] |
| TP / FP / TN / FN | 265 / 452 / 547 / 102 | 141 / 165 / 834 / 226 |

### Training curve (validation split)

| Epoch | Train loss | Val loss | Val AUC |
|---|---|---|---|
| 1 | 0.907 | 0.952 | 0.650 |
| 2 | 0.669 | 1.091 | 0.644 |
| 3 | 0.507 | 1.164 | 0.671 |
| 4 | 0.386 | 1.308 | 0.659 |
| 5 | 0.307 | 1.337 | 0.673 |
| 6 | 0.251 | 1.527 | 0.646 |
| 7 | 0.206 | 1.492 | 0.643 |
| 8 | 0.175 | 1.476 | 0.652 |
| 9 | 0.139 | 1.490 | 0.666 |
| 10 | 0.115 | 1.561 | 0.680 |
| **11 (selected)** | 0.094 | 1.742 | **0.681** |
| 12 | 0.087 | 1.619 | 0.680 |
| 13 | 0.076 | 1.733 | 0.680 |
| 14 | 0.073 | 1.718 | 0.679 |
| 15 | 0.071 | 1.676 | 0.679 |

## Comparison with the targets in `requirements.md`

The targets were AUC > 0.75 and accuracy, sensitivity, specificity and F1 all above 0.70.
**They were not met.** Test AUC is 0.666. At the two thresholds reported, accuracy,
specificity and F1 are below 0.70 at the validation threshold, and sensitivity and F1 are
below 0.70 at 0.5. The targets are left unchanged and the shortfall is reported as measured.

Accuracy at threshold 0.5 (0.709) should not be read as good performance: 71.4% of test
slices are negative, so predicting "no tumor" for every slice would score 71.4%.

## Observations

- The model overfits: training loss falls from 0.91 to 0.07, validation loss rises from 0.95 to
  about 1.7, and validation AUC stays between 0.64 and 0.68 for all 15 epochs.
- Validation AUC (0.681) and test AUC (0.666) are close, so there is no sign of a validation/test
  mismatch. The classifier is weak on both.
- At the validation threshold the classifier flags 54% of all test slices to find 68% of the
  tumor slices; at 0.5 it flags 24% and finds 41%.
- Possible contributors that were not tested: slices at the top and bottom edge of a tumor look
  almost the same as tumor slices but are labeled negative; the 2D input has no information from
  neighboring slices; horizontal flips may be a poor augmentation for abdominal anatomy;
  there are only 197 training cases, and neighboring slices of one case are highly correlated.

## Limitations

- Slice-level task on one public dataset, one fixed split, no external validation.
- Ground-truth masks are used for cropping here; the full pipeline will crop from the predicted
  pancreas, which can only add error.
- No healthy cases, so specificity on tumor-free scans is not measured.
- One training run, no hyperparameter search.
- The test split has been used once for reporting.

## Files

- `docs/results/classification/test/` and `docs/results/classification/val/`: `summary.md`,
  `summary.json`, `roc_points.csv`, `per_slice.csv` (and plots, if present).
- `docs/results/classification/train/`: `metrics.csv`, `run_config.json`.
