# Slice Tumor Classifier Evaluation Summary

- **Checkpoint**: `/kaggle/working/cls_run/best.pt`
- **Split**: `val`
- **Validation Threshold**: `0.044250715523958206`
- **Total Slices**: 1366 (367 pos, 999 neg)
- **Total Cases**: 42
- **Valid Bootstrap Resamples**: 1000 / 1000
- **ROC AUC**: 0.6808 [0.6304, 0.7299]

## Classification Metrics

| Metric | Validation Threshold | Fixed 0.5 Threshold |
|---|---|---|
| Threshold | 0.0443 | 0.5000 |
| Accuracy | 0.5944 [0.5425, 0.6397] | 0.7138 [0.6794, 0.7450] |
| Sensitivity | 0.7221 [0.6496, 0.7904] | 0.3842 [0.2965, 0.4813] |
| Specificity | 0.5475 [0.4712, 0.6149] | 0.8348 [0.7754, 0.8828] |
| Precision (PPV) | 0.3696 [0.3180, 0.4288] | 0.4608 [0.3826, 0.5560] |
| F1 Score | 0.4889 [0.4379, 0.5406] | 0.4190 [0.3491, 0.4867] |

## Confusion Matrix Counts

| Condition | Val Threshold Count | 0.5 Threshold Count |
|---|---|---|
| True Positives (TP) | 265 | 141 |
| False Positives (FP) | 452 | 165 |
| True Negatives (TN) | 547 | 834 |
| False Negatives (FN) | 102 | 226 |
