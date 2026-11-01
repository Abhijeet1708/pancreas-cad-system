# Slice Tumor Classifier Evaluation Summary

- **Checkpoint**: `/kaggle/working/cls_run/best.pt`
- **Split**: `test`
- **Validation Threshold**: `0.044250715523958206`
- **Total Slices**: 1398 (400 pos, 998 neg)
- **Total Cases**: 42
- **Valid Bootstrap Resamples**: 1000 / 1000
- **ROC AUC**: 0.6658 [0.5975, 0.7316]

## Classification Metrics

| Metric | Validation Threshold | Fixed 0.5 Threshold |
|---|---|---|
| Threshold | 0.0443 | 0.5000 |
| Accuracy | 0.5644 [0.5168, 0.6113] | 0.7089 [0.6680, 0.7506] |
| Sensitivity | 0.6825 [0.5721, 0.7749] | 0.4100 [0.2969, 0.5192] |
| Specificity | 0.5170 [0.4440, 0.5890] | 0.8287 [0.7820, 0.8717] |
| Precision (PPV) | 0.3616 [0.3062, 0.4196] | 0.4896 [0.4011, 0.5808] |
| F1 Score | 0.4727 [0.4067, 0.5302] | 0.4463 [0.3461, 0.5373] |

## Confusion Matrix Counts

| Condition | Val Threshold Count | 0.5 Threshold Count |
|---|---|---|
| True Positives (TP) | 273 | 164 |
| False Positives (FP) | 482 | 171 |
| True Negatives (TN) | 516 | 827 |
| False Negatives (FN) | 127 | 236 |
