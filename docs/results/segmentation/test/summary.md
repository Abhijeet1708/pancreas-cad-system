# Segmentation Model Evaluation Summary

- **Checkpoint**: `/kaggle/input/datasets/abhijeetupare/seg-stage2/run/best.pt`
- **Iteration**: 22500
- **Split**: test
- **Requested Cases**: 42
- **Evaluated Cases**: 42
- **Unknown Size Cases**: 0
- **Failed Cases**: 0

## Performance by Tumor Size Group

| Group | N | Pancreas Dice (Mean ± SD [95% CI] / Median) | Tumor Dice (Mean ± SD [95% CI] / Median) | Tumor Detected (Overlap) | Strict Detected (Dice >= 0.10) |
|---|---|---|---|---|---|
| Overall | 42 | 0.762 ± 0.133 [0.720, 0.797] (med 0.795) | 0.323 ± 0.310 [0.232, 0.419] (med 0.243) | 66.7% (28/42) [51.6%, 79.0%] | 59.5% (25/42) [44.5%, 73.0%] |
| Small (<20mm) | 14 | 0.794 ± 0.061 [0.762, 0.824] (med 0.808) | 0.128 ± 0.227 [0.027, 0.252] (med 0.000) | 35.7% (5/14) [16.3%, 61.2%] | 35.7% (5/14) [16.3%, 61.2%] |
| Large (>=20mm) | 28 | 0.746 ± 0.155 [0.685, 0.796] (med 0.787) | 0.421 ± 0.302 [0.309, 0.530] (med 0.537) | 82.1% (23/28) [64.4%, 92.1%] | 71.4% (20/28) [52.9%, 84.7%] |

> Note: Every case in this dataset contains a tumor, so specificity cannot be computed here.
