# Segmentation Model Evaluation Summary

- **Checkpoint**: `/kaggle/input/datasets/abhijeetupare/seg-stage2/run/best.pt`
- **Iteration**: 22500
- **Split**: val
- **Requested Cases**: 42
- **Evaluated Cases**: 42
- **Unknown Size Cases**: 0
- **Failed Cases**: 0

## Performance by Tumor Size Group

| Group | N | Pancreas Dice (Mean ± SD [95% CI] / Median) | Tumor Dice (Mean ± SD [95% CI] / Median) | Tumor Detected (Overlap) | Strict Detected (Dice >= 0.10) |
|---|---|---|---|---|---|
| Overall | 42 | 0.790 ± 0.113 [0.752, 0.820] (med 0.814) | 0.371 ± 0.276 [0.286, 0.453] (med 0.399) | 81.0% (34/42) [66.7%, 90.0%] | 73.8% (31/42) [58.9%, 84.7%] |
| Small (<20mm) | 16 | 0.793 ± 0.123 [0.725, 0.838] (med 0.812) | 0.372 ± 0.280 [0.239, 0.502] (med 0.403) | 81.2% (13/16) [57.0%, 93.4%] | 75.0% (12/16) [50.5%, 89.8%] |
| Large (>=20mm) | 26 | 0.788 ± 0.109 [0.741, 0.824] (med 0.817) | 0.371 ± 0.279 [0.266, 0.478] (med 0.399) | 80.8% (21/26) [62.1%, 91.5%] | 73.1% (19/26) [53.9%, 86.3%] |

> Note: Every case in this dataset contains a tumor, so specificity cannot be computed here.
