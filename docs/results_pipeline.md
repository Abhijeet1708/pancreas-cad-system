# End-to-end pipeline results (three test cases)

Research prototype. Not clinically validated and not for diagnostic use.

## What was run

`python -m src.pipeline.run` on three cases from the **test split**, with the final models:
the segmentation checkpoint from iteration 22,500 and the slice classifier checkpoint from
epoch 11 (validation threshold 0.0443). The pipeline preprocesses the CT volume, segments the
pancreas and tumor with the 3D U-Net, crops every slice with at least 50 predicted pancreas
pixels around the predicted pancreas, scores each slice with the classifier, and computes
Grad-CAM for the top 3 slices.

The cases were picked by rule, not by appearance: the 15th, 50th and 90th percentile of
ground-truth tumor size within the test split. They illustrate behavior and are **not a
performance estimate**; see `results_segmentation.md` and `results_classification.md` for that.

## Results

| Case | Tumor size (equiv. diameter) | Size group in evaluation | Pancreas Dice | Tumor Dice | Tumor volume, predicted vs true (mL) | Segmentation finding | Slices flagged by classifier | Max slice probability |
|---|---|---|---|---|---|---|---|---|
| pancreas_067 | 14.8 mm | small (< 20 mm) | 0.874 | 0.000 | 0.10 vs 1.71 | not found (39 voxels, threshold 100) | 11 of 34 | 0.946 |
| pancreas_198 | 22.6 mm | large (>= 20 mm) | 0.869 | 0.712 | 8.97 vs 6.08 | found (3,587 voxels) | 13 of 26 | 0.9995 |
| pancreas_213 | 34.3 mm | large (>= 20 mm) | 0.823 | 0.869 | 21.37 vs 21.16 | found (8,547 voxels) | 23 of 33 | 0.99998 |

Dice values and true tumor volumes come from the segmentation evaluation of the same cases
(`docs/results/segmentation/test/per_case.csv`). The predicted tumor volumes from the pipeline
match the evaluation's predicted volumes for all three cases.

## Observations

- The small tumor (pancreas_067) was missed by the segmentation: 39 predicted tumor voxels
  against a true volume of 1.71 mL, tumor Dice 0.000. The pancreas itself was outlined well
  (Dice 0.874). This is consistent with the test-split result that small tumors are mostly
  missed (5 of 14 detected).
- The medium case over-segments the tumor (8.97 mL predicted against 6.08 mL true) but overlaps
  it well (Dice 0.712). The large case is close in volume and overlap.
- The classifier flags many slices at its stored threshold (32% to 70% of evaluated slices
  here), so a flagged-slice count is not a detection decision. In the missed small case the
  classifier still flagged 11 of 34 slices while the segmentation found no tumor. The pipeline
  reports the two findings separately and does not combine them into a diagnosis.
- The Grad-CAM overlays show which regions drove the slice classifier's decision. Because the
  classifier is weak (test AUC 0.666), they should not be read as tumor localization.
- Raw-scan input and preprocessed input agree: for pancreas_198, the NIfTI run and the `.npz` run
  give the same image shape (111 x 460 x 460), pancreas volume (84.565 mL) and tumor volume
  (8.9675 mL). One of the three saved figure slices differs (48 instead of 54), probably because
  several slices have probabilities within 0.005 of each other near 1.

## Timing (Tesla T4 GPU)

- Preprocessed `.npz` input, models already loaded: 7.3 to 8.4 s per case. Segmentation
  inference itself took 1.5 to 2.7 s.
- First case in the session: 26.5 s, of which 18.3 s was model loading and GPU warm-up.
- Raw NIfTI input: 18.2 s, of which 14.4 s was reading and resampling the scan.
- CPU timing was not measured.

## Files

`docs/results/pipeline/<case>/result.json` and `docs/results/pipeline/<case>/figures/`
(three slice figures per run: CT crop, predicted segmentation tint, Grad-CAM overlay).
The folder `medium_pancreas_198_from_nifti` is the raw-scan run of the same case.
Each figure carries the footer "Research prototype - not for clinical use". Data credit:
MSD Task07 Pancreas (Antonelli et al., Nature Communications, 2022).
