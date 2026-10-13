"""
Metrics for 3D pancreas and tumor segmentation evaluation.

Conventions
-----------
- Labels:
    0: background
    1: pancreas
    2: tumor
- Dice evaluation regions:
    "pancreas": labels 1 or 2 (whole organ including tumor)
    "tumor": label 2
"""

from typing import Dict, List

import numpy as np

from src.segmentation.data import LABEL_PANCREAS, LABEL_TUMOR


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Dice similarity coefficient between two binary or boolean masks.

    Scoring conventions:
    - Both masks empty: 1.0
    - Exactly one mask empty: 0.0
    - Otherwise: 2 * |pred & gt| / (|pred| + |gt|)

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary/boolean mask.
    gt_mask : np.ndarray
        Ground-truth binary/boolean mask.

    Returns
    -------
    float
        Dice similarity coefficient in [0.0, 1.0].

    Raises
    ------
    ValueError
        If pred_mask and gt_mask have different shapes.
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Shape mismatch: pred_mask {pred_mask.shape} != gt_mask {gt_mask.shape}"
        )

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    pred_sum = int(np.count_nonzero(pred))
    gt_sum = int(np.count_nonzero(gt))

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    intersection = int(np.count_nonzero(pred & gt))
    return float((2.0 * intersection) / (pred_sum + gt_sum))


def region_dice(pred_labels: np.ndarray, gt_labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate Dice scores for pancreas and tumor regions.

    Regions:
    - "pancreas": labels 1 or 2 (whole organ including tumor)
    - "tumor": label 2

    Parameters
    ----------
    pred_labels : np.ndarray
        Predicted integer label volume.
    gt_labels : np.ndarray
        Ground-truth integer label volume.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys "pancreas" and "tumor".

    Raises
    ------
    ValueError
        If pred_labels and gt_labels have different shapes.
    """
    if pred_labels.shape != gt_labels.shape:
        raise ValueError(
            f"Shape mismatch: pred_labels {pred_labels.shape} != gt_labels {gt_labels.shape}"
        )

    pred_pancreas = (pred_labels == LABEL_PANCREAS) | (pred_labels == LABEL_TUMOR)
    gt_pancreas = (gt_labels == LABEL_PANCREAS) | (gt_labels == LABEL_TUMOR)

    pred_tumor = pred_labels == LABEL_TUMOR
    gt_tumor = gt_labels == LABEL_TUMOR

    return {
        "pancreas": dice_score(pred_pancreas, gt_pancreas),
        "tumor": dice_score(pred_tumor, gt_tumor),
    }


def mean_over_cases(list_of_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute per-key mean across a list of metric dictionaries.

    Parameters
    ----------
    list_of_dicts : List[Dict[str, float]]
        Non-empty list of dictionaries with metric scores.

    Returns
    -------
    Dict[str, float]
        Dictionary containing per-key mean scores.

    Raises
    ------
    ValueError
        If list_of_dicts is empty.
    """
    if not list_of_dicts:
        raise ValueError("list_of_dicts cannot be empty")

    keys = sorted({k for d in list_of_dicts for k in d.keys()})
    return {
        k: float(sum(d[k] for d in list_of_dicts if k in d) / len(list_of_dicts))
        for k in keys
    }
