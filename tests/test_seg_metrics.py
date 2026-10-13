"""
Unit tests for segmentation metrics.
"""

import numpy as np
import pytest

from src.segmentation.data import LABEL_BACKGROUND, LABEL_PANCREAS, LABEL_TUMOR
from src.segmentation.metrics import dice_score, mean_over_cases, region_dice


def test_dice_score_identical_and_disjoint() -> None:
    """Test dice_score with identical and completely disjoint masks."""
    mask_a = np.zeros((10, 10, 10), dtype=bool)
    mask_a[2:5, 2:5, 2:5] = True

    mask_b = np.zeros((10, 10, 10), dtype=bool)
    mask_b[6:9, 6:9, 6:9] = True

    # Identical non-empty masks -> 1.0
    assert dice_score(mask_a, mask_a) == 1.0

    # Disjoint non-empty masks -> 0.0
    assert dice_score(mask_a, mask_b) == 0.0


def test_dice_score_hand_computed() -> None:
    """Test dice_score against a known hand-computed overlap."""
    mask_a = np.zeros((10, 10, 10), dtype=bool)
    mask_b = np.zeros((10, 10, 10), dtype=bool)

    # mask_a has 20 voxels
    mask_a[0, 0, 0:20 % 10] = True
    mask_a[0, 1, 0:10] = True  # 10 voxels in slice 1, 10 in slice 0 -> 20 voxels
    # Let's set explicit counts:
    mask_a.fill(False)
    mask_a[0, 0, 0:10] = True  # 10 voxels

    mask_b.fill(False)
    mask_b[0, 0, 5:15 % 10] = True
    mask_b[0, 0, 5:10] = True  # 5 overlap voxels
    mask_b[0, 1, 0:5] = True   # 5 non-overlap voxels -> total 10 voxels

    # |A| = 10, |B| = 10, |A & B| = 5
    # Dice = 2 * 5 / (10 + 10) = 10 / 20 = 0.5
    assert dice_score(mask_a, mask_b) == pytest.approx(0.5)


def test_dice_score_empty_masks() -> None:
    """Test dice_score conventions for both-empty and one-empty masks."""
    empty_mask = np.zeros((10, 10, 10), dtype=bool)
    non_empty_mask = np.zeros((10, 10, 10), dtype=bool)
    non_empty_mask[1, 1, 1] = True

    # Both empty -> 1.0
    assert dice_score(empty_mask, empty_mask) == 1.0

    # Exactly one empty -> 0.0
    assert dice_score(empty_mask, non_empty_mask) == 0.0
    assert dice_score(non_empty_mask, empty_mask) == 0.0


def test_dice_score_shape_mismatch_raises() -> None:
    """Test dice_score raises ValueError on shape mismatch."""
    mask_a = np.zeros((10, 10, 10), dtype=bool)
    mask_b = np.zeros((10, 10, 12), dtype=bool)

    with pytest.raises(ValueError, match="Shape mismatch"):
        dice_score(mask_a, mask_b)


def test_region_dice_definitions_and_pancreas_only_prediction() -> None:
    """
    Test region_dice definitions for pancreas (labels 1 or 2) and tumor (label 2).

    A pancreas-only prediction on a case with tumor must score 0 on tumor region.
    """
    shape = (20, 20, 20)
    gt_labels = np.full(shape, LABEL_BACKGROUND, dtype=np.uint8)
    gt_labels[5:10, 5:10, 5:10] = LABEL_PANCREAS
    gt_labels[10:12, 10:12, 10:12] = LABEL_TUMOR

    # Prediction predicts only pancreas (label 1) across the entire GT pancreas+tumor region
    pred_labels = np.full(shape, LABEL_BACKGROUND, dtype=np.uint8)
    pred_labels[5:10, 5:10, 5:10] = LABEL_PANCREAS
    pred_labels[10:12, 10:12, 10:12] = LABEL_PANCREAS  # tumor predicted as pancreas

    scores = region_dice(pred_labels, gt_labels)

    # Whole organ ("pancreas") covers labels 1 and 2, which matches perfectly -> 1.0
    assert scores["pancreas"] == pytest.approx(1.0)
    # Tumor predicted is empty, while GT tumor is non-empty -> tumor score must be 0.0
    assert scores["tumor"] == 0.0


def test_region_dice_shape_mismatch_raises() -> None:
    """Test region_dice raises ValueError on shape mismatch."""
    pred = np.zeros((10, 10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10, 15), dtype=np.uint8)

    with pytest.raises(ValueError, match="Shape mismatch"):
        region_dice(pred, gt)


def test_mean_over_cases() -> None:
    """Test mean_over_cases calculates per-key means and raises on empty input."""
    case_metrics = [
        {"pancreas": 0.8, "tumor": 0.4},
        {"pancreas": 0.9, "tumor": 0.6},
        {"pancreas": 0.7, "tumor": 0.8},
    ]

    means = mean_over_cases(case_metrics)
    assert means["pancreas"] == pytest.approx(0.8)
    assert means["tumor"] == pytest.approx(0.6)

    # Empty list must raise ValueError
    with pytest.raises(ValueError, match="empty"):
        mean_over_cases([])
