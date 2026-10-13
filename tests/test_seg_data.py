"""
Unit tests for segmentation data utilities.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.segmentation.data import (
    DEFAULT_PATCH_SIZE,
    LABEL_BACKGROUND,
    LABEL_PANCREAS,
    LABEL_TUMOR,
    augment_intensity,
    load_case,
    load_split_ids,
    sample_patch,
)


def test_load_split_ids_success(tmp_path: Path) -> None:
    """Test loading valid split IDs from JSON file."""
    splits_file = tmp_path / "splits.json"
    data = {
        "seed": 42,
        "train": ["case_001", "case_002"],
        "val": ["case_003"],
        "test": ["case_004", "case_005"],
    }
    splits_file.write_text(json.dumps(data), encoding="utf-8")

    train_cases = load_split_ids(splits_file, "train")
    val_cases = load_split_ids(splits_file, "val")
    test_cases = load_split_ids(splits_file, "test")

    assert train_cases == ["case_001", "case_002"]
    assert val_cases == ["case_003"]
    assert test_cases == ["case_004", "case_005"]


def test_load_split_ids_unknown_split_raises(tmp_path: Path) -> None:
    """Test load_split_ids raises ValueError for unknown splits or metadata keys."""
    splits_file = tmp_path / "splits.json"
    data = {
        "seed": 42,
        "train": ["case_001"],
        "val": ["case_002"],
    }
    splits_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_split_ids(splits_file, "non_existent_split")

    with pytest.raises(ValueError):
        load_split_ids(splits_file, "seed")


def test_load_case_success(tmp_path: Path) -> None:
    """Test load_case converts float16 to float32 and preserves uint8 label."""
    npz_file = tmp_path / "case_001.npz"
    image_f16 = np.full((10, 20, 20), 0.5, dtype=np.float16)
    label_u8 = np.ones((10, 20, 20), dtype=np.uint8)
    label_u8[0, 0, 0] = LABEL_TUMOR
    label_u8[1, 1, 1] = LABEL_BACKGROUND

    np.savez(npz_file, image=image_f16, label=label_u8)

    loaded_img, loaded_lbl = load_case(npz_file)

    assert loaded_img.dtype == np.float32
    assert loaded_lbl.dtype == np.uint8
    assert loaded_img.shape == (10, 20, 20)
    assert loaded_lbl.shape == (10, 20, 20)
    assert np.allclose(loaded_img, 0.5)


def test_load_case_shape_mismatch_raises(tmp_path: Path) -> None:
    """Test load_case raises ValueError when image and label shapes differ."""
    npz_file = tmp_path / "case_mismatch.npz"
    image = np.zeros((10, 20, 20), dtype=np.float16)
    label = np.zeros((10, 20, 15), dtype=np.uint8)
    np.savez(npz_file, image=image, label=label)

    with pytest.raises(ValueError, match="Shape mismatch"):
        load_case(npz_file)


def test_load_case_invalid_label_values_raises(tmp_path: Path) -> None:
    """Test load_case raises ValueError when label contains values outside {0, 1, 2}."""
    npz_file = tmp_path / "case_bad_label.npz"
    image = np.zeros((10, 20, 20), dtype=np.float16)
    label = np.full((10, 20, 20), 3, dtype=np.uint8)
    np.savez(npz_file, image=image, label=label)

    with pytest.raises(ValueError, match="values outside"):
        load_case(npz_file)


def test_sample_patch_validation() -> None:
    """Test sample_patch raises ValueError for invalid inputs."""
    img = np.zeros((30, 40, 50), dtype=np.float32)
    lbl = np.zeros((30, 40, 50), dtype=np.uint8)
    lbl_mismatch = np.zeros((30, 40, 60), dtype=np.uint8)

    # Shape mismatch
    with pytest.raises(ValueError, match="does not match"):
        sample_patch(img, lbl_mismatch, patch_size=(10, 10, 10))

    # class_ratios length != 3
    with pytest.raises(ValueError, match="exactly 3 entries"):
        sample_patch(img, lbl, class_ratios=(1, 1))  # type: ignore

    # class_ratios negative entry
    with pytest.raises(ValueError, match="negative"):
        sample_patch(img, lbl, class_ratios=(-1, 1, 2))

    # class_ratios sum is 0
    with pytest.raises(ValueError, match="cannot be zero"):
        sample_patch(img, lbl, class_ratios=(0, 0, 0))


def test_sample_patch_shape_and_dtypes() -> None:
    """Test sample_patch returns exact requested patch size and preserved dtypes."""
    rng = np.random.default_rng(123)
    img = rng.uniform(0.0, 1.0, size=(80, 150, 150)).astype(np.float32)
    lbl = np.zeros((80, 150, 150), dtype=np.uint8)
    lbl[40:50, 70:80, 70:80] = LABEL_PANCREAS

    patch_img, patch_lbl = sample_patch(
        img, lbl, patch_size=DEFAULT_PATCH_SIZE, rng=rng, class_ratios=(1, 1, 2)
    )

    assert patch_img.shape == DEFAULT_PATCH_SIZE
    assert patch_lbl.shape == DEFAULT_PATCH_SIZE
    assert patch_img.dtype == np.float32
    assert patch_lbl.dtype == np.uint8


def test_sample_patch_tumor_forced() -> None:
    """Test that forcing tumor class ensures patch contains tumor voxels."""
    rng = np.random.default_rng(42)
    img = np.zeros((80, 140, 140), dtype=np.float32)
    lbl = np.zeros((80, 140, 140), dtype=np.uint8)
    # Put a small tumor in a corner region
    lbl[10:12, 15:18, 20:25] = LABEL_TUMOR

    for _ in range(10):
        patch_img, patch_lbl = sample_patch(
            img, lbl, patch_size=(32, 64, 64), rng=rng, class_ratios=(0, 0, 1)
        )
        assert np.any(patch_lbl == LABEL_TUMOR)


def test_sample_patch_reproducibility() -> None:
    """Test same seed produces identical patch and different seed produces different patch."""
    img = np.arange(60 * 60 * 60, dtype=np.float32).reshape(60, 60, 60)
    lbl = np.zeros((60, 60, 60), dtype=np.uint8)
    lbl[20:40, 20:40, 20:40] = LABEL_PANCREAS

    rng1 = np.random.default_rng(999)
    p_img1, p_lbl1 = sample_patch(img, lbl, patch_size=(20, 20, 20), rng=rng1)

    rng2 = np.random.default_rng(999)
    p_img2, p_lbl2 = sample_patch(img, lbl, patch_size=(20, 20, 20), rng=rng2)

    rng3 = np.random.default_rng(1000)
    p_img3, _ = sample_patch(img, lbl, patch_size=(20, 20, 20), rng=rng3)

    assert np.array_equal(p_img1, p_img2)
    assert np.array_equal(p_lbl1, p_lbl2)
    assert not np.array_equal(p_img1, p_img3)


def test_sample_patch_volume_smaller_padded() -> None:
    """Test that a volume smaller than patch_size along an axis is padded with 0 to patch_size."""
    rng = np.random.default_rng(77)
    img = np.ones((20, 30, 40), dtype=np.float32)
    lbl = np.ones((20, 30, 40), dtype=np.uint8)
    target_patch_size = (32, 64, 64)

    patch_img, patch_lbl = sample_patch(img, lbl, patch_size=target_patch_size, rng=rng)

    assert patch_img.shape == target_patch_size
    assert patch_lbl.shape == target_patch_size
    # Padded boundary regions must be 0
    assert patch_img[25, 50, 50] == 0.0
    assert patch_lbl[25, 50, 50] == 0


def test_sample_patch_fallback_uniform() -> None:
    """Test sample_patch falls back to uniform center when target class has no voxels."""
    rng = np.random.default_rng(55)
    img = np.zeros((40, 50, 50), dtype=np.float32)
    lbl = np.zeros((40, 50, 50), dtype=np.uint8)  # No tumor voxels at all

    # Request tumor class only
    patch_img, patch_lbl = sample_patch(
        img, lbl, patch_size=(20, 20, 20), rng=rng, class_ratios=(0, 0, 1)
    )
    assert patch_img.shape == (20, 20, 20)
    assert patch_lbl.shape == (20, 20, 20)


def test_augment_intensity_bounds_immutability_and_dtype() -> None:
    """Test augment_intensity preserves input array, clamps to [0, 1], and preserves dtype."""
    rng = np.random.default_rng(42)
    image = np.full((16, 32, 32), 0.5, dtype=np.float32)
    original_copy = image.copy()

    augmented = augment_intensity(image, rng=rng, gaussian_noise=True, noise_std=0.01)

    # Input must be unmodified
    assert np.array_equal(image, original_copy)
    # Output must match dtype
    assert augmented.dtype == np.float32
    # Output must be clipped to [0, 1]
    assert np.all(augmented >= 0.0)
    assert np.all(augmented <= 1.0)
    # Augmented values should differ from constant 0.5
    assert not np.array_equal(augmented, image)

    # Without gaussian noise
    augmented_no_noise = augment_intensity(image, rng=rng, gaussian_noise=False)
    assert augmented_no_noise.dtype == np.float32
    assert np.all(augmented_no_noise >= 0.0)
    assert np.all(augmented_no_noise <= 1.0)
