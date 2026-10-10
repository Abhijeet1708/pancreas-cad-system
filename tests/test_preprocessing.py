import numpy as np
import pytest
import SimpleITK as sitk

from src.preprocessing.preprocess import (
    get_case_id,
    discover_files,
    resample_image,
    process_case
)


def test_get_case_id():
    assert get_case_id("case_001.nii") == "case_001"
    assert get_case_id("case_002.nii.gz") == "case_002"
    assert get_case_id("pancreas_050.nii.gz") == "pancreas_050"


def test_discover_files(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    # Create dummy files
    (images_dir / "case_001.nii.gz").touch()
    (labels_dir / "case_001.nii.gz").touch()

    (images_dir / "case_002.nii").touch()
    (labels_dir / "case_002.nii").touch()

    (images_dir / "case_003_no_label.nii.gz").touch()

    # Ignored files
    (images_dir / "._case_004.nii.gz").touch()
    (labels_dir / "._case_004.nii.gz").touch()

    cases = discover_files(tmp_path)

    assert "case_001" in cases
    assert "image" in cases["case_001"]
    assert "label" in cases["case_001"]

    assert "case_002" in cases
    assert "image" in cases["case_002"]
    assert "label" in cases["case_002"]

    assert "case_003_no_label" in cases
    assert "image" in cases["case_003_no_label"]
    assert "label" not in cases["case_003_no_label"]

    assert "._case_004" not in cases
    assert "case_004" not in cases


def test_resample_image():
    # Create synthetic array: z=5, y=10, x=10
    arr = np.zeros((5, 10, 10), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    # Original spacing: 2.0, 2.0, 5.0
    img.SetSpacing((2.0, 2.0, 5.0))
    img.SetOrigin((10.0, 20.0, 30.0))

    target_spacing = (1.0, 1.0, 2.5)
    res_img = resample_image(img, target_spacing=target_spacing, is_label=False)

    assert res_img.GetSpacing() == target_spacing
    assert res_img.GetOrigin() == (10.0, 20.0, 30.0)
    # Target size should be 2x original size due to half spacing
    # Note: sitk size is (x, y, z)
    assert res_img.GetSize() == (20, 20, 10)


def test_process_case(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    out_dir = tmp_path / "processed"
    images_dir.mkdir()
    labels_dir.mkdir()
    out_dir.mkdir()

    # Synthetic image with specific values to test clipping and normalization
    # -150 should clip to -100 and map to 0.0
    # -100 should map to 0.0
    # 70 should map to (70+100)/340 = 170/340 = 0.5
    # 240 should map to 1.0
    # 300 should clip to 240 and map to 1.0
    img_arr = np.array([[-150, -100], [70, 240], [300, 0]], dtype=np.float32)
    # Shape for sitk is (z, y, x). Here it's 2D but that's fine for testing.
    # We will expand it to 3D to match real-world
    img_arr = img_arr[np.newaxis, ...]

    lbl_arr = np.array([[0, 1], [2, 0], [1, 2]], dtype=np.uint8)[np.newaxis, ...]

    img = sitk.GetImageFromArray(img_arr)
    lbl = sitk.GetImageFromArray(lbl_arr)

    # Needs a 3D spacing to avoid resampling failure if expecting 3 values
    img.SetSpacing((1.0, 1.0, 2.5))
    lbl.SetSpacing((1.0, 1.0, 2.5))

    img_path = images_dir / "case_test.nii.gz"
    lbl_path = labels_dir / "case_test.nii.gz"

    sitk.WriteImage(img, str(img_path))
    sitk.WriteImage(lbl, str(lbl_path))

    process_case(img_path, lbl_path, out_dir, "case_test")

    out_file = out_dir / "case_test.npz"
    assert out_file.exists()

    # Check no tmp files are left behind
    assert len(list(out_dir.glob("*.tmp.npz"))) == 0

    data = np.load(out_file)
    assert "image" in data
    assert "label" in data

    proc_img = data["image"]
    proc_lbl = data["label"]

    assert proc_img.dtype == np.float16
    assert proc_lbl.dtype == np.uint8

    # The output array shape should match input because spacing didn't change
    assert proc_img.shape == proc_lbl.shape
    assert proc_lbl.shape == lbl_arr.shape

    # Check label values preserved
    np.testing.assert_array_equal(proc_lbl, lbl_arr)

    # Check specific mapped values
    # Original: [-150, -100], [70, 240], [300, 0]
    expected_img_vals = [[0.0, 0.0], [0.5, 1.0], [1.0, 100 / 340.0]]
    expected_img = np.array(expected_img_vals, dtype=np.float16)[np.newaxis, ...]
    np.testing.assert_allclose(proc_img, expected_img, rtol=1e-3)


def test_process_case_shape_mismatch(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    out_dir = tmp_path / "processed"
    images_dir.mkdir()
    labels_dir.mkdir()
    out_dir.mkdir()

    img_arr = np.zeros((5, 10, 10), dtype=np.float32)
    lbl_arr = np.zeros((5, 10, 11), dtype=np.uint8)  # Different shape

    img = sitk.GetImageFromArray(img_arr)
    lbl = sitk.GetImageFromArray(lbl_arr)

    img.SetSpacing((1.0, 1.0, 2.5))
    lbl.SetSpacing((1.0, 1.0, 2.5))

    img_path = images_dir / "case_test_mismatch.nii.gz"
    lbl_path = labels_dir / "case_test_mismatch.nii.gz"

    sitk.WriteImage(img, str(img_path))
    sitk.WriteImage(lbl, str(lbl_path))

    with pytest.raises(ValueError, match="Shape mismatch after resampling"):
        process_case(img_path, lbl_path, out_dir, "case_test_mismatch")

    # Assert no final .npz or .tmp.npz left on failure (wait, exception is raised before saving)
    assert not (out_dir / "case_test_mismatch.npz").exists()
    assert not (out_dir / "case_test_mismatch.tmp.npz").exists()
