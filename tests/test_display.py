"""
Unit tests for pipeline display and formatting helpers.

Dependencies: pure numpy and Python standard library only (no torch, matplotlib, or streamlit).
Uses synthetic data only; never reads from or modifies docs/.
"""

import json
from pathlib import Path

import numpy as np

from src.pipeline.display import (
    DISCLAIMER,
    LIMITATIONS_MD,
    format_finding_messages,
    list_example_results,
    summary_rows,
    tint_overlay,
)


def test_list_example_results_valid_and_missing(tmp_path: Path):
    # Case 1: Missing root directory should return []
    missing_dir = tmp_path / "does_not_exist"
    assert list_example_results(missing_dir) == []

    # Create synthetic root
    root = tmp_path / "example_results"
    root.mkdir()

    # Case 2: Subfolder without result.json should be skipped
    empty_sub = root / "case_skipped"
    empty_sub.mkdir()
    (empty_sub / "some_file.txt").write_text("hello", encoding="utf-8")

    # Case 3: Valid subfolder with result.json and figures/
    valid_sub1 = root / "case_001"
    valid_sub1.mkdir()
    res_data1 = {
        "pancreas_volume_ml": 45.2,
        "tumor_volume_ml": 2.1,
        "segmentation_tumor_found": True,
        "n_slices_evaluated": 12,
        "n_slices_flagged": 3,
        "max_slice_probability": 0.85,
        "seconds": 15.4,
        "device": "cpu",
    }
    (valid_sub1 / "result.json").write_text(json.dumps(res_data1), encoding="utf-8")
    fig_dir = valid_sub1 / "figures"
    fig_dir.mkdir()
    (fig_dir / "slice_010.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (fig_dir / "slice_005.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # Case 4: Valid subfolder with figures directly in folder
    valid_sub2 = root / "case_002"
    valid_sub2.mkdir()
    res_data2 = {"pancreas_volume_ml": 38.0, "tumor_volume_ml": 0.0}
    (valid_sub2 / "result.json").write_text(json.dumps(res_data2), encoding="utf-8")
    (valid_sub2 / "slice_001.jpg").write_bytes(b"\xff\xd8\xff")

    results = list_example_results(root)

    # empty_sub is skipped, so results has length 2
    assert len(results) == 2
    names = [r["name"] for r in results]
    assert names == ["case_001", "case_002"]

    # Verify figures are sorted
    case1_entry = results[0]
    assert case1_entry["name"] == "case_001"
    assert case1_entry["result"]["pancreas_volume_ml"] == 45.2
    assert len(case1_entry["figures"]) == 2
    assert case1_entry["figures"][0].name == "slice_005.png"
    assert case1_entry["figures"][1].name == "slice_010.png"

    case2_entry = results[1]
    assert len(case2_entry["figures"]) == 1
    assert case2_entry["figures"][0].name == "slice_001.jpg"


def test_summary_rows_keys_and_formatting():
    # Test with complete result dictionary
    sample_result = {
        "pancreas_volume_ml": 50.1234,
        "tumor_volume_ml": 3.4567,
        "segmentation_tumor_found": True,
        "n_slices_evaluated": 20,
        "n_slices_flagged": 4,
        "max_slice_probability": 0.91234,
        "seconds": 23.456,
        "device": "cuda:0",
    }

    rows = summary_rows(sample_result)
    assert isinstance(rows, list)
    assert len(rows) == 8

    row_dict = {label.lower(): val for label, val in rows}

    # Verify each expected metric is represented
    assert "pancreas volume" in row_dict
    assert row_dict["pancreas volume"] == "50.12 mL"

    assert "tumor volume" in row_dict
    assert row_dict["tumor volume"] == "3.46 mL"

    assert "segmentation finding" in row_dict
    assert row_dict["segmentation finding"] == "Found"

    assert "slices evaluated" in row_dict
    assert row_dict["slices evaluated"] == "20"

    assert "slices flagged" in row_dict
    assert row_dict["slices flagged"] == "4"

    assert "max slice probability" in row_dict
    assert row_dict["max slice probability"] == "0.9123"

    assert "runtime seconds" in row_dict
    assert row_dict["runtime seconds"] == "23.46 s"

    assert "device" in row_dict
    assert row_dict["device"] == "cuda:0"

    # Test with empty/None result dictionary (graceful fallback)
    empty_rows = summary_rows({})
    empty_dict = {label.lower(): val for label, val in empty_rows}
    assert empty_dict["pancreas volume"] == "N/A"
    assert empty_dict["tumor volume"] == "N/A"
    assert empty_dict["segmentation finding"] == "N/A"
    assert empty_dict["slices evaluated"] == "N/A"
    assert empty_dict["slices flagged"] == "N/A"
    assert empty_dict["max slice probability"] == "N/A"
    assert empty_dict["runtime seconds"] == "N/A"
    assert empty_dict["device"] == "unknown"


def test_tint_overlay_shape_dtype_and_colors():
    # Create 3x3 synthetic image and labels
    # Grayscale values in [0.0, 1.0]
    img = np.zeros((3, 3), dtype=np.float32)
    # Background pixel with intensity 0.5 -> gray level 128 (approx)
    img[0, 0] = 0.5
    # Pancreas pixel with intensity 0.0 -> black base
    img[1, 1] = 0.0
    # Tumor pixel with intensity 0.0 -> black base
    img[2, 2] = 0.0

    lbl = np.zeros((3, 3), dtype=np.uint8)
    lbl[0, 0] = 0  # Background
    lbl[1, 1] = 1  # Pancreas (Yellow: [255, 255, 0])
    lbl[2, 2] = 2  # Tumor (Red: [255, 0, 0])

    alpha = 0.4
    overlay = tint_overlay(img, lbl, alpha=alpha)

    # 1. Shape and dtype
    assert overlay.shape == (3, 3, 3)
    assert overlay.dtype == np.uint8

    # 2. Pixel (0, 0) background: gray level round(0.5 * 255) = 128
    bg_pixel = overlay[0, 0]
    expected_gray = int(round(0.5 * 255.0))
    assert np.all(bg_pixel == expected_gray)

    # 3. Pixel (1, 1) pancreas: base 0 blended with Blue [0, 0, 255]
    # R = round((1 - 0.4)*0 + 0.4*0) = 0
    # G = round((1 - 0.4)*0 + 0.4*0) = 0
    # B = round((1 - 0.4)*0 + 0.4*255) = 102
    p_pixel = overlay[1, 1]
    expected_p = np.array([0, 0, 102], dtype=np.uint8)
    assert np.array_equal(p_pixel, expected_p)

    # 4. Pixel (2, 2) tumor: base 0 blended with Red [255, 0, 0]
    # R = round((1 - 0.4)*0 + 0.4*255) = 102
    # G = round((1 - 0.4)*0 + 0.4*0) = 0
    # B = round((1 - 0.4)*0 + 0.4*0) = 0
    t_pixel = overlay[2, 2]
    expected_t = np.array([102, 0, 0], dtype=np.uint8)
    assert np.array_equal(t_pixel, expected_t)

    # Check uint8 image input support
    img_uint8 = (img * 255).astype(np.uint8)
    overlay_uint8 = tint_overlay(img_uint8, lbl, alpha=alpha)
    assert overlay_uint8.shape == (3, 3, 3)
    assert overlay_uint8.dtype == np.uint8
    assert np.array_equal(overlay_uint8[1, 1], expected_p)


def test_format_finding_messages_distinct_and_no_diagnosis():
    # Test positive case
    pos_result = {
        "segmentation_tumor_found": True,
        "tumor_volume_ml": 4.5,
        "pancreas_volume_ml": 62.0,
        "n_slices_evaluated": 15,
        "n_slices_flagged": 3,
        "max_slice_probability": 0.78,
        "classifier_threshold": 0.5,
    }
    seg_msg, cls_msg = format_finding_messages(pos_result)

    assert isinstance(seg_msg, str) and len(seg_msg) > 0
    assert isinstance(cls_msg, str) and len(cls_msg) > 0
    # Messages must be distinct
    assert seg_msg != cls_msg

    # Crucial rule: Neither message must contain the word "diagnosis" (case-insensitive)
    assert "diagnosis" not in seg_msg.lower()
    assert "diagnosis" not in cls_msg.lower()

    # Test negative case with no evaluated slices
    neg_result = {
        "segmentation_tumor_found": False,
        "tumor_volume_ml": 0.0,
        "pancreas_volume_ml": 55.0,
        "n_slices_evaluated": 0,
        "n_slices_flagged": 0,
        "max_slice_probability": None,
        "classifier_threshold": 0.5,
    }
    seg_neg, cls_neg = format_finding_messages(neg_result)
    assert isinstance(seg_neg, str) and len(seg_neg) > 0
    assert isinstance(cls_neg, str) and len(cls_neg) > 0
    assert seg_neg != cls_neg
    assert "diagnosis" not in seg_neg.lower()
    assert "diagnosis" not in cls_neg.lower()


def test_constants_and_limitations_content():
    # Verify DISCLAIMER and LIMITATIONS_MD content
    assert "Research prototype" in DISCLAIMER
    assert "not for clinical use" in DISCLAIMER
    assert "0.762" in LIMITATIONS_MD
    assert "0.323" in LIMITATIONS_MD
    assert "0.666" in LIMITATIONS_MD
    assert "specificity is unknown" in LIMITATIONS_MD
    assert "no external validation" in LIMITATIONS_MD


def test_streamlit_app_compiles_without_import():
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    assert app_path.is_file(), f"File not found: {app_path}"

    source = app_path.read_text(encoding="utf-8")
    # Verify app/streamlit_app.py compiles cleanly as valid Python syntax
    code = compile(source, str(app_path), "exec")
    assert code is not None
