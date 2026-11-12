"""
Tests for segmentation threshold calibration on validation split.
"""

import csv
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
monai = pytest.importorskip("monai")

from src.segmentation.calibrate import (  # noqa: E402
    main,
    select_recommended_bias,
)
from src.segmentation.checkpoint import save_checkpoint  # noqa: E402
from src.segmentation.data import LABEL_PANCREAS, LABEL_TUMOR  # noqa: E402
from src.segmentation.model import build_unet  # noqa: E402


def _create_synthetic_case(npz_path: Path, shape=(16, 32, 32)) -> None:
    """Helper to create a synthetic preprocessed CT case."""
    rng = np.random.default_rng(123)
    image = rng.uniform(0.1, 0.9, size=shape).astype(np.float32)
    label = np.zeros(shape, dtype=np.uint8)
    label[4:12, 8:24, 8:24] = LABEL_PANCREAS
    label[6:10, 12:18, 12:18] = LABEL_TUMOR
    np.savez(npz_path, image=image.astype(np.float16), label=label)


def _create_tiny_checkpoint(ckpt_path: Path, iteration: int = 42) -> None:
    """Helper to save a tiny UNet checkpoint with args."""
    model = build_unet(channels=(4, 8, 16), strides=(2, 2), num_res_units=1)
    state = {
        "model": model.state_dict(),
        "iteration": iteration,
        "args": {
            "channels": [4, 8, 16],
            "strides": [2, 2],
            "num_res_units": 1,
        },
    }
    save_checkpoint(ckpt_path, state)


def test_select_recommended_bias_rule() -> None:
    """
    Test select_recommended_bias on a hand-made table.

    Tests:
    1. Highest tumor Dice satisfying pancreas Dice >= (baseline - 0.01) is chosen.
    2. Candidate with higher tumor Dice is disqualified if pancreas Dice drops > 0.01.
    3. Ties in tumor Dice are broken by smaller absolute bias.
    4. Warning is emitted if bias 0.0 is missing.
    """
    # Baseline at bias 0.0: pancreas 0.850, tumor 0.500
    # Threshold for pancreas dice: 0.850 - 0.01 = 0.840
    table = [
        # Bias -1.0: pancreas 0.860, tumor 0.400 -> valid, tumor dice 0.400
        {"bias": -1.0, "group": "overall", "mean_pancreas_dice": 0.860, "mean_tumor_dice": 0.400},
        # Bias 0.0: baseline, pancreas 0.850, tumor 0.500 -> valid
        {"bias": 0.0, "group": "overall", "mean_pancreas_dice": 0.850, "mean_tumor_dice": 0.500},
        # Bias 0.5: pancreas 0.845, tumor 0.650 -> valid, higher tumor dice
        {"bias": 0.5, "group": "overall", "mean_pancreas_dice": 0.845, "mean_tumor_dice": 0.650},
        # Bias 1.0: pancreas 0.842, tumor 0.650 -> valid, tied with 0.5 on tumor dice (0.650)
        {"bias": 1.0, "group": "overall", "mean_pancreas_dice": 0.842, "mean_tumor_dice": 0.650},
        # Bias 2.0: pancreas 0.835, tumor 0.750 -> DISQUALIFIED (pancreas 0.835 < 0.840)
        {"bias": 2.0, "group": "overall", "mean_pancreas_dice": 0.835, "mean_tumor_dice": 0.750},
    ]

    best_bias, reason = select_recommended_bias(table)
    # Between 0.5 and 1.0: both have tumor Dice 0.650, but |0.5| < |1.0|, so 0.5 wins
    assert best_bias == 0.5
    assert "0.5" in reason
    assert "0.6500" in reason

    # Test missing bias 0.0 warns
    table_no_zero = [
        {"bias": 1.0, "group": "overall", "mean_pancreas_dice": 0.850, "mean_tumor_dice": 0.600},
        {"bias": 2.0, "group": "overall", "mean_pancreas_dice": 0.845, "mean_tumor_dice": 0.700},
    ]
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        bias_res, _ = select_recommended_bias(table_no_zero)
        assert len(recorded) >= 1
        assert "0.0 is not among" in str(recorded[-1].message)
        assert bias_res == 2.0


def test_calibrate_end_to_end_and_empty_val(tmp_path: Path) -> None:
    """
    Test calibrate end-to-end on 2 tiny cases writes all files with expected keys,
    and fails clearly if splits has no val ids.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "calib_out"

    val_cases = ["case_val_1", "case_val_2"]
    for cid in val_cases:
        _create_synthetic_case(data_dir / f"{cid}.npz")

    splits_file = tmp_path / "splits.json"
    splits_file.write_text(
        json.dumps({"seed": 42, "train": [], "val": val_cases, "test": []}),
        encoding="utf-8",
    )

    stats_file = tmp_path / "stats.csv"
    with open(stats_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "pancreas_vox", "tumor_vox", "tumor_ml", "tumor_diam_mm", "bad"])
        writer.writerow(["case_val_1", "1000", "200", "0.5", "14.0", "0"])  # small
        writer.writerow(["case_val_2", "1500", "500", "1.25", "25.0", "0"])  # large

    ckpt_file = tmp_path / "model.pt"
    _create_tiny_checkpoint(ckpt_file, iteration=99)

    argv = [
        "--data-root",
        str(data_dir),
        "--splits",
        str(splits_file),
        "--stats",
        str(stats_file),
        "--checkpoint",
        str(ckpt_file),
        "--out-dir",
        str(out_dir),
        "--biases",
        "-1,0,1",
        "--patch-size",
        "16",
        "32",
        "32",
        "--device",
        "cpu",
    ]

    exit_code = main(argv)
    assert exit_code == 0

    # 1. calibration.csv validation
    calib_csv = out_dir / "calibration.csv"
    assert calib_csv.is_file()
    with open(calib_csv, mode="r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    # 3 biases * 3 groups = 9 rows
    assert len(csv_rows) == 9
    expected_csv_cols = {
        "bias",
        "group",
        "n",
        "mean_pancreas_dice",
        "mean_tumor_dice",
        "median_tumor_dice",
        "detection_rate",
        "strict_detection_rate",
        "mean_volume_ratio",
    }
    assert expected_csv_cols.issubset(set(csv_rows[0].keys()))

    # 2. per_case_bias.csv validation
    per_case_csv = out_dir / "per_case_bias.csv"
    assert per_case_csv.is_file()
    with open(per_case_csv, mode="r", encoding="utf-8") as f:
        pc_rows = list(csv.DictReader(f))
    # 2 cases * 3 biases = 6 rows
    assert len(pc_rows) == 6
    expected_pc_cols = {
        "case",
        "size_group",
        "tumor_diam_mm",
        "bias",
        "pancreas_dice",
        "tumor_dice",
        "tumor_detected",
        "detected_strict",
        "pred_tumor_ml",
        "true_tumor_ml",
    }
    assert expected_pc_cols.issubset(set(pc_rows[0].keys()))
    assert {r["case"] for r in pc_rows} == {"case_val_1", "case_val_2"}

    # 3. calibration.json validation
    calib_json = out_dir / "calibration.json"
    assert calib_json.is_file()
    with open(calib_json, mode="r", encoding="utf-8") as f:
        calib_data = json.load(f)

    assert calib_data["checkpoint_path"] == "model.pt"
    assert calib_data["checkpoint_iteration"] == 99
    assert calib_data["n_requested"] == 2
    assert calib_data["n_evaluated"] == 2
    assert calib_data["failed_cases"] == []
    assert calib_data["biases"] == [-1.0, 0.0, 1.0]
    assert "selection_rule" in calib_data
    assert "recommended_bias" in calib_data
    assert "recommended_bias_reason" in calib_data
    assert isinstance(calib_data["recommended_bias"], (int, float))

    # 4. calibration.md validation
    calib_md = out_dir / "calibration.md"
    assert calib_md.is_file()
    md_content = calib_md.read_text(encoding="utf-8")
    assert "Tumor Decision-Threshold Calibration" in md_content
    assert "Recommended Bias" in md_content
    assert "Selection Rule" in md_content

    # 5. Fails clearly when splits has no val ids
    splits_empty_val = tmp_path / "splits_empty.json"
    splits_empty_val.write_text(
        json.dumps({"seed": 42, "train": [], "val": [], "test": []}),
        encoding="utf-8",
    )
    argv_empty = [
        "--data-root",
        str(data_dir),
        "--splits",
        str(splits_empty_val),
        "--stats",
        str(stats_file),
        "--checkpoint",
        str(ckpt_file),
        "--out-dir",
        str(tmp_path / "calib_empty"),
        "--device",
        "cpu",
    ]
    with pytest.raises(ValueError, match="No validation cases found"):
        main(argv_empty)


def test_calibrate_ignores_test_ids_without_npz(tmp_path: Path) -> None:
    """
    Test calibrate succeeds when splits lists test ids whose .npz do NOT exist on disk,
    proving test ids are never opened, and outputs contain no test case id anywhere.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "calib_guard_test"

    val_cases = ["case_val_only"]
    for cid in val_cases:
        _create_synthetic_case(data_dir / f"{cid}.npz")

    # Nonexistent test cases
    test_cases = ["case_test_phantom_1", "case_test_phantom_2"]

    splits_file = tmp_path / "splits.json"
    splits_file.write_text(
        json.dumps({"seed": 42, "train": [], "val": val_cases, "test": test_cases}),
        encoding="utf-8",
    )

    stats_file = tmp_path / "stats.csv"
    with open(stats_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "pancreas_vox", "tumor_vox", "tumor_ml", "tumor_diam_mm", "bad"])
        writer.writerow(["case_val_only", "1000", "200", "0.5", "14.0", "0"])

    ckpt_file = tmp_path / "model.pt"
    _create_tiny_checkpoint(ckpt_file)

    argv = [
        "--data-root",
        str(data_dir),
        "--splits",
        str(splits_file),
        "--stats",
        str(stats_file),
        "--checkpoint",
        str(ckpt_file),
        "--out-dir",
        str(out_dir),
        "--biases",
        "0,1",
        "--patch-size",
        "16",
        "32",
        "32",
        "--device",
        "cpu",
    ]

    # Must succeed (test .npz files do not exist, so opening them would crash)
    exit_code = main(argv)
    assert exit_code == 0

    # Verify no test case id is in any output file
    for out_name in ["calibration.csv", "per_case_bias.csv", "calibration.json", "calibration.md"]:
        fpath = out_dir / out_name
        assert fpath.is_file()
        content = fpath.read_text(encoding="utf-8")
        for test_cid in test_cases:
            assert test_cid not in content


def test_calibrate_failing_case_records_error_and_exits_code_1(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """Test failing case is recorded in failed_cases, run completes, and exit code is 1."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "calib_fail"

    # Only create case_good
    _create_synthetic_case(data_dir / "case_good.npz")

    val_cases = ["case_good", "case_missing"]
    splits_file = tmp_path / "splits.json"
    splits_file.write_text(
        json.dumps({"seed": 42, "train": [], "val": val_cases, "test": []}),
        encoding="utf-8",
    )

    stats_file = tmp_path / "stats.csv"
    with open(stats_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "pancreas_vox", "tumor_vox", "tumor_ml", "tumor_diam_mm", "bad"])
        writer.writerow(["case_good", "1000", "200", "0.5", "14.0", "0"])
        writer.writerow(["case_missing", "1000", "200", "0.5", "14.0", "0"])

    ckpt_file = tmp_path / "model.pt"
    _create_tiny_checkpoint(ckpt_file)

    argv = [
        "--data-root",
        str(data_dir),
        "--splits",
        str(splits_file),
        "--stats",
        str(stats_file),
        "--checkpoint",
        str(ckpt_file),
        "--out-dir",
        str(out_dir),
        "--biases",
        "0,1",
        "--patch-size",
        "16",
        "32",
        "32",
        "--device",
        "cpu",
    ]

    exit_code = main(argv)
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "WARNING: 1 case(s) failed during calibration!" in captured.err

    # Check calibration.json has failed_cases recorded
    with open(out_dir / "calibration.json", mode="r", encoding="utf-8") as f:
        calib_data = json.load(f)

    assert calib_data["n_requested"] == 2
    assert calib_data["n_evaluated"] == 1
    assert len(calib_data["failed_cases"]) == 1
    assert calib_data["failed_cases"][0]["case_id"] == "case_missing"
