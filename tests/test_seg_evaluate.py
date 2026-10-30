"""
Unit and integration tests for segmentation model evaluation pipeline.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
monai = pytest.importorskip("monai")

from src.segmentation.checkpoint import save_checkpoint  # noqa: E402
from src.segmentation.data import LABEL_PANCREAS, LABEL_TUMOR  # noqa: E402
from src.segmentation.evaluate import (  # noqa: E402
    assign_size_group,
    bootstrap_ci,
    load_dataset_stats,
    main,
    summarize_group,
    wilson_interval,
)
from src.segmentation.model import build_unet  # noqa: E402


def _create_synthetic_case(npz_path: Path, shape=(20, 36, 36)) -> None:
    """Helper to create a synthetic preprocessed CT case."""
    rng = np.random.default_rng(123)
    image = rng.uniform(0.1, 0.9, size=shape).astype(np.float32)
    label = np.zeros(shape, dtype=np.uint8)
    label[5:15, 10:25, 10:25] = LABEL_PANCREAS
    label[8:12, 14:18, 14:18] = LABEL_TUMOR
    np.savez(npz_path, image=image.astype(np.float16), label=label)


def _create_tiny_checkpoint(ckpt_path: Path, iteration: int = 50) -> None:
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


def test_bootstrap_ci_reproducible_and_brackets_mean() -> None:
    """Test bootstrap_ci is reproducible with same seed and brackets sample mean."""
    rng = np.random.default_rng(999)
    values = rng.normal(loc=0.75, scale=0.1, size=40).tolist()
    mean_val = float(np.mean(values))

    ci_1 = bootstrap_ci(values, n_boot=2000, seed=42)
    ci_2 = bootstrap_ci(values, n_boot=2000, seed=42)

    assert ci_1 == ci_2
    assert ci_1[0] <= mean_val <= ci_1[1]

    # Test single-value edge case
    single_ci = bootstrap_ci([0.88], n_boot=100)
    assert single_ci == (0.88, 0.88)

    # Empty raises ValueError
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_wilson_interval_bounds_and_containment() -> None:
    """Test wilson_interval stays within [0, 1] and contains proportion."""
    # (0, n)
    lower_0, upper_0 = wilson_interval(0, 20)
    assert lower_0 == 0.0
    assert 0.0 < upper_0 <= 1.0
    assert lower_0 <= 0.0 <= upper_0

    # (n, n)
    lower_n, upper_n = wilson_interval(20, 20)
    assert 0.0 <= lower_n < 1.0
    assert upper_n == 1.0
    assert lower_n <= 1.0 <= upper_n

    # (5, 10)
    lower_mid, upper_mid = wilson_interval(5, 10)
    assert 0.0 <= lower_mid <= 0.5 <= upper_mid <= 1.0

    # n <= 0 edge case
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_assign_size_group_threshold_and_unknown() -> None:
    """Test assign_size_group boundary at 20.0 and unknown handling."""
    assert assign_size_group(19.999) == "small"
    assert assign_size_group(20.0) == "large"
    assert assign_size_group(35.2) == "large"

    # NaN, Inf, and None
    assert assign_size_group(None) == "unknown"
    assert assign_size_group(float("nan")) == "unknown"
    assert assign_size_group(float("inf")) == "unknown"


def test_summarize_group_on_known_rows() -> None:
    """Test summarize_group on hand-made case rows."""
    rows = [
        {
            "pancreas_dice": 0.80,
            "tumor_dice": 0.60,
            "tumor_detected": True,
            "detected_strict": True,
        },
        {
            "pancreas_dice": 0.90,
            "tumor_dice": 0.40,
            "tumor_detected": True,
            "detected_strict": True,
        },
        {
            "pancreas_dice": 0.70,
            "tumor_dice": 0.05,
            "tumor_detected": True,
            "detected_strict": False,
        },
    ]
    summary = summarize_group(rows, n_boot=1000, seed=42)

    assert summary["n"] == 3
    assert summary["pancreas_dice"]["mean"] == pytest.approx(0.80)
    assert summary["pancreas_dice"]["median"] == pytest.approx(0.80)
    assert summary["tumor_dice"]["mean"] == pytest.approx(0.35)

    assert summary["tumor_detected"]["count"] == 3
    assert summary["tumor_detected"]["rate"] == pytest.approx(1.0)
    assert summary["detected_strict"]["count"] == 2
    assert summary["detected_strict"]["rate"] == pytest.approx(2.0 / 3.0)

    # Valid JSON without NaN
    json_str = json.dumps(summary, allow_nan=False)
    assert "mean" in json_str


def test_summarize_group_zero_rows_returns_none() -> None:
    """Test summarize_group with zero rows returns n=0 and None for stats."""
    summary = summarize_group([])

    assert summary["n"] == 0
    assert summary["pancreas_dice"]["mean"] is None
    assert summary["pancreas_dice"]["std"] is None
    assert summary["pancreas_dice"]["median"] is None
    assert summary["pancreas_dice"]["ci_95"] is None

    assert summary["tumor_dice"]["mean"] is None
    assert summary["tumor_dice"]["std"] is None
    assert summary["tumor_dice"]["median"] is None
    assert summary["tumor_dice"]["ci_95"] is None

    assert summary["tumor_detected"]["count"] == 0
    assert summary["tumor_detected"]["rate"] is None
    assert summary["tumor_detected"]["ci_95"] is None

    assert summary["detected_strict"]["count"] == 0
    assert summary["detected_strict"]["rate"] is None
    assert summary["detected_strict"]["ci_95"] is None

    # Serialization with allow_nan=False must succeed without error
    dumped = json.dumps(summary, allow_nan=False)
    assert "null" in dumped


def test_load_dataset_stats_missing_file(tmp_path: Path) -> None:
    """Test load_dataset_stats returns empty dict when file does not exist."""
    res = load_dataset_stats(tmp_path / "nonexistent.csv")
    assert res == {}


def test_safety_split_test_without_final(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Test --split test is refused with exit code 2 and out-dir is never created
    when --final is omitted.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "never_created_dir"
    ckpt_path = tmp_path / "dummy.pt"
    splits_path = tmp_path / "splits.json"
    stats_path = tmp_path / "stats.csv"

    argv = [
        "--data-root",
        str(data_dir),
        "--splits",
        str(splits_path),
        "--stats",
        str(stats_path),
        "--checkpoint",
        str(ckpt_path),
        "--out-dir",
        str(out_dir),
        "--split",
        "test",
    ]

    exit_code = main(argv)
    assert exit_code == 2
    assert not out_dir.exists()

    captured = capsys.readouterr()
    assert "--split test requires --final flag" in captured.err


def test_end_to_end_evaluation_success(tmp_path: Path) -> None:
    """
    Test end-to-end evaluation on 2 synthetic cases writes per_case.csv,
    summary.json, summary.md, and overlay figures.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "eval_results"

    case_ids = ["case_001", "case_002"]
    for cid in case_ids:
        _create_synthetic_case(data_dir / f"{cid}.npz")

    splits_file = tmp_path / "splits.json"
    splits_file.write_text(
        json.dumps({"seed": 42, "train": [], "val": case_ids, "test": []}),
        encoding="utf-8",
    )

    stats_file = tmp_path / "stats.csv"
    with open(stats_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "pancreas_vox", "tumor_vox", "tumor_ml", "tumor_diam_mm", "bad"])
        writer.writerow(["case_001", "1000", "200", "0.5", "15.0", "0"])  # small
        writer.writerow(["case_002", "1500", "500", "1.25", "25.0", "0"])  # large

    ckpt_file = tmp_path / "model.pt"
    _create_tiny_checkpoint(ckpt_file, iteration=42)

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
        "--split",
        "val",
        "--patch-size",
        "16",
        "32",
        "32",
        "--num-figure-cases",
        "2",
        "--bootstrap-samples",
        "500",
        "--seed",
        "42",
        "--device",
        "cpu",
    ]

    exit_code = main(argv)
    assert exit_code == 0

    # 1. per_case.csv validation
    csv_file = out_dir / "per_case.csv"
    assert csv_file.is_file()
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 2
    assert {r["case_id"] for r in reader} == {"case_001", "case_002"}
    for r in reader:
        assert float(r["pancreas_dice"]) >= 0.0
        assert float(r["tumor_dice"]) >= 0.0
        assert r["tumor_detected"] in ("True", "False")
        assert r["detected_strict"] in ("True", "False")
        assert float(r["pred_tumor_ml"]) >= 0.0
        assert float(r["gt_tumor_ml"]) > 0.0
        assert r["size_group"] in ("small", "large")

    # 2. summary.json validation
    json_file = out_dir / "summary.json"
    assert json_file.is_file()
    with open(json_file, mode="r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["checkpoint_path"] == str(ckpt_file)
    assert summary["checkpoint_iteration"] == 42
    assert summary["split"] == "val"
    assert summary["seed"] == 42
    assert summary["n_boot"] == 500
    assert isinstance(summary["args"], dict)
    assert summary["n_requested"] == 2
    assert summary["n_evaluated"] == 2
    assert summary["n_unknown_size"] == 0
    assert summary["failed_cases"] == []

    assert summary["overall"]["n"] == 2
    assert summary["small"]["n"] == 1
    assert summary["large"]["n"] == 1

    for group in ("overall", "small", "large"):
        assert "pancreas_dice" in summary[group]
        assert "tumor_dice" in summary[group]
        assert "tumor_detected" in summary[group]
        assert "detected_strict" in summary[group]
        assert len(summary[group]["pancreas_dice"]["ci_95"]) == 2

    # 3. summary.md validation
    md_file = out_dir / "summary.md"
    assert md_file.is_file()
    md_text = md_file.read_text(encoding="utf-8")
    assert "Segmentation Model Evaluation Summary" in md_text
    assert "Overall" in md_text
    assert "Small (<20mm)" in md_text
    assert "Large (>=20mm)" in md_text
    assert "specificity cannot be computed here" in md_text

    # 4. Figure generation validation (if matplotlib was available)
    try:
        import matplotlib  # noqa: F401
        fig_dir = out_dir / "figures"
        assert fig_dir.is_dir()
        pngs = list(fig_dir.glob("*.png"))
        assert len(pngs) > 0
    except ImportError:
        pass


def test_unknown_size_group_reported_and_in_overall_only(tmp_path: Path) -> None:
    """
    Test a case with no stats row receives size group 'unknown', is included
    in 'overall' only, and increments n_unknown_size.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "eval_unknown"

    case_ids = ["case_known", "case_no_stats"]
    for cid in case_ids:
        _create_synthetic_case(data_dir / f"{cid}.npz")

    splits_file = tmp_path / "splits.json"
    splits_file.write_text(
        json.dumps({"seed": 42, "train": [], "val": case_ids, "test": []}),
        encoding="utf-8",
    )

    stats_file = tmp_path / "stats.csv"
    with open(stats_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "pancreas_vox", "tumor_vox", "tumor_ml", "tumor_diam_mm", "bad"])
        # Only case_known is in stats
        writer.writerow(["case_known", "1000", "200", "0.5", "12.0", "0"])

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
        "--split",
        "val",
        "--patch-size",
        "16",
        "32",
        "32",
        "--num-figure-cases",
        "0",
        "--bootstrap-samples",
        "200",
        "--device",
        "cpu",
    ]

    exit_code = main(argv)
    assert exit_code == 0

    with open(out_dir / "summary.json", mode="r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["n_requested"] == 2
    assert summary["n_evaluated"] == 2
    assert summary["n_unknown_size"] == 1
    assert summary["overall"]["n"] == 2
    assert summary["small"]["n"] == 1
    assert summary["large"]["n"] == 0
    assert summary["large"]["pancreas_dice"]["mean"] is None

    with open(out_dir / "per_case.csv", mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows_by_id = {r["case_id"]: r for r in rows}
    assert rows_by_id["case_known"]["size_group"] == "small"
    assert rows_by_id["case_no_stats"]["size_group"] == "unknown"
    assert rows_by_id["case_no_stats"]["tumor_diam_mm"] == ""


def test_failed_case_missing_npz_records_error_and_exits_code_1(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """
    Test when one case in the split has no .npz on disk, it appears in failed_cases,
    the other case is evaluated, all output files are written, and exit code is 1.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "eval_failure"

    # Only create case_good.npz; do NOT create case_missing.npz
    _create_synthetic_case(data_dir / "case_good.npz")

    case_ids = ["case_good", "case_missing"]
    splits_file = tmp_path / "splits.json"
    splits_file.write_text(
        json.dumps({"seed": 42, "train": [], "val": case_ids, "test": []}),
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
        "--split",
        "val",
        "--patch-size",
        "16",
        "32",
        "32",
        "--num-figure-cases",
        "0",
        "--bootstrap-samples",
        "200",
        "--device",
        "cpu",
    ]

    exit_code = main(argv)
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "WARNING: 1 case(s) failed during evaluation!" in captured.err

    # Files must still be written
    assert (out_dir / "per_case.csv").is_file()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "summary.md").is_file()

    with open(out_dir / "summary.json", mode="r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["n_requested"] == 2
    assert summary["n_evaluated"] == 1
    assert len(summary["failed_cases"]) == 1
    assert summary["failed_cases"][0]["case_id"] == "case_missing"
    assert "NPZ file not found" in summary["failed_cases"][0]["error"]
