"""
Unit and integration tests for segmentation training, dataset, and resume logic.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
monai = pytest.importorskip("monai")

from src.segmentation.checkpoint import load_checkpoint  # noqa: E402
from src.segmentation.data import LABEL_PANCREAS, LABEL_TUMOR  # noqa: E402
from src.segmentation.train import PatchDataset, main, select_val_cases  # noqa: E402


def _create_synthetic_case(npz_path: Path, shape=(20, 36, 36)) -> None:
    """Helper to create a synthetic preprocessed CT case."""
    rng = np.random.default_rng(123)
    image = rng.uniform(0.1, 0.9, size=shape).astype(np.float32)
    label = np.zeros(shape, dtype=np.uint8)
    label[5:15, 10:25, 10:25] = LABEL_PANCREAS
    label[8:12, 14:18, 14:18] = LABEL_TUMOR
    np.savez(npz_path, image=image.astype(np.float16), label=label)


def test_select_val_cases_evenly_spaced() -> None:
    """Test select_val_cases picks evenly spaced cases across sorted val ids."""
    val_ids = [f"pancreas_{i:03d}" for i in range(10)]

    # 0 or negative returns all
    assert select_val_cases(val_ids, 0) == sorted(val_ids)
    assert select_val_cases(val_ids, -1) == sorted(val_ids)

    # N >= total returns all
    assert select_val_cases(val_ids, 15) == sorted(val_ids)

    # N = 3 picks indices [0, 4, 9]
    assert select_val_cases(val_ids, 3) == ["pancreas_000", "pancreas_004", "pancreas_009"]

    # N = 4 picks indices [0, 3, 6, 9]
    assert select_val_cases(val_ids, 4) == [
        "pancreas_000",
        "pancreas_003",
        "pancreas_006",
        "pancreas_009",
    ]


def test_patch_dataset_shapes_and_dtypes(tmp_path: Path) -> None:
    """Test PatchDataset item shapes, dtypes, and random replacement sampling."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    case_ids = ["case_001", "case_002"]
    for cid in case_ids:
        _create_synthetic_case(data_dir / f"{cid}.npz")

    patch_size = (16, 32, 32)
    patches_per_case = 2
    dataset = PatchDataset(
        case_ids=case_ids,
        data_root=data_dir,
        patches_per_case=patches_per_case,
        patch_size=patch_size,
        seed=42,
    )

    images, labels = dataset[0]

    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert images.shape == (patches_per_case, 1, 16, 32, 32)
    assert labels.shape == (patches_per_case, 16, 32, 32)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_train_main_4_iterations_and_resume(tmp_path: Path) -> None:
    """
    Test main for 4 iterations (val-every 2) creates artifacts, and a second run
    with --max-iterations 8 resumes at iteration 4.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "experiment"

    train_ids = ["train_001", "train_002"]
    val_ids = ["val_001", "val_002"]
    for cid in train_ids + val_ids:
        _create_synthetic_case(data_dir / f"{cid}.npz")

    splits_file = tmp_path / "splits.json"
    splits_data = {
        "seed": 42,
        "train": train_ids,
        "val": val_ids,
        "test": ["test_001"],
    }
    splits_file.write_text(json.dumps(splits_data), encoding="utf-8")

    base_args = [
        "--data-root",
        str(data_dir),
        "--splits",
        str(splits_file),
        "--out-dir",
        str(out_dir),
        "--batch-cases",
        "1",
        "--patches-per-case",
        "2",
        "--num-workers",
        "0",
        "--patch-size",
        "16",
        "32",
        "32",
        "--channels",
        "4",
        "8",
        "16",
        "--strides",
        "2",
        "2",
        "--num-res-units",
        "1",
        "--no-amp",
    ]

    # First run: 4 iterations, val-every 2
    exit_code = main(base_args + ["--max-iterations", "4", "--val-every", "2", "--val-cases", "1"])
    assert exit_code == 0

    assert (out_dir / "last.pt").is_file()
    assert (out_dir / "best.pt").is_file()
    assert (out_dir / "metrics.csv").is_file()
    assert (out_dir / "run_config.json").is_file()

    # Verify run_config.json contains plain types and valid configuration
    with open(out_dir / "run_config.json", mode="r", encoding="utf-8") as f:
        config_data = json.load(f)
    assert config_data["max_iterations"] == 4
    assert isinstance(config_data["data_root"], str)

    ckpt_4 = load_checkpoint(out_dir / "last.pt")
    assert ckpt_4["iteration"] == 4
    assert isinstance(ckpt_4["best_metric"], float)
    assert isinstance(ckpt_4["args"]["data_root"], str)
    assert len(ckpt_4["history"]) == 2

    with open(out_dir / "metrics.csv", mode="r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # Header + 2 iterations (2, 4)
    assert len(rows) == 3
    assert rows[1][0] == "2"
    assert rows[2][0] == "4"

    # Second run: 8 iterations, resumes from 4
    exit_code_resume = main(
        base_args + ["--max-iterations", "8", "--val-every", "2", "--val-cases", "1"]
    )
    assert exit_code_resume == 0

    ckpt_8 = load_checkpoint(out_dir / "last.pt")
    assert ckpt_8["iteration"] == 8
    assert len(ckpt_8["history"]) == 4

    with open(out_dir / "metrics.csv", mode="r", encoding="utf-8") as f:
        rows_after = list(csv.reader(f))
    # Header + 4 iterations (2, 4, 6, 8)
    assert len(rows_after) == 5
    assert [r[0] for r in rows_after[1:]] == ["2", "4", "6", "8"]


def test_train_max_minutes_stops_gracefully_and_saves_last(tmp_path: Path) -> None:
    """Test tiny max-minutes budget terminates gracefully and leaves last.pt."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "timeout_exp"

    train_ids = ["train_001", "train_002"]
    val_ids = ["val_001"]
    for cid in train_ids + val_ids:
        _create_synthetic_case(data_dir / f"{cid}.npz")

    splits_file = tmp_path / "splits.json"
    splits_data = {"seed": 42, "train": train_ids, "val": val_ids, "test": []}
    splits_file.write_text(json.dumps(splits_data), encoding="utf-8")

    args = [
        "--data-root",
        str(data_dir),
        "--splits",
        str(splits_file),
        "--out-dir",
        str(out_dir),
        "--max-iterations",
        "50",
        "--max-minutes",
        "0.00001",
        "--val-every",
        "25",
        "--batch-cases",
        "1",
        "--patches-per-case",
        "1",
        "--num-workers",
        "0",
        "--patch-size",
        "16",
        "32",
        "32",
        "--channels",
        "4",
        "8",
        "16",
        "--strides",
        "2",
        "2",
        "--num-res-units",
        "1",
        "--no-amp",
    ]

    ret = main(args)
    assert ret == 0
    assert (out_dir / "last.pt").is_file()
    ckpt = load_checkpoint(out_dir / "last.pt")
    assert ckpt["iteration"] < 50


def test_training_works_when_test_split_files_missing(tmp_path: Path) -> None:
    """Test that training never attempts to load test split cases."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "no_test_exp"

    train_ids = ["train_001", "train_002"]
    val_ids = ["val_001"]
    for cid in train_ids + val_ids:
        _create_synthetic_case(data_dir / f"{cid}.npz")

    # test case does not exist on disk
    splits_file = tmp_path / "splits.json"
    splits_data = {
        "seed": 42,
        "train": train_ids,
        "val": val_ids,
        "test": ["non_existent_test_case"],
    }
    splits_file.write_text(json.dumps(splits_data), encoding="utf-8")

    args = [
        "--data-root",
        str(data_dir),
        "--splits",
        str(splits_file),
        "--out-dir",
        str(out_dir),
        "--max-iterations",
        "2",
        "--val-every",
        "2",
        "--batch-cases",
        "1",
        "--patches-per-case",
        "1",
        "--num-workers",
        "0",
        "--patch-size",
        "16",
        "32",
        "32",
        "--channels",
        "4",
        "8",
        "16",
        "--strides",
        "2",
        "2",
        "--num-res-units",
        "1",
        "--no-amp",
    ]

    ret = main(args)
    assert ret == 0
    assert (out_dir / "last.pt").is_file()
