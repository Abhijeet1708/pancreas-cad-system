import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.segmentation.splits import (
    load_dataset_stats,
    main,
    split_strata,
    stratify_by_tertiles,
)


def make_synthetic_records(num_cases: int = 60) -> List[Dict[str, Any]]:
    """
    Generate synthetic dataset records for testing.

    Parameters
    ----------
    num_cases : int, default=60
        Number of synthetic records to generate.

    Returns
    -------
    List[Dict[str, Any]]
        Synthetic records with case IDs and increasing tumor diameters.
    """
    return [
        {
            "case": f"synth_{i:03d}",
            "tumor_diam_mm": float(i * 1.5 + 5.0),
        }
        for i in range(num_cases)
    ]


def write_synthetic_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    """
    Write synthetic records to a CSV file.

    Parameters
    ----------
    path : Path
        Destination CSV path.
    records : List[Dict[str, Any]]
        Synthetic records to serialize.
    """
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "case",
            "pancreas_vox",
            "tumor_vox",
            "tumor_ml",
            "tumor_diam_mm",
            "bad_labels",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "case": r["case"],
                "pancreas_vox": 10000,
                "tumor_vox": 2000,
                "tumor_ml": 5.0,
                "tumor_diam_mm": r["tumor_diam_mm"],
                "bad_labels": 0,
            })


def test_no_case_in_multiple_splits():
    """Verify that no case appears in more than one split (pairwise disjoint)."""
    records = make_synthetic_records(60)
    small, medium, large = stratify_by_tertiles(records)
    splits = split_strata([small, medium, large], seed=42)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert train_set.isdisjoint(val_set), "Train and validation splits overlap!"
    assert train_set.isdisjoint(test_set), "Train and test splits overlap!"
    assert val_set.isdisjoint(test_set), "Validation and test splits overlap!"


def test_union_equals_all_cases():
    """Verify that the union of train/val/test equals all input cases with exact count."""
    records = make_synthetic_records(60)
    all_case_ids = {r["case"] for r in records}

    small, medium, large = stratify_by_tertiles(records)
    splits = split_strata([small, medium, large], seed=42)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    combined = train_set | val_set | test_set
    assert combined == all_case_ids, "Union of splits does not match input cases!"

    total_count = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total_count == len(all_case_ids), "Total allocated count does not match input length!"


def test_same_seed_gives_identical_result():
    """Verify that using the same random seed produces identical splits."""
    records = make_synthetic_records(45)
    small, medium, large = stratify_by_tertiles(records)

    splits_a = split_strata([small, medium, large], seed=42)
    splits_b = split_strata([small, medium, large], seed=42)

    assert splits_a == splits_b, "Same seed produced differing split allocations!"


def test_different_seed_gives_different_result():
    """Verify that different random seeds produce different split allocations."""
    records = make_synthetic_records(45)
    small, medium, large = stratify_by_tertiles(records)

    splits_a = split_strata([small, medium, large], seed=42)
    splits_b = split_strata([small, medium, large], seed=999)

    assert splits_a != splits_b, "Different seeds produced identical split allocations!"


def test_each_split_contains_cases_from_every_stratum():
    """Verify that train, val, and test splits each contain cases from every tertile stratum."""
    records = make_synthetic_records(60)
    small, medium, large = stratify_by_tertiles(records)
    strata = [small, medium, large]
    splits = split_strata(strata, seed=42)

    small_set = set(small)
    medium_set = set(medium)
    large_set = set(large)

    for split_name in ("train", "val", "test"):
        split_cases = set(splits[split_name])
        assert not split_cases.isdisjoint(small_set), (
            f"Split '{split_name}' does not contain any cases from the small stratum!"
        )
        assert not split_cases.isdisjoint(medium_set), (
            f"Split '{split_name}' does not contain any cases from the medium stratum!"
        )
        assert not split_cases.isdisjoint(large_set), (
            f"Split '{split_name}' does not contain any cases from the large stratum!"
        )


def test_case_ids_are_sorted_inside_each_split():
    """Verify that case IDs inside each split list are sorted in ascending order."""
    records = make_synthetic_records(30)
    small, medium, large = stratify_by_tertiles(records)
    splits = split_strata([small, medium, large], seed=42)

    for split_name in ("train", "val", "test"):
        case_list = splits[split_name]
        assert case_list == sorted(case_list), f"Cases in split '{split_name}' are not sorted!"


def test_load_dataset_stats_and_tertiles(tmp_path: Path):
    """Verify loading stats from a synthetic CSV and partitioning into tertiles."""
    csv_file = tmp_path / "synthetic_stats.csv"
    records = make_synthetic_records(9)
    write_synthetic_csv(csv_file, records)

    loaded = load_dataset_stats(csv_file)
    assert len(loaded) == 9

    small, medium, large = stratify_by_tertiles(loaded)
    assert len(small) == 3
    assert len(medium) == 3
    assert len(large) == 3


def test_stratify_by_tertiles_error_handling():
    """Verify ValueError is raised when insufficient records are provided."""
    with pytest.raises(ValueError, match="At least 3 records"):
        stratify_by_tertiles([{"case": "c1", "tumor_diam_mm": 5.0}])


def test_cli_execution_with_synthetic_csv(tmp_path: Path):
    """Verify CLI execution end-to-end using temporary synthetic CSV and output paths."""
    stats_csv = tmp_path / "stats.csv"
    out_json = tmp_path / "out" / "splits.json"
    records = make_synthetic_records(30)
    write_synthetic_csv(stats_csv, records)

    main(["--stats", str(stats_csv), "--out", str(out_json), "--seed", "42"])

    assert out_json.is_file(), "Output JSON file was not created by CLI!"

    with open(out_json, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["seed"] == 42
    assert "train" in data
    assert "val" in data
    assert "test" in data
    assert len(data["train"]) + len(data["val"]) + len(data["test"]) == 30
