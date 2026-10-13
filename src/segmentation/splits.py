import argparse
import csv
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union


def load_dataset_stats(csv_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load dataset statistics from a CSV file.

    Parameters
    ----------
    csv_path : Union[str, Path]
        Path to the dataset statistics CSV file.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries containing parsed case information.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Stats file not found: {csv_path}")

    records: List[Dict[str, Any]] = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "case": row["case"],
                "tumor_diam_mm": float(row["tumor_diam_mm"]),
            })
    return records


def stratify_by_tertiles(
    records: List[Dict[str, Any]]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Partition cases into three tertile strata (small, medium, large) by tumor diameter.

    Parameters
    ----------
    records : List[Dict[str, Any]]
        Dataset records containing at least 'case' and 'tumor_diam_mm'.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        Three lists of case IDs corresponding to small, medium, and large
        tumor diameter tertiles, sorted deterministically before shuffling.
    """
    if len(records) < 3:
        raise ValueError(f"At least 3 records are required to form tertiles, got {len(records)}")

    # Sort deterministically by tumor diameter, breaking ties with case ID
    sorted_records = sorted(
        records,
        key=lambda r: (float(r["tumor_diam_mm"]), str(r["case"])),
    )

    n = len(sorted_records)
    n1 = n // 3 + (1 if n % 3 >= 1 else 0)
    n2 = n // 3 + (1 if n % 3 == 2 else 0)

    small = [r["case"] for r in sorted_records[:n1]]
    medium = [r["case"] for r in sorted_records[n1:n1 + n2]]
    large = [r["case"] for r in sorted_records[n1 + n2:]]

    return small, medium, large


def calculate_split_counts(
    stratum_size: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[int, int, int]:
    """
    Calculate the number of cases to allocate to train, validation, and test splits.

    Parameters
    ----------
    stratum_size : int
        Total number of cases in the stratum.
    train_ratio : float, default=0.70
        Target proportion for the training split.
    val_ratio : float, default=0.15
        Target proportion for the validation split.
    test_ratio : float, default=0.15
        Target proportion for the test split.

    Returns
    -------
    Tuple[int, int, int]
        Counts of cases allocated to (train, val, test) guaranteeing exact total sum.
    """
    n_val = round(stratum_size * val_ratio)
    n_test = round(stratum_size * test_ratio)

    if stratum_size >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
        if stratum_size - n_val - n_test < 1:
            n_val = max(1, (stratum_size - 1) // 2)
            n_test = max(1, stratum_size - 1 - n_val)

    n_train = stratum_size - n_val - n_test
    return n_train, n_val, n_test


def split_strata(
    strata: List[List[str]],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Shuffle each stratum with a fixed seed and allocate cases to train/val/test splits.

    Parameters
    ----------
    strata : List[List[str]]
        List of strata, where each stratum contains case IDs.
    seed : int, default=42
        Random seed for reproducible shuffling.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing 'seed', and sorted lists of case IDs for
        'train', 'val', and 'test'.
    """
    rng = random.Random(seed)
    train_cases: List[str] = []
    val_cases: List[str] = []
    test_cases: List[str] = []

    for stratum in strata:
        shuffled = list(stratum)
        rng.shuffle(shuffled)

        n_train, n_val, n_test = calculate_split_counts(len(shuffled))

        train_cases.extend(shuffled[:n_train])
        val_cases.extend(shuffled[n_train:n_train + n_val])
        test_cases.extend(shuffled[n_train + n_val:])

    train_cases.sort()
    val_cases.sort()
    test_cases.sort()

    return {
        "seed": seed,
        "train": train_cases,
        "val": val_cases,
        "test": test_cases,
    }


def generate_splits(
    stats_path: Union[str, Path],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Load dataset stats from CSV and generate reproducible stratified train/val/test splits.

    Parameters
    ----------
    stats_path : Union[str, Path]
        Path to the dataset stats CSV file.
    seed : int, default=42
        Random seed for reproducible shuffling.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing 'seed', 'train', 'val', and 'test'.
    """
    records = load_dataset_stats(stats_path)
    strata = list(stratify_by_tertiles(records))
    return split_strata(strata, seed=seed)


def save_splits(
    splits_data: Dict[str, Any],
    out_path: Union[str, Path],
) -> None:
    """
    Write dataset splits to a JSON file.

    Parameters
    ----------
    splits_data : Dict[str, Any]
        Dictionary containing the splits data.
    out_path : Union[str, Path]
        Path to the output JSON file.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, mode="w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)
        f.write("\n")


def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line argument parser for dataset split generation.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate stratified train/val/test dataset splits by tumor diameter tertiles."
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="docs/dataset_stats.csv",
        help="Path to dataset stats CSV (default: docs/dataset_stats.csv)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="docs/splits.json",
        help="Path to output splits JSON (default: docs/splits.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser


def main(args: Optional[List[str]] = None) -> None:
    """
    Command-line interface entry point.

    Parameters
    ----------
    args : Optional[List[str]], default=None
        Command-line arguments. If None, sys.argv is parsed.
    """
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    splits = generate_splits(parsed_args.stats, seed=parsed_args.seed)
    save_splits(splits, parsed_args.out)
    print(
        f"Wrote splits to {parsed_args.out} (seed={parsed_args.seed}, "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})"
    )


if __name__ == "__main__":
    main()
