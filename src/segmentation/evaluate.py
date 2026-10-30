"""
Evaluation pipeline for 3D pancreas and tumor segmentation models.

Evaluates segmentation predictions against ground truth for validation and test splits,
computes regional Dice scores (pancreas, tumor), tumor detection metrics, tumor volumes,
and aggregates bootstrap confidence intervals across tumor size subgroups.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

from src.segmentation.checkpoint import load_checkpoint
from src.segmentation.data import (
    DEFAULT_PATCH_SIZE,
    LABEL_PANCREAS,
    LABEL_TUMOR,
    load_case,
    load_split_ids,
)
from src.segmentation.metrics import region_dice
from src.segmentation.model import build_unet, predict_volume

VOXEL_VOLUME_ML: float = (1.0 * 1.0 * 2.5) / 1000.0  # 1.0 x 1.0 x 2.5 mm -> 0.0025 mL


def assign_size_group(diam_mm: Optional[float]) -> str:
    """
    Classify tumor diameter into size categories.

    Parameters
    ----------
    diam_mm : Optional[float]
        Tumor maximum diameter in millimeters.

    Returns
    -------
    str
        "small" if diam_mm < 20.0, "large" if diam_mm >= 20.0, or "unknown" if None/NaN.
    """
    if diam_mm is None:
        return "unknown"
    if isinstance(diam_mm, (float, np.floating)) and (math.isnan(diam_mm) or math.isinf(diam_mm)):
        return "unknown"
    if diam_mm < 20.0:
        return "small"
    return "large"


def wilson_interval(
    k: int,
    n: int,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:
    """
    Compute the Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of positive detections or successes.
    n : int
        Total number of trials.
    z : float, default=1.959963984540054
        Normal distribution quantile (default ~1.96 for two-sided 95% CI).

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound) clipped to [0.0, 1.0].
    """
    if n <= 0:
        return 0.0, 0.0

    p = float(k) / float(n)
    denominator = 1.0 + (z * z) / float(n)
    center = (p + (z * z) / (2.0 * float(n))) / denominator
    radicand = (p * (1.0 - p) / float(n)) + ((z * z) / (4.0 * float(n * n)))
    margin = (z * math.sqrt(max(0.0, radicand))) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute a two-sided percentile-bootstrap confidence interval of the sample mean.

    Parameters
    ----------
    values : Sequence[float]
        1D sequence of observed metric values.
    n_boot : int, default=10000
        Number of bootstrap resamples.
    seed : int, default=42
        Seed for reproducible random sampling.
    alpha : float, default=0.05
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound) of the bootstrap percentile interval.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        raise ValueError("values cannot be empty for bootstrap_ci")
    if len(arr) == 1:
        val = float(arr[0])
        return val, val

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = np.mean(arr[indices], axis=1)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))
    return ci_lower, ci_upper


def summarize_group(
    rows: Sequence[Dict[str, Any]],
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Aggregate evaluation metrics for a group of case records.

    Computes n, mean, std, median, and 95% bootstrap CI of the mean for pancreas
    and tumor Dice, along with counts, rates, and 95% Wilson score intervals for
    both tumor detection metrics (overlap and strict Dice >= 0.10).

    When rows is empty, returns n = 0 and None for every statistic.

    Parameters
    ----------
    rows : Sequence[Dict[str, Any]]
        Case result records.
    n_boot : int, default=10000
        Bootstrap samples for Dice mean CIs.
    seed : int, default=42
        Seed for bootstrap sampling.

    Returns
    -------
    Dict[str, Any]
        Dictionary with aggregated summary metrics.
    """
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "pancreas_dice": {
                "mean": None,
                "std": None,
                "median": None,
                "ci_95": None,
            },
            "tumor_dice": {
                "mean": None,
                "std": None,
                "median": None,
                "ci_95": None,
            },
            "tumor_detected": {
                "count": 0,
                "rate": None,
                "ci_95": None,
            },
            "detected_strict": {
                "count": 0,
                "rate": None,
                "ci_95": None,
            },
        }

    pancreas_vals = [float(r["pancreas_dice"]) for r in rows]
    tumor_vals = [float(r["tumor_dice"]) for r in rows]

    p_mean = float(np.mean(pancreas_vals))
    p_std = float(np.std(pancreas_vals, ddof=1)) if n > 1 else 0.0
    p_median = float(np.median(pancreas_vals))
    p_ci = list(bootstrap_ci(pancreas_vals, n_boot=n_boot, seed=seed))

    t_mean = float(np.mean(tumor_vals))
    t_std = float(np.std(tumor_vals, ddof=1)) if n > 1 else 0.0
    t_median = float(np.median(tumor_vals))
    t_ci = list(bootstrap_ci(tumor_vals, n_boot=n_boot, seed=seed))

    det_count = sum(1 for r in rows if r["tumor_detected"])
    det_rate = float(det_count / n)
    det_ci = list(wilson_interval(det_count, n))

    strict_count = sum(1 for r in rows if r["detected_strict"])
    strict_rate = float(strict_count / n)
    strict_ci = list(wilson_interval(strict_count, n))

    return {
        "n": n,
        "pancreas_dice": {
            "mean": p_mean,
            "std": p_std,
            "median": p_median,
            "ci_95": p_ci,
        },
        "tumor_dice": {
            "mean": t_mean,
            "std": t_std,
            "median": t_median,
            "ci_95": t_ci,
        },
        "tumor_detected": {
            "count": det_count,
            "rate": det_rate,
            "ci_95": det_ci,
        },
        "detected_strict": {
            "count": strict_count,
            "rate": strict_rate,
            "ci_95": strict_ci,
        },
    }


def load_dataset_stats(stats_path: Union[str, Path]) -> Dict[str, float]:
    """
    Load mapping from case identifier to tumor diameter in mm from dataset stats CSV.

    Parameters
    ----------
    stats_path : Union[str, Path]
        Path to dataset stats CSV file.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping case IDs to tumor_diam_mm values.
    """
    path = Path(stats_path)
    if not path.is_file():
        return {}

    stats: Dict[str, float] = {}
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row.get("case", "").strip()
            diam_str = row.get("tumor_diam_mm", "").strip()
            if case_id and diam_str:
                try:
                    diam = float(diam_str)
                    stats[case_id] = diam
                    if case_id.endswith(".npz"):
                        stats[case_id[:-4]] = diam
                    else:
                        stats[f"{case_id}.npz"] = diam
                except ValueError:
                    continue
    return stats


def select_figure_cases(
    evaluated_rows: Sequence[Dict[str, Any]],
    num_cases: int,
) -> List[str]:
    """
    Select cases for overlay figure generation spread across the split.

    Ensures inclusion of at least one small (<20mm) and one large (>=20mm) tumor
    if available.

    Parameters
    ----------
    evaluated_rows : Sequence[Dict[str, Any]]
        List of successfully evaluated case records.
    num_cases : int
        Target number of cases to select.

    Returns
    -------
    List[str]
        List of selected case IDs.
    """
    if num_cases <= 0 or not evaluated_rows:
        return []

    small_cases = [r["case_id"] for r in evaluated_rows if r["size_group"] == "small"]
    large_cases = [r["case_id"] for r in evaluated_rows if r["size_group"] == "large"]
    all_cases = [r["case_id"] for r in evaluated_rows]

    selected: List[str] = []
    chosen_set: Set[str] = set()

    if small_cases:
        c = small_cases[0]
        selected.append(c)
        chosen_set.add(c)

    if large_cases and len(selected) < num_cases:
        c = large_cases[0]
        if c not in chosen_set:
            selected.append(c)
            chosen_set.add(c)

    remaining_needed = num_cases - len(selected)
    if remaining_needed > 0:
        candidate_cases = [c for c in all_cases if c not in chosen_set]
        if candidate_cases:
            if remaining_needed >= len(candidate_cases):
                selected.extend(candidate_cases)
            else:
                indices = [
                    int(round(idx))
                    for idx in np.linspace(0, len(candidate_cases) - 1, remaining_needed)
                ]
                for idx in indices:
                    c = candidate_cases[idx]
                    if c not in chosen_set:
                        selected.append(c)
                        chosen_set.add(c)

    return selected[:num_cases]


def save_overlay_figure(
    image: np.ndarray,
    gt_label: np.ndarray,
    pred_label: np.ndarray,
    out_path: Union[str, Path],
    case_id: str,
) -> None:
    """
    Save PNG overlay comparing ground truth and prediction on the axial slice
    with the largest ground-truth tumor area.

    Parameters
    ----------
    image : np.ndarray
        3D CT image volume of shape (Z, Y, X).
    gt_label : np.ndarray
        3D ground-truth integer mask of shape (Z, Y, X).
    pred_label : np.ndarray
        3D predicted integer mask of shape (Z, Y, X).
    out_path : Union[str, Path]
        Output path for the generated PNG.
    case_id : str
        Case identifier for the plot title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure generation.")
        return

    tumor_mask = gt_label == LABEL_TUMOR
    slice_areas = np.sum(tumor_mask, axis=(1, 2))
    if np.any(slice_areas > 0):
        best_z = int(np.argmax(slice_areas))
    else:
        best_z = int(gt_label.shape[0] // 2)

    img_slice = image[best_z]
    gt_slice = gt_label[best_z]
    pred_slice = pred_label[best_z]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Ground truth panel
    axes[0].imshow(img_slice, cmap="gray")
    gt_pancreas = np.ma.masked_where(gt_slice != LABEL_PANCREAS, gt_slice)
    gt_tumor_m = np.ma.masked_where(gt_slice != LABEL_TUMOR, gt_slice)
    axes[0].imshow(gt_pancreas, cmap="winter", alpha=0.45)
    axes[0].imshow(gt_tumor_m, cmap="autumn", alpha=0.6)
    axes[0].set_title(f"{case_id} (Slice {best_z}) - Ground Truth")
    axes[0].axis("off")

    # Prediction panel
    axes[1].imshow(img_slice, cmap="gray")
    pred_pancreas = np.ma.masked_where(pred_slice != LABEL_PANCREAS, pred_slice)
    pred_tumor_m = np.ma.masked_where(pred_slice != LABEL_TUMOR, pred_slice)
    axes[1].imshow(pred_pancreas, cmap="winter", alpha=0.45)
    axes[1].imshow(pred_tumor_m, cmap="autumn", alpha=0.6)
    axes[1].set_title(f"{case_id} (Slice {best_z}) - Prediction")
    axes[1].axis("off")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_markdown_summary(summary_data: Dict[str, Any]) -> str:
    """
    Format evaluation summary metrics into a readable Markdown table.

    Parameters
    ----------
    summary_data : Dict[str, Any]
        Summary metrics dictionary.

    Returns
    -------
    str
        Markdown string.
    """
    lines = [
        "# Segmentation Model Evaluation Summary",
        "",
        f"- **Checkpoint**: `{summary_data.get('checkpoint_path', 'N/A')}`",
        f"- **Iteration**: {summary_data.get('checkpoint_iteration', 'N/A')}",
        f"- **Split**: {summary_data.get('split', 'N/A')}",
        f"- **Requested Cases**: {summary_data.get('n_requested', 0)}",
        f"- **Evaluated Cases**: {summary_data.get('n_evaluated', 0)}",
        f"- **Unknown Size Cases**: {summary_data.get('n_unknown_size', 0)}",
        f"- **Failed Cases**: {len(summary_data.get('failed_cases', []))}",
        "",
        "## Performance by Tumor Size Group",
        "",
        "| Group | N | Pancreas Dice (Mean ± SD [95% CI] / Median) | "
        "Tumor Dice (Mean ± SD [95% CI] / Median) | Tumor Detected (Overlap) | "
        "Strict Detected (Dice >= 0.10) |",
        "|---|---|---|---|---|---|",
    ]

    for grp_key, grp_label in [
        ("overall", "Overall"),
        ("small", "Small (<20mm)"),
        ("large", "Large (>=20mm)"),
    ]:
        grp = summary_data.get(grp_key, {})
        n = grp.get("n", 0)
        if n == 0:
            lines.append(f"| {grp_label} | 0 | N/A | N/A | N/A | N/A |")
            continue

        pd = grp.get("pancreas_dice", {})
        td = grp.get("tumor_dice", {})
        det = grp.get("tumor_detected", {})
        st = grp.get("detected_strict", {})

        p_str = (
            f"{pd['mean']:.3f} ± {pd['std']:.3f} "
            f"[{pd['ci_95'][0]:.3f}, {pd['ci_95'][1]:.3f}] (med {pd['median']:.3f})"
        )
        t_str = (
            f"{td['mean']:.3f} ± {td['std']:.3f} "
            f"[{td['ci_95'][0]:.3f}, {td['ci_95'][1]:.3f}] (med {td['median']:.3f})"
        )
        det_str = (
            f"{det['rate'] * 100:.1f}% ({det['count']}/{n}) "
            f"[{det['ci_95'][0] * 100:.1f}%, {det['ci_95'][1] * 100:.1f}%]"
        )
        st_str = (
            f"{st['rate'] * 100:.1f}% ({st['count']}/{n}) "
            f"[{st['ci_95'][0] * 100:.1f}%, {st['ci_95'][1] * 100:.1f}%]"
        )

        lines.append(f"| {grp_label} | {n} | {p_str} | {t_str} | {det_str} | {st_str} |")

    lines.extend([
        "",
        "> Note: Every case in this dataset contains a tumor, so specificity "
        "cannot be computed here.",
        "",
    ])
    return "\n".join(lines)


def sanitize_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse namespace to plain Python types."""
    cleaned: Dict[str, Any] = {}
    for key, val in vars(args).items():
        if isinstance(val, Path):
            cleaned[key] = str(val)
        elif isinstance(val, (list, tuple)):
            cleaned[key] = [str(item) if isinstance(item, Path) else item for item in val]
        else:
            cleaned[key] = val
    return cleaned


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate 3D pancreas and tumor segmentation model."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Directory of .npz cases")
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("docs/splits.json"),
        help="Path to splits JSON file",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("docs/dataset_stats.csv"),
        help="Path to dataset stats CSV file",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint file")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Evaluation split partition",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        default=False,
        help="Permit evaluation on test split",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=list(DEFAULT_PATCH_SIZE),
        help="Patch size for sliding window inference (z y x)",
    )
    parser.add_argument(
        "--num-figure-cases",
        type=int,
        default=6,
        help="Number of cases to generate overlay figures for",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for 95% CIs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (cuda or cpu)",
    )

    return parser.parse_args(argv)


def evaluate(args: argparse.Namespace) -> int:
    """
    Run evaluation pipeline and write results to out_dir.

    Returns exit code: 0 if all cases succeeded, 1 if any case failed.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_size = (int(args.patch_size[0]), int(args.patch_size[1]), int(args.patch_size[2]))

    # Load checkpoint and rebuild model
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args") or {}
    channels = ckpt_args.get("channels", (16, 32, 64, 128, 256))
    strides = ckpt_args.get("strides", (2, 2, 2, 2))
    num_res_units = ckpt_args.get("num_res_units", 2)

    model = build_unet(
        channels=tuple(channels),
        strides=tuple(strides),
        num_res_units=int(num_res_units),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    iteration = ckpt.get("iteration")

    # Load split IDs and stats
    case_ids = load_split_ids(args.splits, args.split)
    stats_dict = load_dataset_stats(args.stats)
    n_requested = len(case_ids)

    evaluated_rows: List[Dict[str, Any]] = []
    failed_cases: List[Dict[str, str]] = []

    for i, case_id in enumerate(case_ids, start=1):
        print(f"[{i}/{n_requested}] Evaluating {case_id}...")
        try:
            fname = case_id if case_id.endswith(".npz") else f"{case_id}.npz"
            case_path = Path(args.data_root) / fname
            image, gt_label = load_case(case_path)

            pred_label = predict_volume(
                model=model,
                image=image,
                patch_size=patch_size,
                device=device,
            )

            dices = region_dice(pred_label, gt_label)
            pancreas_dice = float(dices["pancreas"])
            tumor_dice = float(dices["tumor"])

            gt_tumor = gt_label == LABEL_TUMOR
            pred_tumor = pred_label == LABEL_TUMOR

            tumor_detected = bool(np.any(pred_tumor & gt_tumor))
            detected_strict = bool(tumor_dice >= 0.10)

            pred_tumor_ml = float(np.count_nonzero(pred_tumor) * VOXEL_VOLUME_ML)
            gt_tumor_ml = float(np.count_nonzero(gt_tumor) * VOXEL_VOLUME_ML)

            diam_mm = stats_dict.get(case_id)
            size_group = assign_size_group(diam_mm)

            row = {
                "case_id": case_id,
                "pancreas_dice": pancreas_dice,
                "tumor_dice": tumor_dice,
                "tumor_detected": tumor_detected,
                "detected_strict": detected_strict,
                "pred_tumor_ml": pred_tumor_ml,
                "gt_tumor_ml": gt_tumor_ml,
                "tumor_diam_mm": diam_mm,
                "size_group": size_group,
            }
            evaluated_rows.append(row)

        except Exception as exc:
            print(f"WARNING: Case {case_id} failed during evaluation: {exc}")
            failed_cases.append({
                "case_id": case_id,
                "error": str(exc),
            })

    n_evaluated = len(evaluated_rows)
    unknown_size_rows = [r for r in evaluated_rows if r["size_group"] == "unknown"]
    n_unknown_size = len(unknown_size_rows)

    small_rows = [r for r in evaluated_rows if r["size_group"] == "small"]
    large_rows = [r for r in evaluated_rows if r["size_group"] == "large"]

    overall_summary = summarize_group(
        evaluated_rows,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )
    small_summary = summarize_group(
        small_rows,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )
    large_summary = summarize_group(
        large_rows,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )

    # 1. Write per_case.csv
    csv_path = out_dir / "per_case.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id",
            "pancreas_dice",
            "tumor_dice",
            "tumor_detected",
            "detected_strict",
            "pred_tumor_ml",
            "gt_tumor_ml",
            "tumor_diam_mm",
            "size_group",
        ])
        for r in evaluated_rows:
            diam_str = f"{r['tumor_diam_mm']:.4f}" if r["tumor_diam_mm"] is not None else ""
            writer.writerow([
                r["case_id"],
                f"{r['pancreas_dice']:.6f}",
                f"{r['tumor_dice']:.6f}",
                str(r["tumor_detected"]),
                str(r["detected_strict"]),
                f"{r['pred_tumor_ml']:.4f}",
                f"{r['gt_tumor_ml']:.4f}",
                diam_str,
                r["size_group"],
            ])

    # 2. Write summary.json
    summary_data = {
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_iteration": iteration,
        "split": args.split,
        "seed": args.seed,
        "n_boot": args.bootstrap_samples,
        "args": sanitize_args(args),
        "n_requested": n_requested,
        "n_evaluated": n_evaluated,
        "n_unknown_size": n_unknown_size,
        "failed_cases": failed_cases,
        "overall": overall_summary,
        "small": small_summary,
        "large": large_summary,
    }
    json_path = out_dir / "summary.json"
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, allow_nan=False)

    # 3. Write summary.md
    md_content = generate_markdown_summary(summary_data)
    md_path = out_dir / "summary.md"
    with open(md_path, mode="w", encoding="utf-8") as f:
        f.write(md_content)

    # 4. Generate figures if requested
    if args.num_figure_cases > 0 and evaluated_rows:
        fig_case_ids = select_figure_cases(evaluated_rows, args.num_figure_cases)
        for fig_cid in fig_case_ids:
            try:
                fname = fig_cid if fig_cid.endswith(".npz") else f"{fig_cid}.npz"
                img, gt = load_case(Path(args.data_root) / fname)
                pred = predict_volume(
                    model=model,
                    image=img,
                    patch_size=patch_size,
                    device=device,
                )
                save_overlay_figure(
                    image=img,
                    gt_label=gt,
                    pred_label=pred,
                    out_path=out_dir / "figures" / f"{fig_cid}.png",
                    case_id=fig_cid,
                )
            except Exception as fig_exc:
                print(f"WARNING: Figure generation failed for {fig_cid}: {fig_exc}")

    if failed_cases:
        sys.stderr.write(
            f"WARNING: {len(failed_cases)} case(s) failed during evaluation!\n"
        )
        return 1

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI and module execution entry point."""
    args = parse_args(argv)

    # Safety check runs FIRST before any file or directory operations
    if args.split == "test" and not args.final:
        sys.stderr.write("Error: --split test requires --final flag.\n")
        return 2

    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
