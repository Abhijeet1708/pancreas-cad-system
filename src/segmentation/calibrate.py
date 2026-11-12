"""
Decision-threshold calibration on validation split for 3D pancreas and tumor segmentation.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import torch

from src.segmentation.checkpoint import load_checkpoint
from src.segmentation.data import (
    DEFAULT_PATCH_SIZE,
    LABEL_TUMOR,
    load_case,
    load_split_ids,
)
from src.segmentation.evaluate import (
    VOXEL_VOLUME_ML,
    assign_size_group,
    load_dataset_stats,
)
from src.segmentation.metrics import region_dice
from src.segmentation.model import build_unet, predict_logits

SELECTION_RULE_TEXT: str = (
    "Among biases whose overall mean pancreas Dice is at least "
    "(value at bias 0.0) - 0.01, pick the highest overall mean tumor Dice; "
    "ties go to the smaller absolute bias."
)


def select_recommended_bias(
    overall_rows: Union[List[Dict[str, Any]], Dict[float, Dict[str, Any]]],
) -> Tuple[float, str]:
    """
    Select recommended logit bias based on validation metrics.

    Parameters
    ----------
    overall_rows : Union[List[Dict[str, Any]], Dict[float, Dict[str, Any]]]
        Either list of summary rows containing bias, mean_pancreas_dice, mean_tumor_dice,
        or dict mapping bias -> {mean_pancreas_dice: ..., mean_tumor_dice: ...}.

    Returns
    -------
    Tuple[float, str]
        (recommended_bias, reason_string)
    """
    stats: Dict[float, Tuple[float, float]] = {}
    if isinstance(overall_rows, dict):
        for b, v in overall_rows.items():
            b_val = float(b)
            p_dice = float(v.get("mean_pancreas_dice", v.get("pancreas_dice", 0.0)))
            t_dice = float(v.get("mean_tumor_dice", v.get("tumor_dice", 0.0)))
            stats[b_val] = (p_dice, t_dice)
    else:
        for r in overall_rows:
            if r.get("group", "overall") != "overall":
                continue
            b_val = float(r["bias"])
            p_dice = float(r.get("mean_pancreas_dice", r.get("pancreas_dice", 0.0)))
            t_dice = float(r.get("mean_tumor_dice", r.get("tumor_dice", 0.0)))
            stats[b_val] = (p_dice, t_dice)

    if not stats:
        raise ValueError("No overall metrics available to select recommended bias.")

    zero_bias_key: Optional[float] = None
    for b in stats:
        if abs(b - 0.0) < 1e-9:
            zero_bias_key = b
            break

    if zero_bias_key is None:
        warnings.warn(
            "Bias 0.0 is not among the evaluated biases! Selection rule requires "
            "baseline pancreas Dice at bias 0.0."
        )
        sys.stderr.write(
            "WARNING: bias 0.0 is not among the evaluated biases! "
            "Selection rule requires baseline pancreas Dice at bias 0.0.\n"
        )
        base_pancreas = min(p for p, _ in stats.values())
        threshold = base_pancreas - 0.01
    else:
        base_pancreas = stats[zero_bias_key][0]
        threshold = base_pancreas - 0.01

    candidates = [
        (b, p, t) for b, (p, t) in stats.items()
        if p >= threshold - 1e-9
    ]

    if not candidates:
        candidates = [(b, p, t) for b, (p, t) in stats.items()]

    candidates.sort(key=lambda item: (-round(item[2], 9), round(abs(item[0]), 9), item[0]))
    best_bias, best_p, best_t = candidates[0]

    reason = (
        f"Selected bias {best_bias}: highest overall mean tumor Dice ({best_t:.4f}) "
        f"satisfying pancreas Dice >= {threshold:.4f} (baseline at 0.0 was {base_pancreas:.4f})"
    )
    return best_bias, reason


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for calibration."""
    parser = argparse.ArgumentParser(
        description="Threshold calibration on validation split."
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
        "--biases",
        type=str,
        default="-1,0,0.5,1,1.5,2,3,4,5,6",
        help="Comma-separated tumor logit bias values",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=list(DEFAULT_PATCH_SIZE),
        help="Patch size for sliding window inference (z y x)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cuda or cpu)",
    )
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    processed_args = []
    i = 0
    while i < len(raw_args):
        if (
            raw_args[i] == "--biases"
            and i + 1 < len(raw_args)
            and not raw_args[i + 1].startswith("--")
        ):
            processed_args.append(f"--biases={raw_args[i + 1]}")
            i += 2
        else:
            processed_args.append(raw_args[i])
            i += 1

    return parser.parse_args(processed_args)


def generate_calibration_markdown(
    summary_rows: List[Dict[str, Any]],
    recommended_bias: float,
    reason: str,
) -> str:
    """Format calibration results into readable Markdown tables."""
    lines = [
        "# Tumor Decision-Threshold Calibration",
        "",
        f"- **Recommended Bias**: `{recommended_bias}`",
        f"- **Selection Reason**: {reason}",
        f"- **Selection Rule**: {SELECTION_RULE_TEXT}",
        "",
        "## Calibration Results by Bias and Tumor Size Group",
        "",
        "| Bias | Group | N | Mean Pancreas Dice | Mean Tumor Dice | "
        "Median Tumor Dice | Tumor Detection Rate | Strict Detection Rate | Mean Volume Ratio |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in summary_rows:
        b = r["bias"]
        grp = r["group"]
        n = r["n"]
        if n == 0:
            lines.append(f"| {b} | {grp} | 0 | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        p_mean = f"{r['mean_pancreas_dice']:.4f}"
        t_mean = f"{r['mean_tumor_dice']:.4f}"
        t_med = f"{r['median_tumor_dice']:.4f}"
        det_r = f"{r['detection_rate'] * 100:.1f}%"
        s_det_r = f"{r['strict_detection_rate'] * 100:.1f}%"
        vol_r = f"{r['mean_volume_ratio']:.4f}"
        lines.append(
            f"| {b} | {grp} | {n} | {p_mean} | {t_mean} | {t_med} | {det_r} | {s_det_r} | {vol_r} |"
        )

    lines.append("")
    return "\n".join(lines)


def calibrate(args: argparse.Namespace) -> int:
    """
    Run threshold calibration across validation split.

    Returns exit code: 0 on success, 1 if any case failed.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_size = (int(args.patch_size[0]), int(args.patch_size[1]), int(args.patch_size[2]))
    biases = [float(b.strip()) for b in args.biases.split(",") if b.strip()]

    if not any(abs(b - 0.0) < 1e-9 for b in biases):
        warnings.warn(
            "Bias 0.0 is not in --biases. Selection rule requires 0.0 for baseline comparison."
        )
        sys.stderr.write(
            "WARNING: bias 0.0 is not among the evaluated biases! "
            "Selection rule requires baseline pancreas Dice at bias 0.0.\n"
        )

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

    # Hard guard: only read "val" IDs
    val_case_ids = load_split_ids(args.splits, "val")
    if not val_case_ids:
        raise ValueError(f"No validation cases found in splits file: {args.splits}")

    stats_dict = load_dataset_stats(args.stats)
    n_requested = len(val_case_ids)

    per_case_rows: List[Dict[str, Any]] = []
    failed_cases: List[Dict[str, str]] = []
    evaluated_cases: List[str] = []

    for idx, case_id in enumerate(val_case_ids, start=1):
        print(f"[{idx}/{n_requested}] Calibrating {case_id}...")
        try:
            fname = case_id if case_id.endswith(".npz") else f"{case_id}.npz"
            case_path = Path(args.data_root) / fname
            image, gt_label = load_case(case_path)

            # Predict raw logits ONCE
            logits = predict_logits(
                model=model,
                image=image,
                patch_size=patch_size,
                device=device,
            )

            diam_mm = stats_dict.get(case_id)
            size_group = assign_size_group(diam_mm)
            gt_tumor = gt_label == LABEL_TUMOR
            true_tumor_ml = float(np.count_nonzero(gt_tumor) * VOXEL_VOLUME_ML)

            evaluated_cases.append(case_id)

            for b in biases:
                logits_b = logits.clone()
                if b != 0.0:
                    logits_b[2] = logits_b[2] + b
                pred_label = torch.argmax(logits_b, dim=0).to(torch.uint8).numpy()

                dices = region_dice(pred_label, gt_label)
                pancreas_dice = float(dices["pancreas"])
                tumor_dice = float(dices["tumor"])

                pred_tumor = pred_label == LABEL_TUMOR
                tumor_detected = bool(np.any(pred_tumor & gt_tumor))
                detected_strict = bool(tumor_dice >= 0.10)
                pred_tumor_ml = float(np.count_nonzero(pred_tumor) * VOXEL_VOLUME_ML)

                per_case_rows.append({
                    "case": case_id,
                    "size_group": size_group,
                    "tumor_diam_mm": diam_mm,
                    "bias": b,
                    "pancreas_dice": pancreas_dice,
                    "tumor_dice": tumor_dice,
                    "tumor_detected": tumor_detected,
                    "detected_strict": detected_strict,
                    "pred_tumor_ml": pred_tumor_ml,
                    "true_tumor_ml": true_tumor_ml,
                })
        except Exception as exc:
            print(f"WARNING: Case {case_id} failed during calibration: {exc}")
            failed_cases.append({
                "case_id": case_id,
                "error": str(exc),
            })

    # 1. Write per_case_bias.csv
    per_case_path = out_dir / "per_case_bias.csv"
    with open(per_case_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
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
        ])
        for r in per_case_rows:
            diam_str = f"{r['tumor_diam_mm']:.4f}" if r["tumor_diam_mm"] is not None else ""
            writer.writerow([
                r["case"],
                r["size_group"],
                diam_str,
                r["bias"],
                f"{r['pancreas_dice']:.6f}",
                f"{r['tumor_dice']:.6f}",
                str(r["tumor_detected"]),
                str(r["detected_strict"]),
                f"{r['pred_tumor_ml']:.4f}",
                f"{r['true_tumor_ml']:.4f}",
            ])

    # 2. Aggregate calibration summary per bias and group
    groups = ["overall", "small", "large"]
    summary_rows: List[Dict[str, Any]] = []

    for b in biases:
        b_rows = [r for r in per_case_rows if abs(r["bias"] - b) < 1e-9]
        for grp in groups:
            if grp == "overall":
                grp_rows = b_rows
            else:
                grp_rows = [r for r in b_rows if r["size_group"] == grp]

            n = len(grp_rows)
            if n == 0:
                summary_rows.append({
                    "bias": b,
                    "group": grp,
                    "n": 0,
                    "mean_pancreas_dice": None,
                    "mean_tumor_dice": None,
                    "median_tumor_dice": None,
                    "detection_rate": None,
                    "strict_detection_rate": None,
                    "mean_volume_ratio": None,
                })
            else:
                p_vals = [r["pancreas_dice"] for r in grp_rows]
                t_vals = [r["tumor_dice"] for r in grp_rows]
                det_cnt = sum(1 for r in grp_rows if r["tumor_detected"])
                st_cnt = sum(1 for r in grp_rows if r["detected_strict"])
                ratios = [
                    (r["pred_tumor_ml"] / r["true_tumor_ml"]) if r["true_tumor_ml"] > 0 else 0.0
                    for r in grp_rows
                ]

                summary_rows.append({
                    "bias": b,
                    "group": grp,
                    "n": n,
                    "mean_pancreas_dice": float(np.mean(p_vals)),
                    "mean_tumor_dice": float(np.mean(t_vals)),
                    "median_tumor_dice": float(np.median(t_vals)),
                    "detection_rate": float(det_cnt / n),
                    "strict_detection_rate": float(st_cnt / n),
                    "mean_volume_ratio": float(np.mean(ratios)),
                })

    # 3. Write calibration.csv
    calib_csv_path = out_dir / "calibration.csv"
    with open(calib_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bias",
            "group",
            "n",
            "mean_pancreas_dice",
            "mean_tumor_dice",
            "median_tumor_dice",
            "detection_rate",
            "strict_detection_rate",
            "mean_volume_ratio",
        ])
        for r in summary_rows:
            if r["n"] == 0:
                writer.writerow([r["bias"], r["group"], 0, "", "", "", "", "", ""])
            else:
                writer.writerow([
                    r["bias"],
                    r["group"],
                    r["n"],
                    f"{r['mean_pancreas_dice']:.6f}",
                    f"{r['mean_tumor_dice']:.6f}",
                    f"{r['median_tumor_dice']:.6f}",
                    f"{r['detection_rate']:.6f}",
                    f"{r['strict_detection_rate']:.6f}",
                    f"{r['mean_volume_ratio']:.6f}",
                ])

    # 4. Selection rule for recommended bias
    recommended_bias: Optional[float] = None
    reason: str = ""
    if evaluated_cases:
        recommended_bias, reason = select_recommended_bias(summary_rows)

    # 5. Write calibration.json
    calib_json_path = out_dir / "calibration.json"
    json_data = {
        "checkpoint_path": Path(args.checkpoint).name,
        "checkpoint_iteration": iteration,
        "n_requested": n_requested,
        "n_evaluated": len(evaluated_cases),
        "failed_cases": failed_cases,
        "biases": biases,
        "selection_rule": SELECTION_RULE_TEXT,
        "recommended_bias": recommended_bias,
        "recommended_bias_reason": reason,
        "calibration": summary_rows,
    }
    with open(calib_json_path, mode="w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, allow_nan=False)

    # 6. Write calibration.md
    md_content = generate_calibration_markdown(
        summary_rows=summary_rows,
        recommended_bias=recommended_bias if recommended_bias is not None else 0.0,
        reason=reason,
    )
    calib_md_path = out_dir / "calibration.md"
    with open(calib_md_path, mode="w", encoding="utf-8") as f:
        f.write(md_content)

    if failed_cases:
        sys.stderr.write(
            f"WARNING: {len(failed_cases)} case(s) failed during calibration!\n"
        )
        return 1

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI and module execution entry point."""
    args = parse_args(argv)
    return calibrate(args)


if __name__ == "__main__":
    sys.exit(main())
