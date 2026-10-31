"""
Evaluation pipeline for slice-level tumor classification.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.classification.model import build_classifier, center_crop
from src.classification.train import compute_auc, sanitize_args


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Compute binary classification metrics at a given probability threshold.

    Parameters
    ----------
    y_true : np.ndarray
        1D binary ground truth labels {0, 1}.
    y_prob : np.ndarray
        1D predicted probabilities in [0, 1].
    threshold : float
        Decision threshold.

    Returns
    -------
    Dict[str, Any]
        Dictionary with accuracy, sensitivity, specificity, precision, f1, and
        confusion counts (tp, fp, tn, fn).
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    n = len(y_true)
    accuracy = float((tp + tn) / n) if n > 0 else None
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None

    if precision is not None and sensitivity is not None and (precision + sensitivity) > 0:
        f1 = float(2.0 * precision * sensitivity / (precision + sensitivity))
    elif precision is not None and sensitivity is not None:
        f1 = 0.0
    else:
        f1 = None

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def case_cluster_bootstrap(
    cases: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    val_threshold: float,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[Dict[str, Any], int]:
    """
    Compute 95% confidence intervals using case-cluster bootstrapping.

    Resamples distinct case identifiers with replacement, preserving within-case
    slice correlations. Replicates with only one class present are skipped.

    Parameters
    ----------
    cases : np.ndarray
        Array of case identifiers for each slice.
    y_true : np.ndarray
        1D binary ground truth labels {0, 1}.
    y_prob : np.ndarray
        1D predicted probabilities.
    val_threshold : float
        Primary evaluation threshold.
    n_boot : int, default=1000
        Number of bootstrap resamples.
    seed : int, default=42
        Random seed for reproducibility.
    alpha : float, default=0.05
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    Tuple[Dict[str, Any], int]
        (ci_dict, n_valid_boot)
    """
    cases = np.asarray(cases)
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    unique_cases = np.unique(cases)
    n_clusters = len(unique_cases)

    if n_clusters == 0:
        return {}, 0

    case_indices = {cid: np.flatnonzero(cases == cid) for cid in unique_cases}
    rng = np.random.default_rng(seed)

    boot_auc: List[float] = []
    boot_val: Dict[str, List[float]] = {
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1": [],
    }
    boot_05: Dict[str, List[float]] = {
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1": [],
    }

    n_valid_boot = 0
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    for _ in range(n_boot):
        sampled_cids = rng.choice(unique_cases, size=n_clusters, replace=True)
        idx_lists = [case_indices[cid] for cid in sampled_cids]
        b_idx = np.concatenate(idx_lists)

        b_y = y_true[b_idx]
        b_p = y_prob[b_idx]

        # Resamples with only one class make AUC undefined; skip them
        if not (np.any(b_y == 1) and np.any(b_y == 0)):
            continue

        b_auc = compute_auc(b_y, b_p)
        if np.isnan(b_auc):
            continue

        n_valid_boot += 1
        boot_auc.append(b_auc)

        m_val = compute_metrics_at_threshold(b_y, b_p, val_threshold)
        for k in boot_val:
            val = m_val.get(k)
            if val is not None and not np.isnan(val):
                boot_val[k].append(float(val))

        m_05 = compute_metrics_at_threshold(b_y, b_p, 0.5)
        for k in boot_05:
            val = m_05.get(k)
            if val is not None and not np.isnan(val):
                boot_05[k].append(float(val))

    # If fewer than 50% of resamples are valid, set CIs to None
    if n_valid_boot < 0.5 * n_boot:
        ci_dict = {
            "auc": None,
            "val_threshold": {k: None for k in boot_val},
            "fixed_05_threshold": {k: None for k in boot_05},
        }
        return ci_dict, n_valid_boot

    def percentile_interval(vals: List[float]) -> Optional[List[float]]:
        if not vals:
            return None
        return [float(np.percentile(vals, lower_pct)), float(np.percentile(vals, upper_pct))]

    ci_dict = {
        "auc": percentile_interval(boot_auc),
        "val_threshold": {k: percentile_interval(boot_val[k]) for k in boot_val},
        "fixed_05_threshold": {k: percentile_interval(boot_05[k]) for k in boot_05},
    }
    return ci_dict, n_valid_boot


def compute_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> List[Tuple[float, float, float]]:
    """
    Generate (threshold, fpr, tpr) points for ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        1D binary ground truth labels {0, 1}.
    y_prob : np.ndarray
        1D predicted probabilities in [0, 1].

    Returns
    -------
    List[Tuple[float, float, float]]
        List of (threshold, fpr, tpr) tuples sorted by fpr.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        return [(1.0, 0.0, 0.0), (0.0, 1.0, 1.0)]

    unique_probs = np.unique(y_prob)
    thresholds = np.concatenate(([float("inf")], np.sort(unique_probs)[::-1], [-float("inf")]))

    points: List[Tuple[float, float, float]] = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = int(np.sum(y_pred[pos_mask] == 1))
        fp = int(np.sum(y_pred[neg_mask] == 1))
        fpr = float(fp / n_neg)
        tpr = float(tp / n_pos)
        points.append((float(thresh), fpr, tpr))

    # Sort primarily by FPR ascending, secondarily by TPR
    points.sort(key=lambda item: (item[1], item[2]))
    return points


def clean_for_json(obj: Any) -> Any:
    """Recursively replace NaN and Inf with None for JSON compliance."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    return obj


def plot_evaluation(
    roc_points: List[Tuple[float, float, float]],
    cm_dict: Dict[str, int],
    auc: Optional[float],
    out_dir: Path,
) -> None:
    """
    Save ROC curve and confusion matrix plots if matplotlib is available.

    Matplotlib is imported strictly inside this function.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # 1. ROC Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    if roc_points:
        fprs = [p[1] for p in roc_points]
        tprs = [p[2] for p in roc_points]
        label = f"ROC (AUC = {auc:.3f})" if (auc is not None and not np.isnan(auc)) else "ROC"
        ax.plot(fprs, tprs, color="blue", lw=2, label=label)
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc.png", dpi=150)
    plt.close(fig)

    # 2. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    cm = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]],
    ])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax.set_yticklabels(["True Neg", "True Pos"])
    ax.set_title("Confusion Matrix")
    for r in range(2):
        for c in range(2):
            color = "white" if cm[r, c] > (cm.max() / 2.0 if cm.max() > 0 else 1) else "black"
            ax.text(c, r, str(cm[r, c]), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion.png", dpi=150)
    plt.close(fig)


def generate_markdown_summary(summary_data: Dict[str, Any]) -> str:
    """Format evaluation summary into a Markdown report."""
    mv = summary_data.get("metrics_at_val_threshold", {})
    m5 = summary_data.get("metrics_at_fixed_05_threshold", {})
    ci = summary_data.get("ci_95", {})
    ci_val = ci.get("val_threshold", {}) if ci else {}
    ci_5 = ci.get("fixed_05_threshold", {}) if ci else {}

    def fmt_val_ci(val: Optional[float], ci_pair: Optional[List[float]]) -> str:
        if val is None:
            return "N/A"
        if ci_pair is not None and len(ci_pair) == 2:
            return f"{val:.4f} [{ci_pair[0]:.4f}, {ci_pair[1]:.4f}]"
        return f"{val:.4f}"

    auc_val = summary_data.get("auc")
    auc_ci = ci.get("auc") if ci else None
    auc_str = fmt_val_ci(auc_val, auc_ci)

    lines = [
        "# Slice Tumor Classifier Evaluation Summary",
        "",
        f"- **Checkpoint**: `{summary_data.get('checkpoint_path', 'N/A')}`",
        f"- **Split**: `{summary_data.get('split', 'N/A')}`",
        f"- **Validation Threshold**: `{summary_data.get('val_threshold', 'N/A')}`",
        f"- **Total Slices**: {summary_data.get('n_slices', 0)} "
        f"({summary_data.get('n_pos', 0)} pos, {summary_data.get('n_neg', 0)} neg)",
        f"- **Total Cases**: {summary_data.get('n_cases', 0)}",
        f"- **Valid Bootstrap Resamples**: {summary_data.get('n_valid_boot', 0)} / "
        f"{summary_data.get('bootstrap_samples', 0)}",
        f"- **ROC AUC**: {auc_str}",
        "",
        "## Classification Metrics",
        "",
        "| Metric | Validation Threshold | Fixed 0.5 Threshold |",
        "|---|---|---|",
        f"| Threshold | {mv.get('threshold', 0.5):.4f} | {m5.get('threshold', 0.5):.4f} |",
        f"| Accuracy | {fmt_val_ci(mv.get('accuracy'), ci_val.get('accuracy'))} | "
        f"{fmt_val_ci(m5.get('accuracy'), ci_5.get('accuracy'))} |",
        f"| Sensitivity | {fmt_val_ci(mv.get('sensitivity'), ci_val.get('sensitivity'))} | "
        f"{fmt_val_ci(m5.get('sensitivity'), ci_5.get('sensitivity'))} |",
        f"| Specificity | {fmt_val_ci(mv.get('specificity'), ci_val.get('specificity'))} | "
        f"{fmt_val_ci(m5.get('specificity'), ci_5.get('specificity'))} |",
        f"| Precision (PPV) | {fmt_val_ci(mv.get('precision'), ci_val.get('precision'))} | "
        f"{fmt_val_ci(m5.get('precision'), ci_5.get('precision'))} |",
        f"| F1 Score | {fmt_val_ci(mv.get('f1'), ci_val.get('f1'))} | "
        f"{fmt_val_ci(m5.get('f1'), ci_5.get('f1'))} |",
        "",
        "## Confusion Matrix Counts",
        "",
        "| Condition | Val Threshold Count | 0.5 Threshold Count |",
        "|---|---|---|",
        f"| True Positives (TP) | {mv.get('tp', 0)} | {m5.get('tp', 0)} |",
        f"| False Positives (FP) | {mv.get('fp', 0)} | {m5.get('fp', 0)} |",
        f"| True Negatives (TN) | {mv.get('tn', 0)} | {m5.get('tn', 0)} |",
        f"| False Negatives (FN) | {mv.get('fn', 0)} | {m5.get('fn', 0)} |",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate ResNet18 slice-level tumor classifier."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to cls_*.npz directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Evaluation split (val or test)",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        default=False,
        help="Permit evaluation on test split",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of case-cluster bootstrap resamples",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (cuda or cpu)",
    )
    return parser.parse_args(argv)


def evaluate(args: argparse.Namespace) -> int:
    """Run classifier evaluation pipeline."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    split_path = Path(args.data_dir) / f"cls_{args.split}.npz"
    if not split_path.is_file():
        raise FileNotFoundError(f"Dataset split file not found: {split_path}")

    with np.load(split_path) as data:
        x_raw = data["X"]
        y_raw = data["y"]
        cases_raw = data["case"]
        zs_raw = data["z"]

    # Rebuild model (pretrained=False) and load checkpoint
    model = build_classifier(pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_threshold = float(ckpt.get("threshold", 0.5))

    # Predict in batches with center crop 128
    all_probs: List[float] = []
    n_slices = len(x_raw)
    batch_size = args.batch_size

    with torch.no_grad():
        for start_idx in range(0, n_slices, batch_size):
            end_idx = min(start_idx + batch_size, n_slices)
            batch_np = x_raw[start_idx:end_idx]
            batch_cropped = center_crop(batch_np, size=128)
            batch_tensor = (
                torch.from_numpy(batch_cropped.astype(np.float32)).unsqueeze(1).to(device)
            )
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_probs.extend(probs.tolist() if probs.ndim == 1 else [float(probs)])

    y_prob = np.array(all_probs, dtype=np.float64)
    y_true = np.asarray(y_raw, dtype=np.uint8)

    # Compute metrics
    auc_val = compute_auc(y_true, y_prob)
    metrics_val = compute_metrics_at_threshold(y_true, y_prob, val_threshold)
    metrics_05 = compute_metrics_at_threshold(y_true, y_prob, 0.5)

    # Case-cluster bootstrap CIs
    ci_dict, n_valid_boot = case_cluster_bootstrap(
        cases=cases_raw,
        y_true=y_true,
        y_prob=y_prob,
        val_threshold=val_threshold,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    n_cases = len(np.unique(cases_raw))

    # 1. Write per_slice.csv
    per_slice_path = out_dir / "per_slice.csv"
    with open(per_slice_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "z", "y", "prob"])
        for c, z, y_val, p_val in zip(cases_raw, zs_raw, y_true, y_prob):
            writer.writerow([str(c), int(z), int(y_val), f"{p_val:.6f}"])

    # 2. Write roc_points.csv
    roc_points = compute_roc_curve(y_true, y_prob)
    roc_csv_path = out_dir / "roc_points.csv"
    with open(roc_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "fpr", "tpr"])
        for thresh, fpr, tpr in roc_points:
            writer.writerow([f"{thresh:.6f}", f"{fpr:.6f}", f"{tpr:.6f}"])

    # 3. Write summary.json
    summary_data = {
        "checkpoint_path": str(args.checkpoint),
        "split": args.split,
        "val_threshold": val_threshold,
        "n_slices": n_slices,
        "n_cases": n_cases,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "bootstrap_samples": args.bootstrap_samples,
        "n_valid_boot": n_valid_boot,
        "auc": auc_val,
        "metrics_at_val_threshold": metrics_val,
        "metrics_at_fixed_05_threshold": metrics_05,
        "ci_95": ci_dict,
        "args": sanitize_args(args),
    }
    json_path = out_dir / "summary.json"
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(clean_for_json(summary_data), f, indent=2, allow_nan=False)

    # 4. Write summary.md
    md_content = generate_markdown_summary(summary_data)
    md_path = out_dir / "summary.md"
    with open(md_path, mode="w", encoding="utf-8") as f:
        f.write(md_content)

    # 5. Plots
    plot_evaluation(
        roc_points=roc_points,
        cm_dict=metrics_val,
        auc=auc_val,
        out_dir=out_dir,
    )

    print(
        f"Evaluation on '{args.split}' complete: {n_slices} slices ({n_pos} pos, {n_neg} neg). "
        f"AUC: {auc_val:.4f} | Accuracy @ {val_threshold:.4f}: {metrics_val['accuracy']:.4f}"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI and module entry point with strict test-split safety check."""
    args = parse_args(argv)

    # Safety check runs FIRST before any file or directory operations
    if args.split == "test" and not args.final:
        sys.stderr.write("Error: --split test requires --final flag.\n")
        return 2

    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
