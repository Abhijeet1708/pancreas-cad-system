"""
Training script for 2D slice-level tumor classifier (ResNet18).
"""

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.classification.model import build_classifier, center_crop


def sanitize_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse namespace to plain Python types for serialization."""
    cleaned: Dict[str, Any] = {}
    for key, val in vars(args).items():
        if isinstance(val, Path):
            cleaned[key] = str(val)
        elif isinstance(val, (list, tuple)):
            cleaned[key] = [str(item) if isinstance(item, Path) else item for item in val]
        else:
            cleaned[key] = val
    return cleaned


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve using Mann-Whitney U rank statistics.

    Parameters
    ----------
    y_true : np.ndarray
        1D binary ground truth labels {0, 1}.
    y_score : np.ndarray
        1D predicted scores or probabilities.

    Returns
    -------
    float
        AUC value in [0.0, 1.0], or NaN if only one class is present.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    pos_mask = y_true == 1
    n_pos = int(np.sum(pos_mask))
    n_neg = int(len(y_true) - n_pos)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    unique_scores, inverse, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(y_score):
        for idx, count in enumerate(counts):
            if count > 1:
                mask = inverse == idx
                ranks[mask] = np.mean(ranks[mask])

    r_pos = float(np.sum(ranks[pos_mask]))
    u = r_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Find threshold maximizing Youden's J statistic (sensitivity + specificity - 1).

    Parameters
    ----------
    y_true : np.ndarray
        1D binary ground truth labels {0, 1}.
    y_prob : np.ndarray
        1D predicted probabilities in [0, 1].

    Returns
    -------
    Tuple[float, float]
        (best_threshold, best_j)
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    unique_probs = np.unique(y_prob)
    if len(unique_probs) > 1:
        midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
        candidates = np.concatenate(([0.0], midpoints, [1.0], [0.5]))
    else:
        candidates = np.array([0.0, 0.5, 1.0])

    candidates = np.unique(candidates)
    best_j = -float("inf")
    best_thresh = 0.5

    for thresh in candidates:
        y_pred = (y_prob >= thresh).astype(int)
        tp = int(np.sum(y_pred[pos_mask] == 1))
        tn = int(np.sum(y_pred[neg_mask] == 0))
        tpr = tp / n_pos
        tnr = tn / n_neg
        j = tpr + tnr - 1.0

        if j > best_j or (j == best_j and abs(thresh - 0.5) < abs(best_thresh - 0.5)):
            best_j = j
            best_thresh = float(thresh)

    return float(best_thresh), float(best_j)


class SliceDataset(Dataset):
    """
    Dataset of 2D axial slice crops.

    During training: extracts a random 128 x 128 crop out of the 160 x 160 window,
    applies random horizontal flip, and applies intensity scale/shift.
    During validation: applies a center 128 x 128 crop.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_train: bool = True,
        seed: int = 42,
    ) -> None:
        self.X = X
        self.y = y
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.X[idx].astype(np.float32)
        target = float(self.y[idx])

        if self.is_train:
            sy = int(self.rng.integers(0, 160 - 128 + 1))
            sx = int(self.rng.integers(0, 160 - 128 + 1))
            cropped = img[sy:sy + 128, sx:sx + 128]

            if self.rng.random() < 0.5:
                cropped = np.ascontiguousarray(np.fliplr(cropped))

            scale = float(self.rng.uniform(0.9, 1.1))
            shift = float(self.rng.uniform(-0.05, 0.05))
            cropped = np.clip(cropped * scale + shift, 0.0, 1.0)
        else:
            cropped = center_crop(img, size=128)

        tensor_x = torch.from_numpy(cropped).unsqueeze(0).float()
        tensor_y = torch.tensor([target], dtype=torch.float32)
        return tensor_x, tensor_y


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train ResNet18 slice-level tumor classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing cls_train.npz and cls_val.npz",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use automatic mixed precision (default: on only with CUDA)",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize with ImageNet weights",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run classifier training."""
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp if args.amp is not None else (device.type == "cuda")

    # Save run_config.json
    config_path = out_dir / "run_config.json"
    with open(config_path, mode="w", encoding="utf-8") as f:
        json.dump(sanitize_args(args), f, indent=2, default=str)

    # Load train and val data (never open cls_test.npz)
    train_path = Path(args.data_dir) / "cls_train.npz"
    val_path = Path(args.data_dir) / "cls_val.npz"

    with np.load(train_path) as data:
        x_train = data["X"]
        y_train = data["y"]

    with np.load(val_path) as data:
        x_val = data["X"]
        y_val = data["y"]

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    pos_weight_val = (float(n_neg) / float(n_pos)) if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)

    train_ds = SliceDataset(x_train, y_train, is_train=True, seed=args.seed)
    val_ds = SliceDataset(x_val, y_val, is_train=False, seed=args.seed + 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = build_classifier(pretrained=args.pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0
    )

    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    metrics_csv_path = out_dir / "metrics.csv"
    with open(metrics_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_auc", "val_accuracy", "threshold"]
        )

    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            train_loss_sum += float(loss.item()) * len(batch_x)
            train_count += len(batch_x)

        scheduler.step()
        train_loss = train_loss_sum / max(1, train_count)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        all_val_probs: list[float] = []
        all_val_targets: list[float] = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss_sum += float(loss.item()) * len(batch_x)
                val_count += len(batch_x)

                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                targets = batch_y.squeeze(-1).cpu().numpy()
                all_val_probs.extend(probs.tolist() if probs.ndim == 1 else [float(probs)])
                all_val_targets.extend(targets.tolist() if targets.ndim == 1 else [float(targets)])

        val_loss = val_loss_sum / max(1, val_count)
        val_targets_arr = np.array(all_val_targets)
        val_probs_arr = np.array(all_val_probs)

        val_auc = compute_auc(val_targets_arr, val_probs_arr)
        best_threshold, _ = find_best_threshold(val_targets_arr, val_probs_arr)
        val_pred = (val_probs_arr >= best_threshold).astype(int)
        val_acc = (
            float(np.mean(val_pred == val_targets_arr))
            if len(val_targets_arr) > 0
            else 0.0
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} - Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Threshold: {best_threshold:.4f}"
        )

        with open(metrics_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_auc:.6f}",
                f"{val_acc:.6f}",
                f"{best_threshold:.6f}",
            ])

        # Checkpoints
        last_ckpt = {
            "model": model.state_dict(),
            "epoch": int(epoch),
            "val_auc": float(val_auc),
            "threshold": float(best_threshold),
            "pretrained_loaded": bool(model.pretrained_loaded),
            "args": sanitize_args(args),
        }
        torch.save(last_ckpt, out_dir / "last.pt")

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_ckpt = {
                "model": model.state_dict(),
                "epoch": int(epoch),
                "val_auc": float(val_auc),
                "threshold": float(best_threshold),
                "pretrained_loaded": bool(model.pretrained_loaded),
                "args": sanitize_args(args),
            }
            torch.save(best_ckpt, out_dir / "best.pt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
