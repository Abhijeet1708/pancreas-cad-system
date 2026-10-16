"""
Resumable training pipeline for 3D pancreas and tumor segmentation.

Supports Kaggle GPU environments with CUDA AMP and CPU fallback for testing and CI.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.segmentation.checkpoint import load_checkpoint, save_checkpoint
from src.segmentation.data import (
    DEFAULT_PATCH_SIZE,
    augment_intensity,
    load_case,
    load_split_ids,
    sample_patch,
)
from src.segmentation.metrics import mean_over_cases, region_dice
from src.segmentation.model import build_loss, build_unet, predict_volume


def sanitize_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Convert argparse namespace to a dictionary containing only standard Python types.

    Converts Path objects to string representations.
    """
    cleaned: Dict[str, Any] = {}
    for key, val in vars(args).items():
        if isinstance(val, Path):
            cleaned[key] = str(val)
        elif isinstance(val, (list, tuple)):
            cleaned[key] = [str(item) if isinstance(item, Path) else item for item in val]
        else:
            cleaned[key] = val
    return cleaned


def select_val_cases(val_ids: Sequence[str], num_cases: int) -> List[str]:
    """
    Select validation case IDs.

    Parameters
    ----------
    val_ids : Sequence[str]
        List of all validation case identifiers.
    num_cases : int
        Number of cases to pick. If <= 0 or >= len(val_ids), all sorted cases are returned.
        Otherwise, selects num_cases evenly spaced across sorted val_ids using
        indices from round(linspace(0, len - 1, num_cases)).

    Returns
    -------
    List[str]
        Selected case identifiers.
    """
    sorted_ids = sorted(val_ids)
    total = len(sorted_ids)
    if num_cases <= 0 or num_cases >= total:
        return sorted_ids
    indices = [int(round(idx)) for idx in np.linspace(0, total - 1, num_cases)]
    return [sorted_ids[i] for i in indices]


class PatchDataset(Dataset):
    """
    Dataset generating 3D patches for segmentation training.

    An item loads one train case using load_case and samples patches_per_case patches
    using sample_patch (class_ratios 1, 1, 2) and augment_intensity.
    """

    def __init__(
        self,
        case_ids: Sequence[str],
        data_root: Union[str, Path],
        patches_per_case: int = 2,
        patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE,
        seed: int = 42,
        start_iteration: int = 0,
        batch_cases: int = 1,
        max_iterations: Optional[int] = None,
    ) -> None:
        if not case_ids:
            raise ValueError("case_ids sequence must not be empty")
        self.case_ids = list(case_ids)
        self.data_root = Path(data_root)
        self.patches_per_case = int(patches_per_case)
        self.patch_size = tuple(patch_size)
        self.seed = int(seed)
        self.start_iteration = int(start_iteration)
        self.batch_cases = int(batch_cases)
        self.max_iterations = max_iterations

    def __len__(self) -> int:
        if self.max_iterations is not None:
            remaining = max(0, self.max_iterations - self.start_iteration)
            return remaining * self.batch_cases
        return len(self.case_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        global_sample_index = (self.start_iteration * self.batch_cases) + index
        rng = np.random.default_rng((self.seed, global_sample_index))

        case_idx = int(rng.integers(0, len(self.case_ids)))
        case_id = self.case_ids[case_idx]
        fname = case_id if case_id.endswith(".npz") else f"{case_id}.npz"
        case_path = self.data_root / fname

        image, label = load_case(case_path)

        patches_img: List[np.ndarray] = []
        patches_lbl: List[np.ndarray] = []

        for _ in range(self.patches_per_case):
            p_img, p_lbl = sample_patch(
                image,
                label,
                patch_size=self.patch_size,
                rng=rng,
                class_ratios=(1, 1, 2),
            )
            p_img = augment_intensity(p_img, rng=rng)
            patches_img.append(p_img[np.newaxis, ...])
            patches_lbl.append(p_lbl)

        img_stack = np.stack(patches_img, axis=0)  # (K, 1, z, y, x)
        lbl_stack = np.stack(patches_lbl, axis=0)  # (K, z, y, x)

        return (
            torch.from_numpy(img_stack).float(),
            torch.from_numpy(lbl_stack).long(),
        )


def compute_lr(
    iteration: int,
    max_iterations: int,
    base_lr: float,
    warmup_iterations: int = 100,
) -> float:
    """
    Compute current learning rate via linear warmup and cosine decay to zero.

    Parameters
    ----------
    iteration : int
        Current iteration index (1-indexed).
    max_iterations : int
        Total training iterations.
    base_lr : float
        Peak learning rate.
    warmup_iterations : int, default=100
        Number of warmup iterations.

    Returns
    -------
    float
        Computed learning rate value.
    """
    if max_iterations <= 0 or iteration <= 0:
        return 0.0

    eff_warmup = (
        min(warmup_iterations, max_iterations // 2)
        if max_iterations < warmup_iterations * 2
        else warmup_iterations
    )
    if eff_warmup > 0 and iteration <= eff_warmup:
        return float(base_lr * (iteration / eff_warmup))

    if iteration >= max_iterations:
        return 0.0

    decay_steps = max_iterations - eff_warmup
    if decay_steps <= 0:
        return 0.0

    progress = (iteration - eff_warmup) / decay_steps
    return float(base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)))


def validate(
    model: torch.nn.Module,
    val_ids: Sequence[str],
    data_root: Union[str, Path],
    patch_size: Tuple[int, int, int],
    device: torch.device,
    sw_batch_size: int = 2,
) -> Tuple[float, float, float]:
    """
    Evaluate validation cases and compute average pancreas/tumor Dice scores.

    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model.
    val_ids : Sequence[str]
        Case IDs to evaluate.
    data_root : Union[str, Path]
        Directory containing .npz cases.
    patch_size : Tuple[int, int, int]
        Patch size for sliding window inference.
    device : torch.device
        Compute device.
    sw_batch_size : int, default=2
        Sliding-window batch size.

    Returns
    -------
    Tuple[float, float, float]
        (pancreas_dice, tumor_dice, overall_score) where score is (pancreas + tumor) / 2.
    """
    root = Path(data_root)
    metrics_list: List[Dict[str, float]] = []
    was_training = model.training
    model.eval()

    try:
        for case_id in val_ids:
            fname = case_id if case_id.endswith(".npz") else f"{case_id}.npz"
            image, label = load_case(root / fname)
            pred = predict_volume(
                model=model,
                image=image,
                patch_size=patch_size,
                device=device,
                sw_batch_size=sw_batch_size,
            )
            metrics_list.append(region_dice(pred, label))

        summary = mean_over_cases(metrics_list)
        pancreas_dice = float(summary["pancreas"])
        tumor_dice = float(summary["tumor"])
        score = float((pancreas_dice + tumor_dice) / 2.0)
        return pancreas_dice, tumor_dice, score
    finally:
        if was_training:
            model.train()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Pancreas/tumor 3D segmentation training")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory of .npz cases")
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("docs/splits.json"),
        help="Path to splits JSON file",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20000,
        help="Maximum training iterations",
    )
    parser.add_argument("--batch-cases", type=int, default=1, help="Number of cases per batch")
    parser.add_argument(
        "--patches-per-case",
        type=int,
        default=2,
        help="Number of patches sampled per case",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument(
        "--val-every",
        type=int,
        default=1000,
        help="Validation evaluation frequency",
    )
    parser.add_argument(
        "--val-cases",
        type=int,
        default=0,
        help="Number of val cases to evaluate (0 = all)",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=0.0,
        help="Time budget in minutes (0 = no limit)",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=list(DEFAULT_PATCH_SIZE),
        help="3D patch size (z y x)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
        help="UNet channels",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[2, 2, 2, 2],
        help="UNet downsampling strides",
    )
    parser.add_argument(
        "--num-res-units",
        type=int,
        default=2,
        help="Number of residual units per block",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable automatic mixed precision (default: on if CUDA is available)",
    )

    return parser.parse_args(argv)


def train(args: argparse.Namespace) -> int:
    """Execute model training and validation loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp is None:
        amp = torch.cuda.is_available()
    else:
        amp = args.amp and torch.cuda.is_available()

    patch_size = (int(args.patch_size[0]), int(args.patch_size[1]), int(args.patch_size[2]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_args = sanitize_args(args)
    config_path = out_dir / "run_config.json"
    with open(config_path, mode="w", encoding="utf-8") as f:
        json.dump(clean_args, f, indent=2, default=str)

    train_ids = load_split_ids(args.splits, "train")
    all_val_ids = load_split_ids(args.splits, "val")
    val_ids = select_val_cases(all_val_ids, args.val_cases)

    model = build_unet(
        channels=args.channels,
        strides=args.strides,
        num_res_units=args.num_res_units,
    ).to(device)

    loss_fn = build_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_iteration = 0
    best_metric = -1.0
    history: List[Dict[str, Any]] = []

    resume_path: Optional[Path] = None
    if args.resume_from is not None and Path(args.resume_from).is_file():
        resume_path = Path(args.resume_from)
    elif (out_dir / "last.pt").is_file():
        resume_path = out_dir / "last.pt"

    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_iteration = int(ckpt["iteration"])
        best_metric = float(ckpt["best_metric"])
        history = list(ckpt.get("history", []))
        print(
            f"Resumed from {resume_path} at iteration {start_iteration} "
            f"(best_metric={best_metric:.4f})"
        )

    if start_iteration >= args.max_iterations:
        print(
            f"Start iteration ({start_iteration}) already reached "
            f"max_iterations ({args.max_iterations}). Nothing to train."
        )
        return 0

    train_dataset = PatchDataset(
        case_ids=train_ids,
        data_root=args.data_root,
        patches_per_case=args.patches_per_case,
        patch_size=patch_size,
        seed=args.seed,
        start_iteration=start_iteration,
        batch_cases=args.batch_cases,
        max_iterations=args.max_iterations,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_cases,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    csv_path = out_dir / "metrics.csv"
    running_losses: List[float] = []
    start_time = time.monotonic()
    last_validated_iteration = -1

    def do_validation(current_iter: int, current_lr: float, elapsed_mins: float) -> None:
        nonlocal best_metric, last_validated_iteration
        p_dice, t_dice, score = validate(
            model=model,
            val_ids=val_ids,
            data_root=args.data_root,
            patch_size=patch_size,
            device=device,
        )
        mean_loss = float(np.mean(running_losses)) if running_losses else 0.0
        running_losses.clear()

        record = {
            "iteration": int(current_iter),
            "mean_train_loss": float(mean_loss),
            "val_pancreas_dice": float(p_dice),
            "val_tumor_dice": float(t_dice),
            "score": float(score),
            "lr": float(current_lr),
            "elapsed_minutes": float(elapsed_mins),
        }
        history.append(record)

        write_header = not csv_path.is_file()
        with open(csv_path, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if write_header:
                writer.writerow([
                    "iteration",
                    "mean_train_loss",
                    "val_pancreas_dice",
                    "val_tumor_dice",
                    "score",
                    "lr",
                    "elapsed_minutes",
                ])
            writer.writerow([
                record["iteration"],
                f"{record['mean_train_loss']:.6f}",
                f"{record['val_pancreas_dice']:.6f}",
                f"{record['val_tumor_dice']:.6f}",
                f"{record['score']:.6f}",
                f"{record['lr']:.6e}",
                f"{record['elapsed_minutes']:.4f}",
            ])

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "iteration": int(current_iter),
            "best_metric": float(best_metric),
            "history": history,
            "args": clean_args,
        }

        if score > best_metric:
            best_metric = score
            state["best_metric"] = float(best_metric)
            save_checkpoint(out_dir / "best.pt", state)

        save_checkpoint(out_dir / "last.pt", state)
        last_validated_iteration = current_iter

        print(
            f"[Validation] Iter {current_iter}: Pancreas Dice={p_dice:.4f}, "
            f"Tumor Dice={t_dice:.4f}, Score={score:.4f} (best={best_metric:.4f})"
        )

    model.train()
    iter_start_time = time.monotonic()

    for step, (batch_images, batch_labels) in enumerate(train_loader):
        iteration = start_iteration + step + 1
        current_lr = compute_lr(iteration, args.max_iterations, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Flatten (batch_cases, K, ...) to (batch_cases * K, ...)
        b_cases, k_patches, c_in, p_z, p_y, p_x = batch_images.shape
        flat_images = batch_images.view(b_cases * k_patches, c_in, p_z, p_y, p_x).to(device)
        flat_labels = batch_labels.view(b_cases * k_patches, 1, p_z, p_y, p_x).to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=amp):
            outputs = model(flat_images)
            loss = loss_fn(outputs, flat_labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = float(loss.item())
        running_losses.append(loss_val)

        step_duration = max(1e-6, time.monotonic() - iter_start_time)
        it_per_sec = 1.0 / step_duration
        iter_start_time = time.monotonic()

        print(
            f"Iter {iteration}/{args.max_iterations} - loss: {loss_val:.4f} "
            f"- lr: {current_lr:.6e} - {it_per_sec:.2f} it/s"
        )

        elapsed_mins = (time.monotonic() - start_time) / 60.0
        is_val_step = (iteration % args.val_every == 0) or (iteration == args.max_iterations)
        if is_val_step:
            do_validation(iteration, current_lr, elapsed_mins)

        if args.max_minutes > 0.0 and elapsed_mins >= args.max_minutes:
            print(
                f"Time budget reached ({elapsed_mins:.2f} >= {args.max_minutes} min). "
                f"Finishing training at iteration {iteration}."
            )
            if last_validated_iteration != iteration:
                do_validation(iteration, current_lr, elapsed_mins)
            else:
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "iteration": int(iteration),
                    "best_metric": float(best_metric),
                    "history": history,
                    "args": clean_args,
                }
                save_checkpoint(out_dir / "last.pt", state)
            return 0

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI and module execution entry point."""
    args = parse_args(argv)
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
