"""
Slice-level tumor dataset construction for 2D pancreas classification.

Note
----
Crops are centered on the ground-truth pancreas centroid for training and
evaluation; the downstream inference pipeline will later use the predicted mask.

Conventions
-----------
- Preprocessed case .npz files contain "image" (float16, values in [0, 1]) and
  "label" (uint8, 0: background, 1: pancreas, 2: tumor) with shape (Z, Y, X).
- Axial slices (axis 0) with at least `min_pancreas_px` pancreas voxels
  (label 1 or 2) are processed.
- Positives (y = 1) must have at least `min_tumor_px` tumor voxels inside the
  central 96 x 96 window of the 160 x 160 crop.
- Negatives (y = 0) have zero tumor voxels (t == 0) over the entire slice.
- Slices containing tumor that fail the positive rule are skipped to eliminate
  label noise caused by spatial cropping.
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.segmentation.data import load_case, load_split_ids


def slice_center(mask2d: np.ndarray) -> Tuple[int, int]:
    """
    Compute (y, x) rounded centroid of nonzero pixels in a 2D mask.

    Parameters
    ----------
    mask2d : np.ndarray
        2D boolean or integer mask of shape (H, W).

    Returns
    -------
    Tuple[int, int]
        (y, x) rounded centroid coordinate. If mask has no nonzero pixels,
        returns the geometric center of the mask.
    """
    coords = np.argwhere(mask2d > 0)
    if coords.size == 0:
        return int(mask2d.shape[0] // 2), int(mask2d.shape[1] // 2)

    mean_y = float(np.mean(coords[:, 0]))
    mean_x = float(np.mean(coords[:, 1]))
    return int(round(mean_y)), int(round(mean_x))


def crop_slice(
    image2d: np.ndarray,
    center_yx: Tuple[int, int],
    size: int = 160,
) -> np.ndarray:
    """
    Extract a size x size window centered on center_yx, shifted to stay inside
    the image; zero-pads symmetrically if the slice is smaller than size.

    Parameters
    ----------
    image2d : np.ndarray
        2D input array of shape (H, W).
    center_yx : Tuple[int, int]
        Desired center (y, x) coordinate.
    size : int, default=160
        Square crop edge length.

    Returns
    -------
    np.ndarray
        Cropped 2D array of shape (size, size) with the same dtype as image2d.
    """
    h, w = image2d.shape[:2]

    pad_top = max(0, (size - h) // 2)
    pad_bottom = max(0, size - h - pad_top)
    pad_left = max(0, (size - w) // 2)
    pad_right = max(0, size - w - pad_left)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        padded = np.pad(
            image2d,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        cy = center_yx[0] + pad_top
        cx = center_yx[1] + pad_left
    else:
        padded = image2d
        cy, cx = center_yx

    cur_h, cur_w = padded.shape[:2]
    start_y = max(0, min(cy - size // 2, cur_h - size))
    start_x = max(0, min(cx - size // 2, cur_w - size))

    return padded[start_y:start_y + size, start_x:start_x + size].copy()


def build_slice_arrays(
    case_ids: Sequence[str],
    data_root: Union[str, Path],
    crop: int = 160,
    min_pancreas_px: int = 50,
    min_tumor_px: int = 30,
    return_stats: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]],
]:
    """
    Extract slice-level arrays across a set of cases.

    Parameters
    ----------
    case_ids : Sequence[str]
        List of case identifiers.
    data_root : Union[str, Path]
        Path to directory containing case .npz files.
    crop : int, default=160
        Crop window size.
    min_pancreas_px : int, default=50
        Minimum pancreas pixels required to process an axial slice.
    min_tumor_px : int, default=30
        Minimum tumor pixels required inside the central 96 x 96 window for y=1.
    return_stats : bool, default=False
        Whether to return an additional dict containing extraction statistics.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X, y, case, z):
        - X: float16 array of shape (N, crop, crop)
        - y: uint8 binary label array of shape (N,)
        - case: string array of shape (N,)
        - z: int32 axial slice indices of shape (N,)
        If return_stats=True, also returns dict of slice counts.
    """
    data_dir = Path(data_root)
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    case_list: List[str] = []
    z_list: List[int] = []

    n_pos = 0
    n_neg = 0
    n_skipped = 0

    inner_size = 96
    inner_start = (crop - inner_size) // 2
    inner_end = inner_start + inner_size

    total_cases = len(case_ids)
    for i, raw_cid in enumerate(case_ids, start=1):
        cid = raw_cid[:-4] if raw_cid.endswith(".npz") else raw_cid
        fname = f"{cid}.npz"
        case_path = data_dir / fname

        image, label = load_case(case_path)
        case_pos = 0
        case_neg = 0
        case_skip = 0

        for z in range(image.shape[0]):
            slice_lbl = label[z]
            pancreas_mask = (slice_lbl == 1) | (slice_lbl == 2)
            if int(np.count_nonzero(pancreas_mask)) < min_pancreas_px:
                continue

            center = slice_center(pancreas_mask)
            t_whole = int(np.count_nonzero(slice_lbl == 2))

            if t_whole == 0:
                y_val = 0
            else:
                cropped_lbl = crop_slice(slice_lbl, center, size=crop)
                inner_lbl = cropped_lbl[inner_start:inner_end, inner_start:inner_end]
                t_inner = int(np.count_nonzero(inner_lbl == 2))
                if t_inner >= min_tumor_px:
                    y_val = 1
                else:
                    n_skipped += 1
                    case_skip += 1
                    continue

            slice_img = image[z]
            cropped_img = crop_slice(slice_img, center, size=crop).astype(np.float16)

            x_list.append(cropped_img)
            y_list.append(y_val)
            case_list.append(cid)
            z_list.append(z)

            if y_val == 1:
                n_pos += 1
                case_pos += 1
            else:
                n_neg += 1
                case_neg += 1

        print(
            f"[{i}/{total_cases}] {cid}: extracted {case_pos} pos, "
            f"{case_neg} neg, {case_skip} skipped"
        )

    if x_list:
        x_arr = np.stack(x_list, axis=0).astype(np.float16)
        y_arr = np.array(y_list, dtype=np.uint8)
        case_arr = np.array(case_list, dtype=str)
        z_arr = np.array(z_list, dtype=np.int32)
    else:
        x_arr = np.empty((0, crop, crop), dtype=np.float16)
        y_arr = np.empty((0,), dtype=np.uint8)
        case_arr = np.empty((0,), dtype=str)
        z_arr = np.empty((0,), dtype=np.int32)

    stats = {
        "pos": n_pos,
        "neg": n_neg,
        "skipped": n_skipped,
        "cases": total_cases,
    }

    if return_stats:
        return x_arr, y_arr, case_arr, z_arr, stats
    return x_arr, y_arr, case_arr, z_arr


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for slice extraction."""
    parser = argparse.ArgumentParser(
        description="Extract 2D slice classification datasets from CT volumes."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Directory of .npz cases")
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("docs/splits.json"),
        help="Path to splits JSON file",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--crop", type=int, default=160, help="Square crop window size")
    parser.add_argument(
        "--min-pancreas-px",
        type=int,
        default=50,
        help="Minimum pancreas pixels required to process an axial slice",
    )
    parser.add_argument(
        "--min-tumor-px",
        type=int,
        default=30,
        help="Minimum tumor pixels inside central 96x96 window for y=1",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for slice dataset creation."""
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        case_ids = load_split_ids(args.splits, split)
        print(f"\nProcessing split '{split}' ({len(case_ids)} cases)...")
        x_arr, y_arr, case_arr, z_arr, stats = build_slice_arrays(
            case_ids=case_ids,
            data_root=args.data_root,
            crop=args.crop,
            min_pancreas_px=args.min_pancreas_px,
            min_tumor_px=args.min_tumor_px,
            return_stats=True,
        )
        out_path = out_dir / f"cls_{split}.npz"
        np.savez_compressed(out_path, X=x_arr, y=y_arr, case=case_arr, z=z_arr)
        print(
            f"Split '{split}': {len(case_ids)} cases, {stats['pos']} positive, "
            f"{stats['neg']} negative, {stats['skipped']} skipped slices. "
            f"Saved to {out_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
