"""
Data utilities for pancreas CT segmentation.

Conventions
-----------
- Preprocessed .npz files contain "image" (float16, values in [0, 1]) and
  "label" (uint8, values 0/1/2).
- Arrays are in (z, y, x) order, spacing 1.0 x 1.0 x 2.5 mm (z is the slice axis).
- Labels:
    0: background
    1: pancreas
    2: tumor
- Dice evaluation regions:
    "pancreas": labels 1 or 2 (whole organ including tumor)
    "tumor": label 2
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

LABEL_BACKGROUND: int = 0
LABEL_PANCREAS: int = 1
LABEL_TUMOR: int = 2
DEFAULT_PATCH_SIZE: Tuple[int, int, int] = (64, 128, 128)


def load_split_ids(splits_path: Union[str, Path], split: str) -> List[str]:
    """
    Load case identifiers for a specified split from a splits JSON file.

    Parameters
    ----------
    splits_path : Union[str, Path]
        Path to the splits JSON file (e.g. docs/splits.json).
    split : str
        Split partition name ("train", "val", or "test").

    Returns
    -------
    List[str]
        List of case identifiers belonging to the specified split.

    Raises
    ------
    FileNotFoundError
        If splits_path does not exist.
    ValueError
        If the split name is unknown or invalid.
    """
    path = Path(splits_path)
    if not path.is_file():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")

    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    if split not in data or split == "seed" or not isinstance(data[split], list):
        valid = [k for k in data.keys() if k != "seed" and isinstance(data[k], list)]
        raise ValueError(f"Unknown split '{split}'. Available splits: {valid}")

    return [str(case_id) for case_id in data[split]]


def load_case(npz_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a preprocessed case volume and label mask from an NPZ file.

    Converts the image from float16 to float32 and keeps the label as uint8.

    Parameters
    ----------
    npz_path : Union[str, Path]
        Path to the preprocessed .npz archive containing 'image' and 'label'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (image, label) where image is float32 with values in [0, 1] and label
        is uint8 with values in {0, 1, 2}.

    Raises
    ------
    FileNotFoundError
        If npz_path does not exist.
    ValueError
        If archive keys are missing, shapes mismatch, or label values are outside {0, 1, 2}.
    """
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(path) as data:
        if "image" not in data or "label" not in data:
            raise ValueError(f"NPZ file must contain 'image' and 'label' arrays: {npz_path}")
        image = data["image"].astype(np.float32)
        label = data["label"].astype(np.uint8)

    if image.shape != label.shape:
        raise ValueError(
            f"Shape mismatch: image shape {image.shape} does not match label shape {label.shape} "
            f"in {npz_path}"
        )

    if np.any((label < 0) | (label > 2)):
        raise ValueError(f"Label array has values outside {{0, 1, 2}} in {npz_path}")

    return image, label


def sample_patch(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE,
    rng: Optional[np.random.Generator] = None,
    class_ratios: Tuple[int, ...] = (1, 1, 2),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a 3D patch from image and label arrays using class-ratio center selection.

    Picks a target class according to class_ratios (background, pancreas, tumor),
    selects a voxel of that class as the patch center (falling back to a uniform random
    center if empty or background search limit reached), and crops a patch of exactly
    patch_size shifted so it stays inside the volume. If the volume is smaller than
    patch_size along an axis, pads with 0.

    Parameters
    ----------
    image : np.ndarray
        Input 3D image array of shape (Z, Y, X).
    label : np.ndarray
        Input 3D label array of shape (Z, Y, X).
    patch_size : Tuple[int, int, int], default=(64, 128, 128)
        Desired patch size in (z, y, x) order.
    rng : Optional[np.random.Generator], default=None
        NumPy random number generator. If None, np.random.default_rng() is used.
    class_ratios : Tuple[int, ...], default=(1, 1, 2)
        Sampling weights for (background, pancreas, tumor).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (image_patch, label_patch) with exact shapes matching patch_size.

    Raises
    ------
    ValueError
        If image and label shapes differ, or class_ratios is invalid.
    """
    if image.shape != label.shape:
        raise ValueError(
            f"Image shape {image.shape} does not match label shape {label.shape}"
        )

    if len(class_ratios) != 3:
        raise ValueError(f"class_ratios must have exactly 3 entries, got {len(class_ratios)}")

    if any(r < 0 for r in class_ratios):
        raise ValueError(f"class_ratios cannot have negative entries, got {class_ratios}")

    total_ratio = sum(class_ratios)
    if total_ratio == 0:
        raise ValueError("class_ratios sum cannot be zero")

    if rng is None:
        rng = np.random.default_rng()

    probs = [r / total_ratio for r in class_ratios]
    classes = [LABEL_BACKGROUND, LABEL_PANCREAS, LABEL_TUMOR]
    chosen_class = int(rng.choice(classes, p=probs))

    if chosen_class == LABEL_BACKGROUND:
        center = tuple(int(rng.integers(0, s)) for s in image.shape)
        if label[center] != LABEL_BACKGROUND:
            for _ in range(49):
                cand = tuple(int(rng.integers(0, s)) for s in image.shape)
                center = cand
                if label[center] == LABEL_BACKGROUND:
                    break
    else:
        flat_indices = np.flatnonzero(label == chosen_class)
        if len(flat_indices) > 0:
            idx = int(rng.choice(flat_indices))
            center = tuple(int(x) for x in np.unravel_index(idx, label.shape))
        else:
            center = tuple(int(rng.integers(0, s)) for s in image.shape)

    pad_width = []
    needs_pad = False
    for s, p in zip(image.shape, patch_size):
        if s < p:
            pad_width.append((0, p - s))
            needs_pad = True
        else:
            pad_width.append((0, 0))

    if needs_pad:
        image = np.pad(image, pad_width, mode="constant", constant_values=0)
        label = np.pad(label, pad_width, mode="constant", constant_values=0)

    slices = []
    for c, s, p in zip(center, image.shape, patch_size):
        start = max(0, min(c - p // 2, s - p))
        end = start + p
        slices.append(slice(start, end))

    image_patch = image[tuple(slices)].copy()
    label_patch = label[tuple(slices)].copy()

    return image_patch, label_patch


def augment_intensity(
    image: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    gaussian_noise: bool = True,
    noise_std: float = 0.01,
) -> np.ndarray:
    """
    Apply intensity scaling, shift, and optional Gaussian noise to an image patch.

    Applies a random scale in [0.9, 1.1], random shift in [-0.05, 0.05], optional
    additive Gaussian noise (std 0.01), and clips output to [0, 1]. Preserves input
    dtype and does not modify the input array.

    Parameters
    ----------
    image : np.ndarray
        Input image patch with values in [0, 1].
    rng : Optional[np.random.Generator], default=None
        NumPy random number generator. If None, np.random.default_rng() is used.
    gaussian_noise : bool, default=True
        Whether to add Gaussian noise.
    noise_std : float, default=0.01
        Standard deviation of Gaussian noise.

    Returns
    -------
    np.ndarray
        Augmented image patch clipped to [0, 1], with the same dtype as image.
    """
    if rng is None:
        rng = np.random.default_rng()

    orig_dtype = image.dtype
    scale = float(rng.uniform(0.9, 1.1))
    shift = float(rng.uniform(-0.05, 0.05))

    out = image.astype(np.float32, copy=True) * scale + shift
    if gaussian_noise and noise_std > 0.0:
        noise = rng.normal(0.0, noise_std, size=image.shape).astype(np.float32)
        out = out + noise

    out = np.clip(out, 0.0, 1.0)
    return out.astype(orig_dtype)
