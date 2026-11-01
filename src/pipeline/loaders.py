"""
Model loading and volume preprocessing utilities for the CAD inference pipeline.
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch

from src.classification.model import build_classifier
from src.preprocessing.preprocess import load_nifti, resample_image
from src.segmentation.model import build_unet


def preprocess_ct_volume(path: Union[str, Path]) -> np.ndarray:
    """
    Load and preprocess a 3D CT volume into float32 (z, y, x) with intensities in [0, 1].

    Accepts .nii / .nii.gz files (resampled to 1.0 x 1.0 x 2.5 mm physical spacing and
    normalized using fixed window [-100, 240] -> (clip + 100) / 340.0) or preprocessed
    .npz files containing an 'image' key (converting float16 to float32).

    Parameters
    ----------
    path : Union[str, Path]
        Path to the CT volume (.nii, .nii.gz, or .npz).

    Returns
    -------
    np.ndarray
        Preprocessed float32 array of shape (z, y, x) in [0, 1].

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the file format is unsupported or required keys are missing.
    """
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"CT volume file not found: {input_path}")

    filename_lower = input_path.name.lower()
    if filename_lower.endswith(".npz"):
        with np.load(input_path) as data:
            if "image" not in data:
                raise ValueError(f"NPZ file does not contain 'image' key: {input_path}")
            img_arr = data["image"].astype(np.float32)
            return img_arr

    if filename_lower.endswith(".nii") or filename_lower.endswith(".nii.gz"):
        img = load_nifti(input_path)
        img_res = resample_image(img, target_spacing=(1.0, 1.0, 2.5), is_label=False)
        img_arr = sitk.GetArrayFromImage(img_res).astype(np.float32)

        # Apply same fixed window as process_case: [-100, 240] -> (clip + 100) / 340.0
        img_arr = np.clip(img_arr, -100.0, 240.0)
        img_arr = (img_arr + 100.0) / 340.0
        return img_arr.astype(np.float32)

    raise ValueError(f"Unsupported CT volume format: {input_path}")


def load_seg_model(
    path: Union[str, Path],
    device: Union[str, torch.device],
) -> torch.nn.Module:
    """
    Rebuild 3D UNet and load weights from checkpoint in eval mode.

    Parameters
    ----------
    path : Union[str, Path]
        Path to segmentation checkpoint (.pt).
    device : Union[str, torch.device]
        Device to map loaded tensors to.

    Returns
    -------
    torch.nn.Module
        3D UNet model configured in eval mode on target device with .iteration set.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {ckpt_path}")

    dev = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
    ckpt_args = ckpt.get("args") or {}
    channels = ckpt_args.get("channels", (16, 32, 64, 128, 256))
    strides = ckpt_args.get("strides", (2, 2, 2, 2))
    num_res_units = ckpt_args.get("num_res_units", 2)

    model = build_unet(
        channels=tuple(channels),
        strides=tuple(strides),
        num_res_units=int(num_res_units),
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.iteration = ckpt.get("iteration")
    return model


def load_cls_model(
    path: Union[str, Path],
    device: Union[str, torch.device],
) -> Tuple[torch.nn.Module, float]:
    """
    Rebuild ResNet18 classifier and load weights from checkpoint in eval mode.

    Parameters
    ----------
    path : Union[str, Path]
        Path to classification checkpoint (.pt).
    device : Union[str, torch.device]
        Device to map loaded tensors to.

    Returns
    -------
    Tuple[torch.nn.Module, float]
        Loaded classifier in eval mode on device, and stored validation threshold.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Classification checkpoint not found: {ckpt_path}")

    dev = torch.device(device)
    model = build_classifier(pretrained=False).to(dev)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if "threshold" in ckpt and ckpt["threshold"] is not None:
        threshold = float(ckpt["threshold"])
        threshold_source = "checkpoint"
    else:
        threshold = 0.5
        threshold_source = "default"

    model.threshold_source = threshold_source
    return model, threshold
