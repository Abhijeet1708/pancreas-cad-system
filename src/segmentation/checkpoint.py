"""
Checkpointing utilities for segmentation model training.
"""

import os
from pathlib import Path
from typing import Any, Dict, Union

import torch


def save_checkpoint(
    path: Union[str, Path],
    state: Dict[str, Any],
) -> None:
    """
    Atomically save a checkpoint dictionary to disk.

    Writes first to `<path>.tmp`, then atomically renames the temporary file
    to the target path via os.replace.

    Parameters
    ----------
    path : Union[str, Path]
        Destination file path.
    state : Dict[str, Any]
        Dictionary holding model, optimizer, scaler, iteration, best_metric,
        history, and args.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(target_path) + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, target_path)


def load_checkpoint(
    path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint dictionary using torch.load default security settings.

    Parameters
    ----------
    path : Union[str, Path]
        Path to checkpoint file.
    map_location : Union[str, torch.device], default="cpu"
        Device to map loaded tensors to.

    Returns
    -------
    Dict[str, Any]
        Loaded checkpoint dictionary.

    Raises
    ------
    FileNotFoundError
        If checkpoint file does not exist.
    """
    target_path = Path(path)
    if not target_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return torch.load(target_path, map_location=map_location)
