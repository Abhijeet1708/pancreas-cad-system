"""
Display and summary formatting utilities for the Pancreas CAD Streamlit application.

Dependencies: pure numpy and standard library only (no streamlit, torch, or matplotlib).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

DISCLAIMER: str = (
    "Research prototype - not for clinical use. Not clinically validated."
)

LIMITATIONS_MD: str = """\
### Limitations and Model Performance

- **Evaluation Split:** Evaluated on the 42-case test split of the MSD Task07 Pancreas
  dataset. Single public dataset; **no external validation** has been performed.
- **3D Segmentation Performance:**
  - Pancreas Dice: **0.762**
  - Tumor Dice: **0.323**
  - Tumor detection rate: **28 of 42 cases (67%)**
    - Small tumors (< 20 mm): **5 of 14 cases** (35.7%)
    - Large tumors (>= 20 mm): **23 of 28 cases** (82.1%)
- **2D Slice Classifier Performance:**
  - Test ROC AUC: **0.666 (weak)**. Slice-level classification serves only as a secondary check.
- **Dataset Limitation:** Every case in the source dataset contains a tumor; therefore,
  **specificity is unknown**.
- **Research Notice:** Research prototype - not for clinical use. Not clinically validated.
"""


def list_example_results(
    root: Union[str, Path]
) -> List[Dict[str, Any]]:
    """
    Discover precomputed pipeline results under root.

    Parameters
    ----------
    root : Union[str, Path]
        Directory containing example case subdirectories.

    Returns
    -------
    List[Dict[str, Any]]
        List of dicts each with keys:
        - "name": subdirectory name
        - "result": parsed dictionary from result.json
        - "figures": sorted list of Path objects to PNG/JPG figures
        Returns [] if root does not exist. Skips subdirectories without result.json.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []

    examples: List[Dict[str, Any]] = []
    for entry in sorted(root_path.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue

        json_path = entry / "result.json"
        if not json_path.is_file():
            continue

        try:
            with open(json_path, mode="r", encoding="utf-8") as f:
                result_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Look for figures in entry / "figures" or entry directly
        fig_dir = entry / "figures"
        figures: List[Path] = []
        if fig_dir.is_dir():
            for p in sorted(fig_dir.iterdir(), key=lambda x: x.name):
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    figures.append(p)

        if not figures:
            for p in sorted(entry.iterdir(), key=lambda x: x.name):
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    figures.append(p)

        examples.append({
            "name": entry.name,
            "result": result_data,
            "figures": figures,
        })

    return examples


def summary_rows(
    result: Dict[str, Any]
) -> List[Tuple[str, str]]:
    """
    Format key CAD metrics into label-value rows for display.

    Parameters
    ----------
    result : Dict[str, Any]
        Pipeline result dictionary (from result.json or analyze_volume).

    Returns
    -------
    List[Tuple[str, str]]
        List of (label, formatted_value_string) pairs for:
        1. Pancreas volume (mL)
        2. Tumor volume (mL)
        3. Segmentation finding ("Found" / "Not found")
        4. Slices evaluated
        5. Slices flagged
        6. Max slice probability
        7. Runtime seconds
        8. Device
    """
    p_vol = result.get("pancreas_volume_ml")
    t_vol = result.get("tumor_volume_ml")
    seg_found = result.get("segmentation_tumor_found")
    n_eval = result.get("n_slices_evaluated")
    n_flag = result.get("n_slices_flagged")
    max_p = result.get("max_slice_probability")
    sec = result.get("seconds")
    dev = result.get("device")

    p_vol_str = f"{float(p_vol):.2f} mL" if p_vol is not None else "N/A"
    t_vol_str = f"{float(t_vol):.2f} mL" if t_vol is not None else "N/A"

    if seg_found is True:
        seg_str = "Found"
    elif seg_found is False:
        seg_str = "Not found"
    else:
        seg_str = "N/A"

    n_eval_str = str(n_eval) if n_eval is not None else "N/A"
    n_flag_str = str(n_flag) if n_flag is not None else "N/A"
    max_p_str = f"{float(max_p):.4f}" if max_p is not None else "N/A"
    sec_str = f"{float(sec):.2f} s" if sec is not None else "N/A"
    dev_str = str(dev) if dev is not None else "unknown"

    return [
        ("Pancreas volume", p_vol_str),
        ("Tumor volume", t_vol_str),
        ("Segmentation finding", seg_str),
        ("Slices evaluated", n_eval_str),
        ("Slices flagged", n_flag_str),
        ("Max slice probability", max_p_str),
        ("Runtime seconds", sec_str),
        ("Device", dev_str),
    ]


def tint_overlay(
    image2d: np.ndarray,
    labels2d: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Produce an RGB uint8 image with pancreas (yellow) and tumor (red) color overlays.

    Parameters
    ----------
    image2d : np.ndarray
        2D axial CT slice (float in [0, 1] or uint8 in [0, 255]).
    labels2d : np.ndarray
        2D label slice with 0: background, 1: pancreas, 2: tumor.
    alpha : float, default=0.4
        Opacity of the color overlays in [0, 1].

    Returns
    -------
    np.ndarray
        RGB uint8 array of shape (H, W, 3). Pure numpy implementation.
    """
    img = np.asarray(image2d)
    lbl = np.asarray(labels2d)

    if img.ndim != 2:
        img = np.squeeze(img)
    if lbl.ndim != 2:
        lbl = np.squeeze(lbl)

    if img.ndim != 2 or lbl.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays, got image shape {image2d.shape} and label shape {labels2d.shape}"
        )
    if img.shape != lbl.shape:
        raise ValueError(
            f"Shape mismatch: image shape {img.shape} != label shape {lbl.shape}"
        )

    if np.issubdtype(img.dtype, np.floating):
        if float(img.max()) <= 1.0:
            gray = np.clip(img * 255.0, 0.0, 255.0).astype(np.float32)
        else:
            gray = np.clip(img, 0.0, 255.0).astype(np.float32)
    else:
        gray = np.clip(img.astype(np.float32), 0.0, 255.0)

    rgb = np.stack([gray, gray, gray], axis=-1)

    p_mask = (lbl == 1)
    t_mask = (lbl == 2)

    # Color targets: Pancreas = Blue [0, 0, 255], Tumor = Red [255, 0, 0]
    # Matches save_slice_figures (cmap="winter" for pancreas, "autumn" for tumor).
    blue = np.array([0.0, 0.0, 255.0], dtype=np.float32)
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)

    out = rgb.copy()
    if np.any(p_mask):
        out[p_mask] = (1.0 - alpha) * rgb[p_mask] + alpha * blue
    if np.any(t_mask):
        out[t_mask] = (1.0 - alpha) * rgb[t_mask] + alpha * red

    return np.clip(np.round(out), 0.0, 255.0).astype(np.uint8)


def format_finding_messages(
    result: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Format separate findings for 3D segmentation and 2D slice classification.

    Parameters
    ----------
    result : Dict[str, Any]
        Pipeline result dictionary.

    Returns
    -------
    Tuple[str, str]
        (segmentation_finding_message, classifier_finding_message).
        Kept strictly distinct; never combined into a single clinical claim and
        does not contain the word 'diagnosis'.
    """
    seg_found = result.get("segmentation_tumor_found", False)
    t_vol = result.get("tumor_volume_ml", 0.0)
    p_vol = result.get("pancreas_volume_ml", 0.0)
    t_vol_val = float(t_vol) if t_vol is not None else 0.0
    p_vol_val = float(p_vol) if p_vol is not None else 0.0

    if seg_found:
        seg_msg = (
            f"Segmentation finding: Potential tumor volume detected "
            f"({t_vol_val:.2f} mL, pancreas volume: {p_vol_val:.2f} mL)."
        )
    else:
        seg_msg = (
            f"Segmentation finding: No significant tumor volume detected "
            f"({t_vol_val:.2f} mL, pancreas volume: {p_vol_val:.2f} mL)."
        )

    n_eval = result.get("n_slices_evaluated", 0)
    n_flag = result.get("n_slices_flagged", 0)
    max_p = result.get("max_slice_probability")
    thresh = result.get("classifier_threshold", 0.5)
    thresh_val = float(thresh) if thresh is not None else 0.5

    if n_eval and int(n_eval) > 0:
        max_p_str = f"{float(max_p):.4f}" if max_p is not None else "N/A"
        cls_msg = (
            f"Classifier finding: {int(n_flag)} of {int(n_eval)} evaluated slices flagged "
            f"above threshold {thresh_val:.4f} (maximum slice probability: {max_p_str}). "
            f"Note: Slice classifier is weak (test AUC 0.666) and serves as an exploratory check."
        )
    else:
        cls_msg = (
            "Classifier finding: No axial slices evaluated (insufficient pancreas volume)."
        )

    return seg_msg, cls_msg
