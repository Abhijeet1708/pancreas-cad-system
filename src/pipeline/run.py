"""
End-to-end inference pipeline: CT -> 3D segmentation -> 2D slice classification -> Grad-CAM.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from src.classification.model import center_crop, gradcam_target_layer
from src.classification.slices import crop_slice, slice_center
from src.explainability.gradcam import compute_gradcam, overlay_heatmap
from src.pipeline.loaders import load_cls_model, load_seg_model, preprocess_ct_volume
from src.segmentation.model import predict_volume

VOXEL_VOLUME_ML: float = (1.0 * 1.0 * 2.5) / 1000.0  # 0.0025 mL


def clean_for_json(obj: Any) -> Any:
    """Recursively convert numpy types and replace NaN/Inf with None."""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    return obj


def save_slice_figures(
    slices_data: Sequence[Dict[str, Any]],
    figures_dir: Union[str, Path],
) -> None:
    """
    Save 3-panel explanation figures for top slices if matplotlib is available.

    Panels:
    1. CT crop (grayscale)
    2. CT crop with predicted pancreas and tumor tint
    3. CT crop with Grad-CAM heatmap overlay
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping figure generation.")
        return

    if not slices_data:
        return

    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for item in slices_data:
        z = item["z"]
        prob = item["probability"]
        crop = item["crop"]
        mask = item.get("predicted_mask_crop", item.get("crop_mask"))
        cam = item["cam"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: CT crop
        axes[0].imshow(crop, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title(f"CT Crop (z={z})")
        axes[0].axis("off")

        # Panel 2: CT crop with segmentation tint
        axes[1].imshow(crop, cmap="gray", vmin=0.0, vmax=1.0)
        p_mask = np.ma.masked_where(mask != 1, mask)
        t_mask = np.ma.masked_where(mask != 2, mask)
        axes[1].imshow(p_mask, cmap="winter", alpha=0.45)
        axes[1].imshow(t_mask, cmap="autumn", alpha=0.6)
        axes[1].set_title("Predicted Segmentation")
        axes[1].axis("off")

        # Panel 3: CT crop with Grad-CAM overlay
        cam_rgb = overlay_heatmap(crop, cam, alpha=0.4)
        axes[2].imshow(cam_rgb)
        axes[2].set_title(f"Grad-CAM (Prob: {prob:.4f})")
        axes[2].axis("off")

        fig.suptitle(f"Slice {z:03d} (Probability: {prob:.4f})", fontsize=14, y=0.98)
        fig.text(
            0.5,
            0.02,
            "Research prototype - not for clinical use",
            ha="center",
            fontsize=10,
            style="italic",
            color="red",
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig_path = fig_dir / f"slice_{z:03d}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def analyze_volume(
    image: np.ndarray,
    seg_model: torch.nn.Module,
    cls_model: torch.nn.Module,
    threshold: float = 0.5,
    device: Union[str, torch.device] = "cpu",
    patch_size: Sequence[int] = (64, 128, 128),
    top_k: int = 3,
    min_tumor_voxels: int = 100,
) -> Dict[str, Any]:
    """
    Analyze preprocessed 3D CT volume without disk I/O.

    Performs 3D segmentation, evaluates axial slices containing pancreas with the
    2D slice classifier, and computes Grad-CAM for the top-k highest-probability slices.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed float32 volume of shape (z, y, x) in [0, 1].
    seg_model : torch.nn.Module
        3D segmentation UNet model in eval mode.
    cls_model : torch.nn.Module
        2D slice classifier in eval mode.
    threshold : float, default=0.5
        Slice classification decision threshold.
    device : Union[str, torch.device], default="cpu"
        Compute device.
    patch_size : Sequence[int], default=(64, 128, 128)
        Sliding-window patch size (z, y, x).
    top_k : int, default=3
        Number of top slices for Grad-CAM explanation.
    min_tumor_voxels : int, default=100
        Minimum tumor voxels required to flag segmentation tumor detection.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - "labels": uint8 array of predicted voxel classes (0: bg, 1: pancreas, 2: tumor).
        - "result": JSON-serializable dictionary with quantitative findings.
        - "slices": per-slice data for top slices (z, probability, crop, mask, cam).
    """
    dev = torch.device(device)
    t_start = time.perf_counter()

    # Stage 1: Segmentation
    t_seg = time.perf_counter()
    labels = predict_volume(
        model=seg_model,
        image=image,
        patch_size=tuple(patch_size),
        device=dev,
    )
    seg_duration = time.perf_counter() - t_seg
    print(f"  Segmentation completed in {seg_duration:.2f}s")

    pancreas_voxels = int(np.count_nonzero((labels == 1) | (labels == 2)))
    tumor_voxels = int(np.count_nonzero(labels == 2))
    pancreas_volume_ml = float(pancreas_voxels * VOXEL_VOLUME_ML)
    tumor_volume_ml = float(tumor_voxels * VOXEL_VOLUME_ML)
    segmentation_tumor_found = bool(tumor_voxels >= min_tumor_voxels)

    # Stage 2: Slice Extraction and Classification
    t_cls = time.perf_counter()
    candidate_slices: List[Dict[str, Any]] = []
    for z in range(image.shape[0]):
        slice_lbl = labels[z]
        pancreas_mask = (slice_lbl == 1) | (slice_lbl == 2)
        if int(np.count_nonzero(pancreas_mask)) >= 50:
            center = slice_center(pancreas_mask)
            crop160 = crop_slice(image[z], center, size=160)
            crop128 = center_crop(crop160, size=128)
            mask160 = crop_slice(slice_lbl, center, size=160)
            mask128 = center_crop(mask160, size=128)
            candidate_slices.append({
                "z": int(z),
                "crop": crop128.astype(np.float32),
                "predicted_mask_crop": mask128.astype(np.uint8),
                "crop_mask": mask128.astype(np.uint8),
            })

    top_slices_info: List[Dict[str, Any]] = []
    n_slices_flagged = 0
    max_slice_probability: Optional[float] = None
    all_probs: List[float] = []

    if candidate_slices:
        crops = np.stack([c["crop"] for c in candidate_slices], axis=0)
        cls_model.eval()
        with torch.no_grad():
            batch_size = 64
            for start_idx in range(0, len(crops), batch_size):
                batch_tensor = (
                    torch.from_numpy(crops[start_idx:start_idx + batch_size])
                    .unsqueeze(1)
                    .float()
                    .to(dev)
                )
                logits = cls_model(batch_tensor)
                probs = torch.sigmoid(logits).view(-1).cpu().numpy()
                all_probs.extend(probs.tolist() if probs.ndim == 1 else [float(probs)])

        for item, p in zip(candidate_slices, all_probs):
            item["probability"] = float(p)

        n_slices_flagged = sum(1 for p in all_probs if p >= threshold)
        max_slice_probability = float(max(all_probs))
        candidate_slices.sort(key=lambda s: s["probability"], reverse=True)
        top_candidates = candidate_slices[:top_k]
        top_slices_info = [
            {"z": s["z"], "probability": s["probability"]} for s in top_candidates
        ]
    else:
        top_candidates = []

    cls_duration = time.perf_counter() - t_cls
    print(
        f"  Slice evaluation ({len(candidate_slices)} slices) completed in {cls_duration:.2f}s"
    )

    # Stage 3: Grad-CAM on Top Slices
    t_cam = time.perf_counter()
    if top_candidates:
        target_layer = gradcam_target_layer(cls_model)
        for item in top_candidates:
            tensor_x = torch.from_numpy(item["crop"]).unsqueeze(0).unsqueeze(0).to(dev)
            cam, _ = compute_gradcam(cls_model, tensor_x, target_layer)
            item["cam"] = cam.astype(np.float32)

    cam_duration = time.perf_counter() - t_cam
    print(f"  Grad-CAM computation for {len(top_candidates)} slices: {cam_duration:.2f}s")

    total_seconds = float(time.perf_counter() - t_start)

    # Build Notes
    notes: List[str] = [
        "Research prototype - not for clinical use",
        "The slice classifier is weak; the segmentation output is the primary finding.",
    ]
    if segmentation_tumor_found:
        notes.append(
            f"Segmentation finding: Tumor detected with {tumor_voxels} voxels "
            f"({tumor_volume_ml:.4f} mL), meeting threshold of {min_tumor_voxels} voxels."
        )
    else:
        notes.append(
            f"Segmentation finding: No significant tumor volume detected ({tumor_voxels} voxels, "
            f"{tumor_volume_ml:.4f} mL < {min_tumor_voxels} voxels threshold)."
        )

    if candidate_slices:
        notes.append(
            f"Classifier finding: {n_slices_flagged} of {len(candidate_slices)} evaluated "
            f"slices flagged above threshold {threshold:.4f} "
            f"(maximum probability: {max_slice_probability:.4f})."
        )
    else:
        notes.append(
            "Classifier finding: No axial slices with >= 50 predicted pancreas pixels found; "
            "no slices evaluated."
        )

    result: Dict[str, Any] = {
        "input": None,
        "seg_checkpoint": None,
        "cls_checkpoint": None,
        "seg_iteration": getattr(seg_model, "iteration", None),
        "classifier_threshold": float(threshold),
        "classifier_threshold_source": getattr(cls_model, "threshold_source", "checkpoint"),
        "device": str(dev),
        "seconds": round(total_seconds, 4),
        "image_shape": [int(s) for s in image.shape],
        "pancreas_volume_ml": pancreas_volume_ml,
        "tumor_volume_ml": tumor_volume_ml,
        "segmentation_tumor_found": segmentation_tumor_found,
        "n_slices_evaluated": len(candidate_slices),
        "n_slices_flagged": n_slices_flagged,
        "max_slice_probability": max_slice_probability,
        "top_slices": top_slices_info,
        "notes": notes,
    }

    return {
        "labels": labels.astype(np.uint8),
        "result": result,
        "slices": top_candidates,
    }


def run_pipeline(
    input_path: Union[str, Path],
    seg_checkpoint: Union[str, Path],
    cls_checkpoint: Union[str, Path],
    out_dir: Union[str, Path],
    device: str = "auto",
    patch_size: Sequence[int] = (64, 128, 128),
    top_k: int = 3,
    min_tumor_voxels: int = 100,
) -> int:
    """
    Execute full inference pipeline from disk and write result files.

    Parameters
    ----------
    input_path : Union[str, Path]
        Path to input CT volume (.nii, .nii.gz, or .npz).
    seg_checkpoint : Union[str, Path]
        Path to segmentation checkpoint (.pt).
    cls_checkpoint : Union[str, Path]
        Path to classification checkpoint (.pt).
    out_dir : Union[str, Path]
        Output directory for result.json, prediction.npz, and figures.
    device : str, default="auto"
        Compute device ("auto", "cpu", or "cuda").
    patch_size : Sequence[int], default=(64, 128, 128)
        Sliding-window patch size (z, y, x).
    top_k : int, default=3
        Number of top slices for explanation.
    min_tumor_voxels : int, default=100
        Minimum tumor voxels threshold for segmentation finding.

    Returns
    -------
    int
        0 on success, 2 on missing input or checkpoint file.
    """
    input_p = Path(input_path)
    seg_p = Path(seg_checkpoint)
    cls_p = Path(cls_checkpoint)

    # Safety checks BEFORE creating out_dir
    if not input_p.is_file():
        sys.stderr.write(f"Error: Input file does not exist: {input_p}\n")
        return 2
    if not seg_p.is_file():
        sys.stderr.write(f"Error: Segmentation checkpoint does not exist: {seg_p}\n")
        return 2
    if not cls_p.is_file():
        sys.stderr.write(f"Error: Classification checkpoint does not exist: {cls_p}\n")
        return 2

    # Device selection
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    if dev.type == "cpu":
        print("Warning: Running inference on CPU; full-volume inference can take several minutes.")

    out_p = Path(out_dir)

    t_all = time.perf_counter()

    # Step (a): Preprocess CT volume
    t_load = time.perf_counter()
    image = preprocess_ct_volume(input_p)
    prep_time = time.perf_counter() - t_load
    print(f"[Stage 1/4] Preprocessed CT volume {image.shape} in {prep_time:.2f}s")

    # Step (b): Load models
    t_models = time.perf_counter()
    seg_model = load_seg_model(seg_p, dev)
    cls_model, cls_threshold = load_cls_model(cls_p, dev)
    print(f"[Stage 2/4] Loaded models in {time.perf_counter() - t_models:.2f}s")

    # Analyze volume
    print("[Stage 3/4] Running 3D segmentation, slice classification, and Grad-CAM...")
    analysis = analyze_volume(
        image=image,
        seg_model=seg_model,
        cls_model=cls_model,
        threshold=cls_threshold,
        device=dev,
        patch_size=tuple(patch_size),
        top_k=top_k,
        min_tumor_voxels=min_tumor_voxels,
    )

    # Step (e): Save to out-dir
    t_save = time.perf_counter()
    out_p.mkdir(parents=True, exist_ok=True)

    result = analysis["result"]
    result["input"] = str(input_p)
    result["seg_checkpoint"] = seg_p.name
    result["cls_checkpoint"] = cls_p.name
    result["seconds"] = round(float(time.perf_counter() - t_all), 4)

    # Save prediction.npz
    np.savez_compressed(out_p / "prediction.npz", labels=analysis["labels"])

    # Save result.json
    with open(out_p / "result.json", mode="w", encoding="utf-8") as f:
        json.dump(clean_for_json(result), f, indent=2, allow_nan=False)

    # Save figures
    save_slice_figures(analysis["slices"], out_p / "figures")
    print(f"[Stage 4/4] Saved outputs to {out_p} in {time.perf_counter() - t_save:.2f}s")
    print(f"Total pipeline execution time: {time.perf_counter() - t_all:.2f}s")

    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end pancreas and tumor CAD inference pipeline."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CT volume file")
    parser.add_argument(
        "--seg-checkpoint",
        type=Path,
        required=True,
        help="Path to segmentation model checkpoint (.pt)",
    )
    parser.add_argument(
        "--cls-checkpoint",
        type=Path,
        required=True,
        help="Path to classification model checkpoint (.pt)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=[64, 128, 128],
        help="Patch size for 3D sliding window inference (z y x)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top probability slices for Grad-CAM explanations",
    )
    parser.add_argument(
        "--min-tumor-voxels",
        type=int,
        default=100,
        help="Minimum tumor voxels required to flag segmentation finding",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for python -m src.pipeline.run."""
    args = parse_args(argv)
    return run_pipeline(
        input_path=args.input,
        seg_checkpoint=args.seg_checkpoint,
        cls_checkpoint=args.cls_checkpoint,
        out_dir=args.out_dir,
        device=args.device,
        patch_size=args.patch_size,
        top_k=args.top_k,
        min_tumor_voxels=args.min_tumor_voxels,
    )


if __name__ == "__main__":
    sys.exit(main())
