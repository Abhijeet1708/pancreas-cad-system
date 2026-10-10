import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import SimpleITK as sitk


def get_config() -> Tuple[Path, Path]:
    """
    Read data and output directories from config.yaml, allowing environment
    variable overrides (PANCREAS_DATA_ROOT, PANCREAS_OUTPUT_DIR).

    Returns:
        Tuple[Path, Path]: The raw data directory and the output processed directory.
    """
    # Resolve config relative to repository root
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config.yaml"

    config = None
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    if config is None:
        config = {}

    raw_dir_str = os.environ.get(
        "PANCREAS_DATA_ROOT", config.get("data", {}).get("raw_dir", "data/raw")
    )
    processed_dir_str = os.environ.get(
        "PANCREAS_OUTPUT_DIR", config.get("data", {}).get("processed_dir", "data/processed")
    )

    raw_dir = Path(raw_dir_str)
    if not raw_dir.is_absolute():
        raw_dir = repo_root / raw_dir

    processed_dir = Path(processed_dir_str)
    if not processed_dir.is_absolute():
        processed_dir = repo_root / processed_dir

    return raw_dir, processed_dir


def get_case_id(filename: str) -> str:
    """
    Extract the case ID from a filename by removing .nii and .nii.gz suffixes.

    Args:
        filename (str): The filename to process.

    Returns:
        str: The extracted case ID.
    """
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    elif filename.endswith(".nii"):
        return filename[:-4]
    return filename


def discover_files(data_root: Path) -> Dict[str, Dict[str, Path]]:
    """
    Find all image and label files in the raw data directory, handling both .nii
    and .nii.gz. Matches them by case ID and ignores files starting with '._'.

    Args:
        data_root (Path): The root directory containing 'images' and 'labels' subdirs.

    Returns:
        Dict[str, Dict[str, Path]]: A dictionary mapping case_id to its image and label paths.
    """
    cases = {}
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"

    # Find images
    if images_dir.exists():
        for p in images_dir.iterdir():
            if p.name.startswith("._") or not (
                p.name.endswith(".nii") or p.name.endswith(".nii.gz")
            ):
                continue
            case_id = get_case_id(p.name)
            if case_id not in cases:
                cases[case_id] = {}
            cases[case_id]["image"] = p

    # Find labels
    if labels_dir.exists():
        for p in labels_dir.iterdir():
            if p.name.startswith("._") or not (
                p.name.endswith(".nii") or p.name.endswith(".nii.gz")
            ):
                continue
            case_id = get_case_id(p.name)
            if case_id not in cases:
                cases[case_id] = {}
            cases[case_id]["label"] = p

    return cases


def load_nifti(filepath: Path) -> sitk.Image:
    """
    Load a NIfTI file using SimpleITK.

    Args:
        filepath (Path): Path to the NIfTI file.

    Returns:
        sitk.Image: The loaded SimpleITK image.
    """
    return sitk.ReadImage(str(filepath))


def resample_image(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 2.5),
    is_label: bool = False
) -> sitk.Image:
    """
    Resample a SimpleITK image to the target spacing.
    Uses linear interpolation for images and nearest-neighbor for labels.
    Calculates output size to preserve physical extent, and keeps origin/direction.

    Args:
        image (sitk.Image): The input image.
        target_spacing (Tuple[float, float, float]): The desired physical spacing (x, y, z).
        is_label (bool): If True, uses nearest neighbor interpolation to preserve integers.

    Returns:
        sitk.Image: The resampled image.
    """
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    # Compute new size to cover the same physical space
    new_size = [
        int(round(orig_sz * orig_spc / targ_spc))
        for orig_sz, orig_spc, targ_spc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1024)

    return resampler.Execute(image)


def process_case(image_path: Path, label_path: Path, output_dir: Path, case_id: str) -> None:
    """
    Process a single case: load, resample, clip HU values to [-100, 240],
    normalize to [0, 1] using fixed window, and save as .npz.

    Args:
        image_path (Path): Path to the input image file.
        label_path (Path): Path to the input label file.
        output_dir (Path): Directory to save the resulting .npz file.
        case_id (str): The case identifier.
    """
    img = load_nifti(image_path)
    lbl = load_nifti(label_path)

    # Resample
    img_res = resample_image(img, is_label=False)
    lbl_res = resample_image(lbl, is_label=True)

    # Extract arrays
    img_arr = sitk.GetArrayFromImage(img_res).astype(np.float32)
    lbl_arr = sitk.GetArrayFromImage(lbl_res).astype(np.uint8)

    if img_arr.shape != lbl_arr.shape:
        raise ValueError(
            f"Shape mismatch after resampling: image {img_arr.shape} vs label {lbl_arr.shape}"
        )

    # Clip and normalize using fixed window [-100, 240]
    # out = (clip(x, -100, 240) + 100) / 340
    img_arr = np.clip(img_arr, -100, 240)
    img_arr = (img_arr + 100) / 340.0
    img_arr = img_arr.astype(np.float16)

    # Save
    tmp_path = output_dir / f"{case_id}.tmp.npz"
    out_path = output_dir / f"{case_id}.npz"
    np.savez_compressed(tmp_path, image=img_arr, label=lbl_arr)
    os.replace(tmp_path, out_path)


def process_batch(data_root: Path, output_dir: Path) -> None:
    """
    Process all matched cases found in data_root and save to output_dir.
    Skips cases that already exist. Warns about unmatched files.
    Reports summary of processed, skipped, and failed cases.

    Args:
        data_root (Path): The root directory for raw data.
        output_dir (Path): The root directory for processed output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Delete leftover temp files
    for tmp_file in output_dir.glob("*.tmp.npz"):
        try:
            tmp_file.unlink()
        except OSError:
            pass

    cases = discover_files(data_root)

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    failed_cases = []

    case_ids = sorted(cases.keys())
    total_cases = len(case_ids)
    print(f"Discovered {total_cases} total unique cases in {data_root}")

    for i, case_id in enumerate(case_ids, 1):
        paths = cases[case_id]
        if "image" not in paths:
            print(f"[{i}/{total_cases}] Warning: Case {case_id} has a label "
                  "but no matching image. Skipping.")
            skipped_count += 1
            continue
        if "label" not in paths:
            print(f"[{i}/{total_cases}] Warning: Case {case_id} has an image "
                  "but no matching label. Skipping.")
            skipped_count += 1
            continue

        out_path = output_dir / f"{case_id}.npz"
        if out_path.exists():
            print(f"[{i}/{total_cases}] Case {case_id} already exists in "
                  "output directory. Skipping.")
            skipped_count += 1
            continue

        try:
            print(f"[{i}/{total_cases}] Processing case {case_id}...")
            process_case(paths["image"], paths["label"], output_dir, case_id)
            processed_count += 1
        except Exception as e:
            print(f"[{i}/{total_cases}] Error processing case {case_id}: {e}")
            failed_count += 1
            failed_cases.append((case_id, str(e)))

    print("\n--- Processing Summary ---")
    print(f"Processed: {processed_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Failed:    {failed_count}")
    if failed_cases:
        print("Failed cases details:")
        for fid, msg in failed_cases:
            print(f"  {fid}: {msg}")


def main() -> None:
    """
    Entry point for the preprocessing script. Parses arguments for data and output
    directories, optionally overriding the config file, and runs the batch process.
    """
    parser = argparse.ArgumentParser(description="Preprocess Pancreas CT Dataset.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data root directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    default_raw, default_out = get_config()

    data_root = Path(args.data_root) if args.data_root else default_raw
    output_dir = Path(args.output_dir) if args.output_dir else default_out

    process_batch(data_root, output_dir)


if __name__ == "__main__":
    main()
