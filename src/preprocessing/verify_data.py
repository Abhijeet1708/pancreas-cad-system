import argparse
from pathlib import Path
import nibabel as nib
import numpy as np

from src.preprocessing.preprocess import get_config, discover_files


def verify_nifti_files(images_dir: Path, labels_dir: Path, sample_size=None):
    # Pass the raw data root (parent of images/labels)
    cases = discover_files(images_dir.parent)

    case_ids = sorted(list(cases.keys()))

    if not case_ids:
        print(f"No valid cases found in {images_dir.parent}")
        return

    print(f"Found {len(case_ids)} unique cases.")

    if sample_size and 0 < sample_size < len(case_ids):
        case_ids = case_ids[:sample_size]
        print(f"Verifying a sample of {sample_size} cases...")

    for cid in case_ids:
        paths = cases[cid]
        img_path = paths.get("image")
        label_path = paths.get("label")

        if not img_path:
            print(f"\nCase: {cid}")
            print("  [ERROR] No image found for this case.")
            continue

        try:
            img = nib.load(str(img_path))
            img_shape = img.shape
            img_spacing = img.header.get_zooms()

            print(f"\nImage: {img_path.name}")
            print(f"  Shape: {img_shape}")
            print(f"  Spacing: {img_spacing}")

            if label_path:
                lbl = nib.load(str(label_path))
                lbl_data = lbl.get_fdata()
                lbl_shape = lbl.shape
                lbl_spacing = lbl.header.get_zooms()
                unique_labels = np.unique(lbl_data)

                print(f"  Label: {label_path.name}")
                print(f"  Label Shape: {lbl_shape}")
                print(f"  Label Spacing: {lbl_spacing}")
                print(f"  Unique labels: {unique_labels}")

                if img_shape != lbl_shape:
                    print(f"  [ERROR] Shape mismatch between image {img_shape} "
                          f"and label {lbl_shape}")
            else:
                print(f"  [WARNING] No corresponding label found for {img_path.name}")

        except Exception as e:
            print(f"[ERROR] Failed to read {img_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Verify NIfTI files.")
    parser.add_argument('--sample', type=int, default=3,
                        help="Number of samples to verify (default: 3). Set to 0 for all.")
    args = parser.parse_args()

    data_raw_dir, _ = get_config()
    images_dir = data_raw_dir / "images"
    labels_dir = data_raw_dir / "labels"

    verify_nifti_files(images_dir, labels_dir, sample_size=args.sample)


if __name__ == "__main__":
    main()
