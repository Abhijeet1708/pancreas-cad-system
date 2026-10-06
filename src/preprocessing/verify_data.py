import os
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np

def verify_nifti_files(images_dir: Path, labels_dir: Path, sample_size=None):
    images = sorted(list(images_dir.glob("*.nii.gz")))
    labels = sorted(list(labels_dir.glob("*.nii.gz")))
    
    if not images:
        print(f"No images found in {images_dir}")
        return
        
    print(f"Found {len(images)} images and {len(labels)} labels.")
    
    if sample_size and sample_size > 0 and sample_size < len(images):
        images = images[:sample_size]
        print(f"Verifying a sample of {sample_size} images...")
        
    for img_path in images:
        label_path = labels_dir / img_path.name
        
        try:
            img = nib.load(str(img_path))
            img_data = img.get_fdata()
            img_shape = img.shape
            img_spacing = img.header.get_zooms()
            
            print(f"\nImage: {img_path.name}")
            print(f"  Shape: {img_shape}")
            print(f"  Spacing: {img_spacing}")
            
            if label_path.exists():
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
                    print(f"  [ERROR] Shape mismatch between image {img_shape} and label {lbl_shape}")
            else:
                print(f"  [WARNING] No corresponding label found for {img_path.name}")
                
        except Exception as e:
            print(f"[ERROR] Failed to read {img_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Verify NIfTI files.")
    parser.add_argument('--sample', type=int, default=3, help="Number of samples to verify (default: 3). Set to 0 for all.")
    args = parser.parse_args()
    
    base_dir = Path(os.path.abspath(__file__)).parent.parent.parent
    data_raw_dir = base_dir / "data" / "raw"
    images_dir = data_raw_dir / "images"
    labels_dir = data_raw_dir / "labels"
    
    verify_nifti_files(images_dir, labels_dir, sample_size=args.sample)
    
if __name__ == "__main__":
    main()
