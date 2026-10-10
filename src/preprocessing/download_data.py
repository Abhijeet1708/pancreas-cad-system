import os
import tarfile
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_and_extract(repo_id: str, filename: str, extract_dir: Path):
    print(f"Downloading {filename} from {repo_id}...")
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

    print(f"Extracting {file_path} to {extract_dir}...")
    with tarfile.open(file_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_dir)


def organize_data(extract_dir: Path, output_dir: Path):
    # Expected structure: extract_dir/Task07_Pancreas/imagesTr and labelsTr
    dataset_dir = extract_dir / "Task07_Pancreas"

    if not dataset_dir.exists():
        found_dirs = [d for d in extract_dir.iterdir() if d.is_dir() and "Pancreas" in d.name]
        if found_dirs:
            dataset_dir = found_dirs[0]
            print(f"Found dataset directory: {dataset_dir.name}")
        else:
            print(f"Error: Could not find Task07_Pancreas or any directory containing "
                  f"'Pancreas' in {extract_dir}.")
            return

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    if not images_tr.exists() or not labels_tr.exists():
        print(f"Warning: The expected directories were not found in {dataset_dir}.")
        return

    print("Moving images...")
    for img_file in images_tr.glob("*.nii.gz"):
        shutil.move(str(img_file), str(out_images / img_file.name))

    print("Moving labels...")
    for lbl_file in labels_tr.glob("*.nii.gz"):
        shutil.move(str(lbl_file), str(out_labels / lbl_file.name))

    print("Cleaning up extracted directory...")
    shutil.rmtree(dataset_dir)
    print(f"Data organized into {output_dir}")


def main():
    repo_id = "qicq1c/Pubilcdataset"
    filename = "10_Decathlon/Task07_Pancreas.tar.gz"

    # Determine base directory
    # Script is in src/preprocessing, so base_dir is 2 levels up
    base_dir = Path(os.path.abspath(__file__)).parent.parent.parent
    data_raw_dir = base_dir / "data" / "raw"
    temp_extract_dir = base_dir / "data" / "temp_extract"

    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    download_and_extract(repo_id, filename, temp_extract_dir)
    organize_data(temp_extract_dir, data_raw_dir)

    # Remove temp extract dir if it's empty or no longer needed
    try:
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
    except Exception as e:
        print(f"Failed to remove temp dir {temp_extract_dir}: {e}")


if __name__ == "__main__":
    main()
