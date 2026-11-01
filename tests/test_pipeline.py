"""
Tests for CAD end-to-end inference pipeline and volume loaders.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sitk = pytest.importorskip("SimpleITK")
monai = pytest.importorskip("monai")
torchvision = pytest.importorskip("torchvision")

from src.classification.model import build_classifier  # noqa: E402
from src.pipeline.loaders import (  # noqa: E402
    load_cls_model,
    load_seg_model,
    preprocess_ct_volume,
)
from src.pipeline.run import analyze_volume, run_pipeline  # noqa: E402
from src.preprocessing.preprocess import process_case  # noqa: E402
from src.segmentation.model import build_unet  # noqa: E402


def test_preprocess_ct_volume_nifti_matches_process_case(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    proc_dir = tmp_path / "processed"
    images_dir.mkdir()
    labels_dir.mkdir()
    proc_dir.mkdir()

    # Synthetic CT HU values spanning below, inside, and above [-100, 240] window
    raw_arr = np.array(
        [
            [[-150.0, -100.0, -50.0], [0.0, 70.0, 150.0], [240.0, 300.0, 500.0]],
            [[-100.0, -20.0, 40.0], [80.0, 120.0, 200.0], [220.0, 240.0, 260.0]],
        ],
        dtype=np.float32,
    )
    lbl_arr = np.array(
        [
            [[0, 1, 1], [0, 2, 2], [0, 0, 0]],
            [[1, 1, 0], [2, 2, 0], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )

    img = sitk.GetImageFromArray(raw_arr)
    lbl = sitk.GetImageFromArray(lbl_arr)
    img.SetSpacing((1.0, 1.0, 2.5))
    lbl.SetSpacing((1.0, 1.0, 2.5))

    img_path = images_dir / "case_001.nii.gz"
    lbl_path = labels_dir / "case_001.nii.gz"
    sitk.WriteImage(img, str(img_path))
    sitk.WriteImage(lbl, str(lbl_path))

    process_case(img_path, lbl_path, proc_dir, "case_001")
    npz_path = proc_dir / "case_001.npz"

    with np.load(npz_path) as data:
        npz_img = data["image"].astype(np.float32)

    # NIfTI loading through preprocess_ct_volume
    prep_vol = preprocess_ct_volume(img_path)
    assert prep_vol.dtype == np.float32
    assert prep_vol.shape == npz_img.shape
    # np.allclose with atol=2e-3 due to float16 compression in npz
    assert np.allclose(prep_vol, npz_img, atol=2e-3)

    # Also test the .npz input path
    prep_npz = preprocess_ct_volume(npz_path)
    assert prep_npz.dtype == np.float32
    assert np.array_equal(prep_npz, npz_img)


def test_analyze_volume_works_with_no_files():
    image = np.zeros((8, 64, 64), dtype=np.float32)
    # Put a pancreas blob with >= 50 pixels in slice 3
    image[3, 20:30, 20:30] = 0.5

    seg_model = build_unet(channels=(4, 8, 16), strides=(2, 2), num_res_units=1)
    cls_model = build_classifier(pretrained=False)

    analysis = analyze_volume(
        image=image,
        seg_model=seg_model,
        cls_model=cls_model,
        threshold=0.5,
        device="cpu",
        patch_size=(8, 32, 32),
        top_k=2,
        min_tumor_voxels=50,
    )

    assert "labels" in analysis
    assert "result" in analysis
    assert "slices" in analysis
    assert isinstance(analysis["labels"], np.ndarray)
    assert analysis["labels"].dtype == np.uint8
    assert "pancreas_volume_ml" in analysis["result"]
    assert "tumor_volume_ml" in analysis["result"]
    assert isinstance(analysis["slices"], list)


def test_run_pipeline_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # 1. Create synthetic input volume
    in_volume = np.zeros((10, 64, 64), dtype=np.float16)
    in_volume[2:6, 20:30, 20:30] = 0.6
    input_file = tmp_path / "sample_volume.npz"
    np.savez_compressed(input_file, image=in_volume)

    # 2. Create tiny checkpoints
    seg_model = build_unet(channels=(4, 8, 16), strides=(2, 2), num_res_units=1)
    seg_ckpt_path = tmp_path / "seg_checkpoint.pt"
    torch.save(
        {
            "iteration": 500,
            "model": seg_model.state_dict(),
            "args": {
                "channels": [4, 8, 16],
                "strides": [2, 2],
                "num_res_units": 1,
            },
        },
        seg_ckpt_path,
    )

    cls_model = build_classifier(pretrained=False)
    cls_ckpt_path = tmp_path / "cls_checkpoint.pt"
    torch.save(
        {
            "model": cls_model.state_dict(),
            "epoch": 3,
            "threshold": 0.45,
            "val_auc": 0.90,
            "args": {},
        },
        cls_ckpt_path,
    )

    # 3. Create fixed mock segmentation with pancreas blob and tumor blob
    mock_labels = np.zeros((10, 64, 64), dtype=np.uint8)
    # Slices 2, 3, 4, 5 each have 100 pancreas pixels (label 1)
    mock_labels[2:6, 20:30, 20:30] = 1
    # Tumor blob on slices 3, 4 (2 slices * 60 voxels = 120 tumor voxels)
    mock_labels[3:5, 22:28, 20:30] = 2

    def mock_predict_volume(model, image, patch_size, device, **kwargs):
        return mock_labels.copy()

    # Monkeypatch predict_volume where it is used in src.pipeline.run
    monkeypatch.setattr("src.pipeline.run.predict_volume", mock_predict_volume)

    out_dir = tmp_path / "pipeline_output"
    exit_code = run_pipeline(
        input_path=input_file,
        seg_checkpoint=seg_ckpt_path,
        cls_checkpoint=cls_ckpt_path,
        out_dir=out_dir,
        device="cpu",
        patch_size=(8, 32, 32),
        top_k=2,
        min_tumor_voxels=100,
    )

    assert exit_code == 0

    # Assert monkeypatch took effect and prediction was saved
    pred_path = out_dir / "prediction.npz"
    assert pred_path.is_file()
    with np.load(pred_path) as pred_data:
        saved_labels = pred_data["labels"]
        assert np.array_equal(saved_labels, mock_labels)

    # Assert result.json has all required keys and correct volumes
    json_path = out_dir / "result.json"
    assert json_path.is_file()
    with open(json_path, mode="r", encoding="utf-8") as f:
        res = json.load(f)

    expected_keys = {
        "input",
        "device",
        "seconds",
        "image_shape",
        "pancreas_volume_ml",
        "tumor_volume_ml",
        "segmentation_tumor_found",
        "n_slices_evaluated",
        "n_slices_flagged",
        "max_slice_probability",
        "top_slices",
        "classifier_threshold",
        "classifier_threshold_source",
        "seg_checkpoint",
        "cls_checkpoint",
        "seg_iteration",
        "notes",
    }
    assert expected_keys.issubset(res.keys())

    # 120 voxels * 0.0025 mL = 0.3 mL
    expected_tumor_voxels = 120
    assert res["tumor_volume_ml"] == pytest.approx(expected_tumor_voxels * 0.0025, rel=1e-5)
    assert res["segmentation_tumor_found"] is True
    assert res["n_slices_evaluated"] == 4
    assert res["classifier_threshold"] == 0.45
    assert res["seg_checkpoint"] == "seg_checkpoint.pt"
    assert res["cls_checkpoint"] == "cls_checkpoint.pt"
    assert res["seg_iteration"] == 500

    # Notes assertions
    notes = res["notes"]
    assert any("Research prototype - not for clinical use" in n for n in notes)
    assert any(
        "slice classifier is weak" in n.lower() and "segmentation output is the primary finding"
        in n.lower()
        for n in notes
    )

    # Figures should exist for top-k slices (top_k=2) when matplotlib is present
    try:
        import matplotlib  # noqa: F401
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False

    if has_matplotlib:
        figures_dir = out_dir / "figures"
        fig_files = list(figures_dir.glob("slice_*.png"))
        assert len(fig_files) == 2


def test_run_pipeline_no_pancreas_predicted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    in_volume = np.zeros((8, 64, 64), dtype=np.float16)
    input_file = tmp_path / "empty_volume.npz"
    np.savez_compressed(input_file, image=in_volume)

    seg_model = build_unet(channels=(4, 8, 16), strides=(2, 2), num_res_units=1)
    seg_ckpt = tmp_path / "seg.pt"
    torch.save(
        {
            "iteration": 1,
            "model": seg_model.state_dict(),
            "args": {"channels": [4, 8, 16], "strides": [2, 2], "num_res_units": 1},
        },
        seg_ckpt,
    )

    cls_model = build_classifier(pretrained=False)
    cls_ckpt = tmp_path / "cls.pt"
    torch.save(
        {"model": cls_model.state_dict(), "threshold": 0.5, "args": {}},
        cls_ckpt,
    )

    # predict_volume returns all zeros
    monkeypatch.setattr(
        "src.pipeline.run.predict_volume",
        lambda *args, **kwargs: np.zeros((8, 64, 64), dtype=np.uint8),
    )

    out_dir = tmp_path / "empty_output"
    exit_code = run_pipeline(
        input_path=input_file,
        seg_checkpoint=seg_ckpt,
        cls_checkpoint=cls_ckpt,
        out_dir=out_dir,
        device="cpu",
        patch_size=(8, 32, 32),
        top_k=3,
        min_tumor_voxels=100,
    )

    assert exit_code == 0
    with open(out_dir / "result.json", mode="r", encoding="utf-8") as f:
        res = json.load(f)

    assert res["n_slices_evaluated"] == 0
    assert res["top_slices"] == []
    assert any("no axial slices" in n.lower() or "no slices" in n.lower() for n in res["notes"])


def test_run_pipeline_missing_input_file(tmp_path: Path):
    missing_input = tmp_path / "nonexistent.npz"
    seg_ckpt = tmp_path / "seg.pt"
    cls_ckpt = tmp_path / "cls.pt"
    seg_ckpt.touch()
    cls_ckpt.touch()

    out_dir = tmp_path / "should_not_exist"
    exit_code = run_pipeline(
        input_path=missing_input,
        seg_checkpoint=seg_ckpt,
        cls_checkpoint=cls_ckpt,
        out_dir=out_dir,
        device="cpu",
    )

    assert exit_code == 2
    assert not out_dir.exists()


def test_load_seg_and_cls_models(tmp_path: Path):
    seg_model = build_unet(channels=(4, 8, 16), strides=(2, 2), num_res_units=1)
    seg_ckpt = tmp_path / "seg.pt"
    torch.save(
        {
            "iteration": 42,
            "model": seg_model.state_dict(),
            "args": {"channels": [4, 8, 16], "strides": [2, 2], "num_res_units": 1},
        },
        seg_ckpt,
    )

    cls_model = build_classifier(pretrained=False)
    cls_ckpt = tmp_path / "cls.pt"
    torch.save(
        {"model": cls_model.state_dict(), "threshold": 0.62, "args": {}},
        cls_ckpt,
    )

    loaded_seg = load_seg_model(seg_ckpt, "cpu")
    assert loaded_seg.training is False
    assert getattr(loaded_seg, "iteration", None) == 42

    loaded_cls, thresh = load_cls_model(cls_ckpt, "cpu")
    assert loaded_cls.training is False
    assert thresh == 0.62
    assert getattr(loaded_cls, "threshold_source", None) == "checkpoint"
