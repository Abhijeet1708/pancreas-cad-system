"""
Tests for slice-level tumor classification (ResNet18) and evaluation pipeline.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from src.classification.evaluate import compute_auc, main as evaluate_main  # noqa: E402
from src.classification.model import (  # noqa: E402
    build_classifier,
    center_crop,
    gradcam_target_layer,
)
from src.classification.slices import (  # noqa: E402
    build_slice_arrays,
    crop_slice,
    slice_center,
)
from src.classification.train import main as train_main  # noqa: E402


def _make_synthetic_case(
    npz_path: Path,
    shape=(4, 180, 180),
    slice_specs=None,
) -> None:
    """Helper to create a synthetic volume with specified slice contents."""
    image = np.full(shape, 0.2, dtype=np.float32)
    label = np.zeros(shape, dtype=np.uint8)

    if slice_specs:
        for z, spec in slice_specs.items():
            if "pancreas" in spec:
                py0, py1, px0, px1 = spec["pancreas"]
                label[z, py0:py1, px0:px1] = 1
                image[z, py0:py1, px0:px1] = 0.5
            if "tumor" in spec:
                ty0, ty1, tx0, tx1 = spec["tumor"]
                label[z, ty0:ty1, tx0:tx1] = 2
                image[z, ty0:ty1, tx0:tx1] = 0.8

    np.savez(npz_path, image=image.astype(np.float16), label=label)


def test_slice_center_and_crop_slice() -> None:
    """Test slice_center centroid and crop_slice window extraction, shifts, and padding."""
    # 1. Centroid calculation
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:30, 40:50] = 1
    cy, cx = slice_center(mask)
    assert 24 <= cy <= 25
    assert 44 <= cx <= 45

    # Empty mask returns geometric center
    empty_mask = np.zeros((80, 120), dtype=np.uint8)
    assert slice_center(empty_mask) == (40, 60)

    # 2. crop_slice shapes and boundary shift
    img = np.arange(200 * 200, dtype=np.float32).reshape(200, 200)

    # Centered crop
    crop_normal = crop_slice(img, (100, 100), size=160)
    assert crop_normal.shape == (160, 160)
    assert crop_normal.dtype == np.float32
    assert crop_normal[80, 80] == img[100, 100]

    # Stays inside when center is near boundary
    crop_topleft = crop_slice(img, (10, 10), size=160)
    assert crop_topleft.shape == (160, 160)
    assert np.array_equal(crop_topleft, img[:160, :160])

    crop_bottomright = crop_slice(img, (195, 195), size=160)
    assert crop_bottomright.shape == (160, 160)
    assert np.array_equal(crop_bottomright, img[40:200, 40:200])

    # 3. crop_slice padding for small images
    small_img = np.ones((60, 80), dtype=np.float16)
    crop_padded = crop_slice(small_img, (30, 40), size=160)
    assert crop_padded.shape == (160, 160)
    assert crop_padded.dtype == np.float16
    assert np.count_nonzero(crop_padded) == 60 * 80
    assert crop_padded[0, 0] == 0.0  # padded region


def test_build_slice_arrays_synthetic(tmp_path: Path) -> None:
    """Test build_slice_arrays positive, negative, and skipped slice counts."""
    case1_path = tmp_path / "case_001.npz"
    case2_path = tmp_path / "case_002.npz"

    # Case 1:
    # Slice 0: no pancreas (< 50 px) -> skipped
    # Slice 1: pancreas 100 px (20:30, 20:30), tumor 0 px -> NEGATIVE (y = 0)
    # Slice 2: pancreas 150 px (40:55, 40:50), tumor 40 px (45:53, 45:50) in center -> POS (y = 1)
    # Slice 3: pancreas 100 px (20:30, 20:30), tumor 15 px (0 < t < 30) -> SKIPPED (ambiguous)
    _make_synthetic_case(
        case1_path,
        shape=(4, 180, 180),
        slice_specs={
            0: {"pancreas": (20, 24, 20, 25)},  # 20 px < 50
            1: {"pancreas": (20, 30, 20, 30)},  # 100 px, 0 tumor
            2: {"pancreas": (40, 55, 40, 50), "tumor": (45, 53, 45, 50)},  # 40 tumor px
            3: {"pancreas": (20, 30, 20, 30), "tumor": (20, 23, 20, 25)},  # 15 tumor px
        },
    )

    # Case 2:
    # Slice 0: pancreas 100 px, tumor 0 px -> NEGATIVE (y = 0)
    # Slice 1: pancreas 150 px, tumor 50 px in center -> POSITIVE (y = 1)
    _make_synthetic_case(
        case2_path,
        shape=(4, 180, 180),
        slice_specs={
            0: {"pancreas": (50, 60, 50, 60)},  # 100 px, 0 tumor
            1: {"pancreas": (60, 75, 60, 70), "tumor": (65, 75, 65, 70)},  # 50 tumor px
        },
    )

    x_arr, y_arr, case_arr, z_arr, stats = build_slice_arrays(
        case_ids=["case_001", "case_002"],
        data_root=tmp_path,
        crop=160,
        min_pancreas_px=50,
        min_tumor_px=30,
        return_stats=True,
    )

    assert len(x_arr) == 4
    assert x_arr.shape == (4, 160, 160)
    assert x_arr.dtype == np.float16
    assert y_arr.dtype == np.uint8

    assert stats["pos"] == 2
    assert stats["neg"] == 2
    assert stats["skipped"] == 1  # Slice 3 of case 1 has ambiguous tumor
    assert list(y_arr) == [0, 1, 0, 1]
    assert list(case_arr) == ["case_001", "case_001", "case_002", "case_002"]
    assert list(z_arr) == [1, 2, 0, 1]


def test_build_slice_arrays_edge_tumor_skipped(tmp_path: Path) -> None:
    """Test refinement 1: synthetic slice whose tumor lies only near crop edge is skipped."""
    case_path = tmp_path / "edge_case.npz"
    # Pancreas centered near (40, 40) (size ~100 px)
    # Crop 160 window centered on (40, 40) will be [0:160, 0:160]
    # Central 96x96 window of this crop is [32:128, 32:128]
    # Place tumor of 40 px at y=140:148, x=140:145 (outside central 96x96)
    _make_synthetic_case(
        case_path,
        shape=(1, 180, 180),
        slice_specs={
            0: {
                "pancreas": (35, 45, 35, 45),  # 100 px
                "tumor": (140, 148, 140, 145),  # 40 px, outside [32:128, 32:128]
            }
        },
    )

    x_arr, y_arr, case_arr, z_arr, stats = build_slice_arrays(
        case_ids=["edge_case"],
        data_root=tmp_path,
        crop=160,
        min_pancreas_px=50,
        min_tumor_px=30,
        return_stats=True,
    )

    # Slice has tumor, but fails the positive central 96x96 rule -> must be skipped
    assert len(x_arr) == 0
    assert stats["pos"] == 0
    assert stats["neg"] == 0
    assert stats["skipped"] == 1


def test_build_classifier_shape_and_gradcam() -> None:
    """Test build_classifier output shape, gradcam target layer, and center crop."""
    model = build_classifier(pretrained=False)
    assert hasattr(model, "backbone")
    assert model.pretrained_loaded is False

    inp = torch.rand(2, 1, 128, 128)
    out = model(inp)
    assert out.shape == (2, 1)

    target_layer = gradcam_target_layer(model)
    assert target_layer is model.backbone.layer4[-1]

    # center_crop tests
    arr = np.zeros((2, 160, 160), dtype=np.float32)
    arr[:, 16:144, 16:144] = 1.0
    cropped = center_crop(arr, size=128)
    assert cropped.shape == (2, 128, 128)
    assert np.all(cropped == 1.0)


def test_train_main_synthetic(tmp_path: Path) -> None:
    """Test train.main on synthetic cls_train.npz and cls_val.npz without cls_test.npz."""
    data_dir = tmp_path / "cls_data"
    out_dir = tmp_path / "cls_out"
    data_dir.mkdir()

    # Create synthetic cls_train.npz
    rng = np.random.default_rng(123)
    x_train = rng.uniform(0.1, 0.9, size=(8, 160, 160)).astype(np.float16)
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
    case_train = np.array([f"case_{i // 2}" for i in range(8)])
    z_train = np.array([i % 2 for i in range(8)], dtype=np.int32)
    np.savez(data_dir / "cls_train.npz", X=x_train, y=y_train, case=case_train, z=z_train)

    # Create synthetic cls_val.npz
    x_val = rng.uniform(0.1, 0.9, size=(4, 160, 160)).astype(np.float16)
    y_val = np.array([0, 1, 0, 1], dtype=np.uint8)
    case_val = np.array(["val_0", "val_0", "val_1", "val_1"])
    z_val = np.array([0, 1, 0, 1], dtype=np.int32)
    np.savez(data_dir / "cls_val.npz", X=x_val, y=y_val, case=case_val, z=z_val)

    # Explicitly verify cls_test.npz does NOT exist
    assert not (data_dir / "cls_test.npz").exists()

    cmd = [
        "--data-dir", str(data_dir),
        "--out-dir", str(out_dir),
        "--epochs", "2",
        "--batch-size", "4",
        "--lr", "1e-4",
        "--seed", "42",
        "--num-workers", "0",
        "--no-amp",
        "--no-pretrained",
    ]
    exit_code = train_main(cmd)
    assert exit_code == 0

    assert (out_dir / "last.pt").is_file()
    assert (out_dir / "best.pt").is_file()
    assert (out_dir / "metrics.csv").is_file()
    assert (out_dir / "run_config.json").is_file()

    # Verify best.pt reloads with default weights_only=True
    ckpt = torch.load(out_dir / "best.pt", weights_only=True)
    assert "model" in ckpt
    assert "threshold" in ckpt
    assert "val_auc" in ckpt
    assert "epoch" in ckpt
    assert "pretrained_loaded" in ckpt
    assert isinstance(ckpt["args"], dict)


def test_evaluate_safety_and_auc_and_e2e(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test safety check, AUC rank statistics, and end-to-end evaluation pipeline."""
    # 1. Safety check refuses test without --final and does NOT create out-dir
    blocked_dir = tmp_path / "blocked_out"
    exit_code = evaluate_main([
        "--data-dir", str(tmp_path),
        "--checkpoint", str(tmp_path / "nonexistent.pt"),
        "--out-dir", str(blocked_dir),
        "--split", "test",
    ])
    assert exit_code == 2
    assert not blocked_dir.exists()
    captured = capsys.readouterr()
    assert "Error: --split test requires --final flag" in captured.err

    # 2. AUC rank statistics function
    # Perfect
    assert compute_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    # Inverted
    assert compute_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    # Ties
    assert compute_auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
    # Single class gives NaN
    assert np.isnan(compute_auc(np.array([1, 1]), np.array([0.2, 0.8])))

    # 3. End-to-end evaluate on synthetic data
    data_dir = tmp_path / "cls_data2"
    out_dir = tmp_path / "eval_out"
    data_dir.mkdir()

    rng = np.random.default_rng(42)
    x_val = rng.uniform(0.1, 0.9, size=(6, 160, 160)).astype(np.float16)
    y_val = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8)
    case_val = np.array(["c1", "c1", "c2", "c2", "c3", "c3"])
    z_val = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    np.savez(data_dir / "cls_val.npz", X=x_val, y=y_val, case=case_val, z=z_val)

    # Build dummy checkpoint
    dummy_model = build_classifier(pretrained=False)
    ckpt_path = tmp_path / "test_model.pt"
    torch.save(
        {
            "model": dummy_model.state_dict(),
            "epoch": 1,
            "val_auc": 0.8,
            "threshold": 0.45,
            "pretrained_loaded": False,
            "args": {},
        },
        ckpt_path,
    )

    eval_cmd = [
        "--data-dir", str(data_dir),
        "--checkpoint", str(ckpt_path),
        "--out-dir", str(out_dir),
        "--split", "val",
        "--bootstrap-samples", "100",
        "--seed", "42",
        "--batch-size", "4",
    ]
    eval_code = evaluate_main(eval_cmd)
    assert eval_code == 0

    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "summary.md").is_file()
    assert (out_dir / "roc_points.csv").is_file()
    assert (out_dir / "per_slice.csv").is_file()

    import json
    with open(out_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["split"] == "val"
    assert summary["val_threshold"] == 0.45
    assert summary["n_slices"] == 6
    assert summary["n_cases"] == 3
    assert summary["n_pos"] == 3
    assert summary["n_neg"] == 3
    assert "auc" in summary
    assert "metrics_at_val_threshold" in summary
    assert "metrics_at_fixed_05_threshold" in summary
    assert "ci_95" in summary
    assert summary["n_valid_boot"] > 0
