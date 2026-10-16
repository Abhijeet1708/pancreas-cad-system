"""
Tests for segmentation model building, loss, prediction, and checkpointing.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
monai = pytest.importorskip("monai")

from src.segmentation.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from src.segmentation.model import build_loss, build_unet, predict_volume  # noqa: E402


def test_build_unet_output_shape() -> None:
    """Test build_unet outputs expected shape (B, 3, z, y, x) on 3D input."""
    model = build_unet(channels=(4, 8, 16), strides=(2, 2), num_res_units=2)
    inputs = torch.randn(2, 1, 16, 32, 32)
    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, 3, 16, 32, 32)


def test_build_loss_finite_on_random_data() -> None:
    """Test build_loss returns a finite scalar loss with DiceCELoss configuration."""
    loss_fn = build_loss()
    preds = torch.randn(2, 3, 16, 32, 32)
    targets = torch.randint(0, 3, (2, 1, 16, 32, 32), dtype=torch.int64)

    loss_val = loss_fn(preds, targets)

    assert torch.isfinite(loss_val)
    assert loss_val.ndim == 0
    assert float(loss_val.item()) > 0.0


def test_predict_volume_shape_dtype_and_values() -> None:
    """
    Test predict_volume returns uint8 with exact input shape and values in 0..2.

    Uses a shape that is not an integer multiple of the patch size.
    """
    model = build_unet(channels=(4, 8, 16), strides=(2, 2), num_res_units=2)
    input_shape = (18, 35, 33)
    patch_size = (16, 32, 32)
    rng = np.random.default_rng(42)
    image = rng.uniform(0.0, 1.0, size=input_shape).astype(np.float32)

    pred = predict_volume(
        model=model,
        image=image,
        patch_size=patch_size,
        device="cpu",
        overlap=0.5,
        sw_batch_size=2,
    )

    assert pred.shape == input_shape
    assert pred.dtype == np.uint8
    assert np.all((pred >= 0) & (pred <= 2))


def test_checkpoint_roundtrip_restores_weights_and_removes_tmp(tmp_path: Path) -> None:
    """Test save_checkpoint and load_checkpoint round-trip preserves weights with no .tmp file."""
    model = build_unet(channels=(4, 8, 16), strides=(2, 2), num_res_units=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    ckpt_path = tmp_path / "model_ckpt.pt"
    tmp_file = tmp_path / "model_ckpt.pt.tmp"

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iteration": 42,
        "best_metric": 0.875,
        "history": [
            {
                "iteration": 42,
                "mean_train_loss": 0.25,
                "val_pancreas_dice": 0.9,
                "val_tumor_dice": 0.85,
                "score": 0.875,
                "lr": 0.0003,
                "elapsed_minutes": 12.5,
            }
        ],
        "args": {"lr": 0.0003, "data_root": "data/cases"},
    }

    save_checkpoint(ckpt_path, state)

    assert ckpt_path.is_file()
    assert not tmp_file.exists()

    loaded = load_checkpoint(ckpt_path, map_location="cpu")

    assert loaded["iteration"] == 42
    assert loaded["best_metric"] == 0.875
    assert len(loaded["history"]) == 1
    assert loaded["args"]["lr"] == 0.0003

    for key, weight in model.state_dict().items():
        assert torch.equal(weight, loaded["model"][key])
