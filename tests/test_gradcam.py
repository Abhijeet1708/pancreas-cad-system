"""
Tests for Grad-CAM explainability and heatmap visualization.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
torchvision = pytest.importorskip("torchvision")

from src.classification.model import build_classifier, gradcam_target_layer  # noqa: E402
from src.explainability.gradcam import compute_gradcam, overlay_heatmap  # noqa: E402


class ToyQuadrantNet(nn.Module):
    """Toy network whose logit depends strictly on the top-left quadrant."""

    def __init__(self) -> None:
        super().__init__()
        self.target = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.target.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        mask = torch.zeros_like(x)
        mask[:, :, :h // 2, :w // 2] = 1.0
        feat = self.target(x * mask)
        return feat.sum().view(1, 1)


class NegativeLogitNet(nn.Module):
    """Toy network that produces negative gradients resulting in an all-zero CAM."""

    def __init__(self) -> None:
        super().__init__()
        self.target = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.target.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.target(x)
        # Negative sign ensures gradient with respect to feat is negative
        return (-feat.sum()).view(1, 1)


def test_gradcam_quadrant_localization():
    model = ToyQuadrantNet()
    x = torch.ones((1, 1, 16, 16), dtype=torch.float32)

    cam, prob = compute_gradcam(model, x, model.target)

    assert isinstance(cam, np.ndarray)
    assert cam.dtype == np.float32
    assert cam.shape == (16, 16)
    assert 0.0 <= prob <= 1.0
    assert not np.isnan(cam).any()
    assert cam.min() >= 0.0 and cam.max() <= 1.0

    # Top-left quadrant mean must be significantly higher than elsewhere
    tl_mean = float(np.mean(cam[:8, :8]))
    other_mask = np.ones((16, 16), dtype=bool)
    other_mask[:8, :8] = False
    other_mean = float(np.mean(cam[other_mask]))

    assert tl_mean > other_mean
    assert tl_mean > 0.8
    assert other_mean == 0.0

    # Verify no forward hooks left on target layer
    assert len(model.target._forward_hooks) == 0


def test_gradcam_all_zeros():
    model = NegativeLogitNet()
    x = torch.ones((1, 1, 16, 16), dtype=torch.float32)

    cam, prob = compute_gradcam(model, x, model.target)

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (16, 16)
    assert not np.isnan(cam).any()
    assert np.all(cam == 0.0)
    assert len(model.target._forward_hooks) == 0


def test_overlay_heatmap():
    img = np.zeros((32, 32), dtype=np.float32)
    cam = np.linspace(0.0, 1.0, 32 * 32).reshape(32, 32).astype(np.float32)

    overlay = overlay_heatmap(img, cam, alpha=0.5)

    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8
    assert overlay.max() > 0


def test_build_classifier_gradcam():
    model = build_classifier(pretrained=False)
    x = torch.rand((1, 1, 128, 128), dtype=torch.float32)
    target_layer = gradcam_target_layer(model)

    cam, prob = compute_gradcam(model, x, target_layer)

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (128, 128)
    assert cam.dtype == np.float32
    assert not np.isnan(cam).any()
    assert 0.0 <= prob <= 1.0
    assert len(target_layer._forward_hooks) == 0


def test_gradcam_inside_torch_no_grad():
    model = build_classifier(pretrained=False)
    x = torch.rand((1, 1, 128, 128), dtype=torch.float32)
    target_layer = gradcam_target_layer(model)

    with torch.no_grad():
        cam, prob = compute_gradcam(model, x, target_layer)

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (128, 128)
    assert not np.isnan(cam).any()
    assert len(target_layer._forward_hooks) == 0


def test_gradcam_with_frozen_model_parameters():
    model = build_classifier(pretrained=False)
    for p in model.parameters():
        p.requires_grad = False

    x = torch.rand((1, 1, 128, 128), dtype=torch.float32)
    target_layer = gradcam_target_layer(model)

    cam, prob = compute_gradcam(model, x, target_layer)

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (128, 128)
    assert not np.isnan(cam).any()
    assert len(target_layer._forward_hooks) == 0


def test_gradcam_restores_train_mode():
    model = build_classifier(pretrained=False)
    model.train()
    assert model.training is True

    x = torch.rand((1, 1, 128, 128), dtype=torch.float32)
    target_layer = gradcam_target_layer(model)

    compute_gradcam(model, x, target_layer)

    assert model.training is True
    assert len(target_layer._forward_hooks) == 0
