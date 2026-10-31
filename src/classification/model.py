"""
ResNet18 classifier and spatial operations for slice-level tumor detection.
"""

from typing import Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class TumorClassifier(nn.Module):
    """
    Slice-level tumor classifier wrapping a torchvision ResNet18 backbone.

    Accepts single-channel inputs of shape (B, 1, H, W) in [0, 1], expands to 3 channels,
    applies ImageNet mean and standard deviation normalization, and outputs a single logit.
    """

    def __init__(self, backbone: nn.Module, pretrained_loaded: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        self.pretrained_loaded = pretrained_loaded
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W) or (B, H, W) with pixel values in [0, 1].

        Returns
        -------
        torch.Tensor
            Logit output of shape (B, 1).
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = (x - self.mean) / self.std
        out = self.backbone(x)
        return out


def build_classifier(pretrained: bool = True) -> TumorClassifier:
    """
    Build a ResNet18 classifier with a single output logit.

    Parameters
    ----------
    pretrained : bool, default=True
        Whether to initialize with ImageNet pretrained weights. If network or
        download fails, falls back to random weights with a warning.

    Returns
    -------
    TumorClassifier
        Initialized classifier module with `backbone` and `pretrained_loaded`.
    """
    pretrained_loaded = False
    if pretrained:
        try:
            try:
                from torchvision.models import ResNet18_Weights

                weights = ResNet18_Weights.DEFAULT
                backbone = models.resnet18(weights=weights)
            except (ImportError, AttributeError):
                backbone = models.resnet18(pretrained=True)
            pretrained_loaded = True
        except Exception as exc:
            warnings.warn(
                f"LOUD WARNING: Could not download or load pretrained weights ({exc}). "
                "Continuing with random initialization!",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                backbone = models.resnet18(weights=None)
            except (TypeError, AttributeError):
                backbone = models.resnet18(pretrained=False)
            pretrained_loaded = False
    else:
        try:
            backbone = models.resnet18(weights=None)
        except (TypeError, AttributeError):
            backbone = models.resnet18(pretrained=False)
        pretrained_loaded = False

    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, 1)
    return TumorClassifier(backbone=backbone, pretrained_loaded=pretrained_loaded)


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """
    Return the target layer for Grad-CAM.

    Parameters
    ----------
    model : nn.Module
        TumorClassifier or module containing `backbone.layer4`.

    Returns
    -------
    nn.Module
        model.backbone.layer4[-1]
    """
    return model.backbone.layer4[-1]


def center_crop(
    x: Union[np.ndarray, torch.Tensor],
    size: int = 128,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Extract a size x size window from the center of (..., H, W).

    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Array or tensor with at least 2 spatial dimensions.
    size : int, default=128
        Square crop edge length.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Cropped array or tensor of shape (..., size, size).
    """
    h, w = x.shape[-2], x.shape[-1]
    if h < size or w < size:
        raise ValueError(f"Input spatial dimensions ({h}, {w}) must be >= crop size ({size})")

    start_y = (h - size) // 2
    start_x = (w - size) // 2
    return x[..., start_y:start_y + size, start_x:start_x + size]
