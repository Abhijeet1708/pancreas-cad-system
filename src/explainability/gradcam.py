"""
Grad-CAM explainability and heatmap visualization for 2D slice classification.
"""

from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def compute_gradcam(
    model: nn.Module,
    x: torch.Tensor,
    target_layer: nn.Module,
) -> Tuple[np.ndarray, float]:
    """
    Compute Grad-CAM heatmap and sigmoid probability for a single 2D input.

    Parameters
    ----------
    model : nn.Module
        Classifier model evaluated in eval mode.
    x : torch.Tensor
        Input tensor of shape (1, 1, H, W) or (1, H, W) in [0, 1].
    target_layer : nn.Module
        Convolutional layer whose feature maps are hooked (e.g. layer4[-1]).

    Returns
    -------
    Tuple[np.ndarray, float]
        - cam: float32 numpy array of shape (H, W) in [0, 1]. All-zero if no positive
          activations, with no NaN values.
        - prob: float representing the sigmoid prediction probability.
    """
    was_training = model.training
    model.eval()

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        activations.append(out)

        def backward_hook(grad: torch.Tensor) -> None:
            gradients.append(grad)

        out.register_hook(backward_hook)

    hook_handle = target_layer.register_forward_hook(forward_hook)

    try:
        with torch.enable_grad():
            x_input = x.detach().clone().requires_grad_(True)
            if x_input.ndim == 3:
                x_input = x_input.unsqueeze(1)

            logits = model(x_input)
            prob = float(torch.sigmoid(logits).view(-1)[0].item())

            model.zero_grad(set_to_none=True)
            logit = logits.view(-1)[0]
            logit.backward()

            h_in, w_in = x_input.shape[-2], x_input.shape[-1]
            if not activations or not gradients:
                return np.zeros((h_in, w_in), dtype=np.float32), prob

            act = activations[0]
            grad = gradients[0]

            weights = torch.mean(grad, dim=(-2, -1), keepdim=True)
            cam = torch.sum(weights * act, dim=1, keepdim=True)
            cam = torch.relu(cam)
            cam = torch.nn.functional.interpolate(
                cam,
                size=(h_in, w_in),
                mode="bilinear",
                align_corners=False,
            )
            cam = cam.squeeze(0).squeeze(0)

            cam_min = cam.min()
            cam_max = cam.max()
            diff = cam_max - cam_min
            if diff > 1e-8:
                cam = (cam - cam_min) / diff
            else:
                cam = torch.zeros_like(cam)

            cam_np = cam.detach().cpu().to(torch.float32).numpy()
            return cam_np, prob
    finally:
        hook_handle.remove()
        if was_training:
            model.train()


def overlay_heatmap(
    image2d: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend 2D grayscale image with a Grad-CAM heatmap using a pure-numpy colormap.

    Parameters
    ----------
    image2d : np.ndarray
        2D grayscale array of shape (H, W). Can be float in [0, 1] or uint8 in [0, 255].
    cam : np.ndarray
        2D Grad-CAM heatmap of shape (H, W) with values in [0, 1].
    alpha : float, default=0.4
        Heatmap overlay opacity in [0.0, 1.0].

    Returns
    -------
    np.ndarray
        Blended RGB image of shape (H, W, 3) and dtype uint8.
    """
    if image2d.dtype == np.uint8:
        base_img = image2d.astype(np.float32) / 255.0
    else:
        base_img = np.clip(image2d.astype(np.float32), 0.0, 1.0)

    if base_img.ndim == 2:
        img_rgb = np.stack([base_img, base_img, base_img], axis=-1)
    else:
        img_rgb = base_img

    cam_clipped = np.clip(cam.astype(np.float32).ravel(), 0.0, 1.0)
    xp = [0.0, 0.25, 0.5, 0.75, 1.0]
    r_pts = [0.0, 0.0, 0.0, 1.0, 1.0]
    g_pts = [0.0, 1.0, 1.0, 1.0, 0.0]
    b_pts = [1.0, 1.0, 0.0, 0.0, 0.0]

    r_ch = np.interp(cam_clipped, xp, r_pts)
    g_ch = np.interp(cam_clipped, xp, g_pts)
    b_ch = np.interp(cam_clipped, xp, b_pts)

    h, w = cam.shape[:2]
    cmap_rgb = np.stack([r_ch, g_ch, b_ch], axis=-1).reshape(h, w, 3)

    blended = (1.0 - alpha) * img_rgb + alpha * cmap_rgb
    out_uint8 = np.clip(np.round(blended * 255.0), 0.0, 255.0).astype(np.uint8)
    return out_uint8
