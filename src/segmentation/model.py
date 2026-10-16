"""
Model architecture, loss, and inference routines for 3D pancreas segmentation.

Conventions
-----------
- Spatial dimensions: 3D (z, y, x).
- Input image: float32 in [0, 1], shape (1, 1, z, y, x) or (z, y, x).
- Labels:
    0: background
    1: pancreas
    2: tumor
- Target label for loss: int64 with channel dimension, shape (B, 1, z, y, x).
"""

from typing import Sequence, Tuple, Union

import monai.inferers
import monai.losses
import monai.networks.nets
import numpy as np
import torch


def build_unet(
    channels: Sequence[int] = (16, 32, 64, 128, 256),
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int = 2,
) -> monai.networks.nets.UNet:
    """
    Construct a 3D UNet model for 3-class segmentation.

    Parameters
    ----------
    channels : Sequence[int], default=(16, 32, 64, 128, 256)
        Feature channels for each level of the encoder/decoder.
    strides : Sequence[int], default=(2, 2, 2, 2)
        Strides between consecutive levels.
    num_res_units : int, default=2
        Number of residual units per block.

    Returns
    -------
    monai.networks.nets.UNet
        Configured 3D UNet with in_channels=1 and out_channels=3.
    """
    return monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=tuple(channels),
        strides=tuple(strides),
        num_res_units=num_res_units,
    )


def build_loss() -> monai.losses.DiceCELoss:
    """
    Construct combined Dice and Cross-Entropy loss for pancreas/tumor segmentation.

    Expects target labels with a channel dimension, i.e. shape (B, 1, z, y, x).

    Returns
    -------
    monai.losses.DiceCELoss
        Loss instance configured with to_onehot_y=True, softmax=True, include_background=False.
    """
    return monai.losses.DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
    )


def predict_volume(
    model: torch.nn.Module,
    image: Union[np.ndarray, torch.Tensor],
    patch_size: Tuple[int, int, int],
    device: Union[str, torch.device],
    overlap: float = 0.5,
    sw_batch_size: int = 2,
) -> np.ndarray:
    """
    Perform sliding-window inference on a 3D volume.

    Parameters
    ----------
    model : torch.nn.Module
        3D segmentation network.
    image : Union[np.ndarray, torch.Tensor]
        Input image volume of shape (z, y, x).
    patch_size : Tuple[int, int, int]
        Patch size for sliding window inference in (z, y, x) order.
    device : Union[str, torch.device]
        Target compute device.
    overlap : float, default=0.5
        Sliding window overlap fraction.
    sw_batch_size : int, default=2
        Batch size of patches during sliding-window inference.

    Returns
    -------
    np.ndarray
        Argmax predicted class map as a uint8 array of shape (z, y, x).
    """
    dev = torch.device(device)
    if isinstance(image, np.ndarray):
        tensor_image = torch.from_numpy(image).float()
    else:
        tensor_image = image.float()

    if tensor_image.ndim == 3:
        tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)
    elif tensor_image.ndim == 4:
        tensor_image = tensor_image.unsqueeze(0)

    tensor_image = tensor_image.to(dev)
    use_cuda = dev.type == "cuda"
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=use_cuda):
                logits = monai.inferers.sliding_window_inference(
                    inputs=tensor_image,
                    roi_size=patch_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                )
            pred = torch.argmax(logits, dim=1).squeeze(0)
            return pred.detach().cpu().to(torch.uint8).numpy()
    finally:
        if was_training:
            model.train()
