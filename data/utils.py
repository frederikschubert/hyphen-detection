from typing import Tuple

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor


def read_patch(image, center, patch_size: int, pad=True):
    image = np.array(image)
    if pad:
        padded_image = pad_image(image, patch_size)
    else:
        padded_image = image
    patch = to_tensor(crop_patch(padded_image, center, patch_size))
    center_mask = create_mask(
        height=image.shape[0],
        width=image.shape[1],
        center=center,
        patch_size=patch_size,
    )
    patch = torch.cat([center_mask, patch])
    return patch


def pad_image(image, patch_size: int):
    half_patch_size = patch_size // 2
    image_padded = np.pad(
        image,
        [
            (half_patch_size, half_patch_size),
            (half_patch_size, half_patch_size),
            (0, 0),
        ],
    )
    return image_padded


def crop_patch(padded_image, center: Tuple[int, int], patch_size: int):
    x, y = center
    patch = padded_image[
        y : y + patch_size,
        x : x + patch_size,
        :,
    ]
    return patch


def create_mask(
    height: int,
    width: int,
    center: Tuple[int, int],
    patch_size: int,
    mask_radius: int = 4,
):
    x_center, y_center = center
    center_mask = np.zeros(shape=[height, width, 1], dtype=np.float32)
    y, x = np.ogrid[-y_center : height - y_center, -x_center : width - x_center]
    mask = x**2 + y**2 <= 4**2
    center_mask[mask] = 1
    center_mask = pad_image(center_mask, patch_size)
    center_mask = crop_patch(center_mask, center, patch_size)
    center_mask = to_tensor(center_mask)
    return center_mask
