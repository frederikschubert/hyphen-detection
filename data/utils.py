from typing import Tuple, List, cast
import cv2

import numpy as np
import torch.nn as nn
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image


def read_patch(image, center, patch_size: int, pad=True):
    image = np.array(image)
    if pad:
        padded_image = pad_image(image, patch_size)
    else:
        padded_image = image
    patch = crop_patch(padded_image, center, patch_size)
    center_mask = create_mask(
        height=image.shape[0],
        width=image.shape[1],
        center=center,
        patch_size=patch_size,
    )
    return to_tensor(patch), to_tensor(center_mask)


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
    mask = x**2 + y**2 <= mask_radius**2
    center_mask[mask] = 1
    center_mask = pad_image(center_mask, patch_size)
    center_mask = crop_patch(center_mask, center, patch_size)
    return center_mask


def get_detection_image(
    image,
    centers,
    predictions,
    labels=[],
    circle_radius: int = 4,
    circle_color: Tuple[int, int, int] = (240, 240, 240),
    circle_thickness: int = 1,
):
    if not labels:
        labels = [-1] * len(centers)
    for center, prediction, label in zip(centers, predictions, labels):
        center = tuple(center)
        image = cv2.circle(
            image,
            center,
            circle_radius,
            circle_color,
            circle_thickness,
        )
        true_positive = prediction == 1 and label == 1
        false_negative = prediction == 0 and label == 1
        if prediction == 1:
            image = cv2.circle(
                image,
                center,
                circle_radius - 1,
                (255, 0, 0),
                -1,
            )
        if true_positive:
            image = cv2.circle(
                image,
                center,
                circle_radius,
                (0, 255, 0),
                circle_thickness,
            )
        if false_negative:
            image = cv2.circle(
                image,
                center,
                circle_radius,
                (255, 0, 0),
                circle_thickness,
            )
    return image


def visualize_predictions(
    model: nn.Module,
    image_path: str,
    centers: List[Tuple[int, int]],
    patch_size: int,
    labels: List[int] = [],
):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    image_patches = torch.stack(
        [torch.cat(read_patch(image, center, patch_size)) for center in centers]
    )
    image_patches = image_patches.to(cast(str, model.device))
    with torch.no_grad():
        predictions = model(image_patches).argmax(dim=-1).cpu().numpy()
    image = get_detection_image(image, centers, predictions, labels)

    return image
