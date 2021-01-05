import multiprocessing as mp
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomAffine,
    ToTensor,
)
from tqdm import tqdm


normalize = Compose([Normalize([0.3616, 0.4328, 0.3969], [0.1581, 0.1880, 0.1735])])
augment = Compose(
    [
        RandomAffine(degrees=180, translate=(0.2, 0.2), fillcolor=None),
        RandomHorizontalFlip(),
    ]
)


def compute_mean_std(dataset):
    loader = DataLoader(
        dataset, batch_size=512, num_workers=os.cpu_count() or 1, pin_memory=True
    )

    mean = 0.0
    std = 0.0
    for images, _ in tqdm(loader, total=len(dataset) // 512):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(dataset)
    std /= len(dataset)
    return mean, std


def imap_progress(f, iterable: List, flatten=False):
    with mp.Pool(processes=os.cpu_count()) as pool:
        results = []
        for result in tqdm(pool.imap(f, iterable), total=len(iterable)):
            if result:
                if flatten:
                    results.extend(result)
                else:
                    results.append(result)
    return results


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
    mask = x ** 2 + y ** 2 <= 4 ** 2
    center_mask[mask] = 1
    center_mask = pad_image(center_mask, patch_size)
    center_mask = crop_patch(center_mask, center, patch_size)
    center_mask = ToTensor()(center_mask)
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
    padded_image = pad_image(image, patch_size)
    image_patches = []
    for center in centers:
        center_mask = create_mask(
            height=image.shape[0],
            width=image.shape[1],
            center=center,
            patch_size=patch_size,
        )
        padded_patch = to_tensor(crop_patch(padded_image, center, patch_size))
        padded_patch = torch.cat([center_mask, padded_patch])
        image_patches.append(padded_patch)
    image_patches = torch.stack(image_patches)
    image_patches = image_patches.to(model.device)
    with torch.no_grad():
        predictions = model(image_patches).argmax(dim=-1).cpu().numpy()
    image = get_detection_image(image, centers, predictions, labels)

    return image
