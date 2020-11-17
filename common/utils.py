import multiprocessing as mp
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
    ToTensor,
)
from tqdm import tqdm

transform = Compose(
    [
        ToTensor(),
        Normalize([4.9835, 5.9437, 5.4343], [22.4497, 26.7665, 24.4718]),
    ]
)

transform_positive = Compose(
    [
        ToTensor(),
        Normalize([4.9835, 5.9437, 5.4343], [22.4497, 26.7665, 24.4718]),
        RandomRotation(180),
        RandomHorizontalFlip(),
    ]
)

transform_test = Compose(
    [ToTensor(), Normalize([4.9835, 5.9437, 5.4343], [22.4497, 26.7665, 24.4718])]
)


def compute_mean_std(dataset, device):
    loader = DataLoader(
        dataset,
        batch_size=512,
        num_workers=os.cpu_count() or 1,
    )

    mean = 0.0
    std = 0.0
    for images, _ in tqdm(loader, total=len(dataset) // 512):
        images = images.to(device)
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


def visualize_predictions(
    model: nn.Module,
    image_path: str,
    centers: List[Tuple[int, int]],
    patch_size: int,
    labels: List[int] = [],
    circle_radius: int = 4,
    circle_color: Tuple[int, int, int] = (240, 240, 240),
    circle_thickness: int = 1,
):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    padded_image = pad_image(image, patch_size)
    image_patches = []
    for center in centers:
        padded_patch = crop_patch(padded_image, center, patch_size)
        padded_patch = transform_test(padded_patch)
        image_patches.append(padded_patch)
    image_patches = torch.stack(image_patches)
    image_patches = image_patches.to(model.device)
    with torch.no_grad():
        predictions = torch.softmax(model(image_patches), dim=-1).cpu().numpy()
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
        positive = prediction[1] > prediction[0]
        true_positive = positive and label == 1
        false_negative = not positive and label == 1
        if positive:
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
