from itertools import product
import multiprocessing as mp
import os
from typing import List, Tuple
from PIL import Image
import cv2

import numpy as np
from torch import nn
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from .hyphen_dataset import transform


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


def visualize_predictions(
    model: nn.Module,
    image_path: str,
    bboxes: List[Tuple[int, int, int, int]],
    patch_size: int,
    threshold: float = 0.8,
    circle_radius: int = 4,
    circle_color: Tuple[int, int, int] = (240, 240, 240),
    circle_thickness: int = 1,
):
    image = Image.open(image_path).convert("RGB")
    image_patches = []
    centers: List[Tuple[int, int]] = []
    for bbox in bboxes:
        padded_image = torch.zeros(3, patch_size, patch_size)
        min_x, max_x, min_y, max_y = bbox
        padded_image[:, : max_y - min_y, : max_x - min_x] = transform(image)[
            :, min_y:max_y, min_x:max_x
        ]
        centers.append(
            (int(min_x + (max_x - min_x) / 2), int(min_y + (max_y - min_y) / 2))
        )
        image_patches.append(padded_image)
    image_patches = torch.stack(image_patches)
    image_patches = image_patches.to(model.device)
    with torch.no_grad():
        predictions = torch.sigmoid(model(image_patches)).cpu().numpy()
    for center, prediction in zip(centers, predictions):
        image = np.array(image)
        image = cv2.circle(
            image,
            center,
            circle_radius,
            circle_color,
            circle_thickness,
        )
        if prediction[1] > threshold:
            image = cv2.circle(
                image,
                center,
                circle_radius - 1,
                (255, 0, 0),
                -1,
            )
    return image
