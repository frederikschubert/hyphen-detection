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
    num_points_x: int = 20,
    num_points_y: int = 17,
    circle_radius: int = 4,
    circle_color: Tuple[int, int, int] = (240, 240, 240),
    circle_thickness: int = 1,
    patch_size: int=40,
):
    image = Image.open(image_path).convert("RGB")
    image_patches = []
    bboxes, centers = get_bboxes_and_centers(
        image.height, image.width, num_points_x, num_points_y
    )
    for bbox in bboxes:
        padded_image = torch.zeros(3, patch_size, patch_size)
        padded_image[:, : bbox[3] - bbox[2], : bbox[1] - bbox[0]] = transform(image)[
            :, bbox[2] : bbox[3], bbox[0] : bbox[1]
        ]
        image_patches.append(padded_image)
    image_patches = torch.stack(image_patches)
    image_patches = image_patches.to(model.device)
    with torch.no_grad():
        predictions = model(image_patches).cpu().numpy()
    for center, prediction in zip(centers, predictions):
        image = np.array(image)
        image = cv2.circle(
            image,
            center,
            circle_radius,
            circle_color,
            circle_thickness,
        )
        if prediction[1] > prediction[0]:
            image = cv2.circle(
                image,
                center,
                circle_radius - 1,
                (255, 0, 0),
                -1,
            )
    return image


def get_bboxes_and_centers(
    height: int, width: int, num_points_x: int, num_points_y: int
):
    bboxes = []
    centers = []
    min_xs = np.round(np.linspace(0, width, num_points_x, endpoint=False))
    min_ys = np.round(np.linspace(0, height, num_points_y, endpoint=False))
    for i, min_x in enumerate(min_xs):
        for j, min_y in enumerate(min_ys):
            max_x = min_xs[i + 1] if i < len(min_xs) - 1 else width
            max_y = min_ys[j + 1] if j < len(min_ys) - 1 else height
            center_x = int(min_x + (max_x - min_x) / 2)
            center_y = int(min_y + (max_y - min_y) / 2)
            centers.append((center_x, center_y))
            bboxes.append((int(min_x), int(max_x), int(min_y), int(max_y)))
    return bboxes, centers
