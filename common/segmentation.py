import os
from itertools import product

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from train import HyphenDetection

from common.hyphen_dataset import read_patch


class PatchDataset(Dataset):
    def __init__(self, image, centers, patch_size):
        self.image = image
        self.centers = centers
        self.patch_size = patch_size

    def __getitem__(self, index):
        return read_patch(self.image, self.centers[index], self.patch_size)

    def __len__(self):
        return len(self.centers)


def get_predictions(model, image, centers, patch_size, threshold, batch_size=128):
    predictions = []
    patches = PatchDataset(image, centers, patch_size)
    patches = DataLoader(
        patches, batch_size=batch_size, num_workers=os.cpu_count() or 1
    )
    for batch in patches:
        batch = batch.to(model.device)
        prediction = F.softmax(model(batch), dim=-1)
        prediction[:, 1] *= 0.0125  # multiply by class frequency
        predictions.append(prediction.argmax(dim=-1).cpu())
    predictions = torch.cat(predictions, dim=0).squeeze()
    # predictions[predictions < threshold] = 0
    # predictions[predictions >= threshold] = 1
    return predictions.numpy()


def create_segmentation(
    model: HyphenDetection,
    image: np.ndarray,
    patch_size: int,
    threshold: float,
    granularity: int,
):
    height, width = image.shape[:-1]
    xs, ys = (
        np.linspace(granularity, width, width // granularity, dtype=np.int),
        np.linspace(granularity, height, height // granularity, dtype=np.int),
    )
    centers = list(product(xs, ys))
    predictions = get_predictions(model, image, centers, patch_size, threshold)
    segmentation = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    # segmentation = np.zeros(shape=[height, width], dtype=np.uint8)
    segmentation_width = granularity // 2
    for center, prediction in zip(centers, predictions):
        x, y = center
        segmentation[
            y - segmentation_width : y + segmentation_width,
            x - segmentation_width : x + segmentation_width,
        ] = prediction
    kernel = np.ones((granularity * 2, granularity * 2), np.uint8)
    segmentation = cv2.morphologyEx(segmentation, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((segmentation_width, segmentation_width), np.uint8)
    segmentation = cv2.morphologyEx(segmentation, cv2.MORPH_ERODE, kernel)
    segmentation = segmentation[:, :, 0]
    return segmentation
