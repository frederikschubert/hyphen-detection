import os
from itertools import product
from typing import List, Tuple, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from train import HyphenDetection


from common.hyphen_dataset import read_patch


class PatchDataset(Dataset):
    def __init__(self, image, centers, patch_size):
        self.image = image
        self.centers = centers
        self.patch_size = patch_size

    def __getitem__(self, index):
        return read_patch(self.image, self.centers[index], self.patch_size, pad=False)

    def __len__(self):
        return len(self.centers)


def get_predictions(
    model: HyphenDetection,
    image: np.ndarray,
    centers: List[Tuple[int, int]],
    patch_size: int,
    batch_size=128,
):
    predictions = []
    patches = PatchDataset(image, centers, patch_size)
    patches = DataLoader(
        patches, batch_size=batch_size, num_workers=os.cpu_count() or 1
    )
    class_weights = (
        model.train_dataset.class_sample_counts + model.val_dataset.class_sample_counts
    )
    for batch in patches:
        batch = batch.to(model.device)
        prediction = F.softmax(model(batch), dim=-1)
        prediction[:, 1] *= (
            class_weights[1] / class_weights[0]
        )  # multiply by class frequency
        prediction = prediction.argmax(dim=-1).cpu()
        predictions.append(prediction)
    predictions = torch.cat(predictions, dim=0).squeeze()
    return predictions.numpy()


def create_segmentation(
    model: HyphenDetection,
    image: np.ndarray,
    patch_size: int,
    granularity: int,
):
    height, width = image.shape[:-1]
    xs, ys = (
        np.linspace(granularity, width, width // granularity, dtype=np.int),
        np.linspace(granularity, height, height // granularity, dtype=np.int),
    )
    centers = cast(List[Tuple[int, int]], list(product(xs, ys)))
    predictions = get_predictions(model, image, centers, patch_size)
    segmentation = np.zeros(shape=[height, width, 3], dtype=np.uint8)
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
