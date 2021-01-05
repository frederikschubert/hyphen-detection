import csv
import os
from typing import List, Literal, Tuple

import numpy as np
from loguru import logger
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.transforms import ToTensor
from tqdm import tqdm

from common.utils import create_mask, crop_patch, pad_image, augment


def read_patch(image, center, patch_size: int):
    image = np.array(image)
    padded_image = pad_image(image, patch_size)
    patch = to_tensor(crop_patch(padded_image, center, patch_size))
    center_mask = create_mask(
        height=image.shape[0],
        width=image.shape[1],
        center=center,
        patch_size=patch_size,
    )
    patch = torch.cat([center_mask, patch])
    return patch


class HyphenDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: Literal["train", "val"] = "train",
        patch_size: int = 80,
    ):
        self.file = os.path.join(path, split, "annotations.csv")
        self.patch_size = patch_size
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.centers: List[Tuple[int, int]] = []
        with open(self.file, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            logger.info("Reading annotations...")
            for row in tqdm(reader):
                self.image_paths.append(row["image_path"])
                self.labels.append(int(row["label"]))
                center = (
                    int(row["x"]),
                    int(row["y"]),
                )
                self.centers.append(center)
        self.labels = np.array(self.labels)
        self.class_sample_counts = np.unique(self.labels, return_counts=True)[1]
        logger.info("Found the following class counts {}", self.class_sample_counts)
        self.weights = np.where(
            self.labels == 0,
            np.ones_like(self.labels, dtype=np.float32)
            * len(self.labels)
            / self.class_sample_counts[0],
            np.ones_like(self.labels, dtype=np.float32)
            * len(self.labels)
            / self.class_sample_counts[1],
        )
        logger.info("Loaded dataset")

    def get_percentage_for_image(self, query_image_path: str):
        labels = self.get_labels_for_image(query_image_path)
        return sum(labels) / len(labels)

    def get_centers_for_image(self, query_image_path: str):
        return [
            center
            for center, image_path in zip(self.centers, self.image_paths)
            if query_image_path in image_path
        ]

    def get_labels_for_image(self, query_image_path: str):
        return [
            label
            for label, image_path in zip(self.labels, self.image_paths)
            if query_image_path in image_path
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        center = self.centers[index]
        image = Image.open(self.image_paths[index]).convert("RGB")
        patch = read_patch(image, center, self.patch_size)
        patch = augment(patch)
        label = self.labels[index]
        return patch, label
