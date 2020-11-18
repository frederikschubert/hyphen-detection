import csv
import os
from typing import List, Literal, Tuple

import numpy as np
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from common.utils import (
    crop_patch,
    pad_image,
    transform,
)


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
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = np.array(image)
        padded_image = pad_image(image, self.patch_size)
        center = self.centers[index]
        label = self.labels[index]
        patch = crop_patch(padded_image, center, self.patch_size)
        patch = transform(patch)
        return patch, label
