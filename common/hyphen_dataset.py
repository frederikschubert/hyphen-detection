import csv
import os
from typing import List, Literal, Tuple

import numpy as np
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from common.utils import augment, read_patch


class HyphenDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: Literal["train", "val"] = "train",
        patch_size: int = 80,
        subsample: bool = False,
    ):
        self.file = os.path.join(path, split, "annotations.csv")
        self.patch_size = patch_size
        self._image_paths: List[str] = []
        self._labels: List[int] = []
        self._centers: List[Tuple[int, int]] = []
        self.subsample = subsample
        with open(self.file, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            logger.info("Reading annotations...")
            for row in tqdm(reader):
                self._image_paths.append(row["image_path"])
                self._labels.append(int(row["label"]))
                center = (
                    int(row["x"]),
                    int(row["y"]),
                )
                self._centers.append(center)
        self._labels = np.array(self._labels)
        self._centers = np.array(self._centers)
        self._image_paths = np.array(self._image_paths)
        self.class_sample_counts = np.unique(self._labels, return_counts=True)[1]
        logger.info("Found the following class counts {}", self.class_sample_counts)
        if self.subsample:
            negative_indices = np.squeeze(np.argwhere(self._labels == 0))
            self.subsampled_indices = np.concatenate(
                [
                    np.random.choice(
                        negative_indices, self.class_sample_counts[1], replace=False
                    ),
                    np.squeeze(np.argwhere(self._labels == 1)),
                ]
            )
            logger.info("Subsampled dataset to size {}", len(self))
        else:
            self.weights = np.where(
                self._labels == 0,
                np.ones_like(self._labels, dtype=np.float32)
                * len(self._labels)
                / self.class_sample_counts[0],
                np.ones_like(self._labels, dtype=np.float32)
                * len(self._labels)
                / self.class_sample_counts[1],
            )
        logger.info("Loaded dataset")

    def get_percentage_for_image(self, query_image_path: str):
        labels = self.get_labels_for_image(query_image_path)
        return sum(labels) / len(labels)

    def get_centers_for_image(self, query_image_path: str):
        return [
            tuple(center.tolist())
            for center, image_path in zip(self._centers, self._image_paths)
            if query_image_path in image_path
        ]

    def get_labels_for_image(self, query_image_path: str):
        return [
            label
            for label, image_path in zip(self._labels, self._image_paths)
            if query_image_path in image_path
        ]

    @property
    def labels(self):
        if self.subsample:
            return self._labels[self.subsampled_indices]
        else:
            return self._labels

    @property
    def image_paths(self):
        if self.subsample:
            return self._image_paths[self.subsampled_indices]
        else:
            return self._image_paths

    @property
    def centers(self):
        if self.subsample:
            return self._centers[self.subsampled_indices]
        else:
            return self._centers

    def __len__(self):
        if self.subsample:
            return len(self.subsampled_indices)
        else:
            return len(self._image_paths)

    def __getitem__(self, index):
        center = self.centers[index]
        image = Image.open(self.image_paths[index]).convert("RGB")
        patch = read_patch(image, center, self.patch_size)
        patch = augment(patch)
        label = self.labels[index]
        return patch, label
