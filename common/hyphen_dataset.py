import csv
import os
from typing import Literal

import numpy as np
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.transforms import (
    Compose,
    Normalize,
    RandomRotation,
    ToTensor,
)
from tqdm import tqdm

transform = Compose(
    [
        ToTensor(),
        Normalize([4.9835, 5.9437, 5.4343], [22.4497, 26.7665, 24.4718]),
        RandomRotation(180),
    ]
)


class HyphenDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: Literal["train", "val"] = "train",
        transform=transform,
    ):
        self.file = os.path.join(path, split, "annotations.csv")
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.bboxes = []
        max_width, max_height = 0, 0
        with open(self.file, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            logger.info("Reading annotations...")
            for row in tqdm(reader):
                self.image_paths.append(row["image_path"])
                self.labels.append(int(row["label"]))
                bbox = [
                    int(row["bbox_min_x"]),
                    int(row["bbox_max_x"]),
                    int(row["bbox_min_y"]),
                    int(row["bbox_max_y"]),
                ]
                self.bboxes.append(bbox)
                width = bbox[1] - bbox[0]
                height = bbox[3] - bbox[2]
                max_width = width if width > max_width else max_width
                max_height = height if height > max_height else max_height
        self.labels = np.array(self.labels)
        class_sample_counts = np.unique(self.labels, return_counts=True)[1]
        logger.info("Found the following class counts {}", class_sample_counts)
        self.weights = np.where(
            self.labels == 0,
            np.ones_like(self.labels, dtype=np.float32) / class_sample_counts[0],
            np.ones_like(self.labels, dtype=np.float32) / class_sample_counts[1],
        )
        self.padded = np.zeros(shape=[3, max_height, max_width], dtype=np.float32)
        logger.info("Loaded dataset")

    def get_bboxes_for_image(self, query_image_path: str):
        return [
            bbox
            for bbox, image_path in zip(self.bboxes, self.image_paths)
            if query_image_path in image_path
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        image = np.array(image)

        bbox = self.bboxes[index]
        label = self.labels[index]

        padded_image = self.padded.copy()
        min_x, max_x, min_y, max_y = bbox
        padded_image[:, : max_y - min_y, : max_x - min_x] = image[
            :, min_y:max_y, min_x:max_x
        ]
        return padded_image, label
