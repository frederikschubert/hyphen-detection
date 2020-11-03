import csv
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from loguru import logger


class HyphenDataset(Dataset):
    def __init__(self, file: str, transform=None):
        self.file = file
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.bboxes = []
        max_width, max_height = 0, 0
        with open(self.file, newline="") as csvfile:
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
        self.padded = np.zeros(shape=[max_height, max_width, 3], dtype=np.float32)

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
        padded_image[: bbox[3] - bbox[2], : bbox[1] - bbox[0]] = image[
            bbox[2] : bbox[3], bbox[0] : bbox[1]
        ]
        padded_image = np.transpose(padded_image, [-1, -2, 0])
        return padded_image, label
