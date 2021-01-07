from common.hyphen_dataset import read_patch
import numpy as np
from tqdm import tqdm
from train import HyphenDetection
from itertools import product
import torch.nn.functional as F


def create_segmentation(
    model: HyphenDetection,
    image: np.ndarray,
    patch_size: int,
    threshold: float,
    granularity: int,
):
    height, width = image.shape[:-1]
    xs, ys = np.linspace(0, width, granularity, dtype=np.int), np.linspace(
        0, height, granularity, dtype=np.int
    )
    centers = product(xs, ys)
    predictions = []
    for center in tqdm(centers):
        patch = read_patch(image, center, patch_size).to(model.device)
        prediction = F.softmax(model(patch).squeeze(), dim=-1)
        predictions.append(1 if prediction[1] > threshold else 0)
    segmentation = np.zeros(shape=[height, width], dtype=np.int)
    for center, prediction in zip(centers, predictions):
        x, y = center
        segmentation[y : y + granularity, x : x + granularity] = prediction
    return segmentation