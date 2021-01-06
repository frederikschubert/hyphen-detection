from collections import OrderedDict
import csv
from PIL import Image
import torch
import math
from common.hyphen_dataset import read_patch
import os
import torch.nn.functional as F

import numpy as np
from common.utils import get_detection_image, visualize_predictions
from pathlib import Path
from typing import Tuple

import wandb
from loguru import logger
from tqdm import tqdm

from train import Arguments, HyphenDetection


def get_artifact_path(
    artifact_name: str = "patch_detector",
    artifact_tag: str = "latest",
    project: str = "frederikschubert/hyphen",
):
    artifact_id = f"{artifact_name}:{artifact_tag}"
    artifact_path = (
        Path(os.getcwd())
        .absolute()
        .joinpath("nobackup")
        .joinpath("artifacts")
        .joinpath(artifact_id)
    )
    if not artifact_path.exists():
        os.makedirs(artifact_path, exist_ok=True)
        artifact = wandb.run.use_artifact(f"{project}/{artifact_id}")
        artifact.download(root=artifact_path)
    return Path(artifact_path)


class EvalArguments(Arguments):
    eval_dir: str = "/home/schubert/projects/hyphen/data/Chile_Input_2/LCN/Biotit"
    artifact_name: str = "patch_detector"
    artifact_tag: str = "latest"
    threshold: float = 0.99
    num_samples: int = np.infty


def main():
    args = EvalArguments().parse_args()
    wandb.init(
        project=args.project,
        tags=args.tags,
        job_type="eval",
        config=args.as_dict(),
        mode="disabled" if args.debug else "run",
    )
    logger.info("Restoring model...")
    model: HyphenDetection = HyphenDetection.load_from_checkpoint(
        str(get_artifact_path(args.artifact_name, args.artifact_tag) / "best.ckpt"),
        dataset=args.dataset,
    )
    model.to("cuda")
    model.eval()
    images = list(Path(args.eval_dir).glob("**/*Laser.bmp"))
    logger.info("Evaluating {} images...", len(images))
    centers = model.val_dataset.get_centers_for_image(model.val_dataset.image_paths[0])
    logger.debug("Centers: {}", len(centers))
    with torch.no_grad():
        with open(
            os.path.join(wandb.run.dir, "percentages.csv"), "w", newline=""
        ) as csv_file:
            fieldnames = [
                "mineral",
                "sample",
                "points",
                "points_on_hyphae",
                "percent",
            ]
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            percentages = []
            rows = []
            for i, image_path in tqdm(enumerate(sorted(images)), total=len(images)):
                logger.trace(image_path)
                image = Image.open(image_path).convert("RGB")
                image = np.array(image)
                predictions = []
                for center in tqdm(centers):
                    patch = read_patch(image, center, args.patch_size).to(model.device)
                    prediction = F.softmax(model(patch).squeeze(), dim=-1)
                    predictions.append(1 if prediction[1] > args.threshold else 0)
                percentage = round(sum(predictions) / len(predictions) * 100, 2)
                percentages.append(percentage)
                row = OrderedDict(
                    {
                        "mineral": os.path.basename(args.eval_dir),
                        "sample": os.path.basename(image_path)
                        .split(".")[0]
                        .split("_Laser")[0],
                        "points": len(centers),
                        "points_on_hyphae": sum(predictions),
                        "percent": percentage,
                    }
                )
                # TODO(frederik): create svg file with markers
                # TODO(frederik): generalize this to multiple minerals
                writer.writerow(row)
                rows.append(list(row.values()))
                csv_file.flush()
                if i > args.num_samples:
                    break
            mean_percentage = round(np.mean(percentages), 2)
            logger.info(
                "Mean Percentage for {} is {}",
                args.eval_dir,
                mean_percentage,
            )
            wandb.log(
                {
                    "mean_percentage": mean_percentage,
                    "results": wandb.Table(columns=fieldnames, data=rows),
                }
            )


if __name__ == "__main__":
    main()
