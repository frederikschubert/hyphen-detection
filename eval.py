from PIL import Image
import torch
import math
from common.hyphen_dataset import read_patch
import os

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

    def process_args(self):
        self.debug = True


def main():
    args = EvalArguments().parse_args()
    wandb.init(
        project=args.project,
        tags=args.tags,
        job_type="eval",
        config=args.as_dict(),
        mode="dryrun" if args.debug else "run",
    )
    logger.info("Restoring model...")
    model: HyphenDetection = HyphenDetection.load_from_checkpoint(
        str(get_artifact_path(args.artifact_name, args.artifact_tag) / "best.ckpt")
    )
    model.to("cuda")
    model.eval()
    images = list(Path(args.eval_dir).glob("**/*Laser.bmp"))
    logger.info("Evaluating {} images...", len(images))
    centers = model.val_dataset.get_centers_for_image(model.val_dataset.image_paths[0])
    logger.debug("Centers: {}", len(centers))
    with torch.no_grad():
        for i, image_path in tqdm(enumerate(images), total=len(images)):
            logger.debug(image_path)
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)
            predictions = []
            for center in tqdm(centers):
                patch = read_patch(image, center, args.patch_size).to(model.device)
                prediction = model(patch).argmax().item()
                predictions.append(prediction)
            logger.info("Percentage: {}", sum(predictions) / len(predictions))
            if args.debug:
                detection_image = get_detection_image(image, centers, predictions)
                wandb.log(
                    {
                        f"sample_{i}": wandb.Image(
                            detection_image, caption=str(image_path)
                        )
                    },
                )
                break


if __name__ == "__main__":
    main()
