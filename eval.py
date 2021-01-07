from common.svg import create_svg
import csv
import os
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from loguru import logger
from PIL import Image
from tqdm import tqdm

from common.hyphen_dataset import read_patch
from common.segmentation import create_segmentation, get_predictions
from common.utils import get_detection_image, visualize_predictions
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
    threshold: float = 0.95
    num_samples: int = np.infty
    create_segmentation: bool = False
    segmentation_granularity: int = 10

    def process_args(self):
        if self.debug:
            self.num_samples = 2


def main():
    args = EvalArguments().parse_args()
    wandb.init(
        project=args.project,
        tags=args.tags,
        job_type="eval",
        config=args.as_dict(),
        mode="disabled" if args.debug else "run",
    )
    if "SLURM_JOB_ID" in os.environ:
        wandb.config.update({"SLURM_JOB_ID": os.environ["SLURM_JOB_ID"]})
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
            grid_images_path = os.path.join(wandb.run.dir, "Bilder_mitRaster")
            os.makedirs(grid_images_path)
            for i, image_path in tqdm(enumerate(sorted(images)), total=len(images)):
                logger.trace(image_path)
                image = Image.open(image_path).convert("RGB")
                image = np.array(image)
                predictions = get_predictions(
                    model, image, centers, args.patch_size, args.threshold
                )
                percentage = round(sum(predictions) / len(predictions) * 100, 2)
                percentages.append(percentage)
                sample_name = (
                    os.path.basename(image_path).split(".")[0].split("_Laser")[0]
                )
                create_svg(
                    os.path.join(grid_images_path, f"{sample_name}.svg"),
                    image_path,
                    centers,
                    predictions,
                )
                row = OrderedDict(
                    {
                        "mineral": os.path.basename(args.eval_dir),
                        "sample": sample_name,
                        "points": len(centers),
                        "points_on_hyphae": sum(predictions),
                        "percent": percentage,
                    }
                )
                if args.create_segmentation:
                    segmentation = create_segmentation(
                        model,
                        image,
                        args.patch_size,
                        args.threshold,
                        args.segmentation_granularity,
                    )
                    wandb.log(
                        {
                            f"{sample_name}_segmentation": wandb.Image(
                                image,
                                masks={
                                    "predictions": {
                                        "mask_data": segmentation,
                                        "class_labels": {0: "background", 1: "hyphae"},
                                    }
                                },
                                caption=sample_name,
                            )
                        }
                    )
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
