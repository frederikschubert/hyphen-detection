from common.utils import visualize_predictions
from pathlib import Path
from typing import Tuple

import wandb
from loguru import logger
from tqdm import tqdm

from train import Arguments, HyphenDetection


class EvalArguments(Arguments):
    eval_dir: str
    model_checkpoint: str
    num_points_x: int = 20
    num_points_y: int = 17
    threshold: float = 0.8
    circle_radius: int = 4
    circle_color: Tuple[int, int, int] = (240, 240, 240)
    circle_thickness: int = 1


def main():
    args = EvalArguments().parse_args()
    wandb.init(
        project=args.project, tags=args.tags, job_type="eval", config=args.as_dict()
    )
    logger.info("Restoring model...")
    model: HyphenDetection = HyphenDetection.load_from_checkpoint(args.model_checkpoint)
    model.eval()
    images = list(Path(args.eval_dir).glob("**/*.png"))
    logger.info("Evaluating {} images...", len(images))
    for image_path in tqdm(images):
        logger.debug(image_path)
        image_with_pred = visualize_predictions(
            model, image_path, args.num_points_x, args.num_points_y
        )
        wandb.log({"result": wandb.Image(image_with_pred)})
        if args.debug:
            break


if __name__ == "__main__":
    main()
