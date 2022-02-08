import hydra
from omegaconf import DictConfig

from common.detection_module import DetectionModule


@hydra.main("configs", "base")
def main(cfg: DictConfig):
    model = DetectionModule.load_from_checkpoint(cfg.checkpoint)
    # TODO(frederik): create dynamic dataset from image directory
    # TODO(frederik): create predictions and persist svgs and numerical summaries


if __name__ == "__main__":
    main()
