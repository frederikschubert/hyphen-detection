import hydra
from omegaconf import DictConfig


@hydra.main("configs", "base")
def main(cfg: DictConfig):
    pass


if __name__ == "__main__":
    main()
