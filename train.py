import logging
import os

import hydra
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as loggers
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from tqdm.contrib.logging import logging_redirect_tqdm
import wandb
from common.detection_module import DetectionModule

log = logging.getLogger(__name__)


@hydra.main("configs", "base")
def main(cfg: DictConfig):
    with logging_redirect_tqdm():
        model = DetectionModule(cfg)
        logger = loggers.WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=cfg.wandb.tags,
            job_type="train",
            name=cfg.wandb.name,
            mode="disabled" if cfg.debug else "run",
            save_dir=os.getcwd(),
            settings=wandb.Settings(start_method="thread"),
        )
        checkpoint_callback = callbacks.ModelCheckpoint(
            dirpath=os.getcwd(),
            filename="patch_detector",
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
        )
        trainer = Trainer(
            gpus=1,
            max_epochs=cfg.epochs,
            logger=logger,
            enable_checkpointing=checkpoint_callback,
            callbacks=[callbacks.LearningRateMonitor(logging_interval="step")],
            fast_dev_run=cfg.debug,
            precision=16 if cfg.fp16 else 32,
        )
        log.info(f"Checkpoint directory {checkpoint_callback.dirpath}")
        trainer.fit(model)


if __name__ == "__main__":
    main()
