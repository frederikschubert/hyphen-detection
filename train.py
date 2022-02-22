import logging
import tempfile

import hydra
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as loggers
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from common.detection_module import DetectionModule

log = logging.getLogger(__name__)


@hydra.main("configs", "base")
def main(cfg: DictConfig):
    model = DetectionModule(cfg)
    logger = loggers.WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        tags=cfg.wandb.tags,
        job_type="train",
        name=cfg.wandb.name,
        mode="disabled" if cfg.debug else "run",
    )
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=tempfile.mkdtemp(),
        filename="patch_detector",
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
    )
    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[callbacks.LearningRateMonitor(logging_interval="step")],
        fast_dev_run=cfg.debug,
        precision=16 if cfg.fp16 else 32,
        # replace_sampler_ddp=not cfg.balance_dataset,
        # accelerator="ddp",
        # TODO(frederik): implement balanced dataset
    )
    log.info("Checkpoint directory {}", checkpoint_callback.dirpath)
    trainer.fit(model)


if __name__ == "__main__":
    main()
