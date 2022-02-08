import logging
import tempfile
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

import pytorch_lightning.loggers as loggers
import pytorch_lightning.callbacks as callbacks
from common.detection_module import DetectionModule

log = logging.getLogger(__name__)


@hydra.main("configs", "base")
def main(cfg: DictConfig):
    model = DetectionModule(cfg)
    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=loggers.WandbLogger(
            project=cfg.project,
            entity=cfg.entity,
            tags=cfg.tags,
            job_type="train",
            name=cfg.name,
            mode="disabled" if cfg.debug else "run",
        ),
        checkpoint_callback=callbacks.ModelCheckpoint(
            dirpath=tempfile.mkdtemp(),
            filename="patch_detector",
            monitor=cfg.target_metric,
            save_top_k=1,
            save_last=True,
        ),
        callbacks=[callbacks.LearningRateMonitor(logging_interval="step")],
        fast_dev_run=cfg.debug,
        precision=16 if cfg.fp16 else 32,
        # replace_sampler_ddp=not cfg.balance_dataset,
        # accelerator="ddp",
    )
    log.info("Checkpoint directory {}", trainer.checkpoint_callback.dirpath)
    trainer.fit(model)


if __name__ == "__main__":
    main()
