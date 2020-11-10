import torch
from torch.utils.data.sampler import WeightedRandomSampler
from common.utils import visualize_predictions
import os
from typing import List, Optional
from loguru import logger
import random

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as loggers
import pytorch_lightning.metrics.functional.classification as metrics
import timm
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common.hyphen_dataset import HyphenDataset
from tap import Tap
from torch import optim
from torch.utils.data.dataloader import DataLoader


class Arguments(Tap):
    project: str = "hyphen"
    tags: List[str] = []
    dataset: str = "./nobackup/dataset/"
    model_name: str = "efficientnet_b0"
    model_checkpoint: Optional[str] = None
    batch_size: int = 256
    learning_rate: float = 1e-3
    debug: bool = False

    def process_args(self):
        if self.debug:
            self.batch_size = 16


class HyphenDetection(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams: Arguments = hparams
        self.save_hyperparameters()

        self.model = timm.create_model(
            self.hparams.model_name, num_classes=2, scriptable=True
        )
        self.train_dataset = HyphenDataset(self.hparams.dataset)
        self.val_dataset = HyphenDataset(self.hparams.dataset, split="val")

    def forward(self, images) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = nn.BCEWithLogitsLoss()(logits, F.one_hot(labels, num_classes=2).float())
        self.log_dict({"loss": loss})
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = nn.BCEWithLogitsLoss()(logits, F.one_hot(labels, num_classes=2).float())
        predictions = logits.argmax(dim=-1)
        accuracy = metrics.accuracy(predictions, labels, num_classes=2)
        if batch_idx == 0:
            image = visualize_predictions(
                self,
                random.choice(self.val_dataset.image_paths),
                patch_size=max(self.train_dataset.padded.shape),
            )
            wandb.log({"sample": wandb.Image(image)})
            wandb.log(
                {"conf_mat": wandb.plot.confusion_matrix(predictions, labels, [0, 1])}
            )
        log = {"val_loss": loss, "accuracy": accuracy}
        self.log_dict(log)
        return log

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=os.cpu_count() or 1,
            sampler=WeightedRandomSampler(
                self.train_dataset.weights, len(self.train_dataset)
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=os.cpu_count() or 1,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]


def main():
    args = Arguments().parse_args()
    model = HyphenDetection(args.as_dict())
    run = wandb.init(
        project=args.project, tags=args.tags, config=args.as_dict(), job_type="train"
    )
    logger = loggers.WandbLogger(experiment=run)
    trainer = pl.Trainer(
        gpus=-1,
        logger=logger,
        checkpoint_callback=callbacks.ModelCheckpoint(
            dirpath=os.path.join(run.dir, "checkpoints"),
            filename="hyphen_checkpoint.ckpt",
            monitor="val_loss",
            save_top_k=1,
        ),
        auto_lr_find=False,
        auto_scale_batch_size=None,
        fast_dev_run=args.debug,
    )
    trainer.tune(model)
    trainer.fit(model)


if __name__ == "__main__":
    main()
