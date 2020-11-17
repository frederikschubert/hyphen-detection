import os
import random
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as loggers
import pytorch_lightning.metrics.functional.classification as metrics
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from loguru import logger
from sklearn.metrics import matthews_corrcoef
from tap import Tap
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from common.hyphen_dataset import HyphenDataset
from common.utils import visualize_predictions


class Arguments(Tap):
    name: Optional[str] = None
    project: str = "hyphen"
    tags: List[str] = []
    dataset: str = "./nobackup/dataset/"
    model_name: str = "efficientnet_b3"
    model_checkpoint: Optional[str] = None
    batch_size: int = 256
    learning_rate: float = 1e-3
    debug: bool = False
    balance_dataset: bool = False
    pretrained: bool = False
    fp16: bool = False
    accumulate_grad_batches: int = 1
    target_metric: str = "matthews_corrcoef"
    patch_size: int = 80

    def process_args(self):
        if self.debug:
            self.batch_size = 16
        if self.fp16:
            self.batch_size *= 2


class HyphenDetection(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams: Arguments = hparams

        self.model = timm.create_model(
            self.hparams.model_name,
            pretrained=self.hparams.pretrained,
            num_classes=2,
            scriptable=True,
        )
        self.train_dataset = HyphenDataset(
            self.hparams.dataset, patch_size=self.hparams.patch_size
        )
        self.val_dataset = HyphenDataset(
            self.hparams.dataset, split="val", patch_size=self.hparams.patch_size
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, images) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.loss(logits, labels)
        self.log_dict({"loss": loss})
        if batch_idx % 100 == 0:
            image_paths = random.choices(self.val_dataset.image_paths, k=4)
            for i, image_path in enumerate(image_paths):
                image = visualize_predictions(
                    self,
                    image_path,
                    centers=self.val_dataset.get_centers_for_image(image_path),
                    labels=self.val_dataset.get_labels_for_image(image_path),
                    patch_size=self.hparams.patch_size,
                )
                wandb.log(
                    {f"sample_{i}": wandb.Image(image, caption=image_path)},
                    commit=False,
                )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.loss(logits, labels)
        self.log("val_loss", loss)
        f1 = metrics.f1_score(logits, labels, 2)
        self.log("f1", f1)
        predictions = logits.argmax(dim=-1)
        return predictions, labels

    def validation_epoch_end(
        self, validation_step_outputs: List[Tuple[torch.Tensor, torch.Tensor]]
    ):
        predictions, labels = [], []
        for p, l in validation_step_outputs:
            predictions.append(p)
            labels.append(l)
        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
        accuracy = metrics.accuracy(predictions, labels, num_classes=2)
        mcc = matthews_corrcoef(labels.cpu().numpy(), predictions.cpu().numpy())
        wandb.log(
            {"conf_mat": wandb.plot.confusion_matrix(predictions, labels, [0, 1])},
            commit=False,
        )
        self.log("accuracy", accuracy)
        self.log("matthews_corrcoef", mcc)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=os.cpu_count() or 1,
            sampler=WeightedRandomSampler(
                self.train_dataset.weights, len(self.train_dataset)
            )
            if self.hparams.balance_dataset
            else None,
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
        return [optimizer], [
            {"scheduler": scheduler, "monitor": self.hparams.target_metric}
        ]


def main():
    args = Arguments().parse_args()
    model = HyphenDetection(args.as_dict())
    run = wandb.init(
        project=args.project,
        tags=args.tags,
        job_type="train",
        name=args.name,
    )
    logger = loggers.WandbLogger(experiment=run)
    trainer = pl.Trainer(
        gpus=-1,
        logger=logger,
        checkpoint_callback=callbacks.ModelCheckpoint(
            dirpath=os.path.join(run.dir, "checkpoints"),
            filename="hyphen_checkpoint.ckpt",
            monitor=args.target_metric,
            save_top_k=1,
        ),
        callbacks=[callbacks.LearningRateMonitor(logging_interval="step")],
        auto_lr_find=False,
        auto_scale_batch_size=None,
        fast_dev_run=args.debug,
        precision=16 if args.fp16 else 32,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    trainer.tune(model)
    trainer.fit(model)


if __name__ == "__main__":
    main()
