import logging
import os
import random
from typing import Any, List, Tuple, cast

import timm
import timm.data
import timm.loss
import timm.optim
import timm.scheduler
import timm.utils
import torch
import torchmetrics
import wandb
from common.transforms import transforms_train, transforms_val
from data.hyphen_dataset import HyphenDataset
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from old.common.utils import visualize_predictions

# Adapted from https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055

log = logging.getLogger(__name__)


class TimmCosineLRSchedulerScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optim, **kwargs):
        self.init_lr = optim.param_groups[0]["lr"]
        self.timmsteplr = timm.scheduler.CosineLRScheduler(optim, **kwargs)
        super().__init__(optim, self)

    def __call__(self, epoch):
        desired_lr = self.timmsteplr.get_epoch_values(epoch)[0]
        mult = desired_lr / self.init_lr
        return mult


class DetectionModule(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.train_loss_fn = timm.loss.BinaryCrossEntropy(**cfg.train_loss_kwargs)
        self.val_loss_fn = torch.nn.CrossEntropyLoss()

        self.model = timm.create_model(
            cfg.model_name, num_classes=cfg.num_classes, **cfg.model_kwargs
        )
        self.train_acc = torchmetrics.Accuracy(
            num_classes=cfg.num_classes, average="macro"
        )
        self.val_acc = torchmetrics.Accuracy(
            num_classes=cfg.num_classes, average="macro"
        )
        self.val_acc_best = torchmetrics.MaxMetric()

        self.dataset = HyphenDataset(
            cfg,
            split="train",
            transforms=transforms_train(),
        )
        self.val_dataset = HyphenDataset(
            cfg,
            split="val",
            transforms=transforms_val(),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            self.logger.experiment.log(
                {
                    "sample_input_batch": wandb.Image(make_grid(x[:, 1:])),
                    "sample_masks": wandb.Image(make_grid(x[:, 0].unsqueeze(1))),
                },
                commit=False,
            )
        if batch_idx % 100 == 0:
            image_paths = random.choices(
                self.val_dataset._image_paths, k=self.cfg.dataset.num_samples
            )
            for i, image_path in enumerate(image_paths):
                image = visualize_predictions(
                    self,
                    image_path,
                    centers=self.val_dataset.get_centers_for_image(image_path),
                    labels=self.val_dataset.get_labels_for_image(image_path),
                    patch_size=self.cfg.dataset.patch_size,
                )
                self.logger.experiment.log(
                    {f"sample_{i}": wandb.Image(image, caption=image_path)},
                    commit=False,
                )
        logits = self.forward(x)
        loss = self.train_loss_fn(logits, y)
        preds = torch.argmax(logits, dim=-1)
        acc = self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.val_loss_fn(logits, y)
        preds = torch.argmax(logits, dim=-1)
        acc = self.val_acc(preds, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        optimizer = self.optimizers()
        if hasattr(optimizer, "sync_lookahead"):
            cast(timm.optim.lookahead.Lookahead, optimizer).sync_lookahead()

    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log(
            "val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True
        )

    def on_train_start(self) -> None:
        trainer: Trainer = self.trainer
        if trainer.num_gpus > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def configure_optimizers(self):
        optimizer = timm.optim.create_optimizer_v2(
            self.model, **self.cfg.optimizer_kwargs
        )

        scheduler = TimmCosineLRSchedulerScheduler(
            optimizer, **self.cfg.scheduler_kwargs
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            num_workers=os.cpu_count() or 1,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=os.cpu_count() or 1,
            pin_memory=True,
        )

    def on_epoch_end(self):
        self.train_acc.reset()
        self.val_acc.reset()
