import os
from typing import Tuple, cast
from omegaconf import DictConfig
import timm
import timm.data
import timm.utils
import timm.loss
import timm.optim
import timm.scheduler
from pytorch_lightning import LightningModule, Trainer
import torch
import torchmetrics
from torch.utils.data import DataLoader
from data.hyphen_dataset import HyphenDataset

# Adapted from https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055


class DetectionModule(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.mixup_fn = timm.data.Mixup(**cfg.mixup_kwargs)
        self.train_loss_fn = timm.loss.BinaryCrossEntropy(**cfg.train_loss_kwargs)
        self.val_loss_fn = torch.nn.CrossEntropyLoss()

        self.model = timm.create_model(
            cfg.model_name, num_classes=cfg.num_classes, **cfg.model_kwargs
        )
        self.train_acc = torchmetrics.Accuracy(num_classes=cfg.num_classes)
        self.val_acc = torchmetrics.Accuracy(num_classes=cfg.num_classes)
        self.val_acc_best = torchmetrics.MaxMetric()

        self.ema_accuracy = torchmetrics.Accuracy(num_classes=cfg.num_classes)

        self.dataset = HyphenDataset(
            cfg,
            split="train",
            transform=timm.data.create_transform(
                input_size=cfg.dataset.patch_size,
                is_training=True,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                auto_augment="rand-m7-mstd0.5-inc1",
            ),
        )
        self.val_dataset = HyphenDataset(
            cfg,
            split="val",
            transform=timm.data.create_transform(
                input_size=cfg.dataset.patch_size,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        mixup_x, mixup_y = self.mixup_fn(x, y)
        logits = self.forward(mixup_x)
        loss = self.train_loss_fn(logits, mixup_y)
        preds = torch.argmax(logits, dim=-1)
        acc = self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        logits = self.forward(x)
        loss = self.val_loss_fn(logits, y)
        preds = torch.argmax(logits, dim=-1)
        acc = self.val_acc(preds, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        ema_preds = self.ema_model(x).argmax(dim=-1)
        self.ema_accuracy.update(ema_preds, y)
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        self.ema_model.update(self.model)
        self.ema_model.eval()

        if hasattr(self.optimizer, "sync_lookahead"):
            cast(timm.optim.lookahead.Lookahead, self.optimizer).sync_lookahead()

    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log(
            "val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True
        )

    def on_train_start(self) -> None:
        trainer: Trainer = self.trainer
        self.ema_model = timm.utils.ModelEmaV2(self.model, **self.cfg.ema_kwargs)
        if trainer.num_gpus > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def configure_optimizers(self):
        optimizer = timm.optim.create_optimizer_v2(
            self.model, **self.cfg.optimizer_kwargs
        )
        scheduler = timm.scheduler.CosineLRScheduler(
            optimizer, **self.cfg.scheduler_kwargs
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
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
        self.ema_accuracy.reset()
