import numpy as np
import tempfile
from numpy.core.arrayprint import array2string
from common.confusion_matrix import confusion_matrix
from common.partially_huberised_cross_entropy import PartiallyHuberisedCrossEntropyLoss
from common.distributed_proxy_sampler import DistributedProxySampler
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as loggers
import pytorch_lightning.metrics.functional.classification as metrics
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid
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
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    debug: bool = False
    balance_dataset: bool = False
    pretrained: bool = False
    fp16: bool = False
    accumulate_grad_batches: int = 1
    target_metric: str = "matthews_corrcoef"
    patch_size: int = 80
    tau: float = 5.0
    num_samples: int = 4

    def process_args(self):
        if self.debug:
            self.batch_size = 16


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
        self.loss = PartiallyHuberisedCrossEntropyLoss(tau=self.hparams.tau)

    def forward(self, images) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        if batch_idx == 0:
            self.logger.experiment.log(
                {"sample_input_batch": wandb.Image(make_grid(images))}, commit=False
            )
        logits = self.forward(images)
        loss = self.loss(logits, labels)
        self.log("loss", loss, sync_dist=True)
        if batch_idx % 100 == 0:
            image_paths = random.choices(
                self.val_dataset.image_paths, k=self.hparams.num_samples
            )
            for i, image_path in enumerate(image_paths):
                image = visualize_predictions(
                    self,
                    image_path,
                    centers=self.val_dataset.get_centers_for_image(image_path),
                    labels=self.val_dataset.get_labels_for_image(image_path),
                    patch_size=self.hparams.patch_size,
                )
                self.logger.experiment.log(
                    {f"sample_{i}": wandb.Image(image, caption=image_path)},
                    commit=False,
                )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.loss(logits, labels)
        self.log("val_loss", loss, sync_dist=True)
        f1 = metrics.f1_score(logits, labels, 2)
        self.log("val_f1", f1, sync_dist=True)
        predictions = logits.argmax(dim=-1)
        return predictions, labels

    def validation_epoch_end(
        self, validation_step_outputs: List[Tuple[torch.Tensor, torch.Tensor]]
    ):
        predictions, labels = [], []
        for p, l in validation_step_outputs:
            predictions.append(p)
            labels.append(l)
        predictions = torch.cat(predictions).cpu()
        labels = torch.cat(labels).cpu()
        accuracy = metrics.accuracy(predictions, labels, num_classes=2)
        self.log("val_accuracy", accuracy)
        labels = labels.numpy()
        predictions = predictions.numpy()
        mcc = torch.Tensor([matthews_corrcoef(labels, predictions)]).type_as(
            validation_step_outputs[0][0]
        )
        self.logger.experiment.log(
            {
                "val_conf_mat": confusion_matrix(
                    predictions, labels, [0, 1], self.logger.experiment
                )
            },
            commit=False,
        )

        self.log("matthews_corrcoef", mcc, sync_dist=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=os.cpu_count() or 1,
            sampler=DistributedProxySampler(
                WeightedRandomSampler(
                    self.train_dataset.weights, len(self.train_dataset)
                )
            )
            if self.hparams.balance_dataset
            else None,
            shuffle=False if self.hparams.balance_dataset else True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=os.cpu_count() or 1,
            pin_memory=True,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate * self.trainer.num_gpus,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [
            {"scheduler": scheduler, "monitor": self.hparams.target_metric}
        ]

    def on_train_end(self):
        artifact = wandb.Artifact("patch_detector", type="model")
        artifact.add_file(
            os.path.join(
                self.trainer.checkpoint_callback.dirpath, "patch_detector.ckpt"
            )
        )
        self.logger.experiment.log_artifact(artifact)


def main():
    args = Arguments().parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = HyphenDetection(args.as_dict())
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=-1,
        logger=loggers.WandbLogger(
            project=args.project,
            tags=args.tags,
            job_type="train",
            name=args.name,
            mode="disabled" if args.debug else "run",
        ),
        checkpoint_callback=callbacks.ModelCheckpoint(
            dirpath=tempfile.mkdtemp(),
            filename="patch_detector",
            monitor=args.target_metric,
            save_top_k=1,
        ),
        callbacks=[callbacks.LearningRateMonitor(logging_interval="step")],
        auto_lr_find=False,
        auto_scale_batch_size=None,
        fast_dev_run=args.debug,
        precision=16 if args.fp16 else 32,
        accumulate_grad_batches=args.accumulate_grad_batches,
        replace_sampler_ddp=False,
        accelerator="ddp",
    )
    trainer.tune(model)
    trainer.fit(model)


if __name__ == "__main__":
    main()
