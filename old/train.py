import os
import random
import tempfile
import math
from typing import List, Optional, Tuple
from loguru import logger
import shtab

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as loggers
import pytorch_lightning.metrics.functional.classification as metrics
import timm
import torch
import wandb
from sklearn.metrics import matthews_corrcoef
from tap import Tap
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import OneCycleLR

from common.confusion_matrix import confusion_matrix
from common.distributed_proxy_sampler import DistributedProxySampler
from common.hyphen_dataset import HyphenDataset
from common.partially_huberised_cross_entropy import PartiallyHuberisedCrossEntropyLoss
from common.utils import compute_mean_std, visualize_predictions


class Arguments(Tap):
    name: Optional[str] = None
    project: str = "hyphen"
    tags: List[str] = []
    dataset: str = "./nobackup/dataset_6/"
    model_name: str = "efficientnet_b3"
    model_checkpoint: Optional[str] = None
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    debug: bool = False
    balance_dataset: bool = False
    pretrained: bool = True
    fp16: bool = True
    accumulate_grad_batches: int = 1
    target_metric: str = "val_matthews_corrcoef"
    patch_size: int = 300
    tau: float = 10.0
    num_samples: int = 4
    assume_label_noise: bool = False
    unbiased_sampling_epochs: int = 0
    subsample: bool = False

    def process_args(self):
        if self.subsample and self.balance_dataset:
            raise ValueError(
                "--subsample and --balance_dataset cannot be true simultaneously"
            )
        super().process_args()

    def configure(self):
        shtab.add_argument_to(self)


class HyphenDetection(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.params: Arguments = Arguments().from_dict(hparams)
        self.save_hyperparameters()

        self.model = timm.create_model(
            self.params.model_name,
            pretrained=self.params.pretrained,
            num_classes=2,
            in_chans=4,
        )
        self.train_dataset = HyphenDataset(
            self.params.dataset,
            patch_size=self.params.patch_size,
            subsample=self.params.subsample,
        )
        self.val_dataset = HyphenDataset(
            self.params.dataset, split="val", patch_size=self.params.patch_size
        )
        self.loss = (
            PartiallyHuberisedCrossEntropyLoss(tau=self.params.tau)
            if self.params.assume_label_noise
            else CrossEntropyLoss()
        )
        self.val_loss = CrossEntropyLoss()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        return self.model(images)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        images, labels = batch
        if batch_idx == 0:
            self.logger.experiment.log(
                {
                    "sample_input_batch": wandb.Image(make_grid(images[:, 1:])),
                    "sample_masks": wandb.Image(make_grid(images[:, 0].unsqueeze(1))),
                },
                commit=False,
            )
        logits = self.forward(images)
        loss = self.loss(logits, labels)
        self.log("loss", loss, sync_dist=True)
        if batch_idx % 100 == 0:
            image_paths = random.choices(
                self.val_dataset._image_paths, k=self.params.num_samples
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
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.val_loss(logits, labels)
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
        mcc = torch.Tensor([matthews_corrcoef(labels, predictions)]).to(self.device)
        self.logger.experiment.log(
            {
                "val_conf_mat": confusion_matrix(
                    predictions, labels, [0, 1], self.logger.experiment
                )
            },
            commit=False,
        )

        self.log("val_matthews_corrcoef", mcc, sync_dist=True)

    def train_dataloader(self):
        enable_unbiased_sampling = (
            self.current_epoch
            >= self.params.epochs - self.params.unbiased_sampling_epochs
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            num_workers=os.cpu_count() or 1,
            sampler=DistributedProxySampler(
                WeightedRandomSampler(
                    np.ones_like(self.train_dataset.weights)
                    if enable_unbiased_sampling
                    else self.train_dataset.weights,
                    len(self.train_dataset),
                )
            )
            if self.params.balance_dataset
            else None,
            shuffle=False if self.params.balance_dataset else True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            num_workers=os.cpu_count() or 1,
            pin_memory=True,
        )

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.params.learning_rate,
            weight_decay=self.params.weight_decay,
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.params.learning_rate * self.trainer.num_gpus,
            total_steps=int(
                self.params.epochs
                * len(self.train_dataset)
                / (self.params.batch_size * self.trainer.num_gpus)
            ),
            pct_start=0.2,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def on_train_end(self):
        if (
            self.trainer.checkpoint_callback
            and self.trainer.checkpoint_callback.best_model_path
        ):
            logger.info("Saving Artifact...")
            artifact = wandb.Artifact("patch_detector", type="model")
            artifact.add_file(
                self.trainer.checkpoint_callback.best_model_path, "best.ckpt"
            )
            if self.trainer.checkpoint_callback.last_model_path:
                artifact.add_file(
                    self.trainer.checkpoint_callback.last_model_path, "last.ckpt"
                )
            self.logger.experiment.log_artifact(artifact)


def main():
    args = Arguments().parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = HyphenDetection(**args.as_dict())
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=-1,
        logger=loggers.WandbLogger(
            project=args.project,
            entity="frederikschubert",
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
            save_last=True,
        ),
        callbacks=[callbacks.LearningRateMonitor(logging_interval="step")],
        auto_lr_find=False,
        auto_scale_batch_size=None,
        fast_dev_run=args.debug,
        precision=16 if args.fp16 else 32,
        accumulate_grad_batches=args.accumulate_grad_batches,
        replace_sampler_ddp=not args.balance_dataset,
        accelerator="ddp",
        reload_dataloaders_every_epoch=args.unbiased_sampling_epochs > 0,
    )
    logger.info("Checkpoint directory {}", trainer.checkpoint_callback.dirpath)
    trainer.tune(model)
    trainer.fit(model)


if __name__ == "__main__":
    main()
