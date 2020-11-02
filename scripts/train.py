import os
from typing import Optional

import pytorch_lightning as pl
import timm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common.hyphen_dataset import HyphenDataset
from common.utils import get_train_val_samplers
from tap import Tap
from torch import optim
from torch.utils.data.dataloader import DataLoader


class Arguments(Tap):
    dataset: str = "./nobackup/dataset/annotations.csv"
    model_name: str = "efficientnet_b0"
    checkpoint: Optional[str] = ""
    batch_size: int = 128
    learning_rate: float = 1e-3
    debug: bool = False
    val_split: float = 0.2
    shuffle: bool = True


class HyphenDetection(pl.LightningModule):
    def __init__(self, args: Arguments):
        super().__init__()
        if type(args) == dict:
            args = Arguments().from_dict(args)
        self.args = args
        self.save_hyperparameters(args.as_dict())
        self.dataset = HyphenDataset(args.dataset)
        self.model = timm.create_model(
            args.model_name, num_classes=2, checkpoint_path=args.checkpoint
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_sampler, self.val_sampler = get_train_val_samplers(
            self.dataset, args.val_split, args.shuffle
        )

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self.forward(images)
        loss = self.criterion(predictions, F.one_hot(labels, num_classes=2).float())
        self.log_dict({"loss": loss})
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self.forward(images)
        loss = self.criterion(predictions, F.one_hot(labels, num_classes=2).float())
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            pin_memory=True,
            num_workers=os.cpu_count(),
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            pin_memory=True,
            num_workers=os.cpu_count(),
            sampler=self.val_sampler,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]


def main():
    args = Arguments().parse_args()
    model = HyphenDetection(args)
    trainer = pl.Trainer(
        gpus=-1,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
