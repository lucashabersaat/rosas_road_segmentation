from typing import Optional
import torchmetrics
import torch
from torch.nn import Module, BCELoss
from torch.nn import functional as F
import pytorch_lightning as pl

from models.unet import UNet
from common.unet_transformer_includes import NoiseRobustDiceLoss


class LitBase(pl.LightningModule):

    def __init__(
            self, config,
            model: Optional[Module] = None,
            loss_fn='bce',
            learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['unet'])
        if model is None:
            model = UNet()
        self.model = model

        self.lr = config["lr"]
        loss_fn = config["loss_fn"]

        if loss_fn == 'bce':
            self.loss_fn = BCELoss()
        elif loss_fn == 'noise_robust_dice':
            self.loss_fn = NoiseRobustDiceLoss()

        #self.accuracy = torchmetrics.classification.accuracy.Accuracy()

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.model(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        #acc = self.accuracy(y_hat, y)
        self.log("ptl/train_loss", loss)
        #self.log("ptl/train_accuracy", acc)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('valid_loss', loss, on_step=True)
        #acc = self.accuracy(y_hat, y)
        return {"val_loss": loss}
    """ 
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    """

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x["val_loss"] for x in outputs]).mean()
        #avg_acc = torch.stack(
        #    [x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        #self.log("ptl/val_accuracy", avg_acc)
