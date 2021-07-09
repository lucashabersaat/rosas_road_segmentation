from typing import Optional
import torch
from torch.nn import Module, BCELoss
import pytorch_lightning as pl

from common.plot_data import show_two_imgs_overlay
from models.unet import UNet
from common.losses import NoiseRobustDiceLoss, DiceLoss


class LitBase(pl.LightningModule):

    def __init__(
            self, config,
            model: Optional[Module] = None,
    ):
        super().__init__()

        if model is None:
            model = UNet()
        self.model = model

        self.lr = config["lr"]
        loss_fn = config["loss_fn"]

        self.divide_into_four = config.get("divide_into_four")
        self.batch_size = config.get("batch_size")
        self.resize_to = config.get("resize_to")

        hyper_parameters = {
            "model": model, "loss_fn": loss_fn, "learning_rate": self.lr, "config": config,
        }

        self.save_hyperparameters(hyper_parameters)

        if loss_fn == 'bce':
            self.loss_fn = BCELoss()
        elif loss_fn == 'noise_robust_dice':
            self.loss_fn = NoiseRobustDiceLoss()
        elif loss_fn == 'dice_loss':
            self.loss_fn = DiceLoss()

        # self.accuracy = torchmetrics.classification.accuracy.Accuracy()

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.model(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        # acc = self.accuracy(y_hat, y)
        self.log("ptl/train_loss", loss)
        # self.log("ptl/train_accuracy", acc)
        self.log('train_loss', loss, on_epoch=True)

        if batch_idx == 0:
            print_y = torch.moveaxis(y.cpu(), 0, -1)
            print_yhat = torch.moveaxis(y_hat.cpu().detach(), 0, -1)

            #show_two_imgs_overlay(print_yhat[0], print_y[0])

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('valid_loss', loss, on_step=True)
        # acc = self.accuracy(y_hat, y)
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
        # avg_acc = torch.stack(
        #    [x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        # self.log("ptl/val_accuracy", avg_acc)
