from typing import Optional

import torch
from torch.nn import Module, BCELoss

import pytorch_lightning as pl

from models.unet import UNet
from common.unet_transformer_includes import NoiseRobustDiceLoss


class LitBase(pl.LightningModule):

    def __init__(
            self,
            model: Optional[Module] = None,
            loss_fn='bce',
            learning_rate: float = 0.0001,
            data_module=None
    ):
        super().__init__()

        if model is None:
            model = UNet()
        self.model = model

        hyper_parameters = {
            "model": model, "loss_fn": loss_fn, "learning_rate": learning_rate,
            "divide_into_four": data_module.divide_into_four, "batch_size": data_module.batch_size
        }
        self.save_hyperparameters(hyper_parameters)

        if loss_fn == 'bce':
            self.loss_fn = BCELoss()
        elif loss_fn == 'noise_robust_dice':
            self.loss_fn = NoiseRobustDiceLoss()

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.model(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
