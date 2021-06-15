from typing import Optional

import torch
from torch.nn import BCELoss

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pytorch_lightning.utilities.cli import LightningCLI

from methods.unet import UNet
from common.lightning.road_data_module import RoadDataModule
from common.predict import predict_and_write_submission


class LitUNet(pl.LightningModule):

    def __init__(
            self,
            unet: Optional[UNet] = None,
            learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['unet'])
        if unet is None:
            unet = UNet()
        self.unet = unet

        self.loss_fn = BCELoss()

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.unet(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.unet(x)
        loss = self.loss_fn(y_hat, y)

        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.unet(x)
        loss = self.loss_fn(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def cli_main():
    model = LitUNet()
    data = RoadDataModule()

    gpu = torch.cuda.is_available()
    trainer = pl.Trainer(gpus=gpu, max_epochs=50)

    trainer.fit(model, data)

    # cli = LightningCLI(LitClassifier, RoadDataModule, seed_everything_default=1234)
    # cli.trainer.test(cli.model, datamodule=cli.datamodule)


def load():
    model = LitUNet.load_from_checkpoint("lightning_logs/version_4/checkpoints/epoch=53-step=1241.ckpt")

    if torch.cuda.is_available():
        model.to(torch.device('cuda'))

    # for name, param in model.state_dict().items():
    #     print(type(param))
    predict_and_write_submission(model, 'lightning_unet')


if __name__ == '__main__':
    cli_lightning_logo()
    # cli_main()
    load()
