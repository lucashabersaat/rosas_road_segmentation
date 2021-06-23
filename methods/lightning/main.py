import torch

import pytorch_lightning as pl

from models.unet import UNet
from models.unet_self_attention import UNetSelfAttention

from common.lightning.base import LitBase

from common.lightning.road_data_module import RoadDataModule
from common.predict import predict_and_write_submission


def fit_normally(model):
    data = RoadDataModule()

    gpu = torch.cuda.is_available()
    trainer = pl.Trainer(gpus=gpu, max_epochs=50)

    trainer.fit(model, data)


def load_model_and_write_submission(path, model_class):
    model = model_class.load_from_checkpoint(path)

    if torch.cuda.is_available():
        model.to(torch.device('cuda'))

    predict_and_write_submission(model, 'lightning_unet')


if __name__ == '__main__':
    # load_model_and_write_submission(LitUNet, "lightning_logs/version_4/checkpoints/epoch=53-step=1241.ckpt")
    fit_normally(LitBase(UNet()))
    fit_normally(LitBase(UNetSelfAttention()))
