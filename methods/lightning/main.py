import sys

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

    name = 'lightning_' + str.lower(arg_model.__class__.__name__)
    predict_and_write_submission(model, name)


def load_model_and_write_submission(path, model_class):
    model = model_class.load_from_checkpoint(path)

    if torch.cuda.is_available():
        model.to(torch.device('cuda'))

    predict_and_write_submission(model, 'lightning_unet')


if __name__ == '__main__':
    """The specific model can be given as argument to the program."""
    if sys.argv is not None:
        model_name = sys.argv[1]
        if model_name == 'unet':
            arg_model = UNet()
        elif model_name == 'self_attention_unet':
            arg_model = UNetSelfAttention()
        else:
            raise Exception('unknown model')
    else:
        arg_model = UNet()

    fit_normally(LitBase(arg_model))
    # load_model_and_write_submission(LitUNet, "lightning_logs/version_4/checkpoints/epoch=53-step=1241.ckpt")
