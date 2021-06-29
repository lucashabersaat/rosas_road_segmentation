import sys

import torch
import pytorch_lightning as pl

from models.unet import UNet
from models.unet_self_attention import UNetSelfAttention
from models.unet_transformer import U_Transformer

from common.lightning.base import LitBase
from common.lightning.road_data_module import RoadDataModule
from common.predict import predict_and_write_submission


def fit_normally(model, data):

    gpu = int(torch.cuda.is_available())
    trainer = pl.Trainer(gpus=gpu, max_epochs=35)

    trainer.fit(model, data)

    name = 'lightning_' + str.lower(model.model.__class__.__name__)
    predict_and_write_submission(model, name, resize_to=192)


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
            model = LitBase(UNet())
            data = RoadDataModule(resize_to=384)
        elif model_name == 'unet_transformer':
            model = LitBase(U_Transformer(3, 1), loss_fn='noise_robust_dice')
            data = RoadDataModule(batch_size=1,  resize_to=192)
        elif model_name == 'self_attention_unet':
            model = LitBase(UNetSelfAttention())
            data = RoadDataModule(resize_to=384)
        else:
            raise Exception('unknown model')
    else:
        model = LitBase(UNet())
        data = RoadDataModule(resize_to=384)

    fit_normally(model, data)
    # load_model_and_write_submission(LitUNet, "lightning_logs_old/version_4/checkpoints/epoch=53-step=1241.ckpt")
