import sys

import torch
import numpy as np
import pytorch_lightning as pl


from models.unet import UNet
from models.unet_transformer import U_Transformer

from common.lightning.base import LitBase
from common.lightning.road_data_module import RoadDataModule
from common.image_data_set import TestImageDataSet
from common.write_data import write_submission


def fit_normally(model, data):
    gpu = int(torch.cuda.is_available())

    trainer = pl.Trainer(gpus=gpu, max_epochs=35, default_root_dir="data")

    # fit
    trainer.fit(model, datamodule=data)

    # predict
    predictions = trainer.predict(model, datamodule=data)
    predictions = [p.cpu().detach().numpy() for p in predictions]

    # put four corresponding images back together again
    if data.divide_into_four:
        predictions = TestImageDataSet.put_back(predictions)

    name = "lightning_" + str.lower(model.model.__class__.__name__)
    write_submission(
        predictions, name, "data/test_images/test_images", data.test_dataset.size
    )


# for later, if you want to load an already trained model
# def load_model_and_write_submission(path, model_class):
#     model = model_class.load_from_checkpoint(path)
#
#     if torch.cuda.is_available():
#         model.to(torch.device('cuda'))
#
#     predict_and_write_submission(model, 'lightning_unet')


if __name__ == "__main__":
    """The specific model can be given as argument to the program."""

    pl.utilities.seed.seed_everything(seed=1337)

    if sys.argv is not None:
        if len(sys.argv) == 1:
            model_name = "unet_transformer"
        else:
            model_name = sys.argv[1]

        if model_name == "unet":
            data = RoadDataModule(resize_to=384)
            model = LitBase(UNet(), data_module=data)
        elif model_name == "unet_transformer":

            data = RoadDataModule(batch_size=1, resize_to=384, divide_into_four=True )
            model = LitBase(
                U_Transformer(3, 1), loss_fn="noise_robust_dice", data_module=data
            )
        else:
            raise Exception("unknown model")
    else:
        model = LitBase(UNet())
        data = RoadDataModule(resize_to=384)

    fit_normally(model, data)
