import os
from argparse import ArgumentParser

import torch
import numpy as np
import pytorch_lightning as pl

from models.unet import UNet
from models.unet_transformer import U_Transformer

from common.lightning.base import LitBase
from common.lightning.road_data_module import RoadDataModule
from common.image_data_set import TestImageDataSet
from common.write_data import write_submission
from common.plot_data import *
from common.postprocess import *


def gpu():
    return int(torch.cuda.is_available())


def predict(trainer, model, data):
    # predict
    predictions = trainer.predict(model, datamodule=data)

    predictions = [p.detach().cpu().numpy() for p in predictions]

    # put four corresponding images back together again
    if data.divide_into_four:
        predictions = TestImageDataSet.put_back(predictions)
    else:
        predictions = np.concatenate(predictions)
        predictions = np.moveaxis(predictions, 1, -1).squeeze()


    post_process = True
    if post_process:
        predictions = blur(predictions)


    # img = np.moveaxis(data.test_dataset.x, 1, -1)
    # show_two_imgs(img[0], predictions)
    # for i in range(0, 90, 5):
    #     show_first_n(img[i:], predictions[i:])


    name = "lightning_" + str.lower(model.model.__class__.__name__)
    write_submission(
        predictions, name, "data/test_images/test_images", data.test_dataset.size
    )


def load_model(version: int):
    no_file_found = Exception("There is no checkpoint file.")

    # get latest version
    if version == -1:
        files = os.listdir("data/lightning_logs/")
        only_versions = filter(lambda s: s[:8] == "version_", files)
        numbers = [int(s[8:]) for s in only_versions]
        version = max(numbers)

    path = "data/lightning_logs/version_" + str(version) + "/checkpoints/"  # 25/checkpoints/epoch=34-step=12599.ckpt"

    if not os.path.exists(path):
        raise no_file_found

    files = os.listdir(path)

    if len(files) == 0:
        raise no_file_found

    last_checkpoint_path = path + files[-1]
    return LitBase.load_from_checkpoint(last_checkpoint_path)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-train', type=str)
    parser.add_argument('-load', dest="load", type=int, nargs='?', const=-1)

    args = parser.parse_args()

    if args.train is not None and args.load is not None:
        raise Exception("Cannot fit a new model and load a model at the same time. Drop one of the arguments.")

    return args


def handle_load(config, version: int):
    lit_model = load_model(version)
    data = RoadDataModule(batch_size=lit_model.batch_size, resize_to=config["resize_to"],
                          divide_into_four=lit_model.divide_into_four)

    if torch.cuda.is_available():
        lit_model.to(torch.device('cuda'))

    return lit_model, data


def handle_train(trainer, config, model_name):
    if model_name is None:
        model_name = "unet_transformer"

    if model_name == "unet":
        config["resize_to"] = 384
        config["divide_into_four"] = False
        model = UNet()
    elif model_name == "unet_transformer":
        # config['loss_fn'] = "noise_robust_dice"
        model = U_Transformer(3, 1)
    else:
        raise Exception("unknown model")

    data = RoadDataModule(batch_size=config["batch_size"], resize_to=config["resize_to"],
                          divide_into_four=config["divide_into_four"])
    lit_model = LitBase(config, model)

    trainer.fit(lit_model, datamodule=data)

    return lit_model, data


if __name__ == "__main__":
    """The specific model can be given as argument to the program."""

    args = get_args()

    pl.utilities.seed.seed_everything(seed=1337)

    # default
    config = {"lr": 0.0001, "loss_fn": "bce", "divide_into_four": False, "batch_size": 1, "resize_to": 192}
    num_epochs = 20

    if args.load is not None:
        trainer = pl.Trainer(gpus=gpu(), default_root_dir="data", logger=False)
        model, data = handle_load(config, args.load)
    else:
        trainer = pl.Trainer(gpus=gpu(), max_epochs=num_epochs, default_root_dir="data")
        model, data = handle_train(trainer, config, args.train)

    predict(trainer, model, data)
