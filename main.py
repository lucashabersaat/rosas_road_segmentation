import os
from argparse import ArgumentParser
import time

import torch
import numpy as np
import pytorch_lightning as pl

from common.lightning.base import LitBase
from common.lightning.road_data_module import RoadDataModule
from common.image_data_set import TestImageDataSet
from common.write_data import write_submission
from common.plot_data import *
from common.postprocess import *
from common.get_model import get_model


def gpu():
    return int(torch.cuda.is_available())


def predict(trainer, model, data):
    """Predict on the test set with given model, apply post processing and write into submission file"""

    predictions = trainer.predict(model, datamodule=data)

    predictions = [p.detach().cpu().numpy() for p in predictions]
    predictions = np.concatenate(predictions)

    predictions = data.test_dataset.reassemble(predictions)
    predictions = np.asarray(predictions).squeeze(1)

    # np.save("predictions.npy", predictions)

    post_process = True
    if post_process:
        predictions = postprocess(predictions)

    name = "lightning_" + str.lower(model.model.__class__.__name__)
    write_submission(
        data.test_dataset.x,
        predictions,
        name,
        "data/test_images",
        (608, 608),
        graph_cut=False,
    )


def load_model(version: int):
    """Load model with given version from 'data/lightning_logs'"""
    no_file_found = Exception("There is no checkpoint file.")

    # get latest version
    if version == -1:
        files = os.listdir("data/lightning_logs/")
        only_versions = filter(lambda s: s[:8] == "version_", files)
        numbers = [int(s[8:]) for s in only_versions]
        version = max(numbers)

    path = "data/lightning_logs/version_" + str(version) + "/checkpoints/"

    if not os.path.exists(path):
        raise no_file_found

    files = os.listdir(path)

    if len(files) == 0:
        raise no_file_found

    last_checkpoint_path = path + files[-1]
    return LitBase.load_from_checkpoint(last_checkpoint_path)


def get_args():
    """Initialize and return program arguments"""
    parser = ArgumentParser()
    parser.add_argument("-train", type=str)
    parser.add_argument("-load", dest="load", type=int, nargs="?", const=-1)

    args = parser.parse_args()

    if args.train is not None and args.load is not None:
        raise Exception(
            "Cannot fit a new model and load a model at the same time. Drop one of the arguments."
        )

    return args


def handle_load(config, version: int):
    """Handle case when a model should be loaded from logs"""
    lit_model = load_model(version)
    data = RoadDataModule(
        batch_size=lit_model.batch_size,
        resize_to=lit_model.resize_to,
        patch_size=config["patch_size"],
        mode=config["mode"],
        variants=config["variants"],
        enhance=config["enhance"],
        offset=config["offset"],
        blend_mode=config["blend_mode"],
        noise=config["noise"],
    )

    if torch.cuda.is_available():
        lit_model.to(torch.device("cuda"))

    return lit_model, data


def handle_train(trainer, config, model_name):
    """Train model with given configs"""
    model = get_model(model_name, config)

    data = RoadDataModule(
        batch_size=config["batch_size"],
        resize_to=config["resize_to"],
        patch_size=config["patch_size"],
        mode=config["mode"],
        variants=config["variants"],
        enhance=config["enhance"],
        offset=config["offset"],
        blend_mode=config["blend_mode"],
        noise=config["noise"],
    )
    lit_model = LitBase(config, model)

    trainer.fit(lit_model, datamodule=data)

    return lit_model, data


if __name__ == "__main__":
    """The specific model can be given as argument to the program."""

    args = get_args()

    pl.utilities.seed.seed_everything(seed=1337)

    # default
    config = {
        "lr": 0.001,
        "loss_fn": "noise_robust_dice",
        "batch_size": 4,
        "num_epochs": 35,
        "resize_to": None,
        "patch_size": 256,
        "mode": "patch",
        "variants": 5,
        "enhance": True,
        "offset": 100,
        "blend_mode": "weighted_average",
        "noise": True,
    }

    if args.load is not None:
        # load
        trainer = pl.Trainer(gpus=gpu(), default_root_dir="data", logger=False)
        model, data = handle_load(config, args.load)
    else:
        # train
        logger = True
        trainer = pl.Trainer(
            gpus=gpu(),
            max_epochs=config["num_epochs"],
            default_root_dir="data",
            logger=logger,
        )
        model, data = handle_train(trainer, config, args.train)

    predict(trainer, model, data)
