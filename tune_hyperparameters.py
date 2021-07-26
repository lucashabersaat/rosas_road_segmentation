from common.lightning.base import LitBase
from common.lightning.road_data_module import RoadDataModule
from ray import tune
import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from common.get_model import get_model
import torch
from argparse import ArgumentParser
import tempfile


def train_segmentation(config, model_name=None, num_epochs=1, num_gpus=0):
    print(config["model_name"])
    model_name = (str)(config["model_name"])
    model = get_model(model_name, config)
    lit_model = LitBase(config, model)
    # print(config["model_name"])
    data = RoadDataModule(batch_size=config["batch_size"], resize_to=config["resize_to"])
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}

    logger = True
    trainer = pl.Trainer(
        gpus=num_gpus,
        max_epochs=num_epochs,
        default_root_dir="data",
        logger=logger,
        callbacks=[TuneReportCallback(metrics, on="validation_end")],

    )
    trainer.fit(lit_model, data)


def get_args():
    """Initialize and return program arguments"""
    parser = ArgumentParser()
    #parser.add_argument("-train", type=str)
    parser.add_argument("-cpu", dest="cpu", type=int, nargs="?", const=-1)

    args = parser.parse_args()

    # if args.train is None:
    #    args.train = "unet"

    if args.cpu is None:
        args.cpu = 16

    return args


if __name__ == "__main__":
    args = get_args()

    pl.utilities.seed.seed_everything(seed=1337)
    # , "acc": "ptl/val_accuracy" should also be logged, for now just loss
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    trainer = pl.Trainer(callbacks=callbacks)

    num_samples = 2
    num_epochs = 1
    gpus_per_trial = int(torch.cuda.is_available())  # set this to higher if using GPU


    #be carefull when changing the config, everything breaksdown, better fix to one value than removing params
    #keeping config across models and other files is the new challenge
    config = {
        # "r2Uet", "attUnet", "r2attUnet", "nestedUnet"
        "model_name": tune.choice(["unet"]), # unet works well now , got some errors about patches with transunet
        "lr": tune.choice([1e-4, 1e-3]),
        "loss_fn": tune.choice(['noise_robust_dice']),
        "batch_size": tune.choice([4]),
        "resize_to": tune.choice([400]),
        "num_epochs": tune.choice([1]),
        "patch_size": tune.choice([256]),
        "divide_into_four": tune.choice([False])
    }

    trainable = tune.with_parameters(
        train_segmentation,
      	#model_name=args.train,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial)

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": args.cpu,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        name="tune_segmentation")

    print("stayin alive, aha aha aha")

    print(analysis.best_config)
