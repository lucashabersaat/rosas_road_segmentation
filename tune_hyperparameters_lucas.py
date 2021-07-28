from common.lightning.base import LitBase
from common.lightning.road_data_module import RoadDataModule
from ray import tune
import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from common.get_model import get_model
import torch
from argparse import ArgumentParser
import tempfile
import os


os.environ["CUDA_LAUNCH_BLOCKING"] = str(1)
def train_segmentation(config, num_epochs=35, num_gpus=1):
    print(config["model_name"])
    model_name = (str)(config["model_name"])
    model = get_model(model_name, config)
    lit_model = LitBase(config, model)
    # print(config["model_name"])
    data = RoadDataModule(batch_size=config["batch_size"],
                     # resize_to=config["resize_to"],
                      patch_size= config["patch_size"],
                      mode=config["mode"],
                      blend_mode=config["blend_mode"],
                      noise=config["noise"])

    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy", "val_iou": "ptl/val_iou"}

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
        args.cpu = 20

    return args


if __name__ == "__main__":
    args = get_args()

    pl.utilities.seed.seed_everything(seed=1337)
    # , "acc": "ptl/val_accuracy" should also be logged, for now just loss
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy", "val_iou": "ptl/val_iou"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    trainer = pl.Trainer(callbacks=callbacks)

    num_samples = 16
    num_epochs = 35
    gpus_per_trial = 1#int(torch.cuda.is_available())  # set this to higher if using GPU


    config_learning_rate = {
        "model_name": tune.choice(["transunet"]),
        "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
        "loss_fn": tune.choice(['noise_robust_dice']),
        "batch_size": tune.choice([4]),
        "num_epochs": tune.choice([num_epochs]),
        "patch_size": tune.choice([256]),
        "mode": tune.choice(["patch"]),
        "blend_mode": tune.choice(["weighted_average"]),
        "noise": tune.choice([True]),
        "variants": tune.choice([5]),
        "enhance": tune.choice([True]),
        "threshold": tune.grid_search([0.2, 0.4, 0.6, 0.8])
    }

    # config = {
    #     "model_name": tune.choice(["transunet"]),
    #     "lr": tune.choice([1e-4]),
    #     "loss_fn": tune.choice(['noise_robust_dice']),
    #     "batch_size": tune.choice([4]),
    #     "num_epochs": tune.choice([num_epochs]),
    #     "patch_size": tune.choice([256]),
    #     "mode": tune.choice(["patch"]),
    #     "variants": tune.choice([5]),
    #     "blend_mode": tune.choice(["weighted_average"]),
    #     "noise": tune.choice([True, False]),
    #     "enhance": tune.choice([True, False]),
    # }

    # set num_samples to 3, if you test this config_patch_size
    config_patch_size = {
        "model_name": tune.choice(["transunet"]),
        "lr": tune.choice([1e-4]),
        "loss_fn": tune.choice(['noise_robust_dice']),
        "batch_size": tune.choice([4]),
        "num_epochs": tune.choice([num_epochs]),
        "patch_size": tune.choice([208, 256, 320]),
        "mode": tune.choice(["patch"]),
        "blend_mode": tune.choice(["weighted_average"]),
        "noise": tune.choice([True]),
        # "threshold": tune.uniform(0.2, 0.8)
    }

    config = config_learning_rate

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
        metric="acc",
        mode="max",
        config=config,
        num_samples=num_samples,
        local_dir="/cluster/scratch/samuelbe",
        name="tune_segmentation_config_learning_rate_lucas")

    print("stayin alive, aha aha aha")

    print(analysis.best_config)
