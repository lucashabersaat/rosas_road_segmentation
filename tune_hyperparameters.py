from common.lightning.base import LitBase
from common.lightning.road_data_module import RoadDataModule
from ray import tune
import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from common.get_model import get_model
import torch


def train_segmentation(config, checkpoint_dir=None, num_epochs=35, num_gpus=1):
    model = get_model("unet", config)
    data = RoadDataModule(
        batch_size = config["batch_size"],
        resize_to = config["resize_to"])

    metrics = {"loss": "ptl/val_loss"}
    lit_model = LitBase(config, model)
    """ 
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        progress_bar_refresh_rate=0,
        callbacks=[TuneReportCallback(metrics, on="validation_end")])
    """
    logger = True
    trainer = pl.Trainer(gpus=int(torch.cuda.is_available()), max_epochs=config["num_epochs"], default_root_dir="data",
                         logger=logger, callbacks=[TuneReportCallback(metrics, on="validation_end")])
    trainer.fit(lit_model, data)


if __name__ == "__main__":

    pl.utilities.seed.seed_everything(seed=1337)
    #, "acc": "ptl/val_accuracy" should also be logged, for now just loss
    metrics = {"loss": "ptl/val_loss"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    trainer = pl.Trainer(callbacks=callbacks)

    num_samples = 4
    num_epochs = 4
    gpus_per_trial = int(torch.cuda.is_available()) # set this to higher if using GPU



    config = {
     #"model_name": tune.choice(["unet", "unet2", "transunet", "r2Uet", "attUnet", "r2attUnet", "nestedUnet"]),
     "lr": tune.loguniform(1e-4, 1e-1),
     "loss_fn": tune.choice(["dice_loss", 'noise_robust_dice']),
     "batch_size": tune.choice([1, 2, 3, 4]),
     "resize_to": tune.choice([384, 192]),
     "num_epochs": tune.choice([35])
    }

    trainable = tune.with_parameters(
        train_segmentation,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial)

    analysis = tune.run(
     trainable,
     config=config,
     num_samples=num_samples)

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        name="tune_segmentation")

    print(analysis.best_config)
