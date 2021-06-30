from base import LitBase
from road_data_module import RoadDataModule
from ray import tune
import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback


def train_segmentation(config, num_epochs=35, num_gpus=0):
    model = LitBase(config)
    dm = RoadDataModule(
        batch_size = config["batch_size"],
        resize_to = config["resize_to"],
        divide_into_four = config["divide_into_four"])
    metrics = {"loss": "ptl/val_loss"}
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        progress_bar_refresh_rate=0,
        callbacks=[TuneReportCallback(metrics, on="validation_end")])
    trainer.fit(model, dm)


if __name__ == "__main__":

    #, "acc": "ptl/val_accuracy" should also be logged, for now just loss
    metrics = {"loss": "ptl/val_loss"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    trainer = pl.Trainer(callbacks=callbacks)

    num_samples = 10
    num_epochs = 10
    gpus_per_trial = 0 # set this to higher if using GPU



    config = {
     "lr": tune.loguniform(1e-4, 1e-1),
     "loss_fn": tune.choice(['bce','noise_robust_dice']),
     "batch_size": tune.choice([1,2,3,4]),
     "resize_to": tune.choice([192, 384]),
     "divide_into_four": tune.choice([True])
    }

    trainable = tune.with_parameters(
        train_segmentation,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial)

    analysis = tune.run(
     trainable,
     config=config,
     num_samples=10)

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
