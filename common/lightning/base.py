from typing import Optional
import torch
from torch.nn import Module, BCELoss
import pytorch_lightning as pl
from common.losses import NoiseRobustDiceLoss, DiceLoss
from common.get_model import get_model



class LitBase(pl.LightningModule):
    def __init__(
        self,
        config,
        model: Optional[Module] = None,
    ):
        super().__init__()

        if isinstance(model, str):
            model_name = model
            self.model = get_model(model_name, config)
        else:
            model_name = str.lower(model.__class__.__name__)
            self.model = model

        self.lr = config["lr"]
        loss_fn = config["loss_fn"]

        self.batch_size = config.get("batch_size")
        self.resize_to = config.get("resize_to")
        self.num_epochs = config.get("num_epochs")

        hyper_parameters = {
            "model": model_name,
            "loss_fn": loss_fn,
            "learning_rate": self.lr,
            "config": config,
        }

        self.save_hyperparameters(hyper_parameters)

        if loss_fn == "bce":
            self.loss_fn = BCELoss()
        elif loss_fn == "noise_robust_dice":
            self.loss_fn = NoiseRobustDiceLoss()
        elif loss_fn == "dice_loss":
            self.loss_fn = DiceLoss()

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.model(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss_fn(y_hat, y)
        acc = LitBase.accuracy(y_hat, y)
        iou = LitBase.IoU(y_hat, y)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", acc)
        self.log("ptl/train_iou", iou)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss_fn(y_hat, y)
        acc = LitBase.accuracy(y_hat, y)
        iou = LitBase.IoU(y_hat, y)

        return {"val_loss": loss, "val_accuracy": acc, "val_iou": iou}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        avg_iou = torch.stack([x["val_iou"] for x in outputs]).mean()

        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc, prog_bar=True)
        self.log("ptl/val_iou", avg_iou, prog_bar=True)

    @staticmethod
    def accuracy(y_hat, y):
        y = torch.flatten(y).type(torch.IntTensor)
        y_hat = torch.flatten(y_hat)

        y_hat = torch.where(y_hat >= 0.5, 1, 0).type(torch.IntTensor)

        fp_fn = torch.count_nonzero(torch.logical_or(y, y_hat))
        tp_tn_fp_fn = len(y_hat)

        return (tp_tn_fp_fn -fp_fn)/ tp_tn_fp_fn

    @staticmethod
    def IoU(y_hat, y):
        """
        Compute the Intersection Over Union, which is TP / (TP + FP + FN).
        meanIoU is the average IoU over all output classes, as we only have one, this is the same.
        """

        y = torch.flatten(y).type(torch.IntTensor)
        y_hat = torch.flatten(y_hat)

        if torch.min(y_hat) < 0 or torch.max(y_hat) > 1:
            print("Not in 0..1 range: ", torch.min(y_hat), torch.max(y_hat))

        y_hat = torch.where(y_hat >= 0.5, 1, 0).type(torch.IntTensor)

        tp = torch.count_nonzero(torch.logical_and(y, y_hat))
        tp_fp_fn = tp + torch.count_nonzero(torch.logical_xor(y, y_hat))

        return tp / tp_fp_fn
