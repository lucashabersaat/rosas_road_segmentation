import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from common.image_data_set import ImageDataSet, TestImageDataSet


class RoadDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        resize_to: int = None,
        patch_size: int = 256,
        mode: str = "none",
        variants: int = 5,
        enhance: bool = True,
        offset: int = 100,
        blend_mode: str = "cover",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_worker = 0

        device = "cuda" if torch.cuda.is_available() else "cpu"

        size_train = 400
        size_test = 608
        if resize_to is not None:
            size_train = resize_to
            size_test = resize_to

        if patch_size is None:
            patch_size = 256

        assert mode in ["none", "breed", "patch", "patch_random"]
        assert blend_mode in ["cover", "average", "weighted_average"]

        self.train_dataset = ImageDataSet(
            path="data/training",
            device=device,
            size=size_train,
            mode=mode,
            variants=variants,
            patch_size=patch_size,
            enhance=enhance,
        )
        self.val_dataset = ImageDataSet(
            path="data/validation",
            device=device,
            size=size_train,
            mode=mode,
            variants=variants,
            patch_size=patch_size,
            enhance=enhance,
        )

        self.test_dataset = TestImageDataSet(
            path="data/test_images",
            device=device,
            size=size_test,
            enhance=enhance,
            patch_size=patch_size,
            offset=offset,
            blend_mode=blend_mode,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_worker,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_worker,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_worker
        )
