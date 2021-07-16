import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import time

from common.image_data_set import ImageDataSet, TestImageDataSet


class RoadDataModule(pl.LightningDataModule):

    def __init__(
            self,
            batch_size: int = 4,
            resize_to=384,
            divide_into_four=True,
            enable_preprocessing=True
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_worker = 0
        self.resize_to = resize_to

        self.divide_into_four = divide_into_four

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_dataset = ImageDataSet(
            "data/training", device, use_patches=False, resize_to=(self.resize_to, self.resize_to),
            divide_into_four=divide_into_four, enable_preprocessing=enable_preprocessing
        )
        self.val_dataset = ImageDataSet(
            "data/validation", device, use_patches=False, resize_to=(self.resize_to, self.resize_to),
            divide_into_four=divide_into_four, enable_preprocessing=enable_preprocessing
        )

        self.test_dataset = TestImageDataSet(
            "data/test_images/test_images", device, use_patches=False, resize_to=(self.resize_to, self.resize_to),
            divide_into_four=divide_into_four, enable_preprocessing=enable_preprocessing
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_worker)
