import math

import torch
import cv2
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np

from common.util import np_to_tensor
from common.read_data import *
from common.plot_data import *


class ImageDataSet(torch.utils.data.Dataset):
    """
    Dataset class that deals with loading the data and making it available by
    index.
    """

    def __init__(self, path, device, resize_to=(400, 400), patch_size=256):
        self.path = path

        self.x, self.y, self.n_samples = None, None, None
        self.n_variants = 8

        self.device = device
        self.resize_to = resize_to
        self.patch_size = patch_size

        self._load()
        self._preprocess()

    def __getitem__(self, item):
        x = np_to_tensor(self.x[item], self.device)
        y = np_to_tensor(self.y[[item]], self.device)

        return x, y

    def __len__(self):
        return self.n_samples

    def _load(self):
        """
        Load loads the images from the data directory, resizes them if
        necessary and stores them in the object.
        """
        self.x = load_all_from_path(os.path.join(self.path, "images"))
        self.y = load_all_from_path(os.path.join(self.path, "groundtruth"))

        self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x])
        self.y = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.y])

        self.x = np.moveaxis(self.x, -1, 1)

        self.n_samples = len(self.x)

    def _preprocess(self):
        """
        Preprocess preprocesses an image by randomly creating multiple variants
        of the same image. The original image gets randomly rotated, cropped and
        its colour modified. The modified image is then inserted into the data
        set.
        """
        x_preprocessed = np.empty(
            (self.n_samples * self.n_variants, 3, self.patch_size, self.patch_size),
            dtype=np.float32,
        )
        y_preprocessed = np.empty(
            (self.n_samples * self.n_variants, self.patch_size, self.patch_size),
            dtype=np.float32,
        )

        for img_index in range(self.n_samples):
            x, y = self.__getitem__(img_index)

            x_padded, y_padded = ImageDataSet._pad(x, y)

            for variant_index in range(self.n_variants):
                x_rotated, y_rotated = ImageDataSet._rotate(x_padded, y_padded)
                x_cropped, y_cropped = ImageDataSet._crop(
                    x_rotated, y_rotated, self.patch_size
                )
                x_flipped, y_flipped = ImageDataSet._flip(x_cropped, y_cropped)

                x_colored = ImageDataSet._color(x_flipped)

                index = (img_index * self.n_variants) + variant_index

                x_preprocessed[index] = x_colored.cpu().numpy()
                y_preprocessed[index] = y_flipped.cpu().numpy()[0]

        self.x = x_preprocessed
        self.y = y_preprocessed

        self.n_samples = len(self.x)

    @staticmethod
    def _pad(x, y):
        """
        Pad mirrors the image and the corresponding groundtruth mask on each
        side. The resulting image is of size (3 * height - 2, 3 * width - 2).
        """
        size = len(x[0]) - 1

        pad = T.Pad(padding=size, padding_mode="reflect")

        return pad(x), pad(y)

    @staticmethod
    def _rotate(x, y):
        """
        Rotate rotates the image randomly with a value picked randomly from the
        range of degrees (-180, +180).
        """
        rotate = T.RandomRotation(degrees=180)

        x_rotated, y_rotated = rotate(torch.stack([x, torch.cat([y, y, y])]))
        y_rotated = y_rotated[0].unsqueeze(0)

        return x_rotated, y_rotated

    @staticmethod
    def _crop(x, y, size):
        """
        Crop expects a padded (and optionally rotated) image. It randomly
        selects a section from the center of the image and crops it. The
        resulting image is of size (size, size).
        """
        orig_size = math.ceil(len(x[0]) / 3)
        pad_size = math.floor(len(x[0]) / 3)
        rand_pos_x, rand_pos_y = torch.randint(
            low=pad_size, high=pad_size + orig_size, size=(2,)
        )

        x_cropped = TF.crop(x, rand_pos_x, rand_pos_y, size, size)
        y_cropped = TF.crop(y, rand_pos_x, rand_pos_y, size, size)

        return x_cropped, y_cropped

    @staticmethod
    def _flip(x, y):
        """
        Flip randomly flips the image.
        """
        flip_horizontal_1 = T.RandomHorizontalFlip(0.5)
        flip_horizontal_2 = T.RandomHorizontalFlip(0.5)
        flip_vertical_1 = T.RandomVerticalFlip(0.5)
        flip_vertical_2 = T.RandomVerticalFlip(0.5)

        flip = T.Compose(
            [flip_horizontal_1, flip_vertical_1, flip_horizontal_2, flip_vertical_2]
        )

        x_flipped, y_flipped = flip(torch.stack([x, torch.cat([y, y, y])]))
        y_flipped = y_flipped[0].unsqueeze(0)

        return x_flipped, y_flipped

    @staticmethod
    def _color(x):
        """
        Colour randomly changes the contrast, brightness and hue of the image.
        """
        color = T.ColorJitter(brightness=0.1, contrast=0.3, hue=0.1)

        return color(x)

    @staticmethod
    def _normalize(x):
        """
        Normalize normalizes the image.
        """
        s = torch.std(x, [1, 2])
        m = torch.mean(x, [1, 2])

        normalize = T.Normalize(m, s)

        return normalize(x)


class TestImageDataSet(torch.utils.data.Dataset):
    """
    Dataset class that deals with loading the data and making it available by
    index.
    """

    def __init__(self, path, device, resize_to=(608, 608), patch_size=256):
        self.path = path

        self.x, self.n_samples = None, None

        self.device = device
        self.resize_to = resize_to
        self.patch_size = patch_size

        self.x_variants = (
            math.ceil((self.resize_to[0] - self.patch_size) / self.patch_size) + 1
        )
        self.y_variants = (
            math.ceil((self.resize_to[1] - self.patch_size) / self.patch_size) + 1
        )
        self.n_variants = self.x_variants * self.y_variants

        self._load()
        self._preprocess()

    def __getitem__(self, item):
        return np_to_tensor(self.x[item], self.device)

    def __len__(self):
        return self.n_samples

    def _load(self):
        """
        Load loads the images from the data directory, resizes them if
        necessary and stores them in the object.
        """
        self.x = load_all_from_path(self.path)
        self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x])
        self.x = np.moveaxis(self.x, -1, 1)

        self.n_samples = len(self.x)

    def _preprocess(self):
        """
        Preprocess preprocesses an image by creating multiple variants of the
        same image. The original image is cropped multiple times.
        """
        x_preprocessed = np.empty(
            (self.n_samples * self.n_variants, 3, self.patch_size, self.patch_size),
            dtype=np.float32,
        )

        index = 0
        for img_index in range(self.n_samples):
            x = self.__getitem__(img_index)

            for pos_x in range(0, self.resize_to[0] - self.patch_size + 1):
                if (
                    pos_x % self.patch_size != 0
                    and pos_x + self.patch_size != self.resize_to[0]
                ):
                    continue

                for pos_y in range(0, self.resize_to[1] - self.patch_size + 1):
                    if (
                        pos_y % self.patch_size != 0
                        and pos_y + self.patch_size != self.resize_to[1]
                    ):
                        continue

                    x_cropped = TF.crop(
                        x, pos_x, pos_y, self.patch_size, self.patch_size
                    )

                    x_preprocessed[index] = x_cropped.cpu().numpy()
                    index += 1

        self.x = x_preprocessed
        self.n_samples = len(self.x)

    def reassemble(self, y):
        """
        Reassemble reassembles patches to a full image.
        """
        n_images = self.n_samples // self.n_variants

        reassembled_images = np.empty(
            [n_images, 1, self.resize_to[0], self.resize_to[1]]
        )

        for img_index in range(n_images):
            for patch_x_index in range(self.x_variants):
                for patch_y_index in range(self.y_variants):
                    x_pos = patch_x_index * self.patch_size
                    if patch_x_index == self.x_variants - 1:
                        x_pos = self.resize_to[0] - self.patch_size

                    y_pos = patch_y_index * self.patch_size
                    if patch_y_index == self.x_variants - 1:
                        y_pos = self.resize_to[1] - self.patch_size

                    patch_index = (
                        img_index * self.n_variants
                        + (patch_x_index * self.x_variants)
                        + patch_y_index
                    )

                    reassembled_images[
                        img_index,
                        :,
                        x_pos : x_pos + self.patch_size,
                        y_pos : y_pos + self.patch_size,
                    ] = y[patch_index, :, :, :]

        return reassembled_images


if __name__ == "__main__":
    dataset = ImageDataSet("data/training", "cpu")
