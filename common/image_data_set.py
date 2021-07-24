import math
import os

import cv2
import numpy as np
from scipy.signal import convolve2d
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from common.util import np_to_tensor
from common.read_data import load_all_from_path
from common.plot_data import (
    show_img,
    show_two_imgs,
    show_two_imgs_overlay,
    show_first_n,
)


class ImageDataSet(torch.utils.data.Dataset):
    """
    Dataset class that deals with loading the training data and making it
    available by index.
    """

    def __init__(
        self,
        path: str,
        device: str,
        size: int = 400,
        mode: str = "none",
        variants: int = 5,
        patch_size: int = 256,
        enhance: bool = True,
        noise: bool = True
    ):
        """
        Init initializes the dataset by loading the images from the file system
        and preprocessing them.
        """
        self.path = path
        self.device = device

        self.x = None
        self.y = None
        self.n = 0

        self.x_preprocessed = None
        self.y_preprocessed = None
        self.n_preprocessed = 0

        self.size = size

        assert mode in ["none", "breed", "patch", "patch_random"]
        self.mode = mode
        self.variants = variants
        self.patch_size = patch_size

        if self.mode == "none":
            self.variants = 1
            self.patch_size = self.size
        if self.mode == "breed":
            self.patch_size = self.size
        elif self.mode == "patch":
            self.variants = 5
        elif self.mode == "patch_random":
            pass

        self.enhance = enhance
        self.noise = noise

        self._load()
        self._preprocess()

    def __getitem__(self, item):
        """
        Getitem returns both the image and the groundtruth of the preprocessed
        image as tensors.
        """
        x = np_to_tensor(self.x_preprocessed[item], self.device)
        y = torch.round(np_to_tensor(self.y_preprocessed[[item]], self.device))

        if self.noise:
            gaussian_noise = 0.02 * np.random.normal(size=x.shape)
            gaussian_noise = np_to_tensor(gaussian_noise, self.device)

            x = torch.add(x, gaussian_noise)

        x = ImageDataSet._normalize(x)

        return x, y

    def __len__(self):
        """
        Len returns the number of preprocessed images.
        """
        return self.n_preprocessed

    def get_original(self, item):
        """
        Get_original returns both the image and the groundtruth of the original
        image as tensors.
        """
        x = np_to_tensor(self.x[item], self.device)
        y = np_to_tensor(self.y[[item]], self.device)

        return x, y

    def _load(self):
        """
        Load loads the images from the data directory, resizes them if
        necessary and stores them in the object.
        """
        self.x = load_all_from_path(os.path.join(self.path, "images"))
        self.y = load_all_from_path(os.path.join(self.path, "groundtruth"))

        self.x = np.stack(
            [cv2.resize(img, dsize=(self.size, self.size)) for img in self.x]
        )
        self.y = np.stack(
            [cv2.resize(img, dsize=(self.size, self.size)) for img in self.y]
        )

        self.x = np.moveaxis(self.x, -1, 1)

        self.n = len(self.x)

    def _preprocess(self):
        """
        Preprocess chooses the preprocessing method and applies the enhancement
        methods.
        """
        if self.mode == "none":
            self._preprocess_none()
        if self.mode == "breed":
            self._preprocess_breed()
        elif self.mode == "patch":
            self._preprocess_patch()
        elif self.mode == "patch_random":
            self._preprocess_patch_random()

        if self.enhance:
            for img_index in range(self.n_preprocessed):
                x, _ = self.__getitem__(img_index)
                self.x_preprocessed[img_index] = ImageDataSet.enhance(x).cpu().numpy()

    def _preprocess_none(self):
        """
        Preprocess_none copies the original data to the preprocessed data
        location without modifying it.
        """
        self.x_preprocessed = self.x.copy()
        self.y_preprocessed = self.y.copy()
        self.n_preprocessed = self.n

    def _preprocess_breed(self):
        """
        Preprocess_breed creates multiple variants of the original images. It
        randomly rotates and flips the image variants and copies them to the
        preprocessed data location.
        """
        x_preprocessed = np.zeros(
            (self.variants * self.n, 3, self.size, self.size),
            dtype=np.float32,
        )
        y_preprocessed = np.zeros(
            (self.variants * self.n, self.size, self.size),
            dtype=np.float32,
        )

        for img_index in range(self.n):
            x, y = self.get_original(img_index)

            for variant_index in range(self.variants):
                x_rotated, y_rotated = ImageDataSet._rotate90(x, y)

                x_flipped, y_flipped = ImageDataSet._flip(x_rotated, y_rotated)

                preprocessed_index = img_index * self.variants + variant_index
                x_preprocessed[preprocessed_index] = x_flipped.cpu().numpy()
                y_preprocessed[preprocessed_index] = y_flipped.cpu().numpy()

        self.x_preprocessed = x_preprocessed
        self.y_preprocessed = y_preprocessed
        self.n_preprocessed = self.n * self.variants

    def _preprocess_patch(self):
        """
        Preprocess_patch preprocesses the original images by creating multiple
        variants of the same image: The original image is cropped into five
        parts, one from each corner and one from the center. Each of these five
        patches gets randomly rotated, flipped, and stored in the preprocessed
        data location.
        """
        x_preprocessed = np.zeros(
            (5 * self.n, 3, self.patch_size, self.patch_size),
            dtype=np.float32,
        )
        y_preprocessed = np.zeros(
            (5 * self.n, self.patch_size, self.patch_size),
            dtype=np.float32,
        )

        for img_index in range(self.n):
            x, y = self.get_original(img_index)

            crop = T.FiveCrop((self.patch_size, self.patch_size))
            x_cropped = crop(x)
            y_cropped = crop(y)

            for i in range(5):
                x_rotated, y_rotated = ImageDataSet._rotate90(
                    x_cropped[i], y_cropped[i]
                )

                x_flipped, y_flipped = ImageDataSet._flip(x_rotated, y_rotated)

                preprocessed_index = 5 * img_index + i
                x_preprocessed[preprocessed_index] = x_flipped.cpu().numpy()
                y_preprocessed[preprocessed_index] = y_flipped.cpu().numpy()

        self.x_preprocessed = x_preprocessed
        self.y_preprocessed = y_preprocessed
        self.n_preprocessed = 5 * self.n

    def _preprocess_patch_random(self):
        """
        Preprocess_patch_random preprocesses the original images by randomly
        creating multiple variants of the same image: The original image gets
        randomly rotated, cropped, and flipped multiple times. The resulting
        images are stored in the preprocessed data location.
        """
        x_preprocessed = np.zeros(
            (self.n * self.variants, 3, self.patch_size, self.patch_size),
            dtype=np.float32,
        )
        y_preprocessed = np.zeros(
            (self.n * self.variants, self.patch_size, self.patch_size),
            dtype=np.float32,
        )

        for img_index in range(self.n):
            x, y = self.get_original(img_index)

            x_padded, y_padded = ImageDataSet._pad(x, y)

            for variant_index in range(self.variants):
                x_rotated, y_rotated = ImageDataSet._rotate(x_padded, y_padded)
                x_cropped, y_cropped = ImageDataSet._crop(
                    x_rotated, y_rotated, self.patch_size
                )
                x_flipped, y_flipped = ImageDataSet._flip(x_cropped, y_cropped)

                preprocessed_index = img_index * self.variants + variant_index
                x_preprocessed[preprocessed_index] = x_flipped.cpu().numpy()
                y_preprocessed[preprocessed_index] = y_flipped.cpu().numpy()

        self.x_preprocessed = x_preprocessed
        self.y_preprocessed = y_preprocessed
        self.n_preprocessed = self.n * self.variants

    @staticmethod
    def enhance(x):
        """
        Enhance applies several filters to an image. The image is blurred and
        sharpened.
        """
        x_blurred = ImageDataSet._blur(x)
        x_sharpened = ImageDataSet._sharpen(x_blurred)

        return x_sharpened

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
    def _rotate90(x, y):
        """
        Rotate rotates the image randomly with a value picked randomly from the
        range of degrees (-180, +180).
        """
        rand_int = torch.randint(low=0, high=4, size=(1,)).item()
        angle = 90 * rand_int

        x_rotated, y_rotated = TF.rotate(torch.stack([x, torch.cat([y, y, y])]), angle)
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
    def _blur(x):
        """
        Colour randomly changes the contrast, brightness and hue of the image.
        """
        blur = T.GaussianBlur(kernel_size=9, sigma=1)

        return blur(x)

    @staticmethod
    def _sharpen(x):
        """
        Colour randomly changes the contrast, brightness and hue of the image.
        """
        filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

        x = np.float32(x.cpu())

        for c in range(len(x)):
            x[c] = convolve2d(x[c], filter, mode='same', boundary='symm')

        return torch.from_numpy(x)

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
    Dataset class that deals with loading the test data and making it available
    by index.
    """

    def __init__(
        self,
        path: str,
        device: str,
        size: int = 400,
        enhance: bool = True,
        patch_size: int = 256,
        offset: int = 100,
        blend_mode: str = "cover",
    ):
        """
        Init initializes the dataset by loading the images from the file system
        and preprocessing them. It splits the images into patches if necessary.
        """
        self.path = path
        self.device = device

        self.x = None
        self.n = 0

        self.x_preprocessed = None
        self.n_preprocessed = 0

        self.size = size

        self.enhance = enhance

        self.patch_size = patch_size
        self.use_patches = self.size != self.patch_size

        self.offset = offset
        assert blend_mode in ["cover", "average", "weighted_average"]
        self.blend_mode = blend_mode

        self.variants = 1
        if self.use_patches:
            self.variants_x = math.ceil((self.size - self.patch_size) / self.offset) + 1
            self.variants_y = self.variants_x
            self.variants = self.variants_x * self.variants_y

        self._load()
        self._preprocess()

    def __getitem__(self, item):
        """
        Getitem returns the preprocessed image as a tensor.
        """
        return np_to_tensor(self.x_preprocessed[item], self.device)

    def __len__(self):
        """
        Len returns the number of preprocessed images.
        """
        return self.n_preprocessed

    def get_original(self, item):
        """
        Get_original returns the original image as a tensor.
        """
        return np_to_tensor(self.x[item], self.device)

    def _load(self):
        """
        Load loads the images from the data directory, resizes them if
        necessary and stores them in the object.
        """
        self.x = load_all_from_path(self.path)

        self.x = np.stack(
            [cv2.resize(img, dsize=(self.size, self.size)) for img in self.x]
        )

        self.x = np.moveaxis(self.x, -1, 1)

        self.n = len(self.x)

    def _preprocess(self):
        """
        Preprocess preprocesses an image by creating multiple variants of the
        same image. The original image is cropped multiple times.
        """
        x_preprocessed = np.zeros(
            (self.n * self.variants, 3, self.patch_size, self.patch_size),
            dtype=np.float32,
        )

        preprocessed_index = 0
        for img_index in range(self.n):
            x = self.get_original(img_index)

            if self.enhance:
                x = ImageDataSet.enhance(x)

            # Do not split image
            if not self.use_patches:
                x_preprocessed[img_index] = x.cpu().numpy()
                continue

            # Split image
            for pos_x in range(0, self.size - self.patch_size + 1):
                if pos_x % self.offset != 0 and pos_x + self.patch_size != self.size:
                    continue

                for pos_y in range(0, self.size - self.patch_size + 1):
                    if (
                        pos_y % self.offset != 0
                        and pos_y + self.patch_size != self.size
                    ):
                        continue

                    x_cropped = TF.crop(
                        x, pos_x, pos_y, self.patch_size, self.patch_size
                    )

                    x_preprocessed[preprocessed_index] = x_cropped.cpu().numpy()
                    preprocessed_index += 1

        self.x_preprocessed = x_preprocessed
        self.n_preprocessed = len(self.x_preprocessed)

    def reassemble(self, y):
        """
        Reassemble reassembles patches to a full image.
        """
        if not self.use_patches:
            return y

        reassembled_images = np.zeros([self.n, 1, self.size, self.size])

        mask = self._get_mask()

        for img_index in range(self.n):
            print(f'Reassembling image {img_index}')

            img_combined = np.zeros([self.size, self.size], dtype=np.ndarray)

            index = img_index * self.variants
            img_patches = y[index : index + self.variants, :, :, :]

            for patch_x_index in range(self.variants_x):
                for patch_y_index in range(self.variants_y):
                    patch_index = patch_x_index * self.variants_y + patch_y_index

                    for patch_x_pos in range(self.patch_size):
                        for patch_y_pos in range(self.patch_size):
                            img_x_pos = patch_x_index * self.offset + patch_x_pos
                            if patch_x_index == self.variants_x - 1:
                                img_x_pos = self.size - self.patch_size + patch_x_pos

                            img_y_pos = patch_y_index * self.offset + patch_y_pos
                            if patch_y_index == self.variants_y - 1:
                                img_y_pos = self.size - self.patch_size + patch_y_pos

                            weight = 1.0
                            if self.blend_mode == "weighted_average":
                                weight = mask[patch_x_pos, patch_y_pos]

                            values = img_combined[img_x_pos, img_y_pos]
                            if isinstance(values, int):
                                values = np.array(
                                    [
                                        [
                                            img_patches[
                                                patch_index, 0, patch_x_pos, patch_y_pos
                                            ],
                                            weight,
                                        ]
                                    ]
                                )
                            else:
                                values = np.concatenate(
                                    (
                                        values,
                                        np.array(
                                            [
                                                [
                                                    img_patches[
                                                        patch_index,
                                                        0,
                                                        patch_x_pos,
                                                        patch_y_pos,
                                                    ],
                                                    weight,
                                                ]
                                            ]
                                        ),
                                    )
                                )
                            img_combined[img_x_pos, img_y_pos] = values

            img_reassembled = np.zeros([1, self.size, self.size], dtype=np.float32)

            if self.blend_mode == "cover":
                for x_pos in range(self.size):
                    for y_pos in range(self.size):
                        img_reassembled[0, x_pos, y_pos] = img_combined[x_pos, y_pos][
                            0
                        ][0]
            elif self.blend_mode in ["average", "weighted_average"]:
                for x_pos in range(self.size):
                    for y_pos in range(self.size):
                        weighted_values = img_combined[x_pos, y_pos]
                        if len(weighted_values) == 1:
                            img_reassembled[0, x_pos, y_pos] = weighted_values[0][0]
                        else:
                            sum_values = 0.0
                            sum_weights = 0.0
                            for value_index in range(len(weighted_values)):
                                sum_values += (
                                    weighted_values[value_index][1]
                                    * weighted_values[value_index][0]
                                )
                                sum_weights += weighted_values[value_index][1]
                            img_reassembled[0, x_pos, y_pos] = sum_values / sum_weights

            reassembled_images[img_index] = img_reassembled

        return reassembled_images

    def _get_mask(self):
        """
        Get_mask returns a weighting mask in the shape of a patch. It weights
        pixels close to the patch border lower than pixels in the center of the
        patch.
        """
        border1 = int(0.1 * self.patch_size)
        border2 = int(0.2 * self.patch_size)

        mask = np.zeros([self.patch_size, self.patch_size])
        mask[:, :] = 0.5
        mask[border1:-border1, border1:-border1] = 0.75
        mask[border2:-border2, border2:-border2] = 1

        return mask


if __name__ == "__main__":
    dataset = ImageDataSet(
        "data/training",
        "cpu",
        size=400,
        mode="none",
        variants=3,
        patch_size=256,
        enhance=True,
        noise=True
    )
    show_two_imgs(dataset[0][0], dataset[0][1])

    # dataset = TestImageDataSet(
    #     "data/test_images",
    #     "cpu",
    #     size=608,
    #     enhance=True,
    #     patch_size=256,
    #     offset=200,
    #     blend_mode="weighted_average",
    # )
    #
    # y = np.zeros([2*dataset.variants, 1, 256, 256])
    # gray = T.Grayscale()
    # for i in range(len(dataset)):
    #     y[i] = gray(dataset[i]).cpu().numpy()
    #
    # reassembled = dataset.reassemble(y)
    # show_img(reassembled[0])
    # show_img(reassembled[1])
