import numpy as np
import torch
import cv2

import torchvision.transforms as T
from torchvision.transforms.functional import adjust_sharpness

from common.plot_data import *


def postprocess(images):
    """Run all post-processing steps. This is only done for the predictions. Not while training."""
    # images = blur(images)
    # images = sharpen(images)
    images = morphological_postprocessing(images, 4)
    return images


def blur(images):
    images = torch.from_numpy(images)

    images_color = torch.stack([images, images, images], 1)

    blurrer = T.GaussianBlur(kernel_size=(21, 21), sigma=5)
    blurred_imgs = blurrer(images_color)
    blurred_imgs = blurred_imgs[:, 0, :, :]

    return blurred_imgs.numpy()


def sharpen(images):
    images = torch.from_numpy(images).unsqueeze(1)

    sharpened = adjust_sharpness(images, 10)

    return sharpened.squeeze().numpy()


def morphological_postprocessing(imgs, iterations):
    """
    Morphological transformation of imgs in order to binarize the image and remove noise through erosion and dilation.
    :param imgs: 3D numpy array of images to process
    :return: postprocessed array
    """
    out = []
    for img in imgs:
        img = img.astype(float)
        # show_img(img)

        kernel = np.ones((3, 3), np.uint8)

        img = cv2.dilate(img, kernel, iterations=iterations)
        img = cv2.erode(img, kernel, iterations=iterations)

        img = cv2.erode(img, kernel, iterations=iterations)
        img = cv2.dilate(img, kernel, iterations=iterations)

        out.append(img)

        # show_img(img)

    out = np.expand_dims(np.stack(out), -1)
    return out
