import torch
from common.plot_data import *

import torchvision.transforms as T


def blur(images):
    images = torch.from_numpy(images)

    images_color = torch.stack([images, images, images], 1)

    blurrer = T.GaussianBlur(kernel_size=(21, 21), sigma=5)
    blurred_imgs = blurrer(images_color)

    blurred_imgs = blurred_imgs[:, 0, :, :]
    show_first_n(blurred_imgs.numpy(), images, title1="PostProcessed", title2="Original")

    return blurred_imgs.numpy()
