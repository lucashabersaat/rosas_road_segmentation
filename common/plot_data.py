import numpy as np
import torch
from matplotlib import pyplot as plt


def show_first_n(imgs1, imgs2, n=5, title1="Image", title2="Mask"):
    # visualizes the first n elements of a series of images and segmentation masks
    imgs_to_draw = min(min(n, len(imgs1)), len(imgs2))

    fig, axs = plt.subplots(2, imgs_to_draw, figsize=(18.5, 6))

    if imgs_to_draw == 1:
        axs[0].imshow(imgs1[0])
        axs[1].imshow(imgs2[0])
        axs[0].set_title(f"{title1}")
        axs[1].set_title(f"{title2}")
        axs[0].set_axis_off()
        axs[1].set_axis_off()
    else:
        for i in range(imgs_to_draw):
            axs[0, i].imshow(imgs1[i])
            axs[1, i].imshow(imgs2[i])
            axs[0, i].set_title(f"{title1} {i}")
            axs[1, i].set_title(f"{title2} {i}")
            axs[0, i].set_axis_off()
            axs[1, i].set_axis_off()

    plt.show()


def show_img(img):
    img = prepare(img)

    plt.imshow(img)
    plt.show()


def show_two_imgs(img1, img2):

    img1 = prepare(img1)
    img2 = prepare(img2)

    fig = plt.figure(figsize=(8, 8))

    fig.add_subplot(1, 2, 1)
    plt.imshow(img1)

    fig.add_subplot(1, 2, 2)
    plt.imshow(img2)

    plt.show()


def show_two_imgs_overlay(img1, overlayed_img):
    img1 = prepare(img1)
    overlayed_img = prepare(overlayed_img)

    plt.imshow(img1)  # I would add interpolation='none'
    plt.imshow(overlayed_img, alpha=0.5)  # interpolation='none'
    plt.show()


def prepare(image):
    """Prepare the image for the plotting. Set type to compatible one and change to HWC from CHW if necessary."""

    if torch.is_tensor(image):
        image = image.cpu().detach().numpy()

    s = image.shape

    if image.dtype == np.float16:
        image = image.astype(float)

    if len(s) == 2:
        return image

    c = 0
    if len(image.shape) == 4:
        c = 1

    if s[c] == s[c + 1] == s[c + 2]:
        print(
            "Same width and height and channel_number are the same. Can't decide how to rearange. Letting it like that.")
        return image

    if s[c + 1] == s[c + 2]:
        return np.moveaxis(image, c, -1)
    else:
        return image
