from torch import nn
from skimage import color
import cv2
import maxflow

import torchvision.transforms as T
from torchvision.transforms.functional import adjust_sharpness

from common.plot_data import *


def postprocess(images):
    """Run all post-processing steps. This is only done for the predictions. Not while training."""
    # images = blur(images)
    # images = sharpen(images)
    images = morphological_postprocessing(images, 10)
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

        out.append(img)

        # show_img(img)

    out = np.expand_dims(np.stack(out), -1)
    return out


def graph_cut(images, originals):

    # sig = nn.Sigmoid()
    # images = sig(images)

    images = 1 / (1 + np.exp(-images))

    binary_masks = np.zeros_like(images)

    for index, image in enumerate(images):

        s = image.size
        l = image.shape[0]
        num_edges = 2 * (l - 1) ** 2 + 2 * (l - 1)

        if image.shape[1] != l:
            raise Exception("Rectangular image? No please not.")

        g = maxflow.Graph[float](s, num_edges)
        nodes = g.add_nodes(s)

        img_flat = image.reshape(-1)
        lab_img = color.rgb2lab(originals[index]).reshape(-1, 3)

        for i in range(img_flat.size):
            p = img_flat[i]

            g.add_tedge(i, 1 - p, p)

            if i % l < l - 1:
                d = color.deltaE_cie76(lab_img[index], lab_img[index + 1])
                g.add_edge(i, i + 1, d, d)
            if i < s - l:
                d = color.deltaE_cie76(lab_img[index], lab_img[index + l])
                g.add_edge(i, i + l, d, d)

        g.maxflow()
        binary_mask = g.get_grid_segments(nodes).reshape(l, l)
        binary_masks[index] = binary_mask

        # print(index, images.shape[0])
        # show_two_imgs_overlay(image, originals[index])
        # show_two_imgs_overlay(binary_masks[index], originals[index])

    return binary_masks
