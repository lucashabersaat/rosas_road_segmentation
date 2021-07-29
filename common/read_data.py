import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image

# some constants
PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.5  # minimum average brightness for a mask patch to be classified as containing road

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


def load_all_from_path(path):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    absolute_path = os.path.join(ROOT_DIR, path)
    return (
        np.stack(
            [np.array(Image.open(f)) for f in sorted(glob(absolute_path + "/*.png"))]
        ).astype(np.float32)
        / 255.0
    )


def image_to_patches(images, masks=None):
    # takes in a 4D np.array containing images and (optionally) a 4D np.array containing the segmentation masks
    # returns a 4D np.array with an ordered sequence of patches extracted from the image and (optionally) a np.array
    # containing labels
    n_images = images.shape[0]  # number of images
    h, w = images.shape[1:3]  # shape of images
    assert (h % PATCH_SIZE) + (
        w % PATCH_SIZE
    ) == 0  # make sure images can be patched exactly

    h_patches = h // PATCH_SIZE
    w_patches = w // PATCH_SIZE
    patches = images.reshape(
        (n_images, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE, -1)
    )
    patches = np.moveaxis(patches, 2, 3)
    patches = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)
    if masks is None:
        return patches

    masks = masks.reshape((n_images, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE, -1))
    masks = np.moveaxis(masks, 2, 3)
    labels = np.mean(masks, (-1, -2, -3)) > CUTOFF  # compute labels
    labels = labels.reshape(-1).astype(np.float32)
    return patches, labels


def show_patched_image(patches, labels, h_patches=25, w_patches=25):
    # reorders a set of patches in their original 2D shape and visualizes them
    fig, axs = plt.subplots(h_patches, w_patches, figsize=(18.5, 18.5))
    for i, (p, l) in enumerate(zip(patches, labels)):
        # the np.maximum operation paints patches labeled as road red
        axs[i // w_patches, i % w_patches].imshow(
            np.maximum(p, np.array([l.item(), 0.0, 0.0]))
        )
        axs[i // w_patches, i % w_patches].set_axis_off()
    plt.show()


# # paths to training and validation datasets
# train_path = "data/training"
# val_path = "data/validation"
#
# train_images = load_all_from_path(os.path.join(train_path, "images"))
# train_masks = load_all_from_path(os.path.join(train_path, "groundtruth"))
# val_images = load_all_from_path(os.path.join(val_path, "images"))
# val_masks = load_all_from_path(os.path.join(val_path, "groundtruth"))
#
# # visualize a few images from the training set
# # show_first_n(train_images, train_masks)
#
#
# # extract all patches and visualize those from the first image
# train_patches, train_labels = image_to_patches(train_images, train_masks)
# val_patches, val_labels = image_to_patches(val_images, val_masks)
#
# # the first image is broken up in the first 25*25 patches
# # show_patched_image(train_patches[:25 * 25], train_labels[:25 * 25])
#
# print(
#     "{0:0.2f}".format(sum(train_labels) / len(train_labels) * 100)
#     + "% of training patches are labeled as 1."
# )
# print(
#     "{0:0.2f}".format(sum(val_labels) / len(val_labels) * 100)
#     + "% of validation patches are labeled as 1."
# )
