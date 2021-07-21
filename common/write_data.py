import re

import numpy as np
from glob import glob
import cv2
import os
import time

from common.read_data import PATCH_SIZE, ROOT_DIR, CUTOFF
from common.plot_data import *
from common.postprocess import *
from common.postprocess import graph_cut


def write_submission(original, all_predictions, name, test_path, size, graph_cut=False):
    """Save the predictions batch wise in submissions."""
    all_test_filenames = sorted(glob(os.path.join(ROOT_DIR, test_path) + "/*.png"))

    num_test_images = len(all_test_filenames)
    batch_size = num_test_images

    file_name = f"data/submissions/{name}_submission.{str(int(time.time()))}.csv"
    print("Writing to", file_name)
    create_empty_submission(submission_filename=file_name)

    for i in range(0, num_test_images, batch_size):
        b = min(batch_size, num_test_images - i)
        predictions = all_predictions[i:i + b]
        test_filenames = all_test_filenames[i:i + b]

        # resize to original
        predictions = np.stack(
            [cv2.resize(img, dsize=size) for img in predictions], 0
        )

        original = np.moveaxis(original, 1, -1)
        original = np.stack(
            [cv2.resize(img, dsize=size) for img in original], 0
        )

        # now compute labels
        if graph_cut:
            predictions = classify_graph_cut(predictions, original)

        predictions = classify_cutoff(predictions, size)

        for i in range(predictions.shape[0]):
            resized = cv2.resize(predictions[i].astype(float), dsize=original[i].shape[1:])
            show_two_imgs_overlay(original[i], resized)
            break

        append_submission(
            predictions,
            test_filenames,
            submission_filename=file_name,
        )


def classify_cutoff(predictions, size):
    predictions = predictions.reshape(
        (-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE)
    )
    predictions = np.moveaxis(predictions, 2, 3)
    return np.round(np.mean(predictions, (-1, -2)) > CUTOFF)


def classify_graph_cut(predictions, original):
    return graph_cut(predictions, original)


def create_empty_submission(submission_filename):
    with open(os.path.join(ROOT_DIR, submission_filename), "w") as f:
        f.write("id,prediction\n")


def create_submission(labels, test_filenames, submission_filename):
    with open(os.path.join(ROOT_DIR, submission_filename), "w") as f:
        f.write("id,prediction\n")
        write_into_file(f, test_filenames, labels)


def append_submission(labels, test_filenames, submission_filename):
    with open(os.path.join(ROOT_DIR, submission_filename), "a") as f:
        write_into_file(f, test_filenames, labels)


def write_into_file(file, test_filenames, labels):
    """Append to the given file the predictions"""
    for fn, patch_array in zip(sorted(test_filenames), labels):
        img_number = int(re.search(r"\d+", fn).group(0))
        for i in range(patch_array.shape[0]):
            for j in range(patch_array.shape[1]):
                file.write(
                    "{:03d}_{}_{},{}\n".format(
                        img_number,
                        j * PATCH_SIZE,
                        i * PATCH_SIZE,
                        int(patch_array[i, j]),
                    )
                )
