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


def write_submission(original, predictions, name, test_path, size, graph_cut=False):
    """Save the predictions batch wise in submissions."""
    test_filenames = sorted(glob(os.path.join(ROOT_DIR, test_path) + "/*.png"))

    file_name = f"data/submissions/{name}_submission.{str(int(time.time()))}.csv"
    print("Writing to", file_name)
    create_empty_submission(submission_filename=file_name)

    if size != original.shape[1:3]:
        # resize to original
        predictions = np.stack([cv2.resize(img, dsize=size) for img in predictions], 0)

        original = np.moveaxis(original, 1, -1)
        original = np.stack([cv2.resize(img, dsize=size) for img in original], 0)

    # now compute labels
    if graph_cut:
        predictions = classify_graph_cut(predictions, original)

    predictions = classify_cutoff(predictions, size, CUTOFF)

    append_submission(
        predictions,
        test_filenames,
        submission_filename=file_name,
    )


def classify_cutoff(predictions, size, cutoff):
    predictions = predictions.reshape(
        (-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE)
    )
    predictions = np.moveaxis(predictions, 2, 3)
    return np.round(np.mean(predictions, (-1, -2)) > cutoff)


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
