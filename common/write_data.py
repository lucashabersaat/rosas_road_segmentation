import re

import numpy as np
from glob import glob
import cv2
import os

from common.read_data import PATCH_SIZE, CUTOFF, ROOT_DIR


def write_submission(all_predictions, name, test_path, size):
    """Save the predictions batch wise in submissions."""
    all_test_filenames = sorted(glob(os.path.join(ROOT_DIR, test_path) + "/*.png"))

    num_test_images = len(all_test_filenames)
    batch_size = 20

    create_empty_submission(submission_filename=f"data/submissions/{name}_submission.csv")

    for i in range(0, num_test_images, batch_size):
        b = min(batch_size, num_test_images - i)
        predictions = all_predictions[i:i + b]
        test_filenames = all_test_filenames[i:i + b]

        predictions = np.stack(
            [cv2.resize(img, dsize=size) for img in predictions], 0
        )  # resize to original

        # now compute labels
        predictions = predictions.reshape(
            (-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE)
        )
        predictions = np.moveaxis(predictions, 2, 3)
        predictions = np.round(np.mean(predictions, (-1, -2)) > CUTOFF)

        append_submission(
            predictions,
            test_filenames,
            submission_filename=f"data/submissions/{name}_submission.csv",
        )


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