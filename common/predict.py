import torch.cuda
import numpy as np
from glob import glob
import cv2
import os

from common.util import np_to_tensor
from common.read_data import load_all_from_path, create_empty_submission, append_submission, \
    PATCH_SIZE, CUTOFF, ROOT_DIR


def predict_and_write_submission(model, name: str):
    """Using the given model, predict the road masks of the test set and save the prediction as valid submission
    with the given name as file name. Write data into file batch-wise to avoid out of memory errors."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # predict on test set
    test_path = "data/test_images/test_images"
    absolute_path = os.path.join(ROOT_DIR, test_path)

    all_test_filenames = sorted(glob(absolute_path + "/*.png"))
    all_test_images = load_all_from_path(test_path)
    batch_size = 10
    num_test_images = len(all_test_filenames)

    create_empty_submission(submission_filename=f"data/submissions/{name}_submission.csv")

    for i in range(0, num_test_images, batch_size):
        b = min(batch_size, num_test_images-i)

        test_filenames = all_test_filenames[i:i+b]
        test_images = all_test_images[i:i+b]

        size = test_images.shape[1:3]

        # we also need to resize the test images. This might not be the best idea depending on their spatial resolution
        test_images = np.stack(
            [cv2.resize(img, dsize=(384, 384)) for img in test_images], 0
        )
        test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)

        test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
        test_pred = np.concatenate(test_pred, 0)
        test_pred = np.moveaxis(test_pred, -1, 1)  # CHW to HWC

        test_pred = np.stack(
            [cv2.resize(img, dsize=size) for img in test_pred], 0
        )  # resize to original

        # now compute labels
        test_pred = test_pred.reshape(
            (-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE)
        )
        test_pred = np.moveaxis(test_pred, 2, 3)
        test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)

        append_submission(
            test_pred,
            test_filenames,
            submission_filename=f"data/submissions/{name}_submission.csv",
        )


