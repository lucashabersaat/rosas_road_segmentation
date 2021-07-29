import cv2
import numpy as np
from sklearn.preprocessing import normalize

from common.plot_data import show_two_imgs_overlay
from common.read_data import load_all_from_path
from common.postprocess import postprocess
from common.write_data import write_submission, classify_cutoff


def load_images(path):
    images = load_all_from_path(path)
    return np.moveaxis(images, -1, 1)


def load_predictions(path):
    return np.load(path)


if __name__ == "__main__":
    images = load_images("data/test_images")
    predictions = load_predictions("../predictions.npy")

    predictions = np.add(predictions, abs(predictions.min()))
    predictions = np.divide(predictions, predictions.max())

    predictions = postprocess(predictions)

    for i in range(5):
        resized = cv2.resize(predictions[i].astype(float), dsize=(608, 608))
        show_two_imgs_overlay(images[i], resized)

    predictions = classify_cutoff(predictions, (608, 608))

    for i in range(5):
        resized = cv2.resize(predictions[i].astype(float), dsize=(608, 608))
        show_two_imgs_overlay(images[i], resized)

    exit()

    write_submission(
        images,
        predictions,
        "bla",
        "data/test_images",
        (608, 608),
        graph_cut=False,
    )
