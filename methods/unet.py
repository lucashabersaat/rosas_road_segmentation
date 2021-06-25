import torch
from torch import nn

from common.read_data import *
from common.util import np_to_tensor, accuracy_fn, print_model_memory
from common.image_data_set import ImageDataSet
from methods.conv_neural_networks import train
from models.unet import UNet, patch_accuracy_fn

if __name__ == "__main__":

    torch.cuda.empty_cache()
    resize_to = 384

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4

    train_dataset = ImageDataSet(
        "data/training", device, use_patches=False, resize_to=(resize_to, resize_to)
    )
    val_dataset = ImageDataSet(
        "data/validation", device, use_patches=False, resize_to=(resize_to, resize_to)
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    model = UNet().to(device)
    loss_fn = nn.BCELoss()
    metric_fns = {"acc": accuracy_fn, "patch_acc": patch_accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 35
    train(
        train_dataloader,
        val_dataloader,
        model,
        loss_fn,
        metric_fns,
        optimizer,
        n_epochs,
    )





    # predict on test set
    test_path = "data/test_images/test_images"
    test_filenames = sorted(glob(test_path + "/*.png"))
    test_images = load_all_from_path(test_path)
    batch_size = test_images.shape[0]
    size = test_images.shape[1:3]

    # we also need to resize the test images. This might not be the best idea depending on their spatial resolution
    test_images = np.stack(
        [cv2.resize(img, dsize=(resize_to, resize_to)) for img in test_images], 0
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

    create_submission(
        test_pred,
        test_filenames,
        submission_filename="data/submissions/unet_submission.csv",
    )
