import torch
from torch import nn
import segmentation_models_pytorch as sm

from common.read_data import *
from methods.conv_neural_networks import ImageDataSet, train, accuracy_fn, np_to_tensor

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = ImageDataSet("data/training", device)
    val_dataset = ImageDataSet("data/validation", device)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=True
    )
    model = sm.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    loss_fn = nn.BCELoss()
    metric_fns = {"acc": accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 20
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
    test_path = os.path.join(ROOT_DIR, "data/test_images/test_images")
    test_filenames = sorted(glob(test_path + "/*.png"))
    test_images = load_all_from_path(test_path)
    test_patches = np.moveaxis(image_to_patches(test_images), -1, 1)  # HWC to CHW
    test_patches = np.reshape(
        test_patches, (38, -1, 3, PATCH_SIZE, PATCH_SIZE)
    )  # split in batches for memory constraints
    test_pred = [
        model(np_to_tensor(batch, device)).detach().cpu().numpy()
        for batch in test_patches
    ]
    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.round(
        test_pred.reshape(
            test_images.shape[0],
            test_images.shape[1] // PATCH_SIZE,
            test_images.shape[1] // PATCH_SIZE,
        )
    )

    create_submission(
        test_pred, test_filenames, submission_filename="data/submissions/cnn_submission.csv"
    )
