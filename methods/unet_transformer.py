from torch import nn

from common.read_data import *
from common.util import *
from common.image_data_set import ImageDataSet
from methods.unet import patch_accuracy_fn
from methods.conv_neural_networks import train
from common.unet_transformer_includes import NoiseRobustDiceLoss
from models.unet_transformer import U_Transformer

if __name__ == "__main__":
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    resize_to = 384

    train_dataset = ImageDataSet(
        "data/training", device, use_patches=False, resize_to=(resize_to, resize_to)
    )

    #print(train_dataset_shape,train_dataset.x.shape)
    a,b,c,d = train_dataset.x.shape
    train_dataset.x = train_dataset.x.reshape(4*a,3,int(c/2),int(d/2))
    #print('train_dataset_shape',train_dataset.x.shape)

    #print(train_dataset.y.shape)
    e,f,g = train_dataset.y.shape
    train_dataset.y = train_dataset.y.reshape(4*e,int(f/2),int(g/2))
    #print(train_dataset.y.shape)

    val_dataset = ImageDataSet(
        "data/validation", device, use_patches=False, resize_to=(resize_to, resize_to)
    )

    #print(val_dataset.x.shape)
    a,b,c,d = val_dataset.x.shape
    val_dataset.x = val_dataset.x.reshape(4*a,3,int(c/2),int(d/2))
    #print(val_dataset.x.shape)

    #print(val_dataset.y.shape)
    a,b,c = val_dataset.y.shape
    val_dataset.y = val_dataset.y.reshape(4*a,int(b/2),int(c/2))
    #print(val_dataset.y.shape)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    model = U_Transformer(3, 1, True).to(device)
    # use same loss as in the repo
    loss_fn = NoiseRobustDiceLoss(eps=1e-7)

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
    test_filenames = sorted(glob(os.path.join(ROOT_DIR, test_path) + "/*.png"))
    test_images = load_all_from_path(test_path)
    batch_size = test_images.shape[0]
    size = test_images.shape[1:3]



    # we also need to resize the test images. This might not be the best idea depending on their spatial resolution
    test_images = np.stack(
        [cv2.resize(img, dsize=(resize_to, resize_to)) for img in test_images], 0
    )
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)

    a,b,c,d = test_images.shape
    test_images = test_images.reshape(4*a,3,int(c/2),int(d/2))

    #output is a list not ndarray, to do reshape to output dim

    test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
    print(test_pred.shape)
    a,b,c,d = test_pred.shape
    test_pred = test_pred.reshape(int(a/4),3, c*2,d*2)
    test_pred = np.concatenate(test_pred, 0)
    print(test_pred.shape)
    test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
    print(test_pred.shape)

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
        submission_filename="data/submissions/unet_transformer_submission.csv",
    )
