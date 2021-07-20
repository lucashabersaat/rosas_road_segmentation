from models.unet import UNet
from models.unet_transformer import U_Transformer
from models.transunet.vit_seg_modeling import CONFIGS, VisionTransformer
from models.unets_to_test import U_Net2, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet


def get_model(model_name, config):
    if model_name is None:
        model_name = "unet_transformer"

    if model_name == "unet":
        config["resize_to"] = 384
        config["divide_into_four"] = False
        model = UNet()
    elif model_name == "unet2":
        config["resize_to"] = 384
        config["divide_into_four"] = False
        model = U_Net2()
        # R2U_Net, AttU_Net, R2AttU_Net, NestedUNet
    elif model_name == "attUnet":
        config["resize_to"] = 384
        config["divide_into_four"] = False
        model = AttU_Net()
    elif model_name == "r2Unet":
        config["resize_to"] = 384
        config["divide_into_four"] = False
        model = R2U_Net()
    elif model_name == "attUnet":
        config["resize_to"] = 384
        config["divide_into_four"] = False
        model = AttU_Net()
    elif model_name == "r2attUnet":
        config["resize_to"] = 384
        config["divide_into_four"] = False
        model = R2AttU_Net()
    elif model_name == "nestedUnet":
        config["resize_to"] = 384
        config["divide_into_four"] = False
        model = NestedUNet()
    elif model_name == "transunet":
        config["resize_to"] = 384
        config["batch_size"] = 4
        config["divide_into_four"] = False
        config['loss_fn'] = "noise_robust_dice"
        transunet_config = CONFIGS['R50-ViT-B_16']
        model = VisionTransformer(transunet_config, img_size=config["resize_to"],
                                  num_classes=transunet_config.n_classes)
    elif model_name == "unet_transformer":
        config['loss_fn'] = "noise_robust_dice"
        config["resize_to"] = 256
        model = U_Transformer(3, 1)
    else:
        raise Exception("unknown model")

    return model
