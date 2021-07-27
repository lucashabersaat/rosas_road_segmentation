from models.unet import UNet
from models.unet_transformer import U_Transformer
from models.transunet.vit_seg_modeling import CONFIGS, VisionTransformer
from models.unets_to_test import U_Net2, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet


def get_model(model_name, config):
    if model_name is None:
        model_name = config["model_name"]

    if model_name == "unet":
        model = UNet()
    elif model_name == "unet2" or model_name == "u_net2":
        model = U_Net2()
    elif model_name == "attUnet":
        model = AttU_Net()
    elif model_name == "r2Unet":
        model = R2U_Net()
    elif model_name == "r2attUnet":
        model = R2AttU_Net()
    elif model_name == "nestedUnet":
        model = NestedUNet()
    elif model_name == "transunet" or model_name == "visiontransformer":
        transunet_config = CONFIGS['R50-ViT-B_16']
        model = VisionTransformer(transunet_config, img_size=config['patch_size'], num_classes=transunet_config.n_classes)
    elif model_name == "unet_transformer":
        config["resize_to"] = 256
        model = U_Transformer(3, 1)
    else:
        raise Exception("unknown model")

    return model
