import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import UNet
from common.graph import EdgeTensorToImg


class GraphUNet(UNet):
    def __init__(self, num_edges, width, height, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__(chs)
        self.edge_module = EdgeTensorToImg(num_edges, width, height)

    def forward(self, x):
        """This has probably some dimension/tensor size issues."""
        x, enc_features = self.encode(x)

        x = self.edge_module(x)

        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)

        return x
