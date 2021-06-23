import torch
from torch import nn

from methods.unet import UNet


class UNetSelfAttention(UNet):
    # UNet-like architecture for single class semantic segmentation.

    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super(UNetSelfAttention, self).__init__(chs)

        last_dim = chs[-1]
        self.q_layer = nn.Linear(last_dim, last_dim)
        self.k_layer = nn.Linear(last_dim, last_dim)
        self.v_layer = nn.Linear(last_dim, last_dim)

        self.self_attention = nn.MultiheadAttention(1024, num_heads=8)

    def forward(self, x):
        x, enc_feature = self.encode(x)

        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)

        x = self.self_attention(q, k, v)

        x = self.decode(x, enc_feature)

        return self.head(x)
