import torch
import torch.nn as nn
import torch.nn.functional as F

from common.graph import EdgeTensorToImg


class GraphNet(nn.Module):
    def __init__(self, num_edges, width, height):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2,)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 45 * 45, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_edges * 5)
        self.edge_module = EdgeTensorToImg(num_edges, width, height)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x), inplace=False))
        x = self.pool(F.relu(self.conv2(x), inplace=False))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x), inplace=False)
        x = F.relu(self.fc2(x), inplace=False)
        x = self.fc3(x)
        x = self.edge_module(x)

        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)

        return x
