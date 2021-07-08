import numpy as np
import torch
from torch import nn

from common.util import init_tensor
from common.plot_data import *

"""
The following modules, assumes the input tensor is a list of edges and converts it into a image.
Each five consecutive values make up an edge. 
- First two are the coordinates of the starting point, 
- second two the end point 
- and the last value is the thickness in pixels.

The idea behind this module, is that some model tries to find this list of edges and the EdgesTensorToImg converts 
it into a nice mask image of roads, which looks much closer to the groundtruth.

Challenge is building a architecture, that does search for this edge representation. 
"""


class EdgeTensorToImg(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLu activations
    def __init__(self, num_edges, output_width, output_height):
        super().__init__()
        self.num_edges = num_edges

        self.edge_size = 5
        self.edges_size = self.edge_size * num_edges

        self.width = output_width
        self.height = output_height

        self.maskEdge = MaskOneEdge(self.width, self.height)

    def forward(self, x: torch.Tensor):

        x = x.squeeze()

        if x.numel() != self.edges_size:
            raise Exception('Tensor has invalid number of elements.')

        output = torch.zeros(self.width, self.height)
        output = init_tensor(output)

        for e in range(0, x.numel(), self.edge_size):
            edge = x[e:e + self.edge_size]
            edge_mask = self.maskEdge(edge)
            output = torch.maximum(output, edge_mask)

        output = output / torch.max(output)

        return output


class MaskOneEdge(nn.Module):
    def __init__(self, w, h):
        super().__init__()
        self.width = w
        self.height = h

        index_y = torch.arange(0, w, 1).repeat(h, 1).resize(w, h, 1)
        index_x = torch.arange(0, h, 1).resize(h, 1).repeat(1, w).resize(w, h, 1)

        coords = torch.cat([index_x, index_y], dim=2).float()
        self.coords = init_tensor(coords)

    def forward(self, edge):
        start_node = edge[0:2]
        end_node = edge[2:4]

        dx = end_node[0] - start_node[0]
        dy = end_node[1] - start_node[1]

        dir = torch.stack([dx, dy])
        dir_norm = dir[0] ** 2 + dir[1] ** 2

        t = (torch.matmul(self.coords, dir) - start_node.dot(dir)) / dir_norm
        double_t = torch.stack([t.clone(), t.clone()], 2)

        f = start_node + double_t * dir

        dist = torch.linalg.norm(self.coords - f, dim=2)

        result = dist / edge[4] * 2

        result = self.plateau_distance_to_line(result) * 2

        result = torch.mul(result, self.plateau_distance_from_ends(t))

        return result

    @staticmethod
    def plateau_distance_to_line(input):
        a = 1
        sigma = 0.01

        return MaskOneEdge.plateau(input, a, sigma)

    @staticmethod
    def plateau_distance_from_ends(input):
        input -= 0.5
        a = 0.5
        sigma = 0.01

        return MaskOneEdge.plateau(input, a, sigma)

    @staticmethod
    def plateau(input, a, sigma):
        erf1 = torch.special.erf((input + a) / sigma)
        erf2 = torch.special.erf((input - a) / sigma)

        return 1 / (4 * a) * (erf1 - erf2)

    @staticmethod
    def plot_plateau():
        start = -2
        end = 2
        step = (end - start) / 1000

        x = []
        y = []
        for i in np.arange(start, end, step):
            x.append(i)
            y.append(MaskOneEdge.plateau_distance_to_line(torch.tensor([i])).numpy())

        plt.plot(x, y)
        plt.show()


if __name__ == "__main__":
    torch.set_printoptions(precision=2, linewidth=200)
    torch.autograd.set_detect_anomaly(True)

    edgesTensor = init_tensor(torch.tensor([0., 0, 100, 100, 10, 100, 0, 10, 5, 2]))
    edgesTensor.requires_grad = True

    module = EdgeTensorToImg(2, 192, 192)
    img_from_tensor = module(edgesTensor)
    img_from_tensor.sum().backward()

    # MaskOneEdge.plot_plateau()

    plt.imshow(img_from_tensor.cpu().detach().numpy(), vmin=0, vmax=1)
    plt.show()
