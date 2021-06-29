import torch
from torch import nn
import math


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def in_boundary(self, width, height):
        return 0 <= self.x < width and 0 <= self.y < height

    @property
    def pos(self):
        return self.x, self.y

    @classmethod
    def size(cls):
        return 2


class Edge:
    def __init__(self, node1: Node, node2: Node, thickness):
        self.node1 = node1
        self.node2 = node2
        self.thickness = thickness

    @classmethod
    def size(cls):
        return 3


class Graph:
    def __init__(self, nodes, edges):
        self.nodes = []
        self.edges = []

        for n in nodes:
            self.add_node(n)

        for e in edges:
            self.add_edge(e)

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_edge(self, edge: Edge):
        if edge.node1 not in self.nodes or edge.node2 not in self.nodes:
            raise Exception('Can only add an edge, if its nodes are already in the graph.')

        self.edges.append(edge)

    def to_image(self, width, height):
        img = torch.zeros(width, height)

        for e in self.edges:
            if not e.node1.in_boundary(width, height) or not e.node2.in_boundary(width, height):
                raise Exception('Nodes of edge not in image boundaries')

            self._edge_indices(e, img)
        return img

    def _edge_indices(self, edge: Edge, img):
        start = edge.node1.pos
        end = edge.node2.pos

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        slope = dy / dx

        slope_norm = math.sqrt(dy * dy + dx * dx)
        perp = [-dy / slope_norm, dx / slope_norm]

        for x in range(start[0], end[0] + 1):
            yy = math.floor(x * slope)

            for y in range(math.ceil(slope)):
                img[x, y + yy] = 1

                for t in range(-edge.thickness, edge.thickness + 1):
                    thick_x = math.ceil(t * perp[0])
                    thick_y = math.ceil(t * perp[1])

                    tmp_x = x + thick_x
                    tmp_y = y + yy + thick_y
                    if 0 <= tmp_x < img.shape[0] and 0 <= tmp_y < img.shape[1]:
                        img[tmp_x, tmp_y] = 1

    def print(self):
        print('Nodes:')
        for n in self.nodes:
            print(n.x, n.y)

        print('Edges:')
        for e in self.edges:
            print(f'From: {e.node1.x}/{e.node1.y} to {e.node2.x}/{e.node2.y}, Thickness: {e.thickness}')


def tensor_to_graph(tensor: torch.Tensor, num_nodes, num_edges):
    nodes_size = Node.size() * num_nodes
    edges_size = Edge.size() * num_edges
    size = nodes_size + edges_size

    if tensor.numel() != size:
        raise Exception('Tensor has invalid number of elements.')

    # if tensor.size() != [1, size]:
    #     raise Exception('Invalid dimension')

    nodes_in_tensor = tensor[:nodes_size]
    nodes = []
    for i in range(0, nodes_size, Node.size()):
        x = nodes_in_tensor[i].item()
        y = nodes_in_tensor[i + 1].item()
        nodes.append(Node(x, y))

    edges_in_tensor = tensor[nodes_size:nodes_size + edges_size]
    edges = []
    for i in range(0, edges_size, Edge.size()):
        node1 = nodes[edges_in_tensor[i].item()]
        node2 = nodes[edges_in_tensor[i + 1].item()]
        thickness = edges_in_tensor[i + 2].item()
        edges.append(Edge(node1, node2, thickness))

    return Graph(nodes, edges)


class TensorToGraphToImg(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLu activations
    def __init__(self, num_nodes, num_edges, output_width, output_height):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.nodes_size = Node.size() * num_nodes
        self.edges_size = Edge.size() * num_edges
        self.size = self.nodes_size + self.edges_size

        self.width = output_width
        self.height = output_height

        self.maskEdge = MaskOneEdgeNode(self.width, self.height)

    def forward(self, x: torch.Tensor):

        if x.numel() != self.size:
            raise Exception('Tensor has invalid number of elements.')

        output = torch.zeros(self.width, self.height)
        nodes = x[:self.nodes_size]
        edges = x[self.nodes_size:self.nodes_size + self.edges_size]

        for e in range(0, edges.numel(), Edge.size()):
            start_node_index = edges[e] * Node.size()
            end_node_index = edges[e + 1] * Node.size()

            edge = edges[e:e + Edge.size()]
            start_node = nodes[start_node_index:start_node_index + Node.size()]
            end_node = nodes[end_node_index:end_node_index + Node.size()]
            edge_mask = self.maskEdge(edge, start_node, end_node)

            output = torch.logical_or(output, edge_mask)

        return output


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

        if x.numel() != self.edges_size:
            raise Exception('Tensor has invalid number of elements.')

        output = torch.zeros(self.width, self.height)

        for e in range(0, x.numel(), self.edge_size):
            edge = x[e:e + self.edge_size]
            edge_mask = self.maskEdge(edge)
            print(edge_mask)
            output = output + edge_mask

        output = output / torch.max(output)

        return output


class MaskOneEdge(nn.Module):
    def __init__(self, w, h):
        super().__init__()
        self.width = w
        self.height = h

        index_y = torch.arange(0, w, 1).repeat(h, 1).resize(w, h, 1)
        index_x = torch.arange(0, h, 1).resize(h, 1).repeat(1, w).resize(w, h, 1)
        self.coords = torch.cat([index_x, index_y], dim=2).float()

    def forward(self, edge):

        start_node = edge[0:2]
        end_node = edge[2:4]

        dx = end_node[0] - start_node[0]
        dy = end_node[1] - start_node[1]

        dir = torch.tensor([dx, dy])
        dir_norm = dir[0] ** 2 + dir[1] ** 2

        t = (torch.matmul(self.coords, dir) - start_node.dot(dir)) / dir_norm
        f = (start_node + t.view(-1, 1) * dir).view(self.width, self.height, 2)
        dist = torch.linalg.norm(self.coords - f, dim=2)

        return 1 - dist / torch.max(dist)


class MaskOneEdgeNode(nn.Module):
    def __init__(self, w, h):
        super().__init__()
        self.width = w
        self.height = h

        index_y = torch.arange(0, w, 1).repeat(h, 1).resize(w, h, 1)
        index_x = torch.arange(0, h, 1).resize(h, 1).repeat(1, w).resize(w, h, 1)
        self.coords = torch.cat([index_x, index_y], dim=2)

    def forward(self, edge, start_node, end_node):
        dx = end_node[0] - start_node[0]
        dy = end_node[1] - start_node[1]

        dir = torch.tensor([dx, dy])
        dir_norm = dir[0] ** 2 + dir[1] ** 2

        t = (torch.matmul(self.coords, dir) - start_node.dot(dir)) / dir_norm
        f = (start_node + t.view(-1, 1) * dir).view(self.width, self.height, 2)
        dist = torch.linalg.norm(self.coords - f, dim=2)

        return dist < edge[2]


torch.set_printoptions(precision=2, linewidth=200)

nodes = [Node(0, 0), Node(10, 12)]
edges = [Edge(nodes[0], nodes[1], 3)]

g = Graph(nodes, edges)

tensor = torch.tensor([0., 0, 10, 10, 7, 0, 10, 5, 0, 1, 1, 2, 3, 1])
tensor.requires_grad = True

# g_from_tensor = tensor_to_graph(tensor, 4, 2)
# g_from_tensor.print()



graphTensor = torch.tensor([0., 0, 10, 10, 1, 7, 0, 10, 5, 2])
graphTensor.requires_grad = True

module = EdgeTensorToImg(2, 10, 10)
img_from_tensor = module(graphTensor)
print(img_from_tensor)
img_from_tensor.backward()
