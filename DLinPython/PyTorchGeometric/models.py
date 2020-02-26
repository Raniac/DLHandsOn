import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.dense import diff_pool

from torch_geometric.utils import add_self_loops

class Net_191106(torch.nn.Module):
    def __init__(self):
        super(Net_191106, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)

## Graph Convolutional Network
class GCNNet(torch.nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(4, 16)
        self.conv2 = GCNConv(16, 64)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)

## Graph Attention Network
class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(4, 8)
        self.conv2 = GATConv(8, 8)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)

## Graph Isomorphism Network
## GIN achieves maximal discriminative power by using injective neighbor aggregation.
class GIN(torch.nn.Module):
    def __init__(self):
        super(GIN, self).__init__()
        nn = Seq(Lin(4, 16), ReLU(), Lin(16, 4))
        self.conv = GINConv(nn, train_eps=True)
        self.feat_after_gap = []

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = global_mean_pool(x, batch)
        self.feat_after_gap.extend(x)

        return F.log_softmax(x, dim=1)

## Hierarchical Graph Representation Learning with Differentiable Pooling
## https://arxiv.org/abs/1806.08804
# TODO: study on GDP inputs parameters
class GDP(torch.nn.Module):
    def __init__(self):
        super(GDP, self).__init__()

    def forward(self, data):
        pass
