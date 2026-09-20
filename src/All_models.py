
import torch
import torch.nn.functional as F
import networkx as nx
from torch import nn
from torch.nn import Linear

import random
import torch_geometric.transforms as T
import math
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, convert, to_dense_adj
from torch_geometric.nn.conv import MessagePassing
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
#from torch_geometric.nn import GATConv
from dgl.nn.pytorch import GraphConv, SAGEConv, TAGConv, GINConv, APPNPConv, GATConv,AvgPooling, MaxPooling, GlobalAttentionPooling



class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, aggr="add"):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True, aggr=aggr))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True, aggr=aggr))
        self.convs.append(SAGEConv(hidden_channels, out_channels, normalize=True, aggr=aggr))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv

class SkipConnSAGE(nn.Module):
    def __init__(self, in_channels, hidden_dimension, out_channels, num_layers, dropout):
        super(SkipConnSAGE, self).__init__()

        self.convs = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_dimension, aggregator_type="mean"))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dimension, hidden_dimension, aggregator_type="mean"))
        self.convs.append(SAGEConv(hidden_dimension, out_channels, aggregator_type="mean"))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, g, x):
        prev_x = None
        for i in range(len(self.convs) - 1):
            prev_x = x
            x = self.convs[i](g, x)
            # Skip Connection
            if i > 0:
                x = x + prev_x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)
        return x

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, num_heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, num_heads))
        self.convs.append(GATConv(hidden_channels * num_heads, out_channels, num_heads))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, g, x):
        for conv in self.convs[:-1]:
            x = F.relu(conv(g, x).view(-1, conv._out_feats))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x).view(-1, self.convs[-1]._out_feats)
        return x

class GCN(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            dropout,
            ngnn_type,
            dataset,
    ):
        super(GCN, self).__init__()

        self.dataset = dataset
        self.convs = torch.nn.ModuleList()

        num_nonl_layers = (
            1 if num_layers <= 2 else 2
        )  # number of nonlinear layers in each conv layer
        if ngnn_type == "input":
            self.convs.append(
                NGNN_GCNConv(
                    in_channels,
                    hidden_channels,
                    hidden_channels,
                    num_nonl_layers,
                )
            )
            for _ in range(num_layers - 2):
                self.convs.append(GraphConv(hidden_channels, hidden_channels))
        elif ngnn_type == "hidden":
            self.convs.append(GraphConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(
                    NGNN_GCNConv(
                        hidden_channels,
                        hidden_channels,
                        hidden_channels,
                        num_nonl_layers,
                    )
                )

        self.convs.append(GraphConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, g, x):
        for conv in self.convs[:-1]:
            x = conv(g, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)
        return x

class NGNN_GCNConv(torch.nn.Module):
    def __init__(
            self, in_channels, hidden_channels, out_channels, num_nonl_layers
    ):
        super(NGNN_GCNConv, self).__init__()
        self.num_nonl_layers = (
            num_nonl_layers  # number of nonlinear layers in each conv layer
        )
        self.conv = GraphConv(in_channels, hidden_channels)
        self.tagConv = TAGConv(hidden_channels, hidden_channels)
        # self.apaConv= APPNPConv(k=10, alpha=0.2)
        self.fc = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        for bias in [self.fc.bias, self.fc2.bias]:
            stdv = 1.0 / math.sqrt(bias.size(0))
            bias.data.uniform_(-stdv, stdv)

    def forward(self, g, x):
        x = self.conv(g, x)
        x = self.tagConv(g, x)
        # x = self.apaConv(g, x)

        if self.num_nonl_layers == 2:
            x = F.relu(x)
            x = self.fc(x)

        x = F.relu(x)
        x = self.fc2(x)
        return x



class NGNN_SAGEConv(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            num_nonl_layers,
            *,
            reduce,
    ):
        super(NGNN_SAGEConv, self).__init__()
        self.num_nonl_layers = (
            num_nonl_layers  # number of nonlinear layers in each conv layer
        )
        self.conv = SAGEConv(in_channels, hidden_channels, reduce)
        self.fc = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        for bias in [self.fc.bias, self.fc2.bias]:
            stdv = 1.0 / math.sqrt(bias.size(0))
            bias.data.uniform_(-stdv, stdv)

    def forward(self, g, x):
        x = self.conv(g, x)

        if self.num_nonl_layers == 2:
            x = F.relu(x)
            x = self.fc(x)

        x = F.relu(x)
        x = self.fc2(x)
        return x


class SAGE_NGNN(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            dropout,
            ngnn_type,
            dataset,
            reduce="mean",
    ):
        super(SAGE_NGNN, self).__init__()

        self.dataset = dataset
        self.convs = torch.nn.ModuleList()

        num_nonl_layers = (
            1 if num_layers <= 2 else 2
        )  # number of nonlinear layers in each conv layer
        if ngnn_type == "input":
            self.convs.append(
                NGNN_SAGEConv(
                    in_channels,
                    hidden_channels,
                    hidden_channels,
                    num_nonl_layers,
                    reduce=reduce,
                )
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    SAGEConv(hidden_channels, hidden_channels, reduce)
                )
        elif ngnn_type == "hidden":
            self.convs.append(SAGEConv(in_channels, hidden_channels, reduce))
            for _ in range(num_layers - 2):
                self.convs.append(
                    NGNN_SAGEConv(
                        hidden_channels,
                        hidden_channels,
                        hidden_channels,
                        num_nonl_layers,
                        reduce=reduce,
                    )
                )

        self.convs.append(SAGEConv(hidden_channels, out_channels, reduce))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, g, x):
        for conv in self.convs[:-1]:
            x = conv(g, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)
        return x

# Define GNN Models

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, out_channels)

    def forward(self, g, x):
        x = F.relu(self.conv1(g, x))
        x = self.conv2(g, x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


class SAGEEncoder(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(SAGEEncoder, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_size, num_classes, aggregator_type='mean')

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        return x
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

class SAGE_mlp(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(SAGE_mlp, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_size, hidden_size, aggregator_type='mean')
        self.fc = Linear(hidden_size, num_classes)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)

        return self.fc(x)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout):
        super(GATEncoder, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, num_heads=num_heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, num_heads=num_heads))
        self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels // num_heads, num_heads=num_heads))
        self.fc = nn.Linear(hidden_channels, out_channels)

        self.dropout = dropout

    def forward(self, g, x):
        for conv in self.convs[:-1]:
            x = conv(g, x)
            x = x.view(-1, conv._out_feats * conv._num_heads)  # Reshape to concatenate the heads
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)
        x = x.view(-1, self.convs[-1]._out_feats * self.convs[-1]._num_heads)  # Reshape to concatenate the heads
        x = self.fc(x)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
class GCNSkipConnection(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCNSkipConnection, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_channels, norm='both', weight=True, bias=True))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels, norm='both', weight=True, bias=True))
        self.convs.append(GraphConv(hidden_channels, out_channels, norm='both', weight=True, bias=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, g, x):
        prev_x = None
        for i in range(len(self.convs) - 1):
            prev_x = x
            x = self.convs[i](g, x)
            # Skip Connection
            if i > 0:
                x = x + prev_x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)
        return x



class MolecularGNN(nn.Module):
    def __init__(
            self,
            in_feats=4,  # Input node feature size (e.g., atom features)
            hidden_feats=128,  # Hidden layer size
            out_feats=256,  # Fixed output embedding size
            num_layers=3,  # Number of GNN layers
            pool_type='mean'  # Pooling type: 'mean', 'max', or 'attention'
    ):
        super(MolecularGNN, self).__init__()

        # GNN Layers (e.g., GraphConv, GAT, GraphSAGE)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_feats if i == 0 else hidden_feats
            self.gnn_layers.append(GraphConv(in_dim, hidden_feats))

        # Pooling Layer
        if pool_type == 'mean':
            self.pool = AvgPooling()
        elif pool_type == 'max':
            self.pool = MaxPooling()
        elif pool_type == 'attention':
            self.pool = GlobalAttentionPooling(nn.Linear(hidden_feats, 1))
        else:
            raise ValueError("Invalid pool_type. Choose 'mean', 'max', or 'attention'.")

        # Optional MLP to refine embedding
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )

    def forward(self, g, node_feats):
        # Update node features through GNN layers
        for layer in self.gnn_layers:
            node_feats = layer(g, node_feats)
            node_feats = F.relu(node_feats)

        # Global pooling to get graph embedding
        graph_embedding = self.pool(g, node_feats)

        # Optional MLP
        graph_embedding = self.mlp(graph_embedding)

        return graph_embedding