import dgl
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import pandas as pd
from sklearn.model_selection import train_test_split

class BioSnapDDIDataset(Dataset):
    def __init__(self, name='biosnapddi', path='../dataset', transform=None, pre_transform=None):
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.path = path
        self.train_df = pd.read_csv(f'{path}/{name}/raw/train.csv')
        self.test_df = pd.read_csv(f'{path}/{name}/raw/test.csv')
        self.process()

    def process(self):
        # Process training data
        self.train_data = self._convert_to_pyg_data(self.train_df)

        # Process test data
        self.test_data = self._convert_to_pyg_data(self.test_df)

    def _convert_to_pyg_data(self, df):
        smiles1 = df['smile1'].tolist()
        smiles2 = df['smile2'].tolist()
        labels = df['label'].tolist()

        unique_smiles = list(set(smiles1 + smiles2))
        smiles_to_index = {smile: idx for idx, smile in enumerate(unique_smiles)}
        node_features = torch.eye(len(unique_smiles))  # One-hot encoding (example)

        edge_index = []
        edge_attr = []
        for i, (smile1, smile2, label) in enumerate(zip(smiles1, smiles2, labels)):
            idx1 = smiles_to_index[smile1]
            idx2 = smiles_to_index[smile2]
            edge_index.append([idx1, idx2])
            edge_attr.append(label)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

        # Store SMILES mapping in the Data object (for later retrieval)
        data.smiles_mapping = smiles_to_index
        data.index_to_smiles = {idx: smile for smile, idx in smiles_to_index.items()}

        return data
    def get(self, idx):
        if idx == 0:
            return self.train_data
        elif idx == 1:
            return self.test_data
        else:
            raise IndexError("Index out of range")

    def len(self):
        return 2  # Train, Test

    def __len__(self):
        return self.len()

    def to_dgl_graph(self, pyg_data):
        g = dgl.graph((pyg_data.edge_index[0], pyg_data.edge_index[1]))
        g.ndata['feat'] = pyg_data.x
        g.edata['label'] = pyg_data.edge_attr

        # Store SMILES mappings in the DGL graph
        g.smiles_to_index = pyg_data.smiles_mapping  # {SMILES: node_id}
        g.index_to_smiles = pyg_data.index_to_smiles  # {node_id: SMILES}

        return g

    def to_dgl_graph_with_node_asgraph(self, pyg_data):
        # Convert PyTorch Geometric data to DGL graph
        g = dgl.graph((pyg_data.edge_index[0], pyg_data.edge_index[1]))
        g.ndata['feat'] = pyg_data.x

        # Create a new graph for each node
        node_graphs = []
        for i in range(g.num_nodes()):
            # Create a new graph with a single node
            node_g = dgl.graph(([], []), num_nodes=1)
            node_g.ndata['feat'] = g.ndata['feat'][i].view(1, -1)
            node_graphs.append(node_g)

        return g, node_graphs

    def get_edge_split(self):
        # Convert PyTorch Geometric data to DGL graphs
        train_g = self.to_dgl_graph(self.train_data)
        test_g = self.to_dgl_graph(self.test_data)

        # Extract positive and negative edges from the training and test sets
        train_pos_edges = self.train_data.edge_index[:, self.train_data.edge_attr == 1].t()
        train_neg_edges = self.train_data.edge_index[:, self.train_data.edge_attr == 0].t()
        test_pos_edges = self.test_data.edge_index[:, self.test_data.edge_attr == 1].t()
        test_neg_edges = self.test_data.edge_index[:, self.test_data.edge_attr == 0].t()

        # Randomly sample 20% of the positive and negative edges for validation
        train_pos_edges, valid_pos_edges = train_test_split(train_pos_edges.numpy(), test_size=0.2, random_state=42)
        train_neg_edges, valid_neg_edges = train_test_split(train_neg_edges.numpy(), test_size=0.2, random_state=42)

        # Convert back to torch tensors
        train_pos_edges = torch.tensor(train_pos_edges, dtype=torch.long)
        valid_pos_edges = torch.tensor(valid_pos_edges, dtype=torch.long)
        train_neg_edges = torch.tensor(train_neg_edges, dtype=torch.long)
        valid_neg_edges = torch.tensor(valid_neg_edges, dtype=torch.long)

        # Create the split_edge dictionary
        split_edge = {
            "train": {"edge": train_pos_edges, "edge_neg": train_neg_edges},
            "valid": {"edge": valid_pos_edges, "edge_neg": valid_neg_edges},
            "test": {"edge": test_pos_edges, "edge_neg": test_neg_edges}
        }

        # Check class imbalance
        self._check_class_imbalance(split_edge)

        return split_edge

    def _check_class_imbalance(self, split_edge):
        # Calculate the number of positive and negative edges in each split
        train_pos_count = split_edge["train"]["edge"].size(0)
        train_neg_count = split_edge["train"]["edge_neg"].size(0)
        valid_pos_count = split_edge["valid"]["edge"].size(0)
        valid_neg_count = split_edge["valid"]["edge_neg"].size(0)
        test_pos_count = split_edge["test"]["edge"].size(0)
        test_neg_count = split_edge["test"]["edge_neg"].size(0)

        # Print the counts
        print(f"Training set - Positive edges: {train_pos_count}, Negative edges: {train_neg_count}")
        print(f"Validation set - Positive edges: {valid_pos_count}, Negative edges: {valid_neg_count}")
        print(f"Test set - Positive edges: {test_pos_count}, Negative edges: {test_neg_count}")

        # Calculate the ratio of positive to negative edges
        train_ratio = train_pos_count / train_neg_count if train_neg_count > 0 else float('inf')
        valid_ratio = valid_pos_count / valid_neg_count if valid_neg_count > 0 else float('inf')
        test_ratio = test_pos_count / test_neg_count if test_neg_count > 0 else float('inf')

        print(f"Training set - Positive to Negative ratio: {train_ratio}")
        print(f"Validation set - Positive to Negative ratio: {valid_ratio}")
        print(f"Test set - Positive to Negative ratio: {test_ratio}")


from smile_to_graph import smiles_to_dgl_graph
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv, TAGConv, GINConv, APPNPConv, GATConv,AvgPooling, MaxPooling, GlobalAttentionPooling

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, AvgPooling, MaxPooling, GlobalAttentionPooling

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
        # Add self-loops to the graph
        g = dgl.add_self_loop(g)

        # Update node features through GNN layers
        for layer in self.gnn_layers:
            node_feats = layer(g, node_feats)
            node_feats = F.relu(node_feats)

        # Global pooling to get graph embedding
        graph_embedding = self.pool(g, node_feats)

        # Optional MLP
        graph_embedding = self.mlp(graph_embedding)

        return graph_embedding
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6
from dgllife.model.gnn.gcn import GCN


class DGL_GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, predictor_dim):
        super(DGL_GCN, self).__init__()
        # Ensure hidden_feats is a list or tuple
        if isinstance(hidden_feats, int):
            hidden_feats = [hidden_feats]

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       activation=[F.relu] * len(hidden_feats),
                       predictor_dim=predictor_dim)

    def forward(self, g, node_feats, edge_feats):
        return self.gnn(g, node_feats, edge_feats)


# Example usage



if __name__ == '__main__':
    dataset = BioSnapDDIDataset(name='biosnapddi')
    g = dataset.to_dgl_graph(dataset.test_data)

    # Get SMILES for node 0
    print(g.index_to_smiles[0])

    # Get all SMILES in the graph
    smiles_list = list(g.index_to_smiles.values())
    smiles_list =smiles_list[1:100]
    node_graphs = [smiles_to_dgl_graph(s) for s in smiles_list]

    print("Graph info:")
    print(f"- Nodes: {g.num_nodes()}, Edges: {g.num_edges()}")
    print(f"- Node features: {g.ndata['feat'].shape}")

    for i, node_g in enumerate(node_graphs):
        print(f"Node {i} graph info:")
        print(f"- Nodes: {node_g.num_nodes()}, Edges: {node_g.num_edges()}")
        print(f"- Node features: {node_g.ndata['feat'].shape}")

    # Hyperparameters
    in_feats = 4  # Number of input features (atom properties)
    hidden_size = 16  # Number of hidden units
    num_classes = 16  # Number of output classes (embedding size)
    lr = 0.01  # Learning rate
    num_epochs = 10  # Number of epochs
    # Initialize the model #ATOM_FDIM + BOND_FDIM
    model = DGL_GCN(in_feats=ATOM_FDIM + BOND_FDIM, hidden_feats=128, predictor_dim=128)
    model = DGL_GCN(in_feats=ATOM_FDIM + BOND_FDIM, hidden_feats=128, predictor_dim=128)

    # Example forward pass
    for node_g in node_graphs:
        features = node_g.ndata['feat']
        outputs = model(node_g, features)
        print(f"Output shape: {outputs.shape}")
"""

    # Initialize the model, loss function, and optimizer
    model = MolecularGNN(in_feats, hidden_size, out_feats=num_classes)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for node_g in node_graphs:
            features = node_g.ndata['feat']
            labels = torch.randn(num_classes)  # Placeholder labels, replace with actual labels if available
            optimizer.zero_grad()
            outputs = model(node_g, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(node_graphs):.4f}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        embeddings = []
        for node_g in node_graphs:
            features = node_g.ndata['feat']
            embedding = model(node_g, features)
            embeddings.append(embedding)
"""