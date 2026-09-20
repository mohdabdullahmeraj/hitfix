import dgl
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from tdc.multi_pred import DDI
import pandas as pd
from dgl.dataloading.negative_sampler import GlobalUniform  # Corrected import

class TDCDDIDataset(Dataset):
    def __init__(self, name='DrugBank', path='./data/', transform=None, pre_transform=None):
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.data = DDI(name=self.name, path=path)
        self.split = self.data.get_split()
        self.process()

    def process(self):
        # Process training data
        train_df = self.split['train']
        self.train_data = self._convert_to_pyg_data(train_df)

        # Process validation data
        val_df = self.split['valid']
        self.val_data = self._convert_to_pyg_data(val_df)

        # Process test data
        test_df = self.split['test']
        self.test_data = self._convert_to_pyg_data(test_df)

    def _convert_to_pyg_data(self, df):
        # Extract SMILES strings and labels
        smiles1 = df['Drug1'].tolist()
        smiles2 = df['Drug2'].tolist()
        labels = df['Y'].tolist()

        # Create node features (for simplicity, using one-hot encoding of SMILES strings)
        unique_smiles = list(set(smiles1 + smiles2))
        smiles_to_index = {smile: idx for idx, smile in enumerate(unique_smiles)}
        node_features = torch.eye(len(unique_smiles))

        # Create edge index and edge attributes
        edge_index = []
        edge_attr = []
        for i, (smile1, smile2, label) in enumerate(zip(smiles1, smiles2, labels)):
            idx1 = smiles_to_index[smile1]
            idx2 = smiles_to_index[smile2]
            edge_index.append([idx1, idx2])
            edge_attr.append(label)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create PyTorch Geometric Data object
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def get(self, idx):
        if idx == 0:
            return self.train_data
        elif idx == 1:
            return self.val_data
        elif idx == 2:
            return self.test_data
        else:
            raise IndexError("Index out of range")

    def len(self):
        return 3  # Train, Val, Test

    def __len__(self):
        return self.len()

    def to_dgl_graph(self, pyg_data):
        # Convert PyTorch Geometric data to DGL graph
        g = dgl.graph((pyg_data.edge_index[0], pyg_data.edge_index[1]))
        g.ndata['feat'] = pyg_data.x
        g.edata['label'] = pyg_data.edge_attr
        return g

    def get_edge_split(self, num_neg_samples=1):
        # Convert PyTorch Geometric data to DGL graphs
        train_g = self.to_dgl_graph(self.train_data)
        val_g = self.to_dgl_graph(self.val_data)
        test_g = self.to_dgl_graph(self.test_data)

        # Generate negative edges for training, validation, and test sets
        neg_sampler = GlobalUniform(k=num_neg_samples)

        # Training set
        train_pos_edges = self.train_data.edge_index.t()
        train_neg_edges = neg_sampler(train_g, train_pos_edges)

        # Validation set
        val_pos_edges = self.val_data.edge_index.t()
        val_neg_edges = neg_sampler(val_g, val_pos_edges)

        # Test set
        test_pos_edges = self.test_data.edge_index.t()
        test_neg_edges = neg_sampler(test_g, test_pos_edges)

        # Create the split_edge dictionary
        split_edge = {
            "eval_train": {"edge": train_pos_edges},
            "valid": {"edge": val_pos_edges, "edge_neg": val_neg_edges},
            "test": {"edge": test_pos_edges, "edge_neg": test_neg_edges}
        }

        return split_edge

    def get_full_data(self):
        # Combine the train, validation, and test data into one unified graph

        # Get individual datasets
        train, val, test = self.train_data, self.val_data, self.test_data

        # Concatenate node features
        x = train.x  # All splits share the same node set in this case

        # Concatenate edge indices and attributes
        edge_index = torch.cat([train.edge_index, val.edge_index, test.edge_index], dim=1)
        edge_attr = torch.cat([train.edge_attr, val.edge_attr, test.edge_attr], dim=0)

        # Return a new combined PyG Data object
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# Usage
# Usage in the main program
if __name__ == '__main__':
    dataset = TDCDDIDataset(name='DrugBank')
    print(f'len {len(dataset)}')

    split_edge = dataset.get_edge_split()

    pos_train_edge = split_edge["eval_train"]["edge"]
    pos_valid_edge = split_edge["valid"]["edge"]
    neg_valid_edge = split_edge["valid"]["edge_neg"]
    pos_test_edge = split_edge["test"]["edge"]
    neg_test_edge = split_edge["test"]["edge_neg"]

    print("Positive Train Edges:", pos_train_edge)
    print("Positive Validation Edges:", pos_valid_edge)
    print("Negative Validation Edges:", neg_valid_edge)
    print("Positive Test Edges:", pos_test_edge)
    print("Negative Test Edges:", neg_test_edge)