
import dgl
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

def make_identity_graph(g):
    """
    Convert a DGL graph into an identity graph (only self-loops).

    Args:
        g (dgl.DGLGraph): Input graph (can have node/edge features).

    Returns:
        dgl.DGLGraph: Identity graph (self-loops only).
    """
    num_nodes = g.num_nodes()

    # Generate self-loop edges (i, i)
    src = torch.arange(num_nodes)
    dst = torch.arange(num_nodes)

    # Create the identity graph
    identity_g = dgl.graph((src, dst), num_nodes=num_nodes)

    # Copy node features if they exist
    if 'feat' in g.ndata:
        identity_g.ndata['feat'] = g.ndata['feat'].clone()

    # Initialize edge features (optional)
    if 'feat' in g.edata:
        feat_dim = g.edata['feat'].shape[1]
        identity_g.edata['feat'] = torch.zeros((num_nodes, feat_dim))

    return identity_g
def atom_features_func(atom):
    # Example feature function - replace with your actual implementation
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic())
    ], dtype=torch.float)




def smiles_to_dgl_graph(smiles):
    try:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # Add hydrogen atoms and compute 2D coordinates
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)

        # Get node features (atom types)
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append(atom.GetAtomicNum())

        # Create edge list
        edge_list = []
        for bond in mol.GetBonds():
            edge_list.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

        # Create DGL graph
        g = dgl.graph((torch.tensor([e[0] for e in edge_list]), torch.tensor([e[1] for e in edge_list])))

        # Add node features
        g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float).view(-1, 1)

        return g

    except Exception as e:
        print(f"Error converting SMILES to graph: {smiles}. Error: {e}")

        validTemplate = 'C[C@H](C=O)CC1CC1'
        g= smiles_to_dgl_graph(validTemplate)

        # Convert to identity graph
        identity_g = make_identity_graph(g)
        return identity_g

# Example usage

if __name__ == '__main__':
    # Convert SMILES to DGL graph
    smiles = "CCO"  # Ethanol
    g = smiles_to_dgl_graph(smiles)

    print("Graph info:")
    print(f"- Nodes: {g.num_nodes()}, Edges: {g.num_edges()}")
    print(f"- Node features: {g.ndata['feat'].shape}")
    print(f"- Edge features: {g.edata['feat'].shape}")

    # Output:
    # Graph info:
    # - Nodes: 3, Edges: 2
    # - Node features: torch.Size([3, 4])
    # - Edge features: torch.Size([2, 4])
    valid_smiles = ["CCOc1ccccc1OCCN[C@H](C)Cc1ccc(OC)c(S(N)(=O)=O)c1",
                    "C[C@@H](Oc1ccc2[nH]nc(/C=C/c3cnn(CCO)c3)c2c1)c1c(Cl)cncc1Cl"]
    graphs = [smiles_to_dgl_graph_old(s) for s in valid_smiles if smiles_to_dgl_graph(s) is not None]
    smiles_list = ["CCO", "CCN", "C1CCCCC1"]  # Ethanol, Ethylamine, Cyclohexane
    #graphs = [safe_smiles_to_dgl_graph(s) for s in valid_smiles]
    node_graphs = [smiles_to_dgl_graph_old(s) for s in valid_smiles]

    print("Graph info:")
    print(f"- Nodes: {g.num_nodes()}, Edges: {g.num_edges()}")
    print(f"- Node features: {g.ndata['feat'].shape}")
    # print(f"- Edge features: {g.edata['feat'].shape}")

    for i, node_g in enumerate(node_graphs):
        print(f"Node {i} graph info:")
        print(f"- Nodes: {node_g.num_nodes()}, Edges: {node_g.num_edges()}")
        print(f"- Node features: {node_g.ndata['feat'].shape}")