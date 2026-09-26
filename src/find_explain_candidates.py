
import torch
import sys
sys.path.insert(0, ".")

from All_models import GCNSkipConnection
from main import LinkPredictor
import dgl
from ogb.linkproppred import DglLinkPropPredDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the ogbl-ddi graph + test split (same data both models were evaluated on)
dataset = DglLinkPropPredDataset(name="ogbl-ddi", root="../dataset")
split_edge = dataset.get_edge_split()
g = dataset[0].to(device)
g = dgl.add_self_loop(g)
test_edges = split_edge["test"]["edge"].to(device)

def load_model(ckpt_path, use_common_neighbors):
    ckpt = torch.load(ckpt_path, map_location=device)
    node_features = ckpt["node_features"].to(device)
    hidden = node_features.size(-1)

    model = GCNSkipConnection(hidden, hidden, hidden, 3, 0.5).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    predictor = LinkPredictor(hidden, hidden, 1, 3, 0.5,
                               extra_feat_dim=(1 if use_common_neighbors else 0)).to(device)
    predictor.load_state_dict(ckpt["predictor_state"])
    predictor.eval()

    return model, predictor, node_features

print("Loading baseline model...")
baseline_model, baseline_predictor, baseline_feat = load_model(
    "/content/drive/MyDrive/HitFix_Research_Project/06_interpretability/checkpoints/baseline_best_model.pt",
    use_common_neighbors=False,
)

print("Loading fixed model...")
fixed_model, fixed_predictor, fixed_feat = load_model(
    "/content/drive/MyDrive/HitFix_Research_Project/06_interpretability/checkpoints/fixed_best_model.pt",
    use_common_neighbors=True,
)

# Build the adjacency matrix for the common-neighbor feature (same as during training)
num_nodes = g.num_nodes()
adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
train_edges = split_edge["train"]["edge"].to(device)
adj[train_edges[:, 0], train_edges[:, 1]] = 1.0
adj[train_edges[:, 1], train_edges[:, 0]] = 1.0

@torch.no_grad()
def score_all_test_edges(model, predictor, feat, use_cn):
    h = model(g, feat)
    edge = test_edges.t()
    scores = []
    for start in range(0, edge.size(1), 50000):
        e0 = edge[0][start:start+50000]
        e1 = edge[1][start:start+50000]
        cn = None
        if use_cn:
            cn = (torch.log1p((adj[e0] * adj[e1]).sum(dim=1)) / 6.2).unsqueeze(-1)
        s = predictor(h[e0], h[e1], cn).squeeze()
        scores.append(s.cpu())
    return torch.cat(scores)

print("Scoring test edges with baseline model...")
baseline_scores = score_all_test_edges(baseline_model, baseline_predictor, baseline_feat, use_cn=False)

print("Scoring test edges with fixed model...")
fixed_scores = score_all_test_edges(fixed_model, fixed_predictor, fixed_feat, use_cn=True)

# Rank each test edge's score against a large random negative sample, to approximate its Hits@K rank
neg_edges = split_edge["test"]["edge_neg"].to(device)
neg_edges = neg_edges[0] if isinstance(neg_edges, (tuple, list)) else neg_edges

@torch.no_grad()
def score_negatives(model, predictor, feat, use_cn, sample_size=20000):
    h = model(g, feat)
    idx = torch.randperm(neg_edges.size(0))[:sample_size]
    sample = neg_edges[idx]
    cn = None
    if use_cn:
        cn = (torch.log1p((adj[sample[:, 0]] * adj[sample[:, 1]]).sum(dim=1)) / 6.2).unsqueeze(-1)
    return predictor(h[sample[:, 0]], h[sample[:, 1]], cn).squeeze().cpu()

print("Scoring negative sample with baseline model...")
baseline_neg_scores = score_negatives(baseline_model, baseline_predictor, baseline_feat, use_cn=False)
print("Scoring negative sample with fixed model...")
fixed_neg_scores = score_negatives(fixed_model, fixed_predictor, fixed_feat, use_cn=True)

def approx_rank(pos_score, neg_scores):
    return (neg_scores > pos_score).sum().item() + 1

print("Computing approximate ranks for each test pair (this may take a minute)...")
candidates = []
for i in range(test_edges.size(0)):
    b_rank = approx_rank(baseline_scores[i], baseline_neg_scores)
    f_rank = approx_rank(fixed_scores[i], fixed_neg_scores)
    if b_rank > 50 and f_rank <= 10:
        improvement = b_rank - f_rank
        candidates.append((i, test_edges[i, 0].item(), test_edges[i, 1].item(), b_rank, f_rank, improvement))

candidates.sort(key=lambda x: -x[5])
print(f"\\nFound {len(candidates)} candidate pairs (baseline rank > 50, fixed rank <= 10)")
print("\\nTop 10 examples (edge_idx, node_A, node_B, baseline_rank, fixed_rank, improvement):")
for c in candidates[:10]:
    print(c)

import json
with open("gnnexplainer_candidates.json", "w") as f:
    json.dump(candidates[:20], f)
print("\\nSaved top 20 candidates to gnnexplainer_candidates.json")
