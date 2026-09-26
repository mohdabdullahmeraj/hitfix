
import torch
import json
import sys
sys.path.insert(0, ".")

from All_models import GCNSkipConnection
from main import LinkPredictor
import dgl
from ogb.linkproppred import DglLinkPropPredDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DglLinkPropPredDataset(name="ogbl-ddi", root="../dataset")
split_edge = dataset.get_edge_split()
g_full = dataset[0].to(device)
g_full = dgl.add_self_loop(g_full)

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

baseline_model, baseline_predictor, baseline_feat = load_model(
    "/content/drive/MyDrive/HitFix_Research_Project/06_interpretability/checkpoints/baseline_best_model.pt", False)
fixed_model, fixed_predictor, fixed_feat = load_model(
    "/content/drive/MyDrive/HitFix_Research_Project/06_interpretability/checkpoints/fixed_best_model.pt", True)

num_nodes = g_full.num_nodes()
adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
train_edges = split_edge["train"]["edge"].to(device)
adj[train_edges[:, 0], train_edges[:, 1]] = 1.0
adj[train_edges[:, 1], train_edges[:, 0]] = 1.0

@torch.no_grad()
def score_pair(model, predictor, feat, g, node_a, node_b, use_cn):
    h = model(g, feat)
    a = torch.tensor([node_a], device=device)
    b = torch.tensor([node_b], device=device)
    cn = None
    if use_cn:
        cn = (torch.log1p((adj[a] * adj[b]).sum(dim=1)) / 6.2).unsqueeze(-1)
    return predictor(h[a], h[b], cn).item()

def get_local_edges(g, node_ids, max_edges=150):
    src, dst = g.edges()
    mask = torch.zeros(src.size(0), dtype=torch.bool, device=device)
    for n in node_ids:
        mask |= (src == n) | (dst == n)
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.size(0) > max_edges:
        idx = idx[torch.randperm(idx.size(0))[:max_edges]]
    return idx, src[idx], dst[idx]

def explain_pair(model, predictor, feat, node_a, node_b, use_cn, label):
    print(f"\n=== Explaining ({node_a}, {node_b}) with {label} model ===")
    base_score = score_pair(model, predictor, feat, g_full, node_a, node_b, use_cn)
    print(f"Full-graph prediction score: {base_score:.4f}")

    edge_idx, src, dst = get_local_edges(g_full, [node_a, node_b])
    print(f"Testing {edge_idx.size(0)} local edges for importance...")

    importances = []
    for i in range(edge_idx.size(0)):
        eids_to_remove = torch.tensor([edge_idx[i].item()], device=device)
        g_perturbed = dgl.remove_edges(g_full, eids_to_remove)
        perturbed_score = score_pair(model, predictor, feat, g_perturbed, node_a, node_b, use_cn)
        importance = base_score - perturbed_score
        importances.append((src[i].item(), dst[i].item(), importance))

    importances.sort(key=lambda x: -x[2])
    print(f"Top 10 most important edges (src, dst, importance = score drop when removed):")
    for imp in importances[:10]:
        print(imp)

    return base_score, importances

with open("gnnexplainer_candidates.json") as f:
    candidates = json.load(f)

chosen = [candidates[0], candidates[5], candidates[3]]

all_results = []
for edge_idx, node_a, node_b, b_rank, f_rank, improvement in chosen:
    print(f"\n{'='*60}")
    print(f"PAIR: nodes ({node_a}, {node_b}) | baseline_rank={b_rank}, fixed_rank={f_rank}")
    print(f"{'='*60}")

    b_score, b_importances = explain_pair(baseline_model, baseline_predictor, baseline_feat, node_a, node_b, False, "BASELINE")
    f_score, f_importances = explain_pair(fixed_model, fixed_predictor, fixed_feat, node_a, node_b, True, "FIXED")

    all_results.append({
        "node_a": node_a, "node_b": node_b,
        "baseline_rank": b_rank, "fixed_rank": f_rank,
        "baseline_score": b_score, "fixed_score": f_score,
        "baseline_top_edges": b_importances[:10],
        "fixed_top_edges": f_importances[:10],
    })

with open("gnnexplainer_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved full explanation results to gnnexplainer_results.json")
