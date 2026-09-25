import argparse
import math
import os
import dgl
from seed_setting_for_reproducibility import seed_torch
import random

import torch
import torch.nn.functional as F
from dgl.dataloading.negative_sampler import GlobalUniform
from dgl.nn.pytorch import GraphConv, SAGEConv, TAGConv, GINConv, APPNPConv
# from torch_geometric.nn import ARMAConv
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from torch.nn import Linear
from torch.utils.data import DataLoader
from biosnap_data_loader import BioSnapDDIDataset
from tdc_data_loader import TDCDDIDataset


from All_models import SAGE,SkipConnSAGE,GAT,GCN,SAGE_NGNN,GCNEncoder, SAGEEncoder, GATEncoder,GCNSkipConnection,SAGE_mlp
from models import MemAGDN,AGDN, DotPredictor, CosPredictor, LinkPredictor,MemoryAttentionGraph,ProposedModel


from TDCEvaluator import TDCDDIEvaluator
import networkx as nx
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch

import torch
from Logger import Logger
from datetime import datetime

# Get the current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class LinkPredictor(torch.nn.Module):
    def __init__(
            self, in_channels, hidden_channels, out_channels, num_layers, dropout, extra_feat_dim=0
    ):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels + extra_feat_dim, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.extra_feat_dim = extra_feat_dim

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, extra_feat=None):
        x = x_i * x_j
        if self.extra_feat_dim > 0 and extra_feat is not None:
            x = torch.cat([x, extra_feat], dim=-1)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, g, x, split_edge, optimizer, batch_size, dataset_name,
          loss_type="bce", focal_gamma=2.0, focal_alpha=0.25,
          asl_gamma_pos=0.0, asl_gamma_neg=4.0, asl_margin=0.05,
          num_neg_samples=1, adj=None):
    model.train()
    predictor.train()

    pos_train_edge = split_edge["train"]["edge"].to(x.device)
    neg_sampler = GlobalUniform(num_neg_samples)
    total_loss = total_examples = 0
    for perm in DataLoader(
            range(pos_train_edge.size(0)), batch_size, shuffle=True
    ):
        optimizer.zero_grad()

        h = model(g, x)

        edge = pos_train_edge[perm].t()

        cn_pos = None
        if adj is not None:
            cn_pos = (torch.log1p((adj[edge[0]] * adj[edge[1]]).sum(dim=1)) / 6.2).unsqueeze(-1) if edge[0].size(0) <= 200000 else None
            if cn_pos is None:
                # batch the CN lookup to avoid a huge intermediate tensor
                cn_chunks = []
                for start in range(0, edge[0].size(0), 50000):
                    e0 = edge[0][start:start+50000]
                    e1 = edge[1][start:start+50000]
                    cn_chunks.append(torch.log1p((adj[e0] * adj[e1]).sum(dim=1)) / 6.2)
                cn_pos = torch.cat(cn_chunks).unsqueeze(-1)
        pos_out = predictor(h[edge[0]], h[edge[1]], cn_pos)

        edge = neg_sampler(g, edge[0])

        cn_neg = None
        if adj is not None:
            cn_chunks = []
            for start in range(0, edge[0].size(0), 50000):
                e0 = edge[0][start:start+50000]
                e1 = edge[1][start:start+50000]
                cn_chunks.append(torch.log1p((adj[e0] * adj[e1]).sum(dim=1)) / 6.2)
            cn_neg = torch.cat(cn_chunks).unsqueeze(-1)
        neg_out = predictor(h[edge[0]], h[edge[1]], cn_neg)
        if num_neg_samples > 1:
            pos_out_expanded = pos_out.repeat_interleave(num_neg_samples, dim=0)
        else:
            pos_out_expanded = pos_out

        if loss_type == "asl":
            pos_loss = -((1 - pos_out) ** asl_gamma_pos) * torch.log(pos_out + 1e-15)
            neg_out_shifted = (neg_out - asl_margin).clamp(min=0)
            neg_loss = -((neg_out_shifted) ** asl_gamma_neg) * torch.log(1 - neg_out_shifted + 1e-15)
            pos_loss = pos_loss.mean()
            neg_loss = neg_loss.mean()
        elif loss_type == "focal":
            pos_loss = -focal_alpha * ((1 - pos_out) ** focal_gamma) * torch.log(pos_out + 1e-15)
            neg_loss = -(1 - focal_alpha) * (neg_out ** focal_gamma) * torch.log(1 - neg_out + 1e-15)
            pos_loss = pos_loss.mean()
            neg_loss = neg_loss.mean()
        else:
            # Original plain BCE (baseline — unchanged from the base paper's code)
            pos_loss = -torch.log(pos_out + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        if dataset_name == "ogbl-ddi":
            torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


def accuracy(pred, label):
    pred_rounded = torch.round(pred)
    accu = torch.eq(pred_rounded, label).sum() / label.shape[0]
    accu = round(accu.item(), 4)
    return accu


@torch.no_grad()
def test(model, predictor, g, x, split_edge, evaluator, batch_size, adj=None):
    model.eval()
    predictor.eval()

    h = model(g, x)

    def safe_get_edge(edge):
        return edge[0] if isinstance(edge, (tuple, list)) else edge

    pos_train_edge = split_edge["eval_train"]["edge"].to(h.device)
    pos_valid_edge = split_edge["valid"]["edge"].to(h.device)
    neg_valid_edge = safe_get_edge(split_edge["valid"]["edge_neg"]).to(h.device)
    pos_test_edge = split_edge["test"]["edge"].to(h.device)
    neg_test_edge = safe_get_edge(split_edge["test"]["edge_neg"]).to(h.device)

    def get_pred(test_edges, h):
        preds = []
        for perm in DataLoader(range(test_edges.size(0)), batch_size):
            edge = test_edges[perm].t()
            cn_feat = None
            if adj is not None:
                cn_feat = (torch.log1p((adj[edge[0]] * adj[edge[1]]).sum(dim=1)) / 6.2).unsqueeze(-1)
            pred = predictor(h[edge[0]], h[edge[1]], cn_feat).squeeze()
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)
            preds.append(pred.cpu())
        return torch.cat(preds, dim=0)

    pos_train_pred = get_pred(pos_train_edge, h)
    pos_valid_pred = get_pred(pos_valid_edge, h)
    neg_valid_pred = get_pred(neg_valid_edge, h)
    pos_test_pred = get_pred(pos_test_edge, h)
    neg_test_pred = get_pred(neg_test_edge, h)

    total_preds = torch.cat((pos_valid_pred, neg_valid_pred), dim=0)
    labels = torch.cat((torch.ones_like(pos_valid_pred), torch.zeros_like(neg_valid_pred)), dim=0)
    acc = accuracy(total_preds, labels)

    results = {}
    for K in [10, 20, 30, 40, 50]:
        evaluator.K = K
        results[f"Hits@{K}"] = (
            evaluator.eval({"y_pred_pos": pos_train_pred, "y_pred_neg": neg_valid_pred})[f"hits@{K}"],
            evaluator.eval({"y_pred_pos": pos_valid_pred, "y_pred_neg": neg_valid_pred})[f"hits@{K}"],
            evaluator.eval({"y_pred_pos": pos_test_pred, "y_pred_neg": neg_test_pred})[f"hits@{K}"],
            acc
        )
    #results["accuracy"]=acc

    return results, h


def to_dgl_graph(pyg_data):
    # Convert PyTorch Geometric data to DGL graph
    g = dgl.graph((pyg_data.edge_index[0], pyg_data.edge_index[1]))
    g.ndata['feat'] = pyg_data.x
    g.edata['label'] = pyg_data.edge_attr
    return g


from dgl.dataloading.negative_sampler import GlobalUniform


def build_split_edge(data, num_neg_samples=1):
    # Convert PyTorch Geometric data to DGL graph
    g = to_dgl_graph(data)

    # Extract positive edges
    pos_edges = data.edge_index.t()

    # Generate negative edges
    neg_sampler = GlobalUniform(k=num_neg_samples)
    neg_edges = neg_sampler(g, pos_edges)

    return {
        'edge': pos_edges,
        'edge_neg': neg_edges
    }


def build_split_edge_full(data_all, train_idx, val_idx, test_idx, num_neg_samples=1):
    g_full = to_dgl_graph(data_all)
    all_pos_edges = data_all.edge_index.t()

    split_edge = {}

    for split, idx in zip(['train', 'valid', 'test'], [train_idx, val_idx, test_idx]):
        pos_edges = all_pos_edges[idx]

        # Use full graph for negative sampling
        neg_sampler = GlobalUniform(k=num_neg_samples)
        neg_edges = neg_sampler(g_full, pos_edges)

        split_edge[split] = {
            'edge': pos_edges,
            'edge_neg': neg_edges
        }

    return split_edge


def main():
    parser = argparse.ArgumentParser(
        description="OGBL,Drugbank,biosnap(Full Batch GCN/GraphSage + NGNN)"
    )

    # dataset setting
    parser.add_argument(
        "--dataset",
        type=str,
        default="biosnapddi",
        choices=["ogbl-ddi", "drugbankddi","biosnapddi"],
    )

    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="GPU device ID. Use -1 for CPU training.",
    )
    parser.add_argument(
        "--model",
        # action="store_true",
        type=str,
        default='gcn_skipconnection',
        choices=["gcn_ngnn_graphconv", "sage_ngnn", "sage_skipconnection", "gcn", "sage", "gat_mlp", "gcn_skipconnection", "sage_nodefeature_mlp", "sage_mlp_virtual_node","magcn"],
        help="If not set, use GCN by default."
    )
    parser.add_argument(
        "--use_sage",
        # action="store_true",
        type=bool,
        default=False,
        help="If not set, use GCN by default.",
    )
    parser.add_argument(
        "--ngnn_type",
        type=str,
        default="input",
        choices=["input", "hidden"],
        help="You can set this value from 'input' or 'hidden' to apply NGNN to different GNN layers.",
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="number of GNN layers"
    )
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64 * 1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    ######
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--n-hidden', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=1)

    parser.add_argument('--input-drop', type=float, default=0.)
    parser.add_argument('--edge-drop', type=float, default=0.)
    parser.add_argument('--attn-drop', type=float, default=0.)
    parser.add_argument('--diffusion-drop', type=float, default=0.)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--output-bn', action='store_true')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--no-dst-attn', action='store_true')
    parser.add_argument('--transition-matrix', type=str, default='gat')
    parser.add_argument('--hop-norm', action='store_true')
    parser.add_argument('--weight-style', type=str, default='HA',
                        choices=['HC', 'HA', 'HA+HC', 'HA1', 'sum', 'max_pool', 'mean_pool', 'lstm'])
    parser.add_argument('--no-pos-emb', action='store_true')
    parser.add_argument('--no-share-weights', action='store_true')
    parser.add_argument('--pre-act', action='store_true')
    # training settings
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    # HitFix Phase 3 — Fix B: focal loss (toggle-able so it can be run as an isolated experiment)
    parser.add_argument("--loss_type", type=str, default="bce", choices=["bce", "focal", "asl"],
                         help="bce = original loss (baseline), focal = focal loss (Fix B)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                         help="focal loss focusing parameter (higher = more focus on hard examples)")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                         help="focal loss balancing parameter for the positive class")
    parser.add_argument("--asl_gamma_pos", type=float, default=0.0,
                         help="ASL focusing parameter for positives (usually low/zero)")
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0,
                         help="ASL focusing parameter for negatives (usually higher than gamma_pos)")
    parser.add_argument("--asl_margin", type=float, default=0.05,
                         help="ASL probability margin — shifts easy-negative loss contribution toward zero")
    parser.add_argument("--num_neg_samples", type=int, default=1,
                         help="number of negative edges sampled per positive edge during training (default 1, matches base paper)")
    parser.add_argument("--use_common_neighbors", action="store_true",
                         help="augment the link predictor with a common-neighbor-count structural feature")
    args = parser.parse_args()
    print(args)

    device = (
        f"cuda:{args.device}"
        if args.device != -1 and torch.cuda.is_available()
        else "cpu"
    )
    device = torch.device(device)

    all_models=[args.model]
    all_data = [args.dataset]

    # Main experiment loop
    for dataset_name in all_data:

        print(f'\n\n=== Processing dataset: {args.dataset} ===')

        # Load dataset once per dataset iteration
        if args.dataset == 'ogbl-ddi':
            dataset = DglLinkPropPredDataset(name=args.dataset, root='../dataset')
            g = dataset[0]
            split_edge = dataset.get_edge_split()
            idx = torch.randperm(split_edge["train"]["edge"].size(0))
            idx = idx[: split_edge["valid"]["edge"].size(0)]
            split_edge["eval_train"] = {"edge": split_edge["train"]["edge"][idx]}
            evaluator = Evaluator(name=dataset.name)

        elif args.dataset == 'biosnapddi' or args.dataset == 'drugbankddi':
            dataset = BioSnapDDIDataset(name=args.dataset)
            split_edge = dataset.get_edge_split()
            g = to_dgl_graph(dataset.get(0))
            idx = torch.randperm(split_edge["train"]["edge"].size(0))
            idx = idx[: split_edge["valid"]["edge"].size(0)]
            split_edge["eval_train"] = {"edge": split_edge["train"]["edge"][idx]}
            evaluator = TDCDDIEvaluator(name='TDC_DDI')

        # === HitFix: common-neighbor structural feature (built once per dataset) ===
        adj = None
        if args.use_common_neighbors:
            num_nodes = g.num_nodes()
            adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
            train_edges_cn = split_edge["train"]["edge"].to(device)
            adj[train_edges_cn[:, 0], train_edges_cn[:, 1]] = 1.0
            adj[train_edges_cn[:, 1], train_edges_cn[:, 0]] = 1.0
        # === END ===

        if dataset.name == "ogbl-ddi":
            emb = torch.nn.Embedding(g.num_nodes(), args.hidden_channels).to(device)
            in_channels = args.hidden_channels
        else:
            emb = torch.nn.Embedding(g.num_nodes(), args.hidden_channels).to(device)
            in_channels = g.ndata["feat"].size(-1) if "feat" in g.ndata else args.hidden_channels

        # Model loop
        for model_name in all_models:
            print(f'\n=== Running model: {model_name} on {args.dataset} ===')
            test_file_result = f"../results/test/{args.dataset}_{model_name}_results_{current_datetime}.csv"

            # Model selection
            augment_embedding = None
            if model_name == 'sage_ngnn':
                model = SAGE_NGNN(
                    in_channels, args.hidden_channels, args.hidden_channels,
                    args.num_layers, args.dropout, args.ngnn_type, dataset.name)
            elif model_name == 'gcn_ngnn_graphconv':
                g = dgl.add_self_loop(g)
                model = GCN(
                    in_channels, args.hidden_channels, args.hidden_channels,
                    args.num_layers, args.dropout, args.ngnn_type, dataset.name)
            elif model_name == 'sage_skipconnection':
                model = SkipConnSAGE(in_channels, args.hidden_channels, args.hidden_channels,
                                     args.num_layers, 0.5).to(device)
            elif model_name == 'sage':
                g = dgl.add_self_loop(g)
                model = SAGEEncoder(in_channels, args.hidden_channels, args.hidden_channels).to(device)
            elif model_name == 'gcn':
                g = dgl.add_self_loop(g)
                model = GCNEncoder(in_channels, args.hidden_channels, args.hidden_channels).to(device)
            elif model_name == 'gcn_skipconnection':
                g = dgl.add_self_loop(g)
                model = GCNSkipConnection(in_channels, args.hidden_channels, args.hidden_channels,
                                          num_layers=3, dropout=0.5).to(device)
            elif model_name == 'gat_mlp':
                g = dgl.add_self_loop(g)
                model = GATEncoder(in_channels=in_channels, hidden_channels=args.hidden_channels,
                                   out_channels=args.hidden_channels, num_layers=3, num_heads=2, dropout=0.5)
            elif model_name == 'sage_mlp_virtual_node':
                g = add_virtual_node_dgle_graph(g)
                model = SAGE_mlp(in_channels, args.hidden_channels, args.hidden_channels).to(device)
            elif model_name == 'sage_nodefeature_mlp':
                g_nx = dgl.to_networkx(g).to_undirected()
                g_nx_simple = nx.Graph(g_nx)
                pagerank = nx.pagerank(g_nx_simple)
                clustering_coef = nx.clustering(g_nx_simple)
                betweeness_centrality = nx.betweenness_centrality(g_nx_simple, k=50)
                degree = dict(g_nx_simple.degree())
                num_nodes = g_nx_simple.number_of_nodes()
                aug_emb = torch.ones(num_nodes, 5, dtype=torch.float64).to(device)
                for i in range(num_nodes):
                    aug_emb[i][0] = degree[i]
                    aug_emb[i][1] = pagerank[i]
                    aug_emb[i][2] = betweeness_centrality[i]
                    aug_emb[i][3] = clustering_coef[i]
                    aug_emb[i][4] = 1.0
                augment_embedding = aug_emb.float()
                model = SAGE_mlp(5, args.hidden_channels, args.hidden_channels).to(device)
                ####This is from other model

            if model_name == 'magcn':
                g = dgl.add_self_loop(g)
                model = MemoryAttentionGraph(in_channels, args.hidden_channels,
                                             args.hidden_channels, 3,
                                             args.n_heads, args.K,
                                             args.dropout, args.input_drop, args.attn_drop,
                                             in_edge_feats=0.0).to(device)




            # Initialize predictor and move to device
            predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1, 3, args.dropout,
                                       extra_feat_dim=(1 if args.use_common_neighbors else 0))
            g, model, predictor = map(lambda x: x.to(device), (g, model, predictor))

            # Initialize loggers for each model-dataset combination
            loggers = {
                "Hits@10": Logger(args.runs, args),
                "Hits@20": Logger(args.runs, args),
                "Hits@30": Logger(args.runs, args),
                "Hits@40": Logger(args.runs, args),
                "Hits@50": Logger(args.runs, args),
            }

            # Training loop
            for run in range(args.runs):
                best_valid_hits20 = -1  # PHASE 2 DIAGNOSIS: track best epoch per run
                random_seed = random.randint(1, 2 ** 6 - 1)
                seed_torch(random_seed)
                model.reset_parameters()
                predictor.reset_parameters()
                if dataset.name == "ogbl-ddi":
                    torch.nn.init.xavier_uniform_(emb.weight)
                    g.ndata["feat"] = emb.weight
                if augment_embedding is not None:
                    g.ndata["feat"] = augment_embedding

                optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(predictor.parameters()) +
                    (list(emb.parameters()) if dataset.name == "ogbl-ddi" else []),
                    lr=args.lr
                )

                for epoch in range(1, 1 + args.epochs):
                    loss = train(model, predictor, g, g.ndata["feat"], split_edge, optimizer, args.batch_size,
                                 args.dataset, loss_type=args.loss_type,
                                 focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
                                 asl_gamma_pos=args.asl_gamma_pos, asl_gamma_neg=args.asl_gamma_neg,
                                 asl_margin=args.asl_margin, num_neg_samples=args.num_neg_samples, adj=adj)

                    if epoch % args.eval_steps == 0:
                        results, h_eval = test(model, predictor, g, g.ndata["feat"], split_edge, evaluator, args.batch_size, adj=adj)

                        # === PHASE 2 DIAGNOSIS: save embeddings + pos/neg pair features at best-valid-Hits@20 epoch ===
                        valid_hits20 = results["Hits@20"][1]
                        if valid_hits20 > best_valid_hits20:
                            best_valid_hits20 = valid_hits20
                            emb_dir = f"../embeddings/{dataset.name}_{model_name}"
                            os.makedirs(emb_dir, exist_ok=True)

                            pos_edge = split_edge["test"]["edge"]
                            neg_edge = split_edge["test"]["edge_neg"]
                            neg_edge = neg_edge[0] if isinstance(neg_edge, (tuple, list)) else neg_edge

                            n_sample = 3000
                            pos_idx = torch.randperm(pos_edge.size(0))[:n_sample]
                            neg_idx = torch.randperm(neg_edge.size(0))[:n_sample]

                            pos_pairs = pos_edge[pos_idx]
                            neg_pairs = neg_edge[neg_idx]
                            pos_feat = (h_eval[pos_pairs[:, 0]] * h_eval[pos_pairs[:, 1]]).detach().cpu()
                            neg_feat = (h_eval[neg_pairs[:, 0]] * h_eval[neg_pairs[:, 1]]).detach().cpu()

                            torch.save({
                                "epoch": epoch,
                                "valid_hits20": valid_hits20,
                                "node_embeddings": h_eval.detach().cpu(),
                                "pos_pair_features": pos_feat,
                                "neg_pair_features": neg_feat,
                            }, f"{emb_dir}/run{run}_best_embeddings.pt")

                            # === PHASE 6: also save the actual trained model + predictor for GNNExplainer ===
                            torch.save({
                                "epoch": epoch,
                                "valid_hits20": valid_hits20,
                                "model_state": model.state_dict(),
                                "predictor_state": predictor.state_dict(),
                            }, f"{emb_dir}/run{run}_best_model.pt")
                            # === END PHASE 6 ===
                        # === END PHASE 2 DIAGNOSIS ===

                        log_file_path = f"../results/training/run{run}_{dataset.name}_{model_name}_train_validation.csv"
                        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

                        with open(log_file_path, "a") as log_file:
                            for key, result in results.items():
                                loggers[key].add_result(run, result)
                                train_hits, valid_hits, test_hits, accuracy = result
                                log_message = (
                                    f"Run: {run + 1:02d}, "
                                    f"Epoch: {epoch:02d}, "
                                    f"Loss: {loss:.4f}, "
                                    f"Train: {100 * train_hits:.2f}%, "
                                    f"Valid: {100 * valid_hits:.2f}%, "
                                    f"Test: {100 * test_hits:.2f}%, "
                                    f"accuracy: {100 * accuracy:.4f}, "
                                    f"Key: {key}\n"
                                )
                                print(log_message.strip())
                                log_file.write(log_message)
                            log_file.write("---\n")
                        print("---")

            # Save results for this model-dataset combination
            os.makedirs(os.path.dirname(test_file_result), exist_ok=True)
            with open(test_file_result, "w") as summary_file:
                for key in loggers.keys():
                    print(key)
                    summary_file.write(f"Key: {key}\n")
                    loggers[key].print_statistics(file=summary_file)
                    summary_file.write("\n---\n")

def add_virtual_node_dgle_graph(G):
    """Adds a virtual node connected to all existing nodes in a DGL graph."""
    num_nodes = G.number_of_nodes()
    virtual_node_id = num_nodes

    # Add 1 new node
    G.add_nodes(1)

    # Get all existing node IDs (excluding the new virtual node)
    existing_nodes = torch.arange(num_nodes)

    # Connect virtual node to all existing nodes (bidirectional)
    src = torch.cat([torch.full((num_nodes,), virtual_node_id), existing_nodes])
    dst = torch.cat([existing_nodes, torch.full((num_nodes,), virtual_node_id)])
    G.add_edges(src, dst)
    return G


if __name__ == "__main__":
    main()
