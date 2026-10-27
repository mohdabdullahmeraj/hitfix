"""
Corrected loader for the BIOSNAP-DDI and DrugBank-DDI CSV datasets.

The reference loader (biosnap_data_loader.BioSnapDDIDataset) has three problems:

1. Train and test files get separate drug numberings, so node i in the training
   graph and node i in the test pairs are usually different drugs. Test pairs are
   then effectively random.
2. Every CSV row (including label-0 "non-interactions") becomes a message-passing
   edge, and the validation split is carved out after the graph is built, so
   validation positives also leak into the graph.
3. Pairs are treated as directed. In DrugBank-DDI many label-0 rows are a real
   interaction with the two drugs swapped (A-B labelled 1, B-A labelled 0), so
   some "negatives" are actually true interactions.

This loader fixes all three:
- one shared, deterministic drug numbering across train, validation and test;
- pairs are unordered; a pair labelled 1 anywhere is a positive, and any
  negative that is a known positive is dropped;
- self-pairs and duplicates are removed; test pairs that also appear in the
  training file are removed from the test set;
- the message-passing graph contains only training-split positives (undirected).

The returned split_edge has the same structure as the reference loader, so the
rest of main.py works unchanged.
"""

import dgl
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


class FixedDDIDataset:
    def __init__(self, name="biosnapddi", path="../dataset", valid_frac=0.2, seed=42, verbose=True,
                 eval_neg="provided"):
        self.name = name
        train_df = pd.read_csv(f"{path}/{name}/raw/train.csv")
        test_df = pd.read_csv(f"{path}/{name}/raw/test.csv")

        # 1) one shared, deterministic drug numbering (sorted, so it never depends on hash order)
        smiles = sorted(set(train_df.smile1) | set(train_df.smile2) | set(test_df.smile1) | set(test_df.smile2))
        self.smiles = smiles
        self.smiles_to_index = {s: i for i, s in enumerate(smiles)}
        self.num_nodes = len(smiles)

        tr = self._to_pairs(train_df)
        te = self._to_pairs(test_df)
        self.stats = {"raw_train_rows": len(train_df), "raw_test_rows": len(test_df)}

        # 2) unordered pairs; a pair labelled 1 anywhere is a positive
        tr_pos = set(map(tuple, tr[tr[:, 2] == 1, :2]))
        tr_neg = set(map(tuple, tr[tr[:, 2] == 0, :2]))
        te_pos = set(map(tuple, te[te[:, 2] == 1, :2]))
        te_neg = set(map(tuple, te[te[:, 2] == 0, :2]))
        all_pos = tr_pos | te_pos
        train_pairs = tr_pos | tr_neg

        clean_tr_neg = tr_neg - all_pos                      # drop "negatives" that are real interactions
        clean_te_pos = te_pos - train_pairs                  # test must be unseen in training file
        clean_te_neg = te_neg - all_pos - train_pairs

        self.stats.update({
            "train_neg_dropped_as_true_interactions": len(tr_neg & all_pos),
            "test_pos_dropped_seen_in_train": len(te_pos & train_pairs),
            "test_neg_dropped": len(te_neg) - len(clean_te_neg),
        })

        to_t = lambda s: torch.tensor(sorted(s), dtype=torch.long).reshape(-1, 2)
        train_pos = to_t(tr_pos)
        train_neg = to_t(clean_tr_neg)
        test_pos = to_t(clean_te_pos)
        test_neg = to_t(clean_te_neg)

        # 3) validation split from the training file (same 80/20, seed 42 as the reference loader)
        tp, vp = train_test_split(train_pos.numpy(), test_size=valid_frac, random_state=seed)
        tn, vn = train_test_split(train_neg.numpy(), test_size=valid_frac, random_state=seed)
        self.split_edge = {
            "train": {"edge": torch.from_numpy(tp), "edge_neg": torch.from_numpy(tn)},
            "valid": {"edge": torch.from_numpy(vp), "edge_neg": torch.from_numpy(vn)},
            "test": {"edge": test_pos, "edge_neg": test_neg},
        }

        # 3b) optional fair evaluation negatives. The provided BIOSNAP negatives almost always contain a
        # drug with no training interactions, so "unseen drug => no interaction" scores well without
        # learning anything. "fair" redraws validation/test negatives from drugs that have training
        # interactions, excluding every known positive; same number of negatives as provided.
        self.stats["eval_neg"] = eval_neg
        if eval_neg == "fair":
            known = all_pos
            tr_e = self.split_edge["train"]["edge"]
            active = torch.unique(tr_e.flatten()).numpy()
            rng = np.random.default_rng(seed)
            used = set()
            for split in ["valid", "test"]:
                need = self.split_edge[split]["edge_neg"].size(0)
                out = []
                while len(out) < need:
                    a = rng.choice(active, size=2 * need)
                    b = rng.choice(active, size=2 * need)
                    for x, y in zip(a, b):
                        if x == y:
                            continue
                        p = (int(min(x, y)), int(max(x, y)))
                        if p in known or p in used:
                            continue
                        used.add(p)
                        out.append(p)
                        if len(out) == need:
                            break
                self.split_edge[split]["edge_neg"] = torch.tensor(out, dtype=torch.long)
        elif eval_neg != "provided":
            raise ValueError("eval_neg must be 'provided' or 'fair'")

        # 4) message-passing graph: training-split positives only, undirected
        src = torch.cat([self.split_edge["train"]["edge"][:, 0], self.split_edge["train"]["edge"][:, 1]])
        dst = torch.cat([self.split_edge["train"]["edge"][:, 1], self.split_edge["train"]["edge"][:, 0]])
        g = dgl.graph((src, dst), num_nodes=self.num_nodes)
        g.ndata["feat"] = torch.eye(self.num_nodes)  # same one-hot node input as the reference loader
        self.graph = g

        if verbose:
            self.report()

    def _to_pairs(self, df):
        """Map rows to (min_idx, max_idx, label), dropping self-pairs and exact duplicates."""
        a = df.smile1.map(self.smiles_to_index).to_numpy()
        b = df.smile2.map(self.smiles_to_index).to_numpy()
        y = df.label.astype(int).to_numpy()
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        arr = np.stack([lo, hi, y], axis=1)
        arr = arr[lo != hi]
        return np.unique(arr, axis=0)

    def get_edge_split(self):
        return self.split_edge

    def report(self):
        se = self.split_edge
        print(f"[FixedDDIDataset:{self.name}] drugs={self.num_nodes} graph_edges(undirected)={se['train']['edge'].size(0)}")
        for k in ["train", "valid", "test"]:
            print(f"  {k:5s}: pos={se[k]['edge'].size(0):7d}  neg={se[k]['edge_neg'].size(0):7d}")
        for k, v in self.stats.items():
            print(f"  {k}: {v}")
