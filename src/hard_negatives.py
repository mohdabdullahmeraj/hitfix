"""
Chemically-informed hard-negative sampling (Tanimoto similarity over Morgan fingerprints).

For a training positive (u, v), a hard negative keeps one drug and swaps the other
for a drug that is chemically similar to it but is not a known interaction partner,
e.g. (u, w) where w is among the top-M most Tanimoto-similar drugs to v.

A fraction `hard_ratio` of negatives is drawn this way; the rest are uniform random
pairs, as in the baseline. Mixing keeps some easy negatives in every batch and limits
the damage from false negatives (similar drugs that truly interact but are unlabelled).

Requires SMILES, so it is available for BIOSNAP-DDI and DrugBank-DDI only.
OGB-DDI ships without molecular structure.
"""

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")


def morgan_matrix(smiles, radius=2, nbits=2048):
    """(N, nbits) uint8 Morgan fingerprint matrix. Unparseable SMILES get an all-zero row."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    out = np.zeros((len(smiles), nbits), dtype=np.uint8)
    n_bad = 0
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            n_bad += 1
            continue
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), out[i])
    return out, n_bad


@torch.no_grad()
def topk_tanimoto(fp, k, device, block=2048):
    """Top-k most similar drugs (excluding self) for every drug, computed in blocks."""
    x = torch.from_numpy(fp).float().to(device)
    cnt = x.sum(1)
    n = x.size(0)
    idx_out = torch.empty((n, k), dtype=torch.long)
    sim_out = torch.empty((n, k))
    for s in range(0, n, block):
        e = min(s + block, n)
        inter = x[s:e] @ x.T
        union = cnt[s:e, None] + cnt[None, :] - inter
        sim = torch.where(union > 0, inter / union.clamp(min=1), torch.zeros_like(inter))
        sim[torch.arange(e - s), torch.arange(s, e)] = -1.0  # exclude self
        v, i = sim.topk(k, dim=1)
        idx_out[s:e], sim_out[s:e] = i.cpu(), v.cpu()
    return idx_out, sim_out


def pair_tanimoto(fp, a, b):
    """Tanimoto similarity for given index pairs (numpy, for diagnostics)."""
    x = fp.astype(bool)
    inter = (x[a] & x[b]).sum(1)
    union = (x[a] | x[b]).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


class TanimotoHardNegativeSampler:
    def __init__(self, smiles, train_pos, num_nodes, device, top_m=50, hard_ratio=0.5,
                 radius=2, nbits=2048, verbose=True):
        self.n = num_nodes
        self.device = device
        self.hard_ratio = hard_ratio
        self.top_m = top_m

        fp, n_bad = morgan_matrix(smiles, radius, nbits)
        self.fp = fp
        cand, cand_sim = topk_tanimoto(fp, top_m, device)
        self.cand = cand.to(device)

        # known positives, both directions, as sorted keys u*N+v for fast membership tests
        tp = train_pos.to(device)
        keys = torch.cat([tp[:, 0] * num_nodes + tp[:, 1], tp[:, 1] * num_nodes + tp[:, 0]])
        self.pos_keys = torch.unique(keys)

        if verbose:
            rng = np.random.default_rng(0)
            ra, rb = rng.integers(0, num_nodes, 20000), rng.integers(0, num_nodes, 20000)
            print(f"[TanimotoHardNeg] drugs={num_nodes} unparseable_smiles={n_bad} top_m={top_m} hard_ratio={hard_ratio}")
            print(f"  mean Tanimoto, drug vs its top-{top_m} neighbours: {cand_sim.mean():.3f}"
                  f"  | random drug pairs: {pair_tanimoto(fp, ra, rb).mean():.3f}")

    def _is_pos(self, a, b):
        return torch.isin(a * self.n + b, self.pos_keys)

    def __call__(self, g, src, dst, k=1):
        """Return (neg_src, neg_dst), k negatives per positive (len(src)*k in total)."""
        src = src.repeat_interleave(k)
        dst = dst.repeat_interleave(k)
        b = src.size(0)
        dev = src.device

        # uniform random pairs (baseline-style)
        ns = torch.randint(0, self.n, (b,), device=dev)
        nd = torch.randint(0, self.n, (b,), device=dev)

        # hard negatives: keep one drug, swap the other for a chemically similar drug
        hard = torch.rand(b, device=dev) < self.hard_ratio
        swap_dst = torch.rand(b, device=dev) < 0.5
        pick = torch.randint(0, self.top_m, (b,), device=dev)
        sim_of_dst = self.cand[dst, pick]
        sim_of_src = self.cand[src, pick]
        hs = torch.where(swap_dst, src, sim_of_src)
        hd = torch.where(swap_dst, sim_of_dst, dst)
        ns = torch.where(hard, hs, ns)
        nd = torch.where(hard, hd, nd)

        # reject self-pairs and known positives; replace with fresh random pairs (two tries)
        for _ in range(2):
            bad = (ns == nd) | self._is_pos(ns, nd)
            if not bad.any():
                break
            nb = int(bad.sum())
            ns[bad] = torch.randint(0, self.n, (nb,), device=dev)
            nd[bad] = torch.randint(0, self.n, (nb,), device=dev)
        keep = ~((ns == nd) | self._is_pos(ns, nd))
        return ns[keep], nd[keep]
