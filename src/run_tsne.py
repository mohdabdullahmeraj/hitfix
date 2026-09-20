
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load what the patch saved
d = torch.load("../embeddings/ogbl-ddi_gcn_skipconnection/run0_best_embeddings.pt")
pos = d["pos_pair_features"].numpy()
neg = d["neg_pair_features"].numpy()

print("Loaded:", pos.shape[0], "positive pairs,", neg.shape[0], "negative pairs")

# Stack them together with labels: 1 = real interaction, 0 = fake
X = np.vstack([pos, neg])
y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

print("Running t-SNE on", X.shape[0], "points (256 dims -> 2 dims)... this can take a few minutes")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", verbose=1)
X_2d = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(8, 8))
plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], s=6, alpha=0.5, label="Negative (fake) pairs", color="tab:red")
plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], s=6, alpha=0.5, label="Positive (real) pairs", color="tab:blue")
plt.legend()
plt.title("t-SNE of pair features (h_i * h_j)\nGCN-SkipConnection on OGB-DDI, best epoch (99)")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.tight_layout()
plt.savefig("tsne_pos_neg_pairs.png", dpi=150)
print("Saved plot to tsne_pair_features.png")
