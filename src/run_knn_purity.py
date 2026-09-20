
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

d = torch.load("../embeddings/ogbl-ddi_gcn_skipconnection/run0_best_embeddings.pt")
pos = d["pos_pair_features"].numpy()
neg = d["neg_pair_features"].numpy()

X = np.vstack([pos, neg])
y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

k = 20
nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
_, indices = nn.kneighbors(X)

purities = []
for i in range(len(X)):
    neighbor_idx = indices[i, 1:]
    same_label_frac = (y[neighbor_idx] == y[i]).mean()
    purities.append(same_label_frac)

purities = np.array(purities)
pos_purity = purities[y == 1].mean()
neg_purity = purities[y == 0].mean()
overall_purity = purities.mean()

print(f"k = {k} nearest neighbors")
print(f"Positive-pair purity: {pos_purity:.3f}")
print(f"Negative-pair purity: {neg_purity:.3f}")
print(f"Overall purity: {overall_purity:.3f}")
print(f"(0.50 = random mixing, 1.00 = perfect separation)")

with open("knn_purity_results.txt", "w") as f:
    f.write(f"k={k}\npositive_purity={pos_purity:.4f}\nnegative_purity={neg_purity:.4f}\noverall_purity={overall_purity:.4f}\n")
