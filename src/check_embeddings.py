
import torch
d = torch.load('../embeddings/ogbl-ddi_gcn_skipconnection/run0_best_embeddings.pt')
print('Saved at epoch:', d['epoch'])
print('Best valid Hits@20 at save time:', d['valid_hits20'])
print('Node embeddings shape:', d['node_embeddings'].shape)
print('Positive pair features shape:', d['pos_pair_features'].shape)
print('Negative pair features shape:', d['neg_pair_features'].shape)
