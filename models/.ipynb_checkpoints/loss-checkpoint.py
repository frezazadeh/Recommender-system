import torch
from torch import Tensor
import numpy as np
from models.gnn import Model
from torch_geometric.data import HeteroData
import random

def bpr_loss(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    """
    BPR loss = -log( sigmoid(pos_score - neg_score) ).
    """
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

def evaluate_bpr_loss(model: Model, full_data: HeteroData, 
                      pos_edge_index: torch.Tensor, 
                      device: torch.device, 
                      num_samples: int = 10000) -> float:
    """
    Compute a BPR-like 'test loss':
      1) sample up to 'num_samples' positive edges from pos_edge_index
      2) sample a negative item for each user
      3) compute BPR loss on these pairs
    """
    model.eval()
    x_dict = model(full_data)
    user_embs = x_dict['user']
    movie_embs = x_dict['movie']

    # Build adjacency for test/val if needed
    user_pos_map_eval = [[] for _ in range(user_embs.size(0))]
    all_user = pos_edge_index[0].tolist()
    all_item = pos_edge_index[1].tolist()
    for u, it in zip(all_user, all_item):
        user_pos_map_eval[u].append(it)

    num_pos = pos_edge_index.size(1)
    chosen = np.random.choice(num_pos, size=min(num_samples, num_pos), replace=False)
    pos_user = pos_edge_index[0, chosen]
    pos_item = pos_edge_index[1, chosen]

    neg_item = []
    for u in pos_user.tolist():
        pos_items = user_pos_map_eval[u]
        while True:
            rand_item = random.randint(0, movie_embs.size(0)-1)
            if rand_item not in pos_items:
                neg_item.append(rand_item)
                break

    pos_user = pos_user.to(device)
    pos_item = pos_item.to(device)
    neg_item = torch.tensor(neg_item, dtype=torch.long, device=device)

    with torch.no_grad():
        pos_scores = model.score(user_embs[pos_user], movie_embs[pos_item])
        neg_scores = model.score(user_embs[pos_user], movie_embs[neg_item])
        loss_val = bpr_loss(pos_scores, neg_scores).item()

    return loss_val
