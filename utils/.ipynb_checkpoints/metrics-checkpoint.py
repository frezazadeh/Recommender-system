import torch
import random
import numpy as np
from models.gnn import Model
from torch_geometric.data import HeteroData



user_pos_items_map_eval = None
def evaluate_auc(model: Model, full_data: HeteroData, 
                 pos_edge_index: torch.Tensor, 
                 num_samples: int = 10000) -> float:
    """
    Approximate AUC: For a sampled set of positive edges (u, i),
    sample a negative item j, compare pos_scores vs neg_scores.
    """
    model.eval()
    x_dict = model(full_data)  # Full-batch embeddings
    user_embs = x_dict['user']
    movie_embs = x_dict['movie']

    global user_pos_items_map_eval
    if user_pos_items_map_eval is None:
        # Build adjacency map for evaluation
        user_pos_items_map_eval = [[] for _ in range(user_embs.size(0))]
        all_user = pos_edge_index[0].tolist()
        all_item = pos_edge_index[1].tolist()
        for u, it in zip(all_user, all_item):
            user_pos_items_map_eval[u].append(it)

    num_pos = pos_edge_index.size(1)
    chosen = np.random.choice(num_pos, size=min(num_samples, num_pos), replace=False)
    pos_user = pos_edge_index[0, chosen]
    pos_item = pos_edge_index[1, chosen]

    neg_item = []
    for u in pos_user.tolist():
        pos_items = user_pos_items_map_eval[u]
        while True:
            rand_item = random.randint(0, movie_embs.size(0)-1)
            if rand_item not in pos_items:
                neg_item.append(rand_item)
                break
    neg_item = torch.tensor(neg_item, dtype=torch.long, device=pos_user.device)

    with torch.no_grad():
        pos_scores = model.score(user_embs[pos_user], movie_embs[pos_item])
        neg_scores = model.score(user_embs[pos_user], movie_embs[neg_item])
        correct = (pos_scores > neg_scores).float().mean().item()  # fraction
    return correct  # Approx AUC
