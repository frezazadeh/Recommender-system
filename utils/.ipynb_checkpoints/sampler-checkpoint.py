import torch
import random
import numpy as np

user_pos_items_map = None
def sample_pos_neg_edges(edge_index: torch.Tensor, 
                         num_users: int, 
                         num_items: int, 
                         batch_size: int):
    """
    Randomly sample 'batch_size' positive edges, then for each user
    sample a negative item not in user's positive set.
    Returns (user_batch, pos_item_batch, neg_item_batch).
    """
    global user_pos_items_map
    if user_pos_items_map is None:
        # Build adjacency once
        user_pos_items_map = [[] for _ in range(num_users)]
        for i in range(edge_index.size(1)):
            u = int(edge_index[0, i])
            it = int(edge_index[1, i])
            user_pos_items_map[u].append(it)

    num_pos_edges = edge_index.size(1)
    chosen = np.random.choice(num_pos_edges, size=batch_size, replace=(num_pos_edges < batch_size))

    user_batch     = edge_index[0, chosen].long()
    pos_item_batch = edge_index[1, chosen].long()

    neg_item_batch = []
    for u in user_batch.tolist():
        pos_items = user_pos_items_map[u]
        while True:
            rand_item = random.randint(0, num_items - 1)
            if rand_item not in pos_items:
                neg_item_batch.append(rand_item)
                break
    neg_item_batch = torch.tensor(neg_item_batch, dtype=torch.long)

    return user_batch, pos_item_batch, neg_item_batch