import torch.nn.functional as F
import torch
from torch import nn, Tensor
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData, download_url, extract_zip

class GNN(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class Model(nn.Module):
    def __init__(self, num_users, num_movies, num_movie_features, hidden_channels):
        super().__init__()
        # Embeddings for users/movies
        self.user_emb = nn.Embedding(num_users, hidden_channels)
        self.movie_emb = nn.Embedding(num_movies, hidden_channels)

        # Linear transform for movie genres
        self.movie_lin = nn.Linear(num_movie_features, hidden_channels)

        # GNN
        self.gnn = GNN(hidden_channels)
        self.hetero_gnn = to_hetero(
            self.gnn,
            metadata=(
                ['user','movie'], 
                [('user','rates','movie'), ('movie','rev_rates','user')]
            )
        )

        # Simple MLP scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, full_data: HeteroData):
        """Compute node embeddings for all 'user' and 'movie' nodes."""
        x_dict = {
            'user': self.user_emb(full_data['user'].node_id),
            'movie': self.movie_lin(full_data['movie'].x) + self.movie_emb(full_data['movie'].node_id),
        }
        x_dict = self.hetero_gnn(x_dict, full_data.edge_index_dict)
        return x_dict
    
    def score(self, user_emb: Tensor, movie_emb: Tensor) -> Tensor:
        """Returns scalar scores for (user_emb, movie_emb)."""
        x = torch.cat([user_emb, movie_emb], dim=-1)
        out = self.scorer(x)
        return out.view(-1)

