import argparse
from data.download import preprocessing
#from utils.sampler import sample_pos_neg_edges
#from utils.evaluator import BPRModelHandler
#from models.loss import evaluate_bpr_loss
from models.train import BPRTrainer
#from utils.metrics import evaluate_auc
from models.gnn import Model
import torch

def main(args):
    num_users, num_movies, movie_feat, train_data, val_data, test_data, train_pos_edge_index, val_pos_edge_index, test_pos_edge_index, genre_names, movie_id_to_name = preprocessing(args.data_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    model = Model(
    num_users=num_users,
    num_movies=num_movies,
    num_movie_features=movie_feat.size(1),
    hidden_channels=args.hidden_channels).to(device)

    train_data = train_data.to(device)
    val_data   = val_data.to(device)
    test_data  = test_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_epochs = args.epochs
    batch_size = args.batch_size
    steps = args.steps  # number of mini-batches per epoch
    train_pos = train_pos_edge_index.to(device)
    patience = args.patience
    data_size = args.data_size
    hidden_channels=args.hidden_channels
    
    trainer=BPRTrainer(model,
                       train_data,
                       val_data,
                       test_data,
                       train_pos,
                       val_pos_edge_index,
                       test_pos_edge_index,
                       optimizer, device,
                       num_epochs,
                       steps,
                       batch_size,
                       num_users,
                       num_movies,
                       patience, 
                       data_size,
                       movie_feat,
                       hidden_channels)
    # Train the model
    trainer.train()

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=str, default='small')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=float, default=30)
    
    args = parser.parse_args()
    main(args)
