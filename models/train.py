import torch
from models.loss import bpr_loss, evaluate_bpr_loss
from utils.sampler import sample_pos_neg_edges
from utils.metrics import evaluate_auc
from models.gnn import Model
import matplotlib.pyplot as plt


class BPRTrainer:
    def __init__(self, model, train_data, val_data, test_data, train_pos,
                 val_pos_edge_index, test_pos_edge_index, optimizer, device,
                 num_epochs, steps, batch_size, num_users, num_movies, patience, data_size, movie_feat, hidden_channels):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_pos = train_pos
        self.val_pos_edge_index = val_pos_edge_index
        self.test_pos_edge_index = test_pos_edge_index
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.steps = steps
        self.batch_size = batch_size
        self.num_users = num_users
        self.num_movies = num_movies
        self.patience = patience
        self.data_size = data_size
        self.movie_feat = movie_feat
        self.hidden_channels = hidden_channels

        self.train_losses = []
        self.test_losses = []
        self.val_aucs = []
        self.best_test_loss = float('inf')
        self.epochs_no_improve = 0

        
    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            total_train_loss = 0.0
            for _ in range(self.steps):
                # Forward pass for each mini-batch
                x_dict = self.model(self.train_data)
                user_embs = x_dict['user']
                movie_embs = x_dict['movie']

                user_batch, pos_item_batch, neg_item_batch = sample_pos_neg_edges(
                    edge_index=self.train_pos,
                    num_users=self.num_users,
                    num_items=self.num_movies,
                    batch_size=self.batch_size
                )
                user_batch = user_batch.to(self.device)
                pos_item_batch = pos_item_batch.to(self.device)
                neg_item_batch = neg_item_batch.to(self.device)

                pos_scores = self.model.score(user_embs[user_batch], movie_embs[pos_item_batch])
                neg_scores = self.model.score(user_embs[user_batch], movie_embs[neg_item_batch])

                loss = bpr_loss(pos_scores, neg_scores)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / self.steps
            self.train_losses.append(avg_train_loss)

            # Compute BPR loss on the test set
            test_bpr_loss = evaluate_bpr_loss(self.model, self.test_data, 
                                                   self.test_pos_edge_index.to(self.device),
                                                   self.device, num_samples=5000)
            self.test_losses.append(test_bpr_loss)

            # Validation AUC after each epoch
            val_auc = evaluate_auc(self.model, self.val_data, 
                                        self.val_pos_edge_index.to(self.device), num_samples=5000)
            self.val_aucs.append(val_auc)

            print(f"Epoch [{epoch}/{self.num_epochs}] "
                  f"Train BPR Loss: {avg_train_loss:.4f}, "
                  f"Test BPR Loss: {test_bpr_loss:.4f} "
                  f"Val AUC={val_auc:.4f}")

            # Early stopping on test loss
            if test_bpr_loss < self.best_test_loss:
                self.best_test_loss = test_bpr_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
        # Plot training vs test BPR Loss
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train BPR Loss')
        plt.plot(self.test_losses, label='Test BPR Loss')
        plt.xlabel('Epoch')
        plt.ylabel('BPR Loss')
        plt.title('Train vs. Test BPR Loss')
        plt.legend()
        
        # Plot validation AUC vs. epoch
        plt.subplot(1, 2, 2)
        plt.plot(self.val_aucs, label='Val AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Validation AUC')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/training_validation_metrics_{self.data_size}.png", dpi=300)
        plt.close()


        # -- Final evaluation on the model we've just trained --
        final_test_bpr_loss = evaluate_bpr_loss(self.model, self.test_data, 
                                                self.test_pos_edge_index.to(self.device),
                                                self.device,
                                                num_samples=5000)
        print(f"\nFinal Test BPR Loss (current model) = {final_test_bpr_loss:.4f}")
        test_auc = evaluate_auc(self.model, self.test_data, self.test_pos_edge_index.to(self.device), num_samples=5000)
        print(f"Test AUC (approx BPR) with current model = {test_auc:.4f}")
        
        # -- Save the trained model (just the state dict) --
        save_path = f"saved_model/bpr_model_{self.data_size}.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
        # -- Load the model into a new instance --
        loaded_model = Model(
            num_users=self.num_users,
            num_movies=self.num_movies,
            num_movie_features=self.movie_feat.size(1),
            hidden_channels=self.hidden_channels
        ).to(self.device)
        
        # Use weights_only=True when loading (PyTorch 2.0+)
        loaded_state_dict = torch.load(save_path, weights_only=True)
        loaded_model.load_state_dict(loaded_state_dict)
        loaded_model.eval()
        print("Model loaded from disk and moved to evaluation mode.")
        
        
        # -- Final evaluation on the loaded model --
        final_test_bpr_loss_loaded = evaluate_bpr_loss(loaded_model, self.test_data, 
                                                       self.test_pos_edge_index.to(self.device),
                                                       self.device,
                                                       num_samples=5000)
        print(f"\nFinal Test BPR Loss (loaded model) = {final_test_bpr_loss_loaded:.4f}")
        final_test_auc_loaded = evaluate_auc(loaded_model, self.test_data, self.test_pos_edge_index.to(self.device), num_samples=5000)
        print(f"Test AUC (approx BPR) with loaded model = {final_test_auc_loaded:.4f}")
