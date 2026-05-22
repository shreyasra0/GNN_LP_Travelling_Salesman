import os
import numpy as np
from gnn.model import GNNModel
from gnn.losses import PruningLoss
from gnn.activations import Activation
from gnn.utils import load_tsp, get_knn_adj, get_edge_labels

def get_data_splits(data_dir, split_ratio=0.8):
    files = [f for f in os.listdir(data_dir) if f.endswith('.tsp')]
    np.random.shuffle(files)
    split = int(len(files) * split_ratio)
    return files[:split], files[split:]

def run_diagnostic(train_files, model, data_dir='data'):
    f = train_files[0]
    file_path = os.path.join(data_dir, f)
    coords = load_tsp(file_path)
    adj = get_knn_adj(coords)
    h = model.forward(coords, adj)
    indices = np.where(adj > 0)
    
    logits = [model.predictor.forward(h[u], h[v]) for u, v in zip(indices[0], indices[1])]
    preds = 1 / (1 + np.exp(-np.clip(np.array(logits), -500, 500)))
    
    print(f"--- Diagnostic: Mean={np.mean(preds):.4f}, Std={np.std(preds):.4f}, Range=[{np.min(preds):.4f}, {np.max(preds):.4f}] ---")

def train_loop(train_files, val_files, epochs=100, initial_lr=0.0001):
    model = GNNModel(in_features=2, hidden_features=16)
    data_dir = 'data'
    lr = initial_lr
    best_loss = float('inf')
    
    for epoch in range(epochs):
        if epoch > 0 and epoch % 10 == 0:
            lr *= 0.9
        
        epoch_loss = 0.0
        for f in train_files:
            file_path = os.path.join(data_dir, f)
            coords = load_tsp(file_path)
            adj = get_knn_adj(coords)
            labels = get_edge_labels(file_path.replace(".tsp", ".opt.tour"), len(coords))
            
            num_nodes = len(coords)
            pos_weight = (num_nodes - 1) / 2
            
            h = model.forward(coords, adj)
            indices = np.where(adj > 0)
            grad_h = np.zeros_like(h)
            num_edges = len(indices[0])
            file_loss = 0.0
            
            for u, v in zip(indices[0], indices[1]):
                logit = model.predictor.forward(h[u], h[v])
                file_loss += PruningLoss.forward(logit, labels[u, v], pos_weight)
                
                loss_grad = PruningLoss.backward(logit, labels[u, v], pos_weight)
                
                grad_norm = np.sqrt(np.sum(loss_grad**2))
                if grad_norm > 1.0:
                    loss_grad = loss_grad * (1.0 / (grad_norm + 1e-6))
                
                du, dv = model.predictor.backward(loss_grad, lr)
                grad_h[u] += du
                grad_h[v] += dv
            
            epoch_loss += (file_loss / num_edges)
            
            grad_h = Activation.backward_leaky_relu(grad_h, model.h1)
            grad_h = model.conv2.backward(grad_h, lr)
            grad_h = Activation.backward_leaky_relu(grad_h, model.h0)
            grad_h = model.conv1.backward(grad_h, lr)
        
        avg_epoch_loss = epoch_loss / len(train_files)
        print(f"Epoch {epoch} | Avg Loss: {avg_epoch_loss:.4f} | LR: {lr:.6f}")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model.save_weights('best_model.npz')
            
        if epoch % 5 == 0:
            run_diagnostic(train_files, model)

if __name__ == "__main__":
    train_files, val_files = get_data_splits('data')
    train_loop(train_files, val_files)