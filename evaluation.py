import numpy as np
import os
from gnn.model import GNNModel
from gnn.utils import load_tsp, get_knn_adj

def get_optimal_edges(tour_file, num_nodes):
    with open(tour_file, 'r') as f:
        lines = f.readlines()
        tour = []
        for line in lines:
            if 'EOF' in line or '-1' in line: break
            try:
                node = int(line.strip()) - 1
                if node >= 0: tour.append(node)
            except ValueError: continue
    
    optimal_edges = set()
    for i in range(len(tour)):
        u, v = tour[i], tour[(i + 1) % len(tour)]
        optimal_edges.add(tuple(sorted((u, v))))
    return optimal_edges

def evaluate_recall(tsp_file, tour_file, model, threshold=0.4):
    coords = load_tsp(tsp_file)
    adj = get_knn_adj(coords)
    
    h = model.forward(coords, adj)
    indices = np.where(adj > 0)
    rows, cols = indices[0], indices[1]
    logits = [model.predictor.forward(h[r], h[c]) for r, c in zip(rows, cols)]
    probs = 1 / (1 + np.exp(-np.clip(np.array(logits).flatten(), -500, 500)))
    
    mask = probs > threshold
    kept_edges = set()
    for r, c in zip(rows[mask], cols[mask]):
        kept_edges.add(tuple(sorted((r, c))))
        
    optimal_edges = get_optimal_edges(tour_file, len(coords))
    
    found = [e for e in optimal_edges if e in kept_edges]
    recall = len(found) / len(optimal_edges)
    
    return recall, len(kept_edges)

if __name__ == "__main__":
    model = GNNModel(in_features=2, hidden_features=16)
    model.load_weights('best_model.npz')
    
    recall, num_kept = evaluate_recall('data/instance_0.tsp', 'data/instance_0.opt.tour', model)
    print(f"Recall: {recall:.4f} | Edges Kept: {num_kept}")