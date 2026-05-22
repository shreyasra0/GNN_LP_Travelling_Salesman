import numpy as np
import glob
from gnn.model import GNNModel
from gnn.utils import load_tsp, get_knn_adj
from evaluation import get_optimal_edges

def fine_grained_analyze(tsp_file, model):
    coords = load_tsp(tsp_file)
    adj = get_knn_adj(coords)
    tour_file = tsp_file.replace('.tsp', '.opt.tour')
    
    h = model.forward(coords, adj)
    indices = np.where(adj > 0)
    rows, cols = indices[0], indices[1]
    logits = [model.predictor.forward(h[r], h[c]) for r, c in zip(rows, cols)]
    probs = 1 / (1 + np.exp(-np.clip(np.array(logits).flatten(), -500, 500)))
    optimal_edges = get_optimal_edges(tour_file, len(coords))
    
    # Analyze the "fringe" - edges dropped between 0.45 and 0.55
    mask_lower = probs > 0.45
    mask_upper = probs > 0.55
    
    dropped_edges = []
    for r, c, p, m1, m2 in zip(rows, cols, probs, mask_lower, mask_upper):
        if m1 and not m2: # Dropped in the transition zone
            edge = tuple(sorted((r, c)))
            is_optimal = edge in optimal_edges
            dropped_edges.append((edge, p, is_optimal))
            
    print(f"Edges dropped between 0.45 and 0.55: {len(dropped_edges)}")
    print(f"Of those, optimal edges lost: {sum(1 for e in dropped_edges if e[2])}")

if __name__ == "__main__":
    model = GNNModel(in_features=2, hidden_features=16)
    model.load_weights('best_model.npz')
    fine_grained_analyze('data/instance_0.tsp', model)