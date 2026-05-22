import numpy as np
from gnn.model import GNNModel
from gnn.utils import load_tsp, get_knn_adj
from lp.graph_utils import get_pruned_graph
from lp.solver import solve_lp_relaxation

class HybridSolver:
    def __init__(self, model_path='best_model.npz'):
        self.model = GNNModel(in_features=2, hidden_features=16)
        self.model.load_weights(model_path)
        
    def solve(self, tsp_file, threshold=0.4):
        coords = load_tsp(tsp_file)
        adj = get_knn_adj(coords)
        
        h = self.model.forward(coords, adj)

        indices = np.where(adj > 0)
        rows, cols = indices[0], indices[1]
        
        logits = [self.model.predictor.forward(h[r], h[c]) for r, c in zip(rows, cols)]
        probs = 1 / (1 + np.exp(-np.clip(np.array(logits), -500, 500)))
        
        edges, costs, _ = get_pruned_graph(coords, adj, probs, threshold=threshold)
        x_val = solve_lp_relaxation(len(coords), edges, costs)
        
        return x_val, edges