import os
import glob
import pulp
import numpy as np
from gnn.model import GNNModel
from gnn.utils import load_tsp, get_knn_adj

def get_optimal_edges(tour_file, num_nodes):
    """Loads true optimal tour and returns a set of undirected edge tuples."""
    with open(tour_file, 'r') as f:
        tour = [int(line.strip()) - 1 for line in f if line.strip().isdigit()]
    tour = [n for n in tour if 0 <= n < num_nodes]
    
    optimal_edges = set()
    for i in range(len(tour)):
        u, v = tour[i], tour[(i + 1) % len(tour)]
        optimal_edges.add(tuple(sorted((u, v))))
    return optimal_edges

def calculate_distance_matrix(coords):
    """Computes Euclidean distance matrix using NumPy vectorization."""
    coords = np.array(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def evaluate_hybrid_recall(tsp_file, tour_file, model, threshold=0.4):
    coords = load_tsp(tsp_file)
    n = len(coords)
    optimal_edges = get_optimal_edges(tour_file, n)
    
    # --- STAGE 1: GNN Edge Filtering ---
    adj = get_knn_adj(coords)
    h = model.forward(coords, adj)
    
    # Extract the distinct row and column index arrays from the tuple
    indices = np.where(adj > 0)
    rows = indices[0]
    cols = indices[1]
    
    # Forward pairs sequentially to match layers.py 1D matrix shape properties
    logits = [model.predictor.forward(h[r], h[c]) for r, c in zip(rows, cols)]
    probs = 1 / (1 + np.exp(-np.clip(np.array(logits).flatten(), -500, 500)))
    
    mask = probs > threshold
    gnn_kept_edges = {tuple(sorted((r, c))) for r, c in zip(rows[mask], cols[mask])}
    
    # --- STAGE 2: LP Baseline Optimization on Filtered Subgraph ---
    prob = pulp.LpProblem("TSP_Hybrid", pulp.LpMinimize)
    
    # LP only considers variables that survived the GNN threshold
    edges = pulp.LpVariable.dicts("edge", gnn_kept_edges, 0, 1, pulp.LpContinuous)
    
    dist = calculate_distance_matrix(coords)
    prob += pulp.lpSum([dist[i][j] * edges[(i, j)] for i, j in gnn_kept_edges])
    
    # Enforce degree constraints using only the surviving sparse edges
    for i in range(n):
        node_edges = []
        for j in range(n):
            edge_key = (min(i, j), max(i, j))
            if edge_key in gnn_kept_edges and i != j:
                node_edges.append(edges[edge_key])
        
        # Guard rail: If GNN completely isolated a node, the problem becomes infeasible
        if not node_edges:
            return 0.0, False  
            
        prob += pulp.lpSum(node_edges) == 2
    
    # Solve without verbose output
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if status != pulp.LpStatusOptimal:
        return 0.0, False  
        
    # --- Final Metric Calculation ---
    hybrid_selected = {(i, j) for (i, j), v in edges.items() if v.varValue and v.varValue > 0.5}
    correct = hybrid_selected.intersection(optimal_edges)
    
    hybrid_recall = len(correct) / len(optimal_edges) if len(optimal_edges) > 0 else 0
    return hybrid_recall, True

if __name__ == "__main__":
    # Initialize and load your GNN Weights
    model = GNNModel(in_features=2, hidden_features=16)
    model.load_weights('best_model.npz')
    
    tsp_files = sorted(glob.glob("data/*.tsp"))
    all_hybrid_recalls = []
    infeasible_count = 0
    
    print(f"{'Instance':<25} | {'Hybrid Recall':<15} | {'Status':<10}")
    print("-" * 55)
    
    for f in tsp_files:
        tour_file = f.replace('.tsp', '.opt.tour')
        recall, success = evaluate_hybrid_recall(f, tour_file, model, threshold=0.4)
        
        if success:
            all_hybrid_recalls.append(recall)
            status_str = "Success"
            recall_str = f"{recall:.4f}"
        else:
            infeasible_count += 1
            status_str = "Infeasible"
            recall_str = "0.0000"  
            
        print(f"{os.path.basename(f):<25} | {recall_str:<15} | {status_str:<10}")
        
    print("-" * 55)
    avg_hybrid_recall = np.mean(all_hybrid_recalls) if all_hybrid_recalls else 0.0
    
    print("FINAL HYBRID SYSTEM REPORT")
    print(f"Total Graphs Checked: {len(tsp_files)}")
    print(f"Successfully Solved: {len(all_hybrid_recalls)}")
    print(f"Infeasible (Filtered too strictly): {infeasible_count}")
    print(f"Average Hybrid Recall (Solved Only): {avg_hybrid_recall:.4%}")
