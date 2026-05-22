import os
import numpy as np
import glob
from gnn.model import GNNModel
from gnn.utils import load_tsp, get_knn_adj

def get_optimal_edges(tour_file, num_nodes):
    with open(tour_file, 'r') as f:
        tour = [int(line.strip()) - 1 for line in f if line.strip().isdigit()]
        tour = [n for n in tour if 0 <= n < num_nodes]
    
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
    kept_edges = {tuple(sorted((r, c))) for r, c in zip(rows[mask], cols[mask])}
    
    optimal_edges = get_optimal_edges(tour_file, len(coords))
    found = [e for e in optimal_edges if e in kept_edges]
    
    recall = len(found) / len(optimal_edges) if len(optimal_edges) > 0 else 0
    return recall, len(kept_edges), len(indices[0]) 

if __name__ == "__main__":
    model = GNNModel(in_features=2, hidden_features=16)
    model.load_weights('best_model.npz')
    
    tsp_files = sorted(glob.glob("data/*.tsp"))
    all_recalls = []
    all_reductions = []
    
    print(f"{'Instance':<20} | {'Recall':<10} | {'Reduction (%)':<15}")
    print("-" * 50)
    
    for f in tsp_files:
        tour_file = f.replace('.tsp', '.opt.tour')
        recall, kept, total = evaluate_recall(f, tour_file, model)
        reduction = 1 - (kept / total)
        
        all_recalls.append(recall)
        all_reductions.append(reduction)
        print(f"{os.path.basename(f):<20} | {recall:.4f} | {reduction:.2%}")

    avg_recall = np.mean(all_recalls)
    avg_reduction = np.mean(all_reductions)

    print("-" * 50)
    print(f"{'AVERAGE':<20} | {avg_recall:.4f} | {avg_reduction:.2%}")
    
    print("\n" + "="*50)
    print("FINAL SUMMARY REPORT")
    print(f"Total Instances Analyzed: {len(tsp_files)}")
    print(f"Average Recall:           {avg_recall:.4%}")
    print(f"Average Edge Reduction:   {avg_reduction:.2%}")
    print("="*50)