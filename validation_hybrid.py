import os
from lp_gnn_hybrid import HybridSolver
from train_gcn import get_data_splits

def run_evaluation(data_dir='data'):
    _, val_files = get_data_splits(data_dir)
    solver = HybridSolver()
    
    results = {}
    for f in val_files:
        print(f"Testing {f}...")
        x_val, edges = solver.solve(os.path.join(data_dir, f))
        
        # Simple metric: how many edges were kept?
        results[f] = len(edges)
        print(f"Edges kept: {len(edges)}")
        
    return results

if __name__ == "__main__":
    run_evaluation()