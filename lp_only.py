import glob
import pulp
import numpy as np
from evaluation import load_tsp, get_optimal_edges

def calculate_distance_matrix(coords):
    # Converts coordinates to numpy array and computes Euclidean matrix
    coords = np.array(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return dist_matrix

def check_baseline_performance():
    all_accuracies = []
    tsp_files = glob.glob("data/*.tsp")

    if not tsp_files:
        print("No .tsp files found in data/ directory.")
        return

    for tsp_file in tsp_files:
        opt_file = tsp_file.replace(".tsp", ".opt.tour")
        
        # 1. Load data
        coords = load_tsp(tsp_file)
        n = len(coords)
        
        # FIXED: Pass 'n' as the second argument required by get_optimal_edges
        optimal_edges = get_optimal_edges(opt_file, n)
        
        # 2. Setup and Solve Raw LP
        prob = pulp.LpProblem("TSP_Baseline", pulp.LpMinimize)
        edges = pulp.LpVariable.dicts("edge", [(i, j) for i in range(n) for j in range(i+1, n)], 0, 1, 'Continuous')
        
        dist = calculate_distance_matrix(coords)
        prob += pulp.lpSum([dist[i][j] * edges[(i, j)] for i, j in edges.keys()])
        
        # Degree constraints
        for i in range(n):
            prob += pulp.lpSum([edges[(min(i, j), max(i, j))] for j in range(n) if i != j]) == 2
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 3. Compare LP output to optimal edges
        lp_selected = {(i, j) for (i, j), v in edges.items() if v.varValue > 0.5}
        
        correct = lp_selected.intersection(optimal_edges)
        accuracy = len(correct) / len(optimal_edges)
        all_accuracies.append(accuracy)
        
    avg_accuracy = sum(all_accuracies) / len(all_accuracies)
    print(f"Average LP baseline recall: {avg_accuracy:.4%}")

if __name__ == "__main__":
    check_baseline_performance()