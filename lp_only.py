import glob
import pulp
import numpy as np
from evaluation import load_tsp, get_optimal_edges

def calculate_distance_matrix(coords):
    coords = np.array(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def check_baseline_performance():
    all_recalls = []
    tsp_files = glob.glob("data/*.tsp")

    if not tsp_files:
        print("No .tsp files found in data/ directory.")
        return

    for tsp_file in tsp_files:
        opt_file = tsp_file.replace(".tsp", ".opt.tour")
        coords = load_tsp(tsp_file)
        n = len(coords)
        optimal_edges = get_optimal_edges(opt_file, n)
        
        # Initialize LP problem
        prob = pulp.LpProblem("TSP_Baseline", pulp.LpMinimize)
        
        # 1. Define decision variables cleanly for undirected graph i < j
        edge_keys = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edges = pulp.LpVariable.dicts("edge", edge_keys, 0, 1, pulp.LpContinuous)
        
        # 2. Objective Function
        dist = calculate_distance_matrix(coords)
        prob += pulp.lpSum([dist[i][j] * edges[(i, j)] for i, j in edge_keys])
        
        # 3. Corrected Degree Constraints: Every node must connect to exactly 2 edges
        for i in range(n):
            node_edges = []
            for j in range(n):
                if i < j:
                    node_edges.append(edges[(i, j)])
                elif i > j:
                    node_edges.append(edges[(j, i)])
            prob += pulp.lpSum(node_edges) == 2
        
        # Solve without verbose output
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract selected edges (thresholded at 0.5)
        lp_selected = {(i, j) for (i, j), v in edges.items() if v.varValue and v.varValue > 0.5}
        
        # Calculate Recall
        correct = lp_selected.intersection(optimal_edges)
        recall = len(correct) / len(optimal_edges) if len(optimal_edges) > 0 else 0
        all_recalls.append(recall)
        
    avg_recall = sum(all_recalls) / len(all_recalls)
    print(f"Average LP baseline recall: {avg_recall:.4%}")

if __name__ == "__main__":
    check_baseline_performance()
