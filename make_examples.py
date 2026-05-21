import os
import numpy as np
from scipy.spatial.distance import pdist, squareform

def solve_2opt_tsp(coords):
    num_nodes = len(coords)
    dist_matrix = squareform(pdist(coords, 'euclidean'))
    tour = list(range(num_nodes))
    
    improved = True
    while improved:
        improved = False
        for i in range(1, num_nodes - 2):
            for j in range(i + 1, num_nodes):
                if j - i == 1: continue
                
                old_dist = dist_matrix[tour[i-1], tour[i]] + dist_matrix[tour[j-1], tour[j]]
                new_dist = dist_matrix[tour[i-1], tour[j-1]] + dist_matrix[tour[i], tour[j]]
                
                if new_dist < old_dist:
                    tour[i:j] = tour[j-1:i-1:-1]
                    improved = True
    return tour

def create_bulk_training_data(num_instances=500, num_cities=50):
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print(f"Generating {num_instances} solved TSP instances...")
    
    for idx in range(num_instances):
        coords = np.random.uniform(0.0, 100.0, (num_cities, 2))
        optimal_tour = solve_2opt_tsp(coords)
        
        tsp_filename = os.path.join(data_dir, f"instance_{idx}.tsp")
        with open(tsp_filename, "w") as f:
            f.write(f"NAME : instance_{idx}\n")
            f.write(f"DIMENSION : {num_cities}\n")
            f.write("TYPE : TSP\n")
            f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
            f.write("NODE_COORD_SECTION\n")
            for node_id, (x, y) in enumerate(coords):
                f.write(f" {node_id + 1} {x:.4f} {y:.4f}\n")
            f.write("EOF\n")
            
        tour_filename = os.path.join(data_dir, f"instance_{idx}.opt.tour")
        with open(tour_filename, "w") as f:
            f.write(f"NAME : instance_{idx}.opt.tour\n")
            f.write("TYPE : TOUR\n")
            f.write(f"DIMENSION : {num_cities}\n")
            f.write("TOUR_SECTION\n")
            for node in optimal_tour:
                f.write(f" {node + 1}\n")
            f.write("-1\n")
            f.write("EOF\n")

    print("Data generation complete.")

if __name__ == "__main__":
    create_bulk_training_data(num_instances=500, num_cities=50)
