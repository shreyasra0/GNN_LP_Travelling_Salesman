import numpy as np
import os
from scipy.spatial.distance import pdist, squareform

def load_tsp(filepath):
    coords = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        start = 0
        for i, line in enumerate(lines):
            if "NODE_COORD_SECTION" in line:
                start = i + 1
                break
        for line in lines[start:]:
            if "EOF" in line: break
            parts = line.split()
            coords.append([float(parts[1]), float(parts[2])])
    
    coords = np.array(coords)
    coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-9)
    return coords

def get_knn_adj(coords, k=10):
    dist_matrix = squareform(pdist(coords, 'euclidean'))
    num_nodes = len(coords)
    adj = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        nearest_indices = np.argsort(dist_matrix[i])[1:k+1]
        adj[i, nearest_indices] = 1
        adj[nearest_indices, i] = 1
    
    adj += np.eye(num_nodes)
    
    degree = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(degree, -0.5, where=degree!=0, out=np.zeros_like(degree))
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def get_edge_labels(tour_filepath, num_nodes):
    tour = []
    with open(tour_filepath, 'r') as f:
        lines = f.readlines()
        start = 0
        for i, line in enumerate(lines):
            if "TOUR_SECTION" in line:
                start = i + 1
                break
        for line in lines[start:]:
            if "-1" in line or "EOF" in line: break
            tour.append(int(line.strip()) - 1)
            
    labels = np.zeros((num_nodes, num_nodes))
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i+1]
        labels[u, v] = labels[v, u] = 1
    labels[tour[-1], tour[0]] = labels[tour[0], tour[-1]] = 1
    return labels