import numpy as np

def get_pruned_graph(coords, adj, probs, threshold=0.5):
    indices = np.where(adj > 0)
    rows = indices[0].flatten()
    cols = indices[1].flatten()
    
    mask = (probs.flatten() > threshold)
    
    u_idx = rows[mask]
    v_idx = cols[mask]
    filtered_probs = probs.flatten()[mask]
    
    edges = list(zip(u_idx, v_idx))
    
    costs = []
    for u, v in edges:
        dist = np.linalg.norm(coords[u] - coords[v])
        costs.append(dist)
        
    return edges, np.array(costs), filtered_probs