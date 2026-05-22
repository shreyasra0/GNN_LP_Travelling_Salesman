import cvxpy as cp
import numpy as np

def solve_lp_relaxation(num_nodes, edges, costs):
    x = cp.Variable(len(edges), nonneg=True)
    
    objective = cp.Minimize(cp.sum(cp.multiply(costs, x)))
    
    constraints = []
    for i in range(num_nodes):
        incident_indices = [idx for idx, (u, v) in enumerate(edges) if u == i or v == i]
        constraints.append(cp.sum(x[incident_indices]) == 2)
    
    constraints.append(x <= 1)
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if x.value is None:
        return None
        
    return np.clip(x.value, 0, 1)