import numpy as np

class GCNLayer:
    def __init__(self, in_features, out_features):
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros(out_features)

    def forward(self, x, adj):
        self.x = x
        I = np.eye(adj.shape[0])
        adj_hat = adj + I
        D = np.sum(adj_hat, axis=1)
        D_inv_sqrt = np.zeros_like(D)
        np.power(D, -0.5, where=D != 0, out=D_inv_sqrt)
        D_mat_inv_sqrt = np.diag(D_inv_sqrt)
        self.adj = D_mat_inv_sqrt @ adj_hat @ D_mat_inv_sqrt
        self.out = np.dot(self.adj, np.dot(x, self.W)) + self.b
        return self.out

    def backward(self, grad_output, lr):
        dW = np.dot(self.x.T, np.dot(self.adj.T, grad_output))
        db = np.sum(grad_output, axis=0)
        dW = np.clip(dW, -1.0, 1.0)
        dx = np.dot(self.adj.T, np.dot(grad_output, self.W.T))
        self.W -= lr * dW
        self.b -= lr * db
        return dx

class EdgePredictor:
    def __init__(self, in_features):
        # Improved initialization: Xavier/Glorot
        limit = np.sqrt(6 / (4 * in_features + 1))
        self.W_edge = np.random.uniform(-limit, limit, (4 * in_features, 1))
        
    def forward(self, h_u, h_v):
        diff = np.abs(h_u - h_v)
        prod = h_u * h_v
        self.combined = np.concatenate([h_u, h_v, diff, prod])
        # Return raw logit for stable gradient computation in Loss
        self.logit = np.dot(self.combined, self.W_edge)
        return self.logit

    def backward(self, grad, lr):
        # Gradient is passed as (pred - target) from PruningLoss
        dW = self.combined.reshape(-1, 1) * grad
        dW = np.clip(dW, -1.0, 1.0)
        d_combined = np.dot(self.W_edge, grad)
        self.W_edge -= lr * dW
        half = len(d_combined) // 4
        return d_combined[:half], d_combined[half:2*half]