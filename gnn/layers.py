import numpy as np

class GCNLayer:
    def __init__(self, in_features, out_features):
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros(out_features)

    def forward(self, x, adj):
        support = np.dot(x, self.W)
        out = np.dot(adj, support) + self.b
        return out

class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class EdgePredictor:
    def __init__(self, in_features):
        self.W_edge = np.random.normal(0, 0.1, (2 * in_features, 1))
        
    def forward(self, h_u, h_v):
        combined = np.concatenate([h_u, h_v])
        logit = np.dot(combined, self.W_edge)
        return Activation.sigmoid(logit)