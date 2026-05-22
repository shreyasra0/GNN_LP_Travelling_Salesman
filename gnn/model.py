import numpy as np
from gnn.layers import GCNLayer, EdgePredictor, Activation

class GNNModel:
    def __init__(self, in_features, hidden_features):
        self.conv1 = GCNLayer(in_features, hidden_features)
        self.conv2 = GCNLayer(hidden_features, hidden_features)
        self.predictor = EdgePredictor(hidden_features)

    def forward(self, x, adj):
        h = self.conv1.forward(x, adj)
        h = Activation.forward_relu(h)
        h = self.conv2.forward(h, adj)
        h = Activation.forward_relu(h)
        return h

    def predict_edge(self, h, u, v):
        return self.predictor.forward(h[u], h[v])