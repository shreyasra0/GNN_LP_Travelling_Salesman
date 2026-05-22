import numpy as np
from gnn.activations import Activation
from gnn.layers import GCNLayer, EdgePredictor

class GNNModel:
    def __init__(self, in_features, hidden_features):
        self.conv1 = GCNLayer(in_features, hidden_features)
        self.conv2 = GCNLayer(hidden_features, hidden_features)
        self.predictor = EdgePredictor(hidden_features)

    def forward(self, x, adj):
        self.h0 = self.conv1.forward(x, adj)
        h = Activation.relu(self.h0)
        self.h1 = self.conv2.forward(h, adj)
        h = Activation.relu(self.h1)
        return h

    def predict_edge(self, h, u, v):
        return self.predictor.forward(h[u], h[v])
        
    def save_weights(self, path='best_model.npz'):
        np.savez(path, 
                 W1=self.conv1.W, b1=self.conv1.b,
                 W2=self.conv2.W, b2=self.conv2.b,
                 W_edge=self.predictor.W_edge)

    def load_weights(self, path='best_model.npz'):
        data = np.load(path)
        self.conv1.W = data['W1']
        self.conv1.b = data['b1']
        self.conv2.W = data['W2']
        self.conv2.b = data['b2']
        self.predictor.W_edge = data['W_edge']