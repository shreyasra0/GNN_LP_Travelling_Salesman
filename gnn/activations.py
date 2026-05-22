import numpy as np

class Activation:
    @staticmethod
    def forward_relu(x):
        return np.maximum(0, x)

    @staticmethod
    def backward_relu(x):
        return (x > 0).astype(float)

    @staticmethod
    def forward_sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def backward_sigmoid(x):
        s = Activation.forward_sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def forward_tanh(x):
        return np.tanh(x)

    @staticmethod
    def backward_tanh(x):
        return 1 - np.tanh(x)**2