import numpy as np

class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def backward_relu(grad, x):
        return grad * (x > 0).astype(float)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, x * alpha)

    @staticmethod
    def backward_leaky_relu(grad, x, alpha=0.01):
        return grad * np.where(x > 0, 1.0, alpha)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def backward_sigmoid(grad, x):
        s = Activation.sigmoid(x)
        return grad * (s * (1 - s))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def backward_tanh(grad, x):
        return grad * (1 - np.tanh(x)**2)