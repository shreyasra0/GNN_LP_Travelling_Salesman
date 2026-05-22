import numpy as np

class PruningLoss:
    @staticmethod
    def forward(logit, y_true, pos_weight, gamma=2.0):
        prob = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))
        
        bce = np.maximum(logit, 0) - logit * y_true + np.log(1 + np.exp(-np.abs(logit)))
        
        p_t = (y_true * prob) + ((1 - y_true) * (1 - prob))
        modulating_factor = (1 - p_t) ** gamma
        
        loss = modulating_factor * bce
        return float(np.sum(pos_weight * loss))

    @staticmethod
    def backward(logit, y_true, pos_weight, gamma=2.0):
        prob = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))
        p_t = (y_true * prob) + ((1 - y_true) * (1 - prob))
        modulating_factor = (1 - p_t) ** gamma
        
        alpha_t = y_true * pos_weight + (1 - y_true)
        grad = alpha_t * modulating_factor * (prob - y_true)
        
        return grad