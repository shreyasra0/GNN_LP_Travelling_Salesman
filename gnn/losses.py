import numpy as np

class PruningLoss:
    @staticmethod
    def forward(y_pred, y_true, fp_penalty=5.0):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        base_loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        fp_mask = (y_true == 0).astype(float)
        fp_loss = fp_penalty * fp_mask * y_pred
        
        return np.mean(base_loss + fp_loss)

    @staticmethod
    def backward(y_pred, y_true, fp_penalty=5.0):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        grad = -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        
        fp_grad = fp_penalty * (y_true == 0).astype(float)
        
        return (grad + fp_grad) / y_true.size