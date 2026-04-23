import numpy as np


class ConformalPredictor:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.all_p_values = []

    def compute_calibration_p_values(self, y_test_pred_prob):
        """
        计算校准集的p-values
        """
        self.all_p_values.append(1 - np.max(y_test_pred_prob, axis=1))

    def cal_p_values(self):
        self.all_p_values = np.concatenate(self.all_p_values)
        threshold_index = int(np.ceil((1 - self.alpha) * len(self.all_p_values)))
        q_hat = np.sort(self.all_p_values)[threshold_index - 1]
        return q_hat

    def predict_with_confidence(self, text, threshold=None):
        pass
