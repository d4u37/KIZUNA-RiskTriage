"""
Calibration methods for Kizuna RiskTriage.
1. Split Conformal Prediction
2. Conformalized Quantile Regression (CQR)
3. Adaptive Conformal Inference (ACI)
"""

import numpy as np
import pandas as pd
from typing import Tuple


class SplitConformal:
    def __init__(self, level=0.95):
        self.level = level
        self.alpha = 1 - level
        self.q_hat = None

    def calibrate(self, cal_residuals):
        n = len(cal_residuals)
        adjusted_level = min(np.ceil((n + 1) * self.level) / n, 1.0)
        abs_residuals = np.abs(cal_residuals)
        self.q_hat = np.quantile(abs_residuals, adjusted_level)
        return self.q_hat

    def predict_interval(self, point_predictions):
        lower = point_predictions - self.q_hat
        upper = point_predictions + self.q_hat
        lower = np.maximum(lower, 0)
        return lower, upper


class CQR:
    def __init__(self, level=0.95):
        self.level = level
        self.alpha = 1 - level
        self.Q = None
        self.cal_scores = None

    def calibrate(self, y_cal, q_lower_cal, q_upper_cal):
        n = len(y_cal)
        scores = np.maximum(q_lower_cal - y_cal, y_cal - q_upper_cal)
        self.cal_scores = scores
        adjusted_level = min(np.ceil((n + 1) * self.level) / n, 1.0)
        self.Q = np.quantile(scores, adjusted_level)
        return self.Q

    def predict_interval(self, q_lower, q_upper):
        lower = q_lower - self.Q
        upper = q_upper + self.Q
        lower = np.maximum(lower, 0)
        upper = np.maximum(upper, lower)
        return lower, upper


class AdaptiveConformalInference:
    def __init__(self, level=0.95, gamma=0.01):
        self.level = level
        self.alpha_target = 1 - level
        self.gamma = gamma
        self.alpha_history = []
        self.coverage_history = []
        self.corrections = []

    def run_online(self, y_true_sequence, q_lower_sequence, q_upper_sequence, cal_scores):
        n_test = len(y_true_sequence)
        aci_lower = np.zeros(n_test)
        aci_upper = np.zeros(n_test)
        alpha_t = self.alpha_target

        for t in range(n_test):
            level_t = np.clip(1 - alpha_t, 0.01, 0.99)
            Q_t = np.quantile(cal_scores, level_t)
            aci_lower[t] = max(q_lower_sequence[t] - Q_t, 0)
            aci_upper[t] = max(q_upper_sequence[t] + Q_t, aci_lower[t])
            covered = int(aci_lower[t] <= y_true_sequence[t] <= aci_upper[t])
            alpha_t = alpha_t + self.gamma * (self.alpha_target - (1 - covered))
            alpha_t = np.clip(alpha_t, 0.001, 0.5)
            self.alpha_history.append(alpha_t)
            self.coverage_history.append(covered)
            self.corrections.append(Q_t)

        return aci_lower, aci_upper

    def get_diagnostics(self):
        return pd.DataFrame({
            'step': range(len(self.alpha_history)),
            'alpha_t': self.alpha_history,
            'covered': self.coverage_history,
            'correction_Q': self.corrections
        })
