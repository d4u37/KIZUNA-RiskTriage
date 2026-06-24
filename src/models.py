"""
Forecasting models for Kizuna RiskTriage.
Implements LightGBM with quantile regression.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Dict, Tuple, Optional


class QuantileForecaster:
    def __init__(self, quantiles=[0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975], lgb_params=None):
        self.quantiles = sorted(quantiles)
        self.models = {}
        self.lgb_params = lgb_params or {
            'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 7,
            'num_leaves': 63, 'min_child_samples': 30, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'verbose': -1, 'n_jobs': -1,
        }

    def fit(self, X_train, y_train):
        print(f"Training {len(self.quantiles)} quantile models...")
        for q in self.quantiles:
            params = self.lgb_params.copy()
            params['objective'] = 'quantile'
            params['alpha'] = q
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            self.models[q] = model
            print(f"  Quantile {q:.3f} trained.")
        print("All quantile models trained.")
        return self

    def predict(self, X):
        predictions = {}
        for q in self.quantiles:
            pred = self.models[q].predict(X)
            pred = np.maximum(pred, 0)
            predictions[f'q_{q}'] = pred
        return pd.DataFrame(predictions)

    def predict_interval(self, X, level=0.95):
        alpha = 1 - level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        lower_q_actual = min(self.quantiles, key=lambda x: abs(x - lower_q))
        upper_q_actual = min(self.quantiles, key=lambda x: abs(x - upper_q))
        point = np.maximum(self.models[0.5].predict(X), 0)
        lower = np.maximum(self.models[lower_q_actual].predict(X), 0)
        upper = np.maximum(self.models[upper_q_actual].predict(X), lower)
        return point, lower, upper


class PointForecaster:
    def __init__(self, lgb_params=None):
        self.lgb_params = lgb_params or {
            'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 7,
            'num_leaves': 63, 'min_child_samples': 30, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'verbose': -1, 'n_jobs': -1,
            'objective': 'regression',
        }
        self.model = None

    def fit(self, X_train, y_train):
        self.model = lgb.LGBMRegressor(**self.lgb_params)
        self.model.fit(X_train, y_train)
        train_pred = self.model.predict(X_train)
        self.residual_std = np.std(y_train - train_pred)
        return self

    def predict(self, X):
        return np.maximum(self.model.predict(X), 0)
