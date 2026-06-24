"""
Calibration and evaluation metrics for Kizuna RiskTriage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def empirical_coverage(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))


def mean_interval_width(lower, upper):
    return np.mean(upper - lower)


def relative_interval_width(lower, upper, point_pred):
    denominator = np.maximum(point_pred, 1.0)
    return (upper - lower) / denominator


def winkler_score(y_true, lower, upper, alpha=0.05):
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    inside = ~below & ~above
    scores = np.zeros_like(y_true, dtype=float)
    scores[inside] = width[inside]
    scores[below] = width[below] + (2 / alpha) * (lower[below] - y_true[below])
    scores[above] = width[above] + (2 / alpha) * (y_true[above] - upper[above])
    return np.mean(scores)


def pinball_loss(y_true, y_pred_quantile, tau):
    error = y_true - y_pred_quantile
    loss = np.where(error >= 0, tau * error, (tau - 1) * error)
    return np.mean(loss)


def calibration_curve_data(y_true, quantile_predictions, levels=None):
    q_cols = [c for c in quantile_predictions.columns if c.startswith('q_')]
    available_quantiles = sorted([float(c.replace('q_', '')) for c in q_cols])
    if levels is None:
        levels = []
        for q in available_quantiles:
            if q > 0.5:
                lower_q = 1 - q
                if lower_q in available_quantiles:
                    levels.append(q - lower_q)
        levels = sorted(set(levels))
    results = []
    for level in levels:
        alpha = 1 - level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        lower_col = f'q_{min(available_quantiles, key=lambda x: abs(x - lower_q))}'
        upper_col = f'q_{min(available_quantiles, key=lambda x: abs(x - upper_q))}'
        lower = quantile_predictions[lower_col].values
        upper = quantile_predictions[upper_col].values
        observed = empirical_coverage(y_true, lower, upper)
        results.append({'nominal_level': level, 'observed_coverage': observed, 'mean_width': np.mean(upper - lower)})
    return pd.DataFrame(results)


def compute_all_metrics(y_true, lower, upper, point_pred, nominal_level=0.95, regime_labels=None):
    alpha = 1 - nominal_level
    metrics = {
        'empirical_coverage': empirical_coverage(y_true, lower, upper),
        'nominal_level': nominal_level,
        'coverage_gap': empirical_coverage(y_true, lower, upper) - nominal_level,
        'mean_interval_width': mean_interval_width(lower, upper),
        'median_interval_width': np.median(upper - lower),
        'winkler_score': winkler_score(y_true, lower, upper, alpha),
        'mean_absolute_error': np.mean(np.abs(y_true - point_pred)),
    }
    if regime_labels is not None:
        calm_mask = regime_labels == 0
        vol_mask = regime_labels == 1
        if calm_mask.sum() > 0:
            metrics['coverage_calm'] = empirical_coverage(y_true[calm_mask], lower[calm_mask], upper[calm_mask])
        if vol_mask.sum() > 0:
            metrics['coverage_volatile'] = empirical_coverage(y_true[vol_mask], lower[vol_mask], upper[vol_mask])
            metrics['conditional_gap'] = metrics.get('coverage_calm', 0) - metrics['coverage_volatile']
    return metrics
