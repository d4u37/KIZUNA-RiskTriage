"""
Risk Tier Classification for Kizuna RiskTriage.
"""

import numpy as np
import pandas as pd
from typing import Optional


class RiskTierClassifier:
    def __init__(self, method='percentile', low_threshold=None, high_threshold=None):
        self.method = method
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.fitted = False

    def fit(self, relative_widths):
        if self.method == 'percentile':
            self.low_threshold = np.percentile(relative_widths, 33)
            self.high_threshold = np.percentile(relative_widths, 67)
        self.fitted = True
        print(f"Risk tier thresholds fitted:")
        print(f"  LOW:    relative_width < {self.low_threshold:.3f}")
        print(f"  MEDIUM: {self.low_threshold:.3f} <= relative_width < {self.high_threshold:.3f}")
        print(f"  HIGH:   relative_width >= {self.high_threshold:.3f}")
        return self

    def classify(self, relative_widths):
        return np.where(relative_widths < self.low_threshold, 'Low',
                        np.where(relative_widths < self.high_threshold, 'Medium', 'High'))

    def get_actions(self, tiers):
        action_map = {
            'Low': 'Auto-replenish (nominal safety stock)',
            'Medium': 'Raise safety stock to calibrated quantile; dashboard flag',
            'High': 'ESCALATE: Human review required; consider dual-sourcing'
        }
        return np.array([action_map[t] for t in tiers])


def validate_risk_tiers(tiers, y_true, upper_bound, point_pred):
    results = []
    for tier in ['Low', 'Medium', 'High']:
        mask = tiers == tier
        if mask.sum() == 0:
            continue
        stockout_rate = np.mean(y_true[mask] > upper_bound[mask])
        mae = np.mean(np.abs(y_true[mask] - point_pred[mask]))
        mape = np.mean(np.abs(y_true[mask] - point_pred[mask]) / (y_true[mask] + 1))
        demand_cv = np.std(y_true[mask]) / (np.mean(y_true[mask]) + 1e-6)
        results.append({
            'Risk Tier': tier, 'Count': int(mask.sum()),
            'Pct of Total': f"{100 * mask.sum() / len(tiers):.1f}%",
            'Stockout Rate': f"{100 * stockout_rate:.1f}%",
            'Mean Abs Error': f"{mae:.2f}", 'MAPE': f"{100 * mape:.1f}%",
            'Demand CV': f"{demand_cv:.3f}"
        })
    df = pd.DataFrame(results)
    stockout_rates = [float(r['Stockout Rate'].replace('%', '')) for r in results]
    if len(stockout_rates) >= 2 and stockout_rates[-1] > stockout_rates[0]:
        print("\n  VALIDATED: High-tier items have higher stockout rates than Low-tier.")
    else:
        print("\n  WARNING: Tier validation may need threshold adjustment.")
    return df


def sensitivity_analysis(relative_widths, y_true, upper_bound, point_pred):
    results = []
    for low_p, high_p in [(25, 60), (30, 65), (33, 67), (35, 70), (40, 75)]:
        low_thresh = np.percentile(relative_widths, low_p)
        high_thresh = np.percentile(relative_widths, high_p)
        tiers = np.where(relative_widths < low_thresh, 'Low',
                         np.where(relative_widths < high_thresh, 'Medium', 'High'))
        for tier in ['Low', 'Medium', 'High']:
            mask = tiers == tier
            if mask.sum() > 0:
                stockout_rate = np.mean(y_true[mask] > upper_bound[mask])
                results.append({'Threshold Setting': f'P{low_p}/P{high_p}', 'Tier': tier,
                                'Stockout Rate': stockout_rate, 'N': int(mask.sum())})
    df = pd.DataFrame(results)
    all_monotonic = True
    for setting in df['Threshold Setting'].unique():
        sub = df[df['Threshold Setting'] == setting]
        rates = sub.set_index('Tier')['Stockout Rate']
        if 'Low' in rates.index and 'High' in rates.index:
            if rates['High'] <= rates['Low']:
                all_monotonic = False
    if all_monotonic:
        print("\n  ROBUST: Tier ordering holds across all threshold settings.")
    return df
