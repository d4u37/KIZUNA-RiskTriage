"""
Inventory Decision Simulation for Kizuna RiskTriage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.stats import norm


class NewsvendorSimulation:
    def __init__(self, holding_cost=1.0, stockout_cost=5.0):
        self.co = holding_cost
        self.cu = stockout_cost

    def simulate_baseline(self, y_true, point_pred, residual_std, service_level=0.95):
        z = norm.ppf(service_level)
        order_qty = np.maximum(point_pred + z * residual_std, 0)
        excess = np.maximum(order_qty - y_true, 0)
        shortage = np.maximum(y_true - order_qty, 0)
        holding_costs = excess * self.co
        stockout_costs = shortage * self.cu
        total_costs = holding_costs + stockout_costs
        fulfilled = np.minimum(order_qty, y_true)
        achieved_service_level = np.sum(fulfilled) / np.sum(y_true)
        stockout_frequency = np.mean(shortage > 0)
        return {
            'method': 'Baseline (Point + z*sigma)',
            'total_cost': np.sum(total_costs),
            'total_holding_cost': np.sum(holding_costs),
            'total_stockout_cost': np.sum(stockout_costs),
            'achieved_service_level': achieved_service_level,
            'stockout_frequency': stockout_frequency,
            'avg_excess_inventory': np.mean(excess),
        }


    def simulate_calibrated(self, y_true, calibrated_upper, point_pred, service_level=0.95):
        order_qty = np.maximum(calibrated_upper.copy(), 0)
        excess = np.maximum(order_qty - y_true, 0)
        shortage = np.maximum(y_true - order_qty, 0)
        holding_costs = excess * self.co
        stockout_costs = shortage * self.cu
        total_costs = holding_costs + stockout_costs
        fulfilled = np.minimum(order_qty, y_true)
        achieved_service_level = np.sum(fulfilled) / np.sum(y_true)
        stockout_frequency = np.mean(shortage > 0)
        return {
            'method': 'Calibrated (CQR+ACI upper bound)',
            'total_cost': np.sum(total_costs),
            'total_holding_cost': np.sum(holding_costs),
            'total_stockout_cost': np.sum(stockout_costs),
            'achieved_service_level': achieved_service_level,
            'stockout_frequency': stockout_frequency,
            'avg_excess_inventory': np.mean(excess),
        }

    def compare(self, y_true, point_pred, residual_std, calibrated_upper, service_level=0.95):
        baseline = self.simulate_baseline(y_true, point_pred, residual_std, service_level)
        calibrated = self.simulate_calibrated(y_true, calibrated_upper, point_pred, service_level)
        comparison = pd.DataFrame([
            {'Metric': 'Achieved Service Level', 'Baseline': f"{100*baseline['achieved_service_level']:.1f}%", 'Calibrated (Ours)': f"{100*calibrated['achieved_service_level']:.1f}%"},
            {'Metric': 'Stockout Frequency', 'Baseline': f"{100*baseline['stockout_frequency']:.1f}%", 'Calibrated (Ours)': f"{100*calibrated['stockout_frequency']:.1f}%"},
            {'Metric': 'Avg Excess Inventory', 'Baseline': f"{baseline['avg_excess_inventory']:.1f}", 'Calibrated (Ours)': f"{calibrated['avg_excess_inventory']:.1f}"},
            {'Metric': 'Total Cost', 'Baseline': f"${baseline['total_cost']:,.0f}", 'Calibrated (Ours)': f"${calibrated['total_cost']:,.0f}"},
        ])
        return comparison

    def robustness_sweep(self, y_true, point_pred, residual_std, calibrated_upper, cost_ratios=None, service_level=0.95):
        if cost_ratios is None:
            cost_ratios = [2, 3, 5, 10, 20]
        results = []
        for ratio in cost_ratios:
            self.co = 1.0
            self.cu = ratio
            baseline = self.simulate_baseline(y_true, point_pred, residual_std, service_level)
            calibrated = self.simulate_calibrated(y_true, calibrated_upper, point_pred, service_level)
            savings_pct = 100 * (baseline['total_cost'] - calibrated['total_cost']) / baseline['total_cost']
            results.append({
                'Cost Ratio (cu/co)': ratio,
                'Baseline Cost': baseline['total_cost'],
                'Calibrated Cost': calibrated['total_cost'],
                'Cost Savings (%)': f"{savings_pct:.1f}%",
                'Calibrated Wins': savings_pct > 0
            })
        df = pd.DataFrame(results)
        wins = df['Calibrated Wins'].sum()
        print(f"\n  Robustness check: Calibrated approach wins in {wins}/{len(cost_ratios)} cost scenarios")
        return df
