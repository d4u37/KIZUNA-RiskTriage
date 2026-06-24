"""
KIZUNA RISKTRIAGE — FULL PIPELINE
Run: python notebooks/full_pipeline.py
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import create_synthetic_m5_like_data, train_test_split_temporal
from src.features import build_feature_matrix, get_feature_columns
from src.models import QuantileForecaster, PointForecaster
from src.calibration import SplitConformal, CQR, AdaptiveConformalInference
from src.metrics import (empirical_coverage, mean_interval_width, winkler_score,
    calibration_curve_data, compute_all_metrics, relative_interval_width)
from src.risk_triage import RiskTierClassifier, validate_risk_tiers, sensitivity_analysis
from src.simulation import NewsvendorSimulation
from src.audit import underestimation_audit, generate_audit_report

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
NOMINAL_LEVEL = 0.95

print("=" * 70)
print("  KIZUNA RISKTRIAGE — Calibrated Uncertainty for Supply Chain Risk")
print("=" * 70)

# STEP 1: DATA
print("\n[STEP 1] DATA LOADING")
df = create_synthetic_m5_like_data(n_items=50, n_days=1000, seed=42)

# STEP 2: FEATURES
print("\n[STEP 2] FEATURE ENGINEERING")
df = build_feature_matrix(df)
feature_cols = get_feature_columns(df)

# STEP 3: SPLIT
print("\n[STEP 3] TEMPORAL SPLIT")
train_df, cal_df, test_df = train_test_split_temporal(df, test_days=180, cal_days=90)
X_train, y_train = train_df[feature_cols].values, train_df['sales'].values
X_cal, y_cal = cal_df[feature_cols].values, cal_df['sales'].values
X_test, y_test = test_df[feature_cols].values, test_df['sales'].values
vol_labels_test = test_df['is_volatile'].values


# STEP 4: MODEL TRAINING
print("\n[STEP 4] MODEL TRAINING")
quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
qf = QuantileForecaster(quantiles=quantiles)
qf.fit(X_train, y_train)
pf = PointForecaster()
pf.fit(X_train, y_train)

cal_preds = qf.predict(X_cal)
test_preds = qf.predict(X_test)
point_pred_test = np.maximum(qf.models[0.5].predict(X_test), 0)
point_pred_cal = np.maximum(qf.models[0.5].predict(X_cal), 0)
q_lower_cal = cal_preds['q_0.025'].values
q_upper_cal = cal_preds['q_0.975'].values
q_lower_test = test_preds['q_0.025'].values
q_upper_test = test_preds['q_0.975'].values

# STEP 5: CALIBRATION
print("\n[STEP 5] CALIBRATION METHODS")
raw_cov = empirical_coverage(y_test, q_lower_test, q_upper_test)
print(f"  Raw QR coverage: {raw_cov:.3f}")

cal_residuals = np.abs(y_cal - point_pred_cal)
sc = SplitConformal(level=NOMINAL_LEVEL)
sc.calibrate(cal_residuals)
sc_lower, sc_upper = sc.predict_interval(point_pred_test)
sc_cov = empirical_coverage(y_test, sc_lower, sc_upper)
print(f"  Split Conformal coverage: {sc_cov:.3f}")

cqr = CQR(level=NOMINAL_LEVEL)
cqr.calibrate(y_cal, q_lower_cal, q_upper_cal)
cqr_lower, cqr_upper = cqr.predict_interval(q_lower_test, q_upper_test)
cqr_cov = empirical_coverage(y_test, cqr_lower, cqr_upper)
print(f"  CQR coverage: {cqr_cov:.3f}")

aci = AdaptiveConformalInference(level=NOMINAL_LEVEL, gamma=0.01)
aci_lower, aci_upper = aci.run_online(y_test, q_lower_test, q_upper_test, cqr.cal_scores)
aci_cov = empirical_coverage(y_test, aci_lower, aci_upper)
print(f"  CQR + ACI coverage: {aci_cov:.3f}")


# STEP 6: CALIBRATION CURVES
print("\n[STEP 6] CALIBRATION CURVES")
cal_curve_data = calibration_curve_data(y_test, test_preds)
print(cal_curve_data.to_string(index=False))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
ax.plot(cal_curve_data['nominal_level'], cal_curve_data['observed_coverage'],
        'ro-', linewidth=2, markersize=8, label='Raw Quantile Regression')
cqr_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
cqr_coverages = []
for level in cqr_levels:
    cqr_temp = CQR(level=level)
    alpha_temp = 1 - level
    lq = f'q_{min(quantiles, key=lambda x: abs(x - alpha_temp/2))}'
    uq = f'q_{min(quantiles, key=lambda x: abs(x - (1-alpha_temp/2)))}'
    cqr_temp.calibrate(y_cal, cal_preds[lq].values, cal_preds[uq].values)
    cl, cu = cqr_temp.predict_interval(test_preds[lq].values, test_preds[uq].values)
    cqr_coverages.append(empirical_coverage(y_test, cl, cu))
ax.plot(cqr_levels, cqr_coverages, 'bs-', linewidth=2, markersize=8, label='CQR (Ours)')
ax.set_xlabel('Nominal Coverage Level')
ax.set_ylabel('Observed Coverage')
ax.set_title('Calibration Curve')
ax.legend()
ax.set_xlim([0.4, 1.0]); ax.set_ylim([0.4, 1.0])
ax.grid(True, alpha=0.3); ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'calibration_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/calibration_curve.png")

# STEP 7: UNDERESTIMATION AUDIT
print("\n[STEP 7] UNDERESTIMATION AUDIT")
intervals_dict = {
    'Raw QR': (q_lower_test, q_upper_test),
    'Split Conformal': (sc_lower, sc_upper),
    'CQR': (cqr_lower, cqr_upper),
    'CQR + ACI': (aci_lower, aci_upper)
}
audit_results = underestimation_audit(y_test, intervals_dict, vol_labels_test, NOMINAL_LEVEL)
report = generate_audit_report(audit_results, NOMINAL_LEVEL)
with open(RESULTS_DIR / 'audit_report.txt', 'w') as f:
    f.write(report)

fig, ax = plt.subplots(figsize=(10, 6))
methods = audit_results['Method'].values
x = np.arange(len(methods))
width = 0.25
ax.bar(x - width, audit_results['Coverage (Overall)'].values, width, label='Overall', color='steelblue')
ax.bar(x, audit_results['Coverage (Calm)'].values, width, label='Calm', color='forestgreen')
ax.bar(x + width, audit_results['Coverage (Volatile)'].values, width, label='Volatile', color='firebrick')
ax.axhline(y=NOMINAL_LEVEL, color='black', linestyle='--', label=f'Target ({NOMINAL_LEVEL:.0%})')
ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha='right')
ax.set_ylabel('Coverage'); ax.set_title('Conditional Coverage Audit')
ax.legend(); ax.set_ylim([0.5, 1.05]); ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'conditional_coverage_audit.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/conditional_coverage_audit.png")


# STEP 8: RISK TIERS
print("\n[STEP 8] RISK TIER CLASSIFICATION")
rel_widths = relative_interval_width(aci_lower, aci_upper, point_pred_test)
tier_clf = RiskTierClassifier(method='percentile')
tier_clf.fit(rel_widths)
tiers = tier_clf.classify(rel_widths)
for tier in ['Low', 'Medium', 'High']:
    print(f"  {tier}: {np.sum(tiers == tier):,} ({100*np.mean(tiers==tier):.1f}%)")
print("\n--- Tier Validation ---")
tier_validation = validate_risk_tiers(tiers, y_test, aci_upper, point_pred_test)
print(tier_validation.to_string(index=False))
sensitivity_analysis(rel_widths, y_test, aci_upper, point_pred_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
tier_counts = pd.Series(tiers).value_counts()
colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
axes[0].pie(tier_counts.values, labels=tier_counts.index,
            colors=[colors[t] for t in tier_counts.index], autopct='%1.1f%%')
axes[0].set_title('Risk Tier Distribution')
tier_val_data = []
for tier in ['Low', 'Medium', 'High']:
    mask = tiers == tier
    if mask.sum() > 0:
        tier_val_data.append({'Tier': tier, 'Stockout Rate': np.mean(y_test[mask] > aci_upper[mask])})
tier_val_df = pd.DataFrame(tier_val_data)
axes[1].bar(tier_val_df['Tier'], tier_val_df['Stockout Rate'],
            color=[colors[t] for t in tier_val_df['Tier']])
axes[1].set_ylabel('Realized Stockout Rate')
axes[1].set_title('Tier Validation: Higher Tier = Higher Stockout Risk')
axes[1].grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'risk_tiers.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/risk_tiers.png")

# STEP 9: INVENTORY SIMULATION
print("\n[STEP 9] INVENTORY SIMULATION")
sim = NewsvendorSimulation(holding_cost=1.0, stockout_cost=5.0)
comparison = sim.compare(y_test, point_pred_test, pf.residual_std, aci_upper, NOMINAL_LEVEL)
print("\n" + comparison.to_string(index=False))
comparison.to_csv(RESULTS_DIR / 'simulation_comparison.csv', index=False)
robustness = sim.robustness_sweep(y_test, point_pred_test, pf.residual_std, aci_upper)
robustness.to_csv(RESULTS_DIR / 'robustness_sweep.csv', index=False)

baseline_results = sim.simulate_baseline(y_test, point_pred_test, pf.residual_std, NOMINAL_LEVEL)
calibrated_results = sim.simulate_calibrated(y_test, aci_upper, point_pred_test, NOMINAL_LEVEL)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
methods_sim = ['Baseline', 'Calibrated (Ours)']
axes[0].bar(methods_sim, [baseline_results['achieved_service_level'], calibrated_results['achieved_service_level']],
            color=['#95a5a6', '#2980b9'])
axes[0].axhline(y=NOMINAL_LEVEL, color='red', linestyle='--', label='Target')
axes[0].set_ylabel('Service Level'); axes[0].set_title('Service Level'); axes[0].legend()
axes[0].set_ylim([0.7, 1.0]); axes[0].grid(True, axis='y', alpha=0.3)
x_cost = np.arange(2); w = 0.35
axes[1].bar(x_cost - w/2, [baseline_results['total_holding_cost'], baseline_results['total_stockout_cost']], w, label='Baseline', color='#95a5a6')
axes[1].bar(x_cost + w/2, [calibrated_results['total_holding_cost'], calibrated_results['total_stockout_cost']], w, label='Calibrated', color='#2980b9')
axes[1].set_xticks(x_cost); axes[1].set_xticklabels(['Holding Cost', 'Stockout Cost'])
axes[1].set_title('Cost Breakdown'); axes[1].legend(); axes[1].grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'inventory_simulation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/inventory_simulation.png")

# STEP 10: METRICS
print("\n[STEP 10] METRICS SUMMARY")
all_metrics = {}
for name, (l, u) in intervals_dict.items():
    all_metrics[name] = compute_all_metrics(y_test, l, u, point_pred_test, NOMINAL_LEVEL, vol_labels_test)
metrics_df = pd.DataFrame(all_metrics).T
print("\n" + metrics_df.round(4).to_string())
metrics_df.to_csv(RESULTS_DIR / 'all_metrics.csv')

# STEP 11: EXAMPLE FORECAST
print("\n[STEP 11] EXAMPLE FORECAST")
items_in_test = test_df['item_id'].unique()
example_item = items_in_test[0]
item_mask = test_df['item_id'].values == example_item
item_y = y_test[item_mask][-60:]
item_point = point_pred_test[item_mask][-60:]
item_aci_lower = aci_lower[item_mask][-60:]
item_aci_upper = aci_upper[item_mask][-60:]
item_tiers = tiers[item_mask][-60:]

fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(range(len(item_y)), item_aci_lower, item_aci_upper, alpha=0.3, color='blue', label='CQR+ACI (calibrated)')
ax.plot(range(len(item_y)), item_y, 'k.-', linewidth=1, markersize=4, label='Actual demand')
ax.plot(range(len(item_y)), item_point, 'b--', linewidth=1, alpha=0.7, label='Point forecast')
for i in range(len(item_tiers)):
    if item_tiers[i] == 'High':
        ax.axvspan(i-0.5, i+0.5, alpha=0.1, color='red')
ax.set_xlabel('Day'); ax.set_ylabel('Demand (units)')
ax.set_title(f'Example Forecast: {example_item}'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'example_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/example_forecast.png")

print("\n" + "=" * 70)
print("  PIPELINE COMPLETE")
print("=" * 70)
print(f"""
KEY FINDINGS:
  Raw QR coverage: {raw_cov:.1%}  |  CQR+ACI coverage: {aci_cov:.1%}
  Volatile coverage (Raw): {audit_results.loc[audit_results['Method']=='Raw QR', 'Coverage (Volatile)'].values[0]:.1%}
  Volatile coverage (ACI): {audit_results.loc[audit_results['Method']=='CQR + ACI', 'Coverage (Volatile)'].values[0]:.1%}
  Baseline service level: {baseline_results['achieved_service_level']:.1%}
  Calibrated service level: {calibrated_results['achieved_service_level']:.1%}
  Cost savings: {100*(baseline_results['total_cost'] - calibrated_results['total_cost'])/baseline_results['total_cost']:.1f}%
""")
print("Done! Ready for submission.")
