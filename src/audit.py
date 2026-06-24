"""
Underestimation Audit for Kizuna RiskTriage.
Implements Objective 5: Check whether the system underestimates uncertainty
in high-volatility periods.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def underestimation_audit(y_true, intervals, volatility_labels, nominal_level=0.95):
    calm_mask = volatility_labels == 0
    vol_mask = volatility_labels == 1
    results = []
    for method_name, (lower, upper) in intervals.items():
        overall_cov = np.mean((y_true >= lower) & (y_true <= upper))
        calm_cov = np.mean((y_true[calm_mask] >= lower[calm_mask]) &
                           (y_true[calm_mask] <= upper[calm_mask])) if calm_mask.sum() > 0 else np.nan
        vol_cov = np.mean((y_true[vol_mask] >= lower[vol_mask]) &
                          (y_true[vol_mask] <= upper[vol_mask])) if vol_mask.sum() > 0 else np.nan
        overall_width = np.mean(upper - lower)
        results.append({
            'Method': method_name,
            'Coverage (Overall)': overall_cov,
            'Coverage (Calm)': calm_cov,
            'Coverage (Volatile)': vol_cov,
            'Gap (Calm - Volatile)': calm_cov - vol_cov if not np.isnan(vol_cov) else np.nan,
            'Width (Overall)': overall_width,
            'Underestimates in Volatility?': 'YES' if vol_cov < nominal_level - 0.05 else 'No'
        })
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("UNDERESTIMATION AUDIT RESULTS")
    print("=" * 70)

    print(f"\nNominal coverage level: {nominal_level:.0%}")
    print(f"Calm periods: {calm_mask.sum()} observations")
    print(f"Volatile periods: {vol_mask.sum()} observations")
    print(f"\nTarget: Coverage should be ~{nominal_level:.0%} in BOTH regimes.\n")
    display_cols = ['Method', 'Coverage (Overall)', 'Coverage (Calm)',
                    'Coverage (Volatile)', 'Gap (Calm - Volatile)',
                    'Underestimates in Volatility?']
    print(df[display_cols].to_string(index=False))
    worst_gap = df['Gap (Calm - Volatile)'].max()
    worst_method = df.loc[df['Gap (Calm - Volatile)'].idxmax(), 'Method']
    best_gap = df['Gap (Calm - Volatile)'].min()
    best_method = df.loc[df['Gap (Calm - Volatile)'].idxmin(), 'Method']
    print(f"\n  Worst: {worst_method} (gap={worst_gap:.3f})")
    print(f"  Best:  {best_method} (gap={best_gap:.3f})")
    print("=" * 70)
    return df


def generate_audit_report(audit_results, nominal_level=0.95):
    report = []
    report.append("UNCERTAINTY UNDERESTIMATION AUDIT — SUMMARY REPORT")
    report.append(f"Target coverage: {nominal_level:.0%}\n")
    for _, row in audit_results.iterrows():
        report.append(f"Method: {row['Method']}")
        report.append(f"  Volatile coverage: {row['Coverage (Volatile)']:.1%}")
        report.append(f"  Gap: {row['Gap (Calm - Volatile)']:.3f}")
        report.append(f"  Underestimates: {row['Underestimates in Volatility?']}\n")
    best_method = audit_results.loc[audit_results['Gap (Calm - Volatile)'].idxmin(), 'Method']
    report.append(f"CONCLUSION: {best_method} provides best conditional calibration.")
    return '\n'.join(report)
