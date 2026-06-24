"""
Data loading and preprocessing for M5 Forecasting dataset.

The M5 dataset contains daily sales of 30,490 Walmart products across 10 stores
over ~5.4 years (2011-01-29 to 2016-06-19).

For hackathon feasibility, we work with a strategic subset:
- Select stores with diverse characteristics
- Focus on product categories with meaningful demand patterns
- Ensure we capture volatility regimes (SNAP, holidays, promotions)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


DATA_DIR = Path(__file__).parent.parent / "data"


def create_synthetic_m5_like_data(n_items: int = 50, n_days: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic data that mimics M5 characteristics for development/demo.

    Generates data with:
    - Weekly seasonality
    - Yearly seasonality
    - Random promotions causing demand spikes
    - SNAP-like benefit days
    - Holiday effects
    - Some intermittent (sparse) demand items
    - Varying volatility across items
    """
    np.random.seed(seed)

    dates = pd.date_range('2013-01-01', periods=n_days, freq='D')
    items = [f"FOOD_{i:03d}" for i in range(n_items)]

    records = []

    for item in items:
        base = np.random.uniform(5, 50)
        noise_scale = np.random.uniform(0.1, 0.5) * base
        is_intermittent = np.random.random() < 0.2

        for t, date in enumerate(dates):
            dow_effect = 1.0 + 0.3 * (date.dayofweek >= 5)
            month_effect = 1.0 + 0.2 * np.sin(2 * np.pi * date.month / 12)
            is_snap = int(date.day <= 10 and np.random.random() < 0.3)
            snap_effect = 1.0 + 0.4 * is_snap
            is_promo = int(np.random.random() < 0.05)
            promo_effect = 1.0 + is_promo * np.random.uniform(0.5, 2.0)
            is_holiday = int(date.month == 12 and date.day >= 20) or \
                         int(date.month == 11 and date.day >= 22 and date.day <= 28) or \
                         int(date.month == 7 and date.day == 4) or \
                         int(date.month == 2 and date.day == 14)
            holiday_effect = 1.0 + 1.5 * int(is_holiday)
            trend = 1.0 + 0.0001 * t

            demand = base * dow_effect * month_effect * snap_effect * promo_effect * holiday_effect * trend
            demand += np.random.normal(0, noise_scale)
            demand = max(0, demand)

            if is_intermittent and np.random.random() < 0.6:
                demand = 0

            demand = int(round(demand))

            records.append({
                'item_id': item,
                'date': date,
                'sales': demand,
                'weekday': date.dayofweek,
                'month': date.month,
                'year': date.year,
                'day_of_month': date.day,
                'is_snap': is_snap,
                'is_promo': is_promo,
                'is_holiday': int(is_holiday),
                'has_event': int(is_holiday) | is_promo,
                'store_id': 'CA_1',
                'cat_id': 'FOODS'
            })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])

    print(f"Synthetic dataset created: {n_items} items, {n_days} days, {len(df):,} rows")
    print(f"  - Intermittent items: ~{int(n_items * 0.2)}")
    print(f"  - Promo days per item: ~{int(n_days * 0.05)}")
    print(f"  - Demand range: {df['sales'].min()} to {df['sales'].max()}")

    return df


def train_test_split_temporal(df: pd.DataFrame,
                              test_days: int = 180,
                              cal_days: int = 90) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/calibration/test split.

    Split structure:
    [======= TRAIN =======][== CALIBRATION ==][==== TEST ====]
    """
    dates = sorted(df['date'].unique())
    n_dates = len(dates)

    test_start_idx = n_dates - test_days
    cal_start_idx = test_start_idx - cal_days

    test_start_date = dates[test_start_idx]
    cal_start_date = dates[cal_start_idx]

    train_df = df[df['date'] < cal_start_date].copy()
    cal_df = df[(df['date'] >= cal_start_date) & (df['date'] < test_start_date)].copy()
    test_df = df[df['date'] >= test_start_date].copy()

    print(f"Temporal split:")
    print(f"  Train: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({train_df['date'].nunique()} days)")
    print(f"  Calibration: {cal_df['date'].min().date()} to {cal_df['date'].max().date()} ({cal_df['date'].nunique()} days)")
    print(f"  Test: {test_df['date'].min().date()} to {test_df['date'].max().date()} ({test_df['date'].nunique()} days)")

    return train_df, cal_df, test_df
