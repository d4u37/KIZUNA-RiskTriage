"""
Feature engineering for demand forecasting.
Creates lag features, rolling statistics, calendar features, and
volatility indicators needed for LightGBM quantile regression.
"""

import pandas as pd
import numpy as np
from typing import List


def create_lag_features(df, lags=[1, 7, 14, 28], target_col='sales'):
    df = df.sort_values(['item_id', 'date']).copy()
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('item_id')[target_col].shift(lag)
    return df


def create_rolling_features(df, windows=[7, 14, 28, 56], target_col='sales'):
    df = df.sort_values(['item_id', 'date']).copy()
    for window in windows:
        rolled = df.groupby('item_id')[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        df[f'rolling_mean_{window}'] = rolled
        rolled_std = df.groupby('item_id')[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std())
        df[f'rolling_std_{window}'] = rolled_std
        rolled_max = df.groupby('item_id')[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max())
        df[f'rolling_max_{window}'] = rolled_max
    return df


def create_calendar_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    df['is_month_start'] = (df['date'].dt.day <= 5).astype(int)
    df['is_month_end'] = (df['date'].dt.day >= 25).astype(int)
    df['quarter'] = df['date'].dt.quarter
    return df


def create_volatility_label(df, window=28, threshold_percentile=75, target_col='sales'):
    df = df.copy()
    df['rolling_vol'] = df.groupby('item_id')[target_col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=7).std())
    df['rolling_cv'] = df['rolling_vol'] / (
        df.groupby('item_id')[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=7).mean()) + 1e-6)
    thresholds = df.groupby('item_id')['rolling_vol'].transform(
        lambda x: x.quantile(threshold_percentile / 100))
    event_cols = [c for c in ['is_snap', 'is_holiday', 'is_promo', 'has_event'] if c in df.columns]
    event_flag = df[event_cols].max(axis=1) if event_cols else 0
    df['is_volatile'] = ((df['rolling_vol'] > thresholds) | (event_flag == 1)).astype(int)
    df['volatility_score'] = df['rolling_cv'].fillna(0)
    return df


def build_feature_matrix(df, target_col='sales'):
    print("Building features...")
    df = create_calendar_features(df)
    print("  Calendar features added.")
    df = create_lag_features(df, lags=[1, 2, 3, 7, 14, 21, 28])
    print("  Lag features added.")
    df = create_rolling_features(df, windows=[7, 14, 28, 56])
    print("  Rolling statistics added.")
    df = create_volatility_label(df, window=28, threshold_percentile=75)
    print("  Volatility labels added.")
    initial_rows = len(df)
    df = df.dropna(subset=['lag_28', 'rolling_mean_28']).reset_index(drop=True)
    print(f"  Dropped {initial_rows - len(df)} rows with insufficient history.")
    print(f"  Final feature matrix: {len(df):,} rows, {len(df.columns)} columns")
    return df


def get_feature_columns(df):
    exclude_cols = {'item_id', 'date', 'sales', 'd', 'id', 'dept_id', 'cat_id',
                    'store_id', 'state_id', 'day_num', 'event_name_1', 'event_type_1',
                    'wm_yr_wk', 'weekday', 'wday', 'is_volatile', 'volatility_score',
                    'rolling_vol', 'rolling_cv', 'snap_CA', 'snap_TX', 'snap_WI'}
    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
    return feature_cols
