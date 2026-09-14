#!/usr/bin/env python3
"""
AI4I 2020 Predictive Maintenance - Data Pipeline
Raw -> Cleaned -> Feature-Ready
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned" / "ai4i2020_cleaned.csv"
FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "ai4i2020_features.csv"

CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_raw():
    """Load raw dataset."""
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded raw data: {df.shape}")
    return df


def clean_data(df):
    """Clean the dataset - no nulls in this dataset, but apply standard cleaning."""
    df = df.copy()
    
    # Rename columns to snake_case for SQL compatibility
    col_map = {
        'UDI': 'udi',
        'Product ID': 'product_id',
        'Type': 'type',
        'Air temperature [K]': 'air_temp_k',
        'Process temperature [K]': 'process_temp_k',
        'Rotational speed [rpm]': 'rotational_speed_rpm',
        'Torque [Nm]': 'torque_nm',
        'Tool wear [min]': 'tool_wear_min',
        'Machine failure': 'machine_failure',
        'TWF': 'twf',
        'HDF': 'hdf',
        'PWF': 'pwf',
        'OSF': 'osf',
        'RNF': 'rnf'
    }
    df = df.rename(columns=col_map)
    
    # Convert type to categorical
    df['type'] = df['type'].astype('category')
    
    # Add derived temperature difference
    df['temp_diff_k'] = df['process_temp_k'] - df['air_temp_k']
    
    # Add power proxy (torque * rotational_speed / 9550 ≈ kW)
    df['power_kw'] = df['torque_nm'] * df['rotational_speed_rpm'] / 9550
    
    # Add tool wear rate (wear per unit time proxy)
    df['wear_rate'] = df['tool_wear_min'] / (df['rotational_speed_rpm'] + 1e-6)
    
    print(f"Cleaned data: {df.shape}")
    return df


def engineer_features(df):
    """Create feature-rich dataset for modeling."""
    df = df.copy()
    
    # Rolling statistics per product_id (proxy for machine)
    # Since we don't have timestamps, use UDI as sequence
    df = df.sort_values('udi').reset_index(drop=True)
    
    # Per-machine rolling stats (using Product ID as machine identifier)
    for col in ['air_temp_k', 'process_temp_k', 'rotational_speed_rpm', 'torque_nm', 'tool_wear_min']:
        df[f'{col}_rolling_mean_10'] = df.groupby('product_id')[col].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )
        df[f'{col}_rolling_std_10'] = df.groupby('product_id')[col].transform(
            lambda x: x.rolling(10, min_periods=1).std()
        )
    
    # Failure lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'machine_failure_lag_{lag}'] = df.groupby('product_id')['machine_failure'].shift(lag).fillna(0).astype(int)
    
    # Specific failure type lags
    for fail_col in ['twf', 'hdf', 'pwf', 'osf', 'rnf']:
        df[f'{fail_col}_lag_1'] = df.groupby('product_id')[fail_col].shift(1).fillna(0).astype(int)
    
    # Cumulative failure counts per machine
    df['cum_machine_failures'] = df.groupby('product_id')['machine_failure'].cumsum().shift(1).fillna(0).astype(int)
    
    # Temperature ratios
    df['temp_ratio'] = df['process_temp_k'] / df['air_temp_k']
    
    # Torque per RPM
    df['torque_per_rpm'] = df['torque_nm'] / (df['rotational_speed_rpm'] + 1e-6)
    
    # Tool wear bins
    df['tool_wear_bin'] = pd.cut(df['tool_wear_min'], bins=[-1, 50, 100, 150, 200, 250], 
                                  labels=['0-50', '51-100', '101-150', '151-200', '201-250'])
    
    # Interaction features
    df['temp_x_speed'] = df['temp_diff_k'] * df['rotational_speed_rpm']
    df['torque_x_wear'] = df['torque_nm'] * df['tool_wear_min']
    
    # Type one-hot encoding
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, type_dummies], axis=1)
    
    print(f"Feature-engineered data: {df.shape}")
    return df


def main():
    print("=" * 60)
    print("AI4I 2020 Predictive Maintenance - Data Pipeline")
    print("=" * 60)
    
    # Step 1: Load raw
    raw = load_raw()
    
    # Step 2: Clean
    cleaned = clean_data(raw)
    cleaned.to_csv(CLEANED_PATH, index=False)
    print(f"Saved cleaned data to: {CLEANED_PATH}")
    
    # Step 3: Feature engineering
    features = engineer_features(cleaned)
    features.to_csv(FEATURE_PATH, index=False)
    print(f"Saved feature data to: {FEATURE_PATH}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Raw:       {raw.shape[0]} rows x {raw.shape[1]} cols")
    print(f"Cleaned:   {cleaned.shape[0]} rows x {cleaned.shape[1]} cols")
    print(f"Features:  {features.shape[0]} rows x {features.shape[1]} cols")
    print(f"\nTarget distribution:")
    print(features['machine_failure'].value_counts())
    print(f"\nFeature columns: {list(features.columns)}")


if __name__ == "__main__":
    main()