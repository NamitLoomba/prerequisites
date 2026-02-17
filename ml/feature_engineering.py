import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_raw_data(filepath="data/raw/synthetic_transactions.csv"):
    """Load the raw synthetic transaction data."""
    df = pd.read_csv(filepath)
    return df

def create_stress_signals(df):
    """
    Create advanced stress signals from raw features.
    These engineered features capture financial behavior patterns.
    """
    df = df.copy()
    
    # Financial Stress Index (composite score)
    df['stress_index'] = (
        df['salary_delay_days'] * 1.5 +
        df['savings_drop_pct'] * 10 +
        df['utility_payment_delay_days'] * 1.0 +
        df['atm_withdrawal_increase'] * 0.8 +
        df['failed_autodebit_count'] * 2.0 +
        df['upi_lending_txn_count'] * 1.2
    )
    
    # Liquidity Ratio (higher = better)
    df['liquidity_ratio'] = (
        1 - df['savings_drop_pct']
    ) / (df['atm_withdrawal_increase'] + 1)
    
    # Payment Reliability Score (lower = worse)
    df['payment_reliability'] = (
        10 - df['utility_payment_delay_days'] - 
        df['failed_autodebit_count'] * 3
    )
    
    # Cash Flow Pressure (higher = worse)
    df['cash_flow_pressure'] = (
        df['salary_delay_days'] * 2 +
        df['atm_withdrawal_increase'] +
        df['upi_lending_txn_count']
    )
    
    # Savings Behavior (higher = concerning)
    df['savings_behavior'] = (
        df['savings_drop_pct'] + 
        df['discretionary_spend_drop_pct']
    )
    
    # Digital Activity Stress
    df['digital_stress'] = (
        df['upi_lending_txn_count'] * 1.5 +
        df['failed_autodebit_count']
    )
    
    return df

def get_feature_columns():
    """Return the list of feature columns for model training."""
    return [
        'salary_delay_days',
        'savings_drop_pct',
        'utility_payment_delay_days',
        'discretionary_spend_drop_pct',
        'atm_withdrawal_increase',
        'upi_lending_txn_count',
        'failed_autodebit_count',
        'stress_index',
        'liquidity_ratio',
        'payment_reliability',
        'cash_flow_pressure',
        'savings_behavior',
        'digital_stress'
    ]

def get_raw_features():
    """Return only raw features (before engineering)."""
    return [
        'salary_delay_days',
        'savings_drop_pct',
        'utility_payment_delay_days',
        'discretionary_spend_drop_pct',
        'atm_withdrawal_increase',
        'upi_lending_txn_count',
        'failed_autodebit_count'
    ]

def process_data(input_df=None, fit_scaler=False, scaler=None):
    """
    Process data with feature engineering.
    
    Args:
        input_df: DataFrame to process (if None, loads raw data)
        fit_scaler: Whether to fit a new scaler
        scaler: Existing scaler to use
    
    Returns:
        Processed DataFrame, fitted scaler
    """
    if input_df is None:
        df = load_raw_data()
    else:
        df = input_df.copy()
    
    # Apply feature engineering
    df = create_stress_signals(df)
    
    feature_cols = get_feature_columns()
    
    # Scale features
    X = df[feature_cols].values
    
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    df[feature_cols] = X_scaled
    
    return df, scaler

def save_processed_data(df, filepath="data/processed/features.csv"):
    """Save processed features to CSV."""
    df.to_csv(filepath, index=False)
    print(f"✅ Processed data saved to {filepath}")

if __name__ == "__main__":
    # Test feature engineering
    df = load_raw_data()
    df_processed, scaler = process_data(df, fit_scaler=True)
    save_processed_data(df_processed)
    print(f"📊 Features created: {len(get_feature_columns())}")
    print(f"📊 Feature columns: {get_feature_columns()}")
