"""
Model Comparison Script
Compare XGBoost vs LightGBM predictions side-by-side
"""

import pandas as pd
from ml.predict import load_model as load_xgb_model
from ml.predict_lightgbm import load_lightgbm_model

def compare_predictions():
    """Compare predictions from both models for the same customer."""
    
    # Load both models
    xgb_predictor = load_xgb_model()
    lgb_predictor = load_lightgbm_model()
    
    # Test customer data
    test_customers = [
        {
            "name": "Low Risk Customer",
            "data": {
                "salary_delay_days": 0,
                "savings_drop_pct": 0.05,
                "utility_payment_delay_days": 0,
                "discretionary_spend_drop_pct": 0.02,
                "atm_withdrawal_increase": 0,
                "upi_lending_txn_count": 0,
                "failed_autodebit_count": 0
            }
        },
        {
            "name": "Medium Risk Customer",
            "data": {
                "salary_delay_days": 2,
                "savings_drop_pct": 0.15,
                "utility_payment_delay_days": 1,
                "discretionary_spend_drop_pct": 0.10,
                "atm_withdrawal_increase": 1,
                "upi_lending_txn_count": 1,
                "failed_autodebit_count": 0
            }
        },
        {
            "name": "High Risk Customer",
            "data": {
                "salary_delay_days": 5,
                "savings_drop_pct": 0.40,
                "utility_payment_delay_days": 4,
                "discretionary_spend_drop_pct": 0.35,
                "atm_withdrawal_increase": 4,
                "upi_lending_txn_count": 3,
                "failed_autodebit_count": 2
            }
        },
        {
            "name": "Critical Risk Customer",
            "data": {
                "salary_delay_days": 10,
                "savings_drop_pct": 0.70,
                "utility_payment_delay_days": 7,
                "discretionary_spend_drop_pct": 0.60,
                "atm_withdrawal_increase": 8,
                "upi_lending_txn_count": 6,
                "failed_autodebit_count": 4
            }
        }
    ]
    
    print("=" * 80)
    print("📊 MODEL COMPARISON: XGBoost vs LightGBM Predictions")
    print("=" * 80)
    print()
    
    # Compare each customer
    for customer in test_customers:
        print(f"👤 {customer['name']}")
        print("-" * 50)
        
        # Get predictions
        xgb_result = xgb_predictor.predict(customer['data'])
        lgb_result = lgb_predictor.predict(customer['data'])
        
        # Display comparison
        print(f"{'Metric':<20} {'XGBoost':<15} {'LightGBM':<15} {'Difference':<12}")
        print("-" * 70)
        print(f"{'Risk Score':<20} {xgb_result['risk_score']:<15.4f} {lgb_result['risk_score']:<15.4f} {abs(xgb_result['risk_score'] - lgb_result['risk_score']):<12.4f}")
        print(f"{'Risk Level':<20} {xgb_result['risk_level']:<15} {lgb_result['risk_level']:<15} {'-':<12}")
        print(f"{'Will Default':<20} {str(xgb_result['will_default']):<15} {str(lgb_result['will_default']):<15} {'-':<12}")
        print(f"{'Confidence':<20} {xgb_result['confidence']:<15.4f} {lgb_result['confidence']:<15.4f} {abs(xgb_result['confidence'] - lgb_result['confidence']):<12.4f}")
        print()

if __name__ == "__main__":
    compare_predictions()