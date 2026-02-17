"""
Test script for LightGBM API integration
Run the API first: python backend/main.py
Then run this script: python test_api_integration.py
"""

import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_models_status():
    """Check which models are available."""
    print("=" * 60)
    print("🔍 Checking Model Status")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/models/status")
    print(json.dumps(response.json(), indent=2))
    print()

def test_xgboost_prediction():
    """Test XGBoost prediction."""
    print("=" * 60)
    print("🤖 Testing XGBoost Prediction")
    print("=" * 60)
    
    customer = {
        "customer_id": "CUST_001",
        "salary_delay_days": 4,
        "savings_drop_pct": 0.30,
        "utility_payment_delay_days": 3,
        "discretionary_spend_drop_pct": 0.25,
        "atm_withdrawal_increase": 3,
        "upi_lending_txn_count": 2,
        "failed_autodebit_count": 1,
        "model_type": "xgboost"
    }
    
    response = requests.post(f"{BASE_URL}/predict-risk", json=customer)
    result = response.json()
    
    print(f"Customer ID: {result['customer_id']}")
    print(f"Model: {result['model_type']}")
    print(f"Risk Score: {result['risk_score']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Will Default: {result['will_default']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Action: {result['recommended_action']}")
    print()

def test_lightgbm_prediction():
    """Test LightGBM prediction."""
    print("=" * 60)
    print("⚡ Testing LightGBM Prediction")
    print("=" * 60)
    
    customer = {
        "customer_id": "CUST_002",
        "salary_delay_days": 4,
        "savings_drop_pct": 0.30,
        "utility_payment_delay_days": 3,
        "discretionary_spend_drop_pct": 0.25,
        "atm_withdrawal_increase": 3,
        "upi_lending_txn_count": 2,
        "failed_autodebit_count": 1,
        "model_type": "lightgbm"
    }
    
    response = requests.post(f"{BASE_URL}/predict-risk", json=customer)
    result = response.json()
    
    print(f"Customer ID: {result['customer_id']}")
    print(f"Model: {result['model_type']}")
    print(f"Risk Score: {result['risk_score']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Will Default: {result['will_default']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Action: {result['recommended_action']}")
    print()

def test_model_comparison():
    """Test comparing both models."""
    print("=" * 60)
    print("📊 Testing Model Comparison")
    print("=" * 60)
    
    customer = {
        "customer_id": "CUST_003",
        "salary_delay_days": 4,
        "savings_drop_pct": 0.30,
        "utility_payment_delay_days": 3,
        "discretionary_spend_drop_pct": 0.25,
        "atm_withdrawal_increase": 3,
        "upi_lending_txn_count": 2,
        "failed_autodebit_count": 1
    }
    
    response = requests.post(f"{BASE_URL}/compare-models", json=customer)
    result = response.json()
    
    print(f"Customer ID: {result['customer_id']}")
    print(f"Score Difference: {result['score_difference']}")
    print(f"Models Agree: {result['agreement']}")
    print()
    
    print("XGBoost Results:")
    xgb = result['xgboost']
    print(f"  Risk Score: {xgb['risk_score']}")
    print(f"  Risk Level: {xgb['risk_level']}")
    print(f"  Will Default: {xgb['will_default']}")
    print()
    
    print("LightGBM Results:")
    lgb = result['lightgbm']
    print(f"  Risk Score: {lgb['risk_score']}")
    print(f"  Risk Level: {lgb['risk_level']}")
    print(f"  Will Default: {lgb['will_default']}")
    print()

def test_batch_prediction():
    """Test batch prediction with LightGBM."""
    print("=" * 60)
    print("📦 Testing Batch Prediction (LightGBM)")
    print("=" * 60)
    
    batch = {
        "customers": [
            {
                "customer_id": "CUST_BATCH_1",
                "salary_delay_days": 1,
                "savings_drop_pct": 0.10,
                "utility_payment_delay_days": 0,
                "discretionary_spend_drop_pct": 0.05,
                "atm_withdrawal_increase": 0,
                "upi_lending_txn_count": 0,
                "failed_autodebit_count": 0,
                "model_type": "lightgbm"
            },
            {
                "customer_id": "CUST_BATCH_2",
                "salary_delay_days": 7,
                "savings_drop_pct": 0.50,
                "utility_payment_delay_days": 5,
                "discretionary_spend_drop_pct": 0.40,
                "atm_withdrawal_increase": 5,
                "upi_lending_txn_count": 4,
                "failed_autodebit_count": 3,
                "model_type": "lightgbm"
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict-batch", json=batch)
    result = response.json()
    
    print(f"Total Customers: {result['total_customers']}")
    print(f"High Risk Count: {result['high_risk_count']}")
    print(f"Critical Risk Count: {result['critical_risk_count']}")
    print()
    
    for pred in result['predictions']:
        print(f"{pred['customer_id']}: {pred['risk_level']} ({pred['risk_score']}) - {pred['model_type']}")
    print()

if __name__ == "__main__":
    try:
        test_models_status()
        test_xgboost_prediction()
        test_lightgbm_prediction()
        test_model_comparison()
        test_batch_prediction()
        
        print("=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API")
        print("Make sure the API is running: python backend/main.py")
    except Exception as e:
        print(f"❌ Error: {e}")
