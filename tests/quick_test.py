"""Quick test to debug the API issue"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

# Test XGBoost prediction
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

print("Sending request...")
print(json.dumps(customer, indent=2))

try:
    response = requests.post(f"{BASE_URL}/predict-risk", json=customer)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Success!")
        print(json.dumps(result, indent=2))
    else:
        print(f"\n❌ Error: {response.status_code}")
        
except Exception as e:
    print(f"\n❌ Exception: {e}")
    import traceback
    traceback.print_exc()
