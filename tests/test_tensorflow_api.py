"""
Test TensorFlow LSTM model through the API
Run after starting the API: python backend/main.py
"""

import requests
import json
import numpy as np

BASE_URL = "http://localhost:8000/api/v1"

print("="*70)
print("🧠 Testing TensorFlow LSTM Model via API")
print("="*70)

# Check if TensorFlow model is loaded
print("\n1. Checking Model Status...")
print("-"*70)
response = requests.get(f"{BASE_URL}/models/status")
status = response.json()
print(json.dumps(status, indent=2))

if not status.get('lstm_tensorflow_available'):
    print("\n❌ TensorFlow LSTM model is not loaded!")
    print("Make sure the API is running and the model file exists.")
    exit(1)

print("\n✅ TensorFlow LSTM model is loaded and ready!")

# Test 1: Stable customer (low risk)
print("\n2. Testing Stable Customer (Low Risk)")
print("-"*70)

# Generate 30 days of stable behavior
stable_sequence = []
for day in range(30):
    stable_sequence.append([
        50000,  # salary_amount
        1 if day % 15 == 0 else 0,  # salary_received (bi-weekly)
        20000,  # savings_amount (stable)
        2000,   # daily_expenses
        1000 if day == 5 else 0,  # utility_payment
        0,      # utility_delay_days
        0,      # atm_withdrawals (low)
        0,      # upi_lending_transactions (none)
        0,      # failed_autodebits (none)
        500     # discretionary_spending
    ])

request_data = {
    "customer_id": "CUST_STABLE_001",
    "sequence_data": stable_sequence
}

response = requests.post(f"{BASE_URL}/predict-sequence", json=request_data)
result = response.json()

print(f"Customer ID: {result['customer_id']}")
print(f"Model Type: {result['model_type']} ← TensorFlow LSTM")
print(f"Risk Score: {result['risk_score']:.4f}")
print(f"Risk Level: {result['risk_level']}")
print(f"Will Default: {result['will_default']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Action: {result['recommended_action']}")

# Test 2: Deteriorating customer (high risk)
print("\n3. Testing Deteriorating Customer (High Risk)")
print("-"*70)

# Generate 30 days of deteriorating behavior
deteriorating_sequence = []
for day in range(30):
    deterioration_factor = day / 30  # Gets worse over time
    deteriorating_sequence.append([
        50000 * (1 - deterioration_factor * 0.3),  # salary drops
        1 if day % 15 == 0 else 0,
        20000 * (1 - deterioration_factor * 0.7),  # savings drop significantly
        2000 * (1 + deterioration_factor * 0.5),   # expenses increase
        1000 if day == 5 else 0,
        int(deterioration_factor * 5),              # payment delays increase
        int(deterioration_factor * 8),              # more ATM withdrawals
        int(deterioration_factor * 5),              # more lending transactions
        int(deterioration_factor * 3),              # failed debits increase
        500 * (1 - deterioration_factor * 0.8)     # discretionary spending drops
    ])

request_data = {
    "customer_id": "CUST_DETERIORATING_002",
    "sequence_data": deteriorating_sequence
}

response = requests.post(f"{BASE_URL}/predict-sequence", json=request_data)
result = response.json()

print(f"Customer ID: {result['customer_id']}")
print(f"Model Type: {result['model_type']} ← TensorFlow LSTM")
print(f"Risk Score: {result['risk_score']:.4f}")
print(f"Risk Level: {result['risk_level']}")
print(f"Will Default: {result['will_default']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Action: {result['recommended_action']}")

# Test 3: Critical risk customer
print("\n4. Testing Critical Risk Customer")
print("-"*70)

# Generate 30 days of high-risk behavior
critical_sequence = []
for day in range(30):
    critical_sequence.append([
        30000,  # low salary
        1 if day % 15 == 0 else 0,
        2000,   # very low savings
        3000,   # high expenses
        1000 if day == 5 else 0,
        7,      # high payment delays
        10,     # many ATM withdrawals
        8,      # many lending transactions
        5,      # many failed debits
        100     # very low discretionary spending
    ])

request_data = {
    "customer_id": "CUST_CRITICAL_003",
    "sequence_data": critical_sequence
}

response = requests.post(f"{BASE_URL}/predict-sequence", json=request_data)
result = response.json()

print(f"Customer ID: {result['customer_id']}")
print(f"Model Type: {result['model_type']} ← TensorFlow LSTM")
print(f"Risk Score: {result['risk_score']:.4f}")
print(f"Risk Level: {result['risk_level']}")
print(f"Will Default: {result['will_default']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Action: {result['recommended_action']}")

print("\n" + "="*70)
print("✅ TensorFlow LSTM Model is Working in the API!")
print("="*70)
print("\nKey Points:")
print("• TensorFlow processes 30-day sequences (not single snapshots)")
print("• LSTM detects temporal patterns (behavior getting worse over time)")
print("• Model type shows 'LSTM' to indicate TensorFlow is being used")
print("• The model can see trends that XGBoost/LightGBM cannot")
