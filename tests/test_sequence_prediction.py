"""
Test sequence model predictions
Run after training completes: python test_sequence_prediction.py
"""

import numpy as np
from ml.sequence_model import load_sequence_model

# Load the trained model
print("Loading sequence model...")
predictor = load_sequence_model()

# Create sample customer sequence data
# Shape: (30 days, 10 features)
# Features: salary_amount, salary_received, savings_amount, daily_expenses, 
#           utility_payment, utility_delay_days, atm_withdrawals, 
#           upi_lending_transactions, failed_autodebits, discretionary_spending

print("\n" + "="*60)
print("Testing Sequence Predictions")
print("="*60)

# Example 1: Stable customer (30 days of good behavior)
print("\n1. Stable Customer (Low Risk)")
stable_customer = np.random.rand(30, 10) * 0.5  # Low values
stable_customer[:, 0] = 50000  # Consistent salary
stable_customer[:, 2] = 20000  # Good savings
stable_customer[:, 6] = 0      # No excessive ATM withdrawals
stable_customer[:, 7] = 0      # No lending transactions

result = predictor.predict_sequence(stable_customer)
print(f"   Risk Score: {result['risk_score']:.4f}")
print(f"   Risk Level: {result['risk_level']}")
print(f"   Will Default: {result['will_default']}")
print(f"   Confidence: {result['confidence']:.4f}")

# Example 2: Deteriorating customer (behavior getting worse over time)
print("\n2. Deteriorating Customer (High Risk)")
deteriorating_customer = np.random.rand(30, 10)

# Simulate deteriorating pattern
for day in range(30):
    deterioration_factor = day / 30  # Gets worse over time
    deteriorating_customer[day, 0] = 50000 * (1 - deterioration_factor * 0.3)  # Salary drops
    deteriorating_customer[day, 2] = 20000 * (1 - deterioration_factor * 0.7)  # Savings drop
    deteriorating_customer[day, 6] = int(deterioration_factor * 5)  # More ATM withdrawals
    deteriorating_customer[day, 7] = int(deterioration_factor * 3)  # More lending transactions
    deteriorating_customer[day, 8] = int(deterioration_factor * 2)  # Failed debits

result = predictor.predict_sequence(deteriorating_customer)
print(f"   Risk Score: {result['risk_score']:.4f}")
print(f"   Risk Level: {result['risk_level']}")
print(f"   Will Default: {result['will_default']}")
print(f"   Confidence: {result['confidence']:.4f}")

# Example 3: Critical risk customer
print("\n3. Critical Risk Customer")
critical_customer = np.random.rand(30, 10)
critical_customer[:, 0] = 30000  # Lower salary
critical_customer[:, 2] = 2000   # Very low savings
critical_customer[:, 6] = 8      # Many ATM withdrawals
critical_customer[:, 7] = 6      # Many lending transactions
critical_customer[:, 8] = 4      # Many failed debits

result = predictor.predict_sequence(critical_customer)
print(f"   Risk Score: {result['risk_score']:.4f}")
print(f"   Risk Level: {result['risk_level']}")
print(f"   Will Default: {result['will_default']}")
print(f"   Confidence: {result['confidence']:.4f}")

print("\n" + "="*60)
print("✅ Predictions Complete!")
print("="*60)
