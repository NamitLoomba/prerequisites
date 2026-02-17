import pandas as pd
import numpy as np
import os

# Ensure output directory exists
os.makedirs("data/raw", exist_ok=True)

np.random.seed(42)

NUM_CUSTOMERS = 500

data = []

for i in range(NUM_CUSTOMERS):
    salary_delay = np.random.randint(0, 10)
    savings_drop = np.round(np.random.uniform(0, 0.5), 2)
    utility_delay = np.random.randint(0, 7)
    discretionary_drop = np.round(np.random.uniform(0, 0.6), 2)
    atm_spike = np.random.randint(0, 5)
    upi_lending = np.random.randint(0, 4)
    failed_autodebit = np.random.randint(0, 3)

    # Risk logic (used to create realistic labels)
    risk_score = (
        0.25 * salary_delay +
        0.30 * savings_drop * 10 +
        0.20 * utility_delay +
        0.15 * atm_spike +
        0.10 * failed_autodebit * 3
    )

    # Lower threshold to generate ~20% defaults
    default = 1 if risk_score > 3.5 else 0

    data.append([
        f"CUST_{i+1}",
        salary_delay,
        savings_drop,
        utility_delay,
        discretionary_drop,
        atm_spike,
        upi_lending,
        failed_autodebit,
        default
    ])

columns = [
    "customer_id",
    "salary_delay_days",
    "savings_drop_pct",
    "utility_payment_delay_days",
    "discretionary_spend_drop_pct",
    "atm_withdrawal_increase",
    "upi_lending_txn_count",
    "failed_autodebit_count",
    "default_next_30_days"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("data/raw/synthetic_transactions.csv", index=False)

print("✅ Synthetic dataset generated successfully")
print(f"📊 Dataset shape: {df.shape}")
print(f"📈 Default rate: {df['default_next_30_days'].mean()*100:.1f}%")
