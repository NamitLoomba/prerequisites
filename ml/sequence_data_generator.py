import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sequence_data(n_customers=1000, sequence_length=30):
    """
    Generate synthetic sequential transaction data for deep learning models.
    
    Args:
        n_customers: Number of customers to generate
        sequence_length: Number of time steps (days) per customer
        
    Returns:
        DataFrame with sequential customer data
    """
    
    print(f"Generating sequential data for {n_customers} customers over {sequence_length} days...")
    
    data = []
    
    for customer_id in range(n_customers):
        # Customer baseline characteristics
        baseline_salary = random.uniform(20000, 100000)
        baseline_savings = random.uniform(5000, 50000)
        baseline_expenses = baseline_salary * random.uniform(0.6, 0.9)
        
        # Determine if this customer will default (create patterns)
        will_default = random.random() < 0.3  # 30% default rate
        
        # Generate sequence
        for day in range(sequence_length):
            # Time-based features
            day_of_month = (day % 30) + 1
            day_of_week = day % 7
            
            # Base salary (with some variation)
            salary_amount = baseline_salary * random.uniform(0.9, 1.1)
            salary_received = 1 if day_of_month in [1, 15] else 0  # Bi-weekly
            
            # Savings pattern
            if will_default and day > sequence_length * 0.7:
                # Deteriorating savings pattern for defaulters
                savings_amount = baseline_savings * max(0.1, 1 - (day / sequence_length) * 1.5)
            else:
                savings_amount = baseline_savings * random.uniform(0.8, 1.2)
            
            # Expense patterns
            if day_of_week in [5, 6]:  # Weekend spending
                daily_expenses = baseline_expenses * 0.8 / 20 * random.uniform(1.2, 2.0)
            else:
                daily_expenses = baseline_expenses * 0.8 / 20 * random.uniform(0.5, 1.2)
            
            # Utility payments (monthly)
            utility_payment = baseline_expenses * 0.2 if day_of_month in [5, 6] else 0
            utility_delay = random.randint(0, 3) if utility_payment > 0 else 0
            
            # ATM withdrawals (increasing for defaulters)
            if will_default and day > sequence_length * 0.6:
                atm_withdrawals = random.randint(2, 5)
            else:
                atm_withdrawals = random.randint(0, 2)
            
            # UPI lending transactions (stress indicator)
            if will_default and day > sequence_length * 0.5:
                upi_lending = random.randint(1, 4)
            else:
                upi_lending = random.randint(0, 1)
            
            # Failed auto-debits (increasing stress)
            if will_default and day > sequence_length * 0.4:
                failed_debits = random.randint(1, 3)
            else:
                failed_debits = random.randint(0, 1)
            
            # Discretionary spending
            discretionary_spend = daily_expenses * random.uniform(0.1, 0.4)
            
            # Create record
            record = {
                'customer_id': customer_id,
                'day': day,
                'salary_amount': salary_amount,
                'salary_received': salary_received,
                'savings_amount': savings_amount,
                'daily_expenses': daily_expenses,
                'utility_payment': utility_payment,
                'utility_delay_days': utility_delay,
                'atm_withdrawals': atm_withdrawals,
                'upi_lending_transactions': upi_lending,
                'failed_autodebits': failed_debits,
                'discretionary_spending': discretionary_spend,
                'will_default': int(will_default)
            }
            
            data.append(record)
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} sequential records")
    print(f"Default rate: {df['will_default'].mean():.2%}")
    
    # Save to file
    df.to_csv('data/raw/sequential_transactions.csv', index=False)
    print("✅ Sequential data saved to data/raw/sequential_transactions.csv")
    
    return df

def create_sequences(df, sequence_length=30, target_days_ahead=7):
    """
    Convert sequential data into training sequences for LSTM.
    
    Args:
        df: DataFrame with sequential data
        sequence_length: Length of input sequences
        target_days_ahead: How many days ahead to predict default
        
    Returns:
        X, y arrays for training
    """
    
    print(f"Creating sequences of length {sequence_length}...")
    
    sequences = []
    targets = []
    
    # Group by customer
    for customer_id in df['customer_id'].unique():
        customer_data = df[df['customer_id'] == customer_id].sort_values('day')
        
        if len(customer_data) < sequence_length + target_days_ahead:
            continue
            
        # Create sequences
        for i in range(len(customer_data) - sequence_length - target_days_ahead + 1):
            # Input sequence
            seq_data = customer_data.iloc[i:i+sequence_length].copy()
            
            # Target: will default in next target_days_ahead days
            future_data = customer_data.iloc[i+sequence_length:i+sequence_length+target_days_ahead]
            target = int(future_data['will_default'].max())  # 1 if any default in future period
            
            # Feature engineering for sequence
            features = seq_data[[
                'salary_amount', 'salary_received', 'savings_amount',
                'daily_expenses', 'utility_payment', 'utility_delay_days',
                'atm_withdrawals', 'upi_lending_transactions', 
                'failed_autodebits', 'discretionary_spending'
            ]].values
            
            sequences.append(features)
            targets.append(target)
    
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f"Created {len(X)} sequences")
    print(f"Target distribution: {np.bincount(y)}")
    
    return X, y

if __name__ == "__main__":
    # Generate sequential data
    df = generate_sequence_data(n_customers=500, sequence_length=60)
    
    # Create training sequences
    X, y = create_sequences(df, sequence_length=30, target_days_ahead=14)
    
    print(f"\nSequence shape: {X.shape}")
    print(f"Target shape: {y.shape}")