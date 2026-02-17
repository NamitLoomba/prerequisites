import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from feast import FeatureStore
import redis
import json

class FeatureStoreManager:
    """Manage real-time features using Feast feature store."""
    
    def __init__(self, repo_path="."):
        self.store = FeatureStore(repo_path=repo_path)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
    def initialize_feature_store(self):
        """Initialize and apply feature definitions."""
        print("🚀 Initializing Feature Store...")
        try:
            # Apply feature definitions
            self.store.apply()
            print("✅ Feature store initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing feature store: {e}")
            raise
    
    def generate_sample_data(self, n_customers=1000):
        """Generate sample data for feature store."""
        print(f"📊 Generating sample data for {n_customers} customers...")
        
        # Customer profile data
        customer_data = []
        for i in range(n_customers):
            customer_data.append({
                'customer_id': i,
                'age': random.randint(18, 75),
                'income_level': random.choice(['Low', 'Medium', 'High', 'Very High']),
                'employment_status': random.choice(['Employed', 'Self-employed', 'Unemployed', 'Retired']),
                'account_age_days': random.randint(30, 3650),
                'total_accounts': random.randint(1, 5),
                'event_timestamp': datetime.now()
            })
        
        # Transaction behavior data
        transaction_data = []
        for i in range(n_customers):
            for day in range(30):  # 30 days of data
                transaction_data.append({
                    'customer_id': i,
                    'daily_transaction_count': random.randint(1, 20),
                    'daily_transaction_amount': random.uniform(100, 5000),
                    'atm_withdrawal_count_today': random.randint(0, 3),
                    'upi_transaction_count_today': random.randint(0, 15),
                    'failed_transaction_count_today': random.randint(0, 2),
                    'salary_received_today': 1 if day % 15 == 0 else 0,
                    'utility_payment_made_today': 1 if day % 30 == 5 else 0,
                    'event_timestamp': datetime.now() - timedelta(days=30-day)
                })
        
        return pd.DataFrame(customer_data), pd.DataFrame(transaction_data)
    
    def ingest_features(self, customer_df, transaction_df):
        """Ingest data into feature store."""
        print("📥 Ingesting features into feature store...")
        
        try:
            # Ingest customer profile features
            self.store.write_to_online_store("customer_profile", customer_df)
            
            # Ingest transaction behavior features
            self.store.write_to_online_store("transaction_behavior", transaction_df)
            
            print("✅ Features ingested successfully")
        except Exception as e:
            print(f"❌ Error ingesting features: {e}")
            raise
    
    def get_online_features(self, customer_ids, feature_refs):
        """
        Get real-time features for customers.
        
        Args:
            customer_ids: List of customer IDs
            feature_refs: List of feature references
            
        Returns:
            Dictionary of features
        """
        try:
            # Get features from online store
            feature_vector = self.store.get_online_features(
                features=feature_refs,
                entity_rows=[{"customer_id": cid} for cid in customer_ids]
            ).to_dict()
            
            return feature_vector
        except Exception as e:
            print(f"❌ Error retrieving online features: {e}")
            return {}
    
    def update_real_time_features(self, customer_id, transaction_data):
        """
        Update real-time features for a customer.
        
        Args:
            customer_id: Customer identifier
            transaction_data: Dictionary with latest transaction data
        """
        try:
            # Update transaction behavior features
            feature_data = {
                'customer_id': customer_id,
                'daily_transaction_count': transaction_data.get('transaction_count', 0),
                'daily_transaction_amount': transaction_data.get('transaction_amount', 0.0),
                'atm_withdrawal_count_today': transaction_data.get('atm_withdrawals', 0),
                'upi_transaction_count_today': transaction_data.get('upi_transactions', 0),
                'failed_transaction_count_today': transaction_data.get('failed_transactions', 0),
                'salary_received_today': transaction_data.get('salary_received', 0),
                'utility_payment_made_today': transaction_data.get('utility_payment', 0),
                'event_timestamp': datetime.now()
            }
            
            # Convert to DataFrame for Feast
            df = pd.DataFrame([feature_data])
            self.store.write_to_online_store("transaction_behavior", df)
            
            print(f"✅ Real-time features updated for customer {customer_id}")
            
        except Exception as e:
            print(f"❌ Error updating real-time features: {e}")
    
    def get_customer_risk_features(self, customer_id):
        """Get all risk-relevant features for a customer."""
        feature_refs = [
            "customer_profile:age",
            "customer_profile:income_level",
            "transaction_behavior:daily_transaction_count",
            "transaction_behavior:daily_transaction_amount",
            "transaction_behavior:atm_withdrawal_count_today",
            "transaction_behavior:upi_transaction_count_today",
            "transaction_behavior:failed_transaction_count_today",
            "transaction_behavior:salary_received_today",
            "stress_indicators:savings_balance_change_pct",
            "stress_indicators:salary_delay_days_current",
            "risk_features:delinquency_risk_score",
            "risk_features:stress_index_current"
        ]
        
        features = self.get_online_features([customer_id], feature_refs)
        return features

def setup_feature_store():
    """Complete setup pipeline for feature store."""
    print("=" * 60)
    print("🚀 FEATURE STORE SETUP PIPELINE")
    print("=" * 60)
    
    # Initialize manager
    manager = FeatureStoreManager()
    
    # Step 1: Initialize feature store
    print("\n📥 Step 1: Initializing feature store...")
    manager.initialize_feature_store()
    
    # Step 2: Generate sample data
    print("\n📊 Step 2: Generating sample data...")
    customer_df, transaction_df = manager.generate_sample_data(n_customers=500)
    
    # Step 3: Ingest features
    print("\n💾 Step 3: Ingesting features...")
    manager.ingest_features(customer_df, transaction_df)
    
    # Step 4: Test feature retrieval
    print("\n🧪 Step 4: Testing feature retrieval...")
    test_customer = 123
    features = manager.get_customer_risk_features(test_customer)
    
    if features:
        print(f"✅ Retrieved features for customer {test_customer}:")
        for key, value in list(features.items())[:5]:  # Show first 5 features
            print(f"   {key}: {value}")
    else:
        print("❌ Failed to retrieve features")
    
    print("\n" + "=" * 60)
    print("✅ Feature Store Setup Complete!")
    print("=" * 60)
    
    return manager

if __name__ == "__main__":
    manager = setup_feature_store()