import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.sequence_data_generator import generate_sequence_data, create_sequences
from ml.sequence_model import SequenceRiskPredictor

def train_sequence_model():
    """Complete pipeline to train sequence model."""
    
    print("=" * 60)
    print("🚀 SEQUENCE MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Generate sequential data
    print("\n📥 Step 1: Generating sequential data...")
    df = generate_sequence_data(n_customers=1000, sequence_length=90)
    
    # Step 2: Create sequences
    print("\n📊 Step 2: Creating training sequences...")
    X, y = create_sequences(df, sequence_length=30, target_days_ahead=14)
    
    # Step 3: Train model
    print("\n🧠 Step 3: Training LSTM model...")
    predictor = SequenceRiskPredictor()
    history = predictor.train_model(X, y, model_type='lstm')
    
    # Step 4: Test prediction
    print("\n🧪 Step 4: Testing prediction...")
    test_sequence = X[0]  # First sequence from test set
    result = predictor.predict_sequence(test_sequence)
    
    print("Prediction Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Sequence Model Training Complete!")
    print("=" * 60)
    
    return predictor, history

def compare_with_traditional_models():
    """Compare sequence model with traditional XGBoost/LightGBM."""
    print("=" * 60)
    print("📊 MODEL COMPARISON: Sequence vs Traditional")
    print("=" * 60)
    
    # Load traditional models
    from ml.predict import load_model as load_xgb
    from ml.predict_lightgbm import load_lightgbm_model
    
    xgb_model = load_xgb()
    lgb_model = load_lightgbm_model()
    seq_model = load_sequence_model()
    
    # Create test customer with deteriorating pattern
    test_customers = [
        {
            "name": "Stable Customer",
            "pattern": "stable"
        },
        {
            "name": "Deteriorating Customer",
            "pattern": "deteriorating"
        }
    ]
    
    for customer in test_customers:
        print(f"\n👤 {customer['name']}")
        print("-" * 40)
        
        if customer['pattern'] == 'stable':
            # Stable pattern - low risk features throughout
            features = {
                "salary_delay_days": 0,
                "savings_drop_pct": 0.05,
                "utility_payment_delay_days": 0,
                "discretionary_spend_drop_pct": 0.02,
                "atm_withdrawal_increase": 0,
                "upi_lending_txn_count": 0,
                "failed_autodebit_count": 0
            }
        else:
            # Deteriorating pattern - increasing risk over time
            features = {
                "salary_delay_days": 8,
                "savings_drop_pct": 0.65,
                "utility_payment_delay_days": 6,
                "discretionary_spend_drop_pct": 0.55,
                "atm_withdrawal_increase": 7,
                "upi_lending_txn_count": 5,
                "failed_autodebit_count": 3
            }
        
        # Traditional models prediction
        xgb_pred = xgb_model.predict(features)
        lgb_pred = lgb_model.predict(features)
        
        # For sequence model, we need to simulate a sequence
        # This is a simplified example - in practice you'd have actual time series
        print(f"XGBoost Risk Score: {xgb_pred['risk_score']:.4f}")
        print(f"LightGBM Risk Score: {lgb_pred['risk_score']:.4f}")
        print(f"Sequence Model: Not directly comparable (needs time series)")
        print("Note: Sequence models require temporal data patterns")

if __name__ == "__main__":
    # Train sequence model
    predictor, history = train_sequence_model()
    
    # Compare models
    print("\n")
    compare_with_traditional_models()