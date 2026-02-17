import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class LightGBMRiskPredictor:
    """LightGBM risk prediction model wrapper."""
    
    def __init__(self, model_path="ml/model_lgb.pkl", scaler_path="ml/scaler_lgb.pkl"):
        """Initialize predictor with trained LightGBM model and scaler."""
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self._load_model()
    
    def _load_model(self):
        """Load trained LightGBM model and scaler."""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("✅ LightGBM Model and scaler loaded successfully")
        except FileNotFoundError as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _create_features(self, data):
        """Create engineered features from raw input (same as XGBoost)."""
        df = pd.DataFrame([data])
        
        # Financial Stress Index
        df['stress_index'] = (
            df['salary_delay_days'] * 1.5 +
            df['savings_drop_pct'] * 10 +
            df['utility_payment_delay_days'] * 1.0 +
            df['atm_withdrawal_increase'] * 0.8 +
            df['failed_autodebit_count'] * 2.0 +
            df['upi_lending_txn_count'] * 1.2
        )
        
        # Liquidity Ratio
        df['liquidity_ratio'] = (1 - df['savings_drop_pct']) / (df['atm_withdrawal_increase'] + 1)
        
        # Payment Reliability
        df['payment_reliability'] = 10 - df['utility_payment_delay_days'] - df['failed_autodebit_count'] * 3
        
        # Cash Flow Pressure
        df['cash_flow_pressure'] = (
            df['salary_delay_days'] * 2 +
            df['atm_withdrawal_increase'] +
            df['upi_lending_txn_count']
        )
        
        # Savings Behavior
        df['savings_behavior'] = df['savings_drop_pct'] + df['discretionary_spend_drop_pct']
        
        # Digital Stress
        df['digital_stress'] = df['upi_lending_txn_count'] * 1.5 + df['failed_autodebit_count']
        
        return df
    
    def _get_feature_columns(self):
        """Return feature column names."""
        return [
            'salary_delay_days', 'savings_drop_pct', 'utility_payment_delay_days',
            'discretionary_spend_drop_pct', 'atm_withdrawal_increase',
            'upi_lending_txn_count', 'failed_autodebit_count',
            'stress_index', 'liquidity_ratio', 'payment_reliability',
            'cash_flow_pressure', 'savings_behavior', 'digital_stress'
        ]
    
    def predict(self, input_data):
        """
        Predict delinquency risk using LightGBM.
        
        Args:
            input_data: dict with raw features
            
        Returns:
            dict with risk_score, risk_level, and prediction
        """
        # Create features
        features = self._create_features(input_data)
        feature_cols = self._get_feature_columns()
        
        # Scale features
        X = features[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Predict
        risk_prob = self.model.predict_proba(X_scaled)[0][1]
        prediction = int(risk_prob >= 0.5)
        
        # Determine risk level
        if risk_prob >= 0.75:
            risk_level = "Critical"
        elif risk_prob >= 0.50:
            risk_level = "High"
        elif risk_prob >= 0.25:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "risk_score": round(risk_prob, 4),
            "risk_level": risk_level,
            "will_default": bool(prediction),
            "confidence": round(max(risk_prob, 1 - risk_prob), 4),
            "model_type": "LightGBM"
        }
    
    def predict_batch(self, input_list):
        """Predict for multiple customers."""
        return [self.predict(data) for data in input_list]


def load_lightgbm_model():
    """Convenience function to load the LightGBM predictor."""
    return LightGBMRiskPredictor()


if __name__ == "__main__":
    # Test LightGBM prediction
    predictor = load_lightgbm_model()
    
    # Sample customer
    test_customer = {
        "salary_delay_days": 4,
        "savings_drop_pct": 0.30,
        "utility_payment_delay_days": 3,
        "discretionary_spend_drop_pct": 0.25,
        "atm_withdrawal_increase": 3,
        "upi_lending_txn_count": 2,
        "failed_autodebit_count": 1
    }
    
    result = predictor.predict(test_customer)
    print("\n📊 LightGBM Prediction Result:")
    for key, value in result.items():
        print(f"   {key}: {value}")