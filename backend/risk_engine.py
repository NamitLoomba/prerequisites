from ml.predict import RiskPredictor
from ml.predict_lightgbm import LightGBMRiskPredictor
from typing import Dict
import os
import numpy as np

# Make TensorFlow/LSTM optional
try:
    from ml.sequence_model import SequenceRiskPredictor
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    SequenceRiskPredictor = None

class RiskEngine:
    """Risk scoring engine that wraps all ML models."""
    
    def __init__(self):
        """Initialize the risk engine with all ML models."""
        self.xgb_predictor = None
        self.lgb_predictor = None
        self.lstm_predictor = None
        self._load_predictors()
    
    def _load_predictors(self):
        """Load XGBoost, LightGBM, and LSTM predictors."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load XGBoost
        try:
            model_path = os.path.join(base_dir, "ml", "model.pkl")
            scaler_path = os.path.join(base_dir, "ml", "scaler.pkl")
            self.xgb_predictor = RiskPredictor(model_path, scaler_path)
            print("✅ XGBoost model loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load XGBoost model: {e}")
        
        # Load LightGBM
        try:
            model_path = os.path.join(base_dir, "ml", "model_lgb.pkl")
            scaler_path = os.path.join(base_dir, "ml", "scaler_lgb.pkl")
            self.lgb_predictor = LightGBMRiskPredictor(model_path, scaler_path)
            print("✅ LightGBM model loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load LightGBM model: {e}")
        
        # Load LSTM Sequence Model (optional - requires TensorFlow)
        if TENSORFLOW_AVAILABLE:
            try:
                model_path = os.path.join(base_dir, "ml", "sequence_model.h5")
                scaler_path = os.path.join(base_dir, "ml", "sequence_scaler.pkl")
                self.lstm_predictor = SequenceRiskPredictor(model_path, scaler_path)
                print("✅ TensorFlow LSTM model loaded successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not load LSTM model: {e}")
        else:
            print("⚠️ TensorFlow not available - LSTM model disabled")
        
        if not self.xgb_predictor and not self.lgb_predictor and not self.lstm_predictor:
            print("⚠️ Risk engine will work in demo mode")
    
    def get_recommendation(self, risk_score: float, risk_level: str) -> str:
        """Get recommended action based on risk level."""
        recommendations = {
            "Critical": "Immediate intervention required - Offer payment holiday",
            "High": "Priority contact - Propose debt consolidation or payment plan",
            "Medium": "Schedule check-in call - Offer financial counseling",
            "Low": "Monitor regularly - Continue standard relationship management"
        }
        return recommendations.get(risk_level, "Monitor regularly")
    
    def score_customer(self, customer_data: Dict) -> Dict:
        """Score a single customer for delinquency risk."""
        if hasattr(customer_data, 'dict'):
            customer_data = customer_data.dict()
        
        customer_id = customer_data.get("customer_id", "UNKNOWN")
        model_type = customer_data.get("model_type", "xgboost")
        
        features = {
            "salary_delay_days": customer_data.get("salary_delay_days", 0),
            "savings_drop_pct": customer_data.get("savings_drop_pct", 0.0),
            "utility_payment_delay_days": customer_data.get("utility_payment_delay_days", 0),
            "discretionary_spend_drop_pct": customer_data.get("discretionary_spend_drop_pct", 0.0),
            "atm_withdrawal_increase": customer_data.get("atm_withdrawal_increase", 0),
            "upi_lending_txn_count": customer_data.get("upi_lending_txn_count", 0),
            "failed_autodebit_count": customer_data.get("failed_autodebit_count", 0)
        }
        
        # Select predictor based on model_type
        if model_type == "lightgbm" and self.lgb_predictor:
            result = self.lgb_predictor.predict(features)
        elif model_type == "xgboost" and self.xgb_predictor:
            result = self.xgb_predictor.predict(features)
        elif model_type == "both":
            # Return comparison of both models
            return self._compare_models(customer_id, features)
        else:
            # Fallback to demo mode
            risk_score = min(1.0, sum(features.values()) / 20)
            risk_level = "Critical" if risk_score >= 0.75 else "High" if risk_score >= 0.5 else "Medium" if risk_score >= 0.25 else "Low"
            result = {
                "risk_score": round(risk_score, 4),
                "risk_level": risk_level,
                "will_default": risk_score >= 0.5,
                "confidence": 0.85,
                "model_type": "demo"
            }
        
        result["recommended_action"] = self.get_recommendation(result["risk_score"], result["risk_level"])
        result["customer_id"] = customer_id
        
        return result
    
    def _compare_models(self, customer_id: str, features: Dict) -> Dict:
        """Compare predictions from both models."""
        xgb_result = None
        lgb_result = None
        
        if self.xgb_predictor:
            xgb_result = self.xgb_predictor.predict(features)
            xgb_result["customer_id"] = customer_id
            xgb_result["recommended_action"] = self.get_recommendation(xgb_result["risk_score"], xgb_result["risk_level"])
        
        if self.lgb_predictor:
            lgb_result = self.lgb_predictor.predict(features)
            lgb_result["customer_id"] = customer_id
            lgb_result["recommended_action"] = self.get_recommendation(lgb_result["risk_score"], lgb_result["risk_level"])
        
        if xgb_result and lgb_result:
            score_diff = abs(xgb_result["risk_score"] - lgb_result["risk_score"])
            agreement = xgb_result["risk_level"] == lgb_result["risk_level"]
            
            return {
                "customer_id": customer_id,
                "xgboost": xgb_result,
                "lightgbm": lgb_result,
                "score_difference": round(score_diff, 4),
                "agreement": agreement
            }
        
        # Fallback if comparison not possible
        return xgb_result or lgb_result or {
            "customer_id": customer_id,
            "risk_score": 0.0,
            "risk_level": "Unknown",
            "will_default": False,
            "confidence": 0.0,
            "model_type": "unavailable",
            "recommended_action": "Models not available"
        }
    
    def score_batch(self, customers: list) -> Dict:
        """Score multiple customers."""
        predictions = []
        high_risk_count = 0
        critical_risk_count = 0
        
        for customer in customers:
            result = self.score_customer(customer)
            predictions.append(result)
            if result["risk_level"] in ["High", "Critical"]:
                high_risk_count += 1
            if result["risk_level"] == "Critical":
                critical_risk_count += 1
        
        return {
            "predictions": predictions,
            "total_customers": len(predictions),
            "high_risk_count": high_risk_count,
            "critical_risk_count": critical_risk_count
        }
    
    def score_sequence(self, customer_id: str, sequence_data) -> Dict:
        """Score a customer using LSTM sequence model (TensorFlow)."""
        if not self.lstm_predictor:
            return {
                "customer_id": customer_id,
                "risk_score": 0.0,
                "risk_level": "Unknown",
                "will_default": False,
                "confidence": 0.0,
                "model_type": "LSTM (unavailable)",
                "recommended_action": "LSTM model not loaded"
            }
        
        # Convert to numpy array
        sequence_array = np.array(sequence_data)
        
        # Predict using TensorFlow LSTM
        result = self.lstm_predictor.predict_sequence(sequence_array)
        
        # Add customer ID and recommendation
        result["customer_id"] = customer_id
        result["recommended_action"] = self.get_recommendation(result["risk_score"], result["risk_level"])
        
        return result

risk_engine = RiskEngine()
