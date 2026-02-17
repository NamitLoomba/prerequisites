import pandas as pd
from feature_store_manager import FeatureStoreManager
from ml.predict import load_model as load_xgb_model
from ml.predict_lightgbm import load_lightgbm_model

class RealTimeRiskEngine:
    """Real-time risk scoring engine using feature store."""
    
    def __init__(self):
        self.feature_manager = FeatureStoreManager()
        self.xgb_model = load_xgb_model()
        self.lgb_model = load_lightgbm_model()
        
    def score_customer_real_time(self, customer_id, transaction_data=None):
        """
        Score customer risk in real-time using feature store.
        
        Args:
            customer_id: Customer identifier
            transaction_data: Optional latest transaction data
            
        Returns:
            Risk assessment with multiple model scores
        """
        
        # Update real-time features if new transaction data provided
        if transaction_data:
            self.feature_manager.update_real_time_features(customer_id, transaction_data)
        
        # Get all relevant features from feature store
        features = self.feature_manager.get_customer_risk_features(customer_id)
        
        if not features or len(features['customer_id']) == 0:
            return {"error": "No features found for customer"}
        
        # Convert features to prediction format
        # This is a simplified mapping - in practice you'd have proper feature engineering
        prediction_features = {
            'salary_delay_days': features.get('stress_indicators:salary_delay_days_current', [0])[0],
            'savings_drop_pct': features.get('stress_indicators:savings_balance_change_pct', [0.0])[0],
            'utility_payment_delay_days': features.get('stress_indicators:utility_payment_delay_current', [0])[0],
            'discretionary_spend_drop_pct': features.get('stress_indicators:discretionary_spending_drop_current', [0.0])[0],
            'atm_withdrawal_increase': features.get('transaction_behavior:atm_withdrawal_count_today', [0])[0],
            'upi_lending_txn_count': features.get('stress_indicators:digital_lending_activity', [0])[0],
            'failed_autodebit_count': features.get('transaction_behavior:failed_transaction_count_today', [0])[0]
        }
        
        # Get predictions from all models
        xgb_result = self.xgb_model.predict(prediction_features)
        lgb_result = self.lgb_model.predict(prediction_features)
        
        # Ensemble prediction
        ensemble_score = (xgb_result['risk_score'] + lgb_result['risk_score']) / 2
        
        # Determine ensemble risk level
        if ensemble_score >= 0.75:
            ensemble_level = "Critical"
        elif ensemble_score >= 0.50:
            ensemble_level = "High"
        elif ensemble_score >= 0.25:
            ensemble_level = "Medium"
        else:
            ensemble_level = "Low"
        
        return {
            "customer_id": customer_id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "models": {
                "xgboost": xgb_result,
                "lightgbm": lgb_result
            },
            "ensemble": {
                "risk_score": round(ensemble_score, 4),
                "risk_level": ensemble_level,
                "confidence": round(max(ensemble_score, 1 - ensemble_score), 4)
            },
            "features_used": len([v for v in prediction_features.values() if v != 0])
        }
    
    def batch_score_customers(self, customer_ids):
        """Score multiple customers in batch."""
        results = []
        for customer_id in customer_ids:
            result = self.score_customer_real_time(customer_id)
            results.append(result)
        return results
    
    def get_feature_freshness(self, customer_id):
        """Get information about feature freshness/timestamps."""
        # This would query the feature store for metadata
        # Simplified implementation
        return {
            "customer_id": customer_id,
            "profile_features_last_updated": "30 days ago",
            "transaction_features_last_updated": "1 hour ago",
            "stress_indicators_last_updated": "6 hours ago",
            "risk_features_last_updated": "1 hour ago"
        }

def demo_real_time_scoring():
    """Demo real-time risk scoring capabilities."""
    print("=" * 70)
    print("🚀 REAL-TIME RISK SCORING DEMO")
    print("=" * 70)
    
    # Initialize engine
    engine = RealTimeRiskEngine()
    
    # Test customers
    test_customers = [1001, 1002, 1003]
    
    print(f"\n📊 Scoring {len(test_customers)} customers in real-time...")
    
    for customer_id in test_customers:
        print(f"\n👤 Customer {customer_id}")
        print("-" * 40)
        
        # Score without new transaction data
        result = engine.score_customer_real_time(customer_id)
        
        if "error" in result:
            print(f"   ❌ {result['error']}")
            continue
            
        # Display results
        print(f"   XGBoost Score: {result['models']['xgboost']['risk_score']:.4f}")
        print(f"   LightGBM Score: {result['models']['lightgbm']['risk_score']:.4f}")
        print(f"   🎯 Ensemble Score: {result['ensemble']['risk_score']:.4f}")
        print(f"   🚨 Risk Level: {result['ensemble']['risk_level']}")
        print(f"   🔧 Features Used: {result['features_used']}")
        
        # Show feature freshness
        freshness = engine.get_feature_freshness(customer_id)
        print(f"   📅 Profile Data: {freshness['profile_features_last_updated']}")
        print(f"   📅 Transaction Data: {freshness['transaction_features_last_updated']}")
    
    print("\n" + "=" * 70)
    print("✅ Real-time Scoring Demo Complete!")
    print("=" * 70)

if __name__ == "__main__":
    demo_real_time_scoring()