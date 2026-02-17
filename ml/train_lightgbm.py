import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_engineering import (
    load_raw_data, 
    process_data, 
    get_feature_columns,
    save_processed_data
)

def train_lightgbm_model():
    """Train LightGBM model for delinquency prediction."""
    
    print("=" * 50)
    print("🚀 Starting LightGBM Model Training Pipeline")
    print("=" * 50)
    
    # Step 1: Load and process data
    print("\n📥 Step 1: Loading raw data...")
    df = load_raw_data()
    print(f"   Loaded {len(df)} records")
    
    print("\n🔧 Step 2: Feature engineering...")
    df_processed, scaler = process_data(df, fit_scaler=True)
    save_processed_data(df_processed)
    
    # Save scaler for later use
    joblib.dump(scaler, "ml/scaler_lgb.pkl")
    print("   ✅ Scaler saved to ml/scaler_lgb.pkl")
    
    # Step 3: Prepare features and target
    print("\n📊 Step 3: Preparing features...")
    feature_cols = get_feature_columns()
    X = df_processed[feature_cols]
    y = df_processed['default_next_30_days']
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    
    # Step 4: Split data
    print("\n✂️ Step 4: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Step 5: Train LightGBM
    print("\n🧠 Step 5: Training LightGBM model...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight='balanced',  # Handle imbalanced classes
        verbose=-1  # Suppress training output
    )
    
    model.fit(X_train, y_train)
    print("   ✅ LightGBM model trained!")
    
    # Step 6: Evaluate
    print("\n📈 Step 6: Model Evaluation...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ap_score = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n   ROC-AUC Score: {roc_auc:.4f}")
    print(f"   Average Precision: {ap_score:.4f}")
    
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    
    print("   Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"   FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Cross-validation
    print("\n🔄 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"   CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Step 7: Feature importance
    print("\n🎯 Feature Importance:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance.head(7).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Step 8: Save model
    print("\n💾 Step 8: Saving model...")
    joblib.dump(model, "ml/model_lgb.pkl")
    print("   ✅ LightGBM model saved to ml/model_lgb.pkl")
    
    print("\n" + "=" * 50)
    print("✅ LightGBM Training Complete!")
    print("=" * 50)
    
    return model, scaler

def compare_models():
    """Compare XGBoost vs LightGBM performance."""
    print("=" * 60)
    print("📊 MODEL COMPARISON: XGBoost vs LightGBM")
    print("=" * 60)
    
    # Load data
    df = load_raw_data()
    df_processed, scaler = process_data(df, fit_scaler=True)
    feature_cols = get_feature_columns()
    X = df_processed[feature_cols]
    y = df_processed['default_next_30_days']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load both models
    xgb_model = joblib.load("ml/model.pkl")
    lgb_model = joblib.load("ml/model_lgb.pkl")
    
    # Evaluate XGBoost
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
    xgb_ap = average_precision_score(y_test, xgb_pred_proba)
    
    # Evaluate LightGBM
    lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_pred_proba)
    lgb_ap = average_precision_score(y_test, lgb_pred_proba)
    
    # Print comparison
    print(f"\n{'Metric':<20} {'XGBoost':<12} {'LightGBM':<12} {'Winner':<10}")
    print("-" * 60)
    print(f"{'ROC-AUC':<20} {xgb_auc:<12.4f} {lgb_auc:<12.4f} {'XGBoost' if xgb_auc >= lgb_auc else 'LightGBM':<10}")
    print(f"{'Avg Precision':<20} {xgb_ap:<12.4f} {lgb_ap:<12.4f} {'XGBoost' if xgb_ap >= lgb_ap else 'LightGBM':<10}")
    
    return {
        'xgboost': {'auc': xgb_auc, 'ap': xgb_ap},
        'lightgbm': {'auc': lgb_auc, 'ap': lgb_ap}
    }

if __name__ == "__main__":
    # Train LightGBM model
    model, scaler = train_lightgbm_model()
    
    # Compare with existing XGBoost model
    print("\n" + "=" * 60)
    comparison = compare_models()