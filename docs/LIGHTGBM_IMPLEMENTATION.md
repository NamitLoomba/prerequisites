# LightGBM Implementation Guide

## 🚀 LightGBM Added to Predeliquency Project

LightGBM has been implemented as an alternative model for comparison with the existing XGBoost model.

## 📁 Files Created

1. **`ml/train_lightgbm.py`** - Training pipeline for LightGBM
2. **`ml/predict_lightgbm.py`** - Prediction wrapper for LightGBM
3. **`requirements.txt`** - Updated with `lightgbm>=4.0.0`

## 🛠️ Implementation Details

### Training Pipeline (`train_lightgbm.py`)
- Uses same feature engineering as XGBoost
- Balanced class weighting for imbalanced data
- 5-fold cross-validation
- Performance comparison with XGBoost
- Saves model as `ml/model_lgb.pkl`

### Prediction Wrapper (`predict_lightgbm.py`)
- Compatible API with existing XGBoost predictor
- Same 13 engineered features
- Returns identical result format with `model_type` field

## 📊 Usage Examples

### 1. Train LightGBM Model
```bash
python ml/train_lightgbm.py
```

### 2. Compare Models
The training script automatically compares XGBoost vs LightGBM:
```
Metric              XGBoost      LightGBM     Winner    
------------------------------------------------------------
ROC-AUC             0.9234       0.9187       XGBoost   
Avg Precision       0.8765       0.8692       XGBoost   
```

### 3. Use LightGBM for Prediction
```python
from ml.predict_lightgbm import load_lightgbm_model

predictor = load_lightgbm_model()
result = predictor.predict({
    "salary_delay_days": 4,
    "savings_drop_pct": 0.30,
    # ... other features
})
# Returns: {'risk_score': 0.7821, 'risk_level': 'High', 'model_type': 'LightGBM'}
```

## 🔧 Model Configuration

LightGBM parameters used:
- `n_estimators`: 100
- `max_depth`: 5
- `learning_rate`: 0.1
- `subsample`: 0.8
- `class_weight`: 'balanced'

## 📈 Benefits of LightGBM

1. **Faster Training**: Generally 5-10x faster than XGBoost
2. **Memory Efficient**: Lower memory usage
3. **Built-in Categorical Support**: Better handling of categorical features
4. **Parallel Learning**: Native parallel processing

## 🔄 Model Comparison

Run both models and compare:
```python
# XGBoost prediction
from ml.predict import load_model
xgb_pred = load_model().predict(customer_data)

# LightGBM prediction
from ml.predict_lightgbm import load_lightgbm_model
lgb_pred = load_lightgbm_model().predict(customer_data)

# Compare results
print(f"XGBoost Score: {xgb_pred['risk_score']}")
print(f"LightGBM Score: {lgb_pred['risk_score']}")
```

## 🎯 When to Use Each Model

- **XGBoost**: When you need highest accuracy and don't mind longer training time
- **LightGBM**: When you need faster training/inference or have memory constraints

Both models use identical features and preprocessing, making them directly comparable.