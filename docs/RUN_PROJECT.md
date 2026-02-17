# How to Run the Pre-Delinquency Risk Platform

## Overview

You have a complete fraud detection system with 3 ML models and a REST API.

## Quick Start

### Option 1: API Only (Recommended)

1. **Start the API server:**
   ```bash
   python backend/main.py
   ```
   
   You should see:
   ```
   ✅ XGBoost model loaded successfully
   ✅ LightGBM model loaded successfully
   INFO: Uvicorn running on http://0.0.0.0:8000
   ```

2. **Test the API (in a new terminal):**
   ```bash
   python test_api_integration.py
   ```

### Option 2: With Frontend Dashboard

1. **Start the API server:**
   ```bash
   python backend/main.py
   ```

2. **Start the dashboard (in a new terminal):**
   ```bash
   streamlit run frontend/dashboard.py
   ```
   
   Opens at: `http://localhost:8501`

## Available Endpoints

### 1. Check Model Status
```bash
curl http://localhost:8000/api/v1/models/status
```

### 2. Predict with XGBoost
```bash
curl -X POST http://localhost:8000/api/v1/predict-risk \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_001",
    "salary_delay_days": 4,
    "savings_drop_pct": 0.30,
    "utility_payment_delay_days": 3,
    "discretionary_spend_drop_pct": 0.25,
    "atm_withdrawal_increase": 3,
    "upi_lending_txn_count": 2,
    "failed_autodebit_count": 1,
    "model_type": "xgboost"
  }'
```

### 3. Predict with LightGBM
```bash
curl -X POST http://localhost:8000/api/v1/predict-risk \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_002",
    "salary_delay_days": 4,
    "savings_drop_pct": 0.30,
    "utility_payment_delay_days": 3,
    "discretionary_spend_drop_pct": 0.25,
    "atm_withdrawal_increase": 3,
    "upi_lending_txn_count": 2,
    "failed_autodebit_count": 1,
    "model_type": "lightgbm"
  }'
```

### 4. Compare Both Models
```bash
curl -X POST http://localhost:8000/api/v1/compare-models \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_003",
    "salary_delay_days": 4,
    "savings_drop_pct": 0.30,
    "utility_payment_delay_days": 3,
    "discretionary_spend_drop_pct": 0.25,
    "atm_withdrawal_increase": 3,
    "upi_lending_txn_count": 2,
    "failed_autodebit_count": 1
  }'
```

## Test Scripts

### Test Traditional Models (XGBoost/LightGBM)
```bash
python test_api_integration.py
```

### Test Sequence Model (LSTM)
```bash
python test_sequence_prediction.py
```

### Compare All Models
```bash
python compare_models.py
```

## Python Usage Examples

### Example 1: Using XGBoost
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict-risk",
    json={
        "customer_id": "CUST_001",
        "salary_delay_days": 4,
        "savings_drop_pct": 0.30,
        "utility_payment_delay_days": 3,
        "discretionary_spend_drop_pct": 0.25,
        "atm_withdrawal_increase": 3,
        "upi_lending_txn_count": 2,
        "failed_autodebit_count": 1,
        "model_type": "xgboost"
    }
)

result = response.json()
print(f"Risk Score: {result['risk_score']}")
print(f"Risk Level: {result['risk_level']}")
```

### Example 2: Using LightGBM
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict-risk",
    json={
        "customer_id": "CUST_002",
        "salary_delay_days": 7,
        "savings_drop_pct": 0.50,
        "utility_payment_delay_days": 5,
        "discretionary_spend_drop_pct": 0.40,
        "atm_withdrawal_increase": 5,
        "upi_lending_txn_count": 4,
        "failed_autodebit_count": 3,
        "model_type": "lightgbm"
    }
)

result = response.json()
print(f"Risk Score: {result['risk_score']}")
print(f"Risk Level: {result['risk_level']}")
```

### Example 3: Using Sequence Model (LSTM)
```python
from ml.sequence_model import load_sequence_model
import numpy as np

# Load model
predictor = load_sequence_model()

# 30 days of customer behavior
customer_sequence = np.random.rand(30, 10)

# Make prediction
result = predictor.predict_sequence(customer_sequence)
print(f"Risk Score: {result['risk_score']}")
print(f"Risk Level: {result['risk_level']}")
```

## Model Comparison

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| XGBoost | Fast | High | General purpose, interpretable |
| LightGBM | Fastest | High | Large datasets, production |
| LSTM Sequence | Slower | Very High | Temporal patterns, trends |

## Retrain Models

### Retrain XGBoost
```bash
python ml/train_model.py
```

### Retrain LightGBM
```bash
python ml/train_lightgbm.py
```

### Retrain Sequence Model
```bash
python ml/train_sequence_model.py
```

## Troubleshooting

### API won't start
- Check if port 8000 is already in use
- Make sure models are trained: `ml/model.pkl`, `ml/model_lgb.pkl`

### Models not found
- Run training scripts first
- Check that model files exist in `ml/` directory

### Import errors
- Install dependencies: `pip install -r requirements.txt`

## Project Structure

```
Predeliquency/
├── backend/
│   ├── main.py              # API server
│   ├── routes.py            # API endpoints
│   ├── risk_engine.py       # Model wrapper
│   └── schemas.py           # Request/response models
├── ml/
│   ├── model.pkl            # XGBoost model
│   ├── model_lgb.pkl        # LightGBM model
│   ├── sequence_model.h5    # LSTM model
│   ├── train_model.py       # Train XGBoost
│   ├── train_lightgbm.py    # Train LightGBM
│   └── train_sequence_model.py  # Train LSTM
├── frontend/
│   └── dashboard.py         # Streamlit dashboard
├── test_api_integration.py  # API tests
└── test_sequence_prediction.py  # Sequence model tests
```

## Next Steps

1. ✅ Start the API: `python backend/main.py`
2. ✅ Test predictions: `python test_api_integration.py`
3. ✅ Integrate into your application
4. ✅ Monitor model performance
5. ✅ Retrain periodically with new data
