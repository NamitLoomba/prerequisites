# LightGBM API Integration Guide

## Overview

The backend API now supports both XGBoost and LightGBM models with the ability to:
- Use either model for predictions
- Compare both models side-by-side
- Check model availability status

## API Endpoints

### 1. Check Model Status
```bash
GET /api/v1/models/status
```

Response:
```json
{
  "xgboost_available": true,
  "lightgbm_available": true,
  "supported_models": ["xgboost", "lightgbm", "both"]
}
```

### 2. Predict Risk (Single Model)
```bash
POST /api/v1/predict-risk
```

Request body:
```json
{
  "customer_id": "CUST_001",
  "salary_delay_days": 4,
  "savings_drop_pct": 0.30,
  "utility_payment_delay_days": 3,
  "discretionary_spend_drop_pct": 0.25,
  "atm_withdrawal_increase": 3,
  "upi_lending_txn_count": 2,
  "failed_autodebit_count": 1,
  "model_type": "lightgbm"
}
```

Response:
```json
{
  "customer_id": "CUST_001",
  "risk_score": 0.7821,
  "risk_level": "High",
  "will_default": true,
  "confidence": 0.7821,
  "recommended_action": "Priority contact - Propose debt consolidation or payment plan",
  "model_type": "LightGBM"
}
```

### 3. Compare Models
```bash
POST /api/v1/compare-models
```

Request body (same as predict-risk, model_type is ignored):
```json
{
  "customer_id": "CUST_001",
  "salary_delay_days": 4,
  "savings_drop_pct": 0.30,
  "utility_payment_delay_days": 3,
  "discretionary_spend_drop_pct": 0.25,
  "atm_withdrawal_increase": 3,
  "upi_lending_txn_count": 2,
  "failed_autodebit_count": 1
}
```

Response:
```json
{
  "customer_id": "CUST_001",
  "xgboost": {
    "customer_id": "CUST_001",
    "risk_score": 0.7845,
    "risk_level": "High",
    "will_default": true,
    "confidence": 0.7845,
    "recommended_action": "Priority contact - Propose debt consolidation or payment plan",
    "model_type": "XGBoost"
  },
  "lightgbm": {
    "customer_id": "CUST_001",
    "risk_score": 0.7821,
    "risk_level": "High",
    "will_default": true,
    "confidence": 0.7821,
    "recommended_action": "Priority contact - Propose debt consolidation or payment plan",
    "model_type": "LightGBM"
  },
  "score_difference": 0.0024,
  "agreement": true
}
```

### 4. Batch Prediction
```bash
POST /api/v1/predict-batch
```

Request body:
```json
{
  "customers": [
    {
      "customer_id": "CUST_001",
      "salary_delay_days": 1,
      "savings_drop_pct": 0.10,
      "utility_payment_delay_days": 0,
      "discretionary_spend_drop_pct": 0.05,
      "atm_withdrawal_increase": 0,
      "upi_lending_txn_count": 0,
      "failed_autodebit_count": 0,
      "model_type": "lightgbm"
    },
    {
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
  ]
}
```

## Model Selection

The `model_type` parameter accepts:
- `"xgboost"` - Use XGBoost model (default)
- `"lightgbm"` - Use LightGBM model
- `"both"` - Compare both models (only for /compare-models endpoint)

## Quick Start

1. Start the API server:
```bash
python backend/main.py
```

2. Test the integration:
```bash
python test_api_integration.py
```

3. Use curl to test:
```bash
# Check model status
curl http://localhost:8000/api/v1/models/status

# Predict with LightGBM
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
    "model_type": "lightgbm"
  }'

# Compare both models
curl -X POST http://localhost:8000/api/v1/compare-models \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_001",
    "salary_delay_days": 4,
    "savings_drop_pct": 0.30,
    "utility_payment_delay_days": 3,
    "discretionary_spend_drop_pct": 0.25,
    "atm_withdrawal_increase": 3,
    "upi_lending_txn_count": 2,
    "failed_autodebit_count": 1
  }'
```

## Python Client Example

```python
import requests

# Initialize
BASE_URL = "http://localhost:8000/api/v1"

# Check available models
status = requests.get(f"{BASE_URL}/models/status").json()
print(f"LightGBM available: {status['lightgbm_available']}")

# Predict with LightGBM
customer = {
    "customer_id": "CUST_001",
    "salary_delay_days": 4,
    "savings_drop_pct": 0.30,
    "utility_payment_delay_days": 3,
    "discretionary_spend_drop_pct": 0.25,
    "atm_withdrawal_increase": 3,
    "upi_lending_txn_count": 2,
    "failed_autodebit_count": 1,
    "model_type": "lightgbm"
}

result = requests.post(f"{BASE_URL}/predict-risk", json=customer).json()
print(f"Risk Score: {result['risk_score']}")
print(f"Risk Level: {result['risk_level']}")

# Compare models
comparison = requests.post(f"{BASE_URL}/compare-models", json=customer).json()
print(f"XGBoost: {comparison['xgboost']['risk_score']}")
print(f"LightGBM: {comparison['lightgbm']['risk_score']}")
print(f"Difference: {comparison['score_difference']}")
```

## Benefits

- **Flexibility**: Choose the best model for your use case
- **Speed**: LightGBM is typically 5-10x faster
- **Comparison**: Validate predictions across models
- **Backward Compatible**: Existing code works without changes (defaults to XGBoost)
