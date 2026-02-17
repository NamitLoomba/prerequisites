from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os

sys.path.insert(0, os.getcwd())

from backend.schemas import RiskInput, RiskOutput, BatchRiskInput, BatchRiskOutput, ComparisonOutput, SequenceRiskInput
from backend.risk_engine import risk_engine
from fastapi import APIRouter, HTTPException

app = FastAPI(title='Pre-Delinquency Risk API', version='1.0.0')

app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

router = APIRouter(prefix='/api/v1')

@router.get('/')
def root():
    return {'status': 'healthy', 'service': 'Pre-Delinquency Risk API'}

@router.post('/predict-risk', response_model=RiskOutput)
def predict_risk(customer: RiskInput):
    """Predict risk using specified model (xgboost or lightgbm)."""
    try:
        result = risk_engine.score_customer(customer.dict())
        return RiskOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/predict-sequence', response_model=RiskOutput)
def predict_sequence(request: SequenceRiskInput):
    """Predict risk using TensorFlow LSTM sequence model (30 days of data)."""
    try:
        result = risk_engine.score_sequence(request.customer_id, request.sequence_data)
        return RiskOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/compare-models', response_model=ComparisonOutput)
def compare_models(customer: RiskInput):
    """Compare predictions from both XGBoost and LightGBM models."""
    try:
        customer_data = customer.dict()
        customer_data['model_type'] = 'both'
        result = risk_engine.score_customer(customer_data)
        return ComparisonOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/predict-batch', response_model=BatchRiskOutput)
def predict_batch(batch_input: BatchRiskInput):
    """Batch prediction using specified model."""
    try:
        customers = [c.dict() for c in batch_input.customers]
        result = risk_engine.score_batch(customers)
        return BatchRiskOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/models/status')
def models_status():
    """Check which models are loaded and available."""
    return {
        'xgboost_available': risk_engine.xgb_predictor is not None,
        'lightgbm_available': risk_engine.lgb_predictor is not None,
        'lstm_tensorflow_available': risk_engine.lstm_predictor is not None,
        'supported_models': ['xgboost', 'lightgbm', 'lstm', 'both']
    }

app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
