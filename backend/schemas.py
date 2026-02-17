from pydantic import BaseModel, Field
from typing import Literal, List

class RiskInput(BaseModel):
    customer_id: str = Field(default='CUST_001')
    salary_delay_days: int = Field(ge=0, le=30)
    savings_drop_pct: float = Field(ge=0.0, le=1.0)
    utility_payment_delay_days: int = Field(ge=0, le=30)
    discretionary_spend_drop_pct: float = Field(ge=0.0, le=1.0)
    atm_withdrawal_increase: int = Field(ge=0)
    upi_lending_txn_count: int = Field(ge=0)
    failed_autodebit_count: int = Field(ge=0)
    model_type: Literal['xgboost', 'lightgbm', 'lstm', 'both'] = Field(default='xgboost', description='Model to use for prediction')

class SequenceRiskInput(BaseModel):
    customer_id: str = Field(default='CUST_001')
    sequence_data: List[List[float]] = Field(description='30 days of customer behavior (30x10 array)')
    # Each day has 10 features: salary_amount, salary_received, savings_amount, daily_expenses,
    # utility_payment, utility_delay_days, atm_withdrawals, upi_lending_transactions, 
    # failed_autodebits, discretionary_spending

class RiskOutput(BaseModel):
    customer_id: str
    risk_score: float
    risk_level: str
    will_default: bool
    confidence: float
    recommended_action: str
    model_type: str

class ComparisonOutput(BaseModel):
    customer_id: str
    xgboost: RiskOutput
    lightgbm: RiskOutput
    score_difference: float
    agreement: bool

class BatchRiskInput(BaseModel):
    customers: list[RiskInput]

class BatchRiskOutput(BaseModel):
    predictions: list[RiskOutput]
    total_customers: int
    high_risk_count: int
    critical_risk_count: int
