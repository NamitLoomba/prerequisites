from feast import Entity, Feature, FeatureView, ValueType, Field
from feast.types import Int64, Float32, String
from datetime import timedelta
import pandas as pd

# Define Entities
customer = Entity(
    name="customer_id",
    value_type=ValueType.INT64,
    description="Customer identifier"
)

# Define Feature Views

# 1. Customer Profile Features
customer_profile_view = FeatureView(
    name="customer_profile",
    entities=["customer_id"],
    ttl=timedelta(days=30),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="income_level", dtype=String),
        Field(name="employment_status", dtype=String),
        Field(name="account_age_days", dtype=Int64),
        Field(name="total_accounts", dtype=Int64),
    ],
    online=True,
    description="Static customer profile information"
)

# 2. Transaction Behavior Features
transaction_behavior_view = FeatureView(
    name="transaction_behavior",
    entities=["customer_id"],
    ttl=timedelta(hours=1),  # Real-time features
    schema=[
        Field(name="daily_transaction_count", dtype=Int64),
        Field(name="daily_transaction_amount", dtype=Float32),
        Field(name="atm_withdrawal_count_today", dtype=Int64),
        Field(name="upi_transaction_count_today", dtype=Int64),
        Field(name="failed_transaction_count_today", dtype=Int64),
        Field(name="salary_received_today", dtype=Int64),
        Field(name="utility_payment_made_today", dtype=Int64),
    ],
    online=True,
    description="Real-time transaction behavior features"
)

# 3. Financial Stress Indicators
stress_indicators_view = FeatureView(
    name="financial_stress_indicators",
    entities=["customer_id"],
    ttl=timedelta(hours=6),
    schema=[
        Field(name="savings_balance_change_pct", dtype=Float32),
        Field(name="salary_delay_days_current", dtype=Int64),
        Field(name="utility_payment_delay_current", dtype=Int64),
        Field(name="discretionary_spending_drop_current", dtype=Float32),
        Field(name="cash_flow_pressure_score", dtype=Float32),
        Field(name="digital_lending_activity", dtype=Int64),
    ],
    online=True,
    description="Dynamic financial stress indicators"
)

# 4. Historical Aggregates
historical_aggregates_view = FeatureView(
    name="historical_aggregates",
    entities=["customer_id"],
    ttl=timedelta(days=7),
    schema=[
        Field(name="avg_monthly_transactions", dtype=Float32),
        Field(name="avg_monthly_balance", dtype=Float32),
        Field(name="transaction_volatility_30d", dtype=Float32),
        Field(name="payment_reliability_score", dtype=Float32),
        Field(name="savings_consistency_score", dtype=Float32),
    ],
    online=True,
    description="Historical behavioral aggregates"
)

# 5. Risk Features (Derived)
risk_features_view = FeatureView(
    name="risk_features",
    entities=["customer_id"],
    ttl=timedelta(hours=1),
    schema=[
        Field(name="delinquency_risk_score", dtype=Float32),
        Field(name="liquidity_ratio_current", dtype=Float32),
        Field(name="stress_index_current", dtype=Float32),
        Field(name="behavioral_anomaly_score", dtype=Float32),
        Field(name="risk_level_category", dtype=String),
    ],
    online=True,
    description="Real-time risk assessment features"
)