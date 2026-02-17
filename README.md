# AI-Powered Pre-Delinquency Risk Platform

An enterprise-grade solution for early detection and prevention of loan delinquency using machine learning.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1579B9?style=for-the-badge)

## Problem Statement

Financial institutions lose billions annually to loan defaults. Traditional risk assessment reacts AFTER delinquency occurs. Our platform predicts delinquency 2-4 weeks BEFORE it happens, enabling proactive intervention.

## Solution

An AI-powered early warning system that:
- Analyzes transaction behavior patterns
- Identifies early stress signals
- Predicts delinquency probability
- Recommends actionable interventions

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Model | XGBoost Classifier |
| Feature Engineering | Custom stress signal creation |
| Backend | FastAPI |
| Frontend | Streamlit |
| Visualization | Plotly |

## Key Features

- **Explainable AI**: Feature importance shows WHY a customer is flagged
- **Risk Scoring**: 0-100% probability of default in 2-4 weeks
- **Intervention Recommendations**: Actionable next steps for each risk level
- **Portfolio Analysis**: Batch processing for portfolio-level assessment
- **Scenario Simulation**: Test different risk scenarios

## Model Details

- **Algorithm**: XGBoost (Gradient Boosted Trees)
- **Prediction Horizon**: 2-4 weeks ahead
- **Features**: 13 engineered features including:
  - Stress index
  - Liquidity ratio
  - Payment reliability score
  - Cash flow pressure
  - Digital stress indicators
- **Training Data**: Synthetic (privacy-safe)

## Getting Started

### Local Installation

`ash
# Clone the repository
git clone https://github.com/yourusername/predelinquency.git
cd predelinquency

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
`

### Deployment

This app is deployed on Streamlit Cloud. Simply connect your GitHub repository to Streamlit Cloud for instant deployment.

## Usage

1. **Single Customer Risk Assessment**: Enter customer financial behavior data to get instant risk score
2. **Portfolio Overview**: Analyze entire customer portfolio for risk distribution
3. **Scenario Simulation**: Test different risk scenarios

## Risk Levels

| Level | Score Range | Action |
|-------|-------------|--------|
| Low | 0-25% | Continue standard monitoring |
| Medium | 25-50% | Schedule check-in call |
| High | 50-75% | Propose payment plan |
| Critical | 75-100% | Offer payment holiday |

## Data Schema

| Feature | Description |
|---------|-------------|
| customer_id | Unique customer identifier |
| salary_delay_days | Days since salary credit delay |
| savings_drop_pct | Week-over-week savings decline |
| utility_payment_delay_days | Days of utility bill delay |
| discretionary_spend_drop_pct | Reduction in lifestyle spending |
| atm_withdrawal_increase | Extra ATM withdrawals |
| upi_lending_txn_count | Lending app transactions |
| failed_autodebit_count | Failed EMI payments |

## Architecture

`
Synthetic Data  Feature Engineering  XGBoost Model  Risk Score  Intervention
`

## Disclaimer

This is a simulated demo using synthetic data for privacy protection. Not for actual financial decision-making.

## License

MIT License

---

Built for Hackathon 2026
