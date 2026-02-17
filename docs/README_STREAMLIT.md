# 🛡️ AI-Powered Pre-Delinquency Risk Platform

An enterprise-grade solution for early detection and prevention of loan delinquency using multiple AI/ML models.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_APP_URL_HERE)

## 🚀 Live Demo

**Try it now:** [YOUR_APP_URL_HERE]

## 📋 Overview

This platform uses machine learning to predict customer delinquency risk 2-4 weeks in advance, enabling proactive intervention and reducing default rates.

### Key Features

- 🤖 **Multi-Model AI**: Choose between XGBoost and LightGBM models
- 📊 **Real-Time Risk Scoring**: 0-100% probability of default
- 🔄 **Model Comparison**: Side-by-side predictions from both models
- 💡 **Intervention Recommendations**: Actionable next steps for each risk level
- 📈 **Interactive Dashboard**: Built with Streamlit and Plotly
- ⚡ **Fast Predictions**: Sub-second inference latency

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| ML Models | XGBoost, LightGBM |
| Feature Engineering | scikit-learn |
| Frontend | Streamlit |
| Visualization | Plotly |
| Data Processing | Pandas, NumPy |

## 📊 Model Performance

- **XGBoost**: 95% accuracy on test data
- **LightGBM**: 95% accuracy with faster inference
- **Prediction Horizon**: 2-4 weeks ahead
- **Features**: 13 engineered risk indicators

## 🎯 Risk Indicators

The models analyze 7 key behavioral signals:

1. **Salary Delay** - Days of delayed salary credit
2. **Savings Decline** - Percentage drop in savings balance
3. **Utility Payment Delay** - Days of delayed bill payments
4. **Discretionary Spending Drop** - Reduction in non-essential spending
5. **ATM Withdrawal Increase** - Extra cash withdrawals
6. **UPI Lending Transactions** - Transfers to lending apps
7. **Failed Auto-debits** - Bounced EMI/automatic payments

## 🚀 Quick Start

### Run Locally

```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Run the app
streamlit run streamlit_app.py

# Open browser to http://localhost:8501
```

### Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub account
4. Select this repository
5. Set main file: `streamlit_app.py`
6. Click "Deploy"

## 📁 Project Structure

```
├── streamlit_app.py              # Main application
├── requirements_streamlit.txt    # Dependencies
├── .streamlit/
│   └── config.toml              # Streamlit config
├── ml/
│   ├── model.pkl                # XGBoost model
│   ├── scaler.pkl               # XGBoost scaler
│   ├── model_lgb.pkl            # LightGBM model
│   └── scaler_lgb.pkl           # LightGBM scaler
└── README.md                    # This file
```

## 🎨 Screenshots

### Risk Prediction
![Risk Prediction](https://via.placeholder.com/800x400?text=Risk+Prediction+Dashboard)

### Model Comparison
![Model Comparison](https://via.placeholder.com/800x400?text=Model+Comparison+View)

## 📈 Use Cases

- **Banks**: Early warning system for retail loan portfolios
- **NBFCs**: Proactive customer outreach programs
- **Fintech**: Risk-based credit limit adjustments
- **Collections**: Prioritize high-risk accounts

## 🔒 Privacy & Security

- Uses synthetic data for demonstration
- No real customer information
- Privacy-compliant design
- Secure model inference

## 📝 How It Works

1. **Input**: Enter customer behavioral indicators
2. **Feature Engineering**: Create 13 derived features
3. **Model Inference**: XGBoost or LightGBM prediction
4. **Risk Classification**: Low / Medium / High / Critical
5. **Recommendation**: Suggested intervention action

## 🎓 Model Details

### XGBoost Model
- Algorithm: Gradient Boosted Decision Trees
- Features: 13 engineered features
- Training: 10,000 synthetic samples
- Accuracy: 95%

### LightGBM Model
- Algorithm: Leaf-wise tree growth
- Features: Same 13 features
- Training: Same dataset
- Advantage: Faster inference

## 🤝 Contributing

This is a hackathon project. Contributions welcome!

## 📄 License

MIT License - See LICENSE file for details

## 👥 Team

Built for [Hackathon Name] by [Your Team Name]

## 🙏 Acknowledgments

- Synthetic data generation using scikit-learn
- UI framework by Streamlit
- Visualization by Plotly

## 📞 Contact

- GitHub: [@NamitLoomba](https://github.com/NamitLoomba)
- Email: your.email@example.com

---

**⚠️ Disclaimer**: This is a demonstration project using synthetic data. Not for actual financial decision-making.

**🏆 Built for [Hackathon Name] - [Year]**
