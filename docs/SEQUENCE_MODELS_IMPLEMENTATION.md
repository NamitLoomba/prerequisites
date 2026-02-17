# Deep Learning Sequence Models Implementation Guide

## 🚀 TensorFlow/Keras Sequence Models Added

Deep learning sequence models have been implemented to detect temporal patterns in customer behavior for delinquency prediction.

## 📁 Files Created

1. **`ml/sequence_data_generator.py`** - Generates synthetic sequential transaction data
2. **`ml/sequence_model.py`** - LSTM and CNN-LSTM models for sequence prediction
3. **`ml/train_sequence_model.py`** - Complete training pipeline
4. **`requirements.txt`** - Updated with `tensorflow>=2.13.0`

## 🛠️ Implementation Details

### Sequence Data Generation
- **Temporal patterns**: 30-90 day customer behavior sequences
- **Features**: Salary, savings, expenses, utility payments, ATM withdrawals, UPI transactions
- **Default patterns**: Deteriorating behavior over time for defaulters
- **Data format**: (samples, timesteps, features)

### Model Architectures

#### 1. LSTM Model
```python
LSTM(64) → LSTM(32) → Dense(32) → Dense(1)
```
- Captures long-term dependencies in customer behavior
- 2-layer LSTM architecture with dropout regularization

#### 2. CNN-LSTM Hybrid
```python
Conv1D(32) → Conv1D(64) → LSTM(64) → LSTM(32) → Dense(32) → Dense(1)
```
- CNN layers detect local patterns (weekly/monthly cycles)
- LSTM layers capture long-term trends
- Best for complex temporal patterns

## 📊 Usage Examples

### 1. Generate and Train Sequence Model
```bash
python ml/train_sequence_model.py
```

### 2. Use Pre-trained Sequence Model
```python
from ml.sequence_model import load_sequence_model
import numpy as np

# Load model
predictor = load_sequence_model()

# Prepare sequence data (30 days of customer behavior)
sequence_data = np.random.rand(30, 10)  # 30 days, 10 features

# Predict
result = predictor.predict_sequence(sequence_data)
print(f"Risk Score: {result['risk_score']}")
print(f"Risk Level: {result['risk_level']}")
```

### 3. Generate Custom Sequential Data
```python
from ml.sequence_data_generator import generate_sequence_data, create_sequences

# Generate data
df = generate_sequence_data(n_customers=500, sequence_length=60)

# Create training sequences
X, y = create_sequences(df, sequence_length=30, target_days_ahead=14)
```

## 🔧 Key Features

### Data Generation
- **Realistic patterns**: Salary cycles, spending behaviors, default indicators
- **Temporal deterioration**: Defaulters show worsening patterns over time
- **Feature richness**: 10 behavioral features per time step

### Model Capabilities
- **Sequence learning**: Captures temporal dependencies
- **Pattern recognition**: Detects early warning signals
- **Flexible architecture**: LSTM or CNN-LSTM options
- **Automatic scaling**: Built-in feature normalization

## 📈 Benefits Over Traditional Models

1. **Temporal Awareness**: Understands *when* behaviors occur, not just *what*
2. **Pattern Detection**: Identifies deteriorating trends before they become critical
3. **Early Warning**: Can detect subtle changes in behavior patterns
4. **Contextual Understanding**: Considers behavioral sequences over time

## 🎯 When to Use Sequence Models

**Use Sequence Models When:**
- You have time-series customer data
- Behavior patterns evolve over time
- Early detection of deteriorating patterns is crucial
- You want to understand temporal risk progression

**Use Traditional Models When:**
- You only have aggregated/static features
- You need fast, simple predictions
- Computational resources are limited
- Interpretability is paramount

## 🔄 Integration with Existing Pipeline

Sequence models complement your existing XGBoost/LightGBM models:

```python
# Traditional approach - static snapshot
xgb_risk = xgb_model.predict(customer_features)

# Sequence approach - temporal analysis
seq_risk = sequence_model.predict_sequence(customer_history)

# Combined approach
final_risk = 0.7 * seq_risk['risk_score'] + 0.3 * xgb_risk['risk_score']
```

## 📊 Expected Performance

Sequence models typically show:
- **10-15% better early detection** for deteriorating patterns
- **Higher precision** for high-risk customers
- **Better recall** for identifying subtle warning signs
- **Complementary insights** to traditional models

## 🚀 Next Steps

1. **Collect real sequential data** from your banking system
2. **Fine-tune models** with actual customer behavior patterns
3. **Implement real-time sequence scoring** in production
4. **Create ensemble** combining sequence + traditional models