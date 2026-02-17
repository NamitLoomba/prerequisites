# TensorFlow in Your Project - Complete Explanation

## Where TensorFlow is Used

TensorFlow is used **ONLY** for the **LSTM Sequence Model** (deep learning for temporal patterns).

```
Your Project Models:
├── XGBoost Model          ❌ Does NOT use TensorFlow (uses XGBoost library)
├── LightGBM Model         ❌ Does NOT use TensorFlow (uses LightGBM library)
└── LSTM Sequence Model    ✅ USES TensorFlow (deep learning)
```

## Visual Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR PROJECT                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Traditional ML Models (No TensorFlow)                      │
│  ┌──────────────┐        ┌──────────────┐                  │
│  │   XGBoost    │        │  LightGBM    │                  │
│  │              │        │              │                  │
│  │ Uses: XGBoost│        │ Uses: LightGBM│                 │
│  │ library      │        │ library      │                  │
│  └──────────────┘        └──────────────┘                  │
│         ↓                        ↓                          │
│    Single snapshot          Single snapshot                │
│    prediction               prediction                      │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Deep Learning Model (Uses TensorFlow)                      │
│  ┌──────────────────────────────────────┐                  │
│  │      LSTM Sequence Model             │                  │
│  │                                       │                  │
│  │  ┌─────────────────────────────┐    │                  │
│  │  │   TensorFlow/Keras          │    │                  │
│  │  │                              │    │                  │
│  │  │  • LSTM Layers (64, 32)     │    │                  │
│  │  │  • Dense Layers             │    │                  │
│  │  │  • Dropout & BatchNorm      │    │                  │
│  │  │  • Adam Optimizer           │    │                  │
│  │  └─────────────────────────────┘    │                  │
│  └──────────────────────────────────────┘                  │
│         ↓                                                    │
│    30-day sequence prediction                               │
│    (temporal patterns)                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## TensorFlow Components in Your Project

### 1. File: `ml/sequence_model.py`

This is where ALL TensorFlow code lives:

```python
import tensorflow as tf                    # ← TensorFlow main library
from tensorflow.keras.models import Sequential  # ← Model architecture
from tensorflow.keras.layers import (
    LSTM,                                  # ← Recurrent neural network layer
    Dense,                                 # ← Fully connected layer
    Dropout,                               # ← Regularization
    BatchNormalization,                    # ← Normalization
    Conv1D,                                # ← Convolutional layer (optional)
    MaxPooling1D                           # ← Pooling layer (optional)
)
from tensorflow.keras.optimizers import Adam  # ← Optimization algorithm
from tensorflow.keras.callbacks import (
    EarlyStopping,                         # ← Stop training early
    ReduceLROnPlateau                      # ← Adjust learning rate
)
```

### 2. What TensorFlow Does

#### Building the Neural Network:
```python
model = Sequential([
    # Layer 1: LSTM with 64 units
    LSTM(64, return_sequences=True, input_shape=(30, 10)),
    BatchNormalization(),
    Dropout(0.2),
    
    # Layer 2: LSTM with 32 units
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),
    
    # Layer 3: Dense layer
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    # Output layer
    Dense(1, activation='sigmoid')  # Probability of default
])
```

#### Training the Model:
```python
model.compile(
    optimizer=Adam(learning_rate=0.001),  # How to update weights
    loss='binary_crossentropy',           # What to minimize
    metrics=['accuracy', 'precision', 'recall']  # What to track
)

# Train for up to 50 epochs
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    callbacks=[EarlyStopping(), ReduceLROnPlateau()]
)
```

#### Making Predictions:
```python
# Load saved model
model = tf.keras.models.load_model("ml/sequence_model.h5")

# Predict
risk_probability = model.predict(customer_sequence)
```

## How Data Flows Through TensorFlow

```
Customer 30-Day History
         ↓
┌────────────────────┐
│  Input: (30, 10)   │  30 days, 10 features per day
└────────────────────┘
         ↓
┌────────────────────┐
│  LSTM Layer (64)   │  ← TensorFlow processes sequences
└────────────────────┘
         ↓
┌────────────────────┐
│  LSTM Layer (32)   │  ← Learns temporal patterns
└────────────────────┘
         ↓
┌────────────────────┐
│  Dense Layer (32)  │  ← Combines features
└────────────────────┘
         ↓
┌────────────────────┐
│  Output: (1)       │  Risk probability: 0.0 to 1.0
└────────────────────┘
```

## Files That Use TensorFlow

| File | Uses TensorFlow? | Purpose |
|------|------------------|---------|
| `ml/sequence_model.py` | ✅ YES | Defines LSTM model architecture |
| `ml/train_sequence_model.py` | ✅ YES | Trains the LSTM model |
| `test_sequence_prediction.py` | ✅ YES | Tests LSTM predictions |
| `ml/train_model.py` | ❌ NO | Trains XGBoost (different library) |
| `ml/train_lightgbm.py` | ❌ NO | Trains LightGBM (different library) |
| `ml/predict.py` | ❌ NO | XGBoost predictions |
| `ml/predict_lightgbm.py` | ❌ NO | LightGBM predictions |
| `backend/main.py` | ❌ NO | API server (no TensorFlow) |

## Why TensorFlow for Sequence Model?

### Traditional ML (XGBoost/LightGBM):
```python
# Single snapshot - no time dimension
customer = {
    "salary_delay_days": 4,
    "savings_drop_pct": 0.30,
    ...
}
prediction = model.predict(customer)  # One point in time
```

### TensorFlow LSTM:
```python
# 30-day sequence - time dimension matters
customer_history = [
    [50000, 1, 20000, ...],  # Day 1
    [50000, 0, 19500, ...],  # Day 2
    [50000, 0, 19000, ...],  # Day 3
    ...                       # Days 4-30
]
prediction = model.predict(customer_history)  # Sees trends over time
```

## TensorFlow Model Architecture Visualization

```
Input Shape: (30 days, 10 features)
         ↓
    ┌─────────┐
    │ LSTM 64 │  ← Remembers patterns from day 1-30
    └─────────┘
         ↓
    ┌─────────┐
    │BatchNorm│  ← Normalizes activations
    └─────────┘
         ↓
    ┌─────────┐
    │Dropout  │  ← Prevents overfitting (20%)
    └─────────┘
         ↓
    ┌─────────┐
    │ LSTM 32 │  ← Refines patterns
    └─────────┘
         ↓
    ┌─────────┐
    │BatchNorm│
    └─────────┘
         ↓
    ┌─────────┐
    │Dropout  │
    └─────────┘
         ↓
    ┌─────────┐
    │Dense 32 │  ← Combines features
    └─────────┘
         ↓
    ┌─────────┐
    │BatchNorm│
    └─────────┘
         ↓
    ┌─────────┐
    │Dropout  │
    └─────────┘
         ↓
    ┌─────────┐
    │Dense 1  │  ← Output: Risk probability
    │(sigmoid)│
    └─────────┘
         ↓
    Risk Score: 0.0 - 1.0
```

## Saved TensorFlow Model

After training, TensorFlow saves the model:

```
ml/sequence_model.h5  (465 KB)
```

This file contains:
- All layer weights and biases
- Model architecture
- Optimizer state
- Training configuration

## How to See TensorFlow in Action

### 1. View Model Architecture:
```python
from ml.sequence_model import load_sequence_model

predictor = load_sequence_model()
predictor.model.summary()  # Shows TensorFlow model structure
```

### 2. Make Predictions:
```python
import numpy as np
from ml.sequence_model import load_sequence_model

# Load TensorFlow model
predictor = load_sequence_model()

# 30 days of customer data
sequence = np.random.rand(30, 10)

# TensorFlow processes this
result = predictor.predict_sequence(sequence)
print(result)
```

### 3. Train the Model:
```bash
python ml/train_sequence_model.py
```

Watch TensorFlow train:
```
Epoch 1/50
940/940 ━━━━━━━━━━━━━━━━━━━━ 24s 20ms/step
Epoch 2/50
940/940 ━━━━━━━━━━━━━━━━━━━━ 17s 16ms/step
...
```

## Key Differences

| Aspect | XGBoost/LightGBM | TensorFlow LSTM |
|--------|------------------|-----------------|
| Library | XGBoost/LightGBM | TensorFlow/Keras |
| Model Type | Decision Trees | Neural Network |
| Input | Single snapshot | Time sequence |
| Training | Fast (seconds) | Slower (minutes) |
| Memory | Low | Higher |
| Temporal Patterns | ❌ No | ✅ Yes |
| File Size | Small (~1MB) | Larger (~465KB) |

## Summary

**TensorFlow is used exclusively for the LSTM Sequence Model** to:
1. Build neural network architecture (LSTM layers)
2. Train on sequential data (30-day patterns)
3. Detect temporal trends (deteriorating behavior)
4. Make predictions on time-series data

**XGBoost and LightGBM don't use TensorFlow** - they use their own libraries for traditional machine learning on single data points.

The sequence model complements the traditional models by adding the ability to see how customer behavior changes over time, not just their current state.
