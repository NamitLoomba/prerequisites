import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, 
    Input, TimeDistributed, Flatten, Conv1D, MaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os

class SequenceRiskPredictor:
    """Deep learning sequence model for delinquency prediction."""
    
    def __init__(self, model_path="ml/sequence_model.h5", scaler_path="ml/sequence_scaler.pkl"):
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        
    def build_lstm_model(self, input_shape):
        """Build LSTM model for sequence prediction."""
        model = Sequential([
            # First LSTM layer
            LSTM(64, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_cnn_lstm_model(self, input_shape):
        """Build CNN-LSTM hybrid model."""
        model = Sequential([
            # CNN layers for local pattern detection
            Conv1D(32, 3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            
            Conv1D(64, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            
            # LSTM layers
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_model(self, X, y, model_type='lstm'):
        """Train sequence model."""
        print("=" * 50)
        print(f"🚀 Training {model_type.upper()} Sequence Model")
        print("=" * 50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Class distribution: {np.bincount(y_train)} / {np.bincount(y_test)}")
        
        # Scale features
        print("\n📊 Scaling features...")
        self.scaler = StandardScaler()
        
        # Reshape for scaling (samples, timesteps, features) -> (samples*timesteps, features)
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        
        # Reshape back
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        X_test_scaled = X_test_scaled.reshape(X_test.shape[0], n_timesteps, n_features)
        
        # Save scaler
        joblib.dump(self.scaler, self.scaler_path)
        print("✅ Scaler saved")
        
        # Build model
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        
        if model_type == 'lstm':
            self.model = self.build_lstm_model(input_shape)
        elif model_type == 'cnn_lstm':
            self.model = self.build_cnn_lstm_model(input_shape)
        else:
            raise ValueError("model_type must be 'lstm' or 'cnn_lstm'")
        
        print(f"\n📊 Model Architecture ({model_type}):")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        print(f"\n🧠 Training {model_type} model...")
        history = self.model.fit(
            X_train_scaled, y_train,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        print("\n📈 Model Evaluation:")
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
            X_test_scaled, y_test, verbose=0
        )
        
        y_pred_proba = self.model.predict(X_test_scaled).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
        
        # Save model
        self.model.save(self.model_path)
        print(f"✅ Model saved to {self.model_path}")
        
        return history
    
    def predict_sequence(self, sequence_data):
        """
        Predict risk for a sequence of customer data.
        
        Args:
            sequence_data: numpy array of shape (timesteps, features)
            
        Returns:
            dict with risk prediction
        """
        if self.model is None or self.scaler is None:
            self.load_model()
        
        # Ensure correct shape
        if len(sequence_data.shape) == 2:
            sequence_data = np.expand_dims(sequence_data, axis=0)
        
        # Scale data
        n_samples, n_timesteps, n_features = sequence_data.shape
        sequence_reshaped = sequence_data.reshape(-1, n_features)
        sequence_scaled = self.scaler.transform(sequence_reshaped)
        sequence_scaled = sequence_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Predict
        risk_prob = self.model.predict(sequence_scaled)[0][0]
        
        # Determine risk level
        if risk_prob >= 0.75:
            risk_level = "Critical"
        elif risk_prob >= 0.50:
            risk_level = "High"
        elif risk_prob >= 0.25:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "risk_score": float(risk_prob),
            "risk_level": risk_level,
            "will_default": bool(risk_prob >= 0.5),
            "confidence": float(max(risk_prob, 1 - risk_prob)),
            "model_type": "LSTM"
        }
    
    def load_model(self):
        """Load trained model and scaler."""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("✅ Sequence model and scaler loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

def load_sequence_model(model_path="ml/sequence_model.h5", scaler_path="ml/sequence_scaler.pkl"):
    """Convenience function to load sequence predictor."""
    predictor = SequenceRiskPredictor(model_path, scaler_path)
    predictor.load_model()
    return predictor

if __name__ == "__main__":
    # This would be used for training
    print("SequenceRiskPredictor - Ready for training")
    print("Use train_model() method with sequential data")