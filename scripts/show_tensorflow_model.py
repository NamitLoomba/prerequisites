"""
Show TensorFlow model architecture and details
"""

from ml.sequence_model import load_sequence_model
import numpy as np

print("="*60)
print("TensorFlow LSTM Model Details")
print("="*60)

# Load the TensorFlow model
print("\n1. Loading TensorFlow model...")
predictor = load_sequence_model()

print("\n2. Model Architecture (TensorFlow/Keras):")
print("-"*60)
predictor.model.summary()

print("\n3. Model Details:")
print("-"*60)
print(f"   Total Parameters: {predictor.model.count_params():,}")
print(f"   Input Shape: (30 days, 10 features)")
print(f"   Output Shape: (1 probability)")
print(f"   Model Type: {type(predictor.model)}")

print("\n4. Layer Information:")
print("-"*60)
for i, layer in enumerate(predictor.model.layers):
    print(f"   Layer {i+1}: {layer.name} ({layer.__class__.__name__})")
    if hasattr(layer, 'units'):
        print(f"           Units: {layer.units}")

print("\n5. Making a Test Prediction with TensorFlow:")
print("-"*60)

# Create sample data
test_sequence = np.random.rand(30, 10)
print(f"   Input: 30 days × 10 features = {test_sequence.shape}")

# TensorFlow processes this
result = predictor.predict_sequence(test_sequence)

print(f"\n   TensorFlow Output:")
print(f"   - Risk Score: {result['risk_score']:.4f}")
print(f"   - Risk Level: {result['risk_level']}")
print(f"   - Will Default: {result['will_default']}")
print(f"   - Model Type: {result['model_type']}")

print("\n6. Model File:")
print("-"*60)
print(f"   Saved at: ml/sequence_model.h5")
print(f"   This is a TensorFlow/Keras HDF5 file")

print("\n" + "="*60)
print("✅ TensorFlow is working in your sequence model!")
print("="*60)
