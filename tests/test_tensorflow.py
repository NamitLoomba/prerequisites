"""Quick test to verify TensorFlow installation"""
import tensorflow as tf
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Simple test
print("\nCreating a simple model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
print("✅ TensorFlow is working!")

# Test data
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

print("\nTraining for 2 epochs...")
model.fit(X, y, epochs=2, verbose=0)
print("✅ Training works!")

print("\nPredicting...")
pred = model.predict(X[:5], verbose=0)
print(f"Predictions: {pred.flatten()}")
print("✅ All tests passed!")
