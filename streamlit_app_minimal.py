import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import numpy as np

st.set_page_config(page_title='Pre-Delinquency Risk Platform', page_icon='🛡️', layout='wide')

st.title('🛡️ AI-Powered Pre-Delinquency Risk Platform')
st.markdown('### Early Intervention System for Financial Risk Management')

# Test if we can load models
try:
    st.write("Testing model loading...")
    model = joblib.load('ml/model.pkl')
    scaler = joblib.load('ml/scaler.pkl')
    st.success("✅ XGBoost model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading XGBoost model: {e}")

try:
    model_lgb = joblib.load('ml/model_lgb.pkl')
    scaler_lgb = joblib.load('ml/scaler_lgb.pkl')
    st.success("✅ LightGBM model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading LightGBM model: {e}")

try:
    import tensorflow as tf
    lstm_model = tf.keras.models.load_model('ml/sequence_model.h5')
    st.success("✅ TensorFlow LSTM model loaded successfully!")
except Exception as e:
    st.warning(f"⚠️ TensorFlow LSTM not available: {e}")

st.write("---")
st.write("If all models loaded successfully, the full app will work!")
st.write("Check the errors above to see what's missing.")
