import streamlit as st

st.title("🛡️ Test App")
st.write("If you see this, Streamlit is working!")

try:
    import pandas as pd
    st.success("✅ Pandas imported")
except Exception as e:
    st.error(f"❌ Pandas error: {e}")

try:
    import numpy as np
    st.success("✅ NumPy imported")
except Exception as e:
    st.error(f"❌ NumPy error: {e}")

try:
    import plotly
    st.success("✅ Plotly imported")
except Exception as e:
    st.error(f"❌ Plotly error: {e}")

try:
    import joblib
    st.success("✅ Joblib imported")
except Exception as e:
    st.error(f"❌ Joblib error: {e}")

try:
    import xgboost
    st.success("✅ XGBoost imported")
except Exception as e:
    st.error(f"❌ XGBoost error: {e}")

try:
    import lightgbm
    st.success("✅ LightGBM imported")
except Exception as e:
    st.error(f"❌ LightGBM error: {e}")

try:
    import sklearn
    st.success("✅ Scikit-learn imported")
except Exception as e:
    st.error(f"❌ Scikit-learn error: {e}")

import os
st.write("Working directory:", os.getcwd())
st.write("Files:", os.listdir('.'))
if os.path.exists('ml'):
    st.write("ML folder contents:", os.listdir('ml'))
