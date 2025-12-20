import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import pickle
import json
import plotly.graph_objects as go

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(page_title="AQI Forecasting", page_icon="🌤️", layout="wide")
st.markdown("<h1 style='text-align:center; color:#70B7FF;'>🌤️ Air Quality Forecasting</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict future air pollution levels</p>", unsafe_allow_html=True)
st.markdown("---")

# ===========================
# PATHS
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_KERAS = os.path.join(BASE_DIR, "models", "best_lstm_model.keras")
MODEL_H5 = os.path.join(BASE_DIR, "models", "best_lstm_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

# ===========================
# BUILD MODEL FUNCTION
# ===========================
def create_model(window_size):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='tanh', input_shape=(window_size, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ===========================
# LOAD MODEL & SCALER
# ===========================
@st.cache_resource
def load_model_and_scaler():
    try:
        import tensorflow as tf
        
        # Load metrics
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        window_size = metrics['window_size']
        
        # Build model and load weights
        model = create_model(window_size)
        
        if os.path.exists(MODEL_H5):
            model.load_weights(MODEL_H5)
        elif os.path.exists(MODEL_KERAS):
            model.load_weights(MODEL_KERAS)
        else:
            raise FileNotFoundError("No model file found")
        
        # Load scaler
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        
        return model, scaler, metrics
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Make sure model files are in the 'models/' folder")
        st.stop()

# ===========================
# LOAD EVERYTHING
# ===========================
model, scaler, metrics = load_model_and_scaler()
WINDOW_SIZE = metrics["window_size"]
TARGET = metrics["target_column"]

# ===========================
# SIDEBAR
# ===========================
with st.sidebar:
    st.markdown("### 🤖 LSTM Model")
    st.success("✅ Model Ready")
    st.markdown("---")
    st.markdown(f"**Target:** {TARGET}")
    st.markdown(f"**Lookback:** {WINDOW_SIZE} days")
    if metrics:
        st.markdown("### 📊 Performance")
        tm = metrics.get('test_metrics', {})
        st.metric("R²", f"{tm.get('r2', 0):.4f}")
        st.metric("MAE", f"{tm.get('mae', 0):.4f}")
        st.metric("RMSE", f"{tm.get('rmse', 0):.4f}")
        st.markdown("---")
with st.sidebar:
    st.markdown("### 📊 AQI Categories")
    st.markdown("""
    - **Good 🌿:** 0 - 50  
    - **Moderate 🌤️:** 51 - 100  
    - **Unhealthy (Sensitive) ⚠️:** 101 - 150  
    - **Unhealthy 🚨:** 151 - 200  
    - **Very Unhealthy 🛑:** 201 - 300  
    - **Hazardous ☠️:** 301+  
    """)

# ===========================
# TABS FOR INPUT
# ===========================
tab1, tab2 = st.tabs(["📤 CSV Upload", "⚡ Quick Demo"])

with tab1:
    st.subheader("📁 Upload CSV")
    uploaded = st.file_uploader(f"CSV with {TARGET} column", type=["csv"])
    
    if uploaded:
        df = pd.read_csv(uploaded)
        if TARGET in df.columns:
            vals = df[TARGET].dropna().values[-WINDOW_SIZE:]
            if len(vals) >= WINDOW_SIZE:
                csv_input = ", ".join([str(round(v, 2)) for v in vals])
                csv_input = st.text_area("Historical Values (Last 30 days)", value=csv_input, height=100, key="csv")
                csv_days = st.slider("Forecast Days", 1, 14, 7, key="csv_days")
            else:
                st.error(f"Need {WINDOW_SIZE} values")
                csv_input = None
        else:
            st.error(f"Column '{TARGET}' not found")
            csv_input = None
    else:
        st.info("Upload CSV file with pollution data")
        csv_input = None

with tab2:
    st.subheader("📝 Manual Input")
    demo_vals = ", ".join([str(round(40 + np.random.randn() * 5, 2)) for _ in range(WINDOW_SIZE)])
    demo_input = st.text_area("Historical Values (Last 30 days)", value=demo_vals, height=100, key="demo")
    demo_days = st.slider("Forecast Days", 1, 14, 7, key="demo_days")

# ===========================
# SELECT INPUT
# ===========================
if csv_input:
    input_vals = csv_input
    forecast_days = csv_days
else:
    input_vals = demo_input
    forecast_days = demo_days

# ===========================
# FORECAST
# ===========================
if st.button("🚀 Forecast", type="primary"):
    try:
        # Parse input values
        values = np.array([float(x.strip()) for x in input_vals.split(",")])
        if len(values) != WINDOW_SIZE:
            st.error(f"Need exactly {WINDOW_SIZE} values, got {len(values)}")
            st.stop()

        # Debug info
        st.info(f"📊 Input range: {values.min():.2f} - {values.max():.2f} ppm")
        st.info(f"🔧 Scaler range: {scaler.data_min_[0]:.2f} - {scaler.data_max_[0]:.2f}")
        
        # Scale input values
        scaled = scaler.transform(values.reshape(-1, 1)).flatten()
        sequence = scaled
        preds_scaled = []

        # Predict for each future day
        for _ in range(forecast_days):
            pred = model.predict(sequence.reshape(1, WINDOW_SIZE, 1), verbose=0)[0, 0]
            preds_scaled.append(pred)
            sequence = np.append(sequence[1:], pred)

        # ✅ CRITICAL: Inverse transform to get original scale predictions
        preds_scaled_array = np.array(preds_scaled).reshape(-1, 1)
        preds_original = scaler.inverse_transform(preds_scaled_array).flatten()
        
        # Debug: Print to verify transformation
        print(f"Scaled predictions: {preds_scaled[:3]}")
        print(f"Original predictions: {preds_original[:3]}")

        # Classification function for air quality status
        def classify(val):
            if val <= 50: return "Good 🌿"
            elif val <= 100: return "Moderate 🌤️"
            elif val <= 150: return "Unhealthy (Sensitive) ⚠️"
            elif val <= 200: return "Unhealthy 🚨"
            elif val <= 300: return "Very Unhealthy 🛑"
            else: return "Hazardous ☠️"

        # Results DataFrame - Displaying original scale values (not scaled)
        df_res = pd.DataFrame({
            "Day": [f"Day {i+1}" for i in range(forecast_days)],
            "O3 Mean (ppm)": [f"{p:.2f}" for p in preds_original],  # ✅ Original scale values
            "Air Quality Status": [classify(float(p)) for p in preds_original]
        })

        # Display results
        st.markdown("### 📋 Forecast Results")
        st.dataframe(df_res, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Prediction", f"{np.mean(preds_original):.2f} ppm")
        with col2:
            st.metric("Min Value", f"{np.min(preds_original):.2f} ppm")
        with col3:
            st.metric("Max Value", f"{np.max(preds_original):.2f} ppm")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ===========================
# FOOTER
# ===========================
st.markdown("---")
st.markdown("<p style='text-align:center;color:#555;'>Developed under the supervision of Engineer: <span style='color:#FF69B4;'>Habiba💖💖💖</span></p>", unsafe_allow_html=True)
