
<p align="center">
  <img src="https://static.vecteezy.com/system/resources/previews/036/669/620/large_2x/colorful-air-quality-index-aqi-gauge-illustration-modern-clean-design-depicting-air-pollution-measurement-and-environmental-health-indicator-vector.jpg" width="800" height="500"/>
</p>

# 🌫️ Air Quality (AQI) Forecasting System

A deep learning-based system for predicting air pollution levels (PM2.5) using LSTM neural networks. This project demonstrates time-series forecasting with real-world environmental data.

---

## 📋 Project Overview

This system analyzes historical air quality data and predicts future PM2.5 concentration levels using Long Short-Term Memory (LSTM) networks. The model considers multiple environmental factors including pollutants, weather conditions, and temporal patterns.

## 📊 Dataset

**US Pollution Data**
   - **Source:** https://www.kaggle.com/datasets/sogun3/uspollution
   - Coverage: Major US cities (2000-2016)

---
### ✨ Key Features

-  **LSTM-based time series forecasting** for air quality prediction
-  **30-day lookback window** for sequential pattern learning
-  **Single pollutant focus** (O3 concentration)
-  **Interactive Streamlit web interface** with 2 input modes
-  **Multi-step forecasting** (1-14 days ahead)
-  **Model performance tracking** (MAE, RMSE, R² metrics)
-  **CSV upload support** for custom data forecasting
-  **Demo mode** with sample data generation

---

## 🏗️ Project Structure

```
aqi_forecasting/
│                 
├── notebook.ipynb    
│
├── models/
│   ├── best_lstm_model.h5       
│   ├── scaler.pkl
│   ├── state_encoder.pkl                  
│   └── metrics.json                  
│
├── app/               
│   └── app.py         
│              
├── requirements.txt                  
└── README.md                         
```

---

## 🌐 Live Demo

You can try the web application here:

Air Quality (AQI) Forecasting App 👉 [Live Demo](https://aqiforecasting-68zsai9abvpjnbbgafzho4.streamlit.app/)

---

**Happy Forecasting! 🌍**
