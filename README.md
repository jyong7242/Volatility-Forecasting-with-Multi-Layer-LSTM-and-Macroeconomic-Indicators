# Volatility-Forecasting-with-Multi-Layer-LSTM-and-Macroeconomic-Indicators

This project focuses on forecasting the next-day realized volatility (`rv5`) of the S&P 500 index using deep learning techniques, specifically multi-layer LSTM networks. The model leverages both historical realized volatility and macroeconomic indicators, including the VIX index and U.S. Treasury rates. This implementation is entirely in Python and utilizes TensorFlow and Keras for deep learning.

## 📌 Objective

Realized volatility is a key metric in risk management, trading, and portfolio allocation. While traditional models such as the HAR model can capture long-memory effects in volatility, they often struggle with complex non-linear dependencies. This project aims to enhance predictive performance using LSTM models and assess the impact of incorporating external financial variables.

## 🧠 Model Summary

- **Baseline Model:** HAR (Heterogeneous Autoregressive) model using lagged volatility values.
- **Deep Learning Model:** Multi-layer LSTM with hyperparameter tuning via Keras Tuner.
- **Input Features:**
  - `rv5` (realized volatility)
  - VIX index (`vixcls`)
  - Interest rates from FRED (DGS1, DGS2, DGS10, DGS30, DTB3)
- **Targets:** One-step-ahead forecast of `rv5` (`rv5_f1`)

## 🧰 Features

- Data collection and preprocessing (Oxford-Man Realized Volatility, FRED)
- Exploratory Data Analysis (EDA)
- Feature engineering: lagged values, moving averages
- StandardScaler normalization
- HAR model (Linear Regression baseline)
- LSTM implementation with:
  - Hyperparameter tuning (sequence length, number of layers, nodes, activation, optimizer)
  - Early stopping and validation
  - Visualization of prediction results and loss curves

## 🔧 Technologies

- Python 3.x
- TensorFlow / Keras
- scikit-learn
- Keras Tuner
- Pandas, NumPy, Matplotlib, Seaborn

## 📊 Results

LSTM models demonstrate improved validation and test MSE when macroeconomic indicators are included, particularly the VIX index. Hyperparameter tuning significantly enhances performance and generalization on out-of-sample data.

| Model              | Validation MSE | Test MSE |
| ------------------ | -------------- | -------- |
| HAR                | 2.05           | 0.23     |
| LSTM (rv5 only)    | ~1.14          | ~0.50    |
| LSTM + VIX + Rates | ~0.76          | ~0.63    |
