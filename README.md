# 📈 Forecasting with ES-RNN: A Hybrid Deep Learning Model

This project implements and evaluates **ES-RNN (Exponential Smoothing Recurrent Neural Network)** — a hybrid architecture that blends traditional exponential smoothing with LSTM-based residual learning — for real-world time series forecasting.

We test its performance on:
- 🛒 **Retail Sales Data**
- 📊 **M3 Monthly Forecasting Benchmark**
- ⚖️ **Compared with NeuralProphet** (baseline)

---

## 📌 Why ES-RNN?

> ES-RNN was introduced as part of the winning solution in the M4 Forecasting Competition. It marries statistical structure with deep learning flexibility to deliver **robust, accurate, and interpretable forecasts**.

### Key Benefits:
- Captures level, trend, seasonality (via exponential smoothing)
- Learns residual patterns with LSTM
- Outperforms naive models and basic neural nets
- Ideal for **noisy, seasonal, or sparse** time series

---

## 🔧 Model Architecture
Original Time Series
↓
Exponential Smoothing → Trend + Seasonality + Residual
↓
Residual → LSTM → Forecast Residual
↓
Final Forecast = Forecast Residual + Trend + Seasonality


---

## 📁 Project Structure

ES-RNN-TimeSeries-Forecasting/
├── ES_RNN_rdebnath.ipynb # Retail sales demo
├── ES_RNN_M3_Benchmark_rdebnath.ipynb # M3 dataset evaluation
├── M3Monthly_Cleaned_Sample.csv # Benchmark dataset
├── esrnn_m3_results.csv # Evaluation metrics
├── ESRNN_M3_Evaluation.csv # R², RMSE, MAE results
├── ES-RNN.pdf # Project slides
└── README.md


---

## 🔬 Methodology

### 1. 📦 Data
- Retail Dataset: 5 years of daily sales for 10 stores × 50 items.
- M3 Benchmark: Monthly time series from various domains (e.g., economics, industry, etc.)

### 2. ⚙️ Preprocessing
- Rolling mean to extract trend
- Residual computation
- MinMax scaling
- Sequence modeling for supervised learning

### 3. 🧠 Model
```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(30, 1)),
    tf.keras.layers.Dense(1)
])
```
Loss: MSE

Optimizer: Adam
---

### 📊 Performance (Retail Data)
Tested on 3 sample monthly time series:
| Metric   | ES-RNN | NeuralProphet |
| -------- | ------ | ------------- |
| RMSE     | 5.70   | 7.75          |
| MAE      | 4.58   | 6.09          |
| MAPE (%) | 25.16  | 32.73         |
| R² Score | 0.31   | -0.25         |

✅ ES-RNN outperformed NeuralProphet on all metrics.
---

### 🧪 M3 Benchmark Results

Tested on 3 sample monthly time series:

| Series ID | RMSE  | MAE   |
| --------- | ----- | ----- |
| M3M\_0001 | 48.07 | 41.95 |
| M3M\_0002 | 44.08 | 36.87 |
| M3M\_0003 | 40.29 | 33.94 |

These results highlight strong generalization with room for tuning on large-scale benchmarks.

### 📈 Visualizations
<p align="center"> <img src="Screenshot 2025-06-24 154320.png" width="600"/> <br><em>ES-RNN Forecast vs Actual</em> </p> <p align="center"> <img src="Screenshot 2025-06-24 154407.png" width="600"/> <br><em>Metric Comparison – ES-RNN vs NeuralProphet</em> </p>

"Hybrid models like ES-RNN offer the best of both worlds — interpretable structure and adaptable learning."
