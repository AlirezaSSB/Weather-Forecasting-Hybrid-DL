# 🌤️ Deep Weather Analytics: Hybrid LSTM-MLP Forecasting Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An advanced, end-to-end meteorological time-series forecasting pipeline. Traditional models often struggle with long-term atmospheric dependencies due to information bottlenecks. This project overcomes that limitation by introducing a **Dual-Branch Hybrid Architecture**, combining the sequential pattern recognition of **LSTM** networks with the calendar-aware deterministic capabilities of **MLP** networks.

The final optimized model is exported to **ONNX** for high-speed, framework-agnostic CPU inference and is deployed via a highly interactive **Streamlit** dashboard.

---

## 🎥 Project Demos
Watch the project in action and see the dashboard capabilities:
* 📺 **[Dashboard Walkthrough & Deep Dive](https://youtu.be/8DA4i7RYp2c?si=6hffuHmAJvRj9RN9)**
* 📱 **[Quick Project Shorts](https://youtube.com/shorts/BdzSYBnM4DQ?si=h5RNs4z_yXi6g9EU)**

---

## 🧠 Architecture Overview
The core of this project is its custom neural network design, which separates the input data into two distinct branches before fusion:
1. **Temporal Branch (LSTM):** Processes a sliding window of historical multivariate data (`Temperature`, `Humidity`, `Wind Speed`, `Pressure`) to extract dynamic patterns.
2. **Static Branch (MLP):** Injects known future calendar covariates (`Month`, `Day of Year`) to maintain strict seasonality awareness.
3. **Fusion Layer:** Concatenates the extracted features from both branches to predict a continuous, multi-step 7-day temperature trajectory.

---

## ✨ Key Features
* **Multi-step Trajectory Forecasting:** Predicts a continuous 7-day weather trend rather than a simple single-point output.
* **Per-Day Confidence Intervals:** Dynamically calculates and visualizes the Root Mean Square Error (RMSE) bounds for each day ahead to accurately represent prediction uncertainty.
* **Blazing Fast Inference (ONNX):** The PyTorch model is traced and compiled into an `.onnx` graph with dynamic batch axes, ensuring rapid execution on standard CPUs without heavy backend dependencies.
* **Interactive UI:** A responsive, dark-themed dashboard built with Streamlit and Plotly for deep historical data exploration and real-time inference.

---

## 🗂️ Repository Structure

```text
.
├── Data/                   # Raw and processed datasets
├── Metrics/                # Evaluation metrics and daily RMSE bounds
├── Models/                 # Exported ONNX inference models
├── Notebooks/              # Jupyter notebooks (EDA, Training, Evaluation)
├── Scalers/                # Serialized MinMax scalers for data normalization
├── Dashboard.py            # Main Streamlit application script
├── LICENSE                 # Project license (MIT)
└── README.md               # Project documentation
```

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/AlirezaSSB/Weather-Forecasting-Hybrid-DL.git
cd Weather-Forecasting-Hybrid-DL
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required packages using pip:
```bash
pip install streamlit pandas numpy plotly onnxruntime scikit-learn torch seaborn matplotlib
```

### 3. Run the Dashboard
Execute the following command to spin up the local Streamlit server:
```bash
streamlit run Dashboard.py
```
*The dashboard will automatically open in your default web browser.*

---

## 📊 Model Performance
The Hybrid LSTM-MLP architecture was evaluated against a standard Baseline LSTM and a Naive Persistence model. By leveraging static calendar features, the hybrid approach achieved the highest accuracy across the 7-day forecasting horizon:

| Model | MAE (°C) | RMSE (°C) | R² Score |
| :--- | :---: | :---: | :---: |
| Naive Persistence | 2.712 | 3.886 | 0.816 |
| Baseline (LSTM only) | 2.102 | 2.948 | 0.894 |
| **Hybrid (LSTM + MLP)** | **1.812** | **2.583** | **0.919** |

*Note: The hybrid model successfully reduces the error accumulation typically seen in later days of the forecast horizon.*

---

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
