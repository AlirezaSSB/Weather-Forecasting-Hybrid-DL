import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import onnxruntime as ort
import pickle
from pathlib import Path

# 1. Application Configuration
st.set_page_config(page_title="Deep Weather Analytics", layout="wide")

WINDOW_SIZE = 14
FORECAST_HORIZON = 7
TS_FEATURES = ["Temperature (C)", "Humidity", "Wind Speed (km/h)", "Pressure (millibars)"]

# `streamlit run dashboard.py` uses whatever folder you launched the command
# from as the working directory - NOT necessarily the folder this script
# lives in. Resolving paths relative to this file makes the app work no
# matter where you run the command from.
BASE_DIR = Path(__file__).resolve().parent

REQUIRED_FILES = {
    "weatherHistory.csv": BASE_DIR / "Data\weatherHistory.csv",
    "weather_hybrid_model_cpu.onnx": BASE_DIR / "Models\weather_hybrid_model_cpu.onnx",
    "scaler_ts.pkl": BASE_DIR / "Scalers\scaler_ts.pkl",
    "scaler_static.pkl": BASE_DIR / "Scalers\scaler_static.pkl",
    "scaler_target.pkl": BASE_DIR / "Scalers\scaler_target.pkl",
    "hybrid_rmse_per_day.pkl": BASE_DIR / "Metrics\hybrid_rmse_per_day.pkl",
}

# 2. Asset Initialization
@st.cache_resource
def load_model_and_scalers():
    session = ort.InferenceSession(str(REQUIRED_FILES["weather_hybrid_model_cpu.onnx"]))
    with open(REQUIRED_FILES["scaler_ts.pkl"], "rb") as f:
        scaler_ts = pickle.load(f)
    with open(REQUIRED_FILES["scaler_static.pkl"], "rb") as f:
        scaler_static = pickle.load(f)
    with open(REQUIRED_FILES["scaler_target.pkl"], "rb") as f:
        scaler_target = pickle.load(f)
    with open(REQUIRED_FILES["hybrid_rmse_per_day.pkl"], "rb") as f:
        rmse_per_day = pickle.load(f)
    return session, scaler_ts, scaler_static, scaler_target, rmse_per_day


@st.cache_data
def load_dataset():
    df_raw = pd.read_csv(REQUIRED_FILES["weatherHistory.csv"])
    df_raw["Formatted Date"] = pd.to_datetime(df_raw["Formatted Date"], utc=True)
    df_raw.sort_values("Formatted Date", inplace=True)
    if "Loud Cover" in df_raw.columns:
        df_raw.drop(columns=["Loud Cover"], inplace=True)
    df_raw.ffill(inplace=True)
    df_raw.bfill(inplace=True)

    # IMPORTANT: must match the training pipeline exactly. The raw file is
    # hourly; the model was trained on DAILY aggregates, so inference has to
    # use the same daily resolution or the 14-step window won't mean 14 days.
    is_sub_daily = df_raw["Formatted Date"].diff().dropna().median() < pd.Timedelta(days=1)
    if is_sub_daily:
        df = (
            df_raw.set_index("Formatted Date")
            .resample("D")
            .agg({c: "mean" for c in TS_FEATURES})
            .dropna()
            .reset_index()
        )
    else:
        df = df_raw[["Formatted Date"] + TS_FEATURES].copy()

    df["Month"] = df["Formatted Date"].dt.month
    df["DayOfYear"] = df["Formatted Date"].dt.dayofyear
    return df


try:
    session, scaler_ts, scaler_static, scaler_target, rmse_per_day = load_model_and_scalers()
    df = load_dataset()
except Exception as e:
    st.error(f"Initialization Error. Ensure all required files are present.\n\nDetails: {e}")
    st.stop()

# 2b. Read the model's ACTUAL expected shapes from the .onnx file itself.
def _static_dim(dim, fallback):
    return dim if isinstance(dim, int) else fallback

_ts_meta = session.get_inputs()[0]      # Time_Series_Input
_static_meta = session.get_inputs()[1]  # Static_Input
_out_meta = session.get_outputs()[0]    # Prediction_Output

MODEL_WINDOW_SIZE = _static_dim(_ts_meta.shape[1], WINDOW_SIZE)
MODEL_N_TS_FEATURES = _static_dim(_ts_meta.shape[2], len(TS_FEATURES))
MODEL_FORECAST_HORIZON = _static_dim(_out_meta.shape[1], FORECAST_HORIZON)

if MODEL_N_TS_FEATURES != len(TS_FEATURES):
    st.error(
        f"Mismatch: the ONNX model expects {MODEL_N_TS_FEATURES} time-series features per day, "
        f"but this dashboard is configured with {len(TS_FEATURES)} ({TS_FEATURES}). "
        "Re-export the model from the notebook, or update TS_FEATURES here to match."
    )
    st.stop()

WINDOW_SIZE = MODEL_WINDOW_SIZE
FORECAST_HORIZON = MODEL_FORECAST_HORIZON

# 3. User Interface (UI)
st.title("🌤️ Intelligent Weather Prediction Dashboard")
st.markdown("LSTM + MLP Hybrid Architecture — daily resolution")

st.sidebar.header("Data Filtering Configuration")
min_date = df["Formatted Date"].min().date()
max_date = df["Formatted Date"].max().date()
selected_dates = st.sidebar.date_input("Select Historical Date Range:", [min_date, max_date])

metrics_list = ["Temperature (C)", "Humidity", "Pressure (millibars)", "Wind Speed (km/h)"]
selected_metric = st.sidebar.selectbox("Select Meteorological Metric:", metrics_list)

with st.sidebar.expander("📈 Model Performance (Training-time Comparison)"):
    try:
        metrics_df = pd.read_csv(BASE_DIR / "Metrics\model_comparison_metrics.csv", index_col="Model")
        st.dataframe(metrics_df.round(3))
        st.caption(
            "Metrics computed on the held-out test set during Phase 6 of the notebook: "
            "Naive Persistence vs Baseline LSTM vs the deployed Hybrid model."
        )
    except Exception:
        st.caption("model_comparison_metrics.csv not found — run Phase 7 of the notebook to generate it.")

# 4. Analytics & Inference
if len(selected_dates) == 2:
    start_date, end_date = selected_dates
    mask = (df["Formatted Date"].dt.date >= start_date) & (df["Formatted Date"].dt.date <= end_date)
    filtered_df = df.loc[mask]

    st.markdown("### 📊 Descriptive Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Value", f"{filtered_df[selected_metric].mean():.2f}")
    col2.metric("Maximum Value", f"{filtered_df[selected_metric].max():.2f}")
    col3.metric("Minimum Value", f"{filtered_df[selected_metric].min():.2f}")

    st.markdown("---")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df["Formatted Date"], y=filtered_df[selected_metric], mode="lines", name=selected_metric))
    fig.update_layout(
        title=f"Historical Daily Trend: {selected_metric}",
        template="plotly_dark", xaxis_title="Date", yaxis_title=selected_metric,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Prediction Engine
    st.markdown("### 🤖 Predictive Analytics Engine")

    if st.button("Execute 7-Day Temperature Forecast"):
        if len(filtered_df) < WINDOW_SIZE:
            st.warning(f"Select a range with at least {WINDOW_SIZE} days of daily data.")
        else:
            # Prepare time-series input: last WINDOW_SIZE days -----------------
            last_window = filtered_df[TS_FEATURES].iloc[-WINDOW_SIZE:].values
            last_window_scaled = scaler_ts.transform(last_window)
            X_ts_input = np.expand_dims(last_window_scaled, axis=0).astype(np.float32)

            # Prepare known-future calendar covariates --------------------------
            last_date = filtered_df["Formatted Date"].iloc[-1]
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, FORECAST_HORIZON + 1)]
            static_array = np.array([[d.month, d.dayofyear] for d in future_dates])
            static_scaled = scaler_static.transform(static_array).flatten().astype(np.float32)
            X_static_input = np.expand_dims(static_scaled, axis=0)

            # Run ONNX inference --------------------------------------------------
            input_name_ts = session.get_inputs()[0].name
            input_name_static = session.get_inputs()[1].name
            ort_inputs = {input_name_ts: X_ts_input, input_name_static: X_static_input}
            pred_scaled = session.run(None, ort_inputs)[0]
            pred_real = scaler_target.inverse_transform(pred_scaled)[0]

            # Per-day confidence bounds (varies by how far ahead we're forecasting)
            upper_bound = pred_real + rmse_per_day
            lower_bound = pred_real - rmse_per_day

            # Results Table ---------------------------------------------------------
            forecast_df = pd.DataFrame({
                "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
                "Forecast (°C)": np.round(pred_real, 2),
                "Lower Bound": np.round(lower_bound, 2),
                "Upper Bound": np.round(upper_bound, 2),
                "±RMSE (°C)": np.round(rmse_per_day, 2),
            })
            st.dataframe(forecast_df.set_index("Date").T)

            # Visualization with Confidence Interval Shading -------------------------
            plot_df = filtered_df.tail(40)  # 40 days of context
            fig_pred = go.Figure()

            fig_pred.add_trace(go.Scatter(
                x=plot_df["Formatted Date"], y=plot_df["Temperature (C)"],
                mode="lines", name="Observed Historical Data", line=dict(color="cyan"),
            ))

            conn_dates = [last_date] + future_dates
            conn_temps = [plot_df["Temperature (C)"].iloc[-1]] + list(pred_real)
            conn_upper = [plot_df["Temperature (C)"].iloc[-1]] + list(upper_bound)
            conn_lower = [plot_df["Temperature (C)"].iloc[-1]] + list(lower_bound)

            fig_pred.add_trace(go.Scatter(
                x=conn_dates + conn_dates[::-1],
                y=conn_upper + conn_lower[::-1],
                fill="toself", fillcolor="rgba(255, 65, 54, 0.2)",
                line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
                name="Per-Day Confidence Interval (±RMSE)",
            ))

            fig_pred.add_trace(go.Scatter(
                x=conn_dates, y=conn_temps, mode="lines+markers",
                name="7-Day Forecast Trajectory",
                marker=dict(color="red", size=8), line=dict(color="red", width=3),
            ))

            fig_pred.update_layout(
                title="Predictive Trajectory with Per-Day Uncertainty Bounds",
                template="plotly_dark", xaxis_title="Date", yaxis_title="Temperature (C)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_pred, use_container_width=True)
else:
    st.warning("Please select a valid start and end date in the sidebar.")