"""
Smart Grid Sensor Network - Real-Time Energy Monitoring Dashboard
COM748 MSc Research Project | Saeed Sarwar Anas (20068400) | Ulster University

Loads the pre-computed results from the LSTM forecasting model (notebook 02)
and the Isolation Forest anomaly detector (notebook 03) and presents them in
one interface.

Run from the PROJECT ROOT (not from inside the dashboard folder):
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ----------------------------------------------------------------------------
# Page configuration  (must be the first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Grid Energy Dashboard",
    page_icon="\u26a1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Paths - resolved relative to the project root so it works from anywhere
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "clean_hourly_power.csv"
PRED_FILE = ROOT / "results" / "predictions.csv"
ANOM_FILE = ROOT / "results" / "detected_anomalies.csv"

# ----------------------------------------------------------------------------
# Cached data loaders
# ----------------------------------------------------------------------------
@st.cache_data
def load_power():
    return pd.read_csv(DATA_FILE, parse_dates=["datetime"], index_col="datetime")

@st.cache_data
def load_predictions():
    df = pd.read_csv(PRED_FILE)
    # drop an unnamed index column if pandas wrote one
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]

@st.cache_data
def load_anomalies():
    return pd.read_csv(ANOM_FILE, parse_dates=["datetime"], index_col="datetime")

def mae(a, p):
    return float(np.mean(np.abs(a - p)))

def rmse(a, p):
    return float(np.sqrt(np.mean((a - p) ** 2)))

# ----------------------------------------------------------------------------
# Load everything, with friendly errors if a file is missing
# ----------------------------------------------------------------------------
missing = [str(f) for f in (DATA_FILE, PRED_FILE, ANOM_FILE) if not f.exists()]
if missing:
    st.error("Could not find these result files:\n\n" + "\n".join(missing))
    st.info(
        "Make sure you run this from the project root and that notebooks "
        "01-03 have been run to generate the data and results:\n\n"
        "`streamlit run dashboard/app.py`"
    )
    st.stop()

power = load_power()
preds = load_predictions()
anoms = load_anomalies()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("Smart Grid")
    st.caption("Industrial Energy Management using IoT and Machine Learning")
    st.markdown("---")
    st.markdown("**Student:** Saeed Sarwar Anas")
    st.markdown("**Student No:** 20068400")
    st.markdown("**Supervisor:** Dr Anwar Haq")
    st.markdown("**Module:** COM748 MSc Research Project")
    st.markdown("**Institution:** Ulster University")
    st.markdown("---")
    st.markdown(
        "[GitHub Repository]"
        "(https://github.com/S1122A/SmartGrid-Dissertation)"
    )
    st.markdown("---")
    st.caption(
        "Displays pre-computed results from the LSTM forecasting model and "
        "the Isolation Forest anomaly detector."
    )

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Smart Grid Energy Monitoring Dashboard")
st.markdown(
    "A unified view of energy demand forecasting and anomaly detection for "
    "low-cost industrial energy management."
)

# ----------------------------------------------------------------------------
# Top metric cards
# ----------------------------------------------------------------------------
lstm_mae = mae(preds["Actual_kW"], preds["LSTM_Predicted_kW"])
lstm_rmse = rmse(preds["Actual_kW"], preds["LSTM_Predicted_kW"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("LSTM Forecast MAE", f"{lstm_mae:.4f} kW")
c2.metric("LSTM Forecast RMSE", f"{lstm_rmse:.4f} kW")
c3.metric("Anomalies Detected", f"{len(anoms):,}")
c4.metric("Records Monitored", f"{len(power):,}")

st.markdown("---")

# ----------------------------------------------------------------------------
# Section 1: Energy consumption with anomalies
# ----------------------------------------------------------------------------
st.subheader("1. Energy Consumption and Detected Anomalies")

min_date = power.index.min().date()
max_date = power.index.max().date()
default_end = min(power.index.min() + pd.Timedelta(days=60),
                  power.index.max()).date()

col_a, col_b = st.columns(2)
start_date = col_a.date_input("Start date", value=min_date,
                              min_value=min_date, max_value=max_date)
end_date = col_b.date_input("End date", value=default_end,
                            min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.warning("Start date must be on or before the end date.")
else:
    mask = (power.index.date >= start_date) & (power.index.date <= end_date)
    window = power.loc[mask]
    amask = (anoms.index.date >= start_date) & (anoms.index.date <= end_date)
    awin = anoms.loc[amask]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=window.index, y=window["Global_active_power"],
        mode="lines", name="Consumption",
        line=dict(color="#4C78A8", width=1)))
    if len(awin) > 0:
        fig1.add_trace(go.Scatter(
            x=awin.index, y=awin["Global_active_power"],
            mode="markers", name=f"Anomaly ({len(awin)})",
            marker=dict(color="#E45756", size=6)))
    fig1.update_layout(
        height=420, margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Date", yaxis_title="Global Active Power (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1))
    st.plotly_chart(fig1, use_container_width=True)
    st.caption(
        f"Showing {len(window):,} hourly records with {len(awin)} anomalies "
        "flagged in the selected window."
    )

st.markdown("---")

# ----------------------------------------------------------------------------
# Section 2: LSTM forecast vs actual
# ----------------------------------------------------------------------------
st.subheader("2. LSTM Demand Forecast vs Actual")

show_baselines = st.checkbox(
    "Show baseline models (Naive, Moving Average)", value=False)

x = list(range(len(preds)))
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x, y=preds["Actual_kW"], mode="lines",
                          name="Actual", line=dict(color="#222222", width=2)))
fig2.add_trace(go.Scatter(x=x, y=preds["LSTM_Predicted_kW"], mode="lines",
                          name="LSTM Forecast",
                          line=dict(color="#4C78A8", width=2)))
if show_baselines:
    fig2.add_trace(go.Scatter(x=x, y=preds["Naive_Predicted_kW"], mode="lines",
                              name="Naive",
                              line=dict(color="#F58518", width=1, dash="dot")))
    fig2.add_trace(go.Scatter(x=x, y=preds["MA_Predicted_kW"], mode="lines",
                              name="Moving Avg",
                              line=dict(color="#54A24B", width=1, dash="dot")))
fig2.update_layout(
    height=420, margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Time step (test-set hours)",
    yaxis_title="Global Active Power (kW)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1))
st.plotly_chart(fig2, use_container_width=True)
st.caption(
    f"First {len(preds)} hours of the held-out test set. The LSTM tracks "
    "actual demand more closely than the baselines."
)

st.markdown("---")

# ----------------------------------------------------------------------------
# Section 3: Forecasting model comparison
# ----------------------------------------------------------------------------
st.subheader("3. Forecasting Model Comparison")

comp = pd.DataFrame({
    "Model": ["Naive", "Moving Average", "LSTM"],
    "MAE": [mae(preds["Actual_kW"], preds["Naive_Predicted_kW"]),
            mae(preds["Actual_kW"], preds["MA_Predicted_kW"]),
            mae(preds["Actual_kW"], preds["LSTM_Predicted_kW"])],
    "RMSE": [rmse(preds["Actual_kW"], preds["Naive_Predicted_kW"]),
             rmse(preds["Actual_kW"], preds["MA_Predicted_kW"]),
             rmse(preds["Actual_kW"], preds["LSTM_Predicted_kW"])],
})

col_c, col_d = st.columns([3, 2])
with col_c:
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=comp["Model"], y=comp["MAE"], name="MAE",
                          marker_color="#4C78A8"))
    fig3.add_trace(go.Bar(x=comp["Model"], y=comp["RMSE"], name="RMSE",
                          marker_color="#F58518"))
    fig3.update_layout(
        height=380, margin=dict(l=20, r=20, t=30, b=20), barmode="group",
        yaxis_title="Error (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1))
    st.plotly_chart(fig3, use_container_width=True)
with col_d:
    st.markdown("**Error metrics (lower is better)**")
    st.dataframe(
        comp.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}"}),
        hide_index=True, use_container_width=True)
    best = comp.loc[comp["MAE"].idxmin(), "Model"]
    st.success(f"Best model: {best} (lowest MAE)")

st.markdown("---")
st.caption(
    "Smart Grid Sensor Network for Industrial Energy Management | "
    "COM748 MSc Research Project | Ulster University"
)
