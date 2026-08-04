"""
Smart Grid Energy Monitoring Dashboard
--------------------------------------
Saeed Sarwar Anas | 20068400 | COM748 | Ulster University

v2 adds a DATASET SELECTOR so the same dashboard can display either the
UCI household benchmark or the Tetouan City cross-validation results.
It reads only pre-computed model outputs, so nothing trains at run time.

Run:  python -m streamlit run dashboard/app.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Smart Grid Energy Monitoring",
    page_icon="⚡",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Dataset registry - each entry points at the pre-computed outputs for one run
# ---------------------------------------------------------------------------
DATASETS = {
    "UCI Household (primary benchmark)": {
        "key": "uci",
        "unit": "kW",
        "power_file": ROOT / "data" / "clean_hourly_power.csv",
        "power_col": "Global_active_power",
        "time_col": "datetime",
        "pred_file": ROOT / "results" / "predictions.csv",
        "anom_file": ROOT / "results" / "detected_anomalies.csv",
        "blurb": "2,075,259 minute-level records (Dec 2006 - Nov 2010) from a single "
                 "French household, resampled to 34,589 hourly records.",
    },
    "Tetouan City (cross-dataset validation)": {
        "key": "tetouan",
        "unit": "kW",
        "power_file": None,                     # full hourly series not published
        "power_col": "Total_Consumption",
        "time_col": "DateTime",
        "pred_file": ROOT / "results" / "tetouan_predictions.csv",
        "anom_file": ROOT / "results" / "tetouan_detected_anomalies.csv",
        "blurb": "52,416 ten-minute records (2017) across three distribution zones of "
                 "Tetouan, Morocco, plus weather, resampled to 8,736 hourly records.",
    },
}

# ---------------------------------------------------------------------------
# Loaders (cached per file path)
# ---------------------------------------------------------------------------
@st.cache_data
def load_csv(path, time_col=None):
    if path is None or not Path(path).exists():
        return None
    if time_col:
        df = pd.read_csv(path, parse_dates=[time_col], index_col=time_col)
        # results files are written in detection order, not date order;
        # sorting is required before any date-range slicing
        return df.sort_index()
    df = pd.read_csv(path)
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]

def mae(a, p):
    return float(np.mean(np.abs(a - p)))

def rmse(a, p):
    return float(np.sqrt(np.mean((a - p) ** 2)))

# ---------------------------------------------------------------------------
# Sidebar - identity and the dataset selector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Smart Grid")
    st.caption("Industrial Energy Management using IoT and Machine Learning")
    st.markdown("---")

    choice = st.selectbox(
        "Dataset",
        list(DATASETS.keys()),
        index=0,
        help="Switch between the primary benchmark and the cross-dataset validation run.",
    )
    CFG = DATASETS[choice]
    st.caption(CFG["blurb"])

    st.markdown("---")
    st.markdown("**Student:** Saeed Sarwar Anas")
    st.markdown("**Student No:** 20068400")
    st.markdown("**Supervisor:** Dr Anwar Haq")
    st.markdown("**Module:** COM748 MSc Research Project")
    st.markdown("**Institution:** Ulster University")
    st.markdown("---")
    st.markdown("[GitHub Repository](https://github.com/S1122A/SmartGrid-Dissertation)")
    st.markdown("---")
    st.caption(
        "Displays pre-computed results from the LSTM forecasting model and the "
        "Isolation Forest anomaly detector. No model is trained at run time."
    )

# ---------------------------------------------------------------------------
# Load the selected dataset's outputs
# ---------------------------------------------------------------------------
preds = load_csv(CFG["pred_file"])
anoms = load_csv(CFG["anom_file"], CFG["time_col"])
power = load_csv(CFG["power_file"], CFG["time_col"]) if CFG["power_file"] else None

if preds is None:
    st.error(
        f"Results file not found: {CFG['pred_file']}\n\n"
        "Run the corresponding notebook first, then reload this page."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Header - always states which dataset is on screen
# ---------------------------------------------------------------------------
st.title("Smart Grid Energy Monitoring Dashboard")
st.markdown(f"**Currently displaying:** {choice}")
st.caption(CFG["blurb"])
st.markdown("---")

# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------
actual = preds["Actual_kW"].values
lstm_p = preds["LSTM_Predicted_kW"].values
lstm_mae, lstm_rmse = mae(actual, lstm_p), rmse(actual, lstm_p)

c1, c2, c3, c4 = st.columns(4)
c1.metric("LSTM Forecast MAE", f"{lstm_mae:,.4f} {CFG['unit']}"
          if CFG["key"] == "uci" else f"{lstm_mae:,.2f} {CFG['unit']}")
c2.metric("LSTM Forecast RMSE", f"{lstm_rmse:,.4f} {CFG['unit']}"
          if CFG["key"] == "uci" else f"{lstm_rmse:,.2f} {CFG['unit']}")
c3.metric("Anomalies Detected", f"{0 if anoms is None else len(anoms):,}")
c4.metric("Records Monitored",
          f"{len(power):,}" if power is not None else f"{len(preds):,} (test window)")
st.caption(
    f"Forecast metrics are computed over the saved {len(preds):,}-hour excerpt of the "
    "held-out test set, which is the window stored in the results file. Figures quoted "
    "over the full test partition may differ slightly."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# 1. Consumption and anomalies
# ---------------------------------------------------------------------------
st.subheader("1. Energy Consumption and Detected Anomalies")

if power is not None:
    col_a, col_b = st.columns(2)
    min_d, max_d = power.index.min().date(), power.index.max().date()
    default_end = min(max_d, (power.index.min() + pd.Timedelta(days=60)).date())
    start_date = col_a.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d)
    end_date = col_b.date_input("End date", value=default_end, min_value=min_d, max_value=max_d)

    s_ts = pd.Timestamp(start_date)
    e_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    if s_ts > e_ts:
        st.warning("Start date is after end date - showing no data.")
    win = power[(power.index >= s_ts) & (power.index < e_ts)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win.index, y=win[CFG["power_col"]],
                             mode="lines", name="Consumption",
                             line=dict(color="#1C7293", width=1)))
    if anoms is not None:
        a_win = anoms[(anoms.index >= s_ts) & (anoms.index < e_ts)]
        if len(a_win):
            fig.add_trace(go.Scatter(x=a_win.index, y=a_win[CFG["power_col"]],
                                     mode="markers", name="Anomaly",
                                     marker=dict(color="#B33A3A", size=7)))
    fig.update_layout(height=420, xaxis_title="Time",
                      yaxis_title=f"Power ({CFG['unit']})",
                      hovermode="x unified", margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
else:
    # Tetouan: the full hourly series is regenerated by notebook 04 rather than
    # stored, so the flagged records are shown on their own timeline.
    st.info(
        "For this dataset the dashboard plots the flagged records directly: the full "
        "hourly series is regenerated by notebook 04 rather than stored in the repository."
    )
    if anoms is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=anoms.index, y=anoms[CFG["power_col"]],
                                 mode="markers", name="Anomaly",
                                 marker=dict(color="#B33A3A", size=8)))
        fig.update_layout(height=420, xaxis_title="Time",
                          yaxis_title=f"Consumption ({CFG['unit']})",
                          margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"{len(anoms)} records flagged by the Isolation Forest. Note the "
            "concentration in the summer months - heat-driven demand peaks."
        )
        with st.expander("View flagged records"):
            st.dataframe(anoms, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# 2. Forecast vs actual
# ---------------------------------------------------------------------------
st.subheader("2. LSTM Demand Forecast vs Actual")
show_baselines = st.checkbox("Show baseline models", value=False)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=preds["Actual_kW"], mode="lines", name="Actual",
                          line=dict(color="#13293D", width=2)))
fig2.add_trace(go.Scatter(y=preds["LSTM_Predicted_kW"], mode="lines", name="LSTM forecast",
                          line=dict(color="#065A82", width=2, dash="dash")))
if show_baselines:
    if "Naive_Predicted_kW" in preds:
        fig2.add_trace(go.Scatter(y=preds["Naive_Predicted_kW"], mode="lines",
                                  name="Naive", line=dict(color="#9FB4C4", width=1)))
    if "MA_Predicted_kW" in preds:
        fig2.add_trace(go.Scatter(y=preds["MA_Predicted_kW"], mode="lines",
                                  name="Moving average", line=dict(color="#C77D0A", width=1)))
fig2.update_layout(height=430, xaxis_title="Hours into the held-out test window",
                   yaxis_title=f"Power ({CFG['unit']})", hovermode="x unified",
                   margin=dict(t=30))
st.plotly_chart(fig2, use_container_width=True)
st.markdown("---")

# ---------------------------------------------------------------------------
# 3. Model comparison
# ---------------------------------------------------------------------------
st.subheader("3. Forecasting Model Comparison")

rows = []
for label, col in [("Naive", "Naive_Predicted_kW"),
                   ("Moving Average", "MA_Predicted_kW"),
                   ("LSTM", "LSTM_Predicted_kW")]:
    if col in preds:
        rows.append({"Model": label,
                     f"MAE ({CFG['unit']})": mae(actual, preds[col].values),
                     f"RMSE ({CFG['unit']})": rmse(actual, preds[col].values)})
comp = pd.DataFrame(rows)

left, right = st.columns([2, 1])
with left:
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=comp["Model"], y=comp[f"MAE ({CFG['unit']})"],
                          name="MAE", marker_color="#065A82"))
    fig3.add_trace(go.Bar(x=comp["Model"], y=comp[f"RMSE ({CFG['unit']})"],
                          name="RMSE", marker_color="#1C7293"))
    fig3.update_layout(height=380, barmode="group",
                       yaxis_title=f"Error ({CFG['unit']})", margin=dict(t=30))
    st.plotly_chart(fig3, use_container_width=True)
with right:
    st.dataframe(comp.set_index("Model").round(4), use_container_width=True)
    best = comp.loc[comp[f"MAE ({CFG['unit']})"].idxmin(), "Model"]
    st.success(f"Lowest error: **{best}**")

st.markdown("---")
st.caption(
    "All figures are computed from stored model outputs. To display a different "
    "dataset, run its notebook to regenerate the results files, then select it "
    "from the sidebar."
)
