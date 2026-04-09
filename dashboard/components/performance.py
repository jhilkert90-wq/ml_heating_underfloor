"""
ML Heating Dashboard - Performance Analytics Component

All data is sourced from the unified thermal state JSON file via
``dashboard.data_service``.  No simulated / demo data is generated.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import sys

# Make data_service importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_service import (
    get_system_metrics,
    get_prediction_history,
    get_parameter_history,
    get_baseline_parameters,
    get_effective_parameters,
    get_channel_summary,
    get_heat_source_channels,
    load_thermal_state,
)

# ------------------------------------------------------------------
# Data helpers – read directly from the real state file
# ------------------------------------------------------------------

def _build_learning_df() -> pd.DataFrame:
    """Build a DataFrame from real prediction history."""
    history = get_prediction_history()
    if not history:
        return pd.DataFrame()

    rows = []
    for entry in history:
        ts = entry.get("timestamp")
        if ts is None:
            continue
        try:
            ts_dt = pd.to_datetime(ts)
        except Exception:
            continue
        rows.append({
            "timestamp": ts_dt,
            "abs_error": abs(entry.get("error", 0.0)),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    # Add a rolling MAE column
    df["rolling_mae"] = df["abs_error"].rolling(window=20, min_periods=1).mean()
    return df


def _build_parameter_df() -> pd.DataFrame:
    """Build a DataFrame from real parameter history."""
    history = get_parameter_history()
    if not history:
        return pd.DataFrame()

    rows = []
    for entry in history:
        ts = entry.get("timestamp")
        if ts is None:
            continue
        try:
            ts_dt = pd.to_datetime(ts)
        except Exception:
            continue
        row: dict = {"timestamp": ts_dt}
        for key in [
            "thermal_time_constant",
            "heat_loss_coefficient",
            "outlet_effectiveness",
            "slab_time_constant_hours",
            "delta_t_floor",
        ]:
            val = entry.get(key)
            if val is not None:
                row[key] = val
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

def render_learning_progress():
    """Render learning progress from real prediction history."""
    st.subheader("📈 Learning Progress Over Time")

    df = _build_learning_df()

    if df.empty:
        st.info("No prediction history available yet. Data appears after the ML system runs.")
        return

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Absolute Prediction Error", "Rolling MAE (20-step)"),
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["abs_error"],
            mode="lines", name="Abs Error",
            line=dict(color="red", width=1),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["rolling_mae"],
            mode="lines", name="Rolling MAE",
            line=dict(color="blue", width=2),
        ),
        row=1, col=2,
    )

    fig.update_layout(height=400, showlegend=True,
                      title_text="ML System Learning Analytics")
    st.plotly_chart(fig, use_container_width=True)

def render_feature_importance():
    """Render thermal parameter comparison: baseline vs effective."""
    st.subheader("🎯 Thermal Parameters: Baseline vs Effective")
    st.caption("Shows how online learning has adjusted the physics model parameters")

    baseline = get_baseline_parameters()
    effective = get_effective_parameters()

    if not baseline or not effective:
        st.info("Parameter data not available.")
        return

    # Filter to numeric parameters only
    param_keys = [k for k in effective if k in baseline and isinstance(baseline.get(k), (int, float))]

    if not param_keys:
        st.info("No numeric parameters found.")
        return

    fig = go.Figure()

    baseline_vals = [baseline[k] for k in param_keys]
    effective_vals = [effective[k] for k in param_keys]

    fig.add_trace(go.Bar(
        y=param_keys, x=baseline_vals,
        orientation="h", name="Baseline",
        marker_color="#1f77b4",
    ))
    fig.add_trace(go.Bar(
        y=param_keys, x=effective_vals,
        orientation="h", name="Effective (baseline + deltas)",
        marker_color="#ff7f0e",
    ))

    fig.update_layout(
        barmode="group",
        title="Thermal Parameter Comparison",
        xaxis_title="Value",
        height=max(300, 40 * len(param_keys)),
    )
    st.plotly_chart(fig, use_container_width=True)

def render_prediction_accuracy():
    """Render prediction error distribution from real data."""
    st.subheader("🎯 Prediction Error Analysis")

    df = _build_learning_df()

    if df.empty:
        st.info("No prediction data available yet.")
        return

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["abs_error"],
        nbinsx=30,
        name="Abs Error Distribution",
        marker_color="green",
    ))
    fig.update_layout(
        xaxis_title="Absolute Error (°C)",
        yaxis_title="Count",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    metrics = get_system_metrics()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE", f"{metrics['mae']:.3f} °C")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.3f} °C")
    with col3:
        st.metric("Predictions", f"{len(df)}")

def render_heat_source_channels():
    """Render heat-source channel performance from real data."""
    st.subheader("🔥 Heat Source Channel Performance")
    st.caption("Per-channel parameter snapshot and recent error metrics")

    channels = get_channel_summary()
    if not channels:
        st.info("No heat source channel data available.")
        return

    for ch in channels:
        with st.expander(f"Channel: **{ch['channel']}**  "
                         f"(history: {ch['history_count']}, "
                         f"recent avg |error|: {ch['recent_avg_abs_error']:.4f}°C)"):
            params = ch.get("parameters", {})
            if params:
                cols = st.columns(len(params))
                for col, (k, v) in zip(cols, params.items()):
                    col.metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))

    # Per-channel error chart
    raw_channels = get_heat_source_channels()
    if not raw_channels:
        return

    fig = go.Figure()
    for name, data in raw_channels.items():
        history = data.get("history", [])
        errors = [h.get("error", 0.0) for h in history]
        if errors:
            fig.add_trace(go.Scatter(
                y=errors, mode="lines", name=name,
            ))
    fig.update_layout(
        title="Prediction Error by Channel",
        xaxis_title="History Index",
        yaxis_title="Error (°C)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

def render_parameter_evolution():
    """Render parameter evolution over time from real parameter history."""
    st.subheader("📊 Parameter Evolution")
    st.caption("How thermal parameters changed through online learning")

    df = _build_parameter_df()

    if df.empty:
        st.info("No parameter history available yet.")
        return

    numeric_cols = [c for c in df.columns if c != "timestamp" and df[c].notna().any()]
    if not numeric_cols:
        st.info("No numeric parameter columns found in history.")
        return

    fig = make_subplots(
        rows=1, cols=1,
    )

    for col_name in numeric_cols:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[col_name],
            mode="lines", name=col_name,
        ))

    fig.update_layout(
        title="Parameter Changes Over Time",
        xaxis_title="Time",
        yaxis_title="Value",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_system_insights():
    """Render system insights from real metrics."""
    st.subheader("🤖 System Insights & Recommendations")

    metrics = get_system_metrics()
    baseline = get_baseline_parameters()

    col1, col2 = st.columns(2)

    with col1:
        st.info("**Performance Summary**")
        st.write(f"• **Learning Cycles**: {metrics['cycle_count']:,}")
        st.write(f"• **Confidence**: {metrics['confidence']:.3f}")
        st.write(f"• **MAE**: {metrics['mae']:.3f} °C")
        st.write(f"• **RMSE**: {metrics['rmse']:.3f} °C")

        source = baseline.get("source", "unknown")
        st.write(f"• **Baseline Source**: {source}")

    with col2:
        st.success("**Recommendations**")
        if metrics["cycle_count"] < 200:
            st.write("• System is still in early learning – allow more cycles")
        elif metrics["mae"] > 0.5:
            st.write("• MAE is high – check sensor calibration")
            st.write("• Consider re-calibrating the baseline")
        else:
            st.write("• System performing well – continue current config")
        if metrics["status"] in ("stale", "unavailable"):
            st.write("• ⚠️ ML system not recently active – check process")


def render_performance():
    """Main performance analytics page."""
    st.header("📈 Performance Analytics")
    st.caption("All data sourced from the unified thermal state file")

    if st.button("🔄 Refresh Analytics"):
        st.rerun()

    render_learning_progress()

    st.divider()

    render_feature_importance()

    st.divider()

    render_prediction_accuracy()

    st.divider()

    render_heat_source_channels()

    st.divider()

    render_parameter_evolution()

    st.divider()

    render_system_insights()
