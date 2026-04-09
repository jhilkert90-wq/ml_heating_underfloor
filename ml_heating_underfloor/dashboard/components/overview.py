"""
ML Heating Dashboard - Overview Component
Real-time monitoring and system status display

All data is read from the unified thermal state JSON via
``dashboard.data_service``.  No simulated / demo data is generated.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add parent directory so ``dashboard`` package is importable in the addon
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_service import (
    get_system_metrics,
    get_prediction_history,
    get_state_file_info,
    load_thermal_state,
)


def get_recent_trend_data():
    """Build a trend DataFrame from real prediction history.

    Returns a :class:`~pandas.DataFrame` with columns
    ``timestamp``, ``confidence``, ``mae`` extracted from the persisted
    prediction history.  Returns an empty DataFrame when no history is
    available.
    """
    history = get_prediction_history()
    if not history:
        return pd.DataFrame(columns=["timestamp", "confidence", "mae"])

    rows = []
    for entry in history:
        ts = entry.get("timestamp")
        if ts is None:
            continue
        try:
            ts_dt = pd.to_datetime(ts)
        except Exception:
            continue
        error = abs(entry.get("error", 0.0))
        context = entry.get("context", {})
        rows.append({
            "timestamp": ts_dt,
            # Use absolute error as proxy for per-step MAE
            "mae": error,
            # Confidence is a system-wide scalar; replicate for charting
            "confidence": 0.0,
        })

    if not rows:
        return pd.DataFrame(columns=["timestamp", "confidence", "mae"])

    df = pd.DataFrame(rows).sort_values("timestamp")
    # Fill confidence from system metrics so the chart is meaningful
    metrics = get_system_metrics()
    df["confidence"] = metrics.get("confidence", 0.0)
    return df

def render_metric_cards():
    """Render system performance metric cards from real state data."""
    metrics = get_system_metrics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Confidence",
            value=f"{metrics['confidence']:.3f}",
        )

    with col2:
        st.metric(
            label="MAE (°C)",
            value=f"{metrics['mae']:.3f}",
        )

    with col3:
        st.metric(
            label="RMSE (°C)",
            value=f"{metrics['rmse']:.3f}",
        )

    with col4:
        st.metric(
            label="Learning Cycles",
            value=f"{metrics['cycle_count']:,}",
        )

def render_performance_trend():
    """Render performance trend chart from real prediction history."""
    st.subheader("Performance Trend")

    df = get_recent_trend_data()

    if df.empty:
        st.info(
            "No performance data available yet. "
            "Data will appear after the ML system starts learning."
        )
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["confidence"],
        mode="lines+markers",
        name="Confidence",
        line=dict(color="#1f77b4", width=2),
        yaxis="y",
    ))

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["mae"],
        mode="lines+markers",
        name="Abs Error (°C)",
        line=dict(color="#ff7f0e", width=2),
        yaxis="y2",
    ))

    fig.update_layout(
        xaxis_title="Time",
        yaxis=dict(
            title="Confidence",
            titlefont=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4"),
        ),
        yaxis2=dict(
            title="Abs Error (°C)",
            titlefont=dict(color="#ff7f0e"),
            tickfont=dict(color="#ff7f0e"),
            overlaying="y",
            side="right",
        ),
        hovermode="x unified",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

def render_system_status():
    """Render current system status from real state data."""
    st.subheader("System Status")

    metrics = get_system_metrics()

    col1, col2 = st.columns(2)

    with col1:
        status = metrics["status"]
        status_map = {
            "active": ("🟢 ML System: Active", st.success),
            "idle": ("🟡 ML System: Idle", st.info),
            "stale": ("🟠 ML System: Stale (no recent run)", st.warning),
        }
        label, writer = status_map.get(
            status, ("🔴 ML System: Unavailable", st.error)
        )
        writer(label)

        if metrics["last_prediction"] > 0:
            st.info(f"🌡️ Last Prediction: {metrics['last_prediction']:.1f}°C")

        # State file info
        file_info = get_state_file_info()
        if file_info:
            st.success(
                f"💾 State: {file_info['size_kb']:.1f} KB  "
                f"({file_info['path']})"
            )
            st.caption(
                f"Updated: {file_info['last_modified'].strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            st.warning("💾 State file not found")

    with col2:
        st.write("**Learning Progress**")
        cycle_count = metrics["cycle_count"]

        if cycle_count < 200:
            st.info("🌱 Initializing (0-200 cycles)")
            progress = cycle_count / 200
        elif cycle_count < 1000:
            st.info("⚙️ Learning (200-1000 cycles)")
            progress = (cycle_count - 200) / 800
        else:
            st.success("✅ Mature (1000+ cycles)")
            progress = 1.0

        st.progress(progress)
        st.write(f"Cycle {cycle_count:,}")

def render_configuration_summary():
    """Render current configuration summary from the real state file."""
    st.subheader("Configuration")

    state = load_thermal_state()
    if state is None:
        st.warning("Thermal state file not available – no configuration to show.")
        return

    baseline = state.get("baseline_parameters", {})

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Baseline Source**")
        source = baseline.get("source", "unknown")
        cal_date = baseline.get("calibration_date", "n/a")
        cal_cycles = baseline.get("calibration_cycles", 0)
        st.write(f"Source: `{source}`")
        st.write(f"Calibration Date: `{cal_date}`")
        st.write(f"Calibration Cycles: `{cal_cycles}`")

    with col2:
        st.write("**Key Thermal Parameters**")
        for key in [
            "thermal_time_constant",
            "heat_loss_coefficient",
            "outlet_effectiveness",
            "slab_time_constant_hours",
        ]:
            val = baseline.get(key)
            if val is not None:
                st.write(f"{key}: `{val}`")


def render_overview():
    """Main overview page."""
    st.header("📊 System Overview")

    if st.button("🔄 Refresh Data"):
        st.rerun()

    render_metric_cards()

    st.divider()

    render_performance_trend()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        render_system_status()

    with col2:
        render_configuration_summary()
