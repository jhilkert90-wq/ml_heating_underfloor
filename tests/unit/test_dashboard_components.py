"""
Tests for dashboard component helper functions.

These tests validate the pure-data helper functions used by the
overview and performance Streamlit components. Streamlit rendering
functions are not tested directly (they require a running Streamlit
server), but the data-building helpers that feed them are tested
thoroughly.
"""

import json
import os
import sys
import pytest
import pandas as pd
from datetime import datetime, timedelta

# Insert dashboard directory so component modules can be imported
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "dashboard")
)
sys.path.insert(
    0, os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, "dashboard", "components"
    )
)


def _write_state(tmp_path, monkeypatch, state_dict):
    """Helper: write a state file and point data_service at it."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state_dict))
    import data_service as ds
    monkeypatch.setattr(ds, "_STATE_FILE_CANDIDATES", [str(path)])
    return path


def _base_state(**overrides):
    """Build a minimal unified thermal state for testing."""
    now = datetime.now()
    state = {
        "metadata": {
            "version": "1.0",
            "format": "unified_thermal_state",
            "created": (now - timedelta(days=7)).isoformat(),
            "last_updated": now.isoformat(),
        },
        "baseline_parameters": {
            "thermal_time_constant": 4.39,
            "heat_loss_coefficient": 0.125,
            "outlet_effectiveness": 0.953,
            "slab_time_constant_hours": 3.19,
            "delta_t_floor": 2.3,
            "equilibrium_ratio": 0.17,
            "total_conductance": 0.8,
            "pv_heat_weight": 0.002,
            "fireplace_heat_weight": 0.387,
            "tv_heat_weight": 0.35,
            "solar_lag_minutes": 45.0,
            "fp_decay_time_constant": 3.91,
            "room_spread_delay_minutes": 18.0,
            "source": "calibrated",
            "calibration_date": (now - timedelta(days=3)).isoformat(),
            "calibration_cycles": 5000,
        },
        "learning_state": {
            "cycle_count": 541,
            "learning_confidence": 3.0,
            "learning_enabled": True,
            "parameter_adjustments": {
                "thermal_time_constant_delta": 0.01,
                "heat_loss_coefficient_delta": -0.002,
                "outlet_effectiveness_delta": 0.003,
                "slab_time_constant_delta": 0.0,
                "delta_t_floor_delta": 0.0,
                "equilibrium_ratio_delta": 0.0,
                "total_conductance_delta": 0.0,
                "pv_heat_weight_delta": 0.0,
                "tv_heat_weight_delta": 0.0,
                "solar_lag_minutes_delta": 0.0,
                "fp_decay_time_constant_delta": 0.0,
                "room_spread_delay_minutes_delta": 0.0,
                "fireplace_heat_weight_delta": 0.0,
            },
            "parameter_bounds": {},
            "heat_source_channels": {
                "heat_pump": {
                    "parameters": {
                        "thermal_time_constant": 4.39,
                        "heat_loss_coefficient": 0.124,
                    },
                    "history_count": 100,
                    "history": [
                        {"error": 0.1, "context": {}},
                        {"error": -0.15, "context": {}},
                        {"error": 0.05, "context": {}},
                    ],
                },
            },
            "prediction_history": [
                {
                    "timestamp": (now - timedelta(hours=2)).isoformat(),
                    "error": 0.12,
                    "context": {"outlet_temp": 38.0, "outdoor_temp": 5.0},
                },
                {
                    "timestamp": (now - timedelta(hours=1)).isoformat(),
                    "error": -0.08,
                    "context": {"outlet_temp": 37.5, "outdoor_temp": 5.5},
                },
                {
                    "timestamp": now.isoformat(),
                    "error": 0.05,
                    "context": {"outlet_temp": 37.0, "outdoor_temp": 6.0},
                },
            ],
            "parameter_history": [
                {
                    "timestamp": (now - timedelta(hours=1)).isoformat(),
                    "thermal_time_constant": 4.39,
                    "heat_loss_coefficient": 0.125,
                    "outlet_effectiveness": 0.953,
                },
                {
                    "timestamp": now.isoformat(),
                    "thermal_time_constant": 4.40,
                    "heat_loss_coefficient": 0.123,
                    "outlet_effectiveness": 0.956,
                },
            ],
        },
        "prediction_metrics": {
            "total_predictions": 1000,
            "accuracy_stats": {
                "mae_all_time": 0.15,
                "rmse_all_time": 0.2,
            },
            "recent_performance": {
                "last_10_mae": 0.11,
            },
        },
        "operational_state": {
            "last_indoor_temp": 22.5,
            "last_prediction": 38.5,
            "last_run_time": now.isoformat(),
            "last_final_temp": 38.5,
        },
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Overview helpers
# ---------------------------------------------------------------------------

class TestOverviewHelpers:
    """Test get_recent_trend_data from overview component."""

    def test_trend_data_returns_dataframe(self, tmp_path, monkeypatch):
        state = _base_state()
        _write_state(tmp_path, monkeypatch, state)

        from overview import get_recent_trend_data
        df = get_recent_trend_data()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "timestamp" in df.columns
        assert "mae" in df.columns
        assert "confidence" in df.columns

    def test_trend_data_empty_when_no_history(self, tmp_path, monkeypatch):
        state = _base_state()
        state["learning_state"]["prediction_history"] = []
        _write_state(tmp_path, monkeypatch, state)

        from overview import get_recent_trend_data
        df = get_recent_trend_data()
        assert df.empty

    def test_trend_data_skips_entries_without_timestamp(self, tmp_path, monkeypatch):
        state = _base_state()
        state["learning_state"]["prediction_history"] = [
            {"error": 0.1},  # no timestamp
            {"timestamp": "2026-04-01T12:00:00", "error": 0.2},
        ]
        _write_state(tmp_path, monkeypatch, state)

        from overview import get_recent_trend_data
        df = get_recent_trend_data()
        assert len(df) == 1

    def test_trend_confidence_from_system_metrics(self, tmp_path, monkeypatch):
        state = _base_state()
        _write_state(tmp_path, monkeypatch, state)

        from overview import get_recent_trend_data
        df = get_recent_trend_data()
        # Confidence should be filled from get_system_metrics
        assert (df["confidence"] == 3.0).all()


# ---------------------------------------------------------------------------
# Performance helpers
# ---------------------------------------------------------------------------

class TestPerformanceHelpers:
    """Test _build_learning_df and _build_parameter_df from performance."""

    def test_learning_df_has_rolling_mae(self, tmp_path, monkeypatch):
        state = _base_state()
        _write_state(tmp_path, monkeypatch, state)

        from performance import _build_learning_df
        df = _build_learning_df()
        assert not df.empty
        assert "rolling_mae" in df.columns
        assert "abs_error" in df.columns

    def test_learning_df_empty_when_no_data(self, tmp_path, monkeypatch):
        state = _base_state()
        state["learning_state"]["prediction_history"] = []
        _write_state(tmp_path, monkeypatch, state)

        from performance import _build_learning_df
        df = _build_learning_df()
        assert df.empty

    def test_parameter_df_has_expected_columns(self, tmp_path, monkeypatch):
        state = _base_state()
        _write_state(tmp_path, monkeypatch, state)

        from performance import _build_parameter_df
        df = _build_parameter_df()
        assert not df.empty
        assert "timestamp" in df.columns
        assert "thermal_time_constant" in df.columns

    def test_parameter_df_tracks_changes(self, tmp_path, monkeypatch):
        state = _base_state()
        _write_state(tmp_path, monkeypatch, state)

        from performance import _build_parameter_df
        df = _build_parameter_df()
        # Should have 2 rows from the parameter_history fixture
        assert len(df) == 2
        # thermal_time_constant went from 4.39 to 4.40
        vals = df["thermal_time_constant"].tolist()
        assert vals[0] == pytest.approx(4.39)
        assert vals[1] == pytest.approx(4.40)

    def test_parameter_df_empty_when_no_data(self, tmp_path, monkeypatch):
        state = _base_state()
        state["learning_state"]["parameter_history"] = []
        _write_state(tmp_path, monkeypatch, state)

        from performance import _build_parameter_df
        df = _build_parameter_df()
        assert df.empty


# ---------------------------------------------------------------------------
# Integration: data service + component agree
# ---------------------------------------------------------------------------

class TestDataIntegration:
    """Verify that data_service and component helpers read the same data."""

    def test_metrics_consistency(self, tmp_path, monkeypatch):
        state = _base_state()
        _write_state(tmp_path, monkeypatch, state)

        from data_service import get_system_metrics
        metrics = get_system_metrics()
        assert metrics["cycle_count"] == 541
        assert metrics["confidence"] == 3.0
        assert metrics["mae"] == 0.15

    def test_effective_parameters_apply_deltas(self, tmp_path, monkeypatch):
        state = _base_state()
        _write_state(tmp_path, monkeypatch, state)

        from data_service import get_effective_parameters
        eff = get_effective_parameters()
        # thermal_time_constant = 4.39 + 0.01 = 4.40
        assert eff["thermal_time_constant"] == pytest.approx(4.40, abs=1e-6)
        # heat_loss_coefficient = 0.125 + (-0.002) = 0.123
        assert eff["heat_loss_coefficient"] == pytest.approx(0.123, abs=1e-6)

    def test_channel_summary_available(self, tmp_path, monkeypatch):
        state = _base_state()
        _write_state(tmp_path, monkeypatch, state)

        from data_service import get_channel_summary
        summary = get_channel_summary()
        assert len(summary) == 1
        assert summary[0]["channel"] == "heat_pump"
        assert summary[0]["history_count"] == 100
