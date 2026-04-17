"""
Tests for the dashboard data_service module.

Validates that the data service correctly reads from a real unified
thermal state JSON structure and returns the expected metric dictionaries,
history lists, and parameter comparisons.
"""

import json
import os
import pytest
from datetime import datetime, timedelta

# The data_service module lives in ``dashboard/`` which is outside
# the ``src`` package.  We add it to sys.path so pytest can import it.
import sys
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "dashboard")
)

from data_service import (
    _find_state_file,
    _shadow_variant,
    load_thermal_state,
    get_system_metrics,
    _empty_metrics,
    get_prediction_history,
    get_parameter_history,
    get_baseline_parameters,
    get_parameter_adjustments,
    get_effective_parameters,
    get_heat_source_channels,
    get_channel_summary,
    get_metadata,
    get_operational_state,
    get_state_file_info,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_state(
    cycle_count=100,
    confidence=2.5,
    mae_all_time=0.15,
    rmse_all_time=0.2,
    last_prediction=38.5,
    last_run_time=None,
    prediction_history=None,
    parameter_history=None,
    heat_source_channels=None,
    baseline_overrides=None,
    delta_overrides=None,
):
    """Build a minimal but realistic unified thermal state dict."""
    if last_run_time is None:
        last_run_time = datetime.now().isoformat()

    state = {
        "metadata": {
            "version": "1.0",
            "format": "unified_thermal_state",
            "created": "2026-01-01T00:00:00",
            "last_updated": datetime.now().isoformat(),
        },
        "baseline_parameters": {
            "thermal_time_constant": 4.39,
            "equilibrium_ratio": 0.17,
            "total_conductance": 0.8,
            "heat_loss_coefficient": 0.125,
            "outlet_effectiveness": 0.953,
            "solar_lag_minutes": 45.0,
            "slab_time_constant_hours": 3.19,
            "pv_heat_weight": 0.002,
            "fireplace_heat_weight": 0.387,
            "tv_heat_weight": 0.35,
            "delta_t_floor": 2.3,
            "fp_decay_time_constant": 3.91,
            "room_spread_delay_minutes": 18.0,
            "source": "calibrated",
            "calibration_date": "2026-01-01T00:00:00",
            "calibration_cycles": 5000,
        },
        "learning_state": {
            "cycle_count": cycle_count,
            "learning_confidence": confidence,
            "learning_enabled": True,
            "parameter_adjustments": {
                "thermal_time_constant_delta": 0.0,
                "heat_loss_coefficient_delta": 0.0,
                "outlet_effectiveness_delta": 0.0,
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
            "heat_source_channels": heat_source_channels or {},
            "prediction_history": prediction_history or [],
            "parameter_history": parameter_history or [],
        },
        "prediction_metrics": {
            "total_predictions": 1000,
            "accuracy_stats": {
                "mae_1h": 0.1,
                "mae_6h": 0.12,
                "mae_24h": 0.14,
                "mae_all_time": mae_all_time,
                "rmse_all_time": rmse_all_time,
            },
            "recent_performance": {
                "last_10_mae": 0.11,
                "last_10_max_error": 0.3,
            },
        },
        "operational_state": {
            "last_indoor_temp": 22.5,
            "last_outdoor_temp": 5.0,
            "last_outlet_temp": 38.0,
            "last_prediction": last_prediction,
            "last_run_time": last_run_time,
            "is_calibrating": False,
            "last_final_temp": 38.5,
        },
    }

    if baseline_overrides:
        state["baseline_parameters"].update(baseline_overrides)
    if delta_overrides:
        state["learning_state"]["parameter_adjustments"].update(delta_overrides)

    return state


@pytest.fixture()
def state_file(tmp_path, monkeypatch):
    """Write a state file and point data_service at it."""
    path = tmp_path / "unified_thermal_state.json"
    state = _make_state()
    path.write_text(json.dumps(state))

    # Patch candidate list to only look at this tmp file
    import data_service as ds
    monkeypatch.setattr(
        ds, "_STATE_FILE_CANDIDATES", [str(path)]
    )
    return path, state


@pytest.fixture()
def missing_state(monkeypatch):
    """Ensure no state file is found."""
    import data_service as ds
    monkeypatch.setattr(ds, "_STATE_FILE_CANDIDATES", ["/nonexistent/x.json"])


# ---------------------------------------------------------------------------
# Tests: load_thermal_state / _find_state_file
# ---------------------------------------------------------------------------

class TestLoadThermalState:
    def test_load_returns_dict(self, state_file):
        path, _ = state_file
        state = load_thermal_state()
        assert isinstance(state, dict)
        assert "metadata" in state

    def test_returns_none_when_missing(self, missing_state):
        assert load_thermal_state() is None

    def test_returns_none_on_corrupt_json(self, tmp_path, monkeypatch):
        path = tmp_path / "bad.json"
        path.write_text("{corrupt")
        import data_service as ds
        monkeypatch.setattr(ds, "_STATE_FILE_CANDIDATES", [str(path)])
        assert load_thermal_state() is None

    def test_prefers_shadow_variant(self, tmp_path, monkeypatch):
        """When a shadow file exists alongside the base file, the shadow
        variant should be returned by ``_find_state_file``."""
        base = tmp_path / "unified_thermal_state.json"
        shadow = tmp_path / "unified_thermal_state_shadow.json"
        base.write_text(json.dumps(_make_state(cycle_count=1)))
        shadow.write_text(json.dumps(_make_state(cycle_count=99)))

        import data_service as ds
        monkeypatch.setattr(
            ds, "_STATE_FILE_CANDIDATES", [str(base)]
        )
        found = _find_state_file()
        assert found == str(shadow)

        state = load_thermal_state()
        assert state["learning_state"]["cycle_count"] == 99

    def test_falls_back_to_base_when_no_shadow(self, tmp_path, monkeypatch):
        """When only the base file exists, it should be returned."""
        base = tmp_path / "unified_thermal_state.json"
        base.write_text(json.dumps(_make_state(cycle_count=7)))

        import data_service as ds
        monkeypatch.setattr(
            ds, "_STATE_FILE_CANDIDATES", [str(base)]
        )
        found = _find_state_file()
        assert found == str(base)


class TestShadowVariant:
    def test_json_suffix(self):
        assert _shadow_variant("/a/b/state.json") == "/a/b/state_shadow.json"

    def test_no_extension(self):
        assert _shadow_variant("/a/b/state") == "/a/b/state_shadow"

    def test_empty_string(self):
        assert _shadow_variant("") == ""

    def test_dotted_directory(self):
        assert (
            _shadow_variant("/path/v1.0/state.json")
            == "/path/v1.0/state_shadow.json"
        )


# ---------------------------------------------------------------------------
# Tests: get_system_metrics
# ---------------------------------------------------------------------------

class TestGetSystemMetrics:
    def test_returns_real_metrics(self, state_file):
        metrics = get_system_metrics()
        assert metrics["cycle_count"] == 100
        assert metrics["confidence"] == 2.5
        assert metrics["mae"] == 0.15
        assert metrics["rmse"] == 0.2
        assert metrics["last_prediction"] == 38.5
        assert metrics["status"] == "active"

    def test_stale_status(self, tmp_path, monkeypatch):
        """A run_time >1 hour ago should be 'stale'."""
        path = tmp_path / "s.json"
        old_time = (datetime.now() - timedelta(hours=2)).isoformat()
        state = _make_state(last_run_time=old_time)
        path.write_text(json.dumps(state))
        import data_service as ds
        monkeypatch.setattr(ds, "_STATE_FILE_CANDIDATES", [str(path)])

        metrics = get_system_metrics()
        assert metrics["status"] == "stale"

    def test_idle_status(self, tmp_path, monkeypatch):
        """A run_time 30 minutes ago should be 'idle'."""
        path = tmp_path / "s.json"
        time_30m_ago = (datetime.now() - timedelta(minutes=30)).isoformat()
        state = _make_state(last_run_time=time_30m_ago)
        path.write_text(json.dumps(state))
        import data_service as ds
        monkeypatch.setattr(ds, "_STATE_FILE_CANDIDATES", [str(path)])

        metrics = get_system_metrics()
        assert metrics["status"] == "idle"

    def test_returns_empty_when_missing(self, missing_state):
        metrics = get_system_metrics()
        assert metrics == _empty_metrics()


# ---------------------------------------------------------------------------
# Tests: prediction & parameter history
# ---------------------------------------------------------------------------

class TestHistory:
    def test_prediction_history_returns_list(self, state_file):
        assert isinstance(get_prediction_history(), list)

    def test_parameter_history_returns_list(self, state_file):
        assert isinstance(get_parameter_history(), list)

    def test_prediction_history_with_entries(self, tmp_path, monkeypatch):
        path = tmp_path / "s.json"
        entries = [
            {"timestamp": "2026-04-01T12:00:00", "error": 0.1, "context": {}},
            {"timestamp": "2026-04-01T12:30:00", "error": -0.2, "context": {}},
        ]
        state = _make_state(prediction_history=entries)
        path.write_text(json.dumps(state))
        import data_service as ds
        monkeypatch.setattr(ds, "_STATE_FILE_CANDIDATES", [str(path)])

        hist = get_prediction_history()
        assert len(hist) == 2
        assert hist[0]["error"] == 0.1

    def test_empty_when_missing(self, missing_state):
        assert get_prediction_history() == []
        assert get_parameter_history() == []


# ---------------------------------------------------------------------------
# Tests: baseline & effective parameters
# ---------------------------------------------------------------------------

class TestParameters:
    def test_baseline_returns_dict(self, state_file):
        bp = get_baseline_parameters()
        assert "thermal_time_constant" in bp
        assert bp["thermal_time_constant"] == 4.39

    def test_adjustments_returns_dict(self, state_file):
        adj = get_parameter_adjustments()
        assert "thermal_time_constant_delta" in adj
        assert adj["thermal_time_constant_delta"] == 0.0

    def test_effective_equals_baseline_when_deltas_zero(self, state_file):
        eff = get_effective_parameters()
        bp = get_baseline_parameters()
        for key, val in eff.items():
            assert val == pytest.approx(bp[key], abs=1e-9)

    def test_effective_applies_deltas(self, tmp_path, monkeypatch):
        path = tmp_path / "s.json"
        state = _make_state(
            delta_overrides={"thermal_time_constant_delta": 0.5}
        )
        path.write_text(json.dumps(state))
        import data_service as ds
        monkeypatch.setattr(ds, "_STATE_FILE_CANDIDATES", [str(path)])

        eff = get_effective_parameters()
        assert eff["thermal_time_constant"] == pytest.approx(4.39 + 0.5)

    def test_empty_when_missing(self, missing_state):
        assert get_baseline_parameters() == {}
        assert get_parameter_adjustments() == {}
        assert get_effective_parameters() == {}


# ---------------------------------------------------------------------------
# Tests: heat source channels
# ---------------------------------------------------------------------------

class TestHeatSourceChannels:
    def test_empty_by_default(self, state_file):
        assert get_heat_source_channels() == {}
        assert get_channel_summary() == []

    def test_channel_summary(self, tmp_path, monkeypatch):
        path = tmp_path / "s.json"
        channels = {
            "heat_pump": {
                "parameters": {"thermal_time_constant": 4.4, "heat_loss_coefficient": 0.12},
                "history_count": 50,
                "history": [
                    {"error": 0.1},
                    {"error": -0.2},
                    {"error": 0.05},
                ],
            },
            "solar": {
                "parameters": {"pv_heat_weight": 0.002},
                "history_count": 10,
                "history": [{"error": 0.3}],
            },
        }
        state = _make_state(heat_source_channels=channels)
        path.write_text(json.dumps(state))
        import data_service as ds
        monkeypatch.setattr(ds, "_STATE_FILE_CANDIDATES", [str(path)])

        summary = get_channel_summary()
        assert len(summary) == 2

        hp = next(c for c in summary if c["channel"] == "heat_pump")
        assert hp["history_count"] == 50
        # avg of abs(0.1) + abs(-0.2) + abs(0.05) = 0.35/3 ≈ 0.1167
        assert hp["recent_avg_abs_error"] == pytest.approx(
            (0.1 + 0.2 + 0.05) / 3, abs=1e-3
        )

    def test_empty_when_missing(self, missing_state):
        assert get_heat_source_channels() == {}
        assert get_channel_summary() == []


# ---------------------------------------------------------------------------
# Tests: metadata & operational state
# ---------------------------------------------------------------------------

class TestMetadataAndOperational:
    def test_metadata(self, state_file):
        md = get_metadata()
        assert md["version"] == "1.0"

    def test_operational_state(self, state_file):
        ops = get_operational_state()
        assert ops["last_indoor_temp"] == 22.5

    def test_empty_when_missing(self, missing_state):
        assert get_metadata() == {}
        assert get_operational_state() == {}


# ---------------------------------------------------------------------------
# Tests: state file info
# ---------------------------------------------------------------------------

class TestStateFileInfo:
    def test_returns_info(self, state_file):
        info = get_state_file_info()
        assert info is not None
        assert info["size_bytes"] > 0
        assert info["size_kb"] > 0
        assert isinstance(info["last_modified"], datetime)

    def test_returns_none_when_missing(self, missing_state):
        assert get_state_file_info() is None
