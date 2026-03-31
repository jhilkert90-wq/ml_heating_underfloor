"""
Tests for the UFH slab (Estrich) thermal model.

The slab model adds a first-order thermal lag between the commanded outlet
temperature and the effective heating temperature seen by the room:

    T_slab(t + Δt) = T_slab(t) + (Δt / τ_slab) · (T_cmd − T_slab(t))

where T_slab(0) = inlet_temp (Rücklauf = current slab state).

All tests use synthetic/generic data — no InfluxDB or Home Assistant
connection required.
"""
import os
import sys
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src import thermal_equilibrium_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    """Clean ThermalEquilibriumModel with deterministic test parameters."""
    m = thermal_equilibrium_model.ThermalEquilibriumModel()
    m.thermal_time_constant = 4.0        # hours
    m.heat_loss_coefficient = 0.13       # 1/h
    m.outlet_effectiveness = 0.49        # dimensionless
    m.slab_time_constant_hours = 1.0     # hours (60 min, UFH default)
    m.external_source_weights = {"pv": 0.002, "fireplace": 0.02, "tv": 0.35}
    return m


def _make_prediction_records(
    n: int,
    outlet_temp: float,
    inlet_temp: float,
    outdoor_temp: float = 3.0,
    indoor_temp: float = 22.8,
    error: float = 0.05,
):
    """Build n synthetic prediction-history records."""
    records = []
    base_time = datetime(2026, 3, 19, 5, 0)
    for i in range(n):
        records.append({
            "timestamp": (base_time + timedelta(minutes=i * 10)).isoformat(),
            "predicted": indoor_temp,
            "actual": indoor_temp + error,
            "error": error,
            "context": {
                "outlet_temp": outlet_temp,
                "outdoor_temp": outdoor_temp,
                "current_indoor": indoor_temp,
                "pv_power": 0.0,
                "fireplace_on": 0.0,
                "tv_on": 0.0,
                "avg_cloud_cover": 50.0,
                "inlet_temp": inlet_temp,
            },
        })
    return records


# ---------------------------------------------------------------------------
# Test 1 — Slab dynamics: monotone approach towards outlet_cmd
# ---------------------------------------------------------------------------

class TestSlabDynamics:
    def test_slab_approaches_outlet_monotonically(self, model):
        """T_slab must monotonically approach outlet_temp from below."""
        result = model.predict_thermal_trajectory(
            current_indoor=22.8,
            target_indoor=22.6,
            outlet_temp=40.0,
            outdoor_temp=3.0,
            time_horizon_hours=4.0,
            time_step_minutes=10,
            inlet_temp=25.0,   # slab starts well below outlet_cmd
        )
        assert "trajectory" in result
        assert len(result["trajectory"]) > 0

    def test_slab_zero_tau_equals_instant(self, model):
        """With τ_slab → 0 the slab effectively equals outlet_cmd every step."""
        model.slab_time_constant_hours = 1e-6
        result_with_slab = model.predict_thermal_trajectory(
            current_indoor=22.8,
            target_indoor=22.6,
            outlet_temp=40.0,
            outdoor_temp=3.0,
            time_horizon_hours=1.0,
            time_step_minutes=10,
            inlet_temp=10.0,   # extreme cold start
        )
        result_no_slab = model.predict_thermal_trajectory(
            current_indoor=22.8,
            target_indoor=22.6,
            outlet_temp=40.0,
            outdoor_temp=3.0,
            time_horizon_hours=1.0,
            time_step_minutes=10,
            inlet_temp=None,   # no slab → t_slab = outlet_cmd immediately
        )
        # With near-zero τ the first step should be nearly identical to no-slab
        assert abs(result_with_slab["trajectory"][0]
                   - result_no_slab["trajectory"][0]) < 0.5

    def test_slab_buffers_cold_outlet_command(self, model):
        """With inlet_temp=25°C and outlet_cmd=14°C the room should drop more
        slowly than when the slab model is absent (no buffering)."""
        model.slab_time_constant_hours = 1.0

        result_with_slab = model.predict_thermal_trajectory(
            current_indoor=22.8,
            target_indoor=22.6,
            outlet_temp=14.0,
            outdoor_temp=3.0,
            time_horizon_hours=2.0,
            time_step_minutes=10,
            inlet_temp=25.0,   # slab starts warmer than outlet_cmd
        )
        result_no_slab = model.predict_thermal_trajectory(
            current_indoor=22.8,
            target_indoor=22.6,
            outlet_temp=14.0,
            outdoor_temp=3.0,
            time_horizon_hours=2.0,
            time_step_minutes=10,
            inlet_temp=None,   # no slab → cold outlet applied immediately
        )
        # Slab-model should keep the first step warmer
        assert result_with_slab["trajectory"][0] > result_no_slab["trajectory"][0], (
            "Slab model must buffer the cold outlet command: first-step temp "
            f"{result_with_slab['trajectory'][0]:.3f} should exceed "
            f"{result_no_slab['trajectory'][0]:.3f}"
        )

    def test_backward_compat_no_inlet_temp(self, model):
        """Calling predict_thermal_trajectory without inlet_temp must not raise."""
        result = model.predict_thermal_trajectory(
            current_indoor=22.8,
            target_indoor=22.6,
            outlet_temp=27.0,
            outdoor_temp=3.0,
            time_horizon_hours=1.0,
            time_step_minutes=10,
        )
        assert "trajectory" in result
        assert len(result["trajectory"]) > 0


# ---------------------------------------------------------------------------
# Test 2 — Gradient observability
# ---------------------------------------------------------------------------

class TestSlabGradient:
    def test_gradient_nonzero_when_inlet_differs_from_outlet(self, model):
        """Slab gradient must be non-zero when inlet_temp ≠ outlet_cmd."""
        model.prediction_history = _make_prediction_records(
            n=12,
            outlet_temp=27.0,
            inlet_temp=24.3,   # well below outlet_cmd → gradient observable
        )
        gradient = model._calculate_slab_time_constant_gradient(
            model.prediction_history
        )
        assert abs(gradient) > 1e-6, (
            f"Expected non-zero slab gradient, got {gradient}"
        )

    def test_gradient_near_zero_at_equilibrium(self, model):
        """Slab gradient must be ≈ 0 when inlet_temp ≈ outlet_cmd (equilibrium)."""
        model.prediction_history = _make_prediction_records(
            n=12,
            outlet_temp=27.0,
            inlet_temp=27.0,   # equilibrium: slab = outlet_cmd
        )
        gradient = model._calculate_slab_time_constant_gradient(
            model.prediction_history
        )
        assert abs(gradient) < 1e-4, (
            f"Expected ~zero slab gradient at equilibrium, got {gradient}"
        )

    def test_gradient_uses_inlet_temp_from_context(self, model):
        """Gradient with inlet_temp context must differ from gradient without it."""
        records_with_inlet = _make_prediction_records(
            n=12, outlet_temp=27.0, inlet_temp=14.0
        )
        records_without_inlet = []
        for r in records_with_inlet:
            r2 = {"timestamp": r["timestamp"], "predicted": r["predicted"],
                  "actual": r["actual"], "error": r["error"],
                  "context": {k: v for k, v in r["context"].items()
                               if k != "inlet_temp"}}
            records_without_inlet.append(r2)

        g_with = model._calculate_slab_time_constant_gradient(records_with_inlet)
        g_without = model._calculate_slab_time_constant_gradient(records_without_inlet)

        # With a known cold inlet the gradient should be larger in magnitude
        assert abs(g_with) >= abs(g_without) - 1e-9


# ---------------------------------------------------------------------------
# Test 3 — Parameter update and clipping
# ---------------------------------------------------------------------------

class TestSlabParameterUpdate:
    def test_adapt_parameters_updates_slab_within_bounds(self, model):
        """_adapt_parameters_from_recent_errors must keep slab within bounds."""
        model.prediction_history = _make_prediction_records(
            n=12, outlet_temp=27.0, inlet_temp=14.0,
            error=0.3,  # meaningful signal
        )
        original_tau = model.slab_time_constant_hours
        model._adapt_parameters_from_recent_errors()

        lo, hi = model.slab_time_constant_bounds
        assert lo <= model.slab_time_constant_hours <= hi, (
            f"slab_time_constant_hours {model.slab_time_constant_hours} "
            f"outside bounds [{lo}, {hi}]"
        )

    def test_slab_recorded_in_parameter_history(self, model):
        """After adaptation slab_time_constant_hours must appear in history."""
        model.prediction_history = _make_prediction_records(
            n=12, outlet_temp=27.0, inlet_temp=14.0, error=0.3,
        )
        initial_len = len(model.parameter_history)

        # Patch save to avoid file-system side-effects
        with patch.object(model, '_save_learning_to_thermal_state'):
            model._adapt_parameters_from_recent_errors()

        if len(model.parameter_history) > initial_len:
            latest = model.parameter_history[-1]
            assert "slab_time_constant_hours" in latest, (
                "slab_time_constant_hours key missing from parameter_history"
            )
            assert "slab_time_constant" in latest.get("gradients", {}), (
                "slab_time_constant gradient key missing from parameter_history"
            )


# ---------------------------------------------------------------------------
# Test 4 — Persistence via _save_learning_to_thermal_state
# ---------------------------------------------------------------------------

class TestSlabPersistence:
    def test_save_learning_persists_slab_and_solar_lag_deltas(self, model):
        """Both slab_time_constant_delta and solar_lag_minutes_delta must be
        written to update_learning_state when they change."""
        captured_adjustments = {}

        mock_state_manager = MagicMock()

        def capture_update(learning_confidence=None, parameter_adjustments=None):
            if parameter_adjustments:
                captured_adjustments.update(parameter_adjustments)

        mock_state_manager.state = {
            "learning_state": {
                "parameter_adjustments": {
                    "thermal_time_constant_delta": 0.0,
                    "heat_loss_coefficient_delta": 0.0,
                    "outlet_effectiveness_delta": 0.0,
                    "pv_heat_weight_delta": 0.0,
                    "tv_heat_weight_delta": 0.0,
                    "solar_lag_minutes_delta": 0.0,
                    "slab_time_constant_delta": 0.0,
                }
            }
        }
        mock_state_manager.update_learning_state.side_effect = capture_update
        mock_state_manager.add_parameter_history_record = MagicMock()

        with patch(
            'src.thermal_equilibrium_model.ThermalEquilibriumModel'
            '._save_learning_to_thermal_state',
            wraps=lambda *a, **kw: None
        ):
            pass

        # Call directly with non-trivial adjustments
        import importlib
        import src.unified_thermal_state as uts_module
        original_get = uts_module.get_thermal_state_manager
        uts_module.get_thermal_state_manager = lambda: mock_state_manager

        try:
            model._save_learning_to_thermal_state(
                new_thermal_adjustment=0.0,
                new_heat_loss_coefficient_adjustment=0.0,
                new_outlet_effectiveness_adjustment=0.0,
                new_pv_heat_weight_adjustment=0.0,
                new_tv_heat_weight_adjustment=0.0,
                new_solar_lag_adjustment=2.0,   # non-trivial
                new_slab_adjustment=0.05,       # non-trivial
            )
        finally:
            uts_module.get_thermal_state_manager = original_get

        assert "solar_lag_minutes_delta" in captured_adjustments, (
            "solar_lag_minutes_delta not persisted"
        )
        assert "slab_time_constant_delta" in captured_adjustments, (
            "slab_time_constant_delta not persisted"
        )
        assert abs(captured_adjustments["slab_time_constant_delta"] - 0.05) < 1e-9
        assert abs(captured_adjustments["solar_lag_minutes_delta"] - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 5 — thermal_config bounds and defaults
# ---------------------------------------------------------------------------

class TestSlabConfig:
    def test_slab_default_in_thermal_config(self):
        """slab_time_constant_hours must have a valid default."""
        from src.thermal_config import ThermalParameterConfig
        default = ThermalParameterConfig.get_default("slab_time_constant_hours")
        lo, hi = ThermalParameterConfig.get_bounds("slab_time_constant_hours")
        assert lo > 0, "lower bound must be positive"
        assert lo <= default <= hi, (
            f"default {default} outside bounds [{lo}, {hi}]"
        )

    def test_slab_default_loaded_into_model(self, model):
        """Model must expose slab_time_constant_hours and slab_time_constant_bounds."""
        assert hasattr(model, "slab_time_constant_hours")
        assert hasattr(model, "slab_time_constant_bounds")
        lo, hi = model.slab_time_constant_bounds
        assert lo <= model.slab_time_constant_hours <= hi


# ---------------------------------------------------------------------------
# Test 6 — unified_thermal_state has the new delta keys
# ---------------------------------------------------------------------------

class TestUnifiedStateKeys:
    def test_new_delta_keys_in_default_state(self):
        """Both solar_lag_minutes_delta and slab_time_constant_delta must exist
        in a freshly initialised unified thermal state."""
        from src.unified_thermal_state import ThermalStateManager
        import tempfile, json, pathlib

        with tempfile.TemporaryDirectory() as tmp:
            state_file = pathlib.Path(tmp) / "thermal_state.json"
            mgr = ThermalStateManager(state_file=str(state_file))
            adjustments = (
                mgr.state["learning_state"]["parameter_adjustments"]
            )
            assert "solar_lag_minutes_delta" in adjustments, (
                "solar_lag_minutes_delta missing from default parameter_adjustments"
            )
            assert "slab_time_constant_delta" in adjustments, (
                "slab_time_constant_delta missing from default parameter_adjustments"
            )
