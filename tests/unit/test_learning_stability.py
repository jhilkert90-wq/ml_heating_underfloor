"""
Tests for learning stability improvements (Scenario 5).

Verifies:
- Dead zone blocks updates when avg |error| < 0.05°C
- HLC update clipped at ±0.005  (MAX_HEAT_LOSS_COEFFICIENT_CHANGE)
- OE  update clipped at ±0.005  (MAX_OUTLET_EFFECTIVENESS_CHANGE)
- pvw update clipped at ±0.0002 (max_weight_change)
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src.thermal_constants import PhysicsConstants


def _make_prediction(error, outdoor_temp=5.0, pv_power=0):
    """Create a minimal prediction_history entry."""
    return {
        "error": error,
        "timestamp": "2026-03-24T12:00:00",
        "abs_error": abs(error),
        "context": {
            "outlet_temp": 40.0,
            "current_indoor": 21.0,
            "outdoor_temp": outdoor_temp,
            "pv_power": pv_power,
            "pv_forecast": [0] * 24,
            "outdoor_forecast": [outdoor_temp] * 24,
            "fireplace_on": 0,
            "tv_on": 0,
            "avg_cloud_cover": 50.0,
            "indoor_temp_delta_60m": 0.0,
            "inlet_temp": 30.0,
            "delta_t": 0.0,
        },
    }


@pytest.fixture
def model():
    """Create model with realistic parameters and enough history."""
    m = ThermalEquilibriumModel()
    # These tests target the legacy global recent-window learner.
    m.orchestrator = None
    m.thermal_time_constant = 3.8
    m.heat_loss_coefficient = 0.146
    m.outlet_effectiveness = 0.936
    m.pv_heat_weight = 0.001
    m.tv_heat_weight = 0.1
    m.learning_rate = 0.01
    m.learning_confidence = 1.0
    return m


def _make_runtime_replay_prediction(parameter_name, climate_mode="heating"):
    """Representative stored prediction record for gradient replay tests."""
    pv_history = [
        0.0, 80.0, 120.0, 180.0, 260.0, 340.0, 430.0, 520.0, 640.0,
        780.0, 920.0, 1040.0, 900.0, 760.0, 620.0, 500.0, 420.0, 360.0,
    ]
    context = {
        "outlet_temp": 30.0,
        "outdoor_temp": 5.0,
        "current_indoor": 21.0,
        "pv_power": pv_history[-1],
        "pv_power_history": pv_history,
        "pv_forecast": [700.0, 900.0, 660.0, 420.0],
        "outdoor_forecast": [4.0, 3.0, 3.0, 4.0],
        "fireplace_on": 0,
        "tv_on": 0,
        "avg_cloud_cover": 35.0,
        "inlet_temp": 27.5,
        "delta_t": 2.5,
        "thermal_power": 1.8,
        "auxiliary_heat": 0.0,
        "climate_mode": climate_mode,
        "indoor_temp_delta_60m": 0.0,
    }

    if parameter_name == "tv_heat_weight":
        context["tv_on"] = 1
    if parameter_name == "solar_lag_minutes":
        context["pv_forecast"] = [1100.0, 950.0, 600.0, 250.0]
        context["pv_power_history"] = [
            0.0, 0.0, 40.0, 120.0, 260.0, 420.0, 650.0, 910.0, 1120.0,
            980.0, 760.0, 540.0, 380.0, 260.0, 190.0, 150.0, 120.0, 90.0,
        ]
        context["pv_power"] = context["pv_power_history"][-1]
    if parameter_name == "slab_time_constant_hours":
        context["outlet_temp"] = 33.0
        context["inlet_temp"] = 28.0
        context["delta_t"] = 3.5
        context["thermal_power"] = 2.6
    if climate_mode == "cooling":
        context.update(
            {
                "outlet_temp": 19.0,
                "outdoor_temp": 29.0,
                "current_indoor": 24.0,
                "outdoor_forecast": [30.0, 31.0, 30.0, 29.0],
                "inlet_temp": 22.0,
                "delta_t": -2.5,
                "thermal_power": -1.3,
                "climate_mode": "cooling",
            }
        )

    return {
        "error": 1.0,
        "timestamp": "2026-05-04T12:00:00",
        "context": context,
    }


def _runtime_signal(model, parameter_name, epsilon, climate_mode="heating"):
    prediction = _make_runtime_replay_prediction(parameter_name, climate_mode)
    gradient = model._calculate_parameter_gradient(
        parameter_name, epsilon, [prediction]
    )
    return abs(gradient * 2 * epsilon)


# ── Dead Zone ────────────────────────────────────────────────────────

class TestDeadZone:
    def test_dead_zone_constant_exists(self):
        assert hasattr(PhysicsConstants, "LEARNING_DEAD_ZONE")
        assert PhysicsConstants.LEARNING_DEAD_ZONE == 0.01

    def test_dead_zone_blocks_learning_below_threshold(self, model):
        """Errors averaging 0.005°C (< 0.01) must NOT change parameters."""
        window = model.recent_errors_window
        model.prediction_history = [_make_prediction(0.005)] * window

        old_hlc = model.heat_loss_coefficient
        old_oe = model.outlet_effectiveness
        old_pvw = model.pv_heat_weight

        model._adapt_parameters_from_recent_errors()

        assert model.heat_loss_coefficient == old_hlc
        assert model.outlet_effectiveness == old_oe
        assert model.pv_heat_weight == old_pvw

    def test_dead_zone_allows_learning_above_threshold(self, model):
        """Errors averaging 0.15°C (> 0.05) MUST trigger parameter updates."""
        window = model.recent_errors_window
        model.prediction_history = [_make_prediction(0.15)] * window

        old_hlc = model.heat_loss_coefficient

        model._adapt_parameters_from_recent_errors()

        # At least HLC should have moved (gradient is non-zero for these inputs)
        assert model.heat_loss_coefficient != old_hlc

    def test_dead_zone_boundary(self, model):
        """Errors exactly at 0.01 should be blocked (< not <=)."""
        window = model.recent_errors_window
        # Average will be exactly 0.009 – just inside dead zone
        model.prediction_history = [_make_prediction(0.009)] * window

        old_hlc = model.heat_loss_coefficient
        model._adapt_parameters_from_recent_errors()
        assert model.heat_loss_coefficient == old_hlc


# ── HLC Clip ─────────────────────────────────────────────────────────

class TestHLCClip:
    def test_max_hlc_change_constant(self):
        assert PhysicsConstants.MAX_HEAT_LOSS_COEFFICIENT_CHANGE == 0.005

    def test_hlc_update_clipped_to_005(self, model):
        """Even with huge error, HLC must not change by more than 0.005."""
        window = model.recent_errors_window
        # Large but sub-catastrophic error drives a big gradient
        model.prediction_history = [_make_prediction(2.0)] * window

        old_hlc = model.heat_loss_coefficient
        model._adapt_parameters_from_recent_errors()

        hlc_change = abs(model.heat_loss_coefficient - old_hlc)
        assert hlc_change <= 0.005 + 1e-9, (
            f"HLC changed by {hlc_change:.6f}, exceeds 0.005 clip"
        )


# ── OE Clip ──────────────────────────────────────────────────────────

class TestOEClip:
    def test_max_oe_change_constant(self):
        assert PhysicsConstants.MAX_OUTLET_EFFECTIVENESS_CHANGE == 0.005

    def test_oe_update_clipped_to_005(self, model):
        """OE must not change by more than 0.005 per step."""
        window = model.recent_errors_window
        model.prediction_history = [_make_prediction(2.0)] * window

        old_oe = model.outlet_effectiveness
        model._adapt_parameters_from_recent_errors()

        oe_change = abs(model.outlet_effectiveness - old_oe)
        assert oe_change <= 0.005 + 1e-9, (
            f"OE changed by {oe_change:.6f}, exceeds 0.005 clip"
        )


# ── PVW Clip ─────────────────────────────────────────────────────────

class TestPVWClip:
    def test_pvw_update_clipped_to_00002(self, model):
        """pvw must not change by more than 0.0002 per step."""
        window = model.recent_errors_window
        # Use daytime conditions with PV so the pvw gradient is non-zero
        model.prediction_history = [
            _make_prediction(2.0, outdoor_temp=5.0, pv_power=3000)
        ] * window

        old_pvw = model.pv_heat_weight
        model._adapt_parameters_from_recent_errors()

        pvw_change = abs(model.pv_heat_weight - old_pvw)
        assert pvw_change <= 0.0002 + 1e-9, (
            f"pvw changed by {pvw_change:.7f}, exceeds 0.0002 clip"
        )


# ── Epsilon Constants ────────────────────────────────────────────────

class TestEpsilonConstants:
    """All 7 learnable-parameter epsilon values must be defined in PhysicsConstants."""

    def test_thermal_time_constant_epsilon_exists(self):
        assert hasattr(PhysicsConstants, "THERMAL_TIME_CONSTANT_EPSILON")
        assert PhysicsConstants.THERMAL_TIME_CONSTANT_EPSILON > 0

    def test_heat_loss_coefficient_epsilon_exists(self):
        assert hasattr(PhysicsConstants, "HEAT_LOSS_COEFFICIENT_EPSILON")
        assert PhysicsConstants.HEAT_LOSS_COEFFICIENT_EPSILON > 0

    def test_outlet_effectiveness_epsilon_exists(self):
        assert hasattr(PhysicsConstants, "OUTLET_EFFECTIVENESS_EPSILON")
        assert PhysicsConstants.OUTLET_EFFECTIVENESS_EPSILON > 0

    def test_pv_heat_weight_epsilon_exists(self):
        assert hasattr(PhysicsConstants, "PV_HEAT_WEIGHT_EPSILON")
        assert PhysicsConstants.PV_HEAT_WEIGHT_EPSILON > 0

    def test_tv_heat_weight_epsilon_exists(self):
        assert hasattr(PhysicsConstants, "TV_HEAT_WEIGHT_EPSILON")
        assert PhysicsConstants.TV_HEAT_WEIGHT_EPSILON > 0

    def test_solar_lag_epsilon_exists(self):
        assert hasattr(PhysicsConstants, "SOLAR_LAG_EPSILON")
        assert PhysicsConstants.SOLAR_LAG_EPSILON > 0

    def test_slab_time_constant_epsilon_exists(self):
        assert hasattr(PhysicsConstants, "SLAB_TIME_CONSTANT_EPSILON")
        assert PhysicsConstants.SLAB_TIME_CONSTANT_EPSILON > 0

    @pytest.mark.parametrize(
        ("parameter_name", "epsilon", "min_signal", "max_signal"),
        [
            (
                "thermal_time_constant",
                PhysicsConstants.THERMAL_TIME_CONSTANT_EPSILON,
                0.05,
                0.5,
            ),
            (
                "heat_loss_coefficient",
                PhysicsConstants.HEAT_LOSS_COEFFICIENT_EPSILON,
                0.05,
                0.5,
            ),
            (
                "outlet_effectiveness",
                PhysicsConstants.OUTLET_EFFECTIVENESS_EPSILON,
                0.05,
                0.5,
            ),
            (
                "pv_heat_weight",
                PhysicsConstants.PV_HEAT_WEIGHT_EPSILON,
                0.05,
                0.5,
            ),
            (
                "tv_heat_weight",
                PhysicsConstants.TV_HEAT_WEIGHT_EPSILON,
                0.05,
                0.25,
            ),
        ],
    )
    def test_core_epsilons_produce_runtime_signal(
        self, model, parameter_name, epsilon, min_signal, max_signal
    ):
        signal = _runtime_signal(model, parameter_name, epsilon)
        assert min_signal <= signal <= max_signal, (
            f"{parameter_name} signal {signal:.6f} outside "
            f"[{min_signal:.3f}, {max_signal:.3f}]"
        )

    def test_solar_lag_epsilon_produces_nonzero_runtime_signal(self, model):
        signal = _runtime_signal(
            model,
            "solar_lag_minutes",
            PhysicsConstants.SOLAR_LAG_EPSILON,
        )
        assert signal > 0.001, (
            f"solar_lag_minutes signal too small: {signal:.6f}"
        )

    def test_slab_epsilon_produces_transient_runtime_signal(self, model):
        signal = _runtime_signal(
            model,
            "slab_time_constant_hours",
            PhysicsConstants.SLAB_TIME_CONSTANT_EPSILON,
        )
        assert signal > 0.02, (
            f"slab_time_constant_hours signal too small: {signal:.6f}"
        )

    def test_cooling_replay_uses_mode_aware_pump_logic(self, model):
        signal = _runtime_signal(
            model,
            "slab_time_constant_hours",
            PhysicsConstants.SLAB_TIME_CONSTANT_EPSILON,
            climate_mode="cooling",
        )
        assert signal > 0.01, (
            "Cooling replay fell back to heating-style pump-off handling "
            f"(signal={signal:.6f})"
        )
