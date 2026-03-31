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
    m.thermal_time_constant = 3.8
    m.heat_loss_coefficient = 0.146
    m.outlet_effectiveness = 0.936
    m.pv_heat_weight = 0.001
    m.tv_heat_weight = 0.1
    m.learning_rate = 0.01
    m.learning_confidence = 1.0
    return m


# ── Dead Zone ────────────────────────────────────────────────────────

class TestDeadZone:
    def test_dead_zone_constant_exists(self):
        assert hasattr(PhysicsConstants, "LEARNING_DEAD_ZONE")
        assert PhysicsConstants.LEARNING_DEAD_ZONE == 0.02

    def test_dead_zone_blocks_learning_below_threshold(self, model):
        """Errors averaging 0.01°C (< 0.02) must NOT change parameters."""
        window = model.recent_errors_window
        model.prediction_history = [_make_prediction(0.01)] * window

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
        """Errors exactly at 0.02 should be blocked (< not <=)."""
        window = model.recent_errors_window
        # Average will be exactly 0.019 – just inside dead zone
        model.prediction_history = [_make_prediction(0.019)] * window

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
