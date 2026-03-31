"""
Test learning isolation for Phase 1: Ensure HP parameters do not change when fireplace or high PV is active.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src import config

def make_context(fireplace_on=0, pv_power=0):
    return {
        "outlet_temp": 40.0,
        "current_indoor": 21.0,
        "outdoor_temp": 5.0,
        "pv_power": pv_power,
        "fireplace_on": fireplace_on,
        "tv_on": 0,
        "avg_cloud_cover": 50.0,
        "inlet_temp": 30.0,
        "delta_t": 5.0,
    }

@pytest.fixture
def model_with_guards(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_HEAT_SOURCE_CHANNELS", True)
    m = ThermalEquilibriumModel()
    m.heat_loss_coefficient = 0.2
    m.outlet_effectiveness = 0.7
    return m

def test_hp_params_unchanged_when_fireplace_active(model_with_guards):
    model = model_with_guards
    old_hlc = model.heat_loss_coefficient
    old_oe = model.outlet_effectiveness
    model.update_prediction_feedback(
        predicted_temp=20.0,
        actual_temp=21.0,
        prediction_context=make_context(fireplace_on=1),
    )
    assert model.heat_loss_coefficient == old_hlc
    assert model.outlet_effectiveness == old_oe

def test_hp_params_unchanged_when_high_pv(model_with_guards):
    model = model_with_guards
    old_hlc = model.heat_loss_coefficient
    old_oe = model.outlet_effectiveness
    model.update_prediction_feedback(
        predicted_temp=20.0,
        actual_temp=21.0,
        prediction_context=make_context(pv_power=1000),
    )
    assert model.heat_loss_coefficient == old_hlc
    assert model.outlet_effectiveness == old_oe

def test_hp_params_update_when_no_contamination(model_with_guards):
    model = model_with_guards
    model.learning_confidence = 1.0
    # Feed enough records to exceed the recent_errors_window threshold
    # so that gradient adaptation is actually triggered.
    ctx = make_context()
    for _ in range(config.RECENT_ERRORS_WINDOW + 1):
        model.update_prediction_feedback(
            predicted_temp=20.0,
            actual_temp=21.0,
            prediction_context=ctx,
        )
    old_hlc = model.heat_loss_coefficient
    old_oe = model.outlet_effectiveness
    model.update_prediction_feedback(
        predicted_temp=20.0,
        actual_temp=21.0,
        prediction_context=ctx,
    )
    # At least one parameter should change
    assert (model.heat_loss_coefficient != old_hlc) or (model.outlet_effectiveness != old_oe)
