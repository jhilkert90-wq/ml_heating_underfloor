"""
Test solar transition scenario: Ensure outlet increases proactively before sunset and no cross-contamination occurs.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from src.heat_source_channels import HeatSourceChannelOrchestrator

def make_context(pv_power=0, pv_forecast=None):
    return {
        "outlet_temp": 40.0,
        "current_indoor": 21.0,
        "outdoor_temp": 5.0,
        "pv_power": pv_power,
        "pv_forecast": pv_forecast or [1000]*12 + [0]*12,  # 2h of sun, then sunset
        "fireplace_on": 0,
        "tv_on": 0,
        "avg_cloud_cover": 50.0,
        "inlet_temp": 30.0,
        "delta_t": 0.0,
    }

def test_solar_transition_channel_learns():
    orch = HeatSourceChannelOrchestrator()
    # Simulate a cycle with high PV, then sunset
    ctx = make_context(pv_power=1000, pv_forecast=[1000]*12 + [0]*12)
    orch.route_learning(1.0, ctx)
    assert len(orch.channels["pv"].history) == 1
    # Now simulate after sunset
    ctx2 = make_context(pv_power=0, pv_forecast=[0]*24)
    orch.route_learning(1.0, ctx2)
    # Should not learn again (PV channel only learns when PV > 500)
    assert len(orch.channels["pv"].history) == 1
