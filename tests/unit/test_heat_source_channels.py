"""
Test Phase 2: Ensure only the correct channel learns from its own events, and no cross-contamination occurs.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from src.heat_source_channels import HeatSourceChannelOrchestrator

def make_context(fireplace_on=0, pv_power=0, tv_on=0):
    return {
        "outlet_temp": 40.0,
        "current_indoor": 21.0,
        "outdoor_temp": 5.0,
        "pv_power": pv_power,
        "fireplace_on": fireplace_on,
        "tv_on": tv_on,
        "avg_cloud_cover": 50.0,
        "inlet_temp": 30.0,
        "delta_t": 0.0,
    }

def test_only_fireplace_channel_learns():
    orch = HeatSourceChannelOrchestrator()
    ctx = make_context(fireplace_on=1)
    orch.route_learning(1.0, ctx)
    assert len(orch.channels["fireplace"].history) == 1
    assert all(len(orch.channels[name].history) == 0 for name in ["heat_pump", "pv", "tv"] if name != "fireplace")

def test_only_pv_channel_learns():
    orch = HeatSourceChannelOrchestrator()
    ctx = make_context(pv_power=1000)
    orch.route_learning(1.0, ctx)
    assert len(orch.channels["pv"].history) == 1
    assert all(len(orch.channels[name].history) == 0 for name in ["heat_pump", "fireplace", "tv"] if name != "pv")

def test_only_tv_channel_learns():
    orch = HeatSourceChannelOrchestrator()
    ctx = make_context(tv_on=1)
    orch.route_learning(1.0, ctx)
    assert len(orch.channels["tv"].history) == 1
    assert all(len(orch.channels[name].history) == 0 for name in ["heat_pump", "fireplace", "pv"] if name != "tv")

def test_only_heat_pump_channel_learns():
    orch = HeatSourceChannelOrchestrator()
    ctx = make_context()
    orch.route_learning(1.0, ctx)
    assert len(orch.channels["heat_pump"].history) == 1
    assert all(len(orch.channels[name].history) == 0 for name in ["fireplace", "pv", "tv"] if name != "heat_pump")
