"""
Test Phase 2: Ensure only the correct channel learns from its own events, and no cross-contamination occurs.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from src import heat_source_channels
from src.heat_source_channels import (
    FireplaceChannel,
    HeatPumpChannel,
    HeatSourceChannelOrchestrator,
    SolarChannel,
    TVChannel,
)
from src.thermal_config import ThermalParameterConfig

def make_context(
    fireplace_on=0,
    pv_power=0,
    tv_on=0,
    heat_pump_active=False,
):
    return {
        "outlet_temp": 40.0 if heat_pump_active else 21.0,
        "current_indoor": 21.0,
        "outdoor_temp": 5.0,
        "pv_power": pv_power,
        "fireplace_on": fireplace_on,
        "tv_on": tv_on,
        "avg_cloud_cover": 50.0,
        "inlet_temp": 30.0 if heat_pump_active else 21.0,
        "delta_t": 5.0 if heat_pump_active else 0.0,
        "thermal_power": 2.0 if heat_pump_active else 0.0,
        "heat_pump_active": heat_pump_active,
    }


@pytest.mark.parametrize(
    ("scenario", "kwargs", "expected_channels"),
    [
        ("none", {"heat_pump_active": False}, set()),
        ("heat_pump", {"heat_pump_active": True}, {"heat_pump"}),
        ("pv", {"pv_power": 1000}, {"pv"}),
        ("fireplace", {"fireplace_on": 1}, {"fireplace"}),
        ("tv", {"tv_on": 1}, {"tv"}),
        ("hp_pv", {"heat_pump_active": True, "pv_power": 1000}, {"pv"}),
        ("hp_fireplace", {"heat_pump_active": True, "fireplace_on": 1}, {"fireplace"}),
        ("hp_tv", {"heat_pump_active": True, "tv_on": 1}, {"tv"}),
        ("pv_fireplace", {"pv_power": 1000, "fireplace_on": 1}, {"pv", "fireplace"}),
        ("pv_tv", {"pv_power": 1000, "tv_on": 1}, {"pv", "tv"}),
        ("fireplace_tv", {"fireplace_on": 1, "tv_on": 1}, {"fireplace", "tv"}),
        (
            "all_four",
            {"heat_pump_active": True, "pv_power": 1000, "fireplace_on": 1, "tv_on": 1},
            {"pv", "fireplace", "tv"},
        ),
    ],
)
def test_channel_routing_matrix(scenario, kwargs, expected_channels):
    orch = HeatSourceChannelOrchestrator()
    ctx = make_context(**kwargs)
    orch.route_learning(1.0, ctx)

    for name, channel in orch.channels.items():
        expected_count = 1 if name in expected_channels else 0
        assert len(channel.history) == expected_count, (
            f"Scenario {scenario}: expected {expected_count} records for {name}, "
            f"got {len(channel.history)}"
        )


def test_total_heat_is_zero_when_no_source_is_active():
    orch = HeatSourceChannelOrchestrator()
    assert orch.total_heat(make_context()) == pytest.approx(0.0)


def test_total_heat_combines_all_active_sources():
    orch = HeatSourceChannelOrchestrator()
    ctx = make_context(
        heat_pump_active=True,
        pv_power=1000,
        fireplace_on=1,
        tv_on=1,
    )
    total = orch.total_heat(ctx)
    expected = sum(
        channel.estimate_heat_contribution(ctx)
        for channel in orch.channels.values()
    )
    assert total == pytest.approx(expected)


@pytest.mark.parametrize(
    ("channel_name", "kwargs", "parameter_name"),
    [
        ("heat_pump", {"heat_pump_active": True}, "outlet_effectiveness"),
        ("pv", {"pv_power": 1200}, "pv_heat_weight"),
        ("fireplace", {"fireplace_on": 1}, "fp_heat_output_kw"),
        ("tv", {"tv_on": 1}, "tv_heat_weight"),
    ],
)
def test_active_channel_self_learning_updates_parameters(
    channel_name,
    kwargs,
    parameter_name,
):
    orch = HeatSourceChannelOrchestrator()
    channel = orch.channels[channel_name]
    initial_value = channel.get_learnable_parameters()[parameter_name]

    for _ in range(6):
        orch.route_learning(1.0, make_context(**kwargs))

    updated_value = channel.get_learnable_parameters()[parameter_name]
    assert updated_value != pytest.approx(initial_value)


def test_no_source_active_does_not_mutate_any_channel():
    orch = HeatSourceChannelOrchestrator()
    initial = orch.get_all_parameters()

    for _ in range(6):
        orch.route_learning(1.0, make_context())

    assert orch.get_all_parameters() == initial


@pytest.mark.parametrize(
    "param_name",
    [
        "delta_t_floor",
        "cloud_factor_exponent",
        "solar_decay_tau_hours",
        "fp_heat_output_kw",
        "fp_decay_time_constant",
        "room_spread_delay_minutes",
    ],
)
def test_channel_specific_defaults_are_registered_in_thermal_config(
    param_name,
):
    default = ThermalParameterConfig.get_default(param_name)
    lower, upper = ThermalParameterConfig.get_bounds(param_name)
    assert lower <= default <= upper


def test_channel_constructors_use_unified_parameter_defaults(monkeypatch):
    defaults = {
        "outlet_effectiveness": 0.61,
        "slab_time_constant_hours": 1.8,
        "delta_t_floor": 3.2,
        "pv_heat_weight": 0.0031,
        "solar_lag_minutes": 72.0,
        "cloud_factor_exponent": 1.4,
        "solar_decay_tau_hours": 0.9,
        "fp_heat_output_kw": 6.4,
        "fp_decay_time_constant": 1.1,
        "room_spread_delay_minutes": 42.0,
        "tv_heat_weight": 0.47,
    }

    monkeypatch.setattr(
        heat_source_channels.thermal_params,
        "get",
        lambda name: defaults[name],
    )

    heat_pump = HeatPumpChannel()
    solar = SolarChannel()
    fireplace = FireplaceChannel()
    tv = TVChannel()

    assert heat_pump.outlet_effectiveness == pytest.approx(0.61)
    assert heat_pump.slab_time_constant_hours == pytest.approx(1.8)
    assert heat_pump.delta_t_floor == pytest.approx(3.2)
    assert solar.pv_heat_weight == pytest.approx(0.0031)
    assert solar.solar_lag_minutes == pytest.approx(72.0)
    assert solar.cloud_factor_exponent == pytest.approx(1.4)
    assert solar.solar_decay_tau_hours == pytest.approx(0.9)
    assert fireplace.fp_heat_output_kw == pytest.approx(6.4)
    assert fireplace.fp_decay_time_constant == pytest.approx(1.1)
    assert fireplace.room_spread_delay_minutes == pytest.approx(42.0)
    assert tv.tv_heat_weight == pytest.approx(0.47)
