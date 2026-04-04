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
    _get_min_records_for_learning,
)
from src.thermal_config import ThermalParameterConfig


@pytest.fixture
def mixed_source_attribution_enabled(monkeypatch):
    monkeypatch.setattr(
        heat_source_channels.config,
        "ENABLE_MIXED_SOURCE_ATTRIBUTION",
        True,
    )


def make_context(
    fireplace_on=0,
    pv_power=0,
    pv_power_current=None,
    tv_on=0,
    heat_pump_active=False,
    avg_cloud_cover=50.0,
    pv_power_history=None,
    delta_t=None,
    thermal_power=None,
):
    resolved_delta_t = 5.0 if delta_t is None and heat_pump_active else (delta_t or 0.0)
    resolved_thermal_power = (
        2.0 if thermal_power is None and heat_pump_active else (thermal_power or 0.0)
    )
    return {
        "outlet_temp": 40.0 if heat_pump_active else 21.0,
        "current_indoor": 21.0,
        "outdoor_temp": 5.0,
        "pv_power": pv_power,
        "pv_power_current": (
            pv_power if pv_power_current is None else pv_power_current
        ),
        "pv_power_history": pv_power_history,
        "fireplace_on": fireplace_on,
        "tv_on": tv_on,
        "avg_cloud_cover": avg_cloud_cover,
        "inlet_temp": 30.0 if heat_pump_active else 21.0,
        "delta_t": resolved_delta_t,
        "thermal_power": resolved_thermal_power,
        "heat_pump_active": heat_pump_active,
    }


@pytest.mark.parametrize(
    ("scenario", "kwargs", "expected_channels"),
    [
        ("none", {"heat_pump_active": False}, {"heat_pump"}),
        ("heat_pump", {"heat_pump_active": True}, {"heat_pump"}),
        ("pv", {"pv_power": 1000}, {"pv"}),
        ("fireplace", {"fireplace_on": 1}, {"fireplace"}),
        ("tv", {"tv_on": 1}, {"tv"}),
        ("hp_pv", {"heat_pump_active": True, "pv_power": 1000}, {"heat_pump", "pv"}),
        ("hp_fireplace", {"heat_pump_active": True, "fireplace_on": 1}, {"fireplace"}),
        ("hp_tv", {"heat_pump_active": True, "tv_on": 1}, {"heat_pump", "tv"}),
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
def test_channel_routing_matrix(
    mixed_source_attribution_enabled,
    scenario,
    kwargs,
    expected_channels,
):
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


def test_mixed_attribution_splits_error_proportionally_between_hp_and_pv(
    mixed_source_attribution_enabled,
):
    orch = HeatSourceChannelOrchestrator()
    context = make_context(heat_pump_active=True, pv_power=2000)

    orch.route_learning(1.0, context)

    hp_history = orch.channels["heat_pump"].history
    pv_history = orch.channels["pv"].history

    assert len(hp_history) == 1
    assert len(pv_history) == 1

    hp_contribution = orch.channels["heat_pump"].estimate_heat_contribution(context)
    pv_contribution = orch.channels["pv"].estimate_heat_contribution(context)
    total = hp_contribution + pv_contribution

    assert hp_history[-1]["error"] == pytest.approx(1.0 * hp_contribution / total)
    assert pv_history[-1]["error"] == pytest.approx(1.0 * pv_contribution / total)
    assert hp_history[-1]["context"]["attribution_applied"] is True
    assert pv_history[-1]["context"]["attribution_applied"] is True


def test_fireplace_freezes_hp_and_renormalizes_mixed_error(
    mixed_source_attribution_enabled,
):
    orch = HeatSourceChannelOrchestrator()
    context = make_context(
        heat_pump_active=True,
        pv_power=2000,
        fireplace_on=1,
        tv_on=1,
    )

    orch.route_learning(2.0, context)

    assert orch.channels["heat_pump"].history == []
    assert len(orch.channels["fireplace"].history) == 1
    assert len(orch.channels["pv"].history) == 1
    assert len(orch.channels["tv"].history) == 1

    fp_contribution = orch.channels["fireplace"].estimate_heat_contribution(context)
    pv_contribution = orch.channels["pv"].estimate_heat_contribution(context)
    tv_contribution = orch.channels["tv"].estimate_heat_contribution(context)
    total = fp_contribution + pv_contribution + tv_contribution

    assert orch.channels["fireplace"].history[-1]["error"] == pytest.approx(
        2.0 * fp_contribution / total
    )
    assert orch.channels["pv"].history[-1]["error"] == pytest.approx(
        2.0 * pv_contribution / total
    )
    assert orch.channels["tv"].history[-1]["error"] == pytest.approx(
        2.0 * tv_contribution / total
    )
    assert (
        orch.channels["fireplace"].history[-1]["context"][
            "heat_pump_frozen_by_fireplace"
        ]
        is True
    )


def test_no_external_source_defaults_learning_to_heat_pump():
    orch = HeatSourceChannelOrchestrator()

    orch.route_learning(0.5, make_context(heat_pump_active=False))

    assert len(orch.channels["heat_pump"].history) == 1
    assert orch.channels["heat_pump"].history[-1]["error"] == pytest.approx(0.5)
    assert orch.channels["pv"].history == []
    assert orch.channels["fireplace"].history == []
    assert orch.channels["tv"].history == []


def test_current_pv_signal_routes_learning_to_pv_when_smoothed_value_is_low(
    mixed_source_attribution_enabled,
):
    orch = HeatSourceChannelOrchestrator()

    orch.route_learning(
        0.75,
        make_context(
            heat_pump_active=False,
            thermal_power=0.0,
            delta_t=0.0,
            pv_power=120.0,
            pv_power_current=850.0,
        ),
    )

    assert orch.channels["pv"].history
    assert orch.channels["heat_pump"].history == []


def test_subthreshold_current_pv_still_falls_back_to_heat_pump():
    orch = HeatSourceChannelOrchestrator()

    orch.route_learning(
        0.5,
        make_context(
            heat_pump_active=False,
            thermal_power=0.0,
            delta_t=0.0,
            pv_power=450.0,
            pv_power_current=450.0,
        ),
    )

    assert orch.channels["pv"].history == []
    assert len(orch.channels["heat_pump"].history) == 1


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

    for _ in range(_get_min_records_for_learning() + 1):
        orch.route_learning(1.0, make_context(**kwargs))

    updated_value = channel.get_learnable_parameters()[parameter_name]
    assert updated_value != pytest.approx(initial_value)


def test_heat_pump_self_learning_updates_hp_owned_model_parameters():
    orch = HeatSourceChannelOrchestrator()
    heat_pump = orch.channels["heat_pump"]
    initial = {
        key: heat_pump.get_learnable_parameters()[key]
        for key in (
            "thermal_time_constant",
            "heat_loss_coefficient",
            "delta_t_floor",
        )
    }

    context = make_context(
        heat_pump_active=True,
        delta_t=4.5,
        thermal_power=2.5,
    )
    for _ in range(_get_min_records_for_learning() + 1):
        orch.route_learning(1.0, context)

    updated = heat_pump.get_learnable_parameters()
    assert updated["thermal_time_constant"] != pytest.approx(
        initial["thermal_time_constant"]
    )
    assert updated["heat_loss_coefficient"] != pytest.approx(
        initial["heat_loss_coefficient"]
    )
    assert updated["delta_t_floor"] > initial["delta_t_floor"]


def test_solar_channel_self_learning_updates_retained_adaptive_parameters():
    orch = HeatSourceChannelOrchestrator()
    solar = orch.channels["pv"]
    initial_lag = solar.solar_lag_minutes
    initial_cloud_exponent = solar.cloud_factor_exponent

    context = make_context(
        pv_power=2000,
        avg_cloud_cover=90.0,
        pv_power_history=[300.0, 500.0, 700.0, 900.0],
    )
    for _ in range(_get_min_records_for_learning() + 1):
        orch.route_learning(1.0, context)

    assert solar.solar_lag_minutes != pytest.approx(initial_lag)
    assert solar.cloud_factor_exponent != pytest.approx(initial_cloud_exponent)


def test_no_source_active_routes_learning_to_heat_pump_channel():
    orch = HeatSourceChannelOrchestrator()
    initial = orch.channels["heat_pump"].get_learnable_parameters().copy()

    n = _get_min_records_for_learning() + 1
    for _ in range(n):
        orch.route_learning(1.0, make_context())

    assert len(orch.channels["heat_pump"].history) == n
    assert orch.channels["pv"].history == []
    assert orch.channels["fireplace"].history == []
    assert orch.channels["tv"].history == []
    assert orch.channels["heat_pump"].get_learnable_parameters() != initial


def test_route_learning_returns_update_event_and_logs_active_channel(caplog):
    orch = HeatSourceChannelOrchestrator()

    with caplog.at_level("INFO"):
        update_events = []
        for _ in range(_get_min_records_for_learning() + 1):
            update_events = orch.route_learning(1.0, make_context(fireplace_on=1))

    assert update_events
    assert update_events[-1]["record_type"] == "channel_update"
    assert update_events[-1]["channel"] == "fireplace"
    assert "fp_heat_output_kw" in update_events[-1]["changes"]
    assert any(
        "🔥 fireplace parameter update:" in record.message
        for record in caplog.records
    )
    assert not any(
        "☀️ pv parameter update:" in record.message
        for record in caplog.records
    )


def test_channel_state_includes_full_history_payload():
    orch = HeatSourceChannelOrchestrator()

    n = _get_min_records_for_learning() + 1
    for index in range(n):
        orch.route_learning(1.0 + index, make_context(fireplace_on=1))

    state = orch.get_channel_state()
    fireplace_state = state["fireplace"]

    assert fireplace_state["history_count"] == n
    assert len(fireplace_state["history"]) == n
    assert fireplace_state["history"][0]["error"] == pytest.approx(1.0)
    assert fireplace_state["history"][-1]["error"] == pytest.approx(float(n))
    assert fireplace_state["history"][-1]["context"]["fireplace_on"] == 1


def test_load_channel_state_restores_history_and_ignores_legacy_missing_history():
    orch = HeatSourceChannelOrchestrator()
    fireplace_history = [
        {
            "error": 0.5,
            "context": make_context(fireplace_on=1),
            "parameters": {"fp_heat_output_kw": 5.0},
        },
        {
            "error": 0.75,
            "context": make_context(fireplace_on=1),
            "parameters": {"fp_heat_output_kw": 5.1},
        },
    ]

    orch.load_channel_state(
        {
            "fireplace": {
                "parameters": {"fp_heat_output_kw": 6.2},
                "history_count": 99,
                "history": fireplace_history,
            },
            "pv": {
                "parameters": {"pv_heat_weight": 0.0042},
                "history_count": 7,
            },
        }
    )

    assert orch.channels["fireplace"].fp_heat_output_kw == pytest.approx(6.2)
    assert orch.channels["fireplace"].history == fireplace_history
    assert orch.channels["pv"].pv_heat_weight == pytest.approx(0.0042)
    assert orch.channels["pv"].history == []


@pytest.mark.parametrize(
    "param_name",
    [
        "thermal_time_constant",
        "heat_loss_coefficient",
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
        "thermal_time_constant": 6.2,
        "heat_loss_coefficient": 0.27,
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

    assert heat_pump.thermal_time_constant == pytest.approx(6.2)
    assert heat_pump.heat_loss_coefficient == pytest.approx(0.27)
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


# ---------------------------------------------------------------------------
#  Phase: Constant consolidation — aligned with canonical sources
# ---------------------------------------------------------------------------

def test_dead_zone_matches_physics_constants():
    """_get_learning_dead_zone() must return PhysicsConstants.LEARNING_DEAD_ZONE."""
    from src.thermal_constants import PhysicsConstants
    assert heat_source_channels._get_learning_dead_zone() == PhysicsConstants.LEARNING_DEAD_ZONE


def test_min_records_matches_recent_errors_window():
    """_get_min_records_for_learning() must return config.RECENT_ERRORS_WINDOW."""
    from src import config as cfg
    assert heat_source_channels._get_min_records_for_learning() == cfg.RECENT_ERRORS_WINDOW


def test_channel_learning_rate_matches_config():
    """_get_channel_learning_rate() must return config.ADAPTIVE_LEARNING_RATE."""
    from src import config as cfg
    assert heat_source_channels._get_channel_learning_rate() == cfg.ADAPTIVE_LEARNING_RATE


# ---------------------------------------------------------------------------
#  Phase: Routing observability — every route_learning() call logged
# ---------------------------------------------------------------------------

def test_route_learning_logs_routing_summary_for_heat_pump_only(caplog):
    """Heat-pump-only routing must emit a 'Channel routing' INFO line."""
    orch = HeatSourceChannelOrchestrator()
    with caplog.at_level("INFO"):
        orch.route_learning(0.5, make_context(heat_pump_active=True))
    assert any(
        "Channel routing" in r.message and "heat_pump" in r.message
        for r in caplog.records
    ), "Expected 'Channel routing' INFO for heat_pump"


def test_route_learning_logs_routing_summary_for_pv(caplog):
    """PV routing must emit a 'Channel routing' INFO line with 'pv'."""
    orch = HeatSourceChannelOrchestrator()
    with caplog.at_level("INFO"):
        orch.route_learning(0.5, make_context(pv_power=1200))
    assert any(
        "Channel routing" in r.message and "pv" in r.message
        for r in caplog.records
    ), "Expected 'Channel routing' INFO for pv"


def test_route_learning_logs_routing_summary_for_multiple_channels(caplog):
    """Multiple external sources produce a combined routing summary."""
    orch = HeatSourceChannelOrchestrator()
    ctx = make_context(pv_power=1200, fireplace_on=1)
    with caplog.at_level("INFO"):
        orch.route_learning(0.5, ctx)
    routing_lines = [
        r.message for r in caplog.records if "Channel routing" in r.message
    ]
    assert routing_lines, "Expected at least one 'Channel routing' INFO line"
    assert "pv" in routing_lines[0]
    assert "fireplace" in routing_lines[0]


def test_no_delta_log_emitted_when_params_unchanged(caplog):
    """When a channel receives an observation but params don't change
    (dead zone), a DEBUG line 'no parameter delta' must be emitted."""
    orch = HeatSourceChannelOrchestrator()
    with caplog.at_level("DEBUG"):
        orch.route_learning(0.001, make_context(heat_pump_active=True))
    assert any(
        "no parameter delta" in r.message for r in caplog.records
    ), "Expected 'no parameter delta' DEBUG when error is tiny"


def test_routing_summary_marks_changed_channels_with_asterisk(caplog):
    """Channels with actual parameter changes should be marked with '*'."""
    orch = HeatSourceChannelOrchestrator()
    ctx = make_context(fireplace_on=1)
    # Feed enough records to trigger a parameter change
    min_records = heat_source_channels._get_min_records_for_learning()
    with caplog.at_level("INFO"):
        for _ in range(min_records + 1):
            orch.route_learning(1.0, ctx)
    routing_lines = [
        r.message for r in caplog.records if "Channel routing" in r.message
    ]
    # At least the last routing line should show fireplace* (changed)
    assert any("*" in line for line in routing_lines), (
        "Expected '*' marker for channels that changed parameters"
    )


# ------------------------------------------------------------------
# Dead-zone gating on raw error
# ------------------------------------------------------------------

def test_dead_zone_gates_on_raw_error_not_attributed(
    mixed_source_attribution_enabled,
):
    """PV should learn when the *raw* prediction error exceeds the dead
    zone, even if the attributed slice is tiny."""
    orch = HeatSourceChannelOrchestrator()
    n = _get_min_records_for_learning() + 1
    raw_error = 0.05  # well above dead zone (0.01)
    ctx = make_context(heat_pump_active=True, pv_power=1000)
    for _ in range(n):
        orch.route_learning(raw_error, ctx)

    pv = orch.channels["pv"]
    # After enough observations with raw error > dead zone,
    # PV parameters should have shifted.
    initial_weight = heat_source_channels._get_parameter_default(
        "pv_heat_weight", 0.002
    )
    assert pv.pv_heat_weight != pytest.approx(initial_weight, abs=1e-9), (
        "PV pv_heat_weight should have changed when raw error exceeds dead zone"
    )


def test_pv_does_not_learn_when_raw_error_below_dead_zone(
    mixed_source_attribution_enabled,
):
    """When even raw error is below dead zone, PV should not learn."""
    orch = HeatSourceChannelOrchestrator()
    n = _get_min_records_for_learning() + 1
    raw_error = 0.005  # below dead zone (0.01)
    ctx = make_context(heat_pump_active=True, pv_power=1000)
    for _ in range(n):
        orch.route_learning(raw_error, ctx)

    pv = orch.channels["pv"]
    initial_weight = heat_source_channels._get_parameter_default(
        "pv_heat_weight", 0.002
    )
    assert pv.pv_heat_weight == pytest.approx(initial_weight, abs=1e-9), (
        "PV should NOT learn when raw error is below dead zone"
    )


def test_cloud_factor_respects_config_gate(monkeypatch):
    """SolarChannel.estimate_heat_contribution should return the
    un-discounted value when CLOUD_COVER_CORRECTION_ENABLED=false."""
    monkeypatch.setattr(
        heat_source_channels.config,
        "CLOUD_COVER_CORRECTION_ENABLED",
        False,
    )
    solar = SolarChannel()
    ctx = make_context(pv_power=2000, avg_cloud_cover=80.0)
    heat = solar.estimate_heat_contribution(ctx)
    # Without cloud factor, heat = pv_power * pv_heat_weight * 1.0
    expected = 2000.0 * solar.pv_heat_weight
    assert heat == pytest.approx(expected, abs=1e-6), (
        f"Cloud correction disabled but heat={heat}, expected={expected}"
    )


def test_cloud_factor_applied_when_enabled(monkeypatch):
    """When cloud correction enabled, estimate_heat_contribution should
    discount by cloud cover."""
    monkeypatch.setattr(
        heat_source_channels.config,
        "CLOUD_COVER_CORRECTION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        heat_source_channels.config,
        "CLOUD_CORRECTION_MIN_FACTOR",
        0.1,
        raising=False,
    )
    solar = SolarChannel()
    ctx_clear = make_context(pv_power=2000, avg_cloud_cover=0.0)
    ctx_cloudy = make_context(pv_power=2000, avg_cloud_cover=80.0)
    heat_clear = solar.estimate_heat_contribution(ctx_clear)
    heat_cloudy = solar.estimate_heat_contribution(ctx_cloudy)
    assert heat_clear > heat_cloudy, (
        "Clear sky should produce more PV heat than overcast"
    )


# ---------------------------------------------------------------------------
# PV routing: max(current, smoothed) for activity threshold
# ---------------------------------------------------------------------------

def test_pv_routing_smoothed_above_threshold(mixed_source_attribution_enabled):
    """When pv_power_current < threshold but pv_power (smoothed) > threshold,
    PV should still be considered active because solar thermal lag means
    heat is still affecting the room."""
    orch = HeatSourceChannelOrchestrator()
    ctx = make_context(
        pv_power=1300,              # smoothed: above 500 W threshold
        pv_power_current=200,       # current: below 500 W threshold
        heat_pump_active=False,
    )
    min_records = _get_min_records_for_learning()
    for ch in orch.channels.values():
        ch.history = [{"error": 0.05, "context": ctx}] * min_records

    events = orch.route_learning(0.05, ctx)
    learned = {e["channel"] for e in events}
    assert "pv" in learned, (
        f"PV smoothed=1300W > 500W threshold, PV should learn. Got: {learned}"
    )


def test_pv_routing_both_below_threshold(mixed_source_attribution_enabled):
    """When both pv_power_current and pv_power are below threshold,
    PV should NOT be active."""
    orch = HeatSourceChannelOrchestrator()
    ctx = make_context(
        pv_power=200,               # smoothed: below 500 W
        pv_power_current=100,       # current: below 500 W
        heat_pump_active=False,
    )
    min_records = _get_min_records_for_learning()
    for ch in orch.channels.values():
        ch.history = [{"error": 0.05, "context": ctx}] * min_records

    events = orch.route_learning(0.05, ctx)
    learned = {e["channel"] for e in events}
    assert "pv" not in learned, (
        f"Both PV values below threshold, PV should NOT learn. Got: {learned}"
    )


def test_pv_routing_current_above_smoothed_below(mixed_source_attribution_enabled):
    """When pv_power_current > threshold but smoothed < threshold,
    PV should still be active (max of the two)."""
    orch = HeatSourceChannelOrchestrator()
    ctx = make_context(
        pv_power=300,               # smoothed: below 500 W
        pv_power_current=800,       # current: above 500 W
        heat_pump_active=False,
    )
    min_records = _get_min_records_for_learning()
    for ch in orch.channels.values():
        ch.history = [{"error": 0.05, "context": ctx}] * min_records

    events = orch.route_learning(0.05, ctx)
    learned = {e["channel"] for e in events}
    assert "pv" in learned, (
        f"PV current=800W > 500W threshold, PV should learn. Got: {learned}"
    )
