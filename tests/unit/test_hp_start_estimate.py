from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src import hp_start_estimate as hse

NOW = datetime(2026, 11, 12, 10, 0, tzinfo=timezone.utc)
HEAT = "sensor.wp_minuten_bis_start_heizen"


def _cfg(**kw):
    base = dict(
        HP_START_ESTIMATE_ENABLED=True,
        HP_START_ESTIMATE_HEATING_ENTITY_ID=HEAT,
        HP_START_ESTIMATE_COOLING_ENTITY_ID="",
        HP_START_ESTIMATE_MAX_AGE_MIN=5.0,
        HP_START_ESTIMATE_MIN_CONFIDENCE=0.3,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _states(state="100", **attrs):
    a = dict(
        status="idle", mode="heating", method="degree_minutes",
        expected_runtime_min=250, confidence=0.8,
        updated_at=(NOW - timedelta(minutes=1)).isoformat(),
    )
    a.update(attrs)
    return {HEAT: {"state": state, "attributes": a}}


@pytest.fixture(autouse=True)
def _reset():
    hse._reset_for_tests()


def test_valid_estimate():
    est = hse.read_hp_start_estimate(_states(), "heating", now=NOW, cfg=_cfg())
    assert est.minutes_to_start == 100 and est.expected_runtime_min == 250
    assert est.known_method


def test_running_is_zero():
    est = hse.read_hp_start_estimate(
        _states("0", status="running"), "heating", now=NOW, cfg=_cfg()
    )
    assert est.minutes_to_start == 0


@pytest.mark.parametrize("states,cfg,reason", [
    (_states(), _cfg(HP_START_ESTIMATE_ENABLED=False), "disabled"),
    (_states(), _cfg(HP_START_ESTIMATE_HEATING_ENTITY_ID=""), "not_configured"),
    ({}, _cfg(), "missing"),
    (_states(mode="cooling"), _cfg(), "wrong_mode"),
    (_states(updated_at=(NOW - timedelta(minutes=30)).isoformat()), _cfg(), "stale"),
    (_states(confidence=0.1), _cfg(), "low_confidence"),
    (_states(status="blocked", confidence="x"), _cfg(), "low_confidence"),
    (_states("abc"), _cfg(), "invalid_state"),
    (_states("-5"), _cfg(), "invalid_state"),
    (_states("unknown", status="running"), _cfg(), "inconsistent"),
    (_states(status="weird"), _cfg(), "invalid_status"),
])
def test_fallback_reasons(states, cfg, reason):
    assert hse.read_hp_start_estimate(states, "heating", now=NOW, cfg=cfg) is None
    assert hse.last_fallback_reason("heating") == reason


def test_unknown_state_idle_means_no_start():
    est = hse.read_hp_start_estimate(
        _states("unknown"), "heating", now=NOW, cfg=_cfg()
    )
    assert est is not None and est.minutes_to_start is None


def test_cooling_independent_of_heating():
    assert hse.read_hp_start_estimate(
        _states(), "cooling", now=NOW, cfg=_cfg()
    ) is None
    assert hse.last_fallback_reason("cooling") == "not_configured"


def test_reader_errors_never_raise_and_trip_breaker():
    class Boom(dict):
        def get(self, *_):
            raise RuntimeError("boom")

    for _ in range(6):
        assert hse.read_hp_start_estimate(Boom(), "heating", now=NOW, cfg=_cfg()) is None
    assert hse.last_fallback_reason("heating") == "circuit_breaker"


def test_diagnostics_reports_source():
    diag = hse.diagnostics(_states(), now=NOW, cfg=_cfg())
    assert diag["heating"]["start_estimate_source"] == "sensor"
    assert diag["cooling"]["start_estimate_source"] == "fallback"


def _traj(model, **kw):
    return model.predict_thermal_trajectory(
        current_indoor=21.0, target_indoor=22.0, outlet_temp=35.0,
        outdoor_temp=5.0, time_horizon_hours=4, time_step_minutes=30,
        inlet_temp=24.0, delta_t_floor=2.0, **kw,
    )["trajectory"]


def test_trajectory_window_gates_hp_contribution():
    from src.thermal_equilibrium_model import ThermalEquilibriumModel

    model = ThermalEquilibriumModel()
    base = _traj(model)
    running = _traj(model, hp_start_delay_h=0.0, hp_runtime_h=None)
    never = _traj(model, hp_start_delay_h=1e6, hp_runtime_h=None)
    later = _traj(model, hp_start_delay_h=2.0, hp_runtime_h=None)
    assert running == pytest.approx(base)
    assert never[-1] < base[-1]
    assert never[0] <= later[0] + 1e-9 and later[-1] < base[-1]
    assert later[-1] > never[-1]


def test_trajectory_window_helper():
    est = hse.read_hp_start_estimate(_states(), "heating", now=NOW, cfg=_cfg())
    win = hse.trajectory_window(est)
    assert win["delay_h"] == pytest.approx(100 / 60) and win["runtime_h"] == pytest.approx(250 / 60)
    assert hse.trajectory_window(None) is None
    blocked = hse.read_hp_start_estimate(
        _states("unknown", status="blocked"), "heating", now=NOW, cfg=_cfg()
    )
    assert hse.trajectory_window(blocked) is None


def test_trajectory_window_preserves_zero_runtime():
    estimate = hse.read_hp_start_estimate(
        _states("0", status="running", expected_runtime_min=0),
        "heating",
        now=NOW,
        cfg=_cfg(),
    )

    assert hse.trajectory_window(estimate) == {
        "delay_h": 0.0,
        "runtime_h": 0.0,
    }


def test_learning_weight():
    def est(state, **kw):
        return hse.read_hp_start_estimate(_states(state, **kw), "heating", now=NOW, cfg=_cfg())

    assert hse.learning_weight(None, 30) is None
    assert hse.learning_weight(est("unknown", status="blocked"), 30) == 0.0
    assert hse.learning_weight(est("10", status="starting_soon"), 30) == pytest.approx(2 / 3)
    assert hse.learning_weight(est("0", status="running"), 30) == 1.0
    assert hse.learning_weight(est("100"), 30) == 1.0


def test_hp_channel_skips_blocked_cycle_and_scales_rate():
    from src.heat_source_channels import HeatPumpChannel

    def run(weight):
        ch = HeatPumpChannel()
        before = ch.outlet_effectiveness
        for _ in range(12):
            ch.record_learning(
                1.0,
                {"delta_t": 3.0, "climate_mode": "heating", "outdoor_temp": 5.0,
                 "hp_run_weight": weight},
            )
        return ch.outlet_effectiveness - before

    assert run(0.0) == 0.0
    assert abs(run(0.5)) <= abs(run(None)) + 1e-12


def test_cost_skip_allowed():
    def est(state, **kw):
        return hse.read_hp_start_estimate(_states(state, **kw), "heating", now=NOW, cfg=_cfg())

    assert hse.cost_skip_allowed(None, 120)
    assert hse.cost_skip_allowed(est("0", status="running"), 120)
    assert hse.cost_skip_allowed(est("60", status="starting_soon"), 120)
    assert not hse.cost_skip_allowed(est("300"), 120)
    assert not hse.cost_skip_allowed(est("unknown"), 120)
