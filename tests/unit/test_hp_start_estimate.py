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
    assert hse.learning_weight(est("100"), 30) == 0.0
    assert hse.learning_weight(est("10"), 30) == pytest.approx(2 / 3)
    assert hse.learning_weight(est("unknown"), 30) == 0.0
    assert hse.learning_weight(est("5", status="unknown"), 30) is None


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


RUNTIME = "sensor.wp_laufzeit_heizen"


def test_runtime_sensor_takes_precedence_and_falls_back():
    cfg = _cfg(HP_START_ESTIMATE_HEATING_RUNTIME_ENTITY_ID=RUNTIME)
    states = _states("0", status="running")
    states[RUNTIME] = {"state": "90"}
    est = hse.read_hp_start_estimate(states, "heating", now=NOW, cfg=cfg)
    assert est.expected_runtime_min == 90 and est.runtime_source == "sensor"
    for bad in ("unavailable", "-5", "abc"):
        states[RUNTIME] = {"state": bad}
        est = hse.read_hp_start_estimate(states, "heating", now=NOW, cfg=cfg)
        assert est.expected_runtime_min == 250
        assert est.runtime_source == "attribute"
    est = hse.read_hp_start_estimate(
        {HEAT: states[HEAT]}, "heating", now=NOW, cfg=cfg
    )
    assert est.runtime_source == "attribute"


def test_valid_runtime_sensor_overrides_negative_attribute():
    cfg = _cfg(HP_START_ESTIMATE_HEATING_RUNTIME_ENTITY_ID=RUNTIME)
    states = _states("0", status="running", expected_runtime_min=-1)
    states[RUNTIME] = {"state": "90"}

    estimate = hse.read_hp_start_estimate(
        states, "heating", now=NOW, cfg=cfg
    )

    assert estimate.expected_runtime_min == 90
    assert estimate.runtime_source == "sensor"


def test_circuit_breaker_locks_again_after_pause():
    bad_cfg = SimpleNamespace(
        HP_START_ESTIMATE_ENABLED=True,
        HP_START_ESTIMATE_HEATING_ENTITY_ID=HEAT,
        HP_START_ESTIMATE_MAX_AGE_MIN="x",
    )
    states = _states()
    later = NOW
    for round_ in range(2):
        for _ in range(5):
            assert hse.read_hp_start_estimate(
                states, "heating", now=later, cfg=bad_cfg
            ) is None
        assert hse.read_hp_start_estimate(
            states, "heating", now=later, cfg=bad_cfg
        ) is None
        assert hse.last_fallback_reason("heating") == "circuit_breaker"
        later = later + timedelta(seconds=601)


def test_candidate_window_shortens_delay_for_higher_outlet_only():
    est = hse.read_hp_start_estimate(
        _states("60", required_supply_c=30.0, supply_actual_c=28.0),
        "heating", now=NOW, cfg=_cfg(),
    )
    window = hse.trajectory_window(est)
    same = hse.candidate_window(
        est, window, candidate_outlet=30.0, reference_outlet=30.0, mode="heating"
    )
    lower = hse.candidate_window(
        est, window, candidate_outlet=28.0, reference_outlet=30.0, mode="heating"
    )
    higher = hse.candidate_window(
        est, window, candidate_outlet=32.0, reference_outlet=30.0, mode="heating"
    )
    assert same == lower == window
    assert higher["delay_h"] == pytest.approx(window["delay_h"] * 2 / 4)
    assert hse.candidate_window(
        None, window, candidate_outlet=32.0, reference_outlet=30.0, mode="heating"
    ) == window


@pytest.mark.parametrize(
    "outlet,required,reference,gm,expected_delay_h",
    [
        (24.0, 24.5, 24.5, -100.0, 1e6),
        (25.0, 24.5, 24.5, -100.0, 1e6),
        (25.04, 24.5, 24.5, -100.0, 1e6),
        (26.0, 24.5, 24.5, -100.0, 100 / 60),
        (27.0, 24.5, 24.5, -100.0, 50 / 60),
        (26.0, 24.5, 24.5, 1000.0, 20.0),
        (24.0, 26.5, 24.5, -100.0, 100 / 60),
        (26.0, 24.5, 24.5, -200.0, 0.0),
        (26.0, 24.5, 24.5, -250.0, 0.0),
    ],
)
def test_idle_candidate_derives_start_from_degree_minutes(
    outlet, required, reference, gm, expected_delay_h,
):
    estimate = hse.read_hp_start_estimate(
        _states(
            "unknown", expected_runtime_min=None,
            required_supply_c=required, supply_actual_c=25.0,
            gm_now=gm, gm_start=-200.0,
        ),
        "heating", now=NOW, cfg=_cfg(),
    )
    window = hse.trajectory_window(estimate)

    result = hse.candidate_window(
        estimate, window, candidate_outlet=outlet,
        reference_outlet=reference, mode="heating",
    )

    assert result["delay_h"] == pytest.approx(expected_delay_h)
    assert result["runtime_h"] is None
    assert window["delay_h"] == 1e6


@pytest.mark.parametrize("attribute,value", [
    ("gm_now", None),
    ("gm_now", float("nan")),
    ("gm_start", None),
    ("gm_start", float("inf")),
    ("supply_actual_c", "invalid"),
    ("required_supply_c", None),
])
def test_idle_candidate_preserves_window_with_invalid_attributes(
    attribute, value, caplog,
):
    attrs = dict(
        required_supply_c=24.5, supply_actual_c=25.0,
        gm_now=-100.0, gm_start=-200.0,
    )
    attrs[attribute] = value
    estimate = hse.read_hp_start_estimate(
        _states("unknown", expected_runtime_min=None, **attrs),
        "heating", now=NOW, cfg=_cfg(),
    )
    window = hse.trajectory_window(estimate)

    with caplog.at_level("DEBUG", logger=hse.LOGGER.name):
        result = hse.candidate_window(
            estimate, window, candidate_outlet=26.0,
            reference_outlet=24.5, mode="heating",
        )

    assert result == window
    assert "Idle candidate start unavailable" in caplog.text


@pytest.mark.parametrize("mode,method,status", [
    ("cooling", "degree_minutes", "idle"),
    ("heating", "hysteresis", "idle"),
    ("heating", "degree_minutes", "blocked"),
    ("heating", "degree_minutes", "unknown"),
])
def test_idle_candidate_does_not_change_other_modes_methods_or_statuses(
    mode, method, status,
):
    estimate = hse.read_hp_start_estimate(
        _states(
            "unknown", method=method, status=status,
            required_supply_c=24.5, supply_actual_c=25.0,
            gm_now=-100.0, gm_start=-200.0,
        ),
        "heating", now=NOW, cfg=_cfg(),
    )
    window = hse.trajectory_window(estimate)

    assert hse.candidate_window(
        estimate, window, candidate_outlet=26.0,
        reference_outlet=24.5, mode=mode,
    ) == window


def test_idle_candidate_start_restores_trajectory_response_after_delay():
    from src.thermal_equilibrium_model import ThermalEquilibriumModel

    estimate = hse.read_hp_start_estimate(
        _states(
            "unknown", expected_runtime_min=None,
            required_supply_c=24.5, supply_actual_c=25.0,
            gm_now=-100.0, gm_start=-200.0,
        ),
        "heating", now=NOW, cfg=_cfg(),
    )
    window = hse.trajectory_window(estimate)
    candidate = hse.candidate_window(
        estimate, window, candidate_outlet=26.0,
        reference_outlet=24.5, mode="heating",
    )
    model = ThermalEquilibriumModel()
    kwargs = dict(
        current_indoor=23.23, target_indoor=22.5, outlet_temp=26.0,
        outdoor_temp=-0.93, time_horizon_hours=4, time_step_minutes=10,
        inlet_temp=25.0, delta_t_floor=2.68,
    )
    passive = model.predict_thermal_trajectory(
        **kwargs, hp_start_delay_h=window["delay_h"],
        hp_runtime_h=window["runtime_h"],
    )["trajectory"]
    started = model.predict_thermal_trajectory(
        **kwargs, hp_start_delay_h=candidate["delay_h"],
        hp_runtime_h=candidate["runtime_h"],
    )["trajectory"]

    assert started[:10] == pytest.approx(passive[:10])
    assert started[-1] > passive[-1]


def test_running_candidate_extends_runtime_with_higher_outlet():
    est = hse.read_hp_start_estimate(
        _states("0", status="running", expected_runtime_min=60,
                required_supply_c=28.0, supply_actual_c=30.0),
        "heating", now=NOW, cfg=_cfg(),
    )
    window = hse.trajectory_window(est)
    same = hse.candidate_window(
        est, window, candidate_outlet=28.0, reference_outlet=28.0, mode="heating"
    )
    higher = hse.candidate_window(
        est, window, candidate_outlet=29.0, reference_outlet=28.0, mode="heating"
    )
    unbounded = hse.candidate_window(
        est, window, candidate_outlet=31.0, reference_outlet=28.0, mode="heating"
    )
    assert same == window
    assert higher["runtime_h"] == pytest.approx(window["runtime_h"] * 2)
    assert unbounded["runtime_h"] is None


def test_running_candidate_shortens_runtime_with_lower_outlet():
    est = hse.read_hp_start_estimate(
        _states("0", status="running", expected_runtime_min=60,
                required_supply_c=28.0, supply_actual_c=30.0),
        "heating", now=NOW, cfg=_cfg(),
    )
    window = hse.trajectory_window(est)
    lower = hse.candidate_window(
        est, window, candidate_outlet=27.0, reference_outlet=28.0,
        mode="heating",
    )

    assert lower["runtime_h"] == pytest.approx(40 / 60)
