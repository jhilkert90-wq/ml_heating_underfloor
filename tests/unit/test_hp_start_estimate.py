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
