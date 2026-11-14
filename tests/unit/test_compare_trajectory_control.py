"""Offline control comparison: no Home Assistant or production-state writes."""

import math
from unittest.mock import patch

import pytest

from validation.compare_trajectory_control import (
    ReplayCycle,
    compare_cycles,
    comfort_metrics,
    optimize_trajectory,
    parse_log,
)


def test_comfort_cost_is_symmetric_and_time_weighted():
    result = comfort_metrics([22.2, 22.8], 22.5, 0.1, 0.5)
    assert result["squared_violation_k2h"] == pytest.approx(0.04)
    assert result["undershoot_kh"] == pytest.approx(0.1)
    assert result["overshoot_kh"] == pytest.approx(0.1)
    assert result["hours_below_band"] == 0.5
    assert result["hours_above_band"] == 0.5


def test_optimizer_scores_whole_trajectory_not_just_endpoint():
    trajectories = {24.0: [21.0, 24.0, 22.5], 25.0: [22.5] * 3}
    result = optimize_trajectory(
        trajectories.__getitem__, lower=24.0, upper=25.0, baseline=24.0,
        current_indoor=22.5, target=22.5, comfort_band=0.1,
        step_hours=1.0, resolution=1.0,
    )
    assert result["outlet"] == 25.0
    assert result["squared_violation_k2h"] == 0.0


@pytest.mark.parametrize(
    ("current", "trajectory", "expected"),
    [(23.2, [23.4] * 4, 24.0), (22.0, [22.0] * 4, 25.3),
     (22.5, [22.5] * 4, 25.3)],
)
def test_flat_response_ties_use_lower_effort_only_during_surplus(
    current, trajectory, expected,
):
    result = optimize_trajectory(
        lambda outlet: trajectory, lower=24.0, upper=26.5, baseline=25.3,
        current_indoor=current, target=22.5, comfort_band=0.1,
        step_hours=1.0,
    )
    assert result["outlet"] == expected


def test_undershoot_uses_lowest_movement_that_enters_comfort_band():
    result = optimize_trajectory(
        lambda outlet: [22.0 + 0.3 * (outlet - 24.0)] * 4,
        lower=24.0, upper=27.0, baseline=24.0,
        current_indoor=22.0, target=22.5, comfort_band=0.1, step_hours=1.0,
    )
    assert result["outlet"] == 25.4
    assert result["squared_violation_k2h"] == 0.0


@pytest.mark.parametrize("trajectory", [[], [math.nan], [math.inf]])
def test_invalid_trajectories_fail_explicitly(trajectory):
    with pytest.raises(ValueError, match="trajectory"):
        comfort_metrics(trajectory, 22.5, 0.1, 1.0)


def _cycle():
    return ReplayCycle(
        timestamp="2026-11-14 20:07:07",
        features={
            "indoor_temp_lag_30m": 23.255, "target_temp": 22.6,
            "outdoor_temp": -0.934375, "inlet_temp": 25.0,
            "outlet_temp": 26.06, "delta_t": 1.06,
            "indoor_temp_delta_60m": 0.035, "pv_now": 0.0,
            "pv_power_history": [0.0] * 18,
            "fireplace_on": 1.0, "tv_on": 0.0,
            "living_room_temp": 24.0,
        },
        parameters={
            "thermal_time_constant": 6.9750585827282565,
            "heat_loss_coefficient": 0.12760530623452535,
            "outlet_effectiveness": 0.7406656828956533,
            "slab_time_constant_hours": 4.012589387530949,
            "delta_t_floor": 2.6805069879163352,
            "solar_lag_minutes": 10.218505876219833,
            "pv_heat_weight": 0.0009835060712585794,
            "fireplace_heat_weight": 2.517588801831192,
            "fp_heat_output_kw": 2.517588801831192,
            "tv_heat_weight": 0.6088893251115983,
            "fp_decay_time_constant": 3.8811454929255564,
            "room_spread_delay_minutes": 18.0,
            "cloud_factor_exponent": 1.0,
            "solar_decay_tau_hours": 1.0015581859251772,
        },
        parameter_source="learning",
        outdoor_forecast=[-0.934375, -0.93, -1.23, -1.53, -2.13],
        pv_forecast=[0.0] * 5,
        actual_indoor=23.23, target_adjusted=22.5,
        baseline=25.3, lower=24.0, upper=35.0,
        logged_raw=26.54, logged_published=26.1,
        hp_status="idle", hp_window={"delay_h": 1e6, "runtime_h": None},
        hold_remaining=0, price_level="expensive",
        logged_trajectory=[23.28] + [23.44] * 23,
        logged_trajectory_outlet=26.5,
    )


def _log_cycle(cycle):
    prefix = f"{cycle.timestamp},000 - DEBUG - "
    lines = [
        "CYCLE 1 START: 20:07:07",
        f"Setting HA state for sensor.ml_heating_learning: "
        f"{ {'state': '5', 'attributes': cycle.parameters}!r}",
        "HP start estimate (heating): status=idle method=degree_minutes "
        "start_in=None min runtime=None min (attribute) confidence=0.60",
        "Using search bounds: outlet_min=24.0°C, outlet_max=35.0°C",
        "Trajectory prediction using forecast arrays: "
        f"outdoor={cycle.outdoor_forecast!r}, PV={cycle.pv_forecast!r}",
        "Prediction: Current 23.23°C → Target 22.6°C (adjusted 22.5°C) "
        "| Calculated outlet: 26.5°C",
        "Gradual control: baseline=25.3°C, limit=1.0 K (calm)",
        f"Setting HA state for sensor.ml_heating_state: "
        f"{ {'state': '0', 'attributes': {'suggested_temp': 26.54, 'final_temp': 26.1}}!r}",
        f"Setting HA state for sensor.ml_heating_features: "
        f"{ {'state': '24', 'attributes': cycle.features}!r}",
        f"Setting HA state for sensor.ml_heating_price_level: "
        f"{ {'attributes': {'price_level': cycle.price_level}}!r}",
        "CYCLE 1 END: 20:07:08",
    ]
    return "\n".join(prefix + line for line in lines)


def test_parser_never_carries_missing_features_into_next_cycle():
    text = "partial log before first cycle\n" + _log_cycle(_cycle())
    text += "\n2026-11-14 20:17:07,000 - DEBUG - CYCLE 2 START: 20:17:07\n"
    text += "2026-11-14 20:17:08,000 - DEBUG - CYCLE 2 END: 20:17:08\n"
    cycles, excluded = parse_log(text)
    assert len(cycles) == 1
    assert cycles[0].logged_raw == 26.54
    assert cycles[0].hp_window == {"delay_h": 1e6, "runtime_h": None}
    assert len(excluded) == 2
    assert "features" in excluded[-1]


def test_parser_uses_decision_time_window_not_later_forecast_sensor_read():
    text = _log_cycle(_cycle())
    text += (
        "\n2026-11-14 20:08:00,000 - DEBUG - HP start estimate (heating): "
        "status=running method=degree_minutes start_in=0.0 min runtime=30.0 min"
    )
    cycles, excluded = parse_log(text)
    assert excluded == []
    assert cycles[0].hp_status == "idle"
    assert cycles[0].hp_window["delay_h"] == 1e6


def test_offline_2007_comparison_uses_same_limits_and_has_no_network_or_state_io(
    tmp_path, monkeypatch, caplog,
):
    from src import config

    monkeypatch.chdir(tmp_path)
    before = config.CLAMP_MIN_ABS
    with patch("requests.sessions.Session.request", side_effect=AssertionError("network")):
        rows = compare_cycles([_cycle()], comfort_band=0.1)

    assert config.CLAMP_MIN_ABS == before
    assert not list(tmp_path.iterdir())
    assert {row["scenario"] for row in rows} == {
        "idle_no_start", "idle_immediate_start",
    }
    no_start = next(row for row in rows if row["scenario"] == "idle_no_start")
    assert no_start["current_raw"] == 24.0
    assert no_start["alternative_raw"] == 24.0
    assert no_start["current_publishable"] == 24.5
    assert no_start["alternative_publishable"] == 24.5
    assert no_start["logged_published"] == 26.1
    assert no_start["current_raw_squared_violation_k2h"] == pytest.approx(
        no_start["alternative_raw_squared_violation_k2h"]
    )
    assert all(row["current_publishable"] <= row["baseline"] for row in rows)
    assert all(row["alternative_publishable"] <= row["baseline"] for row in rows)
    assert not [record for record in caplog.records if record.levelname == "ERROR"]
