import pandas as pd
import pytest

from src.counterfactual_replay import replay_target_hold


def _history(
    *,
    indoor=22.6,
    target=22.6,
    outdoor=5.0,
    external_gain=0.0,
    mode="heating",
    thermal_power=1.8,
    electrical_power=600.0,
):
    return pd.DataFrame(
        {
            "_time": pd.date_range(
                "2026-01-01T00:00:00Z", periods=4, freq="5min"
            ),
            "indoor_temp": [indoor] * 4,
            "target_temp": [target] * 4,
            "outdoor_temp": [outdoor] * 4,
            "external_gain_kw": [external_gain] * 4,
            "mode": [mode] * 4,
            "thermal_power_kw": [thermal_power] * 4,
            "electrical_power_w": [electrical_power] * 4,
        }
    )


def _replay(frame):
    return replay_target_hold(
        frame,
        heat_loss_coefficient=0.1,
        thermal_time_constant_hours=4.0,
        specific_heat_capacity=4.186,
        fallback_cop=3.0,
    )


def test_no_external_gains_replay_uses_historical_target_and_measured_cop():
    result = _replay(_history(target=22.6))

    assert result["complete"]
    assert result["intervals"]["target_temp"].eq(22.6).all()
    assert result["counterfactual_thermal_kwh"] > 0
    assert result["counterfactual_electrical_kwh"] == pytest.approx(
        result["counterfactual_thermal_kwh"] / 3.0
    )
    assert result["measured_cop_count"] == 4


def test_pv_external_gains_reduce_counterfactual_heating_energy():
    no_gain = _replay(_history(external_gain=0.0))
    with_pv = _replay(_history(external_gain=0.5))

    assert with_pv["counterfactual_thermal_kwh"] < no_gain["counterfactual_thermal_kwh"]


def test_initial_above_target_represents_stored_heat_without_heating():
    result = _replay(
        _history(indoor=23.6, target=22.6, outdoor=10.0, thermal_power=0.0)
    )

    assert result["complete"]
    assert result["intervals"]["counterfactual_thermal_power_kw"].iloc[0] == 0.0
    assert (
        result["intervals"]["counterfactual_indoor_temp"].iloc[0] > 22.6
    )
    assert (
        result["intervals"]["counterfactual_indoor_temp"].iloc[-1]
        < result["intervals"]["counterfactual_indoor_temp"].iloc[0]
    )


def test_missing_history_marks_result_incomplete_without_energy_totals():
    frame = _history()
    frame.loc[2, "outdoor_temp"] = float("nan")

    result = _replay(frame)

    assert not result["complete"]
    assert result["missing_count"] == 1
    assert "counterfactual_thermal_kwh" not in result


def test_cooling_mode_replay_estimates_cooling_energy():
    result = _replay(
        _history(
            indoor=24.0,
            target=24.0,
            outdoor=30.0,
            mode="cooling",
            thermal_power=-1.8,
        )
    )

    assert result["complete"]
    assert result["counterfactual_thermal_kwh"] > 0
    assert result["intervals"]["climate_mode"].eq("cooling").all()
    assert result["intervals"]["counterfactual_thermal_power_kw"].gt(0).all()


def test_missing_required_sensor_column_marks_result_incomplete():
    frame = _history().drop(columns=["electrical_power_w"])

    result = _replay(frame)

    assert not result["complete"]
    assert "electrical_power_w" in result["reason"]
