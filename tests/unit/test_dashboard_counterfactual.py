import json
from datetime import date

import pandas as pd
import pytest

from dashboard.components import counterfactual


def _record(timestamp, state):
    return {"last_changed": timestamp.isoformat(), "state": state}


def test_history_builder_aligns_sensor_events_and_marks_invalid_states(monkeypatch):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    names = (
        "indoor_temp",
        "outdoor_temp",
        "target_heating",
        "target_cooling",
        "outlet_temp",
        "inlet_temp",
        "flow_rate",
        "electrical_power_w",
        "pv_power",
        "fireplace_on",
        "tv_on",
    )
    entity_ids = [f"sensor.{name}" for name in names] + ["climate.heating"]
    monkeypatch.setattr(
        counterfactual,
        "_NUMERIC_ENTITIES",
        dict(zip(names, entity_ids)),
    )
    monkeypatch.setattr(counterfactual.config, "HEATING_STATUS_ENTITY_ID", "climate.heating")

    values = {
        "indoor_temp": "22.4",
        "outdoor_temp": "5.0",
        "target_heating": "21.0",
        "target_cooling": "25.0",
        "outlet_temp": "35.0",
        "inlet_temp": "30.0",
        "flow_rate": "60.0",
        "electrical_power_w": "600.0",
        "pv_power": "1000.0",
        "fireplace_on": "off",
        "tv_on": "on",
    }
    raw = []
    for name in names:
        events = [_record(start, values[name])]
        if name == "indoor_temp":
            events.append(_record(start + pd.Timedelta(minutes=5), "unknown"))
            events.append(_record(start + pd.Timedelta(minutes=10), "22.4"))
        raw.append(events)
    raw.append(
        [
            _record(start, "heat"),
            _record(start + pd.Timedelta(minutes=5), "cool"),
        ]
    )

    frame = counterfactual._build_history_frame(
        raw,
        entity_ids,
        start.to_pydatetime(),
        (start + pd.Timedelta(minutes=10)).to_pydatetime(),
    )

    assert frame["indoor_temp"].iloc[0] == 22.4
    assert pd.isna(frame["indoor_temp"].iloc[1])
    assert frame["indoor_temp"].iloc[2] == 22.4
    assert frame["target_temp"].tolist() == [21.0, 25.0, 25.0]
    assert frame["thermal_power_kw"].iloc[0] == pytest.approx(
        5.0 * counterfactual.config.SPECIFIC_HEAT_CAPACITY
    )


def test_state_read_applies_adjustments_and_source_channel_weights(tmp_path):
    path = tmp_path / "thermal.json"
    path.write_text(
        json.dumps(
            {
                "baseline_parameters": {
                    "heat_loss_coefficient": 0.1,
                    "thermal_time_constant": 4.0,
                    "pv_heat_weight": 0.2,
                },
                "learning_state": {
                    "parameter_adjustments": {
                        "heat_loss_coefficient_delta": 0.01,
                        "pv_heat_weight_delta": 0.05,
                    },
                    "heat_source_channels": {
                        "pv": {"parameters": {"pv_heat_weight": 0.4}}
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    parameters = counterfactual._read_state(str(path))

    assert parameters["heat_loss_coefficient"] == 0.11
    assert parameters["thermal_time_constant"] == 4.0
    assert parameters["pv_heat_weight"] == 0.4


def test_make_model_uses_model_weight_defaults_when_state_omits_them(monkeypatch):
    class Model:
        def __init__(self):
            self.external_source_weights = {
                "pv": 0.2,
                "fireplace": 0.3,
                "tv": 0.4,
            }

    monkeypatch.setattr(counterfactual, "ThermalEquilibriumModel", Model)

    model = counterfactual._make_model({})

    assert model.external_source_weights == {
        "pv": 0.2,
        "fireplace": 0.3,
        "tv": 0.4,
    }


def test_external_gain_uses_model_physics_and_pv_lag_history(monkeypatch):
    class Model:
        heat_loss_coefficient = 0.2

        def predict_equilibrium_temperature(self, **kwargs):
            self.inputs = kwargs
            return 8.0

    model = Model()
    monkeypatch.setattr(counterfactual.config, "HISTORY_STEP_MINUTES", 10)

    gain = counterfactual._external_gain_kw(
        model=model,
        pv_values=[0.0, 1.0, 2.0, 3.0, 4.0],
        position=4,
        outlet_temp=35.0,
        outdoor_temp=5.0,
        indoor_temp=22.0,
        fireplace_on=0.0,
        tv_on=1.0,
    )

    assert gain == pytest.approx(0.6)
    assert model.inputs["pv_power"] == [0.0, 2.0, 4.0]
    assert model.inputs["thermal_power"] == 0.0
    assert model.inputs["_suppress_logging"]


def _run_history(start="2026-01-01T23:50:00Z", periods=5):
    return pd.DataFrame(
        {
            "_time": pd.date_range(start, periods=periods, freq="5min"),
            "indoor_temp": [22.6] * periods,
            "target_temp": [22.6] * periods,
            "target_heating": [22.6] * periods,
            "target_cooling": [25.0] * periods,
            "outdoor_temp": [5.0] * periods,
            "outlet_temp": [35.0] * periods,
            "inlet_temp": [30.0] * periods,
            "flow_rate": [60.0] * periods,
            "electrical_power_w": [600.0] * periods,
            "pv_power": [0.0] * periods,
            "fireplace_on": [0.0] * periods,
            "tv_on": [0.0] * periods,
            "thermal_power_kw": [1.8] * periods,
            "mode": ["heating"] * periods,
        }
    )


def _mock_counterfactual_dependencies(monkeypatch, history, parameters=None):
    if parameters is None:
        parameters = {
            "heat_loss_coefficient": 0.1,
            "thermal_time_constant": 4.0,
        }
    monkeypatch.setattr(
        counterfactual, "_fetch_history", lambda *_args: history.copy()
    )
    monkeypatch.setattr(counterfactual, "_find_state_file", lambda: "heating.json")
    monkeypatch.setattr(
        counterfactual, "_find_cooling_state_file", lambda: None
    )
    monkeypatch.setattr(
        counterfactual, "_read_state", lambda _path: parameters.copy()
    )
    monkeypatch.setattr(counterfactual, "_make_model", lambda _parameters: object())
    monkeypatch.setattr(counterfactual, "_external_gain_kw", lambda **_kwargs: 0.0)


def test_run_counterfactual_trims_warmup_and_resums_selected_period(monkeypatch):
    _mock_counterfactual_dependencies(monkeypatch, _run_history())

    result = counterfactual._run_counterfactual(date(2026, 1, 2), date(2026, 1, 2), 3.0)

    assert result["complete"]
    assert result["sample_count"] == 3
    assert result["intervals"]["_time"].min() == pd.Timestamp("2026-01-02T00:00:00Z")
    assert result["actual_thermal_kwh"] == pytest.approx(1.8 * (15.0 / 60.0))
    assert result["actual_thermal_kwh"] == result["intervals"][
        "actual_thermal_energy_kwh"
    ].sum()


def test_run_counterfactual_reports_empty_history(monkeypatch):
    _mock_counterfactual_dependencies(monkeypatch, pd.DataFrame())

    result = counterfactual._run_counterfactual(
        date(2026, 1, 2), date(2026, 1, 2), 3.0
    )

    assert not result["complete"]
    assert "no usable history" in result["reason"]


@pytest.mark.parametrize("invalid_history", ["missing_column", "missing_value"])
def test_run_counterfactual_rejects_incomplete_synchronized_history(
    monkeypatch, invalid_history
):
    history = _run_history()
    if invalid_history == "missing_column":
        history.drop(columns=["pv_power"], inplace=True)
    else:
        history.loc[2, "outdoor_temp"] = float("nan")
    _mock_counterfactual_dependencies(monkeypatch, history)

    result = counterfactual._run_counterfactual(
        date(2026, 1, 2), date(2026, 1, 2), 3.0
    )

    assert not result["complete"]
    assert "missing" in result["reason"].lower() or "incomplete" in result["reason"].lower()


def test_run_counterfactual_reports_heating_parameter_when_unavailable(monkeypatch):
    _mock_counterfactual_dependencies(monkeypatch, _run_history(), parameters={})

    result = counterfactual._run_counterfactual(
        date(2026, 1, 2), date(2026, 1, 2), 3.0
    )

    assert not result["complete"]
    assert "Heating model parameter" in result["reason"]


def test_run_counterfactual_reports_empty_selected_period(monkeypatch):
    history = _run_history(start="2026-01-01T22:00:00Z", periods=5)
    _mock_counterfactual_dependencies(monkeypatch, history)

    result = counterfactual._run_counterfactual(
        date(2026, 1, 2), date(2026, 1, 2), 3.0
    )

    assert not result["complete"]
    assert "selected date range" in result["reason"]
