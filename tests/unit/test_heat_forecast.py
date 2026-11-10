import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.heat_forecast import (
    ForecastStateStore,
    HeatForecastEngine,
    compute_counter_delta,
    cop_calibration_energies,
    cop_curve_from_state,
    fit_energy_weighted_cop,
    forecast_hour_starts,
    hourly_electrical_forecast,
    infer_defrost_counter_inclusion,
    net_cop_energies,
    simulate_heating_energy,
    estimate_defrost_power_history,
    thermal_pv_gain_kw,
)


def test_counter_delta_handles_increments_resets_and_invalid_gaps():
    assert compute_counter_delta(10.0, 10.4, elapsed_hours=1.0) == pytest.approx(0.4)
    assert compute_counter_delta(100.0, 0.2, elapsed_hours=1.0) == pytest.approx(0.2)
    assert compute_counter_delta(10.0, 10.4, elapsed_hours=3.0) is None
    assert compute_counter_delta(10.0, float("nan"), elapsed_hours=1.0) is None
    assert compute_counter_delta(None, 1.0, elapsed_hours=1.0) is None


def test_ha_history_uses_record_entity_and_parses_binary_states():
    from src.heat_forecast import _fetch_ha_history

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    class HistoryHA:
        def get_history_bulk(self, _entities, _start, _stop):
            return [
                [
                    {
                        "entity_id": "sensor.second",
                        "state": "on",
                        "last_changed": start.isoformat(),
                    }
                ],
                [
                    {
                        "entity_id": "sensor.first",
                        "state": "off",
                        "last_changed": start.isoformat(),
                    }
                ],
            ]

    history = _fetch_ha_history(
        HistoryHA(),
        ["sensor.first", "sensor.second", "sensor.omitted"],
        start,
        start + timedelta(hours=1),
    )

    assert history == {
        "sensor.first": [(start, 0.0)],
        "sensor.second": [(start, 1.0)],
        "sensor.omitted": [],
    }


def test_counter_calibration_requires_aligned_interval_endpoints():
    from src.heat_forecast import _calibration_hours

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = {
        "heat": [(start, 0.0), (start + timedelta(hours=1), 1.0)],
        "electricity": [
            (start, 0.0),
            (start + timedelta(hours=1, minutes=30), 0.2),
        ],
        "outdoor": [(start, 2.0), (start + timedelta(hours=1), 2.0)],
    }

    assert _calibration_hours(history, "heat", "electricity", "outdoor") == 0.0


def test_net_cop_keeps_defrost_electricity_in_heating_counter():
    q_net, e_total = net_cop_energies(
        heat_kwh=10.0,
        electricity_kwh=2.0,
        defrost_heat_kwh=0.8,
        defrost_electricity_kwh=0.12,
        defrost_el_in_heating_counter=True,
    )
    assert q_net == pytest.approx(9.2)
    assert e_total == pytest.approx(2.0)


def test_net_cop_adds_separate_defrost_electricity_only_when_not_in_counter():
    _, e_total = net_cop_energies(
        heat_kwh=10.0,
        electricity_kwh=2.0,
        defrost_heat_kwh=0.8,
        defrost_electricity_kwh=0.12,
        defrost_el_in_heating_counter=False,
    )
    assert e_total == pytest.approx(2.12)


def test_cop_calibration_uses_only_valid_net_and_separate_energy_pairs():
    common = {
        "heat_kwh": 10.0,
        "electricity_kwh": 2.0,
        "defrost_heat_kwh": 0.8,
        "defrost_electricity_kwh": 0.12,
    }

    net_included = cop_calibration_energies(
        **common, defrost_el_in_heating_counter=True, defrost_handling="net"
    )
    separate_included = cop_calibration_energies(
        **common, defrost_el_in_heating_counter=True, defrost_handling="separate"
    )
    net_excluded = cop_calibration_energies(
        **common, defrost_el_in_heating_counter=False, defrost_handling="net"
    )
    separate_excluded = cop_calibration_energies(
        **common, defrost_el_in_heating_counter=False, defrost_handling="separate"
    )

    assert net_included == pytest.approx((9.2, 2.0))
    assert separate_included == pytest.approx((9.2, 1.88))
    assert net_excluded == pytest.approx((9.2, 2.12))
    assert separate_excluded == pytest.approx((9.2, 2.0))


def test_auto_defrost_counter_diagnosis_compares_meter_and_power_integral():
    assert infer_defrost_counter_inclusion(
        heating_counter_kwh=0.4,
        defrost_counter_kwh=0.1,
        power_integral_kwh=0.4,
    )
    assert infer_defrost_counter_inclusion(
        heating_counter_kwh=0.3,
        defrost_counter_kwh=0.1,
        power_integral_kwh=0.4,
    ) is False
    assert infer_defrost_counter_inclusion(
        heating_counter_kwh=0.3,
        defrost_counter_kwh=0.1,
        power_integral_kwh=0.35,
    ) is None


def test_interval_learning_diagnoses_auto_meter_and_blends_live_defrost(
    monkeypatch, tmp_path
):
    from src import config

    entity_options = {
        "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID": "sensor.heating_el",
        "HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID": "sensor.heating_th",
        "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID": "sensor.dhw",
        "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID": "sensor.standby",
        "HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID": "sensor.defrost_el",
        "HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID": "sensor.defrost_th",
        "POWER_CONSUMPTION_ENTITY_ID": "sensor.power",
    }
    for key, value in entity_options.items():
        monkeypatch.setattr(config, key, value)
    monkeypatch.setattr(config, "COP_DEFROST_EL_IN_HEATING_COUNTER", "auto")
    monkeypatch.setattr(config, "COP_CALIBRATION_ENABLED", False)
    monkeypatch.setattr(config, "HEAT_FORECAST_ENABLED", True)
    monkeypatch.setattr(config, "COP_DEFROST_HANDLING", "net")
    monkeypatch.setattr(config, "COP_DEFROST_BLEND_COLD_RUN_HOURS", 2.0)
    monkeypatch.setattr(config, "DEFROST_DEFAULT_HEAT_FRACTION", 0.08)
    monkeypatch.setattr(config, "DEFROST_DEFAULT_EL_FRACTION", 0.06)

    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    start = datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC"))
    initial = {
        "heating_el": 10.0,
        "heating_th": 20.0,
        "dhw_el": 0.0,
        "standby_el": 0.0,
        "defrost_el": 1.0,
        "defrost_th": 2.0,
    }
    engine.state["counter_snapshots"] = {
        key: {"value_kwh": value, "time_utc": start.isoformat()}
        for key, value in initial.items()
    }
    engine.state["power_snapshot"] = {
        "power_w": 400.0,
        "time_utc": start.isoformat(),
    }

    for interval in range(1, 4):
        readings = {
            "sensor.heating_el": str(10.0 + interval * 0.4),
            "sensor.heating_th": str(20.0 + interval * 0.8),
            "sensor.dhw": "0",
            "sensor.standby": "0",
            "sensor.defrost_el": str(1.0 + interval * 0.1),
            "sensor.defrost_th": str(2.0 + interval * 0.08),
            "sensor.power": "400",
        }
        states = {
            entity_id: {"state": value} for entity_id, value in readings.items()
        }
        engine._record_interval(
            now=start.replace(hour=interval),
            states=states,
            outdoor_c=0.0,
        )

    assert engine.state["defrost_el_in_heating_counter"] == "true"
    assert engine.state["defrost_el_in_heating_counter_source"] == (
        "power_integral_diagnostic"
    )
    assert engine.state["defrost_heat_fraction_live"] == pytest.approx(0.1)
    assert engine.state["defrost_el_fraction_live"] == pytest.approx(0.25)
    assert engine.state["defrost_blend"] == 1.0


def test_new_hour_anchor_keeps_counter_energy_since_hour_boundary(
    monkeypatch, tmp_path
):
    from src import config

    entities = {
        "heating_el": "sensor.heating_el",
        "heating_th": "sensor.heating_th",
        "dhw_el": "sensor.dhw",
        "standby_el": "sensor.standby",
        "defrost_el": "sensor.defrost_el",
        "defrost_th": "sensor.defrost_th",
    }
    for key, entity_id in entities.items():
        name = {
            "heating_el": "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID",
            "heating_th": "HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID",
            "dhw_el": "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID",
            "standby_el": "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID",
            "defrost_el": "HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID",
            "defrost_th": "HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID",
        }[key]
        monkeypatch.setattr(config, name, entity_id)
    monkeypatch.setattr(config, "POWER_CONSUMPTION_ENTITY_ID", "sensor.power")
    monkeypatch.setattr(config, "HEAT_FORECAST_ENABLED", True)
    monkeypatch.setattr(config, "COP_CALIBRATION_ENABLED", False)

    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    prior_time = datetime(2026, 1, 1, 0, 50, tzinfo=ZoneInfo("UTC"))
    prior_values = {
        "heating_el": 10.0,
        "heating_th": 20.0,
        "dhw_el": 5.0,
        "standby_el": 2.0,
        "defrost_el": 1.0,
        "defrost_th": 3.0,
    }
    increments = {
        "heating_el": 0.2,
        "heating_th": 0.4,
        "dhw_el": 0.1,
        "standby_el": 0.05,
        "defrost_el": 0.02,
        "defrost_th": 0.04,
    }
    engine.state["counter_snapshots"] = {
        key: {"value_kwh": value, "time_utc": prior_time.isoformat()}
        for key, value in prior_values.items()
    }
    current_states = {
        entities[key]: {"state": str(value + increments[key])}
        for key, value in prior_values.items()
    }

    engine._record_interval(
        now=datetime(2026, 1, 1, 1, 10, tzinfo=ZoneInfo("UTC")),
        states=current_states,
        outdoor_c=0.0,
    )

    assert engine.state["hour_anchor"]["energy_so_far"]["heating_el"] == pytest.approx(
        0.1
    )
    assert engine.state["hour_anchor"]["energy_so_far"]["heating_th"] == pytest.approx(
        0.2
    )
    assert engine.state["hour_anchor"]["energy_so_far"]["defrost_th"] == pytest.approx(
        0.02
    )


def test_separate_defrost_forecast_adds_explicit_thermal_and_electrical_energy():
    starts = forecast_hour_starts(
        datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC")), 1
    )
    result = hourly_electrical_forecast(
        starts=starts,
        heat_th_kwh=[2.0],
        outdoor_c=[0.0],
        cop_curve={"intercept": 4.0, "slope": 0.0},
        dhw_el_kwh=[0.0],
        standby_el_kwh=[0.0],
        defrost_handling="separate",
        defrost_heat_fraction=0.1,
        defrost_el_fraction=0.05,
    )
    assert result["forecast_heat_el_kwh"] == pytest.approx([0.5])
    assert result["forecast_defrost_el_kwh"] == pytest.approx([0.0263])


def test_energy_weighted_cop_fit_uses_energy_not_mean_of_hourly_cops():
    samples = [
        {"outdoor_c": 0.0, "heat_kwh": 1.0, "electric_kwh": 0.25},
        {"outdoor_c": 10.0, "heat_kwh": 9.0, "electric_kwh": 1.5},
    ]
    fit = fit_energy_weighted_cop(samples)
    assert fit["samples"] == 2
    assert fit["slope"] == pytest.approx(0.2)
    assert fit["intercept"] == pytest.approx(4.0)


def test_pv_heat_gain_uses_model_weight_and_saturation():
    gain = thermal_pv_gain_kw(5000, weight_kw_per_w=0.002, cap_kw=3.0)

    assert gain == pytest.approx(3.0 * math.tanh(10.0 / 3.0))
    assert gain < 3.0


def test_minute_power_history_estimates_defrost_event_energy_and_fractions():
    start = datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC"))
    power = [(start.replace(minute=minute), 600.0) for minute in range(4)]
    defrost = [
        (start.replace(minute=minute), 1.0 if minute in (1, 2) else 0.0)
        for minute in range(4)
    ]
    thermal = [
        (start.replace(minute=minute), -1200.0 if minute in (1, 2) else 0.0)
        for minute in range(4)
    ]

    estimate = estimate_defrost_power_history(
        electrical_power=power,
        defrost_state=defrost,
        thermal_power=thermal,
        heating_electricity_kwh=0.4,
        heating_heat_kwh=0.8,
    )

    assert estimate["event_count"] == 1
    assert estimate["mean_electricity_kwh_per_event"] == pytest.approx(0.02)
    assert estimate["mean_heat_kwh_per_event"] == pytest.approx(0.04)
    assert estimate["mean_electrical_power_kw"] == pytest.approx(0.6)
    assert estimate["mean_thermal_defrost_power_kw"] == pytest.approx(1.2)
    assert estimate["electricity_fraction"] == pytest.approx(0.05)
    assert estimate["heat_fraction"] == pytest.approx(0.05)


def test_sparse_ha_defrost_history_is_resampled_with_bounded_carry_forward():
    from src.heat_forecast import _resample_history_to_minutes

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    electrical, status, thermal = _resample_history_to_minutes(
        [
            [(start, 600.0), (start + timedelta(minutes=5), 600.0)],
            [(start, 1.0), (start + timedelta(minutes=5), 0.0)],
            [(start, -1200.0), (start + timedelta(minutes=5), -1200.0)],
        ],
        (10, 30, 10),
    )

    estimate = estimate_defrost_power_history(
        electrical_power=electrical,
        defrost_state=status,
        thermal_power=thermal,
        heating_electricity_kwh=1.0,
        heating_heat_kwh=2.0,
    )

    assert len(electrical) == 6
    assert estimate["event_count"] == 1
    assert estimate["mean_electricity_kwh_per_event"] == pytest.approx(0.05)
    assert estimate["mean_heat_kwh_per_event"] == pytest.approx(0.1)
    assert estimate["mean_electrical_power_kw"] == pytest.approx(0.6)
    assert estimate["mean_thermal_defrost_power_kw"] == pytest.approx(1.2)
    assert estimate["electricity_fraction"] == pytest.approx(0.05)
    assert estimate["heat_fraction"] == pytest.approx(0.05)


def test_backfill_seeds_defrost_estimates_from_minute_history(monkeypatch, tmp_path):
    from src import config

    monkeypatch.setattr(config, "POWER_CONSUMPTION_ENTITY_ID", "sensor.power")
    monkeypatch.setattr(config, "THERMAL_POWER_ENTITY_ID", "sensor.thermal_power")
    monkeypatch.setattr(config, "DEFROST_STATUS_ENTITY_ID", "binary_sensor.defrost")
    monkeypatch.setattr(
        config, "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID", "sensor.heat_el"
    )
    monkeypatch.setattr(
        config, "HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID", "sensor.heat_th"
    )
    now = datetime(2026, 1, 1, 12, tzinfo=ZoneInfo("UTC"))
    start = now.replace(hour=10)
    counters = {
        "sensor.heat_el": [(start, 10.0), (now, 10.4)],
        "sensor.heat_th": [(start, 20.0), (now, 20.8)],
    }
    event_minutes = (1, 4, 7, 10, 13)
    minutes = [(start + timedelta(minutes=minute), 240.0) for minute in event_minutes]
    thermal = [
        (start + timedelta(minutes=minute), -480.0)
        for minute in event_minutes
    ]
    states = [
        (start + timedelta(minutes=minute), 1.0) for minute in event_minutes
    ]

    class FakeInflux:
        def fetch_minute_history(
            self, entity_ids, _start, _stop, *, agg_fn, timeout_seconds
        ):
            values = {
                "sensor.power": minutes,
                "sensor.thermal_power": thermal,
                "binary_sensor.defrost": states,
            }
            return {entity: values[entity] for entity in entity_ids}

    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    engine._backfill_defrost_power_history(
        now=now,
        influx_service=FakeInflux(),
        counter_history=counters,
    )

    assert engine.state["defrost_heat_fraction_power"] == pytest.approx(0.05)
    assert engine.state["defrost_el_fraction_power"] == pytest.approx(0.05)
    assert engine.state["defrost_power_history"]["event_count"] == 5


def test_pv_surplus_offset_survives_price_service_failure(monkeypatch, tmp_path):
    from src import config
    from src import pv_trajectory, price_optimizer

    monkeypatch.setattr(config, "ELECTRICITY_PRICE_ENABLED", True)
    monkeypatch.setattr(config, "PRICE_COST_MODE_ENABLED", True)
    monkeypatch.setattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False)
    monkeypatch.setattr(config, "PV_SURPLUS_CHEAP_ENABLED", True)
    monkeypatch.setattr(config, "PV_SURPLUS_CHEAP_THRESHOLD_W", 1000.0)
    monkeypatch.setattr(config, "PRICE_TARGET_OFFSET", 0.5)
    monkeypatch.setattr(config, "HEAT_FORECAST_REUSE_PRICES", False)
    monkeypatch.setattr(
        pv_trajectory,
        "get_forecast_trajectory_state",
        lambda *_args: SimpleNamespace(active=False),
    )

    class BrokenOptimizer:
        def refresh_prices_if_needed(self, _ha_client):
            raise RuntimeError("price API unavailable")

    monkeypatch.setattr(price_optimizer, "PriceOptimizer", BrokenOptimizer)
    starts = forecast_hour_starts(
        datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC")), 1
    )
    offsets, sources = HeatForecastEngine(
        str(tmp_path / "state.json")
    )._target_offsets(
        starts=starts,
        pv_electrical=[2000.0],
        ha_client=object(),
    )

    assert offsets == [0.5]
    assert sources == ["pv_forecast"]


@pytest.mark.parametrize("missing_indoor_and_target", [False, True])
@pytest.mark.parametrize("fireplace_on", [False, True])
def test_update_publishes_hourly_thermal_and_electrical_profiles(
    monkeypatch, tmp_path, missing_indoor_and_target, fireplace_on
):
    from src import config, ha_client, heat_forecast

    entity_options = {
        "OUTDOOR_TEMP_ENTITY_ID": "sensor.outdoor",
        "INDOOR_TEMP_ENTITY_ID": "sensor.indoor",
        "TARGET_INDOOR_TEMP_ENTITY_ID": "sensor.target",
        "HEATING_STATUS_ENTITY_ID": "climate.heating",
        "FIREPLACE_STATUS_ENTITY_ID": "binary_sensor.fireplace",
        "PV_POWER_ENTITY_ID": "sensor.pv",
        "PV_FORECAST_ENTITY_ID": "",
        "FLOW_RATE_ENTITY_ID": "sensor.flow",
        "INLET_TEMP_ENTITY_ID": "sensor.inlet",
        "ACTUAL_OUTLET_TEMP_ENTITY_ID": "sensor.outlet",
        "POWER_CONSUMPTION_ENTITY_ID": "sensor.power",
        "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID": "sensor.heating_el",
        "HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID": "sensor.heating_th",
        "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID": "sensor.dhw",
        "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID": "sensor.standby",
        "HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID": "sensor.defrost_el",
        "HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID": "sensor.defrost_th",
    }
    for key, value in entity_options.items():
        monkeypatch.setattr(config, key, value)
    for key, value in {
        "HEAT_FORECAST_ENABLED": True,
        "HEAT_FORECAST_HORIZON_HOURS": 3,
        "COP_CALIBRATION_ENABLED": False,
        "BUILDING_BALANCE_CAPACITY_MODE": "manual",
        "HEAT_FORECAST_BASE_GAIN_KW": 0.0,
        "HEAT_FORECAST_STEP_MINUTES": 15,
        "ELECTRICITY_PRICE_ENABLED": False,
        "PRICE_COST_MODE_ENABLED": False,
        "PV_TRAJ_FORECAST_MODE_ENABLED": False,
        "PV_SURPLUS_CHEAP_ENABLED": False,
    }.items():
        monkeypatch.setattr(config, key, value)
    monkeypatch.setattr(ha_client, "get_sensor_attributes", lambda _entity: {})

    class FakeHAClient:
        published = None

        def get_state(self, entity_id, states, is_binary=False):
            raw = states.get(entity_id, {}).get("state")
            if is_binary:
                return str(raw).lower() in ("on", "true", "1")
            return raw

        def get_hourly_forecast_strict(self, _hours, starts=None):
            return [0.0, -5.0]

        def set_state(self, entity_id, state, attributes, round_digits=2):
            self.published = (entity_id, state, attributes, round_digits)
            return True

    model = SimpleNamespace(
        external_source_weights={"pv": 0.002},
        solar_decay_tau_hours=0.5,
        heat_loss_coefficient=0.13,
        outlet_effectiveness=0.2,
        slab_time_constant_hours=3.0,
        thermal_time_constant=10.0,
        orchestrator=None,
    )
    wrapper = SimpleNamespace(thermal_model=model, _heating_thermal_model=model)
    states = {
        "sensor.outdoor": {"state": "0"},
        "sensor.indoor": {"state": "20"},
        "sensor.target": {"state": "21"},
        "climate.heating": {"state": "heat"},
        "binary_sensor.fireplace": {"state": "on" if fireplace_on else "off"},
        "sensor.pv": {"state": "0"},
        "sensor.flow": {"state": "5"},
        "sensor.inlet": {"state": "28"},
        "sensor.outlet": {"state": "30"},
        "sensor.power": {"state": "500"},
        "sensor.heating_el": {"state": "10"},
        "sensor.heating_th": {"state": "20"},
        "sensor.dhw": {"state": "3"},
        "sensor.standby": {"state": "1"},
        "sensor.defrost_el": {"state": "0"},
        "sensor.defrost_th": {"state": "0"},
    }
    if missing_indoor_and_target:
        states.pop("sensor.indoor")
        states.pop("sensor.target")
    client = FakeHAClient()
    simulation_fractions = []
    original_simulate = heat_forecast.simulate_heating_energy

    def record_simulation(*args, **kwargs):
        simulation_fractions.append(kwargs.get("first_hour_fraction"))
        return original_simulate(*args, **kwargs)

    monkeypatch.setattr(heat_forecast, "simulate_heating_energy", record_simulation)
    result = HeatForecastEngine(str(tmp_path / "state.json")).update_and_publish(
        ha_client=client,
        states=states,
        wrapper=wrapper,
        now=datetime(2026, 1, 1, 12, 30, tzinfo=ZoneInfo("UTC")),
    )

    assert result is True
    entity_id, _, attributes, _ = client.published
    assert entity_id == config.HEAT_FORECAST_SENSOR_ENTITY_ID
    assert len(attributes["forecast"]) == 3
    assert len(attributes["forecast_heat_th_kwh"]) == 3
    assert attributes["forecast_heat_th_kwh"][1] > 0
    assert attributes["calibration_source"] == "defaults"
    assert attributes["calibration_hours"] == 0
    assert len(attributes["price_source_per_hour"]) == 3
    assert isinstance(attributes["degraded_reasons"], list)
    assert attributes["current_hour_estimate"] == "counter_anchor"
    assert attributes["includes"] == ["heating", "dhw", "standby"]
    assert simulation_fractions == ([0.5, 0.5, 0.5] if fireplace_on else [0.5])
    if fireplace_on:
        assert set(attributes["forecast_fireplace_scenarios"]) == {"3h", "4h"}


def test_cop_curve_decays_stale_bins_for_confidence_and_fit_weight():
    now = datetime(2026, 1, 10, tzinfo=ZoneInfo("UTC"))
    state = {
        "cop_bins": {
            "0": {
                "heat_kwh": 20.0,
                "electric_kwh": 5.0,
                "temp_energy_kwh": 0.0,
                "hours": 10.0,
                "updated_at": "2026-01-08T00:00:00+00:00",
            },
            "10": {
                "heat_kwh": 20.0,
                "electric_kwh": 5.0,
                "temp_energy_kwh": 50.0,
                "hours": 10.0,
                "updated_at": "2026-01-10T00:00:00+00:00",
            },
        }
    }

    curve = cop_curve_from_state(
        state,
        now=now,
        half_life_days=2,
        start_intercept=5.0,
        start_slope=0.2,
        min_hours_per_bin=6,
        min_value=1.5,
        max_value=10.0,
    )

    assert curve["calibrated"] is False
    assert curve["samples"] == 1


def test_heating_energy_is_zero_when_room_is_already_above_target():
    result = simulate_heating_energy(
        outdoor_c=[0.0],
        pv_gain_kw=[0.0],
        fireplace_kw=[0.0],
        target_c=[21.0],
        room_c=22.0,
        slab_c=22.0,
        room_capacity_kwh_k=5.8,
        slab_capacity_kwh_k=3.9,
        heat_loss_kw_k=0.13,
        slab_to_room_kw_k=0.2,
    )
    assert result["heat_kwh"] == [0.0]
    assert result["max_hp_kw"] == 0.0


def test_heating_energy_is_bounded_and_weather_dependent():
    common = {
        "pv_gain_kw": [0.0],
        "fireplace_kw": [0.0],
        "target_c": [21.0],
        "room_c": 20.9,
        "slab_c": 20.9,
        "room_capacity_kwh_k": 5.8,
        "slab_capacity_kwh_k": 0.1,
        "heat_loss_kw_k": 0.01,
        "slab_to_room_kw_k": 0.2,
        "max_hp_kw": 8.0,
    }
    cold = simulate_heating_energy(outdoor_c=[-10.0], **common)
    mild = simulate_heating_energy(outdoor_c=[10.0], **common)
    assert 0.0 <= cold["heat_kwh"][0] <= 8.0
    assert cold["heat_kwh"][0] > mild["heat_kwh"][0]


def test_hour_starts_follow_berlin_dst_and_keep_utc_order():
    before_spring_change = datetime(2026, 3, 29, 0, 0, tzinfo=ZoneInfo("UTC"))
    starts = forecast_hour_starts(before_spring_change, 4)
    assert [item.isoformat() for item in starts] == [
        "2026-03-29T01:00:00+01:00",
        "2026-03-29T03:00:00+02:00",
        "2026-03-29T04:00:00+02:00",
        "2026-03-29T05:00:00+02:00",
    ]


def test_forecast_state_store_round_trips_and_keeps_backup(tmp_path):
    store = ForecastStateStore(str(tmp_path / "heat_forecast.json"))
    state = store.empty()
    state["cop_bins"]["2"] = {"heat_kwh": 3.0}
    store.save(state)
    state["cop_bins"]["2"]["heat_kwh"] = 4.0
    store.save(state)
    assert store.load()["cop_bins"]["2"]["heat_kwh"] == 4.0
    assert (tmp_path / "heat_forecast.json.bak").exists()


def test_defrost_history_carries_state_and_splits_power_gaps():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    power = [
        (start + timedelta(minutes=minute), 600.0)
        for minute in (0, 1, 2, 5, 6)
    ]
    state = [(start + timedelta(minutes=1), 1.0)]

    estimate = estimate_defrost_power_history(
        electrical_power=power,
        defrost_state=state,
        thermal_power=[],
        heating_electricity_kwh=0.4,
        heating_heat_kwh=0.8,
    )

    assert estimate["event_count"] == 2
    assert estimate["total_defrost_electricity_kwh"] == pytest.approx(0.04)
    assert estimate["mean_electricity_kwh_per_event"] == pytest.approx(0.02)
    assert estimate["heat_fraction"] is None


def test_cop_curve_ages_temperature_numerator_without_shifting_temperature():
    now = datetime(2026, 1, 4, tzinfo=timezone.utc)
    curve = cop_curve_from_state(
        {
            "cop_bins": {
                "10": {
                    "heat_kwh": 20.0,
                    "electric_kwh": 5.0,
                    "temp_energy_kwh": 50.0,
                    "hours": 10.0,
                    "updated_at": "2026-01-02T00:00:00+00:00",
                }
            }
        },
        now=now,
        half_life_days=2,
        start_intercept=4.0,
        start_slope=0.0,
        min_hours_per_bin=1.0,
        min_value=1.5,
        max_value=10.0,
    )

    assert curve["temp_range_c"] == pytest.approx([10.0, 10.0])


def test_cop_fit_applies_age_weight_once():
    now = datetime(2026, 1, 4, tzinfo=timezone.utc)
    buckets = {}
    for temperature, cop, updated in (
        (0, 2.0, "2026-01-04T00:00:00+00:00"),
        (5, 6.0, "2026-01-02T00:00:00+00:00"),
        (10, 4.0, "2026-01-04T00:00:00+00:00"),
    ):
        buckets[str(temperature)] = {
            "heat_kwh": cop * 100.0,
            "electric_kwh": 100.0,
            "temp_energy_kwh": temperature * 100.0,
            "hours": 20.0,
            "updated_at": updated,
        }
    curve = cop_curve_from_state(
        {"cop_bins": buckets},
        now=now,
        half_life_days=2,
        start_intercept=4.0,
        start_slope=0.0,
        min_hours_per_bin=1.0,
        min_value=1.5,
        max_value=10.0,
    )
    expected = fit_energy_weighted_cop(
        [
            {"outdoor_c": 0.0, "heat_kwh": 200.0, "electric_kwh": 100.0},
            {"outdoor_c": 5.0, "heat_kwh": 300.0, "electric_kwh": 50.0},
            {"outdoor_c": 10.0, "heat_kwh": 400.0, "electric_kwh": 100.0},
        ]
    )

    assert curve["slope"] == pytest.approx(expected["slope"])
    assert curve["intercept"] == pytest.approx(expected["intercept"])


def test_hourly_cop_uses_configured_bounds_and_separate_defrost_net_convention():
    starts = forecast_hour_starts(datetime(2026, 1, 1, tzinfo=timezone.utc), 1)
    bounded = hourly_electrical_forecast(
        starts=starts,
        heat_th_kwh=[2.0],
        outdoor_c=[0.0],
        cop_curve={"intercept": 12.0, "slope": 0.0},
        dhw_el_kwh=[0.0],
        standby_el_kwh=[0.0],
        cop_min=2.0,
        cop_max=6.0,
    )
    bounded_min = hourly_electrical_forecast(
        starts=starts,
        heat_th_kwh=[2.0],
        outdoor_c=[0.0],
        cop_curve={"intercept": 0.5, "slope": 0.0},
        dhw_el_kwh=[0.0],
        standby_el_kwh=[0.0],
        cop_min=2.0,
        cop_max=6.0,
    )
    separate = hourly_electrical_forecast(
        starts=starts,
        heat_th_kwh=[2.0],
        outdoor_c=[0.0],
        cop_curve={"intercept": 4.0, "slope": 0.0},
        dhw_el_kwh=[0.0],
        standby_el_kwh=[0.0],
        defrost_handling="separate",
        defrost_heat_fraction=0.5,
        defrost_el_fraction=0.1,
    )
    zero_heat_loss = hourly_electrical_forecast(
        starts=starts,
        heat_th_kwh=[2.0],
        outdoor_c=[0.0],
        cop_curve={"intercept": 4.0, "slope": 0.0},
        dhw_el_kwh=[0.0],
        standby_el_kwh=[0.0],
        defrost_handling="separate",
        defrost_heat_fraction=0.0,
        defrost_el_fraction=0.1,
    )

    assert bounded["forecast_heat_el_kwh"] == pytest.approx([0.3333])
    assert bounded_min["forecast_heat_el_kwh"] == pytest.approx([1.0])
    assert separate["forecast_heat_el_kwh"] == pytest.approx([0.5])
    assert separate["forecast_defrost_el_kwh"] == pytest.approx([0.0556])
    assert zero_heat_loss["forecast_defrost_el_kwh"][0] > 0


def test_cop_learning_skips_counter_deltas_from_different_intervals(
    monkeypatch, tmp_path
):
    from src import config

    monkeypatch.setattr(config, "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID", "el")
    monkeypatch.setattr(config, "HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID", "th")
    monkeypatch.setattr(config, "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID", "")
    monkeypatch.setattr(config, "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID", "")
    monkeypatch.setattr(config, "HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID", "")
    monkeypatch.setattr(config, "HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID", "")
    monkeypatch.setattr(config, "POWER_CONSUMPTION_ENTITY_ID", "")
    monkeypatch.setattr(config, "COP_CALIBRATION_ENABLED", True)
    monkeypatch.setattr(config, "HEAT_FORECAST_ENABLED", False)
    monkeypatch.setattr(config, "COP_DEFROST_EL_IN_HEATING_COUNTER", "true")

    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    engine.state["counter_snapshots"] = {
        "heating_el": {"value_kwh": 10.0, "time_utc": start.isoformat()},
        "heating_th": {
            "value_kwh": 20.0,
            "time_utc": (start + timedelta(minutes=10)).isoformat(),
        },
    }

    engine._record_interval(
        now=start + timedelta(minutes=20),
        states={"el": {"state": "10.2"}, "th": {"state": "20.2"}},
        outdoor_c=0.0,
    )

    assert engine.state["cop_bins"] == {}


def test_daily_counter_interval_is_split_at_local_midnight_live_and_backfill(
    monkeypatch, tmp_path
):
    from src import config

    monkeypatch.setattr(config, "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID", "dhw")
    monkeypatch.setattr(config, "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID", "standby")
    monkeypatch.setattr(config, "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID", "heating")
    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    previous = datetime(2026, 1, 1, 22, 30, tzinfo=timezone.utc)
    current = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc)
    snapshots = {
        "dhw_el": {"value_kwh": 2.0, "time_utc": previous.isoformat()},
        "standby_el": {"value_kwh": 1.0, "time_utc": previous.isoformat()},
    }
    engine._record_daily_energy(
        current,
        {"dhw_el": 0.6, "standby_el": 0.2},
        {"dhw_el": 1.0, "standby_el": 1.0},
        snapshots,
    )
    assert engine.state["daily_energy"]["2026-01-01"]["dhw"][23] == pytest.approx(
        0.3
    )
    assert engine.state["daily_energy"]["2026-01-02"]["dhw"][0] == pytest.approx(
        0.3
    )

    backfill = HeatForecastEngine(str(tmp_path / "backfill.json"))
    backfill._backfill_daily_energy(
        {
            "dhw": [(previous, 2.0), (current, 2.6)],
            "standby": [(previous, 1.0), (current, 1.2)],
        },
        current,
    )
    assert backfill.state["daily_energy"]["2026-01-01"]["dhw"][23] == pytest.approx(
        0.3
    )
    assert backfill.state["daily_energy"]["2026-01-02"]["dhw"][0] == pytest.approx(
        0.3
    )


def test_backfill_counts_unchanged_daily_counters_on_observed_short_intervals(
    monkeypatch, tmp_path
):
    from src import config

    monkeypatch.setattr(config, "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID", "dhw")
    monkeypatch.setattr(config, "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID", "standby")
    monkeypatch.setattr(config, "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID", "heating")
    start = datetime(2026, 1, 1, 23, tzinfo=timezone.utc)
    heat_samples = [
        (start + timedelta(hours=hour), float(hour)) for hour in range(25)
    ]
    engine = HeatForecastEngine(str(tmp_path / "state.json"))

    engine._backfill_daily_energy(
        {
            "heating": heat_samples,
            "dhw": [(start, 2.0), (start + timedelta(hours=24), 2.0)],
            "standby": [(start, 1.0), (start + timedelta(hours=24), 1.0)],
        },
        start + timedelta(days=2),
    )

    daily = engine.state["daily_energy"]["2026-01-02"]
    assert daily["dhw_coverage_h"] == pytest.approx(24.0)
    assert daily["standby_coverage_h"] == pytest.approx(24.0)

    gap = HeatForecastEngine(str(tmp_path / "gap.json"))
    gap._backfill_daily_energy(
        {
            "dhw": [(start, 2.0), (start + timedelta(hours=24), 2.0)],
            "standby": [(start, 1.0), (start + timedelta(hours=24), 1.0)],
        },
        start + timedelta(days=2),
    )
    assert gap.state["daily_energy"] == {}


def test_live_daily_allocation_ends_at_counter_sample_time(monkeypatch, tmp_path):
    from src import config

    monkeypatch.setattr(config, "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID", "dhw")
    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    previous = datetime(2026, 1, 1, 22, 30, tzinfo=timezone.utc)
    sample_time = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc)
    engine.state["counter_snapshots"] = {
        "dhw_el": {"value_kwh": 2.6, "time_utc": sample_time.isoformat()}
    }

    engine._record_daily_energy(
        datetime(2026, 1, 2, 2, 30, tzinfo=timezone.utc),
        {"dhw_el": 0.6},
        {"dhw_el": 1.0},
        {"dhw_el": {"value_kwh": 2.0, "time_utc": previous.isoformat()}},
    )

    assert engine.state["daily_energy"]["2026-01-01"]["dhw"][23] == pytest.approx(
        0.3
    )
    assert engine.state["daily_energy"]["2026-01-02"]["dhw"][0] == pytest.approx(
        0.3
    )
    assert engine.state["daily_energy"]["2026-01-02"]["dhw"][3] == 0.0


def test_dhw_daily_budget_caps_every_forecast_slot_today(monkeypatch, tmp_path):
    from src import config

    monkeypatch.setattr(config, "STANDBY_DEFAULT_KW", 0.0)
    starts = [
        datetime(2026, 1, 1, hour, tzinfo=ZoneInfo("Europe/Berlin"))
        for hour in (8, 9, 10)
    ]
    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    engine.state["daily_energy"] = {
        "2026-01-01": {
            "dhw": [0.0] * 24,
            "standby": [0.0] * 24,
            "dhw_coverage_h": 0.0,
            "standby_coverage_h": 0.0,
        }
    }
    engine.state["daily_energy"]["2026-01-01"]["dhw"][8] = 1.0
    engine.state["daily_energy"]["2026-01-01"]["dhw"][9] = 0.1

    dhw, _, _ = engine._dhw_and_standby_profiles(starts)

    assert dhw == pytest.approx([0.0, 0.0, 0.0])


def test_heating_energy_smoothing_preserves_each_six_hour_total():
    from src.heat_forecast import _smooth_heating_energy

    original = [1.0, 1.0, 1.0, 8.0, 1.0, 1.0, 2.0]
    smoothed = _smooth_heating_energy(original)

    assert sum(smoothed[:6]) == pytest.approx(sum(original[:6]))
    assert max(smoothed[:6]) < max(original[:6])
    assert smoothed[6] == original[6]


def test_price_forecast_uses_isolated_optimizer_and_per_day_thresholds(
    monkeypatch, tmp_path
):
    from src import config, price_optimizer

    monkeypatch.setattr(config, "ELECTRICITY_PRICE_ENABLED", True)
    monkeypatch.setattr(config, "PRICE_COST_MODE_ENABLED", False)
    monkeypatch.setattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False)
    monkeypatch.setattr(config, "PV_SURPLUS_CHEAP_ENABLED", False)
    monkeypatch.setattr(config, "PRICE_TARGET_OFFSET", 0.2)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    prices = [0.1 + 0.1 * index for index in range(24)]
    prices += [2.0 + 0.1 * index for index in range(24)]
    entries = [
        (start + timedelta(hours=index), price)
        for index, price in enumerate(prices)
    ]

    def seed_forecast_cache(optimizer, _ha_client):
        optimizer._price_entries = entries
        optimizer._price_tz = timezone.utc

    monkeypatch.setattr(
        price_optimizer.PriceOptimizer,
        "refresh_prices_if_needed",
        seed_forecast_cache,
    )
    controller_optimizer = price_optimizer.PriceOptimizer()
    controller_optimizer._last_price = 12.3
    monkeypatch.setattr(price_optimizer, "_optimizer", controller_optimizer)
    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    starts = [start, start + timedelta(hours=24)]

    offsets, _ = engine._target_offsets(
        starts=starts, pv_electrical=[0.0, 0.0], ha_client=object()
    )

    assert offsets == pytest.approx([0.2, 0.2])
    assert engine._price_optimizer is not controller_optimizer
    assert controller_optimizer._last_price == 12.3


def test_bounded_hourly_prices_reach_explicit_reuse_policy(monkeypatch, tmp_path):
    from src import config, price_optimizer

    monkeypatch.setattr(config, "ELECTRICITY_PRICE_ENABLED", True)
    monkeypatch.setattr(config, "PRICE_COST_MODE_ENABLED", False)
    monkeypatch.setattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False)
    monkeypatch.setattr(config, "PV_SURPLUS_CHEAP_ENABLED", False)
    monkeypatch.setattr(config, "HEAT_FORECAST_REUSE_PRICES", False)
    monkeypatch.setattr(
        price_optimizer.PriceOptimizer, "refresh_prices_if_needed", lambda *_: None
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    optimizer = price_optimizer.PriceOptimizer()
    optimizer._price_entries = [(start, 0.1)]
    optimizer._price_tz = timezone.utc
    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    engine._price_optimizer = optimizer

    _, sources = engine._target_offsets(
        starts=[start, start + timedelta(hours=1)],
        pv_electrical=[0.0, 0.0],
        ha_client=object(),
    )

    assert sources == ["actual", "none"]


def test_price_offset_is_suppressed_during_active_forecast_trajectory(
    monkeypatch, tmp_path
):
    from src import config, price_optimizer, pv_trajectory

    monkeypatch.setattr(config, "ELECTRICITY_PRICE_ENABLED", True)
    monkeypatch.setattr(config, "PRICE_COST_MODE_ENABLED", False)
    monkeypatch.setattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", True)
    monkeypatch.setattr(config, "PV_TRAJ_DISABLE_PRICE_IN_FORECAST_MODE", True)
    monkeypatch.setattr(config, "PV_TRAJ_HEATING_TARGET_OFFSET", 0.2)
    monkeypatch.setattr(config, "PV_SURPLUS_CHEAP_ENABLED", False)
    monkeypatch.setattr(config, "PRICE_TARGET_OFFSET", 0.5)
    monkeypatch.setattr(
        price_optimizer.PriceOptimizer, "refresh_prices_if_needed", lambda *_: None
    )
    monkeypatch.setattr(
        pv_trajectory,
        "get_forecast_trajectory_state",
        lambda *_args: SimpleNamespace(active=True),
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    optimizer = price_optimizer.PriceOptimizer()
    optimizer._price_entries = [
        (start + timedelta(hours=index), 0.1) for index in range(24)
    ]
    optimizer._price_tz = timezone.utc
    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    engine._price_optimizer = optimizer

    offsets, _ = engine._target_offsets(
        starts=[start], pv_electrical=[0.0], ha_client=object()
    )

    assert offsets == [0.2]


def test_strict_weather_forecast_aligns_timestamps_and_preserves_gaps(
    monkeypatch,
):
    from src.ha_client import HAClient
    from src import ha_client

    start = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    forecast = [
        {"datetime": (start - timedelta(hours=1)).isoformat(), "temperature": -5},
        {"datetime": start.isoformat(), "temperature": 1},
        {"datetime": (start + timedelta(hours=2)).isoformat(), "temperature": 3},
    ]

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"service_response": {"weather.home": {"forecast": forecast}}}

    monkeypatch.setattr(ha_client.requests, "post", lambda *_args, **_kwargs: Response())
    client = HAClient("http://fake", "token")

    result = client.get_hourly_forecast_strict(
        3,
        starts=[start, start + timedelta(hours=1), start + timedelta(hours=2)],
    )

    assert result == [1.0, None, 3.0]


def test_current_slot_components_reconcile_to_power_corrected_total(
    monkeypatch, tmp_path
):
    from src import config

    monkeypatch.setattr(config, "COP_DEFROST_HANDLING", "separate")
    monkeypatch.setattr(config, "COP_DEFROST_EL_IN_HEATING_COUNTER", "false")
    monkeypatch.setattr(config, "POWER_CONSUMPTION_ENTITY_ID", "power")
    for key in (
        "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID",
        "HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID",
        "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID",
        "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID",
        "HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID",
        "HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID",
    ):
        monkeypatch.setattr(config, key, key)
    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    electrical = {
        "forecast": [{"value": 0.9}],
        "forecast_total_el_kwh": [0.9],
        "forecast_heat_th_kwh": [1.0],
        "forecast_heat_el_kwh": [0.5],
        "forecast_defrost_el_kwh": [0.1],
        "forecast_dhw_el_kwh": [0.2],
        "forecast_standby_el_kwh": [0.1],
    }

    engine._correct_current_slot(
        now=datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc),
        states={"power": {"state": "0"}},
        ha_client=SimpleNamespace(get_state=lambda entity, states: states[entity]["state"]),
        deltas={},
        electrical=electrical,
        heat_result={"heat_kwh": [1.0]},
        outdoor=[0.0],
        remaining_hours=0.5,
    )

    components = [
        electrical["forecast_heat_el_kwh"][0],
        electrical["forecast_defrost_el_kwh"][0],
        electrical["forecast_dhw_el_kwh"][0],
        electrical["forecast_standby_el_kwh"][0],
    ]
    assert electrical["forecast_total_el_kwh"][0] == pytest.approx(sum(components))
    assert components == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_current_hour_coverage_stays_degraded_without_hour_boundary_anchor(
    monkeypatch, tmp_path
):
    from src import config

    monkeypatch.setattr(config, "COP_DEFROST_HANDLING", "net")
    monkeypatch.setattr(config, "COP_DEFROST_EL_IN_HEATING_COUNTER", "true")
    monkeypatch.setattr(config, "POWER_CONSUMPTION_ENTITY_ID", "power")
    entities = {
        "heating_el": "el",
        "heating_th": "th",
        "dhw_el": "dhw",
        "standby_el": "standby",
        "defrost_el": "defrost_el",
        "defrost_th": "defrost_th",
    }
    for key, entity in entities.items():
        config_key = {
            "heating_el": "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID",
            "heating_th": "HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID",
            "dhw_el": "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID",
            "standby_el": "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID",
            "defrost_el": "HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID",
            "defrost_th": "HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID",
        }[key]
        monkeypatch.setattr(config, config_key, entity)
    now = datetime(2026, 1, 1, 12, 35, tzinfo=timezone.utc)
    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    engine.state["hour_anchor"] = {
        "hour_utc": datetime(2026, 1, 1, 12, tzinfo=timezone.utc).isoformat(),
        "sample_time_utc": datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc).isoformat(),
        "readings": {key: 1.0 for key in entities},
        "energy_so_far": {key: 0.0 for key in entities},
        "energy_coverage": {key: False for key in entities},
    }
    states = {"power": {"state": "500"}}
    for entity in entities.values():
        states[entity] = {"state": "1.01"}
    electrical = {
        "forecast": [{"value": 0.5}],
        "forecast_total_el_kwh": [0.5],
        "forecast_heat_th_kwh": [0.1],
        "forecast_heat_el_kwh": [0.1],
        "forecast_dhw_el_kwh": [0.1],
        "forecast_standby_el_kwh": [0.1],
    }

    engine._correct_current_slot(
        now=now,
        states=states,
        ha_client=SimpleNamespace(get_state=lambda entity, current: current[entity]["state"]),
        deltas={},
        electrical=electrical,
        heat_result={"heat_kwh": [0.1]},
        outdoor=[0.0],
        remaining_hours=25.0 / 60.0,
    )

    assert engine.state["current_hour_estimate"] == "degraded_partial"


def test_partial_counter_coverage_projects_only_uncovered_hour(monkeypatch, tmp_path):
    from src import config

    entities = {
        "heating_el": ("HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID", "el", 10.0),
        "heating_th": ("HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID", "th", 20.0),
        "dhw_el": ("HEAT_FORECAST_DHW_COUNTER_ENTITY_ID", "dhw", 5.0),
        "standby_el": ("HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID", "standby", 2.0),
        "defrost_el": (
            "HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID",
            "defrost_el",
            1.0,
        ),
        "defrost_th": (
            "HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID",
            "defrost_th",
            3.0,
        ),
    }
    for config_key, entity_id, _ in entities.values():
        monkeypatch.setattr(config, config_key, entity_id)
    monkeypatch.setattr(config, "POWER_CONSUMPTION_ENTITY_ID", "power")
    monkeypatch.setattr(config, "COP_DEFROST_HANDLING", "net")
    monkeypatch.setattr(config, "COP_DEFROST_EL_IN_HEATING_COUNTER", "true")
    now = datetime(2026, 1, 1, 10, 45, tzinfo=timezone.utc)
    anchor_time = datetime(2026, 1, 1, 10, 30, tzinfo=timezone.utc)
    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    engine.state["hour_anchor"] = {
        "hour_utc": datetime(2026, 1, 1, 10, tzinfo=timezone.utc).isoformat(),
        "sample_time_utc": anchor_time.isoformat(),
        "reading_times_utc": {
            key: anchor_time.isoformat() for key in entities
        },
        "readings": {key: value for key, (_, _, value) in entities.items()},
        "energy_so_far": {key: 0.0 for key in entities},
        "energy_coverage_hours": {key: 0.0 for key in entities},
        "energy_coverage": {key: False for key in entities},
        "coverage": "partial",
    }
    states = {
        entity: {"state": str(value + (0.25 if key == "heating_el" else 0.0))}
        for key, (_, entity, value) in entities.items()
    }
    states["power"] = {"state": "1000"}
    electrical = {
        "forecast": [{"value": 1.0}],
        "forecast_total_el_kwh": [1.0],
        "forecast_heat_th_kwh": [0.1],
        "forecast_heat_el_kwh": [1.0],
        "forecast_defrost_el_kwh": [0.0],
        "forecast_dhw_el_kwh": [0.0],
        "forecast_standby_el_kwh": [0.0],
    }

    engine._correct_current_slot(
        now=now,
        states=states,
        ha_client=SimpleNamespace(
            get_state=lambda entity, current: current[entity]["state"]
        ),
        deltas={},
        electrical=electrical,
        heat_result={"heat_kwh": [0.1]},
        outdoor=[0.0],
        remaining_hours=0.25,
    )

    assert electrical["forecast_heat_el_kwh"][0] == pytest.approx(0.5)


def test_power_projection_uses_only_remaining_time_and_assigns_active_dhw(
    monkeypatch, tmp_path
):
    from src import config

    monkeypatch.setattr(config, "COP_DEFROST_HANDLING", "net")
    monkeypatch.setattr(config, "COP_DEFROST_EL_IN_HEATING_COUNTER", "true")
    monkeypatch.setattr(config, "POWER_CONSUMPTION_ENTITY_ID", "power")
    monkeypatch.setattr(config, "DHW_STATUS_ENTITY_ID", "dhw_mode")
    entities = {
        "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID": "el",
        "HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID": "th",
        "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID": "dhw",
        "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID": "standby",
        "HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID": "defrost_el",
        "HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID": "defrost_th",
    }
    for key, entity in entities.items():
        monkeypatch.setattr(config, key, entity)
    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    electrical = {
        "forecast": [{"value": 0.3}],
        "forecast_total_el_kwh": [0.3],
        "forecast_heat_th_kwh": [0.0],
        "forecast_heat_el_kwh": [0.1],
        "forecast_dhw_el_kwh": [0.1],
        "forecast_standby_el_kwh": [0.1],
    }

    engine._correct_current_slot(
        now=datetime(2026, 1, 1, 12, 58, tzinfo=timezone.utc),
        states={"power": {"state": "1694"}, "dhw_mode": {"state": "on"}},
        ha_client=SimpleNamespace(
            get_state=lambda entity, current: current[entity]["state"]
        ),
        deltas={},
        electrical=electrical,
        heat_result={"heat_kwh": [0.0]},
        outdoor=[0.0],
        remaining_hours=2.0 / 60.0,
    )

    assert electrical["forecast_total_el_kwh"][0] == pytest.approx(
        1.694 * 2.0 / 60.0, abs=0.0001
    )
    assert electrical["forecast_dhw_el_kwh"][0] > electrical[
        "forecast_standby_el_kwh"
    ][0]
    assert electrical["forecast_heat_el_kwh"][0] == 0.0


def test_explicit_defrost_counter_setting_overrides_unknown(monkeypatch, tmp_path):
    from src import config

    monkeypatch.setattr(config, "COP_DEFROST_EL_IN_HEATING_COUNTER", "true")
    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    engine.state["defrost_el_in_heating_counter"] = "unknown"

    engine._record_interval(
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
        states={},
        outdoor_c=0.0,
    )

    assert engine.state["defrost_el_in_heating_counter"] == "true"
    assert engine.state["defrost_el_in_heating_counter_source"] == "configured"


def test_counter_snapshot_uses_sensor_update_time(monkeypatch, tmp_path):
    from src import config

    monkeypatch.setattr(
        config, "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID", "sensor.heat_el"
    )
    sample_time = datetime(2026, 1, 1, 10, 27, tzinfo=timezone.utc)
    engine = HeatForecastEngine(str(tmp_path / "state.json"))

    engine._record_interval(
        now=datetime(2026, 1, 1, 10, 30, tzinfo=timezone.utc),
        states={
            "sensor.heat_el": {
                "state": "12.5",
                "last_updated": sample_time.isoformat(),
            }
        },
        outdoor_c=0.0,
    )

    assert engine.state["counter_snapshots"]["heating_el"]["time_utc"] == (
        sample_time.isoformat()
    )


@pytest.mark.parametrize(
    ("partial_influx", "expected_source"),
    [(False, "homeassistant"), (True, "mixed")],
)
def test_energy_counter_calibration_falls_back_to_home_assistant(
    monkeypatch, tmp_path, partial_influx, expected_source
):
    from src import config

    entity_values = {
        "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID": "sensor.heat_el",
        "HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID": "sensor.heat_th",
        "HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID": "sensor.defrost_el",
        "HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID": "sensor.defrost_th",
        "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID": "sensor.dhw",
        "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID": "sensor.standby",
    }
    for key, value in entity_values.items():
        monkeypatch.setattr(config, key, value)
    monkeypatch.setattr(config, "COP_CALIBRATION_ENABLED", True)
    monkeypatch.setattr(config, "HEAT_FORECAST_MIN_CALIBRATION_HOURS", 48)
    monkeypatch.setattr(config, "HEAT_FORECAST_INFLUX_TIMEOUT_SECONDS", 60)
    now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    entities = list(entity_values.values()) + ["sensor.outdoor"]

    class BrokenInflux:
        calls = 0

        def fetch_counter_history(self, requested, _start, _stop, **_kwargs):
            self.calls += 1
            if not partial_influx:
                raise TimeoutError("Influx timeout")
            entity = requested[0]
            if entity not in ("sensor.heat_el", "sensor.heat_th", "sensor.outdoor"):
                return {entity: []}
            samples = []
            for hour in range(49):
                stamp = now - timedelta(hours=48 - hour)
                base = -2.0 if entity == "sensor.outdoor" else 0.0
                increment = (
                    0.1
                    if entity == "sensor.heat_el"
                    else 0.3
                    if entity == "sensor.heat_th"
                    else 0.0
                )
                samples.append((stamp, base + hour * increment))
            return {entity: samples}

    class HistoryHA:
        def get_history_bulk(self, requested, start, stop):
            response = []
            for entity in requested:
                index = entities.index(entity) if entity in entities else -1
                records = []
                if index >= 0:
                    for hour in range(49):
                        stamp = now - timedelta(hours=48 - hour)
                        if start <= stamp <= stop:
                            base = 0.0 if entity != "sensor.outdoor" else -2.0
                            increment = (
                                0.1
                                if entity == "sensor.heat_el"
                                else 0.3
                                if entity == "sensor.heat_th"
                                else 0.0
                            )
                            records.append(
                                {
                                    "entity_id": entity,
                                    "state": str(base + hour * increment),
                                    "last_changed": stamp.isoformat(),
                                }
                            )
                response.append(records)
            return response

    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    engine._backfill(
        now=now,
        influx_service=BrokenInflux(),
        outdoor_entity_id="sensor.outdoor",
        ha_client=HistoryHA(),
    )

    assert engine.state["calibration_source"] == expected_source
    assert engine.state["calibration_hours"] >= 48
    assert engine.state["cop_bins"]
    assert engine.state["last_backfill_at"]
    assert engine.state["hour_anchor"]["energy_coverage"]["heating_el"] is True
    engine.state["defrost_power_history_attempted_at"] = now.isoformat()
    engine.store.save(engine.state)
    queries_before_restart = engine.state["backfill_attempt_count"]
    restarted = HeatForecastEngine(str(tmp_path / "state.json"))
    influx_queries = BrokenInflux()
    restarted._backfill(
        now=now + timedelta(hours=1),
        influx_service=influx_queries,
        outdoor_entity_id="sensor.outdoor",
        ha_client=HistoryHA(),
    )
    assert restarted.state["last_backfill_at"] == engine.state["last_backfill_at"]
    assert restarted.state["backfill_attempt_count"] == queries_before_restart
    assert restarted.state["cop_bins"] == engine.state["cop_bins"]
    assert influx_queries.calls > 0
    assert restarted.state["hour_anchor"]["hour_utc"] == (
        (now + timedelta(hours=1)).isoformat()
    )
    assert restarted.state["hour_anchor"]["coverage"] == "counter_anchor"


def test_backfill_uses_recent_pre_boundary_counter_anchor(monkeypatch, tmp_path):
    from src import config

    entities = {
        "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID": "sensor.heat_el",
        "HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID": "sensor.heat_th",
        "HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID": "sensor.defrost_el",
        "HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID": "sensor.defrost_th",
        "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID": "sensor.dhw",
        "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID": "sensor.standby",
    }
    for key, entity in entities.items():
        monkeypatch.setattr(config, key, entity)
    monkeypatch.setattr(config, "COP_CALIBRATION_ENABLED", False)
    monkeypatch.setattr(config, "HEAT_FORECAST_MIN_CALIBRATION_HOURS", 0)
    now = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    history_times = [now - timedelta(minutes=20), now - timedelta(minutes=10)]

    class Influx:
        def fetch_counter_history(self, requested, _start, _stop, **_kwargs):
            return {
                entity: [
                    (stamp, index * 0.1 if "outdoor" not in entity else 2.0)
                    for index, stamp in enumerate(history_times)
                ]
                for entity in requested
            }

    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    engine._backfill(
        now=now,
        influx_service=Influx(),
        outdoor_entity_id="sensor.outdoor",
    )

    assert engine.state["hour_anchor"]["reading_times_utc"]["heating_el"] == (
        history_times[-1].isoformat()
    )
    assert engine.state["hour_anchor"]["energy_coverage"]["heating_el"] is False
    assert engine.state["hour_anchor"]["coverage"] == "counter_anchor"


def test_empty_cop_backfill_is_not_marked_successful(monkeypatch, tmp_path):
    from src import config

    entities = {
        "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID": "sensor.heat_el",
        "HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID": "sensor.heat_th",
        "HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID": "sensor.defrost_el",
        "HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID": "sensor.defrost_th",
        "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID": "sensor.dhw",
        "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID": "sensor.standby",
    }
    for key, entity in entities.items():
        monkeypatch.setattr(config, key, entity)
    monkeypatch.setattr(config, "COP_CALIBRATION_ENABLED", True)
    monkeypatch.setattr(config, "HEAT_FORECAST_MIN_CALIBRATION_HOURS", 48)
    monkeypatch.setattr(config, "HEAT_FORECAST_INFLUX_TIMEOUT_SECONDS", 60)
    now = datetime(2026, 1, 3, 0, tzinfo=timezone.utc)
    history_times = [now - timedelta(hours=48 - hour) for hour in range(49)]

    class Influx:
        def fetch_counter_history(self, requested, _start, _stop, **_kwargs):
            return {
                entity: [
                    (stamp, -2.0 if entity == "sensor.outdoor" else 0.0)
                    for stamp in history_times
                ]
                for entity in requested
            }

    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    engine._backfill(
        now=now,
        influx_service=Influx(),
        outdoor_entity_id="sensor.outdoor",
    )

    assert engine.state["cop_bins"] == {}
    assert engine.state["last_backfill_at"] is None
    assert engine.state["backfill_last_error"]


def test_failed_history_backfill_retries_with_backoff_and_does_not_mark_success(
    monkeypatch, tmp_path
):
    from src import config

    monkeypatch.setattr(config, "COP_CALIBRATION_ENABLED", True)
    monkeypatch.setattr(config, "HEAT_FORECAST_INFLUX_TIMEOUT_SECONDS", 60)
    for key in (
        "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID",
        "HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID",
        "HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID",
        "HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID",
        "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID",
        "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID",
    ):
        monkeypatch.setattr(config, key, f"sensor.{key.lower()}")

    class EmptyInflux:
        calls = 0

        def fetch_counter_history(self, entities, _start, _stop, **_kwargs):
            self.calls += 1
            return {entity: [] for entity in entities}

    class EmptyHA:
        def get_history_bulk(self, entities, _start, _stop):
            return [[] for _ in entities]

    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    influx = EmptyInflux()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for attempt_time in (now, now + timedelta(minutes=4), now + timedelta(minutes=6)):
        engine._backfill(
            now=attempt_time,
            influx_service=influx,
            outdoor_entity_id="sensor.outdoor",
            ha_client=EmptyHA(),
        )

    assert engine.state["backfill_attempt_count"] == 2
    assert engine.state["backfill_last_error"]
    assert engine.state["last_backfill_at"] is None


def test_price_profile_reuses_same_hour_today_until_tomorrow_updates(
    monkeypatch, tmp_path
):
    from src import config, price_optimizer, pv_trajectory

    monkeypatch.setattr(config, "ELECTRICITY_PRICE_ENABLED", True)
    monkeypatch.setattr(config, "PRICE_COST_MODE_ENABLED", False)
    monkeypatch.setattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False)
    monkeypatch.setattr(config, "PV_SURPLUS_CHEAP_ENABLED", False)
    monkeypatch.setattr(
        price_optimizer.PriceOptimizer, "refresh_prices_if_needed", lambda *_: None
    )
    monkeypatch.setattr(
        pv_trajectory,
        "get_forecast_trajectory_state",
        lambda *_args: SimpleNamespace(active=False),
    )
    start = datetime(2026, 1, 1, 0, tzinfo=ZoneInfo("Europe/Berlin"))
    optimizer = price_optimizer.PriceOptimizer()
    optimizer._price_tz = ZoneInfo("Europe/Berlin")
    optimizer._price_entries = [
        (start + timedelta(hours=hour), 0.05 + hour * 0.01)
        for hour in range(24)
    ]
    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    engine._price_optimizer = optimizer
    forecast_starts = [start + timedelta(hours=22 + hour) for hour in range(7)]

    _, first_sources = engine._target_offsets(
        starts=forecast_starts,
        pv_electrical=[0.0] * len(forecast_starts),
        ha_client=object(),
    )
    assert engine._price_source_per_hour[:2] == ["actual", "actual"]
    assert engine._price_source_per_hour[2:] == ["reused_today"] * 5
    assert first_sources[2:] == ["reused_today"] * 5
    assert engine._prices_reused_hours == 5

    optimizer._price_entries.extend(
        (start + timedelta(days=1, hours=hour), 0.2)
        for hour in range(24)
    )
    _, updated_sources = engine._target_offsets(
        starts=forecast_starts,
        pv_electrical=[0.0] * len(forecast_starts),
        ha_client=object(),
    )
    assert engine._price_source_per_hour == ["actual"] * 7
    assert updated_sources == ["actual"] * 7


def test_cost_mode_evaluates_full_reused_price_lookahead(monkeypatch, tmp_path):
    from src import config, price_optimizer, pv_trajectory

    monkeypatch.setattr(config, "ELECTRICITY_PRICE_ENABLED", True)
    monkeypatch.setattr(config, "PRICE_COST_MODE_ENABLED", True)
    monkeypatch.setattr(config, "HEAT_FORECAST_REUSE_PRICES", True)
    monkeypatch.setattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False)
    monkeypatch.setattr(config, "PV_SURPLUS_CHEAP_ENABLED", False)
    monkeypatch.setattr(
        price_optimizer.PriceOptimizer, "refresh_prices_if_needed", lambda *_: None
    )
    monkeypatch.setattr(
        price_optimizer.PriceOptimizer,
        "get_hourly_grid_prices",
        lambda _self, count, _start: [0.1, 0.2] + [None] * (count - 2),
    )
    monkeypatch.setattr(
        pv_trajectory,
        "get_forecast_trajectory_state",
        lambda *_args: SimpleNamespace(active=False),
    )
    observed_entries = []

    def inspect_cost_window(optimizer, *_args, **_kwargs):
        observed_entries.append(list(optimizer._price_entries))
        return {"offset": 0.1}

    monkeypatch.setattr(
        price_optimizer.PriceOptimizer,
        "compute_cost_target_offset",
        inspect_cost_window,
    )
    start = datetime(2026, 1, 1, 0, tzinfo=ZoneInfo("Europe/Berlin"))
    optimizer = price_optimizer.PriceOptimizer()
    optimizer._price_tz = ZoneInfo("Europe/Berlin")
    optimizer._price_entries = [
        (start + timedelta(hours=hour), 0.05 + hour * 0.01)
        for hour in range(24)
    ]
    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    engine._price_optimizer = optimizer
    starts = [start + timedelta(hours=22 + hour) for hour in range(4)]

    engine._target_offsets(
        starts=starts,
        pv_electrical=[0.0] * len(starts),
        ha_client=object(),
    )

    assert observed_entries
    assert any(
        optimizer._to_local(stamp).date() == (start.date() + timedelta(days=1))
        and optimizer._to_local(stamp).hour in (0, 1)
        for stamp, _ in observed_entries[0]
    )
    assert engine._price_source_per_hour == [
        "actual",
        "actual",
        "reused_today",
        "reused_today",
    ]


def test_missing_prices_use_neutral_offsets_and_none_source(monkeypatch, tmp_path):
    from src import config, price_optimizer, pv_trajectory

    monkeypatch.setattr(config, "ELECTRICITY_PRICE_ENABLED", True)
    monkeypatch.setattr(config, "PRICE_COST_MODE_ENABLED", False)
    monkeypatch.setattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False)
    monkeypatch.setattr(config, "PV_SURPLUS_CHEAP_ENABLED", False)
    monkeypatch.setattr(
        price_optimizer.PriceOptimizer, "refresh_prices_if_needed", lambda *_: None
    )
    monkeypatch.setattr(
        pv_trajectory,
        "get_forecast_trajectory_state",
        lambda *_args: SimpleNamespace(active=False),
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    engine = HeatForecastEngine(str(tmp_path / "state.json"))

    offsets, sources = engine._target_offsets(
        starts=[start, start + timedelta(hours=1)],
        pv_electrical=[0.0, 0.0],
        ha_client=object(),
    )

    assert offsets == [0.0, 0.0]
    assert sources == ["none", "none"]
    assert engine._price_source_per_hour == ["none", "none"]


def test_async_forecast_dispatch_does_not_wait_for_history_timeout(monkeypatch):
    import threading
    import time
    from src import config, heat_forecast

    monkeypatch.setattr(config, "HEAT_FORECAST_ENABLED", True)
    monkeypatch.setattr(config, "COP_CALIBRATION_ENABLED", False)
    gate = threading.Event()

    class SlowEngine:
        def update_and_publish(self, **_kwargs):
            gate.wait(60)

    executor = None
    monkeypatch.setattr(heat_forecast, "_forecast_engine", SlowEngine())
    monkeypatch.setattr(heat_forecast, "_forecast_executor", None)
    monkeypatch.setattr(heat_forecast, "_forecast_future", None)
    started = time.monotonic()
    heat_forecast.publish_heat_forecast_async(
        ha_client=object(),
        states={},
        wrapper=object(),
    )
    elapsed = time.monotonic() - started
    future = heat_forecast._forecast_future
    executor = heat_forecast._forecast_executor
    gate.set()
    assert future is not None
    future.result(timeout=2)
    executor.shutdown(wait=True)
    assert elapsed < 5


def test_state_store_migrates_v1_and_recovers_corrupt_primary_from_backup(tmp_path):
    import json

    path = tmp_path / "forecast.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "counter_snapshots": {},
                "cop_bins": {},
                "daily_energy": {},
                "last_backfill_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    store = ForecastStateStore(str(path))
    migrated = store.load()
    assert migrated["schema_version"] == 2
    assert migrated["calibration_source"] == "defaults"
    assert migrated["last_backfill_at"] is None

    store.save(migrated)
    path.write_text("{", encoding="utf-8")
    assert store.load()["schema_version"] == 2


def test_first_forecast_hour_advances_only_its_remaining_fraction():
    arguments = {
        "outdoor_c": [-10.0],
        "pv_gain_kw": [0.0],
        "fireplace_kw": [0.0],
        "target_c": [22.0],
        "room_c": 18.0,
        "slab_c": 18.0,
        "room_capacity_kwh_k": 5.8,
        "slab_capacity_kwh_k": 3.9,
        "heat_loss_kw_k": 0.13,
        "slab_to_room_kw_k": 0.2,
        "max_hp_kw": 8.0,
        "step_minutes": 60,
    }
    full = simulate_heating_energy(**arguments)
    partial = simulate_heating_energy(**arguments, first_hour_fraction=0.25)

    assert partial["heat_kwh"][0] == pytest.approx(full["heat_kwh"][0] * 0.25)
    assert partial["room_end_c"] != pytest.approx(full["room_end_c"])
    assert partial["slab_end_c"] != pytest.approx(full["slab_end_c"])
