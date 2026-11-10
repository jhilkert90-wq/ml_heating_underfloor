import math
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.heat_forecast import (
    ForecastStateStore,
    HeatForecastEngine,
    compute_counter_delta,
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
    assert result["forecast_heat_el_kwh"] == pytest.approx([0.55])
    assert result["forecast_defrost_el_kwh"] == pytest.approx([0.0289])


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
    minutes = [
        (start.replace(minute=minute), 600.0) for minute in range(4)
    ]
    thermal = [
        (start.replace(minute=minute), -1200.0 if minute in (1, 2) else 0.0)
        for minute in range(4)
    ]
    states = [
        (start.replace(minute=minute), 1.0 if minute in (1, 2) else 0.0)
        for minute in range(4)
    ]

    class FakeInflux:
        def fetch_minute_history(self, entity_ids, _start, _stop, *, agg_fn):
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
    assert engine.state["defrost_power_history"]["event_count"] == 1


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

    monkeypatch.setattr(
        price_optimizer, "get_price_optimizer", lambda: BrokenOptimizer()
    )
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


def test_update_publishes_hourly_thermal_and_electrical_profiles(
    monkeypatch, tmp_path
):
    from src import config, ha_client

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

        def get_hourly_forecast_strict(self, _hours):
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
        "binary_sensor.fireplace": {"state": "off"},
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
    client = FakeHAClient()
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
    assert attributes["current_hour_estimate"] == "degraded_missing_counter_anchor"
    assert attributes["includes"] == ["heating", "dhw", "standby"]


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
