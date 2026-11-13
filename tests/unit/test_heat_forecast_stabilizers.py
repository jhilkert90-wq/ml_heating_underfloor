"""Tests for the WP load forecast stabilisation (Prompt 1 steps 1-6)."""

from datetime import datetime, timedelta, timezone

import pytest

from src import forecast_accuracy as fa
from src.heat_forecast import (
    HeatForecastEngine,
    add_cop_observation,
    apply_start_estimate,
    fit_energy_weighted_cop,
    hourly_electrical_forecast,
)
from src.heat_forecast_stabilizers import (
    blend_slot0_projection,
    record_power_sample,
    shift_held_forecast,
    stabilize_window_sum,
    start_window_power_kw,
    windowed_mean_kw,
)

HOUR = datetime(2026, 11, 12, 9, tzinfo=timezone.utc)


def _anchor(at, el, th, offset_min=0):
    stamp = (at + timedelta(minutes=offset_min)).isoformat()
    return {
        "hour_utc": at.isoformat(),
        "coverage": "counter_anchor",
        "reading_times_utc": {"heating_el": stamp, "heating_th": stamp},
        "readings": {"heating_el": el, "heating_th": th},
    }


class TestAnchors:
    def test_tolerance_accepts_small_offset_and_rejects_large(self):
        old, new = _anchor(HOUR, 100.0, 300.0, -3), _anchor(
            HOUR + timedelta(hours=1), 100.5, 302.0, -2
        )
        assert fa.actual_from_anchors(old, new) == (HOUR, 0.5, 2.0)
        far = _anchor(HOUR + timedelta(hours=1), 100.5, 302.0, -20)
        assert fa.actual_from_anchors(old, far) is None

    def test_refresh_interpolates_counter_to_full_hour(self, monkeypatch, tmp_path):
        from src import config

        monkeypatch.setattr(config, "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID", "el")
        monkeypatch.setattr(config, "HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID", "th")
        for key in (
            "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID",
            "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID",
            "HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID",
            "HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID",
        ):
            monkeypatch.setattr(config, key, "")
        now = HOUR + timedelta(minutes=20)

        class Influx:
            gap = 20

            def fetch_counter_history(self, requested, *_a, **_k):
                before = HOUR - timedelta(minutes=10)
                after = before + timedelta(minutes=self.gap)
                return {
                    e: [(before, 10.0), (after, 10.0 + self.gap / 10.0)]
                    for e in requested
                }

        engine = HeatForecastEngine(str(tmp_path / "s.json"))
        influx = Influx()
        engine._refresh_hour_anchor(now=now, influx_service=influx, ha_client=None)
        anchor = engine.state["hour_anchor"]
        assert anchor["readings"]["heating_el"] == pytest.approx(11.0)
        assert anchor["reading_times_utc"]["heating_el"] == HOUR.isoformat()
        assert anchor["energy_coverage"]["heating_el"] is True

        wide = HeatForecastEngine(str(tmp_path / "w.json"))
        influx.gap = 90
        wide._refresh_hour_anchor(now=now, influx_service=influx, ha_client=None)
        assert wide.state["hour_anchor"]["reading_times_utc"][
            "heating_el"
        ] == (HOUR - timedelta(minutes=10)).isoformat()


class TestStartWindow:
    def test_mean_replaces_spike_and_is_capped(self):
        state = {}
        for minute, kw in ((30, 2.0), (20, 2.0), (10, 8.0), (0, 2.0)):
            record_power_sample(
                state, HOUR - timedelta(minutes=minute), el_kw=None, th_kw=kw
            )
        mean = windowed_mean_kw(state, HOUR, key="th", window_minutes=45)
        assert mean == pytest.approx(3.5)
        power, source = start_window_power_kw(
            instantaneous_kw=8.0, mean_kw=mean, reference_kw=2.0,
            cap_factor=1.5, max_kw=8.0,
        )
        assert (power, source) == (3.0, "window_mean_capped")

    def test_runtime_limit_and_blend_out(self):
        base = [0.1] * 8
        result = apply_start_estimate(
            base, [3.0] * 8, minutes_to_start=0, runtime_min=600,
            first_hour_fraction=1.0, max_runtime_h=2.0, blend_hours=2.0,
        )
        assert result[:2] == [3.0, 3.0]
        assert result[2] == pytest.approx(0.1 + 2.9 * 0.75)
        assert result[3] == pytest.approx(0.1 + 2.9 * 0.25)
        assert result[4] == pytest.approx(0.1)


class TestBlockade:
    def test_shift_converts_slot0_rate(self):
        held = shift_held_forecast(
            [0.5, 1.0, 2.0, 3.0], first_hour_fraction_at_generation=0.5,
            hours_elapsed=0, remaining_hours=0.25,
        )
        assert held == pytest.approx([0.25, 1.0, 2.0, 3.0])
        shifted = shift_held_forecast(
            [0.5, 1.0, 2.0, 3.0], first_hour_fraction_at_generation=0.5,
            hours_elapsed=1, remaining_hours=0.5,
        )
        assert shifted == pytest.approx([0.5, 2.0, 3.0, 3.0])

    def test_block_state_grace_and_end(self, tmp_path, monkeypatch):
        from src import config

        monkeypatch.setattr(config, "HEAT_FORECAST_BLOCK_GRACE_MINUTES", 10)
        engine = HeatForecastEngine(str(tmp_path / "s.json"))
        assert engine._update_block_state(now=HOUR, active=True, reason="dhw") == (True, "dhw")
        later = HOUR + timedelta(minutes=5)
        assert engine._update_block_state(now=later, active=False, reason="") == (True, "dhw_grace")
        end = HOUR + timedelta(minutes=20)
        assert engine._update_block_state(now=end, active=False, reason="") == (False, "")

    def test_duplicate_publish_suppressed_only_within_interval(self, tmp_path, monkeypatch):
        from src import config

        monkeypatch.setattr(config, "HEAT_FORECAST_BLOCK_PUBLISH_INTERVAL_SECONDS", 120)
        engine = HeatForecastEngine(str(tmp_path / "s.json"))
        engine.state["last_publish_utc"] = HOUR.isoformat()
        assert engine._publish_recently(HOUR + timedelta(seconds=30)) is True
        assert engine._publish_recently(HOUR + timedelta(seconds=300)) is False

    def test_held_forecast_expires(self, tmp_path, monkeypatch):
        from src import config

        monkeypatch.setattr(config, "HEAT_FORECAST_BLOCK_HOLD_MAX_HOURS", 3.0)
        engine = HeatForecastEngine(str(tmp_path / "s.json"))
        engine.state["last_good_forecast"] = {
            "generated_at": HOUR.isoformat(),
            "first_hour_fraction": 1.0,
            "heat_kwh": [1.0, 2.0, 3.0],
        }
        assert engine._held_forecast(HOUR + timedelta(hours=1), 1.0, 3) == [2.0, 3.0, 3.0]
        assert engine._held_forecast(HOUR + timedelta(hours=5), 1.0, 3) is None


class TestCop:
    def test_weight_share_limits_outlier_bin(self):
        samples = [
            {"outdoor_c": -5.0, "heat_kwh": 30.0, "electric_kwh": 10.0},
            {"outdoor_c": 0.0, "heat_kwh": 35.0, "electric_kwh": 10.0},
            {"outdoor_c": 5.0, "heat_kwh": 40.0, "electric_kwh": 10.0},
            {"outdoor_c": 12.0, "heat_kwh": 400.0, "electric_kwh": 100.0},
        ]
        plain = fit_energy_weighted_cop(samples)
        robust = fit_energy_weighted_cop(samples, max_weight_share=0.3)
        assert abs(robust["slope"] - 0.1) < abs(plain["slope"] - 0.1)

    def test_weight_share_respects_final_clipped_total(self):
        samples = [
            {"outdoor_c": 5.0, "heat_kwh": 8000.0, "electric_kwh": 1000.0},
            {"outdoor_c": 5.0, "heat_kwh": 2.0, "electric_kwh": 1.0},
            {"outdoor_c": 5.0, "heat_kwh": 4.0, "electric_kwh": 1.0},
        ]
        robust = fit_energy_weighted_cop(samples, max_weight_share=0.4)
        dominant_share = (robust["intercept"] - 3.0) / 5.0
        assert dominant_share == pytest.approx(0.4)

    def test_bin_width_groups_temperatures(self):
        state = {}
        for temp in (4.2, 5.7):
            add_cop_observation(
                state, outdoor_c=temp, heat_kwh=3.0, electric_kwh=1.0,
                elapsed_hours=1.0, timestamp=HOUR, half_life_days=45,
                bin_width_k=2.0,
            )
        assert list(state["cop_bins"]) == ["4"]

    def test_online_correction_is_bounded_and_scales_electricity(self):
        state = {"forecast_accuracy": {"history": [
            {"hour_utc": (HOUR - timedelta(hours=i)).isoformat(), "lead": 1,
             "el": 1.0, "th": 3.0, "act_el": 1.5, "act_th": 3.0}
            for i in range(1, 15)
        ]}}
        assert fa.correction_factor(state, HOUR, limit=0.2) == pytest.approx(1.2)
        assert fa.correction_factor(state, HOUR, min_samples=50) == 1.0
        starts = [HOUR]
        kwargs = dict(
            starts=starts, heat_th_kwh=[3.0], outdoor_c=[5.0],
            cop_curve={"intercept": 3.0, "slope": 0.0, "calibrated": False},
            dhw_el_kwh=[0.0], standby_el_kwh=[0.0],
        )
        base = hourly_electrical_forecast(**kwargs)["forecast_heat_el_kwh"][0]
        scaled = hourly_electrical_forecast(
            **kwargs, el_correction_factor=1.2
        )["forecast_heat_el_kwh"][0]
        assert scaled == pytest.approx(base * 1.2)


class TestSlot0AndSum:
    def test_slot0_blend_capped_at_simulation(self):
        assert blend_slot0_projection(
            measured_mean_kw=4.0, model_rate_kw=1.0, weight=0.5, cap_factor=1.5
        ) == pytest.approx(1.5)
        assert blend_slot0_projection(
            measured_mean_kw=1.0, model_rate_kw=2.0, weight=0.5, cap_factor=1.5
        ) == pytest.approx(1.5)

    def test_sum_bound_to_baseline_and_smoothed(self):
        heat = [0.0] + [2.0] * 24
        baseline = [0.0] + [1.0] * 24
        result, info = stabilize_window_sum(heat, baseline, previous_sum=None)
        assert sum(result[1:]) == pytest.approx(24 * 1.15)
        assert info["reason"] == "baseline_upper_bound"
        smooth, info = stabilize_window_sum(
            heat, baseline, previous_sum=24.0, smoothing_alpha=0.5
        )
        assert sum(smooth[1:]) == pytest.approx(24 + 0.5 * (24 * 1.15 - 24))

    def test_jump_below_hysteresis_keeps_previous_sum(self):
        heat = [0.0] + [1.0] * 24
        result, info = stabilize_window_sum(
            heat, heat, previous_sum=24.4, hysteresis=0.03
        )
        assert sum(result[1:]) == pytest.approx(24.4)
        assert info["reason"] == "hysteresis_hold"


def test_dhw_blockade_holds_last_good_simulation(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from src import config, ha_client, heat_forecast

    options = {
        "OUTDOOR_TEMP_ENTITY_ID": "sensor.outdoor",
        "INDOOR_TEMP_ENTITY_ID": "sensor.indoor",
        "TARGET_INDOOR_TEMP_ENTITY_ID": "sensor.target",
        "HEATING_STATUS_ENTITY_ID": "climate.heating",
        "FIREPLACE_STATUS_ENTITY_ID": "binary_sensor.fireplace",
        "PV_POWER_ENTITY_ID": "sensor.pv",
        "PV_FORECAST_ENTITY_ID": "",
        "FLOW_RATE_ENTITY_ID": "",
        "INLET_TEMP_ENTITY_ID": "sensor.inlet",
        "ACTUAL_OUTLET_TEMP_ENTITY_ID": "sensor.outlet",
        "POWER_CONSUMPTION_ENTITY_ID": "sensor.power",
        "HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID": "",
        "HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID": "",
        "HEAT_FORECAST_DHW_COUNTER_ENTITY_ID": "",
        "HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID": "",
        "HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID": "",
        "HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID": "",
        "DHW_STATUS_ENTITY_ID": "binary_sensor.dhw",
        "HEAT_FORECAST_ENABLED": True,
        "HEAT_FORECAST_HORIZON_HOURS": 4,
        "COP_CALIBRATION_ENABLED": False,
        "BUILDING_BALANCE_CAPACITY_MODE": "manual",
        "ELECTRICITY_PRICE_ENABLED": False,
        "PRICE_COST_MODE_ENABLED": False,
        "PV_TRAJ_FORECAST_MODE_ENABLED": False,
        "PV_SURPLUS_CHEAP_ENABLED": False,
        "HP_START_ESTIMATE_ENABLED": False,
    }
    for key, value in options.items():
        monkeypatch.setattr(config, key, value)
    monkeypatch.setattr(ha_client, "get_sensor_attributes", lambda _e: {})

    class Client:
        published = []

        def get_state(self, entity_id, states, is_binary=False):
            raw = states.get(entity_id, {}).get("state")
            return str(raw).lower() in ("on", "true", "1") if is_binary else raw

        def get_hourly_forecast_strict(self, _hours, starts=None):
            return [-5.0, -5.0, -5.0]

        def set_state(self, entity_id, state, attributes, round_digits=2):
            self.published.append(attributes)
            return True

    model = SimpleNamespace(
        external_source_weights={"pv": 0.002}, solar_decay_tau_hours=0.5,
        heat_loss_coefficient=0.13, outlet_effectiveness=0.2,
        slab_time_constant_hours=3.0, thermal_time_constant=10.0,
        orchestrator=None,
    )
    wrapper = SimpleNamespace(thermal_model=model, _heating_thermal_model=model)
    states = {
        "sensor.outdoor": {"state": "-5"}, "sensor.indoor": {"state": "20"},
        "sensor.target": {"state": "21"}, "climate.heating": {"state": "heat"},
        "binary_sensor.fireplace": {"state": "off"}, "sensor.pv": {"state": "0"},
        "sensor.inlet": {"state": "28"}, "sensor.outlet": {"state": "30"},
        "sensor.power": {"state": "500"}, "binary_sensor.dhw": {"state": "off"},
    }
    simulations = []
    original = heat_forecast.simulate_heating_energy
    monkeypatch.setattr(
        heat_forecast, "simulate_heating_energy",
        lambda *a, **k: simulations.append(1) or original(*a, **k),
    )
    monkeypatch.setattr(
        HeatForecastEngine, "_dhw_and_standby_profiles",
        lambda self, starts: ([0.0] * len(starts), [0.01] * len(starts), {}),
    )
    client = Client()
    engine = HeatForecastEngine(str(tmp_path / "state.json"))
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    engine.update_and_publish(ha_client=client, states=states, wrapper=wrapper, now=now)
    normal_runs = len(simulations)
    assert normal_runs >= 1
    good_sum = sum(client.published[-1]["forecast_heat_th_kwh"][1:])

    states["binary_sensor.dhw"] = {"state": "on"}
    states["sensor.inlet"] = {"state": "55"}  # DHW would fake a hot slab
    engine.update_and_publish(
        ha_client=client, states=states, wrapper=wrapper,
        now=now + timedelta(minutes=10),
    )
    assert len(simulations) == normal_runs
    assert sum(client.published[-1]["forecast_heat_th_kwh"][1:]) == pytest.approx(
        good_sum, rel=1e-3
    )

    publishes = len(client.published)
    engine.update_and_publish(
        ha_client=client, states=states, wrapper=wrapper,
        now=now + timedelta(minutes=10, seconds=20),
    )
    assert len(client.published) == publishes
