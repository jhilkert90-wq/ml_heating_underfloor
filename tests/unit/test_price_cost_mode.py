"""Tests for cost-aware price mode (PV / feed-in aware target offset)."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src import config
from src.price_optimizer import HPPowerTracker, PriceOptimizer

UTC = timezone.utc
NOW = datetime(2026, 1, 15, 22, 0, tzinfo=UTC)  # 22:00, night


def _optimizer(prices_by_hour):
    opt = PriceOptimizer()
    start = NOW.replace(hour=0)
    opt._price_entries = [
        (start + timedelta(hours=h), p) for h, p in enumerate(prices_by_hour)
    ]
    opt._price_tz = UTC
    return opt


def _patches(**extra):
    base = dict(
        FEED_IN_TARIFF_EUR_KWH=0.079,
        PRICE_BAND_UP_C=0.5,
        PRICE_BAND_DOWN_C=0.2,
        PRICE_COST_LOOKAHEAD_HOURS=24,
        PRICE_COST_MIN_SPREAD_EUR_KWH=0.02,
    )
    base.update(extra)
    from contextlib import ExitStack
    st = ExitStack()
    for k, v in base.items():
        st.enter_context(patch.object(config, k, v, create=True))
    return st


def _day_prices(night, day):
    """48 h: hours 0-5 and 22-23 night, 6-21 day (both days)."""
    one = [night if (h < 6 or h >= 22) else day for h in range(24)]
    return one + one


class TestCostOffset:
    def test_cheap_night_but_free_pv_day_no_preheat(self):
        # Night grid 0.12, day grid 0.30 but PV covers the HP fully by day
        # → day effective price 0.079 < night 0.12 → no pre-heating at night.
        opt = _optimizer(_day_prices(0.12, 0.30))
        pv_hourly = [0.0] * 8 + [5000.0] * 16  # PV from 06:00 (h=8 from 22:00)
        with _patches():
            res = opt.compute_cost_target_offset(pv_hourly, 1.5, NOW)
        assert res["offset"] <= 0.0

    def test_expensive_day_no_sun_cheap_night_preheats(self):
        opt = _optimizer(_day_prices(0.10, 0.40))
        with _patches():
            res = opt.compute_cost_target_offset([0.0] * 24, 1.5, NOW)
        assert res["offset"] > 0.0
        assert res["offset"] <= 0.5

    def test_expensive_hour_gets_setback_limited_by_band(self):
        opt = _optimizer(_day_prices(0.10, 0.40))
        noon = NOW.replace(hour=10)
        with _patches():
            res = opt.compute_cost_target_offset([0.0] * 24, 1.5, noon)
        assert -0.2 <= res["offset"] < 0.0

    def test_flat_prices_no_shift(self):
        opt = _optimizer([0.25] * 48)
        with _patches():
            res = opt.compute_cost_target_offset([0.0] * 24, 1.5, NOW)
        assert res["offset"] == 0.0

    def test_below_min_spread_no_shift(self):
        opt = _optimizer(_day_prices(0.25, 0.26))
        with _patches():
            res = opt.compute_cost_target_offset([0.0] * 24, 1.5, NOW)
        assert res["offset"] == 0.0

    def test_feed_in_caps_effective_price(self):
        opt = _optimizer([0.05] * 48)  # grid cheaper than feed-in
        with _patches():
            res = opt.compute_cost_target_offset([9000.0] * 24, 1.5, NOW)
        assert res["p_eff_now"] == pytest.approx(0.05)

    def test_horizon_truncated_when_prices_end(self):
        opt = _optimizer([0.1] * 24 + [0.3] * 6)  # data ends 06:00 day 2
        prices = opt.get_hourly_grid_prices(24, NOW)
        assert len(prices) == 8

    def test_no_prices_returns_none(self):
        opt = PriceOptimizer()
        with _patches():
            assert opt.compute_cost_target_offset([], 1.5, NOW) is None


def _feat(**kw):
    base = dict(
        power_consumption=1800.0, thermal_power_kw=5.0, dhw_heating=0.0,
        dhw_disinfection=0.0, dhw_boost_heater=0.0, defrosting=0.0,
        minutes_since_defrost=999.0,
    )
    base.update(kw)
    return base


class TestHPPowerTracker:
    def test_fallback_without_samples(self):
        with patch.object(config, "HP_EXPECTED_POWER_FALLBACK_KW", 2.2, create=True):
            assert HPPowerTracker().expected_power_kw(NOW) == 2.2

    def test_accepts_heating_samples_and_averages(self):
        t = HPPowerTracker()
        assert t.record(_feat(power_consumption=1000.0), "heating", NOW)
        assert t.record(_feat(power_consumption=2000.0), "heating", NOW)
        assert t.expected_power_kw(NOW) == pytest.approx(1.5)

    @pytest.mark.parametrize("flag", [
        "dhw_heating", "dhw_disinfection", "dhw_boost_heater", "defrosting",
    ])
    def test_rejects_dhw_and_defrost(self, flag):
        assert not HPPowerTracker().record(_feat(**{flag: 1.0}), "heating", NOW)

    def test_rejects_defrost_recovery_grace(self):
        assert not HPPowerTracker().record(
            _feat(minutes_since_defrost=5.0), "heating", NOW
        )

    def test_rejects_hp_off_or_not_heating(self):
        t = HPPowerTracker()
        assert not t.record(_feat(power_consumption=50.0), "heating", NOW)
        assert not t.record(_feat(thermal_power_kw=0.0), "heating", NOW)

    def test_cooling_requires_negative_thermal_power(self):
        t = HPPowerTracker()
        assert not t.record(_feat(thermal_power_kw=3.0), "cooling", NOW)
        assert t.record(_feat(thermal_power_kw=-3.0), "cooling", NOW)

    def test_old_samples_pruned(self):
        t = HPPowerTracker()
        t.record(_feat(power_consumption=3000.0), "heating", NOW - timedelta(hours=10))
        t.record(_feat(power_consumption=1000.0), "heating", NOW)
        assert t.expected_power_kw(NOW) == pytest.approx(1.0)


class TestCycleIntegration:
    def test_attach_cost_offset_sets_price_data(self):
        from types import SimpleNamespace
        from src import cycle_routes
        from src.price_optimizer import get_hp_power_tracker

        opt = _optimizer(_day_prices(0.10, 0.40))
        ctx = SimpleNamespace(
            features_dict=dict(_feat(), pv_now_electrical=0.0),
            all_states={},
            climate_mode="heating",
            price_data={"current_price": 0.1, "today": [0.1, 0.4]},
        )
        get_hp_power_tracker()._samples.clear()
        with _patches(), patch(
            "src.price_optimizer.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = NOW
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            cycle_routes._attach_cost_offset(ctx, opt)
        assert "cost_target_offset" in ctx.price_data
        assert ctx.price_data["cost_target_offset"] > 0.0
        get_hp_power_tracker()._samples.clear()


class TestPvForecastAvailability:
    def _states(self, hours, start):
        from datetime import timedelta
        watts = {
            (start + timedelta(minutes=15 * i)).isoformat(): 1000.0 * (i // 4 + 1)
            for i in range(hours * 4)
        }
        return {config.PV_FORECAST_ENTITY_ID: {"attributes": {"watts": watts}}}

    def test_uses_all_available_hours_beyond_trajectory_horizon(self):
        from src.physics_features import parse_hourly_pv_forecast
        with patch.object(config, "PV_FORECAST_ENTITY_ID", "sensor.pv", create=True):
            states = {"sensor.pv": self._states(30, NOW)[config.PV_FORECAST_ENTITY_ID]}
            hourly = parse_hourly_pv_forecast(states, 24, NOW)
        assert len(hourly) == 24  # not limited to 4 or 12

    def test_truncated_where_forecast_ends(self):
        from src.physics_features import parse_hourly_pv_forecast
        with patch.object(config, "PV_FORECAST_ENTITY_ID", "sensor.pv", create=True):
            states = {"sensor.pv": self._states(10, NOW)[config.PV_FORECAST_ENTITY_ID]}
            hourly = parse_hourly_pv_forecast(states, 24, NOW)
        assert 8 <= len(hourly) <= 10

    def test_no_forecast_entity_returns_empty(self):
        from src.physics_features import parse_hourly_pv_forecast
        assert parse_hourly_pv_forecast({}, 24, NOW) == []

    def test_window_cut_to_pv_availability(self):
        opt = _optimizer(_day_prices(0.10, 0.40))
        with _patches():
            res = opt.compute_cost_target_offset([0.0] * 6, 1.5, NOW)
        assert res["hours"] == 6
        assert res["pv_hours_available"] == 6


class TestCostModeSensors:
    def test_publishes_four_individual_sensors(self):
        from src.ha_client import HAClient
        client = HAClient.__new__(HAClient)
        calls = []
        client.set_state = lambda eid, state, attrs, round_digits=None: calls.append((eid, state, attrs))
        info = {
            "p_eff_now": 0.079, "p_grid_now": 0.3, "pv_share_now": 1.0,
            "p_eff_mean": 0.2, "p_eff_min": 0.079, "p_eff_max": 0.4,
            "offset": 0.5, "hp_power_kw": 1.8, "hours": 24.0,
            "price_hours_available": 30.0, "pv_hours_available": 24.0,
        }
        client.publish_cost_mode_sensors(info, hp_samples=5)
        ids = {c[0] for c in calls}
        assert ids == {
            "sensor.ml_heating_price_effective",
            "sensor.ml_heating_cost_offset",
            "sensor.ml_heating_hp_expected_power",
            "sensor.ml_heating_price_window_hours",
        }
        power = next(c for c in calls if c[0].endswith("hp_expected_power"))
        assert power[1] == 1.8 and power[2]["source"] == "measured"
