"""Tests for the electricity price optimizer module.

Covers:
- Price classification (CHEAP / NORMAL / EXPENSIVE)
- Binary search target offset calculation
- Trajectory correction thresholds
- Feature flag disabled → zero behaviour change
- Edge cases (empty prices, flat prices, single price)
- Integration with model_wrapper.calculate_optimal_outlet_temp
- Tibber service cache: refresh logic, time-based lookup, 15/60 min
"""
import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone, date
from unittest.mock import patch, MagicMock

from src.price_optimizer import (
    PriceLevel,
    PriceOptimizer,
    get_price_optimizer,
)
from src import config, model_wrapper, unified_thermal_state
from src.model_wrapper import simplified_outlet_prediction


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def optimizer():
    """A PriceOptimizer with default percentiles (33/67) and offset 0.2."""
    return PriceOptimizer(
        cheap_percentile=33,
        expensive_percentile=67,
        target_offset=0.2,
        expensive_overshoot=0.2,
    )


@pytest.fixture
def daily_prices():
    """Typical Tibber daily price list simulating a realistic day."""
    # 24 hourly prices, rough shape: night is cheap, afternoon expensive
    return [
        0.05, 0.04, 0.03, 0.03, 0.04, 0.06,   # 00-05: night cheap
        0.08, 0.12, 0.15, 0.14, 0.12, 0.10,   # 06-11: morning ramp
        0.09, 0.10, 0.14, 0.18, 0.22, 0.25,   # 12-17: afternoon peak
        0.20, 0.15, 0.12, 0.10, 0.08, 0.06,   # 18-23: evening cooldown
    ]


@pytest.fixture
def clean_state():
    """Clean model wrapper state for integration tests."""
    test_state_file = "thermal_state_price_test.json"
    if os.path.exists(test_state_file):
        os.remove(test_state_file)
    model_wrapper._enhanced_model_wrapper_instance = None
    unified_thermal_state._thermal_state_manager = None
    manager = unified_thermal_state.ThermalStateManager(
        state_file=test_state_file
    )
    unified_thermal_state._thermal_state_manager = manager
    yield
    if os.path.exists(test_state_file):
        os.remove(test_state_file)
    model_wrapper._enhanced_model_wrapper_instance = None
    unified_thermal_state._thermal_state_manager = None


# ──────────────────────────────────────────────────────────────────
# Price Classification
# ──────────────────────────────────────────────────────────────────

class TestClassifyPrice:
    """Test PriceOptimizer.classify_price()."""

    def test_cheap_price_classified_correctly(self, optimizer, daily_prices):
        """Prices in bottom 33% → CHEAP."""
        level = optimizer.classify_price(0.03, daily_prices)
        assert level == PriceLevel.CHEAP

    def test_normal_price_classified_correctly(self, optimizer, daily_prices):
        """Mid-range prices → NORMAL."""
        level = optimizer.classify_price(0.11, daily_prices)
        assert level == PriceLevel.NORMAL

    def test_expensive_price_classified_correctly(self, optimizer, daily_prices):
        """Prices in top 33% → EXPENSIVE."""
        level = optimizer.classify_price(0.22, daily_prices)
        assert level == PriceLevel.EXPENSIVE

    def test_boundary_cheap_threshold(self, optimizer, daily_prices):
        """At exactly the cheap percentile boundary → CHEAP (<=)."""
        prices = np.array(daily_prices)
        cheap_thresh = float(np.percentile(prices, 33))
        level = optimizer.classify_price(cheap_thresh, daily_prices)
        assert level == PriceLevel.CHEAP

    def test_boundary_expensive_threshold(self, optimizer, daily_prices):
        """At exactly the expensive percentile boundary → EXPENSIVE (>=)."""
        prices = np.array(daily_prices)
        exp_thresh = float(np.percentile(prices, 67))
        level = optimizer.classify_price(exp_thresh, daily_prices)
        assert level == PriceLevel.EXPENSIVE

    def test_empty_prices_returns_normal(self, optimizer):
        """Empty price list → defaults to NORMAL."""
        level = optimizer.classify_price(0.10, [])
        assert level == PriceLevel.NORMAL

    def test_single_price_returns_normal(self, optimizer):
        """Only one price available → defaults to NORMAL."""
        level = optimizer.classify_price(0.10, [0.10])
        assert level == PriceLevel.NORMAL

    def test_flat_prices_all_equal(self, optimizer):
        """All prices identical → all thresholds equal → CHEAP (<=)."""
        flat = [0.10] * 24
        level = optimizer.classify_price(0.10, flat)
        # P33 = P67 = 0.10, current <= cheap_threshold → CHEAP
        assert level == PriceLevel.CHEAP

    def test_two_prices_minimum(self, optimizer):
        """Two prices is the minimum for percentile calc."""
        level = optimizer.classify_price(0.05, [0.05, 0.20])
        assert level == PriceLevel.CHEAP

    def test_caches_last_price_and_level(self, optimizer, daily_prices):
        """After classify, _last_price and _last_level are updated."""
        optimizer.classify_price(0.03, daily_prices)
        assert optimizer._last_price == 0.03
        assert optimizer._last_level == PriceLevel.CHEAP

    def test_caches_thresholds(self, optimizer, daily_prices):
        """After classify, _last_thresholds are updated."""
        optimizer.classify_price(0.10, daily_prices)
        cheap_t, exp_t = optimizer._last_thresholds
        assert cheap_t > 0
        assert exp_t > cheap_t


# ──────────────────────────────────────────────────────────────────
# Target Offset
# ──────────────────────────────────────────────────────────────────

class TestGetTargetOffset:
    """Test PriceOptimizer.get_target_offset()."""

    def test_cheap_returns_positive_offset(self, optimizer):
        offset = optimizer.get_target_offset(PriceLevel.CHEAP)
        assert offset == pytest.approx(0.2)

    def test_normal_returns_zero(self, optimizer):
        offset = optimizer.get_target_offset(PriceLevel.NORMAL)
        assert offset == 0.0

    def test_expensive_returns_negative_offset(self, optimizer):
        offset = optimizer.get_target_offset(PriceLevel.EXPENSIVE)
        assert offset == pytest.approx(-0.2)

    def test_custom_offset(self):
        opt = PriceOptimizer(target_offset=0.5)
        assert opt.get_target_offset(PriceLevel.CHEAP) == pytest.approx(0.5)
        assert opt.get_target_offset(PriceLevel.EXPENSIVE) == pytest.approx(-0.5)


# ──────────────────────────────────────────────────────────────────
# Trajectory Thresholds
# ──────────────────────────────────────────────────────────────────

class TestGetTrajectoryThresholds:
    """Test PriceOptimizer.get_trajectory_thresholds()."""

    def test_cheap_same_as_normal(self, optimizer):
        """Cheap uses the same relaxed +0.5 as normal."""
        cheap = optimizer.get_trajectory_thresholds(PriceLevel.CHEAP, 22.0)
        normal = optimizer.get_trajectory_thresholds(PriceLevel.NORMAL, 22.0)
        assert cheap["immediate_overshoot"] == pytest.approx(22.1)
        assert cheap["future_overshoot"] == pytest.approx(22.5)
        assert cheap == normal

    def test_expensive_tighter_future(self, optimizer):
        """Expensive tightens future overshoot to target + 0.2."""
        thresholds = optimizer.get_trajectory_thresholds(
            PriceLevel.EXPENSIVE, 22.0
        )
        assert thresholds["immediate_overshoot"] == pytest.approx(22.1)
        assert thresholds["future_overshoot"] == pytest.approx(22.2)

    def test_normal_default_future(self, optimizer):
        thresholds = optimizer.get_trajectory_thresholds(
            PriceLevel.NORMAL, 21.5
        )
        assert thresholds["future_overshoot"] == pytest.approx(22.0)

    def test_custom_expensive_overshoot(self):
        opt = PriceOptimizer(expensive_overshoot=0.3)
        thresholds = opt.get_trajectory_thresholds(PriceLevel.EXPENSIVE, 22.0)
        assert thresholds["future_overshoot"] == pytest.approx(22.3)


# ──────────────────────────────────────────────────────────────────
# get_price_info export
# ──────────────────────────────────────────────────────────────────

class TestGetPriceInfo:
    """Test PriceOptimizer.get_price_info()."""

    def test_price_info_after_classify(self, optimizer, daily_prices):
        optimizer.classify_price(0.22, daily_prices)
        info = optimizer.get_price_info()
        assert info["price_eur_kwh"] == 0.22
        assert info["price_level"] == "expensive"
        assert info["price_target_offset"] == pytest.approx(-0.2)
        assert info["price_cheap_threshold"] > 0
        assert info["price_expensive_threshold"] > info["price_cheap_threshold"]

    def test_price_info_defaults_before_classify(self, optimizer):
        """Before any classification, defaults are sane."""
        info = optimizer.get_price_info()
        assert info["price_level"] == "normal"
        assert info["price_target_offset"] == 0.0


# ──────────────────────────────────────────────────────────────────
# Feature Flag Disabled → Zero Behaviour Change
# ──────────────────────────────────────────────────────────────────

class TestFeatureFlagDisabled:
    """When ELECTRICITY_PRICE_ENABLED=False, price has no effect."""

    def test_disabled_flag_no_target_shift(self, clean_state):
        """With price disabled, target is not shifted even if price data
        is present in features."""
        wrapper = model_wrapper.get_enhanced_model_wrapper()
        config.CYCLE_INTERVAL_MINUTES = 15

        base_features = {
            "target_temp": 22.0,
            "indoor_temp_lag_30m": 21.5,
            "outdoor_temp": 5.0,
            "_electricity_price": {
                "current_price": 0.03,
                "today": [0.03] * 12 + [0.20] * 12,
                "tomorrow": [],
            },
        }

        with patch.object(config, "ELECTRICITY_PRICE_ENABLED", False):
            _, meta = wrapper.calculate_optimal_outlet_temp(base_features)

        # target_adjusted should equal original when disabled
        assert meta["target_temp_adjusted"] == pytest.approx(22.0)
        assert meta["target_temp_original"] == pytest.approx(22.0)

    def test_disabled_flag_no_price_data(self, clean_state):
        """Without price data, target is unchanged and no errors."""
        wrapper = model_wrapper.get_enhanced_model_wrapper()
        config.CYCLE_INTERVAL_MINUTES = 15

        features = {
            "target_temp": 22.0,
            "indoor_temp_lag_30m": 21.5,
            "outdoor_temp": 5.0,
        }

        with patch.object(config, "ELECTRICITY_PRICE_ENABLED", False):
            _, meta = wrapper.calculate_optimal_outlet_temp(features)

        assert meta["target_temp_adjusted"] == pytest.approx(22.0)


# ──────────────────────────────────────────────────────────────────
# Integration: Price affects outlet temperature
# ──────────────────────────────────────────────────────────────────

class TestPriceAffectsOutlet:
    """When enabled, different price levels produce different outlet temps."""

    def _make_features(self, current_price, daily_prices):
        return {
            "target_temp": 22.0,
            "indoor_temp_lag_30m": 21.8,
            "outdoor_temp": 0.0,
            "_electricity_price": {
                "current_price": current_price,
                "today": daily_prices,
                "tomorrow": [],
            },
        }

    def test_cheap_gives_higher_outlet_than_normal(self, clean_state):
        """CHEAP electricity → shifted target up → higher outlet."""
        wrapper = model_wrapper.get_enhanced_model_wrapper()
        config.CYCLE_INTERVAL_MINUTES = 15

        # Cheap: current at bottom of distribution
        daily = [0.05] * 8 + [0.15] * 8 + [0.25] * 8
        features_cheap = self._make_features(0.05, daily)
        features_normal = self._make_features(0.15, daily)

        with patch.object(config, "ELECTRICITY_PRICE_ENABLED", True):
            outlet_cheap, meta_cheap = (
                wrapper.calculate_optimal_outlet_temp(features_cheap)
            )
            outlet_normal, meta_normal = (
                wrapper.calculate_optimal_outlet_temp(features_normal)
            )

        assert meta_cheap["target_temp_adjusted"] > meta_normal["target_temp_adjusted"]
        assert outlet_cheap > outlet_normal

    def test_expensive_gives_lower_outlet_than_normal(self, clean_state):
        """EXPENSIVE electricity → shifted target down → lower outlet."""
        wrapper = model_wrapper.get_enhanced_model_wrapper()
        config.CYCLE_INTERVAL_MINUTES = 15

        daily = [0.05] * 8 + [0.15] * 8 + [0.25] * 8
        features_expensive = self._make_features(0.25, daily)
        features_normal = self._make_features(0.15, daily)

        with patch.object(config, "ELECTRICITY_PRICE_ENABLED", True):
            outlet_exp, meta_exp = (
                wrapper.calculate_optimal_outlet_temp(features_expensive)
            )
            outlet_normal, meta_normal = (
                wrapper.calculate_optimal_outlet_temp(features_normal)
            )

        assert meta_exp["target_temp_adjusted"] < meta_normal["target_temp_adjusted"]
        assert outlet_exp < outlet_normal

    def test_metadata_contains_price_fields(self, clean_state):
        """Metadata should contain price classification info."""
        wrapper = model_wrapper.get_enhanced_model_wrapper()
        config.CYCLE_INTERVAL_MINUTES = 15

        daily = [0.05] * 8 + [0.15] * 8 + [0.25] * 8
        features = self._make_features(0.05, daily)

        with patch.object(config, "ELECTRICITY_PRICE_ENABLED", True):
            _, meta = wrapper.calculate_optimal_outlet_temp(features)

        assert "price_level" in meta
        assert meta["price_level"] == "cheap"
        assert "price_eur_kwh" in meta
        assert meta["price_eur_kwh"] == pytest.approx(0.05)
        assert "price_target_offset" in meta


# ──────────────────────────────────────────────────────────────────
# Integration: simplified_outlet_prediction passes price_data
# ──────────────────────────────────────────────────────────────────

class TestSimplifiedOutletPredictionWithPrice:
    """Test that simplified_outlet_prediction correctly forwards price_data."""

    def test_price_data_injected_into_features(self, clean_state):
        """price_data parameter should become _electricity_price in features."""
        config.CYCLE_INTERVAL_MINUTES = 15
        features_df = pd.DataFrame([{
            "indoor_temp_lag_30m": 21.5,
            "outdoor_temp": 5.0,
        }])
        price_data = {
            "current_price": 0.20,
            "today": [0.10] * 24,
            "tomorrow": [],
        }

        with patch.object(config, "ELECTRICITY_PRICE_ENABLED", True):
            outlet, confidence, meta = simplified_outlet_prediction(
                features_df, 21.5, 22.0, price_data=price_data
            )

        assert outlet is not None
        assert confidence > 0

    def test_none_price_data_has_no_effect(self, clean_state):
        """price_data=None should not inject _electricity_price."""
        config.CYCLE_INTERVAL_MINUTES = 15
        features_df = pd.DataFrame([{
            "indoor_temp_lag_30m": 21.5,
            "outdoor_temp": 5.0,
        }])

        outlet, confidence, meta = simplified_outlet_prediction(
            features_df, 21.5, 22.0, price_data=None
        )

        assert outlet is not None
        assert meta["target_temp_adjusted"] == pytest.approx(22.0)


# ──────────────────────────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────────────────────────

class TestSingleton:
    """Test the module-level singleton."""

    def test_get_price_optimizer_returns_same_instance(self):
        # Reset singleton
        import src.price_optimizer as po
        po._optimizer = None
        opt1 = get_price_optimizer()
        opt2 = get_price_optimizer()
        assert opt1 is opt2
        po._optimizer = None  # cleanup


# ══════════════════════════════════════════════════════════════════
# Tibber service cache tests
# ══════════════════════════════════════════════════════════════════

# Helpers ─────────────────────────────────────────────────────────

LOCAL_TZ = datetime.now().astimezone().tzinfo  # system local TZ


def _make_hourly_entries(
    day: date, tz: timezone = None, base_price: float = 0.10
) -> list:
    """Generate 24 hourly Tibber-style entries for a given day."""
    if tz is None:
        tz = LOCAL_TZ
    entries = []
    for h in range(24):
        dt = datetime(day.year, day.month, day.day, h, 0, 0, tzinfo=tz)
        price = base_price + 0.01 * h
        entries.append({"start_time": dt.isoformat(), "price": price})
    return entries


def _make_15min_entries(
    day: date, tz: timezone = None, base_price: float = 0.10
) -> list:
    """Generate 96 quarter-hourly entries for a given day."""
    if tz is None:
        tz = LOCAL_TZ
    entries = []
    for slot in range(96):
        h, m = divmod(slot * 15, 60)
        dt = datetime(day.year, day.month, day.day, h, m, 0, tzinfo=tz)
        price = base_price + 0.001 * slot
        entries.append({"start_time": dt.isoformat(), "price": price})
    return entries


# ─────────────────────────────────────────────────────────────────

class TestParsePriceEntries:
    """Test _parse_price_entries with both resolutions."""

    def test_parse_60min_entries(self, optimizer):
        today = date(2026, 4, 14)
        raw = _make_hourly_entries(today)
        now_local = datetime(2026, 4, 14, 10, 0, tzinfo=LOCAL_TZ)
        optimizer._parse_price_entries(raw, now_local)
        assert len(optimizer._price_entries) == 24

    def test_parse_15min_entries(self, optimizer):
        today = date(2026, 4, 14)
        raw = _make_15min_entries(today)
        now_local = datetime(2026, 4, 14, 10, 0, tzinfo=LOCAL_TZ)
        optimizer._parse_price_entries(raw, now_local)
        assert len(optimizer._price_entries) == 96

    def test_parse_sets_cache_metadata(self, optimizer):
        today = date(2026, 4, 14)
        raw = _make_hourly_entries(today)
        now_local = datetime(2026, 4, 14, 10, 0, tzinfo=LOCAL_TZ)
        optimizer._parse_price_entries(raw, now_local)
        assert optimizer._cache_date == today
        assert optimizer._cache_time is not None

    def test_parse_detects_tomorrow(self, optimizer):
        today = date(2026, 4, 14)
        tomorrow = date(2026, 4, 15)
        raw = _make_hourly_entries(today) + _make_hourly_entries(tomorrow)
        now_local = datetime(2026, 4, 14, 14, 0, tzinfo=LOCAL_TZ)
        optimizer._parse_price_entries(raw, now_local)
        assert optimizer._has_tomorrow is True
        assert len(optimizer._price_entries) == 48

    def test_parse_no_tomorrow(self, optimizer):
        today = date(2026, 4, 14)
        raw = _make_hourly_entries(today)
        now_local = datetime(2026, 4, 14, 10, 0, tzinfo=LOCAL_TZ)
        optimizer._parse_price_entries(raw, now_local)
        assert optimizer._has_tomorrow is False

    def test_parse_skips_invalid_entries(self, optimizer):
        today = date(2026, 4, 14)
        raw = _make_hourly_entries(today)
        raw.append({"start_time": "invalid", "price": 0.5})
        raw.append({"price": 0.5})  # missing start_time
        now_local = datetime(2026, 4, 14, 10, 0, tzinfo=LOCAL_TZ)
        optimizer._parse_price_entries(raw, now_local)
        assert len(optimizer._price_entries) == 24  # only valid ones

    def test_parse_sets_price_tz_from_entries(self, optimizer):
        """_price_tz is extracted from the first Tibber entry's tzinfo."""
        mesz = timezone(timedelta(hours=2))
        today = date(2026, 4, 14)
        raw = _make_hourly_entries(today, tz=mesz)
        now_local = datetime(2026, 4, 14, 10, 0, tzinfo=mesz)
        optimizer._parse_price_entries(raw, now_local)
        assert optimizer._price_tz == mesz

    def test_to_local_uses_price_tz(self, optimizer):
        """_to_local converts to the Tibber timezone, not system TZ."""
        mesz = timezone(timedelta(hours=2))
        today = date(2026, 4, 14)
        raw = _make_hourly_entries(today, tz=mesz)
        now_local = datetime(2026, 4, 14, 10, 0, tzinfo=mesz)
        optimizer._parse_price_entries(raw, now_local)

        # A UTC midnight should become 02:00 MESZ
        utc_midnight = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)
        local = optimizer._to_local(utc_midnight)
        assert local.utcoffset() == timedelta(hours=2)
        assert local.hour == 2

    def test_to_local_fallback_when_no_entries(self, optimizer):
        """_to_local falls back to system TZ before any entries are parsed."""
        utc_time = datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc)
        local = optimizer._to_local(utc_time)
        # Should not crash, returns some timezone-aware datetime
        assert local.tzinfo is not None

    def test_day_boundary_with_mesz(self, optimizer):
        """Day filtering uses Tibber's TZ, not UTC — key bug-fix test."""
        mesz = timezone(timedelta(hours=2))
        today = date(2026, 4, 14)
        tomorrow = date(2026, 4, 15)
        raw = _make_hourly_entries(today, tz=mesz) + _make_hourly_entries(
            tomorrow, tz=mesz
        )
        now_local = datetime(2026, 4, 14, 23, 30, tzinfo=mesz)
        optimizer._parse_price_entries(raw, now_local)

        # At 23:30 MESZ (21:30 UTC), today's prices should be April 14
        prices = optimizer.get_today_prices(now_local)
        assert len(prices) == 24


class TestGetCurrentPrice:
    """Test time-based current price lookup."""

    def test_60min_finds_correct_hour(self, optimizer):
        today = date(2026, 4, 14)
        raw = _make_hourly_entries(today)
        now_local = datetime(2026, 4, 14, 10, 0, tzinfo=LOCAL_TZ)
        optimizer._parse_price_entries(raw, now_local)

        # At 10:30, should find the 10:00 entry
        query = datetime(2026, 4, 14, 10, 30, tzinfo=LOCAL_TZ)
        price = optimizer.get_current_price(query)
        expected = 0.10 + 0.01 * 10  # hour 10
        assert price == pytest.approx(expected)

    def test_15min_finds_correct_slot(self, optimizer):
        today = date(2026, 4, 14)
        raw = _make_15min_entries(today)
        now_local = datetime(2026, 4, 14, 10, 0, tzinfo=LOCAL_TZ)
        optimizer._parse_price_entries(raw, now_local)

        # At 10:35 → slot 10:30 (slot index 42)
        query = datetime(2026, 4, 14, 10, 35, tzinfo=LOCAL_TZ)
        price = optimizer.get_current_price(query)
        expected = 0.10 + 0.001 * 42  # slot 42
        assert price == pytest.approx(expected)

    def test_before_first_entry_returns_none(self, optimizer):
        today = date(2026, 4, 14)
        raw = _make_hourly_entries(today)
        now_local = datetime(2026, 4, 14, 10, 0, tzinfo=LOCAL_TZ)
        optimizer._parse_price_entries(raw, now_local)

        # Query at 23:00 the day before → no entry
        query = datetime(2026, 4, 13, 23, 0, tzinfo=LOCAL_TZ)
        assert optimizer.get_current_price(query) is None

    def test_empty_cache_returns_none(self, optimizer):
        assert optimizer.get_current_price() is None

    def test_timezone_aware_comparison(self, optimizer):
        """UTC query against local-tz entries works correctly."""
        today = date(2026, 4, 14)
        raw = _make_hourly_entries(today)
        now_local = datetime(2026, 4, 14, 10, 0, tzinfo=LOCAL_TZ)
        optimizer._parse_price_entries(raw, now_local)

        # Convert a local 10:30 to UTC and query — should find 10:00 entry
        local_time = datetime(2026, 4, 14, 10, 30, tzinfo=LOCAL_TZ)
        query_utc = local_time.astimezone(timezone.utc)
        price = optimizer.get_current_price(query_utc)
        expected = 0.10 + 0.01 * 10
        assert price == pytest.approx(expected)


class TestGetTodayPrices:
    """Test today-only filtering for percentile calculation."""

    def test_filters_to_today_60min(self, optimizer):
        today = date(2026, 4, 14)
        tomorrow = date(2026, 4, 15)
        raw = _make_hourly_entries(today) + _make_hourly_entries(tomorrow)
        now_local = datetime(2026, 4, 14, 14, 0, tzinfo=LOCAL_TZ)
        optimizer._parse_price_entries(raw, now_local)

        query = datetime(2026, 4, 14, 14, 0, tzinfo=LOCAL_TZ)
        prices = optimizer.get_today_prices(query)
        # Should only return today's 24 entries
        assert len(prices) == 24

    def test_filters_to_today_15min(self, optimizer):
        today = date(2026, 4, 14)
        tomorrow = date(2026, 4, 15)
        raw = _make_15min_entries(today) + _make_15min_entries(tomorrow)
        now_local = datetime(2026, 4, 14, 14, 0, tzinfo=LOCAL_TZ)
        optimizer._parse_price_entries(raw, now_local)

        query = datetime(2026, 4, 14, 14, 0, tzinfo=LOCAL_TZ)
        prices = optimizer.get_today_prices(query)
        assert len(prices) == 96

    def test_empty_cache_returns_empty(self, optimizer):
        assert optimizer.get_today_prices() == []


class TestGetPriceDataForFeatures:
    """Test the dict format returned for model_wrapper consumption."""

    def test_returns_correct_format(self, optimizer):
        today = date.today()
        raw = _make_hourly_entries(today)
        now_local = datetime(today.year, today.month, today.day, 10, 0, tzinfo=LOCAL_TZ)
        optimizer._parse_price_entries(raw, now_local)

        data = optimizer.get_price_data_for_features()
        assert data is not None
        assert "current_price" in data
        assert "today" in data
        assert isinstance(data["today"], list)
        assert len(data["today"]) == 24

    def test_returns_none_when_empty(self, optimizer):
        assert optimizer.get_price_data_for_features() is None


class TestRefreshPricesIfNeeded:
    """Test cache refresh trigger logic."""

    def _make_mock_ha(self, entries):
        ha = MagicMock()
        ha.call_tibber_get_prices.return_value = entries
        return ha

    def test_refresh_on_empty_cache(self, optimizer):
        today = date.today()
        ha = self._make_mock_ha(_make_hourly_entries(today))
        optimizer.refresh_prices_if_needed(ha)
        ha.call_tibber_get_prices.assert_called_once()
        assert len(optimizer._price_entries) == 24

    def test_no_refresh_when_fresh(self, optimizer):
        today = date.today()
        tomorrow = today + timedelta(days=1)
        # Include tomorrow so the after-13:00 trigger doesn't fire
        entries = _make_hourly_entries(today) + _make_hourly_entries(tomorrow)
        ha = self._make_mock_ha(entries)
        # First call populates cache
        optimizer.refresh_prices_if_needed(ha)
        # Second call should not re-fetch (cache is fresh)
        optimizer.refresh_prices_if_needed(ha)
        assert ha.call_tibber_get_prices.call_count == 1

    def test_refresh_on_day_change(self, optimizer):
        yesterday = date.today() - timedelta(days=1)
        ha = self._make_mock_ha(_make_hourly_entries(date.today()))
        # Simulate stale cache from yesterday
        optimizer._cache_date = yesterday
        optimizer._cache_time = datetime.now(timezone.utc)
        optimizer._price_entries = [(datetime.now(timezone.utc), 0.1)]
        optimizer.refresh_prices_if_needed(ha)
        ha.call_tibber_get_prices.assert_called_once()

    def test_refresh_after_13_for_tomorrow(self, optimizer):
        today = date.today()
        now = datetime.now(timezone.utc)
        ha = self._make_mock_ha(_make_hourly_entries(today))

        # Pre-populate with today-only cache, set cache_time to now
        optimizer._parse_price_entries(
            _make_hourly_entries(today), datetime.now().astimezone()
        )
        optimizer._has_tomorrow = False
        # Force cache_time to be very recent so age check won't trigger
        optimizer._cache_time = now

        # Only triggers if local hour >= 13: monkey-patch _to_local
        fake_local = datetime.now().astimezone().replace(hour=14, minute=0)
        with patch.object(
            PriceOptimizer, "_to_local", return_value=fake_local
        ):
            optimizer.refresh_prices_if_needed(ha)

        ha.call_tibber_get_prices.assert_called_once()

    def test_service_failure_preserves_cache(self, optimizer):
        today = date.today()
        # Populate cache first
        entries = _make_hourly_entries(today)
        optimizer._parse_price_entries(
            entries, datetime.now().astimezone()
        )
        original_count = len(optimizer._price_entries)

        # Force refresh by clearing cache_date
        optimizer._cache_date = None

        ha = MagicMock()
        ha.call_tibber_get_prices.return_value = None  # service failure
        optimizer.refresh_prices_if_needed(ha)

        # Original cache should remain
        assert len(optimizer._price_entries) == original_count

    def test_auto_detect_first_home(self):
        """call_tibber_get_prices auto-detects first home from dict."""
        from src.ha_client import HAClient
        ha = HAClient.__new__(HAClient)
        ha.url = "http://fake"
        ha.headers = {}

        response = {
            "service_response": {
                "prices": {
                    "My Home": [
                        {"start_time": "2026-04-14 00:00:00+01:00",
                         "price": 0.15},
                    ]
                }
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        mock_resp.raise_for_status = MagicMock()

        with patch("src.ha_client.requests.post", return_value=mock_resp):
            result = ha.call_tibber_get_prices(
                "2026-04-14 00:00:00", "2026-04-16 00:00:00"
            )

        assert result is not None
        assert len(result) == 1
        assert result[0]["price"] == 0.15
