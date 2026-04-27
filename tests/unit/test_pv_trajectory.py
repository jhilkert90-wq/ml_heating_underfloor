"""Unit tests for src/pv_trajectory.py — compute_dynamic_trajectory_steps."""
import pytest
from datetime import datetime
from unittest.mock import patch

from src import config
from src.pv_trajectory import (
    compute_dynamic_trajectory_steps,
    compute_forecast_driven_trajectory_steps,
    is_forecast_trajectory_active,
    _time_of_day_factor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt(hour: int) -> datetime:
    """Build a datetime at the given hour on an arbitrary date."""
    return datetime(2026, 6, 21, hour, 0, 0)


# ---------------------------------------------------------------------------
# Feature flag disabled → returns static TRAJECTORY_STEPS unchanged
# ---------------------------------------------------------------------------

class TestFeatureFlagDisabled:
    def test_disabled_returns_trajectory_steps(self):
        with (
            patch.object(config, "PV_TRAJ_SCALING_ENABLED", False),
            patch.object(config, "TRAJECTORY_STEPS", 5),
        ):
            result = compute_dynamic_trajectory_steps(10000.0, now=_dt(12))
        assert result == 5

    def test_disabled_ignores_pv(self):
        with (
            patch.object(config, "PV_TRAJ_SCALING_ENABLED", False),
            patch.object(config, "TRAJECTORY_STEPS", 3),
        ):
            assert compute_dynamic_trajectory_steps(0.0, now=_dt(12)) == 3
            assert compute_dynamic_trajectory_steps(15000.0, now=_dt(12)) == 3


# ---------------------------------------------------------------------------
# Time-of-day factor
# ---------------------------------------------------------------------------

class TestTimeOfDayFactor:
    def test_morning_hours(self):
        with (
            patch.object(config, "PV_TRAJ_MORNING_FACTOR", 0.5),
            patch.object(config, "PV_TRAJ_MIDDAY_FACTOR", 1.0),
            patch.object(config, "PV_TRAJ_AFTERNOON_FACTOR", 0.75),
            patch.object(config, "PV_TRAJ_NIGHT_FACTOR", 0.0),
        ):
            assert _time_of_day_factor(6) == pytest.approx(0.5)
            assert _time_of_day_factor(10) == pytest.approx(0.5)

    def test_midday_hours(self):
        with (
            patch.object(config, "PV_TRAJ_MORNING_FACTOR", 0.5),
            patch.object(config, "PV_TRAJ_MIDDAY_FACTOR", 1.0),
            patch.object(config, "PV_TRAJ_AFTERNOON_FACTOR", 0.75),
            patch.object(config, "PV_TRAJ_NIGHT_FACTOR", 0.0),
        ):
            assert _time_of_day_factor(11) == pytest.approx(1.0)
            assert _time_of_day_factor(14) == pytest.approx(1.0)

    def test_afternoon_hours(self):
        with (
            patch.object(config, "PV_TRAJ_MORNING_FACTOR", 0.5),
            patch.object(config, "PV_TRAJ_MIDDAY_FACTOR", 1.0),
            patch.object(config, "PV_TRAJ_AFTERNOON_FACTOR", 0.75),
            patch.object(config, "PV_TRAJ_NIGHT_FACTOR", 0.0),
        ):
            assert _time_of_day_factor(15) == pytest.approx(0.75)
            assert _time_of_day_factor(18) == pytest.approx(0.75)

    def test_night_hours(self):
        with (
            patch.object(config, "PV_TRAJ_MORNING_FACTOR", 0.5),
            patch.object(config, "PV_TRAJ_MIDDAY_FACTOR", 1.0),
            patch.object(config, "PV_TRAJ_AFTERNOON_FACTOR", 0.75),
            patch.object(config, "PV_TRAJ_NIGHT_FACTOR", 0.0),
        ):
            assert _time_of_day_factor(19) == pytest.approx(0.0)
            assert _time_of_day_factor(0) == pytest.approx(0.0)
            assert _time_of_day_factor(5) == pytest.approx(0.0)

    def test_boundary_morning_start(self):
        """Hour 6 is morning, hour 5 is night."""
        with (
            patch.object(config, "PV_TRAJ_MORNING_FACTOR", 0.5),
            patch.object(config, "PV_TRAJ_NIGHT_FACTOR", 0.0),
        ):
            assert _time_of_day_factor(6) == pytest.approx(0.5)
            assert _time_of_day_factor(5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_dynamic_trajectory_steps — core interpolation
# ---------------------------------------------------------------------------

_PATCH_ENABLED = patch.object(config, "PV_TRAJ_SCALING_ENABLED", True)


class TestComputeDynamicSteps:

    def _run(self, pv_w, hour, system_kwp=10.0,
             min_s=2, max_s=12,
             morning=0.5, midday=1.0, afternoon=0.75, night=0.0):
        with (
            patch.object(config, "PV_TRAJ_SCALING_ENABLED", True),
            patch.object(config, "PV_TRAJ_SYSTEM_KWP", system_kwp),
            patch.object(config, "PV_TRAJ_MIN_STEPS", min_s),
            patch.object(config, "PV_TRAJ_MAX_STEPS", max_s),
            patch.object(config, "PV_TRAJ_MORNING_FACTOR", morning),
            patch.object(config, "PV_TRAJ_MIDDAY_FACTOR", midday),
            patch.object(config, "PV_TRAJ_AFTERNOON_FACTOR", afternoon),
            patch.object(config, "PV_TRAJ_NIGHT_FACTOR", night),
        ):
            return compute_dynamic_trajectory_steps(pv_w, system_kwp, _dt(hour))

    # --- Night always returns minimum ---
    def test_night_zero_pv_returns_min(self):
        assert self._run(0.0, hour=23) == 2

    def test_night_full_pv_still_returns_min(self):
        # Night factor 0.0 → ratio * 0.0 = 0 → min steps
        assert self._run(10000.0, hour=22) == 2

    # --- Zero PV returns minimum regardless of time ---
    def test_zero_pv_morning_returns_min(self):
        assert self._run(0.0, hour=9) == 2

    def test_zero_pv_midday_returns_min(self):
        assert self._run(0.0, hour=12) == 2

    # --- Full PV at midday returns max ---
    def test_full_pv_midday_returns_max(self):
        # ratio=1.0, factor=1.0 → 2 + round(1.0 * 10) = 12
        assert self._run(10000.0, hour=12) == 12

    # --- Full PV at morning returns mid range ---
    def test_full_pv_morning_clamped(self):
        # ratio=1.0, morning factor=0.5 → 2 + round(0.5 * 10) = 7
        assert self._run(10000.0, hour=8) == 7

    # --- Full PV at afternoon ---
    def test_full_pv_afternoon(self):
        # ratio=1.0, afternoon factor=0.75 → 2 + round(0.75 * 10)
        # Python banker's rounding: round(7.5) = 8 → 2 + 8 = 10
        assert self._run(10000.0, hour=16) == 10

    # --- 15 kWp system example from plan (8 kW at midday) ---
    def test_15kwp_8kw_midday(self):
        # PV ratio = 8000/15000 = 0.533, factor 1.0
        # raw = 2 + round(0.533 * 10) = 2 + round(5.33) = 2 + 5 = 7
        result = self._run(8000.0, hour=12, system_kwp=15.0)
        assert result == 7

    # --- 15 kWp at half capacity in morning ---
    def test_15kwp_half_pv_morning(self):
        # ratio = 7500/15000 = 0.5, factor 0.5
        # raw = 2 + round(0.5 * 0.5 * 10) = 2 + round(2.5) = 2 + 2 = 4
        result = self._run(7500.0, hour=8, system_kwp=15.0)
        assert result == 4

    # --- Result is always within [min, max] ---
    def test_result_clamped_above_max(self):
        # Artificially high PV should not exceed max
        result = self._run(999999.0, hour=12)
        assert result == 12

    def test_result_clamped_below_min(self):
        result = self._run(-100.0, hour=12)
        assert result == 2

    # --- Custom min/max range ---
    def test_custom_min_max(self):
        # min=4, max=8: full PV midday → 4 + round(1.0*4) = 8
        assert self._run(10000.0, hour=12, min_s=4, max_s=8) == 8
        # zero PV → 4
        assert self._run(0.0, hour=12, min_s=4, max_s=8) == 4

    # --- system_kwp default from config ---
    def test_uses_config_system_kwp_when_none(self):
        with (
            patch.object(config, "PV_TRAJ_SCALING_ENABLED", True),
            patch.object(config, "PV_TRAJ_SYSTEM_KWP", 10.0),
            patch.object(config, "PV_TRAJ_MIN_STEPS", 2),
            patch.object(config, "PV_TRAJ_MAX_STEPS", 12),
            patch.object(config, "PV_TRAJ_MIDDAY_FACTOR", 1.0),
            patch.object(config, "PV_TRAJ_MORNING_FACTOR", 0.5),
            patch.object(config, "PV_TRAJ_AFTERNOON_FACTOR", 0.75),
            patch.object(config, "PV_TRAJ_NIGHT_FACTOR", 0.0),
        ):
            # Passing system_kwp=None → should read from config (10.0)
            result = compute_dynamic_trajectory_steps(
                5000.0, system_kwp=None, now=_dt(12)
            )
        # ratio=0.5, factor=1.0 → 2 + round(0.5*10) = 7
        assert result == 7

    # --- Misconfigured max < min is handled gracefully ---
    def test_max_less_than_min_returns_min(self):
        with (
            patch.object(config, "PV_TRAJ_SCALING_ENABLED", True),
            patch.object(config, "PV_TRAJ_SYSTEM_KWP", 10.0),
            patch.object(config, "PV_TRAJ_MIN_STEPS", 8),
            patch.object(config, "PV_TRAJ_MAX_STEPS", 4),  # bad: < min
            patch.object(config, "PV_TRAJ_MIDDAY_FACTOR", 1.0),
            patch.object(config, "PV_TRAJ_MORNING_FACTOR", 0.5),
            patch.object(config, "PV_TRAJ_AFTERNOON_FACTOR", 0.75),
            patch.object(config, "PV_TRAJ_NIGHT_FACTOR", 0.0),
        ):
            result = compute_dynamic_trajectory_steps(
                5000.0, system_kwp=10.0, now=_dt(12)
            )
        # min_steps=8, max_steps clamped to min=8 → always 8
        assert result == 8


# ---------------------------------------------------------------------------
# seasonal_kwp_factor
# ---------------------------------------------------------------------------

from datetime import date
from src.pv_trajectory import seasonal_kwp_factor


class TestSeasonalKwpFactor:
    """Tests for seasonal_kwp_factor() using only stdlib math."""

    # Summer solstice → factor must equal 1.0 (reference point)
    def test_summer_solstice_returns_one(self):
        summer = date(2026, 6, 21)
        factor = seasonal_kwp_factor(summer, latitude_deg=51.0)
        assert factor == pytest.approx(1.0, abs=0.001)

    # Winter solstice → factor clearly below 1.0 for Central Europe
    def test_winter_solstice_below_one(self):
        winter = date(2026, 12, 21)
        factor = seasonal_kwp_factor(winter, latitude_deg=51.0)
        assert factor < 1.0
        assert factor > 0.0

    # Factor always within [min_factor, 1.0]
    def test_factor_bounded_above_by_one(self):
        for doy_date in [date(2026, 1, 1), date(2026, 6, 21), date(2026, 12, 21)]:
            f = seasonal_kwp_factor(doy_date, latitude_deg=48.0)
            assert f <= 1.0 + 1e-9, f"factor={f} exceeds 1.0 for {doy_date}"

    def test_factor_bounded_below_by_min_factor(self):
        winter = date(2026, 12, 21)
        min_f = 0.15
        factor = seasonal_kwp_factor(winter, latitude_deg=65.0, min_factor=min_f)
        assert factor >= min_f

    def test_default_min_factor_applied(self):
        # Default min_factor=0.1 — factor must never be below it
        winter = date(2026, 12, 21)
        factor = seasonal_kwp_factor(winter, latitude_deg=80.0)
        assert factor >= 0.1

    # Equinox: factor should be intermediate (roughly 0.5-0.8 for lat=51)
    def test_equinox_intermediate_value(self):
        equinox = date(2026, 3, 20)
        factor = seasonal_kwp_factor(equinox, latitude_deg=51.0)
        assert 0.3 < factor < 0.95

    # Extreme latitudes don't crash and return bounded values
    def test_equator_latitude(self):
        summer = date(2026, 6, 21)
        factor = seasonal_kwp_factor(summer, latitude_deg=0.0)
        assert 0.0 <= factor <= 1.0

    def test_high_latitude_no_crash(self):
        winter = date(2026, 12, 21)
        factor = seasonal_kwp_factor(winter, latitude_deg=70.0)
        assert 0.0 <= factor <= 1.0

    # Known numerical value for lat=48, Dec 21 (verified against formula)
    def test_known_value_lat48_dec21(self):
        # δ(355) ≈ -23.43°; elev_max = 90 - |48 - (-23.43)| = 90 - 71.43 = 18.57°
        # δ(172) ≈ 23.45°;  elev_max = 90 - |48 - 23.45| = 90 - 24.55 = 65.45°
        # factor = sin(18.57°) / sin(65.45°) ≈ 0.319 / 0.909 ≈ 0.351
        winter = date(2026, 12, 21)
        factor = seasonal_kwp_factor(winter, latitude_deg=48.0, min_factor=0.0)
        assert 0.28 < factor < 0.42, f"Unexpected factor {factor}"


# ---------------------------------------------------------------------------
# compute_dynamic_trajectory_steps with seasonal scaling
# ---------------------------------------------------------------------------

class TestComputeDynamicStepsWithSeasonal:
    """Integration tests for seasonal KWP scaling inside compute_dynamic_trajectory_steps."""

    def _base_patches(self, extra=None):
        """Return a list of patch contexts for the common config attributes."""
        patches = {
            "PV_TRAJ_SCALING_ENABLED": True,
            "PV_TRAJ_SYSTEM_KWP": 10.0,
            "PV_TRAJ_MIN_STEPS": 2,
            "PV_TRAJ_MAX_STEPS": 12,
            "PV_TRAJ_MIDDAY_FACTOR": 1.0,
            "PV_TRAJ_MORNING_FACTOR": 0.5,
            "PV_TRAJ_AFTERNOON_FACTOR": 0.75,
            "PV_TRAJ_NIGHT_FACTOR": 0.0,
            "PV_TRAJ_SEASONAL_SCALING_ENABLED": False,
        }
        if extra:
            patches.update(extra)
        return patches

    def _apply(self, patches):
        """Apply multiple attribute patches via context managers."""
        from contextlib import ExitStack
        stack = ExitStack()
        for attr, val in patches.items():
            stack.enter_context(patch.object(config, attr, val))
        return stack

    def test_seasonal_disabled_behaviour_unchanged(self):
        """When seasonal scaling is off, result matches the non-seasonal path."""
        # Reference: 5000W / 10kWp = 0.5 ratio, midday factor=1.0
        # steps = 2 + round(0.5 * 1.0 * 10) = 7
        ps = self._base_patches()
        with self._apply(ps):
            result = compute_dynamic_trajectory_steps(
                5000.0, system_kwp=10.0,
                now=datetime(2026, 12, 21, 12, 0, 0),
            )
        assert result == 7

    def test_seasonal_enabled_summer_ratio_unchanged(self):
        """On summer solstice, seasonal factor=1.0 → same result as without."""
        ps = self._base_patches({
            "PV_TRAJ_SEASONAL_SCALING_ENABLED": True,
            "PV_TRAJ_LATITUDE": 51.0,
            "PV_TRAJ_SEASONAL_MIN_FACTOR": 0.1,
        })
        # On June 21: factor ≈ 1.0 → effective_kwp ≈ 10.0 → same ratio
        with self._apply(ps):
            result = compute_dynamic_trajectory_steps(
                5000.0, system_kwp=10.0,
                now=datetime(2026, 6, 21, 12, 0, 0),
            )
        # ratio ≈ 0.5, steps = 2 + round(0.5 * 10) = 7
        assert result == 7

    def test_seasonal_enabled_winter_higher_ratio(self):
        """In winter, seasonal factor < 1 → effective_kwp smaller → higher ratio → more steps."""
        # On Dec 21 at lat=51: factor ≈ 0.28-0.35 → effective_kwp ≈ 2.8-3.5 kWp
        # With PV=2000W and 10kWp: without seasonal ratio=0.2
        # With seasonal (factor≈0.31): ratio = 2000 / (10*0.31*1000) ≈ 0.65
        ps = self._base_patches({
            "PV_TRAJ_SEASONAL_SCALING_ENABLED": True,
            "PV_TRAJ_LATITUDE": 51.0,
            "PV_TRAJ_SEASONAL_MIN_FACTOR": 0.1,
        })
        with self._apply(ps):
            steps_winter = compute_dynamic_trajectory_steps(
                2000.0, system_kwp=10.0,
                now=datetime(2026, 12, 21, 12, 0, 0),
            )
        without = self._base_patches()
        with self._apply(without):
            steps_no_seasonal = compute_dynamic_trajectory_steps(
                2000.0, system_kwp=10.0,
                now=datetime(2026, 12, 21, 12, 0, 0),
            )
        assert steps_winter > steps_no_seasonal

    def test_seasonal_enabled_min_factor_prevents_zero(self):
        """Even at extremely high latitudes, min_factor prevents zero denominator."""
        ps = self._base_patches({
            "PV_TRAJ_SEASONAL_SCALING_ENABLED": True,
            "PV_TRAJ_LATITUDE": 85.0,
            "PV_TRAJ_SEASONAL_MIN_FACTOR": 0.1,
        })
        with self._apply(ps):
            # Should not raise ZeroDivisionError
            result = compute_dynamic_trajectory_steps(
                500.0, system_kwp=10.0,
                now=datetime(2026, 12, 21, 12, 0, 0),
            )
        assert 2 <= result <= 12


# ---------------------------------------------------------------------------
# Forecast-driven trajectory mode
# ---------------------------------------------------------------------------

def _fc_patches(extra=None):
    """Return a dict of config patches for forecast-driven mode tests."""
    base = {
        "PV_TRAJ_SCALING_ENABLED": True,
        "PV_TRAJ_FORECAST_MODE_ENABLED": True,
        "PV_TRAJ_THRESHOLD_W": 3000.0,
        "PV_TRAJ_ZERO_W": 50.0,
        "PV_TRAJ_MIN_STEPS": 2,
        "PV_TRAJ_MAX_STEPS": 12,
    }
    if extra:
        base.update(extra)
    return base


def _apply_patches(patches: dict):
    """Context manager that applies multiple ``patch.object(config, ...)`` patches."""
    from contextlib import ExitStack
    stack = ExitStack()
    for attr, val in patches.items():
        stack.enter_context(patch.object(config, attr, val))
    return stack


class TestForecastDrivenTrajectorySteps:
    """Tests for compute_forecast_driven_trajectory_steps() and helpers."""

    # Forecast with 9 daylight hours then night
    _FC_9_THEN_NIGHT = [6000, 5500, 5000, 4000, 3000, 2000, 1000, 200, 60, 0, 0, 0]

    def test_activation_pv_above_threshold_sunset_in_horizon(self):
        """Active when PV >= threshold AND at least one forecast slot <= zero_w."""
        with _apply_patches(_fc_patches()):
            steps = compute_forecast_driven_trajectory_steps(
                5000.0, self._FC_9_THEN_NIGHT
            )
        # 9 consecutive entries > 50 W (within MAX_STEPS=12 horizon)
        assert steps == 9

    def test_no_activation_pv_below_threshold(self):
        """PV below threshold → inactive, returns MIN_STEPS."""
        with _apply_patches(_fc_patches()):
            steps = compute_forecast_driven_trajectory_steps(
                2999.0, self._FC_9_THEN_NIGHT
            )
        assert steps == 2

    def test_no_activation_forecast_no_sunset_in_horizon(self):
        """No sunset within MAX_STEPS horizon → inactive, returns MIN_STEPS."""
        fc_all_high = [8000.0] * 12  # all above zero_w — no sunset
        with _apply_patches(_fc_patches()):
            steps = compute_forecast_driven_trajectory_steps(5000.0, fc_all_high)
        assert steps == 2

    def test_night_mode_pv_below_zero_w(self):
        """Current PV below zero_w → night mode → MIN_STEPS regardless of forecast."""
        fc = [8000.0] * 11 + [0.0]
        with _apply_patches(_fc_patches()):
            steps = compute_forecast_driven_trajectory_steps(40.0, fc)
        assert steps == 2

    def test_step_count_equals_consecutive_pv_hours(self):
        """Steps equal the number of consecutive forecast entries above zero_w."""
        # 5 daylight hours, then night
        fc = [5000, 4000, 3000, 500, 100, 0, 0, 0, 0, 0, 0, 0]
        with _apply_patches(_fc_patches()):
            steps = compute_forecast_driven_trajectory_steps(5000.0, fc)
        assert steps == 5

    def test_steps_at_max_minus_one_is_not_reduced(self):
        """When all but the last horizon slot have PV, step count = MAX_STEPS - 1 (no reduction)."""
        # With MAX_STEPS=4: horizon=[5000, 4000, 3000, 0], consecutive=3
        fc = [5000.0, 4000.0, 3000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        with _apply_patches(_fc_patches({"PV_TRAJ_MIN_STEPS": 2, "PV_TRAJ_MAX_STEPS": 4})):
            steps = compute_forecast_driven_trajectory_steps(5000.0, fc)
        assert steps == 3

    def test_steps_within_bounds_at_max_horizon_minus_one(self):
        """11 consecutive daylight hours (max=12) → 11 steps, within bounds."""
        fc = [6000.0] * 11 + [0.0]  # 11 daylight then night; max=12
        with _apply_patches(_fc_patches()):
            steps = compute_forecast_driven_trajectory_steps(5000.0, fc)
        assert steps == 11

    def test_steps_clamped_to_min(self):
        """Only 1 daylight hour → clamped up to MIN_STEPS=2."""
        fc = [500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        with _apply_patches(_fc_patches({"PV_TRAJ_MIN_STEPS": 2})):
            steps = compute_forecast_driven_trajectory_steps(5000.0, fc)
        assert steps == 2

    def test_boundary_first_forecast_slot_is_zero(self):
        """First forecast slot <= zero_w → remaining_pv_hours=0 → MIN_STEPS."""
        fc = [0, 5000, 5000, 5000, 0, 0, 0, 0, 0, 0, 0, 0]
        with _apply_patches(_fc_patches()):
            steps = compute_forecast_driven_trajectory_steps(5000.0, fc)
        assert steps == 2

    def test_none_forecast_treated_as_empty(self):
        """None forecast (not yet available) → no activation → MIN_STEPS."""
        with _apply_patches(_fc_patches()):
            steps = compute_forecast_driven_trajectory_steps(5000.0, None)
        assert steps == 2

    def test_empty_forecast_list(self):
        """Empty forecast list → no activation → MIN_STEPS."""
        with _apply_patches(_fc_patches()):
            steps = compute_forecast_driven_trajectory_steps(5000.0, [])
        assert steps == 2

    def test_custom_min_max_steps(self):
        """Custom min/max clamps are respected."""
        fc = [5000, 4000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 2 daylight hours
        with _apply_patches(_fc_patches({"PV_TRAJ_MIN_STEPS": 4, "PV_TRAJ_MAX_STEPS": 8})):
            steps = compute_forecast_driven_trajectory_steps(5000.0, fc)
        # 2 < min=4 → clamped to 4
        assert steps == 4

    def test_compute_dynamic_delegates_to_forecast_mode(self):
        """compute_dynamic_trajectory_steps delegates when forecast mode enabled."""
        fc = [5000, 4000, 3000, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 3 remaining hours
        with _apply_patches(_fc_patches()):
            steps = compute_dynamic_trajectory_steps(5000.0, pv_forecast=fc)
        assert steps == 3

    def test_is_forecast_trajectory_active_true(self):
        """is_forecast_trajectory_active returns True when conditions met."""
        fc = self._FC_9_THEN_NIGHT
        with _apply_patches(_fc_patches()):
            result = is_forecast_trajectory_active(5000.0, fc)
        assert result is True

    def test_is_forecast_trajectory_active_pv_below_threshold(self):
        """is_forecast_trajectory_active returns False when PV below threshold."""
        fc = self._FC_9_THEN_NIGHT
        with _apply_patches(_fc_patches()):
            result = is_forecast_trajectory_active(2000.0, fc)
        assert result is False

    def test_is_forecast_trajectory_active_no_sunset(self):
        """is_forecast_trajectory_active returns False when no sunset in horizon."""
        fc = [8000.0] * 12
        with _apply_patches(_fc_patches()):
            result = is_forecast_trajectory_active(5000.0, fc)
        assert result is False

    def test_is_forecast_trajectory_active_mode_disabled(self):
        """is_forecast_trajectory_active returns False when mode disabled."""
        fc = self._FC_9_THEN_NIGHT
        with _apply_patches(_fc_patches({"PV_TRAJ_FORECAST_MODE_ENABLED": False})):
            result = is_forecast_trajectory_active(5000.0, fc)
        assert result is False

    def test_forecast_mode_disabled_falls_back_to_classic(self):
        """With forecast mode off, compute_dynamic_trajectory_steps uses classic algo."""
        fc = self._FC_9_THEN_NIGHT  # should be ignored
        with _apply_patches({
            "PV_TRAJ_SCALING_ENABLED": True,
            "PV_TRAJ_FORECAST_MODE_ENABLED": False,
            "PV_TRAJ_SYSTEM_KWP": 10.0,
            "PV_TRAJ_MIN_STEPS": 2,
            "PV_TRAJ_MAX_STEPS": 12,
            "PV_TRAJ_MIDDAY_FACTOR": 1.0,
            "PV_TRAJ_MORNING_FACTOR": 0.5,
            "PV_TRAJ_AFTERNOON_FACTOR": 0.75,
            "PV_TRAJ_NIGHT_FACTOR": 0.0,
        }):
            steps = compute_dynamic_trajectory_steps(
                5000.0, system_kwp=10.0,
                now=datetime(2026, 6, 21, 12, 0, 0),
                pv_forecast=fc,
            )
        # classic: ratio=0.5, midday factor=1.0 → 2 + round(0.5*10) = 7
        assert steps == 7
