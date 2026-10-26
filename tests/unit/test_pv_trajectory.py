"""Unit tests for src/pv_trajectory.py — compute_dynamic_trajectory_steps."""
import pytest
from unittest.mock import patch

from src import config
from src.pv_trajectory import (
    compute_dynamic_trajectory_steps,
    compute_forecast_driven_trajectory_steps,
    get_forecast_trajectory_state,
    is_forecast_trajectory_active,
)


# ---------------------------------------------------------------------------
# Feature flag disabled → returns static TRAJECTORY_STEPS unchanged
# ---------------------------------------------------------------------------

class TestFeatureFlagDisabled:
    def test_disabled_returns_trajectory_steps(self):
        with (
            patch.object(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False),
            patch.object(config, "TRAJECTORY_STEPS", 5),
        ):
            result = compute_dynamic_trajectory_steps(10000.0)
        assert result == 5

    def test_disabled_ignores_pv(self):
        with (
            patch.object(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False),
            patch.object(config, "TRAJECTORY_STEPS", 3),
        ):
            assert compute_dynamic_trajectory_steps(0.0) == 3
            assert compute_dynamic_trajectory_steps(15000.0) == 3


# ---------------------------------------------------------------------------
# Forecast-driven trajectory mode
# ---------------------------------------------------------------------------

def _fc_patches(extra=None):
    """Return a dict of config patches for forecast-driven mode tests."""
    base = {
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
        # 9 consecutive entries > 50 W → 9 + MIN_STEPS(2) = 11
        assert steps == 11

    def test_no_activation_forecast_1h_below_threshold(self):
        """1h forecast below threshold → inactive, returns MIN_STEPS."""
        fc = [2999.0] + self._FC_9_THEN_NIGHT[1:]
        with _apply_patches(_fc_patches()):
            steps = compute_forecast_driven_trajectory_steps(5000.0, fc)
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

    def test_step_count_equals_consecutive_pv_hours_plus_min_steps(self):
        """Steps equal consecutive forecast entries above zero_w plus MIN_STEPS."""
        # 5 daylight hours, then night
        fc = [5000, 4000, 3000, 500, 100, 0, 0, 0, 0, 0, 0, 0]
        with _apply_patches(_fc_patches()):
            steps = compute_forecast_driven_trajectory_steps(5000.0, fc)
        assert steps == 7  # 5 + MIN_STEPS(2) = 7

    def test_steps_clamped_to_max_when_pv_hours_plus_min_exceeds_max(self):
        """3 daylight hours + min_steps(2) = 5 → clamped to MAX_STEPS=4."""
        # With MAX_STEPS=4: horizon=[5000, 4000, 3000, 0], consecutive=3
        fc = [5000.0, 4000.0, 3000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        with _apply_patches(_fc_patches({"PV_TRAJ_MIN_STEPS": 2, "PV_TRAJ_MAX_STEPS": 4})):
            steps = compute_forecast_driven_trajectory_steps(5000.0, fc)
        assert steps == 4  # 3 + 2 = 5 → clamped to max=4

    def test_steps_within_bounds_at_max_horizon_minus_one(self):
        """11 consecutive daylight hours + MIN_STEPS(2) = 13 → clamped to MAX_STEPS=12."""
        fc = [6000.0] * 11 + [0.0]  # 11 daylight then night; max=12
        with _apply_patches(_fc_patches()):
            steps = compute_forecast_driven_trajectory_steps(5000.0, fc)
        assert steps == 12  # 11 + 2 = 13 → clamped to 12

    def test_steps_one_daylight_hour_plus_min_steps(self):
        """Only 1 daylight hour → 1 + MIN_STEPS(2) = 3 steps."""
        fc = [3500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        with _apply_patches(_fc_patches({"PV_TRAJ_MIN_STEPS": 2})):
            steps = compute_forecast_driven_trajectory_steps(5000.0, fc)
        assert steps == 3  # 1 + 2 = 3

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
        # 2 + min=4 = 6, within [4, 8]
        assert steps == 6

    def test_compute_dynamic_delegates_to_forecast_mode(self):
        """compute_dynamic_trajectory_steps delegates when forecast mode enabled."""
        fc = [5000, 4000, 3000, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 3 remaining hours
        with _apply_patches(_fc_patches()):
            steps = compute_dynamic_trajectory_steps(5000.0, pv_forecast=fc)
        assert steps == 5  # 3 + MIN_STEPS(2) = 5

    def test_forecast_helper_ignores_mode_flag(self):
        """Direct forecast helper still computes forecast-driven steps when mode is off."""
        with _apply_patches(
            _fc_patches(
                {
                    "PV_TRAJ_FORECAST_MODE_ENABLED": False,
                    "TRAJECTORY_STEPS": 4,
                }
            )
        ):
            steps = compute_forecast_driven_trajectory_steps(
                5000.0,
                self._FC_9_THEN_NIGHT,
            )
        assert steps == 11

    def test_is_forecast_trajectory_active_true(self):
        """is_forecast_trajectory_active returns True when conditions met."""
        fc = self._FC_9_THEN_NIGHT
        with _apply_patches(_fc_patches()):
            result = is_forecast_trajectory_active(5000.0, fc)
        assert result is True

    def test_get_forecast_trajectory_state_marks_threshold_active(self):
        """Shared state exposes active/threshold/steps for direct-PV activation."""
        fc = self._FC_9_THEN_NIGHT
        with _apply_patches(_fc_patches()):
            state = get_forecast_trajectory_state(5000.0, fc)
        assert state.active is True
        assert state.forecast_1h_above_threshold is True
        assert state.dynamic_steps == 11
        assert state.dynamic_steps_above_min is True

    def test_is_forecast_trajectory_active_pv_below_threshold(self):
        """is_forecast_trajectory_active returns False when 1h forecast below threshold."""
        fc = [2000.0] + self._FC_9_THEN_NIGHT[1:]
        with _apply_patches(_fc_patches()):
            result = is_forecast_trajectory_active(5000.0, fc)
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

    def test_forecast_mode_disabled_returns_trajectory_steps(self):
        """With forecast mode off, compute_dynamic_trajectory_steps returns static steps."""
        fc = self._FC_9_THEN_NIGHT  # should be ignored
        with _apply_patches({
            "PV_TRAJ_FORECAST_MODE_ENABLED": False,
            "TRAJECTORY_STEPS": 4,
        }):
            steps = compute_dynamic_trajectory_steps(5000.0, pv_forecast=fc)
        assert steps == 4


# ---------------------------------------------------------------------------
# 1h-forecast activation (current PV only used for night stop)
# ---------------------------------------------------------------------------

class TestForecast1hActivation:
    # Next-hour forecast below threshold, sun later
    _FC_LOW_THEN_SUN = [800.0, 1200.0, 4000.0, 5000.0, 5500.0, 4500.0, 3500.0, 2000.0, 60.0, 0.0, 0.0, 0.0]
    _FC_HIGH = [4000.0, 5000.0, 4500.0, 3500.0, 2000.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_active_when_forecast_1h_above_threshold_despite_low_current_pv(self):
        with _apply_patches(_fc_patches()):
            state = get_forecast_trajectory_state(800.0, self._FC_HIGH)
        assert state.active is True
        assert state.forecast_1h_above_threshold is True
        # 6 hours > 50 W + MIN_STEPS(2)
        assert state.dynamic_steps == 8

    def test_inactive_when_forecast_1h_below_threshold_despite_high_current_pv(self):
        with _apply_patches(_fc_patches()):
            state = get_forecast_trajectory_state(5000.0, self._FC_LOW_THEN_SUN)
        assert state.active is False
        assert state.forecast_1h_above_threshold is False
        assert state.dynamic_steps == 2

    def test_threshold_is_inclusive(self):
        fc = [3000.0, 100.0, 0.0, 0.0]
        with _apply_patches(_fc_patches()):
            assert is_forecast_trajectory_active(1000.0, fc) is True

    def test_night_stop_overrides_forecast(self):
        with _apply_patches(_fc_patches()):
            state = get_forecast_trajectory_state(10.0, self._FC_HIGH)
        assert state.active is False
        assert state.dynamic_steps == 2

    def test_no_sunset_in_horizon_inactive(self):
        with _apply_patches(_fc_patches()):
            assert is_forecast_trajectory_active(5000.0, [8000.0] * 12) is False

    def test_empty_forecast_inactive(self):
        with _apply_patches(_fc_patches()):
            assert is_forecast_trajectory_active(5000.0, None) is False
            assert is_forecast_trajectory_active(5000.0, []) is False

    def test_compute_steps_follows_state(self):
        with _apply_patches(_fc_patches()):
            assert compute_forecast_driven_trajectory_steps(800.0, self._FC_HIGH) == 8
            assert compute_forecast_driven_trajectory_steps(5000.0, self._FC_LOW_THEN_SUN) == 2
