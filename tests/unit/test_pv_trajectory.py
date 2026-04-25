"""Unit tests for src/pv_trajectory.py — compute_dynamic_trajectory_steps."""
import pytest
from datetime import datetime
from unittest.mock import patch

from src import config
from src.pv_trajectory import compute_dynamic_trajectory_steps, _time_of_day_factor


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
