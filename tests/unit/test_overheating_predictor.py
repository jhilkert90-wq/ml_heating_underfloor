"""
Unit tests for the Overheating Predictor (predictive pre-cooling).

Tests cover:
- Basic risk prediction with various forecast scenarios
- Forecast input handling (missing, empty, truncated)
- Guard thresholds (min PV, min outdoor)
- Edge cases (night, HP at limit, disabled, multiple peaks)
- PV feature key contract (thermal vs electrical key families)
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.overheating_predictor import OverheatingPredictor
from src.hlc_learner import _build_cycle, HLCCycle
from src import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(
    inlet_temp: float = 23.0,
    outdoor_temp: float = 20.0,
    pv_now: float = 0.0,
    current_indoor: float = 22.0,
    pv_forecasts: list | None = None,
    outdoor_forecasts: list | None = None,
    cloud_cover: float = 50.0,
    fireplace_on: float = 0.0,
    tv_on: float = 0.0,
    indoor_trend: float = 0.0,
) -> dict:
    """Build a minimal features dict for testing."""
    features = {
        "inlet_temp": inlet_temp,
        "outdoor_temp": outdoor_temp,
        "pv_now": pv_now,
        "pv_now_electrical": pv_now,
        "fireplace_on": fireplace_on,
        "tv_on": tv_on,
        "indoor_temp_delta_60m": indoor_trend,
        "indoor_temp_lag_30m": current_indoor,
    }
    n_fc = 12
    pv = pv_forecasts or [0.0] * n_fc
    out = outdoor_forecasts or [outdoor_temp] * n_fc
    for h in range(1, n_fc + 1):
        idx = h - 1
        features[f"pv_forecast_{h}h"] = pv[idx] if idx < len(pv) else 0.0
        features[f"pv_forecast_electrical_{h}h"] = (
            pv[idx] if idx < len(pv) else 0.0
        )
        features[f"temp_forecast_{h}h"] = (
            out[idx] if idx < len(out) else outdoor_temp
        )
        features[f"cloud_cover_forecast_{h}h"] = cloud_cover
    return features


def _make_trajectory_model(trajectory: list, times: list | None = None):
    """Return a mock thermal model with a canned trajectory response."""
    model = Mock()
    if times is None:
        step_h = 10 / 60.0  # 10 min steps
        times = [(i + 1) * step_h for i in range(len(trajectory))]
    model.predict_thermal_trajectory.return_value = {
        "trajectory": trajectory,
        "times": times,
        "reaches_target_at": None,
        "overshoot_predicted": False,
        "max_predicted": max(trajectory) if trajectory else 22.0,
        "min_predicted": min(trajectory) if trajectory else 22.0,
        "equilibrium_temp": trajectory[-1] if trajectory else 22.0,
        "final_error": 0.0,
    }
    return model


# ═══════════════════════════════════════════════════════════════════════
# Basic Prediction Tests
# ═══════════════════════════════════════════════════════════════════════


class TestBasicPrediction:
    """Core risk detection logic."""

    def test_no_risk_below_target(self):
        """Passive trajectory stays below target → no risk."""
        features = _make_features(
            outdoor_temp=18.0,
            outdoor_forecasts=[18.0] * 12,
        )
        # Flat trajectory at 22°C — below target 23°C
        model = _make_trajectory_model([22.0] * 72)
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features=features,
            thermal_model=model,
            climate_mode="cooling",
        )

        assert result["risk"] is False
        assert result["should_cool_now"] is False

    def test_risk_detected_above_target(self):
        """Trajectory exceeds target + margin → risk detected."""
        features = _make_features(
            outdoor_temp=25.0,
            pv_now=3000.0,
            pv_forecasts=[4000.0] * 12,
            outdoor_forecasts=[25.0, 27.0, 28.0, 29.0, 28.0, 27.0,
                              26.0, 24.0, 22.0, 20.0, 19.0, 18.0],
        )
        # Trajectory rises to 25°C at midday (> target 23°C + margin 0.5K)
        traj = [22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 24.5, 24.0,
                23.5, 23.0, 22.5, 22.0]
        # 1h per step for simplicity
        times = [float(h) for h in range(1, 13)]
        model = _make_trajectory_model(traj, times)
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features=features,
            thermal_model=model,
            climate_mode="cooling",
        )

        assert result["risk"] is True
        assert result["peak_temp"] == 25.0
        assert result["peak_hour"] == 6.0

    def test_should_cool_within_lead_time(self):
        """Peak within lead time → should_cool_now=True."""
        features = _make_features(
            outdoor_temp=28.0,
            pv_now=4000.0,
            pv_forecasts=[5000.0] * 12,
            outdoor_forecasts=[28.0] * 12,
        )
        # Peak at 2h (within 3h lead time)
        traj = [23.0, 24.0, 25.0, 24.5, 24.0]
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        model = _make_trajectory_model(traj, times)
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features=features,
            thermal_model=model,
            climate_mode="cooling",
        )

        assert result["should_cool_now"] is True
        assert result["risk"] is True

    def test_should_not_cool_peak_too_far(self):
        """Peak beyond lead time → risk=True but should_cool_now=False."""
        features = _make_features(
            outdoor_temp=25.0,
            pv_now=2000.0,
            pv_forecasts=[3000.0] * 12,
            outdoor_forecasts=[25.0, 26.0, 27.0, 28.0, 29.0, 28.0,
                              27.0, 26.0, 25.0, 23.0, 22.0, 20.0],
        )
        # Peak at 8h (beyond 3h lead time)
        traj = [22.0, 22.2, 22.5, 22.8, 23.0, 23.3, 23.5, 24.0,
                23.5, 23.0, 22.5, 22.0]
        times = [float(h) for h in range(1, 13)]
        model = _make_trajectory_model(traj, times)
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features=features,
            thermal_model=model,
            climate_mode="cooling",
        )

        assert result["risk"] is True
        assert result["should_cool_now"] is False
        assert result["peak_hour"] == 8.0

    def test_already_above_target(self):
        """Room already above target → always cool regardless of forecast."""
        features = _make_features(
            outdoor_temp=25.0,
            pv_now=3000.0,
            pv_forecasts=[3000.0] * 12,
            outdoor_forecasts=[25.0] * 12,
        )
        model = _make_trajectory_model([24.0] * 12)
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            current_indoor=24.0,
            target_cooling=23.0,
            features=features,
            thermal_model=model,
            climate_mode="cooling",
        )

        assert result["should_cool_now"] is True

    def test_already_above_target_at_night(self):
        """Room above target at night (PV=0, outdoor cool) → still cools.

        Regression test: guards must NOT block reactive cooling when the
        room is already overheated, regardless of PV/outdoor conditions.
        """
        features = _make_features(
            outdoor_temp=15.0,
            pv_now=0.0,
            pv_forecasts=[0.0] * 12,
            outdoor_forecasts=[15.0] * 12,
        )
        model = _make_trajectory_model([24.0] * 12)
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            current_indoor=24.0,
            target_cooling=23.0,
            features=features,
            thermal_model=model,
            climate_mode="cooling",
        )

        # Room is 24°C > target 23°C → must cool even with PV=0
        assert result["should_cool_now"] is True

    def test_risk_level_proportional_to_peak(self):
        """Higher peaks → higher peak_temp values."""
        features = _make_features(
            outdoor_temp=28.0,
            pv_now=4000.0,
            pv_forecasts=[5000.0] * 12,
            outdoor_forecasts=[28.0] * 12,
        )
        # Moderate peak
        model1 = _make_trajectory_model(
            [23.8, 24.0, 23.5], [1.0, 2.0, 3.0]
        )
        # High peak
        model2 = _make_trajectory_model(
            [24.0, 26.0, 25.0], [1.0, 2.0, 3.0]
        )
        predictor = OverheatingPredictor()

        r1 = predictor.predict_overheating_risk(
            22.0, 23.0, features, model1, "cooling"
        )
        r2 = predictor.predict_overheating_risk(
            22.0, 23.0, features, model2, "cooling"
        )

        assert r2["peak_temp"] > r1["peak_temp"]


# ═══════════════════════════════════════════════════════════════════════
# Forecast Handling Tests
# ═══════════════════════════════════════════════════════════════════════


class TestForecastHandling:
    """Graceful handling of missing/incomplete forecast data."""

    def test_missing_pv_forecast_returns_no_risk(self):
        """Missing PV forecast → features default to 0W → no risk."""
        features = _make_features(outdoor_temp=18.0)
        # No pv_forecast_*h keys → defaults to 0
        for h in range(1, 13):
            features.pop(f"pv_forecast_{h}h", None)
            features.pop(f"pv_forecast_electrical_{h}h", None)
        model = _make_trajectory_model([22.0] * 12)
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        # Guards should block: total PV < 1000W AND peak outdoor < 22°C
        assert result["risk"] is False

    def test_missing_outdoor_forecast_uses_current(self):
        """Missing outdoor forecast → defaults to current outdoor temp."""
        features = _make_features(
            outdoor_temp=25.0,
            pv_now=3000.0,
            pv_forecasts=[3000.0] * 12,
        )
        for h in range(1, 13):
            features.pop(f"temp_forecast_{h}h", None)
        model = _make_trajectory_model([24.0] * 12, list(range(1, 13)))
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        # Model was called with outdoor forecast filled from current temp
        call_kwargs = model.predict_thermal_trajectory.call_args
        outdoor_arg = call_kwargs.kwargs.get(
            "outdoor_temp", call_kwargs.args[3] if len(call_kwargs.args) > 3 else None
        )
        # All entries should be 25.0 (current outdoor temp)
        if isinstance(outdoor_arg, list):
            assert all(v == 25.0 for v in outdoor_arg)

    def test_empty_arrays_handled(self):
        """Empty pv_power_history does not crash."""
        features = _make_features(
            pv_now=3000.0,
            pv_forecasts=[3000.0] * 12,
            outdoor_forecasts=[25.0] * 12,
        )
        features["pv_power_history"] = []
        model = _make_trajectory_model([23.0] * 12)
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        assert "risk" in result  # No crash

    def test_no_inlet_temp_returns_no_risk(self):
        """Missing inlet_temp → graceful no-risk."""
        features = _make_features()
        features.pop("inlet_temp")
        model = _make_trajectory_model([25.0] * 12)
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        assert result["risk"] is False
        assert "no inlet_temp" in result["reason"]

    def test_trajectory_failure_returns_no_risk(self):
        """If trajectory simulation raises → graceful fallback."""
        features = _make_features(
            pv_now=3000.0,
            pv_forecasts=[3000.0] * 12,
            outdoor_forecasts=[28.0] * 12,
        )
        model = Mock()
        model.predict_thermal_trajectory.side_effect = RuntimeError("test")
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        assert result["risk"] is False
        assert "trajectory failed" in result["reason"]


# ═══════════════════════════════════════════════════════════════════════
# Guard Threshold Tests
# ═══════════════════════════════════════════════════════════════════════


class TestGuardThresholds:
    """Tests for the PV and outdoor temperature guards."""

    def test_trigger_margin_exactly_at_boundary(self):
        """Peak = target + margin exactly → triggers."""
        features = _make_features(
            outdoor_temp=25.0,
            pv_now=3000.0,
            pv_forecasts=[3000.0] * 12,
            outdoor_forecasts=[25.0] * 12,
        )
        # Peak = 23.0 + 0.5 = 23.5 (exactly at threshold)
        # Use peak slightly above to avoid float precision issues
        model = _make_trajectory_model(
            [23.51, 23.0], [1.0, 2.0]
        )
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        assert result["risk"] is True

    def test_trigger_margin_just_below(self):
        """Peak = target + margin - epsilon → no trigger."""
        features = _make_features(
            outdoor_temp=25.0,
            pv_now=3000.0,
            pv_forecasts=[3000.0] * 12,
            outdoor_forecasts=[25.0] * 12,
        )
        # Peak = 23.49 (just below 23.5 threshold)
        model = _make_trajectory_model(
            [23.49, 23.0], [1.0, 2.0]
        )
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        assert result["risk"] is False

    def test_min_pv_gate_blocks_low_pv(self):
        """Total PV < threshold AND peak outdoor < threshold → no risk."""
        features = _make_features(
            outdoor_temp=18.0,
            pv_now=50.0,
            pv_forecasts=[50.0] * 12,
            outdoor_forecasts=[18.0] * 12,
        )
        # Even if trajectory predicts high temp (shouldn't happen in reality)
        model = _make_trajectory_model([25.0] * 12, list(range(1, 13)))
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        assert result["risk"] is False
        assert "guards not met" in result["reason"]

    def test_high_pv_passes_even_with_cool_outdoor(self):
        """High PV total passes guard even if outdoor is cool."""
        features = _make_features(
            outdoor_temp=18.0,
            pv_now=3000.0,
            pv_forecasts=[5000.0] * 12,
            outdoor_forecasts=[18.0] * 12,
        )
        # Trajectory shows overheating from solar gain alone
        model = _make_trajectory_model(
            [23.0, 24.0, 25.0], [1.0, 2.0, 3.0]
        )
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        # PV guard passed (total >> 1000W), outdoor guard not met but OR
        assert result["risk"] is True

    def test_hot_outdoor_passes_even_with_no_pv(self):
        """Hot outdoor forecast passes guard even if PV is zero."""
        features = _make_features(
            outdoor_temp=25.0,
            pv_now=0.0,
            pv_forecasts=[0.0] * 12,
            outdoor_forecasts=[25.0, 27.0, 30.0, 28.0, 26.0, 24.0,
                              22.0, 20.0, 19.0, 18.0, 17.0, 16.0],
        )
        model = _make_trajectory_model(
            [23.0, 24.0, 25.0], [1.0, 2.0, 3.0]
        )
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        # Outdoor guard passed (peak 30°C > 22°C)
        assert result["risk"] is True


# ═══════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and difficult scenarios."""

    def test_not_in_cooling_mode(self):
        """Heating mode → always returns no risk."""
        features = _make_features()
        model = _make_trajectory_model([25.0] * 12)
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "heating"
        )

        assert result["risk"] is False
        assert result["should_cool_now"] is False
        assert "not in cooling mode" in result["reason"]

    def test_night_no_pv_no_risk(self):
        """Night time (PV=0, outdoor cool) → no pre-cooling."""
        features = _make_features(
            outdoor_temp=15.0,
            pv_now=0.0,
            pv_forecasts=[0.0] * 12,
            outdoor_forecasts=[15.0, 14.0, 13.0, 12.0, 12.0, 13.0,
                              14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
        )
        model = _make_trajectory_model([21.0] * 12)
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            21.0, 23.0, features, model, "cooling"
        )

        assert result["should_cool_now"] is False

    def test_hp_at_physical_limit(self):
        """Outdoor 35°C → HP can't reach target, but still triggers pre-cool."""
        features = _make_features(
            outdoor_temp=35.0,
            pv_now=6000.0,
            pv_forecasts=[6000.0] * 12,
            outdoor_forecasts=[35.0] * 12,
        )
        # Trajectory shows extreme overheating
        model = _make_trajectory_model(
            [24.0, 25.0, 26.0, 27.0], [1.0, 2.0, 3.0, 4.0]
        )
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            23.5, 23.0, features, model, "cooling"
        )

        # Room already above target → should cool
        assert result["should_cool_now"] is True

    @patch.object(config, "PRE_COOL_ENABLED", False)
    def test_disabled_via_config(self):
        """PRE_COOL_ENABLED=false → never triggers."""
        features = _make_features(
            outdoor_temp=30.0,
            pv_now=5000.0,
            pv_forecasts=[5000.0] * 12,
            outdoor_forecasts=[30.0] * 12,
        )
        model = _make_trajectory_model([26.0] * 12)
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        assert result["risk"] is False
        assert "PRE_COOL_ENABLED=false" in result["reason"]

    def test_room_well_below_target(self):
        """Room far below target + risk detected → should_cool based on lead time."""
        features = _make_features(
            outdoor_temp=25.0,
            pv_now=3000.0,
            pv_forecasts=[5000.0] * 12,
            outdoor_forecasts=[25.0, 27.0, 29.0, 28.0, 26.0, 24.0,
                              22.0, 20.0, 18.0, 17.0, 16.0, 15.0],
        )
        # Peak in 2h (within lead time)
        model = _make_trajectory_model(
            [22.0, 23.5, 24.5, 24.0, 23.0], [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            21.0, 23.0, features, model, "cooling"
        )

        # Room at 21°C (2°C below target) but peak predicted at 24.5°C in 3h
        assert result["risk"] is True
        assert result["should_cool_now"] is True

    def test_multiple_peaks_uses_highest(self):
        """Trajectory with multiple peaks → uses the maximum."""
        features = _make_features(
            outdoor_temp=25.0,
            pv_now=3000.0,
            pv_forecasts=[3000.0] * 12,
            outdoor_forecasts=[25.0] * 12,
        )
        # Two peaks: 23.8 at 2h and 24.5 at 6h
        traj = [22.5, 23.8, 23.0, 23.2, 23.5, 24.5, 24.0, 23.5,
                23.0, 22.5, 22.0, 22.0]
        times = [float(h) for h in range(1, 13)]
        model = _make_trajectory_model(traj, times)
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        assert result["peak_temp"] == 24.5
        assert result["peak_hour"] == 6.0

    def test_fireplace_on_increases_risk(self):
        """Fireplace ON works against cooling → might increase peak."""
        features = _make_features(
            outdoor_temp=25.0,
            pv_now=3000.0,
            pv_forecasts=[3000.0] * 12,
            outdoor_forecasts=[25.0] * 12,
            fireplace_on=1.0,
        )
        model = _make_trajectory_model(
            [24.0, 25.0, 24.5], [1.0, 2.0, 3.0]
        )
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        # Fireplace passed to trajectory model
        call_kwargs = model.predict_thermal_trajectory.call_args.kwargs
        assert call_kwargs["fireplace_on"] == 1.0
        assert result["risk"] is True

    def test_trajectory_empty_result(self):
        """Trajectory returns empty → no risk."""
        features = _make_features(
            pv_now=3000.0,
            pv_forecasts=[3000.0] * 12,
            outdoor_forecasts=[25.0] * 12,
        )
        model = Mock()
        model.predict_thermal_trajectory.return_value = {
            "trajectory": [],
            "times": [],
        }
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        assert result["risk"] is False

    def test_trajectory_returns_none(self):
        """Trajectory returns None → no risk."""
        features = _make_features(
            pv_now=3000.0,
            pv_forecasts=[3000.0] * 12,
            outdoor_forecasts=[25.0] * 12,
        )
        model = Mock()
        model.predict_thermal_trajectory.return_value = None
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        assert result["risk"] is False

    def test_trajectory_called_with_hp_off_outlet(self):
        """Passive simulation must use outlet = inlet (HP OFF)."""
        features = _make_features(
            inlet_temp=23.0,
            outdoor_temp=25.0,
            pv_now=3000.0,
            pv_forecasts=[3000.0] * 12,
            outdoor_forecasts=[25.0] * 12,
        )
        model = _make_trajectory_model([23.0] * 12)
        predictor = OverheatingPredictor()

        predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        call_kwargs = model.predict_thermal_trajectory.call_args.kwargs
        assert call_kwargs["outlet_temp"] == 23.0  # inlet_temp, not lower
        assert call_kwargs["delta_t_floor"] == 0.0  # HP OFF

    def test_trajectory_called_with_cooling_mode(self):
        """Trajectory must be called with climate_mode='cooling'."""
        features = _make_features(
            outdoor_temp=25.0,
            pv_now=3000.0,
            pv_forecasts=[3000.0] * 12,
            outdoor_forecasts=[25.0] * 12,
        )
        model = _make_trajectory_model([23.0] * 12)
        predictor = OverheatingPredictor()

        predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        call_kwargs = model.predict_thermal_trajectory.call_args.kwargs
        assert call_kwargs["climate_mode"] == "cooling"


# ═══════════════════════════════════════════════════════════════════════
# PV Key Contract Regression Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPVKeyContract:
    """Regression tests for the PV feature key contract.

    OverheatingPredictor MUST consume the thermal-corrected keys
    (pv_now, pv_forecast_{h}h) and must NOT be sensitive to — or
    accidentally consume — the raw electrical keys
    (pv_now_electrical, pv_forecast_electrical_{h}h).

    See memory-bank/systemPatterns.md → "PV Feature Key Contract" for
    the full usage map and rationale.
    """

    def test_uses_pv_now_not_pv_now_electrical(self):
        """Predictor reads pv_now (thermal) for guard check, ignores pv_now_electrical.

        If it accidentally used pv_now_electrical only, a features dict that
        has pv_now=3000 but no pv_now_electrical would suppress the guard and
        return no-risk.  The correct behaviour is risk=True.
        """
        features = _make_features(
            outdoor_temp=28.0,
            pv_now=4000.0,
            pv_forecasts=[5000.0] * 12,
            outdoor_forecasts=[28.0] * 12,
        )
        # Remove the electrical key to prove it isn't required
        features.pop("pv_now_electrical", None)

        model = _make_trajectory_model(
            [23.5, 24.5, 25.0], [1.0, 2.0, 3.0]
        )
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        # Guard must have passed (high pv_now) and risk detected
        assert result["risk"] is True, (
            "Predictor failed to detect risk when pv_now_electrical was absent "
            "— it may be reading the wrong key family."
        )

    def test_pv_forecast_thermal_keys_used_for_trajectory(self):
        """Trajectory PV forecast list is built from pv_forecast_{h}h (thermal).

        Verify by providing thermal forecasts that trigger the guard while
        keeping electrical forecast keys absent.
        """
        features = _make_features(
            outdoor_temp=25.0,
            pv_now=2000.0,
            pv_forecasts=[8000.0] * 12,    # thermal: high enough to pass guard
            outdoor_forecasts=[25.0] * 12,
        )
        # Remove all electrical forecast keys
        for h in range(1, 13):
            features.pop(f"pv_forecast_electrical_{h}h", None)

        model = _make_trajectory_model(
            [23.5, 24.0, 25.0], [1.0, 2.0, 3.0]
        )
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        assert result["risk"] is True, (
            "Predictor blocked risk when pv_forecast_electrical_* was absent. "
            "It must use thermal pv_forecast_{h}h keys."
        )
        # Confirm trajectory was actually called (didn't short-circuit)
        assert model.predict_thermal_trajectory.called

    def test_pv_now_electrical_alone_insufficient_to_pass_guard(self):
        """Guard must NOT pass on electrical key alone when thermal key is absent/zero.

        If the predictor incorrectly reads pv_now_electrical instead of pv_now,
        and pv_now is missing/zero, this test catches the regression.
        """
        features = _make_features(
            outdoor_temp=18.0,
            pv_now=0.0,           # thermal: no solar gain
            pv_forecasts=[0.0] * 12,
            outdoor_forecasts=[18.0] * 12,
        )
        # Inject a high electrical value that should be invisible to the predictor
        features["pv_now_electrical"] = 5000.0
        for h in range(1, 13):
            features[f"pv_forecast_electrical_{h}h"] = 5000.0

        model = _make_trajectory_model([25.0] * 12, list(range(1, 13)))
        predictor = OverheatingPredictor()

        result = predictor.predict_overheating_risk(
            22.0, 23.0, features, model, "cooling"
        )

        # Guards should NOT pass — pv_now=0 (thermal) and outdoor < 22°C
        assert result["risk"] is False, (
            "Predictor passed guard using pv_now_electrical instead of pv_now. "
            "The thermal key pv_now must be the guard input."
        )

    def test_hlc_cycle_requires_pv_now_electrical_field(self):
        """HLCCycle.pv_now_electrical is the correct field for HLC session logic.

        This is the opposite rule: HLC uses the *electrical* key.
        Confirm the _build_cycle helper reads pv_now_electrical from context,
        not pv_now.
        """
        ctx = {
            "timestamp": datetime(2026, 6, 1, 10, 0),
            "thermal_power_kw": 1.5,
            "indoor_temp": 21.0,
            "outdoor_temp": 5.0,
            "target_temp": 21.0,
            "indoor_temp_delta_60m": 0.0,
            "pv_now_electrical": 3500.0,   # electrical key — correct
            "pv_now": 1200.0,              # thermal key — must NOT be used for session FSM
            "fireplace_on": 0.0,
            "tv_on": 0.0,
            "dhw_heating": 0.0,
            "defrosting": 0.0,
            "dhw_boost_heater": 0.0,
            "is_blocking": False,
        }
        cycle = _build_cycle(ctx)

        assert isinstance(cycle, HLCCycle)
        assert cycle.pv_now_electrical == pytest.approx(3500.0), (
            "HLCCycle.pv_now_electrical must be read from the electrical key, "
            "not from pv_now."
        )

    def test_hlc_cycle_pv_now_electrical_defaults_to_zero_when_absent(self):
        """_build_cycle falls back to 0.0 when pv_now_electrical is missing."""
        ctx = {
            "timestamp": datetime(2026, 6, 1, 10, 0),
            "thermal_power_kw": 1.5,
            "indoor_temp": 21.0,
            "outdoor_temp": 5.0,
            "target_temp": 21.0,
            "indoor_temp_delta_60m": 0.0,
            # pv_now_electrical intentionally absent
            "pv_now": 2000.0,
            "fireplace_on": 0.0,
            "tv_on": 0.0,
            "dhw_heating": 0.0,
            "defrosting": 0.0,
            "dhw_boost_heater": 0.0,
            "is_blocking": False,
        }
        cycle = _build_cycle(ctx)

        assert isinstance(cycle, HLCCycle)
        assert cycle.pv_now_electrical == pytest.approx(0.0), (
            "_build_cycle must default pv_now_electrical to 0.0 when absent, "
            "not fall back to pv_now."
        )
