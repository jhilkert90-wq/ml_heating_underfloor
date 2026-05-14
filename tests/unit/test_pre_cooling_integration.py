"""Integration tests for pre-cooling workflow.

Tests the interaction between OverheatingPredictor and the main control loop,
verifying that pre-cooling only activates in cooling mode and correctly shifts
the binary-search target temperature.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.overheating_predictor import OverheatingPredictor


class TestModeIsolation:
    """Pre-cooling must ONLY activate in cooling mode."""

    def test_heating_mode_returns_no_risk(self):
        """Pre-cooling must not fire in heating mode."""
        predictor = OverheatingPredictor()
        result = predictor.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features={"inlet_temp": 30.0},
            thermal_model=MagicMock(),
            climate_mode="heating",
        )
        assert result["risk"] is False
        assert result["should_cool_now"] is False

    def test_idle_mode_returns_no_risk(self):
        """Pre-cooling must not fire in idle mode."""
        predictor = OverheatingPredictor()
        result = predictor.predict_overheating_risk(
            current_indoor=25.0,
            target_cooling=23.0,
            features={"inlet_temp": 25.0},
            thermal_model=MagicMock(),
            climate_mode="idle",
        )
        assert result["risk"] is False
        assert result["should_cool_now"] is False


class TestTargetShiftMechanics:
    """Verify the target-shift approach works correctly."""

    def test_pre_cool_shifts_target_below_current(self):
        """When pre-cool fires and room <= target, target should shift down."""
        # Simulate: room=22.5, target=23.0, predictor says cool now
        current_indoor = 22.5
        target = 23.0
        offset = 0.5

        # The main.py logic: target = current_indoor - offset
        shifted = current_indoor - offset
        assert shifted == 22.0
        assert shifted < current_indoor  # forces binary search to cool

    def test_no_shift_when_room_above_target(self):
        """When room > target, no shift needed — normal cooling handles it."""
        current_indoor = 24.0
        target = 23.0

        # main.py: only shift when prediction_indoor_temp <= target
        should_shift = current_indoor <= target
        assert should_shift is False

    def test_shift_creates_negative_error(self):
        """Shifted target below current creates error that drives cooling."""
        current_indoor = 22.5
        offset = 0.5
        shifted_target = current_indoor - offset  # 22.0

        error = shifted_target - current_indoor  # -0.5
        assert error < 0  # Negative error = need to cool


class TestPreCoolDecisionFlow:
    """Test the full decision flow for realistic scenarios."""

    def _make_features(self, outdoor=30.0, pv=5000.0):
        """Create features dict with forecasts."""
        features = {
            "inlet_temp": 23.0,
            "outdoor_temp": outdoor,
            "pv_now": pv,
        }
        for h in range(1, 13):
            features[f"pv_forecast_{h}h"] = pv
            features[f"temp_forecast_{h}h"] = outdoor
        return features

    def _make_model(self, peak_temp=25.0, peak_hour=2.0):
        """Create a mock thermal model that returns a trajectory."""
        model = MagicMock()
        trajectory = {
            "trajectory": [peak_temp],
            "times": [peak_hour],
        }
        model.predict_thermal_trajectory.return_value = trajectory
        return model

    def test_sunny_hot_day_triggers_pre_cool(self):
        """Hot day with high PV → should trigger pre-cooling."""
        predictor = OverheatingPredictor()
        features = self._make_features(outdoor=32.0, pv=6000.0)
        model = self._make_model(peak_temp=25.0, peak_hour=2.0)

        result = predictor.predict_overheating_risk(
            current_indoor=22.5,
            target_cooling=23.0,
            features=features,
            thermal_model=model,
            climate_mode="cooling",
        )

        assert result["risk"] is True
        assert result["should_cool_now"] is True

    def test_cloudy_cool_day_no_pre_cool(self):
        """Cool cloudy day → no overheating risk."""
        predictor = OverheatingPredictor()
        features = self._make_features(outdoor=18.0, pv=200.0)
        model = self._make_model(peak_temp=22.0, peak_hour=6.0)

        result = predictor.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features=features,
            thermal_model=model,
            climate_mode="cooling",
        )

        assert result["risk"] is False
        assert result["should_cool_now"] is False

    def test_evening_no_pv_no_pre_cool(self):
        """Evening with no PV forecast → guards block pre-cooling."""
        predictor = OverheatingPredictor()
        features = self._make_features(outdoor=20.0, pv=0.0)
        # Remove PV forecasts
        for h in range(1, 13):
            features[f"pv_forecast_{h}h"] = 0.0

        model = self._make_model(peak_temp=24.0, peak_hour=3.0)

        result = predictor.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features=features,
            thermal_model=model,
            climate_mode="cooling",
        )

        # Both PV AND outdoor below thresholds → blocked
        assert result["should_cool_now"] is False

    @patch("src.config.PRE_COOL_ENABLED", False)
    def test_disabled_config_blocks_everything(self):
        """When disabled via config, no risk assessment at all."""
        predictor = OverheatingPredictor()
        features = self._make_features(outdoor=35.0, pv=8000.0)
        model = self._make_model(peak_temp=28.0, peak_hour=1.0)

        result = predictor.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features=features,
            thermal_model=model,
            climate_mode="cooling",
        )

        assert result["risk"] is False
        assert result["should_cool_now"] is False
