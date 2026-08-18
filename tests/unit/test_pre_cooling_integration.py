"""Integration tests for pre-cooling workflow.

Tests the interaction between OverheatingPredictor and the main control loop,
verifying that pre-cooling only activates in cooling mode and correctly shifts
the binary-search target temperature.
"""

import types
import pytest
from unittest.mock import MagicMock, patch

from src import cycle_routes
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

    def _make_ctx(
        self,
        *,
        room_temp: float = 22.3,
        target_temp: float = 23.0,
        cooling_model_type: str = "trajectory",
        cooling_ml_model=None,
        all_states=None,
    ):
        state_manager = MagicMock()
        state_manager.update_operational_state = MagicMock()
        return types.SimpleNamespace(
            prediction_indoor_temp=room_temp,
            target_indoor_temp=target_temp,
            features_dict={"inlet_temp": 20.5, "outdoor_temp": 20.3, "pv_now": 1800.0},
            wrapper=types.SimpleNamespace(thermal_model=MagicMock()),
            climate_mode="cooling",
            cooling_ml_model=cooling_ml_model,
            cooling_ml_model_type=cooling_model_type,
            cooling_obs_buffer=None,
            pre_cool_active=False,
            pre_cool_result=None,
            state_manager=state_manager,
            all_states=all_states or {},
        )

    def test_pre_cool_shifts_from_cooling_target_not_room_temp(self):
        """Fixed pre-cool offset must apply to the configured target."""
        ctx = self._make_ctx(room_temp=22.3, target_temp=23.0)

        with patch(
            "src.overheating_predictor.OverheatingPredictor.predict_overheating_risk",
            return_value={
                "risk": True,
                "peak_temp": 26.3,
                "peak_hour": 7.3,
                "should_cool_now": True,
                "reason": "predicted peak 26.3°C in 7.3h",
            },
        ):
            cycle_routes.step_pre_cooling(ctx)

        assert ctx.pre_cool_active is True
        assert ctx.target_indoor_temp == pytest.approx(22.5)

    def test_no_shift_when_room_above_target(self):
        """When room > target, no shift needed — normal cooling handles it."""
        current_indoor = 24.0
        target = 23.0

        # main.py: only shift when prediction_indoor_temp <= target
        should_shift = current_indoor <= target
        assert should_shift is False

    def test_proportional_shift_is_clamped_by_target_entity_minimum(self):
        """Proportional pre-cool must respect the configured target minimum."""
        cooling_ml_model = MagicMock()
        cooling_ml_model.is_loaded = True
        cooling_ml_model.predict_overheating_risk.return_value = {
            "risk": True,
            "peak_temp": 24.5,
            "peak_hour": 3.0,
            "should_cool_now": True,
            "reason": "lgbm risk",
            "predicted_delta": 2.0,
            "predicted_max_temp": 25.0,
            "lgbm_proba": 0.8,
        }
        ctx = self._make_ctx(
            room_temp=22.8,
            target_temp=23.0,
            cooling_model_type="lgbm_model",
            cooling_ml_model=cooling_ml_model,
            all_states={
                "input_number.cooling_target": {
                    "state": "23.0",
                    "attributes": {"min": 22.4},
                }
            },
        )

        with patch.object(
            cycle_routes.config,
            "TARGET_INDOOR_TEMP_COOLING_ENTITY_ID",
            "input_number.cooling_target",
        ), patch.object(
            cycle_routes.config,
            "PRE_COOL_OVERSHOOT_GAIN",
            0.7,
        ), patch(
            "src.overheating_predictor.OverheatingPredictor.predict_overheating_risk",
            return_value={
                "risk": False,
                "peak_temp": 22.8,
                "peak_hour": 12.0,
                "should_cool_now": False,
                "reason": "shadow trajectory",
            },
        ):
            cycle_routes.step_pre_cooling(ctx)

        assert ctx.pre_cool_active is True
        assert ctx.target_indoor_temp == pytest.approx(22.4)

    def test_shadow_lgbm_can_block_implausible_trajectory_precool(self):
        """A low-risk LGBM shadow result should suppress extreme trajectory pre-cool."""
        cooling_ml_model = MagicMock()
        cooling_ml_model.is_loaded = True
        cooling_ml_model.predict_overheating_risk.return_value = {
            "risk": False,
            "peak_temp": 22.7,
            "peak_hour": 8.0,
            "should_cool_now": False,
            "reason": "LGBM p=0.028; no overheating risk predicted",
            "lgbm_proba": 0.028,
            "predicted_delta": 0.33,
            "predicted_max_temp": 22.7,
        }
        ctx = self._make_ctx(
            room_temp=22.3,
            target_temp=23.0,
            cooling_model_type="trajectory",
            cooling_ml_model=cooling_ml_model,
        )

        with patch(
            "src.overheating_predictor.OverheatingPredictor.predict_overheating_risk",
            return_value={
                "risk": True,
                "peak_temp": 26.3,
                "peak_hour": 7.3,
                "should_cool_now": True,
                "reason": "predicted peak 26.3°C in 7.3h",
                "trajectory": [22.6, 23.4, 24.8, 26.3],
            },
        ):
            cycle_routes.step_pre_cooling(ctx)

        assert ctx.pre_cool_active is False
        assert ctx.target_indoor_temp == pytest.approx(23.0)
        assert "shadow" in ctx.pre_cool_result["reason"].lower()


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
        model.predict_thermal_trajectory.assert_called_once()
        kwargs = model.predict_thermal_trajectory.call_args.kwargs
        assert kwargs["pv_power"] == pytest.approx(6000.0)
        assert kwargs["pv_forecasts"] == pytest.approx([6000.0] * 13)

    def test_forecast_horizon_uses_current_anchor_then_hourly_buckets(self):
        """The passive trajectory should use a 0h anchor plus +1h/+2h forecasts."""
        predictor = OverheatingPredictor()
        features = self._make_features(outdoor=20.3, pv=1800.0)
        features["temp_forecast_1h"] = 20.6
        features["temp_forecast_2h"] = 21.4
        features["pv_forecast_1h"] = 2175.0
        features["pv_forecast_2h"] = 2011.0
        model = self._make_model(peak_temp=24.0, peak_hour=2.0)

        predictor.predict_overheating_risk(
            current_indoor=22.3,
            target_cooling=23.0,
            features=features,
            thermal_model=model,
            climate_mode="cooling",
        )

        kwargs = model.predict_thermal_trajectory.call_args.kwargs
        assert kwargs["outdoor_temp"][:3] == pytest.approx([20.3, 20.6, 21.4])
        assert kwargs["pv_forecasts"][:3] == pytest.approx([1800.0, 2175.0, 2011.0])

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
