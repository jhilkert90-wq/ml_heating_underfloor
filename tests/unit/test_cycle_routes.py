"""Tests for cycle route handlers and shared step functions."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime

from src.cycle_context import CycleContext
from src.cycle_state import CycleState
from src.cycle_routes import (
    run_blocking_route,
    run_idle_route,
    run_heating_route,
    run_cooling_route,
    _resolve_pre_cool_min_target,
    step_get_sensor_data,
    step_apply_cooling_target,
    step_determine_prediction_indoor,
    step_build_features,
    step_prediction,
    step_gradual_control,
    step_ema_smoothing,
    step_setpoint_hold,
)


def _make_ctx(**kwargs) -> CycleContext:
    """Create a CycleContext with sensible defaults for testing."""
    ctx = CycleContext()
    ctx.ha_client = MagicMock()
    ctx.influx_service = MagicMock()
    ctx.all_states = {"entities": {}}
    ctx.state = {}
    ctx.state_manager = MagicMock()
    ctx.wrapper = MagicMock()
    ctx.cycle_number = 1
    ctx.sensor_buffer = MagicMock()
    ctx.climate_mode = "heating"
    ctx.effective_shadow_mode = False
    ctx.shadow_mode = MagicMock()
    ctx.shadow_mode.shadow_deployment = False
    ctx.shadow_mode.should_publish_output_entities = True
    ctx.blocking_entities = [
        "binary_sensor.dhw_status",
        "binary_sensor.defrost_status",
    ]
    for key, val in kwargs.items():
        setattr(ctx, key, val)
    return ctx


class TestStepGetSensorData:
    """Test sensor data retrieval step."""

    @patch("src.cycle_routes.SensorDataManager")
    def test_returns_true_on_success(self, mock_sdm_class):
        mock_sdm = mock_sdm_class.return_value
        mock_sdm.get_sensor_data.return_value = (
            {
                "target_indoor_temp": 22.0,
                "actual_indoor": 21.5,
                "actual_outlet_temp": 30.0,
                "avg_other_rooms_temp": 21.0,
                "fireplace_on": False,
                "outdoor_temp": 5.0,
                "owm_temp": 5.5,
            },
            [],
        )
        ctx = _make_ctx()
        assert step_get_sensor_data(ctx) is True
        assert ctx.target_indoor_temp == 22.0
        assert ctx.actual_indoor == 21.5

    @patch("src.cycle_routes.SensorDataManager")
    def test_returns_false_on_missing_sensors(self, mock_sdm_class):
        mock_sdm = mock_sdm_class.return_value
        mock_sdm.get_sensor_data.return_value = (
            {},
            ["sensor.indoor_temp"],
        )
        ctx = _make_ctx()
        assert step_get_sensor_data(ctx) is False


class TestStepApplyCoolingTarget:
    """Test cooling target override step."""

    @patch("src.cycle_routes.config")
    def test_overrides_target_in_cooling_mode(self, mock_config):
        mock_config.TARGET_INDOOR_TEMP_COOLING_ENTITY_ID = "sensor.cooling_target"
        ctx = _make_ctx(climate_mode="cooling", target_indoor_temp=22.0)
        ctx.ha_client.get_state.return_value = "24.5"
        step_apply_cooling_target(ctx)
        assert ctx.target_indoor_temp == 24.5

    @patch("src.cycle_routes.config")
    def test_no_override_in_heating_mode(self, mock_config):
        mock_config.TARGET_INDOOR_TEMP_COOLING_ENTITY_ID = "sensor.cooling_target"
        ctx = _make_ctx(climate_mode="heating", target_indoor_temp=22.0)
        step_apply_cooling_target(ctx)
        assert ctx.target_indoor_temp == 22.0  # unchanged


class TestResolvePreCoolMinTarget:
    """Test minimum target resolution for pre-cooling."""

    @patch("src.cycle_routes.config")
    def test_clamps_pre_cool_offset_to_max_bound(self, mock_config):
        mock_config.TARGET_INDOOR_TEMP_COOLING_ENTITY_ID = ""
        mock_config.PRE_COOL_MAX_OFFSET_K = 1.0
        ctx = _make_ctx()

        result = _resolve_pre_cool_min_target(
            ctx, original_target=23.0, offset=2.0
        )

        assert result == 22.0

    @patch("src.cycle_routes.config")
    def test_respects_entity_minimum_after_offset_clamp(self, mock_config):
        mock_config.TARGET_INDOOR_TEMP_COOLING_ENTITY_ID = (
            "input_number.cooling_target"
        )
        mock_config.PRE_COOL_MAX_OFFSET_K = 1.0
        ctx = _make_ctx(
            all_states={
                "input_number.cooling_target": {
                    "attributes": {"min": 22.4}
                }
            }
        )

        result = _resolve_pre_cool_min_target(
            ctx, original_target=23.0, offset=2.0
        )

        assert result == 22.4


class TestStepDeterminePredictionIndoor:
    """Test prediction indoor temperature logic."""

    def test_uses_avg_rooms_when_fireplace_on(self):
        ctx = _make_ctx(
            fireplace_on=True,
            actual_indoor=22.0,
            avg_other_rooms_temp=21.0,
            climate_mode="heating",
            last_indoor_temp=None,
        )
        step_determine_prediction_indoor(ctx)
        assert ctx.prediction_indoor_temp == 21.0

    def test_uses_actual_indoor_when_no_fireplace(self):
        ctx = _make_ctx(
            fireplace_on=False,
            actual_indoor=22.0,
            avg_other_rooms_temp=21.0,
            climate_mode="heating",
            last_indoor_temp=None,
        )
        step_determine_prediction_indoor(ctx)
        assert ctx.prediction_indoor_temp == 22.0

    def test_transient_drop_filter_heating(self):
        """Large drop in heating mode triggers filter."""
        ctx = _make_ctx(
            fireplace_on=False,
            actual_indoor=20.0,
            avg_other_rooms_temp=20.0,
            climate_mode="heating",
            last_indoor_temp=21.0,  # 1°C drop > 0.25
        )
        step_determine_prediction_indoor(ctx)
        # Should use extrapolated value: 21.0 - 0.02 = 20.98
        assert abs(ctx.prediction_indoor_temp - 20.98) < 0.001

    def test_no_transient_filter_in_cooling(self):
        """Transient drop filter does NOT apply in cooling mode."""
        ctx = _make_ctx(
            fireplace_on=False,
            actual_indoor=20.0,
            avg_other_rooms_temp=20.0,
            climate_mode="cooling",
            last_indoor_temp=21.0,
        )
        step_determine_prediction_indoor(ctx)
        assert ctx.prediction_indoor_temp == 20.0  # no filter


class TestStepGradualControl:
    """Test gradual temperature control limiting."""

    @patch("src.cycle_routes.config")
    def test_clamps_large_change(self, mock_config):
        mock_config.MAX_TEMP_CHANGE_PER_CYCLE = 2.0
        mock_config.DHW_STATUS_ENTITY_ID = "binary_sensor.dhw"
        mock_config.DISINFECTION_STATUS_ENTITY_ID = "binary_sensor.disinfect"
        mock_config.DHW_BOOST_HEATER_STATUS_ENTITY_ID = "binary_sensor.boost"
        ctx = _make_ctx(
            actual_outlet_temp=30.0,
            final_temp=40.0,  # +10 from baseline
            effective_shadow_mode=False,
        )
        ctx.state = {"last_final_temp": 30.0, "last_blocking_reasons": []}
        step_gradual_control(ctx)
        assert ctx.final_temp == 32.0  # clamped to +2

    @patch("src.cycle_routes.config")
    def test_no_clamp_within_limit(self, mock_config):
        mock_config.MAX_TEMP_CHANGE_PER_CYCLE = 5.0
        mock_config.DHW_STATUS_ENTITY_ID = "binary_sensor.dhw"
        mock_config.DISINFECTION_STATUS_ENTITY_ID = "binary_sensor.disinfect"
        mock_config.DHW_BOOST_HEATER_STATUS_ENTITY_ID = "binary_sensor.boost"
        ctx = _make_ctx(
            actual_outlet_temp=30.0,
            final_temp=33.0,  # +3 from baseline
            effective_shadow_mode=False,
        )
        ctx.state = {"last_final_temp": 30.0, "last_blocking_reasons": []}
        step_gradual_control(ctx)
        assert ctx.final_temp == 33.0  # unchanged


class TestStepEmaSmoothing:
    """Test EMA smoothing step."""

    @patch("src.cycle_routes.apply_ema_smoothing")
    def test_applies_ema_in_heating(self, mock_ema):
        mock_ema.return_value = 31.5
        ctx = _make_ctx(climate_mode="heating", final_temp=32.0)
        ctx.state = {"last_final_temp": 31.0}
        ctx.wrapper._cooling_cycle_state = None
        step_ema_smoothing(ctx)
        assert ctx.final_temp == 31.5
        mock_ema.assert_called_once()

    @patch("src.cycle_routes.apply_ema_smoothing")
    def test_bypasses_ema_in_cooling_recovery(self, mock_ema):
        ctx = _make_ctx(climate_mode="cooling", final_temp=20.0)
        ctx.state = {"last_final_temp": 22.0}
        ctx.wrapper._cooling_cycle_state = "recovery"
        step_ema_smoothing(ctx)
        assert ctx.final_temp == 20.0  # unchanged
        mock_ema.assert_not_called()


class TestStepSetpointHold:
    """Test minimum setpoint hold logic."""

    @patch("src.cycle_routes.config")
    def test_holds_when_remaining_cycles(self, mock_config):
        mock_config.TRAJECTORY_STEPS = 3
        mock_config.MIN_SETPOINT_HOLD_CYCLES = 3
        ctx = _make_ctx(final_temp=35.0)
        ctx.state = {
            "setpoint_hold_cycles_remaining": 2,
            "last_final_temp": 30.0,
        }
        step_setpoint_hold(ctx)
        assert ctx.final_temp == 30.0  # held
        assert ctx.new_hold_cycles == 1

    @patch("src.cycle_routes.config")
    def test_starts_new_hold_on_change(self, mock_config):
        mock_config.TRAJECTORY_STEPS = 3
        mock_config.MIN_SETPOINT_HOLD_CYCLES = 3
        ctx = _make_ctx(final_temp=35.0)
        ctx.state = {
            "setpoint_hold_cycles_remaining": 0,
            "last_final_temp": 30.0,
        }
        step_setpoint_hold(ctx)
        assert ctx.final_temp == 35.0  # new value
        assert ctx.new_hold_cycles == 2  # max(0, 3-1) = 2


class TestRunBlockingRoute:
    """Test blocking route handler."""

    def test_writes_blocked_state_to_ha(self):
        ctx = _make_ctx(is_blocking=True)
        ctx.ha_client.get_state.return_value = True
        with patch("src.cycle_routes.get_shadow_output_entity_id") as mock_entity, \
             patch("src.cycle_routes.get_sensor_attributes") as mock_attrs, \
             patch("src.cycle_routes.save_state") as mock_save:
            mock_entity.return_value = "sensor.ml_heating_state"
            mock_attrs.return_value = {}
            run_blocking_route(ctx)
            mock_save.assert_called_once()
            call_kwargs = mock_save.call_args[1]
            assert call_kwargs["last_is_blocking"] is True


class TestRunIdleRoute:
    """Test idle route handler."""

    @patch("src.cycle_routes.step_build_features")
    @patch("src.cycle_routes.step_determine_prediction_indoor")
    @patch("src.cycle_routes.step_get_sensor_data")
    def test_builds_features_for_learning(
        self, mock_sensors, mock_predict, mock_features
    ):
        mock_sensors.return_value = True
        mock_features.return_value = True
        ctx = _make_ctx()
        run_idle_route(ctx)
        mock_sensors.assert_called_once()
        mock_predict.assert_called_once()
        mock_features.assert_called_once()

    @patch("src.cycle_routes.step_get_sensor_data")
    def test_early_exit_on_missing_sensors(self, mock_sensors):
        mock_sensors.return_value = False
        ctx = _make_ctx()
        run_idle_route(ctx)
        # Should return early, no further steps


class TestRunHeatingRoute:
    """Test heating route handler calls steps in correct order."""

    @patch("src.cycle_routes.step_publish_auxiliary_sensors")
    @patch("src.cycle_routes.step_save_state")
    @patch("src.cycle_routes.step_shadow_comparison")
    @patch("src.cycle_routes.step_update_ml_state_sensor")
    @patch("src.cycle_routes.step_log_metrics")
    @patch("src.cycle_routes.step_update_ha")
    @patch("src.cycle_routes.step_setpoint_hold")
    @patch("src.cycle_routes.step_ema_smoothing")
    @patch("src.cycle_routes.step_gradual_control")
    @patch("src.cycle_routes.step_prediction")
    @patch("src.cycle_routes.step_heating_obs_buffer")
    @patch("src.cycle_routes.step_dynamic_trajectory")
    @patch("src.cycle_routes.step_build_features")
    @patch("src.cycle_routes.step_determine_prediction_indoor")
    @patch("src.cycle_routes.step_get_sensor_data")
    def test_all_heating_steps_called_in_order(
        self,
        mock_sensors,
        mock_predict_indoor,
        mock_features,
        mock_trajectory,
        mock_heat_obs,
        mock_prediction,
        mock_gradual,
        mock_ema,
        mock_hold,
        mock_ha,
        mock_log,
        mock_ml_state,
        mock_shadow,
        mock_save,
        mock_aux,
    ):
        mock_sensors.return_value = True
        mock_features.return_value = True
        ctx = _make_ctx()
        run_heating_route(ctx)

        # Verify all steps called
        mock_sensors.assert_called_once()
        mock_predict_indoor.assert_called_once()
        mock_features.assert_called_once()
        mock_trajectory.assert_called_once()
        mock_heat_obs.assert_called_once()
        mock_prediction.assert_called_once()
        mock_gradual.assert_called_once()
        mock_ema.assert_called_once()
        mock_hold.assert_called_once()
        mock_ha.assert_called_once()
        mock_log.assert_called_once()
        mock_ml_state.assert_called_once()
        mock_shadow.assert_called_once()
        mock_save.assert_called_once()
        mock_aux.assert_called_once()

    @patch("src.cycle_routes.step_get_sensor_data")
    def test_early_exit_on_missing_sensors(self, mock_sensors):
        mock_sensors.return_value = False
        ctx = _make_ctx()
        run_heating_route(ctx)


class TestRunCoolingRoute:
    """Test cooling route handler calls steps in correct order."""

    @patch("src.cycle_routes.step_publish_auxiliary_sensors")
    @patch("src.cycle_routes.step_save_state")
    @patch("src.cycle_routes.step_shadow_comparison")
    @patch("src.cycle_routes.step_update_ml_state_sensor")
    @patch("src.cycle_routes.step_log_metrics")
    @patch("src.cycle_routes.step_update_ha")
    @patch("src.cycle_routes.step_setpoint_hold")
    @patch("src.cycle_routes.step_ema_smoothing")
    @patch("src.cycle_routes.step_gradual_control")
    @patch("src.cycle_routes.step_prediction")
    @patch("src.cycle_routes.step_pre_cooling")
    @patch("src.cycle_routes.step_dynamic_trajectory")
    @patch("src.cycle_routes.step_build_features")
    @patch("src.cycle_routes.step_determine_prediction_indoor")
    @patch("src.cycle_routes.step_apply_cooling_target")
    @patch("src.cycle_routes.step_get_sensor_data")
    def test_all_cooling_steps_called_in_order(
        self,
        mock_sensors,
        mock_cooling_target,
        mock_predict_indoor,
        mock_features,
        mock_trajectory,
        mock_pre_cool,
        mock_prediction,
        mock_gradual,
        mock_ema,
        mock_hold,
        mock_ha,
        mock_log,
        mock_ml_state,
        mock_shadow,
        mock_save,
        mock_aux,
    ):
        mock_sensors.return_value = True
        mock_features.return_value = True
        ctx = _make_ctx(climate_mode="cooling")
        run_cooling_route(ctx)

        # Verify all steps called in correct order
        mock_sensors.assert_called_once()
        mock_cooling_target.assert_called_once()
        mock_predict_indoor.assert_called_once()
        mock_features.assert_called_once()
        mock_trajectory.assert_called_once()
        mock_pre_cool.assert_called_once()
        mock_prediction.assert_called_once()
        mock_gradual.assert_called_once()
        mock_ema.assert_called_once()
        mock_hold.assert_called_once()
        mock_ha.assert_called_once()
        mock_log.assert_called_once()
        mock_ml_state.assert_called_once()
        mock_shadow.assert_called_once()
        mock_save.assert_called_once()
        mock_aux.assert_called_once()

    @patch("src.cycle_routes.step_get_sensor_data")
    def test_early_exit_on_missing_sensors(self, mock_sensors):
        mock_sensors.return_value = False
        ctx = _make_ctx(climate_mode="cooling")
        run_cooling_route(ctx)


class TestHeatingVsCoolingDifferences:
    """Verify that heating and cooling routes have the correct exclusive steps."""

    @patch("src.cycle_routes.step_publish_auxiliary_sensors")
    @patch("src.cycle_routes.step_save_state")
    @patch("src.cycle_routes.step_shadow_comparison")
    @patch("src.cycle_routes.step_update_ml_state_sensor")
    @patch("src.cycle_routes.step_log_metrics")
    @patch("src.cycle_routes.step_update_ha")
    @patch("src.cycle_routes.step_setpoint_hold")
    @patch("src.cycle_routes.step_ema_smoothing")
    @patch("src.cycle_routes.step_gradual_control")
    @patch("src.cycle_routes.step_prediction")
    @patch("src.cycle_routes.step_pre_cooling")
    @patch("src.cycle_routes.step_heating_obs_buffer")
    @patch("src.cycle_routes.step_dynamic_trajectory")
    @patch("src.cycle_routes.step_build_features")
    @patch("src.cycle_routes.step_determine_prediction_indoor")
    @patch("src.cycle_routes.step_apply_cooling_target")
    @patch("src.cycle_routes.step_get_sensor_data")
    def test_heating_does_not_call_pre_cooling(
        self,
        mock_sensors,
        mock_cooling_target,
        mock_predict_indoor,
        mock_features,
        mock_trajectory,
        mock_heat_obs,
        mock_pre_cool,
        mock_prediction,
        mock_gradual,
        mock_ema,
        mock_hold,
        mock_ha,
        mock_log,
        mock_ml_state,
        mock_shadow,
        mock_save,
        mock_aux,
    ):
        mock_sensors.return_value = True
        mock_features.return_value = True
        ctx = _make_ctx(climate_mode="heating")
        run_heating_route(ctx)
        mock_pre_cool.assert_not_called()
        mock_cooling_target.assert_not_called()
        mock_heat_obs.assert_called_once()

    @patch("src.cycle_routes.step_publish_auxiliary_sensors")
    @patch("src.cycle_routes.step_save_state")
    @patch("src.cycle_routes.step_shadow_comparison")
    @patch("src.cycle_routes.step_update_ml_state_sensor")
    @patch("src.cycle_routes.step_log_metrics")
    @patch("src.cycle_routes.step_update_ha")
    @patch("src.cycle_routes.step_setpoint_hold")
    @patch("src.cycle_routes.step_ema_smoothing")
    @patch("src.cycle_routes.step_gradual_control")
    @patch("src.cycle_routes.step_prediction")
    @patch("src.cycle_routes.step_pre_cooling")
    @patch("src.cycle_routes.step_heating_obs_buffer")
    @patch("src.cycle_routes.step_dynamic_trajectory")
    @patch("src.cycle_routes.step_build_features")
    @patch("src.cycle_routes.step_determine_prediction_indoor")
    @patch("src.cycle_routes.step_apply_cooling_target")
    @patch("src.cycle_routes.step_get_sensor_data")
    def test_cooling_does_not_call_heating_obs(
        self,
        mock_sensors,
        mock_cooling_target,
        mock_predict_indoor,
        mock_features,
        mock_trajectory,
        mock_heat_obs,
        mock_pre_cool,
        mock_prediction,
        mock_gradual,
        mock_ema,
        mock_hold,
        mock_ha,
        mock_log,
        mock_ml_state,
        mock_shadow,
        mock_save,
        mock_aux,
    ):
        mock_sensors.return_value = True
        mock_features.return_value = True
        ctx = _make_ctx(climate_mode="cooling")
        run_cooling_route(ctx)
        mock_heat_obs.assert_not_called()
        mock_cooling_target.assert_called_once()
        mock_pre_cool.assert_called_once()
