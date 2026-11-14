"""Tests for cycle route handlers and shared step functions."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime
import time

from src import cycle_routes
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
    step_publish_building_curves,
    step_save_state,
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


def test_near_controller_snapshot_uses_published_target_and_skips_shadow(
    monkeypatch,
):
    from src import forecast_near_controller

    monkeypatch.setattr(
        cycle_routes.config, "HEAT_FORECAST_NEAR_CONTROLLER_ENABLED", True
    )
    monkeypatch.setattr(
        cycle_routes.config, "TARGET_OUTLET_TEMP_ENTITY_ID", "sensor.target"
    )
    monkeypatch.setattr(
        cycle_routes, "get_shadow_output_entity_id", lambda entity: entity
    )
    monkeypatch.setattr(
        cycle_routes, "get_sensor_attributes", lambda _entity: {}
    )
    monkeypatch.setattr(cycle_routes, "save_state", lambda **_kwargs: None)
    snapshots = []
    monkeypatch.setattr(
        forecast_near_controller,
        "save_snapshot",
        lambda wrapper, **kwargs: snapshots.append(kwargs),
    )

    ctx = _make_ctx(
        final_temp=31.0,
        actual_indoor=22.0,
        target_indoor_temp=22.0,
        climate_mode="heating",
    )
    cycle_routes.step_update_ha(ctx)
    step_save_state(ctx)
    assert ctx.ha_client.set_state.call_args.args[1] == 31.0
    assert snapshots[-1]["outlet"] == 31.0

    ctx = _make_ctx(
        final_temp=31.0,
        actual_indoor=22.0,
        target_indoor_temp=22.0,
        climate_mode="heating",
        effective_shadow_mode=True,
    )
    cycle_routes.step_update_ha(ctx)
    step_save_state(ctx)
    assert ctx.ha_client.set_state.call_count == 0
    assert snapshots[-1]["outlet"] is None


def test_hp_start_reference_uses_required_supply_minus_offset(monkeypatch):
    from src import config

    monkeypatch.setattr(config, "HP_START_ESTIMATE_ENABLED", True)
    monkeypatch.setattr(config, "HP_START_ESTIMATE_USE_FOR_TRAJECTORY", True)
    monkeypatch.setattr(config, "HP_START_ESTIMATE_HEATING_ENTITY_ID", "sensor.start")
    monkeypatch.setattr(config, "HP_START_ESTIMATE_REQUIRED_OFFSET_K", 2.0)
    ctx = _make_ctx(
        all_states={
            "sensor.start": {
                "state": "60",
                "attributes": {
                    "status": "idle",
                    "mode": "heating",
                    "method": "degree_minutes",
                    "required_supply_c": 30.0,
                    "supply_actual_c": 28.0,
                    "expected_runtime_min": 120,
                    "confidence": 0.8,
                    "updated_at": datetime.now().astimezone().isoformat(),
                },
            }
        },
        actual_outlet_temp=29.5,
    )

    cycle_routes._attach_hp_start_window(ctx)

    assert ctx.wrapper._hp_start_reference_outlet == 28.0


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


class TestStepPublishBuildingCurves:
    @patch("src.cycle_routes._BUILDING_CURVE_PUBLISHER.publish")
    @patch("src.cycle_routes.config")
    def test_idle_cooling_uses_distinct_heating_and_cooling_targets(
        self, mock_config, mock_publish
    ):
        mock_config.BUILDING_CURVE_ENABLED = True
        mock_config.TARGET_INDOOR_TEMP_ENTITY_ID = "sensor.heating_target"
        mock_config.TARGET_INDOOR_TEMP_COOLING_ENTITY_ID = (
            "sensor.cooling_target"
        )
        mock_config.BUILDING_CURVE_POLY_DEGREE = 4
        ctx = _make_ctx(
            climate_mode="cooling",
            target_indoor_temp=22.0,
            features_dict={},
            actual_indoor=25.0,
            outdoor_temp=18.0,
        )
        ctx.ha_client.get_state.side_effect = lambda entity, *_args, **_kwargs: {
            "sensor.heating_target": "21.5",
            "sensor.cooling_target": "24.0",
        }.get(entity)

        step_publish_building_curves(ctx)

        assert mock_publish.call_args.args[2] == {
            "heating": 21.5,
            "cooling": 24.0,
        }

    @patch("src.cycle_routes._BUILDING_CURVE_PUBLISHER.publish")
    @patch("src.cycle_routes.config")
    def test_balance_observation_does_not_use_flow_rate_as_hvac_off_gate(
        self, mock_config, mock_publish
    ):
        mock_config.BUILDING_CURVE_ENABLED = True
        mock_config.TARGET_INDOOR_TEMP_COOLING_ENTITY_ID = "sensor.cooling_target"
        mock_config.TARGET_INDOOR_TEMP_ENTITY_ID = "sensor.heating_target"
        mock_config.FLOW_RATE_ENTITY_ID = "sensor.flow_rate"
        mock_config.BUILDING_CURVE_POLY_DEGREE = 4
        ctx = _make_ctx(
            climate_mode="heating",
            target_indoor_temp=21.0,
            features_dict={
                "flow_rate": 0.0,
                "thermal_power_kw": 0.0,
                "indoor_temp_delta_60m": 0.1,
                "flow_rate_available": True,
                "inlet_temp_available": True,
                "indoor_history_available": True,
            },
            actual_indoor=22.0,
            outdoor_temp=10.0,
        )
        ctx.ha_client.get_state.return_value = "24.0"

        step_publish_building_curves(ctx, allow_balance_observation=True)

        observation = mock_publish.call_args.kwargs["balance_observation"]
        assert "flow_rate" not in observation
        assert "flow_rate_available" not in observation
        assert all(
            call.args[0] != "sensor.flow_rate"
            for call in ctx.ha_client.get_state.call_args_list
        )

    @patch("src.cycle_routes._BUILDING_CURVE_PUBLISHER.publish")
    @patch("src.cycle_routes.config")
    def test_idle_balance_observation_uses_last_active_cooling_mode(
        self, mock_config, mock_publish
    ):
        mock_config.BUILDING_CURVE_ENABLED = True
        mock_config.TARGET_INDOOR_TEMP_COOLING_ENTITY_ID = "sensor.cooling_target"
        mock_config.TARGET_INDOOR_TEMP_ENTITY_ID = "sensor.heating_target"
        mock_config.BUILDING_CURVE_POLY_DEGREE = 4
        ctx = _make_ctx(
            climate_mode="heating",
            target_indoor_temp=21.0,
            features_dict={
                "flow_rate": 0.0,
                "thermal_power_kw": 0.0,
                "indoor_temp_delta_60m": 0.1,
                "flow_rate_available": True,
                "inlet_temp_available": True,
                "indoor_history_available": True,
            },
            actual_indoor=25.0,
            outdoor_temp=18.0,
        )
        ctx.wrapper.last_active_climate_mode = "cooling"
        ctx.ha_client.get_state.return_value = "24.0"

        step_publish_building_curves(ctx, allow_balance_observation=True)

        observation = mock_publish.call_args.kwargs["balance_observation"]
        assert observation["mode"] == "cooling"

    @patch("src.cycle_routes.save_state")
    def test_idle_state_saves_last_active_mode(self, mock_save):
        ctx = _make_ctx(
            climate_mode="heating", features_dict={}, suggested_temp=35.0
        )
        ctx.wrapper.last_active_climate_mode = "cooling"

        step_save_state(ctx)

        assert mock_save.call_args.kwargs["last_active_climate_mode"] == "cooling"
        assert mock_save.call_args.kwargs["last_computed_temp"] == 35.0


class TestStepRecordBuildingSample:
    def _ctx(
        self,
        flags=(),
        wrapper_modes=("off", "cooling"),
        outlet=30.0,
        inlet=28.0,
    ):
        cfg = cycle_routes.config
        values = {
            cfg.OUTDOOR_TEMP_ENTITY_ID: 8.0,
            cfg.INDOOR_TEMP_ENTITY_ID: 22.0,
            cfg.ACTUAL_OUTLET_TEMP_ENTITY_ID: outlet,
            cfg.INLET_TEMP_ENTITY_ID: inlet,
            cfg.FLOW_RATE_ENTITY_ID: 6.0,
            cfg.PV_POWER_ENTITY_ID: 1500.0,
        }
        active = set(flags)

        def get_state(entity_id, _cache=None, is_binary=False):
            if is_binary:
                return entity_id in active
            return values.get(entity_id)

        ctx = _make_ctx()
        ctx.ha_client.get_state.side_effect = get_state
        ctx.wrapper.last_raw_climate_mode = wrapper_modes[0]
        ctx.wrapper.last_active_climate_mode = wrapper_modes[1]
        return ctx

    def _record(self, ctx):
        with patch.object(
            cycle_routes._BUILDING_CURVE_PUBLISHER, "record_sample"
        ) as record:
            cycle_routes.step_record_building_sample(ctx)
        return record.call_args.args[1]

    def test_idle_sample_uses_last_active_mode_and_signed_thermal_power(self):
        sample = self._record(self._ctx())

        assert sample["mode"] == "cooling"
        assert sample["outdoor"] == 8.0
        assert sample["pv"] == 1500.0
        assert sample["dhw_active"] is False
        assert sample["return_temp"] == 28.0
        assert sample["outlet_temp"] == 30.0
        assert sample["flow"] == 6.0
        assert sample["thermal_power_kw"] == pytest.approx(
            0.2 * cycle_routes.config.SPECIFIC_HEAT_CAPACITY
        )

    def test_defrost_is_recorded_with_negative_thermal_power(self):
        ctx = self._ctx(
            flags=[cycle_routes.config.DEFROST_STATUS_ENTITY_ID],
            outlet=26.0,
            inlet=29.0,
        )

        sample = self._record(ctx)

        assert sample["dhw_active"] is False
        assert sample["thermal_power_kw"] < 0

    @pytest.mark.parametrize(
        "entity_name",
        [
            "DHW_STATUS_ENTITY_ID",
            "DISINFECTION_STATUS_ENTITY_ID",
            "DHW_BOOST_HEATER_STATUS_ENTITY_ID",
        ],
    )
    def test_dhw_type_blocking_is_flagged(self, entity_name):
        entity_id = getattr(cycle_routes.config, entity_name)

        sample = self._record(self._ctx(flags=[entity_id]))

        assert sample["dhw_active"] is True

    def test_missing_hydronic_sensor_leaves_thermal_power_unset(self):
        ctx = self._ctx()
        ctx.ha_client.get_state.side_effect = (
            lambda entity_id, _cache=None, is_binary=False: (
                False
                if is_binary
                else None
                if entity_id == cycle_routes.config.FLOW_RATE_ENTITY_ID
                else 5.0
            )
        )

        assert self._record(ctx)["thermal_power_kw"] is None


class TestBlockingAndGraceRecordSamples:
    @patch("src.cycle_routes.save_state")
    @patch("src.cycle_routes.step_record_building_sample")
    def test_blocking_route_records_a_sample(self, mock_record, _save):
        ctx = _make_ctx()

        run_blocking_route(ctx)

        mock_record.assert_called_once_with(ctx)

    @patch("src.cycle_routes.save_state")
    @patch("src.cycle_routes.step_record_building_sample")
    def test_grace_route_records_a_sample(self, mock_record, _save):
        ctx = _make_ctx(state={"last_final_temp": 30.0})

        cycle_routes.run_grace_period_route(ctx)

        mock_record.assert_called_once_with(ctx)

    @patch("src.cycle_routes.step_record_building_sample")
    @patch("src.cycle_routes.step_get_sensor_data", return_value=True)
    @patch("src.cycle_routes.step_build_features", return_value=False)
    @patch("src.cycle_routes.step_publish_building_curves")
    @patch("src.cycle_routes.step_determine_prediction_indoor")
    def test_idle_route_records_even_when_features_fail(
        self, _predict, _publish, _features, _sensors, mock_record
    ):
        ctx = _make_ctx()

        run_idle_route(ctx)

        mock_record.assert_called_once_with(ctx)


def test_idle_route_publishes_curves_without_balance_observation_on_feature_failure(
    monkeypatch,
):
    ctx = _make_ctx()
    publish = MagicMock()
    monkeypatch.setattr(cycle_routes, "step_get_sensor_data", lambda _ctx: True)
    monkeypatch.setattr(cycle_routes, "step_determine_prediction_indoor", lambda _ctx: None)
    monkeypatch.setattr(cycle_routes, "step_build_features", lambda _ctx: False)
    monkeypatch.setattr(cycle_routes, "step_publish_building_curves", publish)

    run_idle_route(ctx)

    publish.assert_called_once_with(ctx)
    assert not publish.call_args.kwargs.get("allow_balance_observation", False)


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

    @pytest.mark.parametrize("mode", ["heating", "cooling"])
    def test_disabled_clears_pending_hold_but_keeps_hysteresis(self, monkeypatch, mode):
        monkeypatch.setattr(cycle_routes.config, "SETPOINT_HOLD_ENABLED", False)
        monkeypatch.setattr(cycle_routes.config, "OUTLET_HYSTERESIS_K", 0.3)
        monkeypatch.setattr(cycle_routes, "_calm_conditions", lambda ctx: True)
        ctx = _make_ctx(climate_mode=mode, final_temp=30.2)
        ctx.state = {"last_final_temp": 30.0, "setpoint_hold_cycles_remaining": 3}
        step_setpoint_hold(ctx)
        assert ctx.final_temp == 30.0
        assert ctx.new_hold_cycles == 0
        ctx.final_temp = 32.0
        step_setpoint_hold(ctx)
        assert ctx.final_temp == 32.0
        assert ctx.new_hold_cycles == 0

    @pytest.mark.parametrize(
        "dynamic, minimum, maximum, expected",
        [(None, 4, 8, 3), (10, 4, 6, 5), (2, 4, 6, 3),
         (10, 8, 3, 2), (10, 0, 0, 0)],
    )
    def test_hold_duration_obeys_bounds(
        self, monkeypatch, dynamic, minimum, maximum, expected
    ):
        monkeypatch.setattr(cycle_routes.config, "SETPOINT_HOLD_ENABLED", True)
        monkeypatch.setattr(cycle_routes.config, "MIN_SETPOINT_HOLD_CYCLES", minimum)
        monkeypatch.setattr(cycle_routes.config, "MAX_SETPOINT_HOLD_CYCLES", maximum)
        ctx = _make_ctx(final_temp=35.0, dynamic_hold_cycles=dynamic)
        ctx.state = {"last_final_temp": 30.0}
        step_setpoint_hold(ctx)
        assert ctx.new_hold_cycles == expected

    def test_existing_countdown_respects_maximum(self, monkeypatch):
        monkeypatch.setattr(cycle_routes.config, "SETPOINT_HOLD_ENABLED", True)
        monkeypatch.setattr(cycle_routes.config, "MAX_SETPOINT_HOLD_CYCLES", 2)
        ctx = _make_ctx(final_temp=35.0)
        ctx.state = {"last_final_temp": 30.0, "setpoint_hold_cycles_remaining": 8}
        step_setpoint_hold(ctx)
        assert ctx.final_temp == 30.0
        assert ctx.new_hold_cycles == 0

    @pytest.mark.parametrize("mode", ["heating", "cooling"])
    def test_computed_value_is_accepted_after_maximum_cycles(self, monkeypatch, mode):
        monkeypatch.setattr(cycle_routes.config, "SETPOINT_HOLD_ENABLED", True)
        monkeypatch.setattr(cycle_routes.config, "MIN_SETPOINT_HOLD_CYCLES", 2)
        monkeypatch.setattr(cycle_routes.config, "MAX_SETPOINT_HOLD_CYCLES", 3)
        monkeypatch.setattr(cycle_routes.config, "OUTLET_HYSTERESIS_K", 0.0)
        ctx = _make_ctx(climate_mode=mode, final_temp=30.0, dynamic_hold_cycles=8)
        ctx.state = {"last_final_temp": 29.0}
        step_setpoint_hold(ctx)
        assert ctx.new_hold_cycles == 2
        for remaining in (1, 0):
            ctx.state = {
                "last_final_temp": ctx.final_temp,
                "setpoint_hold_cycles_remaining": ctx.new_hold_cycles,
            }
            ctx.final_temp = 32.0
            step_setpoint_hold(ctx)
            assert ctx.final_temp == 30.0
            assert ctx.new_hold_cycles == remaining
        ctx.state["setpoint_hold_cycles_remaining"] = ctx.new_hold_cycles
        ctx.final_temp = 32.0
        step_setpoint_hold(ctx)
        assert ctx.final_temp == 32.0

    def test_forecast_does_not_overwrite_configured_minimum(self, monkeypatch):
        from types import SimpleNamespace
        from src import pv_trajectory

        monkeypatch.setattr(cycle_routes.config, "PV_TRAJ_FORECAST_MODE_ENABLED", True)
        monkeypatch.setattr(cycle_routes.config, "ELECTRICITY_PRICE_ENABLED", False)
        monkeypatch.setattr(cycle_routes.config, "MIN_SETPOINT_HOLD_CYCLES", 4)
        monkeypatch.setattr(cycle_routes.config, "TRAJECTORY_STEPS", 4)
        monkeypatch.setattr(
            pv_trajectory, "get_forecast_trajectory_state",
            lambda *args, **kwargs: SimpleNamespace(dynamic_steps=9),
        )
        ctx = _make_ctx()
        cycle_routes.step_dynamic_trajectory(ctx)
        assert cycle_routes.config.TRAJECTORY_STEPS == 9
        assert cycle_routes.config.MIN_SETPOINT_HOLD_CYCLES == 4
        assert ctx.dynamic_hold_cycles == 9

    @patch("src.cycle_routes.config")
    def test_holds_when_remaining_cycles(self, mock_config):
        mock_config.SETPOINT_HOLD_ENABLED = True
        mock_config.MAX_SETPOINT_HOLD_CYCLES = 20
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
        mock_config.SETPOINT_HOLD_ENABLED = True
        mock_config.MAX_SETPOINT_HOLD_CYCLES = 20
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


class TestOutletStabilisers:
    """Demand-aware slew limit and setpoint hysteresis."""

    def _config(self, mock_config):
        mock_config.MAX_TEMP_CHANGE_PER_CYCLE = 2.0
        mock_config.OUTLET_CALM_SLEW_K = 1.0
        mock_config.OUTLET_HYSTERESIS_K = 0.3
        mock_config.OUTLET_DEMAND_DEFICIT_K = 0.3
        mock_config.OUTLET_DEMAND_TREND_K_PER_H = 0.1
        mock_config.MIN_SETPOINT_HOLD_CYCLES = 4
        mock_config.SETPOINT_HOLD_ENABLED = True
        mock_config.MAX_SETPOINT_HOLD_CYCLES = 20
        mock_config.TRAJECTORY_STEPS = 4
        mock_config.DHW_STATUS_ENTITY_ID = "binary_sensor.dhw"
        mock_config.DISINFECTION_STATUS_ENTITY_ID = "binary_sensor.disinfect"
        mock_config.DHW_BOOST_HEATER_STATUS_ENTITY_ID = "binary_sensor.boost"

    def _ctx(self, indoor, target, final_temp, **state):
        ctx = _make_ctx(
            actual_outlet_temp=26.0,
            actual_indoor=indoor,
            prediction_indoor_temp=indoor,
            target_indoor_temp=target,
            final_temp=final_temp,
            effective_shadow_mode=False,
        )
        ctx.features_dict = {"indoor_temp_delta_60m": 0.0}
        ctx.state = {"last_final_temp": 26.0, **state}
        return ctx

    @patch("src.cycle_routes.config")
    def test_calm_conditions_use_tighter_slew(self, mock_config):
        self._config(mock_config)
        ctx = self._ctx(23.0, 22.5, 29.0, last_target_indoor_temp=22.5)
        step_gradual_control(ctx)
        assert ctx.final_temp == 27.0

    @patch("src.cycle_routes.config")
    def test_demand_uses_full_slew(self, mock_config):
        self._config(mock_config)
        ctx = self._ctx(22.0, 22.5, 29.0, last_target_indoor_temp=22.5)
        step_gradual_control(ctx)
        assert ctx.final_temp == 28.0

    @patch("src.cycle_routes.config")
    def test_target_jump_uses_full_slew(self, mock_config):
        self._config(mock_config)
        ctx = self._ctx(23.0, 22.5, 29.0, last_target_indoor_temp=21.0)
        step_gradual_control(ctx)
        assert ctx.final_temp == 28.0

    @patch("src.cycle_routes.config")
    def test_small_change_is_discarded_when_calm(self, mock_config):
        self._config(mock_config)
        ctx = self._ctx(
            23.0, 22.5, 26.2,
            last_target_indoor_temp=22.5,
            setpoint_hold_cycles_remaining=0,
        )
        step_setpoint_hold(ctx)
        assert ctx.final_temp == 26.0
        assert ctx.new_hold_cycles == 0

    @patch("src.cycle_routes.config")
    def test_small_change_is_kept_on_demand(self, mock_config):
        self._config(mock_config)
        ctx = self._ctx(
            22.0, 22.5, 26.2,
            last_target_indoor_temp=22.5,
            setpoint_hold_cycles_remaining=0,
        )
        step_setpoint_hold(ctx)
        assert ctx.final_temp == 26.2
        assert ctx.new_hold_cycles == 3


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

    def test_rebuilds_heating_features_when_cooling_pump_has_stopped(
        self, monkeypatch
    ):
        ctx = _make_ctx(target_indoor_temp=21.0)
        ctx.wrapper.last_active_climate_mode = "cooling"
        monkeypatch.setattr(cycle_routes, "step_get_sensor_data", lambda _ctx: True)
        monkeypatch.setattr(
            cycle_routes,
            "step_apply_cooling_target",
            lambda cycle_ctx: setattr(cycle_ctx, "target_indoor_temp", 24.0),
        )
        monkeypatch.setattr(
            cycle_routes, "step_determine_prediction_indoor", lambda _ctx: None
        )
        build_modes = []

        def build_features(cycle_ctx):
            build_modes.append((cycle_ctx.climate_mode, cycle_ctx.target_indoor_temp))
            cycle_ctx.features_dict = {
                "thermal_power_kw": 0.0,
                "delta_t": 0.0,
                "inlet_temp": 28.0,
                "outlet_temp": 28.0,
            }
            return True

        monkeypatch.setattr(cycle_routes, "step_build_features", build_features)
        monkeypatch.setattr(cycle_routes, "step_record_building_sample", MagicMock())
        publish_curves = MagicMock()
        publish_features = MagicMock()
        save_state = MagicMock()
        monkeypatch.setattr(
            cycle_routes, "step_publish_building_curves", publish_curves
        )
        monkeypatch.setattr(cycle_routes, "step_publish_features", publish_features)
        monkeypatch.setattr(cycle_routes, "step_save_state", save_state)
        monkeypatch.setattr(cycle_routes, "step_dynamic_trajectory", MagicMock())
        monkeypatch.setattr(cycle_routes, "step_heating_obs_buffer", MagicMock())
        monkeypatch.setattr(cycle_routes, "step_log_metrics", MagicMock())

        run_idle_route(ctx)

        assert build_modes == [("cooling", 24.0), ("heating", 21.0)]
        assert ctx.climate_mode == "heating"
        assert ctx.target_indoor_temp == 21.0
        publish_features.assert_called_once_with(ctx)
        save_state.assert_called_once_with(ctx)
        assert publish_curves.call_args.kwargs["allow_balance_observation"] is True

    @pytest.mark.parametrize(
        ("mode", "thermal_power"),
        [("heating", 0.8), ("cooling", -0.8)],
    )
    def test_continues_active_mode_control_while_heat_pump_runs(
        self, mode, thermal_power, monkeypatch
    ):
        ctx = _make_ctx()
        ctx.wrapper.last_active_climate_mode = mode
        ctx.wrapper._heating_state_manager = ctx.wrapper.state_manager
        ctx.features_dict = {}
        monkeypatch.setattr(cycle_routes, "step_get_sensor_data", lambda _ctx: True)
        monkeypatch.setattr(
            cycle_routes, "step_determine_prediction_indoor", lambda _ctx: None
        )

        def build_features(_ctx):
            _ctx.features_dict = {
                "thermal_power_kw": thermal_power,
                "delta_t": 1.0 if mode == "heating" else -1.0,
                "inlet_temp": 28.0,
                "outlet_temp": 32.0 if mode == "heating" else 20.0,
            }
            return True

        monkeypatch.setattr(cycle_routes, "step_build_features", build_features)
        continue_control = MagicMock()
        monkeypatch.setattr(
            cycle_routes, "_run_active_control_steps", continue_control
        )
        monkeypatch.setattr(
            cycle_routes, "step_publish_building_curves", MagicMock()
        )
        monkeypatch.setattr(cycle_routes, "step_publish_features", MagicMock())
        monkeypatch.setattr(cycle_routes, "step_record_building_sample", MagicMock())
        monkeypatch.setattr(cycle_routes, "step_apply_cooling_target", MagicMock())

        run_idle_route(ctx)

        assert ctx.climate_mode == mode
        ctx.wrapper.set_climate_mode.assert_called_once_with(mode)
        continue_control.assert_called_once_with(ctx)

    @patch("src.cycle_routes.step_build_features")
    @patch("src.cycle_routes.step_publish_features")
    @patch("src.cycle_routes.step_determine_prediction_indoor")
    @patch("src.cycle_routes.step_get_sensor_data")
    def test_builds_features_for_learning(
        self, mock_sensors, mock_predict, mock_publish_features, mock_features
    ):
        mock_sensors.return_value = True
        mock_features.return_value = True
        ctx = _make_ctx()
        run_idle_route(ctx)
        mock_sensors.assert_called_once()
        mock_predict.assert_called_once()
        mock_features.assert_called_once()
        mock_publish_features.assert_called_once_with(ctx)

    @patch("src.cycle_routes.step_get_sensor_data")
    def test_early_exit_on_missing_sensors(self, mock_sensors):
        mock_sensors.return_value = False
        ctx = _make_ctx()
        run_idle_route(ctx)
        # Should return early, no further steps

    @patch("src.cycle_routes.step_publish_building_curves")
    @patch("src.cycle_routes.step_build_features")
    @patch("src.cycle_routes.step_get_sensor_data")
    def test_publishes_building_curves_after_sensor_retrieval(
        self, mock_sensors, mock_features, mock_publish
    ):
        mock_sensors.return_value = True
        mock_features.return_value = True
        run_idle_route(_make_ctx())
        mock_publish.assert_called_once()
        assert mock_publish.call_args.kwargs["allow_balance_observation"] is True


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


class TestStepUpdateHaSmartRounding:
    """Smart rounding must target the price-adjusted indoor target."""

    @pytest.mark.parametrize("climate_mode", ["heating", "cooling"])
    def test_disabled_publishes_exact_final_temperature(
        self, monkeypatch, climate_mode
    ):
        from src.ha_client import HAClient

        monkeypatch.setattr(
            cycle_routes.config, "SMART_ROUNDING_ENABLED", False
        )
        monkeypatch.setattr(cycle_routes.config, "SHADOW_MODE", False)
        ctx = _make_ctx(
            final_temp=26.937,
            climate_mode=climate_mode,
            target_indoor_temp=22.6,
        )
        ctx.ha_client.set_state = MagicMock(
            wraps=HAClient("http://ha.test", "test-token").set_state
        )
        with patch(
            "src.model_wrapper.get_enhanced_model_wrapper"
        ) as get_wrapper, patch(
            "src.prediction_context.prediction_context_manager"
        ) as pcm, patch("src.ha_client.requests.post") as post:
            cycle_routes.step_update_ha(ctx)

        assert ctx.ha_client.set_state.call_args.args[1] == 26.937
        assert ctx.ha_client.set_state.call_args.kwargs["round_digits"] is None
        assert post.call_args.kwargs["json"]["state"] == "26.937"
        assert ctx.applied_outlet_temp == 26.937
        get_wrapper.assert_not_called()
        pcm.create_context.assert_not_called()

    @pytest.mark.parametrize("enabled", [True, False])
    @pytest.mark.parametrize("deployment", [True, False])
    def test_shadow_behavior_is_unchanged(self, monkeypatch, enabled, deployment):
        monkeypatch.setattr(
            cycle_routes.config, "SMART_ROUNDING_ENABLED", enabled
        )
        ctx = _make_ctx(final_temp=26.937, effective_shadow_mode=True)
        ctx.shadow_mode.shadow_deployment = deployment
        with patch(
            "src.model_wrapper.get_enhanced_model_wrapper"
        ) as get_wrapper:
            cycle_routes.step_update_ha(ctx)

        if deployment:
            assert ctx.ha_client.set_state.call_args.args[1] == 26.9
        else:
            ctx.ha_client.set_state.assert_not_called()
        assert ctx.applied_outlet_temp is None
        get_wrapper.assert_not_called()

    @pytest.mark.parametrize("climate_mode", ["heating", "cooling"])
    @pytest.mark.parametrize(
        "adjusted, expected", [(23.1, 27.0), (None, 26.0)]
    )
    def test_uses_adjusted_target(
        self, monkeypatch, adjusted, expected, climate_mode
    ):
        monkeypatch.setattr(
            cycle_routes.config, "SMART_ROUNDING_ENABLED", True
        )
        ctx = _make_ctx(
            final_temp=26.9,
            climate_mode=climate_mode,
            target_indoor_temp=22.6,
            prediction_indoor_temp=22.7,
            outdoor_temp=7.0,
            fireplace_on=False,
            features_dict={"pv_now": 0.0, "pv_power_history": []},
            metadata=(
                {"target_temp_adjusted": adjusted} if adjusted else {}
            ),
        )
        wrapper = MagicMock()
        # floor 26 -> 22.75, ceiling 27 -> 23.05
        wrapper.predict_indoor_temp.side_effect = (
            lambda outlet_temp, **kw: 22.75 if outlet_temp == 26 else 23.05
        )
        pcm = MagicMock()
        pcm.get_thermal_model_params.return_value = {
            "outdoor_temp": 7.0, "pv_power": 0.0,
            "fireplace_on": 0.0, "tv_on": 0.0,
        }
        with patch(
            "src.model_wrapper.get_enhanced_model_wrapper",
            return_value=wrapper,
        ), patch("src.prediction_context.prediction_context_manager", pcm):
            cycle_routes.step_update_ha(ctx)
        assert ctx.ha_client.set_state.call_args[0][1] == expected
        assert ctx.applied_outlet_temp == expected


def test_cost_mode_without_price_enabled_warns(caplog):
    ctx = _make_ctx(features_dict={})
    cycle_routes._COST_MODE_WARNED = False
    with patch.object(
        cycle_routes.config, "PRICE_COST_MODE_ENABLED", True
    ), patch.object(
        cycle_routes.config, "ELECTRICITY_PRICE_ENABLED", False
    ), patch.object(
        cycle_routes.config, "PV_TRAJ_FORECAST_MODE_ENABLED", False
    ):
        cycle_routes.step_dynamic_trajectory(ctx)
    assert "PRICE_COST_MODE_ENABLED has no effect" in caplog.text
