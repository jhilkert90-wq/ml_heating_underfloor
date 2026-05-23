"""Integration tests for the PR2 dispatch wiring in main.py.

These tests verify that the refactored main loop correctly dispatches
to the appropriate route handlers based on cycle state.
"""

from unittest.mock import patch, MagicMock

from src import config
from src.state_manager import SystemState


def _make_all_states(heating_status="heat", ml_control="on"):
    """Create a minimal all_states dict for testing."""
    return {
        config.HEATING_STATUS_ENTITY_ID: {"state": heating_status},
        config.ML_HEATING_CONTROL_ENTITY_ID: {"state": ml_control},
        config.TARGET_INDOOR_TEMP_ENTITY_ID: {"state": "21.0"},
        config.INDOOR_TEMP_ENTITY_ID: {"state": "20.5"},
        config.OUTDOOR_TEMP_ENTITY_ID: {"state": "10.0"},
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID: {"state": "45.0"},
        config.AVG_OTHER_ROOMS_TEMP_ENTITY_ID: {"state": "20.0"},
        config.FIREPLACE_STATUS_ENTITY_ID: {"state": "off"},
        config.OPENWEATHERMAP_TEMP_ENTITY_ID: {"state": "9.0"},
        config.DHW_STATUS_ENTITY_ID: {"state": "off"},
        config.DEFROST_STATUS_ENTITY_ID: {"state": "off"},
        config.DISINFECTION_STATUS_ENTITY_ID: {"state": "off"},
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID: {"state": "off"},
        config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID: {"state": "35.0"},
        config.TV_STATUS_ENTITY_ID: {"state": "off"},
        config.PV_POWER_ENTITY_ID: {"state": "0.0"},
        config.PV_FORECAST_ENTITY_ID: {"state": "0.0", "attributes": {}},
    }


def _get_state_side_effect(entity_id, states_dict, is_binary=False):
    """Standard HA get_state mock."""
    entity_info = states_dict.get(entity_id)
    if not entity_info:
        return None
    state = entity_info.get("state")
    if is_binary:
        return state == "on"
    try:
        return float(state)
    except (ValueError, TypeError):
        return state


class TestDispatchRouting:
    """Verify that the main loop dispatches to the correct route handler."""

    @patch("src.cycle_routes.get_sensor_attributes")
    @patch("src.cycle_routes.save_state")
    @patch("src.cycle_routes.simplified_outlet_prediction")
    @patch("src.cycle_routes.build_physics_features")
    @patch("src.cycle_routes.SensorDataManager")
    @patch("src.pre_dispatch.calculate_thermodynamic_metrics")
    @patch("src.main.create_ha_client")
    @patch("src.main.create_influx_service")
    @patch("src.main.load_state")
    def test_heating_route_dispatched_when_heat_mode(
        self,
        mock_load_state,
        mock_create_influx,
        mock_create_ha,
        mock_calc_metrics,
        mock_sensor_manager,
        mock_build_features,
        mock_prediction,
        mock_save_state,
        mock_get_attributes,
    ):
        """When climate_mode is 'heating' and heating is active, run_heating_route
        is called."""
        all_states = _make_all_states(heating_status="heat")
        mock_ha = MagicMock()
        mock_create_ha.return_value = mock_ha
        mock_ha.get_all_states.side_effect = [
            all_states,
            KeyboardInterrupt("end"),
        ]
        mock_ha.get_state.side_effect = _get_state_side_effect
        mock_get_attributes.side_effect = lambda *a: {}

        mock_influx = MagicMock()
        mock_create_influx.return_value = mock_influx

        mock_load_state.return_value = SystemState()
        mock_build_features.return_value = ({}, [])
        mock_prediction.return_value = (35.0, 0.9, {"predicted_indoor": 21.0})
        mock_calc_metrics.return_value = {}

        with (
            patch("src.model_wrapper.get_enhanced_model_wrapper") as mock_get_wrapper,
            patch("src.main.BlockingStateManager") as mock_bm,
            patch("src.main.HeatingSystemStateChecker") as mock_checker,
            patch("src.pre_dispatch.run_online_learning"),
            patch("src.pre_dispatch.handle_grace_period", return_value=False),
            patch(
                "src.pre_dispatch.check_and_resolve_climate_mode",
                return_value=(True, "heating", MagicMock(), None),
            ),
            patch("src.pre_dispatch.check_blocking_state", return_value=(False, [])),
            patch("src.main.time") as mock_time,
            patch("sys.argv", ["main.py"]),
            patch("src.cycle_routes.run_heating_route") as mock_heating_route,
        ):
            mock_wrapper = MagicMock()
            mock_get_wrapper.return_value = mock_wrapper
            mock_time.time.side_effect = [1000.0 + i for i in range(20)]
            mock_time.sleep.return_value = None
            mock_bm.return_value.poll_for_blocking.side_effect = KeyboardInterrupt

            mock_checker_instance = MagicMock()
            mock_checker.return_value = mock_checker_instance
            mock_checker_instance.get_climate_mode.return_value = "heating"

            from src import main

            try:
                main.main()
            except KeyboardInterrupt:
                pass

            mock_heating_route.assert_called_once()

    @patch("src.cycle_routes.get_sensor_attributes")
    @patch("src.cycle_routes.save_state")
    @patch("src.cycle_routes.simplified_outlet_prediction")
    @patch("src.cycle_routes.build_physics_features")
    @patch("src.cycle_routes.SensorDataManager")
    @patch("src.pre_dispatch.calculate_thermodynamic_metrics")
    @patch("src.main.create_ha_client")
    @patch("src.main.create_influx_service")
    @patch("src.main.load_state")
    def test_idle_route_dispatched_when_heating_inactive(
        self,
        mock_load_state,
        mock_create_influx,
        mock_create_ha,
        mock_calc_metrics,
        mock_sensor_manager,
        mock_build_features,
        mock_prediction,
        mock_save_state,
        mock_get_attributes,
    ):
        """When heating is not active, run_idle_route is dispatched."""
        all_states = _make_all_states(heating_status="off")
        mock_ha = MagicMock()
        mock_create_ha.return_value = mock_ha
        mock_ha.get_all_states.side_effect = [
            all_states,
            KeyboardInterrupt("end"),
        ]
        mock_ha.get_state.side_effect = _get_state_side_effect
        mock_get_attributes.side_effect = lambda *a: {}

        mock_influx = MagicMock()
        mock_create_influx.return_value = mock_influx

        mock_load_state.return_value = SystemState()
        mock_build_features.return_value = ({}, [])
        mock_prediction.return_value = (35.0, 0.9, {})
        mock_calc_metrics.return_value = {}

        with (
            patch("src.model_wrapper.get_enhanced_model_wrapper") as mock_get_wrapper,
            patch("src.main.BlockingStateManager") as mock_bm,
            patch("src.main.HeatingSystemStateChecker") as mock_checker,
            patch("src.pre_dispatch.run_online_learning"),
            patch("src.pre_dispatch.handle_grace_period", return_value=False),
            patch(
                "src.pre_dispatch.check_and_resolve_climate_mode",
                return_value=(False, "heating", MagicMock(), None),
            ),
            patch("src.pre_dispatch.check_blocking_state", return_value=(False, [])),
            patch("src.main.time") as mock_time,
            patch("sys.argv", ["main.py"]),
            patch("src.cycle_routes.run_idle_route") as mock_idle_route,
        ):
            mock_wrapper = MagicMock()
            mock_get_wrapper.return_value = mock_wrapper
            mock_time.time.side_effect = [1000.0 + i for i in range(20)]
            mock_time.sleep.return_value = None
            mock_bm.return_value.poll_for_blocking.side_effect = KeyboardInterrupt

            mock_checker_instance = MagicMock()
            mock_checker.return_value = mock_checker_instance
            mock_checker_instance.get_climate_mode.return_value = "heating"

            from src import main

            try:
                main.main()
            except KeyboardInterrupt:
                pass

            mock_idle_route.assert_called_once()

    @patch("src.cycle_routes.get_sensor_attributes")
    @patch("src.cycle_routes.save_state")
    @patch("src.cycle_routes.simplified_outlet_prediction")
    @patch("src.cycle_routes.build_physics_features")
    @patch("src.cycle_routes.SensorDataManager")
    @patch("src.pre_dispatch.calculate_thermodynamic_metrics")
    @patch("src.main.create_ha_client")
    @patch("src.main.create_influx_service")
    @patch("src.main.load_state")
    def test_blocking_route_dispatched_when_blocked(
        self,
        mock_load_state,
        mock_create_influx,
        mock_create_ha,
        mock_calc_metrics,
        mock_sensor_manager,
        mock_build_features,
        mock_prediction,
        mock_save_state,
        mock_get_attributes,
    ):
        """When blocking is detected, run_blocking_route is dispatched."""
        all_states = _make_all_states(heating_status="heat")
        mock_ha = MagicMock()
        mock_create_ha.return_value = mock_ha
        mock_ha.get_all_states.side_effect = [
            all_states,
            KeyboardInterrupt("end"),
        ]
        mock_ha.get_state.side_effect = _get_state_side_effect
        mock_get_attributes.side_effect = lambda *a: {}

        mock_influx = MagicMock()
        mock_create_influx.return_value = mock_influx

        mock_load_state.return_value = SystemState()
        mock_build_features.return_value = ({}, [])
        mock_prediction.return_value = (35.0, 0.9, {})
        mock_calc_metrics.return_value = {}

        with (
            patch("src.model_wrapper.get_enhanced_model_wrapper") as mock_get_wrapper,
            patch("src.main.BlockingStateManager") as mock_bm,
            patch("src.main.HeatingSystemStateChecker") as mock_checker,
            patch("src.pre_dispatch.run_online_learning"),
            patch("src.pre_dispatch.handle_grace_period", return_value=False),
            patch(
                "src.pre_dispatch.check_and_resolve_climate_mode",
                return_value=(True, "heating", MagicMock(), None),
            ),
            patch(
                "src.pre_dispatch.check_blocking_state",
                return_value=(True, ["defrost"]),
            ),
            patch("src.main.time") as mock_time,
            patch("sys.argv", ["main.py"]),
            patch("src.cycle_routes.run_blocking_route") as mock_blocking_route,
        ):
            mock_wrapper = MagicMock()
            mock_get_wrapper.return_value = mock_wrapper
            mock_time.time.side_effect = [1000.0 + i for i in range(20)]
            mock_time.sleep.return_value = None
            mock_bm.return_value.poll_for_blocking.side_effect = KeyboardInterrupt

            mock_checker_instance = MagicMock()
            mock_checker.return_value = mock_checker_instance
            mock_checker_instance.get_climate_mode.return_value = "heating"

            from src import main

            try:
                main.main()
            except KeyboardInterrupt:
                pass

            mock_blocking_route.assert_called_once()


class TestModeTransitionStateReload:
    """Verify that state is reloaded after climate mode transition."""

    @patch("src.cycle_routes.get_sensor_attributes")
    @patch("src.cycle_routes.save_state")
    @patch("src.cycle_routes.simplified_outlet_prediction")
    @patch("src.cycle_routes.build_physics_features")
    @patch("src.cycle_routes.SensorDataManager")
    @patch("src.pre_dispatch.calculate_thermodynamic_metrics")
    @patch("src.main.create_ha_client")
    @patch("src.main.create_influx_service")
    @patch("src.main.load_state")
    def test_state_reloaded_on_mode_transition(
        self,
        mock_load_state,
        mock_create_influx,
        mock_create_ha,
        mock_calc_metrics,
        mock_sensor_manager,
        mock_build_features,
        mock_prediction,
        mock_save_state,
        mock_get_attributes,
    ):
        """When check_and_resolve_climate_mode returns a reloaded state,
        that state is used for dispatch."""
        all_states = _make_all_states(heating_status="cool")
        mock_ha = MagicMock()
        mock_create_ha.return_value = mock_ha
        mock_ha.get_all_states.side_effect = [
            all_states,
            KeyboardInterrupt("end"),
        ]
        mock_ha.get_state.side_effect = _get_state_side_effect
        mock_get_attributes.side_effect = lambda *a: {}

        mock_influx = MagicMock()
        mock_create_influx.return_value = mock_influx

        # Initial state loaded normally
        initial_state = SystemState(last_indoor_temp=20.0)
        # Reloaded state after mode transition
        reloaded_state = SystemState(last_indoor_temp=22.0)
        mock_load_state.return_value = initial_state
        mock_build_features.return_value = ({}, [])
        mock_prediction.return_value = (28.0, 0.9, {})
        mock_calc_metrics.return_value = {}

        with (
            patch("src.model_wrapper.get_enhanced_model_wrapper") as mock_get_wrapper,
            patch("src.main.BlockingStateManager") as mock_bm,
            patch("src.main.HeatingSystemStateChecker") as mock_checker,
            patch("src.pre_dispatch.run_online_learning"),
            patch("src.pre_dispatch.handle_grace_period", return_value=False),
            patch(
                "src.pre_dispatch.check_and_resolve_climate_mode",
                return_value=(True, "cooling", MagicMock(), reloaded_state),
            ),
            patch("src.pre_dispatch.check_blocking_state", return_value=(False, [])),
            patch("src.main.time") as mock_time,
            patch("sys.argv", ["main.py"]),
            patch("src.cycle_routes.run_cooling_route") as mock_cooling_route,
        ):
            mock_wrapper = MagicMock()
            mock_get_wrapper.return_value = mock_wrapper
            mock_time.time.side_effect = [1000.0 + i for i in range(20)]
            mock_time.sleep.return_value = None
            mock_bm.return_value.poll_for_blocking.side_effect = KeyboardInterrupt

            mock_checker_instance = MagicMock()
            mock_checker.return_value = mock_checker_instance
            mock_checker_instance.get_climate_mode.return_value = "cooling"

            from src import main

            try:
                main.main()
            except KeyboardInterrupt:
                pass

            # Verify cooling route was called
            mock_cooling_route.assert_called_once()
            # The CycleContext passed should use the reloaded state
            ctx = mock_cooling_route.call_args[0][0]
            assert ctx.state == reloaded_state
