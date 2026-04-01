
from unittest.mock import patch, MagicMock
from src import config
from src.state_manager import SystemState
from src.heating_controller import BlockingStateManager

# Since main.py is the entry point, we will test it in an
# integration-style manner. We will mock external dependencies and assert
# that the main function behaves as expected.


@patch("src.main.get_sensor_attributes")
@patch("src.main.build_physics_features", return_value=({}, []))
@patch("src.main.create_ha_client")
@patch("src.main.create_influx_service")
@patch("src.main.simplified_outlet_prediction")
@patch("src.main.load_state")
@patch("src.main.save_state")
def test_main_loop_runs_once(
    mock_save_state,
    mock_load_state,
    mock_simplified_outlet_prediction,
    mock_create_influx_service,
    mock_create_ha_client,
    mock_build_features,
    mock_get_attributes,
):
    """Test that the main loop runs once and calls expected functions."""
    # Arrange
    mock_ha_instance = MagicMock()
    mock_create_ha_client.return_value = mock_ha_instance

    # Ensure get_sensor_attributes returns a fresh dict each time to avoid
    # shared state between calls (e.g. between status sensor and outlet sensor)
    mock_get_attributes.side_effect = lambda *args: {}

    # Mock InfluxService
    mock_influx_instance = MagicMock()
    mock_create_influx_service.return_value = mock_influx_instance

    with patch.object(config, "SHADOW_MODE", False), patch.object(
        config, "INDOOR_TEMP_ENTITY_ID", "sensor.test_indoor_temp"
    ), patch.object(
        config,
        "AVG_OTHER_ROOMS_TEMP_ENTITY_ID",
        "sensor.test_avg_other_rooms_temp",
    ):
        # Define all states for the HA client to return
        all_states = {
            config.HEATING_STATUS_ENTITY_ID: {"state": "heat"},
            config.ML_HEATING_CONTROL_ENTITY_ID: {"state": "on"},
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
            config.PV_FORECAST_ENTITY_ID: {
                "state": "0.0",
                "attributes": {},
            },
        }

        # Mock get_all_states to return our dictionary
        # We use side_effect to raise KeyboardInterrupt on the 3rd call
        # to break the infinite loop in main()
        # Calls:
        # 1. Main loop start
        # 2. sensor_manager.get_sensor_data()
        # 3. Next loop start -> KeyboardInterrupt
        mock_ha_instance.get_all_states.side_effect = [
            all_states,
            all_states,
            KeyboardInterrupt("End of test loop"),
        ]

        # Correctly mock get_state to look up from the all_states dictionary
        def get_state_side_effect(entity_id, states_dict, is_binary=False):
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

        mock_ha_instance.get_state.side_effect = get_state_side_effect

        mock_load_state.return_value = SystemState()
        mock_simplified_outlet_prediction.return_value = (
            35.0,
            0.9,
            {"predicted_indoor": 21.1},
        )

        # Act
        from src import main

        # We need to patch BlockingStateManager.poll_for_blocking to avoid
        # infinite loop or long sleep, but we use the real class for other methods
        with patch.object(main, "time") as mock_time, patch(
            "src.model_wrapper.get_enhanced_model_wrapper"
        ) as mock_get_wrapper, patch.object(
            BlockingStateManager, "poll_for_blocking"
        ) as mock_poll_blocking:

            mock_wrapper = MagicMock()
            mock_get_wrapper.return_value = mock_wrapper
            mock_wrapper.predict_indoor_temp.return_value = 21.0

            # Mock time.time() to advance
            mock_time.time.side_effect = [
                1000.0,
                1001.0,
                1002.0,
                1003.0,
                1004.0,
            ]

            # Mock sleep to do nothing
            mock_time.sleep.return_value = None

            # Run main() - it should catch KeyboardInterrupt and exit
            with patch("sys.argv", ["main.py"]):
                try:
                    main.main()
                except KeyboardInterrupt:
                    pass

    # Assert
    # load_state is called at the start of each loop.
    # 1st call: First iteration (successful)
    # 2nd call: Second iteration (starts, then KeyboardInterrupt at get_all_states)
    assert mock_load_state.call_count == 2
    mock_simplified_outlet_prediction.assert_called_once()
    mock_create_influx_service.assert_called_once()
    assert mock_create_ha_client.call_count > 0
    
    # Verify save_state was called with expected values
    # This confirms data flowed through SensorDataManager to main logic
    mock_save_state.assert_called()
    call_kwargs = mock_save_state.call_args[1]
    assert call_kwargs.get("last_indoor_temp") == 20.5
    assert call_kwargs.get("last_final_temp") == 43.0
    
    # Verify set_state was called to update the target temperature
    # This confirms the prediction result was used
    # Note: The actual call might have attributes populated by get_sensor_attributes
    # so we check for the main arguments and ignore the exact attributes content
    # or use ANY for attributes if needed.
    # Based on main.py:
    # ha_client.set_state(
    #     config.TARGET_OUTLET_TEMP_ENTITY_ID,
    #     smart_rounded_temp,
    #     get_sensor_attributes(config.TARGET_OUTLET_TEMP_ENTITY_ID),
    #     round_digits=None,  # No additional rounding needed
    # )
    mock_ha_instance.set_state.assert_any_call(
        config.TARGET_OUTLET_TEMP_ENTITY_ID,
        43.0,
        {},
        round_digits=None,
    )

    mock_poll_blocking.assert_called_once()
    mock_build_features.assert_called_once()


@patch("src.main.get_sensor_attributes")
@patch("src.main.build_physics_features", return_value=({}, []))
@patch("src.main.create_ha_client")
@patch("src.main.create_influx_service")
@patch("src.main.simplified_outlet_prediction")
@patch("src.main.load_state")
@patch("src.main.save_state")
def test_main_online_learning_passes_previous_cycle_heat_source_context(
    mock_save_state,
    mock_load_state,
    mock_simplified_outlet_prediction,
    mock_create_influx_service,
    mock_create_ha_client,
    mock_build_features,
    mock_get_attributes,
):
    """Previous-cycle fireplace context should flow into online learning."""
    mock_ha_instance = MagicMock()
    mock_create_ha_client.return_value = mock_ha_instance
    mock_get_attributes.side_effect = lambda *args: {}

    mock_influx_instance = MagicMock()
    mock_create_influx_service.return_value = mock_influx_instance

    with patch.object(config, "SHADOW_MODE", False), patch.object(
        config, "INDOOR_TEMP_ENTITY_ID", "sensor.test_indoor_temp"
    ), patch.object(
        config,
        "AVG_OTHER_ROOMS_TEMP_ENTITY_ID",
        "sensor.test_avg_other_rooms_temp",
    ):
        all_states = {
            config.HEATING_STATUS_ENTITY_ID: {"state": "heat"},
            config.ML_HEATING_CONTROL_ENTITY_ID: {"state": "on"},
            config.TARGET_INDOOR_TEMP_ENTITY_ID: {"state": "21.0"},
            config.INDOOR_TEMP_ENTITY_ID: {"state": "21.0"},
            config.OUTDOOR_TEMP_ENTITY_ID: {"state": "10.0"},
            config.ACTUAL_OUTLET_TEMP_ENTITY_ID: {"state": "45.0"},
            config.AVG_OTHER_ROOMS_TEMP_ENTITY_ID: {"state": "19.5"},
            config.FIREPLACE_STATUS_ENTITY_ID: {"state": "on"},
            config.OPENWEATHERMAP_TEMP_ENTITY_ID: {"state": "9.0"},
            config.DHW_STATUS_ENTITY_ID: {"state": "off"},
            config.DEFROST_STATUS_ENTITY_ID: {"state": "off"},
            config.DISINFECTION_STATUS_ENTITY_ID: {"state": "off"},
            config.DHW_BOOST_HEATER_STATUS_ENTITY_ID: {"state": "off"},
            config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID: {"state": "37.0"},
            config.TV_STATUS_ENTITY_ID: {"state": "on"},
            config.PV_POWER_ENTITY_ID: {"state": "0.0"},
            config.PV_FORECAST_ENTITY_ID: {
                "state": "0.0",
                "attributes": {},
            },
        }

        mock_ha_instance.get_all_states.side_effect = [
            all_states,
            all_states,
            KeyboardInterrupt("End of test loop"),
        ]

        def get_state_side_effect(entity_id, states_dict, is_binary=False):
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

        mock_ha_instance.get_state.side_effect = get_state_side_effect

        mock_load_state.return_value = SystemState(
            last_run_features={
                "outdoor_temp": 10.0,
                "pv_now": 0.0,
                "pv_power_history": [],
                "tv_on": 1.0,
                "thermal_power_kw": 0.0,
                "delta_t": 0.0,
                "inlet_temp": 20.0,
                "indoor_temp_gradient": 0.02,
                "indoor_temp_delta_60m": 0.1,
                "living_room_temp": 24.0,
                "cloud_cover_forecast_1h": 40.0,
                "cloud_cover_forecast_2h": 45.0,
                "cloud_cover_forecast_3h": 50.0,
                "cloud_cover_forecast_4h": 55.0,
                "cloud_cover_forecast_5h": 60.0,
                "cloud_cover_forecast_6h": 65.0,
            },
            last_indoor_temp=20.5,
            last_avg_other_rooms_temp=19.5,
            last_fireplace_on=True,
            last_final_temp=34.0,
        )
        mock_simplified_outlet_prediction.return_value = (
            35.0,
            0.9,
            {"predicted_indoor": 21.1},
        )

        from src import main

        with patch.object(main, "time") as mock_time, patch(
            "src.model_wrapper.get_enhanced_model_wrapper"
        ) as mock_get_wrapper, patch.object(
            BlockingStateManager, "poll_for_blocking"
        ) as mock_poll_blocking:

            mock_wrapper = MagicMock()
            mock_get_wrapper.return_value = mock_wrapper
            mock_wrapper.predict_indoor_temp.return_value = 21.0
            mock_wrapper.thermal_model.predict_thermal_trajectory.return_value = {
                "trajectory": [20.8]
            }

            mock_time.time.side_effect = [1000.0 + idx for idx in range(20)]
            mock_time.sleep.return_value = None

            with patch("sys.argv", ["main.py"]):
                try:
                    main.main()
                except KeyboardInterrupt:
                    pass

    mock_wrapper.learn_from_prediction_feedback.assert_called_once()
    learn_call = mock_wrapper.learn_from_prediction_feedback.call_args.kwargs
    learning_context = learn_call["prediction_context"]

    assert learn_call["predicted_temp"] == 20.8
    assert learn_call["actual_temp"] == 21.0
    assert learning_context["outlet_temp"] == 37.0
    assert learning_context["fireplace_on"] == 1.0
    assert learning_context["avg_other_rooms_temp"] == 19.5
    assert learning_context["living_room_temp"] == 24.0
    assert learning_context["tv_on"] == 1.0
    assert learning_context["learning_mode"] == "active_mode_ml_trajectory"
    assert learn_call["effective_shadow_mode"] is False

    mock_save_state.assert_called()
    mock_poll_blocking.assert_called_once()
    mock_build_features.assert_called_once()


@patch("src.main.get_sensor_attributes")
@patch("src.main.build_physics_features", return_value=({}, []))
@patch("src.main.create_ha_client")
@patch("src.main.create_influx_service")
@patch("src.main.simplified_outlet_prediction")
@patch("src.main.load_state")
@patch("src.main.save_state")
def test_main_dynamic_shadow_mode_suppresses_live_control_outputs(
    mock_save_state,
    mock_load_state,
    mock_simplified_outlet_prediction,
    mock_create_influx_service,
    mock_create_ha_client,
    mock_build_features,
    mock_get_attributes,
):
    """ML control OFF should behave as effective shadow mode for outputs."""
    mock_ha_instance = MagicMock()
    mock_create_ha_client.return_value = mock_ha_instance
    mock_get_attributes.side_effect = lambda *args: {}

    mock_influx_instance = MagicMock()
    mock_create_influx_service.return_value = mock_influx_instance

    with patch.object(config, "SHADOW_MODE", False), patch.object(
        config, "INDOOR_TEMP_ENTITY_ID", "sensor.test_indoor_temp"
    ), patch.object(
        config,
        "AVG_OTHER_ROOMS_TEMP_ENTITY_ID",
        "sensor.test_avg_other_rooms_temp",
    ):
        all_states = {
            config.HEATING_STATUS_ENTITY_ID: {"state": "heat"},
            config.ML_HEATING_CONTROL_ENTITY_ID: {"state": "off"},
            config.TARGET_INDOOR_TEMP_ENTITY_ID: {"state": "21.0"},
            config.INDOOR_TEMP_ENTITY_ID: {"state": "21.0"},
            config.OUTDOOR_TEMP_ENTITY_ID: {"state": "10.0"},
            config.ACTUAL_OUTLET_TEMP_ENTITY_ID: {"state": "45.0"},
            config.AVG_OTHER_ROOMS_TEMP_ENTITY_ID: {"state": "19.5"},
            config.FIREPLACE_STATUS_ENTITY_ID: {"state": "on"},
            config.OPENWEATHERMAP_TEMP_ENTITY_ID: {"state": "9.0"},
            config.DHW_STATUS_ENTITY_ID: {"state": "off"},
            config.DEFROST_STATUS_ENTITY_ID: {"state": "off"},
            config.DISINFECTION_STATUS_ENTITY_ID: {"state": "off"},
            config.DHW_BOOST_HEATER_STATUS_ENTITY_ID: {"state": "off"},
            config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID: {"state": "37.0"},
            config.TV_STATUS_ENTITY_ID: {"state": "on"},
            config.PV_POWER_ENTITY_ID: {"state": "0.0"},
            config.PV_FORECAST_ENTITY_ID: {
                "state": "0.0",
                "attributes": {},
            },
        }

        mock_ha_instance.get_all_states.side_effect = [
            all_states,
            all_states,
            KeyboardInterrupt("End of test loop"),
        ]

        def get_state_side_effect(entity_id, states_dict, is_binary=False):
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

        mock_ha_instance.get_state.side_effect = get_state_side_effect

        mock_load_state.return_value = SystemState(
            last_run_features={
                "outdoor_temp": 10.0,
                "pv_now": 0.0,
                "pv_power_history": [],
                "tv_on": 1.0,
                "thermal_power_kw": 0.0,
                "delta_t": 0.0,
                "inlet_temp": 20.0,
                "indoor_temp_gradient": 0.02,
                "indoor_temp_delta_60m": 0.1,
                "living_room_temp": 24.0,
                "cloud_cover_forecast_1h": 40.0,
                "cloud_cover_forecast_2h": 45.0,
                "cloud_cover_forecast_3h": 50.0,
                "cloud_cover_forecast_4h": 55.0,
                "cloud_cover_forecast_5h": 60.0,
                "cloud_cover_forecast_6h": 65.0,
            },
            last_indoor_temp=20.5,
            last_avg_other_rooms_temp=19.5,
            last_fireplace_on=True,
            last_final_temp=34.0,
        )
        mock_simplified_outlet_prediction.return_value = (
            35.0,
            0.9,
            {"predicted_indoor": 21.1},
        )

        from src import main

        with patch.object(main, "time") as mock_time, patch(
            "src.model_wrapper.get_enhanced_model_wrapper"
        ) as mock_get_wrapper, patch.object(
            BlockingStateManager, "poll_for_blocking"
        ) as mock_poll_blocking:
            mock_wrapper = MagicMock()
            mock_get_wrapper.return_value = mock_wrapper
            mock_wrapper.predict_indoor_temp.return_value = 21.0
            mock_wrapper.thermal_model.predict_thermal_trajectory.return_value = {
                "trajectory": [20.8]
            }

            mock_time.time.side_effect = [1000.0 + idx for idx in range(20)]
            mock_time.sleep.return_value = None

            with patch("sys.argv", ["main.py"]):
                try:
                    main.main()
                except KeyboardInterrupt:
                    pass

    mock_wrapper.learn_from_prediction_feedback.assert_called_once()
    learn_call = mock_wrapper.learn_from_prediction_feedback.call_args.kwargs
    assert learn_call["effective_shadow_mode"] is True
    assert learn_call["prediction_context"]["learning_mode"] == (
        "shadow_mode_hc_trajectory"
    )

    target_outlet_calls = [
        call for call in mock_ha_instance.set_state.call_args_list
        if call.args and call.args[0] == config.TARGET_OUTLET_TEMP_ENTITY_ID
    ]
    ml_state_calls = [
        call for call in mock_ha_instance.set_state.call_args_list
        if call.args and call.args[0] == "sensor.ml_heating_state"
    ]

    assert target_outlet_calls == []
    assert ml_state_calls == []
    mock_save_state.assert_called()
    mock_poll_blocking.assert_called_once()


@patch("src.main.get_sensor_attributes")
@patch("src.main.build_physics_features", return_value=({}, []))
@patch("src.main.create_ha_client")
@patch("src.main.create_influx_service")
@patch("src.main.simplified_outlet_prediction")
@patch("src.main.load_state")
@patch("src.main.save_state")
def test_main_shadow_deployment_writes_shadow_output_entities(
    mock_save_state,
    mock_load_state,
    mock_simplified_outlet_prediction,
    mock_create_influx_service,
    mock_create_ha_client,
    mock_build_features,
    mock_get_attributes,
):
    """Static shadow deployment should publish suffixed HA outputs."""
    mock_ha_instance = MagicMock()
    mock_create_ha_client.return_value = mock_ha_instance
    mock_get_attributes.side_effect = lambda *args: {}

    mock_influx_instance = MagicMock()
    mock_create_influx_service.return_value = mock_influx_instance

    with patch.object(config, "SHADOW_MODE", True), patch.object(
        config, "INDOOR_TEMP_ENTITY_ID", "sensor.test_indoor_temp"
    ), patch.object(
        config,
        "AVG_OTHER_ROOMS_TEMP_ENTITY_ID",
        "sensor.test_avg_other_rooms_temp",
    ):
        all_states = {
            config.HEATING_STATUS_ENTITY_ID: {"state": "heat"},
            config.ML_HEATING_CONTROL_ENTITY_ID: {"state": "on"},
            config.TARGET_INDOOR_TEMP_ENTITY_ID: {"state": "21.0"},
            config.INDOOR_TEMP_ENTITY_ID: {"state": "21.0"},
            config.OUTDOOR_TEMP_ENTITY_ID: {"state": "10.0"},
            config.ACTUAL_OUTLET_TEMP_ENTITY_ID: {"state": "45.0"},
            config.AVG_OTHER_ROOMS_TEMP_ENTITY_ID: {"state": "19.5"},
            config.FIREPLACE_STATUS_ENTITY_ID: {"state": "on"},
            config.OPENWEATHERMAP_TEMP_ENTITY_ID: {"state": "9.0"},
            config.DHW_STATUS_ENTITY_ID: {"state": "off"},
            config.DEFROST_STATUS_ENTITY_ID: {"state": "off"},
            config.DISINFECTION_STATUS_ENTITY_ID: {"state": "off"},
            config.DHW_BOOST_HEATER_STATUS_ENTITY_ID: {"state": "off"},
            config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID: {"state": "37.0"},
            config.TV_STATUS_ENTITY_ID: {"state": "on"},
            config.PV_POWER_ENTITY_ID: {"state": "0.0"},
            config.PV_FORECAST_ENTITY_ID: {
                "state": "0.0",
                "attributes": {},
            },
        }

        mock_ha_instance.get_all_states.side_effect = [
            all_states,
            all_states,
            KeyboardInterrupt("End of test loop"),
        ]

        def get_state_side_effect(entity_id, states_dict, is_binary=False):
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

        mock_ha_instance.get_state.side_effect = get_state_side_effect

        mock_load_state.return_value = SystemState(
            last_run_features={
                "outdoor_temp": 10.0,
                "pv_now": 0.0,
                "pv_power_history": [],
                "tv_on": 1.0,
                "thermal_power_kw": 4.2,
                "cop_realtime": 3.5,
                "delta_t": 5.0,
                "flow_rate": 800.0,
                "inlet_temp": 30.0,
                "indoor_temp_gradient": 0.02,
                "indoor_temp_delta_60m": 0.1,
                "living_room_temp": 24.0,
                "cloud_cover_forecast_1h": 40.0,
                "cloud_cover_forecast_2h": 45.0,
                "cloud_cover_forecast_3h": 50.0,
                "cloud_cover_forecast_4h": 55.0,
                "cloud_cover_forecast_5h": 60.0,
                "cloud_cover_forecast_6h": 65.0,
            },
            last_indoor_temp=20.5,
            last_avg_other_rooms_temp=19.5,
            last_fireplace_on=True,
            last_final_temp=34.0,
        )
        mock_simplified_outlet_prediction.return_value = (
            35.0,
            0.9,
            {"predicted_indoor": 21.1},
        )

        from src import main

        with patch.object(main, "time") as mock_time, patch(
            "src.model_wrapper.get_enhanced_model_wrapper"
        ) as mock_get_wrapper, patch.object(
            BlockingStateManager, "poll_for_blocking"
        ) as mock_poll_blocking, patch(
            "src.main.calculate_thermodynamic_metrics",
            return_value={
                "cop_realtime": 3.5,
                "thermal_power_kw": 4.2,
            },
        ):
            mock_wrapper = MagicMock()
            mock_get_wrapper.return_value = mock_wrapper
            mock_wrapper.predict_indoor_temp.return_value = 21.0
            mock_wrapper.thermal_model.predict_thermal_trajectory.return_value = {
                "trajectory": [20.8]
            }

            mock_time.time.side_effect = [1000.0 + idx for idx in range(20)]
            mock_time.sleep.return_value = None

            with patch("sys.argv", ["main.py"]):
                try:
                    main.main()
                except KeyboardInterrupt:
                    pass

    written_entity_ids = [
        call.args[0] for call in mock_ha_instance.set_state.call_args_list
    ]

    assert "sensor.ml_vorlauftemperatur_shadow" in written_entity_ids
    assert "sensor.ml_heating_state_shadow" in written_entity_ids
    assert "sensor.ml_heating_cop_realtime_shadow" in written_entity_ids
    assert "sensor.ml_heating_thermal_power_shadow" in written_entity_ids
    assert config.TARGET_OUTLET_TEMP_ENTITY_ID not in written_entity_ids

    mock_wrapper.learn_from_prediction_feedback.assert_called_once()
    assert (
        mock_wrapper.learn_from_prediction_feedback.call_args.kwargs[
            "effective_shadow_mode"
        ]
        is True
    )
    mock_save_state.assert_called()
    mock_poll_blocking.assert_called_once()
