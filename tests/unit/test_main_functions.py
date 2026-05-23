
import argparse
import pytest
from unittest.mock import patch, MagicMock, ANY
from types import SimpleNamespace
from src import main, config




@patch("src.main.train_thermal_equilibrium_model")
@patch("src.physics_calibration.backup_existing_calibration")
@patch("src.main.load_dotenv")
@patch("src.main.logging")
@patch("src.main.create_influx_service")
def test_main_calibrate_physics(
    mock_create_influx, mock_logging, mock_load_dotenv, mock_backup, mock_train
):
    """Test main function with --calibrate-physics argument."""
    # Arrange
    mock_backup.return_value = "/fake/path"
    mock_train.return_value = True

    # Act
    with patch("sys.argv", ["main.py", "--calibrate-physics"]):
        main.main()

    # Assert
    mock_backup.assert_called_once()
    mock_train.assert_called_once()


@patch("src.main.validate_thermal_model")
@patch("src.main.load_dotenv")
@patch("src.main.logging")
@patch("src.main.create_influx_service")
def test_main_validate_physics(
    mock_create_influx, mock_logging, mock_load_dotenv, mock_validate
):
    """Test main function with --validate-physics argument."""
    # Arrange
    mock_validate.return_value = True

    # Act
    with patch("sys.argv", ["main.py", "--validate-physics"]):
        main.main()

    # Assert
    mock_validate.assert_called_once()


@patch("src.main.BlockingStateManager")
@patch("src.main.HeatingSystemStateChecker")
@patch("src.main.time.sleep")
@patch("src.cycle_routes.save_state")
@patch("src.cycle_routes.simplified_outlet_prediction")
@patch("src.cycle_routes.build_physics_features")
@patch("src.main.create_ha_client")
@patch("src.main.load_state")
@patch("src.main.logging")
@patch("src.main.load_dotenv")
@patch("src.main.create_influx_service")
def test_main_loop_heating_off(
    mock_create_influx,
    mock_load_dotenv,
    mock_logging,
    mock_load_state,
    mock_create_ha_client,
    mock_build_features,
    mock_prediction,
    mock_save_state,
    mock_sleep,
    MockHeatingSystemStateChecker,
    MockBlockingStateManager,
):
    """Test main loop skips when heating is off and loop breaks."""
    # Arrange
    mock_ha_client = MagicMock()
    # On the second run of the main loop, create_ha_client will raise an
    # exception. This will be caught, and then poll_for_blocking will be
    # called, which will raise a second exception to stop the test.
    mock_create_ha_client.side_effect = [mock_ha_client, Exception("Stop loop")]

    # Mock BlockingStateManager
    mock_blocking_manager = MockBlockingStateManager.return_value
    mock_blocking_manager.check_blocking_state.return_value = (False, [])
    mock_blocking_manager.handle_grace_period.return_value = False
    mock_blocking_manager.poll_for_blocking.side_effect = StopIteration("Stop test")

    # Mock HeatingSystemStateChecker
    mock_heating_checker = MockHeatingSystemStateChecker.return_value
    mock_heating_checker.check_heating_active.return_value = False

    mock_load_state.return_value = {}

    # Act
    with patch("src.main.get_sensor_attributes", return_value={}):
        with pytest.raises(StopIteration, match="Stop test"):
            with patch("sys.argv", ["main.py"]):
                main.main()

    # Assert that the main logic was skipped
    mock_build_features.assert_not_called()
    mock_prediction.assert_not_called()
    mock_save_state.assert_not_called()

    # Assert that the system is idle
    # mock_sleep.assert_called_once_with(300)
    # Assert that poll was called once before breaking
    mock_blocking_manager.poll_for_blocking.assert_called_once()


@patch("src.pre_dispatch.resolve_shadow_mode")
@patch("src.cycle_routes.SensorDataManager")
@patch("src.main.BlockingStateManager")
@patch("src.main.HeatingSystemStateChecker")
@patch("src.main.time.sleep")
@patch("src.cycle_routes.save_state")
@patch("src.cycle_routes.simplified_outlet_prediction")
@patch("src.cycle_routes.build_physics_features")
@patch("src.main.create_ha_client")
@patch("src.main.load_state")
@patch("src.cycle_routes.apply_ema_smoothing")
@patch("src.model_wrapper.get_enhanced_model_wrapper")
@patch("src.main.load_dotenv")
@patch("src.main.create_influx_service")
def test_main_skips_ema_during_cooling_recovery(
    mock_create_influx,
    mock_load_dotenv,
    mock_get_wrapper,
    mock_apply_ema,
    mock_load_state,
    mock_create_ha_client,
    mock_build_features,
    mock_prediction,
    mock_save_state,
    mock_sleep,
    MockHeatingSystemStateChecker,
    MockBlockingStateManager,
    MockSensorDataManager,
    mock_resolve_shadow_mode,
):
    """Cooling recovery should bypass apply_ema_smoothing()."""
    mock_influx = MagicMock()
    mock_influx.fetch_recent_history.return_value = {}
    mock_create_influx.return_value = mock_influx

    mock_wrapper = MagicMock()
    mock_wrapper._cooling_cycle_state = "recovery"
    mock_wrapper.state_manager = MagicMock(state_file="/tmp/state.json")
    mock_get_wrapper.return_value = mock_wrapper

    mock_ha_client = MagicMock()
    mock_ha_client.get_all_states.return_value = {"sensor.dummy": {"state": "1"}}
    mock_ha_client.get_state.return_value = None
    mock_create_ha_client.return_value = mock_ha_client

    mock_load_state.return_value = {
        "last_final_temp": 26.0,
        "last_blocking_reasons": [],
    }
    mock_build_features.return_value = (
        {"pv_now": 0.0, "tv_on": 0.0, "inlet_temp": 22.0},
        [],
    )
    mock_prediction.return_value = (22.0, 0.9, {"predicted_indoor": 22.1})

    mock_resolve_shadow_mode.return_value = SimpleNamespace(
        effective_shadow_mode=True,
        should_publish_output_entities=False,
        shadow_deployment=False,
    )

    mock_blocking_manager = MockBlockingStateManager.return_value
    mock_blocking_manager.check_blocking_state.return_value = (False, [])
    mock_blocking_manager.handle_grace_period.return_value = False
    mock_blocking_manager.poll_for_blocking.side_effect = StopIteration("Stop test")

    mock_heating_checker = MockHeatingSystemStateChecker.return_value
    mock_heating_checker.check_heating_active.return_value = True
    mock_heating_checker.get_climate_mode.return_value = "cooling"

    mock_sensor_manager = MockSensorDataManager.return_value
    mock_sensor_manager.get_sensor_data.return_value = (
        {
            "target_indoor_temp": 22.0,
            "actual_indoor": 23.0,
            "actual_outlet_temp": 22.0,
            "avg_other_rooms_temp": 23.0,
            "fireplace_on": False,
            "outdoor_temp": 30.0,
            "owm_temp": 30.0,
        },
        [],
    )

    with patch("src.main.get_sensor_attributes", return_value={}):
        with pytest.raises(StopIteration, match="Stop test"):
            with patch("sys.argv", ["main.py"]):
                main.main()

    mock_apply_ema.assert_not_called()
