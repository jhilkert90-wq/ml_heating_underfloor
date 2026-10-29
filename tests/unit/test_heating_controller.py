
import pytest
from unittest.mock import ANY, Mock, patch
from src import config
from src.heating_controller import (
    BlockingStateManager, SensorDataManager, HeatingSystemStateChecker
)
from src.state_manager import SystemState


# Mock HAClient for testing
@pytest.fixture
def mock_ha_client():
    return Mock()


@pytest.fixture(autouse=True)
def force_active_mode():
    with patch("src.heating_controller.config.SHADOW_MODE", False):
        yield


class TestBlockingStateManager:
    @pytest.fixture
    def blocking_manager(self):
        return BlockingStateManager()

    def test_check_blocking_state_no_blocking(
            self, blocking_manager, mock_ha_client):
        mock_ha_client.get_state.return_value = False
        is_blocking, reasons = blocking_manager.check_blocking_state(
            mock_ha_client, {})
        assert is_blocking is False
        assert len(reasons) == 0

    def test_check_blocking_state_with_blocking(
            self, blocking_manager, mock_ha_client):
        mock_ha_client.get_state.return_value = True
        is_blocking, reasons = blocking_manager.check_blocking_state(
            mock_ha_client, {})
        assert is_blocking is True
        assert len(reasons) > 0

    def test_handle_blocking_state(self, blocking_manager, mock_ha_client):
        with patch('src.heating_controller.save_state') as mock_save_state:
            state = SystemState()
            skip_cycle = blocking_manager.handle_blocking_state(
                mock_ha_client, True, ["reason"], state)
            assert skip_cycle is True
            mock_save_state.assert_called_once()

    def test_handle_blocking_state_ha_error(self, blocking_manager, mock_ha_client):
        """Test handle_blocking_state handles Home Assistant errors."""
        mock_ha_client.set_state.side_effect = Exception("HA API Error")
        with patch("src.heating_controller.save_state") as mock_save_state, patch(
            "src.heating_controller.logging"
        ) as mock_logging:
            state = SystemState()
            skip_cycle = blocking_manager.handle_blocking_state(
                mock_ha_client, True, ["reason"], state
            )
            assert skip_cycle is True
            mock_save_state.assert_called_once()
            mock_logging.debug.assert_called_with(
                "Failed to write BLOCKED state to HA.", exc_info=True
            )

    def test_handle_grace_period_shadow_mode(self, blocking_manager, mock_ha_client):
        """Test that grace period is skipped in shadow mode."""
        with patch("src.heating_controller.config.SHADOW_MODE", True):
            state = SystemState()
            skip_cycle = blocking_manager.handle_grace_period(mock_ha_client, state)
            assert skip_cycle is False

    def test_handle_grace_period_not_blocking_before(
        self, blocking_manager, mock_ha_client
    ):
        """Test grace period is skipped if not blocking before."""
        state = SystemState(last_is_blocking=False)
        skip_cycle = blocking_manager.handle_grace_period(mock_ha_client, state)
        assert skip_cycle is False

    @patch("src.heating_controller.time.time")
    def test_handle_grace_period_expired(
        self, mock_time, blocking_manager, mock_ha_client
    ):
        """Test grace period is skipped if it has expired."""
        mock_time.return_value = 10000
        state = SystemState(last_is_blocking=True, last_blocking_end_time=100)
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))

        with patch("src.heating_controller.config.GRACE_PERIOD_MAX_MINUTES", 5):
            skip_cycle = blocking_manager.handle_grace_period(mock_ha_client, state)
        assert skip_cycle is False

    @patch(
        "src.heating_controller.BlockingStateManager._execute_grace_period",
        return_value=False,
    )
    @patch("src.heating_controller.time.time")
    def test_handle_grace_period_starts(
        self, mock_time, mock_execute, blocking_manager, mock_ha_client
    ):
        """Test that a grace period is started."""
        mock_time.return_value = 100
        state = SystemState(last_is_blocking=True)
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))

        with patch("src.heating_controller.save_state") as mock_save:
            skip = blocking_manager.handle_grace_period(mock_ha_client, state)
            assert skip is True
            mock_save.assert_any_call(last_blocking_end_time=100)
            mock_execute.assert_called_once()

    @patch("src.heating_controller.time.time", return_value=100)
    def test_stagnation_abort_resumes_normal_cycle(
        self, mock_time, blocking_manager, mock_ha_client
    ):
        state = SystemState(last_is_blocking=True, last_blocking_end_time=90)
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        blocking_manager._execute_grace_period = Mock(return_value=True)

        with patch("src.heating_controller.save_state") as mock_save:
            is_grace_period = blocking_manager.handle_grace_period(
                mock_ha_client, state, shadow_mode=False
            )

        assert is_grace_period is False
        mock_save.assert_called_once_with(
            last_is_blocking=False, last_blocking_end_time=None
        )

    @patch("src.heating_controller.time.time", return_value=100)
    @patch("src.heating_controller.get_sensor_attributes", return_value={})
    @patch("src.heating_controller.BlockingStateManager._wait_for_grace_target")
    def test_grace_period_fallback(self, mock_wait, mock_attrs, mock_time, blocking_manager, mock_ha_client):
        """Test grace period fallback logic when sensors are missing."""
        state = SystemState(last_is_blocking=True, last_blocking_end_time=90, last_final_temp=42.0)
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        mock_ha_client.get_state.side_effect = [None, None, None, 40.0]

        with patch("src.heating_controller.save_state"):
            blocking_manager.handle_grace_period(mock_ha_client, state)
            mock_ha_client.set_state.assert_called_with(
                config.TARGET_OUTLET_TEMP_ENTITY_ID,
                42.0,
                {},
                round_digits=0,
            )
            mock_wait.assert_called_once()

    @patch("src.heating_controller.time.time", return_value=100)
    @patch("src.model_wrapper.get_enhanced_model_wrapper")
    @patch("src.physics_features.build_physics_features")
    @patch("src.influx_service.create_influx_service")
    def test_grace_period_intelligent_recovery_no_wait(
        self, mock_influx, mock_build, mock_wrapper_getter, mock_time, blocking_manager, mock_ha_client
    ):
        """Test intelligent recovery when outlet temp is already close to target."""
        state = SystemState(last_is_blocking=True, last_blocking_end_time=90)
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        mock_ha_client.get_state.side_effect = [21.0, 22.0, 5.0, 44.8] + [44.8] * 20

        mock_wrapper = Mock()
        mock_wrapper._calculate_required_outlet_temp.return_value = 45.0
        mock_wrapper_getter.return_value = mock_wrapper
        
        mock_df = Mock()
        mock_df.empty = False
        # Mock iloc[0].to_dict()
        mock_row = Mock()
        mock_row.to_dict.return_value = {"some": "features"}
        # Configure iloc to return mock_row when accessed with index 0
        mock_df.iloc = [mock_row]
        
        mock_build.return_value = (mock_df, {})
    
        with patch("src.heating_controller.save_state"):
            blocking_manager.handle_grace_period(mock_ha_client, state)
            mock_ha_client.set_state.assert_not_called()

    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time")
    def test_wait_for_grace_target_timeout(
        self, mock_time, mock_sleep, blocking_manager, mock_ha_client
    ):
        """Test that the grace period wait times out."""
        mock_time.side_effect = [1000, 1001, 1002, 1003, 1000 + 301] + [1000 + 301] * 20
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        mock_ha_client.get_state.return_value = 50.0

        with patch("src.heating_controller.config.GRACE_PERIOD_MAX_MINUTES", 5):
            blocking_manager._wait_for_grace_target(mock_ha_client, 40.0, True)
        assert mock_sleep.called

    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time", return_value=1000)
    def test_wait_for_grace_target_reaches_target(
        self, mock_time, mock_sleep, blocking_manager, mock_ha_client
    ):
        """Test wait ends when target is reached."""
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        mock_ha_client.get_state.side_effect = [50.0, 40.0] + [40.0] * 20

        blocking_manager._wait_for_grace_target(mock_ha_client, 40.0, True)
        mock_sleep.assert_called_once()

    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time", return_value=1000)
    def test_wait_for_grace_target_blocking_reappears(
        self, mock_time, mock_sleep, blocking_manager, mock_ha_client
    ):
        """Test wait ends when blocking reappears."""
        blocking_manager.check_blocking_state = Mock(side_effect=[(False, []), (True, ["reason"])])
        mock_ha_client.get_state.return_value = 50.0

        with patch("src.heating_controller.save_state") as mock_save:
            blocking_manager._wait_for_grace_target(mock_ha_client, 40.0, True)
            mock_sleep.assert_called_once()
            mock_save.assert_called_once()


    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time")
    def test_wait_for_grace_target_stagnation_aborts(
        self, mock_time, mock_sleep, blocking_manager, mock_ha_client
    ):
        """Outlet stuck above target ends the wait before the timeout."""
        mock_time.side_effect = [1000, 1000] + [1000 + 60 * i for i in range(1, 30)]
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        mock_ha_client.get_state.return_value = 50.0

        with patch("src.heating_controller.config.GRACE_PERIOD_MAX_MINUTES", 15):
            stagnated = blocking_manager._wait_for_grace_target(
                mock_ha_client, 40.0, True
            )
        # The first sample at the start of the wait gives a full four-minute window.
        assert mock_sleep.call_count == 4
        assert stagnated is True

    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time")
    def test_wait_for_grace_target_warmup_stagnation_aborts(
        self, mock_time, mock_sleep, blocking_manager, mock_ha_client
    ):
        """A flat outlet during warm-up ends the wait after the stagnation window."""
        mock_time.side_effect = [1000, 1000] + [1000 + 60 * i for i in range(1, 30)]
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        mock_ha_client.get_state.return_value = 50.0

        with patch("src.heating_controller.config.GRACE_PERIOD_MAX_MINUTES", 15):
            stagnated = blocking_manager._wait_for_grace_target(
                mock_ha_client, 60.0, False
            )

        assert mock_sleep.call_count == 4
        assert stagnated is True

    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time")
    def test_wait_for_grace_target_rising_outlet_is_not_stagnation(
        self, mock_time, mock_sleep, blocking_manager, mock_ha_client
    ):
        """A rising outlet after defrost continues warming instead of aborting."""
        mock_time.side_effect = [1000 + 60 * i for i in range(30)]
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        temps = [50.0 + 0.2 * i for i in range(60)]
        mock_ha_client.get_state.side_effect = lambda *a, **k: temps.pop(0)

        with patch("src.heating_controller.config.GRACE_PERIOD_MAX_MINUTES", 6):
            stagnated = blocking_manager._wait_for_grace_target(
                mock_ha_client, 60.0, False
            )

        assert mock_sleep.call_count >= 5
        assert stagnated is False

    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time")
    def test_wait_for_grace_target_warmup_stagnation_aborts(
        self, mock_time, mock_sleep, blocking_manager, mock_ha_client
    ):
        """A flat outlet during warm-up ends the wait after the stagnation window."""
        mock_time.side_effect = [1000, 1000] + [1000 + 60 * i for i in range(1, 30)]
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        mock_ha_client.get_state.return_value = 50.0

        with patch("src.heating_controller.config.GRACE_PERIOD_MAX_MINUTES", 15):
            blocking_manager._wait_for_grace_target(mock_ha_client, 60.0, False)

        assert mock_sleep.call_count == 4

    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time")
    def test_wait_for_grace_target_rising_outlet_is_not_stagnation(
        self, mock_time, mock_sleep, blocking_manager, mock_ha_client
    ):
        """A rising outlet after defrost continues warming instead of aborting."""
        mock_time.side_effect = [1000 + 60 * i for i in range(30)]
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        temps = [50.0 + 0.2 * i for i in range(60)]
        mock_ha_client.get_state.side_effect = lambda *a, **k: temps.pop(0)

        with patch("src.heating_controller.config.GRACE_PERIOD_MAX_MINUTES", 6):
            blocking_manager._wait_for_grace_target(mock_ha_client, 60.0, False)

        assert mock_sleep.call_count >= 5

    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time")
    def test_wait_for_grace_target_slow_cooling_continues(
        self, mock_time, mock_sleep, blocking_manager, mock_ha_client
    ):
        """Steady cooling is not treated as stagnation."""
        mock_time.side_effect = [1000 + 60 * i for i in range(30)]
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        temps = [50.0 - 0.2 * i for i in range(60)]
        mock_ha_client.get_state.side_effect = lambda *a, **k: temps.pop(0)

        with patch("src.heating_controller.config.GRACE_PERIOD_MAX_MINUTES", 6):
            blocking_manager._wait_for_grace_target(mock_ha_client, 40.0, True)
        # Not aborted at 4 min; only the 6 min timeout ends it
        assert mock_sleep.call_count >= 5

    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time")
    def test_wait_for_grace_target_long_poll_interval_uses_older_reference(
        self, mock_time, mock_sleep, blocking_manager, mock_ha_client
    ):
        """A 300-second poll must compare against a sample before the cutoff."""
        mock_time.side_effect = [1000, 1000, 1300, 1301, 1361] + [1361] * 10
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        outlet_temps = iter([50.0, 49.0, 49.0, 49.0])

        def get_state(entity_id, *_):
            if entity_id == config.ACTUAL_OUTLET_TEMP_ENTITY_ID:
                return next(outlet_temps)
            if entity_id == config.INDOOR_TEMP_ENTITY_ID:
                return 22.0
            if entity_id == config.TARGET_INDOOR_TEMP_ENTITY_ID:
                return 21.0
            return None

        mock_ha_client.get_state.side_effect = get_state
        with (
            patch("src.heating_controller.config.GRACE_PERIOD_MAX_MINUTES", 6),
            patch("src.heating_controller.config.BLOCKING_POLL_INTERVAL_SECONDS", 300),
        ):
            blocking_manager._wait_for_grace_target(mock_ha_client, 40.0, True)

        # A 1 K drop is real cooling; only the 6-minute timeout ends the wait.
        assert mock_sleep.call_count == 3

    @patch("src.heating_controller.time.sleep")
    @patch("src.heating_controller.time.time")
    def test_wait_for_grace_target_timing_drift_uses_sample_before_cutoff(
        self, mock_time, mock_sleep, blocking_manager, mock_ha_client
    ):
        """A sample just after the cutoff must not shorten the measured window."""
        mock_time.side_effect = [1000, 1000, 1100, 1260, 1301] + [1301] * 10
        blocking_manager.check_blocking_state = Mock(return_value=(False, []))
        outlet_temps = iter([50.0, 49.8, 49.6, 49.6])

        def get_state(entity_id, *_):
            if entity_id == config.ACTUAL_OUTLET_TEMP_ENTITY_ID:
                return next(outlet_temps)
            if entity_id == config.INDOOR_TEMP_ENTITY_ID:
                return 22.0
            if entity_id == config.TARGET_INDOOR_TEMP_ENTITY_ID:
                return 21.0
            return None

        mock_ha_client.get_state.side_effect = get_state
        with patch("src.heating_controller.config.GRACE_PERIOD_MAX_MINUTES", 5):
            blocking_manager._wait_for_grace_target(mock_ha_client, 40.0, True)

        # The full-window drop is 0.4 K, so the 0.3 K threshold is not met.
        assert mock_sleep.call_count == 3


class TestSensorDataManager:
    @pytest.fixture
    def sensor_manager(self):
        return SensorDataManager()

    def test_get_critical_sensors_all_present(
            self, sensor_manager, mock_ha_client):
        mock_ha_client.get_state.return_value = 1.0
        sensor_data, missing = sensor_manager.get_critical_sensors(
            mock_ha_client, {})
        assert sensor_data is not None
        assert len(missing) == 0

    def test_get_critical_sensors_some_missing(
            self, sensor_manager, mock_ha_client):
        mock_ha_client.get_state.side_effect = [
            1.0, None, 1.0, 1.0, True, 1.0, 1.0]
        sensor_data, missing = sensor_manager.get_critical_sensors(
            mock_ha_client, {})
        assert sensor_data is None
        assert len(missing) > 0

    def test_handle_missing_sensors_ha_error(self, sensor_manager, mock_ha_client):
        """Test handle_missing_sensors handles Home Assistant errors."""
        mock_ha_client.set_state.side_effect = Exception("HA API Error")
        with patch("src.heating_controller.logging") as mock_logging:
            skip_cycle = sensor_manager.handle_missing_sensors(
                mock_ha_client, ["sensor.missing"]
            )
            assert skip_cycle is True
            mock_logging.debug.assert_called_with(
                "Failed to write NO_DATA state to HA.", exc_info=True
            )


class TestHeatingSystemStateChecker:
    @pytest.fixture
    def state_checker(self):
        return HeatingSystemStateChecker()

    def test_check_heating_active(self, state_checker, mock_ha_client):
        mock_ha_client.get_state.return_value = "heat"
        assert state_checker.check_heating_active(mock_ha_client, {}) is True

    def test_check_heating_inactive(self, state_checker, mock_ha_client):
        mock_ha_client.get_state.return_value = "off"
        assert state_checker.check_heating_active(mock_ha_client, {}) is False

    def test_check_heating_inactive_ha_error(self, state_checker, mock_ha_client):
        """Test check_heating_active handles HA errors when inactive."""
        mock_ha_client.get_state.return_value = "off"
        mock_ha_client.set_state.side_effect = Exception("HA API Error")
        with patch("src.heating_controller.logging") as mock_logging:
            should_continue = state_checker.check_heating_active(mock_ha_client, {})
            assert should_continue is False
            mock_logging.debug.assert_called_with(
                "Failed to write SYSTEM_OFF state to HA.", exc_info=True
            )
