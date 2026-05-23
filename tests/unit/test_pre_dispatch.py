"""Tests for src.pre_dispatch — pre-dispatch step functions."""
from unittest.mock import MagicMock, patch

import pytest

from src.cycle_state import CycleState, determine_cycle_state
from src.loop_state import LoopState
from src.pre_dispatch import (
    check_and_resolve_climate_mode,
    check_blocking_state,
    emit_network_error_state,
    handle_grace_period,
    resolve_shadow_mode_for_cycle,
    update_sensor_buffer_and_thermo,
    validate_sensors_once,
)


class TestUpdateSensorBufferAndThermo:
    """update_sensor_buffer_and_thermo returns False when no states."""

    def test_returns_false_on_empty_states(self):
        ls = LoopState()
        result = update_sensor_buffer_and_thermo(ls, MagicMock(), None, MagicMock())
        assert result is False

    def test_returns_false_on_empty_dict(self):
        ls = LoopState()
        result = update_sensor_buffer_and_thermo(ls, MagicMock(), {}, MagicMock())
        assert result is False


class TestResolveShadowMode:
    """resolve_shadow_mode_for_cycle wraps shadow_mode.resolve_shadow_mode."""

    @patch("src.pre_dispatch.resolve_shadow_mode")
    def test_returns_resolved_mode(self, mock_resolve):
        mock_mode = MagicMock()
        mock_mode.effective_shadow_mode = True
        mock_resolve.return_value = mock_mode
        ha_client = MagicMock()
        ha_client.get_state.return_value = True
        all_states = {"some": "state"}
        shadow_mode, effective = resolve_shadow_mode_for_cycle(ha_client, all_states)
        assert shadow_mode is mock_mode
        assert effective is True


class TestEmitNetworkErrorState:
    """emit_network_error_state does not raise."""

    @patch("src.pre_dispatch.get_sensor_attributes", return_value={})
    def test_no_raise(self, mock_attrs):
        ha_client = MagicMock()
        # Should not raise even if HA call fails
        emit_network_error_state(ha_client)


class TestCheckBlockingState:
    """check_blocking_state returns blocking state tuple."""

    @patch("src.pre_dispatch.BlockingStateManager")
    def test_delegates_to_manager(self, mock_bsm_cls):
        mock_bsm = MagicMock()
        mock_bsm.check_blocking_state.return_value = (True, ["dhw"])
        mock_bsm_cls.return_value = mock_bsm

        ha_client = MagicMock()
        all_states = {}

        result = check_blocking_state(ha_client, all_states)
        assert result == (True, ["dhw"])


class TestCheckAndResolveClimateMode:
    """check_and_resolve_climate_mode determines heating_active and mode."""

    @patch("src.pre_dispatch.HeatingSystemStateChecker")
    @patch("src.pre_dispatch.load_state")
    def test_heating_active(self, mock_load, mock_checker_cls):
        mock_checker = MagicMock()
        mock_checker.check_heating_active.return_value = True
        mock_checker.get_climate_mode.return_value = "heating"
        mock_checker_cls.return_value = mock_checker
        mock_load.return_value = {"last_final_temp": 30.0}

        ha_client = MagicMock()
        all_states = {}
        wrapper = MagicMock()
        wrapper.state_manager = MagicMock()

        result = check_and_resolve_climate_mode(ha_client, all_states, wrapper)
        heating_active, climate_mode, state_mgr, state = result
        assert heating_active is True
        assert climate_mode == "heating"

    @patch("src.pre_dispatch.HeatingSystemStateChecker")
    @patch("src.pre_dispatch.load_state")
    def test_idle_uses_heating_state(self, mock_load, mock_checker_cls):
        """When system is idle, climate_mode forced to 'heating'."""
        mock_checker = MagicMock()
        mock_checker.check_heating_active.return_value = False
        mock_checker_cls.return_value = mock_checker
        mock_load.return_value = {}

        ha_client = MagicMock()
        all_states = {}
        wrapper = MagicMock()
        wrapper.state_manager = MagicMock()

        result = check_and_resolve_climate_mode(ha_client, all_states, wrapper)
        heating_active, climate_mode, state_mgr, state = result
        assert heating_active is False
        # Idle forces heating state manager
        assert climate_mode == "heating"
        wrapper.set_climate_mode.assert_called_once_with("heating")


class TestHandleGracePeriod:
    """handle_grace_period returns correct flags."""

    @patch("src.pre_dispatch.BlockingStateManager")
    def test_not_grace_when_manager_says_no(self, mock_bsm_cls):
        """No grace period if BlockingStateManager says no."""
        mock_bsm = MagicMock()
        mock_bsm.handle_grace_period.return_value = False
        mock_bsm_cls.return_value = mock_bsm

        state = {"last_is_blocking": False}
        is_gp = handle_grace_period(MagicMock(), state, MagicMock(), False)
        assert is_gp is False

    @patch("src.pre_dispatch.save_state")
    @patch("src.pre_dispatch.BlockingStateManager")
    def test_grace_when_manager_says_yes(self, mock_bsm_cls, mock_save):
        """Grace period active when manager detects transition."""
        mock_bsm = MagicMock()
        mock_bsm.handle_grace_period.return_value = True
        mock_bsm_cls.return_value = mock_bsm

        state = {"last_is_blocking": True, "last_final_temp": 28.0}
        is_gp = handle_grace_period(MagicMock(), state, MagicMock(), True)
        assert is_gp is True


class TestGracePeriodDetermineState:
    """Integration: determine_cycle_state with is_grace_period flag."""

    def test_grace_period_state(self):
        result = determine_cycle_state(
            is_blocking=False,
            heating_active=True,
            climate_mode="heating",
            is_grace_period=True,
        )
        assert result == CycleState.GRACE_PERIOD

    def test_normal_heating(self):
        result = determine_cycle_state(
            is_blocking=False,
            heating_active=True,
            climate_mode="heating",
            is_grace_period=False,
        )
        assert result == CycleState.HEATING


class TestValidateSensorsOnce:
    """validate_sensors_once only runs once per process."""

    def test_returns_true_on_empty_states(self):
        """Returns False (not done) when no states available."""
        result = validate_sensors_once(None, MagicMock())
        assert result is False
