"""Tests for src.pre_dispatch — pre-dispatch step functions."""
from unittest.mock import MagicMock, call, patch

import pytest

from src.cycle_state import CycleState, determine_cycle_state
from src.loop_state import LoopState
from src.pre_dispatch import (
    check_and_resolve_climate_mode,
    check_blocking_state,
    emit_network_error_state,
    handle_grace_period,
    initialize_loop_state,
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
        is_gp = handle_grace_period(MagicMock(), state, False)
        assert is_gp is False

    @patch("src.pre_dispatch.BlockingStateManager")
    def test_grace_when_manager_says_yes(self, mock_bsm_cls):
        """Grace period active when manager detects transition."""
        mock_bsm = MagicMock()
        mock_bsm.handle_grace_period.return_value = True
        mock_bsm_cls.return_value = mock_bsm

        state = {"last_is_blocking": True, "last_final_temp": 28.0}
        is_gp = handle_grace_period(MagicMock(), state, True)
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

    def test_returns_false_on_empty_states(self):
        """Returns False (not done) when no states available."""
        result = validate_sensors_once(None, MagicMock())
        assert result is False


class TestInitializeLoopStateClimateMode:
    """initialize_loop_state sets correct climate mode before export_metrics_to_ha."""

    @patch("src.pre_dispatch.HeatingSystemStateChecker")
    @patch("src.model_wrapper.get_enhanced_model_wrapper")
    def test_cooling_mode_set_before_export(
        self, mock_get_wrapper, mock_checker_cls
    ):
        """When HA reports cooling, wrapper.set_climate_mode('cooling') is called
        before wrapper.export_metrics_to_ha()."""
        mock_wrapper = MagicMock()
        mock_wrapper._climate_mode = "heating"
        mock_get_wrapper.return_value = mock_wrapper

        # Capture _climate_mode at the time export is called
        captured_mode = {}

        def _track_mode_on_export():
            captured_mode["mode"] = mock_wrapper._climate_mode

        mock_wrapper.export_metrics_to_ha.side_effect = _track_mode_on_export

        def _set_mode(mode):
            mock_wrapper._climate_mode = mode

        mock_wrapper.set_climate_mode.side_effect = _set_mode

        mock_checker = MagicMock()
        mock_checker.get_climate_mode.return_value = "cooling"
        mock_checker_cls.return_value = mock_checker

        with patch("src.ha_client.create_ha_client") as mock_create_ha:
            mock_ha = MagicMock()
            mock_ha.get_all_states.return_value = {"some": "state"}
            mock_create_ha.return_value = mock_ha

            initialize_loop_state(sensor_buffer=MagicMock(), influx_service=MagicMock())

        # set_climate_mode("cooling") must be called before export_metrics_to_ha
        mock_wrapper.set_climate_mode.assert_called_with("cooling")
        mock_wrapper.export_metrics_to_ha.assert_called_once()
        assert captured_mode.get("mode") == "cooling"

    @patch("src.pre_dispatch.HeatingSystemStateChecker")
    @patch("src.model_wrapper.get_enhanced_model_wrapper")
    def test_falls_back_to_heating_on_ha_error(
        self, mock_get_wrapper, mock_checker_cls
    ):
        """When HA client raises during startup mode detection, wrapper is NOT
        put into an unknown state — set_climate_mode is not called and the
        default 'heating' mode is preserved."""
        mock_wrapper = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper

        with patch("src.ha_client.create_ha_client") as mock_create_ha:
            mock_create_ha.side_effect = OSError("HA unavailable")

            initialize_loop_state(sensor_buffer=MagicMock(), influx_service=MagicMock())

        # Mode detection failed → set_climate_mode should NOT have been called
        mock_wrapper.set_climate_mode.assert_not_called()
        # But export should still proceed
        mock_wrapper.export_metrics_to_ha.assert_called_once()

