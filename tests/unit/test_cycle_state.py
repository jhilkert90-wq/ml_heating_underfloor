"""Tests for the cycle state determination logic."""
import pytest

from src.cycle_state import CycleState, determine_cycle_state


class TestCycleState:
    """Test CycleState enum values."""

    def test_enum_values(self):
        assert CycleState.HEATING.value == "heating"
        assert CycleState.COOLING.value == "cooling"
        assert CycleState.BLOCKING.value == "blocking"
        assert CycleState.IDLE.value == "idle"

    def test_enum_members(self):
        assert len(CycleState) == 4


class TestDetermineCycleState:
    """Test determine_cycle_state function covers all state transitions."""

    def test_blocking_overrides_everything(self):
        """Blocking state takes priority over heating_active and climate_mode."""
        result = determine_cycle_state(
            is_blocking=True,
            heating_active=True,
            climate_mode="heating",
        )
        assert result == CycleState.BLOCKING

    def test_blocking_overrides_cooling(self):
        result = determine_cycle_state(
            is_blocking=True,
            heating_active=True,
            climate_mode="cooling",
        )
        assert result == CycleState.BLOCKING

    def test_blocking_when_system_inactive(self):
        """Blocking even when heating system is off."""
        result = determine_cycle_state(
            is_blocking=True,
            heating_active=False,
            climate_mode="heating",
        )
        assert result == CycleState.BLOCKING

    def test_idle_when_not_active(self):
        """IDLE when heating system is not active and not blocking."""
        result = determine_cycle_state(
            is_blocking=False,
            heating_active=False,
            climate_mode="heating",
        )
        assert result == CycleState.IDLE

    def test_idle_when_not_active_cooling_mode(self):
        """IDLE regardless of climate_mode when system inactive."""
        result = determine_cycle_state(
            is_blocking=False,
            heating_active=False,
            climate_mode="cooling",
        )
        assert result == CycleState.IDLE

    def test_heating_state(self):
        """HEATING when active, not blocking, mode is heating."""
        result = determine_cycle_state(
            is_blocking=False,
            heating_active=True,
            climate_mode="heating",
        )
        assert result == CycleState.HEATING

    def test_cooling_state(self):
        """COOLING when active, not blocking, mode is cooling."""
        result = determine_cycle_state(
            is_blocking=False,
            heating_active=True,
            climate_mode="cooling",
        )
        assert result == CycleState.COOLING

    def test_heating_is_default_for_unknown_mode(self):
        """HEATING as fallback for unknown/None climate_mode."""
        result = determine_cycle_state(
            is_blocking=False,
            heating_active=True,
            climate_mode=None,
        )
        assert result == CycleState.HEATING

    def test_heating_for_empty_string_mode(self):
        result = determine_cycle_state(
            is_blocking=False,
            heating_active=True,
            climate_mode="",
        )
        assert result == CycleState.HEATING

    def test_priority_order_blocking_first(self):
        """Verify priority: blocking > idle > cooling > heating."""
        # All conditions true except blocking
        assert determine_cycle_state(
            is_blocking=False, heating_active=True, climate_mode="cooling"
        ) == CycleState.COOLING

        # Add blocking
        assert determine_cycle_state(
            is_blocking=True, heating_active=True, climate_mode="cooling"
        ) == CycleState.BLOCKING

    def test_priority_idle_over_mode(self):
        """When system is off, mode doesn't matter."""
        assert determine_cycle_state(
            is_blocking=False, heating_active=False, climate_mode="heating"
        ) == CycleState.IDLE
        assert determine_cycle_state(
            is_blocking=False, heating_active=False, climate_mode="cooling"
        ) == CycleState.IDLE
