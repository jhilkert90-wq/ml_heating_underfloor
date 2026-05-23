"""
Cycle state enumeration and determination logic.

This module defines the possible states of the main control loop and provides
the logic to determine which state applies for a given cycle based on system
conditions (blocking, heating active, climate mode).
"""
from enum import Enum


class CycleState(Enum):
    """The four possible states of the main control cycle."""

    HEATING = "heating"
    COOLING = "cooling"
    BLOCKING = "blocking"
    IDLE = "idle"


def determine_cycle_state(
    *,
    is_blocking: bool,
    heating_active: bool,
    climate_mode: str | None,
) -> CycleState:
    """Determine the cycle state from system conditions.

    Parameters
    ----------
    is_blocking:
        True when a blocking process (DHW, defrost, disinfection) is active.
    heating_active:
        True when the heating/cooling system is switched on.
    climate_mode:
        "heating" or "cooling" as detected from the heat pump.

    Returns
    -------
    CycleState
        The definitive state for this cycle.
    """
    if is_blocking:
        return CycleState.BLOCKING

    if not heating_active:
        return CycleState.IDLE

    if climate_mode == "cooling":
        return CycleState.COOLING

    # Default: if heating_active and not cooling, treat as heating
    return CycleState.HEATING
