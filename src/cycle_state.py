"""
Cycle state enumeration and determination logic.

This module defines the possible states of the main control loop and provides
the logic to determine which state applies for a given cycle based on system
conditions (blocking, heating active, climate mode).
"""
from enum import Enum


class CycleState(Enum):
    """The five possible states of the main control cycle."""

    HEATING = "heating"
    COOLING = "cooling"
    BLOCKING = "blocking"
    IDLE = "idle"
    GRACE_PERIOD = "grace_period"


def determine_cycle_state(
    *,
    is_blocking: bool,
    heating_active: bool,
    climate_mode: str | None,
    is_grace_period: bool = False,
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
    is_grace_period:
        True when in the grace period immediately after a blocking event ends.

    Returns
    -------
    CycleState
        The definitive state for this cycle.
    """
    if is_blocking:
        return CycleState.BLOCKING

    if is_grace_period:
        return CycleState.GRACE_PERIOD

    if not heating_active:
        return CycleState.IDLE

    if climate_mode == "cooling":
        return CycleState.COOLING

    # Default: if heating_active and not cooling, treat as heating
    return CycleState.HEATING
