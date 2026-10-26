"""
Dynamic trajectory-step scaling based on PV forecast.

**Forecast-driven mode** (``PV_TRAJ_FORECAST_MODE_ENABLED=true``)
    Step count equals the number of consecutive forecast hours (starting from
    the next hour) with PV production above ``PV_TRAJ_ZERO_W`` **plus**
    ``PV_TRAJ_MIN_STEPS`` reserved slots for the post-sunset period, clamped
    to ``[PV_TRAJ_MIN_STEPS, PV_TRAJ_MAX_STEPS]``::

        steps = min(MAX_STEPS, remaining_pv_hours + MIN_STEPS)

    This maps directly to "how many hours of sun remain today plus a night
    buffer", giving a full planning horizon in the morning and a naturally
    shrinking horizon toward sunset.

    Activation requires **both**:

    * 1h forecast check: ``pv_forecast[0] >= PV_TRAJ_THRESHOLD_W`` (the PV
      forecast for the next hour is high enough). Current PV is not used for
      activation, only for the night stop below.
    * At least one entry within the first ``PV_TRAJ_MAX_STEPS`` forecast slots
      is at or below ``PV_TRAJ_ZERO_W``  (sunset is within the planning horizon)

    ``PV_TRAJ_FORECAST_RESCUE_ENABLED`` / ``PV_TRAJ_RESCUE_MIN_HOURS`` are
    deprecated and no longer have any effect.

    In night mode (``pv_power_w < PV_TRAJ_ZERO_W``) the function returns
    ``PV_TRAJ_MIN_STEPS`` immediately.

    The price offset can be suppressed automatically via
    ``PV_TRAJ_DISABLE_PRICE_IN_FORECAST_MODE`` (default ``true``).
"""
from dataclasses import dataclass
import logging

try:
    from . import config
except ImportError:
    import config  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forecast-driven mode helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastTrajectoryState:
    """Resolved forecast-trajectory state for the current cycle."""

    mode_enabled: bool
    active: bool
    forecast_1h_above_threshold: bool
    sunset_within_horizon: bool
    dynamic_steps: int
    min_steps: int
    max_steps: int

    @property
    def dynamic_steps_above_min(self) -> bool:
        return self.dynamic_steps > self.min_steps


def get_forecast_trajectory_state(
    pv_power_w: float,
    pv_forecast: list[float] | None,
    respect_mode_flag: bool = True,
) -> ForecastTrajectoryState:
    """Return the resolved forecast-driven trajectory state."""
    static_steps = int(getattr(config, "TRAJECTORY_STEPS", 4))
    min_steps = max(1, int(getattr(config, "PV_TRAJ_MIN_STEPS", 2)))
    max_steps = int(getattr(config, "PV_TRAJ_MAX_STEPS", 12))
    if max_steps < min_steps:
        max_steps = min_steps

    if respect_mode_flag and not getattr(
        config,
        "PV_TRAJ_FORECAST_MODE_ENABLED",
        False,
    ):
        return ForecastTrajectoryState(
            mode_enabled=False,
            active=False,
            forecast_1h_above_threshold=False,
            sunset_within_horizon=False,
            dynamic_steps=static_steps,
            min_steps=min_steps,
            max_steps=max_steps,
        )

    zero_w = float(getattr(config, "PV_TRAJ_ZERO_W", 50.0))
    threshold_w = float(getattr(config, "PV_TRAJ_THRESHOLD_W", 3000.0))
    horizon = (pv_forecast or [])[:max_steps]

    if pv_power_w < zero_w:
        return ForecastTrajectoryState(
            mode_enabled=True,
            active=False,
            forecast_1h_above_threshold=False,
            sunset_within_horizon=any(v <= zero_w for v in horizon),
            dynamic_steps=min_steps,
            min_steps=min_steps,
            max_steps=max_steps,
        )

    forecast_1h_above_threshold = bool(horizon) and horizon[0] >= threshold_w
    sunset_within_horizon = any(v <= zero_w for v in horizon)
    active = sunset_within_horizon and forecast_1h_above_threshold

    if active:
        remaining_pv_hours = 0
        for value in horizon:
            if value > zero_w:
                remaining_pv_hours += 1
            else:
                break
        dynamic_steps = int(
            max(min_steps, min(max_steps, remaining_pv_hours + min_steps))
        )
    else:
        dynamic_steps = min_steps

    return ForecastTrajectoryState(
        mode_enabled=True,
        active=active,
        forecast_1h_above_threshold=forecast_1h_above_threshold,
        sunset_within_horizon=sunset_within_horizon,
        dynamic_steps=dynamic_steps,
        min_steps=min_steps,
        max_steps=max_steps,
    )


def is_forecast_trajectory_active(
    pv_power_w: float,
    pv_forecast: list[float] | None,
) -> bool:
    """Return ``True`` when the forecast-driven trajectory mode is activated.

    Activation requires **all** of:

    1. ``PV_TRAJ_FORECAST_MODE_ENABLED=true``
    2. ``pv_forecast[0] >= PV_TRAJ_THRESHOLD_W`` (1h electrical PV forecast)
    3. At least one forecast slot within the first ``PV_TRAJ_MAX_STEPS`` entries
       is at or below ``PV_TRAJ_ZERO_W``  (sunset is within the planning horizon)

    Night mode (``pv_power_w < PV_TRAJ_ZERO_W``) is *not* considered activated
    because step count is already at the minimum.

    Args:
        pv_power_w: Current PV electrical power [W].
        pv_forecast: Hourly PV forecast [W], index 0 = next hour.  ``None``
            or an empty list is treated as an empty horizon (→ not activated).

    Returns:
        ``True`` if forecast mode should drive the trajectory, else ``False``.
    """
    return get_forecast_trajectory_state(pv_power_w, pv_forecast).active


def compute_forecast_driven_trajectory_steps(
    pv_power_w: float,
    pv_forecast: list[float] | None,
) -> int:
    """Compute trajectory steps using the forecast-driven algorithm.

    Step count = consecutive forecast hours starting at the first forecast
    slot (index 0 / next hour) where PV > ``PV_TRAJ_ZERO_W`` **plus**
    ``PV_TRAJ_MIN_STEPS`` reserved slots for the post-sunset period,
    clamped to ``[PV_TRAJ_MIN_STEPS, PV_TRAJ_MAX_STEPS]``::

        steps = clamp(remaining_pv_hours + MIN_STEPS, MIN_STEPS, MAX_STEPS)

    Special cases:

    * ``pv_power_w < PV_TRAJ_ZERO_W`` (night) → ``PV_TRAJ_MIN_STEPS``
    * ``pv_forecast[0] < PV_TRAJ_THRESHOLD_W`` (1h forecast too low)
      → ``PV_TRAJ_MIN_STEPS``
    * No sunset found within ``PV_TRAJ_MAX_STEPS`` horizon → ``PV_TRAJ_MIN_STEPS``

    Args:
        pv_power_w: Current PV electrical power [W].
        pv_forecast: Hourly PV forecast [W], index 0 = next hour.

    Returns:
        Integer step count in ``[PV_TRAJ_MIN_STEPS, PV_TRAJ_MAX_STEPS]``.
    """
    state = get_forecast_trajectory_state(
        pv_power_w,
        pv_forecast,
        respect_mode_flag=False,
    )
    min_steps = state.min_steps
    max_steps = state.max_steps
    zero_w = float(getattr(config, "PV_TRAJ_ZERO_W", 50.0))
    threshold_w = float(getattr(config, "PV_TRAJ_THRESHOLD_W", 3000.0))
    horizon = (pv_forecast or [])[:max_steps]

    # Night: current PV at or below zero threshold
    if pv_power_w < zero_w:
        logger.info(
            "☀️ Forecast trajectory: PV=%.0fW < zero_w=%.0fW → "
            "night mode, %d steps",
            pv_power_w, zero_w, min_steps,
        )
        return state.dynamic_steps

    # 1h forecast below activation threshold
    if not state.forecast_1h_above_threshold:
        logger.info(
            "☀️ Forecast trajectory: 1h forecast=%.0fW < threshold=%.0fW → "
            "inactive, %d steps",
            float(horizon[0]) if horizon else 0.0, threshold_w, min_steps,
        )
        return state.dynamic_steps

    # Activation check: sunset must appear within the planning horizon
    if not state.sunset_within_horizon:
        logger.info(
            "☀️ Forecast trajectory: no sunset within %d-step horizon → "
            "inactive, %d steps",
            max_steps, min_steps,
        )
        return state.dynamic_steps

    remaining_pv_hours = max(0, state.dynamic_steps - min_steps)
    logger.info(
        "☀️ Forecast trajectory: PV=%.0fW, remaining_pv_hours=%d + min_steps=%d → %d steps",
        pv_power_w, remaining_pv_hours, min_steps, state.dynamic_steps,
    )
    return state.dynamic_steps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_dynamic_trajectory_steps(
    pv_power_w: float,
    pv_forecast: list[float] | None = None,
) -> int:
    """Compute the trajectory step count for the current cycle.

    Delegates to :func:`compute_forecast_driven_trajectory_steps` when
    ``PV_TRAJ_FORECAST_MODE_ENABLED`` is ``true``; otherwise returns
    ``config.TRAJECTORY_STEPS`` unchanged.

    Args:
        pv_power_w: Current PV power in Watts.
        pv_forecast: Hourly PV forecast [W], index 0 = next hour.

    Returns:
        Integer step count in ``[PV_TRAJ_MIN_STEPS, PV_TRAJ_MAX_STEPS]``
        when forecast mode is enabled; ``config.TRAJECTORY_STEPS`` otherwise.
    """
    return get_forecast_trajectory_state(
        pv_power_w,
        pv_forecast,
    ).dynamic_steps
