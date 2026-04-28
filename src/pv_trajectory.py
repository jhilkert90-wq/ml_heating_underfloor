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

    * Current PV check: ``pv_power_w >= PV_TRAJ_THRESHOLD_W``  (enough current
      PV) **or** ``PV_TRAJ_FORECAST_RESCUE_ENABLED=true`` and at least
      ``PV_TRAJ_MIN_STEPS`` forecast hours exceed ``PV_TRAJ_THRESHOLD_W``
      (handles passing rain clouds / short-duration dips below threshold).
    * At least one entry within the first ``PV_TRAJ_MAX_STEPS`` forecast slots
      is at or below ``PV_TRAJ_ZERO_W``  (sunset is within the planning horizon)

    In night mode (``pv_power_w < PV_TRAJ_ZERO_W``) the function returns
    ``PV_TRAJ_MIN_STEPS`` immediately.

    The price offset can be suppressed automatically via
    ``PV_TRAJ_DISABLE_PRICE_IN_FORECAST_MODE`` (default ``true``).

When ``PV_TRAJ_FORECAST_MODE_ENABLED`` is ``false``,
``compute_dynamic_trajectory_steps`` returns the current
``config.TRAJECTORY_STEPS`` value unchanged.
"""
import logging

try:
    from . import config
except ImportError:
    import config  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forecast-driven mode helpers
# ---------------------------------------------------------------------------

def is_forecast_trajectory_active(
    pv_power_w: float,
    pv_forecast: list[float] | None,
) -> bool:
    """Return ``True`` when the forecast-driven trajectory mode is activated.

    Activation requires **all** of:

    1. ``PV_TRAJ_FORECAST_MODE_ENABLED=true``
    2. Current PV or forecast rescue check (see below).
    3. At least one forecast slot within the first ``PV_TRAJ_MAX_STEPS`` entries
       is at or below ``PV_TRAJ_ZERO_W``  (sunset is within the planning horizon)

    **Current PV check (condition 2):**
    If ``pv_power_w >= PV_TRAJ_THRESHOLD_W`` the check passes unconditionally.
    If ``pv_power_w < PV_TRAJ_THRESHOLD_W`` (e.g. passing rain cloud) the check
    still passes when ``PV_TRAJ_FORECAST_RESCUE_ENABLED=true`` and at least
    ``PV_TRAJ_MIN_STEPS`` forecast hours exceed ``PV_TRAJ_THRESHOLD_W``.

    Night mode (``pv_power_w < PV_TRAJ_ZERO_W``) is *not* considered activated
    because step count is already at the minimum.

    Args:
        pv_power_w: Current PV electrical power [W].
        pv_forecast: Hourly PV forecast [W], index 0 = next hour.  ``None``
            or an empty list is treated as an empty horizon (→ not activated).

    Returns:
        ``True`` if forecast mode should drive the trajectory, else ``False``.
    """
    if not getattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False):
        return False

    zero_w = float(getattr(config, "PV_TRAJ_ZERO_W", 50.0))
    threshold_w = float(getattr(config, "PV_TRAJ_THRESHOLD_W", 3000.0))
    max_steps = int(getattr(config, "PV_TRAJ_MAX_STEPS", 12))
    min_steps = int(getattr(config, "PV_TRAJ_MIN_STEPS", 2))

    # Night: current PV at or below zero threshold → not activated
    if pv_power_w < zero_w:
        return False

    horizon = (pv_forecast or [])[:max_steps]

    if pv_power_w < threshold_w:
        # Optionally rescue via forecast: if at least min_steps forecast hours
        # exceed the threshold the mode remains active (passing rain cloud).
        if not getattr(config, "PV_TRAJ_FORECAST_RESCUE_ENABLED", True):
            return False
        if sum(1 for v in horizon if v > threshold_w) < min_steps:
            return False

    return any(v <= zero_w for v in horizon)


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
    * ``pv_power_w < PV_TRAJ_THRESHOLD_W`` (insufficient current PV):

      - If ``PV_TRAJ_FORECAST_RESCUE_ENABLED=true`` (default) **and** at least
        ``PV_TRAJ_MIN_STEPS`` forecast hours exceed ``PV_TRAJ_THRESHOLD_W``,
        normal step counting continues (passing rain cloud is ignored).
      - Otherwise → ``PV_TRAJ_MIN_STEPS``

    * No sunset found within ``PV_TRAJ_MAX_STEPS`` horizon → ``PV_TRAJ_MIN_STEPS``

    Args:
        pv_power_w: Current PV electrical power [W].
        pv_forecast: Hourly PV forecast [W], index 0 = next hour.

    Returns:
        Integer step count in ``[PV_TRAJ_MIN_STEPS, PV_TRAJ_MAX_STEPS]``.
    """
    min_steps = int(getattr(config, "PV_TRAJ_MIN_STEPS", 2))
    max_steps = int(getattr(config, "PV_TRAJ_MAX_STEPS", 12))
    if min_steps < 1:
        min_steps = 1
    if max_steps < min_steps:
        max_steps = min_steps

    zero_w = float(getattr(config, "PV_TRAJ_ZERO_W", 50.0))
    threshold_w = float(getattr(config, "PV_TRAJ_THRESHOLD_W", 3000.0))

    # Night: current PV at or below zero threshold
    if pv_power_w < zero_w:
        logger.info(
            "☀️ Forecast trajectory: PV=%.0fW < zero_w=%.0fW → "
            "night mode, %d steps",
            pv_power_w, zero_w, min_steps,
        )
        return min_steps

    # Insufficient current PV — optionally rescue via forecast
    if pv_power_w < threshold_w:
        rescue_enabled = getattr(config, "PV_TRAJ_FORECAST_RESCUE_ENABLED", True)
        if rescue_enabled:
            horizon_rescue = (pv_forecast or [])[:max_steps]
            rescue_hours = sum(1 for v in horizon_rescue if v > threshold_w)
            if rescue_hours >= min_steps:
                logger.info(
                    "☀️ Forecast trajectory: PV=%.0fW < threshold=%.0fW but "
                    "%d forecast hours above threshold → rescued, continuing",
                    pv_power_w, threshold_w, rescue_hours,
                )
            else:
                logger.info(
                    "☀️ Forecast trajectory: PV=%.0fW < threshold=%.0fW, "
                    "only %d forecast hours above threshold (need %d) → "
                    "inactive, %d steps",
                    pv_power_w, threshold_w, rescue_hours, min_steps, min_steps,
                )
                return min_steps
        else:
            logger.info(
                "☀️ Forecast trajectory: PV=%.0fW < threshold=%.0fW → "
                "inactive (rescue disabled), %d steps",
                pv_power_w, threshold_w, min_steps,
            )
            return min_steps

    # Activation check: sunset must appear within the planning horizon
    horizon = (pv_forecast or [])[:max_steps]
    if not any(v <= zero_w for v in horizon):
        logger.info(
            "☀️ Forecast trajectory: no sunset within %d-step horizon → "
            "inactive, %d steps",
            max_steps, min_steps,
        )
        return min_steps

    # Count consecutive hours from the start of the forecast with PV > zero_w
    remaining_pv_hours = 0
    for v in horizon:
        if v > zero_w:
            remaining_pv_hours += 1
        else:
            break  # first night slot reached

    steps = int(max(min_steps, min(max_steps, remaining_pv_hours + min_steps)))
    logger.info(
        "☀️ Forecast trajectory: PV=%.0fW, remaining_pv_hours=%d + min_steps=%d → %d steps",
        pv_power_w, remaining_pv_hours, min_steps, steps,
    )
    return steps


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
    if not getattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False):
        return int(getattr(config, "TRAJECTORY_STEPS", 4))

    return compute_forecast_driven_trajectory_steps(pv_power_w, pv_forecast)

