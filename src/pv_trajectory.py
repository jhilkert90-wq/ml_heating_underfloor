"""
Dynamic trajectory-step scaling based on PV production and time of day.

When ``PV_TRAJ_SCALING_ENABLED`` is ``true`` the system replaces the static
``TRAJECTORY_STEPS`` value with a per-cycle estimate derived from:

1. **PV ratio** — actual PV power relative to the nominal system capacity
   (``PV_TRAJ_SYSTEM_KWP``).  Clamped 0-1.
2. **Time-of-day factor** — four configurable windows:

   ==============================  ==========================================
   Window                          Default factor
   ==============================  ==========================================
   Morning   06:00–10:59           0.5  (sun still rising)
   Midday    11:00–14:59           1.0  (peak production)
   Afternoon 15:00–18:59           0.75 (declining)
   Night     19:00–23:59 and       0.0  (no PV → minimum steps)
             00:00–05:59
   ==============================  ==========================================

3. **Linear interpolation** between ``PV_TRAJ_MIN_STEPS`` and
   ``PV_TRAJ_MAX_STEPS``::

       steps = MIN_STEPS + round(pv_ratio * tod_factor * (MAX_STEPS - MIN_STEPS))

The result is clamped to ``[PV_TRAJ_MIN_STEPS, PV_TRAJ_MAX_STEPS]``.

When the feature is disabled ``compute_dynamic_trajectory_steps`` still
returns the current ``config.TRAJECTORY_STEPS`` value unchanged.
"""
import logging
from datetime import datetime

try:
    from . import config
except ImportError:
    import config  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Time-of-day window boundaries (hours, local time, inclusive lower bound)
# ---------------------------------------------------------------------------
_MORNING_START = 6
_MIDDAY_START = 11
_AFTERNOON_START = 15
_NIGHT_START = 19  # 19:00 – 05:59 is night


def _time_of_day_factor(hour: int) -> float:
    """Return the configured multiplier for *hour* (0–23, local time)."""
    if _MORNING_START <= hour < _MIDDAY_START:
        return float(getattr(config, "PV_TRAJ_MORNING_FACTOR", 0.5))
    elif _MIDDAY_START <= hour < _AFTERNOON_START:
        return float(getattr(config, "PV_TRAJ_MIDDAY_FACTOR", 1.0))
    elif _AFTERNOON_START <= hour < _NIGHT_START:
        return float(getattr(config, "PV_TRAJ_AFTERNOON_FACTOR", 0.75))
    else:
        return float(getattr(config, "PV_TRAJ_NIGHT_FACTOR", 0.0))


def compute_dynamic_trajectory_steps(
    pv_power_w: float,
    system_kwp: float | None = None,
    now: datetime | None = None,
) -> int:
    """Compute the trajectory step count for the current cycle.

    Args:
        pv_power_w: Current PV power in Watts.
        system_kwp: Nominal system capacity in kWp.  Defaults to
            ``config.PV_TRAJ_SYSTEM_KWP``.
        now: Local datetime for time-of-day factor.  Defaults to
            ``datetime.now()``.

    Returns:
        Integer step count in ``[PV_TRAJ_MIN_STEPS, PV_TRAJ_MAX_STEPS]``.
        When ``PV_TRAJ_SCALING_ENABLED`` is ``False`` returns
        ``config.TRAJECTORY_STEPS`` unchanged.
    """
    if not getattr(config, "PV_TRAJ_SCALING_ENABLED", False):
        return int(getattr(config, "TRAJECTORY_STEPS", 4))

    if system_kwp is None:
        system_kwp = float(getattr(config, "PV_TRAJ_SYSTEM_KWP", 10.0))
    if now is None:
        now = datetime.now()

    min_steps = int(getattr(config, "PV_TRAJ_MIN_STEPS", 2))
    max_steps = int(getattr(config, "PV_TRAJ_MAX_STEPS", 12))
    # Guard against misconfiguration
    if min_steps < 1:
        min_steps = 1
    if max_steps < min_steps:
        max_steps = min_steps

    peak_w = system_kwp * 1000.0
    pv_ratio = max(0.0, min(1.0, pv_power_w / peak_w)) if peak_w > 0 else 0.0

    tod_factor = _time_of_day_factor(now.hour)

    raw = min_steps + round(pv_ratio * tod_factor * (max_steps - min_steps))
    steps = int(max(min_steps, min(max_steps, raw)))

    logger.info(
        "☀️ Dynamic trajectory: PV=%.0fW, ratio=%.2f, tod_factor=%.2f "
        "(hour=%d) → %d steps",
        pv_power_w,
        pv_ratio,
        tod_factor,
        now.hour,
        steps,
    )
    return steps
