"""
Dynamic trajectory-step scaling based on PV production and time of day.

When ``PV_TRAJ_SCALING_ENABLED`` is ``true`` the system replaces the static
``TRAJECTORY_STEPS`` value with a per-cycle estimate derived from:

1. **PV ratio** — actual PV power relative to the effective system capacity.
   Clamped 0-1.
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

**Seasonal KWP Scaling (optional)**

When ``PV_TRAJ_SEASONAL_SCALING_ENABLED`` is ``true``, the effective PV peak
used to compute *pv_ratio* is scaled by a seasonal factor derived from the
solar declination at the configured latitude.  This normalises PV production
relative to the summer-solstice maximum so that a clear winter day (full
output for the season) correctly maps to pv_ratio=1.0.

The factor is computed as::

    δ(doy)        = 23.45° × sin(360/365 × (doy − 81))   # solar declination
    elev_max(doy) = 90° − |lat − δ(doy)|                 # noon elevation
    factor        = sin(elev_max_today) / sin(elev_max_june21)

Clamped to ``[PV_TRAJ_SEASONAL_MIN_FACTOR, 1.0]``.

Requires only Python stdlib ``math`` — no external astronomy library needed.

When the feature is disabled ``compute_dynamic_trajectory_steps`` still
returns the current ``config.TRAJECTORY_STEPS`` value unchanged.
"""
import logging
import math
from datetime import date, datetime

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

# Day-of-year for June 21 (summer solstice reference)
_SUMMER_SOLSTICE_DOY = 172


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


def _solar_declination_deg(doy: int) -> float:
    """Return approximate solar declination in degrees for *doy* (1-365)."""
    return 23.45 * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))


def _max_solar_elevation_deg(doy: int, latitude_deg: float) -> float:
    """Return the theoretical maximum solar elevation angle (degrees) at noon.

    Uses the simplified formula::

        elev_max = 90 - |latitude - declination|

    Clamped to [0, 90] to handle polar cases.
    """
    declination = _solar_declination_deg(doy)
    elev = 90.0 - abs(latitude_deg - declination)
    return max(0.0, min(90.0, elev))


def seasonal_kwp_factor(
    current_date: date,
    latitude_deg: float,
    min_factor: float = 0.1,
) -> float:
    """Compute the seasonal scaling factor for PV peak capacity.

    Compares the theoretical maximum solar elevation on *current_date* with
    the summer-solstice maximum (June 21).  The ratio of their sines gives the
    relative clear-sky PV production capacity.

    Args:
        current_date: Date for which to compute the factor.
        latitude_deg: Geographic latitude in decimal degrees (North positive).
        min_factor: Floor value to prevent near-zero results in deep winter.
            Clamped to [0.0, 1.0] internally.

    Returns:
        Float in ``[min_factor, 1.0]`` representing the seasonal scaling.
        Returns 1.0 if the summer-solstice elevation is ≤ 0 (degenerate).
    """
    min_factor = max(0.0, min(1.0, min_factor))
    doy = current_date.timetuple().tm_yday
    elev_today = _max_solar_elevation_deg(doy, latitude_deg)
    elev_summer = _max_solar_elevation_deg(_SUMMER_SOLSTICE_DOY, latitude_deg)

    if elev_summer <= 0.0:
        return 1.0

    # Use sine of elevation angle — proportional to clear-sky irradiance.
    sin_today = math.sin(math.radians(elev_today))
    sin_summer = math.sin(math.radians(elev_summer))

    if sin_summer <= 0.0:
        return 1.0

    factor = sin_today / sin_summer
    return max(min_factor, min(1.0, factor))


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

    # Apply seasonal KWP scaling if enabled
    effective_system_kwp = system_kwp
    _seasonal_factor = 1.0
    if getattr(config, "PV_TRAJ_SEASONAL_SCALING_ENABLED", False):
        latitude = float(getattr(config, "PV_TRAJ_LATITUDE", 51.0))
        min_factor = float(getattr(config, "PV_TRAJ_SEASONAL_MIN_FACTOR", 0.1))
        _seasonal_factor = seasonal_kwp_factor(now.date(), latitude, min_factor)
        effective_system_kwp = system_kwp * _seasonal_factor

    peak_w = effective_system_kwp * 1000.0
    pv_ratio = max(0.0, min(1.0, pv_power_w / peak_w)) if peak_w > 0 else 0.0

    tod_factor = _time_of_day_factor(now.hour)

    raw = min_steps + round(pv_ratio * tod_factor * (max_steps - min_steps))
    steps = int(max(min_steps, min(max_steps, raw)))

    if getattr(config, "PV_TRAJ_SEASONAL_SCALING_ENABLED", False):
        logger.info(
            "☀️ Dynamic trajectory: PV=%.0fW, seasonal_factor=%.2f "
            "(doy=%d, lat=%.1f°), ratio=%.2f, tod_factor=%.2f "
            "(hour=%d) → %d steps",
            pv_power_w,
            _seasonal_factor,
            now.timetuple().tm_yday,
            float(getattr(config, "PV_TRAJ_LATITUDE", 51.0)),
            pv_ratio,
            tod_factor,
            now.hour,
            steps,
        )
    else:
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
