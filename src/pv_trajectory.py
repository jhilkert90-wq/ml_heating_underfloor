"""
Dynamic trajectory-step scaling based on PV production and time of day.

Two modes are available, selected at runtime by configuration:

**Classic mode** (``PV_TRAJ_FORECAST_MODE_ENABLED=false``, default)
    Step count is derived from:

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
    solar declination at the configured latitude.

**Forecast-driven mode** (``PV_TRAJ_FORECAST_MODE_ENABLED=true``)
    Step count equals the number of consecutive forecast hours (starting from
    the next hour) with PV production above ``PV_TRAJ_ZERO_W``.  This maps
    directly to "how many hours of sun remain today", giving a long planning
    horizon in the morning and a naturally shrinking horizon toward sunset.

    Activation requires **both**:

    * ``pv_power_w >= PV_TRAJ_THRESHOLD_W``  (enough current PV)
    * At least one entry within the first ``PV_TRAJ_MAX_STEPS`` forecast slots
      is at or below ``PV_TRAJ_ZERO_W``  (sunset is within the planning horizon)

    In night mode (``pv_power_w < PV_TRAJ_ZERO_W``) the function returns
    ``PV_TRAJ_MIN_STEPS`` immediately.

    This mode ignores ``PV_TRAJ_SYSTEM_KWP``, time-of-day factors, and seasonal
    scaling.  The price offset can be suppressed automatically via
    ``PV_TRAJ_DISABLE_PRICE_IN_FORECAST_MODE`` (default ``true``).

When ``PV_TRAJ_SCALING_ENABLED`` is ``false``,
``compute_dynamic_trajectory_steps`` still returns the current
``config.TRAJECTORY_STEPS`` value unchanged regardless of mode.
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


def _peak_solar_doy(latitude_deg: float) -> int:
    """Return DOY of the solstice that maximises solar elevation for *latitude_deg*.

    June 21 (DOY 172) for the northern hemisphere (lat >= 0);
    December 21 (DOY 355) for the southern hemisphere (lat < 0).
    """
    return 172 if latitude_deg >= 0.0 else 355


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
    the peak-solstice maximum (June 21 for northern latitudes, December 21 for
    southern latitudes).  The ratio of their sines gives the relative
    clear-sky PV production capacity.

    Args:
        current_date: Date for which to compute the factor.
        latitude_deg: Geographic latitude in decimal degrees (North positive,
            South negative).
        min_factor: Floor value to prevent near-zero results in deep winter.
            Clamped to [0.0, 1.0] internally.

    Returns:
        Float in ``[min_factor, 1.0]`` representing the seasonal scaling.
        Returns 1.0 if the peak-solstice elevation is ≤ 0 (degenerate).
    """
    min_factor = max(0.0, min(1.0, min_factor))
    doy = current_date.timetuple().tm_yday
    elev_today = _max_solar_elevation_deg(doy, latitude_deg)
    elev_summer = _max_solar_elevation_deg(_peak_solar_doy(latitude_deg), latitude_deg)

    if elev_summer <= 0.0:
        return 1.0

    # Use sine of elevation angle — proportional to clear-sky irradiance.
    sin_today = math.sin(math.radians(elev_today))
    sin_summer = math.sin(math.radians(elev_summer))

    if sin_summer <= 0.0:
        return 1.0

    factor = sin_today / sin_summer
    return max(min_factor, min(1.0, factor))


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
    2. ``pv_power_w >= PV_TRAJ_THRESHOLD_W``  (enough current PV production)
    3. At least one forecast slot within the first ``PV_TRAJ_MAX_STEPS`` entries
       is at or below ``PV_TRAJ_ZERO_W``  (sunset is within the planning horizon)

    Night mode (``pv_power_w < PV_TRAJ_ZERO_W``) is *not* considered activated
    because step count is already at the minimum.

    Args:
        pv_power_w: Current PV electrical power [W].
        pv_forecast: Hourly PV forecast [W], index 0 = next hour.  ``None``
            or an empty list is treated as all-zero (→ not activated).

    Returns:
        ``True`` if forecast mode should drive the trajectory, else ``False``.
    """
    if not getattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False):
        return False

    zero_w = float(getattr(config, "PV_TRAJ_ZERO_W", 50.0))
    threshold_w = float(getattr(config, "PV_TRAJ_THRESHOLD_W", 3000.0))
    max_steps = int(getattr(config, "PV_TRAJ_MAX_STEPS", 12))

    if pv_power_w < threshold_w:
        return False

    horizon = (pv_forecast or [])[:max_steps]
    return any(v <= zero_w for v in horizon)


def compute_forecast_driven_trajectory_steps(
    pv_power_w: float,
    pv_forecast: list[float] | None,
) -> int:
    """Compute trajectory steps using the forecast-driven algorithm.

    Step count = number of consecutive forecast hours (from hour 1 onward)
    where PV > ``PV_TRAJ_ZERO_W``, clamped to
    ``[PV_TRAJ_MIN_STEPS, PV_TRAJ_MAX_STEPS]``.

    Special cases:

    * ``pv_power_w < PV_TRAJ_ZERO_W`` (night) → ``PV_TRAJ_MIN_STEPS``
    * ``pv_power_w < PV_TRAJ_THRESHOLD_W`` (insufficient PV) → ``PV_TRAJ_MIN_STEPS``
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

    # Insufficient current PV — mode not activated
    if pv_power_w < threshold_w:
        logger.info(
            "☀️ Forecast trajectory: PV=%.0fW < threshold=%.0fW → "
            "inactive, %d steps",
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

    steps = int(max(min_steps, min(max_steps, remaining_pv_hours)))
    logger.info(
        "☀️ Forecast trajectory: PV=%.0fW, remaining_pv_hours=%d → %d steps",
        pv_power_w, remaining_pv_hours, steps,
    )
    return steps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_dynamic_trajectory_steps(
    pv_power_w: float,
    system_kwp: float | None = None,
    now: datetime | None = None,
    pv_forecast: list[float] | None = None,
) -> int:
    """Compute the trajectory step count for the current cycle.

    Delegates to :func:`compute_forecast_driven_trajectory_steps` when
    ``PV_TRAJ_FORECAST_MODE_ENABLED`` is ``true``; otherwise uses the classic
    ``pv_ratio × tod_factor`` interpolation.

    Args:
        pv_power_w: Current PV power in Watts.
        system_kwp: Nominal system capacity in kWp.  Defaults to
            ``config.PV_TRAJ_SYSTEM_KWP``.  Unused in forecast mode.
        now: Local datetime for time-of-day factor.  Defaults to
            ``datetime.now()``.  Unused in forecast mode.
        pv_forecast: Hourly PV forecast [W], index 0 = next hour.  Used only
            in forecast mode; ignored in classic mode.

    Returns:
        Integer step count in ``[PV_TRAJ_MIN_STEPS, PV_TRAJ_MAX_STEPS]``.
        When ``PV_TRAJ_SCALING_ENABLED`` is ``False`` returns
        ``config.TRAJECTORY_STEPS`` unchanged.
    """
    if not getattr(config, "PV_TRAJ_SCALING_ENABLED", False):
        return int(getattr(config, "TRAJECTORY_STEPS", 4))

    # --- Forecast-driven mode ---
    if getattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False):
        return compute_forecast_driven_trajectory_steps(pv_power_w, pv_forecast)

    # --- Classic pv_ratio × tod_factor mode ---
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

