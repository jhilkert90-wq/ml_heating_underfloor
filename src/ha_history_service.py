"""
Home Assistant History Service for calibration data.

Fetches historical sensor data from the HA REST API ``/api/history/period``
endpoint and converts it into a regular 5-minute DataFrame that is
schema-compatible with the InfluxDB ``get_training_data()`` output.

This provides an alternative data source for ``physics_calibration.py``
that does not require an InfluxDB deployment — only a Home Assistant
instance with sufficient recorder retention (4-8 weeks recommended).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from . import config
    from .ha_client import HAClient
except ImportError:
    import config  # type: ignore
    from ha_client import HAClient  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity mapping: config attribute name → short column name used by
# physics_calibration (matches InfluxDB ``get_training_data()`` schema).
# ---------------------------------------------------------------------------
def _build_entity_map() -> Dict[str, str]:
    """
    Build a mapping of full HA entity_id → short column name.

    The short name equals ``entity_id.split('.', 1)[-1]`` which is the
    convention used by ``influx_service.get_training_data()``.
    """
    entity_ids = [
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID,
        config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID,
        config.INDOOR_TEMP_ENTITY_ID,
        config.TV_STATUS_ENTITY_ID,
        config.DHW_STATUS_ENTITY_ID,
        config.DEFROST_STATUS_ENTITY_ID,
        config.DISINFECTION_STATUS_ENTITY_ID,
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
        config.OUTDOOR_TEMP_ENTITY_ID,
        config.PV_POWER_ENTITY_ID,
        config.INLET_TEMP_ENTITY_ID,
        config.FLOW_RATE_ENTITY_ID,
        config.POWER_CONSUMPTION_ENTITY_ID,
        config.FIREPLACE_STATUS_ENTITY_ID,
    ]

    # Optional: living room temp for FP spread calibration
    living_room = getattr(config, "LIVING_ROOM_TEMP_ENTITY_ID", None)
    if living_room:
        entity_ids.append(living_room)

    return {eid: eid.split(".", 1)[-1] for eid in entity_ids}


# ---------------------------------------------------------------------------
# Binary entity detection
# ---------------------------------------------------------------------------
_BINARY_PREFIXES = ("binary_sensor.", "input_boolean.")


def _is_binary_entity(entity_id: str) -> bool:
    return any(entity_id.startswith(p) for p in _BINARY_PREFIXES)


# ---------------------------------------------------------------------------
# State value conversion
# ---------------------------------------------------------------------------
def _parse_state_value(state_str: str, is_binary: bool) -> float:
    """Convert a raw HA state string to a numeric value."""
    if state_str in ("unavailable", "unknown", ""):
        return float("nan")
    if is_binary:
        return 1.0 if state_str in ("on", "True", "true", "1") else 0.0
    try:
        return float(state_str)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Cloud cover proxy from PV clear-sky ratio
# ---------------------------------------------------------------------------
def compute_cloud_proxy(
    pv_series: pd.Series,
    timestamps: pd.DatetimeIndex,
    peak_pv_watts: float = 0.0,
) -> pd.Series:
    """
    Estimate cloud cover percentage from PV power vs clear-sky theoretical.

    Uses the seasonal maximum PV in the DataFrame as clear-sky reference
    when ``peak_pv_watts`` is not supplied.  At night (solar elevation ≤ 0)
    or when no PV reference is available a default of 50% is returned.

    Returns a Series of cloud cover values in [0, 100].
    """
    if peak_pv_watts <= 0:
        peak_pv_watts = pv_series.max()
    if peak_pv_watts <= 0:
        return pd.Series(50.0, index=timestamps, dtype=float)

    # Simple daylight gate: PV > 1% of peak means daytime
    is_day = pv_series > (peak_pv_watts * 0.01)

    ratio = pv_series / peak_pv_watts
    cloud = (1.0 - ratio.clip(0.0, 1.0)) * 100.0

    # At night return 50% (neutral default)
    cloud[~is_day] = 50.0
    return cloud


# ---------------------------------------------------------------------------
# Main conversion helper
# ---------------------------------------------------------------------------
def _ha_history_to_dataframe(
    raw_histories: List[List[Dict[str, Any]]],
    entity_map: Dict[str, str],
    entity_ids: List[str],
    freq: str = "5min",
) -> pd.DataFrame:
    """
    Convert raw HA history JSON into a regular-interval DataFrame.

    Parameters
    ----------
    raw_histories : list of lists
        One inner list per requested entity, each element is a dict with
        at least ``state`` and ``last_changed`` keys.
    entity_map : dict
        Full entity_id → short column name.
    entity_ids : list
        The entity IDs that were requested (same order as raw_histories).
    freq : str
        Target resampling frequency (default ``"5min"``).

    Returns
    -------
    pd.DataFrame with ``_time`` column and one column per entity (short
    name).  Missing values are forward- then backward-filled.
    """
    if not raw_histories:
        return pd.DataFrame()

    # Determine overall time range from the data
    all_times: List[datetime] = []
    per_entity_series: Dict[str, pd.Series] = {}

    for idx, history in enumerate(raw_histories):
        if idx >= len(entity_ids):
            break
        eid = entity_ids[idx]
        col_name = entity_map.get(eid, eid.split(".", 1)[-1])
        is_binary = _is_binary_entity(eid)

        times = []
        values = []
        for record in history:
            ts_str = record.get("last_changed") or record.get("last_updated")
            state = record.get("state", "")
            if ts_str is None:
                continue
            try:
                ts = pd.Timestamp(ts_str).tz_convert("UTC")
            except Exception:
                try:
                    ts = pd.Timestamp(ts_str, tz="UTC")
                except Exception:
                    continue
            val = _parse_state_value(state, is_binary)
            times.append(ts)
            values.append(val)

        if times:
            s = pd.Series(values, index=pd.DatetimeIndex(times, name="_time"))
            # Drop exact duplicates keeping last
            s = s[~s.index.duplicated(keep="last")]
            s = s.sort_index()
            per_entity_series[col_name] = s
            all_times.extend(times)

    if not all_times:
        return pd.DataFrame()

    # Build regular 5-min index spanning the data
    t_min = min(all_times).floor(freq)
    t_max = max(all_times).ceil(freq)
    regular_index = pd.date_range(t_min, t_max, freq=freq, tz="UTC")

    # Reindex each series onto regular grid with forward-fill
    aligned: Dict[str, pd.Series] = {}
    for col, s in per_entity_series.items():
        reindexed = s.reindex(regular_index, method="ffill")
        aligned[col] = reindexed

    df = pd.DataFrame(aligned, index=regular_index)
    df.index.name = "_time"
    df.reset_index(inplace=True)

    # Final fill pass
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_training_data_from_ha(
    lookback_hours: int,
    ha_client: Optional[HAClient] = None,
) -> pd.DataFrame:
    """
    Fetch historical calibration data from the HA REST API.

    Returns a DataFrame with the same schema as
    ``InfluxService.get_training_data()`` so that
    ``physics_calibration.py`` can use either source transparently.

    Parameters
    ----------
    lookback_hours : int
        How many hours of history to request.
    ha_client : HAClient, optional
        An existing client instance.  If *None*, one is created via
        ``create_ha_client()``.

    Returns
    -------
    pd.DataFrame  (empty on failure)
    """
    if ha_client is None:
        from .ha_client import create_ha_client
        ha_client = create_ha_client()

    entity_map = _build_entity_map()
    entity_ids = list(entity_map.keys())

    end_time = datetime.now(tz=timezone.utc)
    start_time = end_time - timedelta(hours=lookback_hours)

    logger.info(
        "Fetching %d hours of history from HA for %d entities …",
        lookback_hours,
        len(entity_ids),
    )

    raw = ha_client.get_history_bulk(entity_ids, start_time, end_time)
    if raw is None:
        logger.error("HA history request returned None")
        return pd.DataFrame()

    logger.info("Received history arrays for %d entities", len(raw))

    df = _ha_history_to_dataframe(raw, entity_map, entity_ids)
    if df.empty:
        logger.error("HA history produced empty DataFrame")
        return df

    # Add cloud cover proxy column from PV data
    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    if pv_col in df.columns and "_time" in df.columns:
        df["cloud_cover_proxy"] = compute_cloud_proxy(
            df[pv_col], df["_time"]
        ).values

    logger.info(
        "HA history DataFrame: %d rows × %d cols (%d hours)",
        len(df),
        len(df.columns),
        lookback_hours,
    )
    return df
