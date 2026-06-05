"""
cooling_correction_ml_calibration.py
--------------------------------------
One-shot training of the LightGBM cooling-correction regressor.

Called via
  ``python -m src.main --calibrate-cooling-correction-ml``
or triggered by the flag file ``/data/config/calibrate_cooling_correction_ml_flag``.

Pipeline
--------
1.  Fetch multi-month historical data (same helper as physics calibration).
2.  Rename entity-ID columns to model-friendly names.
3.  Filter to warm-season rows (AT > COOLING_ML_CORRECTION_WARM_THRESHOLD_C).
4.  Compute derived features (mirrors heating calibration with cooling-specific additions).
5.  Compute residualized regression label:
        adjusted_label[t] = -(T_indoor[t + N_steps] - T_indoor[t]) / S_H_cool
    where S_H_cool uses the cooling outlet-effectiveness (OE_cooling ≈ 0.20).
6.  Forward-looking outlier filtering (fireplace, window-open, PV spike, extreme label).
7.  Train LightGBM regressor (objective = "regression_l1" / MAE).
8.  Save model (joblib) + metadata JSON.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level config import so tests can patch.
try:
    from . import config
except ImportError:
    try:
        import config  # type: ignore
    except ImportError:
        config = None  # type: ignore

# Module-level import of shared training-data export helper.
try:
    from .calibration_data_export import export_training_data
except ImportError:
    try:
        from calibration_data_export import export_training_data  # type: ignore
    except ImportError:
        def export_training_data(*args, **kwargs):  # type: ignore
            return None

# Module-level import of data-fetching helper.
try:
    from src.physics_calibration import fetch_historical_data_for_calibration
except ImportError:
    try:
        from physics_calibration import fetch_historical_data_for_calibration  # type: ignore
    except ImportError:
        def fetch_historical_data_for_calibration(*args, **kwargs):  # type: ignore
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHADING_ACTIVATION_TEMP_C = 23.0


def _json_default(obj):
    """JSON serialiser fallback for numpy scalars."""
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def _compute_s_h(eta: float, u_loss: float, tau_room: float, h: float) -> float:
    """Physics sensitivity S(H) = [η/(η+U)] × [1 − exp(−H/τ)]."""
    denom = eta + u_loss
    if denom < 1e-6 or tau_room < 1e-3:
        return 0.0
    return (eta / denom) * (1.0 - math.exp(-h / tau_room))


def _read_cooling_thermal_params(config) -> tuple[float, float, float]:
    """Read OE_cooling, HLC, τ for the cooling S_H computation.

    Returns (outlet_effectiveness_cooling, heat_loss_coefficient, thermal_time_constant).

    Cooling uses its own OE (typically ~0.20), but HLC and τ are locked
    from the heating calibration (shared building physics).
    """
    try:
        from src.unified_thermal_state import get_thermal_state_manager
        state_manager = get_thermal_state_manager()

        oe_default = float(getattr(config, "COOLING_OUTLET_EFFECTIVENESS", 0.20))
        hlc_default = float(getattr(config, "HEAT_LOSS_COEFFICIENT", 0.124))
        tau_default = float(getattr(config, "THERMAL_TIME_CONSTANT", 4.39))

        def _to_float_or(value, fallback: float) -> float:
            try:
                if value is None:
                    return fallback
                return float(value)
            except (TypeError, ValueError):
                return fallback

        # Try cooling-specific parameters from unified state
        get_computed = getattr(state_manager, "get_computed_parameters", None)
        computed = get_computed() if callable(get_computed) else {}

        # Cooling OE from cooling channel or config default
        state = getattr(state_manager, "state", {}) or {}
        cooling_params = state.get("cooling_parameters", {}) or {}
        oe_cooling = _to_float_or(
            cooling_params.get("outlet_effectiveness"),
            _to_float_or(computed.get("cooling_outlet_effectiveness"), oe_default),
        )

        # HLC and τ from heating calibration (shared physics)
        hlc = _to_float_or(computed.get("heat_loss_coefficient"), hlc_default)
        tau = _to_float_or(computed.get("thermal_time_constant"), tau_default)

        logger.info(
            "Cooling S_H params: OE=%.4f HLC=%.4f τ=%.2fh",
            oe_cooling, hlc, tau,
        )
        return oe_cooling, hlc, tau

    except Exception:
        logger.warning(
            "Could not read unified thermal state; using config defaults for cooling S_H"
        )
        oe = float(getattr(config, "COOLING_OUTLET_EFFECTIVENESS", 0.20))
        hlc = float(getattr(config, "HEAT_LOSS_COEFFICIENT", 0.124))
        tau = float(getattr(config, "THERMAL_TIME_CONSTANT", 4.39))
        return oe, hlc, tau


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def calibrate_cooling_correction_ml(
    lookback_hours: int = 2160,  # 90 days
    cooling_target_c: Optional[float] = None,
) -> bool:
    """
    Train and persist the LightGBM cooling correction regressor.

    Parameters
    ----------
    lookback_hours:
        How many hours of historical data to fetch (default 2160 = 90 days).
    cooling_target_c:
        Cooling target indoor temperature [°C].  When *None* the value is read
        from ``config.TARGET_INDOOR_TEMP_COOLING`` or falls back to 23.0.

    Returns
    -------
    True on success, False on failure.
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        logger.error("numpy/pandas not installed — cannot calibrate")
        return False

    # ── 1. Read config ──────────────────────────────────────────────────
    if cooling_target_c is None:
        cooling_target_c = float(
            getattr(config, "TARGET_INDOOR_TEMP_COOLING",
                    getattr(config, "TARGET_INDOOR_TEMP", 23.0))
        )

    label_horizon_h = int(
        getattr(config, "COOLING_ML_CORRECTION_LABEL_HORIZON_H", 4)
    )
    steps_per_hour = int(getattr(config, "STEPS_PER_HOUR", 6))
    label_horizon_steps = label_horizon_h * steps_per_hour

    warm_threshold = float(
        getattr(config, "COOLING_ML_CORRECTION_WARM_THRESHOLD_C", 18.0)
    )

    at_forecast_str = str(
        getattr(config, "COOLING_ML_CORRECTION_AT_FORECAST_HOURS", "1,2,3,4")
    )
    at_forecast_hours = [int(h.strip()) for h in at_forecast_str.split(",") if h.strip()]

    pv_forecast_str = str(
        getattr(config, "COOLING_ML_CORRECTION_PV_FORECAST_HOURS", "1,2,3,4")
    )
    pv_forecast_hours = [int(h.strip()) for h in pv_forecast_str.split(",") if h.strip()]

    fireplace_lag_str = str(
        getattr(config, "COOLING_ML_CORRECTION_FIREPLACE_LAG_HOURS", "1,2")
    )
    fireplace_lag_hours = [float(h.strip()) for h in fireplace_lag_str.split(",") if h.strip()]

    tv_lag_str = str(
        getattr(config, "COOLING_ML_CORRECTION_TV_LAG_HOURS", "0.5,1")
    )
    tv_lag_hours = [float(h.strip()) for h in tv_lag_str.split(",") if h.strip()]

    # Start date override
    start_date_str = str(
        getattr(config, "COOLING_ML_CORRECTION_CALIBRATION_START_DATE", "")
    )
    if start_date_str:
        try:
            from datetime import datetime as _dt
            start_dt = _dt.strptime(start_date_str.strip(), "%d.%m.%Y")
            now = _dt.now()
            lookback_hours = max(
                int((now - start_dt).total_seconds() / 3600), lookback_hours
            )
            logger.info(
                "Cooling ML calibration start date %s → lookback=%d hours",
                start_date_str, lookback_hours,
            )
        except ValueError:
            logger.warning(
                "Invalid COOLING_ML_CORRECTION_CALIBRATION_START_DATE=%r — "
                "using default lookback=%d", start_date_str, lookback_hours,
            )

    logger.info(
        "=== COOLING CORRECTION ML CALIBRATION START ===\n"
        "  target=%.1f°C  horizon=%dh  lookback=%dh  warm_threshold=%.1f°C",
        cooling_target_c, label_horizon_h, lookback_hours, warm_threshold,
    )

    # ── 2. Fetch historical data ────────────────────────────────────────
    df = fetch_historical_data_for_calibration(
        lookback_hours=lookback_hours,
        purpose="cooling",
    )
    if df is None or df.empty:
        logger.error("No historical data returned — aborting cooling ML calibration")
        return False

    logger.info("Fetched %d rows of historical data", len(df))

    # ── 3. Rename entity-ID columns → model-friendly names ─────────────
    indoor_col = getattr(
        config, "INDOOR_TEMP_ENTITY_ID", "sensor.rt_mittelwert"
    ).split(".", 1)[-1]
    outdoor_col = getattr(
        config, "OUTDOOR_TEMP_ENTITY_ID", "sensor.nibe_bt1_outdoor_temperature"
    ).split(".", 1)[-1]
    outlet_col = getattr(
        config, "OUTLET_TEMP_ENTITY_ID", "sensor.nibe_bt2_supply_temp_s1"
    ).split(".", 1)[-1]
    inlet_col = getattr(
        config, "INLET_TEMP_ENTITY_ID", "sensor.nibe_eb100_ep14_bt3_return_temp"
    ).split(".", 1)[-1]
    flow_col = getattr(
        config, "FLOW_RATE_ENTITY_ID", "input_number.hp_current_flow_rate"
    ).split(".", 1)[-1]
    power_col = getattr(
        config, "POWER_CONSUMPTION_ENTITY_ID", "sensor.nibe_el_leistung"
    ).split(".", 1)[-1]
    pv_col = getattr(
        config, "PV_POWER_ENTITY_ID", "sensor.pv_leistung_gefiltert"
    ).split(".", 1)[-1]
    fireplace_col = getattr(
        config, "FIREPLACE_STATUS_ENTITY_ID", "binary_sensor.fireplace_active"
    ).split(".", 1)[-1]
    tv_col = getattr(
        config, "TV_STATUS_ENTITY_ID", "input_boolean.fernseher"
    ).split(".", 1)[-1]
    living_room_col = getattr(
        config, "LIVING_ROOM_TEMP_ENTITY_ID", "sensor.rt_wz"
    ).split(".", 1)[-1]
    wind_col = getattr(
        config, "WIND_SPEED_ENTITY_ID", "sensor.wind_speed"
    ).split(".", 1)[-1]

    rename_map = {
        indoor_col:      "indoor_temp",
        outdoor_col:     "AT",
        outlet_col:      "VLT",
        inlet_col:       "RLT",
        flow_col:        "flow_rate",
        power_col:       "power_w",
        pv_col:          "PV_Generate",
        fireplace_col:   "fireplace_on",
        tv_col:          "tv_on",
        living_room_col: "living_room_temp",
        wind_col:        "wind_speed",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["indoor_temp", "AT", "VLT", "RLT"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        logger.error(
            "Missing required columns after rename: %s — aborting", missing_cols
        )
        return False

    # Sort by time
    if "_time" in df.columns:
        df = df.sort_values("_time").reset_index(drop=True)

    # Coerce numeric
    for col in ["indoor_temp", "VLT", "RLT", "AT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["indoor_temp", "VLT", "RLT", "AT"], inplace=True)

    # ── 3b. Warm-season filter ──────────────────────────────────────────
    df = df[df["AT"] >= warm_threshold].copy()
    logger.info(
        "After warm-season filter (AT >= %.1f°C): %d rows",
        warm_threshold, len(df),
    )

    if len(df) < 500:
        logger.error(
            "Only %d warm-season rows available — need at least 500. "
            "Increase lookback_hours or adjust COOLING_ML_CORRECTION_WARM_THRESHOLD_C.",
            len(df),
        )
        return False

    df = df.reset_index(drop=True)

    # ── 4. Derived features ─────────────────────────────────────────────
    specific_heat = float(getattr(config, "SPECIFIC_HEAT_CAPACITY", 4.186))

    df["delta_t"] = (
        pd.to_numeric(df["VLT"], errors="coerce")
        - pd.to_numeric(df["RLT"], errors="coerce")
    )
    if "flow_rate" in df.columns:
        df["flow_rate"] = pd.to_numeric(df["flow_rate"], errors="coerce").fillna(0.0)
        df["thermal_power_kw"] = (
            df["flow_rate"] * specific_heat * df["delta_t"] / 60.0
        )
    elif "power_w" in df.columns:
        df["power_w"] = pd.to_numeric(df["power_w"], errors="coerce").fillna(0.0)
        df["thermal_power_kw"] = df["power_w"] / 1000.0
    else:
        df["thermal_power_kw"] = 0.0
    df["thermal_power_w"] = df["thermal_power_kw"] * 1000.0

    df["outlet_indoor_diff"] = df["VLT"] - df["indoor_temp"]
    df["at_delta_indoor"] = df["AT"] - df["indoor_temp"]
    df["indoor_margin"] = cooling_target_c - df["indoor_temp"]

    # Rolling indoor trends
    df["indoor_trend_30m"] = df["indoor_temp"].diff(3)
    df["indoor_trend_1h"] = df["indoor_temp"].diff(steps_per_hour)

    # ── 4b. Standard ML features ────────────────────────────────────────
    if "wind_speed" in df.columns:
        df["wind_speed"] = pd.to_numeric(
            df["wind_speed"], errors="coerce"
        ).fillna(0.0).clip(0, 200)
    else:
        df["wind_speed"] = 0.0

    df["indoor_temp_gradient"] = df["indoor_temp"].diff() * steps_per_hour

    df["is_hp_active"] = (df["delta_t"].abs() > 1.0).astype(float)

    if "_time" in df.columns:
        ts_wd = pd.to_datetime(df["_time"], utc=True)
        df["is_weekend"] = ts_wd.dt.dayofweek.isin([5, 6]).astype(float)
    else:
        df["is_weekend"] = 0.0

    df["thermal_power_rolling_1h"] = df["thermal_power_w"].rolling(
        steps_per_hour, min_periods=1
    ).mean()

    df["indoor_margin_rate"] = df["indoor_margin"].diff() * steps_per_hour

    df["is_overshoot"] = (df["indoor_temp"] > cooling_target_c).astype(float)

    df["d_inlet_temp_60min"] = df["RLT"].diff(steps_per_hour)
    df["is_equilibrium"] = (df["d_inlet_temp_60min"].abs() < 0.3).astype(float)

    # Fireplace and TV features
    for src_col in ["fireplace_on", "tv_on"]:
        if src_col in df.columns:
            df[src_col] = pd.to_numeric(
                df[src_col], errors="coerce"
            ).fillna(0.0).clip(0, 1)
        else:
            df[src_col] = 0.0

    for lag_h in fireplace_lag_hours:
        n_steps = max(1, int(round(lag_h * steps_per_hour)))
        if lag_h == int(lag_h):
            col_name = f"fireplace_lag_{int(lag_h)}h"
        else:
            col_name = f"fireplace_lag_{int(round(lag_h * 60))}m"
        df[col_name] = df["fireplace_on"].rolling(n_steps, min_periods=1).max()

    for lag_h in tv_lag_hours:
        n_steps = max(1, int(round(lag_h * steps_per_hour)))
        if lag_h == int(lag_h):
            col_name = f"tv_lag_{int(lag_h)}h"
        else:
            col_name = f"tv_lag_{int(round(lag_h * 60))}m"
        df[col_name] = df["tv_on"].rolling(n_steps, min_periods=1).max()

    # PV features
    if "PV_Generate" not in df.columns:
        df["PV_Generate"] = 0.0
    else:
        df["PV_Generate"] = pd.to_numeric(
            df["PV_Generate"], errors="coerce"
        ).fillna(0.0)

    df["pv_roll_1h"] = df["PV_Generate"].rolling(steps_per_hour, min_periods=1).mean()
    df["pv_roll_2h"] = df["PV_Generate"].rolling(2 * steps_per_hour, min_periods=1).mean()

    # Temporal cyclical features
    if "_time" in df.columns:
        ts = pd.to_datetime(df["_time"], utc=True)
        hour_frac = ts.dt.hour + ts.dt.minute / 60.0
        doy = ts.dt.dayofyear
    else:
        hour_frac = pd.Series([12.0] * len(df))
        doy = pd.Series([180] * len(df))

    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # AT hindcast substitution
    for h in at_forecast_hours:
        shift = h * steps_per_hour
        df[f"AT_roh_{h}h"] = df["AT"].shift(-shift)

    # PV hindcast substitution
    for h in pv_forecast_hours:
        shift = h * steps_per_hour
        df[f"pv_forecast_{h}h"] = df["PV_Generate"].shift(-shift)
    if "pv_forecast_2h" not in df.columns:
        df["pv_forecast_2h"] = df["PV_Generate"].shift(-2 * steps_per_hour)

    # ── 4c. Physics-motivated features ──────────────────────────────────
    df["heat_loss_driving_force"] = df["indoor_temp"] - df["AT"]
    df["delta_T_indoor_lag1"] = df["indoor_temp"].diff(1).fillna(0.0)

    if "flow_rate" in df.columns:
        specific_heat_j_per_kgk = specific_heat * 1000.0
        df["Q_wp"] = np.where(
            df["flow_rate"] > 0,
            (df["flow_rate"] / 60.0) * (df["VLT"] - df["RLT"]) * specific_heat_j_per_kgk,
            0.0,
        )
    else:
        df["Q_wp"] = 0.0

    df["solar_thermal_proxy"] = df["PV_Generate"] * df["hour_cos"]

    if "pv_forecast_2h" in df.columns:
        df["pv_forecast_delta"] = (df["pv_forecast_2h"] - df["PV_Generate"]).fillna(0.0)
    else:
        df["pv_forecast_delta"] = 0.0

    # ── 4d. Physics interaction features ────────────────────────────────
    df["shading_proxy"] = (
        (df["indoor_temp"] - _SHADING_ACTIVATION_TEMP_C).clip(lower=0.0) * df["PV_Generate"]
    ).fillna(0.0)

    df["heat_loss_interaction"] = (
        (df["indoor_temp"] - df["AT"]) * df["wind_speed"]
    ).fillna(0.0)

    # ── 4e. NB08/NB09-derived features ──────────────────────────────────
    df["cumulative_Q_wp_4h"] = (
        df["Q_wp"].rolling(label_horizon_steps, min_periods=1).sum().fillna(0.0)
    )
    df["indoor_accel"] = df["indoor_trend_1h"].diff(1).fillna(0.0)

    _max_at_h = max(at_forecast_hours) if at_forecast_hours else 4
    _at_max_col = f"AT_roh_{_max_at_h}h"
    if _at_max_col in df.columns:
        df["AT_forecast_trend"] = (df[_at_max_col] - df["AT"]).fillna(0.0)
    else:
        df["AT_forecast_trend"] = 0.0

    df["pv_cumulative_4h"] = (
        df["PV_Generate"].rolling(label_horizon_steps, min_periods=1).sum().fillna(0.0)
    )
    df["thermal_momentum"] = (
        df["thermal_power_rolling_1h"] * df["delta_t"]
    ).fillna(0.0)

    # ── 4f. Trajectory-derived physics features (vectorized) ────────────
    _traj_eta, _traj_u, _traj_tau = _read_cooling_thermal_params(config)
    if _traj_tau < 0.1:
        _traj_tau = float(getattr(config, "THERMAL_TIME_CONSTANT", 4.39))
    if (_traj_eta + _traj_u) < 1e-6:
        _traj_eta = float(getattr(config, "COOLING_OUTLET_EFFECTIVENESS", 0.20))
        _traj_u = float(getattr(config, "HEAT_LOSS_COEFFICIENT", 0.124))

    _traj_denom = _traj_eta + _traj_u
    df["_traj_T_eq"] = (_traj_eta * df["VLT"] + _traj_u * df["AT"]) / _traj_denom

    _traj_H = float(label_horizon_h)
    _traj_steps = int(_traj_H * steps_per_hour)

    _step_hours = 1.0 / steps_per_hour
    _t_first = _step_hours
    _t_last = _traj_H

    _exp_first = np.exp(-_t_first / _traj_tau)
    df["_traj_step_1"] = df["_traj_T_eq"] + (df["indoor_temp"] - df["_traj_T_eq"]) * _exp_first

    _exp_last = np.exp(-_t_last / _traj_tau)
    df["_traj_step_last"] = df["_traj_T_eq"] + (df["indoor_temp"] - df["_traj_T_eq"]) * _exp_last

    df["traj_predicted_error"] = (df["_traj_step_last"] - cooling_target_c).fillna(0.0)
    df["traj_convergence_rate"] = (
        (df["_traj_step_1"] - df["_traj_step_last"]) / max(1, _traj_steps)
    ).fillna(0.0)

    _ratio = (cooling_target_c - df["_traj_T_eq"]) / (df["indoor_temp"] - df["_traj_T_eq"])
    _valid_mask = (_ratio > 0) & (_ratio < 1)
    _log_ratio = np.where(_valid_mask, np.log(_ratio.clip(1e-10, None)), float("nan"))
    df["traj_reaches_target_hours"] = np.where(
        _valid_mask, (-_traj_tau * _log_ratio).clip(0, _traj_H), _traj_H
    )
    df["traj_reaches_target_hours"] = pd.to_numeric(
        df["traj_reaches_target_hours"], errors="coerce"
    ).fillna(_traj_H)

    # Overshoot: For cooling, overshoot means temp below target
    _traj_max = df[["_traj_step_1", "_traj_step_last"]].min(axis=1)
    df["traj_overshoot_magnitude"] = (cooling_target_c - _traj_max).clip(lower=0.0).fillna(0.0)

    df["traj_equilibrium_gap"] = (df["_traj_T_eq"] - cooling_target_c).fillna(0.0)

    # Cleanup internal trajectory columns
    df.drop(columns=["_traj_T_eq", "_traj_step_1", "_traj_step_last"], inplace=True, errors="ignore")

    # ── 5. S_H for cooling ──────────────────────────────────────────────
    eta, u_loss, tau_room = _read_cooling_thermal_params(config)
    s_h = _compute_s_h(eta, u_loss, tau_room, float(label_horizon_h))
    if s_h < 0.05:
        eta_fb = float(getattr(config, "COOLING_OUTLET_EFFECTIVENESS", 0.20))
        u_fb = float(getattr(config, "HEAT_LOSS_COEFFICIENT", 0.124))
        tau_fb = float(getattr(config, "THERMAL_TIME_CONSTANT", 4.39))
        s_h = _compute_s_h(eta_fb, u_fb, tau_fb, float(label_horizon_h))
        logger.warning(
            "S_H from persisted cooling params is degenerate; "
            "fell back to config defaults → S_H=%.4f", s_h,
        )
    if s_h < 0.01:
        logger.error("Cooling S_H=%.6f still degenerate after fallback — aborting", s_h)
        return False
    logger.info(
        "Cooling S_H estimate: OE=%.4f HLC=%.4f τ=%.2fh H=%dh → S_H=%.4f",
        eta, u_loss, tau_room, label_horizon_h, s_h,
    )

    # ── 5b. Residualized label ──────────────────────────────────────────
    future_indoor = df["indoor_temp"].shift(-label_horizon_steps)
    raw_label = -(future_indoor - df["indoor_temp"]) / s_h
    df["label"] = raw_label

    # ── 5c. Forward-looking outlier filtering ────────────────────────────
    n_before = len(df)

    if "fireplace_on" in df.columns:
        fp_fwd = (
            df["fireplace_on"]
            .iloc[::-1]
            .rolling(label_horizon_steps, min_periods=1)
            .max()
            .iloc[::-1]
        )
        fp_mask = fp_fwd > 0.5
        n_fp = int(fp_mask.sum())
        df = df[~fp_mask].copy()
        logger.info("Outlier filter: fireplace forward-look → removed %d rows", n_fp)

    indoor_drop_3 = df["indoor_temp"].diff(3)
    window_mask_base = indoor_drop_3 < -0.3
    window_fwd = (
        window_mask_base.astype(float)
        .iloc[::-1]
        .rolling(label_horizon_steps, min_periods=1)
        .max()
        .iloc[::-1]
    )
    n_win = int((window_fwd > 0.5).sum())
    df = df[window_fwd <= 0.5].copy()
    logger.info("Outlier filter: window-open proxy → removed %d rows", n_win)

    if "PV_Generate" in df.columns and df["PV_Generate"].max() > 0:
        pv_p999 = df["PV_Generate"].quantile(0.999)
        pv_spike_mask = df["PV_Generate"] > pv_p999 * 1.5
        n_pv = int(pv_spike_mask.sum())
        df = df[~pv_spike_mask].copy()
        logger.info("Outlier filter: PV spike → removed %d rows", n_pv)

    extreme_mask = df["label"].abs() > 5.0
    n_ext = int(extreme_mask.sum())
    df = df[~extreme_mask].copy()
    logger.info("Outlier filter: extreme label → removed %d rows", n_ext)

    df = df.reset_index(drop=True)
    logger.info(
        "Outlier filtering total: %d → %d rows (%.1f%% removed)",
        n_before, len(df), 100.0 * (1 - len(df) / max(1, n_before)),
    )

    if len(df) < 100:
        logger.error(
            "Outlier filters removed too many rows (%d remaining) — aborting",
            len(df),
        )
        return False

    # ── 6. Feature set ──────────────────────────────────────────────────
    # No indoor_temp or living_room_temp (redundant with indoor_margin
    # under residualized label).
    feature_cols = [
        "indoor_margin",
        "indoor_trend_30m",
        "indoor_trend_1h",
        "AT",
        "at_delta_indoor",
    ]
    for h in at_forecast_hours:
        feature_cols.append(f"AT_roh_{h}h")

    feature_cols += [
        "VLT",
        "RLT",
        "delta_t",
        "outlet_indoor_diff",
        "thermal_power_w",
        "fireplace_on",
        "tv_on",
    ]

    for lag_h in fireplace_lag_hours:
        if lag_h == int(lag_h):
            feature_cols.append(f"fireplace_lag_{int(lag_h)}h")
        else:
            feature_cols.append(f"fireplace_lag_{int(round(lag_h * 60))}m")

    for lag_h in tv_lag_hours:
        if lag_h == int(lag_h):
            feature_cols.append(f"tv_lag_{int(lag_h)}h")
        else:
            feature_cols.append(f"tv_lag_{int(round(lag_h * 60))}m")

    feature_cols += [
        "PV_Generate",
        "pv_roll_1h",
        "pv_roll_2h",
    ]
    for h in pv_forecast_hours:
        feature_cols.append(f"pv_forecast_{h}h")

    feature_cols += [
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]

    feature_cols += [
        "wind_speed",
        "indoor_temp_gradient",
        "is_hp_active",
        "is_weekend",
        "thermal_power_rolling_1h",
        "indoor_margin_rate",
        "is_overshoot",
    ]

    feature_cols += [
        "d_inlet_temp_60min",
        "is_equilibrium",
    ]

    feature_cols += [
        "heat_loss_driving_force",
        "delta_T_indoor_lag1",
        "Q_wp",
        "solar_thermal_proxy",
        "pv_forecast_delta",
    ]

    feature_cols += [
        "shading_proxy",
        "heat_loss_interaction",
    ]

    feature_cols += [
        "traj_predicted_error",
        "traj_convergence_rate",
        "traj_reaches_target_hours",
        "traj_overshoot_magnitude",
        "traj_equilibrium_gap",
    ]

    feature_cols += [
        "cumulative_Q_wp_4h",
        "indoor_accel",
        "AT_forecast_trend",
        "pv_cumulative_4h",
        "thermal_momentum",
    ]

    # Guard: only keep columns that exist and have > 5% coverage
    available_features = []
    for col in feature_cols:
        if col not in df.columns:
            logger.warning("Feature '%s' not in dataframe — skipping", col)
            continue
        coverage = df[col].notna().mean()
        if coverage < 0.05:
            logger.warning(
                "Feature '%s' coverage %.1f%% < 5%% — skipping",
                col, 100 * coverage,
            )
            continue
        available_features.append(col)

    if len(available_features) < 5:
        logger.error(
            "Too few usable feature columns (%d) — aborting", len(available_features)
        )
        return False

    # ── 7. Drop rows with NaN in features or label ──────────────────────
    df_train = df[available_features + ["label"]].dropna().copy()
    feature_cols = available_features

    logger.info(
        "Training set: %d rows, %d features, label mean=%.3f std=%.3f",
        len(df_train), len(feature_cols),
        df_train["label"].mean(), df_train["label"].std(),
    )

    min_samples = int(
        getattr(config, "COOLING_ML_CORRECTION_MIN_TRAINING_SAMPLES", 200)
    )
    if len(df_train) < min_samples:
        logger.error(
            "Only %d training samples (need %d). "
            "Increase lookback_hours or adjust warm_threshold.",
            len(df_train), min_samples,
        )
        return False

    # ── 8. Train / val split (temporal) ─────────────────────────────────
    val_frac = float(
        getattr(config, "COOLING_ML_CORRECTION_RETRAIN_VAL_FRACTION", 0.25)
    )
    split_idx = int(len(df_train) * (1.0 - val_frac))
    df_fit = df_train.iloc[:split_idx]
    df_val = df_train.iloc[split_idx:]

    X_fit = df_fit[feature_cols]
    y_fit = df_fit["label"]
    X_val = df_val[feature_cols]
    y_val = df_val["label"]

    logger.info(
        "Split: train=%d val=%d (%.0f%%)",
        len(df_fit), len(df_val), 100 * val_frac,
    )

    # ── 9. LightGBM training ───────────────────────────────────────────
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm not installed — cannot train")
        return False

    reg_alpha = float(getattr(config, "COOLING_ML_CORRECTION_REG_ALPHA", 0.1))
    reg_lambda = float(getattr(config, "COOLING_ML_CORRECTION_REG_LAMBDA", 1.0))

    lgb_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "verbosity": -1,
    }

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    y_pred_val = model.predict(X_val)
    val_mae = float(np.mean(np.abs(y_val - y_pred_val)))

    ss_res = float(np.sum((y_val - y_pred_val) ** 2))
    ss_tot = float(np.sum((y_val - y_val.mean()) ** 2))
    val_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

    logger.info(
        "Pre-pruning model: val_MAE=%.4f R²=%.4f",
        val_mae, val_r2,
    )

    # Reconstructed R² (on original label scale, for notebook comparison)
    if s_h > 0.05 and "indoor_margin" in df_val.columns:
        margin_val = df_val["indoor_margin"].values
        y_recon_pred = y_pred_val + margin_val / s_h
        y_recon_true = y_val + margin_val / s_h
        ss_res_r = float(np.sum((y_recon_true - y_recon_pred) ** 2))
        ss_tot_r = float(np.sum((y_recon_true - float(np.mean(y_recon_true))) ** 2))
        recon_r2 = 1.0 - ss_res_r / ss_tot_r if ss_tot_r > 1e-10 else 0.0
        recon_mae = float(np.mean(np.abs(y_recon_true - y_recon_pred)))
        logger.info(
            "Reconstructed (original label scale): MAE=%.4f°C, R²=%.4f",
            recon_mae, recon_r2,
        )

    # Feature importances
    feat_imp_pairs = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    logger.info("Top-10 feature importances:")
    for fname, imp in feat_imp_pairs[:10]:
        logger.info("  %-35s  %6d", fname, imp)

    # ── 10. Optional feature pruning ────────────────────────────────────
    pruning_enabled = bool(
        getattr(config, "COOLING_ML_CORRECTION_FEATURE_PRUNING_ENABLED", True)
    )
    prune_threshold = float(
        getattr(config, "COOLING_ML_CORRECTION_PRUNE_PI_THRESHOLD", 0.0)
    )
    incremental_pruning_enabled = bool(
        getattr(config, "COOLING_ML_CORRECTION_INCREMENTAL_PRUNING_ENABLED", False)
    )
    incremental_pi_threshold = float(
        getattr(config, "COOLING_ML_CORRECTION_INCREMENTAL_PRUNE_PI_THRESHOLD", 0.001)
    )
    pruning_mode = "disabled"
    pruning_dropped_features: list[str] = []
    pruning_steps_applied = 0

    if pruning_enabled and len(feature_cols) > 5:
        from sklearn.inspection import permutation_importance  # type: ignore

        if incremental_pruning_enabled:
            pruning_mode = "incremental"
            current_features = list(feature_cols)
            current_model = model
            current_mae = val_mae
            current_r2 = val_r2
            max_steps = max(0, len(current_features) - 5)

            logger.info(
                "Incremental pruning enabled (PI threshold <= %.4f)",
                incremental_pi_threshold,
            )

            for step_idx in range(1, max_steps + 1):
                step_pi = permutation_importance(
                    current_model,
                    df_val[current_features],
                    y_val,
                    n_repeats=5,
                    scoring="neg_mean_absolute_error",
                    random_state=42,
                )
                step_pairs = sorted(
                    zip(current_features, step_pi.importances_mean),
                    key=lambda x: x[1],
                )
                candidates = [
                    (fname, imp)
                    for fname, imp in step_pairs
                    if imp <= incremental_pi_threshold
                ]
                if not candidates:
                    logger.info(
                        "Incremental pruning converged after %d accepted steps",
                        pruning_steps_applied,
                    )
                    break

                worst_feature, worst_pi = candidates[0]
                remaining_features = [
                    fname for fname in current_features if fname != worst_feature
                ]
                if len(remaining_features) < 5:
                    logger.info(
                        "Incremental pruning stopped: removing '%s' would leave %d features",
                        worst_feature,
                        len(remaining_features),
                    )
                    break

                model_step = lgb.LGBMRegressor(**lgb_params)
                model_step.fit(
                    df_fit[remaining_features],
                    y_fit,
                    eval_set=[(df_val[remaining_features], y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                y_pred_step = model_step.predict(df_val[remaining_features])
                step_mae = float(np.mean(np.abs(y_val - y_pred_step)))
                step_ss_res = float(np.sum((y_val - y_pred_step) ** 2))
                step_r2 = 1.0 - step_ss_res / max(ss_tot, 1e-10)
                regression_pct = (
                    (step_mae - current_mae) / max(current_mae, 1e-10)
                ) * 100.0

                if regression_pct <= 0.5:
                    logger.info(
                        "Incremental pruning step %d accepted: drop '%s' (PI=%.4f), "
                        "MAE %.4f -> %.4f (%+.2f%%), features %d -> %d",
                        step_idx,
                        worst_feature,
                        worst_pi,
                        current_mae,
                        step_mae,
                        regression_pct,
                        len(current_features),
                        len(remaining_features),
                    )
                    current_features = remaining_features
                    current_model = model_step
                    current_mae = step_mae
                    current_r2 = step_r2
                    pruning_steps_applied += 1
                    pruning_dropped_features.append(worst_feature)
                else:
                    logger.info(
                        "Incremental pruning stopped: rejecting drop '%s' (PI=%.4f), "
                        "MAE regression %+.2f%% > 0.5%%",
                        worst_feature,
                        worst_pi,
                        regression_pct,
                    )
                    break

            model = current_model
            feature_cols = current_features
            val_mae = current_mae
            val_r2 = current_r2
            feat_imp_pairs = sorted(
                zip(feature_cols, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        else:
            pruning_mode = "standard"
            pi_result = permutation_importance(
                model, X_val, y_val, n_repeats=5,
                scoring="neg_mean_absolute_error", random_state=42,
            )
            pi_means = pi_result.importances_mean

            keep_mask = pi_means > prune_threshold
            pruned_cols = [c for c, keep in zip(feature_cols, keep_mask) if keep]
            dropped_cols = [c for c, keep in zip(feature_cols, keep_mask) if not keep]

            if dropped_cols and len(pruned_cols) >= 5:
                logger.info(
                    "Pruning: dropping %d features with PI <= %.4f: %s",
                    len(dropped_cols), prune_threshold, dropped_cols,
                )
                model_pruned = lgb.LGBMRegressor(**lgb_params)
                model_pruned.fit(
                    df_fit[pruned_cols], y_fit,
                    eval_set=[(df_val[pruned_cols], y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                y_pred_pruned = model_pruned.predict(df_val[pruned_cols])
                pruned_mae = float(np.mean(np.abs(y_val - y_pred_pruned)))

                regression_pct = (
                    (pruned_mae - val_mae) / max(val_mae, 1e-10)
                ) * 100.0

                if regression_pct <= 0.5:
                    logger.info(
                        "Pruned model accepted: MAE=%.4f (regression=%.2f%%)",
                        pruned_mae, regression_pct,
                    )
                    model = model_pruned
                    feature_cols = pruned_cols
                    val_mae = pruned_mae
                    ss_res_p = float(np.sum((y_val - y_pred_pruned) ** 2))
                    val_r2 = 1.0 - ss_res_p / max(ss_tot, 1e-10)
                    pruning_steps_applied = 1
                    pruning_dropped_features = list(dropped_cols)
                    feat_imp_pairs = sorted(
                        zip(feature_cols, model.feature_importances_),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                else:
                    logger.info(
                        "Pruned model rejected: MAE regression %.2f%% > 0.5%%",
                        regression_pct,
                    )
    elif pruning_enabled:
        pruning_mode = "skipped_too_few_features"

    # ── 11. Optional Optuna HPO ─────────────────────────────────────────
    optuna_enabled = bool(
        getattr(config, "COOLING_ML_CORRECTION_OPTUNA_ENABLED", False)
    )
    if optuna_enabled:
        try:
            import optuna  # type: ignore

            n_trials = int(
                getattr(config, "COOLING_ML_CORRECTION_OPTUNA_N_TRIALS", 20)
            )

            def _objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                    "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 4, 10),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                    "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    "objective": "regression_l1",
                    "metric": "mae",
                    "verbosity": -1,
                }
                m = lgb.LGBMRegressor(**params)
                m.fit(
                    X_fit[feature_cols] if isinstance(X_fit, pd.DataFrame) else df_fit[feature_cols],
                    y_fit,
                    eval_set=[(df_val[feature_cols], y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                return float(np.mean(np.abs(y_val - m.predict(df_val[feature_cols]))))

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize")
            study.optimize(_objective, n_trials=n_trials)

            if study.best_value < val_mae:
                logger.info(
                    "Optuna improved MAE: %.4f → %.4f",
                    val_mae, study.best_value,
                )
                best = study.best_params
                best["objective"] = "regression_l1"
                best["metric"] = "mae"
                best["verbosity"] = -1
                if "lr" in best:
                    best["learning_rate"] = best.pop("lr")
                model = lgb.LGBMRegressor(**best)
                model.fit(
                    df_fit[feature_cols], y_fit,
                    eval_set=[(df_val[feature_cols], y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                y_pred_opt = model.predict(df_val[feature_cols])
                val_mae = float(np.mean(np.abs(y_val - y_pred_opt)))
                ss_res_o = float(np.sum((y_val - y_pred_opt) ** 2))
                val_r2 = 1.0 - ss_res_o / max(ss_tot, 1e-10)
                lgb_params = best
                feat_imp_pairs = sorted(
                    zip(feature_cols, model.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )
            else:
                logger.info(
                    "Optuna did not improve MAE (%.4f vs %.4f) — keeping pre-tuning model",
                    study.best_value, val_mae,
                )
        except ImportError:
            logger.warning("optuna not installed — skipping HPO")

    # ── 11b. Optional cross-validation ──────────────────────────────────
    cv_enabled = bool(
        getattr(config, "COOLING_ML_CORRECTION_CV_ENABLED", False)
    )
    if cv_enabled:
        try:
            from sklearn.model_selection import TimeSeriesSplit  # type: ignore

            n_splits = int(
                getattr(config, "COOLING_ML_CORRECTION_CV_N_SPLITS", 3)
            )
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []
            for train_idx, val_idx in tscv.split(df_train):
                X_cv_train = df_train.iloc[train_idx][feature_cols]
                y_cv_train = df_train.iloc[train_idx]["label"]
                X_cv_val = df_train.iloc[val_idx][feature_cols]
                y_cv_val = df_train.iloc[val_idx]["label"]

                m_cv = lgb.LGBMRegressor(**lgb_params)
                m_cv.fit(
                    X_cv_train, y_cv_train,
                    eval_set=[(X_cv_val, y_cv_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                cv_pred = m_cv.predict(X_cv_val)
                ss_r = float(np.sum((y_cv_val - cv_pred) ** 2))
                ss_t = float(np.sum((y_cv_val - y_cv_val.mean()) ** 2))
                cv_r2 = 1.0 - ss_r / max(ss_t, 1e-10)
                cv_scores.append(cv_r2)

            logger.info(
                "CV R² scores: %s  mean=%.4f ± %.4f",
                [f"{s:.4f}" for s in cv_scores],
                np.mean(cv_scores),
                np.std(cv_scores),
            )
        except ImportError:
            logger.warning("sklearn not available for cross-validation")

    logger.info(
        "=== FINAL MODEL: val_MAE=%.4f R²=%.4f features=%d ===",
        val_mae, val_r2, len(feature_cols),
    )

    # ── 12. Save model + metadata ───────────────────────────────────────
    try:
        import joblib  # type: ignore
    except ImportError:
        logger.error("joblib not installed — cannot save model")
        return False

    model_path = getattr(
        config,
        "COOLING_ML_CORRECTION_MODEL_PATH",
        "/opt/ml_heating/cooling_correction_ml_model.joblib",
    )
    metadata_path = getattr(
        config,
        "COOLING_ML_CORRECTION_METADATA_PATH",
        "/opt/ml_heating/cooling_correction_ml_metadata.json",
    )

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    tmp_model = model_path + ".tmp"
    joblib.dump(model, tmp_model)
    os.replace(tmp_model, model_path)

    metadata = {
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "label_type": "residualized",
        "mode": "cooling",
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "val_mae": val_mae,
        "val_r2": val_r2,
        "n_train": int(len(df_fit)),
        "n_val": int(len(df_val)),
        "label_horizon_h": label_horizon_h,
        "steps_per_hour": steps_per_hour,
        "warm_threshold_c": warm_threshold,
        "cooling_target_c": cooling_target_c,
        "s_h_estimated": s_h,
        "eta": eta,
        "u_loss": u_loss,
        "tau_room": tau_room,
        "lookback_hours": lookback_hours,
        "at_forecast_hours": at_forecast_hours,
        "pv_forecast_hours": pv_forecast_hours,
        "fireplace_lag_hours": fireplace_lag_hours,
        "tv_lag_hours": tv_lag_hours,
        "lgb_params": lgb_params,
        "pruning_mode": pruning_mode,
        "pruning_threshold_standard": prune_threshold,
        "pruning_threshold_incremental": incremental_pi_threshold,
        "incremental_pruning_enabled": incremental_pruning_enabled,
        "pruning_steps_applied": pruning_steps_applied,
        "pruning_dropped_features": pruning_dropped_features,
        "feature_importances": {
            fname: int(imp) for fname, imp in feat_imp_pairs
        },
    }
    tmp_meta = metadata_path + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=_json_default)
    os.replace(tmp_meta, metadata_path)

    logger.info(
        "=== COOLING CORRECTION ML CALIBRATION COMPLETE: "
        "model → %s | MAE=%.4f R²=%.4f ===",
        model_path, val_mae, val_r2,
    )

    # Export training data for offline analysis
    all_available_cols = [c for c in df_train.columns if c != "label"]
    export_training_data(df_train, all_available_cols, "cooling_correction")

    return True
