"""
heating_correction_ml_calibration.py
--------------------------------------
One-shot training of the LightGBM heating-correction regressor.

Called via
  ``python -m src.main --calibrate-heating-correction-ml``
or triggered by the flag file ``/data/config/calibrate_heating_correction_ml_flag``.

Pipeline
--------
1.  Fetch multi-month historical data (same helper as physics calibration).
2.  Rename entity-ID columns to model-friendly names.
3.  Filter to cold-season rows (AT < HEATING_ML_COLD_THRESHOLD_C, default 18 °C).
4.  Compute derived features: indoor_margin, trends, delta_t, fireplace/TV lags,
    cyclical time features, AT hindcast substitution (1–4 h).
5.  Compute regression label: what ΔT_outlet would have zeroed the N-step
    future indoor error?
        label[t] = -(T_indoor[t + N_steps] - T_target[t]) / S_H
    where S_H = (η/(η+U)) × (1 − exp(−H/τ_room)) is computed from the
    unified thermal state (channel-aware when heat-source channels are enabled).
6.  Train LightGBM regressor (objective = "regression_l1" / MAE).
7.  Save model (joblib) + metadata JSON.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level config import so tests can patch `src.heating_correction_ml_calibration.config`.
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

# Module-level import of data-fetching helper so tests can patch it.
# Falls back gracefully to a stub that always returns None when the
# full src package is not on the path (e.g. standalone usage).
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


def _read_baseline_thermal_params(config) -> tuple[float, float, float]:
    """Read η, U, τ from unified thermal state with channel-aware precedence.

    Returns (outlet_effectiveness, heat_loss_coefficient, thermal_time_constant).

    Precedence:
    1) Active heat-pump channel parameters (when channels are enabled)
    2) Baseline + learning adjustments (computed parameters)
    3) Config defaults
    """
    try:
        from src.unified_thermal_state import get_thermal_state_manager
        state_manager = get_thermal_state_manager()

        eta_default = float(getattr(config, "OUTLET_EFFECTIVENESS", 0.830))
        u_default = float(getattr(config, "HEAT_LOSS_COEFFICIENT", 0.124))
        tau_default = float(getattr(config, "THERMAL_TIME_CONSTANT", 4.39))

        def _to_float_or(value, fallback: float) -> float:
            try:
                if value is None:
                    return fallback
                return float(value)
            except (TypeError, ValueError):
                return fallback

        get_computed = getattr(state_manager, "get_computed_parameters", None)
        computed = get_computed() if callable(get_computed) else {}
        eta_computed = _to_float_or(computed.get("outlet_effectiveness"), eta_default)
        u_computed = _to_float_or(computed.get("heat_loss_coefficient"), u_default)
        tau_computed = _to_float_or(computed.get("thermal_time_constant"), tau_default)

        # 1) If channels are enabled and heat_pump channel has parameters,
        # use those first because they represent the actively learned source.
        channels_enabled = bool(
            getattr(config, "ENABLE_HEAT_SOURCE_CHANNELS", False)
        )
        if channels_enabled:
            get_ch_state = getattr(state_manager, "get_heat_source_channel_state", None)
            channels = get_ch_state() if callable(get_ch_state) else {}
            hp_state = (channels or {}).get("heat_pump", {})
            hp_params = hp_state.get("parameters", {})
            hp_history = hp_state.get("history", [])
            hp_history_count = int(hp_state.get("history_count", len(hp_history) or 0))

            # Treat channel as active only when there is evidence it has participated.
            hp_active = hp_history_count > 0 or bool(hp_history)
            if hp_active and hp_params:
                eta = _to_float_or(hp_params.get("outlet_effectiveness"), eta_computed)
                u = _to_float_or(hp_params.get("heat_loss_coefficient"), u_computed)
                tau = _to_float_or(hp_params.get("thermal_time_constant"), tau_computed)
                logger.info(
                    "S_H params source: heat_source_channels.heat_pump.parameters"
                )
                return eta, u, tau

        # 2) Otherwise use baseline + adjustment deltas from unified state.
        if computed:
            eta = eta_computed
            u = u_computed
            tau = tau_computed
            logger.info(
                "S_H params source: unified_state baseline + parameter_adjustments"
            )
            return eta, u, tau

        # Legacy/state-shape fallback if get_computed_parameters is unavailable.
        state = getattr(state_manager, "state", {}) or {}
        baseline = state.get("baseline_parameters", {}) or {}
        learning_state = state.get("learning_state", {}) or {}
        adjustments = learning_state.get("parameter_adjustments", {}) or {}
        eta = float(baseline.get("outlet_effectiveness", eta_default)) + float(
            adjustments.get("outlet_effectiveness_delta", 0.0)
        )
        u = float(baseline.get("heat_loss_coefficient", u_default)) + float(
            adjustments.get("heat_loss_coefficient_delta", 0.0)
        )
        tau = float(baseline.get("thermal_time_constant", tau_default)) + float(
            adjustments.get("thermal_time_constant_delta", 0.0)
        )
        logger.info("S_H params source: unified_state legacy state dict")
        return eta, u, tau
    except Exception:
        logger.warning(
            "Could not read unified thermal state; using config defaults for S_H"
        )
        eta = float(getattr(config, "OUTLET_EFFECTIVENESS", 0.830))
        u = float(getattr(config, "HEAT_LOSS_COEFFICIENT", 0.124))
        tau = float(getattr(config, "THERMAL_TIME_CONSTANT", 4.39))
        return eta, u, tau


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def calibrate_heating_correction_ml(
    lookback_hours: int = 2160,  # 90 days
    heating_target_c: Optional[float] = None,
) -> bool:
    """
    Train and persist the LightGBM heating correction regressor.

    Parameters
    ----------
    lookback_hours:
        Hours of historical data to fetch.
    heating_target_c:
        Indoor target temperature [°C] used for the label calculation.
        Defaults to the value read from the HA state / config at call time.

    Returns
    -------
    bool: True on success, False on failure.
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        logger.error(
            "calibrate_heating_correction_ml: missing dependency — %s", exc
        )
        return False

    logger.info("=== HEATING CORRECTION ML CALIBRATION START ===")

    # ── 0. Parameters ──────────────────────────────────────────────────
    # Resolve lookback_hours from HEATING_ML_CALIBRATION_START_DATE when set.
    _start_date_str = getattr(config, "HEATING_ML_CALIBRATION_START_DATE", "")
    if _start_date_str and _start_date_str.strip():
        _parse_fn = getattr(config, "_parse_heating_start_date", None)
        _start_dt = _parse_fn(_start_date_str) if callable(_parse_fn) else None
        if _start_dt is not None:
            from datetime import timezone as _dt_tz
            _now_utc = datetime.now(_dt_tz.utc)
            _computed_h = math.ceil(
                (_now_utc - _start_dt).total_seconds() / 3600
            )
            if _computed_h > 0:
                lookback_hours = _computed_h
                logger.info(
                    "Resolved lookback_hours=%d from start date '%s'",
                    lookback_hours, _start_date_str,
                )
            else:
                logger.warning(
                    "HEATING_ML_CALIBRATION_START_DATE '%s' is in the future; "
                    "using default lookback_hours=%d",
                    _start_date_str, lookback_hours,
                )
        else:
            logger.warning(
                "HEATING_ML_CALIBRATION_START_DATE '%s' is not a valid DD.MM.YYYY "
                "date; using default lookback_hours=%d",
                _start_date_str, lookback_hours,
            )

    steps_per_hour = round(
        60 / float(getattr(config, "CYCLE_INTERVAL_MINUTES", 10))
    )
    label_horizon_h = int(getattr(config, "HEATING_ML_LABEL_HORIZON_H", 4))
    label_horizon_steps = label_horizon_h * steps_per_hour

    cold_threshold = float(getattr(config, "HEATING_ML_COLD_THRESHOLD_C", 18.0))

    # Heating target for label: fall back to a typical comfort setpoint
    if heating_target_c is None:
        heating_target_c = float(
            getattr(config, "HLC_DEFAULT_TARGET_TEMP", 21.0)
        )

    # AT forecast hours: default 1–4 h
    _at_fc_env = getattr(config, "HEATING_ML_AT_FORECAST_HOURS", "1,2,3,4")
    try:
        at_forecast_hours = [int(x) for x in _at_fc_env.split(",") if x.strip()]
    except ValueError:
        logger.warning(
            "HEATING_ML_AT_FORECAST_HOURS value %r is invalid; using 1,2,3,4",
            _at_fc_env,
        )
        at_forecast_hours = [1, 2, 3, 4]

    # PV forecast hours: default 1–4 h (optional; 0 values when no PV data)
    _pv_fc_env = getattr(config, "HEATING_ML_PV_FORECAST_HOURS", "1,2,3,4")
    try:
        pv_forecast_hours = [int(x) for x in _pv_fc_env.split(",") if x.strip()]
    except ValueError:
        logger.warning(
            "HEATING_ML_PV_FORECAST_HOURS value %r is invalid; using 1,2,3,4",
            _pv_fc_env,
        )
        pv_forecast_hours = [1, 2, 3, 4]

    # Fireplace lag windows [h]: default 1 h and 2 h
    _fp_lag_env = getattr(config, "HEATING_ML_FIREPLACE_LAG_HOURS", "1,2")
    try:
        fireplace_lag_hours = [
            float(x) for x in _fp_lag_env.split(",") if x.strip()
        ]
    except ValueError:
        logger.warning(
            "HEATING_ML_FIREPLACE_LAG_HOURS value %r is invalid; using 1,2",
            _fp_lag_env,
        )
        fireplace_lag_hours = [1.0, 2.0]

    # TV lag windows [h]: default 0.5 h (30 min) and 1 h
    _tv_lag_env = getattr(config, "HEATING_ML_TV_LAG_HOURS", "0.5,1")
    try:
        tv_lag_hours = [float(x) for x in _tv_lag_env.split(",") if x.strip()]
    except ValueError:
        logger.warning(
            "HEATING_ML_TV_LAG_HOURS value %r is invalid; using 0.5,1",
            _tv_lag_env,
        )
        tv_lag_hours = [0.5, 1.0]

    logger.info(
        "Calibration params: label_horizon=%dh (%d steps), cold_threshold=%.1f°C, "
        "target=%.1f°C, steps_per_hour=%d, lookback=%dh, "
        "AT_fc=%s, PV_fc=%s, fp_lag=%s, tv_lag=%s",
        label_horizon_h, label_horizon_steps, cold_threshold,
        heating_target_c, steps_per_hour, lookback_hours,
        at_forecast_hours, pv_forecast_hours, fireplace_lag_hours, tv_lag_hours,
    )

    # ── 1. Fetch historical data ────────────────────────────────────────
    df = fetch_historical_data_for_calibration(
        lookback_hours=lookback_hours,
        purpose="heating",
    )
    if df is None or df.empty:
        logger.error("Calibration aborted: no historical data fetched")
        return False

    logger.info("Fetched %d rows of historical data", len(df))

    # ── 2. Rename entity-ID columns → model-friendly names ─────────────
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
        indoor_col:    "indoor_temp",
        outdoor_col:   "AT",
        outlet_col:    "VLT",
        inlet_col:     "RLT",
        flow_col:      "flow_rate",
        power_col:     "power_w",
        pv_col:        "PV_Generate",
        fireplace_col: "fireplace_on",
        tv_col:        "tv_on",
        living_room_col: "living_room_temp",
        wind_col:      "wind_speed",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["indoor_temp", "AT", "VLT", "RLT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(
            "Calibration aborted: missing required columns: %s", missing
        )
        return False

    # Sort by time
    if "_time" in df.columns:
        df = df.sort_values("_time").reset_index(drop=True)

    # ── 3. Numeric coercion & cold-season filter ────────────────────────
    for col in ["indoor_temp", "AT", "VLT", "RLT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["AT"] < cold_threshold].copy()
    logger.info(
        "After cold-season filter (AT < %.1f°C): %d rows",
        cold_threshold, len(df),
    )

    if len(df) < 500:
        logger.error(
            "Only %d cold-season rows available — need at least 500. "
            "Increase lookback_hours or adjust HEATING_ML_COLD_THRESHOLD_C.",
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

    df["outlet_indoor_diff"] = df["VLT"] - df["indoor_temp"]
    df["at_delta_indoor"] = df["AT"] - df["indoor_temp"]
    df["indoor_margin"] = heating_target_c - df["indoor_temp"]

    # Rolling indoor trends (30 min = 3 steps at 10-min intervals)
    df["indoor_trend_30m"] = df["indoor_temp"].diff(3)
    df["indoor_trend_1h"] = df["indoor_temp"].diff(steps_per_hour)

    # ── 4b. NEW: 8 additional ML correction features ──────────────────
    # Wind speed — fill missing with 0 (calm)
    if "wind_speed" in df.columns:
        df["wind_speed"] = pd.to_numeric(
            df["wind_speed"], errors="coerce"
        ).fillna(0.0).clip(0, 200)
    else:
        df["wind_speed"] = 0.0

    # Indoor temp gradient (°C / 5-min step → °C/h)
    df["indoor_temp_gradient"] = df["indoor_temp"].diff() * steps_per_hour

    # Living room temperature
    if "living_room_temp" in df.columns:
        df["living_room_temp"] = pd.to_numeric(
            df["living_room_temp"], errors="coerce"
        ).fillna(method="ffill").fillna(df["indoor_temp"])
    else:
        df["living_room_temp"] = df["indoor_temp"]

    # Heat pump active — delta_t > 1°C indicates flow
    df["is_hp_active"] = (df["delta_t"].abs() > 1.0).astype(float)

    # Weekend indicator
    if "_time" in df.columns:
        ts_wd = pd.to_datetime(df["_time"], utc=True)
        df["is_weekend"] = ts_wd.dt.dayofweek.isin([5, 6]).astype(float)
    else:
        df["is_weekend"] = 0.0

    # Rolling 1-hour thermal power
    df["thermal_power_rolling_1h"] = df["thermal_power_kw"].rolling(
        steps_per_hour, min_periods=1
    ).mean()

    # Indoor margin rate of change (°C/h)
    df["indoor_margin_rate"] = df["indoor_margin"].diff() * steps_per_hour

    # Overshoot indicator: indoor > target
    df["is_overshoot"] = (df["indoor_temp"] > heating_target_c).astype(float)

    # Slab thermal loading trend over 60 min (steps_per_hour cycles × 10 min):
    # positive → slab absorbing heat; near zero → equilibrium; negative → cool-down
    df["d_inlet_temp_60min"] = df["RLT"].diff(steps_per_hour)
    # Binary flag: |ΔT_rl over 60 min| < 0.3 K → system in thermal steady state
    df["is_equilibrium"] = (df["d_inlet_temp_60min"].abs() < 0.3).astype(float)

    # Fireplace and TV features (binary) — fill missing with 0
    for src_col in ["fireplace_on", "tv_on"]:
        if src_col in df.columns:
            df[src_col] = pd.to_numeric(
                df[src_col], errors="coerce"
            ).fillna(0.0).clip(0, 1)
        else:
            df[src_col] = 0.0

    # Dynamic fireplace lag features (rolling max captures residual heat)
    # Column naming: integer hours → fireplace_lag_Xh; fractional → fireplace_lag_Xm
    for lag_h in fireplace_lag_hours:
        n_steps = max(1, int(round(lag_h * steps_per_hour)))
        if lag_h == int(lag_h):
            col_name = f"fireplace_lag_{int(lag_h)}h"
        else:
            col_name = f"fireplace_lag_{int(round(lag_h * 60))}m"
        df[col_name] = df["fireplace_on"].rolling(n_steps, min_periods=1).max()

    # Dynamic TV lag features
    # Column naming: integer hours → tv_lag_Xh; fractional → tv_lag_Xm
    for lag_h in tv_lag_hours:
        n_steps = max(1, int(round(lag_h * steps_per_hour)))
        if lag_h == int(lag_h):
            col_name = f"tv_lag_{int(lag_h)}h"
        else:
            col_name = f"tv_lag_{int(round(lag_h * 60))}m"
        df[col_name] = df["tv_on"].rolling(n_steps, min_periods=1).max()

    # PV features — optional (fill with 0 when no PV sensor in the dataset)
    if "PV_Generate" not in df.columns:
        df["PV_Generate"] = 0.0
    else:
        df["PV_Generate"] = pd.to_numeric(
            df["PV_Generate"], errors="coerce"
        ).fillna(0.0)

    df["pv_roll_1h"] = df["PV_Generate"].rolling(
        steps_per_hour, min_periods=1
    ).mean()
    df["pv_roll_2h"] = df["PV_Generate"].rolling(
        2 * steps_per_hour, min_periods=1
    ).mean()

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

    # AT hindcast substitution: AT_roh_Xh[t] = AT[t + X_steps]
    for h in at_forecast_hours:
        shift = h * steps_per_hour
        df[f"AT_roh_{h}h"] = df["AT"].shift(-shift)

    # PV hindcast substitution: pv_forecast_Xh[t] = PV_Generate[t + X_steps]
    for h in pv_forecast_hours:
        shift = h * steps_per_hour
        df[f"pv_forecast_{h}h"] = df["PV_Generate"].shift(-shift)
    if "pv_forecast_2h" not in df.columns:
        df["pv_forecast_2h"] = df["PV_Generate"].shift(-2 * steps_per_hour)

    # ── 4c. Physics-motivated features ─────────────────────────────────
    # Newton's law: primary driver of heat loss is T_indoor − T_outdoor
    df["heat_loss_driving_force"] = df["indoor_temp"] - df["AT"]

    # AR momentum: ΔT over 1 row (= 1 cycle ≈ 10 min, matching physics dict's
    # indoor_temp_delta_10m used at inference); calibration data is at cycle resolution
    df["delta_T_indoor_lag1"] = df["indoor_temp"].diff(1).fillna(0.0)

    # Actual heat output in W: flow (L/min → L/s) × ΔT × c_p
    if "flow_rate" in df.columns:
        specific_heat_j_per_kgk = specific_heat * 1000.0
        df["Q_wp"] = np.where(
            df["flow_rate"] > 0,
            (df["flow_rate"] / 60.0)
            * (df["VLT"] - df["RLT"])
            * specific_heat_j_per_kgk,
            0.0,
        )
    else:
        df["Q_wp"] = 0.0

    # Passive solar gain proxy: PV power × cos(hour) encodes sun position/angle
    df["solar_thermal_proxy"] = df["PV_Generate"] * df["hour_cos"]

    # Anticipatory solar: upcoming PV gain minus current (slab time constant ~60–90 min)
    if "pv_forecast_2h" in df.columns:
        df["pv_forecast_delta"] = (
            df["pv_forecast_2h"] - df["PV_Generate"]
        ).fillna(0.0)
    else:
        df["pv_forecast_delta"] = 0.0

    # ── 4d. New physics interaction features ────────────────────────────

    # Continuous shading proxy: indoor_temp > 23°C × PV intensity → solar overheat
    # protection active (shutters closed, Übergangszeit); units: K × kW
    df["shading_proxy"] = (
        (df["indoor_temp"] - 23.0).clip(lower=0.0) * (df["PV_Generate"] / 1000.0)
    ).fillna(0.0)

    # Wind × temperature-difference interaction: approximates convective heat loss
    # term U_eff ≈ U_base + k×wind; linear wind_speed alone has no predictive value
    df["heat_loss_interaction"] = (
        (df["indoor_temp"] - df["AT"]) * df["wind_speed"]
    ).fillna(0.0)

    # ── 5. Regression label construction ───────────────────────────────
    # Estimate S_H from calibrated thermal parameters
    eta, u_loss, tau_room = _read_baseline_thermal_params(config)
    s_h = _compute_s_h(eta, u_loss, tau_room, float(label_horizon_h))
    if s_h < 0.05:
        # Fallback: use S_H computed from config defaults
        _s_h_degenerate = s_h  # save the degenerate value for logging
        eta_fb = float(getattr(config, "OUTLET_EFFECTIVENESS", 0.830))
        u_fb = float(getattr(config, "HEAT_LOSS_COEFFICIENT", 0.124))
        tau_fb = float(getattr(config, "THERMAL_TIME_CONSTANT", 4.39))
        s_h = _compute_s_h(eta_fb, u_fb, tau_fb, float(label_horizon_h))
        logger.warning(
            "S_H from persisted params is %.4f (degenerate); "
            "fell back to config defaults → S_H=%.4f",
            _s_h_degenerate, s_h,
        )
    logger.info(
        "S_H estimate: η=%.4f U=%.4f τ=%.2fh H=%dh → S_H=%.4f",
        eta, u_loss, tau_room, label_horizon_h, s_h,
    )

    # label[t] = −(T_indoor[t + N_steps] − T_target) / S_H
    # Positive label means outlet should have been raised (undershoot),
    # negative label means outlet should have been lowered (overshoot).
    future_indoor = df["indoor_temp"].shift(-label_horizon_steps)
    raw_label = -(future_indoor - heating_target_c) / s_h

    # Rows with trivially small margin contribute a label ≈ 0; use 0 directly
    trivial_mask = df["indoor_margin"].abs() <= 0.05
    raw_label = raw_label.where(~trivial_mask, other=0.0)

    # Clip label to ±5 °C: extreme values indicate DHW, sensor glitch, etc.
    raw_label = raw_label.clip(-5.0, 5.0)
    df["label"] = raw_label

    # ── 6. Feature set ──────────────────────────────────────────────────
    feature_cols = [
        "indoor_temp",
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
        "thermal_power_kw",
        "fireplace_on",
        "tv_on",
    ]

    # Dynamic fireplace lag columns
    for lag_h in fireplace_lag_hours:
        if lag_h == int(lag_h):
            feature_cols.append(f"fireplace_lag_{int(lag_h)}h")
        else:
            feature_cols.append(f"fireplace_lag_{int(round(lag_h * 60))}m")

    # Dynamic TV lag columns
    for lag_h in tv_lag_hours:
        if lag_h == int(lag_h):
            feature_cols.append(f"tv_lag_{int(lag_h)}h")
        else:
            feature_cols.append(f"tv_lag_{int(round(lag_h * 60))}m")

    # PV features (optional — present only when PV data was fetched)
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

    # NEW: 8 additional ML correction features
    feature_cols += [
        "wind_speed",               # PI=0.0000 — linear term; absorbed by heat_loss_interaction
        "indoor_temp_gradient",     # PI=0.0000
        "living_room_temp",
        "is_hp_active",
        "is_weekend",
        "thermal_power_rolling_1h",
        "indoor_margin_rate",       # PI=0.0000
        "is_overshoot",             # PI=0.0000
    ]

    # Slab thermal state features
    feature_cols += [
        "d_inlet_temp_60min",  # PI=0.0000 — ΔT_rl over 60 min: thermal loading trend of the floor slab
        "is_equilibrium",      # PI=0.0000 — 1.0 when |ΔT_rl| < 0.3 K → system in thermal steady state
    ]

    # Physics-motivated features
    feature_cols += [
        "heat_loss_driving_force",  # T_indoor − T_outdoor: Newton heat loss driving force
        "delta_T_indoor_lag1",      # PI=0.0000 — ΔT indoor over 1 cycle: autoregressive momentum
        "Q_wp",                     # PI=0.0000 — Actual heat output in W: flow × ΔT × c_p
        "solar_thermal_proxy",      # PI=-0.0000 — PV × cos(hour): passive solar gain proxy
        "pv_forecast_delta",        # pv_forecast_2h − pv_now: anticipatory solar signal
    ]

    # New physics interaction features (appended after all prior features)
    feature_cols += [
        "shading_proxy",          # max(0, T_indoor−23) × PV/1000: continuous solar shading proxy (K×kW)
        "heat_loss_interaction",  # (T_indoor − AT) × wind_speed: convective heat loss interaction
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
    feature_cols = available_features  # update to what was actually available

    logger.info(
        "Training set: %d rows, %d features, label mean=%.3f std=%.3f",
        len(df_train),
        len(feature_cols),
        df_train["label"].mean(),
        df_train["label"].std(),
    )

    min_samples = int(getattr(config, "HEATING_ML_MIN_TRAINING_SAMPLES", 200))
    if len(df_train) < min_samples:
        logger.error(
            "Only %d training samples (need %d). "
            "Increase lookback_hours or adjust cold_threshold.",
            len(df_train), min_samples,
        )
        return False

    # ── 8. Train / val split (temporal) ────────────────────────────────
    val_fraction = float(getattr(config, "HEATING_ML_RETRAIN_VAL_FRACTION", 0.25))
    n_val = max(1, int(len(df_train) * val_fraction))
    df_val = df_train.iloc[-n_val:].copy()
    df_fit = df_train.iloc[:-n_val].copy()

    X_fit = df_fit[feature_cols].astype(float)
    y_fit = df_fit["label"].values
    X_val = df_val[feature_cols].astype(float)
    y_val = df_val["label"].values

    # ── 9. Train LightGBM regressor ─────────────────────────────────────
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError:
        logger.error("lightgbm not installed — cannot train model")
        return False

    reg_alpha = float(getattr(config, "HEATING_ML_REG_ALPHA", 0.1))
    reg_lambda = float(getattr(config, "HEATING_ML_REG_LAMBDA", 1.0))

    lgb_params = {
        "objective": "regression_l1",  # MAE — robust to outliers
        "metric": "mae",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # ── 9b. Optional: Optuna hyper-parameter optimisation ───────────────
    optuna_enabled = getattr(config, "HEATING_ML_OPTUNA_ENABLED", False)
    if optuna_enabled:
        try:
            import optuna  # type: ignore
            from sklearn.model_selection import TimeSeriesSplit  # type: ignore

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            n_trials = int(getattr(config, "HEATING_ML_OPTUNA_N_TRIALS", 20))

            def _optuna_objective(trial: "optuna.Trial") -> float:
                params = {
                    "objective": "regression_l1",
                    "metric": "mae",
                    "n_estimators": 300,
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.1, log=True
                    ),
                    "max_depth": trial.suggest_int("max_depth", 4, 8),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                    "min_child_samples": trial.suggest_int(
                        "min_child_samples", 10, 50
                    ),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": -1,
                }
                # Use fit split only to avoid leaking holdout-period data into HPO.
                X_all = df_fit[feature_cols].astype(float)
                y_all = df_fit["label"].astype(float)
                max_splits = max(2, min(3, len(X_all) - 1))
                if len(X_all) < 3:
                    return float("inf")
                tscv = TimeSeriesSplit(n_splits=max_splits)
                maes = []
                for tr_idx, va_idx in tscv.split(X_all):
                    X_tr = X_all.iloc[tr_idx]
                    y_tr = y_all.iloc[tr_idx].values
                    X_va = X_all.iloc[va_idx]
                    y_va = y_all.iloc[va_idx].values
                    m = lgb.LGBMRegressor(**params)
                    m.fit(
                        X_tr, y_tr,
                        eval_set=[(X_va, y_va)],
                        callbacks=[
                            lgb.early_stopping(20, verbose=False),
                        ],
                    )
                    preds = m.predict(X_va)
                    maes.append(float(np.mean(np.abs(preds - y_va))))
                return float(np.mean(maes))

            study = optuna.create_study(direction="minimize")
            study.optimize(_optuna_objective, n_trials=n_trials, show_progress_bar=False)

            best = study.best_params
            lgb_params.update(best)
            logger.info(
                "=== OPTUNA BEST PARAMS (n_trials=%d, best_mae=%.4f) ===",
                n_trials, study.best_value,
            )
            for k, v in best.items():
                logger.info("  %-25s  %s", k, v)
        except ImportError:
            logger.warning("optuna not installed — using default hyper-parameters")
        except Exception as exc:
            logger.warning("Optuna optimisation failed: %s — using defaults", exc)

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(20, verbose=False),
            lgb.log_evaluation(50),
        ],
    )

    # ── 10. Validation metrics ──────────────────────────────────────────
    y_pred = model.predict(X_val)
    val_mae = float(np.mean(np.abs(y_pred - y_val)))
    ss_res = float(np.sum((y_val - y_pred) ** 2))
    ss_tot = float(np.sum((y_val - float(np.mean(y_val))) ** 2))
    val_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    logger.info("Val MAE=%.4f°C, Val R²=%.4f", val_mae, val_r2)

    # ── 10a. Optional: Time-series cross-validation metrics ─────────────
    cv_enabled = getattr(config, "HEATING_ML_CV_ENABLED", False)
    if cv_enabled:
        try:
            from sklearn.model_selection import TimeSeriesSplit  # type: ignore
            n_splits = int(getattr(config, "HEATING_ML_CV_N_SPLITS", 3))
            # Compute CV diagnostics on fit split only; keep holdout untouched.
            X_all = df_fit[feature_cols].astype(float)
            y_all = df_fit["label"].astype(float)
            n_splits = min(n_splits, len(X_all) - 1)
            if n_splits < 2:
                logger.warning(
                    "Time-series CV skipped: need at least 3 fit samples, got %d",
                    len(X_all),
                )
            else:
                tscv = TimeSeriesSplit(n_splits=n_splits)
                cv_maes, cv_r2s = [], []
                for tr_idx, va_idx in tscv.split(X_all):
                    X_tr = X_all.iloc[tr_idx]
                    y_tr = y_all.iloc[tr_idx].values
                    X_va = X_all.iloc[va_idx]
                    y_va = y_all.iloc[va_idx].values
                    m = lgb.LGBMRegressor(**lgb_params)
                    m.fit(
                        X_tr, y_tr,
                        eval_set=[(X_va, y_va)],
                        callbacks=[lgb.early_stopping(20, verbose=False)],
                    )
                    preds = m.predict(X_va)
                    fold_mae = float(np.mean(np.abs(preds - y_va)))
                    fold_ss_res = float(np.sum((y_va - preds) ** 2))
                    fold_ss_tot = float(
                        np.sum((y_va - float(np.mean(y_va))) ** 2)
                    )
                    fold_r2 = (
                        1.0 - fold_ss_res / fold_ss_tot
                        if fold_ss_tot > 1e-12 else 0.0
                    )
                    cv_maes.append(fold_mae)
                    cv_r2s.append(fold_r2)
                logger.info(
                    "=== TIME-SERIES CV (%d-fold) ===  "
                    "MAE=%.4f±%.4f  R²=%.4f±%.4f",
                    n_splits,
                    float(np.mean(cv_maes)), float(np.std(cv_maes)),
                    float(np.mean(cv_r2s)), float(np.std(cv_r2s)),
                )
        except ImportError:
            logger.warning("sklearn not available — skipping time-series CV")
        except Exception as exc:
            logger.warning("Time-series CV failed: %s", exc)

    # ── 10b. Feature importance logging ─────────────────────────────────
    # LightGBM built-in feature importance (split-based)
    importances = model.feature_importances_
    feat_imp_pairs = sorted(
        zip(feature_cols, importances), key=lambda x: x[1], reverse=True
    )
    logger.info("=== FEATURE IMPORTANCE (LightGBM split-based) ===")
    for fname, imp in feat_imp_pairs:
        logger.info("  %-30s  %6d", fname, imp)

    # Permutation importance (optional — needs sklearn)
    perm_pairs = None
    try:
        from sklearn.inspection import permutation_importance  # type: ignore
        perm_result = permutation_importance(
            model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=1,
            scoring="neg_mean_absolute_error",
        )
        perm_pairs = sorted(
            zip(feature_cols, perm_result.importances_mean),
            key=lambda x: x[1], reverse=True,
        )
        logger.info("=== PERMUTATION IMPORTANCE (val set, 10 repeats) ===")
        for fname, imp in perm_pairs:
            logger.info("  %-30s  %.4f", fname, imp)
    except ImportError:
        logger.info("sklearn not available — skipping permutation importance")
    except Exception as exc:
        logger.warning("Permutation importance failed: %s", exc)

    # ── 10c. Feature pruning (permutation importance-based) ─────────────
    pruning_enabled = getattr(config, "HEATING_ML_FEATURE_PRUNING_ENABLED", True)
    pi_threshold = float(getattr(config, "HEATING_ML_PRUNE_PI_THRESHOLD", 0.0))

    if pruning_enabled and perm_pairs is not None:
        pruned_features = [
            fname for fname, imp in perm_pairs if imp > pi_threshold
        ]
        dropped = [
            fname for fname, imp in perm_pairs if imp <= pi_threshold
        ]
        # Need at least 5 features after pruning
        if dropped and len(pruned_features) >= 5:
            logger.info(
                "=== FEATURE PRUNING: dropping %d features with PI ≤ %.4f ===",
                len(dropped), pi_threshold,
            )
            for fname in dropped:
                pi_val = next(imp for f, imp in perm_pairs if f == fname)
                logger.info("  ✂️  %-30s  PI=%.4f", fname, pi_val)

            # Retrain on pruned feature set
            X_fit_pruned = df_fit[pruned_features].astype(float)
            X_val_pruned = df_val[pruned_features].astype(float)

            model_pruned = lgb.LGBMRegressor(**lgb_params)
            model_pruned.fit(
                X_fit_pruned, y_fit,
                eval_set=[(X_val_pruned, y_val)],
                callbacks=[
                    lgb.early_stopping(20, verbose=False),
                    lgb.log_evaluation(50),
                ],
            )
            y_pred_pruned = model_pruned.predict(X_val_pruned)
            pruned_mae = float(np.mean(np.abs(y_pred_pruned - y_val)))
            pruned_ss_res = float(np.sum((y_val - y_pred_pruned) ** 2))
            pruned_r2 = (
                1.0 - pruned_ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            )

            # Accept if MAE did not regress beyond 0.5%
            mae_regression_pct = (
                (pruned_mae - val_mae) / val_mae * 100.0
                if val_mae > 1e-12 else 0.0
            )
            logger.info(
                "=== PRUNED vs UNPRUNED: MAE %.4f→%.4f (%+.2f%%), "
                "R² %.4f→%.4f, features %d→%d ===",
                val_mae, pruned_mae, mae_regression_pct,
                val_r2, pruned_r2,
                len(feature_cols), len(pruned_features),
            )

            if mae_regression_pct <= 0.5:
                logger.info(
                    "✅ Pruned model accepted (MAE regression %.2f%% ≤ 0.5%%)",
                    mae_regression_pct,
                )
                model = model_pruned
                feature_cols = pruned_features
                val_mae = pruned_mae
                val_r2 = pruned_r2
                # Recompute feature importance for metadata
                importances = model.feature_importances_
                feat_imp_pairs = sorted(
                    zip(feature_cols, importances),
                    key=lambda x: x[1], reverse=True,
                )
            else:
                logger.info(
                    "⚠️ Pruned model rejected (MAE regression %.2f%% > 0.5%%) "
                    "— keeping unpruned model",
                    mae_regression_pct,
                )
        elif dropped:
            logger.info(
                "⚠️ Feature pruning skipped: only %d features would remain "
                "(need ≥ 5)",
                len(pruned_features),
            )

    # ── 11. Save model + metadata ───────────────────────────────────────
    try:
        import joblib  # type: ignore
    except ImportError:
        logger.error("joblib not installed — cannot save model")
        return False

    model_path = getattr(
        config,
        "HEATING_ML_CORRECTION_MODEL_PATH",
        "/opt/ml_heating/heating_correction_ml_model.joblib",
    )
    metadata_path = getattr(
        config,
        "HEATING_ML_CORRECTION_METADATA_PATH",
        "/opt/ml_heating/heating_correction_ml_metadata.json",
    )

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    tmp_model = model_path + ".tmp"
    joblib.dump(model, tmp_model)
    os.replace(tmp_model, model_path)

    metadata = {
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "val_mae": val_mae,
        "val_r2": val_r2,
        "n_train": int(len(df_fit)),
        "n_val": int(len(df_val)),
        "label_horizon_h": label_horizon_h,
        "steps_per_hour": steps_per_hour,
        "cold_threshold_c": cold_threshold,
        "heating_target_c": heating_target_c,
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
        "feature_importances": {
            fname: int(imp) for fname, imp in feat_imp_pairs
        },
    }
    tmp_meta = metadata_path + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=_json_default)
    os.replace(tmp_meta, metadata_path)

    logger.info(
        "=== HEATING CORRECTION ML CALIBRATION COMPLETE: "
        "model → %s | MAE=%.4f R²=%.4f ===",
        model_path, val_mae, val_r2,
    )

    # ── 12. Export training data for offline HPO / analysis ──────────────
    # Use all columns present in df_train (not just the pruned feature_cols)
    # so offline notebooks can experiment with different pruning thresholds.
    all_available_cols = [c for c in df_train.columns if c != "label"]
    export_training_data(df_train, all_available_cols, "heating")

    return True
