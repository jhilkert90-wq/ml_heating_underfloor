"""
cooling_ml_calibration.py
--------------------------
One-shot training of the LightGBM overheating pre-cooling classifier.

Called via
  ``python -m src.main --calibrate-cooling-ml``
or triggered by the dashboard "Calibrate ML Cooling Model" button (flag file).

Pipeline
--------
1. Fetch multi-month historical data (same helper as physics calibration).
2. Rename entity-ID columns to model-friendly names.
3. Filter to warm-season rows (AT > threshold) where overheating is possible.
4. Compute derived features: thermal_power_kw, delta_t, rolling PV, etc.
5. Hindcast substitution: shift actual AT / PV forward N hours to stand in
   for forecasts (``AT_roh_4h = outdoor_temp shifted back 4 h``).
   All 12 AT forecast hours and all 12 PV forecast hours are included by
   default; controlled by ``COOLING_ML_AT_FORECAST_HOURS`` and
   ``COOLING_ML_PV_FORECAST_HOURS`` env vars (comma-separated lists).
6. Compute rolling-max label: did indoor temp exceed cooling_target within
   PRE_COOL_HORIZON_HOURS?
7. Train LightGBM with class weighting; tune decision threshold on val split.
8. Save model (joblib) + metadata JSON.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Indoor temperature threshold above which solar overheat protection (roller
# shutters / Jalousie) is assumed to be active – matches heating calibration.
_SHADING_ACTIVATION_TEMP_C = 23.0

# ── Calibration hyper-parameters (named constants) ─────────────────────────
# Max allowed AUC drop before preferring raw model over isotonic-calibrated.
_ISOTONIC_AUC_TOLERANCE: float = 0.01
# Minimum samples per fold in threshold cross-validation; below this CV is
# skipped because fold estimates are unreliable with so few samples.
_CV_MIN_FOLD_SIZE: int = 50


class _IsotonicCalibratedModel:
    """Wrap a frozen base classifier with a pre-fitted IsotonicRegression.

    Used as a drop-in replacement for ``CalibratedClassifierCV(cv="prefit")``
    on sklearn versions that removed that argument.  The base model is never
    re-fitted — only the isotonic mapping on top of its probabilities is
    trained, so the deployed model is identical to the one trained on the full
    training split.
    """

    def __init__(self, base_model: Any, isotonic: Any) -> None:
        self._base = base_model
        self._iso = isotonic

    def predict_proba(self, X: Any) -> Any:
        import numpy as np  # type: ignore
        raw = self._base.predict_proba(X)[:, 1]
        cal = self._iso.predict(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: Any) -> Any:
        import numpy as np  # type: ignore
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _resolve_current_cooling_target(config_module: Any, fallback_target_c: float) -> float:
    """Read the current cooling target from Home Assistant, else return fallback."""
    target_entity_id = str(
        getattr(config_module, "TARGET_INDOOR_TEMP_COOLING_ENTITY_ID", "") or ""
    ).strip()
    if not target_entity_id:
        logger.info(
            "Cooling target entity not configured; using fallback cooling_target=%.1f°C",
            fallback_target_c,
        )
        return fallback_target_c

    try:
        try:
            from .ha_client import create_ha_client
        except ImportError:
            from ha_client import create_ha_client  # type: ignore

        ha_client = create_ha_client()
        all_states = ha_client.get_all_states()
        current_target = ha_client.get_state(target_entity_id, all_states)
        if current_target is None:
            raise ValueError("state unavailable")
        resolved_target = float(current_target)
        logger.info(
            "Resolved cooling_target=%.1f°C from HA entity %s",
            resolved_target,
            target_entity_id,
        )
        return resolved_target
    except Exception as exc:
        logger.warning(
            "Failed to resolve cooling target from %s (%s); using fallback %.1f°C",
            target_entity_id,
            exc,
            fallback_target_c,
        )
        return fallback_target_c


def _has_both_classes(y: Any) -> bool:
    """Return True when the label array contains both 0 and 1."""
    import numpy as np  # type: ignore

    try:
        y_arr = np.asarray(y, dtype=int)
    except Exception:
        y_arr = np.asarray(y)
    return len(np.unique(y_arr)) >= 2


def _select_temporal_train_val_split(
    df_train: Any,
    label_col: str,
    val_fraction: float,
) -> tuple[Any, Any]:
    """Choose the closest temporal split that keeps both classes in training."""
    preferred_n_val = max(1, int(len(df_train) * val_fraction))
    preferred_split_idx = len(df_train) - preferred_n_val
    best_split_idx = None
    best_score = None

    for split_idx in range(1, len(df_train)):
        y_fit = df_train[label_col].iloc[:split_idx].values
        y_val = df_train[label_col].iloc[split_idx:].values
        if len(y_val) == 0 or not _has_both_classes(y_fit):
            continue

        score = abs(split_idx - preferred_split_idx)
        if not _has_both_classes(y_val):
            score += len(df_train)

        if best_score is None or score < best_score:
            best_score = score
            best_split_idx = split_idx

    if best_split_idx is None:
        raise ValueError("no temporal split keeps both classes in training")

    return (
        df_train.iloc[:best_split_idx].copy(),
        df_train.iloc[best_split_idx:].copy(),
    )


# Module-level import of shared training-data export helper.
try:
    from .calibration_data_export import export_training_data
except ImportError:
    try:
        from calibration_data_export import export_training_data  # type: ignore
    except ImportError:
        def export_training_data(*args, **kwargs):  # type: ignore
            return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def calibrate_cooling_ml(
    state_manager=None,
    lookback_hours: int = 2160,  # 90 days
    cooling_target_c: Optional[float] = None,
) -> bool:
    """
    Train and persist the LightGBM cooling classifier.

    Physics-derived trajectory features use actively-learned heat pump channel
    parameters (η, U, τ) from the cooling thermal state, not baseline calibration,
    ensuring ML training reflects actual system behavior during the training period.

    Parameters
    ----------
    state_manager:
        Unused (signature compatibility with physics calibration).
    lookback_hours:
        Hours of historical data to fetch.
    cooling_target_c:
        Indoor temperature threshold above which overheating is declared.
        Defaults to ``COOLING_CLAMP_MAX_ABS - 1.0`` (e.g. 24 - 1 = 23°C).

    Returns
    -------
    bool: True on success, False on failure.

    Raises
    ------
    RuntimeError:
        If cooling heat pump channel parameters are not initialized.
        Run ``calibrate_cooling_physics()`` first.
    """
    try:
        from . import config
    except ImportError:
        import config  # type: ignore
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        logger.error("calibrate_cooling_ml: missing dependency — %s", exc)
        return False

    logger.info("=== COOLING ML CALIBRATION START ===")

    # ── 0. Parameters ──────────────────────────────────────────────────
    # Resolve lookback_hours from COOLING_ML_CALIBRATION_START_DATE when set,
    # so the caller does not need to be changed.  The explicit lookback_hours
    # argument (default 2160 h = 90 days) is used as the fallback.
    _start_date_str = getattr(config, "COOLING_ML_CALIBRATION_START_DATE", "")
    if _start_date_str and _start_date_str.strip():
        _parse_fn = getattr(config, "_parse_cooling_start_date", None)
        _start_dt = _parse_fn(_start_date_str) if callable(_parse_fn) else None
        if _start_dt is not None:
            from datetime import timezone as _dt_tz
            _now_utc = datetime.now(_dt_tz.utc)
            _computed_h = math.ceil((_now_utc - _start_dt).total_seconds() / 3600)
            if _computed_h > 0:
                lookback_hours = _computed_h
                logger.info(
                    "Resolved lookback_hours=%d from start date '%s'",
                    lookback_hours, _start_date_str,
                )
            else:
                logger.warning(
                    "COOLING_ML_CALIBRATION_START_DATE '%s' is in the future; "
                    "using default lookback_hours=%d",
                    _start_date_str, lookback_hours,
                )
        else:
            logger.warning(
                "COOLING_ML_CALIBRATION_START_DATE '%s' is not a valid DD.MM.YYYY date; "
                "using default lookback_hours=%d",
                _start_date_str, lookback_hours,
            )

    steps_per_hour = round(60 / float(getattr(config, "CYCLE_INTERVAL_MINUTES", 10)))
    horizon_h = int(getattr(config, "PRE_COOL_HORIZON_HOURS", 12))

    # The LABEL window uses the lead-time horizon, not the full trajectory
    # horizon.  This means label=1 ↔ "overheating within lead_time hours",
    # so that a positive model prediction directly justifies acting NOW
    # (same semantics as the trajectory predictor's lead_time gate).
    lead_time_h = float(getattr(config, "PRE_COOL_LEAD_TIME_HOURS", 3.0))
    label_horizon_h = int(round(lead_time_h))
    label_horizon_steps = label_horizon_h * steps_per_hour
    # The full horizon is still used to generate hindcast forecast features.
    forecast_horizon_steps = horizon_h * steps_per_hour

    if cooling_target_c is None:
        clamp_max = float(getattr(config, "COOLING_CLAMP_MAX_ABS", 24.0))
        cooling_target_c = _resolve_current_cooling_target(config, clamp_max - 1.0)
    logger.info(
        "Calibration params: label_horizon=%dh (%d steps), "
        "forecast_horizon=%dh, cooling_target=%.1f°C, "
        "steps_per_hour=%d, lookback=%dh",
        label_horizon_h, label_horizon_steps,
        horizon_h, cooling_target_c, steps_per_hour, lookback_hours,
    )

    # ── 1. Fetch historical data ────────────────────────────────────────
    try:
        from physics_calibration import fetch_historical_data_for_calibration  # type: ignore
    except ImportError:
        try:
            from src.physics_calibration import fetch_historical_data_for_calibration  # type: ignore
        except ImportError:
            logger.error("Cannot import fetch_historical_data_for_calibration")
            return False

    df = fetch_historical_data_for_calibration(
        lookback_hours=lookback_hours,
        purpose="cooling",
    )
    if df is None or df.empty:
        logger.error("Calibration aborted: no historical data fetched")
        return False

    logger.info("Fetched %d rows of historical data", len(df))

    # ── 2. Rename entity-ID columns → model-friendly names ─────────────
    indoor_col  = getattr(config, "INDOOR_TEMP_ENTITY_ID",  "sensor.rt_mittelwert").split(".", 1)[-1]
    outdoor_col = getattr(config, "OUTDOOR_TEMP_ENTITY_ID", "sensor.nibe_bt1_outdoor_temperature").split(".", 1)[-1]
    outlet_col  = getattr(config, "OUTLET_TEMP_ENTITY_ID",  "sensor.nibe_bt2_supply_temp_s1").split(".", 1)[-1]
    inlet_col   = getattr(config, "INLET_TEMP_ENTITY_ID",   "sensor.nibe_eb100_ep14_bt3_return_temp").split(".", 1)[-1]
    flow_col    = getattr(config, "FLOW_RATE_ENTITY_ID",    "input_number.hp_current_flow_rate").split(".", 1)[-1]
    power_col   = getattr(config, "POWER_CONSUMPTION_ENTITY_ID", "sensor.nibe_el_leistung").split(".", 1)[-1]
    pv_col      = getattr(config, "PV_POWER_ENTITY_ID",     "sensor.pv_leistung_gefiltert").split(".", 1)[-1]
    fireplace_col = getattr(config, "FIREPLACE_STATUS_ENTITY_ID", "binary_sensor.fireplace_active").split(".", 1)[-1]
    tv_col      = getattr(config, "TV_STATUS_ENTITY_ID",    "input_boolean.fernseher").split(".", 1)[-1]
    living_room_col = getattr(config, "LIVING_ROOM_TEMP_ENTITY_ID", "sensor.living_room_temperature").split(".", 1)[-1]
    wind_col    = getattr(config, "WIND_SPEED_ENTITY_ID",   "sensor.wind_speed").split(".", 1)[-1]

    rename_map = {
        indoor_col:  "indoor_temp",
        outdoor_col: "AT",
        outlet_col:  "VLT",
        inlet_col:   "RLT",
        flow_col:    "flow_rate",
        power_col:   "power_w",
        pv_col:      "PV_Generate",
        fireplace_col: "fireplace_on",
        tv_col:      "tv_on",
        living_room_col: "living_room_temp",
        wind_col:    "wind_speed",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["indoor_temp", "AT", "VLT", "RLT", "PV_Generate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Calibration aborted: missing required columns: %s", missing)
        return False

    # Sort by time
    if "_time" in df.columns:
        df = df.sort_values("_time").reset_index(drop=True)

    # ── 3. Numeric coercion & warm-season filter ────────────────────────
    for col in ["indoor_temp", "AT", "VLT", "RLT", "PV_Generate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    warm_threshold = float(getattr(config, "COOLING_ML_WARM_THRESHOLD_C", 10.0))
    df = df[df["AT"] > warm_threshold].copy()
    logger.info("After warm-season filter (AT > %.1f°C): %d rows", warm_threshold, len(df))

    if len(df) < 500:
        logger.error(
            "Only %d warm-season rows available — need at least 500. "
            "Increase lookback_hours or ensure summer data is present.",
            len(df),
        )
        return False

    df = df.reset_index(drop=True)

    # ── 4. Derived features ─────────────────────────────────────────────
    specific_heat = float(getattr(config, "SPECIFIC_HEAT_CAPACITY", 4.186))

    # delta_t and thermal_power_kw
    df["delta_t"] = pd.to_numeric(df["VLT"], errors="coerce") - pd.to_numeric(df["RLT"], errors="coerce")
    if "flow_rate" in df.columns:
        df["flow_rate"] = pd.to_numeric(df["flow_rate"], errors="coerce").fillna(0.0)
        # kW = (L/min × kg/L × kJ/kg·K × K) / 60 s
        df["thermal_power_kw"] = df["flow_rate"] * specific_heat * df["delta_t"] / 60.0
    elif "power_w" in df.columns:
        df["power_w"] = pd.to_numeric(df["power_w"], errors="coerce").fillna(0.0)
        df["thermal_power_kw"] = -df["power_w"] / 1000.0  # negative = cooling
    else:
        df["thermal_power_kw"] = 0.0

    df["outlet_indoor_diff"] = df["VLT"] - df["indoor_temp"]
    df["at_delta_indoor"] = df["AT"] - df["indoor_temp"]
    df["indoor_margin"] = cooling_target_c - df["indoor_temp"]

    # Rolling indoor trends (30 min = 3 steps, 60 min = 6 steps)
    df["indoor_trend_30m"] = df["indoor_temp"].diff(3)
    df["indoor_trend_1h"]  = df["indoor_temp"].diff(steps_per_hour)

    # PV rolling means
    df["PV_Generate"] = pd.to_numeric(df["PV_Generate"], errors="coerce").fillna(0.0)
    df["pv_roll_1h"]  = df["PV_Generate"].rolling(steps_per_hour, min_periods=1).mean()
    df["pv_roll_2h"]  = df["PV_Generate"].rolling(2 * steps_per_hour, min_periods=1).mean()

    # ── 4b. HA context features ─────────────────────────────────────────
    # Wind speed — fill missing with 0 (calm)
    if "wind_speed" in df.columns:
        df["wind_speed"] = pd.to_numeric(
            df["wind_speed"], errors="coerce"
        ).fillna(0.0).clip(0, 200)
    else:
        df["wind_speed"] = 0.0

    # Living room temperature
    if "living_room_temp" in df.columns:
        df["living_room_temp"] = pd.to_numeric(
            df["living_room_temp"], errors="coerce"
        ).fillna(method="ffill").fillna(df["indoor_temp"])
    else:
        df["living_room_temp"] = df["indoor_temp"]

    # Fireplace and TV features (binary) — fill missing with 0
    for src_col in ["fireplace_on", "tv_on"]:
        if src_col in df.columns:
            df[src_col] = pd.to_numeric(
                df[src_col], errors="coerce"
            ).fillna(0.0).clip(0, 1)
        else:
            df[src_col] = 0.0

    # Dynamic fireplace lag features (rolling max captures residual heat)
    _fp_lag_hours = [0.5, 1.0, 2.0]
    for lag_h in _fp_lag_hours:
        n_steps = max(1, int(round(lag_h * steps_per_hour)))
        if lag_h == int(lag_h):
            col_name = f"fireplace_lag_{int(lag_h)}h"
        else:
            col_name = f"fireplace_lag_{int(round(lag_h * 60))}m"
        df[col_name] = df["fireplace_on"].rolling(n_steps, min_periods=1).max()

    # Dynamic TV lag features
    _tv_lag_hours = [0.5, 1.0]
    for lag_h in _tv_lag_hours:
        n_steps = max(1, int(round(lag_h * steps_per_hour)))
        if lag_h == int(lag_h):
            col_name = f"tv_lag_{int(lag_h)}h"
        else:
            col_name = f"tv_lag_{int(round(lag_h * 60))}m"
        df[col_name] = df["tv_on"].rolling(n_steps, min_periods=1).max()

    # ── 4c. Derived physics features ────────────────────────────────────
    # Heat loss driving force (Newton's law)
    df["heat_loss_driving_force"] = df["indoor_temp"] - df["AT"]

    # Indoor temp gradient (°C/h)
    df["indoor_temp_gradient"] = df["indoor_temp"].diff() * steps_per_hour

    # Indoor margin rate of change (°C/h)
    df["indoor_margin_rate"] = df["indoor_margin"].diff() * steps_per_hour

    # AR momentum: ΔT over 1 cycle
    df["delta_T_indoor_lag1"] = df["indoor_temp"].diff(1).fillna(0.0)

    # Slab thermal loading trend over 60 min
    df["d_inlet_temp_60min"] = df["RLT"].diff(steps_per_hour)

    # Binary flag: thermal steady state
    df["is_equilibrium"] = (df["d_inlet_temp_60min"].abs() < 0.3).astype(float)

    # Rolling 1-hour thermal power
    df["thermal_power_rolling_1h"] = df["thermal_power_kw"].rolling(
        steps_per_hour, min_periods=1
    ).mean()

    # Overshoot indicator: indoor > cooling target
    df["is_overshoot"] = (df["indoor_temp"] > cooling_target_c).astype(float)

    # Heat pump active — |delta_t| > 1°C indicates flow
    df["is_hp_active"] = (df["delta_t"].abs() > 1.0).astype(float)

    # Wind × temperature-difference interaction: convective heat loss
    df["heat_loss_interaction"] = (
        (df["indoor_temp"] - df["AT"]) * df["wind_speed"]
    ).fillna(0.0)

    logger.info("Computed HA context + derived physics features")

    # Temporal cyclical features from _time
    if "_time" in df.columns:
        ts = pd.to_datetime(df["_time"], utc=True)
        hour_frac = ts.dt.hour + ts.dt.minute / 60.0
        doy = ts.dt.dayofyear
    else:
        hour_frac = pd.Series([12.0] * len(df))
        doy = pd.Series([180] * len(df))

    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
    df["doy_sin"]  = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"]  = np.cos(2 * np.pi * doy / 365.25)

    # Weekend indicator
    if "_time" in df.columns:
        df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(float)
    else:
        df["is_weekend"] = 0.0

    # Passive solar gain proxy: PV power × cos(hour) encodes sun angle
    df["solar_thermal_proxy"] = df["PV_Generate"] * df["hour_cos"]

    # Continuous shading proxy: overheat-protection active when indoor > 23°C
    df["shading_proxy"] = (
        (df["indoor_temp"] - _SHADING_ACTIVATION_TEMP_C).clip(lower=0.0) * df["PV_Generate"]
    ).fillna(0.0)

    # ── 5. Hindcast substitution for forecast features ──────────────────
    # At calibration time we know the future; shift actual values back to
    # simulate what a perfect forecast would have provided.
    # AT_roh_4h at row t ≈ AT at row t+4h
    # Use the full forecast_horizon_steps so the model can see further ahead.
    forecast_feature_cols: list[str] = []
    for h in range(1, horizon_h + 1):
        shift = h * steps_per_hour
        at_col  = f"AT_roh_{h}h"
        pv_col2 = f"pv_forecast_{h}h"
        df[at_col]  = df["AT"].shift(-shift)
        df[pv_col2] = df["PV_Generate"].shift(-shift)
        forecast_feature_cols.extend([at_col, pv_col2])

    # ── 5b. Forecast noise injection (Prio 5) ─────────────────────────────
    # Add realistic noise to hindcast features to simulate forecast errors
    # at inference time.  This prevents the model from relying on precise
    # future values it won't have at inference.
    _noise_seed = 42
    _rng = np.random.default_rng(_noise_seed)
    for h in range(1, horizon_h + 1):
        at_col = f"AT_roh_{h}h"
        pv_col2 = f"pv_forecast_{h}h"
        if at_col in df.columns:
            # AT noise grows with horizon: std = 0.3 × sqrt(h)
            at_noise = _rng.normal(0, 0.3 * math.sqrt(h), size=len(df))
            df[at_col] = df[at_col] + at_noise
        if pv_col2 in df.columns:
            # PV multiplicative noise grows with horizon: std = 8% × sqrt(h)
            pv_noise = _rng.normal(0, 0.08 * math.sqrt(h), size=len(df))
            df[pv_col2] = df[pv_col2] * (1.0 + pv_noise)
            df[pv_col2] = df[pv_col2].clip(lower=0.0)  # PV can't be negative
    logger.info("Applied forecast noise injection (AT std=0.3√h, PV 8%%√h)")

    # Anticipatory solar: upcoming PV gain minus current (slab thermal lag)
    if "pv_forecast_2h" in df.columns:
        df["pv_forecast_delta"] = (
            df["pv_forecast_2h"] - df["PV_Generate"]
        ).fillna(0.0)
    else:
        df["pv_forecast_delta"] = 0.0

    # ── 5c. Cumulative / integration features (Prio 4) ──────────────────
    # Give the model "total energy build-up" features similar to what the
    # trajectory model computes by integrating over time.
    _cum_pv_cols = [f"pv_forecast_{h}h" for h in range(1, 5)
                    if f"pv_forecast_{h}h" in df.columns]
    if _cum_pv_cols:
        df["cum_pv_forecast_4h"] = df[_cum_pv_cols].sum(axis=1)
    else:
        df["cum_pv_forecast_4h"] = 0.0

    _cum_at_cols = [f"AT_roh_{h}h" for h in range(1, 5)
                    if f"AT_roh_{h}h" in df.columns]
    if _cum_at_cols:
        df["cum_at_excess_4h"] = (df[_cum_at_cols] - cooling_target_c).clip(lower=0).sum(axis=1)
    else:
        df["cum_at_excess_4h"] = 0.0

    _max_at_cols = [f"AT_roh_{h}h" for h in range(1, label_horizon_h + 1)
                    if f"AT_roh_{h}h" in df.columns]
    if _max_at_cols:
        df["max_at_forecast"] = df[_max_at_cols].max(axis=1)
    else:
        df["max_at_forecast"] = df["AT"]

    # indoor_momentum: linear extrapolation 3h ahead based on 1h trend
    df["indoor_momentum"] = df["indoor_trend_1h"].fillna(0.0) * 3.0

    # slab_stored_heat: thermal energy in the floor relative to room
    df["slab_stored_heat"] = (df["VLT"] + df["RLT"]) / 2.0 - df["indoor_temp"]

    logger.info("Computed cumulative/integration features")

    # ── 5d. Trajectory-derived physics features (vectorized) ────────────
    # Analytical Newton-decay approximation of what the trajectory simulator
    # would predict, computed per row using actively-learned heat pump parameters.
    # Uses heat_source_channels["heat_pump"] parameters (not baseline calibration),
    # which reflect the system state during the training period.
    _traj_eta = float(getattr(config, "OUTLET_EFFECTIVENESS", 0.830))
    _traj_u = float(getattr(config, "HEAT_LOSS_COEFFICIENT", 0.124))
    _traj_tau = float(getattr(config, "THERMAL_TIME_CONSTANT", 4.39))

    # Load actively-learned heat pump parameters from cooling thermal state
    try:
        from src.unified_thermal_state_cooling import get_cooling_state_manager
        _state_mgr = get_cooling_state_manager()
        _state_mgr.load_state()
        _hp_params = _state_mgr.state.get("learning_state", {}).get("heat_source_channels", {}).get("heat_pump", {}).get("parameters", {})
        
        if not _hp_params:
            raise RuntimeError(
                "Cooling heat pump channel not initialized. "
                "Run `python -m src.main --calibrate-cooling-physics` first."
            )
        
        # Extract and validate all required parameters
        _required_keys = ["outlet_effectiveness", "heat_loss_coefficient", "thermal_time_constant"]
        _missing_keys = [k for k in _required_keys if k not in _hp_params]
        if _missing_keys:
            raise RuntimeError(
                f"Cooling heat pump parameters incomplete; missing keys: {_missing_keys}. "
                "Run `python -m src.main --calibrate-cooling-physics` first."
            )
        
        # Extract and validate parameter values
        try:
            _traj_eta_loaded = float(_hp_params["outlet_effectiveness"])
            _traj_u_loaded = float(_hp_params["heat_loss_coefficient"])
            _traj_tau_loaded = float(_hp_params["thermal_time_constant"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Cooling heat pump parameters corrupted (non-numeric): {exc}. "
                "Run `python -m src.main --calibrate-cooling-physics` first."
            ) from exc
        
        # Validate parameter ranges (catch NaN, inf, etc.)
        if not (0 < _traj_eta_loaded < 2.0):
            raise RuntimeError(
                f"Invalid outlet_effectiveness {_traj_eta_loaded} (must be 0 < η < 2.0). "
                "Cooling state corrupted."
            )
        if not (0 < _traj_u_loaded < 1.5):
            raise RuntimeError(
                f"Invalid heat_loss_coefficient {_traj_u_loaded} (must be 0 < U < 1.5). "
                "Cooling state corrupted."
            )
        if not (0.1 <= _traj_tau_loaded < 20.0):
            raise RuntimeError(
                f"Invalid thermal_time_constant {_traj_tau_loaded} (must be 0.1 ≤ τ < 20h). "
                "Cooling state corrupted."
            )
        
        # All validations passed — use loaded values
        _traj_eta = _traj_eta_loaded
        _traj_u = _traj_u_loaded
        _traj_tau = _traj_tau_loaded
        
        logger.info(
            "Loaded cooling trajectory parameters from heat_pump channel "
            "(outlet_effectiveness=%.4f, heat_loss_coefficient=%.4f, thermal_time_constant=%.2f)",
            _traj_eta, _traj_u, _traj_tau
        )
    except RuntimeError:
        raise  # Re-raise initialization errors with context
    except Exception as exc:
        # Any other exception (ImportError, KeyError, IOError, etc.) is treated as initialization error
        raise RuntimeError(
            f"Failed to load cooling heat pump parameters: {exc}. "
            "Run `python -m src.main --calibrate-cooling-physics` first."
        ) from exc

    # Sanity guard: τ < 0.1 h should not happen after validation, but check anyway
    if _traj_tau < 0.1:
        logger.error(
            "INTERNAL ERROR: Thermal time constant %.4f passed validation but is < 0.1h. "
            "This indicates corrupted state or validation logic error.",
            _traj_tau
        )
        raise RuntimeError("Invalid thermal time constant passed validation — corrupted state")
    
    # Guard: sum of η + U should be reasonable (shouldn't happen after validation)
    if (_traj_eta + _traj_u) < 1e-6:
        logger.error(
            "INTERNAL ERROR: η+U sum %.2e is too small. This indicates validation logic error.",
            _traj_eta + _traj_u
        )
        raise RuntimeError("Invalid parameter sum passed validation — corrupted state")

    # Equilibrium temperature: T_eq = (η×VLT + U×AT) / (η + U)
    _traj_denom = _traj_eta + _traj_u
    df["_traj_T_eq"] = (
        _traj_eta * df["VLT"] + _traj_u * df["AT"]
    ) / _traj_denom

    # Prediction horizon matches label horizon
    _traj_H = float(label_horizon_h)
    _traj_steps = int(_traj_H * steps_per_hour)

    # Vectorized trajectory: T(t) = T_eq + (T_indoor - T_eq) × exp(-t/τ)
    _step_hours = 1.0 / steps_per_hour
    _exp_first = np.exp(-_step_hours / _traj_tau)
    df["_traj_step_1"] = df["_traj_T_eq"] + (df["indoor_temp"] - df["_traj_T_eq"]) * _exp_first

    _exp_last = np.exp(-_traj_H / _traj_tau)
    df["_traj_step_last"] = df["_traj_T_eq"] + (df["indoor_temp"] - df["_traj_T_eq"]) * _exp_last

    # Feature 1: traj_predicted_error — physics predicted miss of cooling target
    df["traj_predicted_error"] = (df["_traj_step_last"] - cooling_target_c).fillna(0.0)

    # Feature 2: traj_convergence_rate — speed of approach to equilibrium
    df["traj_convergence_rate"] = (
        (df["_traj_step_1"] - df["_traj_step_last"]) / max(1, _traj_steps)
    ).fillna(0.0)

    # Feature 3: traj_reaches_target_hours — analytical time to reach cooling target
    _ratio = (cooling_target_c - df["_traj_T_eq"]) / (df["indoor_temp"] - df["_traj_T_eq"])
    _valid_mask = (_ratio > 0) & (_ratio < 1)
    _reaches_raw = -_traj_tau * np.log(_ratio.where(_valid_mask))
    df["traj_reaches_target_hours"] = _reaches_raw.clip(lower=0.0, upper=_traj_H).fillna(_traj_H)

    # Feature 4: traj_overshoot_magnitude — predicted overshoot above cooling target
    _traj_max = df[["indoor_temp", "_traj_T_eq"]].max(axis=1)
    df["traj_overshoot_magnitude"] = (_traj_max - cooling_target_c).clip(lower=0.0).fillna(0.0)

    # Feature 5: traj_equilibrium_gap — steady-state error (positive = above target = risk)
    df["traj_equilibrium_gap"] = (df["_traj_T_eq"] - cooling_target_c).fillna(0.0)

    # Cleanup temporary columns
    df.drop(columns=["_traj_T_eq", "_traj_step_1", "_traj_step_last"], inplace=True)

    logger.info(
        "Computed trajectory-derived physics features "
        "(η=%.3f U=%.3f τ=%.2fh H=%.0fh)",
        _traj_eta, _traj_u, _traj_tau, _traj_H,
    )

    # ── 6. Define feature set ────────────────────────────────────────────
    # The exact columns are saved to metadata; inference reads them back.
    # AT hindcast hours: controlled by COOLING_ML_AT_FORECAST_HOURS
    # (legacy alias: COOLING_ML_FORECAST_HOURS).  Defaults to all 12 hours.
    # PV hindcast hours: controlled by COOLING_ML_PV_FORECAST_HOURS.
    # Defaults to all 12 hours.
    # The coverage guard in Step 8 silently drops any column whose data
    # coverage is below 5% (e.g. the last N rows which have no future window).
    _default_fc_hours = ",".join(str(h) for h in range(1, horizon_h + 1))
    _at_fc_env = os.getenv(
        "COOLING_ML_AT_FORECAST_HOURS",
        os.getenv("COOLING_ML_FORECAST_HOURS", _default_fc_hours),
    )
    try:
        at_forecast_hours = [int(x) for x in _at_fc_env.split(",") if x.strip()]
    except ValueError:
        logger.warning(
            "COOLING_ML_AT_FORECAST_HOURS value %r is invalid; "
            "falling back to all %d hours",
            _at_fc_env, horizon_h,
        )
        at_forecast_hours = list(range(1, horizon_h + 1))

    _pv_fc_env = os.getenv("COOLING_ML_PV_FORECAST_HOURS", _default_fc_hours)
    try:
        pv_forecast_hours = [int(x) for x in _pv_fc_env.split(",") if x.strip()]
    except ValueError:
        logger.warning(
            "COOLING_ML_PV_FORECAST_HOURS value %r is invalid; "
            "falling back to all %d hours",
            _pv_fc_env, horizon_h,
        )
        pv_forecast_hours = list(range(1, horizon_h + 1))

    logger.info(
        "Forecast feature hours — AT: %s | PV: %s",
        at_forecast_hours,
        pv_forecast_hours,
    )

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
        "PV_Generate",
        "pv_roll_1h",
        "pv_roll_2h",
    ]
    for h in pv_forecast_hours:
        feature_cols.append(f"pv_forecast_{h}h")

    feature_cols += [
        "thermal_power_kw",
        "delta_t",
        "outlet_indoor_diff",
        "VLT",
        "RLT",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        # HA context features
        "wind_speed",
        "living_room_temp",
        "fireplace_on",
        "tv_on",
        # Dynamic lag features
        "fireplace_lag_30m",
        "fireplace_lag_1h",
        "fireplace_lag_2h",
        "tv_lag_30m",
        "tv_lag_1h",
        # Derived physics features
        "heat_loss_driving_force",
        "indoor_temp_gradient",
        "indoor_margin_rate",
        "delta_T_indoor_lag1",
        "d_inlet_temp_60min",
        "is_equilibrium",
        "thermal_power_rolling_1h",
        "is_overshoot",
        "is_hp_active",
        "is_weekend",
        "heat_loss_interaction",
        # Solar / shading features
        "solar_thermal_proxy",
        "shading_proxy",
        "pv_forecast_delta",
        # Cumulative / integration features (Prio 4)
        "cum_pv_forecast_4h",
        "cum_at_excess_4h",
        "max_at_forecast",
        "indoor_momentum",
        "slab_stored_heat",
        # Trajectory-derived physics features
        "traj_predicted_error",
        "traj_convergence_rate",
        "traj_reaches_target_hours",
        "traj_overshoot_magnitude",
        "traj_equilibrium_gap",
    ]

    # ── 7. Label computation ────────────────────────────────────────────
    # label = 1 if max(indoor_temp[t : t + label_horizon_steps]) > cooling_target
    # The label window is PRE_COOL_LEAD_TIME_HOURS (not the full horizon) so
    # a positive prediction means "overheating within lead_time hours → act now".
    # Use the double-reversal rolling trick to look *forward* in time.
    # IMPORTANT: the last `label_horizon_steps` rows have no complete future window;
    # NaN from rolling becomes False via comparison → those rows must stay NaN
    # (pd.NA) so that the subsequent dropna() removes them rather than
    # labelling them as 0 (no overheating) which would add false negatives.
    _label_raw = (
        df["indoor_temp"]
        .iloc[::-1]
        .rolling(label_horizon_steps, min_periods=label_horizon_steps)
        .max()
        .iloc[::-1]
    )
    df["label"] = (
        (_label_raw > cooling_target_c)
        .where(_label_raw.notna())  # keep NaN where rolling was incomplete
        .astype("Int8")             # pandas nullable Int8 preserves pd.NA
    )

    # Regression target: delta_indoor_8h = max(indoor[t:t+8h]) - indoor[t]
    df["delta_indoor_8h"] = (_label_raw - df["indoor_temp"]).where(_label_raw.notna())

    # ── 8. Drop rows with NaN in any feature or label ───────────────────
    # Guard: only keep columns that exist and have >5% coverage
    available_features = []
    for col in feature_cols:
        if col not in df.columns:
            logger.warning("Feature '%s' not in dataframe — skipping", col)
            continue
        coverage = df[col].notna().mean()
        if coverage < 0.05:
            logger.warning("Feature '%s' coverage %.1f%% < 5%% — skipping", col, 100 * coverage)
            continue
        available_features.append(col)

    if len(available_features) < 5:
        logger.error("Too few usable feature columns (%d) — aborting", len(available_features))
        return False

    df_train = df[available_features + ["label", "delta_indoor_8h"]].dropna().copy()
    df_train["label"] = df_train["label"].astype(int)
    feature_cols = available_features  # update to what was actually available

    logger.info(
        "Training set: %d rows, %d features, %.1f%% positive labels",
        len(df_train),
        len(feature_cols),
        100 * df_train["label"].mean(),
    )

    min_samples = int(getattr(config, "COOLING_ML_MIN_TRAINING_SAMPLES", 200))
    if len(df_train) < min_samples:
        logger.error(
            "Only %d training samples (need %d). "
            "Increase lookback_hours or adjust warm_threshold.",
            len(df_train), min_samples,
        )
        return False

    # ── 9. Train / val split (temporal) ────────────────────────────────
    val_fraction = float(getattr(config, "COOLING_ML_RETRAIN_VAL_FRACTION", 0.25))
    total_pos = int(df_train["label"].sum())
    total_neg = int(len(df_train) - total_pos)
    if total_pos == 0 or total_neg == 0:
        logger.error(
            "Cooling ML calibration aborted: dataset contains only one class "
            "(pos=%d neg=%d). Check cooling_target and warm-season filtering.",
            total_pos,
            total_neg,
        )
        return False

    try:
        df_fit, df_val = _select_temporal_train_val_split(
            df_train, "label", val_fraction
        )
    except ValueError:
        logger.error(
            "Cooling ML calibration aborted: unable to create a temporal split "
            "with both classes in training (pos=%d neg=%d). Check cooling_target "
            "and label distribution.",
            total_pos,
            total_neg,
        )
        return False

    X_fit = df_fit[feature_cols].astype(float)
    y_fit = df_fit["label"].values
    X_val = df_val[feature_cols].astype(float)
    y_val = df_val["label"].values
    if not _has_both_classes(y_fit):
        logger.error(
            "Cooling ML calibration aborted: training split is single-class "
            "(pos=%d neg=%d).",
            int(y_fit.sum()),
            int(len(y_fit) - y_fit.sum()),
        )
        return False
    if not _has_both_classes(y_val):
        logger.warning(
            "Cooling ML validation split is single-class (pos=%d neg=%d); "
            "AUC/calibration metrics will be limited.",
            int(y_val.sum()),
            int(len(y_val) - y_val.sum()),
        )

    # ── 9b. Temporal boundary sample weighting (Prio 6) ────────────────
    # Upweight samples at the critical 0→1 label transition boundary and
    # samples where label=1 but indoor < target (conditions building,
    # not yet overheated).  These are the "morning decision" samples.
    sample_weights = np.ones(len(y_fit), dtype=float)
    _label_series = df_fit["label"].values
    _indoor_series = df_fit["indoor_temp"].values if "indoor_temp" in df_fit.columns else None

    # Weight 3.0: label transitions from 0→1 within the next 2h (12 steps)
    _transition_window = 2 * steps_per_hour
    for i in range(len(_label_series)):
        if _label_series[i] == 0:
            # Check if label becomes 1 within next 2h
            future_end = min(i + _transition_window, len(_label_series))
            if any(_label_series[i+1:future_end] == 1):
                sample_weights[i] = 3.0
        elif _label_series[i] == 1 and _indoor_series is not None:
            # Weight 2.0: label=1 but indoor still comfortable (building phase)
            if _indoor_series[i] < cooling_target_c:
                sample_weights[i] = 2.0

    n_upweighted = int((sample_weights > 1.0).sum())
    logger.info(
        "Temporal boundary weighting: %d/%d samples upweighted (%.1f%%)",
        n_upweighted, len(sample_weights), 100 * n_upweighted / max(1, len(sample_weights)),
    )

    # ── 10. Train LightGBM ─────────────────────────────────────────────
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError:
        logger.error("lightgbm not installed — cannot train model")
        return False

    pos = int(y_fit.sum())
    neg = len(y_fit) - pos
    spw = max(1.0, neg / max(1, pos))
    logger.info("Class imbalance: pos=%d neg=%d → scale_pos_weight=%.2f", pos, neg, spw)

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
        "scale_pos_weight": spw,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_fit, y_fit,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(50)],
    )

    # ── 11. Threshold optimisation (Prio 2: F-beta with β=1, cross-validated)
    # Use balanced F1 threshold for precision/recall trade-off.
    # Cross-validate: split training data into 3 temporal folds, find best
    # threshold on each, then average.
    proba_val = model.predict_proba(X_val)[:, 1]
    threshold, best_fbeta = _optimise_threshold_fbeta(y_val, proba_val, beta=1.0)
    logger.info("Optimal F1-threshold=%.4f (val F1=%.4f)", threshold, best_fbeta)

    # Cross-validated threshold confirmation on training folds
    cv_thresholds = _cross_validate_threshold(X_fit, y_fit, model, beta=1.0, n_folds=3)
    if cv_thresholds:
        cv_mean = float(np.mean(cv_thresholds))
        # Weighted average: CV thresholds are more robust (in-sample but
        # temporally diverse), val threshold can be an outlier on a single
        # split.  Weight CV 2:1 over val to stabilise the final threshold.
        threshold = float(np.average(
            [threshold, cv_mean],
            weights=[1.0, 2.0],
        ))
        logger.info(
            "CV thresholds: %s → mean=%.4f | final threshold=%.4f (weighted avg)",
            [f"{t:.4f}" for t in cv_thresholds], cv_mean, threshold,
        )

    # AUC on val
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
        auc = float(roc_auc_score(y_val, proba_val))
    except Exception:
        auc = float("nan")
    logger.info("Val AUC=%.4f", auc)

    # ── 11b. Isotonic probability calibration (Prio 3) ──────────────────
    # Wrap the model with isotonic calibration to produce well-calibrated
    # probabilities.  Fit on the first half of val; evaluate on the second
    # half to avoid overfitting the calibration map to the same data used
    # for AUC comparison.
    calibrated_model = None
    calibrated_auc = float("nan")
    # Split val into calibration-fit and calibration-eval halves
    n_cal_iso = len(X_val) // 2
    X_cal_iso, y_cal_iso = X_val[:n_cal_iso], y_val[:n_cal_iso]
    X_eval_iso, y_eval_iso = X_val[n_cal_iso:], y_val[n_cal_iso:]
    if (
        n_cal_iso < 2
        or len(y_eval_iso) < 2
        or not _has_both_classes(y_cal_iso)
        or not _has_both_classes(y_eval_iso)
    ):
        logger.info(
            "Skipping isotonic calibration: insufficient class diversity in "
            "validation sub-splits"
        )
        auc_eval = auc
    else:
        try:
            from sklearn.calibration import CalibratedClassifierCV  # type: ignore
            # sklearn >=1.6 removed cv="prefit"; fall back to fitting an
            # IsotonicRegression directly on the base model's probabilities so
            # the base model weights are never changed (unlike cv=2 which refits
            # cloned estimators on the calibration split).
            try:
                calibrated_model = CalibratedClassifierCV(
                    estimator=model, method="isotonic", cv="prefit"
                )
                calibrated_model.fit(X_cal_iso, y_cal_iso)
            except (TypeError, ValueError):
                from sklearn.isotonic import IsotonicRegression  # type: ignore
                _iso = IsotonicRegression(out_of_bounds="clip")
                _iso.fit(model.predict_proba(X_cal_iso)[:, 1], y_cal_iso)
                calibrated_model = _IsotonicCalibratedModel(model, _iso)
            proba_cal_eval = calibrated_model.predict_proba(X_eval_iso)[:, 1]
            proba_raw_eval = model.predict_proba(X_eval_iso)[:, 1]
            calibrated_auc = float(roc_auc_score(y_eval_iso, proba_cal_eval))
            auc_eval = float(roc_auc_score(y_eval_iso, proba_raw_eval))
            logger.info(
                "Calibrated model eval-split AUC=%.4f (raw eval-split AUC=%.4f)",
                calibrated_auc, auc_eval,
            )
        except Exception as exc:
            logger.warning("Isotonic calibration failed: %s — using raw model", exc)
            calibrated_model = None
            auc_eval = auc

    # Decision: use calibrated model if it doesn't degrade AUC on held-out eval split
    use_calibrated = (
        calibrated_model is not None
        and not math.isnan(calibrated_auc)
        and not math.isnan(auc_eval)
        and calibrated_auc >= auc_eval - _ISOTONIC_AUC_TOLERANCE
    )
    final_model = calibrated_model if use_calibrated else model
    final_auc = calibrated_auc if use_calibrated else auc
    proba_cal_full = None  # initialise before branch; set inside if use_calibrated
    if use_calibrated:
        # Re-optimise threshold on calibrated probabilities over the full val set
        proba_cal_full = calibrated_model.predict_proba(X_val)[:, 1]
        raw_threshold = threshold
        threshold, best_fbeta = _optimise_threshold_fbeta(y_val, proba_cal_full, beta=1.0)
        # Log isotonic threshold shift
        shift = threshold - raw_threshold
        logger.info(
            "Isotonic threshold shift: raw=%.4f → calibrated=%.4f (Δ=%+.4f)",
            raw_threshold, threshold, shift,
        )
        if abs(shift) > 0.5 * raw_threshold and raw_threshold > 0.01:
            logger.warning(
                "⚠️ Large isotonic threshold shift: %.1f%%",
                100 * abs(shift) / raw_threshold,
            )
        logger.info(
            "Using CALIBRATED model (eval AUC=%.4f) with threshold=%.4f",
            calibrated_auc, threshold,
        )
    else:
        logger.info("Using RAW model (calibration did not improve AUC)")

    # ── 11c. Diagnostic F1 / precision / recall summary ─────────────────
    _diag_proba = proba_cal_full if (use_calibrated and proba_cal_full is not None) else proba_val
    _diag_pred = (_diag_proba >= threshold).astype(int)
    _tp = int(((_diag_pred == 1) & (y_val == 1)).sum())
    _fp = int(((_diag_pred == 1) & (y_val == 0)).sum())
    _fn = int(((_diag_pred == 0) & (y_val == 1)).sum())
    _prec = _tp / max(1, _tp + _fp)
    _rec = _tp / max(1, _tp + _fn)
    _f1 = 2.0 * _prec * _rec / max(1e-9, _prec + _rec)
    _pred_pos_rate = 100.0 * _diag_pred.mean()
    _true_pos_rate = 100.0 * y_val.mean()
    logger.info(
        "Diagnostics: F1=%.4f | Precision=%.4f | Recall=%.4f | "
        "Predicted pos rate=%.1f%% (true=%.1f%%)",
        _f1, _prec, _rec, _pred_pos_rate, _true_pos_rate,
    )

    # ── 12. Save model + metadata ───────────────────────────────────────
    try:
        import joblib  # type: ignore
    except ImportError:
        logger.error("joblib not installed — cannot save model")
        return False

    model_path    = getattr(config, "COOLING_ML_MODEL_PATH",    "/opt/ml_heating/cooling_ml_model.joblib")
    metadata_path = getattr(config, "COOLING_ML_METADATA_PATH", "/opt/ml_heating/cooling_ml_metadata.json")

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    tmp_model = model_path + ".tmp"
    joblib.dump(final_model, tmp_model)
    os.replace(tmp_model, model_path)

    metadata = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "threshold": threshold,
        "val_f1": best_fbeta,
        "roc_auc": final_auc,
        "roc_auc_raw": auc,
        "calibrated": use_calibrated,
        "n_train": int(len(df_fit)),
        "n_val": int(len(df_val)),
        "n_pos": int(pos),
        "n_neg": int(neg),
        "scale_pos_weight": spw,
        "label_horizon_h": label_horizon_h,  # window used for the label (= lead_time_h)
        "forecast_horizon_h": horizon_h,      # window used for hindcast features
        "steps_per_hour": steps_per_hour,
        "cooling_target_c": cooling_target_c,
        "lookback_hours": lookback_hours,
        "lgb_params": lgb_params,
        "threshold_method": "f1_cross_validated",
        "noise_injection": True,
        "temporal_weighting": True,
    }

    # ── 12b. Train LGBMRegressor on delta_indoor_8h ────────────────────
    reg_model_path = model_path.replace(
        "cooling_ml_model.joblib", "cooling_ml_regressor.joblib"
    )
    try:
        y_fit_reg = df_fit["delta_indoor_8h"].values.astype(float)
        y_val_reg = df_val["delta_indoor_8h"].values.astype(float)

        lgb_params_reg = {
            "objective": "regression_l1",
            "metric": "mae",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        reg_model = lgb.LGBMRegressor(**lgb_params_reg)
        reg_model.fit(
            X_fit, y_fit_reg,
            eval_set=[(X_val, y_val_reg)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(50)],
        )

        # Evaluate regression
        from sklearn.metrics import mean_absolute_error  # type: ignore
        delta_pred_val = reg_model.predict(X_val)
        reg_mae = float(mean_absolute_error(y_val_reg, delta_pred_val))

        # Derive binary risk from regression: risk = indoor + Δ > threshold
        indoor_val = df_val["indoor_temp"].values.astype(float)
        predicted_max_val = indoor_val + delta_pred_val
        reg_threshold, reg_f1 = _optimise_regression_threshold(
            y_val, indoor_val, delta_pred_val, cooling_target_c,
        )
        # AUC of regression used as classifier (ranking by delta_pred)
        try:
            reg_auc = float(roc_auc_score(y_val, delta_pred_val))
        except Exception:
            reg_auc = float("nan")

        logger.info(
            "Regressor: MAE=%.4f | AUC=%.4f | reg_threshold=%.2f°C | F1=%.4f",
            reg_mae, reg_auc, reg_threshold, reg_f1,
        )

        # Save regression model
        tmp_reg = reg_model_path + ".tmp"
        joblib.dump(reg_model, tmp_reg)
        os.replace(tmp_reg, reg_model_path)

        metadata["regression_threshold"] = reg_threshold
        metadata["regression_mae"] = reg_mae
        metadata["regression_auc"] = reg_auc
        metadata["regression_f1"] = reg_f1
        metadata["model_approach"] = "dual"
        metadata["reg_model_path"] = reg_model_path
        logger.info("Regressor saved → %s", reg_model_path)

    except Exception:
        logger.exception(
            "Regression model training failed — classifier-only mode"
        )
        metadata["model_approach"] = "classifier_only"

    tmp_meta = metadata_path + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=_json_default)
    os.replace(tmp_meta, metadata_path)

    logger.info(
        "=== COOLING ML CALIBRATION COMPLETE: model → %s | AUC=%.4f threshold=%.4f "
        "calibrated=%s ===",
        model_path, final_auc, threshold, use_calibrated,
    )

    # ── 13. Export training data for offline HPO / analysis ──────────────
    export_training_data(df_train, feature_cols, "cooling")

    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _optimise_threshold_fbeta(y_true, proba, beta: float = 2.0, n_points: int = 200) -> tuple[float, float]:
    """Find threshold that maximises F-beta score on the given split.

    With beta=2 (default), recall is weighted 4× more than precision —
    ensuring the model fires early enough for pre-cooling.
    """
    import numpy as np

    beta_sq = beta * beta
    best_thr = 0.5
    best_fbeta = 0.0
    for thr in np.linspace(0.01, 0.99, n_points):
        y_pred = (proba >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        fbeta = (1 + beta_sq) * prec * rec / max(1e-9, beta_sq * prec + rec)
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_thr = float(thr)
    return best_thr, best_fbeta


# Keep legacy function for backward compat (online retraining buffer uses it)
def _optimise_threshold(y_true, proba, n_points: int = 200) -> tuple[float, float]:
    """Find threshold that maximises F1 score on the given split (legacy)."""
    return _optimise_threshold_fbeta(y_true, proba, beta=1.0, n_points=n_points)


def _optimise_regression_threshold(
    y_true, indoor_temp, delta_pred, cooling_target: float, n_points: int = 200
) -> tuple[float, float]:
    """Find temperature threshold on predicted_max that maximises F1.

    Predicted max = indoor_temp + delta_pred.
    Risk = predicted_max > threshold.
    Sweeps thresholds around the cooling target to find the F1-optimal one.
    """
    import numpy as np

    predicted_max = indoor_temp + delta_pred
    best_thr = cooling_target
    best_f1 = 0.0
    for thr in np.linspace(cooling_target - 2.0, cooling_target + 2.0, n_points):
        y_pred = (predicted_max > thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2.0 * prec * rec / max(1e-9, prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


def _cross_validate_threshold(
    X_train, y_train, model, beta: float = 2.0, n_folds: int = 3
) -> list[float]:
    """Per-segment threshold estimation on the training set.

    Splits X_train into n_folds temporal segments and computes F-beta-optimal
    thresholds using predictions from ``model``, which has already been fit on
    the full training set.  Because the same model is used for all segments
    (i.e. there is no per-fold re-training), these are in-sample predictions
    and the resulting thresholds may be optimistic.  The primary purpose is to
    check whether the validation-set threshold is consistent across temporal
    segments of the training data, not to produce a truly held-out estimate.
    Returns a list of per-segment thresholds.
    """
    import numpy as np

    n = len(y_train)
    fold_size = n // n_folds
    if fold_size < _CV_MIN_FOLD_SIZE:
        return []  # too few samples for meaningful CV

    thresholds: list[float] = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n
        X_fold = X_train[start:end]
        y_fold = y_train[start:end]
        if len(y_fold) < 20 or y_fold.sum() < 1:
            continue
        try:
            proba_fold = model.predict_proba(X_fold)[:, 1]
            thr, _ = _optimise_threshold_fbeta(y_fold, proba_fold, beta=beta)
            thresholds.append(thr)
        except Exception:
            continue
    return thresholds


def _json_default(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
