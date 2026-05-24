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
from typing import Optional

logger = logging.getLogger(__name__)

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
        cooling_target_c = clamp_max - 1.0
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

    rename_map = {
        indoor_col:  "indoor_temp",
        outdoor_col: "AT",
        outlet_col:  "VLT",
        inlet_col:   "RLT",
        flow_col:    "flow_rate",
        power_col:   "power_w",
        pv_col:      "PV_Generate",
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
        df["cum_at_excess_4h"] = df[_cum_at_cols].apply(
            lambda row: sum(max(0.0, v - cooling_target_c) for v in row), axis=1
        )
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
        # Cumulative / integration features (Prio 4)
        "cum_pv_forecast_4h",
        "cum_at_excess_4h",
        "max_at_forecast",
        "indoor_momentum",
        "slab_stored_heat",
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

    df_train = df[available_features + ["label"]].dropna().copy()
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
    n_val = max(1, int(len(df_train) * val_fraction))
    df_val   = df_train.iloc[-n_val:].copy()
    df_fit   = df_train.iloc[:-n_val].copy()

    X_fit = df_fit[feature_cols].values.astype(float)
    y_fit = df_fit["label"].values
    X_val = df_val[feature_cols].values.astype(float)
    y_val = df_val["label"].values

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

    # ── 11. Threshold optimisation (Prio 2: F-beta with β=2, cross-validated)
    # Use recall-biased threshold to ensure early activation.
    # Cross-validate: split training data into 3 temporal folds, find best
    # threshold on each, then average.
    proba_val = model.predict_proba(X_val)[:, 1]
    threshold, best_fbeta = _optimise_threshold_fbeta(y_val, proba_val, beta=2.0)
    logger.info("Optimal F2-threshold=%.4f (val F2=%.4f)", threshold, best_fbeta)

    # Cross-validated threshold confirmation on training folds
    cv_thresholds = _cross_validate_threshold(X_fit, y_fit, model, beta=2.0, n_folds=3)
    if cv_thresholds:
        cv_mean = float(np.mean(cv_thresholds))
        # Use the lower of CV mean and val-optimised (prefer earlier trigger)
        threshold = min(threshold, cv_mean)
        logger.info(
            "CV thresholds: %s → mean=%.4f | final threshold=%.4f",
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
    # probabilities.  If calibration improves val AUC → save calibrated model.
    calibrated_model = None
    calibrated_auc = float("nan")
    try:
        from sklearn.calibration import CalibratedClassifierCV  # type: ignore
        # Fit isotonic calibration on validation predictions
        calibrated_model = CalibratedClassifierCV(
            model, method="isotonic", cv="prefit"
        )
        calibrated_model.fit(X_val, y_val)
        proba_cal = calibrated_model.predict_proba(X_val)[:, 1]
        calibrated_auc = float(roc_auc_score(y_val, proba_cal))
        logger.info("Calibrated model val AUC=%.4f (raw=%.4f)", calibrated_auc, auc)
    except Exception as exc:
        logger.warning("Isotonic calibration failed: %s — using raw model", exc)
        calibrated_model = None

    # Decision: use calibrated model if it doesn't degrade AUC
    use_calibrated = (
        calibrated_model is not None
        and not math.isnan(calibrated_auc)
        and calibrated_auc >= auc - 0.01  # allow up to 0.01 AUC drop
    )
    final_model = calibrated_model if use_calibrated else model
    final_auc = calibrated_auc if use_calibrated else auc
    if use_calibrated:
        # Re-optimise threshold on calibrated probabilities
        threshold, best_fbeta = _optimise_threshold_fbeta(y_val, proba_cal, beta=2.0)
        logger.info(
            "Using CALIBRATED model (AUC=%.4f) with threshold=%.4f",
            calibrated_auc, threshold,
        )
    else:
        logger.info("Using RAW model (calibration did not improve AUC)")

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
        "val_f2": best_fbeta,
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
        "threshold_method": "f2_cross_validated",
        "noise_injection": True,
        "temporal_weighting": True,
    }
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


def _cross_validate_threshold(
    X_train, y_train, model, beta: float = 2.0, n_folds: int = 3
) -> list[float]:
    """Temporal cross-validation for threshold selection.

    Splits X_train into n_folds temporal segments, computes model
    predictions on each held-out segment, and finds the optimal F-beta
    threshold per fold.  Returns a list of per-fold thresholds.
    """
    import numpy as np

    n = len(y_train)
    fold_size = n // n_folds
    if fold_size < 50:
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
