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

    df = fetch_historical_data_for_calibration(lookback_hours=lookback_hours)
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

    warm_threshold = float(getattr(config, "PRE_COOL_MIN_OUTDOOR_FORECAST_C", 22.0)) - 6.0
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

    # ── 6. Define feature set ────────────────────────────────────────────
    # The exact columns are saved to metadata; inference reads them back.
    # AT hindcast hours: controlled by COOLING_ML_AT_FORECAST_HOURS
    # (legacy alias: COOLING_ML_FORECAST_HOURS).  Defaults to all 12 hours.
    # PV hindcast hours: controlled by COOLING_ML_PV_FORECAST_HOURS.
    # Defaults to all 12 hours.
    # The coverage guard in Step 8 silently drops any column whose data
    # coverage is below 5% (e.g. the last N rows which have no future window).
    _at_fc_env = os.getenv(
        "COOLING_ML_AT_FORECAST_HOURS",
        os.getenv("COOLING_ML_FORECAST_HOURS", "1,2,3,4,5,6,7,8,9,10,11,12"),
    )
    try:
        at_forecast_hours = [int(x) for x in _at_fc_env.split(",") if x.strip()]
    except ValueError:
        at_forecast_hours = list(range(1, horizon_h + 1))

    _pv_fc_env = os.getenv("COOLING_ML_PV_FORECAST_HOURS", "1,2,3,4,5,6,7,8,9,10,11,12")
    try:
        pv_forecast_hours = [int(x) for x in _pv_fc_env.split(",") if x.strip()]
    except ValueError:
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
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(50)],
    )

    # ── 11. Threshold optimisation (maximise F1 on val) ────────────────
    proba_val = model.predict_proba(X_val)[:, 1]
    threshold, best_f1 = _optimise_threshold(y_val, proba_val)
    logger.info("Optimal threshold=%.4f (val F1=%.4f)", threshold, best_f1)

    # AUC on val
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
        auc = float(roc_auc_score(y_val, proba_val))
    except Exception:
        auc = float("nan")
    logger.info("Val AUC=%.4f", auc)

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
    joblib.dump(model, tmp_model)
    os.replace(tmp_model, model_path)

    metadata = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "threshold": threshold,
        "val_f1": best_f1,
        "roc_auc": auc,
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
    }
    tmp_meta = metadata_path + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=_json_default)
    os.replace(tmp_meta, metadata_path)

    logger.info(
        "=== COOLING ML CALIBRATION COMPLETE: model → %s | AUC=%.4f threshold=%.4f ===",
        model_path, auc, threshold,
    )
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _optimise_threshold(y_true, proba, n_points: int = 200) -> tuple[float, float]:
    """Find threshold that maximises F1 score on the given split."""
    import numpy as np

    best_thr = 0.5
    best_f1 = 0.0
    for thr in np.linspace(0.01, 0.99, n_points):
        y_pred = (proba >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = 2 * prec * rec / max(1e-9, prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


def _json_default(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
