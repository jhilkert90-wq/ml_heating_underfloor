"""
cooling_ml_model.py
--------------------
LightGBM-based overheating classifier for predictive pre-cooling.

Drop-in complement to ``OverheatingPredictor`` (trajectory-based).
Both return the same result-dict shape so ``main.py`` can switch between
them with a single config flag (``PRE_COOL_MODEL_TYPE``).

Feature mapping
---------------
This module translates the dict returned by ``build_physics_features()``
into the fixed feature vector that was used at training time.  The exact
column order and names are stored in the model metadata JSON so that
notebook-trained and online-retrained models remain compatible.

Sign convention: ``temp_diff_indoor_outdoor`` in physics_features.py is
defined as ``indoor - outdoor``.  The model feature ``at_delta_indoor``
is ``AT - indoor = outdoor - indoor = -temp_diff_indoor_outdoor``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers (joblib / numpy only needed when a model is loaded)
# ---------------------------------------------------------------------------

def _load_joblib():
    try:
        import joblib  # type: ignore
        return joblib
    except ImportError as exc:
        raise ImportError("joblib is required for CoolingMLModel") from exc


def _load_numpy():
    try:
        import numpy as np  # type: ignore
        return np
    except ImportError as exc:
        raise ImportError("numpy is required for CoolingMLModel") from exc


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _pv_roll(physics: dict[str, Any], hours: int, steps_per_hour: int = 6) -> float:
    """Mean raw PV power over the last ``hours`` hours."""
    history: list[float] = (
        physics.get("pv_power_history_electrical")
        or physics.get("pv_power_history")
        or []
    )
    n = hours * steps_per_hour
    if not history:
        return 0.0
    window = history[-n:] if len(history) >= n else history
    return float(sum(window) / len(window)) if window else 0.0


def _doy_sin() -> float:
    doy = datetime.now().timetuple().tm_yday
    return math.sin(2 * math.pi * doy / 365.25)


def _doy_cos() -> float:
    doy = datetime.now().timetuple().tm_yday
    return math.cos(2 * math.pi * doy / 365.25)


def build_feature_vector(
    feature_cols: list[str],
    physics: dict[str, Any],
    current_indoor: float,
    cooling_target: float,
    steps_per_hour: int = 6,
) -> list[float]:
    """
    Construct a feature vector (in ``feature_cols`` order) from
    ``physics_features`` dict, current indoor temp, and cooling target.

    Unknown columns are filled with 0.0 and logged as a warning.
    """
    vec: list[float] = []
    for col in feature_cols:
        val = _extract_feature(col, physics, current_indoor, cooling_target, steps_per_hour)
        vec.append(val)
    return vec


def _extract_feature(
    col: str,
    physics: dict[str, Any],
    current_indoor: float,
    cooling_target: float,
    steps_per_hour: int,
) -> float:
    # ── Derived scalars ────────────────────────────────────────────────
    if col == "indoor_temp":
        return current_indoor
    if col == "indoor_margin":
        # Negative when indoor > target (room too warm)
        return cooling_target - current_indoor
    if col == "indoor_trend_30m":
        return float(physics.get("indoor_temp_delta_30m") or 0.0)
    if col == "indoor_trend_1h":
        return float(physics.get("indoor_temp_delta_60m") or 0.0)
    if col == "AT":
        return float(physics.get("outdoor_temp") or 0.0)
    if col == "at_delta_indoor":
        # AT - indoor = -(indoor - AT) = -temp_diff_indoor_outdoor
        return -float(physics.get("temp_diff_indoor_outdoor") or 0.0)
    if col == "PV_Generate":
        # Prefer raw electrical watts (matches training data scale).
        # Fall back to thermally-corrected pv_now when the electrical key
        # is absent (e.g. PV_TRAJ_FORECAST_MODE_ENABLED=False).
        # Explicit None check so that a valid 0.0 (no PV generation) is kept.
        val = physics.get("pv_now_electrical")
        if val is None:
            val = physics.get("pv_now", 0.0)
        return float(val)
    if col == "pv_roll_1h":
        return _pv_roll(physics, 1, steps_per_hour)
    if col == "pv_roll_2h":
        return _pv_roll(physics, 2, steps_per_hour)
    if col == "thermal_power_kw":
        return float(physics.get("thermal_power_kw") or 0.0)
    if col == "delta_t":
        return float(physics.get("delta_t") or 0.0)
    if col == "outlet_indoor_diff":
        return float(physics.get("outlet_indoor_diff") or 0.0)
    if col == "VLT":
        return float(physics.get("outlet_temp") or 0.0)
    if col == "RLT":
        return float(physics.get("inlet_temp") or 0.0)
    if col == "hour_sin":
        return float(physics.get("hour_sin") or 0.0)
    if col == "hour_cos":
        return float(physics.get("hour_cos") or 0.0)
    if col == "doy_sin":
        return _doy_sin()
    if col == "doy_cos":
        return _doy_cos()

    # ── Dynamic forecast features: AT_roh_Xh → temp_forecast_Xh ──────
    m = re.fullmatch(r"AT_roh_(\d+)h", col)
    if m:
        h = m.group(1)
        return float(physics.get(f"temp_forecast_{h}h") or physics.get("outdoor_temp") or 0.0)

    # ── Dynamic PV forecast: pv_forecast_Xh ───────────────────────────
    m = re.fullmatch(r"pv_forecast_(\d+)h", col)
    if m:
        h = m.group(1)
        # Prefer raw electrical watts (matches training data scale).
        # Fall back to thermally-corrected value when the electrical key
        # is absent (e.g. PV_TRAJ_FORECAST_MODE_ENABLED=False).
        # Explicit None checks so that a valid 0.0 (nighttime) is kept.
        val = physics.get(f"pv_forecast_electrical_{h}h")
        if val is None:
            val = physics.get(f"pv_forecast_{h}h")
        if val is None:
            val = physics.get("pv_now_electrical")
        if val is None:
            val = physics.get("pv_now", 0.0)
        return float(val)

    # ── Cumulative / integration features (Prio 4) ────────────────────
    if col == "cum_pv_forecast_4h":
        total = 0.0
        for h in range(1, 5):
            v = physics.get(f"pv_forecast_electrical_{h}h")
            if v is None:
                v = physics.get(f"pv_forecast_{h}h")
            if v is None:
                v = physics.get("pv_now_electrical")
            if v is None:
                v = physics.get("pv_now", 0.0)
            total += float(v)
        return total

    if col == "cum_at_excess_4h":
        total = 0.0
        for h in range(1, 5):
            at_val = float(physics.get(f"temp_forecast_{h}h") or physics.get("outdoor_temp") or 0.0)
            total += max(0.0, at_val - cooling_target)
        return total

    if col == "max_at_forecast":
        max_at = float(physics.get("outdoor_temp") or 0.0)
        # Look up to 8h ahead (label horizon)
        for h in range(1, 9):
            at_val = float(physics.get(f"temp_forecast_{h}h") or 0.0)
            if at_val > max_at:
                max_at = at_val
        return max_at

    if col == "indoor_momentum":
        # Thermal momentum proxy: 3h extrapolation from 1h trend.
        # 3h chosen because slab thermal mass has ~3h dominant time constant;
        # this approximates where indoor temp will be when cooling effect arrives.
        trend = float(physics.get("indoor_temp_delta_60m") or 0.0)
        return trend * 3.0

    if col == "slab_stored_heat":
        vlt = float(physics.get("outlet_temp") or 0.0)
        rlt = float(physics.get("inlet_temp") or 0.0)
        return (vlt + rlt) / 2.0 - current_indoor

    logger.warning("CoolingMLModel: unknown feature column '%s', filling 0.0", col)
    return 0.0


# ---------------------------------------------------------------------------
# CoolingMLModel
# ---------------------------------------------------------------------------

class CoolingMLModel:
    """
    Thin wrapper around a joblib-serialised LightGBM (or sklearn) classifier.

    Exposes ``predict_overheating_risk()`` with the same result dict as
    ``OverheatingPredictor`` so callers need no special-casing.
    """

    def __init__(self, model_path: str, metadata_path: str, steps_per_hour: int = 6) -> None:
        self._model_path = model_path
        self._metadata_path = metadata_path
        self._steps_per_hour = steps_per_hour

        self._model: Any = None
        self._metadata: dict[str, Any] = {}
        self._feature_cols: list[str] = []
        self._threshold: float = 0.5
        self._loaded = False

    # ------------------------------------------------------------------
    # Load / reload
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load (or reload) model and metadata from disk. Returns True on success."""
        joblib = _load_joblib()
        if not os.path.exists(self._model_path):
            logger.warning(
                "CoolingMLModel: model file not found: %s. "
                "Run --calibrate-cooling-ml to train.",
                self._model_path,
            )
            self._loaded = False
            return False
        if not os.path.exists(self._metadata_path):
            logger.warning(
                "CoolingMLModel: metadata file not found: %s.",
                self._metadata_path,
            )
            self._loaded = False
            return False
        try:
            self._model = joblib.load(self._model_path)
            with open(self._metadata_path, "r", encoding="utf-8") as fh:
                self._metadata = json.load(fh)
            self._feature_cols = self._metadata.get("feature_cols", [])
            self._threshold = float(self._metadata.get("threshold", 0.5))
            self._loaded = True
            logger.info(
                "CoolingMLModel: loaded %s | features=%d threshold=%.4f AUC=%.4f",
                os.path.basename(self._model_path),
                len(self._feature_cols),
                self._threshold,
                self._metadata.get("roc_auc", float("nan")),
            )
            return True
        except Exception:
            logger.exception("CoolingMLModel: failed to load model from %s", self._model_path)
            self._loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_overheating_risk(
        self,
        current_indoor: float,
        target_cooling: float,
        features: dict[str, Any],
        climate_mode: str = "cooling",
    ) -> dict[str, Any]:
        """
        Predict overheating risk using the LGBM classifier.

        Parameters mirror ``OverheatingPredictor.predict_overheating_risk``
        (``thermal_model`` is not needed and omitted).

        Returns
        -------
        dict with keys: risk, peak_temp, peak_hour, hours_until_peak,
        should_cool_now, reason, trajectory, trigger_threshold,
        peak_outdoor, total_pv_forecast, lgbm_proba (extra).
        """
        try:
            from . import config
        except ImportError:
            import config  # type: ignore

        no_risk = self._no_risk_result(target_cooling, features, config)

        if climate_mode != "cooling":
            no_risk["reason"] = "LGBM: not in cooling mode"
            return no_risk

        if not self._loaded:
            no_risk["reason"] = "LGBM: model not loaded — using no-risk fallback"
            return no_risk

        if not self._feature_cols:
            no_risk["reason"] = "LGBM: feature_cols empty — metadata corrupt?"
            return no_risk

        try:
            np = _load_numpy()
            vec = build_feature_vector(
                self._feature_cols,
                features,
                current_indoor,
                target_cooling,
                self._steps_per_hour,
            )
            X = np.array(vec, dtype=float).reshape(1, -1)
            proba = float(self._model.predict_proba(X)[0, 1])
        except Exception:
            logger.exception("CoolingMLModel: inference failed")
            no_risk["reason"] = "LGBM: inference error — no-risk fallback"
            return no_risk

        risk = proba > self._threshold
        trigger_threshold = float(
            getattr(config, "PRE_COOL_TRIGGER_MARGIN_K", 0.5)
        ) + target_cooling
        horizon_h = float(getattr(config, "PRE_COOL_HORIZON_HOURS", 12))
        lead_time = float(getattr(config, "PRE_COOL_LEAD_TIME_HOURS", 3.0))
        at_now = float(features.get("outdoor_temp", 0.0))
        pv_now = float(features.get("pv_now", 0.0))

        # ── should_cool_now logic (mirrors OverheatingPredictor) ───────
        should_cool_now = False
        reason_parts: list[str] = [f"LGBM p={proba:.3f} (thr={self._threshold:.3f})"]

        if current_indoor > target_cooling:
            should_cool_now = True
            reason_parts.append(
                f"room {current_indoor:.1f}°C > target {target_cooling:.1f}°C (reactive)"
            )
        elif risk:
            # Model predicts overheating within PRE_COOL_LEAD_TIME_HOURS
            # (that is the label window used at training time).  A positive
            # prediction already implies imminent overheating, so we act now
            # without an additional lead-time gate.
            should_cool_now = True
            reason_parts.append(
                f"model predicts overheating (p={proba:.3f} > {self._threshold:.3f})"
            )
        else:
            reason_parts.append("no overheating risk predicted")

        peak_hour = (lead_time / 2.0) if risk else horizon_h
        # Rough peak_temp proxy: current indoor + margin deficit as lower bound
        peak_temp_proxy = max(current_indoor, trigger_threshold + 0.1) if risk else current_indoor

        result: dict[str, Any] = {
            "risk": risk,
            "peak_temp": peak_temp_proxy,
            "peak_hour": peak_hour,
            "hours_until_peak": peak_hour,
            "should_cool_now": should_cool_now,
            "reason": "; ".join(reason_parts),
            "trajectory": [],  # not available from classifier
            "trigger_threshold": trigger_threshold,
            "peak_outdoor": at_now,
            "total_pv_forecast": pv_now,
            "lgbm_proba": proba,
        }

        if should_cool_now:
            logger.info("❄️ LGBM PRE-COOL ACTIVATED: %s", result["reason"])
        elif risk:
            logger.info("❄️ LGBM risk detected: %s", result["reason"])
        else:
            logger.debug("LGBM no risk: %s", result["reason"])

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _no_risk_result(
        self, target_cooling: float, features: dict[str, Any], config: Any
    ) -> dict[str, Any]:
        trigger_threshold = (
            float(getattr(config, "PRE_COOL_TRIGGER_MARGIN_K", 0.5)) + target_cooling
        )
        horizon_h = float(getattr(config, "PRE_COOL_HORIZON_HOURS", 12))
        return {
            "risk": False,
            "peak_temp": float(features.get("outdoor_temp", 0.0)),
            "peak_hour": horizon_h,
            "hours_until_peak": horizon_h,
            "should_cool_now": False,
            "reason": "LGBM: no risk",
            "trajectory": [],
            "trigger_threshold": trigger_threshold,
            "peak_outdoor": float(features.get("outdoor_temp", 0.0)),
            "total_pv_forecast": float(features.get("pv_now", 0.0)),
            "lgbm_proba": 0.0,
        }
