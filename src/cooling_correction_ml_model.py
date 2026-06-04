"""
cooling_correction_ml_model.py
--------------------------------
LightGBM-based outlet-temperature correction regressor for cooling mode.

Mirrors ``heating_correction_ml_model.py`` but uses cooling-specific parameters:
- S_H computed from cooling outlet-effectiveness (OE_cooling ≈ 0.20)
- Residualized label reconstruction at inference time
- Cooling target temperature (typically 23.0°C)

Inference usage (inside ``model_wrapper._calculate_cooling_ml_correction()``)
-----------------------------------------------------------------------------
1.  ``CoolingCorrectionMLModel.predict(features, target_indoor)`` → float or None
2.  Caller blends the ML delta with the physics Newton delta using R² as weight.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import datetime
from typing import Any, Optional

from src import config

logger = logging.getLogger(__name__)

_SHADING_ACTIVATION_TEMP_C = 23.0


# ---------------------------------------------------------------------------
# Lazy-import helpers
# ---------------------------------------------------------------------------

def _load_joblib():
    try:
        import joblib  # type: ignore
        return joblib
    except ImportError as exc:
        raise ImportError("joblib is required for CoolingCorrectionMLModel") from exc


def _load_numpy():
    try:
        import numpy as np  # type: ignore
        return np
    except ImportError as exc:
        raise ImportError("numpy is required for CoolingCorrectionMLModel") from exc


def _load_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except ImportError as exc:
        raise ImportError("pandas is required for CoolingCorrectionMLModel") from exc


# ---------------------------------------------------------------------------
# Day-of-year helpers
# ---------------------------------------------------------------------------

def _doy_sin() -> float:
    doy = datetime.now().timetuple().tm_yday
    return math.sin(2 * math.pi * doy / 365.25)


def _doy_cos() -> float:
    doy = datetime.now().timetuple().tm_yday
    return math.cos(2 * math.pi * doy / 365.25)


# ---------------------------------------------------------------------------
# PV rolling helper
# ---------------------------------------------------------------------------

def _pv_roll(physics: dict[str, Any], hours: int, steps_per_hour: int = 6) -> float:
    history = physics.get("pv_power_history_electrical") or physics.get(
        "pv_power_history"
    )
    if not history:
        val = physics.get("pv_now_electrical")
        if val is None:
            val = physics.get("pv_now")
        return float(val) if val is not None else 0.0
    n = max(1, hours * steps_per_hour)
    recent = list(history)[-n:]
    try:
        return float(sum(float(v) for v in recent) / len(recent))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Trajectory-derived feature helpers (inference time)
# ---------------------------------------------------------------------------

def _get_traj_params_cooling(physics: dict[str, Any]) -> tuple[float, float, float]:
    """Return (OE_cooling, HLC, tau) for analytical trajectory approximation."""
    oe = float(getattr(config, "COOLING_OUTLET_EFFECTIVENESS", 0.20))
    hlc = float(getattr(config, "HEAT_LOSS_COEFFICIENT", 0.124))
    tau = float(getattr(config, "THERMAL_TIME_CONSTANT", 4.39))
    return oe, hlc, tau


def _compute_traj_equilibrium_cooling(physics: dict[str, Any]) -> float:
    oe, hlc, tau = _get_traj_params_cooling(physics)
    vlt = physics.get("outlet_temp")
    at = physics.get("outdoor_temp")
    if vlt is None or at is None:
        return 0.0
    denom = oe + hlc
    if denom < 1e-6:
        return float(vlt)
    return (oe * float(vlt) + hlc * float(at)) / denom


def _compute_traj_predicted_error_cooling(
    physics: dict[str, Any], target_indoor: float
) -> float:
    traj = physics.get("_last_trajectory")
    if traj and traj.get("trajectory"):
        return float(traj["trajectory"][-1]) - target_indoor
    _, _, tau = _get_traj_params_cooling(physics)
    t_eq = _compute_traj_equilibrium_cooling(physics)
    indoor = physics.get("indoor_temp") or physics.get("indoor_temp_lag_30m")
    if indoor is None:
        return 0.0
    h = float(getattr(config, "TRAJECTORY_STEPS", 4))
    t_final = t_eq + (float(indoor) - t_eq) * math.exp(-h / tau)
    return t_final - target_indoor


def _compute_traj_convergence_rate_cooling(
    physics: dict[str, Any], target_indoor: float
) -> float:
    traj = physics.get("_last_trajectory")
    if traj and traj.get("trajectory"):
        temps = traj["trajectory"]
        if len(temps) >= 2:
            return (temps[0] - temps[-1]) / len(temps)
        return 0.0
    _, _, tau = _get_traj_params_cooling(physics)
    t_eq = _compute_traj_equilibrium_cooling(physics)
    indoor = physics.get("indoor_temp") or physics.get("indoor_temp_lag_30m")
    if indoor is None:
        return 0.0
    indoor = float(indoor)
    cycle_min = float(getattr(config, "CYCLE_INTERVAL_MINUTES", 10))
    dt_step = cycle_min / 60.0
    h = float(getattr(config, "TRAJECTORY_STEPS", 4))
    n_steps = max(1, int(h * 60 / cycle_min))
    step_1 = t_eq + (indoor - t_eq) * math.exp(-dt_step / tau)
    step_last = t_eq + (indoor - t_eq) * math.exp(-h / tau)
    return (step_1 - step_last) / n_steps


def _compute_traj_reaches_target_hours_cooling(
    physics: dict[str, Any], target_indoor: float
) -> float:
    traj = physics.get("_last_trajectory")
    if traj and traj.get("reaches_target_at") is not None:
        return float(traj["reaches_target_at"])
    _, _, tau = _get_traj_params_cooling(physics)
    t_eq = _compute_traj_equilibrium_cooling(physics)
    indoor = physics.get("indoor_temp") or physics.get("indoor_temp_lag_30m")
    if indoor is None:
        h = float(getattr(config, "TRAJECTORY_STEPS", 4))
        return h
    indoor = float(indoor)
    h = float(getattr(config, "TRAJECTORY_STEPS", 4))
    denom = indoor - t_eq
    if abs(denom) < 1e-6:
        return 0.0
    ratio = (target_indoor - t_eq) / denom
    if ratio <= 0 or ratio >= 1:
        return h
    t_reach = -tau * math.log(ratio)
    return max(0.0, min(h, t_reach))


def _compute_traj_overshoot_magnitude_cooling(
    physics: dict[str, Any], target_indoor: float
) -> float:
    """For cooling, overshoot = temp drops below target."""
    traj = physics.get("_last_trajectory")
    if traj and "min_predicted" in traj:
        return max(0.0, target_indoor - float(traj["min_predicted"]))
    t_eq = _compute_traj_equilibrium_cooling(physics)
    indoor = physics.get("indoor_temp") or physics.get("indoor_temp_lag_30m")
    if indoor is None:
        return 0.0
    trough = min(float(indoor), t_eq)
    return max(0.0, target_indoor - trough)


def _compute_traj_equilibrium_gap_cooling(
    physics: dict[str, Any], target_indoor: float
) -> float:
    traj = physics.get("_last_trajectory")
    if traj and "equilibrium_temp" in traj:
        return float(traj["equilibrium_temp"]) - target_indoor
    return _compute_traj_equilibrium_cooling(physics) - target_indoor


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_cooling_correction_feature(
    col: str,
    physics: dict[str, Any],
    target_indoor: float,
) -> float:
    """
    Map a single feature column name to a float value using the physics dict.
    Mirrors _extract_heating_feature but with cooling-specific parameters.
    """

    # ── Indoor-related features ────────────────────────────────────────
    if col == "indoor_margin":
        indoor = physics.get("indoor_temp")
        if indoor is None:
            indoor = physics.get("indoor_temp_lag_30m")
        if indoor is None:
            return 0.0
        return target_indoor - float(indoor)

    if col == "indoor_trend_30m":
        val = physics.get("indoor_temp_delta_30m")
        if val is not None:
            return float(val)
        return float(physics.get("indoor_temp_gradient", 0.0) or 0.0) * 0.5

    if col == "indoor_trend_1h":
        val = physics.get("indoor_temp_delta_1h")
        if val is not None:
            return float(val)
        return float(physics.get("indoor_temp_gradient", 0.0) or 0.0)

    # ── Outdoor / AT features ──────────────────────────────────────────
    if col == "AT":
        return float(physics.get("outdoor_temp", 0.0) or 0.0)

    if col == "at_delta_indoor":
        outdoor = physics.get("outdoor_temp")
        indoor = physics.get("indoor_temp")
        if indoor is None:
            indoor = physics.get("indoor_temp_lag_30m")
        if outdoor is None or indoor is None:
            return 0.0
        return float(outdoor) - float(indoor)

    # AT forecast
    m_at = re.match(r"AT_roh_(\d+)h", col)
    if m_at:
        h = int(m_at.group(1))
        val = physics.get(f"AT_roh_{h}h")
        if val is not None:
            return float(val)
        return float(physics.get("outdoor_temp", 0.0) or 0.0)

    # ── Hydraulic features ─────────────────────────────────────────────
    if col == "VLT":
        return float(physics.get("outlet_temp", 0.0) or 0.0)

    if col == "RLT":
        return float(physics.get("return_temp", 0.0) or 0.0)

    if col == "delta_t":
        return float(physics.get("delta_t", 0.0) or 0.0)

    if col == "outlet_indoor_diff":
        vlt = physics.get("outlet_temp")
        indoor = physics.get("indoor_temp")
        if indoor is None:
            indoor = physics.get("indoor_temp_lag_30m")
        if vlt is None or indoor is None:
            return 0.0
        return float(vlt) - float(indoor)

    if col == "thermal_power_w":
        val = physics.get("thermal_power_w")
        if val is not None:
            return float(val)
        val_kw = physics.get("thermal_power_kw")
        if val_kw is not None:
            return float(val_kw) * 1000.0
        return 0.0

    # ── Binary / categorical features ─────────────────────────────────
    if col == "fireplace_on":
        return float(physics.get("fireplace_on", 0.0) or 0.0)

    if col == "tv_on":
        return float(physics.get("tv_on", 0.0) or 0.0)

    # Fireplace / TV lag features → fall back to instantaneous flag
    m_fp = re.match(r"fireplace_lag_(\d+[hm])", col)
    if m_fp:
        return float(physics.get("fireplace_on", 0.0) or 0.0)

    m_tv = re.match(r"tv_lag_(\d+[hm])", col)
    if m_tv:
        return float(physics.get("tv_on", 0.0) or 0.0)

    # ── PV features ────────────────────────────────────────────────────
    if col == "PV_Generate":
        val = physics.get("pv_now_electrical")
        if val is None:
            val = physics.get("pv_now")
        return float(val) if val is not None else 0.0

    if col == "pv_roll_1h":
        return _pv_roll(physics, 1)

    if col == "pv_roll_2h":
        return _pv_roll(physics, 2)

    m_pv = re.match(r"pv_forecast_(\d+)h", col)
    if m_pv:
        h = int(m_pv.group(1))
        for key in [f"pv_forecast_electrical_{h}h", f"pv_forecast_{h}h"]:
            val = physics.get(key)
            if val is not None:
                return float(val)
        return 0.0

    # ── Cyclical time features ─────────────────────────────────────────
    if col == "hour_sin":
        val = physics.get("hour_sin")
        if val is not None:
            return float(val)
        now = datetime.now()
        h = now.hour + now.minute / 60.0
        return math.sin(2 * math.pi * h / 24.0)

    if col == "hour_cos":
        val = physics.get("hour_cos")
        if val is not None:
            return float(val)
        now = datetime.now()
        h = now.hour + now.minute / 60.0
        return math.cos(2 * math.pi * h / 24.0)

    if col == "doy_sin":
        return float(physics.get("doy_sin", _doy_sin()) or _doy_sin())

    if col == "doy_cos":
        return float(physics.get("doy_cos", _doy_cos()) or _doy_cos())

    # ── Standard ML features ──────────────────────────────────────────
    if col == "wind_speed":
        return float(physics.get("wind_speed", 0.0) or 0.0)

    if col == "indoor_temp_gradient":
        return float(physics.get("indoor_temp_gradient", 0.0) or 0.0)

    if col == "is_hp_active":
        dt = physics.get("delta_t")
        if dt is None:
            return 0.0
        return 1.0 if abs(float(dt)) > 1.0 else 0.0

    if col == "is_weekend":
        val = physics.get("is_weekend")
        if val is not None:
            return float(val)
        return 1.0 if datetime.now().weekday() >= 5 else 0.0

    if col == "thermal_power_rolling_1h":
        val = physics.get("thermal_power_rolling_1h")
        if val is not None:
            return float(val)
        val_w = physics.get("thermal_power_w")
        if val_w is not None:
            return float(val_w)
        val_kw = physics.get("thermal_power_kw")
        if val_kw is not None:
            return float(val_kw) * 1000.0
        return 0.0

    if col == "indoor_margin_rate":
        return float(physics.get("indoor_margin_rate", 0.0) or 0.0)

    if col == "is_overshoot":
        indoor = physics.get("indoor_temp")
        if indoor is None:
            indoor = physics.get("indoor_temp_lag_30m")
        if indoor is None:
            return 0.0
        return 1.0 if float(indoor) > target_indoor else 0.0

    # ── Slab thermal state features ───────────────────────────────────
    if col == "d_inlet_temp_60min":
        return float(physics.get("d_inlet_temp_60min", 0.0) or 0.0)

    if col == "is_equilibrium":
        val = physics.get("is_equilibrium")
        if val is not None:
            return float(val)
        d_inlet = physics.get("d_inlet_temp_60min")
        if d_inlet is not None:
            return 1.0 if abs(float(d_inlet)) < 0.3 else 0.0
        return 0.0

    # ── Physics-motivated features ────────────────────────────────────
    if col == "heat_loss_driving_force":
        indoor = physics.get("indoor_temp")
        if indoor is None:
            indoor = physics.get("indoor_temp_lag_30m")
        outdoor = physics.get("outdoor_temp")
        if indoor is None or outdoor is None:
            return 0.0
        return float(indoor) - float(outdoor)

    if col == "delta_T_indoor_lag1":
        return float(physics.get("indoor_temp_delta_10m", 0.0) or 0.0)

    if col == "Q_wp":
        val = physics.get("Q_wp")
        if val is not None:
            return float(val)
        flow = physics.get("flow_rate", 0.0) or 0.0
        dt = physics.get("delta_t", 0.0) or 0.0
        cp = float(getattr(config, "SPECIFIC_HEAT_CAPACITY", 4.186)) * 1000.0
        if float(flow) > 0:
            return (float(flow) / 60.0) * float(dt) * cp
        return 0.0

    if col == "solar_thermal_proxy":
        pv = physics.get("pv_now_electrical") or physics.get("pv_now") or 0.0
        now = datetime.now()
        h = now.hour + now.minute / 60.0
        hcos = math.cos(2 * math.pi * h / 24.0)
        return float(pv) * hcos

    if col == "pv_forecast_delta":
        pv_now = physics.get("pv_now_electrical") or physics.get("pv_now") or 0.0
        pv_2h = (
            physics.get("pv_forecast_electrical_2h")
            or physics.get("pv_forecast_2h")
            or pv_now
        )
        return float(pv_2h) - float(pv_now)

    # ── Physics interaction features ──────────────────────────────────
    if col == "shading_proxy":
        indoor = physics.get("indoor_temp")
        if indoor is None:
            indoor = physics.get("indoor_temp_lag_30m")
        if indoor is None:
            return 0.0
        pv = physics.get("pv_now_electrical") or physics.get("pv_now") or 0.0
        return max(0.0, float(indoor) - _SHADING_ACTIVATION_TEMP_C) * float(pv)

    if col == "heat_loss_interaction":
        indoor = physics.get("indoor_temp")
        if indoor is None:
            indoor = physics.get("indoor_temp_lag_30m")
        if indoor is None:
            return 0.0
        outdoor = physics.get("outdoor_temp")
        if outdoor is None:
            return 0.0
        wind = physics.get("wind_speed")
        if wind is None:
            return 0.0
        return (float(indoor) - float(outdoor)) * float(wind)

    # ── Trajectory-derived physics features ────────────────────────────
    if col == "traj_predicted_error":
        return _compute_traj_predicted_error_cooling(physics, target_indoor)
    if col == "traj_convergence_rate":
        return _compute_traj_convergence_rate_cooling(physics, target_indoor)
    if col == "traj_reaches_target_hours":
        return _compute_traj_reaches_target_hours_cooling(physics, target_indoor)
    if col == "traj_overshoot_magnitude":
        return _compute_traj_overshoot_magnitude_cooling(physics, target_indoor)
    if col == "traj_equilibrium_gap":
        return _compute_traj_equilibrium_gap_cooling(physics, target_indoor)

    # ── NB08/NB09-derived features ────────────────────────────────────
    if col == "cumulative_Q_wp_4h":
        return float(physics.get("cumulative_Q_wp_4h", 0.0) or 0.0)

    if col == "indoor_accel":
        return float(physics.get("indoor_accel", 0.0) or 0.0)

    if col == "AT_forecast_trend":
        val = physics.get("AT_forecast_trend")
        if val is not None:
            return float(val)
        outdoor = physics.get("outdoor_temp", 0.0) or 0.0
        for h in [4, 3, 2, 1]:
            fkey = f"AT_roh_{h}h"
            fval = physics.get(fkey)
            if fval is not None:
                return float(fval) - float(outdoor)
        return 0.0

    if col == "pv_cumulative_4h":
        return float(physics.get("pv_cumulative_4h", 0.0) or 0.0)

    if col == "thermal_momentum":
        val = physics.get("thermal_momentum")
        if val is not None:
            return float(val)
        tp = physics.get("thermal_power_rolling_1h", 0.0) or 0.0
        dt = physics.get("delta_t", 0.0) or 0.0
        return float(tp) * float(dt)

    logger.warning(
        "CoolingCorrectionMLModel: unknown feature column '%s', filling 0.0", col
    )
    return 0.0


def build_cooling_correction_feature_vector(
    feature_cols: list[str],
    physics: dict[str, Any],
    target_indoor: float,
) -> list[float]:
    """Construct a feature vector in ``feature_cols`` order."""
    return [
        _extract_cooling_correction_feature(col, physics, target_indoor)
        for col in feature_cols
    ]


# ---------------------------------------------------------------------------
# CoolingCorrectionMLModel
# ---------------------------------------------------------------------------

class CoolingCorrectionMLModel:
    """
    Thin wrapper around a joblib-serialised LightGBM regression model that
    predicts the required outlet-temperature correction (ΔT_outlet in °C)
    for cooling mode.

    Uses residualized label reconstruction:
        full_correction = model.predict(X) - indoor_margin / S_H_cooling
    """

    def __init__(self, model_path: str, metadata_path: str) -> None:
        self._model_path = model_path
        self._metadata_path = metadata_path

        self._model: Any = None
        self._metadata: dict[str, Any] = {}
        self._feature_cols: list[str] = []
        self._r2_score: float = 0.0
        self._label_type: str = ""
        self._s_h: float = 0.0
        self._loaded = False

    # ------------------------------------------------------------------
    # Load / reload
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load (or reload) model and metadata from disk.  Returns True on success."""
        joblib = _load_joblib()
        if not os.path.exists(self._model_path):
            logger.warning(
                "CoolingCorrectionMLModel: model file not found: %s. "
                "Run --calibrate-cooling-correction-ml to train.",
                self._model_path,
            )
            self._loaded = False
            return False
        if not os.path.exists(self._metadata_path):
            logger.warning(
                "CoolingCorrectionMLModel: metadata file not found: %s.",
                self._metadata_path,
            )
            self._loaded = False
            return False
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Trying to unpickle estimator",
                    category=UserWarning,
                )
                self._model = joblib.load(self._model_path)
            with open(self._metadata_path, "r", encoding="utf-8") as fh:
                self._metadata = json.load(fh)
            self._feature_cols = self._metadata.get("feature_cols", [])
            self._r2_score = float(self._metadata.get("val_r2", 0.0))
            self._label_type = self._metadata.get("label_type", "")
            self._s_h = float(self._metadata.get("s_h_estimated", 0.0))
            if self._label_type == "residualized" and self._s_h <= 0.05:
                logger.warning(
                    "CoolingCorrectionMLModel: label_type='residualized' but "
                    "S_H=%.4f (≤0.05) — residualized reconstruction will be skipped",
                    self._s_h,
                )
            self._loaded = True
            logger.info(
                "CoolingCorrectionMLModel: loaded %s | features=%d R²=%.4f MAE=%.4f",
                os.path.basename(self._model_path),
                len(self._feature_cols),
                self._r2_score,
                self._metadata.get("val_mae", float("nan")),
            )
            return True
        except Exception:
            logger.exception(
                "CoolingCorrectionMLModel: failed to load model from %s",
                self._model_path,
            )
            self._loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def r2_score(self) -> float:
        """Validation R² of the trained model (0.0 when not loaded)."""
        return self._r2_score

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        features: dict[str, Any],
        target_indoor: float,
    ) -> Optional[float]:
        """
        Predict the outlet-temperature correction delta [°C] for cooling.

        Parameters
        ----------
        features:      Physics features dict.
        target_indoor: Cooling target temperature [°C].

        Returns
        -------
        float or None
            Predicted ΔT_outlet [°C].
            Returns ``None`` if the model is not loaded or inference fails.
        """
        if not self._loaded:
            return None
        if not self._feature_cols:
            logger.warning(
                "CoolingCorrectionMLModel: feature_cols empty — metadata corrupt?"
            )
            return None
        try:
            np = _load_numpy()
            pd = _load_pandas()
            vec = build_cooling_correction_feature_vector(
                self._feature_cols, features, target_indoor
            )
            X = pd.DataFrame([vec], columns=self._feature_cols)
            delta: float = float(self._model.predict(X)[0])

            # Residualized label reconstruction:
            # adjusted_label = -(T_future - T_current) / S_H
            # original = adjusted + indoor_margin / S_H  (indoor_margin = target - indoor)
            if self._label_type == "residualized" and self._s_h > 0.05:
                indoor_margin = _extract_cooling_correction_feature(
                    "indoor_margin", features, target_indoor
                )
                delta = delta + indoor_margin / self._s_h
                logger.debug(
                    "CoolingCorrectionMLModel: residualized → "
                    "raw=%.3f margin=%.3f s_h=%.3f → full=%.3f°C",
                    float(self._model.predict(X)[0]),
                    indoor_margin, self._s_h, delta,
                )
            else:
                logger.debug(
                    "CoolingCorrectionMLModel: raw ΔT_outlet=%.3f°C (R²=%.4f)",
                    delta, self._r2_score,
                )
            return delta
        except Exception:
            logger.exception("CoolingCorrectionMLModel: inference failed")
            return None
