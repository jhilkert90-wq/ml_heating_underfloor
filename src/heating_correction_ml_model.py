"""
heating_correction_ml_model.py
--------------------------------
LightGBM-based outlet-temperature correction regressor for heating mode.

Mirrors the ``CoolingMLModel`` / ``cooling_ml_model.py`` pattern but predicts
a *regression target* (ΔT_outlet in °C) rather than a binary overheating risk.

Feature mapping
---------------
Translates the dict returned by ``build_physics_features()`` (stored as
``model_wrapper._current_features``) into the feature vector used at training
time.  The exact column order and names are read back from the metadata JSON,
so notebook-trained and service-trained models remain compatible.

Inference usage (inside ``model_wrapper._calculate_ml_correction()``)
----------------------------------------------------------------------
1.  ``HeatingCorrectionMLModel.predict(features, target_indoor)`` → float or None
2.  Caller blends the ML delta with the physics Newton delta using R² as weight.

Note on fireplace/TV lag features
----------------------------------
Rolling-max lag features are computed over history at *training* time.
At *inference* time the physics_features dict exposes only the instantaneous
``fireplace_on`` / ``tv_on`` flags.  As an approximation, any
``fireplace_lag_Xh``, ``fireplace_lag_Xm``, ``tv_lag_Xh``, or ``tv_lag_Xm``
column falls back to the corresponding instantaneous flag.  This is
conservative (slightly underpredicts residual heat after source turns off)
but safe for a first version.

Note on PV features
-------------------
``PV_Generate`` and ``pv_roll_Xh`` read ``pv_now_electrical`` (raw watts,
preferred) or ``pv_now`` (corrected thermal watts) from the physics dict.
``pv_forecast_Xh`` reads ``pv_forecast_electrical_Xh`` first, then falls
back to ``pv_forecast_Xh``, mirroring the cooling ML model.
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

# Indoor temperature threshold above which solar overheat protection (roller
# shutters / Jalousie) is assumed to be active – must match the value used
# in heating_correction_ml_calibration._SHADING_ACTIVATION_TEMP_C.
_SHADING_ACTIVATION_TEMP_C = 23.0


# ---------------------------------------------------------------------------
# Lazy-import helpers (joblib / numpy only needed when a model is loaded)
# ---------------------------------------------------------------------------

def _load_joblib():
    try:
        import joblib  # type: ignore
        return joblib
    except ImportError as exc:
        raise ImportError("joblib is required for HeatingCorrectionMLModel") from exc


def _load_numpy():
    try:
        import numpy as np  # type: ignore
        return np
    except ImportError as exc:
        raise ImportError("numpy is required for HeatingCorrectionMLModel") from exc


def _load_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except ImportError as exc:
        raise ImportError("pandas is required for HeatingCorrectionMLModel") from exc


# ---------------------------------------------------------------------------
# Day-of-year helpers (used when physics dict lacks cyclical time features)
# ---------------------------------------------------------------------------

def _doy_sin() -> float:
    doy = datetime.now().timetuple().tm_yday
    return math.sin(2 * math.pi * doy / 365.25)


def _doy_cos() -> float:
    doy = datetime.now().timetuple().tm_yday
    return math.cos(2 * math.pi * doy / 365.25)


# ---------------------------------------------------------------------------
# PV rolling helper (mirrors cooling_ml_model._pv_roll)
# ---------------------------------------------------------------------------

def _pv_roll(physics: dict[str, Any], hours: int, steps_per_hour: int = 6) -> float:
    """Return mean PV power over the last *hours* from the history list.

    Prefers ``pv_power_history_electrical`` (raw watts) and falls back to
    ``pv_power_history`` (thermally corrected watts) to match training scale.
    """
    history = physics.get("pv_power_history_electrical") or physics.get(
        "pv_power_history"
    )
    if not history:
        # No history available: use instantaneous PV as best approximation
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
# At inference the previous cycle's trajectory result may be stored as
# ``physics["_last_trajectory"]`` (dict with keys: trajectory, times,
# reaches_target_at, max_predicted, min_predicted, equilibrium_temp,
# final_error).  When unavailable, we compute a lightweight analytical
# approximation using the current thermal parameters.

def _get_traj_params(physics: dict[str, Any]) -> tuple[float, float, float]:
    """Return (eta, u_loss, tau) for analytical trajectory approximation."""
    eta = float(getattr(config, "OUTLET_EFFECTIVENESS", 0.830))
    u_loss = float(getattr(config, "HEAT_LOSS_COEFFICIENT", 0.124))
    tau = float(getattr(config, "THERMAL_TIME_CONSTANT", 4.39))
    return eta, u_loss, tau


def _compute_traj_equilibrium(physics: dict[str, Any]) -> float:
    """Compute T_eq = (η×VLT + U×AT) / (η + U)."""
    eta, u_loss, tau = _get_traj_params(physics)
    vlt = physics.get("outlet_temp")
    at = physics.get("outdoor_temp")
    if vlt is None or at is None:
        return 0.0
    denom = eta + u_loss
    if denom < 1e-6:
        return float(vlt)
    return (eta * float(vlt) + u_loss * float(at)) / denom


def _compute_traj_predicted_error(
    physics: dict[str, Any], target_indoor: float
) -> float:
    """Trajectory final temp − target: how far physics expects to miss."""
    traj = physics.get("_last_trajectory")
    if traj and traj.get("trajectory"):
        return float(traj["trajectory"][-1]) - target_indoor
    # Analytical fallback: T(H) = T_eq + (T_indoor - T_eq) × exp(-H/τ)
    _, _, tau = _get_traj_params(physics)
    t_eq = _compute_traj_equilibrium(physics)
    indoor = physics.get("indoor_temp") or physics.get("indoor_temp_lag_30m")
    if indoor is None:
        return 0.0
    h = float(getattr(config, "TRAJECTORY_STEPS", 4))
    t_final = t_eq + (float(indoor) - t_eq) * math.exp(-h / tau)
    return t_final - target_indoor


def _compute_traj_convergence_rate(
    physics: dict[str, Any], target_indoor: float
) -> float:
    """(step_1 − step_last) / n_steps: speed of approach to equilibrium."""
    traj = physics.get("_last_trajectory")
    if traj and traj.get("trajectory"):
        temps = traj["trajectory"]
        if len(temps) >= 2:
            return (temps[0] - temps[-1]) / len(temps)
        return 0.0
    # Analytical fallback
    _, _, tau = _get_traj_params(physics)
    t_eq = _compute_traj_equilibrium(physics)
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


def _compute_traj_reaches_target_hours(
    physics: dict[str, Any], target_indoor: float
) -> float:
    """Analytical time to reach target (capped at horizon)."""
    traj = physics.get("_last_trajectory")
    if traj and traj.get("reaches_target_at") is not None:
        return float(traj["reaches_target_at"])
    # Analytical: t = -τ × ln((T_target - T_eq) / (T_indoor - T_eq))
    _, _, tau = _get_traj_params(physics)
    t_eq = _compute_traj_equilibrium(physics)
    indoor = physics.get("indoor_temp") or physics.get("indoor_temp_lag_30m")
    if indoor is None:
        h = float(getattr(config, "TRAJECTORY_STEPS", 4))
        return h
    indoor = float(indoor)
    h = float(getattr(config, "TRAJECTORY_STEPS", 4))
    denom = indoor - t_eq
    if abs(denom) < 1e-6:
        return 0.0  # already at equilibrium
    ratio = (target_indoor - t_eq) / denom
    if ratio <= 0 or ratio >= 1:
        return h  # target unreachable within horizon
    t_reach = -tau * math.log(ratio)
    return max(0.0, min(h, t_reach))


def _compute_traj_overshoot_magnitude(
    physics: dict[str, Any], target_indoor: float
) -> float:
    """max(0, max_predicted − target): predicted overshoot magnitude."""
    traj = physics.get("_last_trajectory")
    if traj and "max_predicted" in traj:
        return max(0.0, float(traj["max_predicted"]) - target_indoor)
    # Analytical: max(T_indoor, T_eq) − target, clipped to ≥ 0
    t_eq = _compute_traj_equilibrium(physics)
    indoor = physics.get("indoor_temp") or physics.get("indoor_temp_lag_30m")
    if indoor is None:
        return 0.0
    peak = max(float(indoor), t_eq)
    return max(0.0, peak - target_indoor)


def _compute_traj_equilibrium_gap(
    physics: dict[str, Any], target_indoor: float
) -> float:
    """T_eq − T_target: steady-state error signal."""
    traj = physics.get("_last_trajectory")
    if traj and "equilibrium_temp" in traj:
        return float(traj["equilibrium_temp"]) - target_indoor
    return _compute_traj_equilibrium(physics) - target_indoor


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_heating_feature(
    col: str,
    physics: dict[str, Any],
    target_indoor: float,
) -> float:
    """Map a single feature column name to its value from the physics dict.

    Parameters
    ----------
    col:          Feature column name (as stored in model metadata).
    physics:      Dict returned by ``build_physics_features()`` (i.e.
                  ``model_wrapper._current_features``).
    target_indoor: Current heating target temperature [°C].

    Returns
    -------
    float: Feature value; 0.0 for any unknown column (with a warning).
    """
    # ── Temperature scalars ────────────────────────────────────────────
    if col == "indoor_temp":
        # build_physics_features() stores the current indoor temperature as
        # "indoor_temp_lag_30m" (the 30-minute-smoothed lag value used as the
        # prediction baseline) but does NOT create a plain "indoor_temp" key.
        # Fall back to that lag key so inference always produces the real
        # indoor temperature rather than a spurious 0.0.
        val = physics.get("indoor_temp")
        if val is None:
            val = physics.get("indoor_temp_lag_30m")
        return float(val) if val is not None else 0.0
    if col == "indoor_margin":
        # Positive when room is too cold (target > indoor), negative when warm.
        # Same key-fallback as "indoor_temp" above.
        indoor = physics.get("indoor_temp")
        if indoor is None:
            indoor = physics.get("indoor_temp_lag_30m")
        indoor = float(indoor) if indoor is not None else 0.0
        return target_indoor - indoor
    if col == "indoor_trend_30m":
        return float(physics.get("indoor_temp_delta_30m") or 0.0)
    if col == "indoor_trend_1h":
        return float(physics.get("indoor_temp_delta_60m") or 0.0)
    if col == "AT":
        return float(physics.get("outdoor_temp") or 0.0)
    if col == "at_delta_indoor":
        # AT − indoor = −temp_diff_indoor_outdoor
        return -float(physics.get("temp_diff_indoor_outdoor") or 0.0)
    if col == "VLT":
        return float(physics.get("outlet_temp") or 0.0)
    if col == "RLT":
        return float(physics.get("inlet_temp") or 0.0)
    if col == "delta_t":
        return float(physics.get("delta_t") or 0.0)
    if col == "outlet_indoor_diff":
        return float(physics.get("outlet_indoor_diff") or 0.0)
    if col == "thermal_power_w":
        # thermal_power_kw (from physics dict) converted to W
        return float(physics.get("thermal_power_kw") or 0.0) * 1000.0

    # ── External heat sources ──────────────────────────────────────────
    if col == "fireplace_on":
        return float(physics.get("fireplace_on") or 0.0)
    if col == "tv_on":
        return float(physics.get("tv_on") or 0.0)

    # Dynamic fireplace lag: fireplace_lag_Xh or fireplace_lag_Xm
    # → approximated at inference as the instantaneous fireplace_on flag
    if re.fullmatch(r"fireplace_lag_\d+[hm]", col):
        return float(physics.get("fireplace_on") or 0.0)

    # Dynamic TV lag: tv_lag_Xh or tv_lag_Xm
    # → approximated at inference as the instantaneous tv_on flag
    if re.fullmatch(r"tv_lag_\d+[hm]", col):
        return float(physics.get("tv_on") or 0.0)

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

    # Dynamic PV forecast: pv_forecast_Xh
    m_pv = re.fullmatch(r"pv_forecast_(\d+)h", col)
    if m_pv:
        h = m_pv.group(1)
        val = physics.get(f"pv_forecast_electrical_{h}h")
        if val is None:
            val = physics.get(f"pv_forecast_{h}h")
        return float(val) if val is not None else 0.0

    # ── Cyclical time features ─────────────────────────────────────────
    if col == "hour_sin":
        return float(physics.get("hour_sin") or 0.0)
    if col == "hour_cos":
        return float(physics.get("hour_cos") or 0.0)
    if col == "doy_sin":
        return _doy_sin()
    if col == "doy_cos":
        return _doy_cos()

    # ── Dynamic AT forecast: AT_roh_Xh → temp_forecast_Xh ─────────────
    m_at = re.fullmatch(r"AT_roh_(\d+)h", col)
    if m_at:
        h = m_at.group(1)
        return float(
            physics.get(f"temp_forecast_{h}h")
            or physics.get("outdoor_temp")
            or 0.0
        )

    # ── NEW: 8 additional ML correction features ──────────────────────
    if col == "wind_speed":
        # PI=0.0000 — linear term; predictive value captured by heat_loss_interaction; retained for backward compat
        return float(physics.get("wind_speed") or 0.0)
    if col == "indoor_temp_gradient":
        # PI=0.0000
        return float(physics.get("indoor_temp_gradient") or 0.0)
    if col == "living_room_temp":
        val = physics.get("living_room_temp")
        if val is None:
            val = physics.get("indoor_temp") or physics.get("indoor_temp_lag_30m")
        return float(val) if val is not None else 0.0
    if col == "is_hp_active":
        dt = physics.get("delta_t")
        if dt is not None:
            return 1.0 if abs(float(dt)) > 1.0 else 0.0
        return 0.0
    if col == "is_weekend":
        return float(physics.get("is_weekend") or 0.0)
    if col == "thermal_power_rolling_1h":
        # At inference we only have the instantaneous value; convert kW → W
        return float(physics.get("thermal_power_kw") or 0.0) * 1000.0
    if col == "indoor_margin_rate":
        # PI=0.0000
        return float(physics.get("indoor_margin_rate") or 0.0)
    if col == "is_overshoot":
        # PI=0.0000
        indoor = physics.get("indoor_temp") or physics.get("indoor_temp_lag_30m")
        if indoor is not None:
            return 1.0 if float(indoor) > target_indoor else 0.0
        return 0.0

    # ── Slab thermal state features ────────────────────────────────────
    if col == "d_inlet_temp_60min":
        # PI=0.0000
        return float(physics.get("d_inlet_temp_60min") or 0.0)
    if col == "is_equilibrium":
        # PI=0.0000
        return float(physics.get("is_equilibrium") or 0.0)

    # ── physics-motivated features ─────────────────────────────────────
    if col == "heat_loss_driving_force":
        # Newton's law: primary heat loss driver is T_indoor − T_outdoor
        indoor = physics.get("indoor_temp")
        if indoor is None:
            indoor = physics.get("indoor_temp_lag_30m")
        indoor = float(indoor) if indoor is not None else 0.0
        outdoor = float(physics.get("outdoor_temp") or 0.0)
        return indoor - outdoor
    if col == "delta_T_indoor_lag1":
        # PI=0.0000 — AR momentum: ΔT over 1 cycle (10 min) captures thermal inertia
        return float(physics.get("indoor_temp_delta_10m") or 0.0)
    if col == "Q_wp":
        # PI=0.0000 — Actual heat output in W: flow (L/min → L/s) × ΔT × c_p
        flow_lpm = physics.get("flow_rate")
        if not flow_lpm:
            return 0.0
        vlt = physics.get("outlet_temp")
        rlt = physics.get("inlet_temp")
        if vlt is None or rlt is None:
            return 0.0
        specific_heat_kj_per_kgk = float(
            getattr(config, "SPECIFIC_HEAT_CAPACITY", 4.186)
        )
        specific_heat_j_per_kgk = specific_heat_kj_per_kgk * 1000.0
        return (
            (float(flow_lpm) / 60.0)
            * (float(vlt) - float(rlt))
            * specific_heat_j_per_kgk
        )
    if col == "solar_thermal_proxy":
        # PI=-0.0000 — Passive solar gain proxy: PV power × cos(hour) encodes sun angle
        pv = physics.get("pv_now_electrical")
        if pv is None:
            pv = physics.get("pv_now")
        pv = float(pv) if pv is not None else 0.0
        cos_hour = float(physics.get("hour_cos") or 0.0)
        return pv * cos_hour
    if col == "pv_forecast_delta":
        # Anticipatory solar: upcoming PV gain minus current (floor slab lag ~60–90 min)
        pv = physics.get("pv_now_electrical")
        if pv is None:
            pv = physics.get("pv_now")
        pv = float(pv) if pv is not None else 0.0
        fc = physics.get("pv_forecast_electrical_2h")
        if fc is None:
            fc = physics.get("pv_forecast_2h")
        if fc is None:
            return 0.0
        return float(fc) - pv

    # ── New physics interaction features ───────────────────────────────
    if col == "shading_proxy":
        # Continuous solar overheat-protection proxy: T_indoor > 23°C × PV_W
        # Captures roller-shutter / Jalousie activation during Übergangszeit; units: K×W
        indoor = physics.get("indoor_temp")
        if indoor is None:
            indoor = physics.get("indoor_temp_lag_30m")
        if indoor is None:
            return 0.0
        pv = physics.get("pv_now_electrical")
        if pv is None:
            pv = physics.get("pv_now")
        pv = float(pv) if pv is not None else 0.0
        return max(0.0, float(indoor) - _SHADING_ACTIVATION_TEMP_C) * pv
    if col == "heat_loss_interaction":
        # Convective heat loss interaction: (T_indoor − T_outdoor) × wind_speed
        # Wind increases effective U; effect scales with temperature gradient
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
    # At inference time these are computed from the previous cycle's
    # trajectory result (stored as _last_trajectory dict in the physics dict)
    # or from a lightweight analytical approximation using current conditions.
    if col == "traj_predicted_error":
        return _compute_traj_predicted_error(physics, target_indoor)
    if col == "traj_convergence_rate":
        return _compute_traj_convergence_rate(physics, target_indoor)
    if col == "traj_reaches_target_hours":
        return _compute_traj_reaches_target_hours(physics, target_indoor)
    if col == "traj_overshoot_magnitude":
        return _compute_traj_overshoot_magnitude(physics, target_indoor)
    if col == "traj_equilibrium_gap":
        return _compute_traj_equilibrium_gap(physics, target_indoor)

    # ── NB08-derived features ──────────────────────────────────────────
    if col == "cumulative_Q_wp_4h":
        return float(physics.get("cumulative_Q_wp_4h", 0.0) or 0.0)

    if col == "indoor_accel":
        return float(physics.get("indoor_accel", 0.0) or 0.0)

    if col == "AT_forecast_trend":
        val = physics.get("AT_forecast_trend")
        if val is not None:
            return float(val)
        # Fallback: max AT forecast minus current AT
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
        # Fallback: compute from available components
        tp = physics.get("thermal_power_rolling_1h", 0.0) or 0.0
        dt = physics.get("delta_t", 0.0) or 0.0
        return float(tp) * float(dt)

    logger.warning(
        "HeatingCorrectionMLModel: unknown feature column '%s', filling 0.0", col
    )
    return 0.0


def build_heating_feature_vector(
    feature_cols: list[str],
    physics: dict[str, Any],
    target_indoor: float,
) -> list[float]:
    """Construct a feature vector in ``feature_cols`` order."""
    return [
        _extract_heating_feature(col, physics, target_indoor)
        for col in feature_cols
    ]


# ---------------------------------------------------------------------------
# HeatingCorrectionMLModel
# ---------------------------------------------------------------------------

class HeatingCorrectionMLModel:
    """
    Thin wrapper around a joblib-serialised LightGBM regression model that
    predicts the required outlet-temperature correction (ΔT_outlet in °C).

    Exposes ``predict()`` which returns the raw ML correction delta.  The
    caller (``model_wrapper._calculate_ml_correction``) blends this with the
    physics Newton step using the model's R² score as the blend weight.
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
                "HeatingCorrectionMLModel: model file not found: %s. "
                "Run --calibrate-heating-correction-ml to train.",
                self._model_path,
            )
            self._loaded = False
            return False
        if not os.path.exists(self._metadata_path):
            logger.warning(
                "HeatingCorrectionMLModel: metadata file not found: %s.",
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
                    "HeatingCorrectionMLModel: label_type='residualized' but "
                    "S_H=%.4f (≤0.05) — residualized reconstruction will be skipped",
                    self._s_h,
                )
            self._loaded = True
            logger.info(
                "HeatingCorrectionMLModel: loaded %s | features=%d R²=%.4f MAE=%.4f",
                os.path.basename(self._model_path),
                len(self._feature_cols),
                self._r2_score,
                self._metadata.get("val_mae", float("nan")),
            )
            return True
        except Exception:
            logger.exception(
                "HeatingCorrectionMLModel: failed to load model from %s",
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
        Predict the outlet-temperature correction delta [°C].

        Parameters
        ----------
        features:      Physics features dict (``model_wrapper._current_features``).
        target_indoor: Heating target temperature [°C].

        Returns
        -------
        float or None
            Predicted ΔT_outlet [°C] (positive = raise outlet, negative = lower).
            Returns ``None`` if the model is not loaded or inference fails.
        """
        if not self._loaded:
            return None
        if not self._feature_cols:
            logger.warning(
                "HeatingCorrectionMLModel: feature_cols empty — metadata corrupt?"
            )
            return None
        try:
            np = _load_numpy()
            pd = _load_pandas()
            vec = build_heating_feature_vector(
                self._feature_cols, features, target_indoor
            )
            X = pd.DataFrame([vec], columns=self._feature_cols)
            delta: float = float(self._model.predict(X)[0])

            # Residualized label reconstruction:
            # Model was trained on adjusted_label = -(T_future - T_current) / S_H.
            # The original (full) label is: -(T_future - T_target) / S_H.
            # With indoor_margin = target - indoor:
            #   original = adjusted + indoor_margin / S_H
            # So: full_correction = delta + indoor_margin / S_H
            if self._label_type == "residualized" and self._s_h > 0.05:
                indoor_margin = _extract_heating_feature(
                    "indoor_margin", features, target_indoor
                )
                delta = delta + indoor_margin / self._s_h
                logger.debug(
                    "HeatingCorrectionMLModel: residualized → "
                    "raw=%.3f margin=%.3f s_h=%.3f → full=%.3f°C",
                    float(self._model.predict(X)[0]),
                    indoor_margin, self._s_h, delta,
                )
            else:
                logger.debug(
                    "HeatingCorrectionMLModel: raw ΔT_outlet=%.3f°C (R²=%.4f)",
                    delta, self._r2_score,
                )
            return delta
        except Exception:
            logger.exception("HeatingCorrectionMLModel: inference failed")
            return None
