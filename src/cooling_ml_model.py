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


# Indoor temperature threshold above which solar overheat protection
# (roller shutters / Jalousie) is assumed active.
_SHADING_ACTIVATION_TEMP_C = 23.0


# ---------------------------------------------------------------------------
# Trajectory-derived feature helpers (inference time)
# ---------------------------------------------------------------------------
# Mirrors heating_correction_ml_model._compute_traj_* helpers.
# At inference the OverheatingPredictor's trajectory result is injected
# as ``physics["_last_trajectory"]`` by cycle_routes.py.

def _get_traj_params() -> tuple[float, float, float]:
    """Return (eta, u_loss, tau) for analytical trajectory approximation."""
    try:
        from src import config as _cfg
    except ImportError:
        import config as _cfg  # type: ignore
    eta = float(getattr(_cfg, "OUTLET_EFFECTIVENESS", 0.830))
    u_loss = float(getattr(_cfg, "HEAT_LOSS_COEFFICIENT", 0.124))
    tau = float(getattr(_cfg, "THERMAL_TIME_CONSTANT", 4.39))
    return eta, u_loss, tau


def _compute_traj_equilibrium(physics: dict[str, Any]) -> float:
    """Compute T_eq = (η×VLT + U×AT) / (η + U)."""
    eta, u_loss, _ = _get_traj_params()
    vlt = physics.get("outlet_temp")
    at = physics.get("outdoor_temp")
    if vlt is None or at is None:
        return 0.0
    denom = eta + u_loss
    if denom < 1e-6:
        return float(vlt)
    return (eta * float(vlt) + u_loss * float(at)) / denom


def _compute_traj_predicted_error(
    physics: dict[str, Any], target: float
) -> float:
    """Trajectory final temp − target: how far physics expects to miss."""
    traj = physics.get("_last_trajectory")
    if traj and traj.get("trajectory"):
        return float(traj["trajectory"][-1]) - target
    # Analytical fallback
    _, _, tau = _get_traj_params()
    t_eq = _compute_traj_equilibrium(physics)
    indoor = physics.get("indoor_temp") or physics.get("indoor_temp_lag_30m")
    if indoor is None:
        return 0.0
    try:
        from src import config as _cfg
    except ImportError:
        import config as _cfg  # type: ignore
    h = float(getattr(_cfg, "TRAJECTORY_STEPS", 4))
    t_final = t_eq + (float(indoor) - t_eq) * math.exp(-h / tau)
    return t_final - target


def _compute_traj_convergence_rate(
    physics: dict[str, Any], target: float
) -> float:
    """(step_1 − step_last) / n_steps: speed of approach to equilibrium."""
    traj = physics.get("_last_trajectory")
    if traj and traj.get("trajectory"):
        temps = traj["trajectory"]
        if len(temps) >= 2:
            return (temps[0] - temps[-1]) / len(temps)
        return 0.0
    _, _, tau = _get_traj_params()
    t_eq = _compute_traj_equilibrium(physics)
    indoor = physics.get("indoor_temp") or physics.get("indoor_temp_lag_30m")
    if indoor is None:
        return 0.0
    indoor = float(indoor)
    try:
        from src import config as _cfg
    except ImportError:
        import config as _cfg  # type: ignore
    cycle_min = float(getattr(_cfg, "CYCLE_INTERVAL_MINUTES", 10))
    dt_step = cycle_min / 60.0
    h = float(getattr(_cfg, "TRAJECTORY_STEPS", 4))
    n_steps = max(1, int(h * 60 / cycle_min))
    step_1 = t_eq + (indoor - t_eq) * math.exp(-dt_step / tau)
    step_last = t_eq + (indoor - t_eq) * math.exp(-h / tau)
    return (step_1 - step_last) / n_steps


def _compute_traj_reaches_target_hours(
    physics: dict[str, Any], target: float
) -> float:
    """Analytical time to reach target (capped at horizon)."""
    traj = physics.get("_last_trajectory")
    if traj and traj.get("reaches_target_at") is not None:
        return float(traj["reaches_target_at"])
    _, _, tau = _get_traj_params()
    t_eq = _compute_traj_equilibrium(physics)
    indoor = physics.get("indoor_temp") or physics.get("indoor_temp_lag_30m")
    try:
        from src import config as _cfg
    except ImportError:
        import config as _cfg  # type: ignore
    h = float(getattr(_cfg, "TRAJECTORY_STEPS", 4))
    if indoor is None:
        return h
    indoor = float(indoor)
    denom = indoor - t_eq
    if abs(denom) < 1e-6:
        return 0.0
    ratio = (target - t_eq) / denom
    if ratio <= 0 or ratio >= 1:
        return h
    t_reach = -tau * math.log(ratio)
    return max(0.0, min(h, t_reach))


def _compute_traj_overshoot_magnitude(
    physics: dict[str, Any], target: float
) -> float:
    """max(0, max_predicted − target): predicted overshoot magnitude."""
    traj = physics.get("_last_trajectory")
    if traj and "max_predicted" in traj:
        return max(0.0, float(traj["max_predicted"]) - target)
    t_eq = _compute_traj_equilibrium(physics)
    indoor = physics.get("indoor_temp") or physics.get("indoor_temp_lag_30m")
    if indoor is None:
        return 0.0
    peak = max(float(indoor), t_eq)
    return max(0.0, peak - target)


def _compute_traj_equilibrium_gap(
    physics: dict[str, Any], target: float
) -> float:
    """T_eq − T_target: steady-state error signal."""
    traj = physics.get("_last_trajectory")
    if traj and "equilibrium_temp" in traj:
        return float(traj["equilibrium_temp"]) - target
    return _compute_traj_equilibrium(physics) - target


def build_feature_vector(
    feature_cols: list[str],
    physics: dict[str, Any],
    current_indoor: float,
    cooling_target: float,
    steps_per_hour: int = 6,
    label_horizon_h: int = 8,
) -> list[float]:
    """
    Construct a feature vector (in ``feature_cols`` order) from
    ``physics_features`` dict, current indoor temp, and cooling target.

    Unknown columns are filled with 0.0 and logged as a warning.
    """
    vec: list[float] = []
    for col in feature_cols:
        val = _extract_feature(col, physics, current_indoor, cooling_target, steps_per_hour, label_horizon_h)
        vec.append(val)
    return vec


def _extract_feature(
    col: str,
    physics: dict[str, Any],
    current_indoor: float,
    cooling_target: float,
    steps_per_hour: int,
    label_horizon_h: int = 8,
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
            at_h = physics.get(f"temp_forecast_{h}h")
            if at_h is None:
                at_h = physics.get("outdoor_temp")
            if at_h is None:
                at_h = 0.0
            total += max(0.0, float(at_h) - cooling_target)
        return total

    if col == "max_at_forecast":
        max_at = physics.get("outdoor_temp")
        max_at = float(max_at) if max_at is not None else 0.0
        # Look ahead up to label_horizon_h (derived from training metadata)
        for h in range(1, label_horizon_h + 1):
            at_h = physics.get(f"temp_forecast_{h}h")
            if at_h is not None and float(at_h) > max_at:
                max_at = float(at_h)
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

    # ── HA context features ────────────────────────────────────────────
    if col == "wind_speed":
        return float(physics.get("wind_speed") or 0.0)
    if col == "living_room_temp":
        val = physics.get("living_room_temp")
        if val is None:
            val = physics.get("indoor_temp") or physics.get("indoor_temp_lag_30m")
        return float(val) if val is not None else current_indoor
    if col == "fireplace_on":
        return float(physics.get("fireplace_on") or 0.0)
    if col == "tv_on":
        return float(physics.get("tv_on") or 0.0)

    # Dynamic fireplace/TV lag features → approximate with instantaneous flag
    if re.fullmatch(r"fireplace_lag_\d+[hm]", col):
        return float(physics.get("fireplace_on") or 0.0)
    if re.fullmatch(r"tv_lag_\d+[hm]", col):
        return float(physics.get("tv_on") or 0.0)

    # ── Derived physics features ───────────────────────────────────────
    if col == "heat_loss_driving_force":
        outdoor = float(physics.get("outdoor_temp") or 0.0)
        return current_indoor - outdoor
    if col == "indoor_temp_gradient":
        return float(physics.get("indoor_temp_gradient") or 0.0)
    if col == "indoor_margin_rate":
        return float(physics.get("indoor_margin_rate") or 0.0)
    if col == "delta_T_indoor_lag1":
        return float(physics.get("indoor_temp_delta_10m") or 0.0)
    if col == "d_inlet_temp_60min":
        return float(physics.get("d_inlet_temp_60min") or 0.0)
    if col == "is_equilibrium":
        return float(physics.get("is_equilibrium") or 0.0)
    if col == "thermal_power_rolling_1h":
        return float(physics.get("thermal_power_kw") or 0.0)
    if col == "is_overshoot":
        return 1.0 if current_indoor > cooling_target else 0.0
    if col == "is_hp_active":
        dt = physics.get("delta_t")
        if dt is not None:
            return 1.0 if abs(float(dt)) > 1.0 else 0.0
        return 0.0
    if col == "is_weekend":
        return float(physics.get("is_weekend") or 0.0)
    if col == "heat_loss_interaction":
        outdoor = physics.get("outdoor_temp")
        wind = physics.get("wind_speed")
        if outdoor is None or wind is None:
            return 0.0
        return (current_indoor - float(outdoor)) * float(wind)

    # ── Solar / shading features ───────────────────────────────────────
    if col == "solar_thermal_proxy":
        pv = physics.get("pv_now_electrical")
        if pv is None:
            pv = physics.get("pv_now")
        pv = float(pv) if pv is not None else 0.0
        cos_hour = float(physics.get("hour_cos") or 0.0)
        return pv * cos_hour
    if col == "shading_proxy":
        pv = physics.get("pv_now_electrical")
        if pv is None:
            pv = physics.get("pv_now")
        pv = float(pv) if pv is not None else 0.0
        return max(0.0, current_indoor - _SHADING_ACTIVATION_TEMP_C) * pv
    if col == "pv_forecast_delta":
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

    # ── Trajectory-derived physics features ────────────────────────────
    if col == "traj_predicted_error":
        return _compute_traj_predicted_error(physics, cooling_target)
    if col == "traj_convergence_rate":
        return _compute_traj_convergence_rate(physics, cooling_target)
    if col == "traj_reaches_target_hours":
        return _compute_traj_reaches_target_hours(physics, cooling_target)
    if col == "traj_overshoot_magnitude":
        return _compute_traj_overshoot_magnitude(physics, cooling_target)
    if col == "traj_equilibrium_gap":
        return _compute_traj_equilibrium_gap(physics, cooling_target)

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
        self._reg_model: Any = None
        self._metadata: dict[str, Any] = {}
        self._feature_cols: list[str] = []
        self._threshold: float = 0.5
        self._reg_threshold: float = 23.0
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
            self._threshold = float(self._metadata.get("threshold", 0.5))
            self._reg_threshold = float(
                self._metadata.get("regression_threshold", 23.0)
            )
            self._loaded = True

            # Load regression model if available (dual-output mode)
            self._reg_model = None
            reg_path = self._metadata.get("reg_model_path", "")
            if not reg_path:
                # Derive from classifier path
                _derived = self._model_path.replace(
                    "cooling_ml_model.joblib", "cooling_ml_regressor.joblib"
                )
                # Only use derived path if the replacement actually changed something
                if _derived != self._model_path:
                    reg_path = _derived
            if os.path.exists(reg_path):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="Trying to unpickle estimator",
                            category=UserWarning,
                        )
                        self._reg_model = joblib.load(reg_path)
                    logger.info(
                        "CoolingMLModel: regressor loaded from %s "
                        "(threshold=%.2f°C MAE=%.4f)",
                        os.path.basename(reg_path),
                        self._reg_threshold,
                        self._metadata.get("regression_mae", float("nan")),
                    )
                except Exception:
                    logger.warning(
                        "CoolingMLModel: failed to load regressor from %s "
                        "— classifier-only mode",
                        reg_path,
                    )

            logger.info(
                "CoolingMLModel: loaded %s | features=%d threshold=%.4f AUC=%.4f"
                " | regressor=%s",
                os.path.basename(self._model_path),
                len(self._feature_cols),
                self._threshold,
                self._metadata.get("roc_auc", float("nan")),
                "yes" if self._reg_model is not None else "no",
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
            import pandas as pd
            label_horizon_h = int(self._metadata.get("label_horizon_h", 8))
            vec = build_feature_vector(
                self._feature_cols,
                features,
                current_indoor,
                target_cooling,
                self._steps_per_hour,
                label_horizon_h,
            )
            X = pd.DataFrame([vec], columns=self._feature_cols)
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

        # ── Regression prediction (dual-output) ───────────────────────
        delta_pred = 0.0
        predicted_max = current_indoor
        reg_risk = False
        if self._reg_model is not None:
            try:
                delta_pred = float(self._reg_model.predict(X)[0])
                predicted_max = current_indoor + delta_pred
                reg_risk = predicted_max > self._reg_threshold
            except Exception:
                logger.debug("CoolingMLModel: regression prediction failed")

        # ── should_cool_now logic (dual-output strategy) ─────────────
        dual_strategy = str(
            getattr(config, "PRE_COOL_DUAL_OUTPUT_STRATEGY", "classifier_gate")
        )
        should_cool_now = False
        reason_parts: list[str] = [f"LGBM p={proba:.3f} (thr={self._threshold:.3f})"]
        if self._reg_model is not None:
            reason_parts.append(
                f"reg Δ={delta_pred:+.2f} max={predicted_max:.1f}°C"
            )

        if dual_strategy not in ("classifier_gate", "either_triggers"):
            raise ValueError(
                f"Invalid PRE_COOL_DUAL_OUTPUT_STRATEGY: {dual_strategy!r}. "
                "Must be 'classifier_gate' or 'either_triggers'."
            )

        reactive = current_indoor > target_cooling
        if reactive:
            should_cool_now = True
            reason_parts.append(
                f"room {current_indoor:.1f}°C > target {target_cooling:.1f}°C (reactive)"
            )
        elif self._reg_model is not None:
            # Dual-output decision
            if dual_strategy == "either_triggers":
                # Aggressive: either model can trigger pre-cooling
                should_cool_now = risk or reg_risk
            else:
                # classifier_gate (default): classifier must confirm risk;
                # regression provides intensity only
                should_cool_now = risk

            if risk and not reg_risk:
                reason_parts.append("classifier⬆ reg⬇ (disagree)")
            elif not risk and reg_risk:
                reason_parts.append("classifier⬇ reg⬆ (disagree)")

            if should_cool_now:
                reason_parts.append(
                    f"dual-output [{dual_strategy}] → pre-cool"
                )
            else:
                reason_parts.append("no overheating risk predicted")
        elif risk:
            # Classifier-only mode (no regression model)
            should_cool_now = True
            reason_parts.append(
                f"model predicts overheating (p={proba:.3f} > {self._threshold:.3f})"
            )
        else:
            reason_parts.append("no overheating risk predicted")

        peak_hour = (lead_time / 2.0) if risk else horizon_h
        # Use regression prediction for peak_temp when available
        if self._reg_model is not None:
            peak_temp = predicted_max
        else:
            peak_temp = max(current_indoor, trigger_threshold + 0.1) if risk else current_indoor

        result: dict[str, Any] = {
            "risk": risk,
            "peak_temp": peak_temp,
            "peak_hour": peak_hour,
            "hours_until_peak": peak_hour,
            "should_cool_now": should_cool_now,
            "reason": "; ".join(reason_parts),
            "trajectory": [],  # not available from classifier
            "trigger_threshold": trigger_threshold,
            "peak_outdoor": at_now,
            "total_pv_forecast": pv_now,
            "lgbm_proba": proba,
            "predicted_delta": delta_pred,
            "predicted_max_temp": predicted_max,
            "reg_risk": reg_risk,
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
            "predicted_delta": 0.0,
            "predicted_max_temp": 0.0,
            "reg_risk": False,
        }
