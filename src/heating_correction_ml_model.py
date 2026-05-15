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

Note on `fireplace_lag_1h` / `tv_lag_30m`
------------------------------------------
These features are computed as rolling maxima at *training* time (history data
available).  At *inference* time, the physics_features dict only exposes the
instantaneous ``fireplace_on`` / ``tv_on`` flags.  As an approximation we
fall back to the instantaneous value: if the source is currently on the lag is
also 1.0, if off we assume the lag has expired.  This is conservative (slightly
underpredicts residual heat after FP turns off) but safe for a first version.
A future improvement is to maintain a rolling binary history in the wrapper.
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
        return float(physics.get("indoor_temp") or 0.0)
    if col == "indoor_margin":
        # Positive when room is too cold (target > indoor), negative when warm
        indoor = float(physics.get("indoor_temp") or 0.0)
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
    if col == "thermal_power_kw":
        return float(physics.get("thermal_power_kw") or 0.0)

    # ── External heat sources ──────────────────────────────────────────
    if col == "fireplace_on":
        return float(physics.get("fireplace_on") or 0.0)
    if col == "tv_on":
        return float(physics.get("tv_on") or 0.0)
    # Lag features: approximated at inference as the instantaneous value
    # (conservative — slightly underpredicts residual heat, but safe).
    if col == "fireplace_lag_1h":
        return float(physics.get("fireplace_on") or 0.0)
    if col == "tv_lag_30m":
        return float(physics.get("tv_on") or 0.0)

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
    m = re.fullmatch(r"AT_roh_(\d+)h", col)
    if m:
        h = m.group(1)
        return float(
            physics.get(f"temp_forecast_{h}h")
            or physics.get("outdoor_temp")
            or 0.0
        )

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
            self._model = joblib.load(self._model_path)
            with open(self._metadata_path, "r", encoding="utf-8") as fh:
                self._metadata = json.load(fh)
            self._feature_cols = self._metadata.get("feature_cols", [])
            self._r2_score = float(self._metadata.get("val_r2", 0.0))
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
            vec = build_heating_feature_vector(
                self._feature_cols, features, target_indoor
            )
            X = np.array(vec, dtype=float).reshape(1, -1)
            delta: float = float(self._model.predict(X)[0])
            logger.debug(
                "HeatingCorrectionMLModel: raw ΔT_outlet=%.3f°C (R²=%.4f)",
                delta, self._r2_score,
            )
            return delta
        except Exception:
            logger.exception("HeatingCorrectionMLModel: inference failed")
            return None
