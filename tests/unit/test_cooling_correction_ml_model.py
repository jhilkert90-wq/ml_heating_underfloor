"""
Tests for cooling_correction_ml_model.py
"""
from __future__ import annotations

import json
import math
import os
from unittest.mock import MagicMock, patch

import pytest

try:
    import numpy as np
    import pandas as pd
except ImportError:
    pytest.skip("numpy/pandas not installed", allow_module_level=True)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestExtractCoolingCorrectionFeature:
    """Test _extract_cooling_correction_feature for various feature columns."""

    def _extract(self, col, physics, target=23.0):
        from src.cooling_correction_ml_model import _extract_cooling_correction_feature
        return _extract_cooling_correction_feature(col, physics, target)

    def test_indoor_margin(self):
        result = self._extract("indoor_margin", {"indoor_temp": 22.5}, 23.0)
        assert result == pytest.approx(0.5)

    def test_indoor_margin_overshoot(self):
        result = self._extract("indoor_margin", {"indoor_temp": 24.0}, 23.0)
        assert result == pytest.approx(-1.0)

    def test_indoor_margin_fallback_lag(self):
        result = self._extract("indoor_margin", {"indoor_temp_lag_30m": 22.0}, 23.0)
        assert result == pytest.approx(1.0)

    def test_indoor_margin_missing(self):
        result = self._extract("indoor_margin", {}, 23.0)
        assert result == 0.0

    def test_at(self):
        result = self._extract("AT", {"outdoor_temp": 28.0})
        assert result == pytest.approx(28.0)

    def test_at_delta_indoor(self):
        result = self._extract("at_delta_indoor", {"outdoor_temp": 30.0, "indoor_temp": 23.0})
        assert result == pytest.approx(7.0)

    def test_at_forecast(self):
        result = self._extract("AT_roh_4h", {"AT_roh_4h": 32.0})
        assert result == pytest.approx(32.0)

    def test_at_forecast_fallback(self):
        result = self._extract("AT_roh_4h", {"outdoor_temp": 28.0})
        assert result == pytest.approx(28.0)

    def test_vlt(self):
        result = self._extract("VLT", {"outlet_temp": 18.0})
        assert result == pytest.approx(18.0)

    def test_rlt(self):
        result = self._extract("RLT", {"return_temp": 20.0})
        assert result == pytest.approx(20.0)

    def test_delta_t(self):
        result = self._extract("delta_t", {"delta_t": -2.0})
        assert result == pytest.approx(-2.0)

    def test_pv_generate(self):
        result = self._extract("PV_Generate", {"pv_now_electrical": 3000.0})
        assert result == pytest.approx(3000.0)

    def test_pv_generate_fallback(self):
        result = self._extract("PV_Generate", {"pv_now": 2500.0})
        assert result == pytest.approx(2500.0)

    def test_pv_generate_missing(self):
        result = self._extract("PV_Generate", {})
        assert result == 0.0

    def test_fireplace_lag_fallback(self):
        result = self._extract("fireplace_lag_1h", {"fireplace_on": 1.0})
        assert result == pytest.approx(1.0)

    def test_tv_lag_fallback(self):
        result = self._extract("tv_lag_30m", {"tv_on": 1.0})
        assert result == pytest.approx(1.0)

    def test_is_hp_active_on(self):
        result = self._extract("is_hp_active", {"delta_t": -3.0})
        assert result == 1.0

    def test_is_hp_active_off(self):
        result = self._extract("is_hp_active", {"delta_t": 0.5})
        assert result == 0.0

    def test_is_overshoot_true(self):
        result = self._extract("is_overshoot", {"indoor_temp": 24.0}, 23.0)
        assert result == 1.0

    def test_is_overshoot_false(self):
        result = self._extract("is_overshoot", {"indoor_temp": 22.0}, 23.0)
        assert result == 0.0

    def test_heat_loss_driving_force(self):
        result = self._extract("heat_loss_driving_force", {"indoor_temp": 23.0, "outdoor_temp": 30.0})
        assert result == pytest.approx(-7.0)

    def test_shading_proxy(self):
        result = self._extract("shading_proxy", {"indoor_temp": 25.0, "pv_now_electrical": 1000.0})
        assert result == pytest.approx(2.0 * 1000.0)

    def test_shading_proxy_below_threshold(self):
        result = self._extract("shading_proxy", {"indoor_temp": 22.0, "pv_now_electrical": 1000.0})
        assert result == 0.0

    def test_unknown_col_fills_zero(self):
        result = self._extract("nonexistent_feature", {})
        assert result == 0.0

    # NB08/NB09-derived features
    def test_cumulative_Q_wp_4h(self):
        result = self._extract("cumulative_Q_wp_4h", {"cumulative_Q_wp_4h": 5000.0})
        assert result == pytest.approx(5000.0)

    def test_indoor_accel(self):
        result = self._extract("indoor_accel", {"indoor_accel": 0.1})
        assert result == pytest.approx(0.1)

    def test_AT_forecast_trend_direct(self):
        result = self._extract("AT_forecast_trend", {"AT_forecast_trend": 2.0})
        assert result == pytest.approx(2.0)

    def test_AT_forecast_trend_fallback(self):
        result = self._extract("AT_forecast_trend", {"outdoor_temp": 28.0, "AT_roh_4h": 30.0})
        assert result == pytest.approx(2.0)

    def test_pv_cumulative_4h(self):
        result = self._extract("pv_cumulative_4h", {"pv_cumulative_4h": 10000.0})
        assert result == pytest.approx(10000.0)

    def test_thermal_momentum_direct(self):
        result = self._extract("thermal_momentum", {"thermal_momentum": -500.0})
        assert result == pytest.approx(-500.0)

    def test_thermal_momentum_fallback(self):
        result = self._extract("thermal_momentum", {"thermal_power_rolling_1h": 1000.0, "delta_t": -2.0})
        assert result == pytest.approx(-2000.0)


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

class TestBuildCoolingCorrectionFeatureVector:
    def test_respects_col_order(self):
        from src.cooling_correction_ml_model import build_cooling_correction_feature_vector
        cols = ["indoor_margin", "AT", "delta_t"]
        physics = {"indoor_temp": 22.5, "outdoor_temp": 28.0, "delta_t": -2.0}
        vec = build_cooling_correction_feature_vector(cols, physics, 23.0)
        assert len(vec) == 3
        assert vec[0] == pytest.approx(0.5)   # indoor_margin
        assert vec[1] == pytest.approx(28.0)   # AT
        assert vec[2] == pytest.approx(-2.0)   # delta_t

    def test_unknown_col_fills_zero(self):
        from src.cooling_correction_ml_model import build_cooling_correction_feature_vector
        cols = ["indoor_margin", "nonexistent"]
        vec = build_cooling_correction_feature_vector(cols, {"indoor_temp": 22.5}, 23.0)
        assert vec[1] == 0.0


# ---------------------------------------------------------------------------
# CoolingCorrectionMLModel
# ---------------------------------------------------------------------------

class _FakeModel:
    """Simple picklable model for tests."""
    def __init__(self, return_val=0.0):
        self._return_val = return_val

    def predict(self, X):
        return np.array([self._return_val] * len(X))


class _ErrorModel:
    """Picklable model that raises on predict."""
    def predict(self, X):
        raise RuntimeError("boom")


class TestCoolingCorrectionMLModelLoad:
    def test_returns_false_when_model_absent(self, tmp_path):
        from src.cooling_correction_ml_model import CoolingCorrectionMLModel
        model = CoolingCorrectionMLModel(
            str(tmp_path / "missing.joblib"),
            str(tmp_path / "missing.json"),
        )
        assert model.load() is False
        assert model.is_loaded is False

    def test_returns_false_when_metadata_absent(self, tmp_path):
        from src.cooling_correction_ml_model import CoolingCorrectionMLModel
        import joblib
        model_path = str(tmp_path / "model.joblib")
        joblib.dump(_FakeModel(), model_path)
        model = CoolingCorrectionMLModel(model_path, str(tmp_path / "missing.json"))
        assert model.load() is False

    def test_loads_r2_from_metadata(self, tmp_path):
        from src.cooling_correction_ml_model import CoolingCorrectionMLModel
        import joblib

        model_path = str(tmp_path / "model.joblib")
        meta_path = str(tmp_path / "meta.json")
        joblib.dump(_FakeModel(), model_path)
        with open(meta_path, "w") as f:
            json.dump({
                "feature_cols": ["indoor_margin"],
                "val_r2": 0.85,
                "val_mae": 0.15,
                "label_type": "residualized",
                "s_h_estimated": 0.35,
            }, f)

        model = CoolingCorrectionMLModel(model_path, meta_path)
        assert model.load() is True
        assert model.r2_score == pytest.approx(0.85)
        assert model.is_loaded is True


class TestCoolingCorrectionMLModelPredict:
    def test_returns_none_when_not_loaded(self):
        from src.cooling_correction_ml_model import CoolingCorrectionMLModel
        model = CoolingCorrectionMLModel("/nonexistent", "/nonexistent")
        assert model.predict({}, 23.0) is None

    def test_returns_model_prediction(self, tmp_path):
        from src.cooling_correction_ml_model import CoolingCorrectionMLModel
        import joblib

        model_path = str(tmp_path / "model.joblib")
        meta_path = str(tmp_path / "meta.json")
        joblib.dump(_FakeModel(0.5), model_path)
        with open(meta_path, "w") as f:
            json.dump({
                "feature_cols": ["indoor_margin"],
                "val_r2": 0.9,
                "label_type": "",
            }, f)

        model = CoolingCorrectionMLModel(model_path, meta_path)
        model.load()
        result = model.predict({"indoor_temp": 22.5}, 23.0)
        assert result is not None
        assert result == pytest.approx(0.5)

    def test_residualized_reconstruction(self, tmp_path):
        """Residualized model reconstructs full correction."""
        from src.cooling_correction_ml_model import CoolingCorrectionMLModel
        import joblib

        raw_pred = 0.3
        model_path = str(tmp_path / "model.joblib")
        meta_path = str(tmp_path / "meta.json")
        s_h = 0.35
        joblib.dump(_FakeModel(raw_pred), model_path)
        with open(meta_path, "w") as f:
            json.dump({
                "feature_cols": ["indoor_margin"],
                "val_r2": 0.9,
                "label_type": "residualized",
                "s_h_estimated": s_h,
            }, f)

        model = CoolingCorrectionMLModel(model_path, meta_path)
        model.load()

        indoor = 22.5
        target = 23.0
        indoor_margin = target - indoor  # 0.5

        result = model.predict({"indoor_temp": indoor}, target)
        expected = raw_pred - indoor_margin / s_h
        assert result == pytest.approx(expected, rel=1e-3)

    def test_predict_returns_none_on_inference_error(self, tmp_path):
        from src.cooling_correction_ml_model import CoolingCorrectionMLModel
        import joblib

        model_path = str(tmp_path / "model.joblib")
        meta_path = str(tmp_path / "meta.json")
        joblib.dump(_ErrorModel(), model_path)
        with open(meta_path, "w") as f:
            json.dump({"feature_cols": ["indoor_margin"], "val_r2": 0.9}, f)

        model = CoolingCorrectionMLModel(model_path, meta_path)
        model.load()
        assert model.predict({"indoor_temp": 22.5}, 23.0) is None
