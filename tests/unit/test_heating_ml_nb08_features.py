"""
Tests for heating ML model residualized label reconstruction
and NB08-derived feature extractors.
"""
from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import pytest

try:
    import numpy as np
    import pandas as pd
except ImportError:
    pytest.skip("numpy/pandas not installed", allow_module_level=True)


# ---------------------------------------------------------------------------
# NB08-derived feature extractors
# ---------------------------------------------------------------------------

class TestNB08Features:
    """Test the 5 new NB08-derived features in _extract_heating_feature."""

    def _extract(self, col, physics, target=22.6):
        from src.heating_correction_ml_model import _extract_heating_feature
        return _extract_heating_feature(col, physics, target)

    def test_cumulative_Q_wp_4h(self):
        result = self._extract("cumulative_Q_wp_4h", {"cumulative_Q_wp_4h": 12000.0})
        assert result == pytest.approx(12000.0)

    def test_cumulative_Q_wp_4h_missing(self):
        result = self._extract("cumulative_Q_wp_4h", {})
        assert result == 0.0

    def test_indoor_accel(self):
        result = self._extract("indoor_accel", {"indoor_accel": -0.05})
        assert result == pytest.approx(-0.05)

    def test_indoor_accel_missing(self):
        result = self._extract("indoor_accel", {})
        assert result == 0.0

    def test_AT_forecast_trend_direct(self):
        result = self._extract("AT_forecast_trend", {"AT_forecast_trend": 3.0})
        assert result == pytest.approx(3.0)

    def test_AT_forecast_trend_fallback(self):
        """Falls back to AT_roh_4h - outdoor_temp when direct key missing."""
        result = self._extract(
            "AT_forecast_trend",
            {"outdoor_temp": 5.0, "AT_roh_4h": 3.0},
        )
        assert result == pytest.approx(-2.0)

    def test_AT_forecast_trend_no_forecast(self):
        """Returns 0.0 when no forecast available."""
        result = self._extract("AT_forecast_trend", {"outdoor_temp": 5.0})
        assert result == 0.0

    def test_pv_cumulative_4h(self):
        result = self._extract("pv_cumulative_4h", {"pv_cumulative_4h": 8000.0})
        assert result == pytest.approx(8000.0)

    def test_pv_cumulative_4h_missing(self):
        result = self._extract("pv_cumulative_4h", {})
        assert result == 0.0

    def test_thermal_momentum_direct(self):
        result = self._extract("thermal_momentum", {"thermal_momentum": 5000.0})
        assert result == pytest.approx(5000.0)

    def test_thermal_momentum_fallback(self):
        """Computes from thermal_power_rolling_1h × delta_t."""
        result = self._extract(
            "thermal_momentum",
            {"thermal_power_rolling_1h": 2000.0, "delta_t": 3.0},
        )
        assert result == pytest.approx(6000.0)

    def test_thermal_momentum_missing(self):
        result = self._extract("thermal_momentum", {})
        assert result == 0.0


# ---------------------------------------------------------------------------
# Residualized label reconstruction in HeatingCorrectionMLModel
# ---------------------------------------------------------------------------

class _FakeModel:
    def __init__(self, return_val=0.0):
        self._return_val = return_val

    def predict(self, X):
        return np.array([self._return_val] * len(X))


class TestHeatingMLResidualizedReconstruction:
    def test_residualized_reconstruction(self, tmp_path):
        """Model with label_type=residualized reconstructs full correction."""
        from src.heating_correction_ml_model import HeatingCorrectionMLModel
        import joblib

        raw_pred = 0.2
        model_path = str(tmp_path / "model.joblib")
        meta_path = str(tmp_path / "meta.json")
        s_h = 0.492
        joblib.dump(_FakeModel(raw_pred), model_path)
        with open(meta_path, "w") as f:
            json.dump({
                "feature_cols": ["indoor_margin"],
                "val_r2": 0.95,
                "label_type": "residualized",
                "s_h_estimated": s_h,
            }, f)

        model = HeatingCorrectionMLModel(model_path, meta_path)
        model.load()

        indoor = 22.0
        target = 22.6
        indoor_margin = target - indoor  # 0.6

        result = model.predict({"indoor_temp": indoor}, target)
        expected = raw_pred + indoor_margin / s_h
        assert result == pytest.approx(expected, rel=1e-3)

    def test_non_residualized_returns_raw(self, tmp_path):
        """Model without label_type returns raw prediction."""
        from src.heating_correction_ml_model import HeatingCorrectionMLModel
        import joblib

        raw_pred = 0.5
        model_path = str(tmp_path / "model.joblib")
        meta_path = str(tmp_path / "meta.json")
        joblib.dump(_FakeModel(raw_pred), model_path)
        with open(meta_path, "w") as f:
            json.dump({
                "feature_cols": ["indoor_margin"],
                "val_r2": 0.9,
            }, f)

        model = HeatingCorrectionMLModel(model_path, meta_path)
        model.load()
        result = model.predict({"indoor_temp": 22.0}, 22.6)
        assert result == pytest.approx(raw_pred)

    def test_degenerate_s_h_returns_raw(self, tmp_path):
        """Residualized model with degenerate S_H falls back to raw."""
        from src.heating_correction_ml_model import HeatingCorrectionMLModel
        import joblib

        raw_pred = 0.3
        model_path = str(tmp_path / "model.joblib")
        meta_path = str(tmp_path / "meta.json")
        joblib.dump(_FakeModel(raw_pred), model_path)
        with open(meta_path, "w") as f:
            json.dump({
                "feature_cols": ["indoor_margin"],
                "val_r2": 0.9,
                "label_type": "residualized",
                "s_h_estimated": 0.01,  # degenerate
            }, f)

        model = HeatingCorrectionMLModel(model_path, meta_path)
        model.load()
        result = model.predict({"indoor_temp": 22.0}, 22.6)
        # S_H < 0.05 → raw prediction
        assert result == pytest.approx(raw_pred)
