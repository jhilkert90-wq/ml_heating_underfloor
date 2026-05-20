"""
Unit tests for heating_correction_ml_model.py

Covers:
1.  Feature extraction: indoor_margin = target - indoor
2.  Feature extraction: AT_roh_Xh maps to temp_forecast_Xh
3.  Feature extraction: fireplace_lag_1h falls back to fireplace_on
4.  Feature extraction: tv_lag_30m falls back to tv_on
5.  Feature extraction: unknown column fills 0.0 with warning
6.  build_heating_feature_vector: respects feature_cols order
7.  HeatingCorrectionMLModel.load: returns False when model file absent
8.  HeatingCorrectionMLModel.load: sets r2_score from metadata
9.  HeatingCorrectionMLModel.predict: returns None when not loaded
10. HeatingCorrectionMLModel.predict: returns model output when loaded
11. Blend formula w=0 → pure physics delta
12. Blend formula w=1 → pure ML delta
13. Blend formula w=0.5 → weighted average
14. Fallback to physics when ML model returns None
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestExtractHeatingFeature:
    """_extract_heating_feature maps column names to float values."""

    def _call(self, col, physics, target=21.0):
        from src.heating_correction_ml_model import _extract_heating_feature
        return _extract_heating_feature(col, physics, target)

    def test_indoor_margin(self):
        """indoor_margin = target_indoor - indoor_temp"""
        v = self._call("indoor_margin", {"indoor_temp": 20.0}, target=21.0)
        assert v == pytest.approx(1.0)

    def test_indoor_margin_overshoot(self):
        """indoor_margin is negative when room is too warm."""
        v = self._call("indoor_margin", {"indoor_temp": 22.0}, target=21.0)
        assert v == pytest.approx(-1.0)

    def test_indoor_temp(self):
        v = self._call("indoor_temp", {"indoor_temp": 20.5})
        assert v == pytest.approx(20.5)

    def test_indoor_temp_falls_back_to_lag_30m(self):
        """indoor_temp falls back to indoor_temp_lag_30m when indoor_temp absent.

        build_physics_features() does not produce an 'indoor_temp' key; it
        stores the current reading as 'indoor_temp_lag_30m'.  This test
        confirms the inference fallback so the model does not silently receive
        a spurious 0.0.
        """
        v = self._call("indoor_temp", {"indoor_temp_lag_30m": 19.8})
        assert v == pytest.approx(19.8)

    def test_indoor_temp_returns_zero_when_both_keys_absent(self):
        """indoor_temp returns 0.0 when neither key is present."""
        v = self._call("indoor_temp", {})
        assert v == pytest.approx(0.0)

    def test_indoor_margin_falls_back_to_lag_30m(self):
        """indoor_margin = target - indoor_temp, falling back to lag_30m.

        Regression for the inference key-mismatch bug: when indoor_temp is
        absent (runtime build_physics_features dict) the margin must use
        indoor_temp_lag_30m rather than hard-coding 0.0.
        """
        # With fallback: margin = 21.0 - 19.5 = 1.5
        v = self._call("indoor_margin", {"indoor_temp_lag_30m": 19.5}, target=21.0)
        assert v == pytest.approx(1.5)

    def test_at_delta_indoor(self):
        """at_delta_indoor = -temp_diff_indoor_outdoor = AT - indoor"""
        v = self._call("at_delta_indoor", {"temp_diff_indoor_outdoor": 10.0})
        assert v == pytest.approx(-10.0)

    def test_at_roh_forecast(self):
        """AT_roh_2h maps to temp_forecast_2h."""
        v = self._call(
            "AT_roh_2h",
            {"temp_forecast_2h": 5.0, "outdoor_temp": 8.0},
        )
        assert v == pytest.approx(5.0)

    def test_at_roh_fallback_to_outdoor(self):
        """AT_roh_3h falls back to outdoor_temp when specific forecast absent."""
        v = self._call("AT_roh_3h", {"outdoor_temp": 7.0})
        assert v == pytest.approx(7.0)

    def test_fireplace_lag_falls_back_to_fireplace_on(self):
        """fireplace_lag_1h and fireplace_lag_2h use fireplace_on at inference."""
        v1h = self._call("fireplace_lag_1h", {"fireplace_on": 1.0})
        assert v1h == pytest.approx(1.0)
        v2h = self._call("fireplace_lag_2h", {"fireplace_on": 0.0})
        assert v2h == pytest.approx(0.0)
        # fractional: fireplace_lag_30m should also resolve
        v30m = self._call("fireplace_lag_30m", {"fireplace_on": 1.0})
        assert v30m == pytest.approx(1.0)

    def test_tv_lag_falls_back_to_tv_on(self):
        """tv_lag_30m and tv_lag_1h both use tv_on at inference."""
        v30m = self._call("tv_lag_30m", {"tv_on": 1.0})
        assert v30m == pytest.approx(1.0)
        v1h = self._call("tv_lag_1h", {"tv_on": 0.0})
        assert v1h == pytest.approx(0.0)

    def test_pv_generate_prefers_electrical(self):
        """PV_Generate uses pv_now_electrical first, falls back to pv_now."""
        v = self._call("PV_Generate", {"pv_now_electrical": 2500.0, "pv_now": 100.0})
        assert v == pytest.approx(2500.0)

    def test_pv_generate_fallback_to_pv_now(self):
        """PV_Generate uses pv_now when pv_now_electrical absent."""
        v = self._call("PV_Generate", {"pv_now": 300.0})
        assert v == pytest.approx(300.0)

    def test_pv_generate_zero_when_both_absent(self):
        v = self._call("PV_Generate", {})
        assert v == pytest.approx(0.0)

    def test_pv_roll_1h_returns_float(self):
        """pv_roll_1h returns float from history list."""
        from src.heating_correction_ml_model import _extract_heating_feature
        physics = {"pv_power_history_electrical": [1000.0] * 6}
        v = _extract_heating_feature("pv_roll_1h", physics, 21.0)
        assert v == pytest.approx(1000.0)

    def test_pv_roll_2h_returns_float(self):
        from src.heating_correction_ml_model import _extract_heating_feature
        physics = {"pv_power_history_electrical": [500.0] * 12}
        v = _extract_heating_feature("pv_roll_2h", physics, 21.0)
        assert v == pytest.approx(500.0)

    def test_pv_roll_fallback_when_no_history(self):
        """pv_roll_1h falls back to pv_now_electrical when no history."""
        from src.heating_correction_ml_model import _extract_heating_feature
        physics = {"pv_now_electrical": 800.0}
        v = _extract_heating_feature("pv_roll_1h", physics, 21.0)
        assert v == pytest.approx(800.0)

    def test_pv_forecast_prefers_electrical(self):
        """pv_forecast_2h prefers pv_forecast_electrical_2h."""
        v = self._call(
            "pv_forecast_2h",
            {"pv_forecast_electrical_2h": 1200.0, "pv_forecast_2h": 900.0},
        )
        assert v == pytest.approx(1200.0)

    def test_pv_forecast_fallback_to_thermal(self):
        """pv_forecast_3h falls back to pv_forecast_3h when electrical absent."""
        v = self._call("pv_forecast_3h", {"pv_forecast_3h": 450.0})
        assert v == pytest.approx(450.0)

    def test_pv_forecast_zero_when_absent(self):
        v = self._call("pv_forecast_4h", {})
        assert v == pytest.approx(0.0)

    def test_unknown_col_fills_zero(self):
        """Unknown column fills 0.0."""
        v = self._call("totally_unknown_col", {})
        assert v == pytest.approx(0.0)

    # ── NEW: 8 additional ML correction feature handlers ────────────────
    def test_wind_speed(self):
        """wind_speed maps directly from physics dict."""
        v = self._call("wind_speed", {"wind_speed": 5.2})
        assert v == pytest.approx(5.2)

    def test_wind_speed_missing(self):
        """wind_speed returns 0.0 when absent."""
        v = self._call("wind_speed", {})
        assert v == pytest.approx(0.0)

    def test_indoor_temp_gradient(self):
        """indoor_temp_gradient maps directly from physics dict."""
        v = self._call("indoor_temp_gradient", {"indoor_temp_gradient": 0.3})
        assert v == pytest.approx(0.3)

    def test_living_room_temp(self):
        """living_room_temp maps from physics dict."""
        v = self._call("living_room_temp", {"living_room_temp": 22.5})
        assert v == pytest.approx(22.5)

    def test_living_room_temp_fallback(self):
        """living_room_temp falls back to indoor_temp when absent."""
        v = self._call("living_room_temp", {"indoor_temp": 20.0})
        assert v == pytest.approx(20.0)

    def test_living_room_temp_fallback_lag(self):
        """living_room_temp falls back to indoor_temp_lag_30m."""
        v = self._call("living_room_temp", {"indoor_temp_lag_30m": 19.5})
        assert v == pytest.approx(19.5)

    def test_is_hp_active_on(self):
        """is_hp_active = 1.0 when |delta_t| > 1.0."""
        v = self._call("is_hp_active", {"delta_t": 3.5})
        assert v == pytest.approx(1.0)

    def test_is_hp_active_off(self):
        """is_hp_active = 0.0 when |delta_t| <= 1.0."""
        v = self._call("is_hp_active", {"delta_t": 0.5})
        assert v == pytest.approx(0.0)

    def test_is_hp_active_missing(self):
        """is_hp_active = 0.0 when delta_t absent."""
        v = self._call("is_hp_active", {})
        assert v == pytest.approx(0.0)

    def test_is_weekend(self):
        """is_weekend maps from physics dict."""
        v = self._call("is_weekend", {"is_weekend": 1.0})
        assert v == pytest.approx(1.0)

    def test_thermal_power_rolling_1h(self):
        """thermal_power_rolling_1h uses instantaneous thermal_power_kw at inference."""
        v = self._call("thermal_power_rolling_1h", {"thermal_power_kw": 4.5})
        assert v == pytest.approx(4.5)

    def test_indoor_margin_rate(self):
        """indoor_margin_rate maps directly from physics dict."""
        v = self._call("indoor_margin_rate", {"indoor_margin_rate": -0.2})
        assert v == pytest.approx(-0.2)

    def test_is_overshoot_true(self):
        """is_overshoot = 1.0 when indoor > target."""
        v = self._call("is_overshoot", {"indoor_temp": 22.0}, target=21.0)
        assert v == pytest.approx(1.0)

    def test_is_overshoot_false(self):
        """is_overshoot = 0.0 when indoor <= target."""
        v = self._call("is_overshoot", {"indoor_temp": 20.0}, target=21.0)
        assert v == pytest.approx(0.0)

    def test_is_overshoot_fallback_lag(self):
        """is_overshoot uses indoor_temp_lag_30m fallback."""
        v = self._call("is_overshoot", {"indoor_temp_lag_30m": 22.0}, target=21.0)
        assert v == pytest.approx(1.0)

    # ── Slab thermal state features ────────────────────────────────────
    def test_d_inlet_temp_60min(self):
        """d_inlet_temp_60min maps directly from physics dict."""
        v = self._call("d_inlet_temp_60min", {"d_inlet_temp_60min": 0.8})
        assert v == pytest.approx(0.8)

    def test_d_inlet_temp_60min_negative(self):
        """d_inlet_temp_60min preserves negative values (cool-down)."""
        v = self._call("d_inlet_temp_60min", {"d_inlet_temp_60min": -0.5})
        assert v == pytest.approx(-0.5)

    def test_d_inlet_temp_60min_missing(self):
        """d_inlet_temp_60min returns 0.0 when absent."""
        v = self._call("d_inlet_temp_60min", {})
        assert v == pytest.approx(0.0)

    def test_is_equilibrium_one(self):
        """is_equilibrium = 1.0 when set in physics dict."""
        v = self._call("is_equilibrium", {"is_equilibrium": 1.0})
        assert v == pytest.approx(1.0)

    def test_is_equilibrium_zero(self):
        """is_equilibrium = 0.0 when set in physics dict."""
        v = self._call("is_equilibrium", {"is_equilibrium": 0.0})
        assert v == pytest.approx(0.0)

    def test_is_equilibrium_missing(self):
        """is_equilibrium returns 0.0 when absent."""
        v = self._call("is_equilibrium", {})
        assert v == pytest.approx(0.0)

    def test_heat_loss_driving_force_uses_indoor_and_outdoor(self):
        v = self._call(
            "heat_loss_driving_force",
            {"indoor_temp": 21.5, "outdoor_temp": 7.0},
        )
        assert v == pytest.approx(14.5)

    def test_delta_t_indoor_lag1(self):
        v = self._call("delta_T_indoor_lag1", {"indoor_temp_delta_10m": -0.2})
        assert v == pytest.approx(-0.2)
        assert self._call("delta_T_indoor_lag1", {}) == pytest.approx(0.0)

    def test_q_wp_uses_specific_heat_capacity_from_config(self):
        with patch(
            "src.heating_correction_ml_model.config.SPECIFIC_HEAT_CAPACITY", 4.182
        ):
            v = self._call(
                "Q_wp",
                {"flow_rate": 12.0, "outlet_temp": 35.0, "inlet_temp": 30.0},
            )
        # 12 L/min => 0.2 L/s; 0.2 * 5 K * 4182 J/kgK = 4182 W
        assert v == pytest.approx(4182.0)

    def test_q_wp_fallbacks(self):
        assert self._call(
            "Q_wp", {"flow_rate": 0.0, "outlet_temp": 35.0, "inlet_temp": 30.0}
        ) == pytest.approx(0.0)
        assert self._call(
            "Q_wp", {"flow_rate": 12.0, "outlet_temp": 35.0}
        ) == pytest.approx(0.0)

    def test_solar_thermal_proxy(self):
        v = self._call(
            "solar_thermal_proxy",
            {"pv_now_electrical": 1800.0, "hour_cos": 0.5},
        )
        assert v == pytest.approx(900.0)
        assert self._call("solar_thermal_proxy", {}) == pytest.approx(0.0)

    def test_pv_forecast_delta(self):
        v = self._call(
            "pv_forecast_delta",
            {"pv_now_electrical": 600.0, "pv_forecast_electrical_2h": 1600.0},
        )
        assert v == pytest.approx(1000.0)
        assert self._call("pv_forecast_delta", {"pv_now": 600.0}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# build_heating_feature_vector
# ---------------------------------------------------------------------------

class TestBuildHeatingFeatureVector:
    def test_respects_col_order(self):
        from src.heating_correction_ml_model import build_heating_feature_vector
        cols = ["indoor_temp", "AT", "indoor_margin"]
        physics = {"indoor_temp": 20.0, "outdoor_temp": 5.0}
        vec = build_heating_feature_vector(cols, physics, target_indoor=21.0)
        assert len(vec) == 3
        assert vec[0] == pytest.approx(20.0)  # indoor_temp
        assert vec[1] == pytest.approx(5.0)   # AT
        assert vec[2] == pytest.approx(1.0)   # indoor_margin = 21 - 20

    def test_unknown_col_fills_zero(self):
        from src.heating_correction_ml_model import build_heating_feature_vector
        vec = build_heating_feature_vector(
            ["indoor_temp", "BAD_COL"], {"indoor_temp": 19.0}, 21.0
        )
        assert vec[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# HeatingCorrectionMLModel.load
# ---------------------------------------------------------------------------

class TestHeatingMLModelLoad:
    def test_returns_false_when_model_absent(self):
        from src.heating_correction_ml_model import HeatingCorrectionMLModel
        model = HeatingCorrectionMLModel(
            "/nonexistent/model.joblib",
            "/nonexistent/meta.json",
        )
        result = model.load()
        assert result is False
        assert model.is_loaded is False

    def test_returns_false_when_metadata_absent(self, tmp_path):
        """If model file exists but metadata does not → return False."""
        model_file = tmp_path / "model.joblib"
        model_file.write_bytes(b"")  # empty file is enough for the path check
        from src.heating_correction_ml_model import HeatingCorrectionMLModel
        m = HeatingCorrectionMLModel(
            str(model_file), "/nonexistent/meta.json"
        )
        assert m.load() is False

    def test_loads_r2_from_metadata(self, tmp_path):
        """r2_score is read from metadata JSON key 'val_r2'."""
        meta = {"feature_cols": ["indoor_temp", "AT"], "val_r2": 0.75, "val_mae": 0.1}
        model_file = tmp_path / "model.joblib"
        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps(meta))

        mock_model = MagicMock()

        def _fake_joblib_load(path):
            return mock_model

        with patch("joblib.load", _fake_joblib_load), \
             patch("os.path.exists", return_value=True):
            from src.heating_correction_ml_model import HeatingCorrectionMLModel
            m = HeatingCorrectionMLModel(str(model_file), str(meta_file))
            ok = m.load()

        assert ok is True
        assert m.r2_score == pytest.approx(0.75)

    def test_r2_defaults_to_zero_when_not_in_metadata(self, tmp_path):
        """r2_score is 0.0 when metadata lacks 'val_r2'."""
        meta = {"feature_cols": ["indoor_temp"], "val_mae": 0.2}
        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps(meta))
        model_file = tmp_path / "model.joblib"

        def _fake_joblib_load(_):
            return MagicMock()

        with patch("joblib.load", _fake_joblib_load), \
             patch("os.path.exists", return_value=True):
            from src.heating_correction_ml_model import HeatingCorrectionMLModel
            m = HeatingCorrectionMLModel(str(model_file), str(meta_file))
            m.load()

        assert m.r2_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# HeatingCorrectionMLModel.predict
# ---------------------------------------------------------------------------

class TestHeatingMLModelPredict:
    def test_returns_none_when_not_loaded(self):
        from src.heating_correction_ml_model import HeatingCorrectionMLModel
        m = HeatingCorrectionMLModel(
            "/nonexistent/x.joblib", "/nonexistent/x.json"
        )
        result = m.predict({"indoor_temp": 20.0}, target_indoor=21.0)
        assert result is None

    def test_returns_model_prediction(self, tmp_path):
        """predict() returns the regressor output as a float."""
        import numpy as np
        meta = {
            "feature_cols": ["indoor_temp", "AT"],
            "val_r2": 0.6,
            "val_mae": 0.1,
        }
        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps(meta))
        model_file = tmp_path / "model.joblib"

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.5])

        with patch("joblib.load", return_value=mock_model), \
             patch("os.path.exists", return_value=True):
            from src.heating_correction_ml_model import HeatingCorrectionMLModel
            m = HeatingCorrectionMLModel(str(model_file), str(meta_file))
            m.load()
            result = m.predict(
                {"indoor_temp": 20.0, "outdoor_temp": 5.0}, target_indoor=21.0
            )

        assert result == pytest.approx(1.5)

    def test_predict_returns_none_on_inference_error(self, tmp_path):
        meta = {"feature_cols": ["indoor_temp"], "val_r2": 0.5}
        meta_file = tmp_path / "meta.json"
        meta_file.write_text(json.dumps(meta))
        model_file = tmp_path / "model.joblib"

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("boom")

        with patch("joblib.load", return_value=mock_model), \
             patch("os.path.exists", return_value=True):
            from src.heating_correction_ml_model import HeatingCorrectionMLModel
            m = HeatingCorrectionMLModel(str(model_file), str(meta_file))
            m.load()
            result = m.predict({}, target_indoor=21.0)

        assert result is None


# ---------------------------------------------------------------------------
# Blend formula in model_wrapper._calculate_ml_correction
# ---------------------------------------------------------------------------

class TestMLCorrectionBlend:
    """
    Tests for _calculate_ml_correction() in model_wrapper.

    The blend is:
        delta_physics = physics_outlet - outlet_temp
        w = max(0, min(1, r2)) if r2 >= HEATING_ML_BLEND_MIN_R2 else 0
        delta_blend = (1-w)*delta_physics + w*delta_ml
        corrected = outlet_temp + delta_blend   [clamped]
    """

    def setup_method(self):
        from src.model_wrapper import get_enhanced_model_wrapper
        with patch("src.model_wrapper.config.CLAMP_MIN_ABS", 20.0), \
             patch("src.model_wrapper.config.CLAMP_MAX_ABS", 55.0):
            self.wrapper = get_enhanced_model_wrapper()
        for attr in ("_current_indoor", "_current_features"):
            if hasattr(self.wrapper, attr):
                delattr(self.wrapper, attr)
        # Clear class-level singleton cache so each test starts fresh
        from src.model_wrapper import EnhancedModelWrapper
        EnhancedModelWrapper._heating_correction_ml_model = None

    def _mock_trajectory(self):
        return {
            "trajectory": [21.0, 20.9, 20.7, 20.8],
            "reaches_target_at": None,
        }

    @patch("src.model_wrapper.config.TRAJECTORY_STEPS", 4)
    @patch("src.model_wrapper.config.CLAMP_MIN_ABS", 20.0)
    @patch("src.model_wrapper.config.CLAMP_MAX_ABS", 55.0)
    def test_w0_pure_physics(self):
        """When ML model is not loaded, delta_blend == delta_physics."""
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {"indoor_temp_delta_60m": -0.1}

        physics_outlet = 25.6

        with patch.object(
            self.wrapper,
            "_calculate_physics_newton_correction",
            return_value=physics_outlet,
        ), patch.object(
            self.wrapper,
            "_get_heating_correction_ml_model",
            return_value=None,  # model not loaded
        ):
            result = self.wrapper._calculate_ml_correction(
                outlet_temp=25.0,
                trajectory=self._mock_trajectory(),
                target_indoor=21.0,
                cycle_hours=10 / 60,
            )

        assert result == pytest.approx(physics_outlet)

    @patch("src.model_wrapper.config.TRAJECTORY_STEPS", 4)
    @patch("src.model_wrapper.config.CLAMP_MIN_ABS", 20.0)
    @patch("src.model_wrapper.config.CLAMP_MAX_ABS", 55.0)
    @patch("src.model_wrapper.config.HEATING_ML_BLEND_MIN_R2", 0.3)
    def test_w1_pure_ml(self):
        """When R²=1.0, blend weight w=1.0 → corrected = outlet + delta_ml."""
        outlet = 25.0
        delta_ml = 1.2
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {"indoor_temp_delta_60m": -0.1}
        self.wrapper._climate_mode = "heating"

        mock_ml_model = MagicMock()
        mock_ml_model.is_loaded = True
        mock_ml_model.r2_score = 1.0
        mock_ml_model.predict.return_value = delta_ml

        with patch.object(
            self.wrapper,
            "_calculate_physics_newton_correction",
            return_value=26.5,  # physics delta = 1.5
        ), patch.object(
            self.wrapper,
            "_get_heating_correction_ml_model",
            return_value=mock_ml_model,
        ), patch(
            "src.model_wrapper.config.get_outlet_bounds",
            return_value=(20.0, 55.0),
        ):
            result = self.wrapper._calculate_ml_correction(
                outlet_temp=outlet,
                trajectory=self._mock_trajectory(),
                target_indoor=21.0,
                cycle_hours=10 / 60,
            )

        # w=1 → delta_blend = delta_ml → corrected = 25.0 + 1.2 = 26.2
        assert result == pytest.approx(outlet + delta_ml, abs=0.01)

    @patch("src.model_wrapper.config.TRAJECTORY_STEPS", 4)
    @patch("src.model_wrapper.config.CLAMP_MIN_ABS", 20.0)
    @patch("src.model_wrapper.config.CLAMP_MAX_ABS", 55.0)
    @patch("src.model_wrapper.config.HEATING_ML_BLEND_MIN_R2", 0.3)
    def test_w05_blended(self):
        """When R²=0.5, blend weight w=0.5 → weighted average of deltas."""
        outlet = 25.0
        physics_outlet = 26.0  # delta_physics = 1.0
        delta_ml = 0.4
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {"indoor_temp_delta_60m": -0.1}
        self.wrapper._climate_mode = "heating"

        mock_ml_model = MagicMock()
        mock_ml_model.is_loaded = True
        mock_ml_model.r2_score = 0.5
        mock_ml_model.predict.return_value = delta_ml

        with patch.object(
            self.wrapper,
            "_calculate_physics_newton_correction",
            return_value=physics_outlet,
        ), patch.object(
            self.wrapper,
            "_get_heating_correction_ml_model",
            return_value=mock_ml_model,
        ), patch(
            "src.model_wrapper.config.get_outlet_bounds",
            return_value=(20.0, 55.0),
        ):
            result = self.wrapper._calculate_ml_correction(
                outlet_temp=outlet,
                trajectory=self._mock_trajectory(),
                target_indoor=21.0,
                cycle_hours=10 / 60,
            )

        delta_physics = physics_outlet - outlet  # 1.0
        w = 0.5
        expected_delta = (1 - w) * delta_physics + w * delta_ml  # 0.5*1.0 + 0.5*0.4
        assert result == pytest.approx(outlet + expected_delta, abs=0.01)

    @patch("src.model_wrapper.config.TRAJECTORY_STEPS", 4)
    @patch("src.model_wrapper.config.CLAMP_MIN_ABS", 20.0)
    @patch("src.model_wrapper.config.CLAMP_MAX_ABS", 55.0)
    @patch("src.model_wrapper.config.HEATING_ML_BLEND_MIN_R2", 0.3)
    def test_r2_below_threshold_uses_physics(self):
        """When R² < HEATING_ML_BLEND_MIN_R2, w=0 → pure physics output."""
        outlet = 25.0
        physics_outlet = 26.0
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {"indoor_temp_delta_60m": -0.1}
        self.wrapper._climate_mode = "heating"

        mock_ml_model = MagicMock()
        mock_ml_model.is_loaded = True
        mock_ml_model.r2_score = 0.2  # below 0.3 threshold
        mock_ml_model.predict.return_value = 2.0

        with patch.object(
            self.wrapper,
            "_calculate_physics_newton_correction",
            return_value=physics_outlet,
        ), patch.object(
            self.wrapper,
            "_get_heating_correction_ml_model",
            return_value=mock_ml_model,
        ), patch(
            "src.model_wrapper.config.get_outlet_bounds",
            return_value=(20.0, 55.0),
        ):
            result = self.wrapper._calculate_ml_correction(
                outlet_temp=outlet,
                trajectory=self._mock_trajectory(),
                target_indoor=21.0,
                cycle_hours=10 / 60,
            )

        # w=0 → pure physics
        assert result == pytest.approx(physics_outlet, abs=0.01)

    @patch("src.model_wrapper.config.TRAJECTORY_STEPS", 4)
    @patch("src.model_wrapper.config.CLAMP_MIN_ABS", 20.0)
    @patch("src.model_wrapper.config.CLAMP_MAX_ABS", 55.0)
    def test_fallback_to_physics_when_predict_returns_none(self):
        """When predict() returns None, fall back to physics Newton output."""
        outlet = 25.0
        physics_outlet = 26.2
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {"indoor_temp_delta_60m": -0.1}

        mock_ml_model = MagicMock()
        mock_ml_model.is_loaded = True
        mock_ml_model.r2_score = 0.8
        mock_ml_model.predict.return_value = None  # inference failure

        with patch.object(
            self.wrapper,
            "_calculate_physics_newton_correction",
            return_value=physics_outlet,
        ), patch.object(
            self.wrapper,
            "_get_heating_correction_ml_model",
            return_value=mock_ml_model,
        ):
            result = self.wrapper._calculate_ml_correction(
                outlet_temp=outlet,
                trajectory=self._mock_trajectory(),
                target_indoor=21.0,
                cycle_hours=10 / 60,
            )

        assert result == pytest.approx(physics_outlet)
