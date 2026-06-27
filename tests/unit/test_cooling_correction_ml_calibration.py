"""
Tests for cooling_correction_ml_calibration.py
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


def _make_cooling_df(n: int = 700, at_val: float = 25.0):
    """Build a minimal DataFrame accepted by the cooling calibration pipeline."""
    rng = np.random.default_rng(42)
    t = pd.date_range("2024-07-01", periods=n, freq="10min", tz="UTC")
    df = pd.DataFrame({
        "_time":       t,
        "indoor_temp": 23.0 + rng.normal(0, 0.05, n),
        "AT":          np.full(n, at_val),
        "VLT":         rng.normal(18.0, 1.0, n),
        "RLT":         rng.normal(20.0, 1.0, n),
        "flow_rate":   np.full(n, 5.0),
        "fireplace_on": np.zeros(n),
        "tv_on":        np.zeros(n),
    })
    return df


# ---------------------------------------------------------------------------
# _compute_s_h
# ---------------------------------------------------------------------------

class TestCoolingComputeSH:
    def test_formula(self):
        from src.cooling_correction_ml_calibration import _compute_s_h
        oe, hlc, tau, h = 0.20, 0.1206, 4.84, 4.0
        expected = (oe / (oe + hlc)) * (1.0 - math.exp(-h / tau))
        assert _compute_s_h(oe, hlc, tau, h) == pytest.approx(expected, rel=1e-6)

    def test_degenerate_returns_zero(self):
        from src.cooling_correction_ml_calibration import _compute_s_h
        assert _compute_s_h(0.0, 0.0, 4.0, 4.0) == 0.0
        assert _compute_s_h(0.5, 0.5, 0.0, 4.0) == 0.0


class TestReadCoolingThermalParams:
    def test_uses_heat_pump_channel_parameters(self):
        from src.cooling_correction_ml_calibration import _read_cooling_thermal_params

        cfg = type("Cfg", (), {})()

        fake_manager = MagicMock()
        fake_manager.state = {
            "learning_state": {
                "heat_source_channels": {
                    "heat_pump": {
                        "parameters": {
                            "outlet_effectiveness": 0.4808,
                            "heat_loss_coefficient": 0.1342,
                            "thermal_time_constant": 4.8957,
                        }
                    }
                }
            }
        }

        with patch(
            "src.unified_thermal_state_cooling.get_cooling_state_manager",
            return_value=fake_manager,
        ):
            oe, hlc, tau = _read_cooling_thermal_params(cfg)

        fake_manager.load_state.assert_called_once_with()
        assert oe == pytest.approx(0.4808)
        assert hlc == pytest.approx(0.1342)
        assert tau == pytest.approx(4.8957)

    def test_raises_runtime_error_when_heat_pump_channel_missing(self):
        from src.cooling_correction_ml_calibration import _read_cooling_thermal_params

        cfg = type("Cfg", (), {})()

        fake_manager = MagicMock()
        fake_manager.state = {"learning_state": {"heat_source_channels": {}}}

        with patch(
            "src.unified_thermal_state_cooling.get_cooling_state_manager",
            return_value=fake_manager,
        ), pytest.raises(RuntimeError, match="Cooling heat pump channel not initialized"):
            _read_cooling_thermal_params(cfg)

    def test_raises_runtime_error_when_required_key_missing(self):
        from src.cooling_correction_ml_calibration import _read_cooling_thermal_params

        cfg = type("Cfg", (), {})()

        fake_manager = MagicMock()
        fake_manager.state = {
            "learning_state": {
                "heat_source_channels": {
                    "heat_pump": {
                        "parameters": {
                            "outlet_effectiveness": 0.4808,
                            "heat_loss_coefficient": 0.1342,
                        }
                    }
                }
            }
        }

        with patch(
            "src.unified_thermal_state_cooling.get_cooling_state_manager",
            return_value=fake_manager,
        ), pytest.raises(RuntimeError, match="missing keys"):
            _read_cooling_thermal_params(cfg)

    def test_raises_runtime_error_when_parameter_out_of_bounds(self):
        from src.cooling_correction_ml_calibration import _read_cooling_thermal_params

        cfg = type("Cfg", (), {})()

        fake_manager = MagicMock()
        fake_manager.state = {
            "learning_state": {
                "heat_source_channels": {
                    "heat_pump": {
                        "parameters": {
                            "outlet_effectiveness": 0.049,
                            "heat_loss_coefficient": 0.1342,
                            "thermal_time_constant": 4.8957,
                        }
                    }
                }
            }
        }

        with patch(
            "src.unified_thermal_state_cooling.get_cooling_state_manager",
            return_value=fake_manager,
        ), pytest.raises(RuntimeError, match="Invalid cooling outlet_effectiveness"):
            _read_cooling_thermal_params(cfg)


# ---------------------------------------------------------------------------
# Warm-season filter
# ---------------------------------------------------------------------------

class TestWarmSeasonFilter:
    def test_abort_when_too_few_warm_rows(self):
        """Calibration aborts when not enough warm-season rows."""
        from src.cooling_correction_ml_calibration import calibrate_cooling_correction_ml
        df = _make_cooling_df(100, at_val=10.0)  # cold data → filtered out

        with patch(
            "src.cooling_correction_ml_calibration"
            ".fetch_historical_data_for_calibration",
            return_value=df,
        ), patch(
            "src.cooling_correction_ml_calibration._read_cooling_thermal_params",
            return_value=(0.20, 0.1206, 4.84),
        ), patch("src.cooling_correction_ml_calibration.config") as mock_cfg:
            mock_cfg.TARGET_INDOOR_TEMP_COOLING = 23.0
            mock_cfg.COOLING_ML_CORRECTION_LABEL_HORIZON_H = 4
            mock_cfg.STEPS_PER_HOUR = 6
            mock_cfg.COOLING_ML_CORRECTION_WARM_THRESHOLD_C = 18.0
            mock_cfg.COOLING_ML_CORRECTION_CALIBRATION_START_DATE = ""
            mock_cfg.COOLING_ML_CORRECTION_AT_FORECAST_HOURS = "1"
            mock_cfg.COOLING_ML_CORRECTION_PV_FORECAST_HOURS = "1"
            mock_cfg.COOLING_ML_CORRECTION_FIREPLACE_LAG_HOURS = "1"
            mock_cfg.COOLING_ML_CORRECTION_TV_LAG_HOURS = "0.5"
            mock_cfg.SPECIFIC_HEAT_CAPACITY = 4.186
            mock_cfg.COOLING_ML_CORRECTION_MIN_TRAINING_SAMPLES = 50

            result = calibrate_cooling_correction_ml()
            assert result is False


# ---------------------------------------------------------------------------
# Label construction (residualized)
# ---------------------------------------------------------------------------

class TestCoolingLabelConstruction:
    def test_residualized_label_sign(self):
        """When indoor temp rises in future → label < 0 (lower outlet)."""
        s_h = 0.35
        indoor = pd.Series([23.0, 23.0, 23.5, 24.0])
        future = indoor.shift(-2)
        label = -(future - indoor) / s_h
        # At t=0: future=23.5, current=23.0 → label = -(0.5)/0.35 < 0
        assert label.iloc[0] < 0

    def test_residualized_label_no_change(self):
        """When indoor doesn't change → label = 0."""
        s_h = 0.35
        indoor = pd.Series([23.0, 23.0, 23.0, 23.0])
        future = indoor.shift(-2)
        label = -(future - indoor) / s_h
        assert label.iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestCoolingMLCorrectionConfigDefaults:
    def test_cooling_correction_mode_default(self):
        assert config.COOLING_CORRECTION_MODE == "physics"

    def test_warm_threshold_default(self):
        assert config.COOLING_ML_CORRECTION_WARM_THRESHOLD_C == 18.0

    def test_label_horizon_default(self):
        assert config.COOLING_ML_CORRECTION_LABEL_HORIZON_H == 4

    def test_feature_pruning_enabled_default(self):
        assert config.COOLING_ML_CORRECTION_FEATURE_PRUNING_ENABLED is True

    def test_optuna_disabled_default(self):
        assert config.COOLING_ML_CORRECTION_OPTUNA_ENABLED is False

    def test_cv_disabled_default(self):
        assert config.COOLING_ML_CORRECTION_CV_ENABLED is False

    def test_reg_alpha_default(self):
        assert config.COOLING_ML_CORRECTION_REG_ALPHA == pytest.approx(0.1)

    def test_reg_lambda_default(self):
        assert config.COOLING_ML_CORRECTION_REG_LAMBDA == pytest.approx(1.0)

    def test_incremental_pruning_disabled_default(self):
        assert config.COOLING_ML_CORRECTION_INCREMENTAL_PRUNING_ENABLED is False

    def test_incremental_prune_pi_threshold_default(self):
        assert config.COOLING_ML_CORRECTION_INCREMENTAL_PRUNE_PI_THRESHOLD == pytest.approx(0.001)


class TestCoolingIncrementalPruning:
    def test_incremental_pruning_enabled_retrains_stepwise(self):
        """Cooling calibration should retrain at least once in incremental mode."""
        import types

        from src.cooling_correction_ml_calibration import calibrate_cooling_correction_ml

        df = _make_cooling_df(800, at_val=25.0)
        fit_call_count = [0]
        pi_call_count = [0]

        class _FakePIResult:
            def __init__(self, vals):
                self.importances_mean = np.array(vals, dtype=float)

        def _fake_permutation_importance(model, X, y, **kwargs):
            pi_call_count[0] += 1
            n = X.shape[1]
            vals = [0.1] * n
            if pi_call_count[0] == 1 and n > 0:
                vals[0] = -0.1
            return _FakePIResult(vals)

        class _CountingRegressor:
            def __init__(self, **kwargs):
                self.feature_importances_ = np.ones(128, dtype=int)

            def fit(self, X, y, **kw):
                fit_call_count[0] += 1
                self.feature_importances_ = np.ones(X.shape[1], dtype=int)

            def predict(self, X):
                return np.zeros(len(X), dtype=float)

        fake_inspection = types.ModuleType("sklearn.inspection")
        fake_inspection.permutation_importance = _fake_permutation_importance

        with patch(
            "src.cooling_correction_ml_calibration.fetch_historical_data_for_calibration",
            return_value=df,
        ), patch(
            "src.cooling_correction_ml_calibration._read_cooling_thermal_params",
            return_value=(0.20, 0.1206, 4.84),
        ), patch("src.cooling_correction_ml_calibration.config") as mock_cfg:
            mock_cfg.TARGET_INDOOR_TEMP_COOLING = 23.0
            mock_cfg.COOLING_ML_CORRECTION_LABEL_HORIZON_H = 4
            mock_cfg.STEPS_PER_HOUR = 6
            mock_cfg.COOLING_ML_CORRECTION_WARM_THRESHOLD_C = 18.0
            mock_cfg.COOLING_ML_CORRECTION_CALIBRATION_START_DATE = ""
            mock_cfg.COOLING_ML_CORRECTION_AT_FORECAST_HOURS = "1"
            mock_cfg.COOLING_ML_CORRECTION_PV_FORECAST_HOURS = "1"
            mock_cfg.COOLING_ML_CORRECTION_FIREPLACE_LAG_HOURS = "1"
            mock_cfg.COOLING_ML_CORRECTION_TV_LAG_HOURS = "0.5"
            mock_cfg.SPECIFIC_HEAT_CAPACITY = 4.186
            mock_cfg.COOLING_ML_CORRECTION_MIN_TRAINING_SAMPLES = 50
            mock_cfg.COOLING_ML_CORRECTION_RETRAIN_VAL_FRACTION = 0.25
            mock_cfg.COOLING_OUTLET_EFFECTIVENESS = 0.20
            mock_cfg.HEAT_LOSS_COEFFICIENT = 0.1206
            mock_cfg.THERMAL_TIME_CONSTANT = 4.84
            mock_cfg.COOLING_ML_CORRECTION_MODEL_PATH = "/tmp/cml_inc.joblib"
            mock_cfg.COOLING_ML_CORRECTION_METADATA_PATH = "/tmp/cml_inc_meta.json"
            mock_cfg.INDOOR_TEMP_ENTITY_ID = "sensor.indoor_temp"
            mock_cfg.OUTDOOR_TEMP_ENTITY_ID = "sensor.outdoor_temp"
            mock_cfg.OUTLET_TEMP_ENTITY_ID = "sensor.outlet_temp"
            mock_cfg.INLET_TEMP_ENTITY_ID = "sensor.inlet_temp"
            mock_cfg.FLOW_RATE_ENTITY_ID = "sensor.flow_rate"
            mock_cfg.POWER_CONSUMPTION_ENTITY_ID = "sensor.power_w"
            mock_cfg.PV_POWER_ENTITY_ID = "sensor.pv_generate"
            mock_cfg.FIREPLACE_STATUS_ENTITY_ID = "binary_sensor.fireplace_on"
            mock_cfg.TV_STATUS_ENTITY_ID = "binary_sensor.tv_on"
            mock_cfg.WIND_SPEED_ENTITY_ID = "sensor.wind_speed"
            mock_cfg.LIVING_ROOM_TEMP_ENTITY_ID = "sensor.living_room_temp"
            mock_cfg.COOLING_ML_CORRECTION_FEATURE_PRUNING_ENABLED = True
            mock_cfg.COOLING_ML_CORRECTION_PRUNE_PI_THRESHOLD = 0.0
            mock_cfg.COOLING_ML_CORRECTION_INCREMENTAL_PRUNING_ENABLED = True
            mock_cfg.COOLING_ML_CORRECTION_INCREMENTAL_PRUNE_PI_THRESHOLD = 0.001
            mock_cfg.COOLING_ML_CORRECTION_REG_ALPHA = 0.1
            mock_cfg.COOLING_ML_CORRECTION_REG_LAMBDA = 1.0
            mock_cfg.COOLING_ML_CORRECTION_OPTUNA_ENABLED = False
            mock_cfg.COOLING_ML_CORRECTION_CV_ENABLED = False

            mock_lgb = MagicMock()
            mock_lgb.LGBMRegressor = _CountingRegressor
            mock_lgb.early_stopping.return_value = object()

            with patch.dict(
                "sys.modules",
                {
                    "lightgbm": mock_lgb,
                    "sklearn.inspection": fake_inspection,
                },
            ), patch("joblib.dump"), patch("os.replace"):
                result = calibrate_cooling_correction_ml()

        assert result is True
        assert fit_call_count[0] >= 2


# Import config after patches are not needed at module level
from src import config
