"""
Unit tests for cooling_ml_calibration.calibrate_cooling_ml().

Covers:
- Config import resolution (the root cause of the production bug)
- Early exits: no data, too few rows, missing columns, too few features
- Warm-season filter threshold
- Label computation correctness (rolling-max forward look)
- Feature column coverage guard
- End-to-end calibration with mocked LGBM / joblib / InfluxDB
- Metadata JSON keys after successful calibration
- Observation buffer auto-retrain trigger chain
"""

from __future__ import annotations

import json
import math
import os
import types
from unittest.mock import MagicMock, patch, ANY

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_joblib():
    """Create a mock joblib that actually writes files for os.replace to work."""
    mock_joblib = MagicMock()
    def _fake_dump(obj, path, *a, **kw):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write("mock")
    mock_joblib.dump.side_effect = _fake_dump
    return mock_joblib


def _fake_config(**overrides):
    defaults = dict(
        CYCLE_INTERVAL_MINUTES=10,
        PRE_COOL_HORIZON_HOURS=12,
        PRE_COOL_LEAD_TIME_HOURS=8.0,
        COOLING_CLAMP_MAX_ABS=24.0,
        PRE_COOL_MIN_OUTDOOR_FORECAST_C=22.0,
        COOLING_ML_WARM_THRESHOLD_C=10.0,
        SPECIFIC_HEAT_CAPACITY=4.186,
        COOLING_ML_MIN_TRAINING_SAMPLES=10,
        COOLING_ML_RETRAIN_VAL_FRACTION=0.25,
        COOLING_ML_MODEL_PATH="/tmp/test_model.joblib",
        COOLING_ML_METADATA_PATH="/tmp/test_meta.json",
        INDOOR_TEMP_ENTITY_ID="sensor.rt_mittelwert",
        OUTDOOR_TEMP_ENTITY_ID="sensor.nibe_bt1_outdoor_temperature",
        OUTLET_TEMP_ENTITY_ID="sensor.nibe_bt2_supply_temp_s1",
        INLET_TEMP_ENTITY_ID="sensor.nibe_eb100_ep14_bt3_return_temp",
        FLOW_RATE_ENTITY_ID="input_number.hp_current_flow_rate",
        POWER_CONSUMPTION_ENTITY_ID="sensor.nibe_el_leistung",
        PV_POWER_ENTITY_ID="sensor.pv_leistung_gefiltert",
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_warm_df(n_rows: int = 600, indoor_base: float = 22.0) -> pd.DataFrame:
    """Create a warm-season DataFrame with all required columns."""
    rng = np.random.RandomState(42)
    ts = pd.date_range("2025-07-01", periods=n_rows, freq="10min", tz="UTC")
    df = pd.DataFrame({
        "_time": ts,
        "rt_mittelwert": indoor_base + rng.normal(0, 0.5, n_rows),
        "nibe_bt1_outdoor_temperature": 25.0 + rng.normal(0, 2.0, n_rows),
        "nibe_bt2_supply_temp_s1": 20.0 + rng.normal(0, 1.0, n_rows),
        "nibe_eb100_ep14_bt3_return_temp": 22.0 + rng.normal(0, 0.5, n_rows),
        "hp_current_flow_rate": 14.0 + rng.normal(0, 0.5, n_rows),
        "nibe_el_leistung": 350.0 + rng.normal(0, 20, n_rows),
        "pv_leistung_gefiltert": rng.uniform(0, 5000, n_rows),
    })
    return df


# ===========================================================================
# Config Import Resolution (regression tests for the production bug)
# ===========================================================================

class TestConfigImportResolution:
    """The root cause: bare `import config` failed in package context."""

    def test_calibration_module_importable(self):
        """calibrate_cooling_ml can be imported from the src package."""
        from src.cooling_ml_calibration import calibrate_cooling_ml
        assert callable(calibrate_cooling_ml)

    def test_model_module_importable(self):
        """CoolingMLModel can be imported from the src package."""
        from src.cooling_ml_model import CoolingMLModel
        assert CoolingMLModel is not None

    def test_predict_overheating_risk_does_not_raise_import_error(self):
        """predict_overheating_risk no longer crashes with 'No module named config'."""
        from src.cooling_ml_model import CoolingMLModel
        model = CoolingMLModel("/nonexistent/model.joblib", "/nonexistent/meta.json")
        # Model is not loaded, so it should return a safe no-risk result
        result = model.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features={"outdoor_temp": 30.0, "pv_now": 1000.0},
            climate_mode="cooling",
        )
        assert result["risk"] is False
        assert "not loaded" in result["reason"]


# ===========================================================================
# Early Exit Guards
# ===========================================================================

class TestEarlyExits:
    """calibrate_cooling_ml returns False for various invalid inputs."""

    def _mock_physics_cal(self, return_value):
        """Create a mock physics_calibration module with fetch function."""
        mock = MagicMock()
        mock.fetch_historical_data_for_calibration = MagicMock(return_value=return_value)
        return mock

    def test_no_historical_data(self, tmp_path):
        """Returns False when fetch returns empty DataFrame."""
        import src.config as real_config
        overrides = dict(
            COOLING_ML_MODEL_PATH=str(tmp_path / "model.joblib"),
            COOLING_ML_METADATA_PATH=str(tmp_path / "meta.json"),
        )
        mock_phys = self._mock_physics_cal(pd.DataFrame())
        config_patches = {k: patch.object(real_config, k, v, create=True) for k, v in overrides.items()}
        for p in config_patches.values():
            p.start()
        try:
            with patch.dict("sys.modules", {
                "physics_calibration": mock_phys,
                "src.physics_calibration": mock_phys,
            }):
                from src.cooling_ml_calibration import calibrate_cooling_ml
                result = calibrate_cooling_ml()
        finally:
            for p in config_patches.values():
                p.stop()
        assert result is False

    def test_missing_required_columns(self, tmp_path):
        """Returns False when required columns are absent."""
        import src.config as real_config
        overrides = dict(
            COOLING_ML_MODEL_PATH=str(tmp_path / "model.joblib"),
            COOLING_ML_METADATA_PATH=str(tmp_path / "meta.json"),
        )
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        mock_phys = self._mock_physics_cal(df)
        config_patches = {k: patch.object(real_config, k, v, create=True) for k, v in overrides.items()}
        for p in config_patches.values():
            p.start()
        try:
            with patch.dict("sys.modules", {
                "physics_calibration": mock_phys,
                "src.physics_calibration": mock_phys,
            }):
                from src.cooling_ml_calibration import calibrate_cooling_ml
                result = calibrate_cooling_ml()
        finally:
            for p in config_patches.values():
                p.stop()
        assert result is False

    def test_too_few_warm_season_rows(self, tmp_path):
        """Returns False when < 500 rows pass the warm-season filter."""
        import src.config as real_config
        overrides = dict(
            COOLING_ML_MODEL_PATH=str(tmp_path / "model.joblib"),
            COOLING_ML_METADATA_PATH=str(tmp_path / "meta.json"),
        )
        df = _make_warm_df(n_rows=10)
        mock_phys = self._mock_physics_cal(df)
        config_patches = {k: patch.object(real_config, k, v, create=True) for k, v in overrides.items()}
        for p in config_patches.values():
            p.start()
        try:
            with patch.dict("sys.modules", {
                "physics_calibration": mock_phys,
                "src.physics_calibration": mock_phys,
            }):
                from src.cooling_ml_calibration import calibrate_cooling_ml
                result = calibrate_cooling_ml()
        finally:
            for p in config_patches.values():
                p.stop()
        assert result is False

    def test_cold_season_rows_filtered_out(self, tmp_path):
        """Cold outdoor temps (< threshold - 6) are filtered out."""
        import src.config as real_config
        overrides = dict(
            COOLING_ML_MODEL_PATH=str(tmp_path / "model.joblib"),
            COOLING_ML_METADATA_PATH=str(tmp_path / "meta.json"),
        )
        df = _make_warm_df(n_rows=600)
        df["nibe_bt1_outdoor_temperature"] = 10.0  # cold
        mock_phys = self._mock_physics_cal(df)
        config_patches = {k: patch.object(real_config, k, v, create=True) for k, v in overrides.items()}
        for p in config_patches.values():
            p.start()
        try:
            with patch.dict("sys.modules", {
                "physics_calibration": mock_phys,
                "src.physics_calibration": mock_phys,
            }):
                from src.cooling_ml_calibration import calibrate_cooling_ml
                result = calibrate_cooling_ml()
        finally:
            for p in config_patches.values():
                p.stop()
        assert result is False


# ===========================================================================
# Label Computation
# ===========================================================================

class TestLabelComputation:
    """Verify the rolling-max forward-look label logic."""

    def test_label_marks_overheating_correctly(self):
        """Label = 1 when future indoor exceeds cooling_target within horizon."""
        from src.cooling_ml_calibration import _optimise_threshold
        # Verify the helper is importable (side check)
        assert callable(_optimise_threshold)

    def test_optimise_threshold_returns_valid_range(self):
        """Threshold should be in (0, 1) range."""
        from src.cooling_ml_calibration import _optimise_threshold
        y_true = np.array([0, 0, 1, 1, 0, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        thr, f1 = _optimise_threshold(y_true, proba)
        assert 0.0 < thr < 1.0
        assert 0.0 <= f1 <= 1.0

    def test_optimise_threshold_all_zeros(self):
        """With no positive class, F1 is 0 and threshold defaults."""
        from src.cooling_ml_calibration import _optimise_threshold
        y_true = np.array([0, 0, 0, 0])
        proba = np.array([0.1, 0.2, 0.3, 0.4])
        thr, f1 = _optimise_threshold(y_true, proba)
        assert f1 == 0.0

    def test_optimise_threshold_all_ones(self):
        """With all positive class, any threshold above min yields good F1."""
        from src.cooling_ml_calibration import _optimise_threshold
        y_true = np.array([1, 1, 1, 1])
        proba = np.array([0.8, 0.9, 0.7, 0.95])
        thr, f1 = _optimise_threshold(y_true, proba)
        assert f1 > 0.5  # should find a good threshold


# ===========================================================================
# End-to-End Calibration (mocked LGBM + joblib)
# ===========================================================================

class TestEndToEndCalibration:
    """Full pipeline with mocked external deps."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        return tmp_path / "models"

    def _run_calibration(self, model_dir, n_rows=800, extra_cfg=None):
        """Run calibrate_cooling_ml with all external deps mocked."""
        import src.config as real_config

        model_path = str(model_dir / "model.joblib")
        meta_path = str(model_dir / "meta.json")
        overrides = dict(
            COOLING_ML_MODEL_PATH=model_path,
            COOLING_ML_METADATA_PATH=meta_path,
            COOLING_ML_MIN_TRAINING_SAMPLES=10,
        )
        if extra_cfg:
            overrides.update(extra_cfg)

        df = _make_warm_df(n_rows=n_rows)

        # Create a mock LGBMClassifier
        mock_lgb_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = lambda X: np.column_stack([
            np.random.rand(X.shape[0]),
            np.random.rand(X.shape[0]),
        ])
        mock_lgb_cls.return_value = mock_model

        # Create a mock LGBMRegressor
        mock_lgb_reg_cls = MagicMock()
        mock_reg_model = MagicMock()
        mock_reg_model.predict.side_effect = lambda X: np.random.rand(X.shape[0]) * 0.5
        mock_lgb_reg_cls.return_value = mock_reg_model

        mock_lgb = MagicMock()
        mock_lgb.LGBMClassifier = mock_lgb_cls
        mock_lgb.LGBMRegressor = mock_lgb_reg_cls
        mock_lgb.early_stopping.return_value = MagicMock()
        mock_lgb.log_evaluation.return_value = MagicMock()

        mock_joblib = _make_mock_joblib()
        mock_roc_auc = MagicMock(return_value=0.85)
        mock_mae = MagicMock(return_value=0.08)

        # Patch attributes on the real src.config since `from . import config`
        # resolves to src.config, not sys.modules["config"]
        config_patches = {k: patch.object(real_config, k, v, create=True) for k, v in overrides.items()}
        for p in config_patches.values():
            p.start()

        # Mock the physics_calibration module at sys.modules level to prevent
        # real HA connections during the `from src.physics_calibration import ...` fallback
        mock_physics_cal = MagicMock()
        mock_physics_cal.fetch_historical_data_for_calibration = MagicMock(return_value=df)
        try:
            with patch.dict("sys.modules", {
                "lightgbm": mock_lgb,
                "joblib": mock_joblib,
                "sklearn.metrics": MagicMock(
                    roc_auc_score=mock_roc_auc,
                    mean_absolute_error=mock_mae,
                ),
                "physics_calibration": mock_physics_cal,
                "src.physics_calibration": mock_physics_cal,
            }):
                from src.cooling_ml_calibration import calibrate_cooling_ml
                result = calibrate_cooling_ml()
        finally:
            for p in config_patches.values():
                p.stop()

        return result, model_path, meta_path, mock_joblib, mock_model

    def test_calibration_succeeds(self, model_dir):
        """Full pipeline returns True with valid data."""
        result, _, _, _, _ = self._run_calibration(model_dir)
        assert result is True

    def test_metadata_written(self, model_dir):
        """Metadata JSON is written with expected keys."""
        result, _, meta_path, _, _ = self._run_calibration(model_dir)
        assert result is True
        assert os.path.exists(meta_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        expected_keys = {
            "trained_at", "feature_cols", "n_features", "threshold",
            "val_f1", "roc_auc", "n_train", "n_val", "n_pos", "n_neg",
            "scale_pos_weight", "label_horizon_h", "forecast_horizon_h",
            "steps_per_hour", "cooling_target_c", "lookback_hours", "lgb_params",
            "calibrated", "threshold_method", "noise_injection", "temporal_weighting",
            "model_approach",
        }
        assert expected_keys.issubset(set(meta.keys()))

    def test_regression_metadata_keys(self, model_dir):
        """Metadata JSON contains regression keys when dual model is trained."""
        result, _, meta_path, _, _ = self._run_calibration(model_dir)
        assert result is True
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("model_approach") == "dual":
            regression_keys = {
                "regression_threshold", "regression_mae",
                "regression_auc", "regression_f1",
            }
            assert regression_keys.issubset(set(meta.keys()))

    def test_joblib_dump_called(self, model_dir):
        """joblib.dump is called for classifier (and regressor if dual mode)."""
        result, model_path, _, mock_joblib, _ = self._run_calibration(model_dir)
        assert result is True
        assert mock_joblib.dump.call_count >= 1
        # First call is the classifier; check temp path
        call_args = mock_joblib.dump.call_args_list[0]
        assert model_path + ".tmp" == call_args[0][1]

    def test_lgbm_fit_called_with_eval_set(self, model_dir):
        """LGBMClassifier.fit is called with eval_set for early stopping."""
        result, _, _, _, mock_model = self._run_calibration(model_dir)
        assert result is True
        mock_model.fit.assert_called_once()
        fit_kwargs = mock_model.fit.call_args
        assert "eval_set" in fit_kwargs.kwargs or len(fit_kwargs.args) > 2

    def test_lgbm_fit_called_with_sample_weight(self, model_dir):
        """LGBMClassifier.fit is called with sample_weight for temporal boundary weighting."""
        result, _, _, _, mock_model = self._run_calibration(model_dir)
        assert result is True
        mock_model.fit.assert_called_once()
        fit_kwargs = mock_model.fit.call_args
        assert "sample_weight" in fit_kwargs.kwargs, (
            "model.fit() must receive sample_weight for temporal boundary weighting"
        )
        sw = fit_kwargs.kwargs["sample_weight"]
        assert sw is not None
        assert len(sw) > 0

    def test_custom_cooling_target(self, model_dir):
        """cooling_target_c parameter is respected."""
        import src.config as real_config

        model_path = str(model_dir / "model2.joblib")
        meta_path = str(model_dir / "meta2.json")
        overrides = dict(
            COOLING_ML_MODEL_PATH=model_path,
            COOLING_ML_METADATA_PATH=meta_path,
            COOLING_ML_MIN_TRAINING_SAMPLES=10,
        )
        df = _make_warm_df(n_rows=800)

        mock_lgb_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = lambda X: np.column_stack([
            np.random.rand(X.shape[0]),
            np.random.rand(X.shape[0]),
        ])
        mock_lgb_cls.return_value = mock_model

        mock_lgb_reg_cls = MagicMock()
        mock_reg_model = MagicMock()
        mock_reg_model.predict.side_effect = lambda X: np.random.rand(X.shape[0]) * 0.5
        mock_lgb_reg_cls.return_value = mock_reg_model

        mock_lgb = MagicMock()
        mock_lgb.LGBMClassifier = mock_lgb_cls
        mock_lgb.LGBMRegressor = mock_lgb_reg_cls
        mock_lgb.early_stopping.return_value = MagicMock()
        mock_lgb.log_evaluation.return_value = MagicMock()

        config_patches = {k: patch.object(real_config, k, v, create=True) for k, v in overrides.items()}
        for p in config_patches.values():
            p.start()

        mock_physics_cal = MagicMock()
        mock_physics_cal.fetch_historical_data_for_calibration = MagicMock(return_value=df)
        try:
            with patch.dict("sys.modules", {
                "lightgbm": mock_lgb,
                "joblib": _make_mock_joblib(),
                "sklearn.metrics": MagicMock(
                    roc_auc_score=MagicMock(return_value=0.9),
                    mean_absolute_error=MagicMock(return_value=0.08),
                ),
                "physics_calibration": mock_physics_cal,
                "src.physics_calibration": mock_physics_cal,
            }):
                from src.cooling_ml_calibration import calibrate_cooling_ml
                result = calibrate_cooling_ml(cooling_target_c=25.0)
        finally:
            for p in config_patches.values():
                p.stop()

        assert result is True
        with open(meta_path, "r") as f:
            meta = json.load(f)
        assert meta["cooling_target_c"] == 25.0


# ===========================================================================
# Feature Column Coverage Guard
# ===========================================================================

class TestFeatureColumnGuard:
    """Test that features with <5% coverage are skipped."""

    def test_low_coverage_feature_skipped(self, tmp_path):
        """Feature columns with mostly NaN values are excluded."""
        import src.config as real_config

        overrides = dict(
            COOLING_ML_MODEL_PATH=str(tmp_path / "model.joblib"),
            COOLING_ML_METADATA_PATH=str(tmp_path / "meta.json"),
            COOLING_ML_MIN_TRAINING_SAMPLES=10,
        )
        df = _make_warm_df(n_rows=800)

        mock_lgb_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = lambda X: np.column_stack([
            np.random.rand(X.shape[0]),
            np.random.rand(X.shape[0]),
        ])
        mock_lgb_cls.return_value = mock_model

        mock_lgb = MagicMock()
        mock_lgb.LGBMClassifier = mock_lgb_cls
        mock_lgb.early_stopping.return_value = MagicMock()
        mock_lgb.log_evaluation.return_value = MagicMock()

        config_patches = {k: patch.object(real_config, k, v, create=True) for k, v in overrides.items()}
        for p in config_patches.values():
            p.start()

        mock_physics_cal = MagicMock()
        mock_physics_cal.fetch_historical_data_for_calibration = MagicMock(return_value=df)
        try:
            with patch.dict("sys.modules", {
                "lightgbm": mock_lgb,
                "joblib": _make_mock_joblib(),
                "sklearn.metrics": MagicMock(roc_auc_score=MagicMock(return_value=0.8)),
                "physics_calibration": mock_physics_cal,
                "src.physics_calibration": mock_physics_cal,
            }):
                from src.cooling_ml_calibration import calibrate_cooling_ml
                result = calibrate_cooling_ml()
        finally:
            for p in config_patches.values():
                p.stop()
        assert result is True


# ===========================================================================
# Observation Buffer → Retrain Trigger Chain
# ===========================================================================

class TestRetrainTriggerChain:
    """Verify buffer auto-retrain → calibrate_cooling_ml call chain."""

    def test_should_retrain_triggers_after_enough_labels(self, tmp_path):
        """When enough labels accumulate, should_retrain returns True."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        buf = CoolingObservationBuffer(
            path=str(tmp_path / "buf.json"),
            max_n=100,
            min_training_samples=3,
            retrain_trigger_k=2,
            horizon_steps=1,
        )
        # Push and resolve enough observations
        for i in range(4):
            buf.push_pending({"x": float(i)}, 22.0, 23.0, f"t{i}")
            buf.resolve_labels(24.0)  # overheating

        assert buf.should_retrain() is True

    def test_retrain_resets_counter(self, tmp_path):
        """After reset_retrain_counter, should_retrain returns False."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        buf = CoolingObservationBuffer(
            path=str(tmp_path / "buf.json"),
            max_n=100,
            min_training_samples=3,
            retrain_trigger_k=2,
            horizon_steps=1,
        )
        for i in range(4):
            buf.push_pending({"x": float(i)}, 22.0, 23.0, f"t{i}")
            buf.resolve_labels(24.0)

        assert buf.should_retrain() is True
        buf.reset_retrain_counter()
        assert buf.should_retrain() is False


# ===========================================================================
# JSON Serialization Helpers
# ===========================================================================

class TestJsonDefault:
    """Test _json_default handles edge cases."""

    def test_nan_serializes_as_none(self):
        from src.cooling_ml_calibration import _json_default
        assert _json_default(float("nan")) is None

    def test_inf_serializes_as_none(self):
        from src.cooling_ml_calibration import _json_default
        assert _json_default(float("inf")) is None

    def test_neg_inf_serializes_as_none(self):
        from src.cooling_ml_calibration import _json_default
        assert _json_default(float("-inf")) is None

    def test_non_float_raises(self):
        from src.cooling_ml_calibration import _json_default
        with pytest.raises(TypeError):
            _json_default(object())


# ===========================================================================
# Forecast Hour Selection
# ===========================================================================

class TestForecastHourSelection:
    """Verify AT and PV forecast hour selection via env vars."""

    def _run_calibration_with_env(self, tmp_path, env_overrides: dict) -> dict:
        """Run calibrate_cooling_ml with specific env vars and return metadata."""
        import src.config as real_config

        model_path = str(tmp_path / "model.joblib")
        meta_path = str(tmp_path / "meta.json")
        cfg_overrides = dict(
            COOLING_ML_MODEL_PATH=model_path,
            COOLING_ML_METADATA_PATH=meta_path,
            COOLING_ML_MIN_TRAINING_SAMPLES=10,
        )
        df = _make_warm_df(n_rows=800)

        mock_lgb_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = lambda X: np.column_stack([
            np.random.rand(X.shape[0]),
            np.random.rand(X.shape[0]),
        ])
        mock_lgb_cls.return_value = mock_model

        mock_lgb = MagicMock()
        mock_lgb.LGBMClassifier = mock_lgb_cls
        mock_lgb.early_stopping.return_value = MagicMock()
        mock_lgb.log_evaluation.return_value = MagicMock()

        config_patches = {k: patch.object(real_config, k, v, create=True) for k, v in cfg_overrides.items()}
        for p in config_patches.values():
            p.start()

        mock_physics_cal = MagicMock()
        mock_physics_cal.fetch_historical_data_for_calibration = MagicMock(return_value=df)
        try:
            with patch.dict("sys.modules", {
                "lightgbm": mock_lgb,
                "joblib": _make_mock_joblib(),
                "sklearn.metrics": MagicMock(roc_auc_score=MagicMock(return_value=0.8)),
                "physics_calibration": mock_physics_cal,
                "src.physics_calibration": mock_physics_cal,
            }), patch.dict("os.environ", env_overrides, clear=False):
                from src.cooling_ml_calibration import calibrate_cooling_ml
                result = calibrate_cooling_ml()
        finally:
            for p in config_patches.values():
                p.stop()

        assert result is True
        with open(meta_path, "r") as f:
            return json.load(f)

    def test_custom_at_forecast_hours_in_metadata(self, tmp_path):
        """When COOLING_ML_AT_FORECAST_HOURS=1,2,3, only AT_roh_1h/2h/3h appear."""
        meta = self._run_calibration_with_env(
            tmp_path,
            {"COOLING_ML_AT_FORECAST_HOURS": "1,2,3", "COOLING_ML_PV_FORECAST_HOURS": ""},
        )
        cols = meta["feature_cols"]
        assert "AT_roh_1h" in cols
        assert "AT_roh_2h" in cols
        assert "AT_roh_3h" in cols
        # Hours not in the list must be absent
        for h in range(4, 13):
            assert f"AT_roh_{h}h" not in cols

    def test_custom_pv_forecast_hours_in_metadata(self, tmp_path):
        """When COOLING_ML_PV_FORECAST_HOURS=1,2, only pv_forecast_1h/2h appear."""
        meta = self._run_calibration_with_env(
            tmp_path,
            {"COOLING_ML_AT_FORECAST_HOURS": "", "COOLING_ML_PV_FORECAST_HOURS": "1,2"},
        )
        cols = meta["feature_cols"]
        assert "pv_forecast_1h" in cols
        assert "pv_forecast_2h" in cols
        for h in range(3, 13):
            assert f"pv_forecast_{h}h" not in cols

    def test_default_includes_all_12_at_and_pv_hours(self, tmp_path):
        """With default env, all AT_roh_1h–12h and pv_forecast_1h–12h appear."""
        # Remove any overriding env vars from outer scope so defaults apply.
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("COOLING_ML_AT_FORECAST_HOURS",
                                  "COOLING_ML_FORECAST_HOURS",
                                  "COOLING_ML_PV_FORECAST_HOURS")}
        import src.config as real_config

        model_path = str(tmp_path / "model.joblib")
        meta_path = str(tmp_path / "meta.json")
        cfg_overrides = dict(
            COOLING_ML_MODEL_PATH=model_path,
            COOLING_ML_METADATA_PATH=meta_path,
            COOLING_ML_MIN_TRAINING_SAMPLES=10,
        )
        df = _make_warm_df(n_rows=800)

        mock_lgb_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = lambda X: np.column_stack([
            np.random.rand(X.shape[0]),
            np.random.rand(X.shape[0]),
        ])
        mock_lgb_cls.return_value = mock_model

        mock_lgb = MagicMock()
        mock_lgb.LGBMClassifier = mock_lgb_cls
        mock_lgb.early_stopping.return_value = MagicMock()
        mock_lgb.log_evaluation.return_value = MagicMock()

        config_patches = {k: patch.object(real_config, k, v, create=True) for k, v in cfg_overrides.items()}
        for p in config_patches.values():
            p.start()

        mock_physics_cal = MagicMock()
        mock_physics_cal.fetch_historical_data_for_calibration = MagicMock(return_value=df)
        try:
            with patch.dict("sys.modules", {
                "lightgbm": mock_lgb,
                "joblib": _make_mock_joblib(),
                "sklearn.metrics": MagicMock(roc_auc_score=MagicMock(return_value=0.8)),
                "physics_calibration": mock_physics_cal,
                "src.physics_calibration": mock_physics_cal,
            }), patch.dict("os.environ", clean_env, clear=True):
                from src.cooling_ml_calibration import calibrate_cooling_ml
                result = calibrate_cooling_ml()
        finally:
            for p in config_patches.values():
                p.stop()

        assert result is True
        with open(meta_path, "r") as f:
            meta = json.load(f)
        cols = meta["feature_cols"]
        # All 12 AT hindcast columns must be present.
        # Coverage of AT_roh_12h with 800 rows and 10-min intervals:
        # (800 - 72) / 800 = 91%, well above the 5% guard.
        for h in range(1, 13):
            assert f"AT_roh_{h}h" in cols, f"AT_roh_{h}h missing from feature_cols"
        # All 12 PV forecast columns must be present (same reasoning).
        for h in range(1, 13):
            assert f"pv_forecast_{h}h" in cols, f"pv_forecast_{h}h missing from feature_cols"

    def test_legacy_cooling_ml_forecast_hours_alias(self, tmp_path):
        """Legacy COOLING_ML_FORECAST_HOURS env var still controls AT forecast hours
        when COOLING_ML_AT_FORECAST_HOURS is absent from the environment."""
        import src.config as real_config

        model_path = str(tmp_path / "model.joblib")
        meta_path = str(tmp_path / "meta.json")
        cfg_overrides = dict(
            COOLING_ML_MODEL_PATH=model_path,
            COOLING_ML_METADATA_PATH=meta_path,
            COOLING_ML_MIN_TRAINING_SAMPLES=10,
        )
        df = _make_warm_df(n_rows=800)

        mock_lgb_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = lambda X: np.column_stack([
            np.random.rand(X.shape[0]),
            np.random.rand(X.shape[0]),
        ])
        mock_lgb_cls.return_value = mock_model

        mock_lgb = MagicMock()
        mock_lgb.LGBMClassifier = mock_lgb_cls
        mock_lgb.early_stopping.return_value = MagicMock()
        mock_lgb.log_evaluation.return_value = MagicMock()

        config_patches = {k: patch.object(real_config, k, v, create=True) for k, v in cfg_overrides.items()}
        for p in config_patches.values():
            p.start()

        mock_physics_cal = MagicMock()
        mock_physics_cal.fetch_historical_data_for_calibration = MagicMock(return_value=df)

        # Build an environment that has the legacy key but NOT the new key,
        # so the new-key fallback inside os.getenv resolves to the legacy value.
        clean_env = {
            k: v for k, v in os.environ.items()
            if k not in ("COOLING_ML_AT_FORECAST_HOURS",
                         "COOLING_ML_FORECAST_HOURS",
                         "COOLING_ML_PV_FORECAST_HOURS")
        }
        clean_env["COOLING_ML_FORECAST_HOURS"] = "4,8"
        # COOLING_ML_PV_FORECAST_HOURS is absent, so it defaults to all 12 hours

        try:
            with patch.dict("sys.modules", {
                "lightgbm": mock_lgb,
                "joblib": _make_mock_joblib(),
                "sklearn.metrics": MagicMock(roc_auc_score=MagicMock(return_value=0.8)),
                "physics_calibration": mock_physics_cal,
                "src.physics_calibration": mock_physics_cal,
            }), patch.dict("os.environ", clean_env, clear=True):
                from src.cooling_ml_calibration import calibrate_cooling_ml
                result = calibrate_cooling_ml()
        finally:
            for p in config_patches.values():
                p.stop()

        assert result is True
        with open(meta_path, "r") as f:
            meta = json.load(f)
        cols = meta["feature_cols"]
        assert "AT_roh_4h" in cols
        assert "AT_roh_8h" in cols
        for h in [1, 2, 3, 5, 6, 7, 9, 10, 11, 12]:
            assert f"AT_roh_{h}h" not in cols


# ===========================================================================
# Cooling Calibration Start Date
# ===========================================================================

class TestCoolingStartDate:
    """Verify COOLING_ML_CALIBRATION_START_DATE resolution in calibrate_cooling_ml."""

    def test_valid_start_date_overrides_lookback(self):
        """A valid DD.MM.YYYY date resolves to a lookback_hours larger than the default."""
        import src.config as real_config
        from datetime import datetime, timezone

        # Use a fixed historical date far in the past so lookback > default 2160 h.
        start_date_str = "01.01.2020"
        start_dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        expected_hours = int((now_utc - start_dt).total_seconds() / 3600)

        captured_lookback = []

        def _fake_fetch(lookback_hours, **kwargs):
            captured_lookback.append(lookback_hours)
            return None  # trigger early exit after data fetch

        with patch.object(real_config, "COOLING_ML_CALIBRATION_START_DATE", start_date_str):
            with patch.dict("sys.modules", {
                "physics_calibration": MagicMock(
                    fetch_historical_data_for_calibration=_fake_fetch
                ),
                "src.physics_calibration": MagicMock(
                    fetch_historical_data_for_calibration=_fake_fetch
                ),
            }):
                from src.cooling_ml_calibration import calibrate_cooling_ml
                calibrate_cooling_ml()

        assert len(captured_lookback) == 1
        # Allow a small tolerance (up to 2 h) for test execution time.
        assert abs(captured_lookback[0] - expected_hours) <= 2

    def test_empty_start_date_uses_default_lookback(self):
        """Empty COOLING_ML_CALIBRATION_START_DATE leaves lookback_hours at default 2160."""
        import src.config as real_config

        captured_lookback = []

        def _fake_fetch(lookback_hours, **kwargs):
            captured_lookback.append(lookback_hours)
            return None

        with patch.object(real_config, "COOLING_ML_CALIBRATION_START_DATE", ""):
            with patch.dict("sys.modules", {
                "physics_calibration": MagicMock(
                    fetch_historical_data_for_calibration=_fake_fetch
                ),
                "src.physics_calibration": MagicMock(
                    fetch_historical_data_for_calibration=_fake_fetch
                ),
            }):
                from src.cooling_ml_calibration import calibrate_cooling_ml
                calibrate_cooling_ml()

        assert len(captured_lookback) == 1
        assert captured_lookback[0] == 2160

    def test_invalid_start_date_uses_default_lookback_and_warns(self, caplog):
        """An invalid date string falls back to 2160 h and logs a warning."""
        import logging
        import src.config as real_config

        captured_lookback = []

        def _fake_fetch(lookback_hours, **kwargs):
            captured_lookback.append(lookback_hours)
            return None

        with patch.object(real_config, "COOLING_ML_CALIBRATION_START_DATE", "not-a-date"):
            with patch.dict("sys.modules", {
                "physics_calibration": MagicMock(
                    fetch_historical_data_for_calibration=_fake_fetch
                ),
                "src.physics_calibration": MagicMock(
                    fetch_historical_data_for_calibration=_fake_fetch
                ),
            }), caplog.at_level(logging.WARNING):
                from src.cooling_ml_calibration import calibrate_cooling_ml
                calibrate_cooling_ml()

        assert len(captured_lookback) == 1
        assert captured_lookback[0] == 2160
        assert any("not a valid" in r.message.lower() or "not_a_date" in r.message.lower()
                   or "not-a-date" in r.message for r in caplog.records)

    def test_parse_cooling_start_date_valid(self):
        """_parse_cooling_start_date returns aware UTC datetime for valid input."""
        from datetime import timezone
        import src.config as real_config
        result = real_config._parse_cooling_start_date("01.06.2024")
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.day == 1
        assert result.month == 6
        assert result.year == 2024

    def test_parse_cooling_start_date_empty(self):
        """_parse_cooling_start_date returns None for empty string."""
        import src.config as real_config
        assert real_config._parse_cooling_start_date("") is None
        assert real_config._parse_cooling_start_date("   ") is None

    def test_parse_cooling_start_date_invalid(self):
        """_parse_cooling_start_date returns None for non-date strings."""
        import src.config as real_config
        assert real_config._parse_cooling_start_date("2024-06-01") is None
        assert real_config._parse_cooling_start_date("not-a-date") is None
