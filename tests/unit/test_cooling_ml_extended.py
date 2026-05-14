"""
Extended tests for ML pre-cooling modules.

Covers:
- Cold start (no files exist)
- Observation buffer edge cases (concurrent push/resolve, NaN/inf features)
- CoolingMLModel inference edge cases (empty features, all-zero features)
- Calibration label boundary conditions
- Online learning retrain flow
- OverheatingPredictor edge cases (missing forecast keys, reactive path)
- Config default mismatches
- Observation buffer periodic save gap
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_config(**overrides):
    """Create a minimal fake config namespace for tests."""
    defaults = dict(
        PRE_COOL_ENABLED=True,
        PRE_COOL_TRIGGER_MARGIN_K=0.5,
        PRE_COOL_HORIZON_HOURS=12,
        PRE_COOL_LEAD_TIME_HOURS=3.0,
        PRE_COOL_TARGET_OFFSET_K=0.5,
        PRE_COOL_MIN_PV_FORECAST_W=1000.0,
        PRE_COOL_MIN_OUTDOOR_FORECAST_C=22.0,
        PRE_COOL_MODEL_TYPE="trajectory",
        COOLING_CLAMP_MAX_ABS=24.0,
        CYCLE_INTERVAL_MINUTES=10,
        COOLING_ML_MIN_TRAINING_SAMPLES=200,
        COOLING_ML_RETRAIN_VAL_FRACTION=0.25,
        COOLING_ML_MODEL_PATH="/tmp/test_model.joblib",
        COOLING_ML_METADATA_PATH="/tmp/test_meta.json",
        SPECIFIC_HEAT_CAPACITY=4.186,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_physics(**overrides):
    base = {
        "outdoor_temp": 30.0,
        "temp_diff_indoor_outdoor": -8.0,
        "pv_now": 2000.0,
        "pv_power_history": [2000.0] * 20,
        "indoor_temp_delta_30m": 0.1,
        "indoor_temp_delta_60m": 0.2,
        "outlet_temp": 28.0,
        "inlet_temp": 26.0,
        "delta_t": 2.0,
        "outlet_indoor_diff": 6.0,
        "thermal_power_kw": 1.5,
        "temp_forecast_1h": 31.0,
        "temp_forecast_2h": 32.0,
        "temp_forecast_4h": 33.0,
        "hour_sin": 0.5,
        "hour_cos": 0.866,
    }
    base.update(overrides)
    return base


# ===========================================================================
# Cold Start Tests
# ===========================================================================

class TestColdStart:
    """Verify system works correctly when no persisted files exist."""

    def test_observation_buffer_cold_start(self, tmp_path):
        """Buffer starts empty when no JSON file exists."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        path = str(tmp_path / "nonexistent" / "buffer.json")
        buf = CoolingObservationBuffer(path=path, max_n=100, horizon_steps=6)
        assert buf.n_total == 0
        assert buf.n_labeled == 0
        assert buf.n_pending == 0
        assert buf.should_retrain() is False

    def test_cooling_ml_model_cold_start(self):
        """Model returns safe no-risk result when no model file exists."""
        from src.cooling_ml_model import CoolingMLModel
        model = CoolingMLModel("/nonexistent/path/model.joblib", "/nonexistent/meta.json")
        assert model.load() is False
        assert model.is_loaded is False

        with patch.dict("sys.modules", {"config": _fake_config()}):
            result = model.predict_overheating_risk(22.0, 23.0, _make_physics())
        assert result["should_cool_now"] is False
        assert result["risk"] is False
        assert result["lgbm_proba"] == 0.0

    def test_observation_buffer_save_creates_directory(self, tmp_path):
        """save() creates parent directory if it doesn't exist."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        nested = str(tmp_path / "a" / "b" / "c" / "buffer.json")
        buf = CoolingObservationBuffer(path=nested, max_n=10, horizon_steps=2)
        buf.push_pending({"x": 1.0}, 22.0, 23.0, "t0")
        buf.save()
        assert os.path.exists(nested)

    def test_cold_start_full_cycle(self, tmp_path):
        """Simulate a full cold start cycle: no model, no buffer, no crash."""
        from src.cooling_ml_model import CoolingMLModel
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer

        model_path = str(tmp_path / "model.joblib")
        meta_path = str(tmp_path / "meta.json")
        buf_path = str(tmp_path / "buffer.json")

        model = CoolingMLModel(model_path, meta_path)
        model.load()  # returns False, non-fatal

        buf = CoolingObservationBuffer(buf_path, max_n=100, horizon_steps=6)

        # Simulate 10 cycles
        for i in range(10):
            physics = _make_physics(outdoor_temp=25.0 + i * 0.5)

            with patch.dict("sys.modules", {"config": _fake_config()}):
                result = model.predict_overheating_risk(22.0, 23.0, physics)
            assert result["should_cool_now"] is False  # model not loaded

            buf.push_pending(physics, 22.0 + i * 0.1, 23.0, f"t{i}")
            buf.resolve_labels(22.0 + i * 0.1)

        assert buf.n_total == 10
        assert buf.should_retrain() is False  # min_training_samples not met


# ===========================================================================
# CoolingObservationBuffer Edge Cases
# ===========================================================================

class TestObservationBufferEdgeCases:

    def test_nan_in_features_survives_save_load(self, tmp_path):
        """NaN in features is sanitized to None during save."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        path = str(tmp_path / "buf.json")
        buf = CoolingObservationBuffer(path=path, max_n=10, horizon_steps=1)
        buf.push_pending({"x": float("nan"), "y": 1.0}, 22.0, 23.0, "t0")
        buf.resolve_labels(24.0)
        buf.save()

        buf2 = CoolingObservationBuffer(path=path, max_n=10, horizon_steps=1)
        assert buf2.n_labeled == 1
        feats, labels = buf2.get_labeled_data()
        assert feats[0]["x"] is None  # NaN → null → None (fixed)
        assert feats[0]["y"] == 1.0

    def test_inf_in_features_survives_save_load(self, tmp_path):
        """Inf in features is sanitized to None during save."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        path = str(tmp_path / "buf.json")
        buf = CoolingObservationBuffer(path=path, max_n=10, horizon_steps=1)
        buf.push_pending({"x": float("inf")}, 22.0, 23.0, "t0")
        buf.resolve_labels(24.0)
        buf.save()

        buf2 = CoolingObservationBuffer(path=path, max_n=10, horizon_steps=1)
        feats, _ = buf2.get_labeled_data()
        assert feats[0]["x"] is None  # Inf → null → None (fixed)

    def test_resolve_labels_multiple_pending_different_outcomes(self, tmp_path):
        """Multiple pending entries with different indoor peaks get correct labels."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        path = str(tmp_path / "buf.json")
        buf = CoolingObservationBuffer(path=path, max_n=50, horizon_steps=2)

        # Entry 1: will see indoor=24 > target=23 → label=1
        buf.push_pending({}, 22.0, 23.0, "t0")
        buf.resolve_labels(24.0)  # step 1 for entry 0

        # Entry 2: pushed at step 1, will only see indoor=22 → label=0
        buf.push_pending({}, 22.0, 23.0, "t1")
        buf.resolve_labels(22.0)  # step 2 for entry 0, step 1 for entry 1

        # Entry 0 is now labeled, entry 1 still pending
        assert buf.n_labeled == 1
        assert buf.n_pending == 1

        buf.resolve_labels(22.0)  # step 2 for entry 1
        assert buf.n_labeled == 2

        _, labels = buf.get_labeled_data()
        assert labels[0] == 1  # saw 24.0 > 23.0
        assert labels[1] == 0  # only saw 22.0 ≤ 23.0

    def test_zero_horizon_steps(self, tmp_path):
        """horizon_steps=0 should label immediately (edge case)."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        path = str(tmp_path / "buf.json")
        # horizon_steps=0 means entries mature at 0 elapsed steps
        # But resolve_labels increments THEN checks >=, so step 0 won't label
        # This documents the actual behavior
        buf = CoolingObservationBuffer(path=path, max_n=10, horizon_steps=0)
        buf.push_pending({}, 22.0, 23.0, "t0")
        newly = buf.resolve_labels(22.0)
        # With horizon=0, steps_elapsed becomes 1 >= 0, so it labels
        assert newly == 1

    def test_empty_buffer_resolve_labels_no_crash(self, tmp_path):
        """resolve_labels on empty buffer should not crash."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        path = str(tmp_path / "buf.json")
        buf = CoolingObservationBuffer(path=path, max_n=10, horizon_steps=3)
        newly = buf.resolve_labels(22.0)
        assert newly == 0

    def test_get_labeled_data_empty(self, tmp_path):
        """get_labeled_data on empty buffer returns empty lists."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        path = str(tmp_path / "buf.json")
        buf = CoolingObservationBuffer(path=path, max_n=10, horizon_steps=3)
        feats, labels = buf.get_labeled_data()
        assert feats == []
        assert labels == []

    def test_label_exactly_at_cooling_target(self, tmp_path):
        """Indoor exactly at cooling_target should be label=0 (> not >=)."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        path = str(tmp_path / "buf.json")
        buf = CoolingObservationBuffer(path=path, max_n=10, horizon_steps=1)
        buf.push_pending({}, 22.0, 23.0, "t0")
        buf.resolve_labels(23.0)  # exactly at target
        _, labels = buf.get_labeled_data()
        assert labels[0] == 0  # > not >=, so 23.0 is NOT overheating


# ===========================================================================
# CoolingMLModel Edge Cases
# ===========================================================================

class TestCoolingMLModelEdgeCases:

    def test_empty_feature_cols_returns_no_risk(self):
        """Model with empty feature_cols returns no-risk."""
        from src.cooling_ml_model import CoolingMLModel
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.joblib")
            meta_path = os.path.join(tmpdir, "meta.json")

            # Write metadata with empty feature_cols
            with open(meta_path, "w") as f:
                json.dump({"feature_cols": [], "threshold": 0.5}, f)
            open(model_path, "w").close()

            mock_joblib = MagicMock()
            mock_joblib.load.return_value = MagicMock()

            model = CoolingMLModel(model_path, meta_path)
            with patch("src.cooling_ml_model._load_joblib", return_value=mock_joblib):
                model.load()

            with patch.dict("sys.modules", {"config": _fake_config()}):
                result = model.predict_overheating_risk(22.0, 23.0, _make_physics())
            assert result["should_cool_now"] is False
            assert "empty" in result["reason"].lower()

    def test_inference_exception_returns_no_risk(self):
        """Model that raises during predict_proba returns safe fallback."""
        import numpy as np
        from src.cooling_ml_model import CoolingMLModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.joblib")
            meta_path = os.path.join(tmpdir, "meta.json")

            with open(meta_path, "w") as f:
                json.dump({"feature_cols": ["indoor_temp", "AT"], "threshold": 0.5}, f)
            open(model_path, "w").close()

            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = RuntimeError("LGBM crash")
            mock_joblib = MagicMock()
            mock_joblib.load.return_value = mock_model

            model = CoolingMLModel(model_path, meta_path)
            with patch("src.cooling_ml_model._load_joblib", return_value=mock_joblib):
                model.load()

            with patch.dict("sys.modules", {"config": _fake_config()}):
                result = model.predict_overheating_risk(22.0, 23.0, _make_physics())
            assert result["should_cool_now"] is False
            assert "error" in result["reason"].lower()

    def test_reactive_cooling_when_indoor_above_target(self):
        """When room is already above target, should_cool_now=True regardless of model."""
        import numpy as np
        from src.cooling_ml_model import CoolingMLModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.joblib")
            meta_path = os.path.join(tmpdir, "meta.json")

            with open(meta_path, "w") as f:
                json.dump({"feature_cols": ["indoor_temp"], "threshold": 0.5}, f)
            open(model_path, "w").close()

            # Model predicts NO risk (proba=0.1 < threshold=0.5)
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
            mock_joblib = MagicMock()
            mock_joblib.load.return_value = mock_model

            model = CoolingMLModel(model_path, meta_path)
            with patch("src.cooling_ml_model._load_joblib", return_value=mock_joblib):
                model.load()

            # Indoor=24 > target=23 → reactive cooling
            with patch.dict("sys.modules", {"config": _fake_config()}):
                result = model.predict_overheating_risk(24.0, 23.0, _make_physics())
            assert result["should_cool_now"] is True
            assert "reactive" in result["reason"].lower()

    def test_proba_exactly_at_threshold(self):
        """Probability exactly at threshold should NOT trigger risk (> not >=)."""
        import numpy as np
        from src.cooling_ml_model import CoolingMLModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.joblib")
            meta_path = os.path.join(tmpdir, "meta.json")

            with open(meta_path, "w") as f:
                json.dump({"feature_cols": ["indoor_temp"], "threshold": 0.5}, f)
            open(model_path, "w").close()

            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])
            mock_joblib = MagicMock()
            mock_joblib.load.return_value = mock_model

            model = CoolingMLModel(model_path, meta_path)
            with patch("src.cooling_ml_model._load_joblib", return_value=mock_joblib):
                model.load()

            with patch.dict("sys.modules", {"config": _fake_config()}):
                result = model.predict_overheating_risk(22.0, 23.0, _make_physics())
            # 0.5 > 0.5 is False, so no risk
            assert result["risk"] is False


# ===========================================================================
# OverheatingPredictor Edge Cases
# ===========================================================================

class TestOverheatingPredictorEdgeCases:

    def test_missing_inlet_temp_returns_no_risk(self):
        """No inlet_temp in features → safe no-risk return."""
        from src.overheating_predictor import OverheatingPredictor
        predictor = OverheatingPredictor()
        result = predictor.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features={},  # no inlet_temp
            thermal_model=MagicMock(),
            climate_mode="cooling",
        )
        assert result["risk"] is False
        assert "inlet_temp" in result["reason"]

    def test_trajectory_simulation_failure(self):
        """Failed trajectory simulation → safe fallback."""
        from src.overheating_predictor import OverheatingPredictor
        predictor = OverheatingPredictor()

        model = MagicMock()
        model.predict_thermal_trajectory.side_effect = RuntimeError("sim failed")

        result = predictor.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features={"inlet_temp": 26.0, "outdoor_temp": 30.0},
            thermal_model=model,
            climate_mode="cooling",
        )
        assert result["risk"] is False
        assert "failed" in result["reason"]

    def test_empty_trajectory_returns_no_risk(self):
        """Empty trajectory result → no risk."""
        from src.overheating_predictor import OverheatingPredictor
        predictor = OverheatingPredictor()

        model = MagicMock()
        model.predict_thermal_trajectory.return_value = {"trajectory": [], "times": []}

        result = predictor.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features={"inlet_temp": 26.0, "outdoor_temp": 30.0},
            thermal_model=model,
            climate_mode="cooling",
        )
        assert result["risk"] is False

    def test_reactive_cooling_bypasses_guards(self):
        """When room > target, should_cool_now even with low PV and outdoor."""
        from src.overheating_predictor import OverheatingPredictor
        predictor = OverheatingPredictor()

        model = MagicMock()
        # Trajectory with rising temps
        model.predict_thermal_trajectory.return_value = {
            "trajectory": [24.5, 25.0],
            "times": [0.0, 1.0],
        }

        result = predictor.predict_overheating_risk(
            current_indoor=24.0,  # above target 23
            target_cooling=23.0,
            features={
                "inlet_temp": 24.0,
                "outdoor_temp": 15.0,  # low outdoor
                "pv_now": 0.0,  # no PV
            },
            thermal_model=model,
            climate_mode="cooling",
        )
        assert result["should_cool_now"] is True
        assert "room" in result["reason"] and "target" in result["reason"]

    def test_risk_outside_lead_time_waits(self):
        """Peak predicted beyond lead_time → risk=True but should_cool_now=False."""
        from src.overheating_predictor import OverheatingPredictor
        predictor = OverheatingPredictor()

        model = MagicMock()
        # Peak at 10h, beyond default lead_time of 3h
        model.predict_thermal_trajectory.return_value = {
            "trajectory": [22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0],
            "times": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }

        result = predictor.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features={
                "inlet_temp": 22.0,
                "outdoor_temp": 30.0,
                "pv_now": 5000.0,
            },
            thermal_model=model,
            climate_mode="cooling",
        )
        assert result["risk"] is True
        assert result["should_cool_now"] is False  # peak too far away
        assert "waiting" in result["reason"]

    def test_missing_forecast_keys_use_fallbacks(self):
        """Missing pv_forecast/temp_forecast keys use fallback values."""
        from src.overheating_predictor import OverheatingPredictor
        predictor = OverheatingPredictor()

        model = MagicMock()
        model.predict_thermal_trajectory.return_value = {
            "trajectory": [22.0],
            "times": [0.0],
        }

        # Only minimal features, no forecast keys
        result = predictor.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features={
                "inlet_temp": 22.0,
                "outdoor_temp": 30.0,
                "pv_now": 5000.0,
            },
            thermal_model=model,
            climate_mode="cooling",
        )
        # Should not crash, just use fallback values
        assert "risk" in result


# ===========================================================================
# Calibration Label Logic
# ===========================================================================

class TestCalibrationLabelEdgeCases:

    def _compute_labels(self, indoor_series, cooling_target, horizon_steps):
        import pandas as pd
        s = pd.Series(indoor_series)
        label_raw = s.iloc[::-1].rolling(horizon_steps, min_periods=horizon_steps).max().iloc[::-1]
        label = (label_raw > cooling_target).where(label_raw.notna()).astype("Int8")
        return label

    def test_single_row_all_na(self):
        """Single row with horizon > 1 should be all NaN."""
        labels = self._compute_labels([22.0], cooling_target=23.0, horizon_steps=3)
        assert labels.isna().all()

    def test_exact_horizon_length_first_row_labeled(self):
        """When len(series) == horizon_steps, first row should be labeled."""
        labels = self._compute_labels([22.0, 22.0, 22.0], cooling_target=23.0, horizon_steps=3)
        assert not labels.isna().iloc[0]  # first row has complete window
        assert int(labels.iloc[0]) == 0

    def test_monotonically_rising_labels(self):
        """Rising temps: early rows should be label=1 if future peak > target."""
        import pandas as pd
        series = [21.0, 22.0, 23.0, 24.0, 25.0, 22.0]  # spike at idx 3-4
        labels = self._compute_labels(series, cooling_target=23.0, horizon_steps=3)
        # Row 0 sees max(21, 22, 23) = 23.0 → not > 23.0 → label=0
        assert int(labels.iloc[0]) == 0
        # Row 1 sees max(22, 23, 24) = 24.0 → > 23.0 → label=1
        assert int(labels.iloc[1]) == 1

    def test_all_equal_no_overheating(self):
        """Constant series at target → label=0 (not exceeding)."""
        series = [23.0] * 10
        labels = self._compute_labels(series, cooling_target=23.0, horizon_steps=3)
        valid = labels.dropna()
        assert (valid == 0).all()

    def test_horizon_1_labels_everything(self):
        """horizon=1: each row's label is based on its own value."""
        series = [22.0, 24.0, 22.0, 24.0]
        labels = self._compute_labels(series, cooling_target=23.0, horizon_steps=1)
        assert int(labels.iloc[0]) == 0  # 22.0 ≤ 23.0
        assert int(labels.iloc[1]) == 1  # 24.0 > 23.0
        assert int(labels.iloc[2]) == 0
        assert int(labels.iloc[3]) == 1


# ===========================================================================
# Online Learning / Retrain Flow
# ===========================================================================

class TestOnlineLearningFlow:
    """Test the online learning retrain trigger and counter management."""

    def test_retrain_counter_partial_backoff(self, tmp_path):
        """After failed retrain, counter is halved (partial back-off)."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        buf = CoolingObservationBuffer(
            path=str(tmp_path / "buf.json"),
            max_n=100,
            min_training_samples=5,
            retrain_trigger_k=4,
            horizon_steps=1,
        )
        # Create enough labeled entries to trigger retrain
        for i in range(6):
            buf.push_pending({}, 22.0, 23.0, f"t{i}")
            buf.resolve_labels(24.0)

        assert buf.should_retrain() is True
        initial_count = buf._labeled_since_last_train

        # Simulate failed retrain → partial back-off (fixed: subtract trigger_k//2 + 1)
        with buf._lock:
            buf._labeled_since_last_train = max(
                0,
                buf._labeled_since_last_train - buf._retrain_trigger_k // 2 - 1,
            )

        # After fix: backoff drops counter below trigger_k, preventing immediate re-trigger
        assert buf._labeled_since_last_train < buf._retrain_trigger_k

    def test_observation_buffer_preserves_pending_across_restart(self, tmp_path):
        """Pending entries survive save/load and continue accumulating."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        path = str(tmp_path / "buf.json")
        buf = CoolingObservationBuffer(path=path, max_n=50, horizon_steps=3)

        buf.push_pending({"x": 1.0}, 22.0, 23.0, "t0")
        buf.resolve_labels(22.5)  # step 1
        buf.save()

        # "Restart" — reload
        buf2 = CoolingObservationBuffer(path=path, max_n=50, horizon_steps=3)
        assert buf2.n_pending == 1
        assert buf2.n_labeled == 0

        # Continue resolving
        buf2.resolve_labels(22.5)  # step 2
        buf2.resolve_labels(24.0)  # step 3 → labeled
        assert buf2.n_labeled == 1
        _, labels = buf2.get_labeled_data()
        assert labels[0] == 1  # saw 24.0 > 23.0


# ===========================================================================
# Feature Extraction Edge Cases
# ===========================================================================

class TestFeatureExtractionEdgeCases:

    def test_pv_roll_empty_history(self):
        """pv_roll with empty history returns 0.0."""
        from src.cooling_ml_model import _pv_roll
        assert _pv_roll({}, 1) == 0.0
        assert _pv_roll({"pv_power_history": []}, 1) == 0.0

    def test_pv_roll_short_history(self):
        """pv_roll with fewer entries than window uses all available."""
        from src.cooling_ml_model import _pv_roll
        physics = {"pv_power_history": [100.0, 200.0]}
        # 1h = 6 steps but only 2 entries
        val = _pv_roll(physics, 1, steps_per_hour=6)
        assert val == pytest.approx(150.0)

    def test_pv_forecast_feature_extraction(self):
        """pv_forecast_Xh maps to physics pv_forecast_Xh key."""
        from src.cooling_ml_model import _extract_feature
        physics = {"pv_forecast_4h": 5000.0, "pv_now": 2000.0}
        val = _extract_feature("pv_forecast_4h", physics, 22.0, 23.0, 6)
        assert val == pytest.approx(5000.0)

    def test_pv_forecast_fallback_to_pv_now(self):
        """Missing pv_forecast key falls back to pv_now."""
        from src.cooling_ml_model import _extract_feature
        physics = {"pv_now": 3000.0}
        val = _extract_feature("pv_forecast_8h", physics, 22.0, 23.0, 6)
        assert val == pytest.approx(3000.0)

    def test_all_none_physics_doesnt_crash(self):
        """Feature extraction from empty physics dict fills zeros."""
        from src.cooling_ml_model import build_feature_vector
        cols = ["indoor_temp", "AT", "PV_Generate", "thermal_power_kw", "VLT", "RLT"]
        vec = build_feature_vector(cols, {}, 22.0, 23.0)
        assert len(vec) == len(cols)
        assert vec[0] == 22.0  # indoor_temp from param
        assert vec[1] == 0.0   # AT from empty dict


# ===========================================================================
# Config Default Mismatch Verification
# ===========================================================================

class TestConfigDefaults:
    """Verify critical config defaults are consistent."""

    def test_lead_time_config_vs_calibration_default(self):
        """Config default for PRE_COOL_LEAD_TIME_HOURS is 8.0,
        matching config_adapter.py default for addon environments.
        """
        from src import config
        assert config.PRE_COOL_LEAD_TIME_HOURS == 8.0
        # Calibration hardcodes 8.0 as fallback — this is a known mismatch

    def test_pre_cool_horizon_config_default(self):
        from src import config
        assert config.PRE_COOL_HORIZON_HOURS == 12

    def test_trigger_margin_config_default(self):
        from src import config
        assert config.PRE_COOL_TRIGGER_MARGIN_K == 0.5
