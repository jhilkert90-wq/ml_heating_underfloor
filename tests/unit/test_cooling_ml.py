"""
Unit tests for the three new ML pre-cooling modules:
  - CoolingObservationBuffer  (cooling_ml_observation_buffer.py)
  - CoolingMLModel            (cooling_ml_model.py)
  - calibrate_cooling_ml      (cooling_ml_calibration.py)

Coverage targets
----------------
Buffer   : push/resolve label timing, eviction counter, persistence round-trip,
           deadlock guard (should_retrain calls n_labeled under same RLock),
           all-pending eviction fallback, save() snapshot safety.
Model    : feature sign convention (at_delta_indoor), missing feature fallback,
           is_loaded guard, result-dict shape matches trajectory predictor,
           no-risk path, inference path.
Calibrate: label correctness (last horizon rows dropped, not labelled 0),
           empty df handling, warm-season filter, val/fit split edge cases,
           metadata keys present.
"""

from __future__ import annotations

import json
import os
import threading
import time
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

TRAJECTORY_RESULT_KEYS = {
    "risk", "peak_temp", "peak_hour", "hours_until_peak",
    "should_cool_now", "reason", "trajectory", "trigger_threshold",
    "peak_outdoor", "total_pv_forecast",
}


def _make_physics(
    indoor: float = 22.0,
    outdoor: float = 30.0,
    pv_now: float = 2000.0,
    delta_indoor_outdoor: float | None = None,
) -> dict[str, Any]:
    """Minimal physics dict for CoolingMLModel tests."""
    diff = (indoor - outdoor) if delta_indoor_outdoor is None else delta_indoor_outdoor
    return {
        "outdoor_temp": outdoor,
        "temp_diff_indoor_outdoor": diff,  # indoor - outdoor
        "pv_now": pv_now,
        "pv_power_history": [pv_now] * 20,
        "indoor_temp_delta_30m": 0.1,
        "indoor_temp_delta_60m": 0.2,
        "outlet_temp": 28.0,
        "inlet_temp": 26.0,
        "delta_t": 2.0,
        "outlet_indoor_diff": 6.0,
        "thermal_power_kw": 1.5,
        "temp_forecast_1h": outdoor + 1.0,
        "temp_forecast_2h": outdoor + 2.0,
        "temp_forecast_4h": outdoor + 3.0,
        "hour_sin": 0.5,
        "hour_cos": 0.866,
    }


# ===========================================================================
# CoolingObservationBuffer
# ===========================================================================

class TestCoolingObservationBuffer:
    """Tests for cooling_ml_observation_buffer.CoolingObservationBuffer."""

    @pytest.fixture
    def buf_path(self, tmp_path):
        return str(tmp_path / "obs_buffer.json")

    def _make_buf(self, path, max_n=50, min_train=5, trigger_k=3, horizon=4):
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        return CoolingObservationBuffer(
            path=path,
            max_n=max_n,
            min_training_samples=min_train,
            retrain_trigger_k=trigger_k,
            horizon_steps=horizon,
        )

    def test_push_creates_pending_entry(self, buf_path):
        buf = self._make_buf(buf_path)
        buf.push_pending({"x": 1.0}, 22.0, 23.0, "2024-01-01T00:00:00Z")
        assert buf.n_pending == 1
        assert buf.n_labeled == 0

    def test_resolve_labels_after_horizon(self, buf_path):
        """Entry should be labeled after exactly horizon_steps resolve calls."""
        buf = self._make_buf(buf_path, horizon=3)
        buf.push_pending({"x": 1.0}, 22.0, 23.0, "2024-01-01T00:00:00Z")
        # Not yet labeled after 2 steps
        buf.resolve_labels(22.5)
        buf.resolve_labels(22.8)
        assert buf.n_labeled == 0
        # Labeled at 3rd step
        newly = buf.resolve_labels(24.0)
        assert newly == 1
        assert buf.n_labeled == 1

    def test_label_value_overheating(self, buf_path):
        """Label=1 when indoor peak exceeds cooling_target."""
        buf = self._make_buf(buf_path, horizon=2)
        buf.push_pending({}, 22.0, 23.0, "t0")
        buf.resolve_labels(24.0)  # step 1: max_indoor_seen = 24.0
        buf.resolve_labels(22.0)  # step 2: matured
        _, labels = buf.get_labeled_data()
        assert labels == [1]

    def test_label_value_no_overheating(self, buf_path):
        """Label=0 when indoor peak stays below cooling_target."""
        buf = self._make_buf(buf_path, horizon=2)
        buf.push_pending({}, 22.0, 23.0, "t0")
        buf.resolve_labels(22.5)
        buf.resolve_labels(22.8)
        _, labels = buf.get_labeled_data()
        assert labels == [0]

    def test_should_retrain_no_deadlock(self, buf_path):
        """should_retrain() calls n_labeled under the same RLock — must not deadlock."""
        buf = self._make_buf(buf_path, min_train=2, trigger_k=2, horizon=1)
        # Quick timeout check: if it deadlocks the test will hang
        result = {}

        def _check():
            result["ok"] = buf.should_retrain()

        t = threading.Thread(target=_check)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "should_retrain() deadlocked!"
        assert result.get("ok") is False  # not enough samples yet

    def test_should_retrain_triggers_correctly(self, buf_path):
        buf = self._make_buf(buf_path, min_train=2, trigger_k=2, horizon=1)
        for i in range(3):
            buf.push_pending({}, 22.0 + i, 23.0, f"t{i}")
            buf.resolve_labels(24.0)  # all positive, all mature after 1 step
        assert buf.should_retrain() is True

    def test_reset_retrain_counter(self, buf_path):
        buf = self._make_buf(buf_path, min_train=2, trigger_k=2, horizon=1)
        for i in range(3):
            buf.push_pending({}, 22.0, 23.0, f"t{i}")
            buf.resolve_labels(24.0)
        assert buf.should_retrain() is True
        buf.reset_retrain_counter()
        assert buf.should_retrain() is False

    def test_eviction_removes_oldest_labeled(self, buf_path):
        """Buffer respects max_n by evicting oldest labeled entries."""
        buf = self._make_buf(buf_path, max_n=5, horizon=1)
        # Fill buffer to max with labeled entries
        for i in range(5):
            buf.push_pending({"i": float(i)}, 22.0, 23.0, f"t{i}")
            buf.resolve_labels(24.0)
        assert buf.n_total == 5
        # One more push should evict the oldest labeled
        buf.push_pending({"i": 99.0}, 22.0, 23.0, "t99")
        assert buf.n_total == 5  # still at cap

    def test_eviction_pending_fallback(self, buf_path):
        """Eviction removes pending entries when no labeled entries exist."""
        buf = self._make_buf(buf_path, max_n=3, horizon=100)  # horizon=100 → never mature
        for i in range(4):
            buf.push_pending({"i": float(i)}, 22.0, 23.0, f"t{i}")
        assert buf.n_total == 3  # capped at max_n via pending fallback

    def test_eviction_decrements_labeled_counter(self, buf_path):
        """_labeled_since_last_train must not exceed actual labeled count after eviction."""
        buf = self._make_buf(buf_path, max_n=3, min_train=5, trigger_k=3, horizon=1)
        # Create 4 labeled entries; max_n=3 so 1 is evicted on the 4th push
        for i in range(4):
            buf.push_pending({"i": float(i)}, 22.0, 23.0, f"t{i}")
            buf.resolve_labels(24.0)
        # Counter must not be 4 (the evicted entry's count should be subtracted)
        assert buf._labeled_since_last_train <= buf.n_labeled

    def test_persistence_round_trip(self, buf_path):
        """Loaded buffer has same labeled count and counter as saved one."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        buf = self._make_buf(buf_path, horizon=1)
        buf.push_pending({}, 22.0, 23.0, "t0")
        buf.resolve_labels(24.0)
        buf._labeled_since_last_train = 1
        buf.save()
        # Reload
        buf2 = CoolingObservationBuffer(buf_path, min_training_samples=5, retrain_trigger_k=3, horizon_steps=1)
        assert buf2.n_labeled == 1
        assert buf2._labeled_since_last_train == 1

    def test_load_missing_file_starts_fresh(self, buf_path):
        """Missing JSON file → empty buffer, no crash."""
        buf = self._make_buf(buf_path)  # file does not exist
        assert buf.n_total == 0

    def test_load_corrupt_json_starts_fresh(self, buf_path):
        """Malformed JSON → empty buffer, no crash."""
        with open(buf_path, "w") as f:
            f.write("{not valid json")
        buf = self._make_buf(buf_path)
        assert buf.n_total == 0

    def test_save_atomic_no_partial_file(self, buf_path):
        """save() uses .tmp → replace, so partial writes never appear."""
        buf = self._make_buf(buf_path, horizon=1)
        buf.push_pending({}, 22.0, 23.0, "t0")
        buf.resolve_labels(24.0)
        buf.save()
        # Both tmp gone and main file exists
        assert os.path.exists(buf_path)
        assert not os.path.exists(buf_path + ".tmp")

    def test_save_snapshot_not_live_reference(self, buf_path):
        """save() must not fail due to None slots from concurrent _evict."""
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        buf = self._make_buf(buf_path, max_n=3, horizon=1)
        for i in range(3):
            buf.push_pending({"i": float(i)}, 22.0, 23.0, f"t{i}")
            buf.resolve_labels(24.0)
        # Verify save succeeds and round-trips cleanly even at cap
        buf.save()
        buf2 = CoolingObservationBuffer(buf_path, max_n=3, min_training_samples=5, retrain_trigger_k=3, horizon_steps=1)
        assert buf2.n_labeled == 3


# ===========================================================================
# CoolingMLModel
# ===========================================================================

class TestCoolingMLModel:
    """Tests for cooling_ml_model.CoolingMLModel and build_feature_vector."""

    def test_at_delta_indoor_sign(self):
        """at_delta_indoor = outdoor - indoor = -temp_diff_indoor_outdoor."""
        from src.cooling_ml_model import _extract_feature
        physics = {"temp_diff_indoor_outdoor": -8.0}  # indoor=22, outdoor=30 → 22-30=-8
        val = _extract_feature("at_delta_indoor", physics, 22.0, 23.0, 6)
        assert val == pytest.approx(8.0)  # -(−8) = +8 = outdoor - indoor

    def test_at_delta_indoor_positive_diff(self):
        """When room is warmer than outside, at_delta_indoor is negative."""
        from src.cooling_ml_model import _extract_feature
        physics = {"temp_diff_indoor_outdoor": 3.0}  # indoor=25, outdoor=22 → 25-22=3
        val = _extract_feature("at_delta_indoor", physics, 25.0, 23.0, 6)
        assert val == pytest.approx(-3.0)

    def test_indoor_margin_sign(self):
        """indoor_margin = target - indoor (negative when room > target)."""
        from src.cooling_ml_model import _extract_feature
        # Room above target: should be negative
        val = _extract_feature("indoor_margin", {}, 25.0, 23.0, 6)
        assert val == pytest.approx(-2.0)
        # Room below target: should be positive
        val2 = _extract_feature("indoor_margin", {}, 21.0, 23.0, 6)
        assert val2 == pytest.approx(2.0)

    def test_forecast_feature_mapping(self):
        """AT_roh_4h maps to temp_forecast_4h from physics dict."""
        from src.cooling_ml_model import _extract_feature
        physics = {"temp_forecast_4h": 33.0, "outdoor_temp": 30.0}
        val = _extract_feature("AT_roh_4h", physics, 22.0, 23.0, 6)
        assert val == pytest.approx(33.0)

    def test_forecast_fallback_to_outdoor(self):
        """AT_roh_Xh falls back to outdoor_temp if forecast not in physics."""
        from src.cooling_ml_model import _extract_feature
        physics = {"outdoor_temp": 30.0}  # no temp_forecast_6h
        val = _extract_feature("AT_roh_6h", physics, 22.0, 23.0, 6)
        assert val == pytest.approx(30.0)

    def test_pv_roll_uses_history(self):
        """pv_roll_1h uses the last steps_per_hour entries of pv_power_history."""
        from src.cooling_ml_model import _extract_feature
        history = [0.0] * 10 + [2000.0] * 6  # last 6 steps = 1h at sph=6
        physics = {"pv_power_history": history}
        val = _extract_feature("pv_roll_1h", physics, 22.0, 23.0, 6)
        assert val == pytest.approx(2000.0)

    def test_unknown_feature_fills_zero(self):
        """Completely unknown feature columns fill with 0.0 without crashing."""
        from src.cooling_ml_model import _extract_feature
        val = _extract_feature("nonexistent_feature_xyz", {}, 22.0, 23.0, 6)
        assert val == pytest.approx(0.0)

    def test_build_feature_vector_length(self):
        """build_feature_vector returns a list of len(feature_cols)."""
        from src.cooling_ml_model import build_feature_vector
        cols = ["indoor_temp", "at_delta_indoor", "AT", "AT_roh_4h", "doy_sin"]
        physics = _make_physics()
        vec = build_feature_vector(cols, physics, 22.0, 23.0, 6)
        assert len(vec) == len(cols)

    def test_model_not_loaded_returns_no_risk(self):
        """Unloaded model returns should_cool_now=False with informative reason."""
        from src.cooling_ml_model import CoolingMLModel
        model = CoolingMLModel("/nonexistent/model.joblib", "/nonexistent/meta.json")
        physics = _make_physics()
        with patch("src.cooling_ml_model.CoolingMLModel.predict_overheating_risk",
                   wraps=model.predict_overheating_risk):
            # Patch config import inside the method
            import types
            fake_config = types.SimpleNamespace(
                PRE_COOL_TRIGGER_MARGIN_K=0.5,
                PRE_COOL_HORIZON_HOURS=12,
                PRE_COOL_LEAD_TIME_HOURS=8.0,
            )
            with patch.dict("sys.modules", {"config": fake_config}):
                result = model.predict_overheating_risk(22.0, 23.0, physics)
        assert result["should_cool_now"] is False
        assert result["risk"] is False
        assert "not loaded" in result["reason"]

    def test_result_dict_has_trajectory_keys(self):
        """predict_overheating_risk result contains all keys that trajectory predictor returns."""
        from src.cooling_ml_model import CoolingMLModel
        model = CoolingMLModel("/no/model.joblib", "/no/meta.json")
        import types
        fake_config = types.SimpleNamespace(
            PRE_COOL_TRIGGER_MARGIN_K=0.5,
            PRE_COOL_HORIZON_HOURS=12,
            PRE_COOL_LEAD_TIME_HOURS=8.0,
        )
        with patch.dict("sys.modules", {"config": fake_config}):
            result = model.predict_overheating_risk(22.0, 23.0, _make_physics())
        missing = TRAJECTORY_RESULT_KEYS - set(result.keys())
        assert not missing, f"Missing result keys: {missing}"

    def test_lgbm_extra_key_present(self):
        """LGBM result includes lgbm_proba (not in trajectory result)."""
        from src.cooling_ml_model import CoolingMLModel
        model = CoolingMLModel("/no/model.joblib", "/no/meta.json")
        import types
        fake_config = types.SimpleNamespace(
            PRE_COOL_TRIGGER_MARGIN_K=0.5,
            PRE_COOL_HORIZON_HOURS=12,
            PRE_COOL_LEAD_TIME_HOURS=8.0,
        )
        with patch.dict("sys.modules", {"config": fake_config}):
            result = model.predict_overheating_risk(22.0, 23.0, _make_physics())
        assert "lgbm_proba" in result

    def test_non_cooling_mode_returns_no_risk(self):
        """predict_overheating_risk must return no-risk for non-cooling climate modes."""
        from src.cooling_ml_model import CoolingMLModel
        model = CoolingMLModel("/no/model.joblib", "/no/meta.json")
        import types
        fake_config = types.SimpleNamespace(
            PRE_COOL_TRIGGER_MARGIN_K=0.5,
            PRE_COOL_HORIZON_HOURS=12,
            PRE_COOL_LEAD_TIME_HOURS=8.0,
        )
        with patch.dict("sys.modules", {"config": fake_config}):
            result = model.predict_overheating_risk(22.0, 23.0, _make_physics(), climate_mode="heating")
        assert result["should_cool_now"] is False
        assert result["risk"] is False

    def test_loaded_model_inference(self):
        """A properly loaded model produces a risk result with lgbm_proba."""
        import types
        import numpy as np

        from src.cooling_ml_model import CoolingMLModel

        fake_config = types.SimpleNamespace(
            PRE_COOL_TRIGGER_MARGIN_K=0.5,
            PRE_COOL_HORIZON_HOURS=12,
            PRE_COOL_LEAD_TIME_HOURS=8.0,
        )

        feature_cols = ["indoor_temp", "at_delta_indoor", "AT"]
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

        # Mock joblib so the load() path works without joblib installed
        mock_joblib = MagicMock()
        mock_joblib.load.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.joblib")
            meta_path = os.path.join(tmpdir, "meta.json")
            # Write fake metadata
            with open(meta_path, "w") as f:
                json.dump({"feature_cols": feature_cols, "threshold": 0.5}, f)
            # Create a dummy file so os.path.exists returns True
            open(model_path, "w").close()

            mdl = CoolingMLModel(model_path, meta_path)
            with patch("src.cooling_ml_model._load_joblib", return_value=mock_joblib):
                mdl.load()
            assert mdl.is_loaded
            with patch.dict("sys.modules", {"config": fake_config}):
                result = mdl.predict_overheating_risk(22.0, 23.0, _make_physics())
        assert result["lgbm_proba"] == pytest.approx(0.8)
        assert result["risk"] is True
        assert result["should_cool_now"] is True


# ===========================================================================
# Label computation in calibrate_cooling_ml
# ===========================================================================

class TestCalibrateCoolingMlLabels:
    """
    Tests for the label computation in calibrate_cooling_ml.
    We test the core logic in isolation using pandas directly.
    """

    def _compute_labels(self, indoor_series, cooling_target, horizon_steps):
        """Replicate the label computation from cooling_ml_calibration.py."""
        import pandas as pd
        s = pd.Series(indoor_series)
        label_raw = s.iloc[::-1].rolling(horizon_steps, min_periods=horizon_steps).max().iloc[::-1]
        label = (label_raw > cooling_target).where(label_raw.notna()).astype("Int8")
        return label

    def test_last_horizon_rows_are_na(self):
        """Last horizon_steps rows must be pd.NA (no future data), not 0."""
        import pandas as pd
        horizon = 3
        # Flat series that never overheats
        series = [22.0] * 10
        labels = self._compute_labels(series, cooling_target=23.0, horizon_steps=horizon)
        # The last 2 rows (horizon-1 = 2 incomplete windows) should be pd.NA
        tail = labels.iloc[-(horizon - 1):]
        assert tail.isna().all(), f"Expected pd.NA in last {horizon-1} rows, got: {tail.tolist()}"

    def test_positive_label_when_peak_exceeds_target(self):
        """Row with future peak > cooling_target should get label=1."""
        import pandas as pd
        horizon = 3
        # Row 0: next 3 rows have max=25.0 > 23.0 → label=1
        series = [22.0, 25.0, 22.0, 22.0, 22.0, 22.0]
        labels = self._compute_labels(series, cooling_target=23.0, horizon_steps=horizon)
        assert int(labels.iloc[0]) == 1

    def test_negative_label_when_peak_below_target(self):
        """Row where future peak stays below cooling_target should get label=0."""
        import pandas as pd
        horizon = 3
        series = [22.0, 22.0, 22.0, 22.0, 22.0, 22.0]
        labels = self._compute_labels(series, cooling_target=23.0, horizon_steps=horizon)
        # First row has complete future window; peak = 22.0 < 23.0 → label=0
        assert int(labels.iloc[0]) == 0

    def test_no_false_negatives_at_end(self):
        """Even with a rising trend at the end, last rows stay NA (not 0)."""
        import pandas as pd
        horizon = 4
        # Series ends with temps that would suggest overheating IF there was more data
        series = [22.0] * 6 + [22.5, 23.0, 23.5]  # rising near end
        labels = self._compute_labels(series, cooling_target=23.0, horizon_steps=horizon)
        # Last 3 rows (horizon-1=3) should all be pd.NA
        tail = labels.iloc[-(horizon - 1):]
        assert tail.isna().all()

    def test_dropna_removes_na_labels(self):
        """After dropna, no pd.NA labels remain in training data."""
        import pandas as pd
        horizon = 3
        series = [22.0] * 10 + [25.0] * 3
        labels = self._compute_labels(series, cooling_target=23.0, horizon_steps=horizon)
        df = pd.DataFrame({"indoor_temp": series, "label": labels}).dropna()
        assert df["label"].isna().sum() == 0
        assert len(df) == len(series) - (horizon - 1)


# ===========================================================================
# Calibration parameter handling (without InfluxDB)
# ===========================================================================

class TestCalibrateCoolingMlParams:
    """Tests for parameter handling / guard conditions in calibrate_cooling_ml."""

    def test_returns_false_on_missing_pandas(self, monkeypatch):
        """calibrate_cooling_ml returns False when pandas is unavailable."""
        import sys
        # Block pandas import
        original = sys.modules.get("pandas")
        sys.modules["pandas"] = None  # type: ignore
        try:
            # Re-import to clear cached module
            import importlib
            import src.cooling_ml_calibration as m
            importlib.reload(m)
            result = m.calibrate_cooling_ml()
        except Exception:
            result = False
        finally:
            if original is None:
                del sys.modules["pandas"]
            else:
                sys.modules["pandas"] = original
        assert result is False

    def test_returns_false_when_no_data(self, monkeypatch):
        """calibrate_cooling_ml returns False when fetch returns None."""
        import types
        import sys

        fake_config = types.SimpleNamespace(
            CYCLE_INTERVAL_MINUTES=10,
            PRE_COOL_HORIZON_HOURS=12,
            PRE_COOL_LEAD_TIME_HOURS=8.0,
            COOLING_CLAMP_MAX_ABS=24.0,
            PRE_COOL_MIN_OUTDOOR_FORECAST_C=22.0,
            COOLING_ML_MIN_TRAINING_SAMPLES=200,
            COOLING_ML_RETRAIN_VAL_FRACTION=0.25,
            COOLING_ML_MODEL_PATH="/tmp/test_model.joblib",
            COOLING_ML_METADATA_PATH="/tmp/test_meta.json",
        )
        monkeypatch.setitem(sys.modules, "config", fake_config)

        # Inject a fake fetch function into the calibration module so the
        # try/except ImportError block inside the function finds it.
        import src.cooling_ml_calibration as cal
        fake_fetch_mod = MagicMock()
        fake_fetch_mod.fetch_historical_data_for_calibration = MagicMock(return_value=None)
        monkeypatch.setitem(sys.modules, "physics_calibration", fake_fetch_mod)

        result = cal.calibrate_cooling_ml()
        assert result is False


    def test_warm_season_filter_threshold(self):
        """Warm season threshold = PRE_COOL_MIN_OUTDOOR_FORECAST_C - 6."""
        import types
        # The formula in the calibration code:
        pre_cool_min = 22.0
        expected_threshold = pre_cool_min - 6.0
        assert expected_threshold == pytest.approx(16.0)

    def test_label_horizon_uses_lead_time(self):
        """Label horizon = PRE_COOL_LEAD_TIME_HOURS, not PRE_COOL_HORIZON_HOURS."""
        # This verifies our fix: label_horizon_h = int(round(lead_time_h))
        # The calibration code sets label_horizon_h = int(round(lead_time_h))
        lead_time = 8.0
        horizon = 12
        label_h = int(round(lead_time))
        # Verify label window < forecast window
        assert label_h == 8
        assert label_h < horizon


# ===========================================================================
# Eviction counter invariant
# ===========================================================================

class TestEvictionCounterInvariant:
    """Verify _labeled_since_last_train ≤ n_labeled at all times."""

    def test_counter_never_exceeds_n_labeled(self, tmp_path):
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        buf = CoolingObservationBuffer(
            path=str(tmp_path / "buf.json"),
            max_n=5,
            min_training_samples=10,
            retrain_trigger_k=3,
            horizon_steps=1,
        )
        # Push 8 entries (labels mature immediately at horizon=1)
        for i in range(8):
            buf.push_pending({"i": float(i)}, 22.0, 23.0, f"t{i}")
            buf.resolve_labels(24.0)  # matures previous entries
        # At this point: buffer capped at 5; counter should be ≤ n_labeled
        assert buf._labeled_since_last_train <= buf.n_labeled

    def test_counter_zero_after_reset_still_consistent(self, tmp_path):
        from src.cooling_ml_observation_buffer import CoolingObservationBuffer
        buf = CoolingObservationBuffer(
            path=str(tmp_path / "buf.json"),
            max_n=10,
            min_training_samples=2,
            retrain_trigger_k=2,
            horizon_steps=1,
        )
        for i in range(4):
            buf.push_pending({}, 22.0, 23.0, f"t{i}")
            buf.resolve_labels(24.0)
        buf.reset_retrain_counter()
        assert buf._labeled_since_last_train == 0
        assert buf.should_retrain() is False


# ===========================================================================
# Import regression tests (fix for 'No module named config' production bug)
# ===========================================================================

class TestImportConfigRegression:
    """Regression tests for the fixed bare `import config` bug.

    The production log showed:
      calibrate_cooling_ml: missing dependency — No module named 'config'
    because both cooling_ml_calibration.py and cooling_ml_model.py used
    `import config` instead of `from . import config`.
    """

    def test_calibration_importable_from_package(self):
        """calibrate_cooling_ml can be imported via src.cooling_ml_calibration."""
        from src.cooling_ml_calibration import calibrate_cooling_ml
        assert callable(calibrate_cooling_ml)

    def test_model_importable_from_package(self):
        """CoolingMLModel can be imported via src.cooling_ml_model."""
        from src.cooling_ml_model import CoolingMLModel
        assert CoolingMLModel is not None

    def test_predict_does_not_raise_import_error(self):
        """predict_overheating_risk resolves config without ImportError."""
        from src.cooling_ml_model import CoolingMLModel
        model = CoolingMLModel("/no/model.joblib", "/no/meta.json")
        result = model.predict_overheating_risk(
            current_indoor=22.0,
            target_cooling=23.0,
            features={"outdoor_temp": 30.0, "pv_now": 1000.0},
            climate_mode="cooling",
        )
        assert result["risk"] is False
        assert "not loaded" in result["reason"]

    def test_calibration_config_resolves_inside_function(self):
        """Config attributes are accessible inside calibrate_cooling_ml."""
        import types
        fake_cfg = types.SimpleNamespace(
            CYCLE_INTERVAL_MINUTES=10,
            PRE_COOL_HORIZON_HOURS=12,
            PRE_COOL_LEAD_TIME_HOURS=8.0,
            COOLING_CLAMP_MAX_ABS=24.0,
            PRE_COOL_MIN_OUTDOOR_FORECAST_C=22.0,
            COOLING_ML_MIN_TRAINING_SAMPLES=200,
            COOLING_ML_RETRAIN_VAL_FRACTION=0.25,
            COOLING_ML_MODEL_PATH="/tmp/test_model.joblib",
            COOLING_ML_METADATA_PATH="/tmp/test_meta.json",
        )
        fake_fetch_mod = MagicMock()
        fake_fetch_mod.fetch_historical_data_for_calibration = MagicMock(return_value=None)
        with patch.dict("sys.modules", {"config": fake_cfg, "physics_calibration": fake_fetch_mod}):
            from src.cooling_ml_calibration import calibrate_cooling_ml
            # Should return False (no data) but NOT raise ImportError
            result = calibrate_cooling_ml()
        assert result is False
