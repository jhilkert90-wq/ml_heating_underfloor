"""
tests/unit/test_heating_correction_observation_buffer.py
----------------------------------------------------------
Unit tests for HeatingCorrectionObservationBuffer.
Mirrors the structure of TestCoolingObservationBuffer in test_cooling_ml.py.
"""

from __future__ import annotations

import json
import math
import os
import threading

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_buf(path, max_n=50, min_train=5, trigger_k=3, horizon=4):
    from src.heating_correction_ml_observation_buffer import (
        HeatingCorrectionObservationBuffer,
    )
    return HeatingCorrectionObservationBuffer(
        path=str(path),
        max_n=max_n,
        min_training_samples=min_train,
        retrain_trigger_k=trigger_k,
        horizon_steps=horizon,
    )


# ---------------------------------------------------------------------------
# TestHeatingCorrectionObservationBuffer
# ---------------------------------------------------------------------------

class TestHeatingCorrectionObservationBuffer:
    """Tests for HeatingCorrectionObservationBuffer."""

    @pytest.fixture
    def buf_path(self, tmp_path):
        return str(tmp_path / "heating_obs_buffer.json")

    # ------------------------------------------------------------------
    # Basic push / resolve
    # ------------------------------------------------------------------

    def test_push_creates_pending_entry(self, buf_path):
        buf = _make_buf(buf_path)
        buf.push_pending({"x": 1.0}, 20.0, 21.0, "2024-01-01T00:00:00Z")
        assert buf.n_pending == 1
        assert buf.n_labeled == 0

    def test_resolve_labels_after_horizon(self, buf_path):
        """Entry should be labeled after exactly horizon_steps resolve calls."""
        buf = _make_buf(buf_path, horizon=3)
        buf.push_pending({"x": 1.0}, 20.0, 21.0, "2024-01-01T00:00:00Z")
        # Not yet labeled after 2 steps
        buf.resolve_labels(20.5, s_h=0.5)
        buf.resolve_labels(20.8, s_h=0.5)
        assert buf.n_labeled == 0
        # Labeled at 3rd step
        newly = buf.resolve_labels(21.5, s_h=0.5)
        assert newly == 1
        assert buf.n_labeled == 1

    def test_label_value_undershoot(self, buf_path):
        """Positive label when future_indoor < heating_target (undershoot)."""
        # heating_target=21.0, future_indoor=20.0 → -(20.0 - 21.0)/0.5 = +2.0
        buf = _make_buf(buf_path, horizon=1)
        buf.push_pending({}, 20.0, 21.0, "t0")
        buf.resolve_labels(20.0, s_h=0.5)
        _, labels = buf.get_labeled_data()
        assert len(labels) == 1
        assert math.isclose(labels[0], 2.0, abs_tol=1e-6)

    def test_label_value_overshoot(self, buf_path):
        """Negative label when future_indoor > heating_target (overshoot)."""
        # heating_target=21.0, future_indoor=22.0 → -(22.0 - 21.0)/0.5 = -2.0
        buf = _make_buf(buf_path, horizon=1)
        buf.push_pending({}, 21.0, 21.0, "t0")
        buf.resolve_labels(22.0, s_h=0.5)
        _, labels = buf.get_labeled_data()
        assert math.isclose(labels[0], -2.0, abs_tol=1e-6)

    def test_label_value_on_target(self, buf_path):
        """Zero label when future_indoor == heating_target."""
        buf = _make_buf(buf_path, horizon=1)
        buf.push_pending({}, 21.0, 21.0, "t0")
        buf.resolve_labels(21.0, s_h=0.5)
        _, labels = buf.get_labeled_data()
        assert math.isclose(labels[0], 0.0, abs_tol=1e-6)

    def test_label_clipped_to_max(self, buf_path):
        """Label is clipped to ±10 to prevent extreme values."""
        # Very large undershoot → raw_label >> 10
        buf = _make_buf(buf_path, horizon=1)
        buf.push_pending({}, 21.0, 21.0, "t0")
        buf.resolve_labels(0.0, s_h=0.001)  # huge raw label
        _, labels = buf.get_labeled_data()
        assert labels[0] <= 10.0  # clipped at +10

    def test_label_clip_negative(self, buf_path):
        """Large negative labels are also clipped."""
        buf = _make_buf(buf_path, horizon=1)
        buf.push_pending({}, 21.0, 21.0, "t0")
        buf.resolve_labels(100.0, s_h=0.001)  # huge overshoot
        _, labels = buf.get_labeled_data()
        assert labels[0] >= -10.0

    def test_degenerate_s_h_defers_labeling(self, buf_path):
        """When s_h <= 0, labeling is deferred until s_h is valid."""
        buf = _make_buf(buf_path, horizon=1)
        buf.push_pending({}, 20.0, 21.0, "t0")
        # s_h=0 → should defer
        newly = buf.resolve_labels(20.0, s_h=0.0)
        assert newly == 0
        assert buf.n_labeled == 0
        # Next cycle with valid s_h → should label now
        newly = buf.resolve_labels(20.0, s_h=0.5)
        assert newly == 1
        assert buf.n_labeled == 1

    # ------------------------------------------------------------------
    # Retrain triggering
    # ------------------------------------------------------------------

    def test_should_retrain_no_deadlock(self, buf_path):
        """should_retrain() must not deadlock under its own RLock."""
        buf = _make_buf(buf_path, min_train=2, trigger_k=2, horizon=1)
        result = {}

        def _check():
            result["ok"] = buf.should_retrain()

        t = threading.Thread(target=_check)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "should_retrain() deadlocked!"
        assert result.get("ok") is False

    def test_should_retrain_triggers_correctly(self, buf_path):
        buf = _make_buf(buf_path, min_train=2, trigger_k=2, horizon=1)
        for i in range(3):
            buf.push_pending({}, 20.0, 21.0, f"t{i}")
            buf.resolve_labels(21.5, s_h=0.5)
        assert buf.should_retrain() is True

    def test_should_retrain_false_below_min_samples(self, buf_path):
        """Retrain not triggered until min_training_samples is reached."""
        buf = _make_buf(buf_path, min_train=5, trigger_k=2, horizon=1)
        for i in range(3):
            buf.push_pending({}, 20.0, 21.0, f"t{i}")
            buf.resolve_labels(21.5, s_h=0.5)
        assert buf.should_retrain() is False

    def test_reset_retrain_counter(self, buf_path):
        buf = _make_buf(buf_path, min_train=2, trigger_k=2, horizon=1)
        for i in range(3):
            buf.push_pending({}, 20.0, 21.0, f"t{i}")
            buf.resolve_labels(21.5, s_h=0.5)
        assert buf.should_retrain() is True
        buf.reset_retrain_counter()
        assert buf.should_retrain() is False

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def test_eviction_removes_oldest_labeled(self, buf_path):
        """Buffer respects max_n by evicting oldest labeled entries."""
        buf = _make_buf(buf_path, max_n=5, horizon=1)
        for i in range(5):
            buf.push_pending({"i": float(i)}, 20.0, 21.0, f"t{i}")
            buf.resolve_labels(21.5, s_h=0.5)
        assert buf.n_total == 5
        buf.push_pending({"i": 99.0}, 20.0, 21.0, "t99")
        assert buf.n_total == 5

    def test_eviction_pending_fallback(self, buf_path):
        """Eviction removes pending entries when no labeled entries exist."""
        buf = _make_buf(buf_path, max_n=3, horizon=100)  # never mature
        for i in range(4):
            buf.push_pending({"i": float(i)}, 20.0, 21.0, f"t{i}")
        assert buf.n_total == 3

    def test_eviction_decrements_labeled_counter(self, buf_path):
        """_labeled_since_last_train must not exceed actual labeled count."""
        buf = _make_buf(buf_path, max_n=3, min_train=5, trigger_k=3, horizon=1)
        for i in range(4):
            buf.push_pending({"i": float(i)}, 20.0, 21.0, f"t{i}")
            buf.resolve_labels(21.5, s_h=0.5)
        assert buf._labeled_since_last_train <= buf.n_labeled

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def test_persistence_round_trip(self, buf_path):
        """Loaded buffer has same labeled count and counter as saved one."""
        from src.heating_correction_ml_observation_buffer import (
            HeatingCorrectionObservationBuffer,
        )
        buf = _make_buf(buf_path, horizon=1)
        buf.push_pending({}, 20.0, 21.0, "t0")
        buf.resolve_labels(21.5, s_h=0.5)
        buf._labeled_since_last_train = 1
        buf.save()
        buf2 = HeatingCorrectionObservationBuffer(
            str(buf_path), min_training_samples=5, retrain_trigger_k=3, horizon_steps=1
        )
        assert buf2.n_labeled == 1
        assert buf2._labeled_since_last_train == 1

    def test_load_missing_file_starts_fresh(self, buf_path):
        buf = _make_buf(buf_path)
        assert buf.n_total == 0

    def test_load_corrupt_json_starts_fresh(self, buf_path):
        with open(buf_path, "w") as f:
            f.write("{not valid json")
        buf = _make_buf(buf_path)
        assert buf.n_total == 0

    def test_save_atomic_no_partial_file(self, buf_path):
        """save() uses .tmp → replace so partial writes never appear."""
        buf = _make_buf(buf_path, horizon=1)
        buf.push_pending({}, 20.0, 21.0, "t0")
        buf.resolve_labels(21.5, s_h=0.5)
        buf.save()
        assert os.path.exists(buf_path)
        assert not os.path.exists(buf_path + ".tmp")

    def test_save_creates_parent_directory(self, tmp_path):
        """save() creates parent directories when they don't exist."""
        nested = str(tmp_path / "subdir" / "nested" / "buf.json")
        buf = _make_buf(nested, max_n=10, horizon=1)
        buf.push_pending({}, 20.0, 21.0, "t0")
        buf.resolve_labels(21.5, s_h=0.5)
        buf.save()
        assert os.path.exists(nested)

    def test_horizon_steps_preserved_on_reload(self, buf_path):
        """Saved horizon_steps is restored on reload so pending labels age correctly."""
        from src.heating_correction_ml_observation_buffer import (
            HeatingCorrectionObservationBuffer,
        )
        buf = _make_buf(buf_path, horizon=7)
        buf.save()
        buf2 = HeatingCorrectionObservationBuffer(str(buf_path))
        assert buf2._horizon_steps == 7

    def test_save_snapshot_not_live_reference(self, buf_path):
        """save() round-trips cleanly even at buffer cap."""
        from src.heating_correction_ml_observation_buffer import (
            HeatingCorrectionObservationBuffer,
        )
        buf = _make_buf(buf_path, max_n=3, horizon=1)
        for i in range(3):
            buf.push_pending({"i": float(i)}, 20.0, 21.0, f"t{i}")
            buf.resolve_labels(21.5, s_h=0.5)
        buf.save()
        buf2 = HeatingCorrectionObservationBuffer(
            str(buf_path), max_n=3, min_training_samples=5, retrain_trigger_k=3,
            horizon_steps=1,
        )
        assert buf2.n_labeled == 3

    # ------------------------------------------------------------------
    # Multiple observations
    # ------------------------------------------------------------------

    def test_multiple_pending_at_once(self, buf_path):
        """Several pending entries all mature correctly."""
        buf = _make_buf(buf_path, horizon=2)
        for i in range(5):
            buf.push_pending({}, 20.0, 21.0, f"t{i}")
        buf.resolve_labels(21.0, s_h=0.5)  # step 1
        assert buf.n_labeled == 0
        buf.resolve_labels(21.5, s_h=0.5)  # step 2 → all mature
        assert buf.n_labeled == 5

    def test_get_labeled_data_returns_all(self, buf_path):
        """get_labeled_data returns one entry per labeled observation."""
        buf = _make_buf(buf_path, horizon=1)
        for i in range(3):
            buf.push_pending({"f": float(i)}, 20.0, 21.0, f"t{i}")
            buf.resolve_labels(20.5, s_h=0.5)
        feat_dicts, labels = buf.get_labeled_data()
        assert len(feat_dicts) == 3
        assert len(labels) == 3

    def test_label_uses_current_indoor_at_horizon(self, buf_path):
        """Label uses the indoor temperature at the resolve step, not earlier steps."""
        # horizon=2 → label = -(indoor_at_step2 - target) / s_h
        # Step 1: indoor=20.0 (should NOT be used for label)
        # Step 2: indoor=22.0 → label = -(22.0 - 21.0) / 1.0 = -1.0
        buf = _make_buf(buf_path, horizon=2)
        buf.push_pending({}, 20.0, 21.0, "t0")
        buf.resolve_labels(20.0, s_h=1.0)   # step 1 — not matured yet
        buf.resolve_labels(22.0, s_h=1.0)   # step 2 — matured
        _, labels = buf.get_labeled_data()
        assert math.isclose(labels[0], -1.0, abs_tol=1e-6)
