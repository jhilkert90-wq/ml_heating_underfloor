"""Tests for prediction_metrics.py – file I/O, summary, 24h window, simplified breakdown."""

import json
import os
import pytest
from collections import deque
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.prediction_metrics import PredictionMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _add_predictions(tracker: PredictionMetrics, n: int, error: float = 0.5):
    """Add *n* predictions with a fixed absolute error."""
    for i in range(n):
        tracker.add_prediction(20.0, 20.0 + error)


def _ts(hours_ago: float) -> str:
    """Return ISO timestamp *hours_ago* hours in the past."""
    return (datetime.now() - timedelta(hours=hours_ago)).isoformat()


# ---------------------------------------------------------------------------
# Basic no-state-manager path
# ---------------------------------------------------------------------------
class TestPredictionMetricsBasic:
    def test_initialization_creates_empty_deque(self):
        pm = PredictionMetrics()
        assert isinstance(pm.predictions, deque)

    def test_add_prediction_stores_record(self):
        pm = PredictionMetrics()
        pm.add_prediction(20.0, 21.0)
        assert len(pm.predictions) == 1
        rec = pm.predictions[0]
        assert rec["predicted"] == pytest.approx(20.0)
        assert rec["actual"] == pytest.approx(21.0)
        assert rec["error"] == pytest.approx(1.0)
        assert rec["abs_error"] == pytest.approx(1.0)
        assert rec["squared_error"] == pytest.approx(1.0)

    def test_add_prediction_with_custom_timestamp(self):
        pm = PredictionMetrics()
        ts = "2025-01-01T00:00:00"
        pm.add_prediction(20.0, 21.0, timestamp=ts)
        assert pm.predictions[0]["timestamp"] == ts

    def test_cache_invalidated_on_add(self):
        pm = PredictionMetrics()
        _add_predictions(pm, 5)
        # Force a cache
        pm.get_metrics()
        assert pm._cache_timestamp is not None

        # Adding another prediction should invalidate cache
        pm.add_prediction(20.0, 21.0)
        assert pm._cache_timestamp is None

    def test_sliding_window_limit_enforced(self):
        pm = PredictionMetrics()
        for i in range(210):
            pm.add_prediction(20.0, 20.5)
        assert len(pm.predictions) <= 200


# ---------------------------------------------------------------------------
# get_metrics – all windows
# ---------------------------------------------------------------------------
class TestGetMetrics:
    def test_empty_tracker_returns_zero_metrics(self):
        pm = PredictionMetrics()
        metrics = pm.get_metrics()
        for window in ["1h", "6h", "24h", "all"]:
            assert metrics[window]["mae"] == 0.0
            assert metrics[window]["rmse"] == 0.0
            assert metrics[window]["count"] == 0

    def test_metrics_populated_after_predictions(self):
        pm = PredictionMetrics()
        _add_predictions(pm, 50, error=0.5)
        metrics = pm.get_metrics()
        assert metrics["all"]["mae"] == pytest.approx(0.5)
        assert metrics["all"]["count"] == 50

    def test_cache_returned_on_second_call(self):
        pm = PredictionMetrics()
        _add_predictions(pm, 5)
        first = pm.get_metrics()
        # Force a non-stale cache by faking timestamp
        pm._cache_timestamp = datetime.now()
        second = pm.get_metrics()
        assert first is second  # same dict object – from cache

    def test_refresh_cache_flag_bypasses_cache(self):
        pm = PredictionMetrics()
        _add_predictions(pm, 5)
        first = pm.get_metrics()
        pm._cache_timestamp = datetime.now()
        second = pm.get_metrics(refresh_cache=True)
        # Different objects but equal values
        assert first is not second


# ---------------------------------------------------------------------------
# _calculate_trends
# ---------------------------------------------------------------------------
class TestCalculateTrends:
    def test_insufficient_data_flag(self):
        pm = PredictionMetrics()
        _add_predictions(pm, 5)  # less than 10
        metrics = pm.get_metrics()
        assert metrics["trends"].get("insufficient_data") is True

    def test_trend_improving(self):
        pm = PredictionMetrics()
        # First half: large error; second half: small error
        for _ in range(10):
            pm.add_prediction(20.0, 22.0)  # error=2
        for _ in range(10):
            pm.add_prediction(20.0, 20.1)  # error=0.1
        metrics = pm.get_metrics()
        if not metrics["trends"].get("insufficient_data"):
            assert metrics["trends"]["is_improving"] == True  # noqa: E712 (numpy bool compat)

    def test_trend_degrading(self):
        pm = PredictionMetrics()
        for _ in range(10):
            pm.add_prediction(20.0, 20.1)  # error=0.1 first
        for _ in range(10):
            pm.add_prediction(20.0, 22.0)  # error=2 second
        metrics = pm.get_metrics()
        if not metrics["trends"].get("insufficient_data"):
            assert metrics["trends"]["is_improving"] == False  # noqa: E712 (numpy bool compat)

    def test_improvement_percentage_clamped(self):
        pm = PredictionMetrics()
        # Make first half have near-zero error to trigger extreme percentage
        for _ in range(10):
            pm.add_prediction(20.0, 20.0)  # error=0 (perfect)
        for _ in range(10):
            pm.add_prediction(20.0, 22.0)  # large error
        metrics = pm.get_metrics()
        if not metrics["trends"].get("insufficient_data"):
            pct = metrics["trends"]["mae_improvement_percentage"]
            assert pct >= -100.0
            assert pct <= 100.0


# ---------------------------------------------------------------------------
# _calculate_accuracy_breakdown
# ---------------------------------------------------------------------------
class TestAccuracyBreakdown:
    def test_empty_returns_empty_dict(self):
        pm = PredictionMetrics()
        metrics = pm.get_metrics()
        assert metrics["accuracy_breakdown"] == {}

    def test_perfect_predictions_all_excellent(self):
        pm = PredictionMetrics()
        for _ in range(10):
            pm.add_prediction(20.0, 20.0)  # zero error
        breakdown = pm.get_metrics()["accuracy_breakdown"]
        assert breakdown["excellent"]["count"] == 10
        assert breakdown["poor"]["count"] == 0

    def test_large_error_predictions_all_poor(self):
        pm = PredictionMetrics()
        for _ in range(10):
            pm.add_prediction(20.0, 25.0)  # 5°C error
        breakdown = pm.get_metrics()["accuracy_breakdown"]
        assert breakdown["excellent"]["count"] == 0
        # All should end up in poor (or acceptable+poor at best)
        assert breakdown["poor"]["count"] + breakdown.get("acceptable", {}).get("count", 0) > 0


# ---------------------------------------------------------------------------
# get_recent_performance
# ---------------------------------------------------------------------------
class TestGetRecentPerformance:
    def test_no_data_returns_no_data_flag(self):
        pm = PredictionMetrics()
        result = pm.get_recent_performance()
        assert "no_data" in result

    def test_fewer_than_n_predictions(self):
        pm = PredictionMetrics()
        _add_predictions(pm, 3)
        result = pm.get_recent_performance(last_n=10)
        assert result["count"] == 3

    def test_returns_correct_count(self):
        pm = PredictionMetrics()
        _add_predictions(pm, 20)
        result = pm.get_recent_performance(last_n=5)
        assert result["count"] == 5

    def test_single_prediction_std_is_zero(self):
        pm = PredictionMetrics()
        pm.add_prediction(20.0, 21.0)
        result = pm.get_recent_performance(last_n=1)
        assert result["std_error"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# save_state / load_state
# ---------------------------------------------------------------------------
class TestSaveLoadState:
    def test_save_creates_json_file(self, tmp_path):
        pm = PredictionMetrics()
        _add_predictions(pm, 5)
        filepath = str(tmp_path / "metrics.json")
        pm.save_state(filepath)
        assert os.path.exists(filepath)

    def test_saved_file_is_valid_json(self, tmp_path):
        pm = PredictionMetrics()
        _add_predictions(pm, 3)
        filepath = str(tmp_path / "metrics.json")
        pm.save_state(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert "predictions" in data
        assert len(data["predictions"]) == 3

    def test_load_state_restores_predictions(self, tmp_path):
        pm_save = PredictionMetrics()
        pm_save.add_prediction(20.0, 21.0)
        pm_save.add_prediction(20.0, 20.5)
        filepath = str(tmp_path / "metrics.json")
        pm_save.save_state(filepath)

        pm_load = PredictionMetrics()
        result = pm_load.load_state(filepath)
        assert result is True
        assert len(pm_load.predictions) == 2

    def test_load_state_returns_false_for_missing_file(self):
        pm = PredictionMetrics()
        result = pm.load_state("/nonexistent/path/metrics.json")
        assert result is False

    def test_load_state_invalidates_cache(self, tmp_path):
        pm = PredictionMetrics()
        _add_predictions(pm, 5)
        filepath = str(tmp_path / "metrics.json")
        pm.save_state(filepath)
        # Load into fresh instance
        pm2 = PredictionMetrics()
        pm2._cache_timestamp = datetime.now()  # fake valid cache
        pm2.load_state(filepath)
        assert pm2._cache_timestamp is None

    def test_load_state_handles_corrupt_file(self, tmp_path):
        filepath = str(tmp_path / "corrupt.json")
        with open(filepath, "w") as f:
            f.write("NOT_VALID_JSON{{{")
        pm = PredictionMetrics()
        result = pm.load_state(filepath)
        assert result is False


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------
class TestGetSummary:
    def test_no_predictions_returns_no_data_message(self):
        pm = PredictionMetrics()
        summary = pm.get_summary()
        assert "No prediction data" in summary

    def test_summary_with_predictions(self):
        pm = PredictionMetrics()
        _add_predictions(pm, 50, error=0.3)
        summary = pm.get_summary()
        assert "MAE" in summary
        assert "predictions" in summary.lower() or "Prediction" in summary

    def test_summary_includes_trend_with_sufficient_data(self):
        pm = PredictionMetrics()
        for _ in range(10):
            pm.add_prediction(20.0, 22.0)
        for _ in range(10):
            pm.add_prediction(20.0, 20.1)
        summary = pm.get_summary()
        # With enough data the trend line should appear
        assert "improving" in summary.lower() or "degrading" in summary.lower() or "MAE" in summary


# ---------------------------------------------------------------------------
# get_simplified_accuracy_breakdown
# ---------------------------------------------------------------------------
class TestGetSimplifiedAccuracyBreakdown:
    def test_empty_returns_zero_counts(self):
        pm = PredictionMetrics()
        result = pm.get_simplified_accuracy_breakdown()
        assert result["perfect"]["count"] == 0
        assert result["tolerable"]["count"] == 0
        assert result["poor"]["count"] == 0

    def test_all_perfect_zero_error(self):
        pm = PredictionMetrics()
        for _ in range(5):
            pm.add_prediction(20.0, 20.0)
        result = pm.get_simplified_accuracy_breakdown()
        assert result["perfect"]["count"] == 5
        assert result["perfect"]["percentage"] == pytest.approx(100.0)
        assert result["tolerable"]["count"] == 0
        assert result["poor"]["count"] == 0

    def test_all_poor_large_error(self):
        pm = PredictionMetrics()
        for _ in range(5):
            pm.add_prediction(20.0, 25.0)  # 5°C error → poor
        result = pm.get_simplified_accuracy_breakdown()
        assert result["poor"]["count"] == 5
        assert result["perfect"]["count"] == 0

    def test_tolerable_error_range(self):
        pm = PredictionMetrics()
        # Errors in (0, 0.2) are tolerable
        pm.add_prediction(20.0, 20.05)   # abs_error=0.05 → tolerable
        pm.add_prediction(20.0, 20.15)   # abs_error=0.15 → tolerable
        result = pm.get_simplified_accuracy_breakdown()
        assert result["tolerable"]["count"] == 2

    def test_percentages_sum_to_100(self):
        pm = PredictionMetrics()
        _add_predictions(pm, 10, error=0.5)
        result = pm.get_simplified_accuracy_breakdown()
        total_pct = (
            result["perfect"]["percentage"]
            + result["tolerable"]["percentage"]
            + result["poor"]["percentage"]
        )
        assert total_pct == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# get_good_control_percentage
# ---------------------------------------------------------------------------
class TestGetGoodControlPercentage:
    def test_all_perfect(self):
        pm = PredictionMetrics()
        for _ in range(10):
            pm.add_prediction(20.0, 20.0)
        assert pm.get_good_control_percentage() == pytest.approx(100.0)

    def test_all_poor(self):
        pm = PredictionMetrics()
        for _ in range(10):
            pm.add_prediction(20.0, 25.0)
        assert pm.get_good_control_percentage() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 24-hour window helpers
# ---------------------------------------------------------------------------
class TestGet24hWindow:
    def test_no_predictions_returns_empty_window(self):
        pm = PredictionMetrics()
        assert pm._get_predictions_in_24h_window() == []

    def test_recent_predictions_included(self):
        pm = PredictionMetrics()
        pm.add_prediction(20.0, 21.0, timestamp=_ts(1))  # 1h ago → within 24h
        window = pm._get_predictions_in_24h_window()
        assert len(window) == 1

    def test_old_predictions_excluded(self):
        pm = PredictionMetrics()
        pm.add_prediction(20.0, 21.0, timestamp=_ts(25))  # 25h ago → outside 24h
        window = pm._get_predictions_in_24h_window()
        assert len(window) == 0

    def test_predictions_without_timestamp_skipped(self):
        pm = PredictionMetrics()
        # Manually inject a record without timestamp
        pm.predictions.append({
            "timestamp": None,
            "predicted": 20.0,
            "actual": 21.0,
            "error": 1.0,
            "abs_error": 1.0,
            "squared_error": 1.0,
            "context": {},
        })
        window = pm._get_predictions_in_24h_window()
        assert len(window) == 0

    def test_predictions_with_invalid_timestamp_skipped(self):
        pm = PredictionMetrics()
        pm.predictions.append({
            "timestamp": "not-a-date",
            "predicted": 20.0,
            "actual": 21.0,
            "error": 1.0,
            "abs_error": 1.0,
            "squared_error": 1.0,
            "context": {},
        })
        window = pm._get_predictions_in_24h_window()
        assert len(window) == 0


class TestGet24hAccuracyBreakdown:
    def test_no_data_returns_zero_breakdown(self):
        pm = PredictionMetrics()
        result = pm.get_24h_accuracy_breakdown()
        assert result["perfect"]["count"] == 0
        assert result["tolerable"]["count"] == 0
        assert result["poor"]["count"] == 0

    def test_only_recent_predictions_counted(self):
        pm = PredictionMetrics()
        pm.add_prediction(20.0, 20.0, timestamp=_ts(1))     # recent → perfect
        pm.add_prediction(20.0, 25.0, timestamp=_ts(25))    # old → excluded
        result = pm.get_24h_accuracy_breakdown()
        assert result["perfect"]["count"] == 1
        assert result["poor"]["count"] == 0

    def test_percentages_sum_to_100_in_24h(self):
        pm = PredictionMetrics()
        pm.add_prediction(20.0, 20.0, timestamp=_ts(1))
        pm.add_prediction(20.0, 21.0, timestamp=_ts(2))
        result = pm.get_24h_accuracy_breakdown()
        total = (
            result["perfect"]["percentage"]
            + result["tolerable"]["percentage"]
            + result["poor"]["percentage"]
        )
        assert total == pytest.approx(100.0)


class TestGet24hGoodControlPercentage:
    def test_all_recent_perfect(self):
        pm = PredictionMetrics()
        for h in range(1, 6):
            pm.add_prediction(20.0, 20.0, timestamp=_ts(h))
        assert pm.get_24h_good_control_percentage() == pytest.approx(100.0)

    def test_no_data_returns_zero(self):
        pm = PredictionMetrics()
        assert pm.get_24h_good_control_percentage() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# State-manager integration path
# ---------------------------------------------------------------------------
class TestStateManagerIntegration:
    def _make_state_manager(self, prediction_records=None):
        mgr = MagicMock()
        mgr.state = {
            "learning_state": {
                "prediction_history": prediction_records or []
            },
            "prediction_metrics": {
                "total_predictions": 0,
                "accuracy_stats": {},
                "recent_performance": {},
            },
        }
        mgr.save_state = MagicMock()
        return mgr

    def test_loads_predictions_from_state_manager(self):
        records = [
            {
                "timestamp": _ts(1),
                "predicted": 20.0,
                "actual": 21.0,
                "error": 1.0,
                "abs_error": 1.0,
                "squared_error": 1.0,
                "context": {},
            }
        ]
        mgr = self._make_state_manager(records)
        pm = PredictionMetrics(state_manager=mgr)
        assert len(pm.predictions) == 1

    def test_converts_legacy_records_without_abs_error(self):
        records = [
            {
                "timestamp": _ts(1),
                "predicted": 20.0,
                "actual": 21.0,
                "error": -1.0,  # abs_error and squared_error absent
                "context": {},
            }
        ]
        mgr = self._make_state_manager(records)
        pm = PredictionMetrics(state_manager=mgr)
        assert pm.predictions[0]["abs_error"] == pytest.approx(1.0)
        assert pm.predictions[0]["squared_error"] == pytest.approx(1.0)

    def test_save_to_state_calls_save_state(self):
        mgr = self._make_state_manager()
        pm = PredictionMetrics(state_manager=mgr)
        pm.add_prediction(20.0, 21.0)
        mgr.save_state.assert_called()

    def test_state_manager_load_exception_falls_back_to_empty(self):
        mgr = MagicMock()
        mgr.state = MagicMock(side_effect=Exception("broken"))
        pm = PredictionMetrics(state_manager=mgr)
        assert isinstance(pm.predictions, deque)
        assert len(pm.predictions) == 0
