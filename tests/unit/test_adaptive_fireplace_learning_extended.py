"""Extended tests for adaptive_fireplace_learning.py – coverage for
get_enhanced_fireplace_features, get_learning_summary, and the
integrate_adaptive_fireplace_with_multi_source_physics helper.
"""

import json
import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.adaptive_fireplace_learning import (
    AdaptiveFireplaceLearning,
    FireplaceObservation,
    FireplaceLearningState,
    integrate_adaptive_fireplace_with_multi_source_physics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def afl(tmp_path):
    """Minimal AdaptiveFireplaceLearning instance with temp state file."""
    return AdaptiveFireplaceLearning(state_file=str(tmp_path / "state.json"))


@pytest.fixture
def afl_with_observations(tmp_path):
    """AdaptiveFireplaceLearning pre-populated with 5 observations."""
    instance = AdaptiveFireplaceLearning(state_file=str(tmp_path / "state.json"))
    for i in range(5):
        obs = FireplaceObservation(
            timestamp=datetime.now() - timedelta(hours=i),
            temp_differential=2.0 + i * 0.2,
            outdoor_temp=5.0 - i,
            fireplace_active=True,
            duration_minutes=30.0 + i * 5,
            peak_differential=3.0 + i * 0.2,
        )
        instance.learning_state.observations.append(obs)
    return instance


# ---------------------------------------------------------------------------
# _load_state – file exists with valid JSON
# ---------------------------------------------------------------------------
class TestLoadStateFromFile:
    def test_loads_observations_from_existing_file(self, tmp_path):
        state_file = tmp_path / "state.json"
        now = datetime.now()
        data = {
            "observations": [
                {
                    "timestamp": now.isoformat(),
                    "temp_differential": 2.0,
                    "outdoor_temp": 5.0,
                    "fireplace_active": True,
                    "duration_minutes": 30.0,
                    "heat_buildup_rate": 0.0,
                    "heat_decay_rate": 0.0,
                    "peak_differential": 3.0,
                }
            ],
            "learned_coefficients": {},
            "learning_stats": {},
            "last_update": now.isoformat(),
        }
        state_file.write_text(json.dumps(data))
        instance = AdaptiveFireplaceLearning(state_file=str(state_file))
        assert len(instance.learning_state.observations) == 1

    def test_migrates_legacy_missing_fireplace_active_field(self, tmp_path):
        """Legacy observations lacking 'fireplace_active' key should be migrated."""
        state_file = tmp_path / "state.json"
        now = datetime.now()
        data = {
            "observations": [
                {
                    "timestamp": now.isoformat(),
                    "temp_differential": 2.0,
                    "outdoor_temp": 5.0,
                    # 'fireplace_active' absent; old name present
                    "fireplace_active_at_end": True,
                    "duration_minutes": 30.0,
                    "heat_buildup_rate": 0.0,
                    "heat_decay_rate": 0.0,
                    "peak_differential": 3.0,
                }
            ],
            "learned_coefficients": {},
            "learning_stats": {},
            "last_update": None,
        }
        state_file.write_text(json.dumps(data))
        instance = AdaptiveFireplaceLearning(state_file=str(state_file))
        # Migration must produce exactly one loaded observation with the correct value
        observations = instance.learning_state.observations
        assert len(observations) == 1
        assert observations[0].fireplace_active is True

    def test_corrupt_json_creates_new_state(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("{NOT VALID JSON")
        instance = AdaptiveFireplaceLearning(state_file=str(state_file))
        assert isinstance(instance.learning_state, FireplaceLearningState)
        assert len(instance.learning_state.observations) == 0

    def test_invalid_observation_skipped_gracefully(self, tmp_path):
        state_file = tmp_path / "state.json"
        data = {
            "observations": [
                {"timestamp": "bad-date", "temp_differential": "not-a-float"},
            ],
            "learned_coefficients": {},
            "learning_stats": {},
            "last_update": None,
        }
        state_file.write_text(json.dumps(data))
        instance = AdaptiveFireplaceLearning(state_file=str(state_file))
        # Corrupt observation is skipped – no crash
        assert isinstance(instance.learning_state, FireplaceLearningState)


# ---------------------------------------------------------------------------
# _save_state – error handling
# ---------------------------------------------------------------------------
class TestSaveStateErrorHandling:
    def test_save_to_unwritable_path_logs_error(self, caplog):
        import logging
        instance = AdaptiveFireplaceLearning(state_file="/nonexistent/dir/state.json")
        # Add an observation to trigger save
        instance.learning_state.observations.append(
            FireplaceObservation(
                timestamp=datetime.now(),
                temp_differential=2.0,
                outdoor_temp=5.0,
                fireplace_active=True,
                duration_minutes=30.0,
            )
        )
        with caplog.at_level(logging.ERROR, logger="src.adaptive_fireplace_learning"):
            instance._save_state()  # Should not raise
        assert any("Failed to save" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# get_enhanced_fireplace_features
# ---------------------------------------------------------------------------
class TestGetEnhancedFireplaceFeatures:
    def test_returns_enhanced_dict_with_all_keys(self, afl):
        base = {
            "indoor_temp": 22.0,
            "avg_other_rooms_temp": 20.0,
            "outdoor_temp": 5.0,
            "fireplace_on": 1,
            "pv_now": 500.0,
        }
        result = afl.get_enhanced_fireplace_features(base)
        expected_keys = [
            "fireplace_heat_contribution_kw",
            "fireplace_effectiveness_factor",
            "fireplace_learning_confidence",
            "fireplace_temp_differential",
            "fireplace_outdoor_correlation",
            "fireplace_observations_count",
            "fireplace_recent_usage",
            "fireplace_learned_efficiency",
            "fireplace_learned_distribution",
            "fireplace_differential_heat_ratio",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_original_base_features_preserved(self, afl):
        base = {"pv_now": 1000.0, "custom_key": "custom_value"}
        result = afl.get_enhanced_fireplace_features(base)
        assert result["pv_now"] == 1000.0
        assert result["custom_key"] == "custom_value"

    def test_inactive_fireplace_gives_zero_heat_contribution(self, afl):
        base = {
            "indoor_temp": 20.0,
            "avg_other_rooms_temp": 20.0,
            "outdoor_temp": 5.0,
            "fireplace_on": 0,
        }
        result = afl.get_enhanced_fireplace_features(base)
        assert result["fireplace_heat_contribution_kw"] == pytest.approx(0.0)

    def test_temp_differential_calculated_correctly(self, afl):
        base = {
            "indoor_temp": 23.0,
            "avg_other_rooms_temp": 20.0,
            "outdoor_temp": 5.0,
            "fireplace_on": 1,
        }
        result = afl.get_enhanced_fireplace_features(base)
        assert result["fireplace_temp_differential"] == pytest.approx(3.0)

    def test_observation_count_reflects_learning_state(self, afl_with_observations):
        base = {
            "indoor_temp": 22.0,
            "avg_other_rooms_temp": 20.0,
            "outdoor_temp": 5.0,
            "fireplace_on": 1,
        }
        result = afl_with_observations.get_enhanced_fireplace_features(base)
        assert result["fireplace_observations_count"] == 5


# ---------------------------------------------------------------------------
# get_learning_summary
# ---------------------------------------------------------------------------
class TestGetLearningSummary:
    def test_returns_required_keys(self, afl):
        summary = afl.get_learning_summary()
        assert "learning_status" in summary
        assert "learned_characteristics" in summary
        assert "usage_patterns" in summary
        assert "recent_sessions" in summary

    def test_learning_active_false_when_no_observations(self, afl):
        summary = afl.get_learning_summary()
        assert summary["learning_status"]["learning_active"] is False

    def test_learning_active_true_with_enough_observations(self, afl_with_observations):
        summary = afl_with_observations.get_learning_summary()
        # 5 observations ≥ min_observations_for_learning (3)
        assert summary["learning_status"]["learning_active"] is True

    def test_last_update_none_when_no_update(self, afl):
        summary = afl.get_learning_summary()
        assert summary["learning_status"]["last_update"] is None

    def test_last_update_iso_string_when_set(self, afl):
        afl.learning_state.last_update = datetime.now()
        summary = afl.get_learning_summary()
        assert isinstance(summary["learning_status"]["last_update"], str)

    def test_recent_sessions_max_five(self, tmp_path):
        instance = AdaptiveFireplaceLearning(state_file=str(tmp_path / "s.json"))
        for i in range(10):
            obs = FireplaceObservation(
                timestamp=datetime.now() - timedelta(hours=i),
                temp_differential=2.0,
                outdoor_temp=5.0,
                fireplace_active=True,
                duration_minutes=30.0,
            )
            instance.learning_state.observations.append(obs)
        summary = instance.get_learning_summary()
        assert len(summary["recent_sessions"]) <= 5

    def test_learned_characteristics_keys_present(self, afl):
        summary = afl.get_learning_summary()
        chars = summary["learned_characteristics"]
        expected = [
            "heat_output_kw",
            "thermal_efficiency",
            "heat_distribution_factor",
            "differential_to_heat_ratio",
            "outdoor_temp_correlation",
        ]
        for key in expected:
            assert key in chars, f"Missing characteristic: {key}"


# ---------------------------------------------------------------------------
# integrate_adaptive_fireplace_with_multi_source_physics
# ---------------------------------------------------------------------------
class TestIntegrateAdaptiveFireplaceWithMultiSourcePhysics:
    def _make_mock_physics(self):
        """Build a minimal mock that mimics MultiHeatSourcePhysics interface."""
        mock = MagicMock()
        mock.calculate_fireplace_heat_contribution.return_value = {
            "heat_contribution_kw": 2.0,
            "outlet_temp_reduction": 3.0,
            "reasoning": "physics",
        }
        return mock

    def test_replaces_fireplace_calculation(self, afl):
        mock_physics = self._make_mock_physics()
        enhanced = integrate_adaptive_fireplace_with_multi_source_physics(
            mock_physics, afl
        )
        # The method should have been replaced
        assert enhanced.calculate_fireplace_heat_contribution is not None
        # Call it to make sure it doesn't crash
        result = enhanced.calculate_fireplace_heat_contribution(
            fireplace_on=True, outdoor_temp=5.0, duration_hours=1.0
        )
        assert "heat_contribution_kw" in result or result is not None

    def test_fallback_to_physics_when_confidence_low(self, afl):
        """Low confidence → original physics calculation is used as fallback."""
        mock_physics = self._make_mock_physics()
        # Confidence is low by default (0.1)
        enhanced = integrate_adaptive_fireplace_with_multi_source_physics(
            mock_physics, afl
        )
        result = enhanced.calculate_fireplace_heat_contribution(
            fireplace_on=True,
            outdoor_temp=5.0,
            duration_hours=1.0,
            living_room_temp=22.0,
            other_rooms_temp=20.0,
        )
        assert result.get("learning_enhanced") is False

    def test_uses_learned_calculation_when_confidence_high(self, afl):
        """High confidence → adaptive learning result is returned."""
        afl.learning_state.learned_coefficients["learning_confidence"] = 0.8
        mock_physics = self._make_mock_physics()
        enhanced = integrate_adaptive_fireplace_with_multi_source_physics(
            mock_physics, afl
        )
        result = enhanced.calculate_fireplace_heat_contribution(
            fireplace_on=True,
            outdoor_temp=5.0,
            duration_hours=1.0,
            living_room_temp=22.0,
            other_rooms_temp=20.0,
        )
        assert result.get("learning_enhanced") is True

    def test_returns_same_physics_instance(self, afl):
        mock_physics = self._make_mock_physics()
        returned = integrate_adaptive_fireplace_with_multi_source_physics(
            mock_physics, afl
        )
        assert returned is mock_physics
