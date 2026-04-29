"""Extended tests for multi_heat_source_physics.py – integration helpers."""

import pytest
from unittest.mock import MagicMock

from src.multi_heat_source_physics import (
    MultiHeatSourcePhysics,
    _encode_heat_source,
    enhance_physics_features_with_heat_sources,
)


# ---------------------------------------------------------------------------
# _encode_heat_source
# ---------------------------------------------------------------------------
class TestEncodeHeatSource:
    def test_pv_encodes_as_1(self):
        assert _encode_heat_source("PV") == pytest.approx(1.0)

    def test_fireplace_encodes_as_2(self):
        assert _encode_heat_source("Fireplace") == pytest.approx(2.0)

    def test_electronics_encodes_as_3(self):
        assert _encode_heat_source("Electronics") == pytest.approx(3.0)

    def test_system_encodes_as_4(self):
        assert _encode_heat_source("System") == pytest.approx(4.0)

    def test_unknown_encodes_as_0(self):
        assert _encode_heat_source("Unknown") == pytest.approx(0.0)

    def test_empty_string_encodes_as_0(self):
        assert _encode_heat_source("") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# enhance_physics_features_with_heat_sources
# ---------------------------------------------------------------------------
class TestEnhancePhysicsFeaturesWithHeatSources:
    @pytest.fixture
    def physics(self):
        return MultiHeatSourcePhysics()

    def _base_features(self, pv=0.0, fireplace=0, tv=0):
        return {
            "pv_now": pv,
            "fireplace_on": fireplace,
            "tv_on": tv,
            "dhw_heating": 0,
            "dhw_disinfection": 0,
            "dhw_boost_heater": 0,
            "defrosting": 0,
            "indoor_temp_lag_30m": 20.0,
            "outdoor_temp": 5.0,
        }

    def test_returns_dict_with_original_keys(self, physics):
        base = self._base_features()
        result = enhance_physics_features_with_heat_sources(base, physics)
        assert "pv_now" in result
        assert "indoor_temp_lag_30m" in result

    def test_adds_required_enhanced_keys(self, physics):
        base = self._base_features()
        result = enhance_physics_features_with_heat_sources(base, physics)
        required = [
            "pv_heat_contribution_kw",
            "fireplace_heat_contribution_kw",
            "electronics_heat_contribution_kw",
            "total_auxiliary_heat_kw",
            "pv_outlet_reduction",
            "fireplace_outlet_reduction",
            "electronics_outlet_reduction",
            "total_outlet_reduction",
            "heat_source_diversity",
            "heat_source_diversity_factor",
            "system_capacity_reduction_percent",
            "system_auxiliary_heat_kw",
            "system_outlet_adjustment",
            "dominant_heat_source",
            "thermal_balance_score",
            "pv_thermal_effectiveness",
            "fireplace_thermal_buildup",
            "electronics_occupancy_factor",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_original_features_not_mutated(self, physics):
        base = self._base_features(pv=1000.0)
        original_pv = base["pv_now"]
        enhance_physics_features_with_heat_sources(base, physics)
        assert base["pv_now"] == original_pv

    def test_pv_contribution_nonzero_with_pv_power(self, physics):
        base = self._base_features(pv=2000.0)
        result = enhance_physics_features_with_heat_sources(base, physics)
        assert result["pv_heat_contribution_kw"] > 0

    def test_pv_contribution_zero_without_pv_power(self, physics):
        base = self._base_features(pv=0.0)
        result = enhance_physics_features_with_heat_sources(base, physics)
        assert result["pv_heat_contribution_kw"] == pytest.approx(0.0)

    def test_fireplace_contribution_nonzero_when_on(self, physics):
        base = self._base_features(fireplace=1)
        result = enhance_physics_features_with_heat_sources(base, physics)
        assert result["fireplace_heat_contribution_kw"] > 0

    def test_fireplace_contribution_zero_when_off(self, physics):
        base = self._base_features(fireplace=0)
        result = enhance_physics_features_with_heat_sources(base, physics)
        assert result["fireplace_heat_contribution_kw"] == pytest.approx(0.0)

    def test_electronics_contribution_nonzero_when_tv_on(self, physics):
        base = self._base_features(tv=1)
        result = enhance_physics_features_with_heat_sources(base, physics)
        assert result["electronics_heat_contribution_kw"] > 0

    def test_dominant_heat_source_is_numeric(self, physics):
        base = self._base_features(pv=3000.0)
        result = enhance_physics_features_with_heat_sources(base, physics)
        assert isinstance(result["dominant_heat_source"], float)

    def test_thermal_balance_score_is_0_or_1(self, physics):
        base = self._base_features()
        result = enhance_physics_features_with_heat_sources(base, physics)
        assert result["thermal_balance_score"] in (0.0, 1.0)

    def test_total_auxiliary_heat_nonnegative(self, physics):
        base = self._base_features(pv=1000.0, fireplace=1, tv=1)
        result = enhance_physics_features_with_heat_sources(base, physics)
        assert result["total_auxiliary_heat_kw"] >= 0

    def test_dhw_heating_flag_passed_through(self, physics):
        base = self._base_features()
        base["dhw_heating"] = 1
        result = enhance_physics_features_with_heat_sources(base, physics)
        # With DHW, system capacity reduction should be > 0
        assert result["system_capacity_reduction_percent"] >= 0

    def test_works_with_missing_optional_keys(self, physics):
        """Should not crash when some optional keys are absent from base."""
        base = {"pv_now": 500.0}  # Minimal feature dict
        result = enhance_physics_features_with_heat_sources(base, physics)
        assert "total_auxiliary_heat_kw" in result
