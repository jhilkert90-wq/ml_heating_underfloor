"""Tests for outlet search stabilisation (flat objective, demand, state)."""
from unittest.mock import Mock, patch

import pytest

from src import config
from src.outlet_stability import (
    has_heat_demand,
    pick_regularized,
    reference_band,
    reference_outlet,
    slew_limit,
)


class TestHelpers:
    def test_demand_when_room_below_target(self):
        assert has_heat_demand(22.0, 22.5, 0.0, 0.3, 0.1)

    def test_no_demand_when_room_above_target(self):
        assert not has_heat_demand(23.0, 22.5, 0.0, 0.3, 0.1)

    def test_demand_when_falling_towards_target(self):
        assert has_heat_demand(22.45, 22.5, -0.2, 0.3, 0.1)

    def test_falling_far_above_target_is_no_demand(self):
        assert not has_heat_demand(23.2, 22.5, -0.2, 0.3, 0.1)

    def test_reference_matches_base_curve_and_clamps(self):
        ref = reference_outlet(0.0, 22.5, 0.12, 0.83, 20.0, 35.0)
        assert ref == pytest.approx(22.5 * 0.95 / 0.83)
        assert reference_outlet(-30.0, 22.5, 0.12, 0.83, 20.0, 30.0) == 30.0
        assert reference_outlet(0.0, 22.5, 0.12, 0.0, 20.0, 35.0) is None

    def test_band_widens_with_demand(self):
        assert reference_band(False, 4.0, 3.0) == 4.0
        assert reference_band(True, 4.0, 3.0) == 7.0

    def test_pick_prefers_reference_on_flat_scores(self):
        cands = [
            {"outlet": o, "score": 0.03 * abs(o - 34.0)}
            for o in (24.0, 26.0, 28.0, 30.0, 34.0)
        ]
        best = pick_regularized(cands, 26.0, 0.01, 22.0, 30.0)
        assert best["outlet"] == 28.0

    def test_pick_ignores_candidates_outside_band(self):
        cands = [{"outlet": 34.0, "score": 0.0}, {"outlet": 28.0, "score": 0.5}]
        assert pick_regularized(cands, 26.0, 0.01, 22.0, 30.0)["outlet"] == 28.0

    def test_slew_relaxed_on_demand_or_target_jump(self):
        assert slew_limit(2.0, 1.0, demand=False, target_jump=False) == 1.0
        assert slew_limit(2.0, 1.0, demand=True, target_jump=False) == 2.0
        assert slew_limit(2.0, 1.0, demand=False, target_jump=True) == 2.0
        assert slew_limit(0.5, 1.0, demand=False, target_jump=False) == 0.5


class TestFlatSearchRegularisation:
    @pytest.fixture
    def wrapper(self):
        from src.model_wrapper import EnhancedModelWrapper

        return EnhancedModelWrapper()

    def _run(self, wrapper, slope):
        def trajectory(**kwargs):
            end = 22.5 + slope * (kwargs["outlet_temp"] - 30.0)
            return {"trajectory": [end] * 24}

        wrapper.thermal_model.predict_thermal_trajectory = Mock(
            side_effect=trajectory
        )
        wrapper.thermal_model.predict_equilibrium_temperature = Mock()
        wrapper.thermal_model.heat_loss_coefficient = 0.12
        wrapper.thermal_model.outlet_effectiveness = 0.83
        wrapper._log_multi_horizon_predictions = Mock()
        wrapper._current_features = {"inlet_temp": 25.3, "delta_t": 3.0}
        wrapper._get_forecast_conditions = Mock(
            return_value=(0.0, 0.0, [0.0] * 24, [0.0] * 24)
        )
        with patch.object(config, "TRAJECTORY_PREDICTION_ENABLED", False):
            return wrapper._calculate_required_outlet_temp(
                current_indoor=22.6,
                target_indoor=22.5,
                outdoor_temp=0.0,
                thermal_features={"pv_power": 0.0},
            )

    def test_flat_objective_is_pulled_towards_base_curve(self, wrapper):
        result = self._run(wrapper, slope=0.03)
        assert wrapper._search_sensitivity == pytest.approx(0.03, abs=1e-6)
        assert 25.0 < result < 29.5  # unregularised zero crossing is 30 °C

    def test_well_conditioned_search_is_unchanged(self, wrapper, monkeypatch):
        monkeypatch.setattr(config, "OUTLET_REFERENCE_ALWAYS_ACTIVE", False)
        result = self._run(wrapper, slope=0.5)
        assert result == pytest.approx(30.0, abs=0.1)

    def test_always_active_band_limits_well_conditioned_search(
        self, wrapper, monkeypatch
    ):
        monkeypatch.setattr(config, "OUTLET_REFERENCE_ALWAYS_ACTIVE", True)
        result = self._run(wrapper, slope=0.5)
        assert result < 29.5


class TestOperationalStatePersistence:
    def test_keys_missing_in_old_state_files_are_persisted(self, tmp_path):
        from src.unified_thermal_state import ThermalStateManager

        manager = ThermalStateManager(str(tmp_path / "state.json"))
        manager.state["operational_state"].pop(
            "setpoint_hold_cycles_remaining", None
        )
        manager.update_operational_state(
            setpoint_hold_cycles_remaining=3,
            last_climate_mode="heating",
            last_target_indoor_temp=22.5,
            last_predicted_indoor=22.4,
        )
        op = manager.get_operational_state()
        assert op["setpoint_hold_cycles_remaining"] == 3
        assert op["last_climate_mode"] == "heating"
        assert op["last_target_indoor_temp"] == 22.5
        # Learning must re-predict with the applied outlet.
        assert "last_predicted_indoor" not in op

    def test_cooling_manager_persists_hold_cycles(self, tmp_path):
        from src.unified_thermal_state_cooling import (
            CoolingThermalStateManager,
        )

        manager = CoolingThermalStateManager(str(tmp_path / "cool.json"))
        manager.update_operational_state(setpoint_hold_cycles_remaining=2)
        assert (
            manager.get_operational_state()["setpoint_hold_cycles_remaining"]
            == 2
        )


class TestControlQualityMetrics:
    def _metrics(self):
        from src.prediction_metrics import PredictionMetrics

        m = PredictionMetrics(state_manager=Mock())
        m._save_to_state = Mock()
        m.predictions.clear()
        return m

    def test_reports_outlet_steps_and_separates_target_changes(self):
        m = self._metrics()
        rows = [
            (26.0, 0.0, 0.0, 0.02),
            (30.0, 1.0, 0.0, 0.04),
            (26.0, -1.0, 0.5, 0.40),
        ]
        for outlet, deviation, target_change, error in rows:
            m.add_prediction(
                predicted=22.0,
                actual=22.0 + error,
                context={
                    "outlet_temp": outlet,
                    "outlet_applied_vs_computed_k": deviation,
                    "target_temp_change_k": target_change,
                },
            )
        quality = m._calculate_control_quality()
        assert quality["outlet_step_max"] == pytest.approx(4.0)
        assert quality["outlet_step_mean"] == pytest.approx(4.0)
        assert quality["applied_vs_computed_mean"] == pytest.approx(2 / 3)
        assert quality["mae_steady_target"] == pytest.approx(0.03)
        assert quality["mae_target_change"] == pytest.approx(0.40)

    def test_missing_context_yields_none(self):
        quality = self._metrics()._calculate_control_quality()
        assert quality["outlet_step_mean"] is None
        assert quality["mae_target_change"] is None
