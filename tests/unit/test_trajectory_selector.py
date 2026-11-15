"""Tests for the joint trajectory outlet selector (heating_correction_mode: trajectory)."""
from unittest.mock import Mock

import pytest

from src import config
from src.trajectory_selector import comfort_cost, select_trajectory_outlet


def _flat(value):
    return lambda outlet: [value] * 24


class TestComfortCost:
    def test_only_violations_outside_band_cost(self):
        traj = [22.5, 22.6, 22.8, 22.2]
        # 0.2 K above the band at 22.8, 0.2 K below the band at 22.2
        assert comfort_cost(traj, 22.5, 0.1, 0.5) == pytest.approx(
            (0.2 ** 2 + 0.2 ** 2) * 0.5
        )


class TestSelectTrajectoryOutlet:
    def test_picks_outlet_with_lowest_cost(self):
        result = select_trajectory_outlet(
            lambda outlet: [21.0 + 0.1 * (outlet - 24.0)] * 24,
            lower=24.0, upper=40.0, baseline=30.0,
            current_indoor=22.0, target=22.5, step_hours=1 / 6,
        )
        # 22.4 is the lower band edge -> 38.0 °C is the cheapest in-band outlet
        assert result["outlet"] == pytest.approx(38.0)
        assert result["cost"] == pytest.approx(0.0)

    def test_above_band_ties_choose_lowest_outlet(self):
        result = select_trajectory_outlet(
            _flat(23.2), lower=24.0, upper=30.0, baseline=27.0,
            current_indoor=23.23, target=22.5, step_hours=1 / 6,
        )
        assert result["outlet"] == 24.0

    def test_in_band_ties_choose_nearest_baseline(self):
        result = select_trajectory_outlet(
            lambda outlet: [22.5 + 0.3 * (outlet - 26.5)] * 24,
            lower=24.0, upper=30.0, baseline=27.0,
            current_indoor=22.5, target=22.5, step_hours=1 / 6,
        )
        assert result["outlet"] == pytest.approx(26.8)

    def test_bounds_and_baseline_are_candidates(self):
        seen = []

        def simulate(outlet):
            seen.append(outlet)
            return [22.5] * 24

        select_trajectory_outlet(
            simulate, lower=24.03, upper=24.37, baseline=24.25,
            current_indoor=22.5, target=22.5, step_hours=1 / 6,
        )
        assert {24.03, 24.37, 24.25, 24.1, 24.2, 24.3} <= set(seen)

    def test_failed_candidates_are_skipped(self):
        def simulate(outlet):
            if outlet < 26.0:
                raise RuntimeError("boom")
            return [] if outlet > 27.0 else [22.5] * 24

        result = select_trajectory_outlet(
            simulate, lower=24.0, upper=28.0, baseline=24.0,
            current_indoor=22.5, target=22.5, step_hours=1 / 6,
        )
        assert 26.0 <= result["outlet"] <= 27.0

    def test_no_valid_candidate_returns_none(self):
        assert select_trajectory_outlet(
            lambda outlet: [], lower=24.0, upper=25.0, baseline=24.5,
            current_indoor=22.5, target=22.5, step_hours=1 / 6,
        ) is None


class TestWrapperIntegration:
    @pytest.fixture
    def wrapper(self):
        from src.model_wrapper import EnhancedModelWrapper

        return EnhancedModelWrapper()

    def _run(self, wrapper, monkeypatch, response, *, mode, current=22.5,
             target=22.5, inlet=25.0, baseline=27.0, bounds=(24.0, 35.0),
             climate="heating"):
        monkeypatch.setattr(config, "HEATING_CORRECTION_MODE", mode)
        monkeypatch.setattr(config, "get_outlet_bounds", lambda m: bounds)
        monkeypatch.setattr(config, "TRAJECTORY_PREDICTION_ENABLED", False)
        monkeypatch.setattr(config, "TRAJECTORY_SEARCH_ERROR_MODE", "horizon_end")
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 4)
        monkeypatch.setattr(config, "CYCLE_INTERVAL_MINUTES", 10)
        monkeypatch.setattr(config, "OUTLET_REFERENCE_ENABLED", True)
        monkeypatch.setattr(config, "OUTLET_REFERENCE_ALWAYS_ACTIVE", True)
        monkeypatch.setattr(config, "OUTLET_REFERENCE_BAND_K", 2.0)
        monkeypatch.setattr(
            "src.model_wrapper.reference_outlet", lambda *args: 26.5
        )
        wrapper._climate_mode = climate
        wrapper._trajectory_baseline_outlet = baseline
        wrapper._current_features = {"inlet_temp": inlet, "delta_t": 2.68}
        wrapper._log_multi_horizon_predictions = Mock()
        wrapper._get_forecast_conditions = Mock(
            return_value=(0.0, 0.0, [0.0] * 4, [0.0] * 4)
        )
        wrapper.thermal_model.predict_thermal_trajectory = Mock(
            side_effect=lambda **kw: {
                "trajectory": [response(kw["outlet_temp"])] * 24
            }
        )
        wrapper.thermal_model.predict_equilibrium_temperature = Mock()
        return wrapper._calculate_required_outlet_temp(
            current_indoor=current, target_indoor=target, outdoor_temp=0.0,
            thermal_features={"pv_power": 0.0},
        )

    @staticmethod
    def _linear(outlet):
        return 22.5 + 0.3 * (outlet - 26.5)

    def test_trajectory_mode_returns_selector_outlet(
        self, wrapper, monkeypatch, caplog
    ):
        with caplog.at_level("INFO"):
            result = self._run(
                wrapper, monkeypatch, self._linear, mode="trajectory"
            )
        assert result == pytest.approx(26.8)
        assert "using trajectory" in caplog.text

    def test_other_modes_keep_legacy_result_and_log_shadow(
        self, wrapper, monkeypatch, caplog
    ):
        with caplog.at_level("INFO"):
            result = self._run(
                wrapper, monkeypatch, self._linear, mode="physics"
            )
        assert result == pytest.approx(26.5)
        assert "Trajectory selector (shadow)" in caplog.text
        assert "trajectory 26.8°C" in caplog.text
        assert "using legacy" in caplog.text

    def test_upper_candidate_bound_is_inlet_plus_five(
        self, wrapper, monkeypatch
    ):
        from src import trajectory_selector

        selector = Mock(wraps=trajectory_selector.select_trajectory_outlet)
        monkeypatch.setattr(
            "src.model_wrapper.select_trajectory_outlet", selector
        )
        result = self._run(
            wrapper, monkeypatch, lambda outlet: 21.0 + 0.1 * (outlet - 24.0),
            mode="trajectory", current=21.0,
        )
        assert selector.call_args.kwargs["upper"] == pytest.approx(30.0)
        assert result == pytest.approx(30.0)

    def test_without_baseline_ties_fall_back_to_legacy_outlet(
        self, wrapper, monkeypatch
    ):
        result = self._run(
            wrapper, monkeypatch, self._linear, mode="trajectory",
            baseline=None,
        )
        assert result == pytest.approx(26.5)

    def test_selector_failure_keeps_legacy_result(
        self, wrapper, monkeypatch
    ):
        monkeypatch.setattr(
            "src.model_wrapper.select_trajectory_outlet",
            Mock(side_effect=RuntimeError("boom")),
        )
        result = self._run(
            wrapper, monkeypatch, self._linear, mode="trajectory"
        )
        assert result == pytest.approx(26.5)

    def test_cooling_never_runs_selector(self, wrapper, monkeypatch):
        selector = Mock()
        monkeypatch.setattr(
            "src.model_wrapper.select_trajectory_outlet", selector
        )
        self._run(
            wrapper, monkeypatch, lambda outlet: 24.0 + 0.1 * (outlet - 20.0),
            mode="trajectory", current=24.5, target=24.0, inlet=22.0,
            bounds=(18.0, 24.0), climate="cooling",
        )
        selector.assert_not_called()


class TestBaselinePlumbing:
    def test_simplified_prediction_passes_baseline_to_wrapper(
        self, monkeypatch
    ):
        import pandas as pd

        from src import model_wrapper

        wrapper = Mock()
        wrapper.calculate_optimal_outlet_temp.return_value = (26.0, {})
        monkeypatch.setattr(
            model_wrapper, "get_enhanced_model_wrapper", lambda: wrapper
        )
        monkeypatch.setattr(
            model_wrapper, "_calculate_thermal_trust_metrics",
            lambda *args: {},
        )
        model_wrapper.simplified_outlet_prediction(
            pd.DataFrame([{"indoor_temp_lag_30m": 22.0}]), 22.0, 22.5,
            baseline_outlet=27.3,
        )
        assert wrapper._trajectory_baseline_outlet == 27.3

    @pytest.mark.parametrize(
        "state,actual,expected",
        [({"last_final_temp": 26.4}, 25.0, 26.4), ({}, 25.0, 25.0)],
    )
    def test_step_prediction_passes_last_published_outlet(
        self, monkeypatch, state, actual, expected
    ):
        from src import cycle_routes
        from src.cycle_context import CycleContext

        prediction = Mock(return_value=(26.0, 0.9, {}))
        monkeypatch.setattr(
            cycle_routes, "simplified_outlet_prediction", prediction
        )
        ctx = CycleContext()
        ctx.state = state
        ctx.actual_outlet_temp = actual
        ctx.features = None
        ctx.prediction_indoor_temp = 22.0
        ctx.target_indoor_temp = 22.5
        cycle_routes.step_prediction(ctx)
        assert prediction.call_args.kwargs["baseline_outlet"] == expected
