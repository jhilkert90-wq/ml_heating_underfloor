"""
Cooling Mode Tests

Tests for the cooling mode implementation that mirrors heating control
with inverted outlet temperature bounds and mode detection via
HEATING_STATUS_ENTITY_ID.

Key cooling constraints:
- Outlet < inlet (cold water through slab)
- Minimum outlet = 18°C (HP shutdown limit)
- Minimum inlet-outlet delta = 2K
- Safety margin above shutdown to prevent short-cycling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src import config
from src.heating_controller import HeatingSystemStateChecker


# ── Config helper tests ──────────────────────────────────────────────


class TestGetClimateMode:
    """Tests for config.get_climate_mode()."""

    def test_heat_mode(self):
        assert config.get_climate_mode("heat") == "heating"

    def test_auto_mode(self):
        assert config.get_climate_mode("auto") == "heating"

    def test_cool_mode(self):
        assert config.get_climate_mode("cool") == "cooling"

    def test_off_mode(self):
        assert config.get_climate_mode("off") == "off"

    def test_none_returns_off(self):
        assert config.get_climate_mode(None) == "off"

    def test_unknown_returns_off(self):
        assert config.get_climate_mode("dry") == "off"

    def test_case_insensitive(self):
        assert config.get_climate_mode("Cool") == "cooling"
        assert config.get_climate_mode("HEAT") == "heating"
        assert config.get_climate_mode("AUTO") == "heating"


class TestGetOutletBounds:
    """Tests for config.get_outlet_bounds()."""

    def test_heating_bounds(self):
        low, high = config.get_outlet_bounds("heating")
        assert low == config.CLAMP_MIN_ABS
        assert high == config.CLAMP_MAX_ABS

    def test_cooling_bounds_use_full_clamp_range(self):
        """Cooling bounds must use the full CLAMP_MIN_ABS–CLAMP_MAX_ABS range.

        The post-search cooling cycle gate handles HP-cannot-run scenarios;
        the search itself should not pre-constrain with the shutdown margin.
        """
        low, high = config.get_outlet_bounds("cooling")
        assert low == config.COOLING_CLAMP_MIN_ABS
        assert high == config.COOLING_CLAMP_MAX_ABS

    def test_cooling_min_equals_shutdown_limit(self):
        """The effective minimum equals the HP hard shutdown limit."""
        low, _ = config.get_outlet_bounds("cooling")
        assert low == config.COOLING_CLAMP_MIN_ABS

    def test_unknown_mode_returns_heating_bounds(self):
        low, high = config.get_outlet_bounds("off")
        assert low == config.CLAMP_MIN_ABS
        assert high == config.CLAMP_MAX_ABS


class TestGetFallbackOutlet:
    """Tests for config.get_fallback_outlet()."""

    def test_heating_fallback(self):
        assert config.get_fallback_outlet("heating") == 35.0

    def test_cooling_fallback_is_midrange(self):
        expected = (
            config.COOLING_CLAMP_MIN_ABS + config.COOLING_CLAMP_MAX_ABS
        ) / 2.0
        assert config.get_fallback_outlet("cooling") == expected

    def test_cooling_fallback_within_bounds(self):
        fallback = config.get_fallback_outlet("cooling")
        assert fallback >= config.COOLING_CLAMP_MIN_ABS
        assert fallback <= config.COOLING_CLAMP_MAX_ABS


# ── HeatingSystemStateChecker tests ──────────────────────────────────


class TestHeatingSystemStateCheckerCooling:
    """Tests that HeatingSystemStateChecker recognises cooling mode."""

    @pytest.fixture
    def state_checker(self):
        return HeatingSystemStateChecker()

    @pytest.fixture
    def mock_ha_client(self):
        return Mock()

    @pytest.fixture(autouse=True)
    def force_active_mode(self):
        with patch("src.heating_controller.config.SHADOW_MODE", False):
            yield

    def test_cool_mode_is_active(self, state_checker, mock_ha_client):
        """Cooling mode should NOT skip the cycle."""
        mock_ha_client.get_state.return_value = "cool"
        assert (
            state_checker.check_heating_active(mock_ha_client, {}) is True
        )

    def test_get_climate_mode_cooling(self, state_checker, mock_ha_client):
        mock_ha_client.get_state.return_value = "cool"
        mode = state_checker.get_climate_mode(mock_ha_client, {})
        assert mode == "cooling"

    def test_get_climate_mode_heating(self, state_checker, mock_ha_client):
        mock_ha_client.get_state.return_value = "heat"
        mode = state_checker.get_climate_mode(mock_ha_client, {})
        assert mode == "heating"

    def test_get_climate_mode_off(self, state_checker, mock_ha_client):
        mock_ha_client.get_state.return_value = "off"
        mode = state_checker.get_climate_mode(mock_ha_client, {})
        assert mode == "off"


# ── Model wrapper cooling bounds tests ───────────────────────────────


class TestModelWrapperCoolingMode:
    """Tests that EnhancedModelWrapper uses correct bounds in cooling mode."""

    @pytest.fixture
    def wrapper(self):
        from src.model_wrapper import EnhancedModelWrapper

        w = EnhancedModelWrapper()
        return w

    def test_default_mode_is_heating(self, wrapper):
        assert wrapper.climate_mode == "heating"

    def test_set_climate_mode_cooling(self, wrapper):
        wrapper.set_climate_mode("cooling")
        assert wrapper.climate_mode == "cooling"

    def test_set_invalid_mode_defaults_to_heating(self, wrapper):
        wrapper.set_climate_mode("invalid")
        assert wrapper.climate_mode == "heating"

    def test_cooling_binary_search_uses_cooling_bounds(self, wrapper):
        """
        In cooling mode, the binary search should use
        COOLING_CLAMP_MIN_ABS + margin .. min(COOLING_CLAMP_MAX_ABS,
        indoor - delta).
        """
        wrapper.set_climate_mode("cooling")

        # Mock the thermal model to return a simple prediction
        mock_trajectory = {
            "trajectory": [22.0],  # predicted indoor temp
            "timestamps": ["2026-04-07T12:00:00"],
        }
        wrapper.thermal_model.predict_thermal_trajectory = Mock(
            return_value=mock_trajectory
        )
        wrapper.thermal_model.predict_equilibrium_temperature = Mock(
            return_value=22.0
        )
        wrapper._current_features = {
            "inlet_temp": 24.0,
            "delta_t": 3.0,
        }

        # Call the binary search
        result = wrapper._calculate_required_outlet_temp(
            current_indoor=23.5,
            target_indoor=22.0,
            outdoor_temp=30.0,
            thermal_features={
                "pv_power": 0.0,
                "fireplace_on": 0.0,
                "tv_on": 0.0,
            },
        )

        # Result must be within cooling bounds (full range, no margin)
        assert result >= config.COOLING_CLAMP_MIN_ABS, (
            f"Cooling outlet {result} below absolute minimum "
            f"{config.COOLING_CLAMP_MIN_ABS}"
        )
        assert result <= config.COOLING_CLAMP_MAX_ABS, (
            f"Cooling outlet {result} above cooling max "
            f"{config.COOLING_CLAMP_MAX_ABS}"
        )

    def test_cooling_no_viable_range_returns_safe_min(self, wrapper):
        """
        When the room is already cool (indoor - delta < effective_min),
        there is no viable cooling range.  Should return outlet_min
        (the warmest valid cooling outlet), never below effective min.
        """
        wrapper.set_climate_mode("cooling")
        wrapper._current_features = {
            "inlet_temp": 20.0,
            "delta_t": 3.0,
        }

        result = wrapper._calculate_required_outlet_temp(
            current_indoor=20.5,  # Nearly at the HP delta limit
            target_indoor=20.0,
            outdoor_temp=28.0,
            thermal_features={
                "pv_power": 0.0,
                "fireplace_on": 0.0,
                "tv_on": 0.0,
            },
        )

        # Must never go below the absolute cooling minimum.
        assert result >= config.COOLING_CLAMP_MIN_ABS, (
            f"Cooling outlet {result} below absolute minimum "
            f"{config.COOLING_CLAMP_MIN_ABS}"
        )

    def test_heating_mode_uses_standard_bounds(self, wrapper):
        """In heating mode the binary search uses CLAMP_MIN/MAX_ABS."""
        wrapper.set_climate_mode("heating")

        mock_trajectory = {
            "trajectory": [21.0],
            "timestamps": ["2026-04-07T12:00:00"],
        }
        wrapper.thermal_model.predict_thermal_trajectory = Mock(
            return_value=mock_trajectory
        )
        wrapper.thermal_model.predict_equilibrium_temperature = Mock(
            return_value=21.0
        )
        wrapper._current_features = {
            "inlet_temp": 28.0,
            "delta_t": 3.0,
        }

        result = wrapper._calculate_required_outlet_temp(
            current_indoor=20.0,
            target_indoor=21.0,
            outdoor_temp=5.0,
            thermal_features={
                "pv_power": 0.0,
                "fireplace_on": 0.0,
                "tv_on": 0.0,
            },
        )

        assert result >= config.CLAMP_MIN_ABS
        assert result <= config.CLAMP_MAX_ABS

    def test_cooling_outlet_not_tightened_by_inlet_in_search(self, wrapper):
        """Binary search does NOT tighten outlet_max by inlet.

        The post-search RUNNING/RECOVERY gate handles HP-cannot-run
        scenarios when the outlet converges near the inlet.
        """
        wrapper.set_climate_mode("cooling")
        wrapper._current_features = {
            "inlet_temp": 22.0,
            "delta_t": -2.3,
        }
        mock_trajectory = {"trajectory": [22.0], "timestamps": [""]}
        wrapper.thermal_model.predict_thermal_trajectory = Mock(
            return_value=mock_trajectory
        )
        wrapper.thermal_model.predict_equilibrium_temperature = Mock(
            return_value=22.0
        )
        result = wrapper._calculate_required_outlet_temp(
            current_indoor=24.0,
            target_indoor=22.0,
            outdoor_temp=30.0,
            thermal_features={"pv_power": 0.0, "fireplace_on": 0.0, "tv_on": 0.0},
        )
        # Result may be above inlet − delta; the gate will handle it.
        assert result >= config.COOLING_CLAMP_MIN_ABS
        assert result <= config.COOLING_CLAMP_MAX_ABS


# ── Thermal constants tests ──────────────────────────────────────────


class TestThermalConstantsCooling:
    """Verify cooling constants exist in PhysicsConstants."""

    def test_cooling_constants_exist(self):
        from src.thermal_constants import PhysicsConstants

        assert hasattr(PhysicsConstants, "MIN_COOLING_OUTLET_TEMP")
        assert hasattr(PhysicsConstants, "MAX_COOLING_OUTLET_TEMP")
        assert hasattr(PhysicsConstants, "MIN_COOLING_DELTA_K")

    def test_cooling_outlet_range_valid(self):
        from src.thermal_constants import PhysicsConstants

        assert (
            PhysicsConstants.MIN_COOLING_OUTLET_TEMP
            < PhysicsConstants.MAX_COOLING_OUTLET_TEMP
        )

    def test_cooling_min_below_heating_min(self):
        from src.thermal_constants import PhysicsConstants

        assert (
            PhysicsConstants.MIN_COOLING_OUTLET_TEMP
            < PhysicsConstants.MIN_OUTLET_TEMP
        )


# ── State isolation tests ────────────────────────────────────────────


class TestCoolingStateIsolation:
    """Verify that cooling-mode reads/writes use the cooling state manager
    and not the heating one — ensuring the two modes never contaminate each
    other's JSON files."""

    @pytest.fixture
    def wrapper(self, tmp_path):
        """Hermetic wrapper fixture: injects temp-file state managers so the
        tests never touch real filesystem locations or conflict with other
        singleton-using tests."""
        import src.unified_thermal_state as uts
        import src.unified_thermal_state_cooling as utsc
        import src.model_wrapper as mw
        from src.unified_thermal_state import ThermalStateManager
        from src.unified_thermal_state_cooling import CoolingThermalStateManager
        from src.model_wrapper import EnhancedModelWrapper

        heating_file = str(tmp_path / "heating_state.json")
        cooling_file = str(tmp_path / "cooling_state.json")

        # Reset the wrapper singleton first so any lingering instance from a
        # previous test (which may hold stale manager references) is discarded
        # before we inject the new temp-file managers.
        mw._enhanced_model_wrapper_instance = None
        uts._thermal_state_manager = None
        utsc._cooling_state_manager = None

        # Inject fresh temp-file managers so no real paths are touched.
        uts._thermal_state_manager = ThermalStateManager(state_file=heating_file)
        utsc._cooling_state_manager = CoolingThermalStateManager(
            state_file=cooling_file
        )

        w = EnhancedModelWrapper()
        yield w

        # Tear down
        mw._enhanced_model_wrapper_instance = None
        uts._thermal_state_manager = None
        utsc._cooling_state_manager = None

    def test_separate_state_manager_instances(self, wrapper):
        """Heating and cooling managers must be different objects."""
        assert wrapper._heating_state_manager is not wrapper._cooling_state_manager

    def test_separate_thermal_model_instances(self, wrapper):
        """Heating and cooling thermal models must be different objects."""
        assert wrapper._heating_thermal_model is not wrapper._cooling_thermal_model

    def test_heating_mode_uses_heating_manager(self, wrapper):
        """In heating mode (default) the active state manager is the heating one."""
        wrapper.set_climate_mode("heating")
        assert wrapper.state_manager is wrapper._heating_state_manager
        assert wrapper.thermal_model is wrapper._heating_thermal_model
        assert wrapper.prediction_metrics is wrapper._heating_prediction_metrics

    def test_cooling_mode_uses_cooling_manager(self, wrapper):
        """After switching to cooling mode the active state manager is the cooling one."""
        wrapper.set_climate_mode("cooling")
        assert wrapper.state_manager is wrapper._cooling_state_manager
        assert wrapper.thermal_model is wrapper._cooling_thermal_model
        assert wrapper.prediction_metrics is wrapper._cooling_prediction_metrics

    def test_mode_switch_back_to_heating(self, wrapper):
        """Switching back to heating restores the heating pair."""
        wrapper.set_climate_mode("cooling")
        wrapper.set_climate_mode("heating")
        assert wrapper.state_manager is wrapper._heating_state_manager
        assert wrapper.thermal_model is wrapper._heating_thermal_model

    def test_cooling_state_file_differs_from_heating(self, wrapper):
        """The state files used by each manager must be different paths."""
        heating_file = getattr(wrapper._heating_state_manager, "state_file", "")
        cooling_file = getattr(wrapper._cooling_state_manager, "state_file", "")
        assert heating_file != cooling_file

    def test_cooling_thermal_model_injected_with_cooling_manager(self, wrapper):
        """The cooling ThermalEquilibriumModel must have the cooling manager injected."""
        assert (
            wrapper._cooling_thermal_model._state_manager
            is wrapper._cooling_state_manager
        )

    def test_heating_thermal_model_injected_with_heating_manager(self, wrapper):
        """The heating ThermalEquilibriumModel must have the heating manager injected."""
        assert (
            wrapper._heating_thermal_model._state_manager
            is wrapper._heating_state_manager
        )

    def test_update_learning_state_in_cooling_writes_to_cooling_manager(self, wrapper):
        """update_learning_state calls during cooling mode go to the cooling manager."""
        wrapper.set_climate_mode("cooling")

        heating_calls_before = 0
        cooling_calls_before = 0

        with patch.object(
            wrapper._heating_state_manager, "update_learning_state"
        ) as mock_heating_update, patch.object(
            wrapper._cooling_state_manager, "update_learning_state"
        ) as mock_cooling_update:
            wrapper.state_manager.update_learning_state(cycle_count=1)

            assert mock_heating_update.call_count == heating_calls_before
            assert mock_cooling_update.call_count == cooling_calls_before + 1

    def test_add_prediction_record_in_cooling_writes_to_cooling_manager(self, wrapper):
        """add_prediction_record calls during cooling go to the cooling manager."""
        wrapper.set_climate_mode("cooling")

        record = {
            "timestamp": "2026-05-01T12:00:00",
            "predicted": 20.5,
            "actual": 20.2,
            "error": -0.3,
        }
        with patch.object(
            wrapper._heating_state_manager, "add_prediction_record"
        ) as mock_heating_add, patch.object(
            wrapper._cooling_state_manager, "add_prediction_record"
        ) as mock_cooling_add:
            wrapper.state_manager.add_prediction_record(record)

            mock_heating_add.assert_not_called()
            mock_cooling_add.assert_called_once_with(record)

    def test_cycle_count_reloaded_on_mode_switch(self, wrapper):
        """cycle_count is read from the newly active manager on every mode switch."""
        # Manually set a distinctive cycle count in each mock
        wrapper._heating_state_manager.state["learning_state"]["cycle_count"] = 42
        wrapper._cooling_state_manager.state["learning_state"]["cycle_count"] = 7

        wrapper.set_climate_mode("cooling")
        assert wrapper.cycle_count == 7

        wrapper.set_climate_mode("heating")
        assert wrapper.cycle_count == 42



# ── Inlet guard tests (HP idle when outlet > inlet − MIN_COOLING_DELTA_K) ──


class TestCoolingInletGuard:
    """
    Verify that calculate_optimal_outlet_temp() applies the inlet guard in
    cooling mode: when the binary-search result would be within
    MIN_COOLING_DELTA_K of the inlet temperature, it is clamped to inlet_temp
    so the NIBE compressor stays idle (circulator only) instead of receiving
    an un-achievable setpoint.

    Scenario from product context:
        inlet=22°C, MIN_COOLING_DELTA_K=2°C, required outlet=21.5°C
        → gap (0.5°C) < delta (2°C) → clamp to inlet (22°C)
    """

    @pytest.fixture
    def wrapper(self):
        from src.model_wrapper import EnhancedModelWrapper
        w = EnhancedModelWrapper()
        return w

    def _run_optimal(self, wrapper, features, current_indoor, target_indoor,
                     outdoor_temp=30.0, price_data=None):
        """Helper: calls calculate_optimal_outlet_temp and returns outlet."""
        full_features = dict(features)
        full_features.setdefault("indoor_temp_lag_30m", current_indoor)
        full_features.setdefault("target_temp", target_indoor)
        full_features.setdefault("outdoor_temp", outdoor_temp)
        result, _ = wrapper.calculate_optimal_outlet_temp(features=full_features)
        return result

    def _make_features(self, inlet, current_indoor, target_indoor,
                       outdoor_temp=30.0, extra=None):
        """Build a minimal features dict for calculate_optimal_outlet_temp."""
        f = {
            "indoor_temp_lag_30m": current_indoor,
            "target_temp": target_indoor,
            "outdoor_temp": outdoor_temp,
            "inlet_temp": inlet,
            "delta_t": 2.5,
            "pv_now": 0.0,
            "pv_forecast_electrical_1h": 0.0,
            "pv_forecast_1h": 0.0,
            "indoor_temp_delta_60m": 0.0,
        }
        if extra:
            f.update(extra)
        return f

    def test_outlet_clamped_to_inlet_when_gap_too_small(self, wrapper):
        """
        inlet=22, outlet computed to 21.5 → gap=0.5 < delta=2
        → outlet must be clamped to inlet=22 (HP idle).
        """
        wrapper.set_climate_mode("cooling")

        INLET = 22.0
        with patch.object(
            wrapper,
            "_calculate_required_outlet_temp",
            return_value=21.5,
        ):
            wrapper._current_features = {
                "inlet_temp": INLET,
                "delta_t": 2.5,
                "pv_now": 0.0,
                "pv_forecast_electrical_1h": 0.0,
                "pv_forecast_1h": 0.0,
                "indoor_temp_delta_60m": 0.0,
            }
            features = self._make_features(INLET, 23.0, 22.0)
            result, _ = wrapper.calculate_optimal_outlet_temp(
                features=features,
            )

        assert result == INLET, (
            f"Expected outlet clamped to inlet {INLET}°C, got {result}°C"
        )

    def test_outlet_not_clamped_when_gap_sufficient(self, wrapper):
        """
        inlet=22, outlet computed to 19.5 → gap=2.5 ≥ delta=2
        → outlet passes through unchanged.
        """
        wrapper.set_climate_mode("cooling")

        INLET = 22.0
        OUTLET = 19.5  # gap = 2.5 ≥ MIN_COOLING_DELTA_K=2 → no clamp
        with patch.object(
            wrapper,
            "_calculate_required_outlet_temp",
            return_value=OUTLET,
        ):
            wrapper._current_features = {
                "inlet_temp": INLET,
                "delta_t": 2.5,
                "pv_now": 0.0,
                "pv_forecast_electrical_1h": 0.0,
                "pv_forecast_1h": 0.0,
                "indoor_temp_delta_60m": 0.0,
            }
            features = self._make_features(INLET, 23.0, 22.0)
            result, _ = wrapper.calculate_optimal_outlet_temp(
                features=features,
            )

        assert result == OUTLET, (
            f"Expected outlet unchanged at {OUTLET}°C, got {result}°C"
        )

    def test_outlet_clamped_at_exact_delta_boundary(self, wrapper):
        """
        inlet=22, outlet=20 → gap=exactly 2=MIN_COOLING_DELTA_K
        → boundary: outlet passes (just enough delta for HP to run).
        """
        wrapper.set_climate_mode("cooling")

        INLET = 22.0
        OUTLET = 20.0  # gap = exactly 2 = MIN_COOLING_DELTA_K → no clamp
        with patch.object(
            wrapper,
            "_calculate_required_outlet_temp",
            return_value=OUTLET,
        ):
            wrapper._current_features = {
                "inlet_temp": INLET,
                "delta_t": 2.5,
                "pv_now": 0.0,
                "pv_forecast_electrical_1h": 0.0,
                "pv_forecast_1h": 0.0,
                "indoor_temp_delta_60m": 0.0,
            }
            features = self._make_features(INLET, 23.0, 22.0)
            result, _ = wrapper.calculate_optimal_outlet_temp(
                features=features,
            )

        # gap == delta → threshold is NOT exceeded → no clamp
        assert result == OUTLET, (
            f"Expected outlet unchanged at {OUTLET}°C, got {result}°C"
        )

    @patch("src.model_wrapper._is_heat_pump_active", return_value=False)
    def test_running_gate_stays_running_when_hp_is_idle(
        self, _mock_hp_active, wrapper
    ):
        wrapper.set_climate_mode("cooling")
        wrapper._cooling_cycle_state = "running"

        inlet = 22.0
        with patch.object(
            wrapper,
            "_calculate_required_outlet_temp",
            return_value=21.5,
        ):
            features = self._make_features(
                inlet,
                23.0,
                22.0,
                extra={"thermal_power_kw": 0.0, "delta_t": -0.1},
            )
            result, _ = wrapper.calculate_optimal_outlet_temp(features=features)

        assert result == inlet
        assert wrapper._cooling_cycle_state == "running"

    @patch("src.model_wrapper._is_heat_pump_active", return_value=True)
    def test_running_gate_enters_recovery_when_hp_was_running(
        self, _mock_hp_active, wrapper
    ):
        wrapper.set_climate_mode("cooling")
        wrapper._cooling_cycle_state = "running"

        inlet = 22.0
        with patch.object(
            wrapper,
            "_calculate_required_outlet_temp",
            return_value=21.5,
        ):
            features = self._make_features(
                inlet,
                23.0,
                22.0,
                extra={"thermal_power_kw": -1.2, "delta_t": -2.5},
            )
            result, _ = wrapper.calculate_optimal_outlet_temp(features=features)

        assert result == inlet
        assert wrapper._cooling_cycle_state == "recovery"

    def test_inlet_guard_not_applied_in_heating_mode(self, wrapper):
        """
        In heating mode the inlet guard must NOT be applied even when outlet
        < inlet (e.g. a low setpoint during a mild day is intentional in
        heating mode).
        """
        wrapper.set_climate_mode("heating")

        INLET = 30.0
        OUTLET = 28.0  # below inlet but valid in heating mode
        with patch.object(
            wrapper,
            "_calculate_required_outlet_temp",
            return_value=OUTLET,
        ):
            wrapper._current_features = {
                "inlet_temp": INLET,
                "delta_t": 5.0,
                "pv_now": 0.0,
                "pv_forecast_electrical_1h": 0.0,
                "pv_forecast_1h": 0.0,
                "indoor_temp_delta_60m": 0.0,
            }
            features = {
                "indoor_temp_lag_30m": 20.0,
                "target_temp": 21.0,
                "outdoor_temp": 5.0,
                "inlet_temp": INLET,
                "delta_t": 5.0,
                "pv_now": 0.0,
                "pv_forecast_electrical_1h": 0.0,
                "pv_forecast_1h": 0.0,
                "indoor_temp_delta_60m": 0.0,
            }
            result, _ = wrapper.calculate_optimal_outlet_temp(
                features=features,
            )

        # Inlet guard must not change the value in heating mode
        assert result == OUTLET, (
            f"Inlet guard must not apply in heating mode; "
            f"expected {OUTLET}°C, got {result}°C"
        )

    def test_inlet_guard_skipped_when_inlet_unavailable(self, wrapper):
        """
        When inlet_temp is not in features, the guard should not crash and
        the outlet passes through unchanged.
        """
        wrapper.set_climate_mode("cooling")

        OUTLET = 21.5
        with patch.object(
            wrapper,
            "_calculate_required_outlet_temp",
            return_value=OUTLET,
        ):
            wrapper._current_features = {
                # no inlet_temp key
                "delta_t": 2.5,
                "pv_now": 0.0,
                "pv_forecast_electrical_1h": 0.0,
                "pv_forecast_1h": 0.0,
                "indoor_temp_delta_60m": 0.0,
            }
            features = {
                "indoor_temp_lag_30m": 23.0,
                "target_temp": 22.0,
                "outdoor_temp": 30.0,
                "delta_t": 2.5,
                "pv_now": 0.0,
                "pv_forecast_electrical_1h": 0.0,
                "pv_forecast_1h": 0.0,
                "indoor_temp_delta_60m": 0.0,
            }
            result, _ = wrapper.calculate_optimal_outlet_temp(
                features=features,
            )

        assert result == OUTLET, (
            f"Without inlet_temp, outlet should pass through unchanged; "
            f"expected {OUTLET}°C, got {result}°C"
        )
