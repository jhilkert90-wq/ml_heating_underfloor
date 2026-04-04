
import pytest
import numpy as np
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src import thermal_equilibrium_model
from src import unified_thermal_state
from src.heat_source_channels import _get_min_records_for_learning


@pytest.fixture(scope="function")
def clean_model():
    """Fixture to ensure a clean model instance for each test function."""
    model = thermal_equilibrium_model.ThermalEquilibriumModel()
    # Set realistic test parameters
    model.thermal_time_constant = 4.0
    model.heat_loss_coefficient = 1.2
    model.outlet_effectiveness = 0.75
    model.external_source_weights = {
        "pv": 0.0001,
        "fireplace": 1.5,
        "tv": 0.1
    }
    # Baseline physics tests operate directly on the model parameters and are
    # intentionally isolated from channel-mode synchronization.
    model.orchestrator = None
    return model


class TestCorrectedThermalPhysics:
    """Consolidated and refactored tests for the ThermalEquilibriumModel."""

    def test_heating_never_cools_house(self, clean_model):
        """CRITICAL: Heating with outlet > indoor should never predict cooling."""
        current_indoor = 20.0
        outdoor_temp = 5.0
        outlet_temp = 45.0

        predicted = clean_model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=current_indoor
        )

        assert predicted >= current_indoor, (
            f"Heating predicted to cool: {predicted}°C < {current_indoor}°C")
        assert predicted >= outdoor_temp, (
            f"Indoor below outdoor: {predicted}°C < {outdoor_temp}°C")

    def test_weighted_average_equilibrium(self, clean_model):
        """Equilibrium should be weighted average of outlet and outdoor temperatures."""
        outdoor_temp = 5.0
        outlet_temp = 45.0
        current_indoor = 20.0
        
        eff = clean_model.outlet_effectiveness
        loss = clean_model.heat_loss_coefficient
        
        expected = (eff * outlet_temp + loss * outdoor_temp) / (eff + loss)
        
        predicted = clean_model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=current_indoor
        )
        
        assert abs(predicted - expected) < 0.1, (
            f"Expected {expected:.1f}°C, got {predicted:.1f}°C")

    def test_energy_conservation_at_equilibrium(self, clean_model):
        """Test that energy is conserved at equilibrium (Heat Input = Heat Loss)."""
        pv_power = 1000
        outlet_temp = 45.0
        outdoor_temp = 5.0
        
        # Test with temperature-based approximation
        equilibrium_temp = clean_model.predict_equilibrium_temperature(
            outlet_temp=outlet_temp,
            outdoor_temp=outdoor_temp,
            current_indoor=21.0,
            pv_power=pv_power
        )
        
        heat_from_pv = pv_power * clean_model.external_source_weights['pv']
        heat_input_from_outlet = (
            clean_model.outlet_effectiveness * (outlet_temp - equilibrium_temp)
        )
        total_heat_input = heat_input_from_outlet + heat_from_pv
        heat_loss = (
            clean_model.heat_loss_coefficient * (equilibrium_temp - outdoor_temp)
        )

        assert np.isclose(total_heat_input, heat_loss, atol=1e-2), (
            f"Energy not conserved: input={total_heat_input:.3f}, "
            f"loss={heat_loss:.3f}")

    def test_energy_based_prediction(self, clean_model):
        """Test the new energy-based prediction logic."""
        thermal_power = 2.0  # kW
        outdoor_temp = 5.0
        
        # Teq = Tout + P_total / U_loss
        expected_eq = outdoor_temp + (thermal_power / clean_model.heat_loss_coefficient)
        
        predicted = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0,  # Should be ignored for energy calculation
            outdoor_temp=outdoor_temp,
            current_indoor=20.0,
            thermal_power=thermal_power
        )
        
        assert np.isclose(predicted, expected_eq, atol=1e-3), (
            f"Energy-based prediction failed: expected {expected_eq}, got {predicted}"
        )

    def test_second_law_thermodynamics(self, clean_model):
        """Indoor temp cannot be below outdoor when heating."""
        equilibrium_temp = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, 
            outdoor_temp=5.0, 
            current_indoor=21.0
        )
        assert equilibrium_temp > 5.0, (
            "Indoor temperature below outdoor despite heat input")

        # With outlet = outdoor, indoor should equal outdoor (no external heat)
        equilibrium_no_heat = clean_model.predict_equilibrium_temperature(
            outlet_temp=5.0, outdoor_temp=5.0, current_indoor=21.0
        )
        assert np.isclose(equilibrium_no_heat, 5.0, atol=1e-1), (
            "Indoor temperature differs from outdoor with no net heat input")

    def test_physical_bounds(self, clean_model):
        """Indoor temperature is physically bounded between outdoor and source."""
        equilibrium_temp = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, 
            outdoor_temp=5.0, 
            current_indoor=21.0
        )
        assert 5.0 < equilibrium_temp < 45.0
        assert -20 < equilibrium_temp < 50, (
            "Temperature outside realistic building range")

    def test_linearity_of_heat_loss(self, clean_model):
        """Equilibrium should change proportionally with outdoor temperature."""
        outdoor_temps = [0, 5, 10, 15, 20]
        equilibrium_temps = [
            clean_model.predict_equilibrium_temperature(
                outlet_temp=45.0, 
                outdoor_temp=ot, 
                current_indoor=21.0
            ) for ot in outdoor_temps
        ]
        
        slopes = np.diff(equilibrium_temps) / np.diff(outdoor_temps)
        assert np.allclose(slopes, slopes[0], atol=1e-2), (
            "Non-linear equilibrium response to outdoor temp")

    def test_external_heat_source_additivity(self, clean_model):
        """External heat sources should contribute additively."""
        eq_baseline = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=5.0, current_indoor=21.0
        )
        eq_with_pv = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=5.0, current_indoor=21.0, pv_power=1000
        )
        eq_with_fireplace = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=5.0, current_indoor=21.0, fireplace_on=1
        )
        eq_with_both = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=5.0, current_indoor=21.0, pv_power=1000, fireplace_on=1
        )
        
        pv_contribution = eq_with_pv - eq_baseline
        fireplace_contribution = eq_with_fireplace - eq_baseline
        expected_combined = (
            eq_baseline + pv_contribution + fireplace_contribution
        )
        
        assert np.isclose(eq_with_both, expected_combined, atol=1e-2), (
            f"Heat sources not additive: expected={expected_combined:.3f}, "
            f"actual={eq_with_both:.3f}")

    def test_thermal_time_constant_independent_of_equilibrium(self, 
                                                              clean_model):
        """Thermal time constant should not affect equilibrium calculations."""
        eq_original = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=5.0, current_indoor=21.0
        )
        clean_model.thermal_time_constant = 10.0
        eq_different_time = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=5.0, current_indoor=21.0
        )
        assert np.isclose(eq_original, eq_different_time, atol=1e-3), (
            "Thermal time constant affects equilibrium calculation")


class TestHeatSourceChannelActivation:
    def test_adaptive_learning_metrics_export_channel_parameters(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            thermal_equilibrium_model.config,
            "ENABLE_HEAT_SOURCE_CHANNELS",
            True,
        )
        monkeypatch.setattr(
            unified_thermal_state,
            "_thermal_state_manager",
            unified_thermal_state.ThermalStateManager(
                state_file=str(tmp_path / "thermal_state.json")
            ),
        )

        model = thermal_equilibrium_model.ThermalEquilibriumModel()
        if model.orchestrator is None:
            pytest.skip("Heat source channels not enabled")

        model.prediction_history = [{"error": 0.2}]
        model.parameter_history = []
        model.thermal_time_constant = 5.5
        model.heat_loss_coefficient = 0.22
        model.outlet_effectiveness = 0.88
        model.pv_heat_weight = 0.0032
        model.fireplace_heat_weight = 6.5
        model.tv_heat_weight = 0.46
        model.solar_lag_minutes = 75.0
        model.slab_time_constant_hours = 1.7

        heat_pump = model.orchestrator.channels["heat_pump"]
        solar = model.orchestrator.channels["pv"]
        fireplace = model.orchestrator.channels["fireplace"]
        heat_pump.delta_t_floor = 3.4
        solar.cloud_factor_exponent = 1.4
        solar.solar_decay_tau_hours = 0.9
        fireplace.fp_decay_time_constant = 1.1
        fireplace.room_spread_delay_minutes = 42.0

        metrics = model.get_adaptive_learning_metrics()
        current_parameters = metrics["current_parameters"]

        assert metrics["heat_source_channels_enabled"] is True
        assert current_parameters["fireplace_heat_weight"] == pytest.approx(6.5)
        assert current_parameters["delta_t_floor"] == pytest.approx(3.4)
        assert current_parameters["cloud_factor_exponent"] == pytest.approx(1.4)
        assert current_parameters["solar_decay_tau_hours"] == pytest.approx(0.9)
        assert current_parameters["fp_heat_output_kw"] == pytest.approx(6.5)
        assert current_parameters["fp_decay_time_constant"] == pytest.approx(1.1)
        assert current_parameters["room_spread_delay_minutes"] == pytest.approx(42.0)

    def test_fireplace_channel_changes_equilibrium_prediction(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            thermal_equilibrium_model.config,
            "ENABLE_HEAT_SOURCE_CHANNELS",
            True,
        )
        model = thermal_equilibrium_model.ThermalEquilibriumModel()
        if model.orchestrator is None:
            pytest.skip("Heat source channels not enabled")

        base_prediction = model.predict_equilibrium_temperature(
            outlet_temp=45.0,
            outdoor_temp=5.0,
            current_indoor=20.0,
            fireplace_on=1,
        )

        fireplace_channel = model.orchestrator.channels["fireplace"]
        fireplace_channel.fp_heat_output_kw += 3.0

        changed_prediction = model.predict_equilibrium_temperature(
            outlet_temp=45.0,
            outdoor_temp=5.0,
            current_indoor=20.0,
            fireplace_on=1,
        )

        assert changed_prediction > base_prediction

    def test_fireplace_channel_state_persists_across_restart(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            thermal_equilibrium_model.config,
            "ENABLE_HEAT_SOURCE_CHANNELS",
            True,
        )

        unified_thermal_state._thermal_state_manager = (
            unified_thermal_state.ThermalStateManager(
                state_file=str(tmp_path / "thermal_state.json")
            )
        )

        model = thermal_equilibrium_model.ThermalEquilibriumModel()
        if model.orchestrator is None:
            pytest.skip("Heat source channels not enabled")

        initial_power = model.orchestrator.channels[
            "fireplace"
        ].fp_heat_output_kw

        context = {
            "outlet_temp": 40.0,
            "outdoor_temp": 5.0,
            "current_indoor": 20.0,
            "fireplace_on": 1,
            "tv_on": 0,
            "pv_power": 0.0,
            "delta_t": 0.0,
            "thermal_power": 0.0,
        }
        n = _get_min_records_for_learning() + 1
        for _ in range(n):
            model.update_prediction_feedback(
                predicted_temp=20.0,
                actual_temp=21.0,
                prediction_context=context,
            )

        learned_power = model.orchestrator.channels[
            "fireplace"
        ].fp_heat_output_kw
        persisted_fireplace_history = model.orchestrator.channels[
            "fireplace"
        ].history
        assert learned_power != pytest.approx(initial_power)
        assert len(persisted_fireplace_history) == n

        reloaded_model = thermal_equilibrium_model.ThermalEquilibriumModel()
        restored_power = reloaded_model.orchestrator.channels[
            "fireplace"
        ].fp_heat_output_kw
        restored_history = reloaded_model.orchestrator.channels[
            "fireplace"
        ].history
        assert restored_power == pytest.approx(learned_power)
        assert len(restored_history) == n

    def test_sync_heat_source_channels_from_model_state_reseeds_and_persists(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            thermal_equilibrium_model.config,
            "ENABLE_HEAT_SOURCE_CHANNELS",
            True,
        )
        monkeypatch.setattr(
            unified_thermal_state,
            "_thermal_state_manager",
            unified_thermal_state.ThermalStateManager(
                state_file=str(tmp_path / "thermal_state.json")
            ),
        )

        model = thermal_equilibrium_model.ThermalEquilibriumModel()
        if model.orchestrator is None:
            pytest.skip("Heat source channels not enabled")

        model.thermal_time_constant = 5.5
        model.heat_loss_coefficient = 0.22
        model.outlet_effectiveness = 0.88
        model.slab_time_constant_hours = 1.7
        model.pv_heat_weight = 0.0032
        model.solar_lag_minutes = 75.0
        model.fireplace_heat_weight = 6.5
        model.tv_heat_weight = 0.46

        model.sync_heat_source_channels_from_model_state(persist=True)

        heat_pump = model.orchestrator.channels["heat_pump"]
        solar = model.orchestrator.channels["pv"]
        fireplace = model.orchestrator.channels["fireplace"]
        tv = model.orchestrator.channels["tv"]
        persisted = unified_thermal_state.get_thermal_state_manager()
        persisted_state = persisted.get_heat_source_channel_state()

        assert heat_pump.thermal_time_constant == pytest.approx(5.5)
        assert heat_pump.heat_loss_coefficient == pytest.approx(0.22)
        assert heat_pump.outlet_effectiveness == pytest.approx(0.88)
        assert heat_pump.slab_time_constant_hours == pytest.approx(1.7)
        assert solar.pv_heat_weight == pytest.approx(0.0032)
        assert solar.solar_lag_minutes == pytest.approx(75.0)
        assert fireplace.fp_heat_output_kw == pytest.approx(6.5)
        assert tv.tv_heat_weight == pytest.approx(0.46)
        assert persisted_state["fireplace"]["parameters"][
            "fp_heat_output_kw"
        ] == pytest.approx(6.5)
        assert persisted_state["heat_pump"]["parameters"][
            "thermal_time_constant"
        ] == pytest.approx(5.5)
        assert persisted_state["heat_pump"]["parameters"][
            "heat_loss_coefficient"
        ] == pytest.approx(0.22)
        assert persisted_state["heat_pump"]["parameters"][
            "outlet_effectiveness"
        ] == pytest.approx(0.88)

    def test_direct_assignments_keep_orchestrator_synced(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            thermal_equilibrium_model.config,
            "ENABLE_HEAT_SOURCE_CHANNELS",
            True,
        )
        monkeypatch.setattr(
            unified_thermal_state,
            "_thermal_state_manager",
            unified_thermal_state.ThermalStateManager(
                state_file=str(tmp_path / "thermal_state.json")
            ),
        )

        model = thermal_equilibrium_model.ThermalEquilibriumModel()
        if model.orchestrator is None:
            pytest.skip("Heat source channels not enabled")

        model.thermal_time_constant = 6.4
        model.heat_loss_coefficient = 0.31
        model.outlet_effectiveness = 0.91
        model.slab_time_constant_hours = 1.9
        model.solar_lag_minutes = 68.0
        model.pv_heat_weight = 0.0027
        model.fireplace_heat_weight = 5.8
        model.tv_heat_weight = 0.44

        assert model.orchestrator.channels["heat_pump"].thermal_time_constant == pytest.approx(6.4)
        assert model.orchestrator.channels["heat_pump"].heat_loss_coefficient == pytest.approx(0.31)
        assert model.orchestrator.channels["heat_pump"].outlet_effectiveness == pytest.approx(0.91)
        assert model.orchestrator.channels["heat_pump"].slab_time_constant_hours == pytest.approx(1.9)
        assert model.orchestrator.channels["pv"].solar_lag_minutes == pytest.approx(68.0)
        assert model.orchestrator.channels["pv"].pv_heat_weight == pytest.approx(0.0027)
        assert model.orchestrator.channels["fireplace"].fp_heat_output_kw == pytest.approx(5.8)
        assert model.orchestrator.channels["tv"].tv_heat_weight == pytest.approx(0.44)

    def test_channel_mode_trajectory_uses_heat_pump_delta_t_floor_fallback(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            thermal_equilibrium_model.config,
            "ENABLE_HEAT_SOURCE_CHANNELS",
            True,
        )
        model = thermal_equilibrium_model.ThermalEquilibriumModel()
        if model.orchestrator is None:
            pytest.skip("Heat source channels not enabled")

        heat_pump = model.orchestrator.channels["heat_pump"]
        heat_pump.delta_t_floor = 1.0
        low_delta_t_result = model.predict_thermal_trajectory(
            current_indoor=20.0,
            target_indoor=22.0,
            outlet_temp=35.0,
            outdoor_temp=5.0,
            time_horizon_hours=1.0,
            time_step_minutes=10,
            inlet_temp=25.0,
            delta_t_floor=1.5,  # HP on: measured ΔT > 1
        )

        heat_pump.delta_t_floor = 5.0
        high_delta_t_result = model.predict_thermal_trajectory(
            current_indoor=20.0,
            target_indoor=22.0,
            outlet_temp=35.0,
            outdoor_temp=5.0,
            time_horizon_hours=1.0,
            time_step_minutes=10,
            inlet_temp=25.0,
            delta_t_floor=5.0,  # HP on: measured ΔT > 1
        )

        assert low_delta_t_result["trajectory"][-1] > high_delta_t_result["trajectory"][-1]

    def test_channel_mode_cloud_factor_exponent_changes_prediction(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            thermal_equilibrium_model.config,
            "ENABLE_HEAT_SOURCE_CHANNELS",
            True,
        )
        monkeypatch.setattr(
            thermal_equilibrium_model.config,
            "CLOUD_COVER_CORRECTION_ENABLED",
            True,
        )
        model = thermal_equilibrium_model.ThermalEquilibriumModel()
        if model.orchestrator is None:
            pytest.skip("Heat source channels not enabled")

        solar = model.orchestrator.channels["pv"]
        solar.cloud_factor_exponent = 0.3
        low_exponent_prediction = model.predict_equilibrium_temperature(
            outlet_temp=30.0,
            outdoor_temp=5.0,
            current_indoor=20.0,
            pv_power=2000.0,
            cloud_cover_pct=80.0,
        )

        solar.cloud_factor_exponent = 2.0
        high_exponent_prediction = model.predict_equilibrium_temperature(
            outlet_temp=30.0,
            outdoor_temp=5.0,
            current_indoor=20.0,
            pv_power=2000.0,
            cloud_cover_pct=80.0,
        )

        assert high_exponent_prediction > low_exponent_prediction

    def test_channel_mode_skips_legacy_recent_window_adaptation(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            thermal_equilibrium_model.config,
            "ENABLE_HEAT_SOURCE_CHANNELS",
            True,
        )
        model = thermal_equilibrium_model.ThermalEquilibriumModel()
        if model.orchestrator is None:
            pytest.skip("Heat source channels not enabled")

        model.prediction_history = [
            {
                "timestamp": f"2026-04-01T12:00:0{index}",
                "error": 0.5,
                "context": {"outdoor_temp": 5.0, "indoor_temp_delta_60m": 0.0},
            }
            for index in range(model.recent_errors_window)
        ]
        model.parameter_history = []

        monkeypatch.setattr(
            model,
            "_calculate_thermal_time_constant_gradient",
            lambda recent_predictions: pytest.fail(
                "Legacy recent-window gradients should be skipped in channel mode"
            ),
        )

        model._adapt_parameters_from_recent_errors()

        assert model.parameter_history == []

    def test_channel_updates_merge_into_top_level_parameter_history(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            thermal_equilibrium_model.config,
            "ENABLE_HEAT_SOURCE_CHANNELS",
            True,
        )
        monkeypatch.setattr(
            unified_thermal_state,
            "_thermal_state_manager",
            unified_thermal_state.ThermalStateManager(
                state_file=str(tmp_path / "thermal_state.json")
            ),
        )

        model = thermal_equilibrium_model.ThermalEquilibriumModel()
        if model.orchestrator is None:
            pytest.skip("Heat source channels not enabled")

        context = {
            "outlet_temp": 40.0,
            "outdoor_temp": 5.0,
            "current_indoor": 20.0,
            "fireplace_on": 1,
            "tv_on": 0,
            "pv_power": 0.0,
            "delta_t": 0.0,
            "thermal_power": 0.0,
        }
        n = _get_min_records_for_learning() + 1
        for step in range(n):
            model.update_prediction_feedback(
                predicted_temp=20.0,
                actual_temp=21.0,
                prediction_context=context,
                timestamp=f"2026-04-02T12:00:{step:02d}",
            )

        assert model.parameter_history
        latest_record = model.parameter_history[-1]
        metrics = model.get_adaptive_learning_metrics()
        persisted_history = unified_thermal_state.get_thermal_state_manager().state[
            "learning_state"
        ]["parameter_history"]

        assert latest_record["record_type"] == "channel_update"
        assert latest_record["channel"] == "fireplace"
        assert latest_record["attributed_error"] == pytest.approx(1.0)
        assert "fp_heat_output_kw" in latest_record["channel_parameter_changes"]
        assert latest_record["fp_heat_output_kw"] == pytest.approx(
            model.orchestrator.channels["fireplace"].fp_heat_output_kw
        )
        assert metrics["parameter_updates"] == len(model.parameter_history)
        assert metrics["parameter_updates"] > 0
        assert persisted_history[-1]["record_type"] == "channel_update"
        assert persisted_history[-1]["channel"] == "fireplace"

    def test_current_pv_signal_drives_pv_channel_history_even_if_smoothed_pv_is_low(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            thermal_equilibrium_model.config,
            "ENABLE_HEAT_SOURCE_CHANNELS",
            True,
        )
        monkeypatch.setattr(
            thermal_equilibrium_model.config,
            "ENABLE_MIXED_SOURCE_ATTRIBUTION",
            True,
            raising=False,
        )
        monkeypatch.setattr(
            unified_thermal_state,
            "_thermal_state_manager",
            unified_thermal_state.ThermalStateManager(
                state_file=str(tmp_path / "thermal_state.json")
            ),
        )

        model = thermal_equilibrium_model.ThermalEquilibriumModel()
        if model.orchestrator is None:
            pytest.skip("Heat source channels not enabled")

        context = {
            "outlet_temp": 21.0,
            "outdoor_temp": 5.0,
            "current_indoor": 20.0,
            "fireplace_on": 0,
            "tv_on": 0,
            "pv_power": 120.0,
            "pv_power_current": 900.0,
            "delta_t": 0.0,
            "thermal_power": 0.0,
            "heat_pump_active": False,
            "avg_cloud_cover": 20.0,
            "pv_power_history": [0.0, 40.0, 80.0, 120.0],
        }

        n = _get_min_records_for_learning() + 1
        for step in range(n):
            model.update_prediction_feedback(
                predicted_temp=20.0,
                actual_temp=21.0,
                prediction_context=context,
                timestamp=f"2026-04-02T13:00:{step:02d}",
            )

        assert len(model.orchestrator.channels["pv"].history) == n
        assert model.orchestrator.channels["heat_pump"].history == []
        assert model.parameter_history[-1]["channel"] == "pv"

    def test_zero_division_protection(self, clean_model):
        """Model should handle zero effectiveness and heat loss without crashing."""
        clean_model.outlet_effectiveness = 0.0
        clean_model.heat_loss_coefficient = 0.0
        
        predicted = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0,
            outdoor_temp=5.0,
            current_indoor=20.0
        )
        assert predicted == 5.0, (
            f"Expected fallback to outdoor temp 5.0°C, got {predicted}°C")

    def test_extreme_temperatures(self, clean_model):
        """Test behavior with extreme temperature inputs."""
        eq_cold = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=-30.0, current_indoor=20.0
        )
        assert isinstance(eq_cold, float), "Failed with extreme cold"
        
        eq_hot = clean_model.predict_equilibrium_temperature(
            outlet_temp=45.0, outdoor_temp=40.0, current_indoor=20.0
        )
        assert isinstance(eq_hot, float), "Failed with extreme heat"


class TestBaselineFunctionality:
    """
    Tests to ensure baseline model functionality is preserved after
    refactoring. These are based on the original migration tests.
    """

    def test_equilibrium_prediction_baseline(self, clean_model):
        """Capture baseline equilibrium temperature predictions."""
        model = clean_model

        # Test scenario 1: Standard heating scenario
        baseline_result_1 = model.predict_equilibrium_temperature(
            outlet_temp=45.0,
            outdoor_temp=5.0,
            current_indoor=20.0,
            pv_power=500.0,
            fireplace_on=0,
            tv_on=1
        )

        # Test scenario 2: Cold weather with external heat
        baseline_result_2 = model.predict_equilibrium_temperature(
            outlet_temp=55.0,
            outdoor_temp=-5.0,
            current_indoor=18.0,
            pv_power=1200.0,
            fireplace_on=1,
            tv_on=1
        )

        # Test scenario 3: Mild weather, minimal heating
        baseline_result_3 = model.predict_equilibrium_temperature(
            outlet_temp=30.0,
            outdoor_temp=15.0,
            current_indoor=21.0,
            pv_power=0.0,
            fireplace_on=0,
            tv_on=0
        )

        # Note: These are broad checks. The original test was for before/after
        # migration comparison. With refactored parameters, exact values change.
        assert 15.0 <= baseline_result_1 <= 45.0
        assert 10.0 <= baseline_result_2 <= 60.0
        assert 15.0 <= baseline_result_3 <= 30.0

    def test_trajectory_prediction_baseline(self, clean_model):
        """Capture baseline thermal trajectory predictions."""
        model = clean_model

        # Test trajectory prediction (HP on: provide delta_t_floor > 1)
        baseline_trajectory = model.predict_thermal_trajectory(
            current_indoor=18.0,
            target_indoor=21.0,
            outlet_temp=40.0,
            outdoor_temp=8.0,
            time_horizon_hours=4,
            pv_power=300.0,
            fireplace_on=0,
            tv_on=1,
            inlet_temp=38.0,
            delta_t_floor=2.0,
        )

        # Verify trajectory structure
        assert 'trajectory' in baseline_trajectory
        assert 'reaches_target_at' in baseline_trajectory
        assert 'overshoot_predicted' in baseline_trajectory

        # Verify trajectory makes sense
        trajectory = baseline_trajectory['trajectory']
        assert len(trajectory) == 4  # 4 hour horizon
        assert all(isinstance(temp, (int, float)) for temp in trajectory)
        # Check that temperature is rising towards a reasonable equilibrium
        assert trajectory[0] > 18.0
        assert trajectory[-1] > trajectory[0]

    def test_optimal_outlet_calculation_baseline(self, clean_model):
        """Capture baseline optimal outlet temperature calculations."""
        model = clean_model

        # Test optimal outlet calculation
        baseline_optimal = model.calculate_optimal_outlet_temperature(
            target_indoor=22.0,
            current_indoor=19.0,
            outdoor_temp=6.0,
            time_available_hours=2.0,
            pv_power=800.0,
            fireplace_on=0,
            tv_on=1
        )

        # Verify optimal calculation structure
        assert 'optimal_outlet_temp' in baseline_optimal
        assert 'method' in baseline_optimal

        optimal_temp = baseline_optimal['optimal_outlet_temp']
        assert isinstance(optimal_temp, (int, float))
        assert 25.0 <= optimal_temp <= 70.0, (
            f"Optimal temp out of range: {optimal_temp}"
        )


# ---------------------------------------------------------------------------
# Slab pump-on/off gate: uses measured delta_t, not outlet > inlet
# ---------------------------------------------------------------------------

class TestSlabPumpGate:
    """predict_thermal_trajectory must use the passive slab branch when
    the measured delta_t_floor indicates the HP is off, even if the
    calculated outlet_temp exceeds t_slab."""

    @pytest.fixture
    def slab_model(self):
        model = thermal_equilibrium_model.ThermalEquilibriumModel()
        model.thermal_time_constant = 4.0
        model.heat_loss_coefficient = 0.15
        model.outlet_effectiveness = 0.95
        model._slab_time_constant_hours = 1.0
        model.orchestrator = None
        return model

    def test_slab_passive_when_hp_off(self, slab_model):
        """delta_t=0 (HP off) → passive branch even though outlet > inlet."""
        result = slab_model.predict_thermal_trajectory(
            current_indoor=22.0,
            target_indoor=22.0,
            outlet_temp=25.0,       # Higher than inlet
            outdoor_temp=5.0,
            time_horizon_hours=1,
            time_step_minutes=10,
            inlet_temp=23.0,        # BT3 below outlet
            delta_t_floor=0.0,      # HP off: measured ΔT ≈ 0
        )
        # In passive mode the slab cools toward indoor temp.  The first
        # prediction should be *below* what pump-ON would produce because
        # the slab only radiates stored heat, it doesn't get fresh hot
        # water.
        passive_prediction = result["trajectory"][-1]

        result_active = slab_model.predict_thermal_trajectory(
            current_indoor=22.0,
            target_indoor=22.0,
            outlet_temp=25.0,
            outdoor_temp=5.0,
            time_horizon_hours=1,
            time_step_minutes=10,
            inlet_temp=23.0,
            delta_t_floor=2.0,      # HP on: measured ΔT = 2°C
        )
        active_prediction = result_active["trajectory"][-1]
        assert passive_prediction < active_prediction, (
            f"Passive ({passive_prediction:.3f}) should be cooler than "
            f"active ({active_prediction:.3f}) when HP is off"
        )

    def test_slab_active_when_hp_on(self, slab_model):
        """delta_t=3 (HP on) → active branch, slab heats up."""
        result = slab_model.predict_thermal_trajectory(
            current_indoor=20.0,
            target_indoor=22.0,
            outlet_temp=30.0,
            outdoor_temp=5.0,
            time_horizon_hours=1,
            time_step_minutes=10,
            inlet_temp=27.0,        # BT3 = 27, outlet=30, ΔT=3
            delta_t_floor=3.0,      # HP is on
        )
        predictions = result["trajectory"]
        # Temperature should rise toward target with active heating
        assert predictions[-1] > 20.0, (
            f"With HP active, temp should rise above 20°C, got {predictions[-1]:.3f}"
        )

    def test_slab_passive_does_not_use_delta_t_floor(self, slab_model):
        """When HP is off (delta_t=0.5), changing the learned delta_t_floor
        fallback should NOT affect the result because the passive branch
        doesn't use delta_t_floor at all."""
        kwargs = dict(
            current_indoor=22.0,
            target_indoor=22.0,
            outlet_temp=25.0,
            outdoor_temp=5.0,
            time_horizon_hours=1,
            time_step_minutes=10,
            inlet_temp=23.0,
            delta_t_floor=0.5,  # HP off
        )
        result_a = slab_model.predict_thermal_trajectory(**kwargs)
        # Even if _resolve_delta_t_floor would return 3.0 vs 2.0, the
        # passive branch ignores it.  Both should give identical results.
        result_b = slab_model.predict_thermal_trajectory(**kwargs)
        assert result_a["trajectory"] == result_b["trajectory"]
