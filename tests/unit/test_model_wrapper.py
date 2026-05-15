
import pytest
import pandas as pd
import os
from unittest.mock import MagicMock, patch

# Ensure the app's config is loaded before other imports
from src import config, model_wrapper, unified_thermal_state
from src.model_wrapper import (
    simplified_outlet_prediction, get_enhanced_model_wrapper
)

# Set a consistent cycle time for testing
config.CYCLE_INTERVAL_MINUTES = 15


@pytest.fixture(scope="function")
def clean_state():
    """Fixture to ensure a clean state for each test function."""
    test_state_file = "thermal_state.json"

    # Ensure no previous state file exists
    if os.path.exists(test_state_file):
        os.remove(test_state_file)

    # Reset singleton instances in their original modules
    model_wrapper._enhanced_model_wrapper_instance = None
    unified_thermal_state._thermal_state_manager = None

    # Initialize a fresh ThermalStateManager with the test state file
    # This bypasses the default UNIFIED_STATE_FILE path which might point to /opt/ml_heating/
    manager = unified_thermal_state.ThermalStateManager(state_file=test_state_file)
    unified_thermal_state._thermal_state_manager = manager

    yield

    # Clean up after the test
    if os.path.exists(test_state_file):
        os.remove(test_state_file)
    model_wrapper._enhanced_model_wrapper_instance = None
    unified_thermal_state._thermal_state_manager = None


@pytest.fixture
def wrapper_instance(clean_state):
    """Fixture to get a clean EnhancedModelWrapper instance."""
    return get_enhanced_model_wrapper()


class TestEnhancedModelWrapper:
    """Consolidated tests for the EnhancedModelWrapper."""

    def test_initialization(self, wrapper_instance):
        """Test that the wrapper initializes correctly."""
        assert wrapper_instance is not None
        assert wrapper_instance.thermal_model is not None
        assert wrapper_instance.learning_enabled is True
        if config.ENABLE_HEAT_SOURCE_CHANNELS:
            assert wrapper_instance.adaptive_fireplace is None
        # A fresh instance starts with cycle_count 0 from the manager
        assert wrapper_instance.cycle_count == 0

    def test_singleton_pattern(self, clean_state):
        """Test that the singleton pattern works as expected."""
        wrapper1 = get_enhanced_model_wrapper()
        wrapper2 = get_enhanced_model_wrapper()
        assert wrapper1 is wrapper2

    def test_simplified_prediction(self, wrapper_instance):
        """Test the simplified_outlet_prediction function."""
        with patch.object(
            wrapper_instance,
            'calculate_optimal_outlet_temp',
            return_value=(35.0, {'learning_confidence': 4.5}),
        ):
            test_features = pd.DataFrame([{
                'indoor_temp_lag_30m': 20.5,
                'target_temp': 21.0,
                'outdoor_temp': 5.0,
                'pv_now': 1500.0,
                'fireplace_on': 0,
                'tv_on': 1
            }])

            outlet_temp, confidence, metadata = simplified_outlet_prediction(
                test_features, 20.5, 21.0
            )

        assert outlet_temp == 35.0
        assert confidence == 4.5
        # `calculate_optimal_outlet_temp` is mocked
        assert 'prediction_method' not in metadata

    def test_enhanced_prediction(self, wrapper_instance):
        """Test calculate_optimal_outlet_temp with a basic scenario."""
        test_features = {
            'indoor_temp_lag_30m': 20.5,
            'target_temp': 21.0,
            'outdoor_temp': 5.0,
            'pv_now': 0.0,  # No PV so heating is required
            'fireplace_on': 0,
            'tv_on': 0,
        }

        optimal_temp, metadata = (
            wrapper_instance.calculate_optimal_outlet_temp(test_features)
        )

        assert isinstance(optimal_temp, float)
        assert optimal_temp > 21.0  # Should be higher than indoor temp when heating is needed
        assert 'learning_confidence' in metadata
        assert metadata['prediction_method'] == \
            'thermal_equilibrium_single_prediction'

    def test_learning_feedback(self, wrapper_instance):
        """Test the learn_from_prediction_feedback method works."""
        with patch('src.model_wrapper.create_influx_service'), patch(
            'src.model_wrapper.create_ha_client'
        ):
            # Set cycle count to 2 to avoid first-cycle skip
            wrapper_instance.cycle_count = 2

            wrapper_instance.learn_from_prediction_feedback(
                predicted_temp=35.0,
                actual_temp=34.2,
                prediction_context={
                    'indoor_temp': 20.5,
                    'outdoor_temp': 5.0,
                },
            )

        assert wrapper_instance.cycle_count == 3
        # Further assertions would require mocking the thermal model's
        # internal state

    def test_comprehensive_metrics_include_channel_fields(
        self, clean_state, monkeypatch
    ):
        monkeypatch.setattr(config, "ENABLE_HEAT_SOURCE_CHANNELS", True)
        model_wrapper._enhanced_model_wrapper_instance = None

        wrapper = get_enhanced_model_wrapper()
        if wrapper.thermal_model.orchestrator is None:
            pytest.skip("Heat source channels not enabled")

        wrapper.thermal_model.prediction_history = [{"error": 0.2}]
        wrapper.thermal_model.parameter_history = []
        wrapper.thermal_model.fireplace_heat_weight = 6.4

        heat_pump = wrapper.thermal_model.orchestrator.channels["heat_pump"]
        solar = wrapper.thermal_model.orchestrator.channels["pv"]
        fireplace = wrapper.thermal_model.orchestrator.channels["fireplace"]
        heat_pump.delta_t_floor = 3.2
        solar.cloud_factor_exponent = 1.3
        solar.solar_decay_tau_hours = 0.8
        fireplace.fp_decay_time_constant = 1.0
        fireplace.room_spread_delay_minutes = 38.0

        ha_metrics = wrapper.get_comprehensive_metrics_for_ha()

        assert ha_metrics["heat_source_channels_enabled"] is True
        assert ha_metrics["fireplace_heat_weight"] == pytest.approx(6.4)
        assert ha_metrics["delta_t_floor"] == pytest.approx(3.2)
        assert ha_metrics["cloud_factor_exponent"] == pytest.approx(1.3)
        assert ha_metrics["solar_decay_tau_hours"] == pytest.approx(0.8)
        assert ha_metrics["fp_heat_output_kw"] == pytest.approx(6.4)
        assert ha_metrics["fp_decay_time_constant"] == pytest.approx(1.0)
        assert ha_metrics["room_spread_delay_minutes"] == pytest.approx(38.0)

    def test_learning_feedback_only_exports_influx_on_configured_interval(
        self, wrapper_instance, monkeypatch
    ):
        monkeypatch.setattr(
            config,
            "INFLUX_METRICS_EXPORT_INTERVAL_CYCLES",
            5,
            raising=False,
        )

        context = {
            "outlet_temp": 40.0,
            "outdoor_temp": 5.0,
            "current_indoor": 20.0,
            "fireplace_on": 0,
            "pv_power": 0.0,
            "tv_on": 0,
        }

        with patch.object(
            wrapper_instance, "_export_metrics_to_influxdb"
        ) as mock_influx_export, patch.object(
            wrapper_instance, "export_metrics_to_ha"
        ) as mock_ha_export, patch.object(
            wrapper_instance.thermal_model,
            "update_prediction_feedback",
            return_value=0.1,
        ):
            wrapper_instance.cycle_count = 3
            wrapper_instance.learn_from_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.2,
                prediction_context=context,
            )

            mock_influx_export.assert_not_called()
            mock_ha_export.assert_called_once()

            mock_influx_export.reset_mock()
            mock_ha_export.reset_mock()

            wrapper_instance.cycle_count = 4
            wrapper_instance.learn_from_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.2,
                prediction_context=context,
            )

            mock_influx_export.assert_called_once()
            mock_ha_export.assert_called_once()

    def test_learning_feedback_skips_live_ha_export_in_effective_shadow_mode(
        self, wrapper_instance
    ):
        context = {
            "outlet_temp": 40.0,
            "outdoor_temp": 5.0,
            "current_indoor": 20.0,
            "fireplace_on": 0,
            "pv_power": 0.0,
            "tv_on": 0,
        }

        with patch.object(
            wrapper_instance, "_export_metrics_to_influxdb"
        ) as mock_influx_export, patch.object(
            wrapper_instance, "export_metrics_to_ha"
        ) as mock_ha_export, patch.object(
            wrapper_instance.thermal_model,
            "update_prediction_feedback",
            return_value=0.1,
        ):
            wrapper_instance.cycle_count = 3
            wrapper_instance.learn_from_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.2,
                prediction_context=context,
                effective_shadow_mode=True,
            )

            mock_influx_export.assert_not_called()
            mock_ha_export.assert_not_called()
            assert wrapper_instance.cycle_count == 4

    def test_learning_feedback_exports_ha_metrics_in_shadow_deployment(
        self, wrapper_instance, monkeypatch
    ):
        context = {
            "outlet_temp": 40.0,
            "outdoor_temp": 5.0,
            "current_indoor": 20.0,
            "fireplace_on": 0,
            "pv_power": 0.0,
            "tv_on": 0,
        }
        monkeypatch.setattr(config, "SHADOW_MODE", True)

        with patch.object(
            wrapper_instance, "_export_metrics_to_influxdb"
        ) as mock_influx_export, patch.object(
            wrapper_instance, "export_metrics_to_ha"
        ) as mock_ha_export, patch.object(
            wrapper_instance.thermal_model,
            "update_prediction_feedback",
            return_value=0.1,
        ):
            wrapper_instance.cycle_count = 3
            wrapper_instance.learn_from_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=21.2,
                prediction_context=context,
                effective_shadow_mode=True,
            )

            mock_influx_export.assert_not_called()
            mock_ha_export.assert_called_once()

    def test_first_cycle_learning_skip(self, wrapper_instance):
        """Verify that online learning is skipped on the first cycle."""
        with patch('logging.info') as mock_log_info:
            # In a clean state, cycle_count starts at 0, from a fresh state file
            # NOTE: The unified_thermal_state.py _get_default_state initializes
            # cycle_count to 0.
            # However, if the test environment has a lingering state file or if the
            # wrapper initialization logic has changed, this might be different.
            # We explicitly reset it here to ensure the test precondition is met.
            wrapper_instance.cycle_count = 0
            assert wrapper_instance.cycle_count == 0

            initial_params = {
                "thermal_time_constant":
                    wrapper_instance.thermal_model.thermal_time_constant,
                "heat_loss_coefficient":
                    wrapper_instance.thermal_model.heat_loss_coefficient,
                "outlet_effectiveness":
                    wrapper_instance.thermal_model.outlet_effectiveness,
            }

            # First call should be skipped
            wrapper_instance.learn_from_prediction_feedback(
                predicted_temp=22.0,
                actual_temp=21.0,
                prediction_context={
                    'outdoor_temp': 10.0, 'outlet_temp': 40.0,
                    'current_indoor': 20.5
                }
            )

            params_after_first_call = {
                "thermal_time_constant":
                    wrapper_instance.thermal_model.thermal_time_constant,
                "heat_loss_coefficient":
                    wrapper_instance.thermal_model.heat_loss_coefficient,
                "outlet_effectiveness":
                    wrapper_instance.thermal_model.outlet_effectiveness,
            }

        assert initial_params == params_after_first_call
        mock_log_info.assert_any_call(
            "Skipping online learning on the first cycle to ensure stability."
        )
        # The first call increments cycle_count from 0 to 1 and returns
        assert wrapper_instance.cycle_count == 1

    def test_binary_search_heating(self, wrapper_instance):
        """Test the binary search for a heating scenario."""
        # This is an indirect test of _calculate_required_outlet_temp
        test_features = {
            'indoor_temp_lag_30m': 20.0,
            'target_temp': 22.0,  # Heating needed
            'outdoor_temp': 5.0,
        }

        optimal_temp, _ = (
            wrapper_instance.calculate_optimal_outlet_temp(test_features)
        )

        # Expect a relatively high outlet temperature
        assert optimal_temp > 25.0

    def test_binary_search_cooling(self, wrapper_instance):
        """Test the binary search for a cooling scenario."""
        # This is an indirect test of _calculate_required_outlet_temp
        test_features = {
            'indoor_temp_lag_30m': 22.0,
            'target_temp': 21.0,  # Cooling needed
            'outdoor_temp': 25.0,  # Warmer outside
        }

        optimal_temp, _ = (
            wrapper_instance.calculate_optimal_outlet_temp(test_features)
        )

        # Expect a low outlet temperature (close to minimum)
        assert optimal_temp < 30.0

    def test_predict_indoor_temp(self, wrapper_instance):
        """Test the predict_indoor_temp method for smart rounding."""
        predicted_indoor = wrapper_instance.predict_indoor_temp(
            outlet_temp=40.0,
            outdoor_temp=10.0,
            current_indoor=20.0
        )

        assert isinstance(predicted_indoor, float)
        # Should be between current and outlet
        assert 20.0 < predicted_indoor < 40.0

    def test_fireplace_channel_mode_uses_channel_estimate(
        self, clean_state, monkeypatch
    ):
        """Flag-on mode must bypass adaptive fireplace learning entirely."""
        monkeypatch.setattr(config, "ENABLE_HEAT_SOURCE_CHANNELS", True)
        model_wrapper._enhanced_model_wrapper_instance = None

        wrapper = get_enhanced_model_wrapper()
        assert wrapper.adaptive_fireplace is None

        fireplace_channel = wrapper.thermal_model.orchestrator.channels[
            "fireplace"
        ]
        fireplace_channel.fp_heat_output_kw = 7.5

        with patch.object(
            wrapper.thermal_model,
            "predict_thermal_trajectory",
            return_value={"trajectory": [21.5]},
        ) as mocked_trajectory:
            wrapper.predict_indoor_temp(
                outlet_temp=40.0,
                outdoor_temp=5.0,
                current_indoor=20.0,
                fireplace_on=1,
            )

        assert mocked_trajectory.call_args.kwargs["fireplace_power_kw"] == pytest.approx(7.5)

    def test_legacy_fireplace_mode_uses_adaptive_learning(
        self, clean_state, monkeypatch
    ):
        """Flag-off mode keeps adaptive fireplace learning as legacy behavior."""
        mock_learner = MagicMock()
        mock_learner._calculate_learned_heat_contribution.return_value = {
            "heat_contribution_kw": 6.2,
            "learning_confidence": 0.9,
        }

        monkeypatch.setattr(config, "ENABLE_HEAT_SOURCE_CHANNELS", False)

        with patch(
            "src.model_wrapper.AdaptiveFireplaceLearning",
            return_value=mock_learner,
        ):
            model_wrapper._enhanced_model_wrapper_instance = None

            wrapper = get_enhanced_model_wrapper()
            with patch.object(
                wrapper.thermal_model,
                "predict_thermal_trajectory",
                return_value={"trajectory": [21.5]},
            ) as mocked_trajectory:
                wrapper.predict_indoor_temp(
                    outlet_temp=40.0,
                    outdoor_temp=5.0,
                    current_indoor=20.0,
                    fireplace_on=1,
                )

        mock_learner._calculate_learned_heat_contribution.assert_called_once()
        assert mocked_trajectory.call_args.kwargs["fireplace_power_kw"] == pytest.approx(6.2)

    def test_fireplace_learning_integration(self, clean_state, monkeypatch):
        """Legacy flag-off mode still observes adaptive fireplace sessions."""
        monkeypatch.setattr(config, "ENABLE_HEAT_SOURCE_CHANNELS", False)
        model_wrapper._enhanced_model_wrapper_instance = None

        wrapper = get_enhanced_model_wrapper()
        mock_learner = MagicMock()
        wrapper.adaptive_fireplace = mock_learner

        wrapper.cycle_count = 5
        context = {
            'outlet_temp': 40.0,
            'outdoor_temp': 5.0,
            'current_indoor': 20.0,
            'fireplace_on': 1,
            'tv_on': 0,
            'pv_power': 0,
        }

        with patch.object(wrapper, "_export_metrics_to_influxdb"), patch.object(
            wrapper, "export_metrics_to_ha"
        ):
            wrapper.learn_from_prediction_feedback(
                predicted_temp=21.0,
                actual_temp=22.0,
                prediction_context=context,
            )

        mock_learner.observe_fireplace_state.assert_called_once()
        _, kwargs = mock_learner.observe_fireplace_state.call_args
        assert kwargs['living_room_temp'] == 20.0


# ---------------------------------------------------------------------------
# Cloud discount on PV scalar in _extract_thermal_features
# ---------------------------------------------------------------------------

class TestPvScalarCloudDiscount:
    """_extract_thermal_features should derive pv_scalar from the rolling
    pv_power_history window (solar lag) and apply cloud discount so that the
    binary search sees a realistic PV contribution, not a raw sensor spike."""

    @pytest.fixture
    def wrapper(self):
        w = model_wrapper.EnhancedModelWrapper.__new__(
            model_wrapper.EnhancedModelWrapper
        )
        w._avg_cloud_cover = 50.0
        w._cloud_cover_forecast = [50.0] * 6
        return w

    def test_cloud_discount_reduces_pv_scalar(self, wrapper, monkeypatch):
        """With 60% cloud cover, pv_scalar should be substantially less
        than pv_now. Rolling-window average path is exercised."""
        monkeypatch.setattr(config, "CLOUD_COVER_CORRECTION_ENABLED", True)
        monkeypatch.setattr(config, "CLOUD_CORRECTION_MIN_FACTOR", 0.1,
                            raising=False)
        monkeypatch.setattr(config, "PV_TRAJ_ZERO_W", 50.0, raising=False)
        features = {
            "pv_power_history": [4000] * 18,  # 3h of 4kW
            "pv_now": 4000,
            "pv_forecast_1h": 3000.0,  # sun still forecast → rolling-window path
            "cloud_cover_1h": 60.0,
            "avg_cloud_cover": 60.0,
        }
        result = wrapper._extract_thermal_features(features)
        # Rolling avg = 4000. Cloud factor ~0.4 → ~1600.
        assert result["pv_power"] < 4000 * 0.8, (
            f"Expected cloud-discounted PV, got {result['pv_power']}"
        )
        assert result["pv_power"] > 0, "PV should not be zero"

    def test_cloud_discount_clear_sky(self, wrapper, monkeypatch):
        """With 0% cloud cover, pv_scalar should be unchanged (factor=1.0)."""
        monkeypatch.setattr(config, "CLOUD_COVER_CORRECTION_ENABLED", True)
        monkeypatch.setattr(config, "CLOUD_CORRECTION_MIN_FACTOR", 0.1,
                            raising=False)
        monkeypatch.setattr(config, "PV_TRAJ_ZERO_W", 50.0, raising=False)
        features = {
            "pv_power_history": [3000] * 18,
            "pv_now": 3000,
            "pv_forecast_1h": 2500.0,
            "cloud_cover_1h": 0.0,
            "avg_cloud_cover": 0.0,
        }
        result = wrapper._extract_thermal_features(features)
        # Rolling avg = 3000. Cloud factor = 1.0 → pv_power = 3000.
        assert result["pv_power"] == pytest.approx(3000.0, rel=0.01), (
            f"Clear sky should not discount PV, got {result['pv_power']}"
        )

    def test_cloud_discount_disabled(self, wrapper, monkeypatch):
        """When CLOUD_COVER_CORRECTION_ENABLED=False, pv_scalar should be raw."""
        monkeypatch.setattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False)
        monkeypatch.setattr(config, "PV_TRAJ_ZERO_W", 50.0, raising=False)
        features = {
            "pv_power_history": [4000] * 18,
            "pv_now": 4000,
            "pv_forecast_1h": 3500.0,
            "cloud_cover_1h": 80.0,
            "avg_cloud_cover": 80.0,
        }
        result = wrapper._extract_thermal_features(features)
        # Rolling avg = 4000. No cloud discount applied → 4000.
        assert result["pv_power"] == pytest.approx(4000.0, rel=0.01), (
            f"Cloud discount disabled, pv should be raw, got {result['pv_power']}"
        )

    def test_cloud_discount_uses_1h_forecast(self, wrapper, monkeypatch):
        """Should prefer cloud_cover_1h over avg_cloud_cover."""
        monkeypatch.setattr(config, "CLOUD_COVER_CORRECTION_ENABLED", True)
        monkeypatch.setattr(config, "CLOUD_CORRECTION_MIN_FACTOR", 0.1,
                            raising=False)
        monkeypatch.setattr(config, "PV_TRAJ_ZERO_W", 50.0, raising=False)
        features = {
            "pv_power_history": [2000] * 18,
            "pv_now": 2000,
            "pv_forecast_1h": 1800.0,
            "cloud_cover_1h": 0.0,      # Clear 1h forecast
            "avg_cloud_cover": 90.0,    # Heavy avg
        }
        result = wrapper._extract_thermal_features(features)
        # Should use cloud_cover_1h=0 → factor=1.0 → pv=2000
        assert result["pv_power"] == pytest.approx(2000.0, rel=0.01), (
            f"Should use 1h forecast (clear), got {result['pv_power']}"
        )

    def test_pv_scalar_uses_rolling_window_average(self, wrapper, monkeypatch):
        """pv_scalar should be the mean of pv_power_history, not pv_now."""
        monkeypatch.setattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False)
        monkeypatch.setattr(config, "PV_TRAJ_ZERO_W", 50.0, raising=False)
        # History is 2000W but current reading is 4000W
        features = {
            "pv_power_history": [2000] * 12,
            "pv_now": 4000,
            "pv_forecast_1h": 3000.0,  # sun still forecast → use rolling window
        }
        result = wrapper._extract_thermal_features(features)
        # pv_scalar should be the average of history (2000), not pv_now (4000)
        assert result["pv_power"] == pytest.approx(2000.0, rel=0.01), (
            f"Expected rolling-window avg 2000, got {result['pv_power']}"
        )

    def test_pv_scalar_no_history_falls_back_to_pv_now(self, wrapper, monkeypatch):
        """With no pv_power_history, pv_scalar should fall back to pv_now."""
        monkeypatch.setattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False)
        monkeypatch.setattr(config, "PV_TRAJ_ZERO_W", 50.0, raising=False)
        features = {
            "pv_now": 3500,
            "pv_forecast_1h": 3000.0,
        }
        result = wrapper._extract_thermal_features(features)
        assert result["pv_power"] == pytest.approx(3500.0, rel=0.01), (
            f"No history: should use pv_now=3500, got {result['pv_power']}"
        )

    def test_pv_scalar_no_forecast_snaps_to_pv_now(self, wrapper, monkeypatch):
        """When 1h forecast ≤ PV_TRAJ_ZERO_W, pv_scalar must equal pv_now
        regardless of the rolling-window history, and history must be cleared."""
        monkeypatch.setattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False)
        monkeypatch.setattr(config, "PV_TRAJ_ZERO_W", 50.0, raising=False)
        features = {
            "pv_power_history": [3000] * 18,  # stale high history
            "pv_now": 200,
            "pv_forecast_1h": 30.0,  # ≤ PV_TRAJ_ZERO_W → end-of-sun override
        }
        result = wrapper._extract_thermal_features(features)
        assert result["pv_power"] == pytest.approx(200.0, rel=0.01), (
            f"End-of-sun override: expected pv_now=200, got {result['pv_power']}"
        )
        # Rolling window passed into thermal_features must be cleared (None)
        assert result["pv_power_history"] is None, (
            "pv_power_history must be reset to None on end-of-sun override"
        )

    def test_pv_scalar_electrical_forecast_key_takes_precedence(
        self, wrapper, monkeypatch
    ):
        """pv_forecast_electrical_1h should be used for override check when
        present (matches raw-watts scale of pv_now_electrical)."""
        monkeypatch.setattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False)
        monkeypatch.setattr(config, "PV_TRAJ_ZERO_W", 50.0, raising=False)
        features = {
            "pv_power_history": [5000] * 18,   # stale high history
            "pv_now": 100,
            "pv_forecast_1h": 500.0,            # Would keep rolling window
            "pv_forecast_electrical_1h": 20.0,  # Overrides → end-of-sun
        }
        result = wrapper._extract_thermal_features(features)
        assert result["pv_power"] == pytest.approx(100.0, rel=0.01), (
            f"Electrical forecast key should trigger override, got {result['pv_power']}"
        )


# ---------------------------------------------------------------------------
# HP-off binary search: simulated HP-on delta_t
# ---------------------------------------------------------------------------

class TestHpOffSimulatedDeltaT:
    """When HP is off (delta_t < 1.0), the binary search should use the
    learned delta_t_floor (~2.55) so the trajectory simulates pump-on and
    candidates can differentiate.  Without this fix, all candidates
    produce identical predictions → 'unreachable' → outlet spikes to 35°C."""

    @pytest.fixture
    def wrapper(self, clean_state):
        """Wrapper with _current_features simulating HP-off."""
        w = get_enhanced_model_wrapper()
        w.cycle_count = 10
        w._avg_cloud_cover = 50.0
        w._cloud_cover_forecast = [50.0] * 6
        return w

    def test_hp_off_uses_simulated_delta_t(self, wrapper, monkeypatch):
        """With delta_t < 1.0, binary search should pass learned
        delta_t_floor (>= 1.0) so slab pump_on gate opens."""
        # Simulate HP-off features
        wrapper._current_features = {
            "delta_t": 0.2,
            "inlet_temp": 23.5,
            "pv_now": 0.0,
        }

        # Track what delta_t_floor is passed to predict_thermal_trajectory
        captured_dtf = []
        original_predict = wrapper.thermal_model.predict_thermal_trajectory

        def spy_predict(**kwargs):
            captured_dtf.append(kwargs.get("delta_t_floor"))
            return original_predict(**kwargs)

        monkeypatch.setattr(
            wrapper.thermal_model, "predict_thermal_trajectory", spy_predict
        )

        # The learned delta_t_floor should be >= 1.0
        resolved = wrapper.thermal_model._resolve_delta_t_floor(0.2)
        assert resolved >= 1.0, (
            f"Learned delta_t_floor should be >= 1.0, got {resolved}"
        )

        # Run binary search
        result = wrapper._calculate_required_outlet_temp(
            current_indoor=22.8,
            target_indoor=22.6,
            outdoor_temp=5.0,
            thermal_features={
                "pv_power": 0.0,
                "fireplace_on": 0.0,
                "tv_on": 0.0,
                "thermal_power": 0.0,
                "indoor_temp_gradient": 0.0,
                "temp_diff_indoor_outdoor": 17.8,
                "outlet_indoor_diff": 0.7,
            },
        )

        # All trajectory calls should have used the simulated delta_t
        assert len(captured_dtf) > 0, "Binary search should call trajectory"
        for dtf in captured_dtf:
            assert dtf >= 1.0, (
                f"Expected simulated delta_t >= 1.0, got {dtf}"
            )

        # Result should NOT be max outlet (35°C)
        assert result < config.CLAMP_MAX_ABS, (
            f"HP-off should not spike to max outlet {config.CLAMP_MAX_ABS}, "
            f"got {result}"
        )

    def test_hp_on_uses_real_delta_t(self, wrapper, monkeypatch):
        """With delta_t >= 1.0, binary search should use the real value."""
        wrapper._current_features = {
            "delta_t": 3.5,
            "inlet_temp": 30.0,
            "pv_now": 0.0,
        }

        captured_dtf = []
        original_predict = wrapper.thermal_model.predict_thermal_trajectory

        def spy_predict(**kwargs):
            captured_dtf.append(kwargs.get("delta_t_floor"))
            return original_predict(**kwargs)

        monkeypatch.setattr(
            wrapper.thermal_model, "predict_thermal_trajectory", spy_predict
        )

        wrapper._calculate_required_outlet_temp(
            current_indoor=22.0,
            target_indoor=22.6,
            outdoor_temp=5.0,
            thermal_features={
                "pv_power": 0.0,
                "fireplace_on": 0.0,
                "tv_on": 0.0,
                "thermal_power": 2.0,
                "indoor_temp_gradient": 0.0,
                "temp_diff_indoor_outdoor": 17.0,
                "outlet_indoor_diff": 5.0,
            },
        )

        assert len(captured_dtf) > 0, "Binary search should call trajectory"
        for dtf in captured_dtf:
            assert dtf == pytest.approx(3.5, abs=0.01), (
                f"HP-on should use real delta_t=3.5, got {dtf}"
            )

    def test_slab_passive_delta_in_features(self, wrapper):
        """slab_passive_delta should appear in extracted thermal features."""
        w = model_wrapper.EnhancedModelWrapper.__new__(
            model_wrapper.EnhancedModelWrapper
        )
        w._avg_cloud_cover = 50.0
        w._cloud_cover_forecast = [50.0] * 6
        features = {
            "pv_power_history": [0] * 5,
            "pv_now": 0.0,
            "inlet_temp": 24.0,
            "indoor_temp_lag_30m": 22.5,
        }
        result = w._extract_thermal_features(features)
        assert "slab_passive_delta" in result, (
            "slab_passive_delta should be in thermal features"
        )
        assert result["slab_passive_delta"] == pytest.approx(1.5, abs=0.01), (
            f"Expected 24.0 - 22.5 = 1.5, got {result['slab_passive_delta']}"
        )


# ---------------------------------------------------------------------------
# Projected-temp overshoot correction gate
# ---------------------------------------------------------------------------

class TestProjectedTempOvershootGate:
    """Overshoot correction should be skipped when the projected indoor
    temperature falls below target + 0.1°C, meaning the house will
    self-correct in time.

    Projection uses the exponential-decay integral that matches the
    trajectory model:
        projected = current + trend * tau * (1 - exp(-H / tau))
    where tau = TREND_DECAY_TAU_HOURS (default 1.5 h) and H = TRAJECTORY_STEPS.

    At H=4 h, tau=1.5 h the integral factor ≈ 1.396, so a trend of
    -0.3 °C/h yields projected ≈ 22.5 - 0.419 = 22.08 °C < 22.1 °C → skip.
    """

    @pytest.fixture
    def wrapper(self, clean_state):
        w = get_enhanced_model_wrapper()
        w._current_indoor = 22.5
        w._current_features = {
            # -0.3 °C/h: strong enough for exp-decay integral to push
            # projected below target+0.1 (22.08 < 22.1) at H=4 h, τ=1.5 h.
            "indoor_temp_delta_60m": -0.3,
        }
        return w

    def _make_trajectory(self, temps, reaches_target_at=None):
        return {
            "trajectory": temps,
            "times": [i * 0.5 for i in range(1, len(temps) + 1)],
            "reaches_target_at": reaches_target_at,
        }

    def test_skip_overshoot_when_projection_below_target(self, wrapper, monkeypatch):
        """Overshoot correction skipped: exp-decay projected ≈ 22.08 < 22.1."""
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 4)
        monkeypatch.setattr(config, "TREND_DECAY_TAU_HOURS", 1.5)
        trajectory = self._make_trajectory([22.3, 22.2, 22.1, 22.0])

        result = wrapper._calculate_physics_based_correction(
            outlet_temp=28.0,
            trajectory=trajectory,
            target_indoor=22.0,
            cycle_hours=0.167,
        )
        # Should return outlet_temp unchanged (correction skipped)
        assert result == 28.0

    def test_apply_overshoot_when_projection_above_target(self, wrapper, monkeypatch):
        """Overshoot correction applied: trend +0.1 °C/h → projected > 22.1."""
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 4)
        monkeypatch.setattr(config, "TREND_DECAY_TAU_HOURS", 1.5)
        wrapper._current_features = {
            "indoor_temp_delta_60m": 0.1,  # still rising
        }
        trajectory = self._make_trajectory([22.3, 22.4, 22.5, 22.6])

        result = wrapper._calculate_physics_based_correction(
            outlet_temp=28.0,
            trajectory=trajectory,
            target_indoor=22.0,
            cycle_hours=0.167,
        )
        # Should apply correction (outlet lowered)
        assert result < 28.0

    def test_skip_with_zero_trend_but_small_overshoot(self, wrapper, monkeypatch):
        """Zero trend (flat) but current indoor 22.05 — projected = 22.05 < 22.1."""
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 4)
        monkeypatch.setattr(config, "TREND_DECAY_TAU_HOURS", 1.5)
        wrapper._current_indoor = 22.05
        wrapper._current_features = {
            "indoor_temp_delta_60m": 0.0,  # flat
        }
        trajectory = self._make_trajectory([22.15, 22.12, 22.10, 22.05])

        result = wrapper._calculate_physics_based_correction(
            outlet_temp=28.0,
            trajectory=trajectory,
            target_indoor=22.0,
            cycle_hours=0.167,
        )
        # projected = 22.05 + 0 = 22.05 < 22.1 → skip
        assert result == 28.0

    def test_both_violated_max_wins_skips_when_projected_below(self, wrapper, monkeypatch):
        """Both min and max violated, max wins — still skips if projected < target+0.1."""
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 4)
        monkeypatch.setattr(config, "TREND_DECAY_TAU_HOURS", 1.5)
        # Trajectory goes both below and above target
        trajectory = self._make_trajectory([21.8, 22.3, 22.2, 21.9])

        result = wrapper._calculate_physics_based_correction(
            outlet_temp=28.0,
            trajectory=trajectory,
            target_indoor=22.0,
            cycle_hours=0.167,
        )
        # max violation (22.3) > min violation (21.8 vs 21.9 boundary)
        # max_severity = 22.3 - 22.1 = 0.2, min_severity = 21.9 - 21.8 = 0.1
        # max wins → exp-decay projected ≈ 22.08 < 22.1 → skip
        assert result == 28.0


# ---------------------------------------------------------------------------
# Drift detection regression coverage
# ---------------------------------------------------------------------------

class TestPredictionDriftDetection:
    """Regression tests for drift-triggered confidence handling."""

    def test_drift_reset_restores_normal_confidence_cap(self):
        wrapper = model_wrapper.EnhancedModelWrapper.__new__(
            model_wrapper.EnhancedModelWrapper
        )
        wrapper.prediction_metrics = MagicMock()
        wrapper.thermal_model = MagicMock()
        wrapper.state_manager = MagicMock()
        wrapper._drift_counter = 12

        wrapper.prediction_metrics.get_metrics.return_value = {
            "1h": {"mae": 0.6},
            "all": {"mae": 0.5},
        }
        wrapper.thermal_model.learning_confidence = 7.5
        wrapper.thermal_model._max_learning_confidence = 10.0

        wrapper._check_prediction_drift()

        assert wrapper._drift_counter == 0
        assert wrapper.thermal_model._max_learning_confidence == 5.0
        assert wrapper.thermal_model.learning_confidence == 5.0
        wrapper.state_manager.update_learning_state.assert_called_once_with(
            learning_confidence=5.0
        )


# ---------------------------------------------------------------------------
# PV surplus CHEAP soft ramp (Item 2)
# ---------------------------------------------------------------------------

class TestPvSurplusCheapRamp:
    """PV surplus CHEAP offset ramps linearly in the blend zone.

    Tests call calculate_optimal_outlet_temp() so any regression in the
    production ramp path causes a test failure — unlike a helper that
    re-implements the same arithmetic.
    """

    @pytest.fixture
    def wrapper(self, clean_state):
        return get_enhanced_model_wrapper()

    def _features(self, pv_now_elec, target=20.0, indoor=20.0):
        return {
            "indoor_temp_lag_30m": indoor,
            "target_temp": target,
            "outdoor_temp": 10.0,
            "pv_now_electrical": float(pv_now_elec),
            "pv_now": 0.0,
            "pv_power_history": [0.0] * 5,
            # Keep above PV_TRAJ_ZERO_W to avoid end-of-sun snap
            "pv_forecast_1h": 2000.0,
        }

    def _call(self, wrapper, monkeypatch, pv_now_elec,
              threshold=1000, ramp_w=1000, cheap_offset=0.5, target=20.0):
        monkeypatch.setattr(config, "PV_SURPLUS_CHEAP_ENABLED", True,
                            raising=False)
        monkeypatch.setattr(config, "ELECTRICITY_PRICE_ENABLED", False,
                            raising=False)
        monkeypatch.setattr(config, "PV_SURPLUS_CHEAP_THRESHOLD_W", threshold,
                            raising=False)
        monkeypatch.setattr(config, "PV_SURPLUS_CHEAP_RAMP_W", float(ramp_w),
                            raising=False)
        monkeypatch.setattr(config, "PRICE_TARGET_OFFSET", cheap_offset,
                            raising=False)
        _, meta = wrapper.calculate_optimal_outlet_temp(
            self._features(pv_now_elec, target=target)
        )
        return meta

    def test_full_offset_at_threshold(self, wrapper, monkeypatch):
        """At threshold, full cheap_offset is applied."""
        meta = self._call(wrapper, monkeypatch, pv_now_elec=1000,
                          threshold=1000, cheap_offset=0.5)
        assert meta.get("price_level") == "cheap"
        assert meta.get("price_target_offset") == pytest.approx(0.5, abs=0.001)
        assert meta.get("target_temp_adjusted") == pytest.approx(20.5, abs=0.001)

    def test_full_offset_well_above_threshold(self, wrapper, monkeypatch):
        """Well above threshold, full cheap_offset still applied."""
        meta = self._call(wrapper, monkeypatch, pv_now_elec=3000,
                          threshold=1000, cheap_offset=0.5)
        assert meta.get("price_level") == "cheap"
        assert meta.get("price_target_offset") == pytest.approx(0.5, abs=0.001)

    def test_partial_offset_midpoint(self, wrapper, monkeypatch):
        """At the midpoint of the ramp zone, offset ≈ 50% of max."""
        # threshold=1000, ramp_w=1000 → floor=0; midpoint pv_now=500
        meta = self._call(wrapper, monkeypatch, pv_now_elec=500,
                          threshold=1000, ramp_w=1000, cheap_offset=0.4)
        assert meta.get("price_level") == "cheap"
        assert meta.get("price_target_offset") == pytest.approx(0.2, abs=0.005)
        assert meta.get("target_temp_adjusted") == pytest.approx(20.2, abs=0.005)

    def test_partial_offset_three_quarters(self, wrapper, monkeypatch):
        """At 75% of ramp zone, offset ≈ 75% of max."""
        # pv_now=750 → partial = 0.4 * 750/1000 = 0.3
        meta = self._call(wrapper, monkeypatch, pv_now_elec=750,
                          threshold=1000, ramp_w=1000, cheap_offset=0.4)
        assert meta.get("price_level") == "cheap"
        assert meta.get("price_target_offset") == pytest.approx(0.3, abs=0.005)

    def test_zero_offset_below_ramp_floor(self, wrapper, monkeypatch):
        """Below the ramp floor (narrow ramp_w), offset is 0."""
        # threshold=1000, ramp_w=200 → floor=800; pv_now=700 < floor → 0
        meta = self._call(wrapper, monkeypatch, pv_now_elec=700,
                          threshold=1000, ramp_w=200, cheap_offset=0.5)
        assert meta.get("price_target_offset", 0.0) == pytest.approx(0.0,
                                                                      abs=0.001)
        assert meta.get("target_temp_adjusted") == pytest.approx(20.0, abs=0.001)

    def test_ramp_w_larger_than_threshold_clamped(self, wrapper, monkeypatch):
        """ramp_w > threshold is clamped to threshold so floor stays at 0."""
        # ramp_w=2000 > threshold=1000 → clamped to 1000 → floor=0
        # pv_now=100: partial = 1.0 * (100 - 0) / 1000 = 0.1
        meta = self._call(wrapper, monkeypatch, pv_now_elec=100,
                          threshold=1000, ramp_w=2000, cheap_offset=1.0)
        assert meta.get("price_target_offset") == pytest.approx(0.1, abs=0.005)

    def test_disabled_no_offset(self, wrapper, monkeypatch):
        """When PV_SURPLUS_CHEAP_ENABLED=False, no offset is applied."""
        monkeypatch.setattr(config, "PV_SURPLUS_CHEAP_ENABLED", False,
                            raising=False)
        monkeypatch.setattr(config, "ELECTRICITY_PRICE_ENABLED", False,
                            raising=False)
        _, meta = wrapper.calculate_optimal_outlet_temp(
            self._features(pv_now_elec=5000)
        )
        assert meta.get("price_target_offset", 0.0) == pytest.approx(0.0,
                                                                      abs=0.001)
        assert meta.get("target_temp_adjusted") == pytest.approx(20.0, abs=0.001)


# ---------------------------------------------------------------------------
# Overshoot dampening constant (Item 3)
# ---------------------------------------------------------------------------

class TestOvershootDampening:
    """The overshoot branch of _calculate_physics_based_correction uses
    1.0 / max(slab_tau, 1.0) — 2.5× stronger than the old 0.4 constant.

    Tests call the real correction path with controlled thermal-model state so
    a regression back to 0.4 would break them.
    """

    # Known thermal-model values for deterministic calculation:
    #   slab_tau      = 2.0
    #   physics_scale = min((1/0.5)*1.1, 2.5) = 2.2
    #   trajectory    = [22.3, 22.4, 22.5, 22.6], reaches_target_at=None
    #     → max_predicted = 22.6, temp_error = 22.0-22.6 = -0.6
    #     → time_pressure = 1.0 (reaches_target_at is None)
    #     → urgency_multiplier = 1.0 + 2.0*1.0² = 3.0
    #   new overshoot_dampening = 1.0/2.0 = 0.5
    #   new correction = -0.6 * 2.2 * 3.0 * 0.5 = -1.98
    #   old correction = -0.6 * 2.2 * 3.0 * 0.2 = -0.792  (0.4/2.0=0.2)
    _OUTLET_TEMP = 30.0
    _TARGET = 22.0
    _EXPECTED_CORRECTION_NEW = -0.6 * 2.2 * 3.0 * 0.5   # -1.98
    _EXPECTED_CORRECTION_OLD = -0.6 * 2.2 * 3.0 * 0.2   # -0.792

    @pytest.fixture
    def wrapper(self, clean_state):
        w = get_enhanced_model_wrapper()
        w.thermal_model.slab_time_constant_hours = 2.0
        w.thermal_model.outlet_effectiveness = 0.5
        w._current_indoor = 22.5          # current > target → overshoot scenario
        w._current_features = {"indoor_temp_delta_60m": 0.1}
        return w

    def _make_trajectory(self, temps, reaches_target_at=None):
        return {
            "trajectory": temps,
            "times": [i * 0.25 for i in range(1, len(temps) + 1)],
            "reaches_target_at": reaches_target_at,
        }

    def test_overshoot_correction_applied(self, wrapper, monkeypatch):
        """Overshoot path pulls outlet down (result < outlet_temp)."""
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 4)
        result = wrapper._calculate_physics_based_correction(
            outlet_temp=self._OUTLET_TEMP,
            trajectory=self._make_trajectory([22.3, 22.4, 22.5, 22.6]),
            target_indoor=self._TARGET,
            cycle_hours=0.25,
        )
        assert result < self._OUTLET_TEMP, (
            "Overshoot correction must lower the outlet setpoint"
        )

    def test_correction_magnitude_matches_1_0_dampening(
        self, wrapper, monkeypatch
    ):
        """Correction matches 1.0/slab_tau dampening, not the old 0.4/slab_tau."""
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 4)
        result = wrapper._calculate_physics_based_correction(
            outlet_temp=self._OUTLET_TEMP,
            trajectory=self._make_trajectory([22.3, 22.4, 22.5, 22.6]),
            target_indoor=self._TARGET,
            cycle_hours=0.25,
        )
        expected_new = self._OUTLET_TEMP + self._EXPECTED_CORRECTION_NEW
        expected_old = self._OUTLET_TEMP + self._EXPECTED_CORRECTION_OLD
        # Result should be close to the 1.0-dampening expectation
        assert result == pytest.approx(expected_new, abs=0.05), (
            f"Expected correction consistent with 1.0/slab_tau "
            f"(outlet≈{expected_new:.2f}°C), got {result:.2f}°C"
        )
        # And clearly different from the old 0.4-dampening expectation
        assert abs(result - expected_old) > 0.5, (
            f"Result {result:.2f}°C is too close to old 0.4-dampening "
            f"value {expected_old:.2f}°C — constant may not have changed"
        )
