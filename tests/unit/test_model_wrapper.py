
import pytest
import pandas as pd
import os

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

    def test_simplified_prediction(self, wrapper_instance, mocker):
        """Test the simplified_outlet_prediction function."""
        mocker.patch.object(
            wrapper_instance, 'calculate_optimal_outlet_temp',
            return_value=(35.0, {'learning_confidence': 4.5})
        )

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
            'pv_now': 2500.0,
            'fireplace_on': 0,
            'tv_on': 1,
        }

        optimal_temp, metadata = (
            wrapper_instance.calculate_optimal_outlet_temp(test_features)
        )

        assert isinstance(optimal_temp, float)
        assert optimal_temp > 21.0  # Should be higher than indoor temp
        assert 'learning_confidence' in metadata
        assert metadata['prediction_method'] == \
            'thermal_equilibrium_single_prediction'

    def test_learning_feedback(self, wrapper_instance, mocker):
        """Test the learn_from_prediction_feedback method works."""
        # Mock dependencies
        mocker.patch('src.model_wrapper.create_influx_service')
        mocker.patch('src.model_wrapper.create_ha_client')

        # Set cycle count to 2 to avoid first-cycle skip
        wrapper_instance.cycle_count = 2

        wrapper_instance.learn_from_prediction_feedback(
            predicted_temp=35.0,
            actual_temp=34.2,
            prediction_context={'indoor_temp': 20.5, 'outdoor_temp': 5.0}
        )

        assert wrapper_instance.cycle_count == 3
        # Further assertions would require mocking the thermal model's
        # internal state

    def test_first_cycle_learning_skip(self, wrapper_instance, mocker):
        """Verify that online learning is skipped on the first cycle."""
        # Mock logger to check for the specific log message
        mock_log_info = mocker.patch('logging.info')

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
        self, clean_state, monkeypatch, mocker
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
        mocked_trajectory = mocker.patch.object(
            wrapper.thermal_model,
            "predict_thermal_trajectory",
            return_value={"trajectory": [21.5]},
        )

        wrapper.predict_indoor_temp(
            outlet_temp=40.0,
            outdoor_temp=5.0,
            current_indoor=20.0,
            fireplace_on=1,
        )

        assert mocked_trajectory.call_args.kwargs["fireplace_power_kw"] == pytest.approx(7.5)

    def test_legacy_fireplace_mode_uses_adaptive_learning(
        self, clean_state, monkeypatch, mocker
    ):
        """Flag-off mode keeps adaptive fireplace learning as legacy behavior."""
        mock_learner = mocker.Mock()
        mock_learner._calculate_learned_heat_contribution.return_value = {
            "heat_contribution_kw": 6.2,
            "learning_confidence": 0.9,
        }

        monkeypatch.setattr(config, "ENABLE_HEAT_SOURCE_CHANNELS", False)
        mocker.patch("src.model_wrapper.AdaptiveFireplaceLearning", return_value=mock_learner)
        model_wrapper._enhanced_model_wrapper_instance = None

        wrapper = get_enhanced_model_wrapper()
        mocked_trajectory = mocker.patch.object(
            wrapper.thermal_model,
            "predict_thermal_trajectory",
            return_value={"trajectory": [21.5]},
        )

        wrapper.predict_indoor_temp(
            outlet_temp=40.0,
            outdoor_temp=5.0,
            current_indoor=20.0,
            fireplace_on=1,
        )

        mock_learner._calculate_learned_heat_contribution.assert_called_once()
        assert mocked_trajectory.call_args.kwargs["fireplace_power_kw"] == pytest.approx(6.2)

    def test_fireplace_learning_integration(self, clean_state, monkeypatch, mocker):
        """Legacy flag-off mode still observes adaptive fireplace sessions."""
        monkeypatch.setattr(config, "ENABLE_HEAT_SOURCE_CHANNELS", False)
        model_wrapper._enhanced_model_wrapper_instance = None

        wrapper = get_enhanced_model_wrapper()
        mocker.patch.object(wrapper, "_export_metrics_to_influxdb")
        mocker.patch.object(wrapper, "export_metrics_to_ha")
        mock_learner = mocker.Mock()
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

        wrapper.learn_from_prediction_feedback(
            predicted_temp=21.0,
            actual_temp=22.0,
            prediction_context=context,
        )

        mock_learner.observe_fireplace_state.assert_called_once()
        _, kwargs = mock_learner.observe_fireplace_state.call_args
        assert kwargs['living_room_temp'] == 20.0
