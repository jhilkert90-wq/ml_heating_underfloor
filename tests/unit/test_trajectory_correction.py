"""
Unit tests for bidirectional trajectory correction enhancement.

Tests the enhanced PRIORITY 3 logic that detects both temperature drops 
and rises using sensor-aligned 0.1°C boundaries.
"""
import pytest
from unittest.mock import patch

from src.model_wrapper import get_enhanced_model_wrapper


class TestBidirectionalTrajectoryCorrection:
    """Test enhanced PRIORITY 3 bidirectional trajectory correction."""
    
    def setup_method(self):
        """Set up test environment."""
        self._clamp_min_patcher = patch(
            'src.model_wrapper.config.CLAMP_MIN_ABS', 20.0
        )
        self._clamp_max_patcher = patch(
            'src.model_wrapper.config.CLAMP_MAX_ABS', 55.0
        )
        self._clamp_min_patcher.start()
        self._clamp_max_patcher.start()
        self.wrapper = get_enhanced_model_wrapper()
        for attr in ('_current_indoor', '_current_features'):
            if hasattr(self.wrapper, attr):
                delattr(self.wrapper, attr)
        self.base_thermal_features = {
            'pv_power': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }

    def teardown_method(self):
        self._clamp_max_patcher.stop()
        self._clamp_min_patcher.stop()
    
    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_temperature_drop_detection_boundary_exact(self):
        """Test detection of temperature drop at exact boundary (20.9°C)."""
        # Set negative trend so undershoot gate does NOT skip correction
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}
        # Mock trajectory with temperature dropping to exactly 20.9°C
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 20.95, 20.9, 20.9]  # min = 20.9
        }
        
        with patch.object(
            self.wrapper.thermal_model, 
            'predict_thermal_trajectory', 
            return_value=mock_trajectory
        ):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should trigger correction since 20.9 <= 21.0 - 0.1
        assert result > 35.0, "Should apply correction for temp drop to boundary"
    
    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_temperature_drop_detection_below_boundary(self):
        """Test detection of temperature drop below boundary (20.8°C)."""
        # Set negative trend so undershoot gate does NOT skip correction
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}
        # Mock trajectory with temperature dropping below boundary
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 20.9, 20.8, 20.8]  # min = 20.8
        }
        
        with patch.object(
            self.wrapper.thermal_model, 
            'predict_thermal_trajectory', 
            return_value=mock_trajectory
        ):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should trigger correction since 20.8 <= 21.0 - 0.1
        assert result > 35.0, "Should apply correction for temp drop below boundary"
    
    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_temperature_rise_detection_boundary_exact(self):
        """Test detection of temperature rise at exact boundary (21.1°C)."""
        # Set positive trend so projected-temp gate does NOT skip overshoot
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': 0.1}
        # Mock trajectory with temperature rising to exactly 21.1°C
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 21.05, 21.1, 21.1]  # max = 21.1
        }
        
        with patch.object(
            self.wrapper.thermal_model, 
            'predict_thermal_trajectory', 
            return_value=mock_trajectory
        ):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should trigger correction since 21.1 >= 21.0 + 0.1
        assert result < 35.0, "Should apply -ve correction for temp rise to boundary"
    
    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_temperature_rise_detection_above_boundary(self):
        """Test detection of temperature rise above boundary (21.2°C)."""
        # Set positive trend so projected-temp gate does NOT skip overshoot
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': 0.1}
        # Mock trajectory with temperature rising above boundary
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 21.1, 21.2, 21.2]  # max = 21.2
        }
        
        with patch.object(
            self.wrapper.thermal_model, 
            'predict_thermal_trajectory', 
            return_value=mock_trajectory
        ):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should trigger correction since 21.2 >= 21.0 + 0.1
        assert result < 35.0, "Should apply -ve correction for temp rise above boundary"
    
    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_no_correction_within_boundaries(self):
        """Test no correction when temperature stays within boundaries."""
        # Mock trajectory with temperature staying within ±0.1°C
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 20.95, 21.05, 21.0]  # min=20.95, max=21.05
        }
        
        with patch.object(
            self.wrapper.thermal_model, 
            'predict_thermal_trajectory', 
            return_value=mock_trajectory
        ):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should NOT trigger correction since within boundaries
        assert result == 35.0, "Should not apply correction within boundaries"
    
    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_cycle_aligned_trajectory_checking_with_violations(self):
        """Test cycle-time aligned logic checks for violations."""
        # Set negative trend so undershoot gate does NOT skip correction
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}
        # Mock trajectory that reaches target but has boundary violations
        mock_trajectory = {
            'reaches_target_at': 0.8,  # Target reached at 0.8 hours
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 20.8, 20.9, 21.0]  # Contains boundary violation (20.8 <= 20.9)
        }

        with patch.object(
            self.wrapper.thermal_model, 
            'predict_thermal_trajectory', 
            return_value=mock_trajectory
        ):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )

        assert result > 35.0, "Should apply correction for violations"
    
    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_priority_1_does_not_override_when_too_slow(self):
        """Test PRIORITY 1 does not override when target is reached too slowly."""
        # Set negative trend so undershoot gate does NOT skip correction
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}
        # Mock trajectory that reaches target slowly (should NOT override boundary checks)
        mock_trajectory = {
            'reaches_target_at': 2.5,  # Target reached at 2.5 hours (too slow)
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 20.8, 20.8, 20.8]  # Contains boundary violation
        }
        
        with patch.object(
            self.wrapper.thermal_model, 
            'predict_thermal_trajectory', 
            return_value=mock_trajectory
        ):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should apply correction because target reached too slowly
        assert result > 35.0, "Should apply correction when target is reached too slowly"
    
    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_trajectory_correction_handles_boundary_violations(self):
        """Test trajectory correction handles boundary violations."""
        # Set negative trend so undershoot gate does NOT skip correction
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}
        # Mock trajectory with boundary violations (temperature drops and rises)
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 20.5,  # Below target
            'trajectory': [21.0, 21.2, 21.1, 20.5]  # Has rise and drop violations
        }
        
        with patch.object(
            self.wrapper.thermal_model, 
            'predict_thermal_trajectory', 
            return_value=mock_trajectory
        ):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should apply trajectory correction for boundary violations
        # The correction is bounded by physics_correction limits (max +20°C)
        # With minimum meaningful correction of 1.0°C, result should be 36.0°C
        assert result > 35.0, "Should apply +ve correction for boundary violations"
        assert result <= 55.0, "Correction should be bounded by max physics correction"
    
    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_internal_precision_with_sensor_boundaries(self):
        """Test internal precision values with sensor-aligned boundaries."""
        # Set negative trend so undershoot gate does NOT skip correction
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}
        # Mock trajectory with high-precision internal values
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 20.876, 20.923, 20.987]  # min = 20.876 (high precision)
        }
        
        with patch.object(
            self.wrapper.thermal_model, 
            'predict_thermal_trajectory', 
            return_value=mock_trajectory
        ):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )
            
        # Should trigger correction since 20.876 <= 20.9 (21.0 - 0.1)
        assert result > 35.0, "Should handle internal precision with sensor boundaries"
    
    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_correction_reason_messages(self):
        """Test correct reason messages for different scenarios."""
        # Test temperature drop reason
        mock_trajectory_drop = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 20.8]  # Drop scenario
        }
        
        # Test temperature rise reason  
        mock_trajectory_rise = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.0, 21.2]  # Rise scenario
        }
        
        # We can't easily test logging output, but we can verify the method executes
        # without errors for both scenarios
        # Set negative trend so undershoot gate does NOT skip drop correction
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}
        with patch.object(
            self.wrapper.thermal_model, 
            'predict_thermal_trajectory', 
            return_value=mock_trajectory_drop
        ):
            result_drop = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0, current_indoor=21.0, target_indoor=21.0,
                outdoor_temp=5.0, thermal_features=self.base_thermal_features
            )
            
        # Set positive trend so overshoot gate does NOT skip rise correction
        self.wrapper._current_features = {'indoor_temp_delta_60m': 0.1}
        with patch.object(
            self.wrapper.thermal_model, 
            'predict_thermal_trajectory', 
            return_value=mock_trajectory_rise
        ):
            result_rise = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0, current_indoor=21.0, target_indoor=21.0,
                outdoor_temp=5.0, thermal_features=self.base_thermal_features
            )
            
        assert result_drop > 35.0, "Drop scenario should increase outlet temp"
        assert result_rise < 35.0, "Rise scenario should decrease outlet temp"
    
    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_different_target_temperatures(self):
        """Test boundary logic with different target temps."""
        test_cases = [
            (20.0, 19.8, True),   # Target 20.0°C, min 19.8°C → correction
            (20.0, 19.9, True),   # Target 20.0°C, min 19.9°C → correction (boundary)
            (20.0, 20.0, False),  # Target 20.0°C, min 20.0°C → no correction
            (22.5, 22.3, True),   # Target 22.5°C, min 22.3°C → correction
            (22.5, 22.4, True),   # Target 22.5°C, min 22.4°C → correction (boundary)
            (22.5, 22.5, False),  # Target 22.5°C, min 22.5°C → no correction
        ]
        
        for target, min_temp, should_correct in test_cases:
            # Set negative trend so undershoot gate does NOT skip correction
            self.wrapper._current_indoor = target
            self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}
            mock_trajectory = {
                'reaches_target_at': None,
                'equilibrium_temp': target,
                'trajectory': [target, min_temp]
            }
            
            with patch.object(
                self.wrapper.thermal_model, 
                'predict_thermal_trajectory', 
                return_value=mock_trajectory
            ):
                result = self.wrapper._verify_trajectory_and_correct(
                    outlet_temp=35.0,
                    current_indoor=target,
                    target_indoor=target,
                    outdoor_temp=5.0,
                    thermal_features=self.base_thermal_features
                )
                
            if should_correct:
                assert result != 35.0, f"Should correct for target={target}, min={min_temp}"
            else:
                assert result == 35.0, f"Should not correct for target={target}, min={min_temp}"


class TestUndershootGate:
    """Test undershoot gate: skip undershoot correction when indoor is rising
    and the house will self-correct (mirror of the existing overshoot gate)."""

    def setup_method(self):
        """Set up test environment."""
        self._clamp_min_patcher = patch(
            'src.model_wrapper.config.CLAMP_MIN_ABS', 20.0
        )
        self._clamp_max_patcher = patch(
            'src.model_wrapper.config.CLAMP_MAX_ABS', 55.0
        )
        self._clamp_min_patcher.start()
        self._clamp_max_patcher.start()
        self.wrapper = get_enhanced_model_wrapper()
        for attr in ('_current_indoor', '_current_features'):
            if hasattr(self.wrapper, attr):
                delattr(self.wrapper, attr)
        self.base_thermal_features = {
            'pv_power': 0.0,
            'fireplace_on': 0.0,
            'tv_on': 0.0
        }

    def teardown_method(self):
        self._clamp_max_patcher.stop()
        self._clamp_min_patcher.stop()

    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_undershoot_skipped_when_indoor_rising(self):
        """Undershoot correction is skipped when indoor temp is rising
        and the projected indoor will self-correct above target - 0.1°C."""
        # Indoor rising at +0.2°C/h → projected = 21.0 + 4*0.2 = 21.8
        # target - 0.1 = 20.9 → projected 21.8 > 20.9 → skip
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': 0.2}

        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [20.85, 20.8, 20.85, 20.9]  # min = 20.8 < 20.9
        }

        with patch.object(
            self.wrapper.thermal_model,
            'predict_thermal_trajectory',
            return_value=mock_trajectory
        ):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )

        assert result == 35.0, (
            "Should skip undershoot correction when house is self-correcting "
            "(indoor rising)"
        )

    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_undershoot_applied_when_indoor_falling(self):
        """Undershoot correction is NOT skipped when indoor temp is falling
        (house is not self-correcting)."""
        # Indoor falling at -0.1°C/h → projected = 21.0 + 4*(-0.1) = 20.6
        # target - 0.1 = 20.9 → projected 20.6 < 20.9 → apply correction
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}

        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [20.85, 20.8, 20.75, 20.7]  # min = 20.7 < 20.9
        }

        with patch.object(
            self.wrapper.thermal_model,
            'predict_thermal_trajectory',
            return_value=mock_trajectory
        ):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )

        assert result > 35.0, (
            "Should apply undershoot correction when indoor is falling "
            "(not self-correcting)"
        )

    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_undershoot_applied_when_rising_but_projected_still_low(self):
        """Undershoot correction is NOT skipped when indoor is rising but the
        projected temperature is still below the gate threshold."""
        # Indoor rising very slowly at +0.01°C/h → projected = 21.0 + 4*0.01 = 21.04
        # target - 0.1 = 20.9 → projected 21.04 > 20.9 → skip
        # BUT let's test the case where projected is still below threshold:
        # current_indoor = 20.5, trend = +0.05 → projected = 20.5 + 4*0.05 = 20.7
        # target = 21.0, target - 0.1 = 20.9 → projected 20.7 < 20.9 → apply
        self.wrapper._current_indoor = 20.5
        self.wrapper._current_features = {'indoor_temp_delta_60m': 0.05}

        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 20.5,
            'trajectory': [20.4, 20.3, 20.35, 20.4]  # min = 20.3 < 20.9
        }

        with patch.object(
            self.wrapper.thermal_model,
            'predict_thermal_trajectory',
            return_value=mock_trajectory
        ):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=20.5,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )

        assert result > 35.0, (
            "Should apply undershoot correction when rising trend is too weak "
            "to self-correct"
        )

    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_undershoot_gate_both_violated_min_wins_rising(self):
        """When both boundaries are violated and undershoot is more severe,
        the undershoot gate skips correction if indoor is rising enough."""
        # Indoor rising at +0.15°C/h → projected = 21.0 + 4*0.15 = 21.6
        # target - 0.1 = 20.9 → projected 21.6 > 20.9 → skip
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': 0.15}

        # Both boundaries violated: min=20.7 (severity 0.2) > max=21.15 (severity 0.05)
        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [20.7, 21.15, 20.8, 20.9]  # min=20.7, max=21.15
        }

        with patch.object(
            self.wrapper.thermal_model,
            'predict_thermal_trajectory',
            return_value=mock_trajectory
        ):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )

        assert result == 35.0, (
            "Should skip undershoot correction when both violated, min wins, "
            "and indoor is rising"
        )

    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_overshoot_gate_still_works_with_falling_indoor(self):
        """Verify the existing overshoot gate still works correctly:
        overshoot correction is skipped when indoor is falling."""
        # Indoor falling at -0.1°C/h → projected = 21.0 + 4*(-0.1) = 20.6
        # target + 0.1 = 21.1 → projected 20.6 < 21.1 → skip overshoot
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}

        mock_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.15, 21.2, 21.15, 21.1]  # max = 21.2 > 21.1
        }

        with patch.object(
            self.wrapper.thermal_model,
            'predict_thermal_trajectory',
            return_value=mock_trajectory
        ):
            result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )

        assert result == 35.0, (
            "Existing overshoot gate should still skip correction when "
            "indoor is falling"
        )

    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_undershoot_gate_symmetry_with_overshoot_gate(self):
        """The undershoot and overshoot gates should behave symmetrically:
        both skip correction when the house trend will self-correct."""
        # --- Undershoot scenario: indoor rising ---
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': 0.2}

        undershoot_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [20.85, 20.8, 20.85, 20.9]
        }

        with patch.object(
            self.wrapper.thermal_model,
            'predict_thermal_trajectory',
            return_value=undershoot_trajectory
        ):
            undershoot_result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )

        # --- Overshoot scenario: indoor falling ---
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.2}

        overshoot_trajectory = {
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
            'trajectory': [21.15, 21.2, 21.15, 21.1]
        }

        with patch.object(
            self.wrapper.thermal_model,
            'predict_thermal_trajectory',
            return_value=overshoot_trajectory
        ):
            overshoot_result = self.wrapper._verify_trajectory_and_correct(
                outlet_temp=35.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=5.0,
                thermal_features=self.base_thermal_features
            )

        # Both should be skipped (return original outlet_temp)
        assert undershoot_result == 35.0, "Undershoot gate should skip"
        assert overshoot_result == 35.0, "Overshoot gate should skip"


class TestBidirectionalCorrectionIntegration:
    """Integration tests for bidirectional correction in full control flow."""
    
    def setup_method(self):
        """Set up test environment."""
        self.wrapper = get_enhanced_model_wrapper()
    
    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    def test_full_outlet_calculation_with_trajectory_correction(self):
        """Test full outlet temperature calculation with trajectory correction."""
        features = {
            'indoor_temp_lag_30m': 21.0,
            'target_temp': 21.0,
            'outdoor_temp': 5.0,
            'pv_now': 0.0,
            'fireplace_on': 0,
            'tv_on': 0
        }
        
        # Mock binary search convergence
        with patch.object(
            self.wrapper.thermal_model, 
            'predict_equilibrium_temperature', 
            return_value=21.0
        ):
            mock_trajectory = {
                'reaches_target_at': None,
                'equilibrium_temp': 21.0,
                'trajectory': [21.0, 20.8]
            }
            
            with patch.object(
                self.wrapper.thermal_model, 
                'predict_thermal_trajectory', 
                return_value=mock_trajectory
            ):
                outlet_temp, metadata = self.wrapper.calculate_optimal_outlet_temp(features)
                
        # Should return corrected outlet temperature
        assert outlet_temp is not None
        assert isinstance(outlet_temp, float)
        assert metadata['prediction_method'] == 'thermal_equilibrium_single_prediction'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
