
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model_wrapper import EnhancedModelWrapper as ModelWrapper
from src import config

class TestOvershootLogic(unittest.TestCase):
    def setUp(self):
        self.wrapper = ModelWrapper()
        self.wrapper.thermal_model = MagicMock()
        # Ensure predict_thermal_trajectory_with_forecasts is not present so it falls back to predict_thermal_trajectory
        if hasattr(self.wrapper.thermal_model, 'predict_thermal_trajectory_with_forecasts'):
            del self.wrapper.thermal_model.predict_thermal_trajectory_with_forecasts
        
        # Mock config
        config.CYCLE_INTERVAL_MINUTES = 30
        config.TRAJECTORY_STEPS = 4
        
    def test_fast_push_logic(self):
        """
        Test that the system allows a 'fast push' when the immediate cycle is safe
        but future steps might slightly overshoot.
        """
        # Setup inputs
        outlet_temp = 33.1
        current_indoor = 21.0
        target_indoor = 21.2
        outdoor_temp = 8.6
        thermal_features = {"pv_power": 0, "fireplace_on": 0, "tv_on": 0}
        
        # Mock predict_thermal_trajectory to return a trajectory that:
        # 1. Is safe at 30 min (21.15 < 21.3)
        # 2. Overshoots at 60 min (21.33 > 21.3) but is within relaxed bound (21.7)
        
        # The wrapper now calls predict_thermal_trajectory with time_step_minutes=30
        # So we expect steps at 30, 60, 90, ...
        
        mock_trajectory = {
            "trajectory": [21.15, 21.33, 21.4, 21.4],
            "times": [0.5, 1.0, 1.5, 2.0],
            "reaches_target_at": 0.6, # Reaches target shortly after 30 mins
            "overshoot_predicted": True # The model might flag it, but wrapper re-evaluates
        }
        
        self.wrapper.thermal_model.predict_thermal_trajectory.return_value = mock_trajectory
        
        # Mock _get_forecast_conditions to return simple values
        self.wrapper._get_forecast_conditions = MagicMock(return_value=(outdoor_temp, 0, [], []))
        
        # Mock _calculate_physics_based_correction to return a dampened value
        # We want to ensure this is NOT called
        self.wrapper._calculate_physics_based_correction = MagicMock(return_value=32.8)
        
        # Execute
        result = self.wrapper._verify_trajectory_and_correct(
            outlet_temp=outlet_temp,
            current_indoor=current_indoor,
            target_indoor=target_indoor,
            outdoor_temp=outdoor_temp,
            thermal_features=thermal_features
        )
        
        # Verification
        
        # 1. Verify predict_thermal_trajectory was called with correct time step
        # Note: The actual call might include additional arguments or defaults, so we check the critical ones
        call_args = self.wrapper.thermal_model.predict_thermal_trajectory.call_args
        self.assertIsNotNone(call_args, "predict_thermal_trajectory should have been called")
        
        kwargs = call_args[1]
        actual_time_step = kwargs.get('time_step_minutes')
        
        self.assertEqual(actual_time_step, config.CYCLE_INTERVAL_MINUTES,
                        f"Should use configured time step ({config.CYCLE_INTERVAL_MINUTES}), got {actual_time_step}")
        self.assertEqual(kwargs.get('outlet_temp'), outlet_temp)
        self.assertEqual(kwargs.get('current_indoor'), current_indoor)
        
        # 2. Verify NO correction was applied (returned original outlet_temp)
        # If correction was applied, it would return the mocked 32.8
        self.assertEqual(result, outlet_temp, "Should return original outlet temp for fast push scenario")

    def test_immediate_overshoot_prevented(self):
        """
        Test that the system still prevents overshoot if it happens in the immediate cycle.
        """
        # Setup inputs
        outlet_temp = 40.0 # Very hot
        current_indoor = 21.0
        target_indoor = 21.2
        outdoor_temp = 8.6
        thermal_features = {"pv_power": 0, "fireplace_on": 0, "tv_on": 0}
        
        # Mock trajectory that overshoots immediately
        mock_trajectory = {
            "trajectory": [21.4, 21.6, 21.8, 22.0], # 21.4 > 21.2 + 0.1
            "times": [0.5, 1.0, 1.5, 2.0],
            "reaches_target_at": 0.2,
            "overshoot_predicted": True
        }
        
        self.wrapper.thermal_model.predict_thermal_trajectory.return_value = mock_trajectory
        self.wrapper._get_forecast_conditions = MagicMock(return_value=(outdoor_temp, 0, [], []))
        self.wrapper._calculate_physics_based_correction = MagicMock(return_value=35.0)
        
        # Execute
        result = self.wrapper._verify_trajectory_and_correct(
            outlet_temp=outlet_temp,
            current_indoor=current_indoor,
            target_indoor=target_indoor,
            outdoor_temp=outdoor_temp,
            thermal_features=thermal_features
        )
        
        # Verification
        # Should return corrected value
        self.assertEqual(result, 35.0, "Should apply correction for immediate overshoot")

    def test_cooling_min_violates_only_returns_unchanged(self):
        """Cooling mode must skip undershoot-only correction."""
        self.wrapper._climate_mode = "cooling"
        self.wrapper._current_features = {"indoor_temp_delta_60m": 0.4}
        self.wrapper.thermal_model.slab_time_constant_hours = 3.0
        self.wrapper.thermal_model.outlet_effectiveness = 0.5

        trajectory = {"trajectory": [21.8, 21.9, 22.0, 22.0]}
        outlet_temp = 35.0
        result = self.wrapper._calculate_physics_based_correction(
            outlet_temp=outlet_temp,
            trajectory=trajectory,
            target_indoor=22.0,
            cycle_hours=0.5,
        )
        self.assertEqual(
            result,
            outlet_temp,
            "Cooling min_violates-only path should return outlet_temp unchanged",
        )

    def test_cooling_overshoot_still_corrected_when_projected_self_correcting(self):
        """Cooling mode must still correct overshoot even when projected_indoor would skip in heating."""
        self.wrapper._climate_mode = "cooling"
        self.wrapper._current_features = {"indoor_temp_delta_60m": -1.0}
        self.wrapper.thermal_model.slab_time_constant_hours = 3.0
        self.wrapper.thermal_model.outlet_effectiveness = 0.5

        trajectory = {"trajectory": [22.0, 22.2, 22.3, 22.25]}
        outlet_temp = 35.0
        result = self.wrapper._calculate_physics_based_correction(
            outlet_temp=outlet_temp,
            trajectory=trajectory,
            target_indoor=22.0,
            cycle_hours=0.5,
        )
        self.assertLess(
            result,
            outlet_temp,
            "Cooling overshoot path should apply a negative correction",
        )


class TestDisableOvershootCorrectionInForecastMode(unittest.TestCase):
    """Tests for PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION guard in _verify_trajectory_and_correct."""

    def setUp(self):
        self.wrapper = ModelWrapper()
        self.wrapper.thermal_model = MagicMock()
        config.CYCLE_INTERVAL_MINUTES = 30
        config.TRAJECTORY_STEPS = 4
        self.wrapper._get_forecast_conditions = MagicMock(
            return_value=(8.6, 0, [], [])
        )
        self.wrapper._calculate_physics_based_correction = MagicMock(return_value=32.0)

    def _call(self, outlet_temp=35.0):
        return self.wrapper._verify_trajectory_and_correct(
            outlet_temp=outlet_temp,
            current_indoor=21.0,
            target_indoor=21.5,
            outdoor_temp=8.6,
            thermal_features={"pv_power": 0, "fireplace_on": 0, "tv_on": 0},
        )

    def test_flag_true_and_forecast_mode_true_skips_correction(self):
        """When both flags are true, correction is skipped and outlet_temp returned unchanged."""
        config.PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION = True
        config.PV_TRAJ_FORECAST_MODE_ENABLED = True
        try:
            result = self._call(outlet_temp=35.0)
        finally:
            config.PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION = False
            config.PV_TRAJ_FORECAST_MODE_ENABLED = False
        self.assertEqual(result, 35.0)
        self.wrapper.thermal_model.predict_thermal_trajectory.assert_not_called()
        self.wrapper._calculate_physics_based_correction.assert_not_called()

    def test_flag_true_forecast_mode_false_does_not_skip(self):
        """When disable flag is true but forecast mode is off, correction still runs."""
        config.PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION = True
        config.PV_TRAJ_FORECAST_MODE_ENABLED = False
        mock_traj = {
            "trajectory": [22.0, 22.1, 22.2, 22.3],
            "times": [0.5, 1.0, 1.5, 2.0],
            "reaches_target_at": None,
        }
        self.wrapper.thermal_model.predict_thermal_trajectory.return_value = mock_traj
        try:
            self._call(outlet_temp=35.0)
        finally:
            config.PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION = False
        self.wrapper.thermal_model.predict_thermal_trajectory.assert_called_once()

    def test_flag_false_forecast_mode_true_does_not_skip(self):
        """When forecast mode is on but disable flag is false, correction still runs."""
        config.PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION = False
        config.PV_TRAJ_FORECAST_MODE_ENABLED = True
        mock_traj = {
            "trajectory": [22.0, 22.1, 22.2, 22.3],
            "times": [0.5, 1.0, 1.5, 2.0],
            "reaches_target_at": None,
        }
        self.wrapper.thermal_model.predict_thermal_trajectory.return_value = mock_traj
        try:
            self._call(outlet_temp=35.0)
        finally:
            config.PV_TRAJ_FORECAST_MODE_ENABLED = False
        self.wrapper.thermal_model.predict_thermal_trajectory.assert_called_once()

    def test_default_flag_value_is_false(self):
        """Default value of PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION is False."""
        import importlib
        import src.config as cfg_module
        original = os.environ.pop("PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION", None)
        try:
            importlib.reload(cfg_module)
            self.assertFalse(cfg_module.PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION)
        finally:
            if original is not None:
                os.environ["PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION"] = original
            importlib.reload(cfg_module)

    def test_skip_only_when_trajectory_steps_above_min_steps(self):
        """Skip remains active only when TRAJECTORY_STEPS > PV_TRAJ_MIN_STEPS."""
        config.PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION = True
        config.PV_TRAJ_FORECAST_MODE_ENABLED = True
        config.PV_TRAJ_MIN_STEPS = 2
        config.TRAJECTORY_STEPS = 4
        try:
            result = self._call(outlet_temp=35.0)
        finally:
            config.PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION = False
            config.PV_TRAJ_FORECAST_MODE_ENABLED = False
            config.PV_TRAJ_MIN_STEPS = 2
            config.TRAJECTORY_STEPS = 4
        self.assertEqual(result, 35.0)
        self.wrapper.thermal_model.predict_thermal_trajectory.assert_not_called()
        self.wrapper._calculate_physics_based_correction.assert_not_called()

    def test_min_steps_boundary_reenables_correction_path(self):
        """At TRAJECTORY_STEPS == PV_TRAJ_MIN_STEPS, correction path must run."""
        config.PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION = True
        config.PV_TRAJ_FORECAST_MODE_ENABLED = True
        config.PV_TRAJ_MIN_STEPS = 2
        config.TRAJECTORY_STEPS = 2
        mock_traj = {
            "trajectory": [21.0, 21.1],
            "times": [0.5, 1.0],
            "reaches_target_at": None,
        }
        self.wrapper.thermal_model.predict_thermal_trajectory.return_value = mock_traj
        try:
            result = self._call(outlet_temp=35.0)
        finally:
            config.PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION = False
            config.PV_TRAJ_FORECAST_MODE_ENABLED = False
            config.PV_TRAJ_MIN_STEPS = 2
            config.TRAJECTORY_STEPS = 4
        self.assertEqual(
            result,
            self.wrapper._calculate_physics_based_correction.return_value
        )
        self.wrapper.thermal_model.predict_thermal_trajectory.assert_called_once()
        self.wrapper._calculate_physics_based_correction.assert_called_once()


if __name__ == '__main__':
    unittest.main()
