"""
Unit tests for the physics Newton-step heating correction and mode dispatch.

Covers:
1. _calculate_physics_newton_correction() with ε = +0.3 K (undershoot)
2. _calculate_physics_newton_correction() with ε = −0.3 K (overshoot)
3. S_H ≤ 0.01 safety guard falls back to legacy correction
4. Large ε clamped to ±2.5 °C
5. Mode dispatch: "physics" routes to Newton method
6. Mode dispatch: "legacy" routes to legacy method
7. Mode dispatch: "ml" falls back to Newton (with warning)
8. config_adapter maps heating_correction_mode → HEATING_CORRECTION_MODE
"""
import math
import logging
import pytest
from unittest.mock import patch, MagicMock

from src.model_wrapper import get_enhanced_model_wrapper


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# Default thermal params (mirror thermal_config.py DEFAULTS used in the
# analysis document)
ETA = 0.830   # outlet_effectiveness
U   = 0.124   # heat_loss_coefficient
TAU = 4.39    # thermal_time_constant (h)
H   = 4.0     # trajectory_steps (h)

# Expected S_H = [η/(η+U)] × [1 − exp(−H/τ)]
S_H_EXPECTED = (ETA / (ETA + U)) * (1.0 - math.exp(-H / TAU))  # ≈ 0.5202


def _make_wrapper():
    """Return a singleton EnhancedModelWrapper with outlet clamping set."""
    with patch('src.model_wrapper.config.CLAMP_MIN_ABS', 20.0), \
         patch('src.model_wrapper.config.CLAMP_MAX_ABS', 55.0):
        w = get_enhanced_model_wrapper()
    # Ensure no leftover state from previous tests
    for attr in ('_current_indoor', '_current_features'):
        if hasattr(w, attr):
            delattr(w, attr)
    return w


def _patch_thermal_params(wrapper, eta=ETA, u=U, tau=TAU):
    """Patch thermal model parameters on wrapper.thermal_model."""
    wrapper.thermal_model.outlet_effectiveness = eta
    wrapper.thermal_model.heat_loss_coefficient = u
    wrapper.thermal_model.thermal_time_constant = tau


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestPhysicsNewtonCorrection:
    """Tests for _calculate_physics_newton_correction()."""

    def setup_method(self):
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
        _patch_thermal_params(self.wrapper)

    def teardown_method(self):
        self._clamp_min_patcher.stop()
        self._clamp_max_patcher.stop()

    # 1. Undershoot ε = +0.3 K -------------------------------------------------
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_undershoot_0_3k(self):
        """ε = +0.3 K at H=4 h should give ΔT ≈ +0.577 °C."""
        target = 21.0
        outlet = 25.0
        # min of trajectory is 20.7 → ε = 0.3 K
        trajectory = {
            'trajectory': [21.0, 20.9, 20.7, 20.8],
            'reaches_target_at': None,
        }
        # Negative trend so the self-correction gate does NOT skip
        self.wrapper._current_indoor = target
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}

        result = self.wrapper._calculate_physics_newton_correction(
            outlet_temp=outlet,
            trajectory=trajectory,
            target_indoor=target,
            cycle_hours=10 / 60,
        )

        expected_correction = 0.3 / S_H_EXPECTED  # ≈ 0.577
        assert result == pytest.approx(outlet + expected_correction, abs=0.01), (
            f"Expected outlet ≈ {outlet + expected_correction:.3f}°C, "
            f"got {result:.3f}°C"
        )

    # 2. Overshoot ε = −0.3 K --------------------------------------------------
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_overshoot_0_3k(self):
        """ε = −0.3 K at H=4 h should give ΔT ≈ −0.577 °C."""
        target = 21.0
        outlet = 25.0
        # max of trajectory is 21.3 → ε = −0.3 K
        trajectory = {
            'trajectory': [21.0, 21.2, 21.3, 21.2],
            'reaches_target_at': 0.1,
        }
        # Positive trend so the self-correction gate does NOT skip
        self.wrapper._current_indoor = target
        self.wrapper._current_features = {'indoor_temp_delta_60m': +0.3}

        result = self.wrapper._calculate_physics_newton_correction(
            outlet_temp=outlet,
            trajectory=trajectory,
            target_indoor=target,
            cycle_hours=10 / 60,
        )

        expected_correction = -0.3 / S_H_EXPECTED  # ≈ −0.577
        assert result == pytest.approx(outlet + expected_correction, abs=0.01), (
            f"Expected outlet ≈ {outlet + expected_correction:.3f}°C, "
            f"got {result:.3f}°C"
        )

    # 3. S_H ≤ 0.01 safety guard → fallback to legacy -------------------------
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_degenerate_eta_falls_back_to_legacy(self):
        """η = 0 → S_H < 0.01 → method must fall back to legacy."""
        _patch_thermal_params(self.wrapper, eta=0.0)  # Degenerate
        trajectory = {
            'trajectory': [21.0, 20.9, 20.7, 20.8],
            'reaches_target_at': None,
        }
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}

        with patch.object(
            self.wrapper,
            '_calculate_physics_based_correction',
            return_value=26.0,
        ) as mock_legacy:
            result = self.wrapper._calculate_physics_newton_correction(
                outlet_temp=25.0,
                trajectory=trajectory,
                target_indoor=21.0,
                cycle_hours=10 / 60,
            )

        mock_legacy.assert_called_once()
        assert result == 26.0

    # 4. Large ε clamped to ±2.5 °C -------------------------------------------
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_large_undershoot_clamped(self):
        """ε = 5.0 K → raw correction ≈ 9.6 °C, but clamped to 2.5 °C."""
        target = 21.0
        outlet = 25.0
        trajectory = {
            'trajectory': [21.0, 18.0, 16.0, 16.0],  # min=16 → ε=5 K
            'reaches_target_at': None,
        }
        self.wrapper._current_indoor = target
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.5}

        result = self.wrapper._calculate_physics_newton_correction(
            outlet_temp=outlet,
            trajectory=trajectory,
            target_indoor=target,
            cycle_hours=10 / 60,
        )

        assert result == pytest.approx(outlet + 2.5, abs=0.01), (
            f"Clamped correction should be 2.5°C, outlet={result:.3f}°C"
        )


# ---------------------------------------------------------------------------
# Mode dispatch tests
# ---------------------------------------------------------------------------

class TestCorrectionModeDispatch:
    """Tests for the HEATING_CORRECTION_MODE dispatch in
    _verify_trajectory_and_correct()."""

    def setup_method(self):
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
        _patch_thermal_params(self.wrapper)

        self._mock_trajectory = {
            'trajectory': [21.0, 20.9, 20.7, 20.8],
            'reaches_target_at': None,
            'equilibrium_temp': 21.0,
        }

    def teardown_method(self):
        self._clamp_min_patcher.stop()
        self._clamp_max_patcher.stop()

    # 5. "physics" mode routes to Newton method --------------------------------
    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    @patch('src.model_wrapper.config.HEATING_CORRECTION_MODE', 'physics')
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_physics_mode_routes_to_newton(self):
        """HEATING_CORRECTION_MODE='physics' must call Newton method."""
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}

        with patch.object(
            self.wrapper.thermal_model,
            'predict_thermal_trajectory',
            return_value=self._mock_trajectory,
        ), patch.object(
            self.wrapper,
            '_calculate_physics_newton_correction',
            return_value=25.6,
        ) as mock_newton, patch.object(
            self.wrapper,
            '_calculate_physics_based_correction',
        ) as mock_legacy:
            self.wrapper._verify_trajectory_and_correct(
                outlet_temp=25.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=3.0,
                thermal_features={'pv_power': 0.0, 'fireplace_on': 0.0,
                                   'tv_on': 0.0},
            )

        mock_newton.assert_called_once()
        mock_legacy.assert_not_called()

    # 6. "legacy" mode routes to legacy method ---------------------------------
    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    @patch('src.model_wrapper.config.HEATING_CORRECTION_MODE', 'legacy')
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_legacy_mode_routes_to_legacy(self):
        """HEATING_CORRECTION_MODE='legacy' must call legacy method."""
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}

        with patch.object(
            self.wrapper.thermal_model,
            'predict_thermal_trajectory',
            return_value=self._mock_trajectory,
        ), patch.object(
            self.wrapper,
            '_calculate_physics_based_correction',
            return_value=26.3,
        ) as mock_legacy, patch.object(
            self.wrapper,
            '_calculate_physics_newton_correction',
        ) as mock_newton:
            self.wrapper._verify_trajectory_and_correct(
                outlet_temp=25.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=3.0,
                thermal_features={'pv_power': 0.0, 'fireplace_on': 0.0,
                                   'tv_on': 0.0},
            )

        mock_legacy.assert_called_once()
        mock_newton.assert_not_called()

    # 7. "ml" mode falls back to Newton (with warning) -------------------------
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_ml_mode_falls_back_to_newton(self, caplog):
        """_calculate_ml_correction() should warn and delegate to Newton."""
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}
        trajectory = {
            'trajectory': [21.0, 20.9, 20.7, 20.8],
            'reaches_target_at': None,
        }

        with patch.object(
            self.wrapper,
            '_calculate_physics_newton_correction',
            return_value=25.6,
        ) as mock_newton, caplog.at_level(logging.WARNING, logger='src.model_wrapper'):
            result = self.wrapper._calculate_ml_correction(
                outlet_temp=25.0,
                trajectory=trajectory,
                target_indoor=21.0,
                cycle_hours=10 / 60,
            )

        mock_newton.assert_called_once()
        assert result == 25.6
        assert any(
            'not yet implemented' in rec.message.lower()
            or 'falling back' in rec.message.lower()
            for rec in caplog.records
        ), "Expected a warning about ML not being implemented"


# ---------------------------------------------------------------------------
# config_adapter wiring test
# ---------------------------------------------------------------------------

class TestConfigAdapterHeatingCorrectionMode:
    """Test that config_adapter maps heating_correction_mode → env var."""

    def test_maps_physics_mode(self):
        from config_adapter import convert_addon_to_env
        env = convert_addon_to_env({'heating_correction_mode': 'physics'})
        assert env['HEATING_CORRECTION_MODE'] == 'physics'

    def test_maps_legacy_mode(self):
        from config_adapter import convert_addon_to_env
        env = convert_addon_to_env({'heating_correction_mode': 'legacy'})
        assert env['HEATING_CORRECTION_MODE'] == 'legacy'

    def test_maps_ml_mode(self):
        from config_adapter import convert_addon_to_env
        env = convert_addon_to_env({'heating_correction_mode': 'ml'})
        assert env['HEATING_CORRECTION_MODE'] == 'ml'

    def test_default_is_legacy(self):
        """When key is absent, default must be 'legacy'."""
        from config_adapter import convert_addon_to_env
        env = convert_addon_to_env({})
        assert env.get('HEATING_CORRECTION_MODE', 'legacy') == 'legacy'
