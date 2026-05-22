"""
Unit tests for the physics Newton-step heating correction and mode dispatch.

Covers:
1. _calculate_physics_newton_correction() with ε = +0.3 K (undershoot)
2. _calculate_physics_newton_correction() with ε = −0.3 K (overshoot)
3. S_t ≤ 0.01 safety guard falls back to legacy correction
4. Large ε clamped to ±2.5 °C
5. Mode dispatch: "physics" routes to Newton method
6. Mode dispatch: "legacy" routes to legacy method
7. Mode dispatch: "ml" falls back to Newton (with warning)
8. config_adapter maps heating_correction_mode → HEATING_CORRECTION_MODE
9. Mid-horizon PV overshoot uses S(t_worst) not S(H)
10. Undershoot at last step uses S(H) (t_worst = H)
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

# S_H = [η/(η+U)] × [1 − exp(−H/τ)] — used when worst point is at t=H
S_H_EXPECTED = (ETA / (ETA + U)) * (1.0 - math.exp(-H / TAU))  # ≈ 0.5202

# S at t=3h (index 2 out of 4 uniform steps, step size = H/4 = 1h)
# This is what the Newton method uses when the worst point is at step 2.
T_3H = 3.0
S_3H_EXPECTED = (ETA / (ETA + U)) * (1.0 - math.exp(-T_3H / TAU))  # ≈ 0.4306


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

    # 1. Undershoot ε = +0.3 K at step 2 (t=3h) --------------------------------
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_undershoot_0_3k(self):
        """ε = +0.3 K at step 2 (t=3h) should give ΔT = 0.3 / S(3h)."""
        target = 21.0
        outlet = 25.0
        # min of trajectory is 20.7 at index 2 → t_worst = 3h
        # step size = H/n = 4h/4 = 1h; t = (idx+1)*dt = (2+1)*1h = 3h
        # (trajectory model uses one-based times: step k → t = (k+1)*dt)
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

        # t_worst = 3h (index 2, step size = 4/4 = 1h)
        expected_correction = 0.3 / S_3H_EXPECTED
        assert result == pytest.approx(outlet + expected_correction, abs=0.01), (
            f"Expected outlet ≈ {outlet + expected_correction:.3f}°C, "
            f"got {result:.3f}°C (used S(3h)={S_3H_EXPECTED:.4f})"
        )

    # 2. Overshoot ε = −0.3 K at step 2 (t=3h) --------------------------------
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_overshoot_0_3k(self):
        """ε = −0.3 K at step 2 (t=3h) should give ΔT = −0.3 / S(3h)."""
        target = 21.0
        outlet = 25.0
        # max of trajectory is 21.3 at index 2 → t_worst = 3h
        # step size = H/n = 4h/4 = 1h; t = (idx+1)*dt = (2+1)*1h = 3h
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

        expected_correction = -0.3 / S_3H_EXPECTED
        assert result == pytest.approx(outlet + expected_correction, abs=0.01), (
            f"Expected outlet ≈ {outlet + expected_correction:.3f}°C, "
            f"got {result:.3f}°C (used S(3h)={S_3H_EXPECTED:.4f})"
        )

    # 3. S_t ≤ 0.01 safety guard → fallback to legacy -------------------------
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_degenerate_eta_falls_back_to_legacy(self):
        """η = 0 → S(t) < 0.01 → method must fall back to legacy."""
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
        """ε = 5.0 K → raw correction is large but clamped to 2.5 °C."""
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

    # 9. Mid-horizon PV overshoot: S(t_worst) < S(H) → bigger correction ------
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_mid_horizon_pv_overshoot_uses_t_worst(self):
        """PV drives overshoot at t=3h (step 2); S(3h) < S(4h) → larger ΔT.

        This test guards against the regression where S_H was used regardless
        of when the worst point occurred, causing systematic under-correction
        when solar gain peaks mid-trajectory.

        With t_worst=3h: S(3h) = (ETA/(ETA+U)) * (1 - exp(-3/TAU))
        With t_worst=4h: S(4h) = (ETA/(ETA+U)) * (1 - exp(-4/TAU))
        Since 3h < 4h, S(3h) < S(4h), so correction = ε/S(3h) > ε/S(4h).

        NOTE: t_worst=3h is above τ/2≈2.2h so the τ/2 floor does NOT
        trigger here — this test validates the pure mid-horizon correction.
        """
        target = 21.0
        outlet = 25.0
        # Max at index 2 (t=3h for 4 equal steps of 1h each).
        # trajectory dict includes "times" so t_worst is unambiguous.
        trajectory = {
            'trajectory': [21.0, 21.2, 21.4, 21.1],  # max=21.4 at idx 2
            'times': [1.0, 2.0, 3.0, 4.0],
            'reaches_target_at': 0.05,
        }
        # Positive trend so self-correction gate does not skip
        self.wrapper._current_indoor = target
        self.wrapper._current_features = {'indoor_temp_delta_60m': +0.5}

        result = self.wrapper._calculate_physics_newton_correction(
            outlet_temp=outlet,
            trajectory=trajectory,
            target_indoor=target,
            cycle_hours=10 / 60,
        )

        # t_worst = times[2] = 3.0 h
        s_t_worst = S_3H_EXPECTED
        s_h = S_H_EXPECTED
        eps = target - 21.4  # = -0.4 K

        expected_correction_t_worst = eps / s_t_worst   # larger magnitude
        expected_correction_h      = eps / s_h          # smaller magnitude

        # The correction must use t_worst (larger magnitude)
        assert result == pytest.approx(
            outlet + expected_correction_t_worst, abs=0.01
        ), (
            f"Expected S(t_worst=3h)={s_t_worst:.4f} correction "
            f"{expected_correction_t_worst:+.3f}°C, got {result - outlet:+.3f}°C. "
            f"Under-correction from S_H={s_h:.4f} would have been "
            f"{expected_correction_h:+.3f}°C."
        )
        # Sanity: t_worst correction must be larger in magnitude than H correction
        assert abs(expected_correction_t_worst) > abs(expected_correction_h)

    # 10. Error at last step uses S(H) -----------------------------------------
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_undershoot_at_last_step_uses_s_h(self):
        """When worst undershoot is at the last step, S(H) is correct."""
        target = 21.0
        outlet = 25.0
        # Min is at index 3 (last step) → t_worst = H = 4h
        trajectory = {
            'trajectory': [21.0, 20.95, 20.9, 20.7],
            'times': [1.0, 2.0, 3.0, 4.0],
            'reaches_target_at': None,
        }
        self.wrapper._current_indoor = target
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}

        result = self.wrapper._calculate_physics_newton_correction(
            outlet_temp=outlet,
            trajectory=trajectory,
            target_indoor=target,
            cycle_hours=10 / 60,
        )

        # t_worst = 4h = H → should use S_H
        eps = target - 20.7  # = +0.3 K
        expected_correction = eps / S_H_EXPECTED
        assert result == pytest.approx(outlet + expected_correction, abs=0.01), (
            f"Expected S(H=4h)={S_H_EXPECTED:.4f} correction "
            f"{expected_correction:+.3f}°C, got {result - outlet:+.3f}°C."
        )

    # 11. else branch: no violation but target not reached within 3 cycles ----
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_else_branch_applies_correction_when_target_not_reached(self):
        """When no min/max violation but reaches_target_at is None (target
        never reached), the Newton method corrects using the error at the last
        step and uses S(H) as the sensitivity (t_eval = H).

        This test guards the threshold-alignment fix: the else branch now uses
        `reaches_target_at > cycle_hours + tolerance_hours` (3× cycle) instead
        of the original `> cycle_hours` (1× cycle), matching the outer gate.
        """
        target = 21.0
        outlet = 25.0
        # All trajectory points within ±0.1°C of target (no violation),
        # but reaches_target_at = None → correction must still be applied.
        trajectory = {
            'trajectory': [20.95, 20.97, 20.98, 20.96],  # all within ±0.1
            'times': [1.0, 2.0, 3.0, 4.0],
            'reaches_target_at': None,  # never reaches target within horizon
        }
        self.wrapper._current_indoor = target
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.05}

        result = self.wrapper._calculate_physics_newton_correction(
            outlet_temp=outlet,
            trajectory=trajectory,
            target_indoor=target,
            cycle_hours=10 / 60,
        )

        # Error is at last step: ε = 21.0 - 20.96 = 0.04 K
        # Sensitivity at t=H=4h
        eps = target - trajectory['trajectory'][-1]  # +0.04 K
        expected_correction = eps / S_H_EXPECTED
        assert result == pytest.approx(outlet + expected_correction, abs=0.01), (
            f"else-branch expected correction {expected_correction:+.4f}°C "
            f"(S_H={S_H_EXPECTED:.4f}), got {result - outlet:+.4f}°C."
        )

    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_else_branch_no_correction_when_target_reached_in_time(self):
        """When no violation and reaches_target_at ≤ cycle_hours + tolerance,
        the else branch must return outlet_temp unchanged (temp_error = 0).
        """
        target = 21.0
        outlet = 25.0
        cycle_hours = 10 / 60
        tolerance_hours = cycle_hours * 2
        # reaches_target_at is within 3× cycle — should NOT apply correction
        trajectory = {
            'trajectory': [20.95, 20.97, 20.98, 20.96],
            'times': [1.0, 2.0, 3.0, 4.0],
            'reaches_target_at': cycle_hours + tolerance_hours - 0.01,
        }
        self.wrapper._current_indoor = target
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.05}

        result = self.wrapper._calculate_physics_newton_correction(
            outlet_temp=outlet,
            trajectory=trajectory,
            target_indoor=target,
            cycle_hours=cycle_hours,
        )

        assert result == pytest.approx(outlet, abs=0.01), (
            f"No correction expected (target reached in time), got "
            f"Δ={result - outlet:+.4f}°C."
        )

    # 12. τ/2 floor suppresses degenerate early-step correction ----------------
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_tau_half_floor_suppresses_degenerate_correction(self):
        """When worst error is at step 0 (t=1h < τ/2=2.195h), the floor
        re-evaluates ε at the nearest step to τ/2, preventing degenerate
        S(t) ≈ 0 and always-clamped corrections.
        """
        target = 21.0
        outlet = 25.0
        # Undershoot worst at step 0 (t=1h), improving monotonically
        trajectory = {
            'trajectory': [20.5, 20.8, 20.95, 21.0],
            'times': [1.0, 2.0, 3.0, 4.0],
            'reaches_target_at': 4.0,
        }
        self.wrapper._current_indoor = target
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}

        result = self.wrapper._calculate_physics_newton_correction(
            outlet_temp=outlet,
            trajectory=trajectory,
            target_indoor=target,
            cycle_hours=10 / 60,
        )

        # τ/2 = 4.39/2 = 2.195h → nearest step is index 1 (t=2h)
        # ε_floored = 21.0 - 20.8 = +0.2°C (not +0.5°C from step 0)
        # S(τ/2) = (ETA/(ETA+U)) * (1 - exp(-τ/2 / TAU))
        t_floor = TAU * 0.5
        s_floor = (ETA / (ETA + U)) * (1.0 - math.exp(-t_floor / TAU))
        eps_floored = target - 20.8  # +0.2°C
        expected_correction = eps_floored / s_floor

        assert result == pytest.approx(
            outlet + expected_correction, abs=0.05
        ), (
            f"Floor should give ε={eps_floored:+.2f}°C / S(τ/2)={s_floor:.4f} "
            f"= {expected_correction:+.3f}°C, got {result - outlet:+.3f}°C"
        )
        # Must NOT be clamped (without floor it would be +2.5°C clamped)
        assert abs(result - outlet) < 2.5, "Correction should not be clamped"

    # 13. τ/2 floor sign flip suppresses correction entirely -------------------
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_tau_half_floor_sign_flip_suppresses_correction(self):
        """When the trajectory has recovered by τ/2 (sign flips from
        undershoot to overshoot), the correction is suppressed entirely.
        """
        target = 21.0
        outlet = 25.0
        # Undershoot at step 0, but recovered (overshoot) by step 1
        trajectory = {
            'trajectory': [20.5, 21.2, 21.1, 21.05],
            'times': [1.0, 2.0, 3.0, 4.0],
            'reaches_target_at': 1.5,
        }
        self.wrapper._current_indoor = target
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}

        result = self.wrapper._calculate_physics_newton_correction(
            outlet_temp=outlet,
            trajectory=trajectory,
            target_indoor=target,
            cycle_hours=10 / 60,
        )

        # τ/2 = 2.195h → nearest is index 1 (t=2h), temp=21.2 > target
        # Original ε = +0.5 (undershoot), floored ε = -0.2 (overshoot)
        # Sign flip → correction suppressed → returns outlet unchanged
        assert result == pytest.approx(outlet, abs=0.01), (
            f"Sign flip at τ/2 should suppress correction, "
            f"got Δ={result - outlet:+.3f}°C"
        )


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

    @patch('src.model_wrapper.config.TRAJECTORY_PREDICTION_ENABLED', True)
    @patch('src.model_wrapper.config.HEATING_CORRECTION_MODE', 'ml')
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_cooling_mode_ml_override_routes_to_newton(self):
        """Cooling mode must override ML dispatch to Newton correction."""
        self.wrapper.set_climate_mode("cooling")
        self.wrapper._current_indoor = 21.0
        self.wrapper._current_features = {'indoor_temp_delta_60m': -0.1}

        with patch.object(
            self.wrapper.thermal_model,
            'predict_thermal_trajectory',
            return_value=self._mock_trajectory,
        ), patch.object(
            self.wrapper,
            '_calculate_physics_newton_correction',
            return_value=25.4,
        ) as mock_newton, patch.object(
            self.wrapper,
            '_calculate_ml_correction',
        ) as mock_ml:
            self.wrapper._verify_trajectory_and_correct(
                outlet_temp=25.0,
                current_indoor=21.0,
                target_indoor=21.0,
                outdoor_temp=3.0,
                thermal_features={'pv_power': 0.0, 'fireplace_on': 0.0,
                                   'tv_on': 0.0},
            )

        mock_newton.assert_called_once()
        mock_ml.assert_not_called()

    # 7. "ml" mode: no loaded model → falls back to Newton (no crash) ----------
    @patch('src.model_wrapper.config.TRAJECTORY_STEPS', 4)
    def test_ml_mode_falls_back_to_newton_when_model_not_loaded(self):
        """When no model is loaded, _calculate_ml_correction delegates to Newton."""
        from src.model_wrapper import EnhancedModelWrapper
        # Ensure no cached model from other tests
        EnhancedModelWrapper._heating_correction_ml_model = None

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
        ) as mock_newton, patch.object(
            self.wrapper,
            '_get_heating_correction_ml_model',
            return_value=None,
        ):
            result = self.wrapper._calculate_ml_correction(
                outlet_temp=25.0,
                trajectory=trajectory,
                target_indoor=21.0,
                cycle_hours=10 / 60,
            )

        mock_newton.assert_called_once()
        assert result == 25.6


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
        """When key is absent, adapter must still emit 'legacy' explicitly."""
        from config_adapter import convert_addon_to_env
        env = convert_addon_to_env({})
        assert env['HEATING_CORRECTION_MODE'] == 'legacy'
