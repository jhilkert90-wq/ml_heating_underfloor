"""
Tests for the finite-difference Newton correction that uses the trajectory
simulator itself (slab dynamics + pump gate) instead of the analytic
single-zone S(t).
"""
import logging
from unittest.mock import patch

import pytest

from src.model_wrapper import get_enhanced_model_wrapper
from src.thermal_equilibrium_model import ThermalEquilibriumModel


def _sim(slope_on, inlet, base_temp=23.5, drift=-0.02, gate=True):
    """Fake simulator: T_t = base + drift*t + slope_on*(outlet-25.8)*(t/H).

    Outlet <= inlet switches the pump off (no outlet sensitivity).
    """
    n = 24

    def simulate(outlet):
        active = outlet > inlet if gate else True
        return [
            base_temp + drift * i
            + (slope_on * (outlet - 25.8) * (i + 1) / n if active else
               slope_on * (inlet - 25.8) * (i + 1) / n * 0.0)
            for i in range(n)
        ]

    return simulate


class TestFiniteDifferenceNewton:
    def setup_method(self):
        self.clamp = [
            patch("src.model_wrapper.config.CLAMP_MIN_ABS", 20.0),
            patch("src.model_wrapper.config.CLAMP_MAX_ABS", 55.0),
        ]
        for p in self.clamp:
            p.start()
        self.w = get_enhanced_model_wrapper()
        self.w._climate_mode = "heating"
        self.w._current_features = {"inlet_temp": 26.1}

    def teardown_method(self):
        for p in self.clamp:
            p.stop()
        self.w._trajectory_simulator = None

    def _traj(self, simulate, outlet):
        temps = simulate(outlet)
        traj = {"trajectory": temps, "times": [(i + 1) / 6 for i in range(24)]}
        self.w._trajectory_simulator = (traj, simulate)
        return traj

    def _real_traj(self, mode):
        model = ThermalEquilibriumModel(climate_mode=mode)
        model.orchestrator = None
        model.thermal_time_constant = 2.0
        model.heat_loss_coefficient = 1.2
        model.outlet_effectiveness = 1.0
        model.slab_time_constant_hours = 0.5
        self.w.thermal_model = model
        self.w._climate_mode = mode

        if mode == "cooling":
            inlet, outlet, delta_t = 20.0, 20.0, -2.0
            current, outdoor, target, error = 22.0, 22.0, 21.8, 0.5
            bounds = (19.5, 20.5)
        else:
            inlet, outlet, delta_t = 26.0, 26.0, 2.0
            current, outdoor, target, error = 24.0, 24.0, 24.2, -0.5
            bounds = (25.5, 26.5)

        kwargs = {
            "current_indoor": current,
            "target_indoor": target,
            "outdoor_temp": outdoor,
            "time_horizon_hours": 4.0,
            "time_step_minutes": 15,
            "inlet_temp": inlet,
            "delta_t_floor": delta_t,
            "climate_mode": mode,
        }

        def simulate(candidate):
            return model.predict_thermal_trajectory(
                outlet_temp=candidate, **kwargs
            )["trajectory"]

        trajectory = model.predict_thermal_trajectory(
            outlet_temp=outlet, **kwargs
        )
        self.w._trajectory_simulator = (trajectory, simulate)
        self.w._current_features = {"inlet_temp": inlet}
        return outlet, target, error, bounds, trajectory, simulate

    def test_no_simulator_returns_none(self):
        traj = {"trajectory": [23.0, 23.1]}
        self.w._trajectory_simulator = None
        assert self.w._finite_difference_newton_correction(
            28.0, traj, 22.6, -0.5
        ) is None

    def test_simulator_for_other_trajectory_is_ignored(self):
        simulate = _sim(0.5, 26.1)
        self._traj(simulate, 28.0)
        other = {"trajectory": simulate(28.0)}
        assert self.w._finite_difference_newton_correction(
            28.0, other, 22.6, -0.5
        ) is None

    def test_overshoot_lowers_outlet_when_pump_on(self):
        simulate = _sim(0.8, 26.1, base_temp=23.4)
        traj = self._traj(simulate, 29.0)
        result = self.w._finite_difference_newton_correction(
            29.0, traj, 22.6, -0.8
        )
        assert result is not None
        assert result < 29.0
        # Corrected outlet must remove (most of) the modelled overshoot
        assert max(simulate(result)) < max(traj["trajectory"])

    @pytest.mark.parametrize("mode", ["heating", "cooling"])
    def test_real_trajectory_reduces_effort_past_initial_inlet(self, mode):
        outlet, target, error, bounds, trajectory, simulate = (
            self._real_traj(mode)
        )
        with patch(
            "src.model_wrapper.config.get_outlet_bounds",
            return_value=bounds,
        ):
            result = self.w._finite_difference_newton_correction(
                outlet, trajectory, target, error
            )

        assert result is not None
        if mode == "heating":
            assert result < outlet
            assert max(simulate(result)) < max(trajectory["trajectory"])
        else:
            assert result > outlet
            assert min(simulate(result)) > min(trajectory["trajectory"])
        assert bounds[0] <= result <= bounds[1]
        probe_trajectory = simulate(
            outlet - 1.0 if mode == "heating" else outlet + 1.0
        )
        assert probe_trajectory[0] == pytest.approx(
            trajectory["trajectory"][0]
        )
        assert probe_trajectory[-1] != pytest.approx(
            trajectory["trajectory"][-1]
        )

    @pytest.mark.parametrize("mode", ["heating", "cooling"])
    def test_correction_does_not_cross_target_on_other_side(self, mode):
        # Decaying trajectory that already ends on target: a full Newton
        # step would undershoot (heating) / overshoot (cooling) later.
        sign = 1.0 if mode == "heating" else -1.0
        n, target = 24, 22.5

        def simulate(outlet):
            return [
                target + sign * (0.9 - 0.04 * i)
                + 0.2 * (outlet - 26.3) * (i + 1) / n
                for i in range(n)
            ]

        self.w._climate_mode = mode
        self.w._current_features = {"inlet_temp": 23.8}
        traj = {"trajectory": simulate(26.3)}
        self.w._trajectory_simulator = (traj, simulate)
        error = -sign * 0.9
        with patch(
            "src.model_wrapper.config.get_outlet_bounds",
            return_value=(10.0, 40.0),
        ):
            result = self.w._finite_difference_newton_correction(
                26.3, traj, target, error
            )
        assert result is not None
        temps = simulate(result)
        if mode == "heating":
            assert min(temps) >= target - 0.1 - 1e-6
            assert result < 26.3
        else:
            assert max(temps) <= target + 0.1 + 1e-6
            assert result > 26.3
        assert abs(result - 26.3) < 1.0

    def test_pump_off_overshoot_keeps_outlet(self):
        # Outlet below inlet: the simulator has no outlet sensitivity, so
        # lowering further cannot help.
        simulate = _sim(0.8, 26.1, base_temp=23.4)
        traj = self._traj(simulate, 25.8)
        result = self.w._finite_difference_newton_correction(
            25.8, traj, 22.6, -0.8
        )
        assert result == pytest.approx(25.8)

    def test_pump_off_undershoot_probes_into_active_region(self):
        simulate = _sim(0.8, 26.1, base_temp=22.0)
        traj = self._traj(simulate, 25.8)
        result = self.w._finite_difference_newton_correction(
            25.8, traj, 22.6, 0.6
        )
        assert result is not None and result > 25.8

    def test_correction_is_clamped(self):
        simulate = _sim(0.05, 26.1, base_temp=20.0)
        traj = self._traj(simulate, 30.0)
        result = self.w._finite_difference_newton_correction(
            30.0, traj, 22.6, 2.6
        )
        assert result == pytest.approx(32.5, abs=1e-6)

    def test_noise_in_pump_off_branch_does_not_raise_outlet(self):
        def simulate(outlet):
            # Pump off everywhere; tiny negative numerical noise vs outlet
            return [23.5 - 0.02 * i - 0.0001 * (outlet - 25.8) for i in range(24)]

        traj = self._traj(simulate, 25.8)
        result = self.w._finite_difference_newton_correction(
            25.8, traj, 22.6, -0.9
        )
        assert result == pytest.approx(25.8)

    def test_newton_entrypoint_prefers_fd(self):
        simulate = _sim(0.8, 26.1, base_temp=23.4)
        traj = self._traj(simulate, 29.0)
        traj["reaches_target_at"] = None
        self.w._current_indoor = 23.4
        self.w._current_features = {
            "inlet_temp": 26.1, "indoor_temp_delta_60m": 0.3,
        }
        with patch(
            "src.model_wrapper.config.TRAJECTORY_STEPS", 4
        ):
            result = self.w._calculate_physics_newton_correction(
                29.0, traj, 22.6, 10 / 60
            )
        assert result < 29.0
        assert result >= 26.1 - 1e-6
