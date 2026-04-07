"""Tests for Phase 3: Fireplace decay + spread wired into predictions."""

import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.thermal_equilibrium_model import ThermalEquilibriumModel


# ===================================================================
# predict_equilibrium_temperature – fireplace_decay_kw parameter
# ===================================================================
class TestEquilibriumDecay:
    """Verify that fireplace_decay_kw flows into the equilibrium prediction."""

    def setup_method(self):
        self.model = ThermalEquilibriumModel()
        self.model.heat_loss_coefficient = 0.2

    def test_decay_adds_heat_when_fp_off(self):
        """FP off + decay_kw > 0 → hotter equilibrium than without."""
        base = self.model.predict_equilibrium_temperature(
            outlet_temp=30.0,
            outdoor_temp=5.0,
            current_indoor=20.0,
            thermal_power=2.0,
            fireplace_on=0,
            fireplace_decay_kw=0.0,
        )
        with_decay = self.model.predict_equilibrium_temperature(
            outlet_temp=30.0,
            outdoor_temp=5.0,
            current_indoor=20.0,
            thermal_power=2.0,
            fireplace_on=0,
            fireplace_decay_kw=1.5,
        )
        assert with_decay > base + 0.5

    def test_decay_zero_same_as_before(self):
        """Zero decay should not change the prediction."""
        base = self.model.predict_equilibrium_temperature(
            outlet_temp=30.0,
            outdoor_temp=5.0,
            current_indoor=20.0,
            thermal_power=2.0,
            fireplace_on=0,
        )
        explicit_zero = self.model.predict_equilibrium_temperature(
            outlet_temp=30.0,
            outdoor_temp=5.0,
            current_indoor=20.0,
            thermal_power=2.0,
            fireplace_on=0,
            fireplace_decay_kw=0.0,
        )
        assert abs(base - explicit_zero) < 0.01

    def test_decay_stacks_with_fp_on(self):
        """When FP is on AND decay > 0, both contribute."""
        fp_on_only = self.model.predict_equilibrium_temperature(
            outlet_temp=30.0,
            outdoor_temp=5.0,
            current_indoor=20.0,
            thermal_power=2.0,
            fireplace_on=1,
            fireplace_power_kw=3.0,
            fireplace_decay_kw=0.0,
        )
        fp_on_plus_decay = self.model.predict_equilibrium_temperature(
            outlet_temp=30.0,
            outdoor_temp=5.0,
            current_indoor=20.0,
            thermal_power=2.0,
            fireplace_on=1,
            fireplace_power_kw=3.0,
            fireplace_decay_kw=1.0,
        )
        assert fp_on_plus_decay > fp_on_only


# ===================================================================
# predict_thermal_trajectory – fireplace_decay_kw in trajectory
# ===================================================================
class TestTrajectoryDecay:
    """Verify that fireplace_decay_kw flows through trajectory simulation."""

    def setup_method(self):
        self.model = ThermalEquilibriumModel()
        self.model.heat_loss_coefficient = 0.2

    def test_trajectory_with_decay_warmer(self):
        """Trajectory with FP decay should be warmer than without."""
        base = self.model.predict_thermal_trajectory(
            current_indoor=20.0,
            target_indoor=21.0,
            outlet_temp=30.0,
            outdoor_temp=5.0,
            time_horizon_hours=2.0,
            time_step_minutes=10,
            thermal_power=2.0,
            fireplace_on=0,
            fireplace_decay_kw=0.0,
        )
        with_decay = self.model.predict_thermal_trajectory(
            current_indoor=20.0,
            target_indoor=21.0,
            outlet_temp=30.0,
            outdoor_temp=5.0,
            time_horizon_hours=2.0,
            time_step_minutes=10,
            thermal_power=2.0,
            fireplace_on=0,
            fireplace_decay_kw=2.0,
        )
        # Final point of trajectory with decay should be higher
        base_final = base["trajectory"][-1]
        decay_final = with_decay["trajectory"][-1]
        assert decay_final > base_final + 0.1


# ===================================================================
# model_wrapper – _calculate_fireplace_power_kw returns (power, decay)
# ===================================================================
class TestModelWrapperDecay:
    """Verify model wrapper returns tuple (power, decay_kw)."""

    @patch("src.model_wrapper.ThermalEquilibriumModel")
    @patch("src.model_wrapper.get_thermal_state_manager")
    @patch("src.model_wrapper.PredictionMetrics")
    def test_fp_on_returns_power_zero_decay(self, mock_pm, mock_mgr, mock_tem):
        """When FP is ON, returns (power_kw, 0.0)."""
        mock_mgr.return_value.get_learning_metrics.return_value = {
            "current_cycle_count": 0
        }
        mock_orch = MagicMock()
        mock_channel = MagicMock()
        mock_channel.estimate_heat_contribution.return_value = 5.0
        mock_orch.channels = {"fireplace": mock_channel}
        mock_tem.return_value.orchestrator = mock_orch

        from src.model_wrapper import EnhancedModelWrapper as ModelWrapper
        wrapper = ModelWrapper.__new__(ModelWrapper)
        wrapper.thermal_model = mock_tem.return_value
        wrapper.adaptive_fireplace = None
        wrapper._fireplace_last_on_time = None
        wrapper._fireplace_on_since = None

        power, decay = wrapper._calculate_fireplace_power_kw(
            current_indoor=20.0,
            outdoor_temp=5.0,
            fireplace_on=1.0,
        )
        assert power == 5.0
        assert decay == 0.0

    @patch("src.model_wrapper.ThermalEquilibriumModel")
    @patch("src.model_wrapper.get_thermal_state_manager")
    @patch("src.model_wrapper.PredictionMetrics")
    def test_fp_off_recent_returns_decay(self, mock_pm, mock_mgr, mock_tem):
        """When FP was recently on and now off, returns (None, decay_kw)."""
        mock_mgr.return_value.get_learning_metrics.return_value = {
            "current_cycle_count": 0
        }
        mock_channel = MagicMock()
        mock_channel.estimate_decay_contribution.return_value = 2.5
        mock_orch = MagicMock()
        mock_orch.channels = {"fireplace": mock_channel}
        mock_tem.return_value.orchestrator = mock_orch

        from src.model_wrapper import EnhancedModelWrapper as ModelWrapper
        wrapper = ModelWrapper.__new__(ModelWrapper)
        wrapper.thermal_model = mock_tem.return_value
        wrapper.adaptive_fireplace = None
        wrapper._fireplace_last_on_time = datetime.now() - timedelta(minutes=15)
        wrapper._fireplace_on_since = None

        power, decay = wrapper._calculate_fireplace_power_kw(
            current_indoor=20.0,
            outdoor_temp=5.0,
            fireplace_on=0.0,
        )
        assert power is None
        assert decay == 2.5
        mock_channel.estimate_decay_contribution.assert_called_once()

    @patch("src.model_wrapper.ThermalEquilibriumModel")
    @patch("src.model_wrapper.get_thermal_state_manager")
    @patch("src.model_wrapper.PredictionMetrics")
    def test_fp_off_long_ago_no_decay(self, mock_pm, mock_mgr, mock_tem):
        """When FP was off for hours, decay is negligible → 0."""
        mock_mgr.return_value.get_learning_metrics.return_value = {
            "current_cycle_count": 0
        }
        mock_channel = MagicMock()
        mock_channel.estimate_decay_contribution.return_value = 0.001  # negligible
        mock_orch = MagicMock()
        mock_orch.channels = {"fireplace": mock_channel}
        mock_tem.return_value.orchestrator = mock_orch

        from src.model_wrapper import EnhancedModelWrapper as ModelWrapper
        wrapper = ModelWrapper.__new__(ModelWrapper)
        wrapper.thermal_model = mock_tem.return_value
        wrapper.adaptive_fireplace = None
        wrapper._fireplace_last_on_time = datetime.now() - timedelta(hours=5)
        wrapper._fireplace_on_since = None

        power, decay = wrapper._calculate_fireplace_power_kw(
            current_indoor=20.0,
            outdoor_temp=5.0,
            fireplace_on=0.0,
        )
        assert power is None
        assert decay == 0.0  # below 0.01 threshold

    @patch("src.model_wrapper.ThermalEquilibriumModel")
    @patch("src.model_wrapper.get_thermal_state_manager")
    @patch("src.model_wrapper.PredictionMetrics")
    def test_fp_never_on_no_decay(self, mock_pm, mock_mgr, mock_tem):
        """When FP was never on, returns (None, 0.0)."""
        mock_mgr.return_value.get_learning_metrics.return_value = {
            "current_cycle_count": 0
        }

        from src.model_wrapper import EnhancedModelWrapper as ModelWrapper
        wrapper = ModelWrapper.__new__(ModelWrapper)
        wrapper.thermal_model = mock_tem.return_value
        wrapper.adaptive_fireplace = None
        wrapper._fireplace_last_on_time = None
        wrapper._fireplace_on_since = None

        power, decay = wrapper._calculate_fireplace_power_kw(
            current_indoor=20.0,
            outdoor_temp=5.0,
            fireplace_on=0.0,
        )
        assert power is None
        assert decay == 0.0


# ===================================================================
# estimate_decay_contribution – unit test on FireplaceChannel
# ===================================================================
class TestFireplaceChannelDecay:
    """Direct test of the (previously dead) estimate_decay_contribution."""

    def test_exponential_decay(self):
        from src.heat_source_channels import FireplaceChannel
        ch = FireplaceChannel.__new__(FireplaceChannel)
        ch.fp_heat_output_kw = 5.0
        ch.fp_decay_time_constant = 0.75  # hours

        # At t=0 → 0 (guard)
        assert ch.estimate_decay_contribution(0, {}) == 0.0

        # At τ → 5·exp(-1) ≈ 1.84
        val = ch.estimate_decay_contribution(0.75, {})
        expected = 5.0 * math.exp(-1)
        assert abs(val - expected) < 0.01

        # At 3τ → ~0.25
        val_3tau = ch.estimate_decay_contribution(2.25, {})
        assert val_3tau < 0.3
        assert val_3tau > 0.0
