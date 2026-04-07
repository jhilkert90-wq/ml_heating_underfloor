"""Tests for Phase 6: Wire + persist calibrated channel parameters."""

import pytest

from src.heat_source_channels import HeatSourceChannelOrchestrator
from src.physics_calibration import _apply_channel_params


class TestApplyChannelParams:
    """Verify _apply_channel_params sets attributes on channels."""

    def test_apply_fp_decay(self):
        orch = HeatSourceChannelOrchestrator()
        _apply_channel_params(orch, {"fp_decay_time_constant": 1.25})
        assert orch.channels["fireplace"].fp_decay_time_constant == 1.25

    def test_apply_room_spread_delay(self):
        orch = HeatSourceChannelOrchestrator()
        _apply_channel_params(orch, {"room_spread_delay_minutes": 45.0})
        assert orch.channels["fireplace"].room_spread_delay_minutes == 45.0

    def test_apply_delta_t_floor(self):
        orch = HeatSourceChannelOrchestrator()
        _apply_channel_params(orch, {"delta_t_floor": 2.5})
        assert orch.channels["heat_pump"].delta_t_floor == 2.5

    def test_apply_cloud_factor(self):
        orch = HeatSourceChannelOrchestrator()
        _apply_channel_params(orch, {"cloud_factor_exponent": 1.5})
        assert orch.channels["pv"].cloud_factor_exponent == 1.5

    def test_apply_solar_decay(self):
        orch = HeatSourceChannelOrchestrator()
        _apply_channel_params(orch, {"solar_decay_tau_hours": 0.8})
        assert orch.channels["pv"].solar_decay_tau_hours == 0.8

    def test_apply_all_at_once(self):
        orch = HeatSourceChannelOrchestrator()
        params = {
            "fp_decay_time_constant": 1.1,
            "room_spread_delay_minutes": 50.0,
            "delta_t_floor": 3.0,
            "cloud_factor_exponent": 1.3,
            "solar_decay_tau_hours": 0.7,
        }
        _apply_channel_params(orch, params)
        assert orch.channels["fireplace"].fp_decay_time_constant == 1.1
        assert orch.channels["fireplace"].room_spread_delay_minutes == 50.0
        assert orch.channels["heat_pump"].delta_t_floor == 3.0
        assert orch.channels["pv"].cloud_factor_exponent == 1.3
        assert orch.channels["pv"].solar_decay_tau_hours == 0.7

    def test_empty_params_no_change(self):
        orch = HeatSourceChannelOrchestrator()
        before = orch.channels["fireplace"].fp_decay_time_constant
        _apply_channel_params(orch, {})
        assert orch.channels["fireplace"].fp_decay_time_constant == before


class TestSyncRoundTrip:
    """Verify channel params survive export/import cycle."""

    def test_export_includes_channel_params(self):
        orch = HeatSourceChannelOrchestrator()
        orch.channels["fireplace"].fp_decay_time_constant = 1.5
        orch.channels["pv"].cloud_factor_exponent = 1.8
        orch.channels["heat_pump"].delta_t_floor = 2.8
        exported = orch.export_model_parameters()
        assert exported["fp_decay_time_constant"] == 1.5
        assert exported["cloud_factor_exponent"] == 1.8
        assert exported["delta_t_floor"] == 2.8

    def test_sync_from_model_parameters_loads_channel_params(self):
        orch = HeatSourceChannelOrchestrator()
        params = {
            "thermal_time_constant": 4.0,
            "heat_loss_coefficient": 0.2,
            "outlet_effectiveness": 0.55,
            "slab_time_constant_hours": 1.0,
            "pv_heat_weight": 0.005,
            "solar_lag_minutes": 90.0,
            "fireplace_heat_weight": 5.0,
            "tv_heat_weight": 0.2,
            "fp_decay_time_constant": 1.2,
            "room_spread_delay_minutes": 40.0,
            "delta_t_floor": 2.5,
            "cloud_factor_exponent": 1.4,
            "solar_decay_tau_hours": 0.6,
        }
        orch.sync_from_model_parameters(params)
        assert orch.channels["fireplace"].fp_decay_time_constant == 1.2
        assert orch.channels["fireplace"].room_spread_delay_minutes == 40.0
        assert orch.channels["heat_pump"].delta_t_floor == 2.5
        assert orch.channels["pv"].cloud_factor_exponent == 1.4
        assert orch.channels["pv"].solar_decay_tau_hours == 0.6

    def test_full_round_trip(self):
        """Export → new orchestrator → sync → values match."""
        orch1 = HeatSourceChannelOrchestrator()
        orch1.channels["fireplace"].fp_decay_time_constant = 0.9
        orch1.channels["pv"].solar_decay_tau_hours = 0.4
        orch1.channels["heat_pump"].delta_t_floor = 3.1

        exported = orch1.export_model_parameters()

        orch2 = HeatSourceChannelOrchestrator()
        orch2.sync_from_model_parameters(exported)

        assert orch2.channels["fireplace"].fp_decay_time_constant == 0.9
        assert orch2.channels["pv"].solar_decay_tau_hours == 0.4
        assert orch2.channels["heat_pump"].delta_t_floor == 3.1
