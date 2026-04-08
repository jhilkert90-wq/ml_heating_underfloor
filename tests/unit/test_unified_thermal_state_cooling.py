"""
Tests for CoolingThermalStateManager.

Validates cooling-specific baseline parameters, independent learning state,
buffer snapshot persistence, and calibration tracking.
"""

import pytest
import os
import src.unified_thermal_state_cooling

from src.unified_thermal_state_cooling import (
    CoolingThermalStateManager,
    get_cooling_state_manager,
)
from src.thermal_config import ThermalParameterConfig


@pytest.fixture
def temp_state_file(tmp_path):
    return tmp_path / "test_cooling_state.json"


@pytest.fixture
def cooling_manager(temp_state_file):
    """Fresh CoolingThermalStateManager per test."""
    src.unified_thermal_state_cooling._cooling_state_manager = None
    return CoolingThermalStateManager(state_file=str(temp_state_file))


class TestCoolingThermalStateManager:
    """Core CoolingThermalStateManager tests."""

    def test_default_state_structure(self, cooling_manager):
        state = cooling_manager._get_default_state()
        assert "metadata" in state
        assert "baseline_parameters" in state
        assert "learning_state" in state
        assert "prediction_metrics" in state
        assert "buffer_state" in state
        assert "operational_state" in state

    def test_metadata_format_is_cooling(self, cooling_manager):
        assert (
            cooling_manager.state["metadata"]["format"]
            == "unified_thermal_state_cooling"
        )

    def test_baseline_uses_cooling_defaults(self, cooling_manager):
        baseline = cooling_manager.state["baseline_parameters"]
        cd = ThermalParameterConfig.COOLING_DEFAULTS
        assert baseline["thermal_time_constant"] == cd['thermal_time_constant']
        assert baseline["slab_time_constant_hours"] == cd['slab_time_constant_hours']
        assert baseline["outlet_effectiveness"] == cd['outlet_effectiveness']
        assert baseline["pv_heat_weight"] == cd['pv_heat_weight']

    def test_cooling_defaults_differ_from_heating(self, cooling_manager):
        """Key cooling defaults should differ from their heating counterparts."""
        cd = ThermalParameterConfig.COOLING_DEFAULTS
        hd = ThermalParameterConfig.DEFAULTS
        assert cd['thermal_time_constant'] != hd['thermal_time_constant']
        assert cd['slab_time_constant_hours'] != hd['slab_time_constant_hours']
        assert cd['outlet_effectiveness'] != hd['outlet_effectiveness']

    def test_save_and_load(self, cooling_manager, temp_state_file):
        cooling_manager.state["operational_state"]["last_indoor_temp"] = 25.0
        cooling_manager.save_state()

        reloaded = CoolingThermalStateManager(state_file=str(temp_state_file))
        assert reloaded.state["operational_state"]["last_indoor_temp"] == 25.0

    def test_set_calibrated_baseline(self, cooling_manager):
        params = {
            "heat_loss_coefficient": 0.10,
            "outlet_effectiveness": 0.85,
            "thermal_time_constant": 2.5,
        }
        cooling_manager.set_calibrated_baseline(params, calibration_cycles=50)

        state = cooling_manager.get_current_parameters()
        assert state["baseline_parameters"]["source"] == "calibrated"
        assert state["baseline_parameters"]["heat_loss_coefficient"] == 0.10
        assert state["baseline_parameters"]["calibration_cycles"] == 50

    def test_calibration_resets_learning_deltas(self, cooling_manager):
        cooling_manager.update_learning_state(
            parameter_adjustments={"equilibrium_ratio_delta": 0.05}
        )
        cooling_manager.set_calibrated_baseline(
            {"thermal_time_constant": 2.0}, calibration_cycles=10
        )
        adj = cooling_manager.state["learning_state"]["parameter_adjustments"]
        assert adj["equilibrium_ratio_delta"] == 0.0


class TestCoolingLearningState:
    """Online learning state tests."""

    def test_update_learning_state(self, cooling_manager):
        cooling_manager.update_learning_state(
            cycle_count=5, learning_confidence=4.0
        )
        metrics = cooling_manager.get_learning_metrics()
        assert metrics["current_cycle_count"] == 5
        assert metrics["learning_confidence"] == 4.0

    def test_add_prediction_record(self, cooling_manager):
        cooling_manager.add_prediction_record(
            {"predicted": 23.0, "actual": 23.2}
        )
        history = cooling_manager.state["learning_state"]["prediction_history"]
        assert len(history) == 1
        assert history[0]["predicted"] == 23.0

    def test_prediction_history_capped(self, cooling_manager):
        for i in range(210):
            cooling_manager.add_prediction_record({"i": i})
        history = cooling_manager.state["learning_state"]["prediction_history"]
        assert len(history) == 200

    def test_heat_source_channel_roundtrip(self, cooling_manager):
        channel_state = {
            "heat_pump": {"parameters": {"hp_weight": 0.9}},
        }
        cooling_manager.set_heat_source_channel_state(channel_state)
        assert cooling_manager.get_heat_source_channel_state() == channel_state

    def test_reset_learning_preserves_baseline(self, cooling_manager):
        cooling_manager.set_calibrated_baseline(
            {"thermal_time_constant": 2.5}, calibration_cycles=30
        )
        cooling_manager.update_learning_state(learning_confidence=1.0)
        cooling_manager.reset_learning_state()

        metrics = cooling_manager.get_learning_metrics()
        assert metrics["learning_confidence"] == ThermalParameterConfig.COOLING_DEFAULTS['learning_confidence']
        assert metrics["calibration_cycles"] == 30

    def test_get_computed_parameters(self, cooling_manager):
        cooling_manager.update_learning_state(
            parameter_adjustments={"equilibrium_ratio_delta": 0.02}
        )
        computed = cooling_manager.get_computed_parameters()
        cd = ThermalParameterConfig.COOLING_DEFAULTS
        assert computed["equilibrium_ratio"] == pytest.approx(
            cd['equilibrium_ratio'] + 0.02
        )


class TestCoolingBufferState:
    """Buffer snapshot persistence tests."""

    def test_default_buffer_state_empty(self, cooling_manager):
        buf = cooling_manager.state["buffer_state"]
        assert buf["sensor_snapshots"] == {}
        assert buf["last_snapshot_time"] is None

    def test_save_and_get_buffer_snapshot(self, cooling_manager):
        snapshots = {
            "sensor.outlet_temp": [
                ("2026-04-07T12:00:00+00:00", 20.5),
                ("2026-04-07T12:05:00+00:00", 20.3),
            ],
            "sensor.flow_rate": [
                ("2026-04-07T12:00:00+00:00", 120.0),
            ],
        }
        cooling_manager.save_buffer_snapshot(snapshots)

        loaded = cooling_manager.get_buffer_snapshot()
        assert "sensor.outlet_temp" in loaded
        assert len(loaded["sensor.outlet_temp"]) == 2
        assert loaded["sensor.flow_rate"][0][1] == 120.0

    def test_buffer_snapshot_persists_across_reload(
        self, cooling_manager, temp_state_file
    ):
        snapshots = {"sensor.inlet_temp": [("2026-04-07T12:00:00+00:00", 22.0)]}
        cooling_manager.save_buffer_snapshot(snapshots)

        reloaded = CoolingThermalStateManager(state_file=str(temp_state_file))
        loaded = reloaded.get_buffer_snapshot()
        assert loaded["sensor.inlet_temp"][0][1] == 22.0

    def test_buffer_snapshot_timestamp_set(self, cooling_manager):
        cooling_manager.save_buffer_snapshot({"s": []})
        assert cooling_manager.state["buffer_state"]["last_snapshot_time"] is not None


class TestCoolingOperationalState:
    """Operational state tests."""

    def test_update_operational_state(self, cooling_manager):
        cooling_manager.update_operational_state(last_prediction=21.0)
        op = cooling_manager.get_operational_state()
        assert op["last_prediction"] == 21.0

    def test_set_calibration_mode(self, cooling_manager, temp_state_file):
        cooling_manager.save_state()
        cooling_manager.set_calibration_mode(True)
        reloaded = CoolingThermalStateManager(state_file=str(temp_state_file))
        assert reloaded.state["operational_state"]["is_calibrating"] is True


class TestCoolingSingleton:
    """Singleton accessor tests."""

    def test_singleton_returns_same_instance(self):
        src.unified_thermal_state_cooling._cooling_state_manager = None
        m1 = get_cooling_state_manager()
        m2 = get_cooling_state_manager()
        assert m1 is m2

    def test_singleton_uses_shadow_isolated_path(self, tmp_path, monkeypatch):
        src.unified_thermal_state_cooling._cooling_state_manager = None
        base = tmp_path / "cooling_state.json"
        monkeypatch.setattr(
            "src.shadow_mode.config.SHADOW_MODE", True
        )
        monkeypatch.setattr(
            "src.shadow_mode.config.UNIFIED_STATE_FILE_COOLING",
            str(base),
        )
        mgr = get_cooling_state_manager()
        mgr.save_state()
        assert mgr.state_file == str(tmp_path / "cooling_state_shadow.json")
        assert os.path.exists(tmp_path / "cooling_state_shadow.json")


class TestCoolingBackup:
    """Backup tests."""

    def test_create_backup(self, cooling_manager, temp_state_file):
        cooling_manager.save_state()
        success, backup_file = cooling_manager.create_backup("test_backup")
        assert success
        assert os.path.exists(backup_file)


class TestThermalConfigCoolingDefaults:
    """Tests for ThermalParameterConfig cooling defaults."""

    def test_get_cooling_default(self):
        val = ThermalParameterConfig.get_cooling_default('thermal_time_constant')
        assert val == ThermalParameterConfig.COOLING_DEFAULTS['thermal_time_constant']

    def test_get_cooling_default_unknown_raises(self):
        with pytest.raises(KeyError):
            ThermalParameterConfig.get_cooling_default('no_such_param')

    def test_get_cooling_bounds(self):
        lo, hi = ThermalParameterConfig.get_cooling_bounds('thermal_time_constant')
        assert lo < hi

    def test_get_cooling_bounds_unknown_raises(self):
        with pytest.raises(KeyError):
            ThermalParameterConfig.get_cooling_bounds('no_such_param')

    def test_get_all_cooling_defaults(self):
        all_cd = ThermalParameterConfig.get_all_cooling_defaults()
        assert 'thermal_time_constant' in all_cd
        assert 'slab_time_constant_hours' in all_cd
