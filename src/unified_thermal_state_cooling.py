"""
Unified Thermal State Management — Cooling Mode

Dedicated state file for cooling-mode operation.  Mirrors the heating
``ThermalStateManager`` but carries cooling-specific baseline parameters,
an independent online-learning state, its own calibration record, and a
sensor-buffer snapshot so the system can resume cooling after restart
without losing recent sensor context.

The cooling state lives in a separate JSON file (default:
``unified_thermal_state_cooling.json``) so that heating and cooling
learning histories never cross-contaminate.
"""

import json
import os
import logging
from copy import deepcopy
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np

try:
    from .thermal_config import ThermalParameterConfig
    from .shadow_mode import get_effective_cooling_state_file
except ImportError:
    from thermal_config import ThermalParameterConfig
    from shadow_mode import get_effective_cooling_state_file


class CoolingThermalStateManager:
    """Cooling-mode thermal state manager.

    Provides a self-contained JSON persistence layer with:
    * **Cooling baseline parameters** — sensible defaults for underfloor
      slab cooling (tighter outlet range, faster slab τ, etc.).
    * **Online learning state** — independent cycle count, confidence,
      and parameter adjustments that only update during cooling cycles.
    * **Calibration record** — tracks when cooling parameters were last
      calibrated and how many cycles were used.
    * **Buffer state** — snapshot of recent sensor readings so cooling
      can resume after a service restart without a cold start.
    * **Operational state** — last-known temperatures and run metadata
      specific to cooling operation.
    """

    def __init__(self, state_file: Optional[str] = None):
        self.state_file = state_file or get_effective_cooling_state_file()
        self.state = self._get_default_state()

        if os.path.exists(self.state_file):
            self.load_state()
        else:
            logging.info("🆕 Creating new cooling thermal state file")

    # ------------------------------------------------------------------
    # Default state
    # ------------------------------------------------------------------

    def _get_default_state(self) -> Dict[str, Any]:
        """Return the default cooling state structure."""
        cd = ThermalParameterConfig.COOLING_DEFAULTS
        cb = ThermalParameterConfig.COOLING_BOUNDS
        return {
            "metadata": {
                "version": "1.0",
                "format": "unified_thermal_state_cooling",
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
            },
            "baseline_parameters": {
                "thermal_time_constant": cd['thermal_time_constant'],
                "equilibrium_ratio": cd['equilibrium_ratio'],
                "total_conductance": cd['total_conductance'],
                "heat_loss_coefficient": cd['heat_loss_coefficient'],
                "outlet_effectiveness": cd['outlet_effectiveness'],
                "solar_lag_minutes": cd['solar_lag_minutes'],
                "slab_time_constant_hours": cd['slab_time_constant_hours'],
                "pv_heat_weight": cd['pv_heat_weight'],
                "fireplace_heat_weight": cd['fireplace_heat_weight'],
                "tv_heat_weight": cd['tv_heat_weight'],
                "delta_t_floor": cd['delta_t_floor'],
                "fp_decay_time_constant": cd['fp_decay_time_constant'],
                "room_spread_delay_minutes": cd['room_spread_delay_minutes'],
                "source": "config_defaults",
                "calibration_date": None,
                "calibration_cycles": 0,
            },
            "learning_state": {
                "cycle_count": 0,
                "learning_confidence": cd['learning_confidence'],
                "learning_enabled": True,
                "parameter_adjustments": {
                    "equilibrium_ratio_delta": 0.0,
                    "total_conductance_delta": 0.0,
                    "heat_loss_coefficient_delta": 0.0,
                    "outlet_effectiveness_delta": 0.0,
                    "thermal_time_constant_delta": 0.0,
                    "pv_heat_weight_delta": 0.0,
                    "tv_heat_weight_delta": 0.0,
                    "solar_lag_minutes_delta": 0.0,
                    "slab_time_constant_delta": 0.0,
                    "delta_t_floor_delta": 0.0,
                    "fp_decay_time_constant_delta": 0.0,
                    "room_spread_delay_minutes_delta": 0.0,
                },
                "parameter_bounds": {
                    "equilibrium_ratio": list(cb['equilibrium_ratio']),
                    "total_conductance": list(cb['total_conductance']),
                },
                "heat_source_channels": {},
                "prediction_history": [],
                "parameter_history": [],
            },
            "prediction_metrics": {
                "total_predictions": 0,
                "accuracy_stats": {
                    "mae_1h": 0.0,
                    "mae_6h": 0.0,
                    "mae_24h": 0.0,
                    "mae_all_time": 0.0,
                    "rmse_all_time": 0.0,
                },
                "recent_performance": {
                    "last_10_mae": 0.0,
                    "last_10_max_error": 0.0,
                },
            },
            "buffer_state": {
                "sensor_snapshots": {},
                "last_snapshot_time": None,
            },
            "operational_state": {
                "last_indoor_temp": None,
                "last_outdoor_temp": None,
                "last_outlet_temp": None,
                "last_prediction": None,
                "last_run_time": None,
                "is_calibrating": False,
                "last_run_features": None,
                "last_final_temp": None,
                "last_avg_other_rooms_temp": None,
                "last_fireplace_on": False,
                "last_is_blocking": False,
                "last_blocking_reasons": [],
                "last_blocking_end_time": None,
            },
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_state(self) -> bool:
        """Load cooling state from JSON file."""
        try:
            with open(self.state_file, 'r') as f:
                loaded_state = json.load(f)

            self.state = self._merge_with_defaults(loaded_state)
            self.state["metadata"]["last_updated"] = datetime.now().isoformat()

            # Persist merged state so any new schema keys are written.
            self.save_state()

            logging.info(
                "✅ Loaded cooling thermal state from %s", self.state_file
            )
            return True

        except FileNotFoundError:
            logging.info(
                "📋 No existing cooling state file at %s", self.state_file
            )
            return False
        except Exception as e:
            logging.error("❌ Failed to load cooling state: %s", e)
            self.state = self._get_default_state()
            return False

    def save_state(self) -> bool:
        """Atomically persist cooling state to JSON."""
        import tempfile
        try:
            self.state["metadata"]["last_updated"] = datetime.now().isoformat()
            serializable = self._convert_numpy_types(self.state)

            dir_path = os.path.dirname(self.state_file) or '.'
            os.makedirs(dir_path, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                'w', dir=dir_path, delete=False, suffix='.tmp'
            ) as tmp_f:
                try:
                    import fcntl
                    fcntl.flock(tmp_f.fileno(), fcntl.LOCK_EX)
                except (ImportError, AttributeError, OSError):
                    pass
                json.dump(serializable, tmp_f, indent=2)
                tmp_path = tmp_f.name

            os.replace(tmp_path, self.state_file)
            logging.debug(
                "💾 Saved cooling thermal state to %s", self.state_file
            )
            return True

        except Exception as e:
            logging.error("❌ Failed to save cooling state: %s", e)
            return False

    # ------------------------------------------------------------------
    # Schema migration
    # ------------------------------------------------------------------

    def _merge_with_defaults(self, loaded: Dict) -> Dict:
        """Merge loaded state with defaults to handle schema migrations."""
        merged = self._get_default_state()

        def merge_dict(default: Dict, loaded: Dict) -> Dict:
            for key, value in loaded.items():
                if key in default:
                    if isinstance(default[key], dict) and isinstance(
                        value, dict
                    ):
                        default[key] = merge_dict(default[key], value)
                    else:
                        default[key] = value
                else:
                    default[key] = value
            return default

        return merge_dict(merged, loaded)

    # ------------------------------------------------------------------
    # Numpy serialization helper
    # ------------------------------------------------------------------

    def _convert_numpy_types(self, obj) -> Any:
        """Convert numpy / pandas types to native Python for JSON."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_numpy_types(i) for i in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # ------------------------------------------------------------------
    # Baseline parameters
    # ------------------------------------------------------------------

    def set_calibrated_baseline(
        self, parameters: Dict[str, float], calibration_cycles: int = 0
    ) -> None:
        """Set calibrated cooling baseline parameters."""
        baseline = self.state["baseline_parameters"]

        for key in (
            "thermal_time_constant",
            "equilibrium_ratio",
            "total_conductance",
            "heat_loss_coefficient",
            "outlet_effectiveness",
            "pv_heat_weight",
            "fireplace_heat_weight",
            "tv_heat_weight",
            "solar_lag_minutes",
            "slab_time_constant_hours",
            "delta_t_floor",
            "fp_decay_time_constant",
            "room_spread_delay_minutes",
        ):
            if key in parameters:
                baseline[key] = parameters[key]

        baseline["source"] = "calibrated"
        baseline["calibration_date"] = datetime.now().isoformat()
        baseline["calibration_cycles"] = calibration_cycles

        # Reset learning adjustments after re-calibration.
        for k in self.state["learning_state"]["parameter_adjustments"]:
            self.state["learning_state"]["parameter_adjustments"][k] = 0.0
        self.state["learning_state"]["heat_source_channels"] = {}

        logging.info(
            "🎯 Set calibrated cooling baseline (cycles: %s)",
            calibration_cycles,
        )
        self.save_state()

    def get_current_parameters(self) -> Dict[str, Any]:
        """Return the complete cooling state structure."""
        return self.state.copy()

    def get_computed_parameters(self) -> Dict[str, float]:
        """Return effective cooling parameters (baseline + learning deltas)."""
        baseline = self.state["baseline_parameters"]
        adj = self.state["learning_state"]["parameter_adjustments"]
        cd = ThermalParameterConfig.COOLING_DEFAULTS

        return {
            "thermal_time_constant": (
                baseline["thermal_time_constant"]
                + adj.get("thermal_time_constant_delta", 0.0)
            ),
            "equilibrium_ratio": (
                baseline["equilibrium_ratio"]
                + adj["equilibrium_ratio_delta"]
            ),
            "total_conductance": (
                baseline["total_conductance"]
                + adj["total_conductance_delta"]
            ),
            "heat_loss_coefficient": (
                baseline["heat_loss_coefficient"]
                + adj.get("heat_loss_coefficient_delta", 0.0)
            ),
            "outlet_effectiveness": (
                baseline["outlet_effectiveness"]
                + adj.get("outlet_effectiveness_delta", 0.0)
            ),
            "pv_heat_weight": (
                baseline["pv_heat_weight"]
                + adj.get("pv_heat_weight_delta", 0.0)
            ),
            "fireplace_heat_weight": baseline["fireplace_heat_weight"],
            "tv_heat_weight": (
                baseline["tv_heat_weight"]
                + adj.get("tv_heat_weight_delta", 0.0)
            ),
            "solar_lag_minutes": (
                baseline.get("solar_lag_minutes", cd['solar_lag_minutes'])
                + adj.get("solar_lag_minutes_delta", 0.0)
            ),
            "slab_time_constant_hours": (
                baseline.get(
                    "slab_time_constant_hours",
                    cd['slab_time_constant_hours'],
                )
                + adj.get("slab_time_constant_delta", 0.0)
            ),
        }

    # ------------------------------------------------------------------
    # Learning state
    # ------------------------------------------------------------------

    def update_learning_state(
        self,
        cycle_count: Optional[int] = None,
        learning_confidence: Optional[float] = None,
        parameter_adjustments: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update cooling-mode learning state."""
        ls = self.state["learning_state"]

        if cycle_count is not None:
            ls["cycle_count"] = cycle_count
        if learning_confidence is not None:
            ls["learning_confidence"] = learning_confidence
        if parameter_adjustments is not None:
            for param, delta in parameter_adjustments.items():
                ls["parameter_adjustments"][param] = delta

        self.save_state()

    def add_prediction_record(self, prediction_record: Dict) -> None:
        """Append a prediction record to cooling history."""
        history = self.state["learning_state"]["prediction_history"]
        history.append(prediction_record)
        if len(history) > 200:
            self.state["learning_state"]["prediction_history"] = history[-200:]
        self.state["prediction_metrics"]["total_predictions"] += 1

    def add_parameter_history_record(self, parameter_record: Dict) -> None:
        """Append a parameter history record."""
        history = self.state["learning_state"]["parameter_history"]
        history.append(parameter_record)
        if len(history) > 500:
            self.state["learning_state"]["parameter_history"] = history[-500:]

    def get_heat_source_channel_state(self) -> Dict[str, Any]:
        """Get persisted cooling heat-source channel state."""
        return deepcopy(
            self.state["learning_state"].get("heat_source_channels", {})
        )

    def set_heat_source_channel_state(
        self, channel_state: Dict[str, Any]
    ) -> None:
        """Persist cooling heat-source channel state."""
        self.state["learning_state"]["heat_source_channels"] = deepcopy(
            channel_state
        )
        self.save_state()

    # ------------------------------------------------------------------
    # Buffer state
    # ------------------------------------------------------------------

    def save_buffer_snapshot(
        self,
        sensor_snapshots: Dict[str, List[Tuple[str, float]]],
    ) -> None:
        """Persist a snapshot of sensor buffer readings.

        ``sensor_snapshots`` maps sensor_id → list of
        ``(iso_timestamp, value)`` pairs.
        """
        self.state["buffer_state"]["sensor_snapshots"] = sensor_snapshots
        self.state["buffer_state"][
            "last_snapshot_time"
        ] = datetime.now().isoformat()
        self.save_state()

    def get_buffer_snapshot(
        self,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Return the last persisted sensor buffer snapshot."""
        return deepcopy(
            self.state["buffer_state"].get("sensor_snapshots", {})
        )

    # ------------------------------------------------------------------
    # Operational state
    # ------------------------------------------------------------------

    def update_operational_state(self, **kwargs) -> None:
        """Update cooling operational state."""
        op = self.state["operational_state"]
        for key, value in kwargs.items():
            if key in op:
                op[key] = value
        op["last_run_time"] = datetime.now().isoformat()

    def get_operational_state(self) -> Dict[str, Any]:
        """Return current cooling operational state."""
        return self.state["operational_state"].copy()

    def set_calibration_mode(self, is_calibrating: bool) -> None:
        """Set cooling calibration mode flag."""
        self.state["operational_state"]["is_calibrating"] = is_calibrating
        self.save_state()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_learning_metrics(self) -> Dict[str, Any]:
        """Return comprehensive cooling learning metrics."""
        ls = self.state["learning_state"]
        baseline = self.state["baseline_parameters"]
        metrics = self.state["prediction_metrics"]

        return {
            "baseline_source": baseline["source"],
            "calibration_date": baseline["calibration_date"],
            "calibration_cycles": baseline["calibration_cycles"],
            "current_cycle_count": ls["cycle_count"],
            "learning_confidence": ls["learning_confidence"],
            "total_predictions": metrics["total_predictions"],
            "current_parameters": self.get_current_parameters(),
            "parameter_adjustments": ls["parameter_adjustments"].copy(),
            "heat_source_channels": self.get_heat_source_channel_state(),
            "accuracy_stats": metrics["accuracy_stats"].copy(),
            "learning_enabled": ls["learning_enabled"],
        }

    # ------------------------------------------------------------------
    # Reset / backup
    # ------------------------------------------------------------------

    def reset_learning_state(self, keep_baseline: bool = True) -> None:
        """Reset cooling learning state."""
        if keep_baseline:
            ls = self.state["learning_state"]
            ls["cycle_count"] = 0
            ls["learning_confidence"] = (
                ThermalParameterConfig.COOLING_DEFAULTS['learning_confidence']
            )
            ls["prediction_history"] = []
            ls["parameter_history"] = []
            ls["heat_source_channels"] = {}
            for key in ls["parameter_adjustments"]:
                ls["parameter_adjustments"][key] = 0.0
        else:
            self.state = self._get_default_state()

        logging.info(
            "🔄 Reset cooling learning state (baseline preserved: %s)",
            keep_baseline,
        )
        self.save_state()

    def create_backup(self, backup_name: str = "") -> tuple[bool, str]:
        """Create a backup of the cooling state."""
        try:
            if not backup_name:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f'cooling_state_backup_{timestamp}'
            if not backup_name.endswith('.json'):
                backup_name += '.json'

            backup_dir = os.path.dirname(self.state_file)
            backup_file = os.path.join(backup_dir, backup_name)

            import shutil
            shutil.copy2(self.state_file, backup_file)
            logging.info("🔄 Created cooling state backup: %s", backup_name)
            return True, backup_file

        except Exception as e:
            logging.error("❌ Failed to create cooling state backup: %s", e)
            return False, str(e)


# ------------------------------------------------------------------
# Global singleton
# ------------------------------------------------------------------

_cooling_state_manager: Optional[CoolingThermalStateManager] = None


def get_cooling_state_manager() -> CoolingThermalStateManager:
    """Return the global cooling thermal state manager (singleton)."""
    global _cooling_state_manager
    if _cooling_state_manager is None:
        _cooling_state_manager = CoolingThermalStateManager()
    return _cooling_state_manager


def save_cooling_state(**kwargs) -> None:
    """Convenience: update operational state and persist."""
    manager = get_cooling_state_manager()
    manager.update_operational_state(**kwargs)
    manager.save_state()


def load_cooling_state() -> Dict[str, Any]:
    """Convenience: load and return cooling state dict."""
    manager = get_cooling_state_manager()
    manager.load_state()
    return manager.state
