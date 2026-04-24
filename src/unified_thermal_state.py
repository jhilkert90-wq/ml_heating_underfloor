"""
Unified Thermal State Management - Single JSON File Architecture

This module provides a clean, unified approach to managing all thermal system state
in a single JSON file, eliminating the need for pickle files and multiple JSON files.

Key features:
- Single source of truth for all thermal parameters
- Human-readable JSON format
- Version control friendly
- Calibrated baseline + online learning state
- Robust error handling and validation
- Compressed history storage (prediction, parameter, channel histories)
"""

import json
import os
import logging
from copy import deepcopy
from typing import Dict, Any, Optional, Set
from datetime import datetime
import numpy as np

# Import centralized thermal configuration
try:
    from .thermal_config import ThermalParameterConfig
    from .shadow_mode import get_effective_unified_state_file
except ImportError:
    from thermal_config import ThermalParameterConfig
    from shadow_mode import get_effective_unified_state_file


# ---------------------------------------------------------------------------
# Context / history field allow-lists
# ---------------------------------------------------------------------------
# Context fields persisted in prediction_history.  Includes both the subset
# needed for gradient replay AND decision-relevant sensor readings so the
# history makes the system's choices visible/auditable.
PREDICTION_CONTEXT_KEYS: Set[str] = {
    # Gradient replay essentials
    "outlet_temp", "outdoor_temp", "pv_power", "fireplace_on", "tv_on",
    "current_indoor", "thermal_power", "avg_cloud_cover", "inlet_temp",
    "delta_t", "outdoor_forecast", "pv_forecast",
    # Decision-visibility readings
    "target_temp", "heat_pump_active", "indoor_temp_delta_60m",
    "living_room_temp", "indoor_temp_gradient",
}

# Fields kept in parameter_history entries.  Triple-stored duplicates
# (parameters_before, parameters_after, channel_parameter_changes) are
# removed — the ``changes`` dict plus the flat snapshot already capture
# everything.
PARAMETER_HISTORY_KEYS: Set[str] = {
    "timestamp", "record_type", "channel", "learning_rate",
    "learning_confidence", "avg_recent_error", "raw_prediction_error",
    "attributed_error", "attribution_applied", "active_contributions",
    "heat_pump_frozen_by_fireplace", "gradients", "changes",
    # Full parameter snapshot — all channel params for visibility
    "thermal_time_constant", "heat_loss_coefficient", "outlet_effectiveness",
    "pv_heat_weight", "tv_heat_weight", "solar_lag_minutes",
    "slab_time_constant_hours", "delta_t_floor",
    "cloud_factor_exponent", "solar_decay_tau_hours",
    "fp_heat_output_kw", "fp_decay_time_constant",
    "room_spread_delay_minutes",
}

# Context fields kept in heat-source channel history entries.
CHANNEL_HISTORY_CONTEXT_KEYS: Set[str] = {
    "raw_prediction_error", "attributed_error", "attribution_applied",
    "active_contributions", "heat_pump_frozen_by_fireplace",
    "fireplace_on", "delta_t", "avg_cloud_cover",
    "pv_power", "pv_power_current", "pv_power_history",
    "thermal_power", "heat_pump_active",
    # Decision-visibility readings
    "outlet_temp", "outdoor_temp", "current_indoor", "inlet_temp",
    "target_temp", "tv_on",
}


def _slim_prediction_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Return a trimmed copy of a prediction context for persistence."""
    return {k: v for k, v in context.items() if k in PREDICTION_CONTEXT_KEYS}


def _slim_parameter_history_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a trimmed copy of a parameter-history record for persistence."""
    return {k: v for k, v in record.items() if k in PARAMETER_HISTORY_KEYS}


def _slim_channel_history_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a trimmed copy of a channel-history entry for persistence.

    Keeps ``error``, slimmed ``context``, current ``parameters``, and
    ``changes``.  Drops the redundant ``parameters_before`` and
    ``parameters_after`` (derivable from ``parameters`` and ``changes``).
    """
    slimmed: Dict[str, Any] = {"error": entry.get("error")}
    ctx = entry.get("context")
    if ctx is not None:
        slimmed["context"] = {
            k: v for k, v in ctx.items() if k in CHANNEL_HISTORY_CONTEXT_KEYS
        }
    # Keep channel parameters and change deltas for visibility
    if "parameters" in entry:
        slimmed["parameters"] = entry["parameters"]
    if "changes" in entry:
        slimmed["changes"] = entry["changes"]
    return slimmed


class ThermalStateManager:
    """Unified thermal state manager using single JSON file.

    Replaces legacy pickle files and separate JSON configurations
    with single thermal_state.json containing everything.
    """

    def __init__(self, state_file: Optional[str] = None):
        self.state_file = state_file or get_effective_unified_state_file()
        self.state = self._get_default_state()

        # Load existing state if available
        if os.path.exists(self.state_file):
            self.load_state()
        else:
            logging.info("🆕 Creating new unified thermal state file")

    def _get_default_state(self) -> Dict[str, Any]:
        """Get default thermal state structure."""
        return {
            "metadata": {
                "version": "1.0",
                "format": "unified_thermal_state",
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "baseline_parameters": {
                "thermal_time_constant":
                ThermalParameterConfig.get_default('thermal_time_constant'),
                "equilibrium_ratio":
                ThermalParameterConfig.get_default('equilibrium_ratio'),
                "total_conductance":
                ThermalParameterConfig.get_default('total_conductance'),
                "heat_loss_coefficient":
                ThermalParameterConfig.get_default('heat_loss_coefficient'),
                "outlet_effectiveness":
                ThermalParameterConfig.get_default('outlet_effectiveness'),
                "solar_lag_minutes":
                ThermalParameterConfig.get_default('solar_lag_minutes'),
                "slab_time_constant_hours":
                ThermalParameterConfig.get_default('slab_time_constant_hours'),
                "pv_heat_weight":
                ThermalParameterConfig.get_default('pv_heat_weight'),
                "fireplace_heat_weight":
                ThermalParameterConfig.get_default('fireplace_heat_weight'),
                "tv_heat_weight":
                ThermalParameterConfig.get_default('tv_heat_weight'),
                "delta_t_floor":
                ThermalParameterConfig.get_default('delta_t_floor'),
                "fp_decay_time_constant":
                ThermalParameterConfig.get_default('fp_decay_time_constant'),
                "room_spread_delay_minutes":
                ThermalParameterConfig.get_default(
                    'room_spread_delay_minutes'),
                "source": "config_defaults",
                "calibration_date": None,
                "calibration_cycles": 0
            },
            "learning_state": {
                "cycle_count": 0,
                "learning_confidence": 3.0,
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
                    "room_spread_delay_minutes_delta": 0.0
                },
                "parameter_bounds": {
                    "equilibrium_ratio":
                    list(ThermalParameterConfig.get_bounds(
                        'equilibrium_ratio')),
                    "total_conductance":
                    list(ThermalParameterConfig.get_bounds(
                        'total_conductance'))
                },
                "heat_source_channels": {},
                "prediction_history": [],
                "parameter_history": []
            },
            "prediction_metrics": {
                "total_predictions": 0,
                "accuracy_stats": {
                    "mae_1h": 0.0,
                    "rmse_1h": 0.0,
                    "mae_6h": 0.0,
                    "rmse_6h": 0.0,
                    "mae_24h": 0.0,
                    "rmse_24h": 0.0,
                    "mae_all_time": 0.0,
                    "rmse_all_time": 0.0
                },
                "recent_performance": {
                    "last_10_mae": 0.0,
                    "last_10_max_error": 0.0,
                    "last_10_count": 0
                }
            },
            "operational_state": {
                "last_indoor_temp": None,
                "last_outdoor_temp": None,
                "last_outlet_temp": None,
                "last_prediction": None,
                "last_run_time": None,
                "is_calibrating": False,
                # Added missing fields required by main.py learning logic
                "last_run_features": None,
                "last_final_temp": None,
                "last_avg_other_rooms_temp": None,
                "last_fireplace_on": False,
                "last_is_blocking": False,
                "last_blocking_reasons": [],
                "last_blocking_end_time": None
            }
        }

    def load_state(self) -> bool:
        """Load thermal state from JSON file."""
        try:
            with open(self.state_file, 'r') as f:
                loaded_state = json.load(f)

            # Validate and merge with default structure to handle schema migrations
            self.state = self._merge_with_defaults(loaded_state)

            # Migrate: compress legacy verbose history entries in-place
            self._compress_existing_histories()

            self.state["metadata"]["last_updated"] = datetime.now().isoformat()

            # Persist immediately so any new schema keys (migrations) are written
            # to the file without waiting for the next learning cycle save.
            self.save_state()

            logging.info("✅ Loaded unified thermal state from %s", self.state_file)
            return True

        except FileNotFoundError:
            logging.info("📋 No existing thermal state file found at %s",
                         self.state_file)
            return False
        except Exception as e:
            logging.error("❌ Failed to load thermal state: %s", e)
            self.state = self._get_default_state()
            return False

    def save_state(self) -> bool:
        """Save thermal state to JSON file (atomic write)."""
        import tempfile
        try:
            # Update metadata
            self.state["metadata"]["last_updated"] = datetime.now().isoformat()

            # Convert numpy types to native Python for JSON serialization
            serializable_state = self._convert_numpy_types(self.state)

            # Atomic write: write to temp file, then os.replace for safety
            dir_path = os.path.dirname(self.state_file) or '.'
            os.makedirs(dir_path, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                'w', dir=dir_path, delete=False, suffix='.tmp'
            ) as tmp_f:
                try:
                    import fcntl
                    fcntl.flock(tmp_f.fileno(), fcntl.LOCK_EX)
                except (ImportError, AttributeError, OSError):
                    pass  # fcntl not available on Windows
                json.dump(serializable_state, tmp_f, indent=2)
                tmp_path = tmp_f.name

            os.replace(tmp_path, self.state_file)

            logging.debug("💾 Saved unified thermal state to %s", self.state_file)
            return True

        except Exception as e:
            logging.error("❌ Failed to save thermal state: %s", e)
            return False

    def _merge_with_defaults(self, loaded_state: Dict) -> Dict:
        """Merge loaded state with default structure to handle schema changes."""
        merged = self._get_default_state()

        # Recursively merge loaded data
        def merge_dict(default: Dict, loaded: Dict) -> Dict:
            for key, value in loaded.items():
                if key in default:
                    if isinstance(default[key], dict) and \
                       isinstance(value, dict):
                        default[key] = merge_dict(default[key], value)
                    else:
                        default[key] = value
                else:
                    # Handle new keys from loaded state
                    default[key] = value
            return default

        return merge_dict(merged, loaded_state)

    def _compress_existing_histories(self) -> None:
        """Migrate legacy verbose history entries to compressed format.

        This runs once on load and trims fields that are no longer
        persisted.  It is idempotent: already-compressed entries pass
        through unchanged.
        """
        ls = self.state.get("learning_state", {})

        # 1. Compress prediction_history contexts
        pred_hist = ls.get("prediction_history", [])
        for i, entry in enumerate(pred_hist):
            ctx = entry.get("context")
            if ctx is not None:
                entry["context"] = _slim_prediction_context(ctx)
            # Drop fields that are never read after storage
            for key in ("model_internal_prediction",
                        "parameters_at_prediction",
                        "shadow_mode", "system_state",
                        "learning_quality", "abs_error",
                        "squared_error"):
                entry.pop(key, None)

        # 2. Compress parameter_history entries
        param_hist = ls.get("parameter_history", [])
        ls["parameter_history"] = [
            _slim_parameter_history_record(record)
            for record in param_hist
        ]

        # 3. Compress heat-source channel history entries
        channels = ls.get("heat_source_channels", {})
        for ch_name, ch_data in channels.items():
            history = ch_data.get("history")
            if isinstance(history, list):
                ch_data["history"] = [
                    _slim_channel_history_entry(entry)
                    for entry in history
                    if isinstance(entry, dict)
                ]

    def _convert_numpy_types(self, obj) -> Any:
        """Convert numpy types and pandas objects to native types."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value)
                    for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)  # Convert numpy boolean to Python boolean
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):
            try:
                if len(obj) == 1:
                    return obj.to_dict(orient='records')[0]
                else:
                    return obj.to_dict(orient='records')
            except Exception:
                logging.warning("Failed to convert pandas object to dict: %s",
                                type(obj))
                return None
        elif hasattr(obj, '__dict__') and not isinstance(
                obj, (str, int, float, bool)):
            logging.warning("Cannot serialize complex object of type %s, "
                            "storing as None", type(obj))
            return None
        else:
            return obj

    # === BASELINE PARAMETERS MANAGEMENT ===

    def set_calibrated_baseline(self, parameters: Dict[str, float],
                               calibration_cycles: int = 0) -> None:
        """Set calibrated baseline parameters from calibration process."""
        baseline = self.state["baseline_parameters"]

        # Update thermal parameters
        if "thermal_time_constant" in parameters:
            baseline["thermal_time_constant"] = parameters["thermal_time_constant"]
        if "equilibrium_ratio" in parameters:
            baseline["equilibrium_ratio"] = parameters["equilibrium_ratio"]
        if "total_conductance" in parameters:
            baseline["total_conductance"] = parameters["total_conductance"]
        if "heat_loss_coefficient" in parameters:
            baseline["heat_loss_coefficient"] = \
                parameters["heat_loss_coefficient"]

        if "outlet_effectiveness" in parameters:
            baseline["outlet_effectiveness"] = \
                parameters["outlet_effectiveness"]

        # Update heat source weights if provided
        if "pv_heat_weight" in parameters:
            baseline["pv_heat_weight"] = parameters["pv_heat_weight"]
        if "fireplace_heat_weight" in parameters:
            baseline["fireplace_heat_weight"] = \
                parameters["fireplace_heat_weight"]
        if "tv_heat_weight" in parameters:
            baseline["tv_heat_weight"] = parameters["tv_heat_weight"]
        if "solar_lag_minutes" in parameters:
            baseline["solar_lag_minutes"] = parameters["solar_lag_minutes"]
        if "slab_time_constant_hours" in parameters:
            baseline["slab_time_constant_hours"] = \
                parameters["slab_time_constant_hours"]
        if "delta_t_floor" in parameters:
            baseline["delta_t_floor"] = parameters["delta_t_floor"]
        if "fp_decay_time_constant" in parameters:
            baseline["fp_decay_time_constant"] = \
                parameters["fp_decay_time_constant"]
        if "room_spread_delay_minutes" in parameters:
            baseline["room_spread_delay_minutes"] = \
                parameters["room_spread_delay_minutes"]

        # Update metadata
        baseline["source"] = "calibrated"
        baseline["calibration_date"] = datetime.now().isoformat()
        baseline["calibration_cycles"] = calibration_cycles

        # Reset learning adjustments when new baseline is set
        self.state["learning_state"]["parameter_adjustments"] = {
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
            "room_spread_delay_minutes_delta": 0.0
        }
        self.state["learning_state"]["heat_source_channels"] = {}

        logging.info("🎯 Set calibrated baseline parameters (cycles: %s)",
                     calibration_cycles)
        self.save_state()

    def get_current_parameters(self) -> Dict[str, Any]:
        """Get complete thermal state structure."""
        # Return the complete state structure that thermal_equilibrium_model expects
        return self.state.copy()

    def get_computed_parameters(self) -> Dict[str, float]:
        """Get current thermal parameters (baseline + learning adjustments)."""
        baseline = self.state["baseline_parameters"]
        adjustments = self.state["learning_state"]["parameter_adjustments"]

        return {
            "thermal_time_constant": (baseline["thermal_time_constant"] +
                                      adjustments.get("thermal_time_constant_delta", 0.0)),
            "equilibrium_ratio": (baseline["equilibrium_ratio"] +
                                  adjustments["equilibrium_ratio_delta"]),
            "total_conductance": (baseline["total_conductance"] +
                                  adjustments["total_conductance_delta"]),
            "heat_loss_coefficient": (baseline["heat_loss_coefficient"] +
                                  adjustments.get("heat_loss_coefficient_delta", 0.0)),
            "outlet_effectiveness": (baseline["outlet_effectiveness"] +
                                  adjustments.get("outlet_effectiveness_delta", 0.0)),
            "pv_heat_weight": (baseline["pv_heat_weight"] +
                               adjustments.get("pv_heat_weight_delta", 0.0)),
            "fireplace_heat_weight": baseline["fireplace_heat_weight"],
            "tv_heat_weight": (baseline["tv_heat_weight"] +
                               adjustments.get("tv_heat_weight_delta", 0.0)),
            "solar_lag_minutes": (baseline.get("solar_lag_minutes",
                                   ThermalParameterConfig.get_default('solar_lag_minutes')) +
                                   adjustments.get("solar_lag_minutes_delta", 0.0)),
            "slab_time_constant_hours": (baseline.get("slab_time_constant_hours",
                                          ThermalParameterConfig.get_default('slab_time_constant_hours')) +
                                          adjustments.get("slab_time_constant_delta", 0.0))
        }

    # === LEARNING STATE MANAGEMENT ===

    def update_learning_state(self, cycle_count: int = None,
                            learning_confidence: float = None,
                            parameter_adjustments:
                            Dict[str, float] = None) -> None:
        """Update learning state parameters."""
        learning_state = self.state["learning_state"]

        if cycle_count is not None:
            learning_state["cycle_count"] = cycle_count
        if learning_confidence is not None:
            learning_state["learning_confidence"] = learning_confidence
        if parameter_adjustments is not None:
            for param, delta in parameter_adjustments.items():
                # Accept known keys plus new learnable params (slab, solar_lag)
                learning_state["parameter_adjustments"][param] = delta

        self.save_state()

    def add_prediction_record(self, prediction_record: Dict) -> None:
        """Add a prediction record to history.

        The record's ``context`` is slimmed down to the fields actually
        needed for gradient replay, removing ~60% of per-entry bloat.
        """
        # Slim the context before persisting
        slimmed = prediction_record.copy()
        if "context" in slimmed:
            slimmed["context"] = _slim_prediction_context(slimmed["context"])
        # Drop fields that are never read after storage
        for key in ("model_internal_prediction", "parameters_at_prediction",
                     "shadow_mode", "system_state", "learning_quality",
                     "abs_error", "squared_error"):
            slimmed.pop(key, None)

        history = self.state["learning_state"]["prediction_history"]
        history.append(slimmed)

        # Keep manageable history size (sliding window)
        if len(history) > 200:
            self.state["learning_state"]["prediction_history"] = history[-200:]

        # Update prediction count
        self.state["prediction_metrics"]["total_predictions"] += 1

    def add_parameter_history_record(self, parameter_record: Dict) -> None:
        """Add a parameter history record.

        Redundant fields (parameters_before, parameters_after,
        channel_parameter_changes, and flat duplicate params) are stripped
        before persisting.
        """
        slimmed = _slim_parameter_history_record(parameter_record)
        history = self.state["learning_state"]["parameter_history"]
        history.append(slimmed)

        # Keep manageable history size (sliding window)
        if len(history) > 500:
            self.state["learning_state"]["parameter_history"] = history[-500:]

    def get_heat_source_channel_state(self) -> Dict[str, Any]:
        """Get persisted heat-source channel state."""
        return deepcopy(
            self.state["learning_state"].get("heat_source_channels", {})
        )

    def set_heat_source_channel_state(self, channel_state: Dict[str, Any]) -> None:
        """Persist heat-source channel state.

        Channel history entries are slimmed before persisting: redundant
        ``parameters_before`` and ``parameters_after`` are dropped (they
        are derivable from ``parameters`` + ``changes``).  Context is
        trimmed to the fields used by channel learning and decision
        visibility.
        """
        compressed = {}
        for ch_name, ch_data in channel_state.items():
            compressed_ch = {}
            for key, value in ch_data.items():
                if key == "history" and isinstance(value, list):
                    compressed_ch[key] = [
                        _slim_channel_history_entry(entry)
                        for entry in value
                        if isinstance(entry, dict)
                    ]
                else:
                    compressed_ch[key] = deepcopy(value)
            compressed[ch_name] = compressed_ch

        self.state["learning_state"]["heat_source_channels"] = compressed
        self.save_state()

    # === OPERATIONAL STATE MANAGEMENT ===

    def update_operational_state(self, **kwargs) -> None:
        """Update operational state parameters."""
        operational = self.state["operational_state"]

        for key, value in kwargs.items():
            if key in operational:
                # Prevent last_run_features from being stored as a
                # non-dict type (e.g. DataFrame, JSON string, Series).
                # This is the root cause of the double-encoding bug
                # where features_dict gets serialized as a JSON string
                # and then re-serialized by json.dump().
                if key == "last_run_features" and value is not None:
                    if isinstance(value, str):
                        import json as _json
                        try:
                            value = _json.loads(value)
                        except (ValueError, TypeError):
                            logging.warning(
                                "last_run_features was a non-JSON string, "
                                "discarding"
                            )
                            value = {}
                    if hasattr(value, "to_dict") and not isinstance(
                        value, dict
                    ):
                        try:
                            if hasattr(value, "iloc"):
                                value = value.iloc[0].to_dict()
                            else:
                                value = value.to_dict()
                        except (
                            AttributeError,
                            IndexError,
                            KeyError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            logging.warning(
                                "Failed to normalize last_run_features via "
                                "to_dict(): %s",
                                exc,
                            )
                            value = {}
                    if not isinstance(value, dict):
                        logging.warning(
                            "last_run_features has unexpected type %s, "
                            "discarding",
                            type(value).__name__,
                        )
                        value = {}
                operational[key] = value

        operational["last_run_time"] = datetime.now().isoformat()

    def get_operational_state(self) -> Dict[str, Any]:
        """Get current operational state."""
        return self.state["operational_state"].copy()

    def set_calibration_mode(self, is_calibrating: bool) -> None:
        """Set calibration mode flag."""
        self.state["operational_state"]["is_calibrating"] = is_calibrating
        self.save_state()

    # === METRICS AND REPORTING ===

    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive learning metrics for monitoring."""
        learning_state = self.state["learning_state"]
        baseline = self.state["baseline_parameters"]
        metrics = self.state["prediction_metrics"]

        return {
            "baseline_source": baseline["source"],
            "calibration_date": baseline["calibration_date"],
            "calibration_cycles": baseline["calibration_cycles"],
            "current_cycle_count": learning_state["cycle_count"],
            "learning_confidence": learning_state["learning_confidence"],
            "total_predictions": metrics["total_predictions"],
            "current_parameters": self.get_current_parameters(),
            "parameter_adjustments":
            learning_state["parameter_adjustments"].copy(),
            "heat_source_channels": self.get_heat_source_channel_state(),
            "accuracy_stats": metrics["accuracy_stats"].copy(),
            "learning_enabled": learning_state["learning_enabled"]
        }

    def reset_learning_state(self, keep_baseline: bool = True) -> None:
        """Reset learning state while optionally preserving baseline."""
        if keep_baseline:
            # Reset only learning components
            self.state["learning_state"]["cycle_count"] = 0
            self.state["learning_state"]["learning_confidence"] = 3.0
            self.state["learning_state"]["prediction_history"] = []
            self.state["learning_state"]["parameter_history"] = []
            self.state["learning_state"]["heat_source_channels"] = {}

            # Reset parameter adjustments
            adjustments = self.state["learning_state"]["parameter_adjustments"]
            for key in adjustments:
                adjustments[key] = 0.0
        else:
            # Reset everything including baseline
            self.state = self._get_default_state()

        logging.info("🔄 Reset learning state (baseline preserved: %s)",
                     keep_baseline)
        self.save_state()

    # === BACKUP AND RESTORE FUNCTIONALITY ===

    def create_backup(self, backup_name: str = "") -> tuple[bool, str]:
        """Create a backup of the current thermal state."""
        try:
            if not backup_name:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f'thermal_state_backup_{timestamp}'

            # Ensure backup has .json extension
            if not backup_name.endswith('.json'):
                backup_name += '.json'

            backup_path = os.path.dirname(self.state_file)
            backup_file = os.path.join(backup_path, backup_name)

            # Copy current state to backup file
            import shutil
            shutil.copy2(self.state_file, backup_file)

            logging.info("🔄 Created thermal state backup: %s", backup_name)
            return True, backup_file

        except Exception as e:
            logging.error("❌ Failed to create thermal state backup: %s", e)
            return False, str(e)

    def restore_from_backup(self, backup_file: str) -> tuple[bool, str]:
        """Restore thermal state from a backup file."""
        try:
            if not os.path.exists(backup_file):
                return False, f"Backup file not found: {backup_file}"

            # Create safety backup before restore
            safety_backup_name = \
                f'pre_restore_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            self.create_backup(safety_backup_name)

            # Copy backup to current state file
            import shutil
            shutil.copy2(backup_file, self.state_file)

            # Reload state from restored file
            self.load_state()

            logging.info("🔄 Restored thermal state from backup: %s",
                         os.path.basename(backup_file))
            return (True, "Thermal state restored successfully. "
                    f"Safety backup created as {safety_backup_name}.json")

        except Exception as e:
            logging.error("❌ Failed to restore from backup: %s", e)
            return False, str(e)

    def list_backups(self) -> list[dict]:
        """List all available thermal state backups."""
        try:
            backup_path = os.path.dirname(self.state_file)
            backups = []

            # Look for backup files in same directory as state file
            for file in os.listdir(backup_path):
                if file.startswith('thermal_state_backup_') and \
                   file.endswith('.json'):
                    file_path = os.path.join(backup_path, file)
                    stat = os.stat(file_path)

                    backups.append({
                        'name': file,
                        'path': file_path,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_mtime),
                        'type': 'thermal_state_backup'
                    })

            return sorted(backups, key=lambda x: x['created'], reverse=True)

        except Exception as e:
            logging.error("❌ Failed to list backups: %s", e)
            return []


# Global instance for easy access
_thermal_state_manager = None


def get_thermal_state_manager() -> ThermalStateManager:
    """Get global thermal state manager instance."""
    global _thermal_state_manager
    if _thermal_state_manager is None:
        _thermal_state_manager = ThermalStateManager()
    return _thermal_state_manager


def save_thermal_state(**kwargs) -> None:
    """Convenience function to save thermal state."""
    manager = get_thermal_state_manager()
    manager.update_operational_state(**kwargs)
    manager.save_state()


def load_thermal_state() -> Dict[str, Any]:
    """Convenience function to load thermal state."""
    manager = get_thermal_state_manager()
    manager.load_state()
    return manager.state