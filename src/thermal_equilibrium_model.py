"""
Thermal Equilibrium Model with Adaptive Learning.

This module defines the core physics-based model for predicting thermal
equilibrium and adapting its parameters in real-time based on prediction
accuracy. It combines a heat balance equation with a gradient-based
learning mechanism to continuously improve its accuracy.
"""

import numpy as np
import logging
from typing import Dict, List, Optional

# MIGRATION: Use unified thermal parameter system
try:
    from .thermal_parameters import thermal_params  # type: ignore
    from .thermal_constants import PhysicsConstants  # type: ignore
    from .thermal_config import ThermalParameterConfig  # type: ignore
    from . import config  # type: ignore
except ImportError:
    # Direct import fallback for notebooks and standalone tests
    from thermal_parameters import thermal_params  # type: ignore
    from thermal_constants import PhysicsConstants  # type: ignore
    from thermal_config import ThermalParameterConfig  # type: ignore
    import config  # type: ignore


class ThermalEquilibriumModel:
    """
    A physics-based thermal model that predicts indoor temperature equilibrium
    and adapts its parameters based on real-world feedback.
    """

    def __init__(self):
        # Initialize default attributes to ensure existence
        self.solar_lag_minutes = config.SOLAR_LAG_MINUTES
        self.slab_time_constant_hours = config.SLAB_TIME_CONSTANT_HOURS

        # Load calibrated parameters first, fallback to config defaults
        self._load_thermal_parameters()

        # Phase 2-4: Heat Source Channel Orchestrator (Steps 10-11)
        # Routes learning updates to the correct channel, preventing
        # cross-contamination between heat sources.
        self.orchestrator = None
        if config.ENABLE_HEAT_SOURCE_CHANNELS:
            try:
                from src.heat_source_channels import (
                    HeatSourceChannelOrchestrator,
                )

                self.orchestrator = HeatSourceChannelOrchestrator()
                self._initialize_heat_source_channels()
            except Exception as exc:
                logging.warning(
                    "⚠️ Could not init heat source orchestrator: %s", exc
                )

        # self.outdoor_coupling = config.OUTDOOR_COUPLING
        # thermal_bridge_factor removed in Phase 2: was not used in
        # calculations

    def _load_thermal_parameters(self):
        """
        Load thermal parameters with proper baseline + adjustments.
        This ensures trained parameters persist across restarts.
        """
        try:
            # Try to load calibrated parameters from unified thermal state
            try:
                from .unified_thermal_state import get_thermal_state_manager
            except ImportError:
                from unified_thermal_state import get_thermal_state_manager

            state_manager = get_thermal_state_manager()
            thermal_state = state_manager.get_current_parameters()

            # Check for calibrated parameters in baseline_parameters section
            baseline_params = thermal_state.get("baseline_parameters", {})
            if baseline_params.get("source") == "calibrated":
                # Load baseline + adjustments for trained parameters
                learning_state = thermal_state.get("learning_state", {})
                adjustments = learning_state.get("parameter_adjustments", {})

                # Apply learning adjustments to baseline
                self.thermal_time_constant = (
                    baseline_params["thermal_time_constant"]
                    + adjustments.get("thermal_time_constant_delta", 0.0)
                )
                self.heat_loss_coefficient = (
                    baseline_params["heat_loss_coefficient"]
                    + adjustments.get("heat_loss_coefficient_delta", 0.0)
                )
                self._baseline_heat_loss_coefficient = (
                    baseline_params["heat_loss_coefficient"]
                )
                self.outlet_effectiveness = (
                    baseline_params["outlet_effectiveness"]
                    + adjustments.get("outlet_effectiveness_delta", 0.0)
                )

                self.solar_lag_minutes = (
                    baseline_params.get(
                        "solar_lag_minutes", config.SOLAR_LAG_MINUTES
                    )
                    + adjustments.get("solar_lag_minutes_delta", 0.0)
                )

                self.slab_time_constant_hours = (
                    baseline_params.get(
                        "slab_time_constant_hours",
                        config.SLAB_TIME_CONSTANT_HOURS,
                    )
                    + adjustments.get("slab_time_constant_delta", 0.0)
                )

                self.external_source_weights = {
                    "pv": baseline_params.get(
                        "pv_heat_weight", config.PV_HEAT_WEIGHT
                    ) + adjustments.get("pv_heat_weight_delta", 0.0),
                    "fireplace": baseline_params.get(
                        "fireplace_heat_weight", config.FIREPLACE_HEAT_WEIGHT,
                    ) + adjustments.get("fireplace_heat_weight_delta", 0.0),
                    "tv": baseline_params.get(
                        "tv_heat_weight", config.TV_HEAT_WEIGHT
                    ) + adjustments.get("tv_heat_weight_delta", 0.0),
                }

                self.delta_t_floor = (
                    baseline_params.get(
                        "delta_t_floor",
                        ThermalParameterConfig.get_default('delta_t_floor'),
                    )
                    + adjustments.get("delta_t_floor_delta", 0.0)
                )
                self.fp_decay_time_constant = (
                    baseline_params.get(
                        "fp_decay_time_constant",
                        ThermalParameterConfig.get_default(
                            'fp_decay_time_constant'),
                    )
                    + adjustments.get(
                        "fp_decay_time_constant_delta", 0.0)
                )
                self.room_spread_delay_minutes = (
                    baseline_params.get(
                        "room_spread_delay_minutes",
                        ThermalParameterConfig.get_default(
                            'room_spread_delay_minutes'),
                    )
                    + adjustments.get(
                        "room_spread_delay_minutes_delta", 0.0)
                )

                logging.info(
                    "🎯 Loading CALIBRATED thermal parameters "
                    "(baseline + learning adjustments):"
                )
                logging.info(
                    "   heat_loss_coefficient: %.4f + %.5f = %.4f",
                    baseline_params["heat_loss_coefficient"],
                    adjustments.get("heat_loss_coefficient_delta", 0.0),
                    self.heat_loss_coefficient,
                )
                logging.info(
                    "   outlet_effectiveness: %.3f + %.3f = %.3f",
                    baseline_params["outlet_effectiveness"],
                    adjustments.get("outlet_effectiveness_delta", 0.0),
                    self.outlet_effectiveness,
                )
                logging.info(
                    "   pv_heat_weight: %.4f",
                    self.external_source_weights["pv"]
                )

                # Validate parameters using schema validator
                try:
                    from thermal_state_validator import (
                        validate_thermal_state_safely,
                    )

                    if not validate_thermal_state_safely(thermal_state):
                        logging.warning(
                            "⚠️ Thermal state validation failed, but core "
                            "parameters were loaded. Retaining calibrated "
                            "parameters and relying on safety clamping."
                        )
                        # We do NOT revert to defaults here because we
                        # successfully loaded the baseline parameters above.
                        # Reverting would lose calibration due to minor schema
                        # mismatches (e.g. missing optional fields).
                        # The code below will clamp parameters to safe bounds.
                except ImportError:
                    logging.debug("Schema validation not available")

                # Initialize learning attributes
                self._initialize_learning_attributes()

                # Restore learning history from saved state
                self.learning_confidence = max(
                    learning_state.get("learning_confidence", 3.0), 0.1
                )
                self.prediction_history = list(learning_state.get(
                    "prediction_history", []
                ))
                # Use list() to get an independent copy – avoids the shared-
                # reference trap where self.parameter_history and
                # state["learning_state"]["parameter_history"] are the same
                # object, causing the trimming slice
                # (self.parameter_history = self.parameter_history[-500:])
                # to silently disconnect the two lists and lose future records.
                self.parameter_history = list(learning_state.get(
                    "parameter_history", []
                ))

                logging.info(
                    "   - Restored learning confidence: %.3f",
                    self.learning_confidence,
                )
                logging.info(
                    "   - Restored prediction history: %d records",
                    len(self.prediction_history),
                )
                logging.info(
                    "   - Restored parameter history: %d records",
                    len(self.parameter_history),
                )

                # EXTRA SAFETY: Clamp parameters to current bounds
                # This handles cases where bounds were tightened in config
                # but old parameters are still within the old (wider) bounds
                # but outside the new (tighter) bounds.
                try:
                    # Clamp heat_loss_coefficient
                    hcl_bounds = ThermalParameterConfig.get_bounds(
                        "heat_loss_coefficient"
                    )
                    if not (
                        hcl_bounds[0]
                        <= self.heat_loss_coefficient
                        <= hcl_bounds[1]
                    ):
                        logging.warning(
                            "⚠️ Clamping heat_loss_coefficient %.4f to "
                            "bounds %s",
                            self.heat_loss_coefficient,
                            hcl_bounds,
                        )
                        self.heat_loss_coefficient = max(
                            hcl_bounds[0],
                            min(self.heat_loss_coefficient, hcl_bounds[1]),
                        )

                    # Clamp outlet_effectiveness
                    oe_bounds = ThermalParameterConfig.get_bounds(
                        "outlet_effectiveness"
                    )
                    if not (
                        oe_bounds[0]
                        <= self.outlet_effectiveness
                        <= oe_bounds[1]
                    ):
                        logging.warning(
                            "⚠️ Clamping outlet_effectiveness %.4f to "
                            "bounds %s",
                            self.outlet_effectiveness,
                            oe_bounds,
                        )
                        self.outlet_effectiveness = max(
                            oe_bounds[0],
                            min(self.outlet_effectiveness, oe_bounds[1]),
                        )

                except Exception as e:
                    logging.error(f"Error during parameter clamping: {e}")

                # STABILITY FIX: Detect and reset corrupted parameters on load
                if self._detect_parameter_corruption():
                    logging.warning(
                        "🗑️ Detected corrupted parameters on load. "
                        "Resetting to defaults and clearing learning state."
                    )
                    logging.warning(
                        "   Corrupted values - Heat Loss: %.4f, "
                        "Effectiveness: %.4f",
                        self.heat_loss_coefficient,
                        self.outlet_effectiveness
                    )

                    # CRITICAL: Reset persistent state to prevent reloading
                    # bad deltas
                    
                    # Check if baseline itself is corrupted
                    # If baseline is bad, we MUST wipe it, otherwise it
                    # persists
                    baseline_hl = baseline_params.get(
                        "heat_loss_coefficient", 0
                    )
                    baseline_oe = baseline_params.get(
                        "outlet_effectiveness", 1
                    )

                    is_baseline_corrupted = (
                        baseline_hl > 1.2
                        or baseline_oe < 0.4
                        or (
                            baseline_hl > 0.8 and baseline_oe < 0.35
                        )  # Specific bad combo
                    )

                    if is_baseline_corrupted:
                        logging.error(
                            "🚨 BASELINE IS CORRUPTED! (HL=%.2f, Eff=%.2f). "
                            "Wiping entire thermal state including baseline.",
                            baseline_hl,
                            baseline_oe,
                        )
                        state_manager.reset_learning_state(keep_baseline=False)
                    else:
                        logging.warning(
                            "⚠️ Baseline seems okay, only resetting "
                            "learning state."
                        )
                        state_manager.reset_learning_state(keep_baseline=True)

                    self._load_config_defaults()
                    # CRITICAL: Return to avoid using corrupted state
                    return

            else:
                # Use config defaults
                self._load_config_defaults()
                logging.info(
                    "⚙️ Loading DEFAULT config parameters "
                    "(no calibration found)"
                )

        except Exception as e:
            # Fallback to config defaults if thermal state unavailable
            logging.warning(f"⚠️ Failed to load calibrated parameters: {e}")
            self._load_config_defaults()
            logging.info("⚙️ Using config defaults as fallback")

    def _load_config_defaults(self):
        """MIGRATED: Load thermal parameters from unified parameter system."""
        self.thermal_time_constant = thermal_params.get(
            "thermal_time_constant"
        )
        self.heat_loss_coefficient = thermal_params.get(
            "heat_loss_coefficient"
        )
        self._baseline_heat_loss_coefficient = self.heat_loss_coefficient
        self.outlet_effectiveness = thermal_params.get("outlet_effectiveness")
        self.solar_lag_minutes = config.SOLAR_LAG_MINUTES
        self.slab_time_constant_hours = config.SLAB_TIME_CONSTANT_HOURS

        self.external_source_weights = {
            "pv": thermal_params.get("pv_heat_weight"),
            "fireplace": thermal_params.get("fireplace_heat_weight"),
            "tv": thermal_params.get("tv_heat_weight"),
        }

        self.delta_t_floor = ThermalParameterConfig.get_default(
            'delta_t_floor')
        self.fp_decay_time_constant = ThermalParameterConfig.get_default(
            'fp_decay_time_constant')
        self.room_spread_delay_minutes = ThermalParameterConfig.get_default(
            'room_spread_delay_minutes')

        # Initialize remaining attributes
        self._initialize_learning_attributes()

    def _initialize_heat_source_channels(self) -> None:
        """Seed and restore heat-source channel state."""
        if self.orchestrator is None:
            return

        self._seed_orchestrator_from_model_state()

        try:
            try:
                from .unified_thermal_state import get_thermal_state_manager
            except ImportError:
                from unified_thermal_state import get_thermal_state_manager

            state_manager = get_thermal_state_manager()
            persisted_state = state_manager.get_heat_source_channel_state()
            if persisted_state:
                # Pass calibration date so load_channel_state can decide
                # whether a fresh calibration should override channel-
                # learned params or whether to restore them.
                thermal_state = state_manager.get_current_parameters()
                cal_date = thermal_state.get(
                    "baseline_parameters", {}
                ).get("calibration_date")
                self.orchestrator.load_channel_state(
                    persisted_state,
                    baseline_calibration_date=cal_date,
                )
                logging.info(
                    "🔥 Restored heat-source channel state from unified thermal state"
                )
        except Exception as exc:
            logging.warning(
                "⚠️ Failed to restore heat-source channel state: %s", exc
            )

        self._sync_model_from_orchestrator()

    def _seed_orchestrator_from_model_state(self) -> None:
        """Initialize channel parameters from the current model state."""
        if self.orchestrator is None:
            return

        self.orchestrator.sync_from_model_parameters(
            {
                "thermal_time_constant": self.thermal_time_constant,
                "heat_loss_coefficient": self.heat_loss_coefficient,
                "outlet_effectiveness": self.outlet_effectiveness,
                "slab_time_constant_hours": self.slab_time_constant_hours,
                "pv_heat_weight": self.pv_heat_weight,
                "solar_lag_minutes": self.solar_lag_minutes,
                "fireplace_heat_weight": self.fireplace_heat_weight,
                "tv_heat_weight": self.tv_heat_weight,
                "delta_t_floor": self.delta_t_floor,
                "fp_decay_time_constant": self.fp_decay_time_constant,
                "room_spread_delay_minutes":
                    self.room_spread_delay_minutes,
            }
        )

    def sync_heat_source_channels_from_model_state(
        self, persist: bool = False
    ) -> None:
        """Reseed channel mode from the current model values.

        Calibration code mutates model parameters directly. When channel mode
        is enabled, predictions subsequently sync from the orchestrator, so we
        must explicitly reseed the orchestrator after those mutations.
        """
        if self.orchestrator is None:
            return

        self._seed_orchestrator_from_model_state()
        self._sync_model_from_orchestrator()
        if persist:
            self._persist_heat_source_channel_state()

    def _sync_model_from_orchestrator(self) -> None:
        """Treat channel state as the source of truth when enabled."""
        if self.orchestrator is None:
            return

        parameters = self.orchestrator.export_model_parameters()
        self.thermal_time_constant = parameters["thermal_time_constant"]
        self.heat_loss_coefficient = parameters["heat_loss_coefficient"]
        self.outlet_effectiveness = parameters["outlet_effectiveness"]
        self.slab_time_constant_hours = parameters[
            "slab_time_constant_hours"
        ]
        self.pv_heat_weight = parameters["pv_heat_weight"]
        self.solar_lag_minutes = parameters["solar_lag_minutes"]
        self.fireplace_heat_weight = parameters["fireplace_heat_weight"]
        self.tv_heat_weight = parameters["tv_heat_weight"]

    def _persist_heat_source_channel_state(
        self, parameter_history_records: Optional[List[Dict]] = None
    ) -> None:
        """Persist channel state and any new channel update records."""
        if self.orchestrator is None:
            return

        try:
            try:
                from .unified_thermal_state import get_thermal_state_manager
            except ImportError:
                from unified_thermal_state import get_thermal_state_manager

            state_manager = get_thermal_state_manager()
            if parameter_history_records:
                for parameter_record in parameter_history_records:
                    state_manager.add_parameter_history_record(
                        parameter_record
                    )
            state_manager.set_heat_source_channel_state(
                self.orchestrator.get_channel_state()
            )
        except Exception as exc:
            logging.warning(
                "⚠️ Failed to persist heat-source channel state: %s", exc
            )

    def _sync_orchestrator_parameter_if_needed(
        self, values: Dict[str, float]
    ) -> None:
        """Keep channel mode aligned when tracked model parameters are assigned directly."""
        orchestrator = getattr(self, "orchestrator", None)
        if orchestrator is None:
            return
        orchestrator.sync_from_model_parameters(values)

    def _get_channel_parameter_value(
        self, channel_name: str, parameter_name: str, fallback: float
    ) -> float:
        """Read a live channel-owned parameter without trusting stale exports."""
        orchestrator = getattr(self, "orchestrator", None)
        if orchestrator is None:
            return fallback

        channel = orchestrator.channels.get(channel_name)
        if channel is None or not hasattr(channel, parameter_name):
            return fallback

        try:
            return float(getattr(channel, parameter_name))
        except (TypeError, ValueError):
            return fallback

    def _resolve_delta_t_floor(self, observed_delta_t: float) -> float:
        """Prefer the observed loop delta-T when present, else fall back to HP channel state."""
        if observed_delta_t >= 1.0:
            return observed_delta_t
        return self._get_channel_parameter_value(
            "heat_pump", "delta_t_floor", 2.0
        )

    @property
    def thermal_time_constant(self) -> float:
        """Get thermal time constant."""
        return getattr(self, "_thermal_time_constant", 0.0)

    @thermal_time_constant.setter
    def thermal_time_constant(self, value: float):
        """Set thermal time constant and keep channel mode in sync."""
        self._thermal_time_constant = value
        self._sync_orchestrator_parameter_if_needed(
            {"thermal_time_constant": value}
        )

    @property
    def heat_loss_coefficient(self) -> float:
        """Get heat loss coefficient."""
        return getattr(self, "_heat_loss_coefficient", 0.0)

    @heat_loss_coefficient.setter
    def heat_loss_coefficient(self, value: float):
        """Set heat loss coefficient and keep channel mode in sync."""
        self._heat_loss_coefficient = value
        self._sync_orchestrator_parameter_if_needed(
            {"heat_loss_coefficient": value}
        )

    @property
    def outlet_effectiveness(self) -> float:
        """Get outlet effectiveness."""
        return getattr(self, "_outlet_effectiveness", 0.0)

    @outlet_effectiveness.setter
    def outlet_effectiveness(self, value: float):
        """Set outlet effectiveness and keep channel mode in sync."""
        self._outlet_effectiveness = value
        self._sync_orchestrator_parameter_if_needed(
            {"outlet_effectiveness": value}
        )

    @property
    def slab_time_constant_hours(self) -> float:
        """Get slab time constant."""
        return getattr(self, "_slab_time_constant_hours", 0.0)

    @slab_time_constant_hours.setter
    def slab_time_constant_hours(self, value: float):
        """Set slab time constant and keep channel mode in sync."""
        self._slab_time_constant_hours = value
        self._sync_orchestrator_parameter_if_needed(
            {"slab_time_constant_hours": value}
        )

    @property
    def solar_lag_minutes(self) -> float:
        """Get solar lag minutes."""
        return getattr(self, "_solar_lag_minutes", 0.0)

    @solar_lag_minutes.setter
    def solar_lag_minutes(self, value: float):
        """Set solar lag and keep channel mode in sync."""
        self._solar_lag_minutes = value
        self._sync_orchestrator_parameter_if_needed(
            {"solar_lag_minutes": value}
        )

    @property
    def pv_heat_weight(self) -> float:
        """Get PV heat weight."""
        return self.external_source_weights.get("pv", 0.0)

    @pv_heat_weight.setter
    def pv_heat_weight(self, value: float):
        """Set PV heat weight."""
        self.external_source_weights["pv"] = value
        self._sync_orchestrator_parameter_if_needed({"pv_heat_weight": value})

    @property
    def tv_heat_weight(self) -> float:
        """Get TV heat weight."""
        return self.external_source_weights.get("tv", 0.0)

    @tv_heat_weight.setter
    def tv_heat_weight(self, value: float):
        """Set TV heat weight."""
        self.external_source_weights["tv"] = value
        self._sync_orchestrator_parameter_if_needed({"tv_heat_weight": value})

    @property
    def fireplace_heat_weight(self) -> float:
        """Get Fireplace heat weight."""
        return self.external_source_weights.get("fireplace", 0.0)

    @fireplace_heat_weight.setter
    def fireplace_heat_weight(self, value: float):
        """Set Fireplace heat weight."""
        self.external_source_weights["fireplace"] = value
        self._sync_orchestrator_parameter_if_needed(
            {"fireplace_heat_weight": value}
        )

    def _initialize_learning_attributes(self):
        """Initialize adaptive learning and other attributes."""
        self.adaptive_learning_enabled = True
        self.safety_margin = PhysicsConstants.DEFAULT_SAFETY_MARGIN
        self.prediction_horizon_hours = (
            PhysicsConstants.DEFAULT_PREDICTION_HORIZON
        )
        self.momentum_decay_rate = PhysicsConstants.MOMENTUM_DECAY_RATE

        self.learning_rate = (
            thermal_params.get("adaptive_learning_rate")
            or PhysicsConstants.DEFAULT_LEARNING_RATE
        )
        self.equilibrium_samples = []
        self.trajectory_samples = []
        self.overshoot_events = []

        self.prediction_errors = []
        self.mode_switch_history = []
        self.overshoot_prevention_count = 0

        self.prediction_history: List[Dict] = []
        self.parameter_history: List[Dict] = []
        self.learning_confidence = PhysicsConstants.INITIAL_LEARNING_CONFIDENCE
        self.min_learning_rate = (
            thermal_params.get("min_learning_rate")
            or PhysicsConstants.MIN_LEARNING_RATE
        )
        self.max_learning_rate = (
            thermal_params.get("max_learning_rate")
            or PhysicsConstants.MAX_LEARNING_RATE
        )
        self.confidence_decay_rate = PhysicsConstants.CONFIDENCE_DECAY_RATE
        self.confidence_boost_rate = PhysicsConstants.CONFIDENCE_BOOST_RATE
        self.recent_errors_window = config.RECENT_ERRORS_WINDOW

        self.thermal_time_constant_bounds = ThermalParameterConfig.get_bounds(
            "thermal_time_constant"
        )
        self.heat_loss_coefficient_bounds = ThermalParameterConfig.get_bounds(
            "heat_loss_coefficient"
        )
        self.outlet_effectiveness_bounds = ThermalParameterConfig.get_bounds(
            "outlet_effectiveness"
        )
        self.pv_heat_weight_bounds = ThermalParameterConfig.get_bounds(
            "pv_heat_weight"
        )
        self.tv_heat_weight_bounds = ThermalParameterConfig.get_bounds(
            "tv_heat_weight"
        )
        self.solar_lag_minutes_bounds = ThermalParameterConfig.get_bounds(
            "solar_lag_minutes"
        )
        self.slab_time_constant_bounds = ThermalParameterConfig.get_bounds(
            "slab_time_constant_hours"
        )
        # Clamp persisted value to current bounds (e.g. after a bounds update)
        self.slab_time_constant_hours = float(np.clip(
            self.slab_time_constant_hours,
            self.slab_time_constant_bounds[0],
            self.slab_time_constant_bounds[1],
        ))

        self.parameter_stability_threshold = (
            PhysicsConstants.THERMAL_STABILITY_THRESHOLD
        )
        self.error_improvement_threshold = (
            PhysicsConstants.ERROR_IMPROVEMENT_THRESHOLD
        )

    def _calculate_effective_solar(
        self, pv_input, lag_minutes: float = None
    ) -> float:
        """
        Calculate effective solar power using a rolling average window.
        Uses fractional steps to ensure differentiability for optimization.

        Args:
            pv_input: Current PV power (float) or history list (List[float])
            lag_minutes: Window size in minutes (defaults to self.solar_lag_minutes)

        Returns:
            Effective PV power (float)
        """
        if lag_minutes is None:
            lag_minutes = self.solar_lag_minutes

        # Handle scalar input or empty list
        if isinstance(pv_input, (int, float)):
            return float(pv_input)
        if not pv_input:
            return 0.0

        # Avoid division by zero for very small lags
        if lag_minutes < 1e-3:
            return float(pv_input[-1])

        # Calculate number of steps to average (fractional)
        step_minutes = config.HISTORY_STEP_MINUTES
        float_steps = lag_minutes / step_minutes

        # Split into full steps and fractional remainder
        num_full_steps = int(float_steps)
        fraction = float_steps - num_full_steps

        total_val = 0.0

        # Add full steps (most recent N steps)
        if num_full_steps > 0:
            full_window = pv_input[-num_full_steps:]
            total_val += sum(full_window)

        # Add fractional step (the one just before the full window)
        if fraction > 0:
            # Index for the partial step is -(num_full_steps + 1)
            partial_idx = -(num_full_steps + 1)
            if abs(partial_idx) <= len(pv_input):
                total_val += pv_input[partial_idx] * fraction
            # If history isn't long enough, we assume 0 for the missing
            # past data

        return float(total_val / float_steps)

    def _calculate_cloud_factor(self, cloud_cover_pct: float) -> float:
        """
        Calculate cloud cover adjustment factor for PV heat weight.

        This factor modulates the PV contribution based on cloud coverage,
        accounting for the difference between direct sunlight and diffuse
        radiation. Higher cloud cover reduces the effective PV power
        using an exponential (power law) relationship.

        Formula: cloud_factor = (1.0 - cloud_cover_pct / 100.0)
        - 0% clouds → 1.0 (full direct + diffuse solar)
        - 50% clouds → 0.5 (mixed, some clouds)
        - 95% clouds → 0.05 (mostly diffuse, minimal direct)
        - 100% clouds → 0.0 (no direct sunlight)

        This linear model provides a simpler approximation compared to the previous exponential model,
        as cloud cover affects direct radiation linearly while diffuse component approaches zero.
        component approaches zero.

        Args:
            cloud_cover_pct: Cloud cover percentage (0-100)

        Returns:
            Cloud factor scalar (0.0-1.0) to multiply with pv_heat_weight
        """
        if not getattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False):
            return 1.0

        # Clamp to valid range
        cloud_pct = max(0.0, min(100.0, cloud_cover_pct))
        cloud_exponent = self._get_channel_parameter_value(
            "pv", "cloud_factor_exponent", 1.0
        )
        cloud_factor = 1.0 - (cloud_pct / 100.0) ** cloud_exponent

        min_factor = float(getattr(config, "CLOUD_CORRECTION_MIN_FACTOR", 0.1))
        min_factor = max(0.0, min(1.0, min_factor))
        return max(min_factor, cloud_factor)

    def predict_equilibrium_temperature(
        self,
        outlet_temp: float,
        outdoor_temp: float,
        current_indoor: float,
        pv_power=0,
        fireplace_on: float = 0,
        tv_on: float = 0,
        thermal_power: float = None,
        auxiliary_heat: float = 0.0,
        _suppress_logging: bool = False,
        fireplace_power_kw: float = None,
        cloud_cover_pct: float = 50.0,
        fireplace_decay_kw: float = 0.0,
    ) -> float:
        """
        Predict equilibrium temperature using standard heat balance physics.
        Supports both temperature-based approximation and energy-based
        modeling.
        """
        self._sync_model_from_orchestrator()

        # Calculate effective solar power with lag
        effective_pv = self._calculate_effective_solar(pv_power)
        
        # Apply cloud cover adjustment to PV heat weight
        # This accounts for the difference between direct sunlight and diffuse radiation
        cloud_factor = self._calculate_cloud_factor(cloud_cover_pct)
        pv_weight_adjusted = self.external_source_weights.get("pv", 0.0) * cloud_factor
        heat_from_pv = effective_pv * pv_weight_adjusted
        if not _suppress_logging and getattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False):
            logging.debug(
                "☁️ Cloud cover=%.0f%%, factor=%.3f, PV weight %.5f → %.5f, "
                "heat_from_pv=%.4f kW (effective_pv=%.0fW)",
                cloud_cover_pct,
                cloud_factor,
                self.external_source_weights.get("pv", 0.0),
                pv_weight_adjusted,
                heat_from_pv,
                effective_pv,
            )

        if fireplace_power_kw is not None:
            heat_from_fireplace = fireplace_power_kw
        else:
            heat_from_fireplace = (
                fireplace_on
                * self.external_source_weights.get("fireplace", 0.0)
            )
        # Add residual decay heat after fireplace is turned off
        heat_from_fireplace += fireplace_decay_kw

        heat_from_tv = tv_on * self.external_source_weights.get("tv", 0.0)

        external_thermal_power = (
            heat_from_pv + heat_from_fireplace + heat_from_tv + auxiliary_heat
        )

        # Energy-based physical modeling (Preferred if thermal_power is
        # available)
        if thermal_power is not None:
            # Teq = Tout + (P_total / U_loss)
            # P_total = P_thermal + P_external
            total_power = thermal_power + external_thermal_power

            if self.heat_loss_coefficient > 0:
                equilibrium_temp = outdoor_temp + (
                    total_power / self.heat_loss_coefficient
                )
            else:
                equilibrium_temp = outdoor_temp

            if not _suppress_logging:
                logging.debug(
                    "⚡ Energy physics: thermal_power=%.3f, ext_power=%.3f, "
                    "U_loss=%.3f, equilibrium=%.2f°C",
                    thermal_power,
                    external_thermal_power,
                    self.heat_loss_coefficient,
                    equilibrium_temp,
                )
            return equilibrium_temp

        # Fallback: Temperature-based approximation
        # RESTORED: Differential-based effectiveness scaling
        # Currently disabled (factor=0.0) — kept for future re-enabling
        if outlet_temp > current_indoor:
            temp_diff = max(1.0, outlet_temp - current_indoor)
            scaling_factor = 1.0 + (temp_diff - 25.0) * 0.0  # DISABLED
            effective_outlet_effectiveness = self.outlet_effectiveness * max(
                0.8, scaling_factor
            )
        else:
            effective_outlet_effectiveness = self.outlet_effectiveness

        total_conductance = (
            self.heat_loss_coefficient + effective_outlet_effectiveness
        )
        if total_conductance > 0:
            equilibrium_temp = (
                effective_outlet_effectiveness * outlet_temp
                + self.heat_loss_coefficient * outdoor_temp
                + external_thermal_power
            ) / total_conductance
        else:
            equilibrium_temp = outdoor_temp

        if outlet_temp > outdoor_temp:
            equilibrium_temp = max(outdoor_temp, equilibrium_temp)
        elif outlet_temp < outdoor_temp:
            equilibrium_temp = min(outdoor_temp, equilibrium_temp)
        else:
            if total_conductance > 0:
                equilibrium_temp = (
                    outdoor_temp + external_thermal_power / total_conductance
                )
            else:
                equilibrium_temp = outdoor_temp

        if not _suppress_logging:
            logging.debug(
                "🔬 Equilibrium physics: outlet=%.1f°C, outdoor=%.1f°C, "
                "heat_loss_coeff=%.3f, outlet_eff=%.3f, "
                "equilibrium=%.2f°C",
                outlet_temp,
                outdoor_temp,
                self.heat_loss_coefficient,
                effective_outlet_effectiveness,
                equilibrium_temp,
            )

        return equilibrium_temp

    def update_prediction_feedback(
        self,
        predicted_temp: float,
        actual_temp: float,
        prediction_context: Dict,
        timestamp: Optional[str] = None,
        is_blocking_active: bool = False,
    ):
        """
        Update the model with real-world feedback to enable adaptive learning.
        """
        self._sync_model_from_orchestrator()

        if not self.adaptive_learning_enabled:
            return

        if is_blocking_active:
            # Check if this is a safety floor event (outlet temp clamped to min)
            # If so, we should still learn because the model might be
            # over-optimistic
            outlet_temp = prediction_context.get("outlet_temp")
            is_safety_floor = False
            # Assuming 25 is safety floor
            if outlet_temp is not None and outlet_temp <= 25.0:
                is_safety_floor = True

            if not is_safety_floor:
                logging.debug(
                    "⏸️ Skipping learning during blocking event (DHW/Defrost)"
                )
                return
            else:
                logging.info("⚠️ Learning enabled during safety floor event")

        if self._detect_parameter_corruption():
            logging.warning(
                "🛑 Parameter corruption detected - learning DISABLED"
            )
            return

        # TRANSIENT DISTURBANCE GUARD: Detect sudden implausible indoor temp
        # drops (door/window opened) and skip learning for this cycle.
        # A physically normal house cools at most ~0.15°C in 10 minutes.
        # Drops > 0.25°C/cycle are almost certainly transient disturbances
        # that would inject large artificial errors into the gradient descent.
        _current_indoor = prediction_context.get("current_indoor")
        if _current_indoor is not None and actual_temp is not None:
            _temp_drop = _current_indoor - actual_temp
            if _temp_drop > 0.25:
                logging.warning(
                    "🚪 Transient disturbance guard: indoor temp dropped "
                    "%.3f°C in one cycle (%.2f → %.2f) — skipping learning "
                    "(door/window likely opened)",
                    _temp_drop,
                    _current_indoor,
                    actual_temp,
                )
                return None

        outlet_temp = prediction_context.get("outlet_temp")
        if outlet_temp is None:
            logging.warning(
                "No outlet_temp in prediction context, skipping learning"
            )
            return

        current_indoor = prediction_context.get("current_indoor")
        if current_indoor is None:
            logging.warning(
                "No current_indoor in prediction context, skipping learning"
            )
            return

        prediction_error = actual_temp - predicted_temp

        pv_input = prediction_context.get("pv_power_history")
        if pv_input is None:
            pv_input = prediction_context.get("pv_power", 0)

        # Build forecast-aware outdoor array for trajectory-aligned learning.
        # Uses the same horizon (TRAJECTORY_STEPS) as the optimization so that
        # gradients capture the effect of future PV / weather on the outlet
        # decision that was made.
        _outdoor_now = prediction_context.get("outdoor_temp", 10.0)
        _outdoor_forecast = prediction_context.get("outdoor_forecast")
        if _outdoor_forecast:
            _outdoor_arr = [_outdoor_now] + list(_outdoor_forecast)
        else:
            _outdoor_arr = _outdoor_now  # scalar fallback

        _pv_forecast = prediction_context.get("pv_forecast")

        predicted_trajectory = self.predict_thermal_trajectory(
            current_indoor=current_indoor,
            target_indoor=current_indoor,
            outlet_temp=outlet_temp,
            outdoor_temp=_outdoor_arr,
            time_horizon_hours=float(config.TRAJECTORY_STEPS),
            time_step_minutes=config.CYCLE_INTERVAL_MINUTES,
            pv_power=pv_input,
            pv_forecasts=_pv_forecast,
            fireplace_on=prediction_context.get("fireplace_on", 0),
            tv_on=prediction_context.get("tv_on", 0),
            thermal_power=prediction_context.get("thermal_power"),
            cloud_cover_pct=prediction_context.get("avg_cloud_cover", 50.0),
        )
        predicted_temp_at_cycle_end = predicted_trajectory["trajectory"][-1]

        system_state = (
            "shadow_mode_physics" if config.SHADOW_MODE else "active_mode"
        )

        prediction_record = {
            "timestamp": timestamp,
            "predicted": predicted_temp,
            "actual": actual_temp,
            "error": prediction_error,
            "context": prediction_context.copy(),
            "model_internal_prediction": predicted_temp_at_cycle_end,
            "parameters_at_prediction": {
                "thermal_time_constant": self.thermal_time_constant,
                "heat_loss_coefficient": self.heat_loss_coefficient,
                "outlet_effectiveness": self.outlet_effectiveness,
                "solar_lag_minutes": self.solar_lag_minutes,
            },
            "shadow_mode": config.SHADOW_MODE,
            "system_state": system_state,
            "learning_quality": self._assess_learning_quality(
                prediction_context, prediction_error
            ),
            "fireplace_active": bool(
                prediction_context.get("fireplace_on", 0)
            ),
            "pump_off": prediction_context.get("delta_t", 999) < 1.0,
        }

        self.prediction_history.append(prediction_record)

        if len(self.prediction_history) > config.MAX_PREDICTION_HISTORY:
            self.prediction_history = self.prediction_history[-config.MAX_PREDICTION_HISTORY:]

        # Step 10: Route learning through orchestrator for channel isolation.
        # Each channel independently learns from its own active periods,
        # preventing cross-contamination (e.g. fireplace heat misattributed
        # to outlet effectiveness).
        if self.orchestrator is not None:
            update_events = self.orchestrator.route_learning(
                error=prediction_error,
                context=prediction_context or {},
            )
            self._sync_model_from_orchestrator()
            channel_history_records = self._build_channel_parameter_history_records(
                update_events=update_events,
                timestamp=timestamp,
            )
            for history_record in channel_history_records:
                self._append_parameter_history_record(history_record)
            self._persist_heat_source_channel_state(
                parameter_history_records=channel_history_records,
            )

        if len(self.prediction_history) >= self.recent_errors_window:
            error_magnitude = abs(prediction_error)
            if error_magnitude < PhysicsConstants.ERROR_THRESHOLD_CONFIDENCE:
                # Good prediction
                self.learning_confidence *= self.confidence_boost_rate
            elif error_magnitude > PhysicsConstants.ERROR_THRESHOLD_HIGH:
                # Bad prediction
                self.learning_confidence *= self.confidence_decay_rate

            self.learning_confidence = float(
                np.clip(self.learning_confidence, 0.1, 5.0)
            )
            self._adapt_parameters_from_recent_errors()
            self._sync_model_from_orchestrator()
            self._persist_heat_source_channel_state()

        logging.debug(
            "Prediction feedback: error=%.3f°C, confidence=%.3f",
            prediction_error,
            self.learning_confidence,
        )
        return prediction_error

    def _assess_learning_quality(
        self, prediction_context: Dict, prediction_error: float
    ) -> str:
        """Assess the quality of this learning opportunity."""
        try:
            temp_gradient = abs(
                prediction_context.get("indoor_temp_gradient", 0.0)
            )
            is_stable = temp_gradient < 0.1
            error_magnitude = abs(prediction_error)

            if (
                error_magnitude < PhysicsConstants.ERROR_THRESHOLD_LOW
                and is_stable
            ):
                return "excellent"
            elif (
                error_magnitude < PhysicsConstants.ERROR_THRESHOLD_MEDIUM
                and is_stable
            ):
                return "good"
            elif error_magnitude < PhysicsConstants.ERROR_THRESHOLD_LOW:
                return "fair"
            elif is_stable:
                return "fair"
            else:
                return "poor"
        except Exception:
            return "unknown"

    def _adapt_parameters_from_recent_errors(self):
        """Adapt model parameters with corrected gradient calculations."""
        if self.orchestrator is not None:
            logging.debug(
                "Skipping legacy global adaptation: channel mode owns recent-window learning"
            )
            return

        recent_predictions = self.prediction_history[
            -self.recent_errors_window:
        ]

        if len(recent_predictions) < self.recent_errors_window:
            return

        if self._detect_parameter_corruption():
            logging.warning(
                "🛑 Parameter corruption detected in adaptation - "
                "learning DISABLED"
            )
            return

        recent_errors = [abs(p["error"]) for p in recent_predictions]

        # Dead zone: skip learning when errors are below sensor noise
        avg_recent_error = np.mean(recent_errors)
        if avg_recent_error < PhysicsConstants.LEARNING_DEAD_ZONE:
            logging.debug(
                "🎯 Dead zone: avg error %.4f < %.2f — skipping update",
                avg_recent_error, PhysicsConstants.LEARNING_DEAD_ZONE,
            )
            return

        has_catastrophic_error = any(error >= 5.0 for error in recent_errors)

        if has_catastrophic_error:
            # Check if this is due to PV over-estimation (predicted >> actual)
            # In this case, we WANT to learn (reduce PV weight)
            raw_errors = [p["error"] for p in recent_predictions]
            has_large_negative_error = any(e <= -5.0 for e in raw_errors)

            if has_large_negative_error:
                logging.warning(
                    "⚠️ Large negative error (over-prediction) detected - "
                    "Allowing learning to correct model."
                )
            else:
                max_error = max(recent_errors)
                logging.warning(
                    "🛑 Blocking parameter updates due to catastrophic error "
                    "(%.1f°C)",
                    max_error,
                )
                return

        thermal_gradient = self._calculate_thermal_time_constant_gradient(
            recent_predictions
        )
        heat_loss_coefficient_gradient = (
            self._calculate_heat_loss_coefficient_gradient(recent_predictions)
        )
        outlet_effectiveness_gradient = (
            self._calculate_outlet_effectiveness_gradient(recent_predictions)
        )
        pv_heat_weight_gradient = (
            self._calculate_pv_heat_weight_gradient(recent_predictions)
        )
        tv_heat_weight_gradient = (
            self._calculate_tv_heat_weight_gradient(recent_predictions)
        )
        solar_lag_gradient = self._calculate_solar_lag_gradient(
            recent_predictions
        )
        slab_gradient = self._calculate_slab_time_constant_gradient(
            recent_predictions
        )
        logging.debug(
            "Slab gradient: %.6f (τ_slab=%.3fh, bounds=[%.2f, %.2f])",
            slab_gradient, self.slab_time_constant_hours,
            self.slab_time_constant_bounds[0], self.slab_time_constant_bounds[1],
        )

        # FIREPLACE LEARNING GUARD: Zero HP-related gradients when
        # fireplace was active during the learning window — fireplace
        # heat contaminates OE/HLC gradient signals.
        has_fireplace = any(
            p.get("fireplace_active", False)
            or p.get("context", {}).get("fireplace_on", 0) > 0
            for p in recent_predictions
        )
        if has_fireplace:
            logging.info(
                "🔥 Fireplace active in recent window — skipping HP "
                "parameter gradients (OE, HLC, τ, slab_τ)"
            )
            thermal_gradient = 0.0
            heat_loss_coefficient_gradient = 0.0
            outlet_effectiveness_gradient = 0.0
            slab_gradient = 0.0

        current_learning_rate = self._calculate_adaptive_learning_rate()

        old_thermal_time_constant = self.thermal_time_constant
        old_heat_loss_coefficient = self.heat_loss_coefficient
        old_outlet_effectiveness = self.outlet_effectiveness
        old_pv_heat_weight = self.pv_heat_weight
        old_tv_heat_weight = self.tv_heat_weight
        old_solar_lag = self.solar_lag_minutes
        old_slab_time_constant = self.slab_time_constant_hours

        thermal_update = current_learning_rate * thermal_gradient
        heat_loss_coefficient_update = (
            current_learning_rate * heat_loss_coefficient_gradient
        )
        outlet_effectiveness_update = (
            current_learning_rate * outlet_effectiveness_gradient
        )

        # COLD WEATHER PROTECTION:
        # If outdoor temperature is dropping rapidly (e.g. night), the house
        # thermal mass (inertia) keeps it warm. The model might misinterpret
        # this "staying warm" as "better insulation" (lower heat loss coeff)
        # or "better radiators" (higher outlet effectiveness).
        # We must prevent parameters from drifting incorrectly during these
        # transient events.

        # Calculate average outdoor temp from recent history
        recent_outdoor = [
            p["context"].get("outdoor_temp", 0) for p in recent_predictions
        ]
        avg_outdoor = np.mean(recent_outdoor)

        # If it's cold outside, be skeptical of "better performance"
        if avg_outdoor < PhysicsConstants.COLD_WEATHER_PROTECTION_THRESHOLD:
            damping_factor = PhysicsConstants.COLD_WEATHER_DAMPING_FACTOR
            if (
                avg_outdoor
                < PhysicsConstants.EXTREME_COLD_PROTECTION_THRESHOLD
            ):
                damping_factor = PhysicsConstants.EXTREME_COLD_DAMPING_FACTOR

            # 1. Protect HLC (prevent reduction)
            # update > 0 means we subtract, so HLC goes down
            if heat_loss_coefficient_update > 0:
                heat_loss_coefficient_update *= damping_factor
                logging.debug(
                    "❄️ Cold weather protection: Dampening HLC reduction by "
                    "%.2f (outdoor=%.1f°C)",
                    damping_factor,
                    avg_outdoor,
                )

            # 2. Protect OE (prevent increase)
            # update < 0 means we subtract a negative, so OE goes up
            if outlet_effectiveness_update < 0:
                outlet_effectiveness_update *= damping_factor
                logging.debug(
                    "❄️ Cold weather protection: Dampening OE increase by "
                    "%.2f (outdoor=%.1f°C)",
                    damping_factor,
                    avg_outdoor,
                )

        # INDOOR TREND PROTECTION:
        # Guards the 0..10°C outdoor regime that cold-weather protection misses.
        # Uses indoor_temp_delta_60m (°C over last 60 min) stored in the context
        # by main.py from physics_features.py.
        #
        # Sign convention (same as cold weather block above):
        #   heat_loss_coefficient_update > 0  →  subtracted → HLC goes DOWN
        #   heat_loss_coefficient_update < 0  →  subtracted → HLC goes UP
        #   outlet_effectiveness_update  < 0  →  subtracted → OE goes UP
        #   outlet_effectiveness_update  > 0  →  subtracted → OE goes DOWN
        recent_indoor_deltas = [
            p["context"].get("indoor_temp_delta_60m", 0.0)
            for p in recent_predictions
        ]
        avg_indoor_delta_60m = np.mean(recent_indoor_deltas)

        # COOLING guard: indoor falling → don't let model conclude
        # "better insulation" (HLC↓) or "more effective radiators" (OE↑)
        if avg_indoor_delta_60m < PhysicsConstants.INDOOR_COOLING_TREND_THRESHOLD:
            cool_damp = PhysicsConstants.INDOOR_COOLING_DAMPING_FACTOR
            if heat_loss_coefficient_update > 0:
                heat_loss_coefficient_update *= cool_damp
                logging.debug(
                    "🌡️ Cooling trend protection: Dampening HLC reduction "
                    "(indoor Δ60m=%.3f°C, damp=%.2f)",
                    avg_indoor_delta_60m,
                    cool_damp,
                )
            if outlet_effectiveness_update < 0:
                outlet_effectiveness_update *= cool_damp
                logging.debug(
                    "🌡️ Cooling trend protection: Dampening OE increase "
                    "(indoor Δ60m=%.3f°C, damp=%.2f)",
                    avg_indoor_delta_60m,
                    cool_damp,
                )

        # WARMING guard: indoor rising → don't let model conclude
        # "worse insulation" (HLC↑) or "less effective radiators" (OE↓)
        elif avg_indoor_delta_60m > PhysicsConstants.INDOOR_WARMING_TREND_THRESHOLD:
            warm_damp = PhysicsConstants.INDOOR_WARMING_DAMPING_FACTOR
            if heat_loss_coefficient_update < 0:
                heat_loss_coefficient_update *= warm_damp
                logging.debug(
                    "🌡️ Warming trend protection: Dampening HLC increase "
                    "(indoor Δ60m=%.3f°C, damp=%.2f)",
                    avg_indoor_delta_60m,
                    warm_damp,
                )
            if outlet_effectiveness_update > 0:
                outlet_effectiveness_update *= warm_damp
                logging.debug(
                    "🌡️ Warming trend protection: Dampening OE decrease "
                    "(indoor Δ60m=%.3f°C, damp=%.2f)",
                    avg_indoor_delta_60m,
                    warm_damp,
                )

        # HIGH-PV DAMPENING: When significant PV is present, solar heat
        # can be misattributed to HP effectiveness. Dampen OE/HLC updates.
        recent_pv = [
            p["context"].get("pv_power", 0) for p in recent_predictions
        ]
        avg_pv = np.mean(recent_pv) if recent_pv else 0
        if avg_pv > 500:
            pv_damp = 0.3
            heat_loss_coefficient_update *= pv_damp
            outlet_effectiveness_update *= pv_damp
            logging.debug(
                "☀️ High-PV dampening: avg PV=%.0fW > 500W — "
                "dampening HLC/OE updates by %.0f%%",
                avg_pv, (1 - pv_damp) * 100,
            )

        pv_heat_weight_update = (
            current_learning_rate * pv_heat_weight_gradient
        )
        tv_heat_weight_update = (
            current_learning_rate * tv_heat_weight_gradient
        )

        # BOUND SATURATION PROTECTION:
        # When pv_heat_weight is at (or near) a bound AND the gradient wants
        # to push it further into the bound, the remaining prediction error
        # cannot be absorbed by pv_heat_weight. Without protection the error
        # leaks into HLC and OE, causing catastrophic drift (e.g. HLC → 0.01).
        #
        # Detection: current value within 5% of bound range AND gradient
        # direction would push it further into the bound.
        # Action: dampen HLC and OE updates by 90% to prevent error spillover.
        pv_range = self.pv_heat_weight_bounds[1] - self.pv_heat_weight_bounds[0]
        pv_near_upper = (
            self.pv_heat_weight
            >= self.pv_heat_weight_bounds[1] - 0.05 * pv_range
        )
        pv_near_lower = (
            self.pv_heat_weight
            <= self.pv_heat_weight_bounds[0] + 0.05 * pv_range
        )
        # update is subtracted: positive update → pv_heat_weight decreases
        # negative update → pv_heat_weight increases
        pv_saturated_upper = pv_near_upper and pv_heat_weight_update < 0
        pv_saturated_lower = pv_near_lower and pv_heat_weight_update > 0

        if pv_saturated_upper or pv_saturated_lower:
            bound_label = "upper" if pv_saturated_upper else "lower"
            saturation_damp = 0.1  # reduce to 10%
            heat_loss_coefficient_update *= saturation_damp
            outlet_effectiveness_update *= saturation_damp
            logging.warning(
                "☀️ PV bound saturation protection: pv_heat_weight=%.6f "
                "near %s bound (%.6f) — dampening HLC/OE updates by %.0f%%",
                self.pv_heat_weight,
                bound_label,
                (
                    self.pv_heat_weight_bounds[1]
                    if pv_saturated_upper
                    else self.pv_heat_weight_bounds[0]
                ),
                (1 - saturation_damp) * 100,
            )

        max_heat_loss_coefficient_change = (
            PhysicsConstants.MAX_HEAT_LOSS_COEFFICIENT_CHANGE
        )
        heat_loss_coefficient_update = np.clip(
            heat_loss_coefficient_update,
            -max_heat_loss_coefficient_change,
            max_heat_loss_coefficient_change,
        )
        max_outlet_effectiveness_change = (
            PhysicsConstants.MAX_OUTLET_EFFECTIVENESS_CHANGE
        )
        outlet_effectiveness_update = np.clip(
            outlet_effectiveness_update,
            -max_outlet_effectiveness_change,
            max_outlet_effectiveness_change,
        )
        max_thermal_time_constant_change = (
            PhysicsConstants.MAX_THERMAL_TIME_CONSTANT_CHANGE
        )
        thermal_update = np.clip(
            thermal_update,
            -max_thermal_time_constant_change,
            max_thermal_time_constant_change,
        )

        # Clip PV/TV/solar_lag updates (limit max change per step)
        max_weight_change = 0.0001
        pv_heat_weight_update = np.clip(
            pv_heat_weight_update, -max_weight_change, max_weight_change
        )
        tv_heat_weight_update = np.clip(
            tv_heat_weight_update, -max_weight_change, max_weight_change
        )
        solar_lag_update = current_learning_rate * solar_lag_gradient * 100.0
        solar_lag_update = np.clip(solar_lag_update, -5.0, 5.0)

        slab_update = current_learning_rate * slab_gradient
        slab_update = np.clip(slab_update, -0.05, 0.05)

        self.thermal_time_constant = float(
            np.clip(
                self.thermal_time_constant - thermal_update,
                self.thermal_time_constant_bounds[0],
                self.thermal_time_constant_bounds[1],
            )
        )

        self.heat_loss_coefficient = float(
            np.clip(
                self.heat_loss_coefficient - heat_loss_coefficient_update,
                self.heat_loss_coefficient_bounds[0],
                self.heat_loss_coefficient_bounds[1],
            )
        )

        # HLC DRIFT SAFETY FLOOR: prevent runaway drift below 85% of
        # calibrated baseline
        hlc_floor = self._baseline_heat_loss_coefficient * 0.85
        if self.heat_loss_coefficient < hlc_floor:
            logging.warning(
                "🛑 HLC drift floor: %.4f < %.4f (85%% of baseline "
                "%.4f) — clamping",
                self.heat_loss_coefficient,
                hlc_floor,
                self._baseline_heat_loss_coefficient,
            )
            self.heat_loss_coefficient = hlc_floor

        self.outlet_effectiveness = float(
            np.clip(
                self.outlet_effectiveness - outlet_effectiveness_update,
                self.outlet_effectiveness_bounds[0],
                self.outlet_effectiveness_bounds[1],
            )
        )

        self.pv_heat_weight = float(
            np.clip(
                self.pv_heat_weight - pv_heat_weight_update,
                self.pv_heat_weight_bounds[0],
                self.pv_heat_weight_bounds[1],
            )
        )

        self.tv_heat_weight = float(
            np.clip(
                self.tv_heat_weight - tv_heat_weight_update,
                self.tv_heat_weight_bounds[0],
                self.tv_heat_weight_bounds[1],
            )
        )

        self.solar_lag_minutes = float(
            np.clip(
                self.solar_lag_minutes - solar_lag_update,
                self.solar_lag_minutes_bounds[0],
                self.solar_lag_minutes_bounds[1],
            )
        )

        self.slab_time_constant_hours = float(
            np.clip(
                self.slab_time_constant_hours - slab_update,
                self.slab_time_constant_bounds[0],
                self.slab_time_constant_bounds[1],
            )
        )

        thermal_change = abs(
            self.thermal_time_constant - old_thermal_time_constant
        )
        heat_loss_coefficient_change = abs(
            self.heat_loss_coefficient - old_heat_loss_coefficient
        )
        outlet_effectiveness_change = abs(
            self.outlet_effectiveness - old_outlet_effectiveness
        )
        pv_heat_weight_change = abs(self.pv_heat_weight - old_pv_heat_weight)
        tv_heat_weight_change = abs(self.tv_heat_weight - old_tv_heat_weight)
        solar_lag_change = abs(self.solar_lag_minutes - old_solar_lag)
        slab_change = abs(self.slab_time_constant_hours - old_slab_time_constant)

        self._append_parameter_history_record(
            {
                "timestamp": recent_predictions[-1]["timestamp"],
                "thermal_time_constant": self.thermal_time_constant,
                "heat_loss_coefficient": self.heat_loss_coefficient,
                "outlet_effectiveness": self.outlet_effectiveness,
                "pv_heat_weight": self.pv_heat_weight,
                "tv_heat_weight": self.tv_heat_weight,
                "solar_lag_minutes": self.solar_lag_minutes,
                "slab_time_constant_hours": self.slab_time_constant_hours,
                "learning_rate": current_learning_rate,
                "learning_confidence": self.learning_confidence,
                "avg_recent_error": np.mean(
                    [abs(p["error"]) for p in recent_predictions]
                ),
                "gradients": {
                    "thermal": thermal_gradient,
                    "heat_loss_coefficient": heat_loss_coefficient_gradient,
                    "outlet_effectiveness": outlet_effectiveness_gradient,
                    "pv_heat_weight": pv_heat_weight_gradient,
                    "tv_heat_weight": tv_heat_weight_gradient,
                    "solar_lag": solar_lag_gradient,
                    "slab_time_constant": slab_gradient,
                },
                "changes": {
                    "thermal": thermal_change,
                    "heat_loss_coefficient": heat_loss_coefficient_change,
                    "outlet_effectiveness": outlet_effectiveness_change,
                    "pv_heat_weight": pv_heat_weight_change,
                    "tv_heat_weight": tv_heat_weight_change,
                    "solar_lag": solar_lag_change,
                    "slab_time_constant": slab_change,
                },
            }
        )

        if (
            thermal_change > 0.001
            or heat_loss_coefficient_change > 0.0001
            or outlet_effectiveness_change > 0.0001
            or pv_heat_weight_change > 0.0001
            or tv_heat_weight_change > 0.0001
            or solar_lag_change > 0.1
            or slab_change > 0.001
        ):
            logging.info(
                "Adaptive learning update: "
                "thermal: %.2f→%.2f (Δ%+.3f), "
                "heat_loss_coeff: %.4f→%.4f (Δ%+.5f), "
                "outlet_eff: %.3f→%.3f (Δ%+.3f), "
                "pv_weight: %.3f→%.3f (Δ%+.3f), "
                "tv_weight: %.3f→%.3f (Δ%+.3f), "
                "solar_lag: %.1f→%.1f (Δ%+.2f), "
                "slab_tau: %.3f→%.3f (Δ%+.4f)",
                old_thermal_time_constant,
                self.thermal_time_constant,
                thermal_change,
                old_heat_loss_coefficient,
                self.heat_loss_coefficient,
                heat_loss_coefficient_change,
                old_outlet_effectiveness,
                self.outlet_effectiveness,
                outlet_effectiveness_change,
                old_pv_heat_weight,
                self.pv_heat_weight,
                pv_heat_weight_change,
                old_tv_heat_weight,
                self.tv_heat_weight,
                tv_heat_weight_change,
                old_solar_lag,
                self.solar_lag_minutes,
                solar_lag_change,
                old_slab_time_constant,
                self.slab_time_constant_hours,
                slab_change,
            )

            self._save_learning_to_thermal_state(
                self.thermal_time_constant - old_thermal_time_constant,
                self.heat_loss_coefficient - old_heat_loss_coefficient,
                self.outlet_effectiveness - old_outlet_effectiveness,
                self.pv_heat_weight - old_pv_heat_weight,
                self.tv_heat_weight - old_tv_heat_weight,
                new_solar_lag_adjustment=self.solar_lag_minutes - old_solar_lag,
                new_slab_adjustment=self.slab_time_constant_hours - old_slab_time_constant,
            )
        else:
            logging.debug(
                "Micro learning update: thermal_Δ=%+.5f, "
                "heat_loss_coeff_Δ=%+.7f, outlet_eff_Δ=%+.5f, "
                "pv_Δ=%+.5f, tv_Δ=%+.5f",
                thermal_change,
                heat_loss_coefficient_change,
                outlet_effectiveness_change,
                pv_heat_weight_change,
                tv_heat_weight_change,
            )

    def _calculate_parameter_gradient(
        self,
        parameter_name: str,
        epsilon: float,
        recent_predictions: List[Dict],
    ) -> float:
        """
        Generic finite-difference gradient calculation for any parameter.

        IMPORTANT: thermal_power is intentionally NOT passed to the trajectory
        here.  When thermal_power is non-None, predict_equilibrium_temperature
        uses the energy formula  T_eq = T_outdoor + P/HLC, which does NOT
        depend on outlet_temp or outlet_effectiveness at all.  Passing
        thermal_power therefore always produces a zero finite-difference for
        outlet_effectiveness (and a formula-mismatch gradient for others).
        The temperature-based formula (outlet × OE, outdoor × HLC) correctly
        expresses how each parameter influences the indoor temperature and is
        the right model for gradient-based learning.
        """
        gradient_sum = 0.0
        count = 0

        original_value = getattr(self, parameter_name)

        # HP parameters whose gradients must be shielded from contaminated
        # records (fireplace active or pump off).
        _hp_params = {
            "thermal_time_constant", "heat_loss_coefficient",
            "outlet_effectiveness", "slab_time_constant_hours",
        }
        is_hp_param = parameter_name in _hp_params

        for pred in recent_predictions:
            context = pred["context"]

            if not all(
                key in context
                for key in ["outlet_temp", "outdoor_temp", "current_indoor"]
            ):
                continue

            # LEARNING GUARD: skip contaminated records for HP params
            if is_hp_param:
                if pred.get("fireplace_active", False):
                    continue
                if pred.get("pump_off", False):
                    continue

            # PUMP-OFF FIX: when pump was off, use inlet_temp instead of
            # stale Sollwert so the slab model correctly simulates pump-OFF.
            effective_outlet = context["outlet_temp"]
            if (
                context.get("delta_t", 999) < 1.0
                or context.get("thermal_power_kw", 999) < 0.2
            ):
                effective_outlet = context.get(
                    "inlet_temp", context["outlet_temp"]
                )
            # Defensive fallback: if effective_outlet is None after the
            # pump-off fallback (e.g. both outlet_temp and inlet_temp are
            # missing from a legacy context), use current_indoor so the
            # gradient still contributes instead of crashing or silently
            # dropping the record.
            if effective_outlet is None:
                effective_outlet = context["current_indoor"]

            # Build forecast-aware outdoor array matching the optimization
            # horizon (TRAJECTORY_STEPS).  Fall back to scalar if the context
            # was recorded before forecast arrays were stored.
            _g_outdoor_now = context["outdoor_temp"]
            _g_outdoor_fc = context.get("outdoor_forecast")
            _g_outdoor = (
                [_g_outdoor_now] + list(_g_outdoor_fc)
                if _g_outdoor_fc
                else _g_outdoor_now
            )
            _g_pv_fc = context.get("pv_forecast")
            _g_horizon = float(config.TRAJECTORY_STEPS)

            setattr(self, parameter_name, original_value + epsilon)
            pred_plus_trajectory = self.predict_thermal_trajectory(
                current_indoor=context["current_indoor"],
                target_indoor=context["current_indoor"],
                outlet_temp=effective_outlet,
                outdoor_temp=_g_outdoor,
                time_horizon_hours=_g_horizon,
                time_step_minutes=config.CYCLE_INTERVAL_MINUTES,
                pv_power=context.get("pv_power", 0),
                pv_forecasts=_g_pv_fc,
                fireplace_on=context.get("fireplace_on", 0),
                tv_on=context.get("tv_on", 0),
                cloud_cover_pct=context.get("avg_cloud_cover", 50.0),
                inlet_temp=context.get("inlet_temp"),
                delta_t_floor=context.get("delta_t", 0.0),
                # thermal_power deliberately omitted: energy-mode formula
                # T=outdoor+P/HLC bypasses outlet_effectiveness entirely,
                # producing zero gradient for OE and an inconsistent gradient
                # for other params.  Use temperature-based formula instead.
            )
            pred_plus = pred_plus_trajectory["trajectory"][-1]

            setattr(self, parameter_name, original_value - epsilon)
            pred_minus_trajectory = self.predict_thermal_trajectory(
                current_indoor=context["current_indoor"],
                target_indoor=context["current_indoor"],
                outlet_temp=effective_outlet,
                outdoor_temp=_g_outdoor,
                time_horizon_hours=_g_horizon,
                time_step_minutes=config.CYCLE_INTERVAL_MINUTES,
                pv_power=context.get("pv_power", 0),
                pv_forecasts=_g_pv_fc,
                fireplace_on=context.get("fireplace_on", 0),
                tv_on=context.get("tv_on", 0),
                cloud_cover_pct=context.get("avg_cloud_cover", 50.0),
                inlet_temp=context.get("inlet_temp"),
                delta_t_floor=context.get("delta_t", 0.0),
                # thermal_power deliberately omitted (see above)
            )
            pred_minus = pred_minus_trajectory["trajectory"][-1]

            setattr(self, parameter_name, original_value)

            finite_diff = (pred_plus - pred_minus) / (2 * epsilon)
            error = np.clip(pred["error"], -2.0, 2.0)
            gradient = -finite_diff * error
            gradient_sum += gradient
            count += 1

        return gradient_sum / count if count > 0 else 0.0

    def _calculate_thermal_time_constant_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """
        Calculate thermal time constant gradient.
        """
        return self._calculate_parameter_gradient(
            "thermal_time_constant",
            PhysicsConstants.THERMAL_TIME_CONSTANT_EPSILON,
            recent_predictions,
        )

    def _calculate_heat_loss_coefficient_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """
        Calculate heat loss coefficient gradient.
        """
        return self._calculate_parameter_gradient(
            "heat_loss_coefficient",
            PhysicsConstants.HEAT_LOSS_COEFFICIENT_EPSILON,
            recent_predictions,
        )

    def _calculate_outlet_effectiveness_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """
        Calculate outlet effectiveness gradient.
        """
        return self._calculate_parameter_gradient(
            "outlet_effectiveness",
            PhysicsConstants.OUTLET_EFFECTIVENESS_EPSILON,
            recent_predictions,
        )

    def _calculate_pv_heat_weight_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """
        Calculate PV heat weight gradient.

        ZERO-PV GUARD: When current PV power AND all PV forecasts are zero
        across all recent records (nighttime / evening), the finite-difference
        perturbation of pv_heat_weight produces only floating-point noise —
        no real physical signal.  Amplified by even moderate prediction errors
        this noise can push pv_heat_weight to its bounds within ~1 hour.
        Return exactly 0.0 in that case.

        NOTE: pv_power_history is intentionally NOT checked.  It contains
        stale daytime readings that linger in the trailing window but do NOT
        influence the trajectory finite-difference (which only uses current +
        forecast PV).  Without this, the guard would fail during evening
        twilight when history still has afternoon values.
        """
        all_pv_zero = True
        for pred in recent_predictions:
            ctx = pred.get("context", {})
            # Check current PV power
            pv_now = ctx.get("pv_power", 0)
            if isinstance(pv_now, (int, float)) and pv_now > 0:
                all_pv_zero = False
                break
            # Check PV forecasts stored in context
            pv_fc = ctx.get("pv_forecast")
            if pv_fc and any(v > 0 for v in pv_fc):
                all_pv_zero = False
                break

        if all_pv_zero:
            logging.debug(
                "🌙 Zero-PV guard: all recent PV power and forecasts are "
                "zero — suppressing pv_heat_weight gradient"
            )
            return 0.0

        return self._calculate_parameter_gradient(
            "pv_heat_weight",
            PhysicsConstants.PV_HEAT_WEIGHT_EPSILON,
            recent_predictions,
        )

    def _calculate_tv_heat_weight_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """
        Calculate TV heat weight gradient.
        """
        return self._calculate_parameter_gradient(
            "tv_heat_weight",
            PhysicsConstants.TV_HEAT_WEIGHT_EPSILON,
            recent_predictions,
        )

    def _calculate_solar_lag_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """Calculate solar lag gradient (epsilon=5 minutes)."""
        return self._calculate_parameter_gradient(
            "solar_lag_minutes", 5.0, recent_predictions
        )

    def _calculate_slab_time_constant_gradient(
        self, recent_predictions: List[Dict]
    ) -> float:
        """Calculate slab time-constant gradient (epsilon=0.1 h).

        Uses finite-difference via _calculate_parameter_gradient, exactly as
        all other learnable parameters.  The gradient is non-zero only when
        inlet_temp != outlet_cmd in the stored prediction contexts, i.e.
        during transient heating/cooling phases.
        """
        return self._calculate_parameter_gradient(
            "slab_time_constant_hours", 0.1, recent_predictions
        )

    def _calculate_adaptive_learning_rate(self) -> float:
        """
        Calculate adaptive learning rate based on model performance.
        """
        base_rate = (
            max(self.learning_rate, self.min_learning_rate)
            * self.learning_confidence
        )

        heat_loss_coefficient_std = 0.0
        thermal_time_constant_std = 0.0
        outlet_effectiveness_std = 0.0
        recent_params = self._get_recent_parameter_snapshot_records(3)
        if len(recent_params) >= 3:
            heat_loss_coefficient_std = np.std(
                [p["heat_loss_coefficient"] for p in recent_params]
            )
            thermal_time_constant_std = np.std(
                [p["thermal_time_constant"] for p in recent_params]
            )
            outlet_effectiveness_std = np.std(
                [p["outlet_effectiveness"] for p in recent_params]
            )

        if (
            heat_loss_coefficient_std > 0.02
            or outlet_effectiveness_std > 0.05
            or thermal_time_constant_std > 0.5
        ):
            base_rate *= 0.01
            logging.debug(
                (
                    "⚠️ Parameter oscillation detected (heat_loss_coeff=%.3f, "
                    "outlet_eff=%.3f, thermal=%.3f), "
                    "reducing learning rate by 99%%"
                ),
                heat_loss_coefficient_std,
                outlet_effectiveness_std,
                thermal_time_constant_std,
            )
        elif (
            heat_loss_coefficient_std > 0.01
            or outlet_effectiveness_std > 0.02
            or thermal_time_constant_std > 0.2
        ):
            base_rate *= 0.1

        if self.prediction_history:
            recent_errors = [
                abs(p["error"]) for p in self.prediction_history[-5:]
            ]
            has_catastrophic_error = any(
                error >= 5.0 for error in recent_errors
            )

            if has_catastrophic_error:
                # Check if this is due to PV over-estimation (predicted >>
                # actual) In this case, we WANT to learn (reduce PV weight)
                # We need to check the sign of the error.
                # recent_errors contains absolute errors, so we need to check
                # the raw errors.
                raw_errors = [p["error"] for p in self.prediction_history[-5:]]
                # Error = Actual - Predicted.
                # If Predicted >> Actual, Error is large negative.
                # So we check if we have large negative errors.

                has_large_negative_error = any(e <= -5.0 for e in raw_errors)

                if has_large_negative_error:
                    logging.warning(
                        "⚠️ Large negative error (over-prediction) detected - "
                        "Allowing learning to correct model."
                    )
                    # Allow learning, maybe even boost it?
                    base_rate *= 2.0
                else:
                    base_rate = 0.0
                    max_error = max(recent_errors)
                    logging.warning(
                        "🛑 Catastrophic error (%.1f°C) - learning DISABLED",
                        max_error,
                    )

            elif len(self.prediction_history) >= 5:
                avg_error = np.mean(recent_errors)
                if config.SHADOW_MODE:
                    # In shadow mode, we want to learn faster, even from large
                    # errors, now that the physics are corrected. The
                    # confidence mechanism will naturally temper the learning
                    # rate.
                    if avg_error > 2.0:
                        base_rate /= 1.5  # Less aggressive reduction
                else:
                    # Boost learning rate for significant errors to adapt
                    # faster
                    if avg_error > 3.0:
                        base_rate *= 5.0
                    elif (
                        avg_error > PhysicsConstants.ERROR_THRESHOLD_VERY_HIGH
                    ):
                        base_rate *= PhysicsConstants.ERROR_BOOST_FACTOR_HIGH
                    elif avg_error > PhysicsConstants.ERROR_THRESHOLD_HIGH:
                        base_rate *= PhysicsConstants.ERROR_BOOST_FACTOR_MEDIUM
                    elif avg_error > PhysicsConstants.ERROR_THRESHOLD_MEDIUM:
                        base_rate *= PhysicsConstants.ERROR_BOOST_FACTOR_LOW

        return np.clip(base_rate, 0.0, self.max_learning_rate)

    def predict_thermal_trajectory(
        self,
        current_indoor,
        target_indoor,
        outlet_temp,
        outdoor_temp,
        time_horizon_hours=None,
        time_step_minutes=60,
        weather_forecasts=None,
        pv_forecasts=None,
        thermal_time_constant=None,
        inlet_temp=None,
        **external_sources,
    ):
        """
        Predict temperature trajectory over time horizon.
        """
        self._sync_model_from_orchestrator()

        if time_horizon_hours is None:
            time_horizon_hours = int(self.prediction_horizon_hours)

        trajectory = []
        current_temp = current_indoor

        pv_power = external_sources.get("pv_power", 0)
        fireplace_on = external_sources.get("fireplace_on", 0)
        tv_on = external_sources.get("tv_on", 0)
        thermal_power = external_sources.get("thermal_power", None)
        auxiliary_heat = external_sources.get("auxiliary_heat", 0.0)
        fireplace_power_kw = external_sources.get("fireplace_power_kw", None)
        cloud_cover_pct = external_sources.get("cloud_cover_pct", 50.0)
        fireplace_decay_kw = float(external_sources.get("fireplace_decay_kw", 0.0))
        # Steady-state temperature drop across floor loop: ΔT = BT2 − BT3
        # = thermal_power / (flow_rate × C_P).  Passed from caller as
        # delta_t_floor; defaults to 0 to preserve backward-compat behaviour.
        measured_delta_t = float(external_sources.get("delta_t_floor", 0.0))
        delta_t_floor = self._resolve_delta_t_floor(measured_delta_t)
        # _resolve_delta_t_floor already substitutes the HP channel's learned
        # value (~2 °C) when measured < 1.0, so no hardcoded fallback needed.

        time_step_hours = time_step_minutes / 60.0
        num_steps = int(time_horizon_hours * 60 / time_step_minutes)
        if num_steps == 0:
            num_steps = 1
            
        # Handle outdoor temperature forecast (array or scalar)
        if isinstance(outdoor_temp, (list, np.ndarray)):
            # Interpolate array input to simulation steps
            source_len = len(outdoor_temp)
            if source_len == num_steps:
                outdoor_forecasts = list(outdoor_temp)
            else:
                # Source points are assumed to be hourly: index 0=t=0h, 1=t=1h, ...
                # Interpolate only over [0, time_horizon_hours] to respect the horizon.
                source_hours = np.arange(source_len)  # [0, 1, 2, ..., source_len-1]
                target_hours = np.linspace(0, time_horizon_hours, num_steps)
                # Clip to avoid extrapolating beyond available forecast data
                target_hours_clipped = np.clip(target_hours, 0, source_hours[-1])
                outdoor_forecasts = np.interp(
                    target_hours_clipped, source_hours, outdoor_temp
                ).tolist()
        elif weather_forecasts:
            # Legacy support for separate argument
            source_len = len(weather_forecasts)
            # Assume weather_forecasts are hourly if not specified, map to
            # steps. If we assume weather_forecasts matches the horizon hours:
            source_times = np.linspace(0, time_horizon_hours, source_len)
            target_times = np.linspace(0, time_horizon_hours, num_steps)
            outdoor_forecasts = np.interp(
                target_times, source_times, weather_forecasts
            ).tolist()
        else:
            outdoor_forecasts = [outdoor_temp] * num_steps

        # Handle PV power using rolling buffer (for solar lag computation)
        # pv_power may be a list (historical), a scalar, or from pv_forecasts
        PV_BUFFER_MAX = 18  # 3 hours at 10-minute steps
        if isinstance(pv_power, (list, np.ndarray)):
            pv_buffer = list(pv_power)[-PV_BUFFER_MAX:]
            last_pv_val = float(pv_buffer[-1]) if pv_buffer else 0.0
        else:
            last_pv_val = float(pv_power) if pv_power else 0.0
            pv_buffer = [last_pv_val] * PV_BUFFER_MAX

        # Build future PV values from forecasts.
        # Source times are whole hours (0, 1, 2, ...) matching the hourly
        # forecast array layout. Clip target times so we never extrapolate
        # beyond the available forecast window (same approach as outdoor).
        if pv_forecasts:
            source_len = len(pv_forecasts)
            source_hours = np.arange(source_len)  # [0, 1, 2, ..., source_len-1]
            target_times = np.linspace(0, time_horizon_hours, num_steps)
            target_times_clipped = np.clip(target_times, 0, source_hours[-1])
            pv_future_values = np.interp(
                target_times_clipped, source_hours, pv_forecasts
            ).tolist()
        else:
            pv_future_values = [last_pv_val] * num_steps

        for step in range(num_steps):
            future_outdoor = outdoor_forecasts[step]
            next_pv_val = pv_future_values[step]
            pv_buffer.append(next_pv_val)
            if len(pv_buffer) > PV_BUFFER_MAX:
                pv_buffer.pop(0)

            # --- Slab (Estrich) model ---
            # T_slab tracks the return temperature (BT3 / Rücklauf).
            # T_slab(0) = inlet_temp; when None the slab model is inactive.
            #
            # Pump-ON  (BT2 > BT3 AND measured ΔT ≥ 1): forced convection
            #   drives BT3 toward BT2 − ΔT_floor.
            # Pump-OFF (measured ΔT < 1 OR BT2 ≤ BT3): NIBE shuts down;
            #   BT3 cools passively toward indoor air temperature.
            #
            # Using measured_delta_t as the gate prevents the pump-ON branch
            # from firing when the HP is actually off but outlet happens to
            # read higher than inlet due to sensor values or calculated targets.
            if step == 0:
                t_slab = (
                    float(inlet_temp)
                    if inlet_temp is not None
                    else float(outlet_temp)
                )
            pump_on = (
                float(outlet_temp) > t_slab and measured_delta_t >= 1.0
            )
            if pump_on:
                alpha = min(1.0, time_step_hours / self.slab_time_constant_hours)
                t_slab_target = float(outlet_temp) - delta_t_floor
                t_slab = t_slab + alpha * (t_slab_target - t_slab)
                # Floor emits heat based on mean water temp across the loop
                t_effective = (float(outlet_temp) + t_slab) / 2.0
            else:  # pump off: passive cooling toward indoor air
                alpha_passive = 1.0 - np.exp(
                    -time_step_hours / self.thermal_time_constant
                )
                t_slab = t_slab + alpha_passive * (current_temp - t_slab)
                # No flow — floor radiates from slab mass only
                t_effective = t_slab

            equilibrium_temp = self.predict_equilibrium_temperature(
                outlet_temp=t_effective,
                outdoor_temp=future_outdoor,
                current_indoor=current_temp,
                pv_power=list(pv_buffer),
                fireplace_on=fireplace_on,
                tv_on=tv_on,
                thermal_power=thermal_power,
                auxiliary_heat=auxiliary_heat,
                _suppress_logging=True,
                fireplace_power_kw=fireplace_power_kw,
                cloud_cover_pct=cloud_cover_pct,
                fireplace_decay_kw=fireplace_decay_kw,
            )

            time_constant_hours = (
                thermal_time_constant
                if thermal_time_constant is not None
                else self.thermal_time_constant
            )
            approach_factor = 1 - np.exp(
                -time_step_hours / time_constant_hours
            )
            temp_change = (equilibrium_temp - current_temp) * approach_factor

            if step > 0:
                momentum_factor = np.exp(
                    -step * time_step_hours * self.momentum_decay_rate
                )
                temp_change *= 1.0 - momentum_factor * 0.2

            current_temp = current_temp + temp_change
            trajectory.append(current_temp)

        reaches_target_at = None
        sensor_precision_tolerance = 0.1  # °C — must match binary search convergence tolerance

        for i, temp in enumerate(trajectory):
            if abs(temp - target_indoor) < sensor_precision_tolerance:
                reaches_target_at = (i + 1) * time_step_hours
                break

        if (
            trajectory
            and abs(trajectory[0] - target_indoor) < sensor_precision_tolerance
        ):
            cycle_hours = config.CYCLE_INTERVAL_MINUTES / 60.0
            if reaches_target_at is not None:
                reaches_target_at = min(reaches_target_at, cycle_hours)
            else:
                reaches_target_at = cycle_hours

        overshoot_predicted = False
        max_predicted = max(trajectory) if trajectory else current_indoor

        if target_indoor > current_indoor:
            overshoot_predicted = max_predicted > (
                target_indoor + self.safety_margin
            )
        else:
            min_predicted = min(trajectory) if trajectory else current_indoor
            overshoot_predicted = min_predicted < (
                target_indoor - self.safety_margin
            )

        return {
            "trajectory": trajectory,
            "times": [
                (step + 1) * time_step_hours for step in range(num_steps)
            ],
            "reaches_target_at": reaches_target_at,
            "overshoot_predicted": overshoot_predicted,
            "max_predicted": max(trajectory) if trajectory else current_indoor,
            "min_predicted": min(trajectory) if trajectory else current_indoor,
            "equilibrium_temp": (
                trajectory[-1] if trajectory else current_indoor
            ),
            "final_error": (
                abs(trajectory[-1] - target_indoor)
                if trajectory
                else abs(current_indoor - target_indoor)
            ),
        }

    def calculate_optimal_outlet_temperature(
        self,
        target_indoor,
        current_indoor,
        outdoor_temp,
        time_available_hours=1.0,
        config_override=None,
        **external_sources,
    ):
        """
        Calculate optimal outlet temperature to reach target indoor
        temperature.
        """
        self._sync_model_from_orchestrator()

        pv_power = external_sources.get(
            "pv_power", external_sources.get("pv_now", 0)
        )
        fireplace_on = external_sources.get("fireplace_on", 0)
        tv_on = external_sources.get("tv_on", 0)

        temp_change_required = target_indoor - current_indoor

        if abs(temp_change_required) < 0.1:
            outlet_temp = self._calculate_equilibrium_outlet_temperature(
                target_indoor, outdoor_temp, pv_power, fireplace_on, tv_on
            )
            return {
                "optimal_outlet_temp": outlet_temp,
                "method": "equilibrium_maintenance",
                "temp_change_required": temp_change_required,
                "time_available": time_available_hours,
            }

        method = "direct_calculation"
        heat_loss_coefficient = self.heat_loss_coefficient
        outlet_effectiveness = self.outlet_effectiveness
        if config_override:
            heat_loss_coefficient = config_override.get(
                "heat_loss_coefficient", heat_loss_coefficient
            )
            outlet_effectiveness = config_override.get(
                "outlet_effectiveness", outlet_effectiveness
            )
        total_conductance = heat_loss_coefficient + outlet_effectiveness

        external_heating = (
            pv_power * self.external_source_weights.get("pv", 0.0)
            + fireplace_on * self.external_source_weights.get("fireplace", 0.0)
            + tv_on * self.external_source_weights.get("tv", 0.0)
        )

        if outlet_effectiveness <= 0:
            return None

        optimal_outlet = (
            target_indoor * total_conductance
            - heat_loss_coefficient * outdoor_temp
            - external_heating
        ) / outlet_effectiveness
        required_equilibrium = target_indoor

        min_outlet = max(outdoor_temp + 5, 25.0)
        max_outlet = 70.0

        optimal_outlet_bounded = max(
            min_outlet, min(optimal_outlet, max_outlet)
        )

        if optimal_outlet_bounded < outdoor_temp:
            fallback_outlet = self._calculate_equilibrium_outlet_temperature(
                target_indoor, outdoor_temp, pv_power, fireplace_on, tv_on
            )
            return {
                "optimal_outlet_temp": fallback_outlet,
                "method": "fallback_equilibrium",
                "reason": "unrealistic_outlet_temp",
                "original_calculation": optimal_outlet,
                "temp_change_required": temp_change_required,
                "time_available": time_available_hours,
            }

        return {
            "optimal_outlet_temp": optimal_outlet_bounded,
            "method": method,
            "required_equilibrium": required_equilibrium,
            "temp_change_required": temp_change_required,
            "time_available": time_available_hours,
            "external_heating": external_heating,
            "bounded": optimal_outlet != optimal_outlet_bounded,
            "original_calculation": optimal_outlet,
        }

    def _calculate_equilibrium_outlet_temperature(
        self, target_temp, outdoor_temp, pv_power=0, fireplace_on=0, tv_on=0
    ):
        """
        Calculate outlet temperature needed for equilibrium at target temp.
        """
        self._sync_model_from_orchestrator()

        external_heating = (
            pv_power * self.external_source_weights["pv"]
            + fireplace_on * self.external_source_weights["fireplace"]
            + tv_on * self.external_source_weights["tv"]
        )

        if self.outlet_effectiveness <= 0:
            return 35.0

        total_conductance = (
            self.heat_loss_coefficient + self.outlet_effectiveness
        )
        equilibrium_outlet = (
            target_temp * total_conductance
            - self.heat_loss_coefficient * outdoor_temp
            - external_heating
        ) / self.outlet_effectiveness

        min_outlet = max(outdoor_temp + 5, 25.0)
        max_outlet = 65.0

        return max(min_outlet, min(equilibrium_outlet, max_outlet))

    def calculate_physics_aware_thresholds(self, *args, **kwargs):
        """Keep original threshold calculation method unchanged"""
        pass

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get relative importance of different heat sources/parameters.
        Acts as a compatibility layer for the legacy feature importance metric.
        """
        # Normalize weights to sum to 1.0 for relative importance
        weights = {
            "pv_power": self.external_source_weights.get("pv", 0.0),
            "fireplace": self.external_source_weights.get("fireplace", 0.0),
            "tv_power": self.external_source_weights.get("tv", 0.0),
            # Proxy for outdoor influence
            "outdoor_temp": self.heat_loss_coefficient,
            # Proxy for target influence
            "target_temp": self.outlet_effectiveness,
        }

        total = sum(abs(v) for v in weights.values())
        if total > 0:
            return {k: abs(v) / total for k, v in weights.items()}
        return weights

    def _get_current_export_parameters(self) -> Dict[str, float]:
        """Build a runtime parameter snapshot for HA and Influx exports."""
        current_parameters = {
            "thermal_time_constant": self.thermal_time_constant,
            "heat_loss_coefficient": self.heat_loss_coefficient,
            "outlet_effectiveness": self.outlet_effectiveness,
            "pv_heat_weight": self.pv_heat_weight,
            "fireplace_heat_weight": self.fireplace_heat_weight,
            "tv_heat_weight": self.tv_heat_weight,
            "solar_lag_minutes": self.solar_lag_minutes,
            "slab_time_constant_hours": self.slab_time_constant_hours,
        }

        if self.orchestrator is None:
            return current_parameters

        # Pull channel-only values from the live orchestrator so exports
        # always reflect the active runtime authority, not stale persisted
        # state.
        channel_parameters = self.orchestrator.get_all_parameters()
        heat_pump_params = channel_parameters.get("heat_pump", {})
        solar_params = channel_parameters.get("pv", {})
        fireplace_params = channel_parameters.get("fireplace", {})

        current_parameters.update(
            {
                "delta_t_floor": heat_pump_params.get("delta_t_floor", 0.0),
                "cloud_factor_exponent": solar_params.get(
                    "cloud_factor_exponent", 1.0
                ),
                "solar_decay_tau_hours": solar_params.get(
                    "solar_decay_tau_hours", 0.0
                ),
                "fp_heat_output_kw": fireplace_params.get(
                    "fp_heat_output_kw", self.fireplace_heat_weight
                ),
                "fp_decay_time_constant": fireplace_params.get(
                    "fp_decay_time_constant", 0.0
                ),
                "room_spread_delay_minutes": fireplace_params.get(
                    "room_spread_delay_minutes", 0.0
                ),
            }
        )

        return current_parameters

    def _append_parameter_history_record(self, parameter_record: Dict) -> None:
        """Append a parameter-history record and keep the in-memory list capped."""
        self.parameter_history.append(parameter_record)
        if len(self.parameter_history) > config.MAX_PARAMETER_HISTORY:
            self.parameter_history = self.parameter_history[
                -config.MAX_PARAMETER_HISTORY:
            ]

    def _normalize_parameter_snapshot(
        self, parameters: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Normalize a parameter snapshot to plain float values."""
        normalized: Dict[str, float] = {}
        for name, value in (parameters or {}).items():
            try:
                normalized[name] = float(value)
            except (TypeError, ValueError):
                continue
        return normalized

    def _normalize_channel_parameter_changes(
        self, changes: Optional[Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """Normalize structured channel change snapshots for persistence."""
        normalized: Dict[str, Dict[str, float]] = {}
        for name, change in (changes or {}).items():
            if not isinstance(change, dict):
                continue
            try:
                normalized[name] = {
                    "before": float(change["before"]),
                    "after": float(change["after"]),
                    "delta": float(change["delta"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
        return normalized

    def _build_channel_parameter_history_records(
        self, update_events: List[Dict], timestamp: Optional[str]
    ) -> List[Dict]:
        """Build top-level parameter-history records from channel update events."""
        if not update_events:
            return []

        current_snapshot = self._get_current_export_parameters()
        records: List[Dict] = []

        for event in update_events:
            structured_changes = self._normalize_channel_parameter_changes(
                event.get("changes")
            )
            try:
                attributed_error = float(
                    event.get("attributed_error", event.get("error", 0.0))
                )
            except (TypeError, ValueError):
                attributed_error = 0.0
            try:
                raw_prediction_error = float(
                    event.get("raw_prediction_error", attributed_error)
                )
            except (TypeError, ValueError):
                raw_prediction_error = attributed_error

            active_contributions: Dict[str, float] = {}
            for name, value in (event.get("active_contributions") or {}).items():
                try:
                    active_contributions[name] = float(value)
                except (TypeError, ValueError):
                    continue

            history_record = {
                "timestamp": timestamp,
                "record_type": event.get("record_type", "channel_update"),
                "channel": event.get("channel"),
                "learning_rate": event.get("learning_rate"),
                "learning_confidence": self.learning_confidence,
                "avg_recent_error": abs(attributed_error),
                "raw_prediction_error": raw_prediction_error,
                "attributed_error": attributed_error,
                "attribution_applied": bool(
                    event.get("attribution_applied", False)
                ),
                "active_contributions": active_contributions,
                "heat_pump_frozen_by_fireplace": bool(
                    event.get("heat_pump_frozen_by_fireplace", False)
                ),
                "parameters_before": self._normalize_parameter_snapshot(
                    event.get("parameters_before")
                ),
                "parameters_after": self._normalize_parameter_snapshot(
                    event.get("parameters_after")
                ),
                "channel_parameter_changes": structured_changes,
                "gradients": {},
                "changes": {
                    name: abs(change["delta"])
                    for name, change in structured_changes.items()
                },
            }
            history_record.update(current_snapshot)
            records.append(history_record)

        return records

    def _get_recent_parameter_snapshot_records(self, limit: int) -> List[Dict]:
        """Return recent history records that contain the core parameter snapshot."""
        if limit <= 0:
            return []

        records: List[Dict] = []
        required_keys = (
            "thermal_time_constant",
            "heat_loss_coefficient",
            "outlet_effectiveness",
        )

        for record in reversed(self.parameter_history):
            try:
                for key in required_keys:
                    float(record[key])
            except (KeyError, TypeError, ValueError):
                continue

            records.append(record)
            if len(records) >= limit:
                break

        return list(reversed(records))

    def get_adaptive_learning_metrics(self) -> Dict:
        """
        ENHANCED: Get metrics with additional debugging info.
        """
        # Allow export even with minimal history for debugging
        if len(self.prediction_history) < 1:
            return {
                "insufficient_data": True,
                "history_len": len(self.prediction_history),
                "required_len": 1
            }

        recent_errors = [
            abs(p["error"]) for p in self.prediction_history[-20:]
        ]
        all_errors = [abs(p["error"]) for p in self.prediction_history]

        if len(recent_errors) >= 10:
            first_half_errors = recent_errors[: len(recent_errors) // 2]
            second_half_errors = recent_errors[len(recent_errors) // 2:]
            error_improvement = np.mean(first_half_errors) - np.mean(
                second_half_errors
            )
        else:
            error_improvement = 0.0

        recent_params = self._get_recent_parameter_snapshot_records(5)
        if len(recent_params) >= 5:
            thermal_stability = np.std(
                [p["thermal_time_constant"] for p in recent_params]
            )
            heat_loss_coefficient_stability = np.std(
                [p["heat_loss_coefficient"] for p in recent_params]
            )
            outlet_effectiveness_stability = np.std(
                [p["outlet_effectiveness"] for p in recent_params]
            )
            recent_gradients = recent_params[-1].get("gradients", {})
        else:
            thermal_stability = 0.0
            heat_loss_coefficient_stability = 0.0
            outlet_effectiveness_stability = 0.0
            recent_gradients = {}

        return {
            "total_predictions": len(self.prediction_history),
            "parameter_updates": len(self.parameter_history),
            "update_percentage": (
                len(self.parameter_history)
                / len(self.prediction_history)
                * 100
                if self.prediction_history
                else 0
            ),
            "avg_recent_error": np.mean(recent_errors),
            "avg_all_time_error": np.mean(all_errors),
            "error_improvement_trend": error_improvement,
            "learning_confidence": self.learning_confidence,
            "current_learning_rate": self._calculate_adaptive_learning_rate(),
            "heat_source_channels_enabled": self.orchestrator is not None,
            "thermal_time_constant_stability": thermal_stability,
            "heat_loss_coefficient_stability": heat_loss_coefficient_stability,
            "outlet_effectiveness_stability": outlet_effectiveness_stability,
            "recent_gradients": recent_gradients,
            "current_parameters": self._get_current_export_parameters(),
            "fixes_applied": "VERSION_WITH_CORRECTED_GRADIENTS",
        }

    def _save_learning_to_thermal_state(
        self,
        new_thermal_adjustment,
        new_heat_loss_coefficient_adjustment,
        new_outlet_effectiveness_adjustment,
        new_pv_heat_weight_adjustment=0.0,
        new_tv_heat_weight_adjustment=0.0,
        new_solar_lag_adjustment=0.0,
        new_slab_adjustment=0.0,
    ):
        """
        Save learned parameter adjustments to unified thermal state.
        """
        try:
            try:
                from .unified_thermal_state import get_thermal_state_manager
            except ImportError:
                from unified_thermal_state import get_thermal_state_manager

            state_manager = get_thermal_state_manager()
            learning_state = state_manager.state.get("learning_state", {})

            current_deltas = learning_state.get("parameter_adjustments", {})
            current_thermal_delta = current_deltas.get(
                "thermal_time_constant_delta", 0.0
            )
            current_heat_loss_coefficient_delta = current_deltas.get(
                "heat_loss_coefficient_delta", 0.0
            )
            current_outlet_effectiveness_delta = current_deltas.get(
                "outlet_effectiveness_delta", 0.0
            )
            current_pv_heat_weight_delta = current_deltas.get(
                "pv_heat_weight_delta", 0.0
            )
            current_tv_heat_weight_delta = current_deltas.get(
                "tv_heat_weight_delta", 0.0
            )
            current_solar_lag_delta = current_deltas.get(
                "solar_lag_minutes_delta", 0.0
            )
            current_slab_delta = current_deltas.get(
                "slab_time_constant_delta", 0.0
            )

            updated_thermal_delta = (
                current_thermal_delta + new_thermal_adjustment
            )
            updated_heat_loss_coefficient_delta = (
                current_heat_loss_coefficient_delta
                + new_heat_loss_coefficient_adjustment
            )
            updated_outlet_effectiveness_delta = (
                current_outlet_effectiveness_delta
                + new_outlet_effectiveness_adjustment
            )
            updated_pv_heat_weight_delta = (
                current_pv_heat_weight_delta + new_pv_heat_weight_adjustment
            )
            updated_tv_heat_weight_delta = (
                current_tv_heat_weight_delta + new_tv_heat_weight_adjustment
            )
            updated_solar_lag_delta = (
                current_solar_lag_delta + new_solar_lag_adjustment
            )
            updated_slab_delta = (
                current_slab_delta + new_slab_adjustment
            )

            if (
                abs(new_thermal_adjustment) > 0.001
                or abs(new_heat_loss_coefficient_adjustment) > 0.0001
                or abs(new_outlet_effectiveness_adjustment) > 0.0001
                or abs(new_pv_heat_weight_adjustment) > 0.0001
                or abs(new_tv_heat_weight_adjustment) > 0.0001
                or abs(new_solar_lag_adjustment) > 0.1
                or abs(new_slab_adjustment) > 0.001
            ):
                # Persist the latest parameter record BEFORE save_state is
                # triggered by update_learning_state. This fixes the shared-
                # reference bug: after the in-memory list is trimmed
                # (self.parameter_history = self.parameter_history[-config.MAX_PARAMETER_HISTORY:]),
                # it is no longer the same object as
                # state["learning_state"]["parameter_history"], so appends
                # to self.parameter_history are never written to JSON.
                # add_parameter_history_record also keeps the JSON list
                # capped at 500 (same as the in-memory cap).
                if self.parameter_history:
                    state_manager.add_parameter_history_record(
                        self.parameter_history[-1]
                    )

                state_manager.update_learning_state(
                    learning_confidence=self.learning_confidence,
                    parameter_adjustments={
                        "thermal_time_constant_delta": updated_thermal_delta,
                        "heat_loss_coefficient_delta": (
                            updated_heat_loss_coefficient_delta
                        ),
                        "outlet_effectiveness_delta": (
                            updated_outlet_effectiveness_delta
                        ),
                        "pv_heat_weight_delta": updated_pv_heat_weight_delta,
                        "tv_heat_weight_delta": updated_tv_heat_weight_delta,
                        "solar_lag_minutes_delta": updated_solar_lag_delta,
                        "slab_time_constant_delta": updated_slab_delta,
                    },
                )
                logging.debug(
                    (
                        "💾 Accumulated learning deltas: "
                        "thermal_Δ=%+.3f (+%+.3f), "
                        "heat_loss_coeff_Δ=%+.5f (+%+.5f), "
                        "outlet_eff_Δ=%+.3f (+%+.3f), "
                        "pv_weight_Δ=%+.3f (+%+.3f), "
                        "tv_weight_Δ=%+.3f (+%+.3f), "
                        "solar_lag_Δ=%+.3f (+%+.3f), "
                        "slab_tau_Δ=%+.4f (+%+.4f)"
                    ),
                    updated_thermal_delta,
                    new_thermal_adjustment,
                    updated_heat_loss_coefficient_delta,
                    new_heat_loss_coefficient_adjustment,
                    updated_outlet_effectiveness_delta,
                    new_outlet_effectiveness_adjustment,
                    updated_pv_heat_weight_delta,
                    new_pv_heat_weight_adjustment,
                    updated_tv_heat_weight_delta,
                    new_tv_heat_weight_adjustment,
                    updated_solar_lag_delta,
                    new_solar_lag_adjustment,
                    updated_slab_delta,
                    new_slab_adjustment,
                )
            else:
                state_manager.update_learning_state(
                    learning_confidence=self.learning_confidence
                )
                logging.debug(
                    f"💾 Updated learning confidence: "
                    f"{self.learning_confidence:.3f} "
                    f"(no significant parameter changes)"
                )
        except Exception as e:
            logging.error(
                f"❌ Failed to save learning to thermal state: {e}"
            )

    def _detect_parameter_corruption(self) -> bool:
        """
        Detect if parameters are in a corrupted state.
        """
        # Check heat_loss_coefficient bounds
        hcl_bounds = ThermalParameterConfig.get_bounds("heat_loss_coefficient")
        if not (hcl_bounds[0] <= self.heat_loss_coefficient <= hcl_bounds[1]):
            logging.warning(
                "heat_loss_coefficient %s is outside of bounds %s",
                self.heat_loss_coefficient,
                hcl_bounds,
            )
            return True

        # Check outlet_effectiveness bounds
        oe_bounds = ThermalParameterConfig.get_bounds("outlet_effectiveness")
        if not (oe_bounds[0] <= self.outlet_effectiveness <= oe_bounds[1]):
            logging.warning(
                "outlet_effectiveness %s is outside of bounds %s",
                self.outlet_effectiveness,
                oe_bounds,
            )
            return True

        # Check for physically impossible combinations (drift detection)
        # High heat loss + low effectiveness = broken physics model
        # Also catch extreme heat loss regardless of effectiveness
        if self.heat_loss_coefficient > 1.8:
            logging.warning(
                "⚠️ Extreme heat loss detected (%.2f). Resetting to prevent "
                "runaway heating.",
                self.heat_loss_coefficient,
            )
            return True

        if (
            self.heat_loss_coefficient > 1.2
            and self.outlet_effectiveness < 0.4
        ):
            logging.warning(
                "⚠️ Parameter drift detected: High heat loss (%.2f) "
                "with low effectiveness (%.2f) indicates broken learning "
                "state",
                self.heat_loss_coefficient,
                self.outlet_effectiveness,
            )
            return True

        # Enhanced check for "borderline bad" combination that causes 65C
        # Lowered threshold to catch more cases (0.8 -> 0.6)
        if (
            self.heat_loss_coefficient > 0.6
            and self.outlet_effectiveness < 0.35
        ):
            logging.warning(
                "⚠️ Parameter drift detected: Moderate heat loss (%.2f) "
                "with very low effectiveness (%.2f) causes extreme "
                "predictions",
                self.heat_loss_coefficient,
                self.outlet_effectiveness,
            )
            return True

        # Check thermal_time_constant bounds
        ttc_bounds = ThermalParameterConfig.get_bounds("thermal_time_constant")
        if not (ttc_bounds[0] <= self.thermal_time_constant <= ttc_bounds[1]):
            logging.warning(
                "thermal_time_constant %s is outside of bounds %s",
                self.thermal_time_constant,
                ttc_bounds,
            )
            return True

        return False

    def reset_adaptive_learning(self):
        """Reset adaptive learning state with aggressive initial settings."""
        self._load_config_defaults()
        self.prediction_history = []
        self.parameter_history = []
        self.learning_confidence = 3.0  # Start with high confidence
        logging.info(
            "Adaptive learning state reset with aggressive settings"
        )
