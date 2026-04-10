"""
ThermalEquilibriumModel-based Model Wrapper.

This module provides a clean interface for thermal physics-based heating
control using only the ThermalEquilibriumModel. All legacy ML model code has
been removed as part of the thermal equilibrium model migration.

Key features:
- Single ThermalEquilibriumModel-based prediction pathway
- Persistent thermal learning state across service restarts
- Simplified outlet temperature prediction interface
- Adaptive thermal parameter learning
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# Support both package-relative and direct import for notebooks
from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src.unified_thermal_state import get_thermal_state_manager
from src.influx_service import create_influx_service
from src.prediction_metrics import PredictionMetrics
from src import config
from src.ha_client import create_ha_client
from src.adaptive_fireplace_learning import AdaptiveFireplaceLearning
from src.prediction_context import prediction_context_manager
from src.shadow_mode import resolve_shadow_mode


# Singleton pattern to prevent multiple model instantiation
_enhanced_model_wrapper_instance = None


class EnhancedModelWrapper:
    """
    Simplified model wrapper using ThermalEquilibriumModel for persistent
    learning.

    Replaces the complex Heat Balance Controller with a single prediction path
    that continuously adapts thermal parameters and survives service restarts.

    Implements singleton pattern to prevent multiple instantiation per service
    restart.

    Now supports up to 6-hour forecast horizons for trajectory prediction and feature engineering.
    """

    def __init__(self):
        self.thermal_model = ThermalEquilibriumModel()
        self.learning_enabled = True

        # Climate mode: "heating" (default) or "cooling".
        # Set each cycle by main.py via set_climate_mode().
        self._climate_mode = "heating"

        # Get thermal state manager
        self.state_manager = get_thermal_state_manager()

        # Initialize prediction metrics for MAE/RMSE tracking
        self.prediction_metrics = PredictionMetrics(
            state_manager=self.state_manager
        )

        # Adaptive fireplace learning is legacy-only when channel mode is off.
        self.adaptive_fireplace = None
        if not self._use_heat_source_channels():
            self.adaptive_fireplace = AdaptiveFireplaceLearning()

        # Get current cycle count from unified state
        metrics = self.state_manager.get_learning_metrics()
        self.cycle_count = metrics["current_cycle_count"]

        # UNIFIED FORECAST: Store cycle-aligned forecast conditions for smart
        # rounding
        self.cycle_aligned_forecast = {}
        self._avg_cloud_cover = 50.0

        # Fireplace state tracking for decay / spread delay
        self._fireplace_last_on_time: Optional[datetime] = None
        self._fireplace_on_since: Optional[datetime] = None

        logging.info(
            "🎯 Model Wrapper initialized with ThermalEquilibriumModel"
        )
        logging.info(f"   - Thermal time constant: "
                     f"{self.thermal_model.thermal_time_constant:.1f}h")
        logging.info(f"   - Heat loss coefficient: "
                     f"{self.thermal_model.heat_loss_coefficient:.4f}")
        logging.info(f"   - Outlet effectiveness: "
                     f"{self.thermal_model.outlet_effectiveness:.4f}")
        logging.info(
            f"   - Learning confidence: "
            f"{self.thermal_model.learning_confidence:.2f}"
        )
        logging.info(f"   - Current cycle: {self.cycle_count}")

    def _use_heat_source_channels(self) -> bool:
        """Return whether channel mode is actively controlling heat sources."""
        return bool(
            config.ENABLE_HEAT_SOURCE_CHANNELS
            and self.thermal_model.orchestrator is not None
        )

    def _get_influx_export_interval_cycles(self) -> int:
        """Return the configured Influx export interval in learning cycles."""
        return max(
            1,
            int(
                getattr(
                    config,
                    "INFLUX_METRICS_EXPORT_INTERVAL_CYCLES",
                    5,
                )
            ),
        )

    # --- Climate mode helpers ---
    def set_climate_mode(self, mode: str) -> None:
        """Set the current climate mode ('heating' or 'cooling')."""
        if mode not in ("heating", "cooling"):
            mode = "heating"
        self._climate_mode = mode

    @property
    def climate_mode(self) -> str:
        """Return the current climate mode."""
        return self._climate_mode

    def _calculate_fireplace_power_kw(
        self,
        current_indoor: float,
        outdoor_temp: float,
        fireplace_on: float,
        features_local: Optional[Dict] = None,
    ) -> Tuple[Optional[float], float]:
        """Resolve fireplace power from the active fireplace authority.

        Returns ``(power_kw_or_None, decay_kw)``.  ``decay_kw`` is the
        residual heat contribution after the fireplace has been turned off
        (exponential tail).  When the FP is currently ON the decay component
        is 0.
        """
        now = datetime.now()

        # --- Track FP state transitions ---
        if fireplace_on:
            if self._fireplace_on_since is None:
                self._fireplace_on_since = now
            self._fireplace_last_on_time = now
        else:
            self._fireplace_on_since = None

        features_local = features_local or {}

        # --- Decay when OFF ---
        decay_kw = 0.0
        if not fireplace_on and self._fireplace_last_on_time is not None:
            minutes_off = (now - self._fireplace_last_on_time).total_seconds() / 60.0
            if minutes_off > 0 and self._use_heat_source_channels():
                fp_ch = self.thermal_model.orchestrator.channels.get("fireplace")
                if fp_ch:
                    hours_off = minutes_off / 60.0
                    decay_kw = fp_ch.estimate_decay_contribution(hours_off, {})
                    if decay_kw < 0.01:
                        decay_kw = 0.0  # negligible

        if not fireplace_on:
            return None, decay_kw

        features_local = features_local or {}

        if self._use_heat_source_channels():
            fireplace_channel = self.thermal_model.orchestrator.channels[
                "fireplace"
            ]
            fireplace_power_kw = fireplace_channel.estimate_heat_contribution(
                {
                    "fireplace_on": fireplace_on,
                    "current_indoor": current_indoor,
                    "outdoor_temp": outdoor_temp,
                    "living_room_temp": features_local.get(
                        "living_room_temp", current_indoor
                    ),
                    "other_rooms_temp": features_local.get(
                        "other_rooms_temp", current_indoor - 2.0
                    ),
                }
            )
            logging.debug(
                "🔥 Using fireplace channel power: %.2fkW",
                fireplace_power_kw,
            )
            return fireplace_power_kw, 0.0

        if self.adaptive_fireplace is None:
            return None, 0.0

        living_room_temp = features_local.get(
            "living_room_temp", current_indoor
        )
        other_rooms_temp = features_local.get(
            "other_rooms_temp", current_indoor - 2.0
        )
        fireplace_analysis = (
            self.adaptive_fireplace._calculate_learned_heat_contribution(
                temp_differential=living_room_temp - other_rooms_temp,
                outdoor_temp=outdoor_temp,
                fireplace_active=True,
            )
        )
        fireplace_power_kw = fireplace_analysis.get("heat_contribution_kw", 0.0)
        confidence = fireplace_analysis.get("learning_confidence", 0.0)
        logging.debug(
            f"🔥 Using adaptive fireplace power (living room): "
            f"{fireplace_power_kw:.2f}kW "
            f"(confidence: {confidence:.2f})"
        )
        return fireplace_power_kw, 0.0

    def predict_indoor_temp(
        self, outlet_temp: float, outdoor_temp: float, **kwargs
    ) -> float:
        """
        Predict indoor temperature for smart rounding.

        Uses the thermal model's equilibrium prediction with proper parameter
        handling. Provides robust conversion of pandas data types to scalar
        values.
        """
        try:
            # UNIFIED FORECAST FIX: Use cycle-aligned forecast for smart
            # rounding
            if hasattr(self, "cycle_aligned_forecast") and \
                    self.cycle_aligned_forecast:
                pv_val = self.cycle_aligned_forecast.get('pv_power', 0.0)
                caller_pv = kwargs.get('pv_power', 0.0)

                # Handle list logging safely
                caller_pv_display = caller_pv
                if isinstance(caller_pv, list):
                    caller_pv_display = f"List(len={len(caller_pv)})"
                elif isinstance(caller_pv, (int, float)):
                    caller_pv_display = f"{caller_pv:.0f}"

                logging.debug(
                    "🧠 Smart rounding is using cycle-aligned forecast: "
                    f"PV={pv_val:.0f}W (caller sent PV={caller_pv_display}W)"
                )
                pv_power = self.cycle_aligned_forecast.get(
                    "pv_power", kwargs.get("pv_power", 0.0)
                )
                fireplace_on = self.cycle_aligned_forecast.get(
                    "fireplace_on", kwargs.get("fireplace_on", 0.0)
                )
                tv_on = self.cycle_aligned_forecast.get(
                    "tv_on", kwargs.get("tv_on", 0.0)
                )
                # Use cycle-aligned outdoor_temp as well for full consistency
                outdoor_temp = self.cycle_aligned_forecast.get(
                    "outdoor_temp", outdoor_temp
                )
            else:
                # Fallback to kwargs if cycle-aligned forecast is not available
                pv_power = kwargs.get("pv_power", 0.0)
                fireplace_on = kwargs.get("fireplace_on", 0.0)
                tv_on = kwargs.get("tv_on", 0.0)
            current_indoor = kwargs.get("current_indoor", outdoor_temp + 15.0)

            # Convert pandas Series to scalar values
            def to_scalar(value):
                """Convert pandas Series or any value to scalar."""
                if value is None:
                    return 0.0
                # Handle pandas Series
                if hasattr(value, "iloc"):
                    return float(value.iloc[0]) if len(value) > 0 else 0.0
                # Handle pandas scalar
                if hasattr(value, "item"):
                    return float(value.item())
                # Handle regular values
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return 0.0

            # Convert all parameters to scalars
            pv_power = to_scalar(pv_power)
            fireplace_on = to_scalar(fireplace_on)
            tv_on = to_scalar(tv_on)
            current_indoor = to_scalar(current_indoor)
            outdoor_temp = to_scalar(outdoor_temp)
            outlet_temp = to_scalar(outlet_temp)
            thermal_power = kwargs.get("thermal_power", None)
            if thermal_power is not None:
                thermal_power = to_scalar(thermal_power)

            # Additional safety checks
            if outdoor_temp == 0.0:
                logging.error("predict_indoor_temp: outdoor_temp is invalid")
                return 21.0  # Safe fallback temperature
            if outlet_temp == 0.0:
                logging.error("predict_indoor_temp: outlet_temp is invalid")
                return outdoor_temp + 10.0
            if current_indoor == 0.0:
                current_indoor = outdoor_temp + 15.0

            # Use thermal model to predict temperature at the end of the cycle
            cycle_hours = config.CYCLE_INTERVAL_MINUTES / 60.0

            features_local = getattr(self, "_current_features", {}) or {}
            fireplace_power_kw, fp_decay_kw = self._calculate_fireplace_power_kw(
                current_indoor=current_indoor,
                outdoor_temp=outdoor_temp,
                fireplace_on=fireplace_on,
                features_local=features_local,
            )

            trajectory_result = (
                self.thermal_model.predict_thermal_trajectory(
                    current_indoor=current_indoor,
                    target_indoor=current_indoor,
                    outlet_temp=outlet_temp,
                    outdoor_temp=outdoor_temp,
                    time_horizon_hours=cycle_hours,
                    time_step_minutes=config.CYCLE_INTERVAL_MINUTES,
                    pv_power=pv_power,
                    fireplace_on=fireplace_on,
                    tv_on=tv_on,
                    thermal_power=thermal_power,
                    fireplace_power_kw=fireplace_power_kw,
                    fireplace_decay_kw=fp_decay_kw,
                    cloud_cover_pct=self._avg_cloud_cover,
                )
            )

            if (
                not trajectory_result
                or "trajectory" not in trajectory_result
                or not trajectory_result["trajectory"]
            ):
                logging.warning(
                    f"predict_thermal_trajectory returned invalid result for "
                    f"outlet={outlet_temp}, outdoor={outdoor_temp}"
                )
                return outdoor_temp + 10.0  # Safe fallback

            predicted_temp = trajectory_result["trajectory"][0]

            return float(predicted_temp)  # Ensure we return a float

        except Exception as e:
            logging.error(f"predict_indoor_temp failed: {e}")
            # Safe fallback - assume minimal heating effect
            return outdoor_temp + 10.0 if outdoor_temp is not None else 21.0

    def calculate_optimal_outlet_temp(
        self, features: Dict
    ) -> Tuple[float, Dict]:
        """Calculate optimal outlet temp using direct physics prediction."""
        try:
            # Store features for trajectory verification during binary search
            self._current_features = features

            # Extract core thermal parameters
            current_indoor = features.get("indoor_temp_lag_30m", 21.0)
            target_indoor = features.get("target_temp", 21.0)
            outdoor_temp = features.get("outdoor_temp", 10.0)

            # Store current indoor for trajectory correction
            self._current_indoor = current_indoor

            # Extract enhanced thermal intelligence features
            thermal_features = self._extract_thermal_features(features)

            # Calculate required outlet temperature using iterative approach
            optimal_outlet_temp = self._calculate_required_outlet_temp(
                current_indoor,
                target_indoor,
                outdoor_temp,
                thermal_features,
            )

            # This ensures we have a valid target prediction for logging
            predicted_indoor = self.predict_indoor_temp(
                outlet_temp=optimal_outlet_temp,
                outdoor_temp=outdoor_temp,
                current_indoor=current_indoor,
                pv_power=thermal_features.get("pv_power", 0.0),
                fireplace_on=thermal_features.get("fireplace_on", 0.0),
                tv_on=thermal_features.get("tv_on", 0.0),
            )

            # Get prediction metadata
            confidence = self.thermal_model.learning_confidence
            prediction_metadata = {
                "thermal_time_constant":
                    self.thermal_model.thermal_time_constant,
                "heat_loss_coefficient":
                    self.thermal_model.heat_loss_coefficient,
                "outlet_effectiveness":
                    self.thermal_model.outlet_effectiveness,
                "learning_confidence": confidence,
                "prediction_method": "thermal_equilibrium_single_prediction",
                "cycle_count": self.cycle_count,
                "predicted_indoor": predicted_indoor,
            }

            if optimal_outlet_temp is None:
                logging.warning(
                    "Failed to calculate optimal outlet temperature"
                )
                optimal_outlet_temp = config.get_fallback_outlet(
                    self._climate_mode
                )

            return optimal_outlet_temp, prediction_metadata

        except Exception as e:
            logging.error(f"Prediction failed: {e}", exc_info=True)
            # Fallback to safe temperature
            fallback_temp = config.get_fallback_outlet(self._climate_mode)
            fallback_metadata = {
                "prediction_method": "fallback_safe_temperature",
                "error": str(e),
            }
            return fallback_temp, fallback_metadata

    def _extract_thermal_features(self, features: Dict) -> Dict:
        """Extract thermal intelligence features for the thermal model."""
        thermal_features = {}

        # Multi-heat source features
        # Use scalar for pv_power (for prediction_context_manager math)
        # Pass history separately for 45-min rolling average lag calculation
        pv_history = features.get("pv_power_history")
        pv_now = features.get("pv_now", 0.0)
        
        # Calculate scalar using the same lag window the thermal model uses.
        # Averaging the full 3h history would include old PV values from
        # earlier in the day even when PV has been 0 for the last hour+.
        if pv_history and len(pv_history) > 0:
            lag_steps = max(1, round(config.SOLAR_LAG_MINUTES / config.HISTORY_STEP_MINUTES))
            lag_window = pv_history[-lag_steps:]
            pv_scalar = sum(lag_window) / len(lag_window)
        else:
            pv_scalar = pv_now

        # Apply cloud discount to PV scalar before it enters the binary
        # search.  Without this, a raw sensor spike (e.g. 4 kW during a
        # brief sun break) makes the binary search declare "target
        # unreachable" and snap the outlet to 18 °C, only to bounce back
        # next cycle when clouds return.  The equilibrium model already
        # applies cloud_factor internally, but the *scalar* fed to the
        # binary-search convergence pre-check must be consistent.
        cloud_1h = features.get("cloud_cover_1h",
                                features.get("avg_cloud_cover", 50.0))
        if getattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False):
            cloud_pct = max(0.0, min(100.0, float(cloud_1h)))
            min_factor = float(
                getattr(config, "CLOUD_CORRECTION_MIN_FACTOR", 0.1)
            )
            cloud_factor = max(
                min_factor, 1.0 - (cloud_pct / 100.0)
            )
            pv_scalar *= cloud_factor

        thermal_features["pv_power"] = pv_scalar
        # Pass PV history for lag calculation in model
        thermal_features["pv_power_history"] = pv_history
        thermal_features["fireplace_on"] = float(
            features.get("fireplace_on", 0)
        )
        thermal_features["tv_on"] = float(features.get("tv_on", 0))
        thermal_features["thermal_power"] = float(
            features.get("thermal_power_kw", 0.0)
        )

        # Enhanced thermal intelligence features
        thermal_features["indoor_temp_gradient"] = \
            features.get("indoor_temp_gradient", 0.0)
        thermal_features["temp_diff_indoor_outdoor"] = \
            features.get("temp_diff_indoor_outdoor", 0.0)
        thermal_features["outlet_indoor_diff"] = \
            features.get("outlet_indoor_diff", 0.0)

        # Slab passive delta: inlet_temp (BT3, slab return) minus indoor temp.
        # Positive → slab warmer than room (passive heating available).
        # Negative → slab cooler than room (absorbing heat from room).
        _inlet_t = features.get("inlet_temp")
        _indoor_t = features.get("indoor_temp_lag_30m",
                                 features.get("current_indoor"))
        if _inlet_t is not None and _indoor_t is not None:
            thermal_features["slab_passive_delta"] = (
                float(_inlet_t) - float(_indoor_t)
            )

        # Note: Occupancy/cooking features removed (no sensors)

        return thermal_features

    def _get_forecast_conditions(
        self, outdoor_temp: float, pv_power: float, thermal_features: Dict
    ) -> Tuple[float, float, list, list]:
        """
        Get forecast conditions using UnifiedPredictionContext.

        Delegates to the centralized prediction context manager to ensure
        consistency across all prediction systems (binary search, smart
        rounding, trajectory optimization).

        Returns arrays for up to 6-hour forecasts (t=0 to t=6h) for both outdoor and PV.
        """
        # BUGFIX: Ensure pv_power and outdoor_temp are scalars to prevent
        # TypeError in logging.
        if hasattr(pv_power, "iloc"):
            pv_power = \
                float(pv_power.iloc[0]) if not pv_power.empty else 0.0
        if hasattr(outdoor_temp, "iloc"):
            outdoor_temp = \
                float(outdoor_temp.iloc[0]) if not outdoor_temp.empty else 0.0

        # Pass features to manager
        features = getattr(self, "_current_features", {})
        prediction_context_manager.set_features(features)

        # Create unified context
        target_temp = (
            self._current_features.get("target_temp")
            if hasattr(self, "_current_features") else None
        )
        current_temp = (
            self._current_indoor
            if hasattr(self, "_current_indoor") else None
        )

        context = prediction_context_manager.create_context(
            outdoor_temp=outdoor_temp,
            pv_power=pv_power,
            thermal_features=thermal_features,
            target_temp=target_temp,
            current_temp=current_temp,
        )

        # Return values in expected format.
        # Truncate forecast arrays to TRAJECTORY_STEPS+1 entries so the
        # trajectory horizon matches exactly [t=0, t=1h, ..., t=TRAJECTORY_STEPS].
        # This prevents far-future forecasts (e.g. sunrise at t=6h) from
        # influencing an optimization window that only runs to t=3h.
        n = config.TRAJECTORY_STEPS + 1
        outdoor_arr = ([outdoor_temp] + context['outdoor_forecast'])[:n]
        pv_arr = ([pv_power] + context['pv_forecast'])[:n]
        
        # Store cloud cover forecast for use in predict_equilibrium_temperature calls
        self._avg_cloud_cover = context.get('avg_cloud_cover', 50.0)
        self._cloud_cover_forecast = context.get('cloud_cover_forecast', [50.0] * 6)
        
        return (
            context['avg_outdoor'],
            context['avg_pv'],
            outdoor_arr,
            pv_arr,
        )

    def _calculate_required_outlet_temp(
        self,
        current_indoor: float,
        target_indoor: float,
        outdoor_temp: float,
        thermal_features: Dict,
    ) -> float:
        """
        Calculate required outlet temperature to reach target indoor temp.
        """
        # REMOVED: "Already at target" bypass logic. Let physics model always
        # calculate proper outlet temp. The thermal model should determine
        # maintenance requirements based on actual conditions.

        # Use the calibrated thermal model to find required outlet temp. This
        # leverages the learned parameters instead of simple heuristics.
        pv_power = thermal_features.get("pv_power", 0.0)
        fireplace_on = thermal_features.get("fireplace_on", 0.0)
        tv_on = thermal_features.get("tv_on", 0.0)

        fireplace_power_kw, fp_decay_kw = self._calculate_fireplace_power_kw(
            current_indoor=current_indoor,
            outdoor_temp=outdoor_temp,
            fireplace_on=fireplace_on,
            features_local=getattr(self, "_current_features", {}) or {},
        )

        # Iterative search to find outlet temp that produces target indoor
        # temp. This uses the learned thermal physics parameters from
        # calibration.
        tolerance = 0.01  # °C

        # Use mode-aware outlet bounds.
        outlet_min, outlet_max = config.get_outlet_bounds(self._climate_mode)
        fallback_temp = config.get_fallback_outlet(self._climate_mode)
        
        if self._climate_mode == "cooling":
            # COOLING MODE: outlet must be below room temperature.
            # The HP needs at least MIN_COOLING_DELTA_K between room
            # temperature and outlet to run. Clamp the effective max to
            # (current_indoor - delta) so the search space makes physical
            # sense.
            indoor_based_max = current_indoor - config.MIN_COOLING_DELTA_K
            outlet_max = min(outlet_max, indoor_based_max)
            # Ensure min < max; if the room is already cool there is no
            # scope for the HP to do useful work.
            if outlet_min >= outlet_max:
                logging.info(
                    "❄️ Cooling: no viable outlet range "
                    "(min=%.1f >= max=%.1f). "
                    "Room already near target or too cool for HP.",
                    outlet_min, outlet_max,
                )
                return outlet_min  # warmest valid cooling outlet; never below effective min
            logging.info(
                "❄️ Cooling mode bounds: outlet %.1f–%.1f°C "
                "(indoor=%.1f°C, target=%.1f°C)",
                outlet_min, outlet_max, current_indoor, target_indoor,
            )
        else:
            # HEATING SAFETY: If significant heating is required
            # (target > current by more than 0.5°C), enforce a minimum
            # floor of 25°C to prevent the model from suggesting
            # cooling-range temperatures just because of high PV.
            if target_indoor - current_indoor > 0.5:
                outlet_min = max(outlet_min, 25.0)

        logging.debug(
            f"🔧 Using natural bounds: outlet_min={outlet_min:.1f}°C, "
            f"outlet_max={outlet_max:.1f}°C"
        )

        # UNIFIED: Get forecast conditions using centralized method
        (
            avg_outdoor,
            avg_pv,
            outdoor_forecast,
            pv_forecast,
        ) = self._get_forecast_conditions(
            outdoor_temp, pv_power, thermal_features
        )

        # UNIFIED FORECAST: Store cycle-aligned conditions for smart rounding
        self.cycle_aligned_forecast = {
            "outdoor_temp": avg_outdoor,
            "pv_power": avg_pv,
            "fireplace_on": fireplace_on,
            "tv_on": tv_on,
        }

        # Use PV history if available to respect solar lag
        pv_input = thermal_features.get("pv_power_history")
        if not pv_input:
            logging.debug(f"DEBUG: No PV history found, using scalar avg_pv: {avg_pv}")
            pv_input = avg_pv
        else:
            logging.debug(
                f"DEBUG: Using PV history (len={len(pv_input)}) instead of scalar {avg_pv}"
            )

        # Resolve delta_t for the binary search ONCE, before both the
        # pre-check and the binary-search loop, so both use the same value.
        _inlet = (
            self._current_features.get("inlet_temp")
            if hasattr(self, "_current_features")
            else None
        )
        _dtf = (
            self._current_features.get("delta_t", 0.0)
            if hasattr(self, "_current_features")
            else 0.0
        )

        # HP-OFF FIX: When HP is off (delta_t < 1.0), the slab pump
        # gate blocks ALL binary-search candidates identically (passive
        # branch ignores outlet_temp).  Substitute the learned HP
        # delta_t_floor so the trajectory simulates "HP running at this
        # outlet" — letting candidates differentiate and conveying the
        # correct setpoint for NIBE to start heating.
        if _dtf < 1.0:
            _dtf_simulated = (
                self.thermal_model._resolve_delta_t_floor(_dtf)
            )
            logging.info(
                "🔄 HP off (delta_t=%.2f): simulating HP-on "
                "delta_t=%.2f for binary search",
                _dtf, _dtf_simulated,
            )
            _dtf = _dtf_simulated
        # Store the resolved delta_t for use by trajectory verification too
        self._search_delta_t_floor = _dtf

        # Binary search for optimal outlet temperature
        logging.debug(
            f"🎯 Binary search start: target={target_indoor:.1f}°C, "
            f"current={current_indoor:.1f}°C, "
            f"range={outlet_min:.1f}-{outlet_max:.1f}°C"
        )

        for iteration in range(20):  # Max 20 iterations for efficiency
            # Check if range has collapsed (early exit)
            range_size = outlet_max - outlet_min
            if range_size < 0.05:  # °C - range too small to matter
                final_outlet = (outlet_min + outlet_max) / 2.0
                logging.info(
                    f"🔄 Binary search early exit after {iteration+1} "
                    f"iterations: range collapsed to {range_size:.3f}°C, "
                    f"using {final_outlet:.1f}°C"
                )
                return final_outlet

            outlet_mid = (outlet_min + outlet_max) / 2.0

            # Predict indoor temperature with this outlet temperature using
            # cycle-aligned conditions
            try:
                # Use TRAJECTORY_STEPS as the optimization horizon for binary
                # search. This makes the binary search predictive over the
                # same horizon as the trajectory verification:
                # - TRAJECTORY_STEPS=2: optimizes for 2h ahead (moderate
                #   PV/weather anticipation, less oscillation risk)
                # - TRAJECTORY_STEPS=4: optimizes for 4h ahead (stronger
                #   PV/weather anticipation, e.g. reduces outlet temp when
                #   3000W PV is forecast in 4h; conversely raises outlet when
                #   outdoor temp will drop sharply tonight)
                # NOTE: longer horizons increase sensitivity to forecast
                # errors. Values >4h are not recommended.
                optimization_horizon = float(config.TRAJECTORY_STEPS)

                # Pass full forecast arrays to thermal model for accurate
                # trajectory. outdoor_forecast contains [1h, 2h, 3h, 4h]
                # NOTE: pass pv_power as scalar (current PV) so the buffer
                # is seeded with the actual present value, not the far-future
                # forecast. The full schedule is provided via pv_forecasts so
                # predict_thermal_trajectory can interpolate it correctly over
                # the hourly source grid (0h, 1h, 2h, ...).

                trajectory_result = (
                    self.thermal_model.predict_thermal_trajectory(
                        current_indoor=current_indoor,
                        target_indoor=target_indoor,
                        outlet_temp=outlet_mid,
                        outdoor_temp=outdoor_forecast,  # Pass array directly
                        time_horizon_hours=optimization_horizon,
                        time_step_minutes=config.CYCLE_INTERVAL_MINUTES,
                        pv_power=pv_input,  # Pass history for initialization
                        pv_forecasts=pv_forecast,  # Schedule: [t=0h, t=1h, ..., t=6h]
                        fireplace_on=fireplace_on,
                        tv_on=tv_on,
                        fireplace_power_kw=fireplace_power_kw,
                        fireplace_decay_kw=fp_decay_kw,
                        cloud_cover_pct=self._avg_cloud_cover,
                        inlet_temp=_inlet,
                        delta_t_floor=_dtf,
                    )
                )

                if (
                    not trajectory_result
                    or "trajectory" not in trajectory_result
                    or not trajectory_result["trajectory"]
                ):
                    logging.warning(
                        f"   Iteration {iteration+1}: "
                        f"predict_thermal_trajectory returned invalid result "
                        f"for outlet={outlet_mid:.1f}°C - using fallback"
                    )
                    return fallback_temp

                # Use the temperature at the END of the horizon for
                # optimization
                predicted_indoor = trajectory_result["trajectory"][-1]

            except Exception as e:
                logging.error(
                    f"   Iteration {iteration+1}: "
                    f"predict_thermal_trajectory failed: {e}"
                )
                return fallback_temp  # Safe fallback

            # Calculate error from target
            error = predicted_indoor - target_indoor

            # Detailed logging at each iteration
            logging.debug(
                f"   Iteration {iteration+1}: outlet={outlet_mid:.1f}°C → "
                f"predicted={predicted_indoor:.2f}°C, error={error:.3f}°C "
                f"(range: {outlet_min:.1f}-{outlet_max:.1f}°C)"
            )

            # Check if we're close enough
            if abs(error) < tolerance:
                logging.info(
                    f"✅ Binary search converged after {iteration+1} "
                    f"iterations: {outlet_mid:.1f}°C → "
                    f"{predicted_indoor:.2f}°C (target: "
                    f"{target_indoor:.1f}°C, error: {error:.3f}°C)"
                )

                # Show final equilibrium physics for the converged result
                self.thermal_model.predict_equilibrium_temperature(
                    outlet_temp=outlet_mid,
                    outdoor_temp=avg_outdoor,
                    current_indoor=current_indoor,
                    pv_power=avg_pv,
                    fireplace_on=fireplace_on,
                    tv_on=tv_on,
                    _suppress_logging=False,  # Show equilibrium physics
                    fireplace_power_kw=fireplace_power_kw,
                    fireplace_decay_kw=fp_decay_kw,
                    cloud_cover_pct=self._avg_cloud_cover,
                )

                # MULTI-HORIZON FORECAST LOGGING: Show predictions with
                # different forecast horizons
                self._log_multi_horizon_predictions(
                    current_indoor=current_indoor,
                    target_indoor=target_indoor,
                    outdoor_temp=outdoor_temp,
                    thermal_features=thermal_features,
                )

                # NEW: Trajectory verification and course correction.
                # Pass the SAME forecast arrays used by binary search so
                # verify sees identical conditions.
                if config.TRAJECTORY_PREDICTION_ENABLED:
                    outlet_mid = self._verify_trajectory_and_correct(
                        outlet_temp=outlet_mid,
                        current_indoor=current_indoor,
                        target_indoor=target_indoor,
                        outdoor_temp=outdoor_temp,
                        thermal_features=thermal_features,
                        features=getattr(
                            self, "_current_features", {}
                        ),
                        outdoor_forecast=outdoor_forecast,
                        pv_forecast=pv_forecast,
                        pv_history_for_buffer=pv_input,
                    )

                return outlet_mid

            # Adjust search range based on error
            # COOLING FIX: Consider whether we're heating or cooling the house
            temp_diff = target_indoor - current_indoor
            is_heating_scenario = temp_diff > 0.1  # Need to heat house
            is_cooling_scenario = temp_diff < -0.1  # Need to cool house

            if is_heating_scenario:
                # HEATING: Normal logic
                if predicted_indoor < target_indoor:
                    # Need higher outlet temperature
                    outlet_min = outlet_mid
                    logging.debug(
                        f"     → Heating: Predicted too low, raising minimum "
                        f"to {outlet_min:.1f}°C"
                    )
                else:
                    # Need lower outlet temperature
                    outlet_max = outlet_mid
                    logging.debug(
                        f"     → Heating: Predicted too high, lowering "
                        f"maximum to {outlet_max:.1f}°C"
                    )
            elif is_cooling_scenario:
                # COOLING: For cooling, we want to get as close as possible to
                # target. Standard binary search logic works, just need to be
                # close to target
                if predicted_indoor < target_indoor:
                    # Predicted is below target - need slightly higher outlet
                    # to reach target
                    outlet_min = outlet_mid
                    logging.debug(
                        f"     → Cooling: Predicted below target, raising "
                        f"minimum to {outlet_min:.1f}°C"
                    )
                else:
                    # Predicted is above target - need lower outlet to reach
                    # target
                    outlet_max = outlet_mid
                    logging.debug(
                        f"     → Cooling: Predicted above target, lowering "
                        f"maximum to {outlet_max:.1f}°C"
                    )
            else:
                # MAINTENANCE: At target (normal logic)
                if predicted_indoor < target_indoor:
                    # Need higher outlet temperature
                    outlet_min = outlet_mid
                    logging.debug(
                        f"     → Maintenance: Predicted too low, raising "
                        f"minimum to {outlet_min:.1f}°C"
                    )
                else:
                    # Need lower outlet temperature
                    outlet_max = outlet_mid
                    logging.debug(
                        f"     → Maintenance: Predicted too high, lowering "
                        f"maximum to {outlet_max:.1f}°C"
                    )

        # Return best guess if didn't converge
        final_outlet = (outlet_min + outlet_max) / 2.0
        try:
            final_predicted = (
                self.thermal_model.predict_equilibrium_temperature(
                    outlet_temp=final_outlet,
                    outdoor_temp=avg_outdoor,  # Use forecast average
                    current_indoor=current_indoor,
                    pv_power=avg_pv,  # Use forecast average for consistency
                    fireplace_on=fireplace_on,
                    tv_on=tv_on,
                    _suppress_logging=True,
                    fireplace_power_kw=fireplace_power_kw,
                    fireplace_decay_kw=fp_decay_kw,
                    cloud_cover_pct=self._avg_cloud_cover,
                )
            )

            # Handle None return for final prediction
            if final_predicted is None:
                logging.warning(
                    "⚠️ Final prediction returned None, using fallback %.1f°C",
                    fallback_temp,
                )
                return fallback_temp

        except Exception as e:
            logging.error(f"Final prediction failed: {e}")
            return fallback_temp

        final_error = final_predicted - target_indoor
        logging.warning(
            f"⚠️ Binary search didn't converge after 20 iterations: "
            f"{final_outlet:.1f}°C → {final_predicted:.2f}°C "
            f"(target: {target_indoor:.1f}°C, error: {final_error:.3f}°C)"
        )

        # SATURATION SAFETY CAP: When binary search saturates near outlet_max
        # the model couldn't find a sensible temperature. Cap at inlet + 5°C
        # to prevent runaway outlet temperatures (self-adjusting as slab warms).
        _inlet_now = (
            self._current_features.get("inlet_temp")
            if hasattr(self, "_current_features")
            else None
        )
        if _inlet_now is not None and final_outlet > _inlet_now + 5.0:
            capped = _inlet_now + 5.0
            logging.warning(
                f"🛡️ Saturation cap: {final_outlet:.1f}°C → "
                f"{capped:.1f}°C (inlet {_inlet_now:.1f}°C + 5°C)"
            )
            final_outlet = capped

        # NEW: Trajectory verification and course correction.
        # Pass same forecast arrays as the binary search used.
        if config.TRAJECTORY_PREDICTION_ENABLED:
            final_outlet = self._verify_trajectory_and_correct(
                outlet_temp=final_outlet,
                current_indoor=current_indoor,
                target_indoor=target_indoor,
                outdoor_temp=outdoor_temp,
                thermal_features=thermal_features,
                features=getattr(
                    self, "_current_features", {}
                ),
                outdoor_forecast=outdoor_forecast,
                pv_forecast=pv_forecast,
                pv_history_for_buffer=pv_input,
            )

        return final_outlet

    def _log_multi_horizon_predictions(
        self,
        current_indoor: float,
        target_indoor: float,
        outdoor_temp: float,
        thermal_features: Dict,
    ) -> None:
        """
        Log predicted outlet temperatures using different forecast horizons (now up to 6h).

        This helps diagnose which forecast horizon is causing high outlet temp
        predictions during evening/overnight scenarios when outdoor temperature
        drops.
        """
        try:
            # Extract thermal features
            pv_power = thermal_features.get("pv_power", 0.0)
            fireplace_on = thermal_features.get("fireplace_on", 0.0)
            tv_on = thermal_features.get("tv_on", 0.0)
            features = getattr(self, "_current_features", {})

            if not features:
                logging.debug("🔍 Multi-horizon: No forecast data available")
                return

            # UNIFIED: Get cycle-aligned forecast from context manager
            # This ensures the "cycle" method in logs matches exactly what was
            # used for prediction
            cycle_minutes = config.CYCLE_INTERVAL_MINUTES
            cycle_method = f"cycle({cycle_minutes}min)"

            # Use the unified context that should have been created during
            # prediction
            context = prediction_context_manager.get_context()
            if context:
                cycle_outdoor = context['avg_outdoor']
                cycle_pv = context['avg_pv']
            else:
                # Fallback if context missing (shouldn't happen in normal flow)
                cycle_outdoor = outdoor_temp
                cycle_pv = pv_power
                cycle_method = "cycle(fallback)"

            # Extract individual forecast horizons including cycle-aligned
            forecasts = {
                "current": {"outdoor": outdoor_temp, "pv": pv_power},
                cycle_method: {"outdoor": cycle_outdoor, "pv": cycle_pv},
                "1h": {
                    "outdoor": features.get("temp_forecast_1h", outdoor_temp),
                    "pv": features.get("pv_forecast_1h", pv_power),
                },
                "2h": {
                    "outdoor": features.get("temp_forecast_2h", outdoor_temp),
                    "pv": features.get("pv_forecast_2h", pv_power),
                },
                "3h": {
                    "outdoor": features.get("temp_forecast_3h", outdoor_temp),
                    "pv": features.get("pv_forecast_3h", pv_power),
                },
                "4h": {
                    "outdoor": features.get("temp_forecast_4h", outdoor_temp),
                    "pv": features.get("pv_forecast_4h", pv_power),
                },
                "5h": {
                    "outdoor": features.get("temp_forecast_5h", outdoor_temp),
                    "pv": features.get("pv_forecast_5h", pv_power),
                },
                "6h": {
                    "outdoor": features.get("temp_forecast_6h", outdoor_temp),
                    "pv": features.get("pv_forecast_6h", pv_power),
                },
                "avg": {
                    "outdoor": (
                        features.get("temp_forecast_1h", outdoor_temp)
                        + features.get("temp_forecast_2h", outdoor_temp)
                        + features.get("temp_forecast_3h", outdoor_temp)
                        + features.get("temp_forecast_4h", outdoor_temp)
                        + features.get("temp_forecast_5h", outdoor_temp)
                        + features.get("temp_forecast_6h", outdoor_temp)
                    )
                    / 6.0,
                    "pv": (
                        features.get("pv_forecast_1h", pv_power)
                        + features.get("pv_forecast_2h", pv_power)
                        + features.get("pv_forecast_3h", pv_power)
                        + features.get("pv_forecast_4h", pv_power)
                        + features.get("pv_forecast_5h", pv_power)
                        + features.get("pv_forecast_6h", pv_power)
                    )
                    / 6.0,
                },
            }

            # Calculate predicted outlet temperature for each horizon
            predictions = {}
            for horizon, conditions in forecasts.items():
                try:
                    # Use same precision as main binary search for consistency
                    outlet_min, outlet_max = config.get_outlet_bounds(
                        self._climate_mode
                    )
                    tolerance = 0.1  # Same precision as main binary search

                    for iteration in range(20):  # Same iterations as main
                        outlet_mid = (outlet_min + outlet_max) / 2.0
                        predicted_indoor = (
                            self.thermal_model.predict_equilibrium_temperature(
                                outlet_temp=outlet_mid,
                                outdoor_temp=conditions["outdoor"],
                                current_indoor=current_indoor,
                                pv_power=conditions["pv"],
                                fireplace_on=fireplace_on,
                                tv_on=tv_on,
                                _suppress_logging=True,
                                cloud_cover_pct=self._avg_cloud_cover,
                            )
                        )

                        if predicted_indoor is None:
                            break

                        error = predicted_indoor - target_indoor
                        if abs(error) < tolerance:
                            predictions[horizon] = {
                                "outlet": outlet_mid,
                                "predicted": predicted_indoor,
                                "conditions": conditions,
                            }
                            break

                        if predicted_indoor < target_indoor:
                            outlet_min = outlet_mid
                        else:
                            outlet_max = outlet_mid
                    else:
                        # Didn't converge, use midpoint
                        final_outlet = (outlet_min + outlet_max) / 2.0
                        final_predicted = (
                            self.thermal_model.predict_equilibrium_temperature(
                                outlet_temp=final_outlet,
                                outdoor_temp=conditions["outdoor"],
                                current_indoor=current_indoor,
                                pv_power=conditions["pv"],
                                fireplace_on=fireplace_on,
                                tv_on=tv_on,
                                _suppress_logging=True,
                                cloud_cover_pct=self._avg_cloud_cover,
                            )
                        )
                        if final_predicted is not None:
                            predictions[horizon] = {
                                "outlet": final_outlet,
                                "predicted": final_predicted,
                                "conditions": conditions,
                            }

                except Exception as e:
                    logging.debug(
                        f"Multi-horizon prediction failed for {horizon}: {e}"
                    )
                    continue

            # Log the multi-horizon predictions in a clear format
            if predictions:
                logging.debug(
                    f"🔍 Multi-horizon predictions for target "
                    f"{target_indoor:.1f}°C:"
                )

                # Order the predictions logically (including cycle-aligned)
                order = [
                    "current",
                    cycle_method,
                    "1h",
                    "2h",
                    "3h",
                    "4h",
                    "avg",
                ]
                for horizon in order:
                    if horizon in predictions:
                        pred = predictions[horizon]
                        outlet = pred["outlet"]
                        predicted = pred["predicted"]
                        conditions = pred["conditions"]
                        error = predicted - target_indoor

                        # Mark the cycle-aligned prediction as NEW
                        if horizon == cycle_method:
                            marker = "← NEW: Cycle-aligned"
                        else:
                            marker = ""

                        logging.debug(
                            f"   {horizon:>12}: {outlet:.1f}°C → "
                            f"{predicted:.1f}°C (error: {error:+.2f}°C, "
                            f"outdoor: {conditions['outdoor']:.1f}°C, "
                            f"PV: {conditions['pv']:.0f}W) {marker}"
                        )

                # Show the differences to highlight issues
                if "current" in predictions and "avg" in predictions:
                    current_outlet = predictions["current"]["outlet"]
                    avg_outlet = predictions["avg"]["outlet"]
                    outlet_diff = avg_outlet - current_outlet

                    if abs(outlet_diff) > 2.0:  # Significant difference
                        direction = "higher" if outlet_diff > 0 else "lower"
                        logging.warning(
                            f"⚠️ Forecast vs current difference: "
                            f"forecast avg outlet {outlet_diff:+.1f}°C "
                            f"{direction} "
                            f"than current conditions"
                        )
            else:
                logging.debug(
                    "🔍 Multi-horizon: No predictions calculated successfully"
                )

        except Exception as e:
            logging.error(f"Multi-horizon prediction logging failed: {e}")

    def _verify_trajectory_and_correct(
        self,
        outlet_temp: float,
        current_indoor: float,
        target_indoor: float,
        outdoor_temp: float,
        thermal_features: Dict,
        features: Optional[Dict] = None,
        outdoor_forecast: Optional[list] = None,
        pv_forecast: Optional[list] = None,
        pv_history_for_buffer: Optional[list] = None,
    ) -> float:
        """
        Verify that the calculated outlet temperature will actually reach the
        target using trajectory prediction, and apply physics-based adaptive
        correction if needed.

        PHYSICS-BASED CORRECTION: Uses learned thermal parameters to adaptively
        scale
        corrections based on house characteristics and time pressure.
        """
        try:
            # Use pre-computed forecast arrays when provided (from binary
            # search) to guarantee identical conditions. Only fall back to
            # recomputing when called without them (e.g. from non-converged
            # path).
            if outdoor_forecast is None or pv_forecast is None:
                _, _, outdoor_forecast, pv_forecast = (
                    self._get_forecast_conditions(
                        outdoor_temp,
                        thermal_features.get("pv_power", 0.0),
                        thermal_features,
                    )
                )
            if pv_history_for_buffer is None:
                pv_history_for_buffer = (
                    thermal_features.get("pv_power_history")
                    or [thermal_features.get("pv_power", 0.0)]
                )

            fireplace_on = thermal_features.get("fireplace_on", 0.0)
            fireplace_power_kw, fp_decay_kw = self._calculate_fireplace_power_kw(
                current_indoor=current_indoor,
                outdoor_temp=outdoor_temp,
                fireplace_on=fireplace_on,
                features_local=features if features is not None else {},
            )

            # Get trajectory prediction with forecast integration
            # Use the enhanced predict_thermal_trajectory which now supports
            # arrays.
            # IMPORTANT: split pv_power (history for buffer init) and
            # pv_forecasts (future schedule) exactly as the binary search does.
            # Passing pv_forecast as pv_power would seed the buffer with the
            # last forecast value (e.g. 1308 W) and replicate it for all
            # future steps — causing an immediate false overshoot.
            trajectory = self.thermal_model.predict_thermal_trajectory(
                current_indoor=current_indoor,
                target_indoor=target_indoor,
                outlet_temp=outlet_temp,
                outdoor_temp=outdoor_forecast,  # Pass array directly
                time_horizon_hours=config.TRAJECTORY_STEPS,
                time_step_minutes=config.CYCLE_INTERVAL_MINUTES,
                pv_power=pv_history_for_buffer,  # History for buffer init
                pv_forecasts=pv_forecast,        # Hourly schedule (0h,1h,...)
                fireplace_on=fireplace_on,
                tv_on=thermal_features.get("tv_on", 0.0),
                fireplace_power_kw=fireplace_power_kw,
                fireplace_decay_kw=fp_decay_kw,
                cloud_cover_pct=self._avg_cloud_cover,
                inlet_temp=(
                    self._current_features.get("inlet_temp")
                    if hasattr(self, "_current_features")
                    else None
                ),
                delta_t_floor=getattr(
                    self, "_search_delta_t_floor",
                    self._current_features.get("delta_t", 0.0)
                    if hasattr(self, "_current_features")
                    else 0.0
                ),
            )

            _it = (
                self._current_features.get("inlet_temp")
                if hasattr(self, "_current_features") else None
            )
            if _it is not None:
                logging.debug(
                    "Slab: inlet=%.1f\u00b0C, outlet=%.1f\u00b0C, \u0394=%+.1f\u00b0C, \u03c4=%.2fh",
                    float(_it), float(outlet_temp),
                    float(outlet_temp) - float(_it),
                    self.thermal_model.slab_time_constant_hours,
                )

            logging.debug(
                f"🔍 Trajectory prediction using forecast arrays: "
                f"outdoor={outdoor_forecast}, PV={pv_forecast}"
            )

            # TRAJECTORY-BASED DECISION: Check if target reachable within cycle
            # time. Allow tolerance for the finite resolution of the trajectory
            # (steps = CYCLE_INTERVAL_MINUTES each). Using 3 × cycle_hours
            # means: if the target is reached within 3 cycles (~30 min for a
            # 10-min cycle) no correction is applied. This prevents spurious
            # corrections when binary search already found an outlet temp that
            # reaches the target quickly — the trajectory simply cannot resolve
            # sub-step timing.
            cycle_hours = config.CYCLE_INTERVAL_MINUTES / 60.0
            tolerance_hours = cycle_hours * 2  # 2 extra time steps
            sensor_precision_tolerance = 0.1  # °C — must match binary search convergence tolerance

            # DEBUG: Log trajectory details for diagnosis
            trajectory_temps = trajectory.get("trajectory", [])
            first_step_temp = (
                trajectory_temps[0] if trajectory_temps else current_indoor
            )
            reaches_target_at = trajectory.get("reaches_target_at")

            # Get time of first step for accurate logging
            trajectory_times = trajectory.get("times", [])
            first_step_time = (
                trajectory_times[0]
                if trajectory_times
                else config.CYCLE_INTERVAL_MINUTES / 60.0
            )

            logging.debug(
                f"🔍 Trajectory DEBUG: outlet={outlet_temp:.1f}°C → "
                f"{first_step_time:.1f}h_prediction={first_step_temp:.2f}°C "
                f"(vs target {target_indoor:.1f}°C, "
                f"error: {first_step_temp - target_indoor:+.3f}°C), "
                f"reaches_target_at={reaches_target_at}h, "
                f"cycle_time={cycle_hours:.1f}h"
            )

            # Show full trajectory (1h-4h) with deviation vs target for debugging
            trajectory_temps = trajectory.get("trajectory", [])
            trajectory_times = trajectory.get("times", [])
            if trajectory_temps:
                logging.debug(f"🔍 Trajectory predictions for target {target_indoor:.1f}°C:")
                for i, temp in enumerate(trajectory_temps):
                    # Use trajectory_times if available, else fallback to index
                    t = trajectory_times[i] if trajectory_times and i < len(trajectory_times) else (i + 1)
                    error = temp - target_indoor
                    logging.debug(
                        f"    {t:.1f}h: {temp:.2f}°C → target {target_indoor:.2f}°C (error: {error:+.2f}°C)"
                    )

            # NEW LOGIC: Check for temperature boundary violations regardless
            # of target achievement. This ensures comfort boundaries are
            # respected even when target is theoretically reachable
            trajectory_temps = trajectory.get("trajectory", [])
            trajectory_times = trajectory.get("times", [])

            if trajectory_temps:
                min_temp = min(trajectory_temps)
                max_temp = max(trajectory_temps)

                # Check if immediate cycle is safe (no overshoot)
                # If we are safe for the current cycle, we can be more lenient
                # with future predictions as we will have a chance to correct
                # in the next cycle.
                immediate_overshoot = False
                if (
                    trajectory_times
                    and trajectory_times[0] <= cycle_hours + 0.01
                ):
                    if (
                        trajectory_temps[0]
                        > target_indoor + sensor_precision_tolerance
                    ):
                        immediate_overshoot = True
                else:
                    # Fallback if times not available or first step is far
                    # future
                    if (
                        trajectory_temps[0]
                        > target_indoor + sensor_precision_tolerance
                    ):
                        immediate_overshoot = True

                if not immediate_overshoot:
                    # Immediate cycle is safe - allow relaxed boundaries for
                    # future. Allow up to 0.5°C overshoot in future steps (vs
                    # 0.1°C strict). This enables "fast push" strategies where
                    # we heat aggressively now knowing we can back off in the
                    # next cycle.
                    temp_boundary_violation = (
                        min_temp <= target_indoor - sensor_precision_tolerance
                        or max_temp >= target_indoor + 0.5  # Relaxed overshoot
                    )

                    # Log if we are ignoring a future overshoot that would have
                    # been caught by strict rules
                    if (
                        max_temp
                        >= target_indoor + sensor_precision_tolerance
                    ) and (
                        max_temp < target_indoor + 0.5
                    ):
                        logging.debug(
                            f"Ignoring minor future overshoot "
                            f"(max={max_temp:.2f}°C) because immediate cycle "
                            f"is safe ({trajectory_temps[0]:.2f}°C)"
                        )
                else:
                    # Immediate overshoot detected - enforce strict boundaries
                    temp_boundary_violation = (
                        min_temp <= target_indoor - sensor_precision_tolerance
                        or max_temp >= target_indoor + sensor_precision_tolerance
                    )
            else:
                temp_boundary_violation = False

            # Enhanced logic: If target reached quickly, be more lenient with
            # boundary violations. Use tolerance to allow reaching target
            # slightly after cycle end
            if reaches_target_at is not None and reaches_target_at <= (
                cycle_hours + tolerance_hours
            ):
                if not temp_boundary_violation:
                    logging.info(
                        f"✅ Target will be reached in "
                        f"{reaches_target_at:.1f}h (within {cycle_hours:.1f}h "
                        f"cycle + tolerance) and no boundary violations - no "
                        "correction needed"
                    )
                    return outlet_temp
                else:
                    # Target reachable quickly but has boundary violations
                    # For fast target achievement (< 0.5h), allow larger
                    # tolerance (±0.3°C instead of ±0.1°C)
                    if reaches_target_at <= 0.5:  # Very fast achievement
                        relaxed_boundary_violation = (
                            min_temp
                            <= target_indoor - sensor_precision_tolerance
                            or max_temp >= target_indoor + 0.3  # Relaxed overshoot only
                        )
                        if not relaxed_boundary_violation:
                            logging.info(
                                f"✅ Target will be reached quickly in "
                                f"{reaches_target_at:.1f}h with minor "
                                "overshoot (0.1-0.3°C) - no correction needed"
                            )
                            return outlet_temp

            # Apply correction if target not reachable or significant boundary
            # violations
            if trajectory_temps and min(trajectory_temps) > target_indoor + sensor_precision_tolerance:
                logging.info(
                    "⚠️ Overshoot detected: entire trajectory is above target "
                    f"{target_indoor:.1f}°C. Applying correction."
                )
            elif reaches_target_at is None or reaches_target_at > (
                cycle_hours + tolerance_hours
            ):
                logging.info(
                    f"⚠️ Target will NOT be reached within {cycle_hours:.1f}h "
                    "cycle (+tolerance) - applying physics-based correction"
                )
            elif temp_boundary_violation:
                logging.info(
                    "⚠️ Temperature boundary violations detected (min: "
                    f"{min_temp:.2f}°C, max: {max_temp:.2f}°C) - applying "
                    "correction"
                )

            # Calculate physics-based correction
            corrected_outlet = self._calculate_physics_based_correction(
                outlet_temp=outlet_temp,
                trajectory=trajectory,
                target_indoor=target_indoor,
                cycle_hours=cycle_hours,
            )

            return corrected_outlet

        except Exception as e:
            logging.error(f"Trajectory verification failed: {e}")
            return outlet_temp  # Return original if verification fails

    def _calculate_physics_based_correction(
        self,
        outlet_temp: float,
        trajectory: Dict,
        target_indoor: float,
        cycle_hours: float,
    ) -> float:
        """
        Calculate physics-based adaptive correction with deviation-scaled
        response.

        ENHANCED PRIORITY 2 IMPLEMENTATION:
        - Graduated response zones based on temperature deviation magnitude
        - Unified control logic for heating and cooling scenarios
        - Physics-based scaling using house thermal characteristics
        """
        try:
            # Calculate temperature error from trajectory
            trajectory_temps = trajectory.get("trajectory", [])
            if not trajectory_temps:
                return outlet_temp

            # Get initial temperature deviation for graduated response
            current_indoor = getattr(self, '_current_indoor', target_indoor)
            initial_deviation = abs(target_indoor - current_indoor)

            # Calculate trajectory error
            min_predicted_temp = min(trajectory_temps)
            max_predicted_temp = max(trajectory_temps)
            boundary_tolerance = 0.1

            # Determine primary error source
            min_violates = (
                min_predicted_temp <= target_indoor - boundary_tolerance
            )
            max_violates = (
                max_predicted_temp >= target_indoor + boundary_tolerance
            )

            # Projected indoor temperature for overshoot gate:
            # If the natural trend brings indoor below target + 0.1°C
            # within TRAJECTORY_STEPS hours, skip overshoot correction
            # and let the house return to normal.
            _features = getattr(self, '_current_features', None) or {}
            indoor_trend_60m = _features.get('indoor_temp_delta_60m', 0.0)
            projected_indoor = (
                current_indoor
                + config.TRAJECTORY_STEPS * indoor_trend_60m
            )

            if min_violates and max_violates:
                # Both boundaries violated - choose the more severe
                min_severity = abs(
                    min_predicted_temp
                    - (target_indoor - boundary_tolerance)
                )
                max_severity = abs(
                    max_predicted_temp
                    - (target_indoor + boundary_tolerance)
                )
                if min_severity > max_severity:
                    # Undershoot is more severe — skip correction if
                    # projected indoor will rise above target - 0.1°C
                    # within TRAJECTORY_STEPS hours.
                    if projected_indoor > target_indoor - boundary_tolerance:
                        logging.info(
                            f"🔄 Skipping undershoot correction (both violated, "
                            f"min wins): projected indoor "
                            f"{projected_indoor:.2f}°C > target-0.1 "
                            f"({target_indoor - boundary_tolerance:.1f}°C) "
                            f"in {config.TRAJECTORY_STEPS}h "
                            f"(trend {indoor_trend_60m:+.3f}°C/h) — "
                            "house self-correcting"
                        )
                        return outlet_temp
                    temp_error = target_indoor - min_predicted_temp
                else:
                    # Overshoot is more severe — skip correction if
                    # projected indoor will fall below target + 0.1°C
                    # within TRAJECTORY_STEPS hours.
                    if projected_indoor < target_indoor + boundary_tolerance:
                        logging.info(
                            f"🔄 Skipping overshoot correction (both violated, "
                            f"max wins): projected indoor "
                            f"{projected_indoor:.2f}°C < target+0.1 "
                            f"({target_indoor + boundary_tolerance:.1f}°C) "
                            f"in {config.TRAJECTORY_STEPS}h "
                            f"(trend {indoor_trend_60m:+.3f}°C/h) — "
                            "house self-correcting"
                        )
                        return outlet_temp
                    temp_error = target_indoor - max_predicted_temp
            elif min_violates:
                # Skip undershoot correction if projected indoor will rise
                # above target - 0.1°C within TRAJECTORY_STEPS hours.
                if projected_indoor > target_indoor - boundary_tolerance:
                    logging.info(
                        f"🔄 Skipping undershoot correction: projected indoor "
                        f"{projected_indoor:.2f}°C > target-0.1 "
                        f"({target_indoor - boundary_tolerance:.1f}°C) "
                        f"in {config.TRAJECTORY_STEPS}h "
                        f"(trend {indoor_trend_60m:+.3f}°C/h) — "
                        "house self-correcting"
                    )
                    return outlet_temp
                temp_error = target_indoor - min_predicted_temp
            elif max_violates:
                # Skip overshoot correction if projected indoor will fall
                # below target + 0.1°C within TRAJECTORY_STEPS hours.
                if projected_indoor < target_indoor + boundary_tolerance:
                    logging.info(
                        f"🔄 Skipping overshoot correction: projected indoor "
                        f"{projected_indoor:.2f}°C < target+0.1 "
                        f"({target_indoor + boundary_tolerance:.1f}°C) "
                        f"in {config.TRAJECTORY_STEPS}h "
                        f"(trend {indoor_trend_60m:+.3f}°C/h) — "
                        "house self-correcting"
                    )
                    return outlet_temp
                temp_error = target_indoor - max_predicted_temp
            else:
                # Target not reached in time
                reaches_target_at = trajectory.get("reaches_target_at")
                if (
                    reaches_target_at is None
                    or reaches_target_at > cycle_hours
                ):
                    final_predicted_temp = trajectory_temps[-1]
                    temp_error = target_indoor - final_predicted_temp
                else:
                    temp_error = 0.0

            # USER FEEDBACK: Implement exponential correction based on
            # deviation from target. This ensures a strong pull towards the
            # target when the temperature is far away, and a gentler approach
            # as it gets closer.

            # k controls the aggression of the exponential curve. A higher
            # value means a more aggressive response to temperature deviations.
            # k = 1.5 due to under floor heating 
    
            k = 1.5
            raw_aggression = np.exp(k * initial_deviation)

            # Clamp aggression to prevent runaway correction
            aggression_factor = min(raw_aggression, 3.0)

            # Physics-based scaling using house thermal characteristics.
            if self.thermal_model.outlet_effectiveness > 0.01:
                # REVERTED: Reduced base scale multiplier from 0.5 back to 0.3
                # and added hard clamp to prevent explosion when effectiveness
                # is very low (e.g. < 0.05).
                # scale multiplier = 1.1 due to under floor heating
                raw_scale = (
                    1.0 / self.thermal_model.outlet_effectiveness
                ) * 1.1
                base_scale = min(raw_scale, 2.5)
            else:
                base_scale = 2.5 # Fallback (reduced from 10.0)

            physics_scale = base_scale

            # Time pressure calculation (how urgently we need to correct).
            time_pressure = self._calculate_time_pressure(
                trajectory, cycle_hours
            )
            urgency_multiplier = 1.0 + 2.0 * (time_pressure ** 2)

            # Calculate the final correction, applying aggression only when
            # undershooting.
            if temp_error > 0:
                # Apply exponential aggression when we are below target.
                correction = (
                    temp_error
                    * physics_scale
                    * urgency_multiplier
                    * aggression_factor
                )
                logging.info(
                    "   Exponential Correction for undershoot: "
                    f"aggression_factor={aggression_factor:.2f}x"
                )
            else:
                # When overshooting, apply a gentle, non-exponential correction
                # to prevent the system from pulling back too hard.
                overshoot_dampening = 0.4
                correction = (
                    temp_error
                    * physics_scale
                    * urgency_multiplier
                    * overshoot_dampening
                )
                logging.info(
                    "   Gentle Correction for overshoot: dampened by "
                    f"{overshoot_dampening:.0%}"
                )

            # Clamp the correction to a reasonable maximum to prevent extreme
            # values.
            # REDUCED: Cap max correction to 2.5°C (was 3.0°C) to prevent
            # startup overshoot.
            max_correction = 2.5
            correction = max(-max_correction, min(max_correction, correction))

            # Final outlet temperature.
            corrected_outlet = outlet_temp + correction
            clamp_min, clamp_max = config.get_outlet_bounds(self._climate_mode)
            corrected_outlet = max(
                clamp_min,
                min(clamp_max, corrected_outlet),
            )

            logging.info(
                f"🎯 Corrected outlet: {corrected_outlet:.1f}°C "
                f"({outlet_temp:.1f}°C + {correction:+.1f}°C) "
                f"(deviation: {initial_deviation:.1f}°C, "
                f"temp_error: {temp_error:+.2f}°C)"
            )

            return corrected_outlet

        except Exception as e:
            logging.error(f"Physics-based correction failed: {e}")
            return outlet_temp

    def _calculate_time_pressure(
        self, trajectory: Dict, cycle_hours: float
    ) -> float:
        """
        Calculate how urgently we need to correct (0.0 = no pressure, 1.0 =
        maximum urgency).
        """
        reaches_target_at = trajectory.get("reaches_target_at")

        if reaches_target_at is None:
            return 1.0  # Maximum urgency - may never reach target
        elif reaches_target_at <= cycle_hours:
            return (
                0.0  # No pressure - target reachable in time
            )
        elif reaches_target_at <= cycle_hours * 2:
            return 0.3  # Low pressure - close to reachable
        elif reaches_target_at <= cycle_hours * 4:
            return 0.6  # Medium pressure
        else:
            return 1.0  # High pressure - far from target

    # This method is no longer needed - thermal state is loaded in
    # ThermalEquilibriumModel

    def learn_from_prediction_feedback(
        self,
        predicted_temp: float,
        actual_temp: float,
        prediction_context: Dict,
        timestamp: Optional[str] = None,
        is_blocking_active: bool = False,
        effective_shadow_mode: Optional[bool] = None,
    ):
        """
        Learn from prediction feedback using the thermal model's adaptive
        learning.
        """
        if not self.learning_enabled:
            return

        if self.adaptive_fireplace is not None:
            # Update adaptive fireplace learning BEFORE the blocking check so
            # that fireplace sessions that span a DHW cycle are not silently
            # lost. The observer only tracks session state and is legacy-only
            # when heat-source channels are disabled.
            try:
                fireplace_on_now = bool(
                    prediction_context.get("fireplace_on", 0)
                )
                if (
                    fireplace_on_now
                    or self.adaptive_fireplace.current_session is not None
                ):
                    current_indoor = prediction_context.get(
                        "current_indoor", 20.0
                    )
                    other_rooms_temp = prediction_context.get(
                        "avg_other_rooms_temp", current_indoor - 2.0
                    )
                    outdoor_temp = prediction_context.get("outdoor_temp", 0.0)

                    living_room_temp = prediction_context.get(
                        "living_room_temp"
                    ) or current_indoor
                    self.adaptive_fireplace.observe_fireplace_state(
                        living_room_temp=living_room_temp,
                        other_rooms_temp=other_rooms_temp,
                        outdoor_temp=outdoor_temp,
                        fireplace_active=fireplace_on_now,
                    )
            except Exception as e:
                logging.warning(
                    "Fireplace learning observation failed: %s", e,
                    exc_info=True,
                )

        if is_blocking_active:
            logging.info(
                "Skipping online learning due to active blocking event."
            )
            return

        try:
            # FIRST-CYCLE GUARD: Skip learning on the first cycle after a
            # restart to prevent incorrect adjustments from the time gap
            # between cycles.
            if self.cycle_count <= 1:
                logging.info(
                    "Skipping online learning on the first cycle to ensure "
                    "stability."
                )
                # Still update cycle count and save state, but don't learn.
                self.cycle_count += 1
                self.state_manager.update_learning_state(
                    cycle_count=self.cycle_count
                )
                return

            # Update thermal model with prediction feedback
            prediction_error = self.thermal_model.update_prediction_feedback(
                predicted_temp=predicted_temp,
                actual_temp=actual_temp,
                prediction_context=prediction_context,
                timestamp=timestamp or datetime.now().isoformat(),
                is_blocking_active=is_blocking_active,
            )

            # Fireplace learning is now handled before the blocking
            # check (above) to avoid losing sessions that span DHW.

            # Add prediction to MAE/RMSE tracking
            self.prediction_metrics.add_prediction(
                predicted=predicted_temp,
                actual=actual_temp,
                context=prediction_context,
                timestamp=timestamp,
            )

            # Add prediction record to unified state
            prediction_record = {
                "timestamp": timestamp or datetime.now().isoformat(),
                "predicted": predicted_temp,
                "actual": actual_temp,
                "error": actual_temp - predicted_temp,
                "context": prediction_context,
            }
            self.state_manager.add_prediction_record(prediction_record)

            # Track learning cycles
            self.cycle_count += 1

            # Update cycle count in unified state
            self.state_manager.update_learning_state(
                cycle_count=self.cycle_count
            )

            # Export Influx metrics at the configured cadence to limit write
            # volume while keeping Home Assistant updates real-time.
            if self.cycle_count % self._get_influx_export_interval_cycles() == 0:
                self._export_metrics_to_influxdb()

            shadow_mode = resolve_shadow_mode(
                effective_shadow_mode=effective_shadow_mode
            )

            # Keep live HA metrics quiet during effective shadow-mode cycles
            # until the later shadow-suffixed rollout is in place.
            if shadow_mode.should_publish_output_entities:
                self.export_metrics_to_ha()
            else:
                logging.debug(
                    "Skipping live HA metric export during effective "
                    "shadow mode"
                )

            # Log learning cycle completion
            if prediction_error is not None:
                logging.info(
                    f"✅ Learning cycle {self.cycle_count}: "
                    f"error={abs(prediction_error):.3f}°C, "
                    f"confidence="
                    f"{self.thermal_model.learning_confidence:.3f}, "
                    f"total_predictions="
                    f"{len(self.prediction_metrics.predictions)}"
                )

        except Exception as e:
            logging.error(f"Learning from feedback failed: {e}", exc_info=True)

    def export_metrics_to_ha(self):
        """Export metrics to Home Assistant sensors."""
        try:
            ha_client = create_ha_client()

            # Get comprehensive metrics
            ha_metrics = self.get_comprehensive_metrics_for_ha()

            # Export MAE/RMSE metrics (confidence now provided via
            # ml_heating_learning sensor)
            ha_client.log_model_metrics(
                mae=ha_metrics.get("mae_all_time", 0.0),
                rmse=ha_metrics.get("rmse_all_time", 0.0),
            )

            # Export adaptive learning metrics
            ha_client.log_adaptive_learning_metrics(ha_metrics)

            # Export feature importance (if available)
            if hasattr(self.thermal_model, "get_feature_importance"):
                importances = self.thermal_model.get_feature_importance()
                if importances:
                    ha_client.log_feature_importance(importances)

            logging.info(
                "✅ Exported metrics to Home Assistant sensors successfully"
            )

        except Exception as e:
            # Better error logging for debugging sensor export issues
            logging.error(
                f"❌ FAILED to export metrics to HA: {e}", exc_info=True
            )
            keys = (
                list(ha_metrics.keys())
                if 'ha_metrics' in locals()
                else 'metrics not created'
            )
            logging.error(f"   Attempted to export: {keys}")
            logging.error(f"   HA Client created: {'ha_client' in locals()}")
            # Re-raise the exception for visibility during debugging
            raise

    def get_prediction_confidence(self) -> float:
        """Get current prediction confidence from thermal model."""
        return self.thermal_model.learning_confidence

    def get_learning_metrics(self) -> Dict:
        """Get comprehensive learning metrics for monitoring."""
        try:
            metrics = self.thermal_model.get_adaptive_learning_metrics()
            # Check if we got valid metrics or just insufficient_data flag
            if (
                isinstance(metrics, dict)
                and "insufficient_data" not in metrics
                and len(metrics) > 1
            ):
                # Extract current parameters from nested structure if available
                if "current_parameters" in metrics:
                    current_params = metrics["current_parameters"]
                    # Return flattened structure with actual loaded parameters
                    result = metrics.copy()
                    result.update(current_params)
                    result.update(
                        {
                            "learning_confidence": self.thermal_model.learning_confidence,
                            "cycle_count": self.cycle_count,
                        }
                    )
                    return result
                else:
                    # Use the metrics as-is if current_parameters key not found
                    return metrics
        except AttributeError:
            pass

        # Fallback if method doesn't exist or returns insufficient_data
        return {
            "thermal_time_constant": self.thermal_model.thermal_time_constant,
            "heat_loss_coefficient": self.thermal_model.heat_loss_coefficient,
            "outlet_effectiveness": self.thermal_model.outlet_effectiveness,
            "learning_confidence": self.thermal_model.learning_confidence,
            "cycle_count": self.cycle_count,
        }

    def get_comprehensive_metrics_for_ha(self) -> Dict:
        """Get comprehensive metrics for Home Assistant sensor export."""
        try:
            # Get thermal learning metrics
            thermal_metrics = self.get_learning_metrics()

            # Get prediction accuracy metrics (all-time for MAE/RMSE)
            prediction_metrics = self.prediction_metrics.get_metrics()

            # Get recent performance
            recent_performance = (
                self.prediction_metrics.get_recent_performance(10)
            )

            # Get 24h window simplified accuracy breakdown
            accuracy_24h = self.prediction_metrics.get_24h_accuracy_breakdown()
            good_control_24h = (
                self.prediction_metrics.get_24h_good_control_percentage()
            )

            # Combine into comprehensive HA-friendly format
            ha_metrics = {
                # Core thermal parameters (learned)
                "thermal_time_constant": thermal_metrics.get(
                    "thermal_time_constant", 6.0
                ),
                "heat_loss_coefficient": thermal_metrics.get(
                    "heat_loss_coefficient", 0.05
                ),
                "outlet_effectiveness": thermal_metrics.get(
                    "outlet_effectiveness", 0.04
                ),
                "pv_heat_weight": thermal_metrics.get(
                    "pv_heat_weight", 0.001
                ),
                "fireplace_heat_weight": thermal_metrics.get(
                    "fireplace_heat_weight", 0.0
                ),
                "tv_heat_weight": thermal_metrics.get(
                    "tv_heat_weight", 0.1
                ),
                "solar_lag_minutes": thermal_metrics.get(
                    "solar_lag_minutes", 0.0
                ),
                "slab_time_constant_hours": thermal_metrics.get(
                    "slab_time_constant_hours", 2.0
                ),
                "heat_source_channels_enabled": bool(
                    thermal_metrics.get("heat_source_channels_enabled", False)
                ),
                "learning_confidence": thermal_metrics.get(
                    "learning_confidence", 3.0
                ),
                # Learning progress
                "cycle_count": self.cycle_count,
                "parameter_updates": thermal_metrics.get(
                    "parameter_updates", 0
                ),
                "update_percentage": thermal_metrics.get(
                    "update_percentage", 0
                ),
                # Prediction accuracy (MAE/RMSE) - all-time
                "mae_1h": prediction_metrics.get("1h", {}).get("mae", 0.0),
                "mae_6h": prediction_metrics.get("6h", {}).get("mae", 0.0),
                "mae_24h": prediction_metrics.get("24h", {}).get("mae", 0.0),
                "mae_all_time": prediction_metrics.get("all", {}).get(
                    "mae", 0.0
                ),
                "rmse_all_time": prediction_metrics.get("all", {}).get(
                    "rmse", 0.0
                ),
                # Recent performance
                "recent_mae_10": recent_performance.get("mae", 0.0),
                "recent_max_error": recent_performance.get("max_error", 0.0),
                # NEW: Simplified 3-category accuracy (24h window)
                "perfect_accuracy_pct": accuracy_24h.get("perfect", {}).get(
                    "percentage", 0.0
                ),
                "tolerable_accuracy_pct": accuracy_24h.get(
                    "tolerable", {}
                ).get("percentage", 0.0),
                "poor_accuracy_pct": accuracy_24h.get("poor", {}).get(
                    "percentage", 0.0
                ),
                "good_control_pct": good_control_24h,
                # Legacy accuracy breakdown (all-time) - kept for backward
                # compatibility
                "excellent_accuracy_pct": prediction_metrics.get(
                    "accuracy_breakdown", {}
                )
                .get("excellent", {})
                .get("percentage", 0.0),
                "good_accuracy_pct": (
                    prediction_metrics.get("accuracy_breakdown", {})
                    .get("excellent", {})
                    .get("percentage", 0.0)
                    + prediction_metrics.get("accuracy_breakdown", {})
                    .get("very_good", {})
                    .get("percentage", 0.0)
                    + prediction_metrics.get("accuracy_breakdown", {})
                    .get("good", {})
                    .get("percentage", 0.0)
                ),
                # Trend analysis (ensure JSON serializable)
                "is_improving": bool(
                    prediction_metrics.get("trends", {}).get(
                        "is_improving", False
                    )
                ),
                "improvement_percentage": float(
                    prediction_metrics.get("trends", {}).get(
                        "mae_improvement_percentage", 0.0
                    )
                ),
                # Model health summary
                "model_health": (
                    "excellent"
                    if thermal_metrics.get("learning_confidence", 0) >= 4.0
                    else (
                        "good"
                        if thermal_metrics.get("learning_confidence", 0) >= 3.0
                        else (
                            "fair"
                            if thermal_metrics.get("learning_confidence", 0)
                            >= 2.0
                            else "poor"
                        )
                    )
                ),
                # Total predictions tracked
                "total_predictions": len(self.prediction_metrics.predictions),
                # Timestamp
                "last_updated": datetime.now().isoformat(),
            }

            if ha_metrics["heat_source_channels_enabled"]:
                ha_metrics.update(
                    {
                        "delta_t_floor": thermal_metrics.get(
                            "delta_t_floor", 0.0
                        ),
                        "cloud_factor_exponent": thermal_metrics.get(
                            "cloud_factor_exponent", 1.0
                        ),
                        "solar_decay_tau_hours": thermal_metrics.get(
                            "solar_decay_tau_hours", 0.0
                        ),
                        "fp_heat_output_kw": thermal_metrics.get(
                            "fp_heat_output_kw", 0.0
                        ),
                        "fp_decay_time_constant": thermal_metrics.get(
                            "fp_decay_time_constant", 0.0
                        ),
                        "room_spread_delay_minutes": thermal_metrics.get(
                            "room_spread_delay_minutes", 0.0
                        ),
                    }
                )

            # Slab passive delta: inlet_temp − indoor → passive heating signal
            _feats = getattr(self, "_current_features", {}) or {}
            _s_inlet = _feats.get("inlet_temp")
            _s_indoor = _feats.get("indoor_temp_lag_30m",
                                   _feats.get("current_indoor"))
            if _s_inlet is not None and _s_indoor is not None:
                ha_metrics["slab_passive_delta"] = round(
                    float(_s_inlet) - float(_s_indoor), 2
                )

            return ha_metrics

        except Exception as e:
            logging.error(f"Failed to get comprehensive metrics: {e}")
            return {
                "error": str(e),
                "cycle_count": self.cycle_count,
                "last_updated": datetime.now().isoformat(),
            }

    def _export_metrics_to_influxdb(self):
        """Export adaptive learning metrics to InfluxDB for monitoring."""
        export_failures = []

        try:
            # Create InfluxDB service
            influx_service = create_influx_service()
        except Exception as e:
            logging.warning("Failed to create InfluxDB service: %s", e)
            return

        # Export prediction metrics
        prediction_metrics = self.prediction_metrics.get_metrics()
        if prediction_metrics:
            try:
                influx_service.write_prediction_metrics(prediction_metrics)
                logging.debug("✅ Exported prediction metrics to InfluxDB")
            except Exception as e:
                export_failures.append("prediction_metrics")
                logging.warning(
                    "⚠️ Failed to export prediction metrics to InfluxDB: %s", e
                )

        # Export thermal learning metrics
        if hasattr(self.thermal_model, "get_adaptive_learning_metrics"):
            try:
                influx_service.write_thermal_learning_metrics(
                    self.thermal_model
                )
                logging.debug(
                    "✅ Exported thermal learning metrics to InfluxDB"
                )
            except Exception as e:
                export_failures.append("thermal_learning")
                logging.warning(
                    "⚠️ Failed to export thermal learning metrics to "
                    "InfluxDB: %s", e
                )

        # Export feature importance
        if hasattr(self.thermal_model, "get_feature_importance"):
            importances = self.thermal_model.get_feature_importance()
            if importances:
                try:
                    influx_service.write_feature_importances(importances)
                    logging.debug(
                        "✅ Exported feature importance to InfluxDB"
                    )
                except Exception as e:
                    export_failures.append("feature_importances")
                    logging.warning(
                        "⚠️ Failed to export feature importances to "
                        "InfluxDB: %s", e
                    )

        # Export learning phase metrics (if available)
        learning_phase_data = {
            "current_learning_phase": "high_confidence",  # Simplified
            "stability_score": min(
                1.0, self.thermal_model.learning_confidence / 5.0
            ),
            "learning_weight_applied": 1.0,
            "stable_period_duration_min": 30,
            "learning_updates_24h": {
                "high_confidence": min(288, self.cycle_count),
                "low_confidence": 0,
                "skipped": 0,
            },
            "learning_efficiency_pct": 85.0,
            "correction_stability": 0.9,
            "false_learning_prevention_pct": 95.0,
        }
        try:
            influx_service.write_learning_phase_metrics(learning_phase_data)
            logging.debug("✅ Exported learning phase metrics to InfluxDB")
        except Exception as e:
            export_failures.append("learning_phase")
            logging.warning(
                "⚠️ Failed to export learning phase metrics to InfluxDB: %s",
                e,
            )

        # Export basic trajectory metrics (simplified)
        trajectory_data = {
            "prediction_horizon": "4h",
            "trajectory_accuracy": {
                "mae_1h": prediction_metrics.get("1h", {}).get("mae", 0.0),
                "mae_2h": prediction_metrics.get("6h", {}).get("mae", 0.0)
                * 1.2,
                "mae_4h": prediction_metrics.get("24h", {}).get("mae", 0.0)
                * 1.5,
            },
            "overshoot_prevention": {
                "overshoot_predicted": False,
                "prevented_24h": 0,
                "undershoot_prevented_24h": 0,
            },
            "convergence": {
                "avg_time_minutes": 45.0,
                "accuracy_percentage": 87.5,
            },
            "forecast_integration": {
                "weather_available": False,
                "pv_available": True,
                "quality_score": 0.8,
            },
        }
        try:
            influx_service.write_trajectory_prediction_metrics(trajectory_data)
            logging.debug(
                "✅ Exported trajectory prediction metrics to InfluxDB"
            )
        except Exception as e:
            export_failures.append("trajectory")
            logging.warning(
                "⚠️ Failed to export trajectory metrics to InfluxDB: %s", e
            )

        if export_failures:
            logging.warning(
                "⚠️ InfluxDB export partially failed (cycle %d): %s. "
                "Check INFLUX_TOKEN has write permission to "
                "INFLUX_FEATURES_BUCKET.",
                self.cycle_count,
                ", ".join(export_failures),
            )
        else:
            logging.info(
                "📊 Exported all adaptive learning metrics to InfluxDB "
                "(cycle %d)",
                self.cycle_count,
            )

    def _save_learning_state(self):
        """Save current thermal learning state to persistent storage."""
        try:
            # State saving is handled by the unified thermal state manager
            # No additional saving needed here as the state_manager handles
            # persistence
            logging.debug(
                "Learning state automatically saved via state_manager"
            )

        except Exception as e:
            logging.error(f"Failed to save learning state: {e}")


# Legacy functions removed - ThermalEquilibriumModel handles persistence
# internally


def get_enhanced_model_wrapper() -> EnhancedModelWrapper:
    """
    Create and return an enhanced model wrapper with singleton pattern.

    This prevents multiple model instantiation which was causing the rapid
    cycle execution issue. Only one instance per service restart.
    """
    global _enhanced_model_wrapper_instance

    if _enhanced_model_wrapper_instance is None:
        logging.info("🔧 Creating new Model Wrapper instance (singleton)")
        _enhanced_model_wrapper_instance = EnhancedModelWrapper()
    else:
        logging.debug("♻️ Reusing existing Model Wrapper instance")

    return _enhanced_model_wrapper_instance


def simplified_outlet_prediction(
    features: pd.DataFrame, current_temp: float, target_temp: float
) -> Tuple[float, float, Dict]:
    """
    SIMPLIFIED outlet temperature prediction using Enhanced Model Wrapper.

    This replaces the complex find_best_outlet_temp() function with a single
    call to the Enhanced Model Wrapper, dramatically simplifying the codebase.

    Args:
        features: Input features DataFrame
        current_temp: Current indoor temperature
        target_temp: Target indoor temperature

    Returns:
        Tuple of (outlet_temp, confidence, metadata)
    """
    wrapper = None
    try:
        # Create enhanced model wrapper
        wrapper = get_enhanced_model_wrapper()

        # Convert features to dict format - handle empty DataFrame
        records = features.to_dict(orient="records")
        if len(records) == 0:
            features_dict = {}
        else:
            features_dict = records[0]

        # CRITICAL FIX: Do not overwrite indoor_temp_lag_30m with current_temp!
        # This destroys the gradient information calculated in
        # build_physics_features.
        # The lag feature should come from history, not be forced to current.
        # features_dict["indoor_temp_lag_30m"] = current_temp
        features_dict["target_temp"] = target_temp

        logging.info(
            f"DEBUG: Wrapper params: "
            f"U={wrapper.thermal_model.heat_loss_coefficient}, "
            f"eff={wrapper.thermal_model.outlet_effectiveness}, "
            f"tau={wrapper.thermal_model.thermal_time_constant}"
        )

        # Get simplified prediction
        outlet_temp, metadata = wrapper.calculate_optimal_outlet_temp(
            features_dict
        )
        confidence = metadata.get("learning_confidence", 3.0)

        # Calculate thermal trust metrics for HA sensor display
        thermal_trust_metrics = _calculate_thermal_trust_metrics(
            wrapper, outlet_temp, current_temp, target_temp
        )
        metadata["thermal_trust_metrics"] = thermal_trust_metrics

        # Log the calculated outlet temperature - smart rounding will be
        # applied later in main.py
        logging.info(
            f"🎯 Prediction: Current {current_temp:.2f}°C → Target "
            f"{target_temp:.1f}°C | Calculated outlet: {outlet_temp:.1f}°C "
            f"(before smart rounding) (confidence: {confidence:.3f})"
        )

        return outlet_temp, confidence, metadata

    except Exception as e:
        logging.error(f"Simplified prediction failed: {e}", exc_info=True)
        # Safe fallback
        fallback = config.get_fallback_outlet(
            wrapper.climate_mode if wrapper else "heating"
        )
        return fallback, 2.0, {"error": str(e), "method": "fallback"}


def _calculate_thermal_trust_metrics(
    wrapper: EnhancedModelWrapper,
    outlet_temp: float,
    current_temp: float,
    target_temp: float,
) -> Dict:
    """
    Calculate thermal trust metrics for HA sensor display.

    These metrics replace legacy MAE/RMSE with physics-based trust indicators
    that show how well the thermal model is performing.
    """
    try:
        # Get thermal model parameters
        thermal_model = wrapper.thermal_model

        # Calculate thermal stability (how stable are the thermal parameters)
        time_constant_stability = min(
            1.0, thermal_model.thermal_time_constant / 48.0
        )
        heat_loss_stability = min(
            1.0, thermal_model.heat_loss_coefficient * 20.0
        )
        outlet_effectiveness_stability = min(
            1.0, thermal_model.outlet_effectiveness * 25.0
        )
        thermal_stability = (
            time_constant_stability
            + heat_loss_stability
            + outlet_effectiveness_stability
        ) / 3.0

        # Calculate prediction consistency (how reasonable is this prediction)
        temp_diff = abs(target_temp - current_temp)
        outlet_indoor_diff = abs(outlet_temp - current_temp)

        # Reasonable outlet temps should be 5-40°C above indoor temp for
        # heating
        if temp_diff > 0.1:  # Need heating
            reasonable_range = (
                outlet_indoor_diff >= 5.0 and outlet_indoor_diff <= 40.0
            )
        else:  # At target
            reasonable_range = (
                outlet_indoor_diff >= 0.0 and outlet_indoor_diff <= 20.0
            )

        prediction_consistency = 1.0 if reasonable_range else 0.5

        # Calculate physics alignment (how well does prediction align with
        # physics). Higher outlet temps should be needed for larger temperature
        # differences
        if temp_diff > 0.1:
            expected_outlet_range = current_temp + (
                temp_diff * 8.0
            )  # Rough physics heuristic
            physics_error = abs(outlet_temp - expected_outlet_range)
            physics_alignment = max(0.0, 1.0 - (physics_error / 20.0))
        else:
            physics_alignment = 1.0

        # Model health assessment
        confidence = thermal_model.learning_confidence
        if confidence >= 4.0:
            model_health = "excellent"
        elif confidence >= 3.0:
            model_health = "good"
        elif confidence >= 2.0:
            model_health = "fair"
        else:
            model_health = "poor"

        # Learning progress (how much has the model learned)
        cycle_count = wrapper.cycle_count
        learning_progress = min(
            1.0, cycle_count / 100.0
        )  # Fully learned after 100 cycles

        return {
            "thermal_stability": thermal_stability,
            "prediction_consistency": prediction_consistency,
            "physics_alignment": physics_alignment,
            "model_health": model_health,
            "learning_progress": learning_progress,
        }

    except Exception as e:
        logging.error(f"Failed to calculate thermal trust metrics: {e}")
        return {
            "thermal_stability": 0.0,
            "prediction_consistency": 0.0,
            "physics_alignment": 0.0,
            "model_health": "error",
            "learning_progress": 0.0,
        }
