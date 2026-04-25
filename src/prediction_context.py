"""
Unified Prediction Context Service

This module provides a centralized service for creating consistent
environmental contexts across all prediction systems (binary search,
smart rounding, trajectory).

Eliminates the inconsistency where:
- Binary search used forecast-based conditions
- Smart rounding used current-only conditions  
- Trajectory prediction used forecast-based conditions

Now all three systems use identical environmental data through this service.
"""

import logging
from typing import Dict, Tuple, List, Optional

from . import config


class UnifiedPredictionContext:
    """
    Centralized service for creating consistent prediction contexts.
    
    Ensures binary search, smart rounding, and trajectory prediction all use
    identical environmental conditions for their thermal model predictions.
    """
    
    @staticmethod
    def create_prediction_context(
        features: Optional[Dict],
        outdoor_temp: float,
        pv_power: float,
        thermal_features: Dict,
        target_temp: Optional[float] = None,
        current_temp: Optional[float] = None,
    ) -> Dict:
        """
        Create a unified prediction context that all systems should use.

        This replaces the individual context creation in each system with
        a single source of truth for environmental conditions.

        Args:
            features: Full feature dictionary with forecast data
            outdoor_temp: Current outdoor temperature
            pv_power: Current PV power
            thermal_features: Thermal features dict

        Returns:
            Dict with standardized prediction context including:
            - avg_outdoor: Cycle-aligned outdoor temp (forecast if available)
            - avg_pv: Cycle-aligned PV power (forecast if available)
            - outdoor_forecast: TRAJECTORY_STEPS-hour outdoor temperature forecast array
            - pv_forecast: TRAJECTORY_STEPS-hour PV power forecast array
            - fireplace_on: Fireplace status
            - tv_on: TV status
            - use_forecasts: Boolean indicating if forecasts were used
        """
        context = {
            "target_temp": target_temp,
            "current_temp": current_temp,
        }

        # Get cycle time from config
        cycle_minutes = config.CYCLE_INTERVAL_MINUTES
        cycle_hours = cycle_minutes / 60.0

        # Validate cycle time against reasonable limits
        max_reasonable_cycle = 180
        if cycle_minutes > max_reasonable_cycle:
            cycle_hours = max_reasonable_cycle / 60.0

        # Extract forecast data if available
        outdoor_forecast = []
        pv_forecast = []
        cloud_cover_forecast = []

        if features:
            # Extract up to TRAJECTORY_STEPS-hour forecasts dynamically
            n_fc = config.TRAJECTORY_STEPS
            _cc_default = 0.0

            outdoor_forecast = [
                features.get(f'temp_forecast_{h}h', outdoor_temp)
                for h in range(1, n_fc + 1)
            ]
            pv_forecast = [
                features.get(f'pv_forecast_{h}h', pv_power)
                for h in range(1, n_fc + 1)
            ]
            # Extract cloud cover forecasts (0-100%)
            # Default 0% (clear sky) — when CLOUD_COVER_CORRECTION_ENABLED=false
            # physics_features.py already sends 0.0 values.
            cloud_cover_forecast = [
                features.get(f'cloud_cover_forecast_{h}h', _cc_default)
                for h in range(1, n_fc + 1)
            ]

            # Calculate cycle-aligned forecast: pick the slot whose hour index
            # best matches the cycle length, capped at the last available slot.
            if cycle_hours <= 1.0:
                # 0-60min cycles: use average over the cycle
                # Assuming linear interpolation between current and 1h forecast
                # The average condition during the cycle is the value at the midpoint.
                # For a 30 min cycle (0.5h), midpoint is 15 min (0.25h).
                # Weight = midpoint / 1h = (cycle_hours / 2) / 1 = cycle_hours / 2
                weight = cycle_hours / 2.0
                avg_outdoor = (
                    outdoor_temp * (1 - weight) + outdoor_forecast[0] * weight
                )
                avg_pv = pv_power * (1 - weight) + pv_forecast[0] * weight
                avg_cloud_cover = cloud_cover_forecast[0]
            else:
                # For cycle_hours > 1: round to nearest hour, cap at n_fc, floor at 0
                hour_idx = max(0, min(int(round(cycle_hours)), n_fc) - 1)
                avg_outdoor = outdoor_forecast[hour_idx]
                avg_pv = pv_forecast[hour_idx]
                avg_cloud_cover = cloud_cover_forecast[hour_idx]

            use_forecasts = True

            logging.info(
                f"🌡️ Using cycle-aligned forecast ({cycle_minutes}min): "
                f"outdoor={avg_outdoor:.1f}°C "
                f"(vs current {outdoor_temp:.1f}°C), "
                f"PV={avg_pv:.0f}W "
                f"(vs current {pv_power:.0f}W)"
            )
        else:
            # No forecast data available, use current values
            n_fc = config.TRAJECTORY_STEPS
            avg_outdoor = outdoor_temp
            avg_pv = pv_power
            outdoor_forecast = [outdoor_temp] * n_fc
            pv_forecast = [pv_power] * n_fc
            cloud_cover_forecast = [0.0] * n_fc  # Default clear sky
            avg_cloud_cover = 0.0
            use_forecasts = False
            logging.debug(
                f"🌡️ Using current conditions (no forecasts): "
                f"outdoor={outdoor_temp:.1f}°C, PV={pv_power:.0f}W"
            )
        
        # Build unified context
        context = {
            'avg_outdoor': avg_outdoor,
            'avg_pv': avg_pv,
            'outdoor_forecast': outdoor_forecast,
            'pv_forecast': pv_forecast,
            'cloud_cover_forecast': cloud_cover_forecast,
            'avg_cloud_cover': avg_cloud_cover,
            'fireplace_on': thermal_features.get('fireplace_on', 0.0),
            'tv_on': thermal_features.get('tv_on', 0.0),
            'use_forecasts': use_forecasts,
            'current_outdoor': outdoor_temp,
            'current_pv': pv_power,
            'target_temp': target_temp,
            'current_temp': current_temp
        }
        
        return context
    
    @staticmethod
    def get_thermal_model_params(context: Dict) -> Dict:
        """
        Extract thermal model parameters from unified context.
        
        This ensures all thermal model calls use identical parameters
        regardless of which system (binary search, smart rounding, trajectory)
        is making the call.
        
        Args:
            context: Unified prediction context from
                create_prediction_context()

        Returns:
            Dict with thermal model parameters
        """
        return {
            'outdoor_temp': context['avg_outdoor'],  # Use forecast average
            'pv_power': context['avg_pv'],          # Use forecast average  
            'fireplace_on': context['fireplace_on'],
            'tv_on': context['tv_on']
        }


class PredictionContextManager:
    """
    Manager class that maintains prediction context state and provides
    convenient access methods for different prediction systems.
    """
    
    def __init__(self):
        self._current_context: Optional[Dict] = None
        self._features: Optional[Dict] = None
    
    def set_features(self, features: Dict) -> None:
        """Store features for context creation."""
        self._features = features
        
    def create_context(self, outdoor_temp: float, pv_power: float,
                       thermal_features: Dict,
                       target_temp: Optional[float] = None,
                       current_temp: Optional[float] = None) -> Dict:
        """
        Create and store unified prediction context.

        This context will be used by all prediction systems to ensure
        consistency.
        """
        self._current_context = (
            UnifiedPredictionContext.create_prediction_context(
                features=self._features,
                outdoor_temp=outdoor_temp,
                pv_power=pv_power,
                thermal_features=thermal_features,
                target_temp=target_temp,
                current_temp=current_temp
            )
        )
        return self._current_context
    
    def get_context(self) -> Optional[Dict]:
        """Get the current unified context."""
        return self._current_context
    
    def get_thermal_model_params(self) -> Dict:
        """Get thermal model parameters from current context."""
        if self._current_context is None:
            raise ValueError(
                "No prediction context available. Call create_context() first."
            )

        return UnifiedPredictionContext.get_thermal_model_params(
            self._current_context
        )

    def get_forecast_arrays(self) -> Tuple[List[float], List[float]]:
        """Get forecast arrays for trajectory prediction."""
        if self._current_context is None:
            raise ValueError(
                "No prediction context available. Call create_context() first."
            )

        return (
            self._current_context['outdoor_forecast'],
            self._current_context['pv_forecast']
        )
    
    def uses_forecasts(self) -> bool:
        """Check if the current context uses forecast data."""
        if self._current_context is None:
            return False
        return self._current_context.get('use_forecasts', False)


# Global instance for easy access across modules
prediction_context_manager = PredictionContextManager()