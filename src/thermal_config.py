"""
Central Thermal Parameter Configuration - Single Source of Truth

This module provides a centralized configuration for all thermal parameters,
eliminating the architectural anti-pattern of having duplicate parameter
definitions scattered across multiple files.

Key benefits:
- Single source of truth for all thermal parameters
- Consistent bounds across calibration and runtime systems
- Built-in validation and type safety
- Clear documentation for each parameter
- Easy maintenance and updates
"""

from typing import Dict, Tuple


class ThermalParameterConfig:
    """
    Centralized thermal parameter configuration for the ML heating system.

    This class provides defaults, bounds, and validation for all thermal
    parameters used throughout the system, ensuring consistency between
    calibration, runtime operation, and validation systems.
    """

    # Default parameter values optimized for moderate insulation houses
    # UPDATED FOR TDD COMPLIANCE
    # physically reasonable ranges with realistic heat balance
    DEFAULTS = {
        'outlet_temp_max': 35.0,           # °C
        'outlet_temp_min': 0.0,           # °C
        'thermal_time_constant': 4.0,      # hours
        'equilibrium_ratio': 0.17,         # dimensionless
        'total_conductance': 0.8,         # 1/hour
        'pv_heat_weight': 0.0002,           # °C/W
        'fireplace_heat_weight': 1.0,      # °C
        'tv_heat_weight': 0.35,             # °C
        'fp_heat_output_kw': 3.0,          # kW
        'fp_decay_time_constant': 2.0,    # hours
        'room_spread_delay_minutes': 30.0,  # minutes
        'adaptive_learning_rate': 0.01,
        'learning_confidence': 3.0,
        'min_learning_rate': 0.001,
        'max_learning_rate': 0.1,
        'heat_loss_coefficient': 0.15,      # 1/hour (Corrected baseline)
        'outlet_effectiveness': 0.93,      # dimensionless
        'delta_t_floor': 2.4,              # °C
        'cloud_factor_exponent': 1.0,      # dimensionless
        'solar_lag_minutes': 45.0,         # minutes
        'solar_decay_tau_hours': 1.0,      # hours
        'slab_time_constant_hours': 1.0,  # hours (UFH slab forced-convection time constant; data-fitted from 180m² floor)
    }

    # Parameter bounds (min, max) for optimization and validation
    # These bounds are designed to allow realistic parameter exploration
    # while preventing physically impossible values
    BOUNDS = {
        'outlet_temp_max': (30.0, 70.0),   # °C
        'outlet_temp_min': (0.0, 30.0),   # °C
        'thermal_time_constant': (3.0, 100.0),     # Hours
        'equilibrium_ratio': (0.1, 0.9),         # dimensionless
        'total_conductance': (0.1, 0.8),         # 1/hour
        'pv_heat_weight': (0.0001, 0.005),       # W/°C
        'fireplace_heat_weight': (0.01, 6.0),    # 1/°C
        'tv_heat_weight': (0.05, 1.5),           # W/°C
        'fp_heat_output_kw': (0.5, 15.0),        # kW
        'fp_decay_time_constant': (0.1, 5.0),    # hours
        'room_spread_delay_minutes': (0.0, 180.0),  # minutes
        'adaptive_learning_rate': (0.001, 0.1),
        'learning_confidence': (1.0, 5.0),
        'min_learning_rate': (0.0001, 0.01),
        'max_learning_rate': (0.01, 0.2),
        'heat_loss_coefficient': (0.01, 1.2),
        'outlet_effectiveness': (0.3, 2.0),
        'delta_t_floor': (0.0, 10.0),      # °C
        'cloud_factor_exponent': (0.1, 3.0),  # dimensionless
        'solar_lag_minutes': (0.0, 180.0),  # minutes
        'solar_decay_tau_hours': (0.0, 3.0),  # hours
        'slab_time_constant_hours': (0.5, 3.0),  # hours (forced convection: 0.5h min to 3h max)
    }

    # Parameter descriptions for documentation and debugging
    DESCRIPTIONS = {
        'outlet_temp_max': 'Maximum outlet temperature (°C)',
        'outlet_temp_min': 'Minimum outlet temperature (°C)',
        'thermal_time_constant':
            'Time constant for thermal equilibrium (hours)',
        'equilibrium_ratio': 'Equilibrium ratio (dimensionless)',
        'total_conductance': 'Total conductance (1/hour)',
        'pv_heat_weight': 'PV power heating contribution (W/°C)',
        'fireplace_heat_weight': 'Fireplace heating contribution (1/°C)',
        'tv_heat_weight': 'TV/appliance heating contribution (W/°C)',
        'fp_heat_output_kw': 'Fireplace channel startup heat output (kW)',
        'fp_decay_time_constant': 'Fireplace channel heat decay time constant (hours)',
        'room_spread_delay_minutes': 'Fireplace channel room-spread delay (minutes)',
        'adaptive_learning_rate':
            'Adaptive learning rate for parameter adjustment',
        'learning_confidence': 'Confidence in the current learned parameters',
        'min_learning_rate': 'Minimum learning rate',
        'max_learning_rate': 'Maximum learning rate',
        'heat_loss_coefficient': 'Heat loss coefficient (1/hour)',
        'outlet_effectiveness': 'Outlet effectiveness (dimensionless)',
        'delta_t_floor': 'Heat-pump loop delta-T floor used by the HP channel (°C)',
        'cloud_factor_exponent': 'Solar channel cloud-cover attenuation exponent',
        'solar_lag_minutes': 'Effective delay/smoothing window for solar gain',
        'solar_decay_tau_hours': 'Solar channel residual-heat decay time constant (hours)',
        'slab_time_constant_hours': 'UFH slab (Estrich) first-order thermal time constant (hours)',
    }

    # Parameter units for display and logging
    UNITS = {
        'outlet_temp_max': '°C',
        'outlet_temp_min': '°C',
        'thermal_time_constant': 'hours',
        'equilibrium_ratio': 'dimensionless',
        'total_conductance': '1/hour',
        'pv_heat_weight': 'W/°C',
        'fireplace_heat_weight': '1/°C',
        'tv_heat_weight': 'W/°C',
        'fp_heat_output_kw': 'kW',
        'fp_decay_time_constant': 'hours',
        'room_spread_delay_minutes': 'minutes',
        'adaptive_learning_rate': 'dimensionless',
        'learning_confidence': 'dimensionless',
        'min_learning_rate': 'dimensionless',
        'max_learning_rate': 'dimensionless',
        'heat_loss_coefficient': '1/hour',
        'outlet_effectiveness': 'dimensionless',
        'delta_t_floor': '°C',
        'cloud_factor_exponent': 'dimensionless',
        'solar_lag_minutes': 'minutes',
        'solar_decay_tau_hours': 'hours',
        'slab_time_constant_hours': 'hours',
    }

    @classmethod
    def get_default(cls, param_name: str) -> float:
        """
        Get the default value for a thermal parameter.

        Args:
            param_name: Name of the thermal parameter

        Returns:
            Default value for the parameter

        Raises:
            KeyError: If parameter name is not recognized
        """
        if param_name not in cls.DEFAULTS:
            raise KeyError(f"Unknown thermal parameter: {param_name}")
        return cls.DEFAULTS[param_name]

    @classmethod
    def get_bounds(cls, param_name: str) -> Tuple[float, float]:
        """
        Get the bounds (min, max) for a thermal parameter.

        Args:
            param_name: Name of the thermal parameter

        Returns:
            Tuple of (min_value, max_value)

        Raises:
            KeyError: If parameter name is not recognized
        """
        if param_name not in cls.BOUNDS:
            raise KeyError(f"Unknown thermal parameter: {param_name}")
        return cls.BOUNDS[param_name]

    @classmethod
    def validate_parameter(cls, param_name: str, value: float) -> bool:
        """
        Validate that a parameter value is within acceptable bounds.

        Args:
            param_name: Name of the thermal parameter
            value: Value to validate

        Returns:
            True if value is within bounds, False otherwise

        Raises:
            KeyError: If parameter name is not recognized
        """
        min_val, max_val = cls.get_bounds(param_name)
        return min_val <= value <= max_val

    @classmethod
    def clamp_parameter(cls, param_name: str, value: float) -> float:
        """
        Clamp a parameter value to be within acceptable bounds.

        Args:
            param_name: Name of the thermal parameter
            value: Value to clamp

        Returns:
            Value clamped to be within bounds

        Raises:
            KeyError: If parameter name is not recognized
        """
        min_val, max_val = cls.get_bounds(param_name)
        return max(min_val, min(value, max_val))

    @classmethod
    def get_description(cls, param_name: str) -> str:
        """
        Get a human-readable description of a thermal parameter.

        Args:
            param_name: Name of the thermal parameter

        Returns:
            Description string

        Raises:
            KeyError: If parameter name is not recognized
        """
        if param_name not in cls.DESCRIPTIONS:
            raise KeyError(f"Unknown thermal parameter: {param_name}")
        return cls.DESCRIPTIONS[param_name]

    @classmethod
    def get_unit(cls, param_name: str) -> str:
        """
        Get the unit string for a thermal parameter.

        Args:
            param_name: Name of the thermal parameter

        Returns:
            Unit string

        Raises:
            KeyError: If parameter name is not recognized
        """
        if param_name not in cls.UNITS:
            raise KeyError(f"Unknown thermal parameter: {param_name}")
        return cls.UNITS[param_name]

    @classmethod
    def get_all_defaults(cls) -> Dict[str, float]:
        """
        Get all default parameter values.

        Returns:
            Dictionary mapping parameter names to default values
        """
        return cls.DEFAULTS.copy()

    @classmethod
    def get_all_bounds(cls) -> Dict[str, Tuple[float, float]]:
        """
        Get all parameter bounds.

        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        return cls.BOUNDS.copy()

    @classmethod
    def get_parameter_info(cls, param_name: str) -> Dict:
        """
        Get comprehensive information about a thermal parameter.

        Args:
            param_name: Name of the thermal parameter

        Returns:
            Dictionary with default, bounds, description, and unit

        Raises:
            KeyError: If parameter name is not recognized
        """
        return {
            'default': cls.get_default(param_name),
            'bounds': cls.get_bounds(param_name),
            'description': cls.get_description(param_name),
            'unit': cls.get_unit(param_name)
        }

    @classmethod
    def get_all_parameter_info(cls) -> Dict[str, Dict]:
        """
        Get comprehensive information about all thermal parameters.

        Returns:
            Dictionary mapping parameter names to their info dictionaries
        """
        return {
            param_name: cls.get_parameter_info(param_name)
            for param_name in cls.DEFAULTS.keys()
        }


# Convenience functions for backward compatibility
def get_thermal_default(param_name: str) -> float:
    """Convenience function to get thermal parameter default."""
    return ThermalParameterConfig.get_default(param_name)


def get_thermal_bounds(param_name: str) -> Tuple[float, float]:
    """Convenience function to get thermal parameter bounds."""
    return ThermalParameterConfig.get_bounds(param_name)


def validate_thermal_parameter(param_name: str, value: float) -> bool:
    """Convenience function to validate thermal parameter."""
    return ThermalParameterConfig.validate_parameter(param_name, value)
