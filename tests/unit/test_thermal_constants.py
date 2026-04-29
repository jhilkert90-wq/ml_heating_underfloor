"""Tests for thermal_constants.py – ThermalUnits, ThermalParameterValidator, helpers."""

import logging
import pytest

from src.thermal_constants import (
    PhysicsConstants,
    ThermalParameterValidator,
    ThermalUnits,
    format_thermal_state,
    validate_thermal_parameters,
)


# ---------------------------------------------------------------------------
# PhysicsConstants – basic sanity checks
# ---------------------------------------------------------------------------
class TestPhysicsConstants:
    def test_temperature_bounds_are_sane(self):
        assert PhysicsConstants.MIN_BUILDING_TEMP < 0
        assert PhysicsConstants.MAX_BUILDING_TEMP > 40
        assert PhysicsConstants.MIN_OUTLET_TEMP < PhysicsConstants.MAX_OUTLET_TEMP

    def test_heat_loss_bounds_positive(self):
        assert PhysicsConstants.MIN_HEAT_LOSS_COEFF > 0
        assert PhysicsConstants.MAX_HEAT_LOSS_COEFF > PhysicsConstants.MIN_HEAT_LOSS_COEFF

    def test_learning_rate_bounds(self):
        assert PhysicsConstants.MIN_LEARNING_RATE < PhysicsConstants.DEFAULT_LEARNING_RATE
        assert PhysicsConstants.DEFAULT_LEARNING_RATE < PhysicsConstants.MAX_LEARNING_RATE

    def test_error_thresholds_ordered(self):
        assert (
            PhysicsConstants.ERROR_THRESHOLD_LOW
            < PhysicsConstants.ERROR_THRESHOLD_MEDIUM
            < PhysicsConstants.ERROR_THRESHOLD_HIGH
            < PhysicsConstants.ERROR_THRESHOLD_VERY_HIGH
        )


# ---------------------------------------------------------------------------
# ThermalUnits.get_unit
# ---------------------------------------------------------------------------
class TestThermalUnitsGetUnit:
    def test_known_parameter_returns_unit(self):
        assert ThermalUnits.get_unit("indoor_temperature") == "°C"

    def test_unknown_parameter_returns_unknown(self):
        assert ThermalUnits.get_unit("nonexistent_param") == "unknown"

    def test_outlet_temperature_unit(self):
        assert ThermalUnits.get_unit("outlet_temperature") == "°C"

    def test_pv_power_unit(self):
        assert ThermalUnits.get_unit("pv_power") == "W"


# ---------------------------------------------------------------------------
# ThermalUnits.get_range
# ---------------------------------------------------------------------------
class TestThermalUnitsGetRange:
    def test_known_parameter_returns_tuple(self):
        lo, hi = ThermalUnits.get_range("thermal_time_constant")
        assert lo == PhysicsConstants.MIN_TIME_CONSTANT
        assert hi == PhysicsConstants.MAX_TIME_CONSTANT

    def test_unknown_parameter_returns_inf_range(self):
        lo, hi = ThermalUnits.get_range("nonexistent_param")
        assert lo == float("-inf")
        assert hi == float("inf")


# ---------------------------------------------------------------------------
# ThermalUnits.validate_parameter
# ---------------------------------------------------------------------------
class TestThermalUnitsValidateParameter:
    def test_valid_indoor_temperature(self):
        assert ThermalUnits.validate_parameter("indoor_temperature", 20.0) is True

    def test_invalid_indoor_temperature_too_high(self):
        assert ThermalUnits.validate_parameter("indoor_temperature", 100.0) is False

    def test_invalid_indoor_temperature_too_low(self):
        assert ThermalUnits.validate_parameter("indoor_temperature", -50.0) is False

    def test_boundary_value_min_inclusive(self):
        lo, _ = ThermalUnits.get_range("indoor_temperature")
        assert ThermalUnits.validate_parameter("indoor_temperature", lo) is True

    def test_boundary_value_max_inclusive(self):
        _, hi = ThermalUnits.get_range("indoor_temperature")
        assert ThermalUnits.validate_parameter("indoor_temperature", hi) is True

    def test_unknown_parameter_returns_true_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = ThermalUnits.validate_parameter("unknown_param", 42.0)
        assert result is True
        assert "unknown_param" in caplog.text

    def test_valid_heat_loss_coefficient(self):
        assert ThermalUnits.validate_parameter("heat_loss_coefficient", 1.0) is True

    def test_invalid_heat_loss_coefficient_negative(self):
        assert ThermalUnits.validate_parameter("heat_loss_coefficient", -0.5) is False

    def test_valid_outlet_effectiveness(self):
        assert ThermalUnits.validate_parameter("outlet_effectiveness", 0.85) is True

    def test_valid_pv_power(self):
        assert ThermalUnits.validate_parameter("pv_power", 5000.0) is True


# ---------------------------------------------------------------------------
# ThermalUnits.validate_parameters
# ---------------------------------------------------------------------------
class TestThermalUnitsValidateParameters:
    def test_all_valid_numeric_params(self):
        params = {
            "indoor_temperature": 20.0,
            "heat_loss_coefficient": 1.5,
            "outlet_effectiveness": 0.9,
        }
        results = ThermalUnits.validate_parameters(params)
        assert all(results.values())

    def test_mixed_valid_invalid(self):
        params = {
            "indoor_temperature": 20.0,   # valid
            "heat_loss_coefficient": 99.9,  # invalid – way above max
        }
        results = ThermalUnits.validate_parameters(params)
        assert results["indoor_temperature"] is True
        assert results["heat_loss_coefficient"] is False

    def test_non_numeric_values_pass_validation(self):
        params = {"label": "some_string", "flag": True}
        results = ThermalUnits.validate_parameters(params)
        assert results["label"] is True
        assert results["flag"] is True

    def test_integer_values_accepted(self):
        params = {"pv_power": 1000}
        results = ThermalUnits.validate_parameters(params)
        assert results["pv_power"] is True

    def test_empty_dict_returns_empty_result(self):
        assert ThermalUnits.validate_parameters({}) == {}


# ---------------------------------------------------------------------------
# ThermalUnits.format_parameter
# ---------------------------------------------------------------------------
class TestThermalUnitsFormatParameter:
    def test_temperature_formatted_with_one_decimal(self):
        formatted = ThermalUnits.format_parameter("indoor_temperature", 20.123)
        assert "20.1" in formatted
        assert "°C" in formatted

    def test_coefficient_formatted_with_three_decimals(self):
        formatted = ThermalUnits.format_parameter("heat_loss_coefficient", 1.2345)
        assert "1.234" in formatted

    def test_effectiveness_formatted_with_two_decimals(self):
        formatted = ThermalUnits.format_parameter("outlet_effectiveness", 0.856)
        assert "0.86" in formatted

    def test_pv_power_formatted_with_zero_decimals(self):
        formatted = ThermalUnits.format_parameter("pv_power", 1234.5)
        assert "1234" in formatted or "1235" in formatted  # rounded to 0dp
        assert "W" in formatted

    def test_unknown_unit_falls_back_to_three_decimal_format(self):
        formatted = ThermalUnits.format_parameter("learning_rate", 0.01234)
        assert "0.012" in formatted


# ---------------------------------------------------------------------------
# ThermalParameterValidator.validate_heat_balance_parameters
# ---------------------------------------------------------------------------
class TestThermalParameterValidatorHeatBalance:
    def setup_method(self):
        self.validator = ThermalParameterValidator()

    def test_valid_parameters_return_true(self):
        result = self.validator.validate_heat_balance_parameters(
            heat_loss_coeff=1.5,
            outlet_effectiveness=0.8,
            external_weights={"pv": 0.001, "tv": 0.3},
        )
        assert result is True
        assert self.validator.validation_errors == []

    def test_out_of_range_hlc_adds_error(self):
        result = self.validator.validate_heat_balance_parameters(
            heat_loss_coeff=0.0,  # 0 is below min → invalid
            outlet_effectiveness=0.8,
            external_weights={},
        )
        assert result is False
        assert len(self.validator.validation_errors) > 0

    def test_zero_outlet_effectiveness_adds_error(self):
        result = self.validator.validate_heat_balance_parameters(
            heat_loss_coeff=1.0,
            outlet_effectiveness=0.0,  # zero → must be positive
            external_weights={},
        )
        assert result is False

    def test_very_high_hlc_adds_warning_but_not_error(self):
        result = self.validator.validate_heat_balance_parameters(
            heat_loss_coeff=6.0,   # > 5.0 → warning threshold
            outlet_effectiveness=0.8,
            external_weights={},
        )
        # Value is still within RANGES (min=0.1, max=10.0)
        assert result is True
        assert len(self.validator.validation_warnings) > 0

    def test_low_outlet_effectiveness_adds_warning(self):
        result = self.validator.validate_heat_balance_parameters(
            heat_loss_coeff=1.0,
            outlet_effectiveness=0.2,  # < 0.3 → warning
            external_weights={},
        )
        assert result is True
        assert len(self.validator.validation_warnings) > 0

    def test_errors_cleared_on_each_call(self):
        # First call with error
        self.validator.validate_heat_balance_parameters(
            heat_loss_coeff=0.0,
            outlet_effectiveness=0.8,
            external_weights={},
        )
        assert len(self.validator.validation_errors) > 0

        # Second call with valid params clears errors
        self.validator.validate_heat_balance_parameters(
            heat_loss_coeff=1.0,
            outlet_effectiveness=0.8,
            external_weights={},
        )
        assert self.validator.validation_errors == []


# ---------------------------------------------------------------------------
# ThermalParameterValidator.validate_temperature_inputs
# ---------------------------------------------------------------------------
class TestThermalParameterValidatorTemperatureInputs:
    def setup_method(self):
        self.validator = ThermalParameterValidator()

    def test_valid_temperatures_return_true(self):
        result = self.validator.validate_temperature_inputs(
            indoor=20.0, outdoor=5.0, outlet=45.0
        )
        assert result is True

    def test_outlet_below_outdoor_while_heating_adds_error(self):
        result = self.validator.validate_temperature_inputs(
            indoor=22.0, outdoor=10.0, outlet=8.0  # outlet < outdoor while heating
        )
        assert result is False
        assert len(self.validator.validation_errors) > 0

    def test_indoor_much_colder_than_outdoor_adds_warning(self):
        result = self.validator.validate_temperature_inputs(
            indoor=0.0, outdoor=25.0, outlet=45.0
        )
        # Might be valid values but unusual → warning
        assert len(self.validator.validation_warnings) > 0

    def test_high_outlet_low_indoor_adds_warning(self):
        result = self.validator.validate_temperature_inputs(
            indoor=18.0,   # < 25
            outdoor=5.0,
            outlet=65.0,   # > 60
        )
        assert len(self.validator.validation_warnings) > 0

    def test_out_of_range_temperature_adds_error(self):
        result = self.validator.validate_temperature_inputs(
            indoor=200.0,   # way outside range
            outdoor=5.0,
            outlet=45.0,
        )
        assert result is False


# ---------------------------------------------------------------------------
# ThermalParameterValidator.get_validation_report
# ---------------------------------------------------------------------------
class TestThermalParameterValidatorReport:
    def setup_method(self):
        self.validator = ThermalParameterValidator()

    def test_report_with_no_errors_no_warnings(self):
        self.validator.validate_heat_balance_parameters(1.0, 0.8, {})
        report = self.validator.get_validation_report()
        assert "✅" in report

    def test_report_contains_error_messages(self):
        self.validator.validate_heat_balance_parameters(0.0, 0.8, {})
        report = self.validator.get_validation_report()
        assert "VALIDATION ERRORS" in report or "❌" in report

    def test_report_contains_warning_messages(self):
        self.validator.validate_heat_balance_parameters(6.0, 0.8, {})
        report = self.validator.get_validation_report()
        assert "VALIDATION WARNINGS" in report or "⚠️" in report


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
class TestConvenienceFunctions:
    def test_validate_thermal_parameters_all_valid(self):
        params = {
            "indoor_temperature": 20.0,
            "outlet_effectiveness": 0.85,
        }
        assert validate_thermal_parameters(params) is True

    def test_validate_thermal_parameters_one_invalid(self):
        params = {
            "indoor_temperature": 200.0,  # out of range
        }
        assert validate_thermal_parameters(params) is False

    def test_format_thermal_state_returns_string(self):
        params = {
            "indoor_temperature": 20.0,
            "outlet_effectiveness": 0.85,
            "label": "test",
        }
        output = format_thermal_state(params)
        assert isinstance(output, str)
        assert "indoor_temperature" in output
        assert "outlet_effectiveness" in output
        assert "label" in output

    def test_format_thermal_state_non_numeric_passthrough(self):
        params = {"mode": "heating"}
        output = format_thermal_state(params)
        assert "heating" in output
