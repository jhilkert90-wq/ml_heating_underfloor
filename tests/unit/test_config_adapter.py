from config_adapter import convert_addon_to_env


def test_convert_addon_to_env_maps_learning_threshold_options():
    env = convert_addon_to_env(
        {
            "learning_dead_zone": 0.02,
            "pv_learning_threshold": 75,
            "recent_errors_window": 7,
            "cooling_physics_min_outdoor_rolling_24h_c": 19.5,
            "outlet_effectiveness_mode": "curve",
        }
    )

    assert env["LEARNING_DEAD_ZONE"] == "0.02"
    assert env["PV_LEARNING_THRESHOLD"] == "75"
    assert env["RECENT_ERRORS_WINDOW"] == "7"
    assert env["COOLING_PHYSICS_MIN_OUTDOOR_ROLLING_24H_C"] == "19.5"
    assert env["OUTLET_EFFECTIVENESS_MODE"] == "curve"


def test_convert_addon_to_env_maps_forecast_and_direct_power_options():
    env = convert_addon_to_env(
        {
            "heat_forecast_enabled": True,
            "heat_forecast_electricity_counter_entity": "sensor.heating_kwh",
            "heat_forecast_defrost_electricity_counter_entity": "sensor.defrost_kwh",
            "cop_calibration_enabled": True,
            "cop_defrost_el_in_heating_counter": "auto",
            "thermal_power_entity": "sensor.heat_power",
            "thermal_power_source": "direct",
        }
    )

    assert env["HEAT_FORECAST_ENABLED"] == "true"
    assert env["HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID"] == "sensor.heating_kwh"
    assert env["HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID"] == "sensor.defrost_kwh"
    assert env["COP_CALIBRATION_ENABLED"] == "true"
    assert env["COP_DEFROST_EL_IN_HEATING_COUNTER"] == "auto"
    assert env["THERMAL_POWER_ENTITY_ID"] == "sensor.heat_power"
    assert env["THERMAL_POWER_SOURCE"] == "direct"
