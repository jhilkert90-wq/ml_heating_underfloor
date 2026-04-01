# Home Assistant Sensors & Inputs for Calibration

This document lists all Home Assistant (HA) sensors and inputs required by the ML
heating system for real-time control, adaptive learning, and physics calibration.

## Required Sensors (Critical)

| Config Variable | Default Entity ID | Type | Description |
|---|---|---|---|
| `INDOOR_TEMP_ENTITY_ID` | `sensor.kuche_temperatur` | Temperature (°C) | Current indoor temperature — primary feedback signal |
| `OUTDOOR_TEMP_ENTITY_ID` | `sensor.thermometer_waermepume_kompensiert` | Temperature (°C) | Outdoor temperature for heat-loss calculation |
| `ACTUAL_OUTLET_TEMP_ENTITY_ID` | `sensor.hp_outlet_temp` | Temperature (°C) | Measured heat-pump outlet (Vorlauf BT2) |
| `INLET_TEMP_ENTITY_ID` | `sensor.hp_inlet_temp` | Temperature (°C) | Measured heat-pump inlet / return (Rücklauf BT3) — used for slab model |
| `TARGET_INDOOR_TEMP_ENTITY_ID` | `input_number.target_indoor_temp` | Temperature (°C) | User-set desired indoor temperature |
| `TARGET_OUTLET_TEMP_ENTITY_ID` | `sensor.ml_vorlauftemperatur` | Temperature (°C) | **Output**: ML-computed outlet setpoint written back to HA |

## Heat-Pump Status Sensors

| Config Variable | Default Entity ID | Type | Description |
|---|---|---|---|
| `DHW_STATUS_ENTITY_ID` | `binary_sensor.hp_dhw_heating_status` | Binary | Domestic hot water heating active (blocks heating cycle) |
| `DEFROST_STATUS_ENTITY_ID` | `binary_sensor.hp_defrosting_status` | Binary | Defrost cycle active (blocks heating cycle) |
| `DISINFECTION_STATUS_ENTITY_ID` | `binary_sensor.hp_legionella_prevention` | Binary | Legionella prevention active (blocks heating cycle) |
| `DHW_BOOST_HEATER_STATUS_ENTITY_ID` | `binary_sensor.hp_dhw_boost_heater` | Binary | DHW boost heater active (blocks heating cycle) |
| `ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID` | `sensor.hp_target_temp_circuit1` | Temperature (°C) | HP's own target outlet (used to detect external overrides) |
| `HEATING_STATUS_ENTITY_ID` | `climate.heizung_2` | Climate | HP heating system climate entity |

## Heat-Source Channel Sensors

These sensors feed the decomposed heat-source learning channels
(see `src/heat_source_channels.py`).

| Config Variable | Default Entity ID | Type | Channel | Description |
|---|---|---|---|---|
| `PV_POWER_ENTITY_ID` | `sensor.power_pv` | Power (W) | SolarChannel | Current PV production — determines solar heat contribution |
| `PV_FORECAST_ENTITY_ID` | `sensor.energy_production_today_4` | Forecast | SolarChannel | PV production forecast for proactive pre-sunset / sunrise control |
| `FIREPLACE_STATUS_ENTITY_ID` | `binary_sensor.fireplace_active` | Binary | FireplaceChannel | Fireplace active — isolates fireplace heat from HP learning |
| `TV_STATUS_ENTITY_ID` | `input_boolean.fernseher` | Binary | TVChannel | TV/electronics on — minor heat source |
| `LIVING_ROOM_TEMP_ENTITY_ID` | `sensor.living_room_temperature` | Temperature (°C) | FireplaceChannel | Living room temp for fireplace heat differential |
| `AVG_OTHER_ROOMS_TEMP_ENTITY_ID` | `sensor.avg_other_rooms_temp` | Temperature (°C) | FireplaceChannel | Average other rooms temp (fireplace spread detection) |

When `ENABLE_HEAT_SOURCE_CHANNELS=true`, the fireplace sensors above feed `FireplaceChannel`, whose state is persisted in `learning_state.heat_source_channels`. When the flag is `false`, the same fireplace sensors are consumed by the legacy `AdaptiveFireplaceLearning` fallback instead.

## Performance & Flow Sensors

| Config Variable | Default Entity ID | Type | Description |
|---|---|---|---|
| `FLOW_RATE_ENTITY_ID` | `sensor.hp_current_flow_rate` | Flow (l/h) | Underfloor heating flow rate — thermal power calculation |
| `POWER_CONSUMPTION_ENTITY_ID` | `sensor.power_wp` | Power (W) | HP electrical power consumption — COP calculation |

## Weather & Forecast Sensors

| Config Variable | Default Entity ID | Type | Description |
|---|---|---|---|
| `OPENWEATHERMAP_TEMP_ENTITY_ID` | `sensor.openweathermap_temperature` | Temperature (°C) | Weather service outdoor temp for forecast calibration |

## User Control Inputs

| Config Variable | Default Entity ID | Type | Description |
|---|---|---|---|
| `ML_HEATING_CONTROL_ENTITY_ID` | `input_boolean.ml_heating_control` | Boolean | Master on/off for ML heating control |
| `SOLAR_CORRECTION_ENTITY_ID` | `input_number.ml_heating_solar_correction` | Number | Manual solar correction factor (default 1.0) |

## ML Output Sensors

| Config Variable | Default Entity ID | Type | Description |
|---|---|---|---|
| `TARGET_OUTLET_TEMP_ENTITY_ID` | `sensor.ml_vorlauftemperatur` | Temperature (°C) | Computed outlet setpoint |
| `MAE_ENTITY_ID` | `sensor.ml_model_mae` | Number | Model mean absolute error |
| `RMSE_ENTITY_ID` | `sensor.ml_model_rmse` | Number | Model root mean squared error |

## Which Sensors Are Used for What

### Real-Time Control Loop
- Indoor temp, outdoor temp, outlet/inlet temp, target indoor temp
- PV power, fireplace status, TV status
- All blocking-state sensors (DHW, defrost, disinfection, boost)

### Adaptive Learning (per-cycle parameter updates)
- Indoor temp (prediction vs actual)
- Outlet/inlet temp (slab model gradient)
- PV power (solar channel learning, gating HP learning when PV > 500 W)
- Fireplace status and room temperatures (fireplace channel learning in channel mode, legacy adaptive fireplace learning when channel mode is off)
- TV status (TV channel learning)

### Physics Calibration (batch calibration from InfluxDB history)
- Indoor temp, outlet temp, outdoor temp (transient period detection)
- Inlet temp (effective temperature calculation)
- PV power, fireplace status, TV status (multi-source attribution)
- Flow rate, power consumption (COP and thermal power)

### Solar Transition Scenarios (evening / morning)
- PV power + PV forecast → SolarChannel.predict_future_contribution()
- Inlet temp → slab residual heat after HP reduces
- Outdoor temp → heat loss calculation

## Minimum Viable Sensor Set

For basic operation without all features:

1. **Indoor temperature** (required)
2. **Outdoor temperature** (required)
3. **Outlet temperature** (required)
4. **Target indoor temperature** (required)
5. **Inlet temperature** (strongly recommended — enables slab model)
6. **PV power** (recommended — prevents solar heat misattribution)
7. **Fireplace status** (recommended if you have a fireplace)

All other sensors enhance accuracy but the system degrades gracefully
without them.
