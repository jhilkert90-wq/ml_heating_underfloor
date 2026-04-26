# ML Heating — Complete Parameter Reference

This document provides a full description of every configuration parameter available in
`ml_heating_underfloor/config.yaml` (Home Assistant add-on) and the equivalent environment
variables in `.env` (standalone/Docker deployments).

**Two configuration surfaces:**

| Surface | Where | Used by |
|---|---|---|
| `config.yaml` → HA add-on UI | `ml_heating_underfloor/config.yaml` | HA add-on Configuration tab |
| `.env` file | Root `.env` (copy from `.env_sample`) | Standalone / Docker / development |

The `config_adapter.py` layer automatically maps add-on `config.yaml` option names to the
environment variables consumed by `src/config.py`, so every option available in the add-on
UI is also available as an env var with the equivalent uppercase name.

---

## Parameter Callouts

> ⚙️ **Must Configure** — You must set these to match your Home Assistant setup.

> 🔄 **Auto-Calibrated** — These values are learned from your historical data. The default shown
> is a sensible starting point for underfloor heating but will be overwritten after calibration.

> 🧪 **Advanced** — These are internal model parameters. Leave them at their defaults unless you
> have a specific reason to change them and understand the thermal physics involved.

---

## 1. Core Entity Configuration

These are the most critical parameters. You **must** set all entity IDs to match your Home Assistant setup.

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `target_indoor_temp_entity` | `TARGET_INDOOR_TEMP_ENTITY_ID` | string | — | ⚙️ The `input_number` or sensor holding the desired indoor setpoint (e.g. `input_number.soll_rt`). ML Heating reads this as its temperature target. |
| `indoor_temp_entity` | `INDOOR_TEMP_ENTITY_ID` | string | — | ⚙️ Primary indoor temperature sensor the system tries to maintain. Use a reliable representative sensor (e.g. average of main living areas). |
| `outdoor_temp_entity` | `OUTDOOR_TEMP_ENTITY_ID` | string | — | ⚙️ Outdoor temperature sensor, ideally compensated and located near the heat pump. Main weather input for the thermal model. |
| `heating_control_entity` | `HEATING_STATUS_ENTITY_ID` | string | — | ⚙️ The climate or sensor entity for your heating system. ML Heating reads its state to detect `heat`, `auto`, or `cool` modes. Control is skipped when the state is off. |
| `outlet_temp_entity` | `ACTUAL_OUTLET_TEMP_ENTITY_ID` | string | — | ⚙️ The heat pump's actual water outlet (supply) temperature sensor. |
| `inlet_temp_entity` | `INLET_TEMP_ENTITY_ID` | string | — | ⚙️ The heat pump's water inlet (return) temperature sensor. Used with the outlet sensor to compute delta-T. |
| `flow_rate_entity` | `FLOW_RATE_ENTITY_ID` | string | — | ⚙️ Flow rate sensor in litres per minute. Combined with delta-T to compute actual thermal power. |
| `power_consumption_entity` | `POWER_CONSUMPTION_ENTITY_ID` | string | — | ⚙️ Heat pump electrical power consumption sensor in Watts. Used for efficiency monitoring. |
| `specific_heat_capacity` | `SPECIFIC_HEAT_CAPACITY` | float (1–10) | `4.186` | Specific heat capacity of the heat-transfer fluid in kJ/kg·K. Use `4.186` for pure water. Adjust slightly for glycol/water mixtures. |
| `target_outlet_temp_entity` | `TARGET_OUTLET_TEMP_ENTITY_ID` | string | — | ⚙️ The entity ML Heating **writes** its calculated target outlet temperature to. In shadow mode a `_shadow` suffix is appended automatically. |
| `actual_target_outlet_temp_entity` | `ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID` | string | — | ⚙️ The entity ML Heating **reads** to find out what outlet temperature was actually applied, for learning. **Active mode**: same as `target_outlet_temp_entity`. **Shadow mode**: point to the heat curve's entity so the model learns from heat-curve decisions. |
| `openweathermap_temp_entity` | `OPENWEATHERMAP_TEMP_ENTITY_ID` | string | — | ⚙️ A weather entity (e.g. `weather.home`) or forecast temperature sensor. Used for trajectory prediction so the model anticipates upcoming temperature changes. |
| `avg_other_rooms_temp_entity` | `AVG_OTHER_ROOMS_TEMP_ENTITY_ID` | string | — | Sensor providing the average temperature of rooms **not** affected by the fireplace. Used as the indoor reference during fireplace events to avoid incorrect learning. |
| `pv_forecast_entity` | `PV_FORECAST_ENTITY_ID` | string | — | HA sensor with today's PV power forecast in `watts` attributes at 15-minute resolution (e.g. from Forecast.Solar integration). Used for solar-aware trajectory planning. |
| `living_room_temp_entity` | `LIVING_ROOM_TEMP_ENTITY_ID` | string | — | Living room temperature sensor. Used exclusively for fireplace detection analysis. |

---

## 2. External Heat Sources

Configure these if your home has PV solar, a fireplace, or significant electronics heat.

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `pv_power_entity` | `PV_POWER_ENTITY_ID` | string | — | ⚙️ Sensor providing current total PV generation in Watts. The thermal model uses this to account for solar heat gain through windows. |
| `solar_correction_entity` | `SOLAR_CORRECTION_ENTITY_ID` | string | — | `input_number` entity (0–100 %) for dynamically adjusting how much PV power is credited as indoor heat gain. Useful when shading reduces actual solar gain. |
| `solar_correction_default_percent` | `SOLAR_CORRECTION_DEFAULT_PERCENT` | float (0–100) | `100.0` | Fallback solar correction % when the correction entity is unavailable. 100 % = full credit; 50 % = half credited. |
| `solar_correction_min_percent` | `SOLAR_CORRECTION_MIN_PERCENT` | float (0–100) | `0.0` | Lower clamp for the solar correction entity. |
| `solar_correction_max_percent` | `SOLAR_CORRECTION_MAX_PERCENT` | float (0–100) | `100.0` | Upper clamp for the solar correction entity. |
| `fireplace_status_entity` | `FIREPLACE_STATUS_ENTITY_ID` | string | — | `binary_sensor` that is `on` when the fireplace or wood stove is active. ML Heating switches to an alternative indoor reference and activates the fireplace heat channel. |
| `tv_status_entity` | `TV_STATUS_ENTITY_ID` | string | — | `input_boolean` or `binary_sensor` that is `on` when the TV or other significant electronics are on. The model accounts for this as a small additive heat source. |

---

## 3. Blocking Detection

ML Heating pauses prediction and learning during these events. Correct entity IDs are required
to avoid corrupted learning data.

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `dhw_status_entity` | `DHW_STATUS_ENTITY_ID` | string | — | ⚙️ `binary_sensor` that is `on` when Domestic Hot Water is being heated. System pauses until DHW is done and outlet temperature stabilises. |
| `defrost_status_entity` | `DEFROST_STATUS_ENTITY_ID` | string | — | ⚙️ `binary_sensor` that is `on` during a heat pump defrost cycle. A longer recovery grace period (`defrost_recovery_grace_minutes`) is applied after defrost ends. |
| `disinfection_status_entity` | `DISINFECTION_STATUS_ENTITY_ID` | string | — | `binary_sensor` that is `on` during a DHW tank thermal disinfection (Legionella protection) cycle. |
| `dhw_boost_heater_entity` | `DHW_BOOST_HEATER_STATUS_ENTITY_ID` | string | — | `binary_sensor` that is `on` when the DHW electric boost heater is active. |

---

## 4. ML Learning Parameters

Control how the model collects history and predicts.

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `history_steps` | `HISTORY_STEPS` | int (3–12) | `6` | Number of historical time slices used as features. More steps = richer context but more memory. |
| `history_step_minutes` | `HISTORY_STEP_MINUTES` | int (5–30) | `10` | Time interval in minutes between each historical step. Match to your data collection granularity. |
| `prediction_horizon_steps` | `PREDICTION_HORIZON_STEPS` | int (12–96) | `24` | How many 5-minute steps ahead the model predicts during calibration. 24 = 120 min. |
| `training_lookback_hours` | `TRAINING_LOOKBACK_HOURS` | int (24–2000) | `1800` | Hours of historical data used for initial calibration. More data = better initial parameters. |
| `cycle_interval_minutes` | `CYCLE_INTERVAL_MINUTES` | int (5–60) | `10` | Minutes between each learn-and-predict cycle. Longer = clearer learning signal; shorter = more responsive but noisier. |
| `max_temp_change_per_cycle` | `MAX_TEMP_CHANGE_PER_CYCLE` | int (1–10) | `2` | Maximum outlet temperature change (°C) allowed in one cycle. Prevents abrupt jumps that could stress the heat pump. |
| `training_data_source` | `TRAINING_DATA_SOURCE` | list | `auto` | Source for initial calibration data. `auto` tries InfluxDB first, then HA history API. `influx` forces InfluxDB. `ha_history` forces HA history API. |

---

## 5. Safety Configuration

Absolute temperature bounds enforced at all times in heating mode.

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `clamp_min_abs` | `CLAMP_MIN_ABS` | float (10–25) | `18.0` | ⚙️ Absolute minimum outlet temperature in heating mode (°C). For underfloor heating keep at 18–25°C. The system will never send a lower setpoint. |
| `clamp_max_abs` | `CLAMP_MAX_ABS` | float (25–55) | `35.0` | ⚙️ Absolute maximum outlet temperature in heating mode (°C). For underfloor heating keep at or below 35°C to protect the floor construction. |

---

## 6. Cooling Mode Configuration

Used only when `heating_control_entity` reports `cool` mode.

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `cooling_clamp_min_abs` | `COOLING_CLAMP_MIN_ABS` | float (14–22) | `18.0` | Absolute minimum outlet temperature in cooling mode (°C). The HP shuts down at this limit. |
| `cooling_clamp_max_abs` | `COOLING_CLAMP_MAX_ABS` | float (18–28) | `24.0` | Maximum outlet temperature in cooling mode (°C). Must be above inlet temp + min_cooling_delta_k. |
| `min_cooling_delta_k` | `MIN_COOLING_DELTA_K` | float (0.5–5) | `2.0` | Minimum inlet–outlet temperature difference (K) required for the HP to operate in cooling. |
| `cooling_shutdown_margin_k` | `COOLING_SHUTDOWN_MARGIN_K` | float (0–3) | `1.0` | Safety margin (K) above the cooling shutdown limit. ML targets at least this margin above `cooling_clamp_min_abs` to prevent short-cycling. |

---

## 7. InfluxDB Configuration

Required for historical calibration. Optional for ongoing operation (HA history API can be used instead).

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `influx_url` | `INFLUX_URL` | string | — | Full URL of your InfluxDB instance including protocol and port (e.g. `http://192.168.0.10:8086`). |
| `influx_token` | `INFLUX_TOKEN` | string | — | API token with read access to the HA data bucket and write access to the features bucket. Leave empty to disable InfluxDB. |
| `influx_org` | `INFLUX_ORG` | string | — | Your InfluxDB organisation name (found in InfluxDB settings). |
| `influx_bucket` | `INFLUX_BUCKET` | string | — | The InfluxDB bucket where HA historical sensor data is stored. Used for initial calibration. |
| `influx_features_bucket` | `INFLUX_FEATURES_BUCKET` | string | `ml_heating_features` | InfluxDB bucket ML Heating writes generated feature metrics to. A `_shadow` suffix is appended in shadow mode. |
| `influx_metrics_export_interval_cycles` | `INFLUX_METRICS_EXPORT_INTERVAL_CYCLES` | int (1–100) | `5` | How many learning cycles pass between each InfluxDB metrics export. Higher = less write load. |

---

## 8. Model Management

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `auto_backup_enabled` | `AUTO_BACKUP_ENABLED` | bool | `true` | When enabled, the thermal state file is backed up before each model update, allowing recovery from a learning regression. |
| `backup_retention_days` | `BACKUP_RETENTION_DAYS` | int (1–365) | `30` | Number of days to retain model backup files before automatic deletion. |
| `unified_state_file` | `UNIFIED_STATE_FILE` | string | `/config/ml_heating/unified_thermal_state.json` | Path to the thermal state JSON file (heating mode). Use `/config/ml_heating/` to store in the HA config folder (accessible via File Editor add-on). |
| `unified_state_file_cooling` | `UNIFIED_STATE_FILE_COOLING` | string | `/config/ml_heating/unified_thermal_state_cooling.json` | Path to the thermal state JSON file for cooling mode. Kept separate to prevent cross-mode learning contamination. |

---

## 9. Dashboard Configuration

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `dashboard_update_interval` | `DASHBOARD_UPDATE_INTERVAL` | int (5–300) | `30` | How often (seconds) the embedded Streamlit dashboard refreshes. Lower = more responsive but higher CPU use. |
| `show_advanced_metrics` | `SHOW_ADVANCED_METRICS` | bool | `true` | When enabled, the dashboard shows detailed per-channel learning rates, thermal parameter history, and prediction accuracy breakdowns. |
| `dashboard_theme` | `DASHBOARD_THEME` | list | `auto` | Visual theme for the dashboard: `auto` (follows system preference), `light`, or `dark`. |

---

## 10. Development Settings

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `enable_dev_api` | `ENABLE_DEV_API` | bool | `false` | Enables an HTTP API endpoint for external tooling. Leave disabled in normal operation. |
| `dev_api_key` | `DEV_API_KEY` | string | `""` | Authentication key for the development API (requires `enable_dev_api`). Leave empty to disable. |
| `log_level` | `LOG_LEVEL` | list | `INFO` | Minimum log severity: `DEBUG` (very verbose), `INFO` (normal), `WARNING`, or `ERROR`. |
| `debug` | `DEBUG` | bool | `false` | When `true`, logs detailed feature vectors, binary search iterations, and model decisions every cycle. Equivalent to setting `log_level=DEBUG`. |

---

## 11. Performance Tuning

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `confidence_threshold` | `CONFIDENCE_THRESHOLD` | float (0.1–10) | `2.0` | Minimum learning confidence before ML controls heating. Below this threshold the system falls back to a simple heat curve. Increase if ML takes over too early; decrease if it stays in fallback too long. |
| `trajectory_steps` | `TRAJECTORY_STEPS` | int (2–12) | `4` | Number of hours the thermal trajectory optimizer plans ahead. 4 = 4-hour planning horizon. More steps = more proactive pre-heating but higher compute. |
| `grace_period_max_minutes` | `GRACE_PERIOD_MAX_MINUTES` | int (10–120) | `15` | Maximum time (minutes) to wait for outlet temperature to stabilise after a blocking event before resuming ML control. |
| `defrost_recovery_grace_minutes` | `DEFROST_RECOVERY_GRACE_MINUTES` | int (15–120) | `45` | Additional grace period (minutes) after a defrost cycle. Defrost chills the slab; this window lets it fully recover before the model learns again. |
| `blocking_poll_interval_seconds` | `BLOCKING_POLL_INTERVAL_SECONDS` | int (30–300) | `60` | How frequently (seconds) blocking sensors are checked while waiting in idle or grace periods. |
| `trend_decay_tau_hours` | `TREND_DECAY_TAU_HOURS` | float (0.1–24) | `1.5` | 🧪 Exponential decay time constant (hours) for the indoor temperature momentum bias injected into trajectory predictions. Higher = longer trend persistence. |

---

## 12. Shadow Mode

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `shadow_mode` | `SHADOW_MODE` | bool | `true` | When `true`, ML Heating calculates and learns but does **not** apply outlet temperatures to the heating system. HA outputs and InfluxDB buckets are isolated with a `_shadow` suffix. **Recommended starting configuration.** |
| `ml_heating_control_entity` | `ML_HEATING_CONTROL_ENTITY_ID` | string | — | An `input_boolean` in HA. ON = ML actively controls heating. OFF = shadow mode. Can be toggled from the HA dashboard without restarting the add-on. |

---

## 13. Thermal Equilibrium Model Parameters

> 🔄 **Auto-Calibrated**: All parameters in this section are automatically determined from your
> historical heating data during initial calibration and then refined through online learning.
> The defaults shown are good starting points for underfloor heating systems with ~150 m² floor
> area but will be overwritten after `--calibrate-physics`. **Do not adjust manually** unless you
> deeply understand the thermal physics and have a specific reason.

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `thermal_time_constant` | `THERMAL_TIME_CONSTANT` | float (1–100) | `4.39` | 🔄 Building thermal response time in hours. Underfloor heating systems typically have values of 3–6 h due to the thermal mass of the slab. |
| `heat_loss_coefficient` | `HEAT_LOSS_COEFFICIENT` | float (0.005–1.2) | `0.12` | 🔄 Rate of heat loss per degree of indoor–outdoor temperature difference (1/hour). Reflects building insulation quality. |
| `outlet_effectiveness` | `OUTLET_EFFECTIVENESS` | float (0.01–2) | `0.95` | 🔄 Efficiency of heat transfer from the outlet water to the indoor air (dimensionless). Near 1.0 is typical for a well-designed system. |
| `outdoor_coupling` | `OUTDOOR_COUPLING` | float (0.1–0.8) | `0.3` | 🔄 Direct influence of outdoor temperature on indoor temperature (via infiltration/ventilation). |
| `thermal_bridge_factor` | `THERMAL_BRIDGE_FACTOR` | float (0.01–0.5) | `0.1` | 🔄 Correction factor for thermal bridging losses in the building envelope. |
| `equilibrium_ratio` | `EQUILIBRIUM_RATIO` | float (0.05–0.9) | `0.17` | 🔄 Internal ratio parameter of the thermal equilibrium equation. |
| `total_conductance` | `TOTAL_CONDUCTANCE` | float (0.1–0.8) | `0.8` | 🔄 Total thermal conductance of the building (1/hour). Reflects combined effect of all building elements. |
| `slab_time_constant_hours` | `SLAB_TIME_CONSTANT_HOURS` | float (0.5–6) | `3.19` | 🔄 First-order thermal time constant of the UFH slab (Estrich) in hours. Models the delay between water circuit and room temperature. |
| `solar_lag_minutes` | `SOLAR_LAG_MINUTES` | float (0–180) | `45.0` | 🔄 Delay in minutes between PV power generation and indoor temperature rise from solar gain. |
| `cloud_correction_min_factor` | `CLOUD_CORRECTION_MIN_FACTOR` | float (0–1) | `0.1` | 🧪 Minimum attenuation factor applied to solar gain estimates on cloudy days. Prevents over-discounting. |

---

## 14. External Heat Source Weights

> 🔄 **Auto-Calibrated**: All weights in this section are automatically learned from data.

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `pv_heat_weight` | `PV_HEAT_WEIGHT` | float (0.0001–0.01) | `0.00207` | 🔄 Indoor temperature rise (°C) per Watt of PV power per model step. Very small values are normal (~0.002). |
| `fireplace_heat_weight` | `FIREPLACE_HEAT_WEIGHT` | float (0.01–10) | `0.387` | 🔄 Temperature contribution of the fireplace to the indoor model when active (°C). |
| `tv_heat_weight` | `TV_HEAT_WEIGHT` | float (0.01–1.5) | `0.35` | 🔄 Temperature contribution of TV/electronics when on (°C). |
| `delta_t_floor` | `DELTA_T_FLOOR` | float (0–10) | `2.3` | 🔄 Minimum heat pump outlet–inlet delta (°C) substituted in the HP channel when the HP is off (measured delta-T < 1°C). Prevents false outlet spikes in passive slab mode. |
| `fp_decay_time_constant` | `FP_DECAY_TIME_CONSTANT` | float (0.1–10) | `3.91` | 🔄 Exponential decay time constant (hours) for fireplace heat after the fireplace turns off. |
| `room_spread_delay_minutes` | `ROOM_SPREAD_DELAY_MINUTES` | float (0–180) | `18.0` | 🔄 Delay (minutes) before fireplace heat is assumed to spread to the rest of the house. |

---

## 15. Adaptive Learning Parameters

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `adaptive_learning_rate` | `ADAPTIVE_LEARNING_RATE` | float (0.001–0.2) | `0.01` | 🧪 Base learning rate for thermal parameter adaptation. Conservative (0.01) is recommended for underfloor heating due to its slow response. |
| `min_learning_rate` | `MIN_LEARNING_RATE` | float (0.0001–0.1) | `0.001` | 🧪 Floor for the adaptive learning rate. Prevents learning from stopping completely. |
| `max_learning_rate` | `MAX_LEARNING_RATE` | float (0.001–0.5) | `0.01` | 🧪 Ceiling for the adaptive learning rate. Prevents over-aggressive updates during unusual conditions. |
| `learning_confidence` | `LEARNING_CONFIDENCE` | float (0.1–10) | `3.0` | 🧪 Starting confidence score for new or reset models (scale 0.1–10.0). The model builds confidence through accurate predictions. |
| `recent_errors_window` | `RECENT_ERRORS_WINDOW` | int (5–50) | `10` | 🧪 Number of recent prediction errors analysed when deciding whether to update parameters. |
| `learning_dead_zone` | `LEARNING_DEAD_ZONE` | float (0–1) | `0.01` | 🧪 Prediction errors smaller than this threshold (°C) are treated as zero and do not trigger parameter updates. Prevents drift from sensor noise. |
| `pv_learning_threshold` | `PV_LEARNING_THRESHOLD` | float (0–5000) | `50` | 🧪 Minimum PV power (W) before the PV heat channel activates for learning. |

---

## 16. Hybrid Learning Strategy

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `hybrid_learning_enabled` | `HYBRID_LEARNING_ENABLED` | bool | `true` | 🧪 Enables intelligent learning-phase classification. Stable periods get full weight; transitions and chaotic periods get reduced or zero weight. |
| `stability_classification_enabled` | `STABILITY_CLASSIFICATION_ENABLED` | bool | `true` | 🧪 Enables automatic classification of learning periods as stable, transitional, or chaotic. |
| `high_confidence_weight` | `HIGH_CONFIDENCE_WEIGHT` | float (0–2) | `1.0` | 🧪 Learning weight during stable, high-confidence periods. 1.0 = full weight. |
| `low_confidence_weight` | `LOW_CONFIDENCE_WEIGHT` | float (0–1) | `0.3` | 🧪 Learning weight during controlled transitions. 0.3 = 30% of normal update rate. |
| `learning_phase_skip_weight` | `LEARNING_PHASE_SKIP_WEIGHT` | float (0–1) | `0.0` | 🧪 Learning weight during chaotic or unreliable periods. 0.0 = skip learning. |

---

## 17. Prediction Metrics Tracking

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `prediction_metrics_enabled` | `PREDICTION_METRICS_ENABLED` | bool | `true` | Enables prediction accuracy tracking published to HA sensors. Keep enabled for performance monitoring. |
| `metrics_window_1h` | `METRICS_WINDOW_1H` | int (6–24) | `12` | 🧪 Number of 5-minute samples in the 1-hour rolling metrics window (12 = 60 min ÷ 5 min). |
| `metrics_window_6h` | `METRICS_WINDOW_6H` | int (36–144) | `72` | 🧪 Number of 5-minute samples in the 6-hour rolling metrics window. |
| `metrics_window_24h` | `METRICS_WINDOW_24H` | int (144–576) | `288` | 🧪 Number of 5-minute samples in the 24-hour rolling metrics window. |
| `prediction_accuracy_threshold` | `PREDICTION_ACCURACY_THRESHOLD` | float (0.1–1) | `0.3` | 🧪 Maximum prediction error (°C) counted as 'accurate'. |
| `mae_entity` | `MAE_ENTITY_ID` | string | `sensor.ml_model_mae` | HA sensor entity where the Mean Absolute Error is published. |
| `rmse_entity` | `RMSE_ENTITY_ID` | string | `sensor.ml_model_rmse` | HA sensor entity where the Root Mean Squared Error is published. |

---

## 18. Trajectory Prediction

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `trajectory_prediction_enabled` | `TRAJECTORY_PREDICTION_ENABLED` | bool | `true` | Enables multi-step thermal trajectory prediction with forecast integration. Strongly recommended for underfloor heating. |
| `weather_forecast_integration` | `WEATHER_FORECAST_INTEGRATION` | bool | `true` | Integrates weather forecast data into trajectory predictions. Essential for proactive pre-heating before cold spells. Requires `openweathermap_temp_entity`. |
| `pv_forecast_integration` | `PV_FORECAST_INTEGRATION` | bool | `true` | Integrates PV forecast data into trajectory predictions. Recommended for PV homes. Requires `pv_forecast_entity`. |
| `solar_correction_enabled` | `SOLAR_CORRECTION_ENABLED` | bool | `true` | Enables the solar correction factor to adjust PV power credited as indoor heat gain. |
| `cloud_cover_correction_enabled` | `CLOUD_COVER_CORRECTION_ENABLED` | bool | `false` | 🧪 Enables cloud-cover attenuation of PV-based solar gain estimates. Experimental — default is `false`. |
| `overshoot_detection_enabled` | `OVERSHOOT_DETECTION_ENABLED` | bool | `true` | Enables trajectory-based overshoot gates that skip corrections when indoor temperature is already moving in the right direction. Prevents overcorrection oscillation. |

---

## 19. Advanced Learning Features

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `enable_multi_lag_learning` | `ENABLE_MULTI_LAG_LEARNING` | bool | `true` | 🧪 Enables time-delayed learning for PV, fireplace, and TV to capture realistic thermal delays (e.g. PV warming peaks 60–90 min after production). |
| `pv_lag_steps` | `PV_LAG_STEPS` | int (1–8) | `4` | 🧪 Number of 30-minute lag steps for PV heat learning. 4 = 2-hour look-back. |
| `fireplace_lag_steps` | `FIREPLACE_LAG_STEPS` | int (1–8) | `4` | 🧪 Number of 30-minute lag steps for fireplace heat learning. |
| `tv_lag_steps` | `TV_LAG_STEPS` | int (1–6) | `2` | 🧪 Number of 30-minute lag steps for TV/electronics heat learning. |
| `enable_seasonal_adaptation` | `ENABLE_SEASONAL_ADAPTATION` | bool | `true` | 🧪 Enables automatic seasonal adjustment of thermal parameters. Eliminates manual recalibration between winter and summer. |
| `seasonal_learning_rate` | `SEASONAL_LEARNING_RATE` | float (0.001–0.1) | `0.01` | 🧪 Learning rate for the seasonal adaptation component. Keep low (0.01) to prevent rapid seasonal drift. |
| `min_seasonal_samples` | `MIN_SEASONAL_SAMPLES` | int (50–500) | `100` | 🧪 Minimum samples required before seasonal adaptation activates. Ensures the model has a solid baseline first. |
| `enable_summer_learning` | `ENABLE_SUMMER_LEARNING` | bool | `true` | 🧪 Enables learning from periods when the HVAC is off (typically summer) for a cleaner external heat source signal. |

---

## 20. Historical Calibration System

> 🧪 These parameters control the physics-based calibration run (`--calibrate-physics`). Default values are suitable for most setups.

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `calibration_baseline_file` | `CALIBRATION_BASELINE_FILE` | string | `/data/calibrated_baseline.json` | Path to the JSON file where calibrated physics baseline parameters are stored after `--calibrate-physics`. |
| `stability_temp_change_threshold` | `STABILITY_TEMP_CHANGE_THRESHOLD` | float (0.05–0.5) | `0.1` | 🧪 Maximum indoor temperature change (°C) for a period to be considered 'stable' during calibration. |
| `min_stable_period_minutes` | `MIN_STABLE_PERIOD_MINUTES` | int (15–120) | `30` | 🧪 Minimum duration (minutes) a stable period must last to be included in calibration data. |
| `optimization_method` | `OPTIMIZATION_METHOD` | string | `L-BFGS-B` | 🧪 Scipy optimisation method for physics calibration. `L-BFGS-B` handles bounded problems efficiently. |
| `pv_calibration_indoor_ceiling` | `PV_CALIBRATION_INDOOR_CEILING` | float (18–30) | `23.0` | 🧪 Indoor temperatures at or above this value are excluded from PV calibration (automated blinds likely closed). Set just below your blind trigger threshold. |

---

## 21. Delta Temperature Forecast Calibration

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `enable_delta_forecast_calibration` | `ENABLE_DELTA_FORECAST_CALIBRATION` | bool | `true` | 🧪 Enables local calibration of weather forecast data using measured offsets. Corrects systematic biases between the weather station and your location. |
| `delta_calibration_max_offset` | `DELTA_CALIBRATION_MAX_OFFSET` | float (1–20) | `10.0` | 🧪 Maximum allowed temperature offset (°C) during delta calibration. Safety cap preventing unrealistic corrections. |

---

## 22. Learning History Sizes

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `max_prediction_history` | `MAX_PREDICTION_HISTORY` | int (100–2000) | `700` | 🧪 Maximum number of prediction records kept in memory for accuracy analysis. |
| `max_parameter_history` | `MAX_PARAMETER_HISTORY` | int (100–2000) | `700` | 🧪 Maximum number of parameter update records kept in memory for trend analysis. |

---

## 23. Indoor Trend Protection

> 🧪 These parameters prevent HLC/OE drift when the indoor temperature moves due to setpoint changes rather than heating behaviour.

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `indoor_cooling_trend_threshold` | `INDOOR_COOLING_TREND_THRESHOLD` | float (-1–0) | `-0.05` | 🧪 Rate-of-change (°C/step) below which indoor temp is classified as 'cooling' due to a setpoint reduction. Learning is dampened. |
| `indoor_cooling_damping_factor` | `INDOOR_COOLING_DAMPING_FACTOR` | float (0–1) | `0.3` | 🧪 Multiplier for learning updates during a detected cooling trend. 0.3 = 30% of normal rate. |
| `indoor_warming_trend_threshold` | `INDOOR_WARMING_TREND_THRESHOLD` | float (0–1) | `0.10` | 🧪 Rate-of-change (°C/step) above which indoor temp is classified as 'warming' due to a setpoint increase. |
| `indoor_warming_damping_factor` | `INDOOR_WARMING_DAMPING_FACTOR` | float (0–1) | `0.3` | 🧪 Multiplier for learning updates during a detected warming trend. 0.3 = 30% of normal rate. |

---

## 24. Heat Source Channel Architecture

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `enable_heat_source_channels` | `ENABLE_HEAT_SOURCE_CHANNELS` | bool | `true` | 🧪 Enables decomposed heat-source learning with independent channels for HP, solar, fireplace, and TV. Prevents cross-contamination. Strongly recommended. |
| `enable_mixed_source_attribution` | `ENABLE_MIXED_SOURCE_ATTRIBUTION` | bool | `false` | 🧪 Enables proportional attribution of heat between simultaneously active sources. Experimental. |
| `pv_room_decay_multiplier` | `PV_ROOM_DECAY_MULTIPLIER` | float (0.1–10) | `2.0` | 🧪 Multiplier × `thermal_time_constant` = hours HP learning is frozen after PV drops below threshold. |
| `decay_cancel_margin` | `DECAY_CANCEL_MARGIN` | float (0–2) | `0.1` | 🧪 Indoor temp margin (°C) above target at which PV/FP decay is cancelled early, indicating residual heat has dissipated. |

---

## 25. Electricity Price Optimisation (Tibber)

Requires the Tibber integration in Home Assistant.

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `electricity_price_enabled` | `ELECTRICITY_PRICE_ENABLED` | bool | `false` | Master switch for Tibber price integration. When enabled, the indoor target shifts up during cheap periods and down during expensive periods. |
| `price_cheap_percentile` | `PRICE_CHEAP_PERCENTILE` | int (10–50) | `33` | Prices below this percentile of today's distribution are 'cheap' and trigger a positive setpoint offset. |
| `price_expensive_percentile` | `PRICE_EXPENSIVE_PERCENTILE` | int (50–90) | `67` | Prices above this percentile are 'expensive' and trigger a negative setpoint offset. |
| `price_target_offset` | `PRICE_TARGET_OFFSET` | float (0–1) | `0.2` | How much (°C) the indoor target shifts up during cheap and down during expensive periods. |
| `price_expensive_overshoot` | `PRICE_EXPENSIVE_OVERSHOOT` | float (0–1) | `0.2` | Tighter future overshoot threshold (°C) during expensive periods, reducing outlet temperature earlier. |
| `price_cache_refresh_minutes` | `PRICE_CACHE_REFRESH_MINUTES` | int (15–1440) | `60` | How often (minutes) prices are re-fetched from Tibber. Also refreshes after 13:00 for tomorrow's prices. |

---

## 26. PV Surplus Optimisation

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `pv_surplus_cheap_enabled` | `PV_SURPLUS_CHEAP_ENABLED` | bool | `false` | When enabled, the indoor target is raised by `price_target_offset` whenever PV production exceeds `pv_surplus_cheap_threshold_w`, independently of Tibber. |
| `pv_surplus_cheap_threshold_w` | `PV_SURPLUS_CHEAP_THRESHOLD_W` | int (100–20000) | `3000` | PV power threshold (W) above which surplus heating activates. Roughly 20% of your rated capacity (e.g. 3000 W for a 15 kWp system). |

---

## 27. Setpoint Stability

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `min_setpoint_hold_cycles` | `MIN_SETPOINT_HOLD_CYCLES` | int (0–20) | `4` | Minimum cycles a computed outlet setpoint is held before the optimiser produces a new value. Defaults to `trajectory_steps` so the setpoint is stable for the full planning horizon. Set to 0 to recompute every cycle. |

---

## 28. Dynamic Trajectory Scaling

> 🧪 Scales the planning horizon dynamically based on current PV production and time of day. Disabled by default.

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `pv_traj_scaling_enabled` | `PV_TRAJ_SCALING_ENABLED` | bool | `false` | 🧪 When enabled, `trajectory_steps` and the setpoint hold duration are recomputed each cycle from actual PV production and time of day. |
| `pv_traj_system_kwp` | `PV_TRAJ_SYSTEM_KWP` | float (1–100) | `10.0` | 🧪 Your PV installation's rated peak capacity in kWp (e.g. `15.0` for 15 kWp). Used to normalise PV power to a 0–1 ratio. |
| `pv_traj_min_steps` | `PV_TRAJ_MIN_STEPS` | int (2–12) | `2` | 🧪 Minimum planning horizon in hours at zero PV (night / overcast). Must be ≤ `pv_traj_max_steps`. |
| `pv_traj_max_steps` | `PV_TRAJ_MAX_STEPS` | int (2–12) | `12` | 🧪 Maximum planning horizon in hours at full rated PV output. |
| `pv_traj_morning_factor` | `PV_TRAJ_MORNING_FACTOR` | float (0–1) | `0.5` | 🧪 PV ratio multiplier during morning ramp-up (06:00–10:59). Moderate commitment while the sun is still rising. |
| `pv_traj_midday_factor` | `PV_TRAJ_MIDDAY_FACTOR` | float (0–1) | `1.0` | 🧪 PV ratio multiplier during peak production (11:00–14:59). Full planning horizon allowed. |
| `pv_traj_afternoon_factor` | `PV_TRAJ_AFTERNOON_FACTOR` | float (0–1) | `0.75` | 🧪 PV ratio multiplier during afternoon decline (15:00–18:59). Slightly shorter horizon. |
| `pv_traj_night_factor` | `PV_TRAJ_NIGHT_FACTOR` | float (0–1) | `0.0` | 🧪 PV ratio multiplier at night (19:00–05:59). Forces minimum trajectory steps. |

---

## 29. Seasonal KWP Scaling

> 🧪 Normalises PV production relative to the summer-solstice maximum so a clear winter day correctly maps to `pv_ratio=1.0`.

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `pv_traj_seasonal_scaling_enabled` | `PV_TRAJ_SEASONAL_SCALING_ENABLED` | bool | `false` | 🧪 When enabled, the effective PV peak is scaled by a seasonal factor derived from solar declination at your latitude. |
| `pv_traj_latitude` | `PV_TRAJ_LATITUDE` | float (-90–90) | `51.0` | 🧪 Geographic latitude of your PV installation in decimal degrees, North positive. Examples: 48.0 Munich, 51.5 Berlin, 47.8 Zurich, 52.4 Amsterdam. |
| `pv_traj_seasonal_min_factor` | `PV_TRAJ_SEASONAL_MIN_FACTOR` | float (0–1) | `0.1` | 🧪 Floor value for the seasonal scaling factor. Prevents near-zero denominators at high latitudes in deep winter. |

---

## 30. Outlet Smoothing

| Parameter | Env Var | Type | Default | Description |
|---|---|---|---|---|
| `outlet_smoothing_alpha` | `OUTLET_SMOOTHING_ALPHA` | float (0–1) | `0.3` | Exponential moving average (EMA) smoothing factor for the outlet temperature setpoint. 0.0 = maximum smoothing; 1.0 = no smoothing (instant response). |
| `outlet_smoothing_bypass` | `OUTLET_SMOOTHING_BYPASS` | float (0.5–5) | `2.0` | When the required setpoint change exceeds this threshold (°C), EMA smoothing is bypassed for an immediate response (e.g. after defrost recovery). |

---

## See Also

- [README.md](../README.md) — Installation, architecture overview, and configuration quick-start
- [docs/SHADOW_MODE_USER_GUIDE.md](SHADOW_MODE_USER_GUIDE.md) — How to set up and use shadow mode safely
- [docs/THERMAL_PARAMETER_REFERENCE.md](THERMAL_PARAMETER_REFERENCE.md) — Deep dive into the thermal model physics
- [docs/PRICE_OPTIMIZATION_INTEGRATION.md](PRICE_OPTIMIZATION_INTEGRATION.md) — Tibber electricity price integration setup
- [.env_sample](../.env_sample) — Annotated environment variable template for standalone deployments
