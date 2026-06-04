"""
Centralized Configuration for the ML Heating Script.

This file consolidates all user-configurable parameters for the application.
It uses the `dotenv` library to load settings from a `.env` file, allowing for
easy management of sensitive information like API keys and environment-specific
settings without hardcoding them into the source.

The configuration is organized into logical sections:
- API Credentials and Endpoints
- File Paths for persistent data
- Model & History Parameters for feature engineering
- Home Assistant Entity IDs (Core, Blocking, and Additional Sensors)
- Tuning & Debug Parameters for runtime behavior
- Metrics Entity IDs for performance monitoring

It is crucial to create a `.env` file and customize these settings, especially
the `HASS_URL`, `HASS_TOKEN`, and all `*_ENTITY_ID` variables, to match your
specific Home Assistant setup.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists.
load_dotenv()

# --- API Credentials and Endpoints ---
# This section contains the connection details for external services.


# Detect if running in Home Assistant addon environment
def _is_addon_environment():
    """Detect if running in Home Assistant addon environment."""
    # Only treat as addon if SUPERVISOR_TOKEN is set AND HASS_TOKEN is not
    # explicitly provided. An explicit HASS_TOKEN means the user configured
    # standalone Docker mode, even if SUPERVISOR_TOKEN happens to be present.
    return (
        os.getenv("SUPERVISOR_TOKEN") is not None
        and os.getenv("HASS_TOKEN") is None
    )


# Detect if running in a notebook/analysis environment
def _is_notebook_environment():
    """Detect if running in a Jupyter notebook or analysis script."""
    # Check for common notebook indicators or explicit env var
    return (
        os.getenv("ML_HEATING_ENV") == "notebook"
        or "ipykernel" in str(os.environ.get("modules", ""))
    )


# For Home Assistant addon, uses internal supervisor API;
# for standalone, uses .env
if _is_addon_environment():
    HASS_URL: str = os.getenv("HASS_URL", "http://supervisor/core")
    HASS_TOKEN: str = os.getenv("SUPERVISOR_TOKEN", "").strip()
else:
    # Default to localhost, but allow override for notebooks
    default_url = "http://localhost:8123"
    if _is_notebook_environment():
        # In notebooks, we might want to fail gracefully or warn if not set
        pass

    HASS_URL = os.getenv("HASS_URL", default_url)
    HASS_TOKEN = os.getenv("HASS_TOKEN", "").strip()


HASS_HEADERS: dict[str, str] = {
    "Authorization": f"Bearer {HASS_TOKEN}",
    "Content-Type": "application/json",
}
INFLUX_URL: str = os.getenv("INFLUX_URL", "https://influxdb.erbehome.de")
INFLUX_TOKEN: str = os.getenv("INFLUX_TOKEN", "")
INFLUX_ORG: str = os.getenv("INFLUX_ORG", "erbehome")
INFLUX_BUCKET: str = os.getenv("INFLUX_BUCKET", "home_assistant/autogen")

INFLUX_FEATURES_BUCKET: str = os.getenv(
    "INFLUX_FEATURES_BUCKET", "ml_heating_features"
)

# --- File Paths ---
UNIFIED_STATE_FILE: str = os.getenv(
    "UNIFIED_STATE_FILE", "/opt/ml_heating/unified_thermal_state.json"
)
UNIFIED_STATE_FILE_COOLING: str = os.getenv(
    "UNIFIED_STATE_FILE_COOLING",
    "/opt/ml_heating/unified_thermal_state_cooling.json",
)
CALIBRATION_BASELINE_FILE: str = os.getenv(
    "CALIBRATION_BASELINE_FILE", "/opt/ml_heating/calibrated_baseline.json"
)

# --- Model & History Parameters ---
# These parameters control time windows for feature creation and prediction.
# HISTORY_STEPS: Number of historical time slices to use for features.
# HISTORY_STEP_MINUTES: The interval in minutes between each history step.
# PREDICTION_HORIZON_STEPS: How many steps into the future the model
# predicts.
HISTORY_STEPS: int = int(os.getenv("HISTORY_STEPS", "6"))
HISTORY_STEP_MINUTES: int = int(os.getenv("HISTORY_STEP_MINUTES", "10"))
# Prediction horizon used during calibration to determine future target.
PREDICTION_HORIZON_STEPS: int = int(
    os.getenv("PREDICTION_HORIZON_STEPS", "24")
)
# The number of hours of historical data to use for initial training.
TRAINING_LOOKBACK_HOURS: int = int(os.getenv("TRAINING_LOOKBACK_HOURS", "168"))

# Training data source for calibration: "influx", "ha_history", or "auto"
# "auto" tries InfluxDB first, falls back to HA history API.
TRAINING_DATA_SOURCE: str = os.getenv("TRAINING_DATA_SOURCE", "auto")

# --- Core Entity IDs ---
# These are the most critical entities for the script's operation.
# **It is essential to update these to match your Home Assistant setup.**
# TARGET_INDOOR_TEMP_ENTITY_ID: The desired indoor temperature (e.g., from
# a thermostat).
# INDOOR_TEMP_ENTITY_ID: The current actual indoor temperature.
# ACTUAL_OUTLET_TEMP_ENTITY_ID: The current measured boiler outlet
# temperature.
# TARGET_OUTLET_TEMP_ENTITY_ID: The sensor this script will create/update
# with its calculated temperature.
TARGET_INDOOR_TEMP_ENTITY_ID: str = os.getenv(
    "TARGET_INDOOR_TEMP_ENTITY_ID",
    "input_number.hp_auto_correct_target",
)
# Separate target temperature entity for cooling mode.
# If set, cooling mode reads its target from this entity instead of
# TARGET_INDOOR_TEMP_ENTITY_ID.  When empty, the heating target is used.
TARGET_INDOOR_TEMP_COOLING_ENTITY_ID: str = os.getenv(
    "TARGET_INDOOR_TEMP_COOLING_ENTITY_ID", ""
)
INDOOR_TEMP_ENTITY_ID: str = os.getenv(
    "INDOOR_TEMP_ENTITY_ID", "sensor.kuche_temperatur"
)
ACTUAL_OUTLET_TEMP_ENTITY_ID: str = os.getenv(
    "ACTUAL_OUTLET_TEMP_ENTITY_ID", "sensor.hp_outlet_temp"
)
INLET_TEMP_ENTITY_ID: str = os.getenv(
    "INLET_TEMP_ENTITY_ID", "sensor.hp_inlet_temp"
)
FLOW_RATE_ENTITY_ID: str = os.getenv(
    "FLOW_RATE_ENTITY_ID", "sensor.hp_current_flow_rate"
)
POWER_CONSUMPTION_ENTITY_ID: str = os.getenv(
    "POWER_CONSUMPTION_ENTITY_ID", "sensor.power_wp"
)
SPECIFIC_HEAT_CAPACITY: float = float(
    os.getenv("SPECIFIC_HEAT_CAPACITY", "4.186")
)
# The entity the script will write the final calculated temperature to.
TARGET_OUTLET_TEMP_ENTITY_ID: str = os.getenv(
    "TARGET_OUTLET_TEMP_ENTITY_ID", "sensor.ml_vorlauftemperatur"
)
# The entity to read what outlet temperature was actually set (for learning).
# In active mode: should match TARGET_OUTLET_TEMP_ENTITY_ID
# In shadow mode: reads the heat curve's setting
ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID: str = os.getenv(
    "ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID", "sensor.hp_target_temp_circuit1"
)

# --- Blocking & Status Entity IDs ---
# These binary sensors can pause the script's operation. For example, if the
# system is busy with Domestic Hot Water (DHW) heating, the script will wait.
DHW_STATUS_ENTITY_ID: str = os.getenv(
    "DHW_STATUS_ENTITY_ID", "binary_sensor.hp_dhw_heating_status"
)
DEFROST_STATUS_ENTITY_ID: str = os.getenv(
    "DEFROST_STATUS_ENTITY_ID", "binary_sensor.hp_defrosting_status"
)
DISINFECTION_STATUS_ENTITY_ID: str = os.getenv(
    "DISINFECTION_STATUS_ENTITY_ID",
    "binary_sensor.hp_dhw_tank_disinfection_status",
)
DHW_BOOST_HEATER_STATUS_ENTITY_ID: str = os.getenv(
    "DHW_BOOST_HEATER_STATUS_ENTITY_ID",
    "binary_sensor.hp_dhw_boost_heater_status",
)

# --- Additional Sensor IDs ---
# These entities provide extra context to the model as features. The more
# relevant data the model has, the better its predictions can be.
TV_STATUS_ENTITY_ID: str = os.getenv(
    "TV_STATUS_ENTITY_ID", "input_boolean.fernseher"
)
OUTDOOR_TEMP_ENTITY_ID: str = os.getenv(
    "OUTDOOR_TEMP_ENTITY_ID", "sensor.thermometer_waermepume_kompensiert"
)
PV_POWER_ENTITY_ID: str = os.getenv(
    "PV_POWER_ENTITY_ID", "sensor.power_pv"
)
WIND_SPEED_ENTITY_ID: str = os.getenv(
    "WIND_SPEED_ENTITY_ID", "sensor.wind_speed"
)
SOLAR_CORRECTION_ENTITY_ID: str = os.getenv(
    "SOLAR_CORRECTION_ENTITY_ID", "input_number.ml_heating_solar_correction"
)
SOLAR_CORRECTION_DEFAULT_PERCENT: float = float(
    os.getenv("SOLAR_CORRECTION_DEFAULT_PERCENT", "100.0")
)
SOLAR_CORRECTION_MIN_PERCENT: float = float(
    os.getenv("SOLAR_CORRECTION_MIN_PERCENT", "0.0")
)
SOLAR_CORRECTION_MAX_PERCENT: float = float(
    os.getenv("SOLAR_CORRECTION_MAX_PERCENT", "100.0")
)

# Living room temperature sensor (used for fireplace analysis only)
LIVING_ROOM_TEMP_ENTITY_ID: str = os.getenv(
    "LIVING_ROOM_TEMP_ENTITY_ID", "sensor.living_room_temperature"
)

# PV forecast sensor (HA attributes 'watts' available in 15-min steps)
PV_FORECAST_ENTITY_ID: str = os.getenv(
    "PV_FORECAST_ENTITY_ID", "sensor.energy_production_today_4"
)
HEATING_STATUS_ENTITY_ID: str = os.getenv(
    "HEATING_STATUS_ENTITY_ID", "climate.heizung_2"
)
OPENWEATHERMAP_TEMP_ENTITY_ID: str = os.getenv(
    "OPENWEATHERMAP_TEMP_ENTITY_ID", "sensor.openweathermap_temperature"
)
AVG_OTHER_ROOMS_TEMP_ENTITY_ID: str = os.getenv(
    "AVG_OTHER_ROOMS_TEMP_ENTITY_ID", "sensor.avg_other_rooms_temp"
)
FIREPLACE_STATUS_ENTITY_ID: str = os.getenv(
    "FIREPLACE_STATUS_ENTITY_ID", "binary_sensor.fireplace_active"
)

# --- Electricity Price Integration ---
# Prices are fetched via the tibber.get_prices HA service call (PriceOptimizer).
ELECTRICITY_PRICE_ENABLED: bool = (
    os.getenv("ELECTRICITY_PRICE_ENABLED", "false").lower() == "true"
)
# Percentile thresholds for classifying price levels from daily prices.
PRICE_CHEAP_PERCENTILE: float = float(
    os.getenv("PRICE_CHEAP_PERCENTILE", "33")
)
PRICE_EXPENSIVE_PERCENTILE: float = float(
    os.getenv("PRICE_EXPENSIVE_PERCENTILE", "67")
)
# Target temperature offset (°C) applied during cheap/expensive periods.
PRICE_TARGET_OFFSET: float = float(
    os.getenv("PRICE_TARGET_OFFSET", "0.2")
)
# Future trajectory overshoot threshold during expensive periods (°C above
# target).  Tighter than the normal 0.5°C to reduce outlet earlier.
PRICE_EXPENSIVE_OVERSHOOT: float = float(
    os.getenv("PRICE_EXPENSIVE_OVERSHOOT", "0.2")
)
# How often (minutes) to re-fetch prices from the Tibber service.
# Also refreshes after 13:00 if tomorrow's prices are not yet cached.
PRICE_CACHE_REFRESH_MINUTES: int = int(
    os.getenv("PRICE_CACHE_REFRESH_MINUTES", "60")
)
# PV surplus cheap override: when current PV power (W) exceeds this threshold
# the target is shifted by +PRICE_TARGET_OFFSET (same as CHEAP price), even
# without a Tibber price feed.  Set to -1 or 0 to disable.
PV_SURPLUS_CHEAP_ENABLED: bool = (
    os.getenv("PV_SURPLUS_CHEAP_ENABLED", "false").lower() == "true"
)
PV_SURPLUS_CHEAP_THRESHOLD_W: int = int(
    os.getenv("PV_SURPLUS_CHEAP_THRESHOLD_W", "3000")
)
# Blend zone width (W) below PV_SURPLUS_CHEAP_THRESHOLD_W for soft-ramp.
# In the range [threshold - ramp_w, threshold] the CHEAP offset scales
# linearly from 0 to full.  Defaults to threshold (i.e. ramp starts at 0 W).
PV_SURPLUS_CHEAP_RAMP_W: float = float(
    os.getenv("PV_SURPLUS_CHEAP_RAMP_W", str(PV_SURPLUS_CHEAP_THRESHOLD_W))
)

# --- Forecast-Driven Trajectory Mode ---
# When PV_TRAJ_FORECAST_MODE_ENABLED is true, trajectory steps are derived
# from remaining PV forecast hours (consecutive hours until PV drops to
# PV_TRAJ_ZERO_W) plus PV_TRAJ_MIN_STEPS reserved for the post-sunset
# period: steps = clamp(remaining_pv_hours + MIN_STEPS, MIN, MAX).
PV_TRAJ_MIN_STEPS: int = int(os.getenv("PV_TRAJ_MIN_STEPS", "2"))
PV_TRAJ_MAX_STEPS: int = int(os.getenv("PV_TRAJ_MAX_STEPS", "12"))
PV_TRAJ_FORECAST_MODE_ENABLED: bool = (
    os.getenv("PV_TRAJ_FORECAST_MODE_ENABLED", "false").lower() == "true"
)
# Minimum current PV electrical power [W] required to activate the forecast
# trajectory.  Below this threshold the mode is inactive → PV_TRAJ_MIN_STEPS.
PV_TRAJ_THRESHOLD_W: float = float(os.getenv("PV_TRAJ_THRESHOLD_W", "3000.0"))
# PV power [W] at or below which a forecast slot is treated as "night" / PV≈0.
PV_TRAJ_ZERO_W: float = float(os.getenv("PV_TRAJ_ZERO_W", "50.0"))
# When true (default), suppress the electricity price target-temperature offset
# while the forecast trajectory is active so it does not interfere with the
# pre-heat plan.
PV_TRAJ_DISABLE_PRICE_IN_FORECAST_MODE: bool = (
    os.getenv("PV_TRAJ_DISABLE_PRICE_IN_FORECAST_MODE", "true").lower() == "true"
)
# When true (default), a temporary drop of pv_now below PV_TRAJ_THRESHOLD_W
# (e.g. passing rain cloud) does not immediately collapse the trajectory to
# PV_TRAJ_MIN_STEPS.  Instead, the forecast is consulted: if at least
# PV_TRAJ_MIN_STEPS forecast hours remain above PV_TRAJ_THRESHOLD_W the
# algorithm continues with normal step counting.  Set to false to require
# current PV above the threshold at all times.
PV_TRAJ_FORECAST_RESCUE_ENABLED: bool = (
    os.getenv("PV_TRAJ_FORECAST_RESCUE_ENABLED", "true").lower() == "true"
)
# Minimum forecast hours above PV_TRAJ_THRESHOLD_W required for the forecast
# rescue to fire.  Previously hardcoded to PV_TRAJ_MIN_STEPS, which caused the
# rescue to fail during gradual late-afternoon PV decline.  Default: 1 — even a
# single future hour above threshold keeps the trajectory active.
PV_TRAJ_RESCUE_MIN_HOURS: int = int(
    os.getenv("PV_TRAJ_RESCUE_MIN_HOURS", "1")
)
# When true, skip the overshoot/undershoot outlet-temperature correction only
# while forecast mode is active AND dynamic trajectory steps are above the
# minimum floor (TRAJECTORY_STEPS > PV_TRAJ_MIN_STEPS). At the minimum floor,
# correction is re-enabled. Default: false (correction active).
PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION: bool = (
    os.getenv("PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION", "false").lower() == "true"
)

# --- Output Sensors ---
FEATURES_ENTITY_ID: str = os.getenv(
    "FEATURES_ENTITY_ID", "sensor.ml_heating_features"
)

# --- Tuning & Debug Parameters ---
# DEBUG: Set to "1" to enable verbose logging for development.
# CONFIDENCE_THRESHOLD: A critical tuning parameter. If the model's
#   normalized confidence score (0-1) drops below this threshold, the system
#   will fall back to the baseline temperature. A lower threshold means the
#   system is more tolerant of model uncertainty.
# HEAT_BALANCE_MODE: Enable the intelligent heat balance controller that uses
#   trajectory prediction instead of smoothing. Uses 3-phase control:
#   Charging (>0.5°C error), Balancing (0.2-0.5°C), Maintenance (<0.2°C).
# TRAJECTORY_STEPS: Number of hours to predict in trajectory optimization.
# CYCLE_INTERVAL_MINUTES: The time in minutes between each full cycle of
#   learning and prediction. A longer interval (e.g., 10-15 mins) provides a
#   clearer learning signal, while a shorter one is more responsive.
# MAX_TEMP_CHANGE_PER_CYCLE: The maximum allowable integer change (in degrees)
#   for the outlet temperature setpoint in a single cycle. This prevents
#   abrupt changes and is required as the heatpump only accepts full degrees.
DEBUG: bool = os.getenv("DEBUG", "0") == "1"
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.2"))
TRAJECTORY_STEPS: int = int(os.getenv("TRAJECTORY_STEPS", "4"))
CYCLE_INTERVAL_MINUTES: int = int(os.getenv("CYCLE_INTERVAL_MINUTES", "10"))
MAX_TEMP_CHANGE_PER_CYCLE: int = int(
    os.getenv("MAX_TEMP_CHANGE_PER_CYCLE", "2")
)
# Minimum number of cycles a computed setpoint is held before the optimizer
# is allowed to produce a different value.  Defaults to TRAJECTORY_STEPS so
# the setpoint is stable for at least the full planning horizon.
# Set to 0 to disable (recompute every cycle).
MIN_SETPOINT_HOLD_CYCLES: int = max(
    0, int(os.getenv("MIN_SETPOINT_HOLD_CYCLES", str(TRAJECTORY_STEPS)))
)
TREND_DECAY_TAU_HOURS: float = max(
    0.1, float(os.getenv("TREND_DECAY_TAU_HOURS", "1.5"))
)
OUTLET_SMOOTHING_ALPHA: float = float(
    os.getenv("OUTLET_SMOOTHING_ALPHA", "0.3")
)
OUTLET_SMOOTHING_BYPASS: float = float(
    os.getenv("OUTLET_SMOOTHING_BYPASS", "2.0")
)
# Heating correction algorithm selector.
# "legacy"  — current empirical formula (default, preserves existing behaviour).
# "physics" — horizon-aware Newton step ΔT = ε / S(t_worst), where
#             S(t) = [η/(η+U)] × [1 − exp(−t/τ)] is evaluated at the time of
#             the worst trajectory violation (recommended after calibration).
# "ml"      — LightGBM regressor (requires calibration run, future feature).
HEATING_CORRECTION_MODE: str = os.getenv("HEATING_CORRECTION_MODE", "legacy")
# Maximum minutes to wait during the grace period after blocking ends.
GRACE_PERIOD_MAX_MINUTES: int = int(
    os.getenv("GRACE_PERIOD_MAX_MINUTES", "15")
)
# Extra grace period after HP defrost cycles.  Defrost steals heat from the
# slab; the HP then re-heats the slab before the room reaches true steady
# state.  Periods inside this window are excluded from OE / HLC calibration.
DEFROST_RECOVERY_GRACE_MINUTES: int = int(
    os.getenv("DEFROST_RECOVERY_GRACE_MINUTES", "45")
)

# How often (seconds) to poll blocking entities during the idle period.
# A value of 60 means we check the blocking state once per minute.
BLOCKING_POLL_INTERVAL_SECONDS: int = int(
    os.getenv("BLOCKING_POLL_INTERVAL_SECONDS", "60")
)

# --- Metrics Entity IDs ---
# These entities are created in Home Assistant to allow real-time monitoring
# of the model's performance and health.
# Note: Model confidence is now provided via sensor.ml_heating_learning.state
MAE_ENTITY_ID: str = os.getenv("MAE_ENTITY_ID", "sensor.ml_model_mae")
RMSE_ENTITY_ID: str = os.getenv("RMSE_ENTITY_ID", "sensor.ml_model_rmse")

# --- Multi-Lag Learning Configuration ---
# Enable time-delayed learning for external heat sources (PV, fireplace, TV)
# to capture realistic time delays (e.g., PV warming peaks 60-90min
# after production)
ENABLE_MULTI_LAG_LEARNING: bool = (
    os.getenv("ENABLE_MULTI_LAG_LEARNING", "true").lower() == "true"
)
PV_LAG_STEPS: int = int(os.getenv("PV_LAG_STEPS", "4"))
FIREPLACE_LAG_STEPS: int = int(os.getenv("FIREPLACE_LAG_STEPS", "4"))
TV_LAG_STEPS: int = int(os.getenv("TV_LAG_STEPS", "2"))

# --- Seasonal Adaptation Configuration ---
# Enable automatic seasonal learning to eliminate need for recalibration
# between winter and summer. Uses cos/sin modulation.
ENABLE_SEASONAL_ADAPTATION: bool = (
    os.getenv("ENABLE_SEASONAL_ADAPTATION", "true").lower() == "true"
)
SEASONAL_LEARNING_RATE: float = float(
    os.getenv("SEASONAL_LEARNING_RATE", "0.01")
)
MIN_SEASONAL_SAMPLES: int = int(os.getenv("MIN_SEASONAL_SAMPLES", "100"))

# --- Summer Learning Configuration ---
# Enable learning from periods when HVAC is off (typically summer) for
# cleaner signal of external source effects
ENABLE_SUMMER_LEARNING: bool = (
    os.getenv("ENABLE_SUMMER_LEARNING", "true").lower() == "true"
)

# --- Shadow Mode Configuration ---
# SHADOW_MODE: When true, ML runs in observation mode without affecting heating
# - ML predictions are calculated but not sent to heating system
# - No HA sensors are updated (target temp, confidence, MAE, RMSE, state)
# - System learns from heat curve's actual control decisions
# - Performance comparison logging between ML vs heat curve
SHADOW_MODE: bool = os.getenv("SHADOW_MODE", "false").lower() == "true"

# --- ML Heating Control Entity ---
# ML_HEATING_CONTROL_ENTITY_ID: HA input_boolean to enable/disable ML control
# - When ON: ML actively controls heating (Active Mode)
# - When OFF: Shadow mode (ML observes only, doesn't control)
# - Note: SHADOW_MODE environment variable overrides this setting
ML_HEATING_CONTROL_ENTITY_ID: str = os.getenv(
    "ML_HEATING_CONTROL_ENTITY_ID",
    "input_boolean.ml_heating"
)

# --- Thermal Equilibrium Model Parameters ---
# DEPRECATED: These parameters are now managed by the unified
# ThermalParameterManager and are sourced from src/thermal_config.py.
# Environment variables set for these will be loaded by the manager.

# Adaptive Learning Parameters (Priority 3 - Advanced Tuning)
# Error analysis window size
RECENT_ERRORS_WINDOW: int = int(os.getenv("RECENT_ERRORS_WINDOW", "10"))
LEARNING_DEAD_ZONE: float = float(os.getenv("LEARNING_DEAD_ZONE", "0.01"))
PV_LEARNING_THRESHOLD: float = float(os.getenv("PV_LEARNING_THRESHOLD", "50"))

# --- Hybrid Learning Strategy (Phase 2 Enhancement) ---
# Enable intelligent learning phase classification with weighted periods
HYBRID_LEARNING_ENABLED: bool = (
    os.getenv("HYBRID_LEARNING_ENABLED", "true").lower() == "true"
)
STABILITY_CLASSIFICATION_ENABLED: bool = (
    os.getenv("STABILITY_CLASSIFICATION_ENABLED", "true").lower() == "true"
)
HIGH_CONFIDENCE_WEIGHT: float = float(
    os.getenv("HIGH_CONFIDENCE_WEIGHT", "1.0")
)
LOW_CONFIDENCE_WEIGHT: float = float(
    os.getenv("LOW_CONFIDENCE_WEIGHT", "0.3")
)
LEARNING_PHASE_SKIP_WEIGHT: float = float(
    os.getenv("LEARNING_PHASE_SKIP_WEIGHT", "0.0")
)

# --- MAE/RMSE Tracking System (Phase 2 Enhancement) ---
# Enable comprehensive prediction accuracy tracking
PREDICTION_METRICS_ENABLED: bool = (
    os.getenv("PREDICTION_METRICS_ENABLED", "true").lower() == "true"
)
METRICS_WINDOW_1H: int = int(os.getenv("METRICS_WINDOW_1H", "12"))
METRICS_WINDOW_6H: int = int(os.getenv("METRICS_WINDOW_6H", "72"))
METRICS_WINDOW_24H: int = int(os.getenv("METRICS_WINDOW_24H", "288"))
PREDICTION_ACCURACY_THRESHOLD: float = float(
    os.getenv("PREDICTION_ACCURACY_THRESHOLD", "0.3")
)

# --- Trajectory Prediction Enhancement (Phase 2) ---
# Advanced trajectory prediction with forecast integration
TRAJECTORY_PREDICTION_ENABLED: bool = (
    os.getenv("TRAJECTORY_PREDICTION_ENABLED", "true").lower() == "true"
)
WEATHER_FORECAST_INTEGRATION: bool = (
    os.getenv("WEATHER_FORECAST_INTEGRATION", "true").lower() == "true"
)
PV_FORECAST_INTEGRATION: bool = (
    os.getenv("PV_FORECAST_INTEGRATION", "true").lower() == "true"
)
SOLAR_CORRECTION_ENABLED: bool = (
    os.getenv("SOLAR_CORRECTION_ENABLED", "true").lower() == "true"
)
CLOUD_COVER_CORRECTION_ENABLED: bool = (
    os.getenv("CLOUD_COVER_CORRECTION_ENABLED", "false").lower() == "true"
)
OVERSHOOT_DETECTION_ENABLED: bool = (
    os.getenv("OVERSHOOT_DETECTION_ENABLED", "true").lower() == "true"
)

# --- Historical Calibration System (Phase 0) ---
# Physics-based historical parameter optimization
STABILITY_TEMP_CHANGE_THRESHOLD: float = float(
    os.getenv("STABILITY_TEMP_CHANGE_THRESHOLD", "0.1")
)
MIN_STABLE_PERIOD_MINUTES: int = int(
    os.getenv("MIN_STABLE_PERIOD_MINUTES", "30")
)
OPTIMIZATION_METHOD: str = os.getenv("OPTIMIZATION_METHOD", "L-BFGS-B")
# Calibration method for train_thermal_equilibrium_model().
# "scipy" (default): multi-pass L-BFGS-B joint optimisation.
# "physics": fully analytical, sequential physics-direct path (no scipy).
CALIBRATION_METHOD: str = os.getenv("CALIBRATION_METHOD", "scipy")

# Minimum 24-hour rolling mean outdoor temperature [°C] for cooling physics
# calibration.  Only historical rows where the 24h rolling mean of
# outdoor_temp exceeds this threshold are used for calibrating the cooling
# thermal equilibrium model.  Default 16°C ensures only genuine warm-season
# data enters the cooling calibration pipeline.
COOLING_PHYSICS_MIN_OUTDOOR_ROLLING_24H_C: float = float(
    os.getenv("COOLING_PHYSICS_MIN_OUTDOOR_ROLLING_24H_C", "16.0")
)

# Indoor temperature ceiling for PV calibration periods.
# Periods with indoor_temp >= this value are excluded from PV Pass 2
# because automated blinds likely closed, blocking solar gain while PV
# stays high on the roof.  Set just below your blind trigger threshold.
PV_CALIBRATION_INDOOR_CEILING: float = float(
    os.getenv("PV_CALIBRATION_INDOOR_CEILING", "23.0")
)

# --- Thermal Power Gate Thresholds ---
# Standardised thresholds applied consistently across calibration, session
# learning, and runtime HP-active detection.
#
# HEATING_MIN_THERMAL_POWER_KW
#   Minimum water-side thermal power [kW] accepted as genuine *heating*.
#   Used as a quality gate in HLC calibration (both historical and session-
#   based), in physics_calibration stable-period filters, and in the session
#   learner's per-cycle filter.  Windows/cycles below this threshold are
#   rejected because they are indistinguishable from pump-recirculation
#   standby, forward-filled data gaps, or low-load slab warm-up.
#   Default 0.5 kW aligns with the long-standing physics_calibration.py
#   usage and the slab-tau / HP-startup detection code.
HEATING_MIN_THERMAL_POWER_KW: float = float(
    os.getenv("HEATING_MIN_THERMAL_POWER_KW", "0.5")
)
# COOLING_MIN_THERMAL_POWER_KW
#   Minimum thermal power [kW] accepted as genuine *cooling* (always negative
#   in cooling mode: outlet < inlet).  Used in cooling-side calibration to
#   reject low-load or standby rows.
#   Default -0.5 kW (symmetric with the heating threshold).
COOLING_MIN_THERMAL_POWER_KW: float = float(
    os.getenv("COOLING_MIN_THERMAL_POWER_KW", "-0.5")
)
# HP_ACTIVE_MIN_POWER_KW
#   Noise-floor threshold [kW] for detecting whether the heat pump is
#   *running at all* at runtime.  Semantically different from the calibration
#   thresholds above: this catches any non-zero pump output so that learning
#   and channel attribution are activated even at minimum HP capacity.
#   Used in heat_source_channels._is_heat_pump_active() and
#   temperature_control._perform_learning().
#   Default 0.05 kW (50 W), the approximate standby recirculation level.
HP_ACTIVE_MIN_POWER_KW: float = float(
    os.getenv("HP_ACTIVE_MIN_POWER_KW", "0.05")
)

# --- HLC Validation Gates ---
# Shared quality thresholds used by both the day-level session learner
# (_close_day) and the historical calibration function (calibrate_hlc).
# Maximum mean PV power [W] allowed in a validation window/day.
HLC_PV_MAX_W: float = float(os.getenv("HLC_PV_MAX_W", "50.0"))
# Maximum allowed |mean_indoor − target_temp| [K] (equilibrium gate).
HLC_MAX_INDOOR_DELTA: float = float(os.getenv("HLC_MAX_INDOOR_DELTA", "0.3"))
# Maximum allowed |indoor_temp_delta_60m| [K] (stability gate).
HLC_MAX_TREND: float = float(os.getenv("HLC_MAX_TREND", "0.2"))
# Outdoor temperature range for valid calibration [°C].
HLC_OUTDOOR_TEMP_MIN: float = float(os.getenv("HLC_OUTDOOR_TEMP_MIN", "-10.0"))
HLC_OUTDOOR_TEMP_MAX: float = float(os.getenv("HLC_OUTDOOR_TEMP_MAX", "15.0"))
# Minimum required (T_target − T_outdoor) [K] to ensure real heating demand.
HLC_MIN_HEATING_DEMAND_K: float = float(
    os.getenv("HLC_MIN_HEATING_DEMAND_K", "1.0")
)
# Default indoor target temperature [°C] used when the target_temp sensor
# is unavailable.  Keeps the indoor_far_from_target and low_heating_demand
# quality gates active for HLC regression.
HLC_DEFAULT_TARGET_TEMP: float = float(
    os.getenv("HLC_DEFAULT_TARGET_TEMP", "22.6")
)


# --- Historical HLC Calibration ---
# Number of hours of historical data to fetch for HLC calibration.
HLC_CALIBRATION_LOOKBACK_HOURS: int = int(
    os.getenv("HLC_CALIBRATION_LOOKBACK_HOURS", "720")
)
# Minimum number of stable periods required for a reliable estimate.
HLC_CALIBRATION_MIN_PERIODS: int = int(
    os.getenv("HLC_CALIBRATION_MIN_PERIODS", "20")
)
# Window size in 5-minute rows for the HLC calibration.
# calibrate_hlc() processes data in non-overlapping blocks of this size.
# Default 12 rows = 60 minutes, which better approximates thermal equilibrium
# for buildings with multi-hour thermal time constants.
HLC_WINDOW_SIZE_ROWS: int = int(os.getenv("HLC_WINDOW_SIZE_ROWS", "12"))
# Minimum water-side flow rate [L/min] required to treat a window as active
# heating.  Windows below this threshold are rejected with "flow_too_low" to
# prevent forward-filled standby periods from passing quality gates.
HLC_MIN_FLOW_RATE_LPM: float = float(os.getenv("HLC_MIN_FLOW_RATE_LPM", "0.5"))
# Minimum thermal power for the calibration window is now governed by the
# shared HEATING_MIN_THERMAL_POWER_KW variable defined above.
# When true, calibrate_hlc also fits Q = HLC*ΔT + Q0 (with intercept) and
# logs Q0 as a diagnostic — a large |Q0| flags data contamination.
HLC_REGRESSION_INTERCEPT: bool = (
    os.getenv("HLC_REGRESSION_INTERCEPT", "false").lower() in ("1", "true", "yes")
)

# --- Delta Temperature Forecast Calibration ---
# Enable local calibration of weather forecasts using measured temperature
# offset. This corrects for systematic biases between weather station and
# actual location.
ENABLE_DELTA_FORECAST_CALIBRATION: bool = (
    os.getenv("ENABLE_DELTA_FORECAST_CALIBRATION", "true").lower() == "true"
)
# Maximum allowed temperature offset to prevent unrealistic corrections.
DELTA_CALIBRATION_MAX_OFFSET: float = float(
    os.getenv("DELTA_CALIBRATION_MAX_OFFSET", "10.0")
)

# Absolute clamp values for outlet temperature (heating mode)
CLAMP_MIN_ABS: float = float(os.getenv("CLAMP_MIN_ABS", "25.0"))
CLAMP_MAX_ABS: float = float(os.getenv("CLAMP_MAX_ABS", "55.0"))

# --- Cooling Mode Configuration ---
# Cooling outlet temperature bounds.
# COOLING_CLAMP_MIN_ABS: Absolute minimum outlet temp in cooling mode.
#   The heat pump shuts down when outlet reaches this value.
# COOLING_CLAMP_MAX_ABS: Maximum outlet temp in cooling mode.
#   Typically near inlet temp; the HP needs at least MIN_COOLING_DELTA_K
#   between inlet and outlet to operate.
COOLING_CLAMP_MIN_ABS: float = float(os.getenv("COOLING_CLAMP_MIN_ABS", "18.0"))
COOLING_CLAMP_MAX_ABS: float = float(os.getenv("COOLING_CLAMP_MAX_ABS", "24.0"))
# Minimum delta between inlet and outlet for heat pump operation in cooling (K)
MIN_COOLING_DELTA_K: float = float(os.getenv("MIN_COOLING_DELTA_K", "2.0"))
# Safety margin above the HP shutdown limit to prevent short-cycling.
# The ML will target at least this margin above COOLING_CLAMP_MIN_ABS
# to avoid the HP hitting the hard 18°C limit and shutting down.
COOLING_SHUTDOWN_MARGIN_K: float = float(
    os.getenv("COOLING_SHUTDOWN_MARGIN_K", "1.0")
)

# --- Pre-Cooling (Predictive Overheating Prevention) ---
# When enabled in cooling mode, the system runs a passive trajectory
# simulation (HP OFF) using PV + outdoor forecasts.  If the trajectory
# predicts indoor temperature will exceed the cooling target, the system
# activates the heat pump proactively — before the room actually overheats.
PRE_COOL_ENABLED: bool = (
    os.getenv("PRE_COOL_ENABLED", "true").lower() == "true"
)
# Trigger pre-cooling when predicted peak exceeds cooling target + this margin [K].
PRE_COOL_TRIGGER_MARGIN_K: float = float(
    os.getenv("PRE_COOL_TRIGGER_MARGIN_K", "0.5")
)
# How many hours ahead to simulate the passive trajectory.
PRE_COOL_HORIZON_HOURS: int = int(
    os.getenv("PRE_COOL_HORIZON_HOURS", "12")
)
# Start pre-cooling this many hours before the predicted peak.
PRE_COOL_LEAD_TIME_HOURS: float = float(
    os.getenv("PRE_COOL_LEAD_TIME_HOURS", "8.0")
)
# How much to shift the binary-search target temperature down [K] to trigger
# the heat pump when the room hasn't overheated yet.
PRE_COOL_TARGET_OFFSET_K: float = float(
    os.getenv("PRE_COOL_TARGET_OFFSET_K", "0.5")
)
# Minimum total PV forecast [W] over the horizon to consider overheating risk.
# Prevents triggering on cold rainy days where PV forecast is near zero.
PRE_COOL_MIN_PV_FORECAST_W: float = float(
    os.getenv("PRE_COOL_MIN_PV_FORECAST_W", "1000.0")
)
# Minimum peak outdoor temperature forecast [°C] to consider overheating risk.
PRE_COOL_MIN_OUTDOOR_FORECAST_C: float = float(
    os.getenv("PRE_COOL_MIN_OUTDOOR_FORECAST_C", "22.0")
)

# --- ML-Based Pre-Cooling Model (LightGBM Overheating Classifier) ---
# Active pre-cooling strategy selector.
# "trajectory" = physics simulation (BOTH: default, no model file required).
# "lgbm_model" = trained LightGBM classifier (MODEL-BASED: requires calibration).
# Tooltip: BOTH — controls which method is active vs. shadow.
PRE_COOL_MODEL_TYPE: str = os.getenv("PRE_COOL_MODEL_TYPE", "trajectory")

# Proportional pre-cooling: scale offset by regression-predicted overshoot.
PRE_COOL_PROPORTIONAL: bool = (
    os.getenv("PRE_COOL_PROPORTIONAL", "true").lower() == "true"
)
PRE_COOL_MIN_OFFSET_K: float = float(
    os.getenv("PRE_COOL_MIN_OFFSET_K", "0.2")
)
PRE_COOL_MAX_OFFSET_K: float = float(
    os.getenv("PRE_COOL_MAX_OFFSET_K", "1.0")
)
PRE_COOL_OVERSHOOT_GAIN: float = float(
    os.getenv("PRE_COOL_OVERSHOOT_GAIN", "0.7")
)
# Dual-output strategy: "classifier_gate" (conservative, default) or
# "either_triggers" (aggressive, catches more events).
PRE_COOL_DUAL_OUTPUT_STRATEGY: str = os.getenv(
    "PRE_COOL_DUAL_OUTPUT_STRATEGY", "classifier_gate"
)

_UNIFIED_STATE_DIR: str = os.path.dirname(
    os.getenv("UNIFIED_STATE_FILE", "/opt/ml_heating/unified_thermal_state.json")
)
# Path to trained LightGBM classifier (joblib). Tooltip: MODEL-BASED only.
COOLING_ML_MODEL_PATH: str = os.getenv(
    "COOLING_ML_MODEL_PATH",
    os.path.join(_UNIFIED_STATE_DIR, "cooling_ml_model.joblib"),
)
# Path to model metadata JSON (feature list, threshold, AUC). Tooltip: MODEL-BASED only.
COOLING_ML_METADATA_PATH: str = os.getenv(
    "COOLING_ML_METADATA_PATH",
    os.path.join(_UNIFIED_STATE_DIR, "cooling_ml_metadata.json"),
)
# Path to trained LightGBM regressor (joblib) for dual-output mode.
COOLING_ML_REGRESSOR_PATH: str = os.getenv(
    "COOLING_ML_REGRESSOR_PATH",
    os.path.join(_UNIFIED_STATE_DIR, "cooling_ml_regressor.joblib"),
)
# Path to observation buffer JSON for sliding-window online learning.
# Tooltip: MODEL-BASED only.
COOLING_ML_OBSERVATION_BUFFER_PATH: str = os.getenv(
    "COOLING_ML_OBSERVATION_BUFFER_PATH",
    os.path.join(_UNIFIED_STATE_DIR, "cooling_ml_obs_buffer.json"),
)
# Minimum labeled observations required before first train / retrain.
# Tooltip: MODEL-BASED only. Default 200.
COOLING_ML_MIN_TRAINING_SAMPLES: int = int(
    os.getenv("COOLING_ML_MIN_TRAINING_SAMPLES", "200")
)
# Number of new labeled observations that triggers an automatic retrain.
# Tooltip: MODEL-BASED only. Default 50.
COOLING_ML_RETRAIN_TRIGGER_K: int = int(
    os.getenv("COOLING_ML_RETRAIN_TRIGGER_K", "50")
)
# Rolling buffer size: keep the last N labeled observations for retraining.
# Tooltip: MODEL-BASED only. Default 500.
COOLING_ML_BUFFER_MAX_N: int = int(os.getenv("COOLING_ML_BUFFER_MAX_N", "500"))
# Fraction of buffer held out for threshold optimisation during each retrain.
# Tooltip: MODEL-BASED only. Default 0.25.
COOLING_ML_RETRAIN_VAL_FRACTION: float = float(
    os.getenv("COOLING_ML_RETRAIN_VAL_FRACTION", "0.25")
)
# Default comma-separated list of forecast hours used for AT and PV hindcast
# features during cooling ML calibration.  Derived from PRE_COOL_LEAD_TIME_HOURS
# so only hours within the label window are included by default.
_COOLING_ML_DEFAULT_FORECAST_HOURS = ",".join(
    str(h) for h in range(1, int(PRE_COOL_LEAD_TIME_HOURS) + 1)
)

# Comma-separated list of forecast hours to use as outdoor-temperature hindcast
# features (AT_roh_Xh) during calibration.  All 12 hours are included by default
# so the model can see the full daily temperature cycle ahead.
# Legacy env var COOLING_ML_FORECAST_HOURS is honoured as an alias.
# Tooltip: MODEL-BASED only.
COOLING_ML_AT_FORECAST_HOURS: str = os.getenv(
    "COOLING_ML_AT_FORECAST_HOURS",
    os.getenv("COOLING_ML_FORECAST_HOURS", _COOLING_ML_DEFAULT_FORECAST_HOURS),
)
# Backward-compat alias: resolved from COOLING_ML_AT_FORECAST_HOURS (which already
# consumed the legacy COOLING_ML_FORECAST_HOURS env var as its own fallback).
COOLING_ML_FORECAST_HOURS: str = COOLING_ML_AT_FORECAST_HOURS
# Comma-separated list of forecast hours to use as PV-power hindcast features
# (pv_forecast_Xh) during calibration.  All 12 hours are included by default.
# Tooltip: MODEL-BASED only.
COOLING_ML_PV_FORECAST_HOURS: str = os.getenv(
    "COOLING_ML_PV_FORECAST_HOURS", _COOLING_ML_DEFAULT_FORECAST_HOURS
)

# Minimum outdoor temperature [°C] to include rows in the cooling ML training
# set.  A lower value (e.g. 10°C) adds shoulder-season data with mostly
# negative labels, improving class balance.
COOLING_ML_WARM_THRESHOLD_C: float = float(
    os.getenv("COOLING_ML_WARM_THRESHOLD_C", "10.0")
)

# Earliest date for cooling ML training data.  Format: DD.MM.YYYY (e.g. 01.06.2024).
# When set, calibrate_cooling_ml() computes lookback_hours as (now − start_date).
# Leave empty to use the default 2160 h (90-day) lookback.
# Tooltip: MODEL-BASED only.
COOLING_ML_CALIBRATION_START_DATE: str = os.getenv(
    "COOLING_ML_CALIBRATION_START_DATE", ""
)

# --- ML-Based Heating Correction (LightGBM Regressor) ---
# Outdoor temperature ceiling [°C] for the heating-season filter.
# Only rows with AT < this threshold are used for training.
HEATING_ML_COLD_THRESHOLD_C: float = float(
    os.getenv("HEATING_ML_COLD_THRESHOLD_C", "18.0")
)
# Earliest date for heating ML correction training data.  Format: DD.MM.YYYY.
# When set, calibrate_heating_correction_ml() computes lookback_hours as
# (now − start_date).  Leave empty to use the default 2160 h (90-day) lookback.
HEATING_ML_CALIBRATION_START_DATE: str = os.getenv(
    "HEATING_ML_CALIBRATION_START_DATE", ""
)
# Comma-separated list of forecast hours to use as AT hindcast features
# (AT_roh_Xh) during heating correction calibration.
HEATING_ML_AT_FORECAST_HOURS: str = os.getenv(
    "HEATING_ML_AT_FORECAST_HOURS", "1,2,3,4"
)
# Comma-separated list of forecast hours to use as PV hindcast features
# (pv_forecast_Xh) during heating correction calibration.
# PV solar gain can be significant on sunny winter days even with AT < 18 °C.
HEATING_ML_PV_FORECAST_HOURS: str = os.getenv(
    "HEATING_ML_PV_FORECAST_HOURS", "1,2,3,4"
)
# Comma-separated list of lag window sizes in HOURS for fireplace residual-heat
# features (fireplace_lag_Xh).  Each value generates one rolling-max feature.
# Larger windows let the model learn longer decay tails after fireplace is off.
HEATING_ML_FIREPLACE_LAG_HOURS: str = os.getenv(
    "HEATING_ML_FIREPLACE_LAG_HOURS", "1,2"
)
# Comma-separated list of lag window sizes in HOURS for TV residual-heat features.
# Use fractional hours for sub-hour windows (e.g. "0.5" → tv_lag_30m).
HEATING_ML_TV_LAG_HOURS: str = os.getenv(
    "HEATING_ML_TV_LAG_HOURS", "0.5,1"
)
# Path to the trained LightGBM heating correction regressor (joblib).
HEATING_ML_CORRECTION_MODEL_PATH: str = os.getenv(
    "HEATING_ML_CORRECTION_MODEL_PATH",
    os.path.join(_UNIFIED_STATE_DIR, "heating_correction_ml_model.joblib"),
)
# Path to heating ML model metadata JSON (feature list, R², MAE).
HEATING_ML_CORRECTION_METADATA_PATH: str = os.getenv(
    "HEATING_ML_CORRECTION_METADATA_PATH",
    os.path.join(_UNIFIED_STATE_DIR, "heating_correction_ml_metadata.json"),
)
# Minimum cold-season rows required before training is accepted.
HEATING_ML_MIN_TRAINING_SAMPLES: int = int(
    os.getenv("HEATING_ML_MIN_TRAINING_SAMPLES", "200")
)
# Fraction of the training set held out for validation metrics.
HEATING_ML_RETRAIN_VAL_FRACTION: float = float(
    os.getenv("HEATING_ML_RETRAIN_VAL_FRACTION", "0.25")
)
# Label lookahead horizon for training [hours].  Should match TRAJECTORY_STEPS.
HEATING_ML_LABEL_HORIZON_H: int = int(
    os.getenv("HEATING_ML_LABEL_HORIZON_H", "4")
)
# Path to the online-learning observation buffer JSON for the heating correction
# regressor.  Defaults to the same directory as UNIFIED_STATE_FILE so all
# runtime state lives in one place.
HEATING_ML_OBSERVATION_BUFFER_PATH: str = os.getenv(
    "HEATING_ML_OBSERVATION_BUFFER_PATH",
    os.path.join(_UNIFIED_STATE_DIR, "heating_correction_ml_obs_buffer.json"),
)
# Number of new labeled observations that triggers an automatic retrain.
HEATING_ML_RETRAIN_TRIGGER_K: int = int(
    os.getenv("HEATING_ML_RETRAIN_TRIGGER_K", "50")
)
# Rolling buffer size: keep the last N labeled observations for retraining.
HEATING_ML_BUFFER_MAX_N: int = int(os.getenv("HEATING_ML_BUFFER_MAX_N", "500"))
# --- Feature pruning (permutation importance-based) ---
# When true, features with permutation importance <= the threshold are removed
# and the model is retrained on the pruned feature set.  Only accepted if the
# pruned model's MAE does not regress beyond 0.5%.
HEATING_ML_FEATURE_PRUNING_ENABLED: bool = (
    os.getenv("HEATING_ML_FEATURE_PRUNING_ENABLED", "true").lower() == "true"
)
# Permutation importance cutoff.  Features at or below this value are pruned.
HEATING_ML_PRUNE_PI_THRESHOLD: float = float(
    os.getenv("HEATING_ML_PRUNE_PI_THRESHOLD", "0.0")
)
# --- LightGBM regularization ---
HEATING_ML_REG_ALPHA: float = float(
    os.getenv("HEATING_ML_REG_ALPHA", "0.1")
)
HEATING_ML_REG_LAMBDA: float = float(
    os.getenv("HEATING_ML_REG_LAMBDA", "1.0")
)
# --- Optuna hyper-parameter optimisation (optional, adds training time) ---
HEATING_ML_OPTUNA_ENABLED: bool = (
    os.getenv("HEATING_ML_OPTUNA_ENABLED", "false").lower() == "true"
)
HEATING_ML_OPTUNA_N_TRIALS: int = int(
    os.getenv("HEATING_ML_OPTUNA_N_TRIALS", "20")
)
# --- Time-series cross-validation ---
HEATING_ML_CV_ENABLED: bool = (
    os.getenv("HEATING_ML_CV_ENABLED", "false").lower() == "true"
)
HEATING_ML_CV_N_SPLITS: int = int(
    os.getenv("HEATING_ML_CV_N_SPLITS", "3")
)

# ============================================================================
# Cooling ML Correction (LightGBM Regressor)
# ============================================================================
# Analogous to the heating ML correction but for cooling mode.
# Uses a residualized label: adjusted_label = -(T_future - T_current) / S_H_cool
# where S_H_cool uses the cooling outlet-effectiveness (OE_cooling ≈ 0.20).

# Cooling outlet effectiveness — calibrated from cooling training data.
COOLING_OUTLET_EFFECTIVENESS: float = float(
    os.getenv("COOLING_OUTLET_EFFECTIVENESS", "0.20")
)

# Cooling correction algorithm selector.
# "physics" — physics Newton step (default).
# "ml"      — LightGBM regressor (requires cooling correction calibration).
COOLING_CORRECTION_MODE: str = os.getenv("COOLING_CORRECTION_MODE", "physics")

# Minimum outdoor temperature for warm-season filtering (cooling data).
COOLING_ML_CORRECTION_WARM_THRESHOLD_C: float = float(
    os.getenv("COOLING_ML_CORRECTION_WARM_THRESHOLD_C", "18.0")
)
# Earliest date for cooling correction ML training data.  Format: DD.MM.YYYY.
COOLING_ML_CORRECTION_CALIBRATION_START_DATE: str = os.getenv(
    "COOLING_ML_CORRECTION_CALIBRATION_START_DATE", ""
)
# Comma-separated AT forecast hours for cooling correction features.
COOLING_ML_CORRECTION_AT_FORECAST_HOURS: str = os.getenv(
    "COOLING_ML_CORRECTION_AT_FORECAST_HOURS", "1,2,3,4"
)
# Comma-separated PV forecast hours for cooling correction features.
COOLING_ML_CORRECTION_PV_FORECAST_HOURS: str = os.getenv(
    "COOLING_ML_CORRECTION_PV_FORECAST_HOURS", "1,2,3,4"
)
# Comma-separated fireplace lag hours for cooling correction features.
COOLING_ML_CORRECTION_FIREPLACE_LAG_HOURS: str = os.getenv(
    "COOLING_ML_CORRECTION_FIREPLACE_LAG_HOURS", "1,2"
)
# Comma-separated TV lag hours for cooling correction features.
COOLING_ML_CORRECTION_TV_LAG_HOURS: str = os.getenv(
    "COOLING_ML_CORRECTION_TV_LAG_HOURS", "0.5,1"
)
# Label lookahead horizon [hours].
COOLING_ML_CORRECTION_LABEL_HORIZON_H: int = int(
    os.getenv("COOLING_ML_CORRECTION_LABEL_HORIZON_H", "4")
)
# Path to trained LightGBM cooling correction regressor (joblib).
COOLING_ML_CORRECTION_MODEL_PATH: str = os.getenv(
    "COOLING_ML_CORRECTION_MODEL_PATH",
    os.path.join(_UNIFIED_STATE_DIR, "cooling_correction_ml_model.joblib"),
)
# Path to cooling ML model metadata JSON.
COOLING_ML_CORRECTION_METADATA_PATH: str = os.getenv(
    "COOLING_ML_CORRECTION_METADATA_PATH",
    os.path.join(_UNIFIED_STATE_DIR, "cooling_correction_ml_metadata.json"),
)
# Path to the online-learning observation buffer JSON.
COOLING_ML_CORRECTION_OBS_BUFFER_PATH: str = os.getenv(
    "COOLING_ML_CORRECTION_OBS_BUFFER_PATH",
    os.path.join(_UNIFIED_STATE_DIR, "cooling_correction_ml_obs_buffer.json"),
)
# Minimum warm-season rows required before training is accepted.
COOLING_ML_CORRECTION_MIN_TRAINING_SAMPLES: int = int(
    os.getenv("COOLING_ML_CORRECTION_MIN_TRAINING_SAMPLES", "200")
)
# Fraction of training data held out for validation.
COOLING_ML_CORRECTION_RETRAIN_VAL_FRACTION: float = float(
    os.getenv("COOLING_ML_CORRECTION_RETRAIN_VAL_FRACTION", "0.25")
)
# Number of new labeled observations that triggers automatic retrain.
COOLING_ML_CORRECTION_RETRAIN_TRIGGER_K: int = int(
    os.getenv("COOLING_ML_CORRECTION_RETRAIN_TRIGGER_K", "50")
)
# Rolling buffer size: keep the last N labeled observations.
COOLING_ML_CORRECTION_BUFFER_MAX_N: int = int(
    os.getenv("COOLING_ML_CORRECTION_BUFFER_MAX_N", "500")
)
# Feature pruning (permutation importance-based).
COOLING_ML_CORRECTION_FEATURE_PRUNING_ENABLED: bool = (
    os.getenv("COOLING_ML_CORRECTION_FEATURE_PRUNING_ENABLED", "true").lower() == "true"
)
COOLING_ML_CORRECTION_PRUNE_PI_THRESHOLD: float = float(
    os.getenv("COOLING_ML_CORRECTION_PRUNE_PI_THRESHOLD", "0.0")
)
# LightGBM regularization.
COOLING_ML_CORRECTION_REG_ALPHA: float = float(
    os.getenv("COOLING_ML_CORRECTION_REG_ALPHA", "0.1")
)
COOLING_ML_CORRECTION_REG_LAMBDA: float = float(
    os.getenv("COOLING_ML_CORRECTION_REG_LAMBDA", "1.0")
)
# Optuna HPO (optional).
COOLING_ML_CORRECTION_OPTUNA_ENABLED: bool = (
    os.getenv("COOLING_ML_CORRECTION_OPTUNA_ENABLED", "false").lower() == "true"
)
COOLING_ML_CORRECTION_OPTUNA_N_TRIALS: int = int(
    os.getenv("COOLING_ML_CORRECTION_OPTUNA_N_TRIALS", "20")
)
# Time-series cross-validation.
COOLING_ML_CORRECTION_CV_ENABLED: bool = (
    os.getenv("COOLING_ML_CORRECTION_CV_ENABLED", "false").lower() == "true"
)
COOLING_ML_CORRECTION_CV_N_SPLITS: int = int(
    os.getenv("COOLING_ML_CORRECTION_CV_N_SPLITS", "3")
)

# Earliest date for heating physics calibration training data.  Format: DD.MM.YYYY.
# When set, train_thermal_equilibrium_model() / calibrate_thermal_model_physics()
# compute lookback_hours as (now − start_date).
# Leave empty to use the default TRAINING_LOOKBACK_HOURS.
PHYSICS_CALIBRATION_START_DATE: str = os.getenv(
    "PHYSICS_CALIBRATION_START_DATE", ""
)

# Earliest date for cooling physics calibration training data.  Format: DD.MM.YYYY.
# When set, calibrate_cooling_physics() computes lookback_hours as (now − start_date).
# Leave empty to use the default TRAINING_LOOKBACK_HOURS × 2 lookback.
COOLING_PHYSICS_CALIBRATION_START_DATE: str = os.getenv(
    "COOLING_PHYSICS_CALIBRATION_START_DATE", ""
)


def _parse_physics_start_date(date_str: Optional[str]) -> "Optional[datetime]":
    """Parse DD.MM.YYYY string to a timezone-aware UTC datetime, or return None.

    Used by the heating and cooling physics calibration paths.
    """
    from datetime import datetime, timezone  # local import avoids circular issues
    s = (date_str or "").strip()
    if not s:
        return None
    try:
        dt = datetime.strptime(s, "%d.%m.%Y")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# Alias used by the cooling physics calibration path.
_parse_cooling_physics_start_date = _parse_physics_start_date


def _parse_heating_start_date(date_str: str) -> "Optional[datetime]":
    """Parse DD.MM.YYYY string to a timezone-aware UTC datetime, or return None.

    Mirror of ``_parse_cooling_start_date`` for the heating ML calibration.
    """
    from datetime import datetime, timezone  # local import avoids circular issues
    s = (date_str or "").strip()
    if not s:
        return None
    try:
        dt = datetime.strptime(s, "%d.%m.%Y")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_cooling_start_date(date_str: str) -> "Optional[datetime]":
    """Parse DD.MM.YYYY string to a timezone-aware UTC datetime, or return None.

    Returns
    -------
    datetime or None
        Timezone-aware UTC datetime for the start of the given date,
        or ``None`` if ``date_str`` is empty or not a valid DD.MM.YYYY string.
    """
    from datetime import datetime, timezone  # local import avoids circular issues
    s = (date_str or "").strip()
    if not s:
        return None
    try:
        dt = datetime.strptime(s, "%d.%m.%Y")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# Thermal Model Parameters
PV_HEAT_WEIGHT: float = float(os.getenv("PV_HEAT_WEIGHT", "0.0020704649305198215"))
FIREPLACE_HEAT_WEIGHT: float = float(os.getenv("FIREPLACE_HEAT_WEIGHT", "0.387"))
TV_HEAT_WEIGHT: float = float(os.getenv("TV_HEAT_WEIGHT", "0.35"))
THERMAL_TIME_CONSTANT: float = float(os.getenv("THERMAL_TIME_CONSTANT", "4.390554703745845"))
HEAT_LOSS_COEFFICIENT: float = float(os.getenv("HEAT_LOSS_COEFFICIENT", "0.1245214561975565"))
OUTLET_EFFECTIVENESS: float = float(os.getenv("OUTLET_EFFECTIVENESS", "0.9526723072021629"))
DELTA_T_FLOOR: float = float(os.getenv("DELTA_T_FLOOR", "2.3"))
FP_DECAY_TIME_CONSTANT: float = float(os.getenv("FP_DECAY_TIME_CONSTANT", "3.9144707244638868"))
ROOM_SPREAD_DELAY_MINUTES: float = float(os.getenv("ROOM_SPREAD_DELAY_MINUTES", "18.0"))
CLOUD_FACTOR_EXPONENT: float = float(os.getenv("CLOUD_FACTOR_EXPONENT", "1.0"))
SOLAR_DECAY_TAU_HOURS: float = float(os.getenv("SOLAR_DECAY_TAU_HOURS", "1.0"))
EQUILIBRIUM_RATIO: float = float(os.getenv("EQUILIBRIUM_RATIO", "0.17"))
TOTAL_CONDUCTANCE: float = float(os.getenv("TOTAL_CONDUCTANCE", "0.8"))
ADAPTIVE_LEARNING_RATE: float = float(
    os.getenv("ADAPTIVE_LEARNING_RATE", "0.01")
)
LEARNING_CONFIDENCE: float = float(os.getenv("LEARNING_CONFIDENCE", "3.0"))
MIN_LEARNING_RATE: float = float(os.getenv("MIN_LEARNING_RATE", "0.001"))
MAX_LEARNING_RATE: float = float(os.getenv("MAX_LEARNING_RATE", "0.1"))
SOLAR_LAG_MINUTES: float = float(os.getenv("SOLAR_LAG_MINUTES", "45.0"))
CLOUD_CORRECTION_MIN_FACTOR: float = float(
    os.getenv("CLOUD_CORRECTION_MIN_FACTOR", "0.1")
)
SLAB_TIME_CONSTANT_HOURS: float = float(os.getenv("SLAB_TIME_CONSTANT_HOURS", "3.19"))
MAX_PREDICTION_HISTORY: int = int(os.getenv("MAX_PREDICTION_HISTORY", "700"))
MAX_PARAMETER_HISTORY: int = int(os.getenv("MAX_PARAMETER_HISTORY", "700"))
INFLUX_METRICS_EXPORT_INTERVAL_CYCLES: int = int(
    os.getenv("INFLUX_METRICS_EXPORT_INTERVAL_CYCLES", "5")
)
INDOOR_COOLING_TREND_THRESHOLD: float = float(os.getenv("INDOOR_COOLING_TREND_THRESHOLD", "-0.05"))
INDOOR_COOLING_DAMPING_FACTOR: float = float(os.getenv("INDOOR_COOLING_DAMPING_FACTOR", "0.3"))
INDOOR_WARMING_TREND_THRESHOLD: float = float(os.getenv("INDOOR_WARMING_TREND_THRESHOLD", "0.10"))
INDOOR_WARMING_DAMPING_FACTOR: float = float(os.getenv("INDOOR_WARMING_DAMPING_FACTOR", "0.3"))

# --- Heat Source Channel Architecture (Phase 2) ---
# Enable decomposed heat-source learning with independent channels for
# heat pump, solar, fireplace, and TV.  When enabled, learning guards
# prevent cross-contamination between heat sources.
ENABLE_HEAT_SOURCE_CHANNELS: bool = (
    os.getenv("ENABLE_HEAT_SOURCE_CHANNELS", "true").lower() == "true"
)
ENABLE_MIXED_SOURCE_ATTRIBUTION: bool = (
    os.getenv("ENABLE_MIXED_SOURCE_ATTRIBUTION", "false").lower()
    == "true"
)

# --- Decay Window & Safety Thresholds ---
# Multiplier applied to thermal_time_constant for PV room-heat decay window.
# After PV drops below threshold, HP learning is frozen for
# thermal_time_constant × PV_ROOM_DECAY_MULTIPLIER hours (~8.7h at τ=4.37h).
PV_ROOM_DECAY_MULTIPLIER: float = float(
    os.getenv("PV_ROOM_DECAY_MULTIPLIER", "2.0")
)
# Indoor temp margin above target_temp at which PV/FP decay is cancelled
# early — residual heat from the external source has dissipated.
DECAY_CANCEL_MARGIN: float = float(
    os.getenv("DECAY_CANCEL_MARGIN", "0.1")
)


# --- Climate Mode Helpers ---
def get_climate_mode(heating_state: str | None) -> str:
    """Determine climate mode from the HEATING_STATUS_ENTITY_ID state.

    Returns:
        "heating" if state is "heat" or "auto",
        "cooling" if state is "cool",
        "off" otherwise.
    """
    if heating_state is None:
        return "off"
    state_lower = str(heating_state).lower().strip()
    if state_lower in ("heat", "auto"):
        return "heating"
    if state_lower == "cool":
        return "cooling"
    return "off"


def get_outlet_bounds(climate_mode: str) -> tuple[float, float]:
    """Return (min, max) outlet temperature bounds for the given mode.

    In heating mode the outlet is above room temperature.
    In cooling mode the outlet is below room temperature.
    """
    if climate_mode == "cooling":
        # Return the full physical range.  The post-search cooling cycle
        # gate (RUNNING/RECOVERY) handles HP-cannot-run scenarios;
        # COOLING_SHUTDOWN_MARGIN_K is only used in the RECOVERY→RUNNING
        # transition check.
        return COOLING_CLAMP_MIN_ABS, COOLING_CLAMP_MAX_ABS
    return CLAMP_MIN_ABS, CLAMP_MAX_ABS


def get_fallback_outlet(climate_mode: str) -> float:
    """Return a safe fallback outlet temperature for the given mode."""
    if climate_mode == "cooling":
        # Mid-range cooling outlet — safe and not too aggressive.
        return (COOLING_CLAMP_MIN_ABS + COOLING_CLAMP_MAX_ABS) / 2.0
    return 35.0
