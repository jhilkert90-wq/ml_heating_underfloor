#!/usr/bin/env python3
"""
ML Heating Add-on Configuration Adapter

Maps Home Assistant add-on configuration options to environment variables
for compatibility with the existing ML heating system.
"""

import json
import os
import shlex
import sys
from pathlib import Path
from datetime import datetime
import shutil

def log_info(message):
    """Log info message"""
    print(f"[INFO] {message}")


def log_warning(message):
    """Log warning message"""
    print(f"[WARNING] {message}")


def log_error(message):
    """Log error message"""
    print(f"[ERROR] {message}")


def load_addon_config():
    """Load Home Assistant add-on configuration"""
    try:
        with open('/data/options.json', 'r') as f:
            config = json.load(f)
        log_info("Add-on configuration loaded successfully")
        return config
    except FileNotFoundError:
        log_error("/data/options.json not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log_error(f"Invalid JSON in options.json: {e}")
        sys.exit(1)

def setup_data_directories():
    """Create necessary data directories with proper permissions"""
    directories = [
        '/data/models',
        '/data/backups',
        '/data/logs',
        '/data/config',
        # /config/ml_heating is the default location for the unified thermal
        # state file in HA addon deployments (accessible via File Editor).
        # Created preemptively so the runtime can write to it on first start.
        '/config/ml_heating',
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        log_info(f"Created directory: {directory}")


def convert_addon_to_env(config):
    """Convert add-on options to environment variables for existing ML system.

    Maps every config.yaml option to the environment variable name expected by
    ``src/config.py``.  Keys that are not present in ``config`` are silently
    skipped so that both the legacy ``ml_heating`` / ``ml_heating_dev`` config
    schemas and the comprehensive ``ml_heating_underfloor`` schema work.
    """

    # Home Assistant API configuration (internal supervisor access)
    env_vars = {
        'HASS_URL': 'http://supervisor/core',
        'HASS_TOKEN': os.environ.get('SUPERVISOR_TOKEN', ''),

        # --- Core Entity Mappings (must match src/config.py variable names) --
        'TARGET_INDOOR_TEMP_ENTITY_ID': config.get(
            'target_indoor_temp_entity', ''
        ),
        'INDOOR_TEMP_ENTITY_ID': config.get('indoor_temp_entity', ''),
        'OUTDOOR_TEMP_ENTITY_ID': config.get('outdoor_temp_entity', ''),
        'HEATING_STATUS_ENTITY_ID': config.get('heating_control_entity', ''),
        'ACTUAL_OUTLET_TEMP_ENTITY_ID': config.get('outlet_temp_entity', ''),
        'INLET_TEMP_ENTITY_ID': config.get('inlet_temp_entity', ''),
        'FLOW_RATE_ENTITY_ID': config.get('flow_rate_entity', ''),
        'POWER_CONSUMPTION_ENTITY_ID': config.get(
            'power_consumption_entity', ''
        ),
        'SPECIFIC_HEAT_CAPACITY': str(
            config.get('specific_heat_capacity', 4.186)
        ),
        'TARGET_OUTLET_TEMP_ENTITY_ID': config.get(
            'target_outlet_temp_entity', ''
        ),
        'ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID': config.get(
            'actual_target_outlet_temp_entity', ''
        ),
        'OPENWEATHERMAP_TEMP_ENTITY_ID': config.get(
            'openweathermap_temp_entity', ''
        ),
        'AVG_OTHER_ROOMS_TEMP_ENTITY_ID': config.get(
            'avg_other_rooms_temp_entity', ''
        ),
        'PV_FORECAST_ENTITY_ID': config.get('pv_forecast_entity', ''),
        'LIVING_ROOM_TEMP_ENTITY_ID': config.get(
            'living_room_temp_entity', ''
        ),

        # --- External Heat Sources ---
        'PV_POWER_ENTITY_ID': config.get('pv_power_entity', ''),
        'SOLAR_CORRECTION_ENTITY_ID': config.get(
            'solar_correction_entity', ''
        ),
        'SOLAR_CORRECTION_DEFAULT_PERCENT': str(
            config.get('solar_correction_default_percent', 100.0)
        ),
        'SOLAR_CORRECTION_MIN_PERCENT': str(
            config.get('solar_correction_min_percent', 0.0)
        ),
        'SOLAR_CORRECTION_MAX_PERCENT': str(
            config.get('solar_correction_max_percent', 100.0)
        ),
        'FIREPLACE_STATUS_ENTITY_ID': config.get(
            'fireplace_status_entity', ''
        ),
        # tv_status_entity → TV_STATUS_ENTITY_ID (config.py name)
        'TV_STATUS_ENTITY_ID': config.get(
            'tv_status_entity',
            config.get('tv_power_entity', '')
        ),

        # --- Blocking Detection ---
        'DHW_STATUS_ENTITY_ID': config.get('dhw_status_entity', ''),
        'DEFROST_STATUS_ENTITY_ID': config.get('defrost_status_entity', ''),
        'DISINFECTION_STATUS_ENTITY_ID': config.get(
            'disinfection_status_entity', ''
        ),
        'DHW_BOOST_HEATER_STATUS_ENTITY_ID': config.get(
            'dhw_boost_heater_entity', ''
        ),

        # --- ML Learning Parameters ---
        'HISTORY_STEPS': str(config.get('history_steps', 6)),
        'HISTORY_STEP_MINUTES': str(config.get('history_step_minutes', 10)),
        'PREDICTION_HORIZON_STEPS': str(
            config.get('prediction_horizon_steps', 24)
        ),
        'TRAINING_LOOKBACK_HOURS': str(
            config.get('training_lookback_hours', 168)
        ),
        'CYCLE_INTERVAL_MINUTES': str(
            config.get('cycle_interval_minutes', 10)
        ),
        'MAX_TEMP_CHANGE_PER_CYCLE': str(
            config.get('max_temp_change_per_cycle', 2)
        ),
        'TRAINING_DATA_SOURCE': config.get('training_data_source', 'auto'),

        # --- Safety Configuration ---
        'CLAMP_MIN_ABS': str(config.get('clamp_min_abs', 18.0)),
        'CLAMP_MAX_ABS': str(config.get('clamp_max_abs', 35.0)),

        # --- Cooling Mode ---
        'COOLING_CLAMP_MIN_ABS': str(
            config.get('cooling_clamp_min_abs', 18.0)
        ),
        'COOLING_CLAMP_MAX_ABS': str(
            config.get('cooling_clamp_max_abs', 24.0)
        ),
        'MIN_COOLING_DELTA_K': str(
            config.get('min_cooling_delta_k', 2.0)
        ),
        'COOLING_SHUTDOWN_MARGIN_K': str(
            config.get('cooling_shutdown_margin_k', 1.0)
        ),

        # --- InfluxDB Configuration ---
        # config.py expects INFLUX_URL (full URL), INFLUX_TOKEN, INFLUX_ORG,
        # INFLUX_BUCKET.  The underfloor config uses these directly.
        # Legacy ml_heating configs use influxdb_host/port; we build the URL.
        'INFLUX_URL': config.get(
            'influx_url',
            'http://{}:{}'.format(
                config.get('influxdb_host', 'a0d7b954-influxdb'),
                config.get('influxdb_port', 8086)
            )
        ),
        'INFLUX_TOKEN': config.get(
            'influx_token',
            config.get('influxdb_token', '')
        ),
        'INFLUX_ORG': config.get(
            'influx_org',
            config.get('influxdb_org', '')
        ),
        'INFLUX_BUCKET': config.get(
            'influx_bucket',
            config.get('influxdb_bucket', 'home_assistant')
        ),
        'INFLUX_FEATURES_BUCKET': config.get(
            'influx_features_bucket', 'ml_heating_features'
        ),
        'INFLUX_METRICS_EXPORT_INTERVAL_CYCLES': str(
            config.get('influx_metrics_export_interval_cycles', 5)
        ),

        # --- Add-on specific paths ---
        # unified_state_file defaults to /config/ml_heating/ so it is
        # accessible from the HA File Editor add-on (same directory as
        # configuration.yaml). Users can override via config option.
        'UNIFIED_STATE_FILE': config.get(
            'unified_state_file',
            '/config/ml_heating/unified_thermal_state.json'
        ),
        'UNIFIED_STATE_FILE_COOLING': config.get(
            'unified_state_file_cooling',
            '/config/ml_heating/unified_thermal_state_cooling.json'
        ),
        'CALIBRATION_BASELINE_FILE': config.get(
            'calibration_baseline_file', '/data/calibrated_baseline.json'
        ),
        'BACKUP_DIR': '/data/backups',
        'LOG_FILE_PATH': '/data/logs/ml_heating.log',

        # --- Performance Tuning ---
        'CONFIDENCE_THRESHOLD': str(
            config.get('confidence_threshold', 2.0)
        ),
        'TRAJECTORY_STEPS': str(config.get('trajectory_steps', 4)),
        'GRACE_PERIOD_MAX_MINUTES': str(
            config.get('grace_period_max_minutes', 15)
        ),
        'DEFROST_RECOVERY_GRACE_MINUTES': str(
            config.get('defrost_recovery_grace_minutes', 45)
        ),
        'BLOCKING_POLL_INTERVAL_SECONDS': str(
            config.get('blocking_poll_interval_seconds', 60)
        ),

        # --- Shadow Mode ---
        'SHADOW_MODE': str(config.get('shadow_mode', False)).lower(),
        'ML_HEATING_CONTROL_ENTITY_ID': config.get(
            'ml_heating_control_entity', 'input_boolean.ml_heating'
        ),

        # --- Logging and Development ---
        'LOG_LEVEL': config.get('log_level', 'INFO'),
        'DEBUG': '1' if config.get('debug', False) else '0',
        'ENABLE_DEV_API': str(config.get('enable_dev_api', False)),
        'DEV_API_KEY': config.get('dev_api_key', ''),

        # --- Dashboard ---
        'DASHBOARD_UPDATE_INTERVAL': str(
            config.get('dashboard_update_interval', 30)
        ),
        'SHOW_ADVANCED_METRICS': str(
            config.get('show_advanced_metrics', True)
        ),
        'DASHBOARD_THEME': config.get('dashboard_theme', 'auto'),

        # --- Model Management ---
        'AUTO_BACKUP_ENABLED': str(config.get('auto_backup_enabled', True)),
        'BACKUP_RETENTION_DAYS': str(config.get('backup_retention_days', 30)),

        # --- Thermal Model Parameters ---
        # Defaults are calibrated baseline values used when no calibration
        # file or unified thermal state file is available.
        'THERMAL_TIME_CONSTANT': str(
            config.get('thermal_time_constant', 4.390554703745845)
        ),
        'HEAT_LOSS_COEFFICIENT': str(
            config.get('heat_loss_coefficient', 0.1245214561975565)
        ),
        'OUTLET_EFFECTIVENESS': str(
            config.get('outlet_effectiveness', 0.9526723072021629)
        ),
        'OUTDOOR_COUPLING': str(config.get('outdoor_coupling', 0.3)),
        'THERMAL_BRIDGE_FACTOR': str(
            config.get('thermal_bridge_factor', 0.1)
        ),
        'EQUILIBRIUM_RATIO': str(config.get('equilibrium_ratio', 0.17)),
        'TOTAL_CONDUCTANCE': str(config.get('total_conductance', 0.8)),
        'SLAB_TIME_CONSTANT_HOURS': str(
            config.get('slab_time_constant_hours', 3.19)
        ),
        'SOLAR_LAG_MINUTES': str(config.get('solar_lag_minutes', 45.0)),
        'CLOUD_CORRECTION_MIN_FACTOR': str(
            config.get('cloud_correction_min_factor', 0.1)
        ),

        # --- External Heat Source Weights ---
        'PV_HEAT_WEIGHT': str(
            config.get('pv_heat_weight', 0.0020704649305198215)
        ),
        'FIREPLACE_HEAT_WEIGHT': str(
            config.get('fireplace_heat_weight', 0.387)
        ),
        'TV_HEAT_WEIGHT': str(config.get('tv_heat_weight', 0.35)),
        'DELTA_T_FLOOR': str(config.get('delta_t_floor', 2.3)),
        'FP_DECAY_TIME_CONSTANT': str(
            config.get('fp_decay_time_constant', 3.9144707244638868)
        ),
        'ROOM_SPREAD_DELAY_MINUTES': str(
            config.get('room_spread_delay_minutes', 18.0)
        ),

        # --- Adaptive Learning Parameters ---
        'ADAPTIVE_LEARNING_RATE': str(
            config.get('adaptive_learning_rate', 0.01)
        ),
        'MIN_LEARNING_RATE': str(config.get('min_learning_rate', 0.001)),
        'MAX_LEARNING_RATE': str(config.get('max_learning_rate', 0.01)),
        'LEARNING_CONFIDENCE': str(config.get('learning_confidence', 3.0)),
        'RECENT_ERRORS_WINDOW': str(
            config.get('recent_errors_window', 10)
        ),
        'LEARNING_DEAD_ZONE': str(config.get('learning_dead_zone', 0.01)),
        'PV_LEARNING_THRESHOLD': str(
            config.get('pv_learning_threshold', 50)
        ),

        # --- Hybrid Learning Strategy ---
        'HYBRID_LEARNING_ENABLED': str(
            config.get('hybrid_learning_enabled', True)
        ).lower(),
        'STABILITY_CLASSIFICATION_ENABLED': str(
            config.get('stability_classification_enabled', True)
        ).lower(),
        'HIGH_CONFIDENCE_WEIGHT': str(
            config.get('high_confidence_weight', 1.0)
        ),
        'LOW_CONFIDENCE_WEIGHT': str(
            config.get('low_confidence_weight', 0.3)
        ),
        'LEARNING_PHASE_SKIP_WEIGHT': str(
            config.get('learning_phase_skip_weight', 0.0)
        ),

        # --- Prediction Metrics ---
        'PREDICTION_METRICS_ENABLED': str(
            config.get('prediction_metrics_enabled', True)
        ).lower(),
        'METRICS_WINDOW_1H': str(config.get('metrics_window_1h', 12)),
        'METRICS_WINDOW_6H': str(config.get('metrics_window_6h', 72)),
        'METRICS_WINDOW_24H': str(config.get('metrics_window_24h', 288)),
        'PREDICTION_ACCURACY_THRESHOLD': str(
            config.get('prediction_accuracy_threshold', 0.3)
        ),
        'MAE_ENTITY_ID': config.get('mae_entity', 'sensor.ml_model_mae'),
        'RMSE_ENTITY_ID': config.get('rmse_entity', 'sensor.ml_model_rmse'),

        # --- Trajectory Prediction ---
        'TRAJECTORY_PREDICTION_ENABLED': str(
            config.get('trajectory_prediction_enabled', True)
        ).lower(),
        'WEATHER_FORECAST_INTEGRATION': str(
            config.get('weather_forecast_integration', True)
        ).lower(),
        'PV_FORECAST_INTEGRATION': str(
            config.get('pv_forecast_integration', True)
        ).lower(),
        'SOLAR_CORRECTION_ENABLED': str(
            config.get('solar_correction_enabled', True)
        ).lower(),
        'CLOUD_COVER_CORRECTION_ENABLED': str(
            config.get('cloud_cover_correction_enabled', False)
        ).lower(),
        'OVERSHOOT_DETECTION_ENABLED': str(
            config.get('overshoot_detection_enabled', True)
        ).lower(),

        # --- Multi-Lag Learning ---
        'ENABLE_MULTI_LAG_LEARNING': str(
            config.get('enable_multi_lag_learning', True)
        ).lower(),
        'PV_LAG_STEPS': str(config.get('pv_lag_steps', 4)),
        'FIREPLACE_LAG_STEPS': str(config.get('fireplace_lag_steps', 4)),
        'TV_LAG_STEPS': str(config.get('tv_lag_steps', 2)),

        # --- Seasonal Adaptation ---
        'ENABLE_SEASONAL_ADAPTATION': str(
            config.get('enable_seasonal_adaptation',
                       config.get('seasonal_learning_enabled', True))
        ).lower(),
        'SEASONAL_LEARNING_RATE': str(
            config.get('seasonal_learning_rate', 0.01)
        ),
        'MIN_SEASONAL_SAMPLES': str(
            config.get('min_seasonal_samples', 100)
        ),

        # --- Summer Learning ---
        'ENABLE_SUMMER_LEARNING': str(
            config.get('enable_summer_learning', True)
        ).lower(),

        # --- Historical Calibration ---
        'STABILITY_TEMP_CHANGE_THRESHOLD': str(
            config.get('stability_temp_change_threshold', 0.1)
        ),
        'MIN_STABLE_PERIOD_MINUTES': str(
            config.get('min_stable_period_minutes', 30)
        ),
        'OPTIMIZATION_METHOD': config.get('optimization_method', 'L-BFGS-B'),
        'PV_CALIBRATION_INDOOR_CEILING': str(
            config.get('pv_calibration_indoor_ceiling', 23.0)
        ),

        # --- Delta Forecast Calibration ---
        'ENABLE_DELTA_FORECAST_CALIBRATION': str(
            config.get('enable_delta_forecast_calibration', True)
        ).lower(),
        'DELTA_CALIBRATION_MAX_OFFSET': str(
            config.get('delta_calibration_max_offset', 10.0)
        ),

        # --- Learning History ---
        'MAX_PREDICTION_HISTORY': str(
            config.get('max_prediction_history', 700)
        ),
        'MAX_PARAMETER_HISTORY': str(
            config.get('max_parameter_history', 700)
        ),

        # --- Indoor Trend Protection ---
        'INDOOR_COOLING_TREND_THRESHOLD': str(
            config.get('indoor_cooling_trend_threshold', -0.05)
        ),
        'INDOOR_COOLING_DAMPING_FACTOR': str(
            config.get('indoor_cooling_damping_factor', 0.3)
        ),
        'INDOOR_WARMING_TREND_THRESHOLD': str(
            config.get('indoor_warming_trend_threshold', 0.10)
        ),
        'INDOOR_WARMING_DAMPING_FACTOR': str(
            config.get('indoor_warming_damping_factor', 0.3)
        ),

        # --- Heat Source Channels ---
        'ENABLE_HEAT_SOURCE_CHANNELS': str(
            config.get('enable_heat_source_channels', True)
        ).lower(),
        'ENABLE_MIXED_SOURCE_ATTRIBUTION': str(
            config.get('enable_mixed_source_attribution', False)
        ).lower(),

        # --- Electricity Price Optimization ---
        'ELECTRICITY_PRICE_ENABLED': str(
            config.get('electricity_price_enabled', False)
        ).lower(),
        'PRICE_CHEAP_PERCENTILE': str(
            config.get('price_cheap_percentile', 33)
        ),
        'PRICE_EXPENSIVE_PERCENTILE': str(
            config.get('price_expensive_percentile', 67)
        ),
        'PRICE_TARGET_OFFSET': str(
            config.get('price_target_offset', 0.2)
        ),
        'PRICE_EXPENSIVE_OVERSHOOT': str(
            config.get('price_expensive_overshoot', 0.2)
        ),
        'PRICE_CACHE_REFRESH_MINUTES': str(
            config.get('price_cache_refresh_minutes', 60)
        ),
        'PV_SURPLUS_CHEAP_ENABLED': str(
            config.get('pv_surplus_cheap_enabled', False)
        ).lower(),
        'PV_SURPLUS_CHEAP_THRESHOLD_W': str(
            config.get('pv_surplus_cheap_threshold_w', 3000)
        ),
        'MIN_SETPOINT_HOLD_CYCLES': str(
            config.get('min_setpoint_hold_cycles', 4)
        ),

        # --- Outlet Smoothing ---
        'OUTLET_SMOOTHING_ALPHA': str(
            config.get('outlet_smoothing_alpha', 0.3)
        ),
        'OUTLET_SMOOTHING_BYPASS': str(
            config.get('outlet_smoothing_bypass', 2.0)
        ),
    }

    # Set environment variables for the ML system
    for key, value in env_vars.items():
        if value is not None and value != '':
            os.environ[key] = str(value)

    # Write env vars to a shell-sourceable file so that child processes
    # started later by run.sh (e.g. ``python3 -m src.main``) inherit them.
    env_file_path = "/data/config/env_vars"
    try:
        os.makedirs(os.path.dirname(env_file_path), exist_ok=True)
        with open(env_file_path, "w") as fh:
            for key, value in env_vars.items():
                if value is not None and value != '':
                    fh.write(f"export {key}={shlex.quote(str(value))}\n")
        # Restrict permissions – the file may contain tokens.
        os.chmod(env_file_path, 0o600)
        log_info(f"Environment file written to {env_file_path}")
    except Exception as e:
        log_warning(f"Failed to write environment file: {e}")

    log_info(
        f"Set {len([v for v in env_vars.values() if v])} environment variables"
    )
    return env_vars


def import_existing_model(source_path):
    """Import existing unified state from standalone installation"""
    try:
        if not os.path.exists(source_path):
            log_warning(f"State file not found: {source_path}")
            return False
            
        # Backup existing state if present
        target_path = '/data/models/unified_thermal_state.json'
        if os.path.exists(target_path):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"pre_import_{timestamp}.json"
            backup_path = f"/data/backups/{backup_name}"
            shutil.copy(target_path, backup_path)
            log_info(f"Backed up existing state to {backup_path}")

        # Import new state
        shutil.copy(source_path, target_path)
        log_info(f"State imported successfully from {source_path}")
        return True

    except Exception as e:
        log_error(f"State import failed: {e}")
        return False


def validate_configuration(config):
    """Validate critical configuration settings"""
    required_entities = [
        ('target_indoor_temp_entity', 'Target Indoor Temperature Entity'),
        ('indoor_temp_entity', 'Indoor Temperature Entity'),
        ('outdoor_temp_entity', 'Outdoor Temperature Entity'), 
        ('heating_control_entity', 'Heating Control Entity'),
        ('outlet_temp_entity', 'Outlet Temperature Entity')
    ]
    
    missing_entities = []
    for key, description in required_entities:
        if not config.get(key) or config.get(key).strip() == '':
            missing_entities.append(description)
    
    if missing_entities:
        log_error("Missing required entity configurations:")
        for entity in missing_entities:
            log_error(f"  - {entity}")
        log_error(
            "Please configure all required entities in the add-on settings"
        )
        return False

    # Validate numeric ranges
    learning_rate = config.get('adaptive_learning_rate', 0.01)
    if not (0.001 <= learning_rate <= 0.1):
        log_warning(
            f"Learning rate {learning_rate} outside recommended range "
            "[0.001, 0.1]"
        )

    max_temp = config.get('safety_max_temp', 25.0)
    min_temp = config.get('safety_min_temp', 18.0)
    if max_temp <= min_temp:
        log_error(
            f"Safety max temp ({max_temp}) must be greater than "
            f"min temp ({min_temp})"
        )
        return False

    log_info("Configuration validation passed")
    return True


def save_config_backup(config):
    """Save configuration backup for troubleshooting"""
    try:
        config_backup = {
            'timestamp': datetime.now().isoformat(),
            'addon_version': '1.0.0',
            'config': config
        }
        
        backup_path = '/data/config/addon_config_backup.json'
        with open(backup_path, 'w') as f:
            json.dump(config_backup, f, indent=2)
        
        log_info(f"Configuration backup saved to {backup_path}")
        
    except Exception as e:
        log_warning(f"Failed to save config backup: {e}")


def initialize_addon_environment():
    """Initialize add-on environment for ML system"""
    try:
        log_info("Initializing ML Heating Add-on environment...")
        
        # Load add-on configuration
        addon_config = load_addon_config()
        
        # Validate configuration
        if not validate_configuration(addon_config):
            log_error("Configuration validation failed")
            return False
        
        # Setup directories
        setup_data_directories()
        
        # Save config backup
        save_config_backup(addon_config)
        
        # Convert to environment variables
        env_vars = convert_addon_to_env(addon_config)

        # Import existing model if specified
        if (addon_config.get('import_existing_model') and
                addon_config.get('existing_model_path')):
            import_existing_model(addon_config['existing_model_path'])

        log_info("Add-on environment initialized successfully")
        return env_vars

    except Exception as e:
        log_error(f"Add-on environment initialization failed: {e}")
        return False


def main():
    """Main configuration adapter function"""
    try:
        log_info("ML Heating Add-on Configuration Adapter Starting...")
        
        # Initialize environment
        env_vars = initialize_addon_environment()
        
        if not env_vars:
            log_error("Environment initialization failed")
            sys.exit(1)
        
        log_info("Configuration adapter completed successfully")
        
        # Print summary for debugging
        addon_config = load_addon_config()
        log_info("=== Configuration Summary ===")
        log_info(
            f"Target Indoor Temp Entity: "
            f"{addon_config.get('target_indoor_temp_entity')}"
        )
        log_info(
            f"Indoor Temp Entity: {addon_config.get('indoor_temp_entity')}"
        )
        log_info(
            f"Outdoor Temp Entity: {addon_config.get('outdoor_temp_entity')}"
        )
        log_info(f"Learning Rate: {addon_config.get('adaptive_learning_rate')}")
        log_info(
            f"Cycle Interval: "
            f"{addon_config.get('cycle_interval_minutes')} minutes"
        )
        log_info("=============================")

    except Exception as e:
        log_error(f"Configuration adapter failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
