"""
This module is the central entry point and main control loop for the
application.

It orchestrates the entire process of data collection, prediction, and action
using the enhanced physics-based heating model. The script operates in a
continuous loop, performing the following key steps in each iteration:

1.  **Initialization**: Loads the physics model and application state.
2.  **Data Fetching**: Gathers the latest sensor data from Home Assistant.
3.  **Feature Engineering**: Builds a feature set from current and historical
    data.
4.  **Prediction**: Uses the physics model to find the optimal heating
    temperature.
5.  **Action**: Sets the new target temperature in Home Assistant.
6.  **State Persistence**: Saves the current state for the next cycle.
"""
import argparse
import logging
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from . import config
from .thermal_constants import PhysicsConstants
from .ha_client import create_ha_client, get_sensor_attributes
from .influx_service import create_influx_service
from .physics_calibration import (
    train_thermal_equilibrium_model,
    validate_thermal_model,
)
from .state_manager import load_state
from .heating_controller import (
    BlockingStateManager,
    HeatingSystemStateChecker,
)
from .sensor_buffer import SensorBuffer
from .shadow_mode import get_shadow_output_entity_id
from .hlc_learner import calibrate_hlc
from .cycle_state import CycleState, determine_cycle_state


def _bool_arg(parsed_args, name: str) -> bool:
    value = getattr(parsed_args, name, False)
    return value if isinstance(value, bool) else False


def _str_arg(parsed_args, name: str) -> str | None:
    value = getattr(parsed_args, name, None)
    return value if isinstance(value, str) else None


def main():
    """
    The main function that orchestrates the heating control logic.

    This function initializes the system, enters a continuous loop to
    monitor and control the heating, and handles command-line arguments
    for modes like initial training.
    """
    parser = argparse.ArgumentParser(description="Heating Controller")
    parser.add_argument(
        "--calibrate-physics",
        action="store_true",
        help="Calibrate the physics model using the method set in CALIBRATION_METHOD config.",
    )
    parser.add_argument(
        "--calibrate-physics-direct",
        action="store_true",
        help="Calibrate the physics model using the physics-direct analytical path (no scipy).",
    )
    parser.add_argument(
        "--calibrate-physics-export-only",
        action="store_true",
        help="Export calibration data to CSV and exit (no optimisation).",
    )
    parser.add_argument(
        "--validate-physics",
        action="store_true",
        help="Test model behavior and exit.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--list-backups", action="store_true", help="List available backups."
    )
    parser.add_argument(
        "--restore-backup", type=str, help="Restore from a backup file."
    )
    parser.add_argument(
        "--calibrate-cooling-ml",
        action="store_true",
        help="Train the LightGBM overheating classifier for ML-based pre-cooling and exit.",
    )
    parser.add_argument(
        "--calibrate-cooling-physics",
        action="store_true",
        help="Calibrate the cooling thermal equilibrium model from warm-season data and exit.",
    )
    parser.add_argument(
        "--calibrate-heating-correction-ml",
        action="store_true",
        help="Train the LightGBM heating-correction regressor and exit.",
    )
    parser.add_argument(
        "--calibrate-cooling-correction-ml",
        action="store_true",
        help="Train the LightGBM cooling-correction regressor and exit.",
    )
    args = parser.parse_args()
    # Load environment variables and configure logging.
    load_dotenv()
    log_level = (
        logging.DEBUG if _bool_arg(args, "debug") or config.DEBUG else logging.INFO
    )

    # Configure logging to ensure output goes to stdout for systemd capture
    import sys

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,  # Explicitly output to stdout for systemd
        force=True,  # Force reconfigure if already configured
    )

    # Suppress verbose logging from underlying libraries.
    logging.getLogger("requests").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.INFO)

    # --- Initialization ---
    # ThermalEquilibriumModel is now loaded directly in model_wrapper.py
    # Shadow mode comparison metrics (no longer tracking MAE/RMSE)
    shadow_ml_error_sum = 0.0
    shadow_hc_error_sum = 0.0
    shadow_comparison_count = 0

    influx_service = create_influx_service()

    # --- HLC Calibration Flag Detection ---
    _hlc_flag = "/data/config/hlc_calibrate_flag"
    if os.path.exists(_hlc_flag):
        logging.info("🔬 HLC calibrate flag detected — running one-shot calibration")
        try:
            os.remove(_hlc_flag)
        except OSError as _flag_err:
            logging.error(
                "❌ Could not remove HLC flag %s — skipping calibration "
                "to avoid infinite loop: %s", _hlc_flag, _flag_err
            )
            _hlc_flag = None  # signal: skip calibration
        if _hlc_flag is not None:
            result = calibrate_hlc(influx_service=influx_service)
            if result.get("success"):
                logging.info("✅ %s", result["message"])
            else:
                logging.warning("⚠️ HLC calibration failed: %s", result.get("message"))

    # --- Cooling ML Calibration Flag Detection ---
    _cooling_ml_flag = "/data/config/calibrate_cooling_ml_flag"
    if os.path.exists(_cooling_ml_flag):
        logging.info("🤖 Cooling ML calibrate flag detected — running LGBM training")
        try:
            os.remove(_cooling_ml_flag)
        except OSError as _flag_err:
            logging.error(
                "❌ Could not remove cooling ML flag %s — skipping to avoid loop: %s",
                _cooling_ml_flag, _flag_err,
            )
            _cooling_ml_flag = None
        if _cooling_ml_flag is not None:
            try:
                from .cooling_ml_calibration import calibrate_cooling_ml
                _ok = calibrate_cooling_ml()
                if _ok:
                    logging.info("✅ Cooling ML model calibrated successfully")
                else:
                    logging.error("❌ Cooling ML calibration failed — check logs")
            except Exception as _cml_err:
                logging.error("❌ Cooling ML calibration error: %s", _cml_err, exc_info=True)

    # --- Cooling Physics Calibration Flag Detection ---
    _cooling_physics_flag = "/data/config/calibrate_cooling_physics_flag"
    if os.path.exists(_cooling_physics_flag):
        logging.info(
            "🔬 Cooling physics calibrate flag detected — running cooling thermal calibration"
        )
        try:
            os.remove(_cooling_physics_flag)
        except OSError as _flag_err:
            logging.error(
                "❌ Could not remove cooling physics flag %s — skipping to avoid loop: %s",
                _cooling_physics_flag, _flag_err,
            )
            _cooling_physics_flag = None
        if _cooling_physics_flag is not None:
            try:
                from .physics_calibration_cooling import calibrate_cooling_physics
                _ok = calibrate_cooling_physics()
                if _ok:
                    logging.info("✅ Cooling physics model calibrated successfully")
                else:
                    logging.error("❌ Cooling physics calibration failed — check logs")
            except Exception as _cp_err:
                logging.error(
                    "❌ Cooling physics calibration error: %s", _cp_err, exc_info=True
                )

    # --- Heating Correction ML Calibration Flag Detection ---
    _heating_ml_flag = "/data/config/calibrate_heating_correction_ml_flag"
    if os.path.exists(_heating_ml_flag):
        logging.info(
            "🤖 Heating correction ML calibrate flag detected — running LGBM training"
        )
        try:
            os.remove(_heating_ml_flag)
        except OSError as _flag_err:
            logging.error(
                "❌ Could not remove heating ML flag %s — skipping to avoid loop: %s",
                _heating_ml_flag, _flag_err,
            )
            _heating_ml_flag = None
        if _heating_ml_flag is not None:
            try:
                from .heating_correction_ml_calibration import (
                    calibrate_heating_correction_ml,
                )
                _ok = calibrate_heating_correction_ml()
                if _ok:
                    logging.info(
                        "✅ Heating correction ML model calibrated successfully"
                    )
                else:
                    logging.error(
                        "❌ Heating correction ML calibration failed — check logs"
                    )
            except Exception as _hml_err:
                logging.error(
                    "❌ Heating correction ML calibration error: %s",
                    _hml_err, exc_info=True,
                )

    # --- Cooling Correction ML Calibration Flag Detection ---
    _cooling_ml_corr_flag = "/data/config/calibrate_cooling_correction_ml_flag"
    if os.path.exists(_cooling_ml_corr_flag):
        logging.info(
            "🧊 Cooling correction ML calibrate flag detected — running LGBM training"
        )
        try:
            os.remove(_cooling_ml_corr_flag)
        except OSError as _flag_err:
            logging.error(
                "❌ Could not remove cooling ML correction flag %s — skipping to avoid loop: %s",
                _cooling_ml_corr_flag, _flag_err,
            )
            _cooling_ml_corr_flag = None
        if _cooling_ml_corr_flag is not None:
            try:
                from .cooling_correction_ml_calibration import (
                    calibrate_cooling_correction_ml,
                )
                _ok = calibrate_cooling_correction_ml()
                if _ok:
                    logging.info(
                        "✅ Cooling correction ML model calibrated successfully"
                    )
                else:
                    logging.error(
                        "❌ Cooling correction ML calibration failed — check logs"
                    )
            except Exception as _cml_err:
                logging.error(
                    "❌ Cooling correction ML calibration error: %s",
                    _cml_err, exc_info=True,
                )

    # --- Physics-Direct Calibration Flag Detection ---
    _physics_direct_flag = "/data/config/calibrate_physics_direct_flag"
    if os.path.exists(_physics_direct_flag):
        logging.info(
            "🔬 Physics-direct calibrate flag detected — running analytics calibration"
        )
        try:
            os.remove(_physics_direct_flag)
        except OSError as _flag_err:
            logging.error(
                "❌ Could not remove physics-direct flag %s — skipping "
                "to avoid infinite loop: %s", _physics_direct_flag, _flag_err
            )
            _physics_direct_flag = None
        if _physics_direct_flag is not None:
            try:
                from .physics_calibration_direct import calibrate_thermal_model_physics
                result_model = calibrate_thermal_model_physics()
                if result_model is not None:
                    logging.info(
                        "✅ Physics-direct calibration completed — "
                        "restart the service to apply the new parameters"
                    )
                else:
                    logging.error("❌ Physics-direct calibration returned None")
            except Exception as _phys_err:
                logging.error(
                    "❌ Physics-direct calibration error: %s", _phys_err, exc_info=True
                )

    # --- Scipy Heating Thermal Calibration Flag Detection ---
    _recalibrate_flag = "/data/config/recalibrate_flag"
    if os.path.exists(_recalibrate_flag):
        logging.info(
            "🔬 Recalibrate flag detected — running scipy heating thermal calibration"
        )
        try:
            os.remove(_recalibrate_flag)
        except OSError as _flag_err:
            logging.error(
                "❌ Could not remove recalibrate flag %s — skipping "
                "to avoid infinite loop: %s", _recalibrate_flag, _flag_err
            )
            _recalibrate_flag = None
        if _recalibrate_flag is not None:
            try:
                from .physics_calibration import (
                    backup_existing_calibration,
                    train_thermal_equilibrium_model,
                )
                backup_path = backup_existing_calibration()
                if backup_path:
                    logging.info(
                        "✅ Previous thermal state backed up: %s",
                        os.path.basename(backup_path),
                    )
                _ok = train_thermal_equilibrium_model(method="scipy")
                if _ok:
                    logging.info("✅ Scipy heating thermal calibration completed successfully")
                else:
                    logging.error("❌ Scipy heating thermal calibration failed — check logs")
            except Exception as _recal_err:
                logging.error(
                    "❌ Scipy heating thermal calibration error: %s", _recal_err, exc_info=True
                )

    # --- InfluxDB Write Permission Check ---
    # Verify early that the token can write to the features bucket
    try:
        influx_service.check_write_permission()
    except Exception as e:
        logging.warning("InfluxDB write permission check skipped: %s", e)

    # --- Sensor Buffer Initialization ---
    # Initialize the circular buffer for sensor smoothing
    sensor_buffer = SensorBuffer(max_age_minutes=120)

    # Hydrate buffer from InfluxDB (Startup only)
    try:
        logging.info("💧 Hydrating sensor buffer from InfluxDB...")
        # Define sensors to hydrate
        hydration_sensors = [
            config.INDOOR_TEMP_ENTITY_ID,
            config.ACTUAL_OUTLET_TEMP_ENTITY_ID,
            config.TARGET_OUTLET_TEMP_ENTITY_ID,
            config.OUTDOOR_TEMP_ENTITY_ID,
            config.INLET_TEMP_ENTITY_ID,
            config.FLOW_RATE_ENTITY_ID,
        ]

        # Fetch raw history
        history_data = influx_service.fetch_recent_history(
            hydration_sensors,
            lookback_minutes=120
        )

        # Hydrate the buffer
        sensor_buffer.hydrate(history_data)
        logging.info("✅ Sensor buffer hydrated successfully")

    except Exception as e:
        logging.warning(
            f"⚠️ Buffer hydration failed: {e}. "
            "Starting with empty buffer (Cold Start Mode)."
        )

    # --- Shadow Mode Status ---
    if config.SHADOW_MODE:
        logging.info(
            "🔍 SHADOW MODE ENABLED: ML will observe and learn without "
            "affecting heating control"
        )
        logging.info("   - ML predictions calculated but not sent to HA")
        logging.info(
            "   - No HA sensor updates (confidence, MAE, RMSE, state)"
        )
        logging.info(
            "   - Learning from heat curve's actual control decisions"
        )
        logging.info("   - Performance comparison logging active")
    else:
        logging.info("🎯 ACTIVE MODE: ML actively controls heating system")

    # --- Thermal Model Calibration ---
    if _bool_arg(args, "calibrate_physics"):
        try:
            from .physics_calibration import backup_existing_calibration

            logging.info("=== CALIBRATING THERMAL EQUILIBRIUM MODEL ===")

            # Create backup before calibration
            logging.info("Step 0: Creating backup before calibration...")
            backup_path = backup_existing_calibration()
            if backup_path:

                logging.info(
                    "✅ Previous thermal state backed up: %s",
                    os.path.basename(backup_path),
                )
            else:
                logging.info("ℹ️ No existing thermal state found to backup")

            # Use CALIBRATION_METHOD from config as the default; --calibrate-physics
            # respects the configured method.
            result = train_thermal_equilibrium_model(
                method=getattr(config, "CALIBRATION_METHOD", "scipy")
            )
            if result:
                logging.info("✅ Thermal model calibrated successfully!")
                logging.info(
                    "🔄 Restart ml_heating to use trained thermal model"
                )
            else:
                logging.error("❌ Thermal model calibration failed")
        except Exception as e:
            logging.error(
                "Thermal model calibration error: %s", e, exc_info=True
            )
        return

    # --- Physics-Direct Calibration (explicit CLI flag) ---
    if _bool_arg(args, "calibrate_physics_direct"):
        try:
            from .physics_calibration_direct import calibrate_thermal_model_physics

            logging.info("=== CALIBRATING THERMAL EQUILIBRIUM MODEL (PHYSICS-DIRECT) ===")
            result = calibrate_thermal_model_physics()
            if result:
                logging.info("✅ Physics-direct calibration completed successfully!")
                logging.info("🔄 Restart ml_heating to use trained thermal model")
            else:
                logging.error("❌ Physics-direct calibration failed")
        except Exception as e:
            logging.error(
                "Physics-direct calibration error: %s", e, exc_info=True
            )
        return

    # --- Export Calibration Data Only ---
    if _bool_arg(args, "calibrate_physics_export_only"):
        try:
            from .physics_calibration import (
                fetch_historical_data_for_calibration,
            )
            import json as _json

            export_dir = os.path.dirname(config.UNIFIED_STATE_FILE)
            os.makedirs(export_dir, exist_ok=True)

            logging.info("=== EXPORTING CALIBRATION DATA ===")
            logging.info("Export directory: %s", export_dir)

            # 1. Fetch training data
            df = fetch_historical_data_for_calibration(
                lookback_hours=config.TRAINING_LOOKBACK_HOURS,
            )
            if df is None or df.empty:
                logging.error("❌ No calibration data available")
                return

            csv_path = os.path.join(export_dir, "calibration_data.csv")
            df.to_csv(csv_path, index=False)
            logging.info(
                "✅ Exported %d rows × %d cols to %s",
                len(df), len(df.columns), csv_path,
            )

            # 2. Export config values needed by standalone calibration
            config_export = {
                "INDOOR_TEMP_ENTITY_ID": config.INDOOR_TEMP_ENTITY_ID,
                "ACTUAL_OUTLET_TEMP_ENTITY_ID": config.ACTUAL_OUTLET_TEMP_ENTITY_ID,
                "ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID": config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID,
                "OUTDOOR_TEMP_ENTITY_ID": config.OUTDOOR_TEMP_ENTITY_ID,
                "PV_POWER_ENTITY_ID": config.PV_POWER_ENTITY_ID,
                "TV_STATUS_ENTITY_ID": config.TV_STATUS_ENTITY_ID,
                "FIREPLACE_STATUS_ENTITY_ID": config.FIREPLACE_STATUS_ENTITY_ID,
                "INLET_TEMP_ENTITY_ID": config.INLET_TEMP_ENTITY_ID,
                "FLOW_RATE_ENTITY_ID": config.FLOW_RATE_ENTITY_ID,
                "POWER_CONSUMPTION_ENTITY_ID": config.POWER_CONSUMPTION_ENTITY_ID,
                "DHW_STATUS_ENTITY_ID": config.DHW_STATUS_ENTITY_ID,
                "DEFROST_STATUS_ENTITY_ID": config.DEFROST_STATUS_ENTITY_ID,
                "DISINFECTION_STATUS_ENTITY_ID": config.DISINFECTION_STATUS_ENTITY_ID,
                "DHW_BOOST_HEATER_STATUS_ENTITY_ID": config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
                "LIVING_ROOM_TEMP_ENTITY_ID": getattr(config, "LIVING_ROOM_TEMP_ENTITY_ID", ""),
                "SPECIFIC_HEAT_CAPACITY": float(config.SPECIFIC_HEAT_CAPACITY),
                "GRACE_PERIOD_MAX_MINUTES": float(config.GRACE_PERIOD_MAX_MINUTES),
                "CLOUD_COVER_CORRECTION_ENABLED": bool(
                    getattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False),
                ),
                "PV_CALIBRATION_INDOOR_CEILING": float(
                    getattr(config, "PV_CALIBRATION_INDOOR_CEILING", 23.0),
                ),
                "TRAINING_LOOKBACK_HOURS": int(config.TRAINING_LOOKBACK_HOURS),
            }
            cfg_path = os.path.join(export_dir, "calibration_config.json")
            with open(cfg_path, "w") as f:
                _json.dump(config_export, f, indent=2)
            logging.info("✅ Exported config to %s", cfg_path)

            # 3. Copy unified thermal state if it exists
            state_src = config.UNIFIED_STATE_FILE
            if os.path.exists(state_src):
                logging.info(
                    "✅ Unified thermal state already at %s", state_src,
                )
            else:
                logging.info(
                    "ℹ️ No unified thermal state found at %s", state_src,
                )

            logging.info("=== EXPORT COMPLETE ===")
            logging.info(
                "Copy these files to your laptop and run:\n"
                "  python physics_calibration_standalone.py "
                "--data %s --config %s",
                csv_path, cfg_path,
            )
        except Exception as e:
            logging.error(
                "Calibration export error: %s", e, exc_info=True,
            )
        return

    # --- Cooling ML Calibration (CLI) ---
    if _bool_arg(args, "calibrate_cooling_ml"):
        logging.info("=== COOLING ML CALIBRATION (CLI) ===")
        try:
            from .cooling_ml_calibration import calibrate_cooling_ml
            _ok = calibrate_cooling_ml()
            if _ok:
                logging.info("✅ Cooling ML model trained and saved successfully")
            else:
                logging.error("❌ Cooling ML calibration failed")
        except Exception as _cml_exc:
            logging.error("Cooling ML calibration error: %s", _cml_exc, exc_info=True)
        return

    # --- Cooling Physics Calibration (CLI) ---
    if _bool_arg(args, "calibrate_cooling_physics"):
        logging.info("=== COOLING PHYSICS CALIBRATION (CLI) ===")
        try:
            from .physics_calibration_cooling import calibrate_cooling_physics
            _ok = calibrate_cooling_physics()
            if _ok:
                logging.info("✅ Cooling physics model calibrated successfully")
            else:
                logging.error("❌ Cooling physics calibration failed")
        except Exception as _cp_exc:
            logging.error("Cooling physics calibration error: %s", _cp_exc, exc_info=True)
        return

    # --- Heating Correction ML Calibration (CLI) ---
    if _bool_arg(args, "calibrate_heating_correction_ml"):
        logging.info("=== HEATING CORRECTION ML CALIBRATION (CLI) ===")
        try:
            from .heating_correction_ml_calibration import (
                calibrate_heating_correction_ml,
            )
            _ok = calibrate_heating_correction_ml()
            if _ok:
                logging.info(
                    "✅ Heating correction ML model trained and saved successfully"
                )
            else:
                logging.error("❌ Heating correction ML calibration failed")
        except Exception as _hml_exc:
            logging.error(
                "Heating correction ML calibration error: %s",
                _hml_exc, exc_info=True,
            )
        return

    # --- Cooling Correction ML Calibration (CLI) ---
    if _bool_arg(args, "calibrate_cooling_correction_ml"):
        logging.info("=== COOLING CORRECTION ML CALIBRATION (CLI) ===")
        try:
            from .cooling_correction_ml_calibration import (
                calibrate_cooling_correction_ml,
            )
            _ok = calibrate_cooling_correction_ml()
            if _ok:
                logging.info(
                    "✅ Cooling correction ML model trained and saved successfully"
                )
            else:
                logging.error("❌ Cooling correction ML calibration failed")
        except Exception as _cml_exc:
            logging.error(
                "Cooling correction ML calibration error: %s",
                _cml_exc, exc_info=True,
            )
        return

    # --- Thermal Model Validation ---
    if _bool_arg(args, "validate_physics"):
        try:
            result = validate_thermal_model()
            if result:
                logging.info("✅ Thermal model validation passed!")
            else:
                logging.error("❌ Thermal model validation failed!")
        except Exception as e:
            logging.error(
                "Thermal model validation error: %s", e, exc_info=True
            )
        return

    if _bool_arg(args, "list_backups"):
        from .unified_thermal_state import get_thermal_state_manager
        import json
        state_manager = get_thermal_state_manager()
        backups = state_manager.list_backups()
        if backups:
            print("Available backups:")
            # print backups in a json format so it is easy to parse
            print(json.dumps(backups, indent=2, default=str))
        else:
            print("No backups found.")
        return

    restore_backup = _str_arg(args, "restore_backup")
    if restore_backup:
        from .unified_thermal_state import get_thermal_state_manager
        state_manager = get_thermal_state_manager()
        success, message = state_manager.restore_from_backup(
            restore_backup
        )
        if success:
            print(f"Successfully restored from backup: {restore_backup}")
            print(message)
        else:
            print(f"Failed to restore from backup: {restore_backup}")
            print(message)
        return

    # --- Main Control Loop ---
    # This loop runs indefinitely, performing one full cycle of learning and
    # prediction every CYCLE_INTERVAL_MINUTES.

    from .pre_dispatch import (
        initialize_loop_state,
        update_sensor_buffer_and_thermo,
        resolve_shadow_mode_for_cycle,
        run_online_learning,
        handle_grace_period,
        validate_sensors_once,
        emit_network_error_state,
        check_blocking_state,
        check_and_resolve_climate_mode,
    )
    from .cycle_context import CycleContext
    from .cycle_routes import (
        run_blocking_route,
        run_grace_period_route,
        run_idle_route,
        run_heating_route,
        run_cooling_route,
    )

    loop = initialize_loop_state(
        sensor_buffer=sensor_buffer,
        influx_service=influx_service,
    )

    # Apply the mode-specific feature profile for this process lifetime.
    # This overlays heating_profile / cooling_profile settings from options.json
    # onto the config module globals before the first cycle begins.
    from .mode_profiles import apply_profile as _apply_mode_profile
    _apply_mode_profile(loop.wrapper.climate_mode)

    while True:
        try:
            # --- Cycle start ---
            cycle_number, cycle_start_time, cycle_start_datetime = (
                loop.increment_cycle()
            )

            if loop.last_cycle_end_time is not None:
                interval_since_last = cycle_start_time - loop.last_cycle_end_time
                logging.debug(
                    "🔄 CYCLE %d START: %s (interval: %.1fmin since last cycle)",
                    cycle_number,
                    cycle_start_datetime.strftime("%H:%M:%S"),
                    interval_since_last / 60,
                )
            else:
                logging.debug(
                    "🔄 CYCLE %d START: %s (first cycle)",
                    cycle_number,
                    cycle_start_datetime.strftime("%H:%M:%S"),
                )

            # --- Fetch HA data ---
            from .model_wrapper import get_enhanced_model_wrapper as _get_wrapper

            ha_client = create_ha_client()
            all_states = ha_client.get_all_states()

            # --- Network error check (must be before anything using all_states) ---
            if not all_states:
                logging.warning(
                    "Could not fetch states from HA, skipping cycle."
                )
                emit_network_error_state(ha_client)
                time.sleep(PhysicsConstants.RETRY_DELAY_SECONDS)
                continue

            # --- Early climate mode detection ---
            # Detect climate mode BEFORE loading state so the correct
            # state file (heating vs cooling) is used from the start.
            _early_checker = HeatingSystemStateChecker()
            climate_mode = _early_checker.get_climate_mode(
                ha_client, all_states
            )
            _wrapper = _get_wrapper()
            _wrapper.set_climate_mode(climate_mode or "heating")
            # Normalize: set_climate_mode coerces unsupported values (e.g. "off")
            # to "heating", so read back the actual mode the wrapper is using.
            climate_mode = _wrapper.climate_mode
            _active_state_manager = _wrapper.state_manager

            # --- Load state from the mode-correct state file ---
            state = load_state(state_manager=_active_state_manager)

            # --- One-time sensor validation ---
            if not loop.sensor_validation_done:
                if validate_sensors_once(all_states, ha_client):
                    loop.sensor_validation_done = True

            # --- Sensor buffer + thermodynamic metrics ---
            thermodynamic_metrics_written = update_sensor_buffer_and_thermo(
                loop, ha_client, all_states, influx_service
            )

            # --- Shadow mode resolution ---
            shadow_mode, effective_shadow_mode = resolve_shadow_mode_for_cycle(
                ha_client, all_states
            )

            # --- Blocking check ---
            is_blocking, blocking_reasons = check_blocking_state(
                ha_client, all_states
            )

            # --- Online learning from previous cycle ---
            # Skip until at least one cycle has completed in this process.
            # `cycle_number > 1` is not sufficient because cycle 1 may have
            # exited early (e.g. network error → continue), leaving
            # last_cycle_end_time as None and persisted state stale.
            if loop.last_cycle_end_time is not None:
                run_online_learning(
                    ha_client=ha_client,
                    all_states=all_states,
                    state=state,
                    effective_shadow_mode=effective_shadow_mode,
                    climate_mode=climate_mode,
                    wrapper=loop.wrapper,
                )
            else:
                logging.debug(
                    "Skipping online learning on first cycle after boot"
                )

            # --- Grace period check ---
            is_grace_period = handle_grace_period(
                ha_client, state, effective_shadow_mode
            )

            if is_grace_period:
                # Dispatch to grace period route
                ctx = CycleContext(
                    cycle_number=cycle_number,
                    cycle_start_time=cycle_start_time,
                    cycle_start_datetime=cycle_start_datetime,
                    ha_client=ha_client,
                    influx_service=influx_service,
                    all_states=all_states,
                    state=state,
                    state_manager=_active_state_manager,
                    wrapper=loop.wrapper,
                    climate_mode=climate_mode,
                    is_blocking=is_blocking,
                    blocking_reasons=blocking_reasons,
                    effective_shadow_mode=effective_shadow_mode,
                    shadow_mode=shadow_mode,
                    sensor_buffer=loop.sensor_buffer,
                    blocking_entities=loop.blocking_entities,
                )
                run_grace_period_route(ctx)
                # Grace period skips normal dispatch — go to end-of-loop
            else:
                # --- Check heating active + resolve climate mode ---
                heating_active, climate_mode, _active_state_manager, reloaded_state = (
                    check_and_resolve_climate_mode(
                        ha_client, all_states, loop.wrapper
                    )
                )
                if reloaded_state is not None:
                    state = reloaded_state

                if not heating_active:
                    # --- IDLE dispatch ---
                    cycle_state = CycleState.IDLE
                    logging.debug("🔄 Cycle state: %s", cycle_state.value)

                    ctx = CycleContext(
                        cycle_number=cycle_number,
                        cycle_start_time=cycle_start_time,
                        cycle_start_datetime=cycle_start_datetime,
                        ha_client=ha_client,
                        influx_service=influx_service,
                        all_states=all_states,
                        state=state,
                        state_manager=_active_state_manager,
                        wrapper=loop.wrapper,
                        climate_mode=climate_mode,
                        is_blocking=False,
                        blocking_reasons=[],
                        effective_shadow_mode=effective_shadow_mode,
                        shadow_mode=shadow_mode,
                        last_indoor_temp=state.get("last_indoor_temp"),
                        sensor_buffer=loop.sensor_buffer,
                        cooling_ml_model=loop.cooling_ml_model,
                        cooling_obs_buffer=loop.cooling_obs_buffer,
                        cooling_ml_model_type=loop.cooling_ml_model_type,
                        heating_obs_buffer=loop.heating_obs_buffer,
                        blocking_entities=loop.blocking_entities,
                        thermodynamic_metrics_written_in_sensor_update=thermodynamic_metrics_written,
                    )
                    run_idle_route(ctx)
                elif is_blocking:
                    # --- BLOCKING dispatch ---
                    cycle_state = CycleState.BLOCKING
                    logging.debug("🔄 Cycle state: %s", cycle_state.value)

                    ctx = CycleContext(
                        cycle_number=cycle_number,
                        cycle_start_time=cycle_start_time,
                        cycle_start_datetime=cycle_start_datetime,
                        ha_client=ha_client,
                        influx_service=influx_service,
                        all_states=all_states,
                        state=state,
                        state_manager=_active_state_manager,
                        wrapper=loop.wrapper,
                        climate_mode=climate_mode,
                        is_blocking=True,
                        blocking_reasons=blocking_reasons,
                        effective_shadow_mode=effective_shadow_mode,
                        shadow_mode=shadow_mode,
                        sensor_buffer=loop.sensor_buffer,
                        blocking_entities=loop.blocking_entities,
                    )
                    run_blocking_route(ctx)
                else:
                    # --- HEATING or COOLING dispatch ---
                    cycle_state = determine_cycle_state(
                        is_blocking=False,
                        heating_active=True,
                        climate_mode=climate_mode,
                    )
                    logging.debug("🔄 Cycle state: %s", cycle_state.value)

                    ctx = CycleContext(
                        cycle_number=cycle_number,
                        cycle_start_time=cycle_start_time,
                        cycle_start_datetime=cycle_start_datetime,
                        ha_client=ha_client,
                        influx_service=influx_service,
                        all_states=all_states,
                        state=state,
                        state_manager=_active_state_manager,
                        wrapper=loop.wrapper,
                        climate_mode=climate_mode,
                        is_blocking=False,
                        blocking_reasons=[],
                        effective_shadow_mode=effective_shadow_mode,
                        shadow_mode=shadow_mode,
                        last_indoor_temp=state.get("last_indoor_temp"),
                        sensor_buffer=loop.sensor_buffer,
                        cooling_ml_model=loop.cooling_ml_model,
                        cooling_obs_buffer=loop.cooling_obs_buffer,
                        cooling_ml_model_type=loop.cooling_ml_model_type,
                        heating_obs_buffer=loop.heating_obs_buffer,
                        blocking_entities=loop.blocking_entities,
                        thermodynamic_metrics_written_in_sensor_update=thermodynamic_metrics_written,
                    )

                    if cycle_state == CycleState.COOLING:
                        run_cooling_route(ctx)
                    else:
                        run_heating_route(ctx)

        except Exception as e:
            logging.error("Error in main loop: %s", e, exc_info=True)
            try:
                ha_client = create_ha_client()
                heating_state_entity_id = get_shadow_output_entity_id(
                    "sensor.ml_heating_state"
                )
                attributes_state = get_sensor_attributes(
                    heating_state_entity_id
                )
                attributes_state.update(
                    {
                        "state_description": "Model error",
                        "last_error": str(e),
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                    }
                )
                ha_client.set_state(
                    heating_state_entity_id,
                    7,
                    attributes_state,
                    round_digits=None,
                )
            except Exception:
                logging.debug(
                    "Failed to write MODEL_ERROR state to HA.", exc_info=True
                )

        # --- Cycle end debug logging ---
        cycle_end_time = time.time()
        cycle_duration = cycle_end_time - cycle_start_time
        cycle_end_datetime = datetime.now()

        logging.debug(
            "✅ CYCLE %d END: %s (duration: %.1fs)",
            cycle_number,
            cycle_end_datetime.strftime("%H:%M:%S"),
            cycle_duration,
        )

        loop.last_cycle_end_time = cycle_end_time

        # Poll for blocking events during the idle period so defrost
        # starts/ends are detected quickly. This call will block until the
        # next cycle is due, or until a blocking event starts or ends.
        logging.debug(
            "💤 POLLING START: Waiting %dmin until next cycle...",
            PhysicsConstants.CYCLE_INTERVAL_MINUTES,
        )
        blocking_manager = BlockingStateManager()
        blocking_manager.poll_for_blocking(ha_client, state, sensor_buffer)

        poll_end_time = time.time()
        poll_duration = poll_end_time - cycle_end_time
        logging.debug(
            "⏰ POLLING END: Waited %.1fmin, starting next cycle...",
            poll_duration / 60,
        )


if __name__ == "__main__":
    main()

