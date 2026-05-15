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

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from . import config
from .thermal_constants import PhysicsConstants
from .physics_features import build_physics_features
from .ha_client import create_ha_client, get_sensor_attributes
from .influx_service import create_influx_service
from .model_wrapper import simplified_outlet_prediction
from .physics_calibration import (
    train_thermal_equilibrium_model,
    validate_thermal_model,
)
from .physics_features import calculate_thermodynamic_metrics
from .state_manager import load_state, save_state
from .heating_controller import (
    BlockingStateManager,
    SensorDataManager,
    HeatingSystemStateChecker,
)
from .sensor_buffer import SensorBuffer
from .shadow_mode import get_shadow_output_entity_id, resolve_shadow_mode
from .temperature_control import apply_ema_smoothing
from .hlc_learner import HLCSessionLearner, calibrate_hlc


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
        "--calibrate-hlc",
        action="store_true",
        help="Run one-shot HLC calibration from historical data and exit.",
    )
    parser.add_argument(
        "--calibrate-cooling-ml",
        action="store_true",
        help="Train the LightGBM overheating classifier for ML-based pre-cooling and exit.",
    )
    parser.add_argument(
        "--calibrate-heating-correction-ml",
        action="store_true",
        help="Train the LightGBM heating-correction regressor and exit.",
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

    # --- HLC Learner Initialization ---
    _hlc_session_learner: HLCSessionLearner | None = None
    if getattr(config, "HLC_SESSION_ENABLED", False):
        _hlc_session_learner = HLCSessionLearner()
        _restored = _hlc_session_learner.load_session_records()
        if _restored:
            logging.info(
                "📡 HLC session learner: loaded %d session records", _restored
            )
        else:
            logging.info("📡 HLC session learner: started fresh (no prior records)")

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

    # --- One-shot HLC Calibration ---
    if _bool_arg(args, "calibrate_hlc"):
        logging.info("=== ONE-SHOT HLC CALIBRATION ===")
        result = calibrate_hlc()
        if result.get("success"):
            logging.info("✅ %s", result["message"])
        else:
            logging.error("❌ HLC calibration failed: %s", result.get("message"))
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
    # prediction every 5 minutes.

    # Initialize the model and export initial metrics to HA
    from .model_wrapper import get_enhanced_model_wrapper

    wrapper = get_enhanced_model_wrapper()
    try:
        wrapper.export_metrics_to_ha()
        logging.info("✅ Initial metrics exported to HA successfully.")
    except Exception as e:
        logging.error(
            f"❌ FAILED to export initial metrics to HA: {e}", exc_info=True
        )

    # --- Startup Sensor Validation ---
    # Will run once on the first cycle when all_states is available.
    _sensor_validation_done = False

    # Define blocking_entities outside try block so it's available in
    # exception handler
    blocking_entities = [
        config.DHW_STATUS_ENTITY_ID,
        config.DEFROST_STATUS_ENTITY_ID,
        config.DISINFECTION_STATUS_ENTITY_ID,
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
    ]

    # Cycle timing debug variables
    cycle_number = 0
    last_cycle_end_time = None

    # --- Cooling ML model + observation buffer (lazy init once) ---
    _cooling_ml_model = None
    _cooling_obs_buffer = None
    _cooling_ml_model_type = getattr(config, "PRE_COOL_MODEL_TYPE", "trajectory")
    if getattr(config, "PRE_COOL_ENABLED", True):
        _steps_per_hour = round(60 / float(getattr(config, "CYCLE_INTERVAL_MINUTES", 10)))
        try:
            from .cooling_ml_model import CoolingMLModel
            from .cooling_ml_observation_buffer import CoolingObservationBuffer
            _horizon_h = int(getattr(config, "PRE_COOL_HORIZON_HOURS", 12))
            _cooling_ml_model = CoolingMLModel(
                model_path=getattr(config, "COOLING_ML_MODEL_PATH", "/opt/ml_heating/cooling_ml_model.joblib"),
                metadata_path=getattr(config, "COOLING_ML_METADATA_PATH", "/opt/ml_heating/cooling_ml_metadata.json"),
                steps_per_hour=_steps_per_hour,
            )
            _cooling_ml_model.load()  # non-fatal if file missing
            _cooling_obs_buffer = CoolingObservationBuffer(
                path=getattr(config, "COOLING_ML_OBSERVATION_BUFFER_PATH", "/opt/ml_heating/cooling_ml_obs_buffer.json"),
                max_n=int(getattr(config, "COOLING_ML_BUFFER_MAX_N", 500)),
                min_training_samples=int(getattr(config, "COOLING_ML_MIN_TRAINING_SAMPLES", 200)),
                retrain_trigger_k=int(getattr(config, "COOLING_ML_RETRAIN_TRIGGER_K", 50)),
                horizon_steps=_horizon_h * _steps_per_hour,
            )
        except Exception as _cml_init_err:
            logging.warning("Cooling ML init failed (non-fatal): %s", _cml_init_err)

    # --- Heating Correction ML observation buffer (always init) ---
    # Collects heating-mode observations regardless of HEATING_CORRECTION_MODE
    # so the buffer accumulates data even before the ML mode is enabled.
    _heating_obs_buffer = None
    _heating_obs_steps_per_hour = round(60 / float(getattr(config, "CYCLE_INTERVAL_MINUTES", 10)))
    try:
        from .heating_correction_ml_observation_buffer import (
            HeatingCorrectionObservationBuffer,
        )
        _heating_label_h = int(getattr(config, "HEATING_ML_LABEL_HORIZON_H", 4))
        _heating_obs_buffer = HeatingCorrectionObservationBuffer(
            path=getattr(
                config,
                "HEATING_ML_OBSERVATION_BUFFER_PATH",
                "/opt/ml_heating/heating_correction_ml_obs_buffer.json",
            ),
            max_n=int(getattr(config, "HEATING_ML_BUFFER_MAX_N", 500)),
            min_training_samples=int(
                getattr(config, "HEATING_ML_MIN_TRAINING_SAMPLES", 200)
            ),
            retrain_trigger_k=int(
                getattr(config, "HEATING_ML_RETRAIN_TRIGGER_K", 50)
            ),
            horizon_steps=_heating_label_h * _heating_obs_steps_per_hour,
        )
    except Exception as _hml_buf_init_err:
        logging.warning(
            "Heating correction obs buffer init failed (non-fatal): %s",
            _hml_buf_init_err,
        )

    while True:
        try:
            # CYCLE START DEBUG LOGGING
            cycle_number += 1
            cycle_start_time = time.time()
            cycle_start_datetime = datetime.now()

            # Calculate interval since last cycle
            if last_cycle_end_time is not None:
                interval_since_last = cycle_start_time - last_cycle_end_time
                logging.debug(
                    f"🔄 CYCLE {cycle_number} START: "
                    f"{cycle_start_datetime.strftime('%H:%M:%S')} "
                    f"(interval: {interval_since_last/60:.1f}min since "
                    f"last cycle)"
                )
            else:
                logging.debug(
                    f"🔄 CYCLE {cycle_number} START: "
                    f"{cycle_start_datetime.strftime('%H:%M:%S')} "
                    f"(first cycle)"
                )
            # Initialize shadow mode tracking for this cycle
            shadow_mode_active = (
                config.TARGET_OUTLET_TEMP_ENTITY_ID
                != config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID
            )

            # Load the application state at the beginning of each cycle.
            # Use the wrapper's currently-active state manager so a cooling-
            # mode cycle reads from the cooling file even before climate mode
            # is re-detected below.  On the very first cycle the wrapper
            # defaults to the heating manager, which is the correct fallback.
            from .model_wrapper import get_enhanced_model_wrapper as _get_wrapper
            _active_state_manager = _get_wrapper().state_manager
            state = load_state(state_manager=_active_state_manager)
            # Create a new Home Assistant client for each cycle.
            ha_client = create_ha_client()
            # Fetch all states from Home Assistant at once to minimize
            # API calls.
            all_states = ha_client.get_all_states()

            thermodynamic_metrics_written_in_sensor_update = False

            # --- One-time Startup Sensor Validation ---
            # Verify configured sensor entity IDs exist in HA on first
            # successful fetch. Missing sensors (especially fireplace/TV)
            # lead to permanently empty learning channels.
            if all_states and not _sensor_validation_done:
                try:
                    # all_states can be a list of dicts (HA API) or a dict
                    # of dicts (test mocks) — extract entity IDs from both.
                    if isinstance(all_states, list):
                        _known_ids = {
                            s.get("entity_id")
                            for s in all_states
                            if isinstance(s, dict)
                        }
                    elif isinstance(all_states, dict):
                        _known_ids = set(all_states.keys())
                    else:
                        _known_ids = set()

                    _critical_sensors = {
                        "INDOOR_TEMP": config.INDOOR_TEMP_ENTITY_ID,
                        "OUTDOOR_TEMP": config.OUTDOOR_TEMP_ENTITY_ID,
                        "OUTLET_TEMP": config.ACTUAL_OUTLET_TEMP_ENTITY_ID,
                        "TARGET_INDOOR": config.TARGET_INDOOR_TEMP_ENTITY_ID,
                        "HEATING_STATUS": config.HEATING_STATUS_ENTITY_ID,
                    }
                    _optional_sensors = {
                        "FIREPLACE": config.FIREPLACE_STATUS_ENTITY_ID,
                        "TV": config.TV_STATUS_ENTITY_ID,
                        "INLET_TEMP": config.INLET_TEMP_ENTITY_ID,
                        "PV_POWER": config.PV_POWER_ENTITY_ID,
                        "LIVING_ROOM": config.LIVING_ROOM_TEMP_ENTITY_ID,
                    }
                    _missing_critical = {
                        name: eid
                        for name, eid in _critical_sensors.items()
                        if eid not in _known_ids
                    }
                    _missing_optional = {
                        name: eid
                        for name, eid in _optional_sensors.items()
                        if eid not in _known_ids
                    }
                    if _missing_critical:
                        logging.error(
                            "🚨 CRITICAL sensors not found in HA! "
                            "Learning and control will be impaired: %s",
                            _missing_critical,
                        )
                    if _missing_optional:
                        logging.warning(
                            "⚠️ Optional sensors not found in HA — "
                            "associated learning channels will remain "
                            "empty: %s",
                            _missing_optional,
                        )
                    if not _missing_critical and not _missing_optional:
                        logging.info(
                            "✅ All configured sensor entity IDs "
                            "verified in HA."
                        )
                    _sensor_validation_done = True
                except Exception as e:
                    logging.warning(
                        "Startup sensor validation failed: %s", e
                    )

            # --- Update Sensor Buffer ---
            if all_states:
                current_time = datetime.now(timezone.utc)
                
                # Helper to safely get float state
                def get_float_state(entity_id):
                    try:
                        val = ha_client.get_state(entity_id, all_states)
                        return float(val) if val is not None else None
                    except (ValueError, TypeError):
                        return None

                # Push new readings to buffer
                buffer_updates = {
                    config.INDOOR_TEMP_ENTITY_ID: get_float_state(
                        config.INDOOR_TEMP_ENTITY_ID
                    ),
                    config.ACTUAL_OUTLET_TEMP_ENTITY_ID: get_float_state(
                        config.ACTUAL_OUTLET_TEMP_ENTITY_ID
                    ),
                    config.TARGET_OUTLET_TEMP_ENTITY_ID: get_float_state(
                        config.TARGET_OUTLET_TEMP_ENTITY_ID
                    ),
                    config.OUTDOOR_TEMP_ENTITY_ID: get_float_state(
                        config.OUTDOOR_TEMP_ENTITY_ID
                    ),
                    config.INLET_TEMP_ENTITY_ID: get_float_state(
                        config.INLET_TEMP_ENTITY_ID
                    ),
                    config.FLOW_RATE_ENTITY_ID: get_float_state(
                        config.FLOW_RATE_ENTITY_ID
                    ),
                }

                for entity_id, value in buffer_updates.items():
                    if value is not None:
                        sensor_buffer.add_reading(
                            entity_id, value, current_time
                        )

                # --- Real-time Thermodynamic Sensors ---
                # Calculate and export COP and Thermal Power immediately so
                # they are available even if the cycle is skipped (e.g.
                # blocking/idle).
                try:
                    # Get power consumption which isn't in the buffer updates
                    power_consumption = get_float_state(
                        config.POWER_CONSUMPTION_ENTITY_ID
                    )

                    # Use values we just fetched for the buffer
                    current_outlet = buffer_updates.get(
                        config.ACTUAL_OUTLET_TEMP_ENTITY_ID
                    )
                    current_inlet = buffer_updates.get(
                        config.INLET_TEMP_ENTITY_ID
                    )
                    current_flow = buffer_updates.get(
                        config.FLOW_RATE_ENTITY_ID
                    )

                    # Only calculate if we have the minimum required data
                    # (outlet temp)
                    if current_outlet is not None:
                        thermo_metrics = calculate_thermodynamic_metrics(
                            outlet_temp=current_outlet,
                            inlet_temp=current_inlet,
                            flow_rate=current_flow,
                            power_consumption=power_consumption
                        )

                        # Export to Home Assistant
                        # 1. COP
                        cop_entity_id = get_shadow_output_entity_id(
                            "sensor.ml_heating_cop_realtime"
                        )
                        ha_client.set_state(
                            cop_entity_id,
                            thermo_metrics["cop_realtime"],
                            get_sensor_attributes(cop_entity_id),
                            round_digits=2
                        )

                        # 2. Thermal Power
                        thermal_power_entity_id = get_shadow_output_entity_id(
                            "sensor.ml_heating_thermal_power"
                        )
                        ha_client.set_state(
                            thermal_power_entity_id,
                            thermo_metrics["thermal_power_kw"],
                            get_sensor_attributes(thermal_power_entity_id),
                            round_digits=3
                        )

                        try:
                            influx_service.write_thermodynamic_metrics(
                                thermo_metrics
                            )
                            thermodynamic_metrics_written_in_sensor_update = True
                        except Exception as e:
                            error_msg = str(e)
                            if "unauthorized" in error_msg.lower() or "401" in error_msg:
                                logging.error(
                                    "Failed to write thermodynamic metrics: %s. "
                                    "Check that INFLUX_TOKEN has write permission to "
                                    "INFLUX_FEATURES_BUCKET.",
                                    e,
                                )
                            else:
                                logging.warning(
                                    "Failed to log thermodynamic metrics: %s", e
                                )

                        logging.debug(
                            "Thermodynamic sensors updated: COP=%.2f, "
                            "Power=%.3fkW",
                            thermo_metrics["cop_realtime"],
                            thermo_metrics["thermal_power_kw"]
                        )
                except Exception as e:
                    logging.warning(
                        "Failed to update thermodynamic sensors: %s", e
                    )

            # --- Determine Shadow Mode Early (needed for grace period) ---
            # Read input_boolean.ml_heating to determine control mode
            ml_heating_enabled = None
            if all_states:
                ml_heating_enabled = ha_client.get_state(
                    config.ML_HEATING_CONTROL_ENTITY_ID,
                    all_states,
                    is_binary=True
                )

            # Shadow mode is active when:
            # - Config SHADOW_MODE=true (override), OR
            # - ML heating boolean is OFF/unavailable
            if ml_heating_enabled is None:
                if all_states:  # Only warn if we could fetch states
                    logging.warning(
                        "Cannot read %s, defaulting to shadow mode",
                        config.ML_HEATING_CONTROL_ENTITY_ID,
                    )
                ml_heating_enabled = False

            shadow_mode = resolve_shadow_mode(
                ml_heating_enabled=ml_heating_enabled
            )
            effective_shadow_mode = shadow_mode.effective_shadow_mode

            if not all_states:
                logging.warning(
                    "Could not fetch states from HA, skipping cycle."
                )
                # Emit NETWORK_ERROR state to Home Assistant
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
                            "state_description": "Network Error",
                            "last_updated": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                    )
                    ha_client.set_state(
                        heating_state_entity_id,
                        3,
                        attributes_state,
                        round_digits=None,
                    )
                except Exception:
                    logging.debug(
                        "Failed to write NETWORK_ERROR state to HA.",
                        exc_info=True,
                    )
                time.sleep(PhysicsConstants.RETRY_DELAY_SECONDS)
                continue

            # --- Check for blocking modes (DHW, Defrost) ---
            # Skip the control logic if the heat pump is busy with other
            # tasks like heating domestic hot water (DHW) or defrosting.
            # blocking_entities already defined outside try block. Build a
            # list of active blocking reasons so we can distinguish
            # DHW-like (long) blockers from short ones like defrost.
            blocking_manager = BlockingStateManager()
            is_blocking, blocking_reasons = (
                blocking_manager.check_blocking_state(
                    ha_client, all_states
                )
            )

            # Determine climate mode EARLY — before learning — so that the
            # learning context can pass the correct mode to
            # _is_heat_pump_active() and other mode-aware helpers.
            _early_checker = HeatingSystemStateChecker()
            climate_mode = _early_checker.get_climate_mode(
                ha_client, all_states
            )

            # --- Step 1: Online Learning from Previous Cycle ---
            # Learn from the results of the previous cycle. This allows the
            # model to continuously adapt to the actual house behavior,
            # whether running in active mode (model controls heating) or
            # shadow mode (heat curve controls heating).
            last_run_features = state.get("last_run_features")
            last_indoor_temp = state.get("last_indoor_temp")
            last_final_temp_stored = state.get("last_final_temp")
            last_avg_other_rooms_temp = state.get("last_avg_other_rooms_temp")

            if (
                last_run_features is not None
                and last_indoor_temp is not None
                and last_final_temp_stored is not None
            ):

                # Read the actual target outlet temp that was applied.
                # This reads what temperature was actually set by either:
                # - The model in active mode
                # - The heat curve in shadow mode
                # By reading it now (start of next cycle), we give it time
                # to update after the previous cycle's set_state call.
                actual_applied_temp = ha_client.get_state(
                    config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID, all_states
                )

                if actual_applied_temp is None:
                    actual_applied_temp = last_final_temp_stored

                # Get current indoor temperature to calculate actual change
                current_indoor = ha_client.get_state(
                    config.INDOOR_TEMP_ENTITY_ID, all_states
                )

                if current_indoor is not None:
                    actual_indoor_change = current_indoor - last_indoor_temp
                    previous_cycle_climate_mode = (
                        state.get("last_climate_mode") or climate_mode
                    )

                    # Create learning features with the actual outlet temp
                    # that was applied
                    # Handle case where last_run_features might be stored as
                    # string
                    if isinstance(last_run_features, str):
                        logging.error(
                            "ERROR: last_run_features corrupted as string - "
                            "attempting to recover"
                        )
                        try:
                            # Try to parse as JSON if it's a string
                            # representation
                            import json

                            last_run_features = json.loads(last_run_features)
                            logging.info(
                                "✅ Recovered features from JSON string"
                            )
                        except (json.JSONDecodeError, TypeError):
                            logging.error(
                                "❌ Cannot recover features from string, "
                                "using empty dict"
                            )
                            last_run_features = {}

                    if isinstance(last_run_features, pd.DataFrame):
                        learning_features = last_run_features.copy().to_dict(
                            orient="records"
                        )[0]
                    elif isinstance(last_run_features, dict):
                        learning_features = last_run_features.copy()
                    else:
                        learning_features = (
                            last_run_features.copy()
                            if last_run_features
                            else {}
                        )

                    persisted_target_temp = state.get(
                        "last_target_indoor_temp"
                    )
                    if persisted_target_temp is None:
                        persisted_target_temp = learning_features.get(
                            "target_temp"
                        )

                    if persisted_target_temp is None:
                        target_entity_id = config.TARGET_INDOOR_TEMP_ENTITY_ID
                        if (
                            previous_cycle_climate_mode == "cooling"
                            and getattr(
                                config,
                                "TARGET_INDOOR_TEMP_COOLING_ENTITY_ID",
                                "",
                            )
                        ):
                            target_entity_id = (
                                config.TARGET_INDOOR_TEMP_COOLING_ENTITY_ID
                            )
                        persisted_target_temp = ha_client.get_state(
                            target_entity_id, all_states
                        )

                    if persisted_target_temp is not None:
                        target_indoor_temp = float(persisted_target_temp)
                    else:
                        target_indoor_temp = last_indoor_temp
                        logging.debug(
                            "target_indoor_temp unavailable for previous cycle, using last_indoor_temp fallback: %.2f",
                            last_indoor_temp,
                        )

                    learning_features["target_temp"] = target_indoor_temp

                    learning_features["outlet_temp"] = actual_applied_temp
                    learning_features["outlet_temp_sq"] = (
                        actual_applied_temp**2
                    )
                    learning_features["outlet_temp_cub"] = (
                        actual_applied_temp**3
                    )

                    # Online learning is now handled by
                    # ThermalEquilibriumModel in model_wrapper
                    try:
                        # Import and create model wrapper for learning
                        from .model_wrapper import get_enhanced_model_wrapper

                        # Create wrapper instance
                        wrapper = get_enhanced_model_wrapper()
                        wrapper.set_climate_mode(
                            previous_cycle_climate_mode
                        )

                        # Prepare prediction context for learning
                        pv_history = learning_features.get("pv_power_history")
                        pv_now_learn = learning_features.get("pv_now", 0.0)
                        # If actual PV is zero (sun has set), use zero instead of lagged values
                        if pv_now_learn == 0:
                            pv_scalar_learn = 0.0
                        else:
                            pv_scalar_learn = (sum(pv_history) / len(pv_history)) if (pv_history and len(pv_history) > 0) else pv_now_learn
                        # Extract cloud cover forecasts and calculate average
                        cloud_cover_forecasts = [
                            learning_features.get(f"cloud_cover_forecast_{h}h", 50.0)
                            for h in range(1, 7)
                        ]
                        avg_cloud_cover = (
                            sum(cloud_cover_forecasts) / len(cloud_cover_forecasts)
                            if cloud_cover_forecasts else 50.0
                        )

                        prediction_context = {
                            "outlet_temp": actual_applied_temp,
                            "outdoor_temp": learning_features.get(
                                "outdoor_temp", 10.0
                            ),
                            "pv_power": pv_scalar_learn,
                            "pv_power_current": pv_now_learn,
                            "pv_power_history": pv_history,
                            "fireplace_on": float(
                                state.get("last_fireplace_on", False)
                            ),
                            "tv_on": learning_features.get("tv_on", 0.0),
                            "current_indoor": last_indoor_temp,
                            "avg_other_rooms_temp": last_avg_other_rooms_temp,
                            "thermal_power": learning_features.get(
                                "thermal_power_kw", None
                            ),
                            # HP active detection is now mode-aware via
                            # climate_mode — no explicit flag needed.
                            # _is_heat_pump_active() checks thermal_power,
                            # delta_t, and outlet/indoor thresholds with
                            # correct signs for heating vs cooling.
                            "climate_mode": previous_cycle_climate_mode,
                            # Pass auxiliary heat if available in features
                            "auxiliary_heat": learning_features.get(
                                "total_auxiliary_heat_kw", 0.0
                            ),
                            # Add target indoor temp for learning/cold weather logic
                            "target_temp": target_indoor_temp,
                            # Add cloud cover data for weather-aware learning
                            "avg_cloud_cover": avg_cloud_cover,
                            "cloud_cover_forecast": cloud_cover_forecasts,
                            # inlet_temp = Rücklauf = current slab state;
                            # used by slab time-constant gradient learning
                            "inlet_temp": learning_features.get("inlet_temp"),
                            # BT2 - BT3 steady-state floor loop offset;
                            # used by slab Euler step and gradient learning
                            "delta_t": learning_features.get("delta_t", 0.0),
                            # Indoor temperature trend for learning protection guards
                            "indoor_temp_gradient": learning_features.get("indoor_temp_gradient", 0.0),
                            "indoor_temp_delta_60m": learning_features.get("indoor_temp_delta_60m", 0.0),
                            # Living room temp (sensor.rt_wz) for fireplace learning
                            "living_room_temp": learning_features.get("living_room_temp"),
                            # Forecast arrays for trajectory-aligned gradient learning.
                            # These allow the gradient calculation to use the same
                            # future-aware horizon as the optimization (TRAJECTORY_STEPS).
                            "outdoor_forecast": [
                                learning_features.get(f"temp_forecast_{h}h", learning_features.get("outdoor_temp", 10.0))
                                for h in range(1, config.TRAJECTORY_STEPS + 1)
                            ],
                            "pv_forecast": [
                                learning_features.get(f"pv_forecast_{h}h", 0.0)
                                for h in range(1, config.TRAJECTORY_STEPS + 1)
                            ],
                        }

                        # FIXED SHADOW MODE LEARNING: Only learn from shadow
                        # mode when actually in shadow mode
                        # In ACTIVE MODE: Learn from ML's own decisions
                        # (even with smart rounding)
                        # In SHADOW MODE: Learn from heat curve decisions
                        was_shadow_mode_cycle = effective_shadow_mode

                        try:
                            # UNIFIED LEARNING: Always use trajectory
                            # prediction for learning to ensure consistency
                            # with the control loop's prediction method.
                            was_shadow_mode_cycle = effective_shadow_mode

                            if was_shadow_mode_cycle:
                                learning_mode = "shadow_mode_hc_trajectory"
                                log_msg = (
                                    "🔍 SHADOW MODE LEARNING (trajectory): "
                                    "Predicting indoor temp from heat "
                                    f"curve's {actual_applied_temp}°C "
                                    "outlet setting"
                                )
                            else:
                                learning_mode = "active_mode_ml_trajectory"
                                log_msg = (
                                    "🎯 ACTIVE MODE LEARNING (trajectory): "
                                    "Verifying ML prediction accuracy for "
                                    f"{actual_applied_temp}°C outlet setting"
                                )
                            logging.debug(log_msg)

                            # Use persisted prediction from previous cycle
                            # when available (active mode only). This avoids
                            # re-running the trajectory and uses the exact
                            # value the model committed to.
                            _stored_pred = state.get(
                                "last_predicted_indoor"
                            )
                            if (
                                _stored_pred is not None
                                and not was_shadow_mode_cycle
                            ):
                                model_predicted_temp = float(_stored_pred)
                                learning_mode = (
                                    "active_mode_persisted_prediction"
                                )
                                logging.debug(
                                    "♻️ Using persisted predicted indoor "
                                    "%.2f°C from previous cycle "
                                    "(skipping trajectory re-run)",
                                    model_predicted_temp,
                                )
                            else:
                                # Fallback: re-run trajectory (shadow mode
                                # or first cycle without stored prediction).

                                # Build forecast-aware outdoor array:
                                # [current, +1h, +2h, ..., +TRAJECTORY_STEPS h]
                                _learn_outdoor_now = prediction_context.get(
                                    "outdoor_temp", 10.0
                                )
                                _learn_outdoor_arr = (
                                    [_learn_outdoor_now]
                                    + prediction_context.get(
                                        "outdoor_forecast",
                                        [_learn_outdoor_now] * config.TRAJECTORY_STEPS,
                                    )
                                )
                                _learn_pv_forecast = prediction_context.get(
                                    "pv_forecast", None
                                )
                                trajectory = (
                                    wrapper.thermal_model
                                    .predict_thermal_trajectory(
                                        current_indoor=last_indoor_temp,
                                        target_indoor=last_indoor_temp,
                                        outlet_temp=actual_applied_temp,
                                        outdoor_temp=_learn_outdoor_arr,
                                        time_horizon_hours=float(
                                            config.TRAJECTORY_STEPS
                                        ),
                                        time_step_minutes=config.CYCLE_INTERVAL_MINUTES,
                                        pv_power=prediction_context.get(
                                            "pv_power", 0.0
                                        ),
                                        pv_forecasts=_learn_pv_forecast,
                                        fireplace_on=prediction_context.get(
                                            "fireplace_on", 0.0
                                        ),
                                        tv_on=prediction_context.get("tv_on", 0.0),
                                        cloud_cover_pct=prediction_context.get(
                                            "avg_cloud_cover", 50.0
                                        ),
                                        inlet_temp=prediction_context.get(
                                            "inlet_temp"
                                        ),
                                        delta_t_floor=prediction_context.get(
                                            "delta_t", 0.0
                                        ),
                                        # thermal_power intentionally omitted: the
                                        # instantaneous sensor reading is near-zero
                                        # during transitions and would trigger the
                                        # energy-only formula (T_eq = T_outdoor +
                                        # P/HLC), ignoring outlet_temp entirely and
                                        # producing a ~5°C equilibrium. Use the
                                        # outlet-based formula instead.
                                        thermal_power=None,
                                    )
                                )

                                predicted_indoor_temp = (
                                    trajectory["trajectory"][0]
                                    if trajectory and trajectory.get("trajectory")
                                    else last_indoor_temp
                                )

                                if predicted_indoor_temp is None:
                                    logging.warning(
                                        "Skipping online learning (%s): "
                                        "prediction returned None",
                                        learning_mode
                                    )
                                    continue

                                model_predicted_temp = predicted_indoor_temp

                        except Exception as e:
                            logging.warning(
                                "Skipping online learning: thermal model "
                                f"prediction error: {e}"
                            )
                            continue

                        # Enhanced prediction context with learning mode info
                        enhanced_prediction_context = prediction_context.copy()
                    
                        # Check if the previous cycle was a grace period
                        # passthrough. If so, override the learning mode to
                        # reflect this
                        if (
                            isinstance(last_run_features, dict)
                            and last_run_features.get("learning_mode")
                            == "grace_period_passthrough"
                        ):
                            learning_mode = "grace_period_passthrough"
                            
                        enhanced_prediction_context[
                            "learning_mode"
                        ] = learning_mode
                        enhanced_prediction_context[
                            "was_shadow_mode_cycle"
                        ] = was_shadow_mode_cycle
                        enhanced_prediction_context[
                            "ml_calculated_temp"
                        ] = last_final_temp_stored
                        enhanced_prediction_context[
                            "hc_applied_temp"
                        ] = actual_applied_temp

                        # Call the learning feedback method with the correct
                        # prediction context
                        wrapper.learn_from_prediction_feedback(
                            predicted_temp=model_predicted_temp,
                            actual_temp=current_indoor,
                            prediction_context=enhanced_prediction_context,
                            timestamp=datetime.now().isoformat(),
                            is_blocking_active=is_blocking,
                            effective_shadow_mode=effective_shadow_mode,
                        )

                        logging.debug(
                            "✅ Online learning: applied_temp=%.1f°C, "
                            "actual_change=%.3f°C, cycle=%d",
                            actual_applied_temp,
                            actual_indoor_change,
                            wrapper.cycle_count,
                        )
                    except Exception as e:
                        logging.warning(
                            "Online learning failed: %s", e, exc_info=True
                        )

                    # Shadow mode error tracking removed - handled by
                    # ThermalEquilibriumModel. Use the shared shadow-mode
                    # decision for comparison logging below.
                    if (
                        effective_shadow_mode
                        and actual_applied_temp != last_final_temp_stored
                    ):
                        logging.debug(
                            "Shadow mode: ML would set %.1f°C, HC set %.1f°C",
                            last_final_temp_stored,
                            actual_applied_temp,
                        )
                else:
                    logging.debug(
                        "Skipping online learning: current indoor temp "
                        "unavailable"
                    )
            else:
                logging.debug(
                    "Skipping online learning: no data from previous cycle"
                )

            # --- Grace Period after Blocking ---
            # Use modular blocking state manager for cleaner code
            # organization
            blocking_manager = BlockingStateManager()
            is_grace_period = blocking_manager.handle_grace_period(
                ha_client, state, shadow_mode=effective_shadow_mode
            )
            
            if is_grace_period:
                # We are in grace period, but we continue to allow passive
                # learning (feature calculation and state saving)
                logging.info("⏳ Grace period active - Passive learning mode")
                # We will skip the CONTROL part later

                # FIX: Do NOT overwrite last_final_temp with current actual
                # outlet temp. This causes "state poisoning" where a low
                # actual temp (e.g. 25C) becomes the target for the next
                # cycle if sensors fail or another grace period occurs.
                # Instead, preserve the previous valid target.
                preserved_target = state.get("last_final_temp")
                if preserved_target is None:
                    # Fallback only if no previous state exists
                    try:
                        preserved_target = float(
                            ha_client.get_state(
                                config.ACTUAL_OUTLET_TEMP_ENTITY_ID, all_states
                            )
                        )
                    except (ValueError, TypeError):
                        preserved_target = 20.0 # Safe fallback

                logging.info(
                    "Preserving last_final_temp=%.1f°C during grace period",
                    preserved_target
                )

                # Save state to ensure next cycle has valid 'last_run_features'
                # Mark this as a grace period passthrough so the next cycle
                # knows this wasn't a calculated target
                save_state(
                    state_manager=_active_state_manager,
                    last_final_temp=preserved_target,
                    last_is_blocking=False,  # Grace period is not blocking
                    # Preserve end time
                    last_blocking_end_time=state.last_blocking_end_time,
                    last_run_features={
                        "learning_mode": "grace_period_passthrough"
                    },
                )
                continue

            # --- Check if heating system is active ---
            heating_checker = HeatingSystemStateChecker()
            if not heating_checker.check_heating_active(ha_client, all_states):
                time.sleep(PhysicsConstants.RETRY_DELAY_SECONDS)
                continue

            # --- Determine climate mode (heating vs cooling) ---
            climate_mode = heating_checker.get_climate_mode(
                ha_client, all_states
            )
            # Propagate mode to model wrapper so binary search uses correct
            # outlet bounds and fallbacks.
            from .model_wrapper import get_enhanced_model_wrapper as _get_wrapper
            _wrapper = _get_wrapper()
            _prev_state_manager = _wrapper.state_manager
            _wrapper.set_climate_mode(climate_mode)
            # Re-resolve after mode swap so all saves below use the correct file.
            _active_state_manager = _wrapper.state_manager
            # Reload operational state whenever the active state file changes
            # (heating→cooling AND cooling→heating transitions) so this cycle
            # starts with the correct context (last_final_temp,
            # setpoint_hold_cycles_remaining, etc.).
            if _active_state_manager is not _prev_state_manager:
                state = load_state(state_manager=_active_state_manager)
                logging.info(
                    "♻️ Climate mode transition → reloaded operational state "
                    "from %s",
                    _active_state_manager.state_file,
                )
            if climate_mode == "cooling":
                logging.info(
                    "❄️ COOLING MODE: ML will calculate cooling outlet "
                    "temperature (outlet < inlet) — using cooling state %s",
                    _active_state_manager.state_file,
                )

            if is_blocking:
                logging.info(
                    "Blocking process active (DHW/Defrost), skipping."
                )
                try:
                    heating_state_entity_id = get_shadow_output_entity_id(
                        "sensor.ml_heating_state"
                    )
                    blocking_reasons = [
                        e
                        for e in blocking_entities
                        if ha_client.get_state(e, all_states, is_binary=True)
                    ]
                    attributes_state = get_sensor_attributes(
                        heating_state_entity_id
                    )
                    attributes_state.update(
                        {
                            "state_description": (
                                "Blocking activity - Skipping"
                            ),
                            "blocking_reasons": blocking_reasons,
                            "last_updated": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                    )
                    ha_client.set_state(
                        heating_state_entity_id,
                        2,
                        attributes_state,
                        round_digits=None,
                    )
                except Exception:
                    logging.debug(
                        "Failed to write BLOCKED state to HA.", exc_info=True
                    )
                # Save the blocking state for the next cycle (preserve
                # last_final_temp and record which entities caused the
                # blocking so we can avoid learning from DHW-like cycles).
                save_state(
                    state_manager=_active_state_manager,
                    last_is_blocking=True,
                    last_final_temp=state.get("last_final_temp"),
                    last_blocking_reasons=blocking_reasons,
                    last_blocking_end_time=None,
                )
                time.sleep(PhysicsConstants.RETRY_DELAY_SECONDS)
                continue

            # --- Get current sensor values ---
            sensor_manager = SensorDataManager()
            sensor_data, missing_sensors = sensor_manager.get_sensor_data(
                ha_client, cycle_number
            )

            if missing_sensors:
                sensor_manager.handle_missing_sensors(
                    ha_client, missing_sensors
                )
                time.sleep(PhysicsConstants.RETRY_DELAY_SECONDS)
                continue

            # Unpack sensor data
            target_indoor_temp = sensor_data["target_indoor_temp"]
            actual_indoor = sensor_data["actual_indoor"]
            actual_outlet_temp = sensor_data["actual_outlet_temp"]
            avg_other_rooms_temp = sensor_data["avg_other_rooms_temp"]
            fireplace_on = sensor_data["fireplace_on"]
            outdoor_temp = sensor_data["outdoor_temp"]
            owm_temp = sensor_data["owm_temp"]

            # In cooling mode, use the cooling-specific target entity if
            # configured.  Fall back to the standard heating target otherwise.
            if (
                climate_mode == "cooling"
                and getattr(config, "TARGET_INDOOR_TEMP_COOLING_ENTITY_ID", "")
            ):
                _cooling_target = ha_client.get_state(
                    config.TARGET_INDOOR_TEMP_COOLING_ENTITY_ID, all_states
                )
                if _cooling_target is not None:
                    try:
                        target_indoor_temp = float(_cooling_target)
                        logging.info(
                            "❄️ Using cooling target entity: %.1f°C",
                            target_indoor_temp,
                        )
                    except (TypeError, ValueError):
                        logging.warning(
                            "❄️ Cooling target entity returned "
                            "non-numeric value '%s' — using heating "
                            "target %.1f°C instead.",
                            _cooling_target,
                            float(target_indoor_temp),
                        )

            # --- Step 1: State Retrieval ---
            # Heat balance controller doesn't use prediction history anymore.
            # Removed prediction_history retrieval.

            # --- Step 2: Feature Building ---
            # Gathers all necessary data points (current sensor values,
            # historical data from InfluxDB, etc.) and transforms them into a
            # feature vector. This vector is the input the model will use to
            # make its next prediction.
            if fireplace_on:
                prediction_indoor_temp = avg_other_rooms_temp
                logging.debug(
                    "Fireplace ON. Using avg other rooms temp for prediction."
                )
            else:
                prediction_indoor_temp = actual_indoor
                logging.debug(
                    "Fireplace is OFF. Using main indoor temp for prediction."
                )

            # TRANSIENT DROP FILTER: If indoor temp dropped more than 0.25°C
            # since last cycle (door/window opened), use a gentle cooling
            # estimate instead of the raw dropped reading.  This prevents the
            # heat pump from starting unnecessarily — the house will recover
            # within ~30 min after the door closes.
            # NOTE: Only applies in HEATING mode.  In cooling mode a temp drop
            # is normal (HP is actively cooling); a door/window opening would
            # cause a RISE (warm outdoor air), not a drop.
            if (
                climate_mode != "cooling"
                and last_indoor_temp is not None
                and prediction_indoor_temp is not None
            ):
                _drop = last_indoor_temp - prediction_indoor_temp
                if _drop > 0.25:
                    _extrapolated = last_indoor_temp - 0.02
                    logging.warning(
                        "🚪 Transient drop filter: indoor temp dropped "
                        "%.3f°C (%.2f → %.2f). Using extrapolated temp "
                        "%.2f instead to prevent unnecessary heating.",
                        _drop,
                        last_indoor_temp,
                        prediction_indoor_temp,
                        _extrapolated,
                    )
                    prediction_indoor_temp = _extrapolated

            # --- Step 3a: Feature building ---
            # When PV_TRAJ_FORECAST_MODE_ENABLED is true, physics_features.py
            # internally expands the forecast horizon to
            # max(TRAJECTORY_STEPS, PV_TRAJ_MAX_STEPS) via _n_fc_full, so all
            # keys needed for any dynamic step count are always present.
            # There is no need to pre-mutate config.TRAJECTORY_STEPS here.
            features, outlet_history = build_physics_features(
                ha_client,
                influx_service,
                sensor_buffer,
                climate_mode=climate_mode,
                target_indoor_temp_override=target_indoor_temp,
            )
            # Handle both DataFrame and dict features properly
            if isinstance(features, pd.DataFrame):
                # Convert DataFrame to dict for safe access
                features_dict = (
                    features.iloc[0].to_dict() if not features.empty else {}
                )
            else:
                features_dict = features if isinstance(features, dict) else {}
            if features is None:
                logging.warning("Feature building failed, skipping cycle.")
                time.sleep(PhysicsConstants.RETRY_DELAY_SECONDS)
                continue

            # --- HLC Learner: push cycle data ---
            _hlc_cycle_ctx = None
            if _hlc_session_learner is not None:
                _hlc_cycle_ctx = {
                    "timestamp": datetime.now(),
                    "thermal_power_kw": features_dict.get("thermal_power_kw"),
                    "indoor_temp": actual_indoor,
                    "outdoor_temp": outdoor_temp,
                    "target_temp": target_indoor_temp,
                    "indoor_temp_delta_60m": features_dict.get(
                        "indoor_temp_delta_60m", 0.0
                    ),
                    "pv_now_electrical": features_dict.get(
                        "pv_now_electrical", 0.0
                    ),
                    "fireplace_on": float(fireplace_on) if fireplace_on else 0.0,
                    "tv_on": features_dict.get("tv_on", 0.0),
                    "dhw_heating": features_dict.get("dhw_heating", 0.0),
                    "defrosting": features_dict.get("defrosting", 0.0),
                    "dhw_boost_heater": features_dict.get(
                        "dhw_boost_heater", 0.0
                    ),
                    "is_blocking": bool(is_blocking),
                }
                try:
                    _session_result = _hlc_session_learner.push_cycle(
                        _hlc_cycle_ctx
                    )
                    if _session_result.get("session_closed"):
                        if _session_result.get("session_validated"):
                            _n_sessions = _session_result["session_records"]
                            logging.info(
                                "📡 HLC session: session record added (total: %d)",
                                _n_sessions,
                            )
                            if _n_sessions >= config.HLC_SESSION_MIN_SESSIONS:
                                _sess_ok, _sess_msg = (
                                    _hlc_session_learner.apply_to_thermal_state(
                                        thermal_state_manager=_active_state_manager
                                    )
                                )
                                if _sess_ok:
                                    logging.info(
                                        "📡 HLC session: %s", _sess_msg
                                    )
                                else:
                                    logging.debug(
                                        "📡 HLC session apply skipped: %s",
                                        _sess_msg,
                                    )
                        else:
                            logging.debug(
                                "📡 HLC session: session rejected — %s",
                                _session_result.get("reject_reason", "unknown"),
                            )
                except Exception as _hlc_sess_exc:
                    logging.debug(
                        "HLC session push failed: %s", _hlc_sess_exc
                    )

            # --- Step 3: Prediction ---
            # Dynamic trajectory scaling: now that pv_now is available from
            # features, compute the effective TRAJECTORY_STEPS for this cycle.
            # When PV_TRAJ_FORECAST_MODE_ENABLED is true, physics_features.py
            # fetches forecasts up to PV_TRAJ_MAX_STEPS, so all horizon keys
            # are present in features_dict regardless of TRAJECTORY_STEPS.
            _pv_forecast_traj: list[float] | None = None
            if getattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False):
                try:
                    from .pv_trajectory import compute_dynamic_trajectory_steps
                    # Use raw electrical output (not thermally-corrected) to
                    # measure actual solar availability for horizon scaling.
                    _pv_now_traj = float(
                        features_dict.get("pv_now_electrical", 0.0)
                    )
                    # Build hourly PV forecast list for forecast-driven mode.
                    # Use electrical (uncorrected) forecast keys to match the
                    # scale of pv_now_electrical and PV_TRAJ_THRESHOLD_W.
                    _pv_forecast_traj = [
                        float(features_dict.get(
                            f"pv_forecast_electrical_{h}h",
                            features_dict.get(f"pv_forecast_{h}h", 0.0),
                        ))
                        for h in range(
                            1, int(getattr(config, "PV_TRAJ_MAX_STEPS", 12)) + 1
                        )
                    ]
                    _dyn_steps = compute_dynamic_trajectory_steps(
                        _pv_now_traj,
                        pv_forecast=_pv_forecast_traj,
                    )
                    config.TRAJECTORY_STEPS = _dyn_steps
                    config.MIN_SETPOINT_HOLD_CYCLES = _dyn_steps
                except Exception as _exc:
                    logging.warning(
                        "Dynamic trajectory scaling failed: %s", _exc
                    )

            # Read electricity price for price-aware optimization
            price_data = None
            if getattr(config, "ELECTRICITY_PRICE_ENABLED", False):
                try:
                    from .price_optimizer import get_price_optimizer
                    optimizer = get_price_optimizer()
                    optimizer.refresh_prices_if_needed(ha_client)
                    price_data = optimizer.get_price_data_for_features()
                except Exception as exc:
                    logging.warning("Failed to read electricity price: %s", exc)

            # In forecast-driven trajectory mode, optionally suppress the price
            # offset so it does not interfere with the pre-heat plan.
            if (
                price_data is not None
                and getattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False)
                and getattr(
                    config, "PV_TRAJ_DISABLE_PRICE_IN_FORECAST_MODE", True
                )
            ):
                try:
                    from .pv_trajectory import is_forecast_trajectory_active
                    # Reuse the forecast list built during trajectory scaling;
                    # fall back to building it on demand if scaling was skipped.
                    _fc_pv_now = float(
                        features_dict.get("pv_now_electrical", 0.0)
                    )
                    _fc_forecast = _pv_forecast_traj if _pv_forecast_traj is not None else [
                        float(features_dict.get(
                            f"pv_forecast_electrical_{h}h",
                            features_dict.get(f"pv_forecast_{h}h", 0.0),
                        ))
                        for h in range(
                            1,
                            int(getattr(config, "PV_TRAJ_MAX_STEPS", 12)) + 1,
                        )
                    ]
                    if is_forecast_trajectory_active(
                        _fc_pv_now,
                        _fc_forecast,
                    ):
                        price_data = None
                        logging.info(
                            "☀️ Forecast trajectory active: "
                            "price offset suppressed"
                        )
                except Exception as _exc:
                    logging.debug(
                        "Forecast trajectory price suppression check "
                        "failed: %s",
                        _exc,
                    )

            # temperature prediction. This replaces the complex Heat Balance
            # Controller with a single prediction call.
            error_target_vs_actual = (
                target_indoor_temp - prediction_indoor_temp
            )

            # --- Pre-Cooling: Predictive Overheating Prevention ---
            # Supports two strategies via PRE_COOL_MODEL_TYPE:
            #   "trajectory" (default) — physics simulation (OverheatingPredictor)
            #   "lgbm_model" — LightGBM classifier (CoolingMLModel)
            # The inactive strategy runs as shadow and logs its signal.
            # Every cooling-mode cycle the observation buffer receives a feature
            # snapshot; labels are resolved horizon_steps later for auto-retrain.
            _pre_cool_active = False
            _pre_cool_result = None
            if (
                climate_mode == "cooling"
                and getattr(config, "PRE_COOL_ENABLED", True)
            ):
                try:
                    from .overheating_predictor import OverheatingPredictor
                    _pre_cool_predictor = OverheatingPredictor()
                    _traj_result = _pre_cool_predictor.predict_overheating_risk(
                        current_indoor=prediction_indoor_temp,
                        target_cooling=target_indoor_temp,
                        features=features_dict,
                        thermal_model=_wrapper.thermal_model,
                        climate_mode=climate_mode,
                    )

                    # LGBM result (only when model loaded)
                    _lgbm_result = None
                    if _cooling_ml_model is not None and _cooling_ml_model.is_loaded:
                        _lgbm_result = _cooling_ml_model.predict_overheating_risk(
                            current_indoor=prediction_indoor_temp,
                            target_cooling=target_indoor_temp,
                            features=features_dict,
                            climate_mode=climate_mode,
                        )

                    # Select active vs. shadow strategy
                    if _cooling_ml_model_type == "lgbm_model" and _lgbm_result is not None:
                        _pre_cool_result = _lgbm_result
                        logging.debug(
                            "❄️ SHADOW (trajectory): risk=%s peak=%.1f°C in %.1fh",
                            _traj_result.get("risk"),
                            _traj_result.get("peak_temp", 0),
                            _traj_result.get("peak_hour", 0),
                        )
                    else:
                        _pre_cool_result = _traj_result
                        if _lgbm_result is not None:
                            logging.debug(
                                "❄️ SHADOW (lgbm): risk=%s p=%.3f should_cool=%s",
                                _lgbm_result.get("risk"),
                                _lgbm_result.get("lgbm_proba", 0.0),
                                _lgbm_result.get("should_cool_now"),
                            )

                    # Observation buffer: accumulate features + resolve labels
                    if _cooling_obs_buffer is not None:
                        import datetime as _dt
                        _cooling_obs_buffer.push_pending(
                            features=features_dict,
                            indoor_temp=prediction_indoor_temp,
                            cooling_target=target_indoor_temp,
                            timestamp=_dt.datetime.utcnow().isoformat() + "Z",
                        )
                        _newly_labeled = _cooling_obs_buffer.resolve_labels(prediction_indoor_temp)
                        if _newly_labeled > 0:
                            logging.debug(
                                "Cooling obs buffer: resolved %d new labels this cycle",
                                _newly_labeled,
                            )
                        # Persist after each push/resolve cycle so pending entries
                        # (and their evolving label state) survive restarts.
                        try:
                            _cooling_obs_buffer.save()
                        except Exception as _save_err:
                            logging.warning(
                                "Cooling obs buffer save failed (non-fatal): %s",
                                _save_err,
                            )
                        # Auto-retrain when enough new labeled samples accumulated
                        if _cooling_obs_buffer.should_retrain():
                            logging.info(
                                "🤖 Cooling ML: retrain trigger reached "
                                "(%d labeled) — starting retrain",
                                _cooling_obs_buffer.n_labeled,
                            )
                            try:
                                from .cooling_ml_calibration import calibrate_cooling_ml
                                if calibrate_cooling_ml():
                                    _cooling_ml_model.load()
                                    logging.info("✅ Cooling ML model retrained and reloaded")
                                    # Only reset + persist on success so a transient
                                    # failure (InfluxDB down, etc.) retries next cycle.
                                    _cooling_obs_buffer.reset_retrain_counter()
                                    _cooling_obs_buffer.save()
                                else:
                                    logging.warning(
                                        "Cooling ML retrain returned False — "
                                        "will retry after %d more new observations",
                                        max(1, _cooling_obs_buffer._retrain_trigger_k // 2 + 1),
                                    )
                                    # Partial back-off: reduce the counter so we retry
                                    # sooner than a full retrain_trigger_k wait, but ensure
                                    # we drop below trigger_k to prevent immediate re-trigger.
                                    with _cooling_obs_buffer._lock:
                                        _cooling_obs_buffer._labeled_since_last_train = max(
                                            0,
                                            _cooling_obs_buffer._labeled_since_last_train
                                            - _cooling_obs_buffer._retrain_trigger_k // 2 - 1,
                                        )
                            except Exception as _retrain_err:
                                logging.warning(
                                    "Cooling ML retrain failed (non-fatal): %s", _retrain_err
                                )

                    if (
                        _pre_cool_result.get("should_cool_now")
                        and prediction_indoor_temp <= target_indoor_temp
                    ):
                        _offset = float(
                            getattr(
                                config, "PRE_COOL_TARGET_OFFSET_K", 0.5
                            )
                        )
                        _original_target = target_indoor_temp
                        target_indoor_temp = (
                            prediction_indoor_temp - _offset
                        )
                        _pre_cool_active = True
                        logging.info(
                            "❄️ PRE-COOL [%s]: target shifted %.1f → %.1f°C "
                            "(room %.1f°C, predicted peak %.1f°C in "
                            "%.1fh). Reason: %s",
                            _cooling_ml_model_type,
                            _original_target,
                            target_indoor_temp,
                            prediction_indoor_temp,
                            _pre_cool_result.get("peak_temp", 0),
                            _pre_cool_result.get("peak_hour", 0),
                            _pre_cool_result.get("reason", ""),
                        )
                    # Persist pre-cooling state for dashboard / restart
                    try:
                        _active_state_manager.update_operational_state(
                            pre_cool_active=_pre_cool_active,
                            pre_cool_peak_temp=float(
                                _pre_cool_result.get("peak_temp", 0)
                            ),
                            pre_cool_peak_hour=float(
                                _pre_cool_result.get("peak_hour", 0)
                            ),
                            pre_cool_risk=bool(
                                _pre_cool_result.get("risk", False)
                            ),
                        )
                    except Exception:
                        pass
                except Exception as _pre_cool_exc:
                    logging.debug(
                        "Pre-cooling check failed: %s", _pre_cool_exc
                    )

            # --- Heating Correction ML: Observation Buffer (push + resolve + retrain) ---
            # Mirrors the cooling obs buffer pattern: the entire block is gated on
            # climate_mode == "heating" so that:
            #   • resolve_labels is never called with summer/cooling indoor temps,
            #     which would assign garbage labels to pending heating observations.
            #   • retrain is only triggered during a heating cycle.
            if _heating_obs_buffer is not None and climate_mode == "heating":
                import datetime as _dt
                try:
                    # Compute S_H from current calibrated thermal parameters.
                    # Uses the same formula as calibrate_heating_correction_ml().
                    from .heating_correction_ml_calibration import (
                        _compute_s_h,
                        _read_baseline_thermal_params,
                    )
                    _hob_eta, _hob_u, _hob_tau = _read_baseline_thermal_params(config)
                    _hob_label_h = float(getattr(config, "HEATING_ML_LABEL_HORIZON_H", 4))
                    _hob_s_h = _compute_s_h(_hob_eta, _hob_u, _hob_tau, _hob_label_h)

                    # Push a new pending observation every heating cycle.
                    _hob_features = features_dict if isinstance(features_dict, dict) else {}
                    _heating_obs_buffer.push_pending(
                        features=_hob_features,
                        indoor_temp=prediction_indoor_temp,
                        heating_target=target_indoor_temp,
                        timestamp=_dt.datetime.utcnow().isoformat() + "Z",
                    )

                    # Resolve pending labels (advances age for pending entries).
                    _hob_newly_labeled = _heating_obs_buffer.resolve_labels(
                        prediction_indoor_temp, _hob_s_h
                    )
                    if _hob_newly_labeled > 0:
                        logging.debug(
                            "Heating obs buffer: resolved %d new label(s) this cycle",
                            _hob_newly_labeled,
                        )

                    # Persist after push/resolve so pending entries survive restarts.
                    try:
                        _heating_obs_buffer.save()
                    except Exception as _hob_save_err:
                        logging.warning(
                            "Heating obs buffer save failed (non-fatal): %s",
                            _hob_save_err,
                        )

                    # Auto-retrain when enough new labeled samples accumulated.
                    if _heating_obs_buffer.should_retrain():
                        logging.info(
                            "🤖 Heating ML: retrain trigger reached "
                            "(%d labeled) — starting retrain",
                            _heating_obs_buffer.n_labeled,
                        )
                        try:
                            from .heating_correction_ml_calibration import (
                                calibrate_heating_correction_ml,
                            )
                            if calibrate_heating_correction_ml():
                                # Invalidate the in-memory singleton so the next
                                # inference call hot-reloads the retrained model.
                                from .model_wrapper import EnhancedModelWrapper
                                EnhancedModelWrapper._heating_correction_ml_model = None
                                logging.info(
                                    "✅ Heating correction ML model retrained and reloaded"
                                )
                                _heating_obs_buffer.reset_retrain_counter()
                                _heating_obs_buffer.save()
                            else:
                                logging.warning(
                                    "Heating ML retrain returned False — "
                                    "will retry after %d more new observations",
                                    max(1, _heating_obs_buffer.retrain_trigger_k // 2 + 1),
                                )
                                # Partial back-off so we retry sooner than a full K wait
                                # but don't immediately re-trigger.
                                with _heating_obs_buffer._lock:
                                    _heating_obs_buffer._labeled_since_last_train = max(
                                        0,
                                        _heating_obs_buffer._labeled_since_last_train
                                        - _heating_obs_buffer.retrain_trigger_k // 2 - 1,
                                    )
                        except Exception as _hob_retrain_err:
                            logging.warning(
                                "Heating ML retrain failed (non-fatal): %s",
                                _hob_retrain_err,
                            )
                except Exception as _hob_err:
                    logging.debug(
                        "Heating obs buffer cycle failed (non-fatal): %s", _hob_err
                    )

            suggested_temp, confidence, metadata = (
                simplified_outlet_prediction(
                    features, prediction_indoor_temp, target_indoor_temp,
                    price_data=price_data,
                )
            )
            final_temp = suggested_temp

            # Log simplified prediction info
            logging.debug(
                "Model Wrapper: temp=%.1f°C, error=%.3f°C, confidence=%.3f",
                suggested_temp,
                abs(error_target_vs_actual),
                confidence,
            )

            # --- Gradual Temperature Control ---
            if actual_outlet_temp is not None:
                max_change = config.MAX_TEMP_CHANGE_PER_CYCLE
                original_temp = final_temp  # Keep a copy for logging

                last_blocking_reasons = (
                    state.get("last_blocking_reasons", []) or []
                )
                last_final_temp = state.get("last_final_temp")

                # DHW-like blockers that should keep the soft-start behavior
                dhw_like_blockers = {
                    config.DHW_STATUS_ENTITY_ID,
                    config.DISINFECTION_STATUS_ENTITY_ID,
                    config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
                }

                # SHADOW MODE FIX: In shadow mode, the baseline for gradual
                # control should be the actual heat curve temperature from
                # the last cycle, not the ML's (potentially wrong)
                # prediction.
                if effective_shadow_mode:
                    baseline = actual_outlet_temp
                    logging.info(
                        "Gradual control baseline in shadow mode set to "
                        "actual_outlet_temp: %.1f°C", baseline
                    )
                elif last_final_temp is not None:
                    baseline = last_final_temp
                    if any(
                        b in dhw_like_blockers
                        for b in last_blocking_reasons
                    ):
                        baseline = actual_outlet_temp
                else:
                    baseline = actual_outlet_temp

                # Calculate the difference from the chosen baseline
                delta = final_temp - baseline
                # Clamp the delta to the maximum allowed change
                if abs(delta) > max_change:
                    final_temp = baseline + np.clip(
                        delta, -max_change, max_change
                    )
                    logging.info("--- Gradual Temperature Control ---")
                    logging.info(
                        "Change from baseline %.1f°C to suggested %.1f°C "
                        "exceeds max change of %.1f°C. Capping at %.1f°C.",
                        baseline,
                        original_temp,
                        max_change,
                        final_temp,
                    )

            # --- EMA outlet smoothing ---
            last_final = state.get("last_final_temp")
            final_temp = apply_ema_smoothing(final_temp, last_final)

            # --- Minimum Setpoint Hold ---
            # Prevent the setpoint from changing more often than every
            # MIN_SETPOINT_HOLD_CYCLES cycles so the trajectory optimizer's
            # plan is not undermined by per-cycle micro-adjustments.
            # NOTE: config.MIN_SETPOINT_HOLD_CYCLES may have been updated
            # above by dynamic trajectory scaling; min_hold is only used when
            # starting a *new* hold (the else branch), never during an active
            # hold countdown, so there is no mid-countdown mutation issue.
            hold_remaining = state.get("setpoint_hold_cycles_remaining", 0) or 0
            min_hold = int(getattr(
                config, "MIN_SETPOINT_HOLD_CYCLES", config.TRAJECTORY_STEPS
            ))
            held_temp = state.get("last_final_temp")
            if hold_remaining > 0 and held_temp is not None:
                logging.info(
                    "⏱️ Setpoint hold: keeping %.1f°C for %d more cycle(s) "
                    "(computed=%.1f°C)",
                    held_temp, hold_remaining, final_temp,
                )
                final_temp = held_temp
                new_hold_cycles = hold_remaining - 1
            else:
                # Only start a new hold when the setpoint actually changes.
                # If the optimizer produced the same temperature as before,
                # leave the counter at 0 so the next cycle can update freely.
                setpoint_changed = (
                    held_temp is None
                    or abs(final_temp - held_temp) > PhysicsConstants.SETPOINT_CHANGE_THRESHOLD_C
                )
                new_hold_cycles = max(0, min_hold - 1) if setpoint_changed else 0

            # Final prediction is now handled by ThermalEquilibriumModel in
            # model_wrapper
            # Use confidence metadata for predicted indoor temp if available
            predicted_indoor = metadata.get(
                "predicted_indoor", prediction_indoor_temp
            )

            # --- Step 4: Update Home Assistant and Log ---
            # The calculated `final_temp` is sent to Home Assistant to
            # control the boiler. Other metrics like model confidence, MAE,
            # and feature importances are also published to HA for
            # monitoring. In shadow mode, skip all HA sensor updates to
            # avoid interference.

            target_output_entity_id = get_shadow_output_entity_id(
                config.TARGET_OUTLET_TEMP_ENTITY_ID
            )
            if effective_shadow_mode and not shadow_mode.shadow_deployment:
                logging.info(
                    "🔍 SHADOW MODE: ML prediction calculated but not "
                    "applied to heating system"
                )
                logging.info(
                    "   Final temp: %.1f°C (calculated but not sent to HA)",
                    final_temp,
                )
            else:
                if not effective_shadow_mode:
                    # Apply smart rounding: test floor vs ceiling to see which
                    # gets closer to target
                    floor_temp = np.floor(final_temp)
                    ceiling_temp = np.ceil(final_temp)

                    if floor_temp == ceiling_temp:
                        # Already an integer
                        smart_rounded_temp = int(final_temp)
                        logging.debug(
                            f"Smart rounding: {final_temp:.2f}°C is already "
                            "integer"
                        )
                    else:
                        # Test both options using the thermal model to see which
                        # gets closer to target
                        try:
                            from .model_wrapper import get_enhanced_model_wrapper

                            wrapper = get_enhanced_model_wrapper()

                            # Create test contexts for floor and ceiling
                            # temperatures
                            pv_hist = (
                                features_dict.get("pv_power_history", [])
                                if isinstance(features_dict, dict)
                                else []
                            )
                            pv_now_test = (
                                features_dict.get("pv_now", 0.0)
                                if isinstance(features_dict, dict)
                                else 0.0
                            )
                            # If actual PV is zero (sun has set), use zero instead of lagged values
                            if pv_now_test == 0:
                                pv_val = 0.0
                            else:
                                pv_val = (sum(pv_hist) / len(pv_hist)) if (pv_hist and len(pv_hist) > 0) else pv_now_test
                            test_context_floor = {
                                "outlet_temp": floor_temp,
                                "outdoor_temp": outdoor_temp,
                                "pv_power": pv_val,
                                "pv_power_history": pv_hist,
                                "fireplace_on": fireplace_on,
                                "tv_on": (
                                    features_dict.get("tv_on", 0.0)
                                    if isinstance(features_dict, dict)
                                    else 0.0
                                ),
                            }

                            test_context_ceiling = test_context_floor.copy()
                            test_context_ceiling["outlet_temp"] = ceiling_temp

                            # UNIFIED CONTEXT: Use same forecast-based
                            # conditions as binary search
                            from .prediction_context import (
                                prediction_context_manager
                            )

                            # Set up unified prediction context (same as binary
                            # search uses)

                            pv_hist_smart = features_dict.get("pv_power_history", [])
                            pv_now_smart = features_dict.get("pv_now", 0.0)
                            # If actual PV is zero (sun has set), use zero instead of lagged values
                            if pv_now_smart == 0:
                                pv_smart_scalar = 0.0
                            else:
                                pv_smart_scalar = (sum(pv_hist_smart) / len(pv_hist_smart)) if (pv_hist_smart and len(pv_hist_smart) > 0) else pv_now_smart
                            thermal_features = {
                                "pv_power": pv_smart_scalar,
                                "pv_power_history": pv_hist_smart,
                                "fireplace_on": (
                                    float(fireplace_on)
                                    if fireplace_on is not None
                                    else 0.0
                                ),
                                "tv_on": features_dict.get("tv_on", 0.0),
                            }

                            prediction_context_manager.set_features(features_dict)
                            unified_context = (
                                prediction_context_manager.create_context(
                                    outdoor_temp=outdoor_temp,
                                    pv_power=thermal_features["pv_power"],
                                    thermal_features=thermal_features,
                                    target_temp=target_indoor_temp,
                                    current_temp=prediction_indoor_temp
                                )
                            )

                            thermal_params = (
                                prediction_context_manager
                                .get_thermal_model_params()
                            )

                            # Get predictions using UNIFIED forecast-based
                            # parameters
                            floor_predicted = wrapper.predict_indoor_temp(
                                outlet_temp=floor_temp,
                                outdoor_temp=thermal_params["outdoor_temp"],
                                current_indoor=prediction_indoor_temp,
                                pv_power=thermal_params["pv_power"],
                                fireplace_on=thermal_params["fireplace_on"],
                                tv_on=thermal_params["tv_on"],
                            )
                            ceiling_predicted = wrapper.predict_indoor_temp(
                                outlet_temp=ceiling_temp,
                                outdoor_temp=thermal_params["outdoor_temp"],
                                current_indoor=prediction_indoor_temp,
                                pv_power=thermal_params["pv_power"],
                                fireplace_on=thermal_params["fireplace_on"],
                                tv_on=thermal_params["tv_on"],
                            )

                            # Handle None returns from predict_indoor_temp
                            if (
                                floor_predicted is None
                                or ceiling_predicted is None
                            ):
                                logging.warning(
                                    "Smart rounding: predict_indoor_temp "
                                    "returned None, using fallback"
                                )
                                smart_rounded_temp = round(final_temp)
                                logging.debug(
                                    f"Smart rounding fallback: {final_temp:.2f}°C "
                                    f"→ {smart_rounded_temp}°C"
                                )
                            else:
                                # Calculate errors from target
                                floor_error = abs(
                                    floor_predicted - target_indoor_temp
                                )
                                ceiling_error = abs(
                                    ceiling_predicted - target_indoor_temp
                                )

                                if floor_error <= ceiling_error:
                                    smart_rounded_temp = int(floor_temp)
                                    chosen = "floor"
                                else:
                                    smart_rounded_temp = int(ceiling_temp)
                                    chosen = "ceiling"

                                logging.debug(
                                    f"Smart rounding: {final_temp:.2f}°C → "
                                    f"{smart_rounded_temp}°C (chose {chosen}: "
                                    f"floor→{floor_predicted:.2f}°C "
                                    f"[err={floor_error:.3f}], "
                                    f"ceiling→{ceiling_predicted:.2f}°C "
                                    f"[err={ceiling_error:.3f}], "
                                    f"target={target_indoor_temp:.1f}°C)"
                                )
                        except Exception as e:
                            # Fallback to regular rounding if smart rounding
                            # fails
                            smart_rounded_temp = round(final_temp)
                            logging.warning(
                                f"Smart rounding failed ({e}), using regular "
                                f"rounding: {final_temp:.2f}°C → "
                                f"{smart_rounded_temp}°C"
                            )
                else:
                    logging.info(
                        "🔍 SHADOW DEPLOYMENT: Publishing ML recommendation "
                        "to %s",
                        target_output_entity_id,
                    )

                logging.debug("Setting target outlet temp")
                ha_client.set_state(
                    target_output_entity_id,
                    round(final_temp, 1),
                    get_sensor_attributes(target_output_entity_id),
                    round_digits=None,  # No additional rounding needed
                )

            # --- Log Metrics ---
            # Metrics logging now handled by ThermalEquilibriumModel in
            # model_wrapper
            if not effective_shadow_mode:
                logging.debug("Logging thermal model metrics")
                # Confidence is logged via simplified_outlet_prediction
                # metadata
                # Feature importances are handled by ThermalEquilibriumModel
                # Learning metrics are exported by ThermalEquilibriumModel

            # --- Log Thermodynamic Metrics (Feature/Sensor-Update) ---
            # Log COP, Power, and Delta T to InfluxDB for efficiency tracking
            if not thermodynamic_metrics_written_in_sensor_update:
                try:
                    thermodynamic_metrics = {
                        "cop_realtime": features_dict.get("cop_realtime", 0.0),
                        "thermal_power_kw": features_dict.get(
                            "thermal_power_kw", 0.0
                        ),
                        "delta_t": features_dict.get("delta_t", 0.0),
                        "flow_rate": features_dict.get("flow_rate", 0.0),
                        "inlet_temp": features_dict.get("inlet_temp", 0.0),
                    }
                    influx_service.write_thermodynamic_metrics(
                        thermodynamic_metrics
                    )
                except Exception as e:
                    error_msg = str(e)
                    if "unauthorized" in error_msg.lower() or "401" in error_msg:
                        logging.error(
                            "Failed to write thermodynamic metrics: %s. "
                            "Check that INFLUX_TOKEN has write permission to "
                            "INFLUX_FEATURES_BUCKET.",
                            e,
                        )
                    else:
                        logging.warning(
                            "Failed to log thermodynamic metrics: %s", e
                        )


            # --- Update ML State sensor ---
            # Skip ML state sensor updates in shadow mode
            if shadow_mode.should_publish_output_entities:
                try:
                    # Get thermal model trust metrics from
                    # ThermalEquilibriumModel
                    thermal_trust_metrics = metadata.get(
                        "thermal_trust_metrics", {}
                    )

                    heating_state_entity_id = get_shadow_output_entity_id(
                        "sensor.ml_heating_state"
                    )

                    attributes_state = get_sensor_attributes(
                        heating_state_entity_id
                    )
                    attributes_state.update(
                        {
                            "state_description": "Confidence - Too Low"
                            if confidence < config.CONFIDENCE_THRESHOLD
                            else "OK - Prediction done",
                            "confidence": round(confidence, 4),
                            "suggested_temp": round(suggested_temp, 2),
                            "final_temp": round(final_temp, 2),
                            "predicted_indoor": round(predicted_indoor, 2),
                            "last_prediction_time": (
                                datetime.now(timezone.utc).isoformat()
                            ),
                            "temperature_error": round(
                                abs(error_target_vs_actual), 3
                            ),
                            "pre_cool_active": _pre_cool_active,
                            "pre_cool_peak_temp": round(
                                float(
                                    _pre_cool_result.get("peak_temp", 0)
                                ), 1
                            ) if _pre_cool_result else None,
                            "pre_cool_peak_hour": round(
                                float(
                                    _pre_cool_result.get("peak_hour", 0)
                                ), 1
                            ) if _pre_cool_result else None,
                            # Note: ThermalEquilibriumModel trust metrics
                            # moved to sensor.ml_heating_learning to
                            # eliminate redundancy
                        }
                    )
                    ha_client.set_state(
                        heating_state_entity_id,
                        1 if confidence < config.CONFIDENCE_THRESHOLD else 0,
                        attributes_state,
                        round_digits=None,
                    )
                except Exception:
                    logging.debug(
                        "Failed to write ML state to HA.", exc_info=True
                    )
            else:
                logging.debug(
                    "🔍 SHADOW MODE: Skipping ML state sensor updates"
                )

            # --- Shadow Mode Status Logging ---
            if shadow_mode.shadow_deployment:
                logging.debug(
                    "🔍 SHADOW MODE: Enabled via config (SHADOW_MODE=true)"
                )
            elif not ml_heating_enabled:
                logging.debug(
                    "🔍 SHADOW MODE: ML control disabled via %s",
                    config.ML_HEATING_CONTROL_ENTITY_ID,
                )
            else:
                logging.debug(
                    "✅ ACTIVE MODE: ML controlling heating via %s",
                    config.ML_HEATING_CONTROL_ENTITY_ID,
                )

            # --- Shadow Mode Comparison Logging ---
            if effective_shadow_mode:
                # Read what the heat curve actually set
                heat_curve_temp = ha_client.get_state(
                    config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID, all_states
                )

                if (
                    heat_curve_temp is not None
                    and heat_curve_temp != final_temp
                ):
                    # Simple comparison without model prediction
                    logging.debug(
                        "SHADOW MODE: ML would set %.1f°C, HC set %.1f°C | "
                        "Target: %.1f°C",
                        final_temp,
                        heat_curve_temp,
                        target_indoor_temp,
                    )

            # Shadow metrics now handled by ThermalEquilibriumModel

            # Use the actual rounded temperature that was applied to HA
            applied_temp = (
                final_temp if not config.SHADOW_MODE else final_temp
            )

            thermal_params = {}
            # Calculate what the applied temperature will actually predict
            try:
                if not effective_shadow_mode and "wrapper" in locals():
                    # Get prediction for the applied smart-rounded temperature
                    pv_hist_applied = features_dict.get("pv_power_history", [])
                    pv_now_applied = features_dict.get("pv_now", 0.0)
                    # If actual PV is zero (sun has set), use zero instead of lagged values
                    if pv_now_applied == 0:
                        pv_applied = 0.0
                    else:
                        pv_applied = (sum(pv_hist_applied) / len(pv_hist_applied)) if (pv_hist_applied and len(pv_hist_applied) > 0) else pv_now_applied
                    
                    applied_prediction = wrapper.predict_indoor_temp(
                        outlet_temp=applied_temp,
                        outdoor_temp=thermal_params.get(
                            "outdoor_temp", outdoor_temp
                        ),
                        current_indoor=prediction_indoor_temp,
                        pv_power=thermal_params.get(
                            "pv_power",
                            pv_applied
                        ),
                        fireplace_on=thermal_params.get(
                            "fireplace_on", fireplace_on
                        ),
                        tv_on=thermal_params.get(
                            "tv_on", features_dict.get("tv_on", 0.0)
                        ),
                    )
                    if applied_prediction is None:
                        applied_prediction = predicted_indoor  # Fallback
                else:
                    # Shadow mode or wrapper not available
                    applied_prediction = predicted_indoor
            except Exception as e:
                logging.warning(f"Failed to get applied temp prediction: {e}")
                applied_prediction = predicted_indoor

            log_message = (
                "Target: %.1f°C | Suggested: %.1f°C | Applied: %.1f°C | "
                "Actual Indoor: %.2f°C | Predicted Indoor: %.2f°C | "
                "Confidence: %.3f"
            )
            logging.debug(
                log_message,
                target_indoor_temp,
                suggested_temp,
                applied_temp,
                actual_indoor,
                applied_prediction,  # Now shows prediction for applied temp
                confidence,
            )

            log_message = (
                "Target: %.1f°C | Suggested: %.1f°C | Applied: %.1f°C | "
                "Actual Indoor: %.2f°C | Predicted Indoor: %.2f°C | "
                "Confidence: %.3f"
            )
            logging.info(
                log_message,
                target_indoor_temp,
                suggested_temp,
                applied_temp,
                actual_indoor,
                applied_prediction,  # Now shows prediction for applied temp
                confidence,
            )

            # --- Step 6: State Persistence for Next Run ---
            # Model saving is now handled by ThermalEquilibriumModel in
            # model_wrapper

            # The features and indoor temperature from the *current* run are
            # saved to a file. This data will be loaded at the start of the
            # next loop iteration to be used in the "Online Learning" step.
            # Note: We save final_temp here, but will read the actual
            # applied temp from ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID at the
            # start of the next cycle (after it has had time to update).
            # Inject fireplace_on into features_dict so it is
            # available in learning_features on the next cycle.
            features_dict["fireplace_on"] = (
                float(fireplace_on) if fireplace_on else 0.0
            )
            state_to_save = {
                "last_run_features": features_dict,
                "last_indoor_temp": actual_indoor,
                "last_avg_other_rooms_temp": avg_other_rooms_temp,
                "last_fireplace_on": fireplace_on,
                "last_final_temp": final_temp,
                "last_climate_mode": climate_mode,
                "last_target_indoor_temp": (
                    float(target_indoor_temp)
                    if target_indoor_temp is not None
                    else None
                ),
                "last_predicted_indoor": (
                    applied_prediction
                    if 'applied_prediction' in locals()
                    else None
                ),
                "last_is_blocking": is_blocking,
                "last_blocking_reasons": (
                    blocking_reasons if is_blocking else []
                ),
                "setpoint_hold_cycles_remaining": new_hold_cycles,
            }
            save_state(state_manager=_active_state_manager, **state_to_save)
            # Update in-memory state so the idle poll uses fresh data
            state.update(state_to_save)

            # --- Publish auxiliary sensors ---
            try:
                ha_client.publish_last_run_features(features_dict)
            except Exception:
                logging.debug(
                    "Failed to publish features sensor.", exc_info=True
                )

            if price_data is not None:
                try:
                    # Price info is flattened into metadata by
                    # calculate_optimal_outlet_temp
                    price_info = {
                        k: metadata.get(k)
                        for k in (
                            "price_eur_kwh",
                            "price_level",
                            "price_cheap_threshold",
                            "price_expensive_threshold",
                            "price_target_offset",
                        )
                        if metadata.get(k) is not None
                    }
                    if price_info:
                        ha_client.publish_price_level(price_info)
                except Exception:
                    logging.debug(
                        "Failed to publish price level sensor.",
                        exc_info=True,
                    )

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

        # CYCLE END DEBUG LOGGING
        cycle_end_time = time.time()
        cycle_duration = cycle_end_time - cycle_start_time
        cycle_end_datetime = datetime.now()

        logging.debug(
            f"✅ CYCLE {cycle_number} END: "
            f"{cycle_end_datetime.strftime('%H:%M:%S')} "
            f"(duration: {cycle_duration:.1f}s)"
        )

        last_cycle_end_time = cycle_end_time

        # Poll for blocking events during the idle period so defrost
        # starts/ends are detected quickly. This call will block until the
        # next cycle is due, or until a blocking event starts or ends.
        logging.debug(
            f"💤 POLLING START: Waiting "
            f"{PhysicsConstants.CYCLE_INTERVAL_MINUTES}min "
            "until next cycle..."
        )
        blocking_manager.poll_for_blocking(ha_client, state, sensor_buffer)

        poll_end_time = time.time()
        poll_duration = poll_end_time - cycle_end_time
        logging.debug(
            f"⏰ POLLING END: Waited {poll_duration/60:.1f}min, starting "
            "next cycle..."
        )


if __name__ == "__main__":
    main()

