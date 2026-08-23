"""
Pre-dispatch step functions for the main control loop.

These functions run *before* the state router dispatches to a route handler.
They handle:
- Sensor buffer updates and thermodynamic sensor export
- Shadow mode resolution
- Online learning from previous cycle
- Grace period detection and state preservation

All functions accept the LoopState (cross-cycle) and per-cycle runtime values,
keeping the main loop body thin.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from . import config
from .ha_client import get_sensor_attributes
from .heating_controller import BlockingStateManager, HeatingSystemStateChecker
from .loop_state import LoopState
from .physics_features import calculate_thermodynamic_metrics
from .shadow_mode import get_shadow_output_entity_id, resolve_shadow_mode
from .state_manager import load_state


# ---------------------------------------------------------------------------
# Sensor buffer + thermodynamic metrics
# ---------------------------------------------------------------------------


def update_sensor_buffer_and_thermo(
    loop: LoopState,
    ha_client: Any,
    all_states: Any,
    influx_service: Any,
) -> bool:
    """Push new readings to sensor buffer and export thermodynamic sensors.

    Returns True if thermodynamic metrics were successfully written to InfluxDB.
    """
    thermodynamic_written = False

    if not all_states:
        return thermodynamic_written

    current_time = datetime.now(timezone.utc)

    def get_float_state(entity_id: str) -> float | None:
        try:
            val = ha_client.get_state(entity_id, all_states)
            return float(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    # Push new readings to buffer
    buffer_updates = {
        config.INDOOR_TEMP_ENTITY_ID: get_float_state(config.INDOOR_TEMP_ENTITY_ID),
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID: get_float_state(
            config.ACTUAL_OUTLET_TEMP_ENTITY_ID
        ),
        config.TARGET_OUTLET_TEMP_ENTITY_ID: get_float_state(
            config.TARGET_OUTLET_TEMP_ENTITY_ID
        ),
        config.OUTDOOR_TEMP_ENTITY_ID: get_float_state(config.OUTDOOR_TEMP_ENTITY_ID),
        config.INLET_TEMP_ENTITY_ID: get_float_state(config.INLET_TEMP_ENTITY_ID),
        config.FLOW_RATE_ENTITY_ID: get_float_state(config.FLOW_RATE_ENTITY_ID),
    }

    for entity_id, value in buffer_updates.items():
        if value is not None:
            loop.sensor_buffer.add_reading(entity_id, value, current_time)

    # --- Real-time Thermodynamic Sensors ---
    try:
        power_consumption = get_float_state(config.POWER_CONSUMPTION_ENTITY_ID)
        current_outlet = buffer_updates.get(config.ACTUAL_OUTLET_TEMP_ENTITY_ID)
        current_inlet = buffer_updates.get(config.INLET_TEMP_ENTITY_ID)
        current_flow = buffer_updates.get(config.FLOW_RATE_ENTITY_ID)

        if current_outlet is not None:
            thermo_metrics = calculate_thermodynamic_metrics(
                outlet_temp=current_outlet,
                inlet_temp=current_inlet,
                flow_rate=current_flow,
                power_consumption=power_consumption,
            )

            # Export to Home Assistant
            cop_entity_id = get_shadow_output_entity_id(
                "sensor.ml_heating_cop_realtime"
            )
            ha_client.set_state(
                cop_entity_id,
                thermo_metrics["cop_realtime"],
                get_sensor_attributes(cop_entity_id),
                round_digits=2,
            )

            thermal_power_entity_id = get_shadow_output_entity_id(
                "sensor.ml_heating_thermal_power"
            )
            ha_client.set_state(
                thermal_power_entity_id,
                thermo_metrics["thermal_power_kw"],
                get_sensor_attributes(thermal_power_entity_id),
                round_digits=3,
            )

            try:
                influx_service.write_thermodynamic_metrics(thermo_metrics)
                thermodynamic_written = True
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
                    logging.warning("Failed to log thermodynamic metrics: %s", e)

            logging.debug(
                "Thermodynamic sensors updated: COP=%.2f, Power=%.3fkW",
                thermo_metrics["cop_realtime"],
                thermo_metrics["thermal_power_kw"],
            )
    except Exception as e:
        logging.warning("Failed to update thermodynamic sensors: %s", e)

    return thermodynamic_written


# ---------------------------------------------------------------------------
# Shadow mode resolution
# ---------------------------------------------------------------------------


def resolve_shadow_mode_for_cycle(ha_client: Any, all_states: Any):
    """Determine effective shadow mode for this cycle.

    Returns (shadow_mode_obj, effective_shadow_mode_bool).
    """
    ml_heating_enabled = None
    if all_states:
        ml_heating_enabled = ha_client.get_state(
            config.ML_HEATING_CONTROL_ENTITY_ID,
            all_states,
            is_binary=True,
        )

    if ml_heating_enabled is None:
        if all_states:
            logging.warning(
                "Cannot read %s, defaulting to shadow mode",
                config.ML_HEATING_CONTROL_ENTITY_ID,
            )
        ml_heating_enabled = False

    shadow_mode = resolve_shadow_mode(ml_heating_enabled=ml_heating_enabled)
    return shadow_mode, shadow_mode.effective_shadow_mode


# ---------------------------------------------------------------------------
# Online learning from previous cycle
# ---------------------------------------------------------------------------


def run_online_learning(
    ha_client: Any,
    all_states: Any,
    state: dict,
    effective_shadow_mode: bool,
    climate_mode: str,
    wrapper: Any,
) -> None:
    """Execute online learning from the previous cycle's results.

    This allows the thermal model to continuously adapt to actual house
    behavior, whether running in active mode or shadow mode.
    """
    last_run_features = state.get("last_run_features")
    last_indoor_temp = state.get("last_indoor_temp")
    last_final_temp_stored = state.get("last_final_temp")
    last_avg_other_rooms_temp = state.get("last_avg_other_rooms_temp")

    if (
        last_run_features is None
        or last_indoor_temp is None
        or last_final_temp_stored is None
    ):
        logging.debug("Skipping online learning: no data from previous cycle")
        return

    # Read the actual target outlet temp that was applied
    actual_applied_temp = ha_client.get_state(
        config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID, all_states
    )
    if actual_applied_temp is None:
        actual_applied_temp = last_final_temp_stored

    # Get current indoor temperature to calculate actual change
    current_indoor = ha_client.get_state(config.INDOOR_TEMP_ENTITY_ID, all_states)

    if current_indoor is None:
        logging.debug("Skipping online learning: current indoor temp unavailable")
        return

    actual_indoor_change = current_indoor - last_indoor_temp
    previous_cycle_climate_mode = state.get("last_climate_mode") or climate_mode

    # Handle corrupted last_run_features
    if isinstance(last_run_features, str):
        logging.error(
            "ERROR: last_run_features corrupted as string - attempting to recover"
        )
        try:
            last_run_features = json.loads(last_run_features)
            logging.info("✅ Recovered features from JSON string")
        except (json.JSONDecodeError, ValueError):
            logging.error("❌ Cannot recover features from string, using empty dict")
            last_run_features = {}

    if isinstance(last_run_features, pd.DataFrame):
        learning_features = last_run_features.copy().to_dict(orient="records")[0]
    elif isinstance(last_run_features, dict):
        learning_features = last_run_features.copy()
    else:
        learning_features = last_run_features.copy() if last_run_features else {}

    # Determine target indoor temp for learning context
    persisted_target_temp = state.get("last_target_indoor_temp")
    if persisted_target_temp is None:
        persisted_target_temp = learning_features.get("target_temp")

    if persisted_target_temp is None:
        target_entity_id = config.TARGET_INDOOR_TEMP_ENTITY_ID
        if previous_cycle_climate_mode == "cooling" and getattr(
            config, "TARGET_INDOOR_TEMP_COOLING_ENTITY_ID", ""
        ):
            target_entity_id = config.TARGET_INDOOR_TEMP_COOLING_ENTITY_ID
        persisted_target_temp = ha_client.get_state(target_entity_id, all_states)

    if persisted_target_temp is not None:
        target_indoor_temp = float(persisted_target_temp)
    else:
        target_indoor_temp = last_indoor_temp
        logging.debug(
            "target_indoor_temp unavailable for previous cycle, "
            "using last_indoor_temp fallback: %.2f",
            last_indoor_temp,
        )

    learning_features["target_temp"] = target_indoor_temp
    learning_features["outlet_temp"] = actual_applied_temp
    learning_features["outlet_temp_sq"] = actual_applied_temp**2
    learning_features["outlet_temp_cub"] = actual_applied_temp**3

    # Build prediction context for learning
    try:
        wrapper.set_climate_mode(previous_cycle_climate_mode)

        pv_history = learning_features.get("pv_power_history")
        pv_now_learn = learning_features.get("pv_now", 0.0)
        if pv_now_learn == 0:
            pv_scalar_learn = 0.0
        else:
            pv_scalar_learn = (
                (sum(pv_history) / len(pv_history))
                if (pv_history and len(pv_history) > 0)
                else pv_now_learn
            )

        cloud_cover_forecasts = [
            learning_features.get(f"cloud_cover_forecast_{h}h", 50.0)
            for h in range(1, 7)
        ]
        avg_cloud_cover = (
            sum(cloud_cover_forecasts) / len(cloud_cover_forecasts)
            if cloud_cover_forecasts
            else 50.0
        )

        prediction_context = {
            "outlet_temp": actual_applied_temp,
            "outdoor_temp": learning_features.get("outdoor_temp", 10.0),
            "pv_power": pv_scalar_learn,
            "pv_power_current": pv_now_learn,
            "pv_power_history": pv_history,
            "fireplace_on": float(state.get("last_fireplace_on", False)),
            "tv_on": learning_features.get("tv_on", 0.0),
            "current_indoor": last_indoor_temp,
            "avg_other_rooms_temp": last_avg_other_rooms_temp,
            "thermal_power": learning_features.get("thermal_power_kw", None),
            "climate_mode": previous_cycle_climate_mode,
            "auxiliary_heat": learning_features.get("total_auxiliary_heat_kw", 0.0),
            "target_temp": target_indoor_temp,
            "avg_cloud_cover": avg_cloud_cover,
            "cloud_cover_forecast": cloud_cover_forecasts,
            "inlet_temp": learning_features.get("inlet_temp"),
            "delta_t": learning_features.get("delta_t", 0.0),
            "indoor_temp_gradient": learning_features.get(
                "indoor_temp_gradient", 0.0
            ),
            "indoor_temp_delta_60m": learning_features.get(
                "indoor_temp_delta_60m", 0.0
            ),
            "living_room_temp": learning_features.get("living_room_temp"),
            "outdoor_forecast": [
                learning_features.get(
                    f"temp_forecast_{h}h",
                    learning_features.get("outdoor_temp", 10.0),
                )
                for h in range(1, config.TRAJECTORY_STEPS + 1)
            ],
            "pv_forecast": [
                learning_features.get(f"pv_forecast_{h}h", 0.0)
                for h in range(1, config.TRAJECTORY_STEPS + 1)
            ],
        }

        was_shadow_mode_cycle = effective_shadow_mode

        # Determine learning mode and get model prediction
        try:
            if was_shadow_mode_cycle:
                learning_mode = "shadow_mode_hc_trajectory"
            else:
                learning_mode = "active_mode_ml_trajectory"

            # Use persisted prediction from previous cycle when available
            _stored_pred = state.get("last_predicted_indoor")
            if _stored_pred is not None and not was_shadow_mode_cycle:
                model_predicted_temp = float(_stored_pred)
                learning_mode = "active_mode_persisted_prediction"
                logging.debug(
                    "♻️ Using persisted predicted indoor %.2f°C from previous cycle",
                    model_predicted_temp,
                )
            else:
                # Fallback: re-run trajectory prediction
                _learn_outdoor_now = prediction_context.get("outdoor_temp", 10.0)
                _learn_outdoor_arr = [_learn_outdoor_now] + prediction_context.get(
                    "outdoor_forecast",
                    [_learn_outdoor_now] * config.TRAJECTORY_STEPS,
                )
                _learn_pv_forecast = prediction_context.get("pv_forecast", None)
                trajectory = wrapper.thermal_model.predict_thermal_trajectory(
                    current_indoor=last_indoor_temp,
                    target_indoor=last_indoor_temp,
                    outlet_temp=actual_applied_temp,
                    outdoor_temp=_learn_outdoor_arr,
                    time_horizon_hours=float(config.TRAJECTORY_STEPS),
                    time_step_minutes=config.CYCLE_INTERVAL_MINUTES,
                    pv_power=prediction_context.get("pv_power", 0.0),
                    pv_forecasts=_learn_pv_forecast,
                    fireplace_on=prediction_context.get("fireplace_on", 0.0),
                    tv_on=prediction_context.get("tv_on", 0.0),
                    cloud_cover_pct=prediction_context.get("avg_cloud_cover", 50.0),
                    inlet_temp=prediction_context.get("inlet_temp"),
                    delta_t_floor=prediction_context.get("delta_t", 0.0),
                    thermal_power=None,
                )

                predicted_indoor_temp = (
                    trajectory["trajectory"][0]
                    if trajectory and trajectory.get("trajectory")
                    else last_indoor_temp
                )

                if predicted_indoor_temp is None:
                    logging.warning(
                        "Skipping online learning (%s): prediction returned None",
                        learning_mode,
                    )
                    return

                model_predicted_temp = predicted_indoor_temp

        except Exception as e:
            logging.warning("Skipping online learning: thermal model prediction error: %s", e)
            return

        # Check if previous cycle was a grace period passthrough
        enhanced_prediction_context = prediction_context.copy()
        if (
            isinstance(last_run_features, dict)
            and last_run_features.get("learning_mode") == "grace_period_passthrough"
        ):
            learning_mode = "grace_period_passthrough"

        enhanced_prediction_context["learning_mode"] = learning_mode
        enhanced_prediction_context["was_shadow_mode_cycle"] = was_shadow_mode_cycle
        enhanced_prediction_context["ml_calculated_temp"] = last_final_temp_stored
        enhanced_prediction_context["hc_applied_temp"] = actual_applied_temp

        # Execute learning
        wrapper.learn_from_prediction_feedback(
            predicted_temp=model_predicted_temp,
            actual_temp=current_indoor,
            prediction_context=enhanced_prediction_context,
            timestamp=datetime.now().isoformat(),
            is_blocking_active=False,
            effective_shadow_mode=effective_shadow_mode,
        )

        logging.debug(
            "✅ Online learning: applied_temp=%.1f°C, actual_change=%.3f°C, cycle=%d",
            actual_applied_temp,
            actual_indoor_change,
            wrapper.cycle_count,
        )
    except Exception as e:
        logging.warning("Online learning failed: %s", e, exc_info=True)

    # Shadow mode error tracking
    if effective_shadow_mode and actual_applied_temp != last_final_temp_stored:
        logging.debug(
            "Shadow mode: ML would set %.1f°C, HC set %.1f°C",
            last_final_temp_stored,
            actual_applied_temp,
        )


# ---------------------------------------------------------------------------
# Grace period handling
# ---------------------------------------------------------------------------


def handle_grace_period(
    ha_client: Any,
    state: dict,
    effective_shadow_mode: bool,
) -> bool:
    """Detect whether a post-blocking grace period is active.

    Returns True if grace period is active (caller should dispatch to
    run_grace_period_route, which handles state persistence).
    Returns False if no grace period is active.
    """
    blocking_manager = BlockingStateManager()
    is_grace_period = blocking_manager.handle_grace_period(
        ha_client, state, shadow_mode=effective_shadow_mode
    )

    return is_grace_period


# ---------------------------------------------------------------------------
# Startup sensor validation
# ---------------------------------------------------------------------------


def validate_sensors_once(
    all_states: Any,
    ha_client: Any,
) -> bool:
    """Run one-time startup sensor validation. Returns True when done."""
    if not all_states:
        return False

    try:
        if isinstance(all_states, list):
            _known_ids = {
                s.get("entity_id") for s in all_states if isinstance(s, dict)
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
                "associated learning channels will remain empty: %s",
                _missing_optional,
            )
        if not _missing_critical and not _missing_optional:
            logging.info("✅ All configured sensor entity IDs verified in HA.")
        return True
    except Exception as e:
        logging.warning("Startup sensor validation failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Network error state
# ---------------------------------------------------------------------------


def emit_network_error_state(ha_client: Any) -> None:
    """Write NETWORK_ERROR state to HA when states cannot be fetched."""
    try:
        from .ha_client import create_ha_client

        _ha_client = create_ha_client()
        heating_state_entity_id = get_shadow_output_entity_id(
            "sensor.ml_heating_state"
        )
        attributes_state = get_sensor_attributes(heating_state_entity_id)
        attributes_state.update(
            {
                "state_description": "Network Error",
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        )
        _ha_client.set_state(
            heating_state_entity_id,
            3,
            attributes_state,
            round_digits=None,
        )
    except Exception:
        logging.debug("Failed to write NETWORK_ERROR state to HA.", exc_info=True)


# ---------------------------------------------------------------------------
# Blocking / heating active checks
# ---------------------------------------------------------------------------


def check_blocking_state(
    ha_client: Any, all_states: Any
) -> tuple[bool, list]:
    """Check for blocking modes. Returns (is_blocking, blocking_reasons)."""
    blocking_manager = BlockingStateManager()
    return blocking_manager.check_blocking_state(ha_client, all_states)


def check_and_resolve_climate_mode(
    ha_client: Any,
    all_states: Any,
    wrapper: Any,
) -> tuple[bool, str, Any, Any]:
    """Check heating system active and resolve climate mode.

    Returns:
        (heating_active, climate_mode, active_state_manager, state_or_None)
        If state_or_None is not None, it means the state file changed and
        state was reloaded.
    """
    heating_checker = HeatingSystemStateChecker()
    heating_active = heating_checker.check_heating_active(ha_client, all_states)

    if not heating_active:
        # For IDLE state, use heating state manager
        # (per requirement: idle saves in heating unified thermal state)
        wrapper.set_climate_mode("heating")
        return False, "heating", wrapper.state_manager, None

    climate_mode = heating_checker.get_climate_mode(ha_client, all_states)
    _prev_state_manager = wrapper.state_manager
    wrapper.set_climate_mode(climate_mode)
    _active_state_manager = wrapper.state_manager

    # Reload state if state file changed (mode transition)
    reloaded_state = None
    if _active_state_manager is not _prev_state_manager:
        reloaded_state = load_state(state_manager=_active_state_manager)
        logging.info(
            "♻️ Climate mode transition → reloaded operational state from %s",
            _active_state_manager.state_file,
        )

        # --- Warm restart on genuine mode change ---
        # Trigger sys.exit(0) so supervisord restarts the process with a fresh
        # module environment and the correct mode profile applied at startup.
        # A sentinel file prevents an infinite restart loop: if the sentinel
        # already records the target mode, we already restarted for this
        # transition and should proceed normally.
        _prev_mode = (reloaded_state or {}).get("last_climate_mode")
        _sentinel = "/data/config/warm_restart_mode_sentinel"
        _sentinel_content: str | None = None
        _sentinel_read_error = False
        try:
            if os.path.exists(_sentinel):
                with open(_sentinel, encoding="utf-8") as _sf:
                    _sentinel_content = _sf.read().strip()
        except OSError as _se:
            logging.warning(
                "⚠️ Warm-restart sentinel read error %s: %s — skipping restart this cycle",
                _sentinel,
                _se,
            )
            _sentinel_read_error = True

        if not _sentinel_read_error:
            if _sentinel_content == climate_mode:
                # We already warm-restarted for this transition — clear the
                # sentinel so the next cycle proceeds without triggering again.
                try:
                    os.remove(_sentinel)
                except OSError:
                    pass
                logging.debug(
                    "🔄 Warm-restart sentinel cleared for mode '%s'", climate_mode
                )
            elif _prev_mode and _prev_mode != climate_mode:
                # Genuine transition: write sentinel then exit so supervisord
                # restarts the process in the new mode.
                _sentinel_written = False
                try:
                    os.makedirs(os.path.dirname(_sentinel), exist_ok=True)
                    with open(_sentinel, "w", encoding="utf-8") as _sf:
                        _sf.write(climate_mode)
                    _sentinel_written = True
                except OSError as _we:
                    logging.warning(
                        "⚠️ Could not write warm-restart sentinel %s: %s — skipping restart",
                        _sentinel,
                        _we,
                    )
                if _sentinel_written:
                    logging.info(
                        "🔄 Climate mode changed %s → %s — warm restart to apply new profile settings",
                        _prev_mode,
                        climate_mode,
                    )
                    sys.exit(0)

    if climate_mode == "cooling":
        logging.info(
            "❄️ COOLING MODE: ML will calculate cooling outlet "
            "temperature (outlet < inlet) — using cooling state %s",
            _active_state_manager.state_file,
        )

    return True, climate_mode, _active_state_manager, reloaded_state


# ---------------------------------------------------------------------------
# LoopState initialization factory
# ---------------------------------------------------------------------------


def initialize_loop_state(
    sensor_buffer: Any,
    influx_service: Any,
) -> LoopState:
    """Create and populate a LoopState with all runtime objects.

    Handles the conditional lazy initialization of cooling ML model,
    cooling observation buffer, and heating observation buffer.
    """
    from .model_wrapper import get_enhanced_model_wrapper

    wrapper = get_enhanced_model_wrapper()
    detected_mode = "heating"
    try:
        from .ha_client import create_ha_client

        ha_client = create_ha_client()
        all_states = ha_client.get_all_states()
        detected_mode = HeatingSystemStateChecker().get_climate_mode(
            ha_client, all_states
        )
        wrapper.set_climate_mode(detected_mode)
        logging.info(
            "🌡️ initialize_loop_state: climate mode detected as '%s' before initial export.",
            detected_mode,
        )
    except Exception as mode_err:
        logging.warning(
            "⚠️ Could not detect climate mode at startup (defaulting to 'heating'): %s",
            mode_err,
        )

    # Apply mode profile before initializing mode-dependent runtime components.
    from .mode_profiles import apply_profile as _apply_mode_profile

    _apply_mode_profile(detected_mode)
    try:
        wrapper.export_metrics_to_ha()
        logging.info("✅ Initial metrics exported to HA successfully.")
    except Exception as e:
        logging.error(
            "❌ FAILED to export initial metrics to HA: %s", e, exc_info=True
        )

    loop = LoopState(
        sensor_buffer=sensor_buffer,
        influx_service=influx_service,
        wrapper=wrapper,
        blocking_entities=[
            config.DHW_STATUS_ENTITY_ID,
            config.DEFROST_STATUS_ENTITY_ID,
            config.DISINFECTION_STATUS_ENTITY_ID,
            config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
        ],
    )

    # --- Cooling ML model + observation buffer (lazy init once) ---
    loop.cooling_ml_model_type = getattr(config, "PRE_COOL_MODEL_TYPE", "trajectory")
    if getattr(config, "PRE_COOL_ENABLED", True):
        _steps_per_hour = round(
            60 / float(getattr(config, "CYCLE_INTERVAL_MINUTES", 10))
        )
        try:
            from .cooling_ml_model import CoolingMLModel
            from .cooling_ml_observation_buffer import CoolingObservationBuffer

            _horizon_h = int(getattr(config, "PRE_COOL_HORIZON_HOURS", 12))
            loop.cooling_ml_model = CoolingMLModel(
                model_path=getattr(
                    config,
                    "COOLING_ML_MODEL_PATH",
                    "/opt/ml_heating/cooling_ml_model.joblib",
                ),
                metadata_path=getattr(
                    config,
                    "COOLING_ML_METADATA_PATH",
                    "/opt/ml_heating/cooling_ml_metadata.json",
                ),
                steps_per_hour=_steps_per_hour,
            )
            loop.cooling_ml_model.load()
            loop.cooling_obs_buffer = CoolingObservationBuffer(
                path=getattr(
                    config,
                    "COOLING_ML_OBSERVATION_BUFFER_PATH",
                    "/opt/ml_heating/cooling_ml_obs_buffer.json",
                ),
                max_n=int(getattr(config, "COOLING_ML_BUFFER_MAX_N", 500)),
                min_training_samples=int(
                    getattr(config, "COOLING_ML_MIN_TRAINING_SAMPLES", 200)
                ),
                retrain_trigger_k=int(
                    getattr(config, "COOLING_ML_RETRAIN_TRIGGER_K", 50)
                ),
                horizon_steps=_horizon_h * _steps_per_hour,
            )
        except Exception as _cml_init_err:
            logging.warning(
                "Cooling ML init failed (non-fatal): %s", _cml_init_err
            )

    # --- Heating Correction ML observation buffer (always init) ---
    _heating_obs_steps_per_hour = round(
        60 / float(getattr(config, "CYCLE_INTERVAL_MINUTES", 10))
    )
    try:
        from .heating_correction_ml_observation_buffer import (
            HeatingCorrectionObservationBuffer,
        )

        _heating_label_h = int(
            getattr(config, "HEATING_ML_LABEL_HORIZON_H", 4)
        )
        loop.heating_obs_buffer = HeatingCorrectionObservationBuffer(
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

    return loop
