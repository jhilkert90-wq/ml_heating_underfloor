"""
Cycle route handlers for each system state.

Each route handler encapsulates the logic for one system state (HEATING,
COOLING, BLOCKING, IDLE).  The main loop determines the state once, then
calls the appropriate route.

Shared steps that run across multiple states are defined as standalone
functions and called explicitly from each route that needs them, giving
full control over execution order.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .cycle_context import CycleContext
from .ha_client import get_sensor_attributes
from .heating_controller import SensorDataManager
from .model_wrapper import simplified_outlet_prediction
from .physics_features import build_physics_features
from .shadow_mode import get_shadow_output_entity_id
from .state_manager import save_state
from .temperature_control import apply_ema_smoothing
from .thermal_constants import PhysicsConstants


# ---------------------------------------------------------------------------
# Shared step functions (used by multiple routes)
# ---------------------------------------------------------------------------


def step_get_sensor_data(ctx: CycleContext) -> bool:
    """Fetch current sensor values.  Returns False if critical sensors missing."""
    sensor_manager = SensorDataManager()
    sensor_data, missing_sensors = sensor_manager.get_sensor_data(
        ctx.ha_client, ctx.cycle_number
    )
    if missing_sensors:
        sensor_manager.handle_missing_sensors(ctx.ha_client, missing_sensors)
        return False

    ctx.sensor_data = sensor_data
    ctx.target_indoor_temp = sensor_data["target_indoor_temp"]
    ctx.actual_indoor = sensor_data["actual_indoor"]
    ctx.actual_outlet_temp = sensor_data["actual_outlet_temp"]
    ctx.avg_other_rooms_temp = sensor_data["avg_other_rooms_temp"]
    ctx.fireplace_on = sensor_data["fireplace_on"]
    ctx.outdoor_temp = sensor_data["outdoor_temp"]
    ctx.owm_temp = sensor_data["owm_temp"]
    return True


def step_apply_cooling_target(ctx: CycleContext) -> None:
    """In cooling mode, override target from cooling-specific entity."""
    if (
        ctx.climate_mode == "cooling"
        and getattr(config, "TARGET_INDOOR_TEMP_COOLING_ENTITY_ID", "")
    ):
        _cooling_target = ctx.ha_client.get_state(
            config.TARGET_INDOOR_TEMP_COOLING_ENTITY_ID, ctx.all_states
        )
        if _cooling_target is not None:
            try:
                ctx.target_indoor_temp = float(_cooling_target)
                logging.info(
                    "❄️ Using cooling target entity: %.1f°C",
                    ctx.target_indoor_temp,
                )
            except (TypeError, ValueError):
                logging.warning(
                    "❄️ Cooling target entity returned non-numeric value '%s'"
                    " — using heating target %.1f°C instead.",
                    _cooling_target,
                    float(ctx.target_indoor_temp),
                )


def step_determine_prediction_indoor(ctx: CycleContext) -> None:
    """Choose prediction indoor temp (fireplace-aware + transient drop filter)."""
    if ctx.fireplace_on:
        ctx.prediction_indoor_temp = ctx.avg_other_rooms_temp
        logging.debug("Fireplace ON. Using avg other rooms temp for prediction.")
    else:
        ctx.prediction_indoor_temp = ctx.actual_indoor
        logging.debug("Fireplace is OFF. Using main indoor temp for prediction.")

    # TRANSIENT DROP FILTER: Only applies in HEATING mode.
    if (
        ctx.climate_mode != "cooling"
        and ctx.last_indoor_temp is not None
        and ctx.prediction_indoor_temp is not None
    ):
        _drop = ctx.last_indoor_temp - ctx.prediction_indoor_temp
        if _drop > 0.25:
            _extrapolated = ctx.last_indoor_temp - 0.02
            logging.warning(
                "🚪 Transient drop filter: indoor temp dropped "
                "%.3f°C (%.2f → %.2f). Using extrapolated temp "
                "%.2f instead to prevent unnecessary heating.",
                _drop,
                ctx.last_indoor_temp,
                ctx.prediction_indoor_temp,
                _extrapolated,
            )
            ctx.prediction_indoor_temp = _extrapolated


def step_build_features(ctx: CycleContext) -> bool:
    """Build physics features.  Returns False if features unavailable."""
    features, outlet_history = build_physics_features(
        ctx.ha_client,
        ctx.influx_service,
        ctx.sensor_buffer,
        climate_mode=ctx.climate_mode,
        target_indoor_temp_override=ctx.target_indoor_temp,
    )
    if features is None:
        logging.warning("Feature building failed, skipping cycle.")
        return False

    ctx.features = features
    ctx.outlet_history = outlet_history

    # Convert DataFrame to dict for safe access
    if isinstance(features, pd.DataFrame):
        ctx.features_dict = (
            features.iloc[0].to_dict() if not features.empty else {}
        )
    else:
        ctx.features_dict = features if isinstance(features, dict) else {}
    return True


def step_dynamic_trajectory(ctx: CycleContext) -> None:
    """Dynamic trajectory scaling (PV-aware). Heating-specific but shared infra."""
    _pv_forecast_traj: list[float] | None = None
    if getattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False):
        try:
            from .pv_trajectory import compute_dynamic_trajectory_steps

            _pv_now_traj = float(
                ctx.features_dict.get("pv_now_electrical", 0.0)
            )
            _pv_forecast_traj = [
                float(
                    ctx.features_dict.get(
                        f"pv_forecast_electrical_{h}h",
                        ctx.features_dict.get(f"pv_forecast_{h}h", 0.0),
                    )
                )
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
            logging.warning("Dynamic trajectory scaling failed: %s", _exc)

    # Read electricity price
    ctx.price_data = None
    if getattr(config, "ELECTRICITY_PRICE_ENABLED", False):
        try:
            from .price_optimizer import get_price_optimizer

            optimizer = get_price_optimizer()
            optimizer.refresh_prices_if_needed(ctx.ha_client)
            ctx.price_data = optimizer.get_price_data_for_features()
        except Exception as exc:
            logging.warning("Failed to read electricity price: %s", exc)

    # Suppress price in forecast trajectory mode
    if (
        ctx.price_data is not None
        and getattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False)
        and getattr(config, "PV_TRAJ_DISABLE_PRICE_IN_FORECAST_MODE", True)
    ):
        try:
            from .pv_trajectory import is_forecast_trajectory_active

            _fc_pv_now = float(
                ctx.features_dict.get("pv_now_electrical", 0.0)
            )
            _fc_forecast = (
                _pv_forecast_traj
                if _pv_forecast_traj is not None
                else [
                    float(
                        ctx.features_dict.get(
                            f"pv_forecast_electrical_{h}h",
                            ctx.features_dict.get(f"pv_forecast_{h}h", 0.0),
                        )
                    )
                    for h in range(
                        1,
                        int(getattr(config, "PV_TRAJ_MAX_STEPS", 12)) + 1,
                    )
                ]
            )
            if is_forecast_trajectory_active(_fc_pv_now, _fc_forecast):
                ctx.price_data = None
                logging.info(
                    "☀️ Forecast trajectory active: price offset suppressed"
                )
        except Exception as _exc:
            logging.debug(
                "Forecast trajectory price suppression check failed: %s", _exc
            )


def step_prediction(ctx: CycleContext) -> None:
    """Run simplified_outlet_prediction."""
    error_target_vs_actual = ctx.target_indoor_temp - ctx.prediction_indoor_temp

    suggested_temp, confidence, metadata = simplified_outlet_prediction(
        ctx.features,
        ctx.prediction_indoor_temp,
        ctx.target_indoor_temp,
        price_data=ctx.price_data,
    )
    ctx.suggested_temp = suggested_temp
    ctx.final_temp = suggested_temp
    ctx.confidence = confidence
    ctx.metadata = metadata
    ctx.predicted_indoor = metadata.get(
        "predicted_indoor", ctx.prediction_indoor_temp
    )

    logging.debug(
        "Model Wrapper: temp=%.1f°C, error=%.3f°C, confidence=%.3f",
        suggested_temp,
        abs(error_target_vs_actual),
        confidence,
    )


def _resolve_pre_cool_min_target(
    ctx: CycleContext, original_target: float, offset: float
) -> float:
    """Resolve the minimum allowed indoor target for pre-cooling."""
    entity_minimum = None
    entity_id = getattr(config, "TARGET_INDOOR_TEMP_COOLING_ENTITY_ID", "")
    if entity_id:
        entity_state = (ctx.all_states or {}).get(entity_id) or {}
        attrs = entity_state.get("attributes") or {}
        minimum = attrs.get("min")
        if minimum is not None:
            try:
                entity_minimum = float(minimum)
            except (TypeError, ValueError):
                logging.debug(
                    "Ignoring non-numeric cooling target min attribute %r",
                    minimum,
                )
    max_offset = min(
        float(getattr(config, "PRE_COOL_MAX_OFFSET_K", offset)),
        float(offset),
    )
    target = original_target - max_offset
    return max(entity_minimum, target) if entity_minimum is not None else target


def _apply_shadow_pre_cool_guard(
    ctx: CycleContext, trajectory_result: dict, lgbm_result: dict | None
) -> dict:
    """Suppress implausible trajectory-triggered pre-cool when shadow disagrees."""
    if (
        not lgbm_result
        or not getattr(
            config, "PRE_COOL_SHADOW_DISAGREEMENT_GUARD_ENABLED", True
        )
        or ctx.cooling_ml_model_type != "trajectory"
        or not trajectory_result.get("should_cool_now")
        or ctx.prediction_indoor_temp > ctx.target_indoor_temp
    ):
        return trajectory_result

    lgbm_proba = float(lgbm_result.get("lgbm_proba", 0.0))
    lgbm_peak = float(
        lgbm_result.get(
            "predicted_max_temp",
            lgbm_result.get("peak_temp", ctx.prediction_indoor_temp),
        )
    )
    trajectory_peak = float(
        trajectory_result.get("peak_temp", ctx.prediction_indoor_temp)
    )
    peak_gap = trajectory_peak - lgbm_peak
    max_lgbm_proba = float(
        getattr(config, "PRE_COOL_SHADOW_MAX_LGBM_PROBA", 0.10)
    )
    min_peak_gap = float(
        getattr(config, "PRE_COOL_SHADOW_MIN_PEAK_GAP_K", 2.0)
    )
    if (
        lgbm_result.get("should_cool_now")
        or lgbm_result.get("risk")
        or lgbm_proba > max_lgbm_proba
        or peak_gap < min_peak_gap
    ):
        return trajectory_result

    blocked = dict(trajectory_result)
    blocked["risk"] = False
    blocked["should_cool_now"] = False
    blocked["shadow_blocked"] = True
    blocked["shadow_peak_temp"] = lgbm_peak
    blocked["shadow_lgbm_proba"] = lgbm_proba
    _existing_reason = trajectory_result.get("reason", "")
    _shadow_reason = (
        f"blocked by LGBM shadow (p={lgbm_proba:.3f}, "
        f"peak={lgbm_peak:.1f}°C, Δpeak={peak_gap:.1f}K)"
    )
    blocked["reason"] = (
        f"{_existing_reason}; {_shadow_reason}"
        if _existing_reason
        else _shadow_reason
    )
    logging.info(
        "❄️ PRE-COOL shadow guard blocked trajectory trigger: %s",
        blocked["reason"],
    )
    return blocked


def step_gradual_control(ctx: CycleContext) -> None:
    """Apply gradual temperature change limiting."""
    if ctx.actual_outlet_temp is None:
        return

    max_change = config.MAX_TEMP_CHANGE_PER_CYCLE
    original_temp = ctx.final_temp

    last_blocking_reasons = ctx.state.get("last_blocking_reasons", []) or []
    last_final_temp = ctx.state.get("last_final_temp")

    dhw_like_blockers = {
        config.DHW_STATUS_ENTITY_ID,
        config.DISINFECTION_STATUS_ENTITY_ID,
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID,
    }

    if ctx.effective_shadow_mode:
        baseline = ctx.actual_outlet_temp
        logging.info(
            "Gradual control baseline in shadow mode set to "
            "actual_outlet_temp: %.1f°C",
            baseline,
        )
    elif last_final_temp is not None:
        baseline = last_final_temp
        if any(b in dhw_like_blockers for b in last_blocking_reasons):
            baseline = ctx.actual_outlet_temp
    else:
        baseline = ctx.actual_outlet_temp

    delta = ctx.final_temp - baseline
    if abs(delta) > max_change:
        ctx.final_temp = baseline + np.clip(delta, -max_change, max_change)
        logging.info("--- Gradual Temperature Control ---")
        logging.info(
            "Change from baseline %.1f°C to suggested %.1f°C "
            "exceeds max change of %.1f°C. Capping at %.1f°C.",
            baseline,
            original_temp,
            max_change,
            ctx.final_temp,
        )


def step_ema_smoothing(ctx: CycleContext) -> None:
    """Apply EMA smoothing (bypassed in cooling recovery)."""
    last_final = ctx.state.get("last_final_temp")
    _cooling_recovery_active = (
        ctx.climate_mode == "cooling"
        and getattr(ctx.wrapper, "_cooling_cycle_state", None) == "recovery"
    )
    if _cooling_recovery_active:
        logging.debug(
            "❄️ Cooling recovery: bypassing EMA smoothing "
            "(preserving inlet_temp=%.1f°C)",
            ctx.final_temp,
        )
    else:
        ctx.final_temp = apply_ema_smoothing(ctx.final_temp, last_final)


def step_setpoint_hold(ctx: CycleContext) -> None:
    """Minimum setpoint hold logic."""
    hold_remaining = ctx.state.get("setpoint_hold_cycles_remaining", 0) or 0
    min_hold = int(
        getattr(config, "MIN_SETPOINT_HOLD_CYCLES", config.TRAJECTORY_STEPS)
    )
    held_temp = ctx.state.get("last_final_temp")

    if hold_remaining > 0 and held_temp is not None:
        logging.info(
            "⏱️ Setpoint hold: keeping %.1f°C for %d more cycle(s) "
            "(computed=%.1f°C)",
            held_temp,
            hold_remaining,
            ctx.final_temp,
        )
        ctx.final_temp = held_temp
        ctx.new_hold_cycles = hold_remaining - 1
    else:
        setpoint_changed = (
            held_temp is None
            or abs(ctx.final_temp - held_temp)
            > PhysicsConstants.SETPOINT_CHANGE_THRESHOLD_C
        )
        ctx.new_hold_cycles = max(0, min_hold - 1) if setpoint_changed else 0


def step_update_ha(ctx: CycleContext) -> None:
    """Update Home Assistant with the final temperature and metrics."""
    from .model_wrapper import get_enhanced_model_wrapper
    from .prediction_context import prediction_context_manager

    target_output_entity_id = get_shadow_output_entity_id(
        config.TARGET_OUTLET_TEMP_ENTITY_ID
    )

    if ctx.effective_shadow_mode and not ctx.shadow_mode.shadow_deployment:
        logging.info(
            "🔍 SHADOW MODE: ML prediction calculated but not "
            "applied to heating system"
        )
        logging.info(
            "   Final temp: %.1f°C (calculated but not sent to HA)",
            ctx.final_temp,
        )
    else:
        smart_rounded_temp = round(ctx.final_temp, 1)
        if not ctx.effective_shadow_mode:
            # Smart rounding
            floor_temp = np.floor(ctx.final_temp)
            ceiling_temp = np.ceil(ctx.final_temp)

            if floor_temp == ceiling_temp:
                smart_rounded_temp = int(ctx.final_temp)
            else:
                try:
                    wrapper = get_enhanced_model_wrapper()
                    pv_hist = (
                        ctx.features_dict.get("pv_power_history", [])
                        if isinstance(ctx.features_dict, dict)
                        else []
                    )
                    pv_now_test = (
                        ctx.features_dict.get("pv_now", 0.0)
                        if isinstance(ctx.features_dict, dict)
                        else 0.0
                    )
                    if pv_now_test == 0:
                        pv_val = 0.0
                    else:
                        pv_val = (
                            (sum(pv_hist) / len(pv_hist))
                            if (pv_hist and len(pv_hist) > 0)
                            else pv_now_test
                        )

                    prediction_context_manager.set_features(ctx.features_dict)
                    thermal_features = {
                        "pv_power": pv_val,
                        "pv_power_history": pv_hist,
                        "fireplace_on": (
                            float(ctx.fireplace_on)
                            if ctx.fireplace_on is not None
                            else 0.0
                        ),
                        "tv_on": ctx.features_dict.get("tv_on", 0.0),
                    }
                    prediction_context_manager.create_context(
                        outdoor_temp=ctx.outdoor_temp,
                        pv_power=thermal_features["pv_power"],
                        thermal_features=thermal_features,
                        target_temp=ctx.target_indoor_temp,
                        current_temp=ctx.prediction_indoor_temp,
                    )
                    thermal_params = (
                        prediction_context_manager.get_thermal_model_params()
                    )

                    floor_predicted = wrapper.predict_indoor_temp(
                        outlet_temp=floor_temp,
                        outdoor_temp=thermal_params["outdoor_temp"],
                        current_indoor=ctx.prediction_indoor_temp,
                        pv_power=thermal_params["pv_power"],
                        fireplace_on=thermal_params["fireplace_on"],
                        tv_on=thermal_params["tv_on"],
                    )
                    ceiling_predicted = wrapper.predict_indoor_temp(
                        outlet_temp=ceiling_temp,
                        outdoor_temp=thermal_params["outdoor_temp"],
                        current_indoor=ctx.prediction_indoor_temp,
                        pv_power=thermal_params["pv_power"],
                        fireplace_on=thermal_params["fireplace_on"],
                        tv_on=thermal_params["tv_on"],
                    )

                    if floor_predicted is None or ceiling_predicted is None:
                        smart_rounded_temp = round(ctx.final_temp)
                    else:
                        floor_error = abs(
                            floor_predicted - ctx.target_indoor_temp
                        )
                        ceiling_error = abs(
                            ceiling_predicted - ctx.target_indoor_temp
                        )
                        if floor_error <= ceiling_error:
                            smart_rounded_temp = int(floor_temp)
                        else:
                            smart_rounded_temp = int(ceiling_temp)

                        logging.debug(
                            "Smart rounding: %.2f°C → %d°C "
                            "(floor→%.2f°C [err=%.3f], "
                            "ceiling→%.2f°C [err=%.3f], "
                            "target=%.1f°C)",
                            ctx.final_temp,
                            smart_rounded_temp,
                            floor_predicted,
                            floor_error,
                            ceiling_predicted,
                            ceiling_error,
                            ctx.target_indoor_temp,
                        )
                except Exception as e:
                    smart_rounded_temp = round(ctx.final_temp)
                    logging.warning(
                        "Smart rounding failed (%s), using regular "
                        "rounding: %.2f°C → %d°C",
                        e,
                        ctx.final_temp,
                        smart_rounded_temp,
                    )
        else:
            logging.info(
                "🔍 SHADOW DEPLOYMENT: Publishing ML recommendation to %s",
                target_output_entity_id,
            )

        logging.debug("Setting target outlet temp")
        ctx.ha_client.set_state(
            target_output_entity_id,
            float(smart_rounded_temp),
            get_sensor_attributes(target_output_entity_id),
            round_digits=None,
        )


def step_log_metrics(ctx: CycleContext) -> None:
    """Log thermodynamic metrics to InfluxDB."""
    if not ctx.effective_shadow_mode:
        logging.debug("Logging thermal model metrics")

    if not ctx.thermodynamic_metrics_written_in_sensor_update:
        try:
            thermodynamic_metrics = {
                "cop_realtime": ctx.features_dict.get("cop_realtime", 0.0),
                "thermal_power_kw": ctx.features_dict.get(
                    "thermal_power_kw", 0.0
                ),
                "delta_t": ctx.features_dict.get("delta_t", 0.0),
                "flow_rate": ctx.features_dict.get("flow_rate", 0.0),
                "inlet_temp": ctx.features_dict.get("inlet_temp", 0.0),
            }
            ctx.influx_service.write_thermodynamic_metrics(
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


def step_update_ml_state_sensor(ctx: CycleContext) -> None:
    """Update the ML state sensor in Home Assistant."""
    if ctx.shadow_mode.should_publish_output_entities:
        try:
            heating_state_entity_id = get_shadow_output_entity_id(
                "sensor.ml_heating_state"
            )
            attributes_state = get_sensor_attributes(heating_state_entity_id)
            attributes_state.update(
                {
                    "state_description": "Confidence - Too Low"
                    if ctx.confidence < config.CONFIDENCE_THRESHOLD
                    else "OK - Prediction done",
                    "confidence": round(ctx.confidence, 4),
                    "suggested_temp": round(ctx.suggested_temp, 2),
                    "final_temp": round(ctx.final_temp, 2),
                    "predicted_indoor": round(ctx.predicted_indoor, 2),
                    "last_prediction_time": (
                        datetime.now(timezone.utc).isoformat()
                    ),
                    "temperature_error": round(
                        abs(ctx.target_indoor_temp - ctx.prediction_indoor_temp),
                        3,
                    ),
                    "pre_cool_active": ctx.pre_cool_active,
                    "pre_cool_peak_temp": round(
                        float(ctx.pre_cool_result.get("peak_temp", 0)), 1
                    )
                    if ctx.pre_cool_result
                    else None,
                    "pre_cool_peak_hour": round(
                        float(ctx.pre_cool_result.get("peak_hour", 0)), 1
                    )
                    if ctx.pre_cool_result
                    else None,
                }
            )
            ctx.ha_client.set_state(
                heating_state_entity_id,
                1 if ctx.confidence < config.CONFIDENCE_THRESHOLD else 0,
                attributes_state,
                round_digits=None,
            )
        except Exception:
            logging.debug("Failed to write ML state to HA.", exc_info=True)
    else:
        logging.debug("🔍 SHADOW MODE: Skipping ML state sensor updates")


def step_shadow_comparison(ctx: CycleContext) -> None:
    """Log shadow mode comparison between ML and heat curve."""
    if not ctx.effective_shadow_mode:
        return

    heat_curve_temp = ctx.ha_client.get_state(
        config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID, ctx.all_states
    )
    if heat_curve_temp is not None and heat_curve_temp != ctx.final_temp:
        logging.debug(
            "SHADOW MODE: ML would set %.1f°C, HC set %.1f°C | "
            "Target: %.1f°C",
            ctx.final_temp,
            heat_curve_temp,
            ctx.target_indoor_temp,
        )


def step_save_state(ctx: CycleContext) -> None:
    """Persist cycle state for next iteration."""
    ctx.features_dict["fireplace_on"] = (
        float(ctx.fireplace_on) if ctx.fireplace_on else 0.0
    )
    state_to_save = {
        "last_run_features": ctx.features_dict,
        "last_indoor_temp": ctx.actual_indoor,
        "last_avg_other_rooms_temp": ctx.avg_other_rooms_temp,
        "last_fireplace_on": ctx.fireplace_on,
        "last_final_temp": ctx.final_temp,
        "last_climate_mode": ctx.climate_mode,
        "last_target_indoor_temp": (
            float(ctx.target_indoor_temp)
            if ctx.target_indoor_temp is not None
            else None
        ),
        "last_predicted_indoor": ctx.predicted_indoor,
        "last_is_blocking": ctx.is_blocking,
        "last_blocking_reasons": (
            ctx.blocking_reasons if ctx.is_blocking else []
        ),
        "setpoint_hold_cycles_remaining": ctx.new_hold_cycles,
    }
    save_state(state_manager=ctx.state_manager, **state_to_save)
    ctx.state.update(state_to_save)


def step_publish_auxiliary_sensors(ctx: CycleContext) -> None:
    """Publish feature and price sensors."""
    try:
        ctx.ha_client.publish_last_run_features(ctx.features_dict)
    except Exception:
        logging.debug("Failed to publish features sensor.", exc_info=True)

    if ctx.price_data is not None:
        try:
            price_info = {
                k: ctx.metadata.get(k)
                for k in (
                    "price_eur_kwh",
                    "price_level",
                    "price_cheap_threshold",
                    "price_expensive_threshold",
                    "price_target_offset",
                )
                if ctx.metadata.get(k) is not None
            }
            if price_info:
                ctx.ha_client.publish_price_level(price_info)
        except Exception:
            logging.debug(
                "Failed to publish price level sensor.", exc_info=True
            )


# ---------------------------------------------------------------------------
# Pre-cooling step (cooling-only)
# ---------------------------------------------------------------------------


def step_pre_cooling(ctx: CycleContext) -> None:
    """Predictive overheating prevention (cooling mode only)."""
    if not getattr(config, "PRE_COOL_ENABLED", True):
        return

    try:
        from .overheating_predictor import OverheatingPredictor

        _pre_cool_predictor = OverheatingPredictor()
        _traj_result = _pre_cool_predictor.predict_overheating_risk(
            current_indoor=ctx.prediction_indoor_temp,
            target_cooling=ctx.target_indoor_temp,
            features=ctx.features_dict,
            thermal_model=ctx.wrapper.thermal_model,
            climate_mode=ctx.climate_mode,
        )

        # LGBM result
        _lgbm_result = None
        if (
            ctx.cooling_ml_model is not None
            and ctx.cooling_ml_model.is_loaded
        ):
            # Inject trajectory from OverheatingPredictor so that LGBM model
            # can use trajectory-derived features (traj_predicted_error, etc.)
            _lgbm_features = ctx.features_dict
            if _traj_result and _traj_result.get("trajectory"):
                _lgbm_features = dict(ctx.features_dict)
                _lgbm_features["_last_trajectory"] = {
                    "trajectory": _traj_result["trajectory"],
                    "max_predicted": _traj_result.get("peak_temp"),
                    "reaches_target_at": _traj_result.get("peak_hour"),
                }
            _lgbm_result = ctx.cooling_ml_model.predict_overheating_risk(
                current_indoor=ctx.prediction_indoor_temp,
                target_cooling=ctx.target_indoor_temp,
                features=_lgbm_features,
                climate_mode=ctx.climate_mode,
            )

        # Select active vs. shadow strategy
        if (
            ctx.cooling_ml_model_type == "lgbm_model"
            and _lgbm_result is not None
        ):
            ctx.pre_cool_result = _lgbm_result
            logging.debug(
                "❄️ SHADOW (trajectory): risk=%s peak=%.1f°C in %.1fh",
                _traj_result.get("risk"),
                _traj_result.get("peak_temp", 0),
                _traj_result.get("peak_hour", 0),
            )
        else:
            ctx.pre_cool_result = _traj_result
            if _lgbm_result is not None:
                logging.debug(
                    "❄️ SHADOW (lgbm): risk=%s p=%.3f should_cool=%s",
                    _lgbm_result.get("risk"),
                    _lgbm_result.get("lgbm_proba", 0.0),
                    _lgbm_result.get("should_cool_now"),
                )
                ctx.pre_cool_result = _apply_shadow_pre_cool_guard(
                    ctx, ctx.pre_cool_result, _lgbm_result
                )

        # Observation buffer: accumulate features + resolve labels
        if ctx.cooling_obs_buffer is not None:
            ctx.cooling_obs_buffer.push_pending(
                features=ctx.features_dict,
                indoor_temp=ctx.prediction_indoor_temp,
                cooling_target=ctx.target_indoor_temp,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            _newly_labeled = ctx.cooling_obs_buffer.resolve_labels(
                ctx.prediction_indoor_temp
            )
            if _newly_labeled > 0:
                logging.debug(
                    "Cooling obs buffer: resolved %d new labels this cycle",
                    _newly_labeled,
                )
            try:
                ctx.cooling_obs_buffer.save()
            except Exception as _save_err:
                logging.warning(
                    "Cooling obs buffer save failed (non-fatal): %s",
                    _save_err,
                )
            # Auto-retrain
            if ctx.cooling_obs_buffer.should_retrain():
                logging.info(
                    "🤖 Cooling ML: retrain trigger reached "
                    "(%d labeled) — starting retrain",
                    ctx.cooling_obs_buffer.n_labeled,
                )
                try:
                    from .cooling_ml_calibration import calibrate_cooling_ml

                    if calibrate_cooling_ml():
                        ctx.cooling_ml_model.load()
                        logging.info(
                            "✅ Cooling ML model retrained and reloaded"
                        )
                        ctx.cooling_obs_buffer.reset_retrain_counter()
                        ctx.cooling_obs_buffer.save()
                    else:
                        logging.warning(
                            "Cooling ML retrain returned False — "
                            "will retry after %d more new observations",
                            max(
                                1,
                                ctx.cooling_obs_buffer._retrain_trigger_k // 2
                                + 1,
                            ),
                        )
                        with ctx.cooling_obs_buffer._lock:
                            ctx.cooling_obs_buffer._labeled_since_last_train = max(
                                0,
                                ctx.cooling_obs_buffer._labeled_since_last_train
                                - ctx.cooling_obs_buffer._retrain_trigger_k
                                // 2
                                - 1,
                            )
                except Exception as _retrain_err:
                    logging.warning(
                        "Cooling ML retrain failed (non-fatal): %s",
                        _retrain_err,
                    )

        # Apply pre-cool target shift
        _offset = 0.0  # initialise; set below if should_cool_now
        if (
            ctx.pre_cool_result.get("should_cool_now")
            and ctx.prediction_indoor_temp <= ctx.target_indoor_temp
        ):
            # Proportional offset from regression (if available), else fixed
            _use_proportional = (
                getattr(config, "PRE_COOL_PROPORTIONAL", True)
                and ctx.pre_cool_result.get("predicted_delta", 0.0) > 0.0
            )
            if _use_proportional:
                _predicted_max = float(
                    ctx.pre_cool_result.get("predicted_max_temp", 0)
                )
                _overshoot = _predicted_max - ctx.target_indoor_temp
                _gain = float(
                    getattr(config, "PRE_COOL_OVERSHOOT_GAIN", 0.7)
                )
                _min_k = float(
                    getattr(config, "PRE_COOL_MIN_OFFSET_K", 0.2)
                )
                _max_k = float(
                    getattr(config, "PRE_COOL_MAX_OFFSET_K", 1.0)
                )
                _offset = max(_min_k, min(_max_k, _overshoot * _gain))
            else:
                _offset = float(
                    getattr(config, "PRE_COOL_TARGET_OFFSET_K", 0.5)
                )
            _original_target = ctx.target_indoor_temp
            _min_target = _resolve_pre_cool_min_target(
                ctx, _original_target, _offset
            )
            ctx.target_indoor_temp = _min_target
            ctx.pre_cool_active = True
            logging.info(
                "❄️ PRE-COOL [%s]: target shifted %.1f → %.1f°C "
                "(room %.1f°C, offset=%.2fK%s, predicted peak %.1f°C in %.1fh). "
                "Reason: %s",
                ctx.cooling_ml_model_type,
                _original_target,
                ctx.target_indoor_temp,
                ctx.prediction_indoor_temp,
                _offset,
                " [proportional]" if _use_proportional else " [fixed]",
                ctx.pre_cool_result.get("peak_temp", 0),
                ctx.pre_cool_result.get("peak_hour", 0),
                ctx.pre_cool_result.get("reason", ""),
            )

        # Persist pre-cooling state
        try:
            ctx.state_manager.update_operational_state(
                pre_cool_active=ctx.pre_cool_active,
                pre_cool_peak_temp=float(
                    ctx.pre_cool_result.get("peak_temp", 0)
                ),
                pre_cool_peak_hour=float(
                    ctx.pre_cool_result.get("peak_hour", 0)
                ),
                pre_cool_risk=bool(ctx.pre_cool_result.get("risk", False)),
                pre_cool_offset_k=float(_offset) if ctx.pre_cool_active else 0.0,
                pre_cool_predicted_max=float(
                    ctx.pre_cool_result.get("predicted_max_temp", 0)
                ) if ctx.pre_cool_active else 0.0,
                pre_cool_shadow_peak_temp=float(
                    _lgbm_result.get(
                        "predicted_max_temp",
                        _lgbm_result.get("peak_temp", 0),
                    )
                ) if _lgbm_result is not None else 0.0,
                pre_cool_shadow_lgbm_proba=float(
                    _lgbm_result.get("lgbm_proba", 0.0)
                ) if _lgbm_result is not None else 0.0,
            )
        except Exception:
            pass
    except Exception as _pre_cool_exc:
        logging.debug("Pre-cooling check failed: %s", _pre_cool_exc)


# ---------------------------------------------------------------------------
# Heating observation buffer step (heating-only)
# ---------------------------------------------------------------------------


def step_heating_obs_buffer(ctx: CycleContext) -> None:
    """Heating correction ML observation buffer (push + resolve + retrain)."""
    if ctx.heating_obs_buffer is None:
        return

    try:
        from .heating_correction_ml_calibration import (
            _compute_s_h,
            _read_baseline_thermal_params,
        )

        _hob_eta, _hob_u, _hob_tau = _read_baseline_thermal_params(config)
        _hob_label_h = float(
            getattr(config, "HEATING_ML_LABEL_HORIZON_H", 4)
        )
        _hob_s_h = _compute_s_h(_hob_eta, _hob_u, _hob_tau, _hob_label_h)

        # Push pending observation
        _hob_features = (
            ctx.features_dict if isinstance(ctx.features_dict, dict) else {}
        )
        ctx.heating_obs_buffer.push_pending(
            features=_hob_features,
            indoor_temp=ctx.prediction_indoor_temp,
            heating_target=ctx.target_indoor_temp,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Resolve labels
        _hob_newly_labeled = ctx.heating_obs_buffer.resolve_labels(
            ctx.prediction_indoor_temp, _hob_s_h
        )
        if _hob_newly_labeled > 0:
            logging.debug(
                "Heating obs buffer: resolved %d new label(s) this cycle",
                _hob_newly_labeled,
            )

        # Persist
        try:
            ctx.heating_obs_buffer.save()
        except Exception as _hob_save_err:
            logging.warning(
                "Heating obs buffer save failed (non-fatal): %s",
                _hob_save_err,
            )

        # Auto-retrain
        if ctx.heating_obs_buffer.should_retrain():
            logging.info(
                "🤖 Heating ML: retrain trigger reached "
                "(%d labeled) — starting retrain",
                ctx.heating_obs_buffer.n_labeled,
            )
            try:
                from .heating_correction_ml_calibration import (
                    calibrate_heating_correction_ml,
                )

                if calibrate_heating_correction_ml():
                    from .model_wrapper import EnhancedModelWrapper

                    EnhancedModelWrapper._heating_correction_ml_model = None
                    logging.info(
                        "✅ Heating correction ML model retrained and reloaded"
                    )
                    ctx.heating_obs_buffer.reset_retrain_counter()
                    ctx.heating_obs_buffer.save()
                else:
                    logging.warning(
                        "Heating ML retrain returned False — "
                        "will retry after %d more new observations",
                        max(
                            1,
                            ctx.heating_obs_buffer.retrain_trigger_k // 2 + 1,
                        ),
                    )
                    with ctx.heating_obs_buffer._lock:
                        ctx.heating_obs_buffer._labeled_since_last_train = max(
                            0,
                            ctx.heating_obs_buffer._labeled_since_last_train
                            - ctx.heating_obs_buffer.retrain_trigger_k // 2
                            - 1,
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


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


def run_blocking_route(ctx: CycleContext) -> None:
    """BLOCKING state: DHW / Defrost / Disinfection active.

    Actions:
    1. Write BLOCKED state to HA sensor
    2. Save blocking state for next cycle
    """
    logging.info("Blocking process active (DHW/Defrost), skipping.")
    try:
        heating_state_entity_id = get_shadow_output_entity_id(
            "sensor.ml_heating_state"
        )
        blocking_reasons = [
            e
            for e in ctx.blocking_entities
            if ctx.ha_client.get_state(e, ctx.all_states, is_binary=True)
        ]
        ctx.blocking_reasons = blocking_reasons
        attributes_state = get_sensor_attributes(heating_state_entity_id)
        attributes_state.update(
            {
                "state_description": "Blocking activity - Skipping",
                "blocking_reasons": blocking_reasons,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        )
        ctx.ha_client.set_state(
            heating_state_entity_id,
            2,
            attributes_state,
            round_digits=None,
        )
    except Exception:
        logging.debug("Failed to write BLOCKED state to HA.", exc_info=True)

    # Save blocking state for next cycle
    save_state(
        state_manager=ctx.state_manager,
        last_is_blocking=True,
        last_final_temp=ctx.state.get("last_final_temp"),
        last_blocking_reasons=ctx.blocking_reasons,
        last_blocking_end_time=None,
    )


def run_grace_period_route(ctx: CycleContext) -> None:
    """GRACE_PERIOD state: transition after blocking ends.

    Preserves the previous valid target temperature to avoid state poisoning.
    Marks this cycle as a grace period passthrough so the next cycle's
    online learning knows not to learn from a non-calculated target.
    Uses the heating state manager (same as IDLE).
    """
    logging.info("⏳ Grace period active - Passive learning mode")

    preserved_target = ctx.state.get("last_final_temp")
    if preserved_target is None:
        try:
            preserved_target = float(
                ctx.ha_client.get_state(
                    config.ACTUAL_OUTLET_TEMP_ENTITY_ID, ctx.all_states
                )
            )
        except (ValueError, TypeError):
            preserved_target = 20.0

    logging.info(
        "Preserving last_final_temp=%.1f°C during grace period", preserved_target
    )

    save_state(
        state_manager=ctx.state_manager,
        last_final_temp=preserved_target,
        last_is_blocking=False,
        last_blocking_end_time=ctx.state.get("last_blocking_end_time"),
        last_run_features={"learning_mode": "grace_period_passthrough"},
    )


def run_idle_route(ctx: CycleContext) -> None:
    """IDLE state: system not active.

    Full feature calculation and state saving so that:
    - Online learning (pre-dispatch) has valid last_run_features next cycle
    - The heating observation buffer still resolves labels
    - Thermal model keeps learning even when HP is off

    State is saved to the heating unified thermal state file.
    """
    if not step_get_sensor_data(ctx):
        return

    step_determine_prediction_indoor(ctx)

    if not step_build_features(ctx):
        return

    # Dynamic trajectory / price still calculated for feature completeness.
    # NOTE: step_dynamic_trajectory mutates config.TRAJECTORY_STEPS and
    # config.MIN_SETPOINT_HOLD_CYCLES and may trigger electricity-price
    # network calls. These side effects are intentional so that the next
    # HEATING/COOLING cycle begins with an up-to-date trajectory length.
    step_dynamic_trajectory(ctx)

    # Heating obs buffer: resolve labels even while idle so pending
    # observations get their labels assigned (the indoor temp is still
    # changing due to thermal decay).
    step_heating_obs_buffer(ctx)

    # Log metrics (thermodynamic sensors useful even in idle)
    step_log_metrics(ctx)

    # Save state so next cycle's online learning has valid features.
    # IDLE always uses the heating state manager.
    step_save_state(ctx)

    logging.debug(
        "IDLE: system inactive, features built and state saved for learning"
    )


def run_heating_route(ctx: CycleContext) -> None:
    """HEATING state: active heating mode.

    Steps executed in order:
    1. Get sensor data
    2. Determine prediction indoor temp (with transient drop filter)
    3. Build features
    4. Dynamic trajectory / price
    5. Heating obs buffer (push + resolve + retrain)
    6. Prediction
    7. Gradual control
    8. EMA smoothing
    9. Setpoint hold
    10. Update HA
    11. Log metrics
    12. Update ML state sensor
    13. Shadow comparison
    14. Save state
    15. Publish auxiliary sensors
    """
    if not step_get_sensor_data(ctx):
        return

    step_determine_prediction_indoor(ctx)

    if not step_build_features(ctx):
        return

    step_dynamic_trajectory(ctx)
    step_heating_obs_buffer(ctx)
    step_prediction(ctx)
    step_gradual_control(ctx)
    step_ema_smoothing(ctx)
    step_setpoint_hold(ctx)
    step_update_ha(ctx)
    step_log_metrics(ctx)
    step_update_ml_state_sensor(ctx)
    step_shadow_comparison(ctx)
    step_save_state(ctx)
    step_publish_auxiliary_sensors(ctx)


def run_cooling_route(ctx: CycleContext) -> None:
    """COOLING state: active cooling mode.

    Steps executed in order:
    1. Get sensor data
    2. Apply cooling target override
    3. Determine prediction indoor temp (no transient drop filter)
    4. Build features
    5. Dynamic trajectory / price
    6. Pre-cooling (overheating prediction + obs buffer)
    7. Prediction
    8. Gradual control
    9. EMA smoothing (bypassed during recovery)
    10. Setpoint hold
    11. Update HA
    12. Log metrics
    13. Update ML state sensor
    14. Shadow comparison
    15. Save state
    16. Publish auxiliary sensors
    """
    if not step_get_sensor_data(ctx):
        return

    step_apply_cooling_target(ctx)
    step_determine_prediction_indoor(ctx)

    if not step_build_features(ctx):
        return

    step_dynamic_trajectory(ctx)
    step_pre_cooling(ctx)
    step_prediction(ctx)
    step_gradual_control(ctx)
    step_ema_smoothing(ctx)
    step_setpoint_hold(ctx)
    step_update_ha(ctx)
    step_log_metrics(ctx)
    step_update_ml_state_sensor(ctx)
    step_shadow_comparison(ctx)
    step_save_state(ctx)
    step_publish_auxiliary_sensors(ctx)
