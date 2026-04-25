"""
Enhanced feature builder for RealisticPhysicsModel with thermal momentum
analysis.

This module builds comprehensive physics features including:
- Original 19 features for backward compatibility
- NEW: 15 thermal momentum and extended lag features for enhanced temperature
  stability
- Thermal gradient analysis for overshoot prevention
- Extended lag features (10m, 60m) for thermal mass understanding
- Cyclical time encoding for daily/seasonal patterns
- Outlet effectiveness analysis for heat transfer optimization

Total: 34 sophisticated thermal intelligence features for ±0.1°C control
precision.
"""
import logging
import math
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

# Support both package-relative and direct import
try:
    from . import config
    from .ha_client import HAClient
    from .influx_service import InfluxService
    from .sensor_buffer import SensorBuffer
except ImportError:
    import config  # type: ignore
    from ha_client import HAClient  # type: ignore
    from influx_service import InfluxService  # type: ignore
    from sensor_buffer import SensorBuffer  # type: ignore


def calculate_thermodynamic_metrics(
    outlet_temp: float,
    inlet_temp: Optional[float],
    flow_rate: Optional[float],
    power_consumption: Optional[float],
) -> dict:
    """
    Calculate thermodynamic metrics (COP, Power, Delta T).
    """
    # Process inputs
    inlet_temp_f = float(inlet_temp) if inlet_temp is not None else outlet_temp
    flow_rate_f = float(flow_rate) if flow_rate is not None else 0.0
    power_consumption_f = (
        float(power_consumption) if power_consumption is not None else 0.0
    )

    # Delta T
    delta_t = outlet_temp - inlet_temp_f

    # Thermal Power (kW)
    # Q = m * c * dT
    # Flow rate L/min -> kg/s (approx / 60)
    # Specific heat capacity kJ/kg*K
    thermal_power_kw = (
        (flow_rate_f / 60.0) * config.SPECIFIC_HEAT_CAPACITY * delta_t
    )

    # COP
    electrical_power_kw = power_consumption_f / 1000.0
    if electrical_power_kw > 0.1:
        cop_realtime = thermal_power_kw / electrical_power_kw
    else:
        cop_realtime = 0.0

    return {
        "inlet_temp": inlet_temp_f,
        "flow_rate": flow_rate_f,
        "power_consumption": power_consumption_f,
        "delta_t": delta_t,
        "thermal_power_kw": thermal_power_kw,
        "cop_realtime": cop_realtime,
    }


def build_physics_features(
    ha_client: HAClient,
    influx_service: InfluxService,
    sensor_buffer: Optional[SensorBuffer] = None,
) -> Tuple[Optional[pd.DataFrame], list[float]]:
    """
    Build enhanced features for RealisticPhysicsModel with thermal momentum.
    
    Creates 40 physics features including original 19 + 15 new thermal features
    + 6 thermodynamic:
    - Core temperatures: outlet, inlet, indoor_lag_30m, target, outdoor
    - System states: dhw_heating, dhw_disinfection, dhw_boost_heater,
      defrosting
    - External sources: pv_now, fireplace_on, tv_on
    - Forecasts: temp_forecast_1h-12h, pv_forecast_1h-12h
    - NEW: Thermal momentum, extended lag, delta analysis, cyclical time
    - THERMODYNAMIC: Delta T, Thermal Power, COP
    
    Args:
        ha_client: Home Assistant client
        influx_service: InfluxDB service
        sensor_buffer: Optional in-memory buffer for sensor smoothing
        
    Returns:
        DataFrame with single row containing all features, or None if missing
    """
    # Fetch current sensor values
    all_states = ha_client.get_all_states()

    actual_indoor = ha_client.get_state(
        config.INDOOR_TEMP_ENTITY_ID, all_states
    )
    # Fetch living room temp for fireplace analysis
    living_room_temp = ha_client.get_state(
        config.LIVING_ROOM_TEMP_ENTITY_ID, all_states
    )
    outdoor_temp = ha_client.get_state(
        config.OUTDOOR_TEMP_ENTITY_ID, all_states
    )
    outlet_temp = ha_client.get_state(
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID, all_states
    )
    target_indoor_temp = ha_client.get_state(
        config.TARGET_INDOOR_TEMP_ENTITY_ID, all_states
    )
    
    # Fetch thermodynamic sensors (New in Feature/Sensor-Update)
    inlet_temp = ha_client.get_state(config.INLET_TEMP_ENTITY_ID, all_states)
    flow_rate = ha_client.get_state(config.FLOW_RATE_ENTITY_ID, all_states)
    power_consumption = ha_client.get_state(
        config.POWER_CONSUMPTION_ENTITY_ID, all_states
    )

    # Apply sensor smoothing if buffer is available
    if sensor_buffer:
        # Indoor Temp (15 min smoothing)
        avg_indoor = sensor_buffer.get_average(
            config.INDOOR_TEMP_ENTITY_ID, 15
        )
        if avg_indoor is not None:
            actual_indoor = avg_indoor

        # Outdoor Temp (30 min smoothing)
        avg_outdoor = sensor_buffer.get_average(
            config.OUTDOOR_TEMP_ENTITY_ID, 30
        )
        if avg_outdoor is not None:
            outdoor_temp = avg_outdoor

        # Outlet Temp (5 min smoothing)
        avg_outlet = sensor_buffer.get_average(
            config.ACTUAL_OUTLET_TEMP_ENTITY_ID, 5
        )
        if avg_outlet is not None:
            outlet_temp = avg_outlet

        # Inlet Temp (5 min smoothing)
        avg_inlet = sensor_buffer.get_average(config.INLET_TEMP_ENTITY_ID, 5)
        if avg_inlet is not None:
            inlet_temp = avg_inlet

        # Flow Rate (5 min smoothing)
        avg_flow = sensor_buffer.get_average(config.FLOW_RATE_ENTITY_ID, 5)
        if avg_flow is not None:
            flow_rate = avg_flow

    # Check for missing critical data
    if None in [actual_indoor, outdoor_temp, outlet_temp, target_indoor_temp]:
        logging.error("Missing critical sensor data. Cannot build features.")
        return None, []
    
    # Get extended history for thermal momentum features
    # Need 18 steps for 180-minute solar lag buffer (18 * 10min = 180min)
    extended_steps = max(18, config.HISTORY_STEPS)
    outlet_history = influx_service.fetch_outlet_history(extended_steps)
    indoor_history = influx_service.fetch_indoor_history(extended_steps)
    pv_history = influx_service.fetch_pv_history(extended_steps)
    
    if len(indoor_history) < 6 or len(outlet_history) < 3:
        logging.error(
            "Insufficient history for enhanced features. "
            f"Need 6 indoor + 3 outlet, got {len(indoor_history)} "
            f"+ {len(outlet_history)}."
        )
        return None, []
    
    # System states (binary) - default to False if unavailable
    dhw_heating = ha_client.get_state(
        config.DHW_STATUS_ENTITY_ID, all_states, is_binary=True
    ) or False
    dhw_disinfection = ha_client.get_state(
        config.DISINFECTION_STATUS_ENTITY_ID, all_states, is_binary=True
    ) or False
    dhw_boost_heater = ha_client.get_state(
        config.DHW_BOOST_HEATER_STATUS_ENTITY_ID, all_states, is_binary=True
    ) or False
    defrosting = ha_client.get_state(
        config.DEFROST_STATUS_ENTITY_ID, all_states, is_binary=True
    ) or False
    
    # External heat sources
    # Sum all PV power sources
    pv_now = ha_client.get_state(config.PV_POWER_ENTITY_ID, all_states) or 0.0
    pv_now = float(pv_now)

    # Solar correction helper (input_number percentage from Home Assistant)
    solar_correction_enabled = getattr(config, "SOLAR_CORRECTION_ENABLED", True)
    solar_percent_default = getattr(
        config, "SOLAR_CORRECTION_DEFAULT_PERCENT", 100.0
    )
    solar_percent_min = getattr(config, "SOLAR_CORRECTION_MIN_PERCENT", 0.0)
    solar_percent_max = getattr(config, "SOLAR_CORRECTION_MAX_PERCENT", 100.0)
    solar_correction_percent = float(solar_percent_default)
    solar_correction_factor = 1.0

    if solar_correction_enabled:
        solar_percent_raw = ha_client.get_state(
            config.SOLAR_CORRECTION_ENTITY_ID, all_states
        )
        if solar_percent_raw is None:
            logging.warning(
                "Solar correction sensor unavailable (%s). Using default %.1f%%",
                config.SOLAR_CORRECTION_ENTITY_ID,
                solar_percent_default,
            )
        else:
            try:
                solar_correction_percent = float(solar_percent_raw)
            except (TypeError, ValueError):
                logging.warning(
                    "Invalid solar correction value '%s'. Using default %.1f%%",
                    solar_percent_raw,
                    solar_percent_default,
                )
                solar_correction_percent = float(solar_percent_default)

        solar_correction_percent = max(
            solar_percent_min,
            min(solar_percent_max, solar_correction_percent),
        )
        solar_correction_factor = solar_correction_percent / 100.0
    
    fireplace_on = ha_client.get_state(
        config.FIREPLACE_STATUS_ENTITY_ID, all_states, is_binary=True
    ) or False
    tv_on = ha_client.get_state(
        config.TV_STATUS_ENTITY_ID, all_states, is_binary=True
    ) or False
    
    # PV Forecasts with correct 'watts' attribute parsing (support up to TRAJECTORY_STEPS hours)
    _n_fc = config.TRAJECTORY_STEPS
    pv_forecasts = [0.0] * _n_fc
    if config.PV_FORECAST_ENTITY_ID:
        try:
            from datetime import timezone, timedelta
            pv_state = all_states.get(config.PV_FORECAST_ENTITY_ID)
            pv_forecast_data = (
                pv_state.get("attributes") if pv_state else None
            )
            # CORRECT: Look for 'watts' attribute, not 'forecast'
            if pv_forecast_data and "watts" in pv_forecast_data:
                now = datetime.now(timezone.utc)
                watts_data = pv_forecast_data["watts"]
                # Convert timestamp strings to pandas timestamps
                forecast_dict = {}
                for ts_str, watts in watts_data.items():
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        forecast_dict[pd.Timestamp(ts)] = float(watts)
                    except Exception as e_parse:
                        logging.debug(
                            f"Could not parse timestamp {ts_str}: {e_parse}"
                        )
                        continue
                if forecast_dict:
                    s = pd.Series(forecast_dict, dtype=float).sort_index()
                    # Calculate hourly averages for next TRAJECTORY_STEPS hours
                    hourly = []
                    for hour in range(1, _n_fc + 1):
                        hour_start = now + timedelta(hours=hour)
                        hour_end = hour_start + timedelta(hours=1)
                        hour_entries = []
                        for ts, watts in s.items():
                            ts_ts = pd.Timestamp(ts)  # type: ignore
                            ts_utc = (
                                ts_ts.tz_convert('UTC')
                                if ts_ts.tz
                                else ts_ts.tz_localize('UTC')
                            )
                            if hour_start <= ts_utc < hour_end:
                                hour_entries.append(watts)
                        if hour_entries:
                            avg_watts = sum(hour_entries) / len(hour_entries)
                            hourly.append(round(avg_watts, 1))
                        else:
                            hourly.append(0.0)
                    # Pad to _n_fc if less
                    while len(hourly) < _n_fc:
                        hourly.append(hourly[-1] if hourly else 0.0)
                    pv_forecasts = hourly[:_n_fc]
                    logging.debug(
                        f"PV forecast parsed successfully: {pv_forecasts}W"
                    )
                else:
                    logging.debug("No valid PV forecast timestamps parsed")
            else:
                keys = (
                    list(pv_forecast_data.keys())
                    if pv_forecast_data
                    else 'no attributes'
                )
                logging.debug(
                    f"PV forecast entity missing 'watts' attribute: {keys}"
                )
        except Exception as e:
            logging.debug(f"Could not fetch PV forecast: {e}")
            pv_forecasts = [0.0] * _n_fc

    def _scale_pv_value(value: object) -> float:
        try:
            return float(value) * solar_correction_factor
        except (TypeError, ValueError):
            return 0.0

    # Apply global solar correction consistently to all PV inputs.
    pv_now *= solar_correction_factor
    pv_forecasts = [_scale_pv_value(p) for p in pv_forecasts]
    pv_history = [_scale_pv_value(p) for p in pv_history]

    logging.debug(
        "Solar correction: enabled=%s, percent=%.1f%%, factor=%.3f",
        solar_correction_enabled,
        solar_correction_percent,
        solar_correction_factor,
    )
    
    # Convert to float for calculations first
    actual_indoor_f = (
        float(actual_indoor) if actual_indoor is not None else 20.0
    )
    outdoor_temp_f = (
        float(outdoor_temp) if outdoor_temp is not None else 0.0
    )
    outlet_temp_f = (
        float(outlet_temp) if outlet_temp is not None else 30.0
    )
    target_temp_f = (
        float(target_indoor_temp) if target_indoor_temp is not None else 20.0
    )

    # Process thermodynamic sensors
    thermo_metrics = calculate_thermodynamic_metrics(
        outlet_temp_f,
        inlet_temp,
        flow_rate,
        power_consumption
    )

    inlet_temp_f = thermo_metrics["inlet_temp"]
    flow_rate_f = thermo_metrics["flow_rate"]
    power_consumption_f = thermo_metrics["power_consumption"]
    delta_t = thermo_metrics["delta_t"]
    thermal_power_kw = thermo_metrics["thermal_power_kw"]
    cop_realtime = thermo_metrics["cop_realtime"]

    # Get calibrated temperature forecasts using delta correction (support up to TRAJECTORY_STEPS hours)
    try:
        # Check if delta calibration is enabled and available
        if (
            hasattr(config, 'ENABLE_DELTA_FORECAST_CALIBRATION') and
            config.ENABLE_DELTA_FORECAST_CALIBRATION and
            hasattr(ha_client, 'get_calibrated_hourly_forecast')
        ):
            temp_forecasts = ha_client.get_calibrated_hourly_forecast(
                current_outdoor_temp=outdoor_temp_f,
                enable_delta_calibration=True
            )
        else:
            temp_forecasts = ha_client.get_hourly_forecast()
        # Ensure we have a valid list of forecasts
        if not isinstance(temp_forecasts, list) or len(temp_forecasts) < _n_fc:
            # Fallback to default values if forecasts are invalid
            temp_forecasts = [outdoor_temp_f] * _n_fc
        # Pad to _n_fc if less
        while len(temp_forecasts) < _n_fc:
            temp_forecasts.append(temp_forecasts[-1] if temp_forecasts else outdoor_temp_f)
    except Exception as e:
        logging.debug(f"Could not fetch temperature forecasts: {e}")
        temp_forecasts = [outdoor_temp_f] * _n_fc
    
    # Get cloud cover forecasts (only when cloud correction is enabled)
    cloud_cover_forecasts = [0.0] * _n_fc  # Default: clear sky (no correction)
    if getattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False):
        try:
            if hasattr(ha_client, 'get_hourly_cloud_cover'):
                cloud_cover_forecasts = ha_client.get_hourly_cloud_cover()
                # Ensure we have a valid list
                if not isinstance(cloud_cover_forecasts, list) or len(cloud_cover_forecasts) < _n_fc:
                    cloud_cover_forecasts = [0.0] * _n_fc
                # Pad to _n_fc if less
                while len(cloud_cover_forecasts) < _n_fc:
                    cloud_cover_forecasts.append(cloud_cover_forecasts[-1] if cloud_cover_forecasts else 0.0)
        except Exception as e:
            logging.debug(f"Could not fetch cloud cover forecasts: {e}")
            cloud_cover_forecasts = [0.0] * _n_fc

        avg_cc = sum(cloud_cover_forecasts) / len(cloud_cover_forecasts)
        _cc_labels = " ".join(
            f"{h}h={cloud_cover_forecasts[h - 1]:.0f}%" for h in range(1, _n_fc + 1)
        )
        logging.debug(
            "☁️ Cloud cover features: %s (avg=%.1f%%)",
            _cc_labels,
            avg_cc,
        )

    # Get current time for cyclical encoding
    now = datetime.now()
    current_hour = now.hour
    current_month = now.month
    
    # Calculate time period for gradient (in hours)
    time_period = config.HISTORY_STEP_MINUTES / 60.0

    # FIX: Handle missing history (defaults) to prevent artificial gradients
    # If history is default (21.0) but actual is different, clamp to actual
    # to avoid a fake massive temperature drop skewing gradient/lag features.
    hist_start = float(indoor_history[0])
    if hist_start == 21.0 and abs(actual_indoor_f - 21.0) > 0.5:
        hist_start = actual_indoor_f

    hist_10m = float(indoor_history[-1])
    if hist_10m == 21.0 and abs(actual_indoor_f - 21.0) > 0.5:
        hist_10m = actual_indoor_f

    hist_60m = float(indoor_history[-6])
    if hist_60m == 21.0 and abs(actual_indoor_f - 21.0) > 0.5:
        hist_60m = actual_indoor_f

    hist_30m = float(indoor_history[-3])
    if hist_30m == 21.0 and abs(actual_indoor_f - 21.0) > 0.5:
        hist_30m = actual_indoor_f

    # Build enhanced feature dictionary with thermal momentum features
    features = {
        # Living room temp for fireplace analysis
        'living_room_temp': float(living_room_temp) if living_room_temp is not None else None,
        # === ORIGINAL 19 FEATURES (for backward compatibility) ===
        # Core temperatures
        'outlet_temp': outlet_temp_f,
        'indoor_temp_lag_30m': hist_30m,  # 30 min ago
        'target_temp': target_temp_f,
        'outdoor_temp': outdoor_temp_f,
        # System states
        'dhw_heating': float(dhw_heating),
        'dhw_disinfection': float(dhw_disinfection),
        'dhw_boost_heater': float(dhw_boost_heater),
        'defrosting': float(defrosting),
        # External heat sources
        'pv_now': float(pv_now),
        'solar_correction_percent': float(solar_correction_percent),
        'solar_correction_factor': float(solar_correction_factor),
        'solar_correction_enabled': float(solar_correction_enabled),
        'fireplace_on': float(fireplace_on),
        'tv_on': float(tv_on),
        # Weather forecasts (1-TRAJECTORY_STEPS hours)
        **{f'temp_forecast_{h}h': float(temp_forecasts[h - 1]) for h in range(1, _n_fc + 1)},
        # PV forecasts (1-TRAJECTORY_STEPS hours)
        **{f'pv_forecast_{h}h': float(pv_forecasts[h - 1]) for h in range(1, _n_fc + 1)},
        # Cloud cover forecasts (1-TRAJECTORY_STEPS hours, 0-100%)
        **{f'cloud_cover_forecast_{h}h': float(cloud_cover_forecasts[h - 1]) for h in range(1, _n_fc + 1)},
        # P0 Priority: Thermal momentum analysis (3 features)
        'temp_diff_indoor_outdoor': actual_indoor_f - outdoor_temp_f,
        'indoor_temp_gradient': ((actual_indoor_f - hist_start) / time_period),
        'outlet_indoor_diff': outlet_temp_f - actual_indoor_f,
        # P0 Priority: Extended lag features (4 features)
        'indoor_temp_lag_10m': hist_10m,  # 10 min ago
        'indoor_temp_lag_60m': hist_60m,  # 60 min ago
        'outlet_temp_lag_30m': float(outlet_history[-3]),  # 30 min ago
        'outlet_temp_change': outlet_temp_f - float(outlet_history[-1]),
        # P1 Priority: Delta analysis (3 features)
        'indoor_temp_delta_10m': actual_indoor_f - hist_10m,
        'indoor_temp_delta_30m': actual_indoor_f - hist_30m,
        'indoor_temp_delta_60m': actual_indoor_f - hist_60m,
        # P1 Priority: Cyclical time encoding (4 features)
        'hour_sin': math.sin(2 * math.pi * current_hour / 24),
        'hour_cos': math.cos(2 * math.pi * current_hour / 24),
        'month_sin': math.sin(2 * math.pi * (current_month - 1) / 12),
        'month_cos': math.cos(2 * math.pi * (current_month - 1) / 12),
        # P2 Priority: Outlet effectiveness analysis (1 feature)
        'outlet_effectiveness_ratio': ((actual_indoor_f - target_temp_f) / max(0.1, outlet_temp_f - actual_indoor_f)),
        # === THERMODYNAMIC FEATURES (Feature/Sensor-Update) ===
        'inlet_temp': inlet_temp_f,
        'flow_rate': flow_rate_f,
        'power_consumption': power_consumption_f,
        'delta_t': delta_t,
        'thermal_power_kw': thermal_power_kw,
        'cop_realtime': cop_realtime,
        # === WEEK 4 ENHANCED FORECAST FEATURES ===
        # Enhanced forecast analysis (3 new features)
        # °C/hour trend (use up to TRAJECTORY_STEPS hours)
        'temp_trend_forecast': ((temp_forecasts[_n_fc - 1] - outdoor_temp_f) / _n_fc),
        # Simple heating demand (use TRAJECTORY_STEPS hours)
        'heating_demand_forecast': max(0.0, (21.0 - temp_forecasts[_n_fc - 1]) * 0.1),
        # Net thermal load (use TRAJECTORY_STEPS hours)
        'combined_forecast_thermal_load': (
            max(0.0, (21.0 - temp_forecasts[_n_fc - 1]) * 0.1)
            - (pv_forecasts[_n_fc - 1] * 0.001)
        ),
        # PV power history for solar lag computation
        'pv_power_history': pv_history,
    }
    return pd.DataFrame([features]), outlet_history
