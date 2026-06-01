"""
Physics-Direct Calibration for the Cooling Thermal Equilibrium Model.

Adapts the heating physics calibration to estimate thermal parameters from
warm-season data for the **cooling** mode.  Parameters are persisted to the
cooling thermal state (``unified_thermal_state_cooling.json``) so the cooling
cycle route can use a calibrated equilibrium model.

Key differences from the heating calibration:
* **Data filter**: Only rows where the 24-hour rolling mean outdoor
  temperature exceeds ``COOLING_PHYSICS_MIN_OUTDOOR_ROLLING_24H_C`` (default
  16 °C) are used.
* **Stable period detection**: Includes both active cooling (outlet < indoor)
  and passive cooling periods (HP off, slab still below room temp).
* **OE quality gate**: Inverted — requires ``T_indoor − T_outlet > 0`` (HP
  is actively cooling or slab absorbs heat from room).
* **Fireplace/TV weights**: Not calibrated — defaults are used (these
  appliances are irrelevant in summer/cooling season).
* **Slab time constant**: Heating-calibrated value is used as prior/fallback;
  only overridden if cooling data yields a high-confidence result.
* **HLC**: Re-calibrated from warm-season data; falls back to the heating
  value if insufficient data.

Called via:
    ``python -m src.main --calibrate-cooling-physics``
or triggered by the dashboard flag file.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from . import config
    from .thermal_equilibrium_model import ThermalEquilibriumModel
    from .thermal_config import ThermalParameterConfig
    from .unified_thermal_state import get_thermal_state_manager
    from .unified_thermal_state_cooling import get_cooling_state_manager
    from .physics_calibration import (
        fetch_historical_data_for_calibration,
        backup_existing_calibration,
        calculate_cooling_time_constant,
        calibrate_delta_t_floor,
        calibrate_solar_decay_tau,
        filter_pv_decay_periods,
        filter_transient_periods,
        calibrate_transient_parameters,
    )
    from .physics_calibration_direct import (
        _calibrate_solar_lag_xcorr,
        _calibrate_slab_tau_grid_search,
        _residual_heat_source_weight,
    )
except ImportError:
    import config  # type: ignore
    from thermal_equilibrium_model import ThermalEquilibriumModel  # type: ignore
    from thermal_config import ThermalParameterConfig  # type: ignore
    from unified_thermal_state import get_thermal_state_manager  # type: ignore
    from unified_thermal_state_cooling import get_cooling_state_manager  # type: ignore
    from physics_calibration import (  # type: ignore
        fetch_historical_data_for_calibration,
        backup_existing_calibration,
        calculate_cooling_time_constant,
        calibrate_delta_t_floor,
        calibrate_solar_decay_tau,
        filter_pv_decay_periods,
        filter_transient_periods,
        calibrate_transient_parameters,
    )
    from physics_calibration_direct import (  # type: ignore
        _calibrate_solar_lag_xcorr,
        _calibrate_slab_tau_grid_search,
        _residual_heat_source_weight,
    )

try:
    from scipy.optimize import minimize_scalar
except ImportError:
    minimize_scalar = None


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Samples per hour in the training data (5-min intervals)
_SAMPLES_PER_HOUR: int = 12

# 24h rolling window size (in 5-min samples)
_ROLLING_24H_SAMPLES: int = 24 * _SAMPLES_PER_HOUR  # 288

# Minimum PV electrical power [W] to treat a period as "PV active"
_MIN_PV_POWER_W: float = 100.0

# Minimum stable periods required for calibration
_MIN_STABLE_PERIODS: int = 20

# Minimum R² for slab-tau to override the heating fallback
_SLAB_TAU_MIN_R2: float = 0.7


# ---------------------------------------------------------------------------
# Data filtering for cooling
# ---------------------------------------------------------------------------

def _apply_outdoor_rolling_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to rows where 24h rolling mean outdoor > threshold.

    The threshold is read from ``config.COOLING_PHYSICS_MIN_OUTDOOR_ROLLING_24H_C``
    (default 16 °C).

    Parameters
    ----------
    df : pd.DataFrame
        Raw historical DataFrame with outdoor temperature column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame (may be empty).
    """
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    if outdoor_col not in df.columns:
        logging.error("Outdoor temp column '%s' missing — cannot filter", outdoor_col)
        return pd.DataFrame()

    threshold = float(getattr(
        config, "COOLING_PHYSICS_MIN_OUTDOOR_ROLLING_24H_C", 16.0
    ))

    # Compute 24h rolling mean
    outdoor_numeric = pd.to_numeric(df[outdoor_col], errors="coerce")
    rolling_mean = outdoor_numeric.rolling(
        window=_ROLLING_24H_SAMPLES, min_periods=_ROLLING_24H_SAMPLES // 2
    ).mean()

    mask = rolling_mean > threshold
    n_before = len(df)
    df_filtered = df[mask].copy()
    n_after = len(df_filtered)

    logging.info(
        "24h rolling outdoor filter (>%.1f°C): %d → %d rows (%.1f%% retained)",
        threshold, n_before, n_after,
        100.0 * n_after / max(n_before, 1),
    )
    return df_filtered


# ---------------------------------------------------------------------------
# Cooling-adapted stable period detection
# ---------------------------------------------------------------------------

def filter_stable_periods_cooling(df: pd.DataFrame) -> list:
    """Filter for stable cooling periods (active + passive).

    Unlike the heating version, this includes:
    * **Active cooling**: HP running, outlet < indoor (HP cools slab)
    * **Passive cooling**: HP off, but slab temp still below room temp
      (slab absorbs heat from room)

    Quality gates:
    * Indoor temperature stable (range < threshold in window)
    * No blocking states (DHW, defrost, etc.)
    * Outlet temperature stable (std < 2 °C)
    """
    logging.info("=== FILTERING FOR STABLE COOLING PERIODS ===")

    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    inlet_col = config.INLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    flow_rate_col = config.FLOW_RATE_ENTITY_ID.split(".", 1)[-1]

    temp_change_threshold = getattr(
        config, "STABILITY_TEMP_CHANGE_THRESHOLD", 0.2
    )
    min_duration = getattr(config, "MIN_STABLE_PERIOD_MINUTES", 20)
    window_size = int(min_duration) // 5

    stable_periods = []
    filter_stats = {
        'total_checked': 0, 'missing_data': 0, 'temp_unstable': 0,
        'outlet_unstable': 0, 'passed': 0,
    }

    for i in range(window_size, len(df) - window_size):
        filter_stats['total_checked'] += 1

        window_start = i - window_size // 2
        window_end = i + window_size // 2
        window = df.iloc[window_start:window_end]

        # Indoor temperature stability
        indoor_temps = pd.to_numeric(
            window[indoor_col], errors="coerce"
        ).dropna() if indoor_col in window.columns else pd.Series(dtype=float)

        if len(indoor_temps) < window_size * 0.8:
            filter_stats['missing_data'] += 1
            continue

        temp_range = indoor_temps.max() - indoor_temps.min()
        if temp_range > temp_change_threshold:
            filter_stats['temp_unstable'] += 1
            continue

        # Outlet stability
        if outlet_col in window.columns:
            outlet_temps = pd.to_numeric(
                window[outlet_col], errors="coerce"
            ).dropna()
            if len(outlet_temps) >= window_size * 0.8 and outlet_temps.std() > 2.0:
                filter_stats['outlet_unstable'] += 1
                continue

        # Extract center row values
        center_row = df.iloc[i]

        indoor_temp = _safe_float(center_row.get(indoor_col))
        outdoor_temp = _safe_float(center_row.get(outdoor_col))
        outlet_temp = _safe_float(center_row.get(outlet_col))
        inlet_temp = _safe_float(center_row.get(inlet_col))
        pv_power = _safe_float(center_row.get(pv_col), default=0.0)
        flow_rate = _safe_float(center_row.get(flow_rate_col), default=0.0)

        if indoor_temp is None or outdoor_temp is None or outlet_temp is None:
            filter_stats['missing_data'] += 1
            continue

        # For cooling: outlet should be at or below indoor (active or passive)
        # We include passive periods (HP off) unlike heating which requires
        # HP to be actively running
        # Accept both: outlet < indoor (cooling) AND small positive diff (passive)
        # Reject only: outlet >> indoor (heating mode data leaked through)
        if outlet_temp > indoor_temp + 1.0:
            # This looks like heating, not cooling — skip
            continue

        # Compute thermal power
        thermal_power_kw = 0.0
        if inlet_temp is not None and flow_rate is not None and flow_rate > 0:
            delta_t = outlet_temp - inlet_temp
            thermal_power_kw = (flow_rate / 60.0) * config.SPECIFIC_HEAT_CAPACITY * delta_t

        # Effective temperature (BT2+BT3)/2 when inlet available
        effective_temp = outlet_temp
        if inlet_temp is not None:
            effective_temp = (outlet_temp + inlet_temp) / 2.0

        period = {
            'indoor_temp': indoor_temp,
            'outdoor_temp': outdoor_temp,
            'outlet_temp': outlet_temp,
            'effective_temp': effective_temp,
            'inlet_temp': inlet_temp,
            'pv_power': pv_power,
            'flow_rate': flow_rate,
            'thermal_power_kw': thermal_power_kw,
            'fireplace_on': 0,  # ignored for cooling
            'tv_on': 0,  # ignored for cooling
        }

        stable_periods.append(period)
        filter_stats['passed'] += 1

    logging.info(
        "Cooling stable period filter: %d checked, %d passed "
        "(missing=%d, temp_unstable=%d, outlet_unstable=%d)",
        filter_stats['total_checked'],
        filter_stats['passed'],
        filter_stats['missing_data'],
        filter_stats['temp_unstable'],
        filter_stats['outlet_unstable'],
    )
    return stable_periods


def _safe_float(val, default=None) -> Optional[float]:
    """Safely convert a value to float, returning default on failure."""
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Cooling-adapted OE estimation
# ---------------------------------------------------------------------------

def _filter_cooling_active_periods(stable_periods: list) -> list:
    """Return periods where cooling is actively happening (outlet < indoor).

    Accepts both active HP cooling and passive slab absorption.
    """
    return [
        p for p in stable_periods
        if p.get('indoor_temp', 20) > p.get('outlet_temp', 25)
        and p.get('pv_power', 0) < _MIN_PV_POWER_W
    ]


def _calibrate_oe_cooling(
    stable_periods: list, hlc: float
) -> Optional[float]:
    """Estimate outlet_effectiveness from cooling periods.

    In cooling mode the equilibrium equation is the same:
        T_eq = (OE × T_outlet + HLC × T_outdoor) / (OE + HLC)

    But the drive is inverted: T_indoor > T_outlet (heat flows from room
    into cooler slab/water).

    OE = HLC × (T_outdoor − T_indoor) / (T_outlet − T_indoor)

    Since T_outdoor > T_indoor in cooling season and T_outlet < T_indoor:
    * numerator (T_outdoor − T_indoor) > 0
    * denominator (T_outlet − T_indoor) < 0
    → raw OE would be negative.

    Rearranging correctly for cooling equilibrium where room temp is
    maintained by balance of outdoor heat gain and slab cooling:
        T_indoor = (OE × T_outlet + HLC × T_outdoor) / (OE + HLC)

    Solving for OE:
        T_indoor × (OE + HLC) = OE × T_outlet + HLC × T_outdoor
        OE × T_indoor + HLC × T_indoor = OE × T_outlet + HLC × T_outdoor
        OE × (T_indoor − T_outlet) = HLC × (T_outdoor − T_indoor)
        OE = HLC × (T_outdoor − T_indoor) / (T_indoor − T_outlet)

    Both numerator and denominator are positive in cooling mode.
    """
    logging.info("=== COOLING OE ESTIMATION ===")

    if hlc <= 0:
        logging.warning("⚠️ OE calibration skipped — HLC ≤ 0")
        return None

    cooling_periods = _filter_cooling_active_periods(stable_periods)
    if not cooling_periods:
        logging.warning("⚠️ No cooling-active periods for OE calibration")
        return None

    oe_values = []
    weights = []

    for p in cooling_periods:
        t_in = p.get("indoor_temp")
        t_out = p.get("outdoor_temp")
        t_outlet = p.get("effective_temp", p.get("outlet_temp"))

        if t_in is None or t_out is None or t_outlet is None:
            continue

        # Cooling drive: room is warmer than outlet
        drive = t_in - t_outlet
        if drive <= 0.2:
            continue  # Insufficient cooling drive

        # Outdoor heat gain: outdoor warmer than indoor
        delta_outdoor = t_out - t_in
        if delta_outdoor <= 0:
            continue  # No outdoor heat gain — unusual for cooling season

        oe = hlc * delta_outdoor / drive

        # Validate against cooling bounds
        bounds = ThermalParameterConfig.get_cooling_bounds("outlet_effectiveness")
        if bounds[0] <= oe <= bounds[1]:
            oe_values.append(oe)
            weights.append(drive)

    if len(oe_values) < 10:
        logging.warning(
            "⚠️ Insufficient cooling periods for OE: %d (need ≥10)",
            len(oe_values),
        )
        return None

    # Weighted median
    sorted_pairs = sorted(zip(oe_values, weights), key=lambda x: x[0])
    sorted_oe = [x[0] for x in sorted_pairs]
    sorted_w = [x[1] for x in sorted_pairs]
    cum_w = np.cumsum(sorted_w)
    total_w = cum_w[-1]
    median_idx = np.searchsorted(cum_w, total_w / 2.0)
    median_idx = min(median_idx, len(sorted_oe) - 1)
    oe_estimate = sorted_oe[median_idx]

    logging.info(
        "✅ Cooling OE estimate: %.4f kW/K (from %d periods, weighted median)",
        oe_estimate, len(oe_values),
    )
    return oe_estimate


# ---------------------------------------------------------------------------
# Cooling-adapted HLC estimation
# ---------------------------------------------------------------------------

def _calibrate_hlc_cooling(stable_periods: list) -> Optional[float]:
    """Estimate HLC from cooling-season data.

    In cooling mode HLC represents heat gain from outdoor air into the
    building.  We use HP-off periods where the building is warming up
    passively from outdoor heat:

        Q_gain = HLC × (T_outdoor − T_indoor)

    Or from active-cooling periods where slab cooling balances outdoor
    heat gain at equilibrium:

        OE × (T_indoor − T_outlet) ≈ HLC × (T_outdoor − T_indoor)
        → HLC ≈ OE × (T_indoor − T_outlet) / (T_outdoor − T_indoor)

    Since we don't know OE yet at this stage, we use a simpler approach:
    forced-through-origin OLS regression of thermal load vs temperature
    difference, similar to the heating HLC calibration but with inverted
    sign conventions.
    """
    logging.info("=== COOLING HLC ESTIMATION (warm-season data) ===")

    # Use periods where flow rate > 0 (HP running in cooling mode)
    cooling_active = [
        p for p in stable_periods
        if p.get('thermal_power_kw', 0) < config.COOLING_MIN_THERMAL_POWER_KW
        and p.get('indoor_temp', 20) > p.get('outlet_temp', 25)
        and p.get('outdoor_temp', 20) > p.get('indoor_temp', 20)
    ]

    if len(cooling_active) < 15:
        logging.warning(
            "⚠️ Insufficient cooling-active periods for HLC: %d (need ≥15)",
            len(cooling_active),
        )
        return None

    # At cooling equilibrium:
    #   cooling_power (W→K) ≈ HLC × ΔT_outdoor_indoor
    # thermal_power_kw is negative in cooling; the slab absorbs heat from room
    # |thermal_power_kw| is the cooling rate
    # At steady state: cooling_rate = heat_gain_rate
    # |Q_cooling| = HLC × (T_outdoor - T_indoor)

    x_vals = []  # ΔT = T_outdoor - T_indoor
    y_vals = []  # |thermal_power_kw| (cooling rate)

    for p in cooling_active:
        dt = p['outdoor_temp'] - p['indoor_temp']
        if dt <= 0.5:
            continue  # Need meaningful temperature difference
        q_cool = abs(p['thermal_power_kw'])
        x_vals.append(dt)
        y_vals.append(q_cool)

    if len(x_vals) < 10:
        logging.warning("⚠️ Insufficient valid HLC samples: %d", len(x_vals))
        return None

    x = np.array(x_vals)
    y = np.array(y_vals)

    # Forced-through-origin OLS: HLC = Σ(x*y) / Σ(x²)
    hlc = float(np.sum(x * y) / np.sum(x * x))

    # Validate against cooling bounds
    bounds = ThermalParameterConfig.get_cooling_bounds("heat_loss_coefficient")
    if not (bounds[0] <= hlc <= bounds[1]):
        logging.warning(
            "⚠️ Cooling HLC %.5f outside bounds [%.4f, %.4f] — clamping",
            hlc, bounds[0], bounds[1],
        )
        hlc = max(bounds[0], min(hlc, bounds[1]))

    # Compute R²
    y_pred = hlc * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

    logging.info(
        "✅ Cooling HLC = %.5f kW/K (R²=%.3f, n=%d)",
        hlc, r2, len(x_vals),
    )
    return hlc


# ---------------------------------------------------------------------------
# Cooling PV weight estimation
# ---------------------------------------------------------------------------

def _filter_cooling_pv_periods(stable_periods: list) -> list:
    """Return periods with PV active during cooling season."""
    return [
        p for p in stable_periods
        if p.get('pv_power', 0) > _MIN_PV_POWER_W
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def calibrate_cooling_physics(
    state_manager=None,
) -> Optional[ThermalEquilibriumModel]:
    """Calibrate thermal parameters for cooling mode using physics-direct methods.

    Fetches warm-season historical data, filters to rows with 24h rolling
    mean outdoor temp > threshold, and estimates all thermal parameters
    using cooling-adapted physics.  Results are persisted to the cooling
    thermal state.

    Parameters
    ----------
    state_manager:
        Optional CoolingThermalStateManager.  Defaults to the cooling
        singleton when *None*.

    Returns
    -------
    ThermalEquilibriumModel or None
        Configured model on success, *None* on failure.
    """
    logging.info(
        "=== COOLING THERMAL EQUILIBRIUM MODEL CALIBRATION (PHYSICS-DIRECT) ==="
    )

    # Resolve cooling state manager
    cooling_sm = state_manager if state_manager is not None else get_cooling_state_manager()

    # Load heating state for fallback values
    heating_sm = get_thermal_state_manager()
    heating_params: Dict[str, float] = heating_sm.state.get("baseline_parameters", {})

    def _heating_fallback(key: str) -> float:
        """Return heating-calibrated value if available, else cooling default."""
        val = heating_params.get(key)
        try:
            fval = float(val)  # type: ignore[arg-type]
            if not np.isnan(fval):
                bounds = ThermalParameterConfig.get_cooling_bounds(key)
                if bounds[0] <= fval <= bounds[1]:
                    return fval
        except (TypeError, ValueError, KeyError):
            pass
        return ThermalParameterConfig.get_cooling_default(key)

    def _cooling_fallback(key: str) -> float:
        """Return cooling state value > heating fallback > config default."""
        # First try existing cooling calibration
        cooling_baseline = cooling_sm.state.get("baseline_parameters", {})
        val = cooling_baseline.get(key)
        try:
            fval = float(val)  # type: ignore[arg-type]
            if not np.isnan(fval) and cooling_baseline.get("source") == "calibrated":
                return fval
        except (TypeError, ValueError):
            pass
        return _heating_fallback(key)

    # --- Fetch historical data ---
    logging.info("Fetching historical data for cooling calibration...")
    df = fetch_historical_data_for_calibration(
        lookback_hours=getattr(config, "TRAINING_LOOKBACK_HOURS", 168) * 2,
        purpose="cooling",
    )
    if df is None or df.empty:
        logging.error("❌ Failed to fetch historical data")
        return None

    logging.info("✅ Retrieved %d samples (%.1f hours)", len(df), len(df) / _SAMPLES_PER_HOUR)

    # --- Apply 24h rolling outdoor temperature filter ---
    df = _apply_outdoor_rolling_filter(df)
    if df.empty or len(df) < 100:
        logging.error(
            "❌ Insufficient warm-season data after outdoor temp filter: %d rows",
            len(df),
        )
        return None

    # --- Filter stable cooling periods ---
    stable_periods = filter_stable_periods_cooling(df)
    if len(stable_periods) < _MIN_STABLE_PERIODS:
        logging.error(
            "❌ Insufficient stable cooling periods: %d (need at least %d)",
            len(stable_periods), _MIN_STABLE_PERIODS,
        )
        return None
    logging.info("✅ Found %d stable cooling periods", len(stable_periods))

    # --- Step 1: HLC (re-calibrate from warm-season data) ---
    logging.info("Step 1: Cooling HLC estimation...")
    hlc = _calibrate_hlc_cooling(stable_periods)
    if hlc is None:
        logging.warning(
            "⚠️ Cooling HLC calibration failed — using heating fallback"
        )
        hlc = _heating_fallback("heat_loss_coefficient")

    # --- Step 2: OE ---
    logging.info("Step 2: Cooling OE estimation...")
    oe = _calibrate_oe_cooling(stable_periods, hlc)
    if oe is None:
        logging.warning("⚠️ Cooling OE calibration failed — using fallback")
        oe = _cooling_fallback("outlet_effectiveness")

    # --- Step 3: Thermal time constant ---
    # Use heating value as prior; only override if cooling data is good
    logging.info("Step 3: Thermal time constant (heating fallback + cooling check)...")
    tau = _heating_fallback("thermal_time_constant")

    # Try cooling curves (HP-off periods where building warms)
    tau_cooling, tau_r2 = calculate_cooling_time_constant(df)
    if tau_cooling is not None and tau_r2 is not None and tau_r2 > _SLAB_TAU_MIN_R2:
        # High confidence — override
        tau = tau_cooling
        logging.info(
            "✅ Cooling τ_room = %.2fh (R²=%.3f) — overrides heating fallback",
            tau, tau_r2,
        )
    else:
        logging.info(
            "Using heating-calibrated τ_room = %.2fh (cooling R²=%.3f insufficient)",
            tau, tau_r2 or 0.0,
        )

    # --- Step 4: PV heat weight ---
    logging.info("Step 4: PV heat weight (from cooling periods with PV)...")
    pv_periods = _filter_cooling_pv_periods(stable_periods)
    pv_weight = _residual_heat_source_weight(
        pv_periods, "pv", hlc, oe, min_periods=15, percentile=50.0
    )
    if pv_weight is None:
        logging.warning("⚠️ PV weight calibration failed — using fallback")
        pv_weight = _cooling_fallback("pv_heat_weight")

    # --- Steps 5-6: Fireplace/TV weights — use defaults (not calibrated) ---
    logging.info("Steps 5-6: Fireplace/TV weights — using defaults (not relevant for cooling)")
    fp_weight = ThermalParameterConfig.get_cooling_default("fireplace_heat_weight")
    tv_weight = ThermalParameterConfig.get_cooling_default("tv_heat_weight")

    # --- Step 7: Solar lag ---
    logging.info("Step 7: Solar lag (cross-correlation on cooling data)...")
    solar_lag = _calibrate_solar_lag_xcorr(df, hlc, oe)
    if solar_lag is None:
        logging.warning("⚠️ Solar lag calibration failed — using fallback")
        solar_lag = _cooling_fallback("solar_lag_minutes")

    # --- Step 8: delta_t_floor ---
    logging.info("Step 8: delta_t_floor (cooling periods)...")
    delta_t_floor_val = calibrate_delta_t_floor(stable_periods)
    if delta_t_floor_val is None:
        logging.warning("⚠️ delta_t_floor calibration failed — using fallback")
        delta_t_floor_val = _cooling_fallback("delta_t_floor")

    # --- Step 9: Slab time constant ---
    # Use heating value as prior; only override if cooling result is high confidence
    logging.info("Step 9: Slab time constant (heating fallback + cooling check)...")
    slab_tau = _heating_fallback("slab_time_constant_hours")

    slab_tau_cooling = _calibrate_slab_tau_grid_search(df, delta_t_floor=delta_t_floor_val)
    if slab_tau_cooling is not None:
        # The grid search succeeded — validate reasonableness
        bounds = ThermalParameterConfig.get_cooling_bounds("slab_time_constant_hours")
        if bounds[0] <= slab_tau_cooling <= bounds[1]:
            slab_tau = slab_tau_cooling
            logging.info(
                "✅ Cooling slab τ = %.2fh — overrides heating fallback",
                slab_tau,
            )
        else:
            logging.warning(
                "⚠️ Cooling slab τ %.2fh outside bounds — keeping heating fallback %.2fh",
                slab_tau_cooling, slab_tau,
            )
    else:
        logging.info("Using heating-calibrated slab τ = %.2fh", slab_tau)

    # --- Step 10-11: FP decay / room spread — skip (not relevant for cooling) ---
    fp_decay_tau = ThermalParameterConfig.get_cooling_default("fp_decay_time_constant")
    room_spread_delay = ThermalParameterConfig.get_cooling_default("room_spread_delay_minutes")

    # --- Step 12: Cloud factor exponent — use default ---
    cloud_exponent = ThermalParameterConfig.get_cooling_default("cloud_factor_exponent")

    # --- Step 13: Solar decay tau ---
    logging.info("Step 13: Solar decay tau...")
    pv_decay_periods = filter_pv_decay_periods(df)
    solar_decay_tau = calibrate_solar_decay_tau(pv_decay_periods)
    if solar_decay_tau is None:
        solar_decay_tau = _cooling_fallback("solar_decay_tau_hours")

    # --- Assemble calibrated parameters ---
    calibrated_params: Dict[str, float] = {
        "heat_loss_coefficient": hlc,
        "outlet_effectiveness": oe,
        "thermal_time_constant": tau,
        "pv_heat_weight": pv_weight,
        "fireplace_heat_weight": fp_weight,
        "tv_heat_weight": tv_weight,
        "solar_lag_minutes": solar_lag,
        "delta_t_floor": delta_t_floor_val,
        "slab_time_constant_hours": slab_tau,
        "fp_decay_time_constant": fp_decay_tau,
        "room_spread_delay_minutes": room_spread_delay,
    }

    # --- Build thermal model ---
    logging.info("Building cooling thermal model with calibrated parameters...")
    thermal_model = ThermalEquilibriumModel()
    thermal_model.thermal_time_constant = tau
    thermal_model.heat_loss_coefficient = hlc
    thermal_model.outlet_effectiveness = oe
    thermal_model.external_source_weights["pv"] = pv_weight
    thermal_model.external_source_weights["fireplace"] = fp_weight
    thermal_model.external_source_weights["tv"] = tv_weight
    thermal_model.solar_lag_minutes = solar_lag
    thermal_model.slab_time_constant_hours = slab_tau
    thermal_model.delta_t_floor = delta_t_floor_val
    thermal_model.fp_decay_time_constant = fp_decay_tau
    thermal_model.room_spread_delay_minutes = room_spread_delay
    thermal_model.learning_confidence = 3.0

    # --- Log summary ---
    logging.info("\n=== COOLING PHYSICS-DIRECT CALIBRATED PARAMETERS ===")
    logging.info("  heat_loss_coefficient:  %.5f kW/K", hlc)
    logging.info("  outlet_effectiveness:   %.4f kW/K", oe)
    logging.info("  thermal_time_constant:  %.2f h", tau)
    logging.info("  pv_heat_weight:         %.6f kW/W", pv_weight)
    logging.info("  fireplace_heat_weight:  %.3f kW (default)", fp_weight)
    logging.info("  tv_heat_weight:         %.3f kW (default)", tv_weight)
    logging.info("  solar_lag_minutes:      %.1f min", solar_lag)
    logging.info("  delta_t_floor:          %.2f °C", delta_t_floor_val)
    logging.info("  slab_time_constant:     %.2f h", slab_tau)
    logging.info("  fp_decay_time_constant: %.2f h (default)", fp_decay_tau)
    logging.info("  room_spread_delay:      %.0f min (default)", room_spread_delay)

    # --- Persist to cooling thermal state ---
    logging.info("Saving cooling physics-direct parameters to cooling thermal state...")
    try:
        cooling_sm.set_calibrated_baseline(
            calibrated_params,
            calibration_cycles=len(stable_periods),
        )
        logging.info(
            "✅ Cooling physics-direct parameters saved to cooling thermal state"
        )
    except Exception as exc:
        logging.error("❌ Failed to save cooling parameters: %s", exc)

    return thermal_model
