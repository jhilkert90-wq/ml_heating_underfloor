"""
Physics-Direct Calibration for the Thermal Equilibrium Model.

This module provides a physics-first, sequential calibration path that
estimates every thermal parameter from first principles.  Each parameter is
derived from the underlying heat-balance physics and locked before the next
one is estimated, breaking the inter-parameter degeneracy that can trap
joint optimization.

Where a closed-form analytical estimate is numerically fragile (e.g. OE
with small temperature drives), a scipy refinement pass is used to produce
the best physically correct result.

The user can choose between this path and the existing scipy optimizer via the
``CALIBRATION_METHOD`` config variable (``"physics"`` or ``"scipy"``) or via
the dashboard toggle.

Calibration sequence
--------------------
The equilibrium equation used throughout is::

    T_eq = (OE × T_outlet + HLC × T_outdoor + P_external) / (OE + HLC)

where OE (outlet effectiveness) and HLC (heat-loss coefficient) are both in
kW/K.  Each step locks a parameter before estimating the next:

1.  **HLC** — forced-through-origin OLS via ``calibrate_hlc()``.
2.  **OE** — analytical initial guess + 1-D scipy refinement (HLC locked).
3.  **τ_room** — transient calibration (primary) or cooling curves (fallback).
4.  **pv_heat_weight** — residual energy balance on PV-on periods.
5.  **fireplace_heat_weight** — residual energy balance on FP-on periods.
6.  **tv_heat_weight** — residual energy balance on TV-on periods.
7.  **solar_lag_minutes** — cross-correlation of PV vs d(residual)/dt.
8.  **delta_t_floor** — P25 percentile via ``calibrate_delta_t_floor()``.
9.  **slab_time_constant_hours** — 1-D grid search over [0.1, 4.0] h.
10. **fp_decay_time_constant** — log-linear OLS via ``calibrate_fp_decay_tau()``.
11. **room_spread_delay_minutes** — cross-correlation via ``calibrate_room_spread_delay()``.
12. **cloud_factor_exponent** — log-OLS (only when CLOUD_COVER_CORRECTION_ENABLED).
13. **solar_decay_tau_hours** — log-linear OLS via ``calibrate_solar_decay_tau()``.

Required sensors (same as the existing scipy path)
--------------------------------------------------
flow_rate, inlet_temp, outlet_temp, indoor_temp, outdoor_temp, pv_power,
fireplace_status, tv_status.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

try:
    from scipy.optimize import minimize_scalar
except ImportError:
    minimize_scalar = None
    logging.warning("scipy not available — OE refinement will use analytical estimate only")

try:
    from . import config
    from .thermal_equilibrium_model import ThermalEquilibriumModel
    from .thermal_config import ThermalParameterConfig
    from .unified_thermal_state import get_thermal_state_manager
    from .state_manager import save_state
    from .physics_calibration import (
        fetch_historical_data_for_calibration,
        filter_stable_periods,
        backup_existing_calibration,
        calculate_cooling_time_constant,
        calibrate_delta_t_floor,
        calibrate_fp_decay_tau,
        calibrate_room_spread_delay,
        calibrate_solar_decay_tau,
        filter_fp_decay_periods,
        filter_fp_spread_periods,
        filter_pv_decay_periods,
        _filter_hp_only_periods,
        _apply_channel_params,
        filter_transient_periods,
        calibrate_transient_parameters,
    )
    from .hlc_learner import calibrate_hlc
except ImportError:
    import config  # type: ignore
    from thermal_equilibrium_model import ThermalEquilibriumModel  # type: ignore
    from thermal_config import ThermalParameterConfig  # type: ignore
    from unified_thermal_state import get_thermal_state_manager  # type: ignore
    from state_manager import save_state  # type: ignore
    from physics_calibration import (  # type: ignore
        fetch_historical_data_for_calibration,
        filter_stable_periods,
        backup_existing_calibration,
        calculate_cooling_time_constant,
        calibrate_delta_t_floor,
        calibrate_fp_decay_tau,
        calibrate_room_spread_delay,
        calibrate_solar_decay_tau,
        filter_fp_decay_periods,
        filter_fp_spread_periods,
        filter_pv_decay_periods,
        _filter_hp_only_periods,
        _apply_channel_params,
        filter_transient_periods,
        calibrate_transient_parameters,
    )
    from hlc_learner import calibrate_hlc  # type: ignore


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Minimum PV electrical power [W] to treat a period as "PV active"
_MIN_PV_POWER_W: float = 100.0

# Minimum PV power [W] for a cross-correlation lag sample to count
_MIN_PV_ACTIVE_FOR_LAG_W: float = 50.0

# Minimum periods with active PV for the solar lag cross-correlation to run
_MIN_PV_SAMPLES_FOR_LAG: int = 20

# Sanity bounds for the per-window PV weight estimator [kW/W]
_PV_WEIGHT_MIN_KW_PER_W: float = 0.0001
_PV_WEIGHT_MAX_KW_PER_W: float = 0.005

# Upper bound for auxiliary heat-source contribution [kW] (FP / TV)
_MAX_AUX_HEAT_SOURCE_KW: float = 10.0

# Assumed samples per hour in the training data (5-min intervals)
_SAMPLES_PER_HOUR: int = 12


def _refine_oe_scipy(
    hp_periods, hlc: float, oe_initial: float,
) -> Optional[float]:
    """Refine outlet_effectiveness via 1-D scipy optimization.

    Locks HLC and optimizes OE only by minimizing MAE between the
    temperature-based equilibrium prediction and actual indoor temperature
    across HP-only stable periods.  This mirrors the scipy calibration
    path's "Pass 1 OE-only" approach but starts from the physics-derived
    analytical initial guess.

    Parameters
    ----------
    hp_periods:
        HP-only period dicts from ``_filter_hp_only_periods()``.
    hlc:
        Heat-loss coefficient [kW/K], locked from Step 1.
    oe_initial:
        Analytical OE estimate used as initial guess.

    Returns
    -------
    float or None
        Refined OE [kW/K], or *None* if scipy unavailable or optimization fails.
    """
    if minimize_scalar is None:
        logging.info("scipy not available — skipping OE refinement")
        return None

    if len(hp_periods) < 10:
        return None

    # Sub-sample for speed (every 3rd period)
    sampled = hp_periods[::3] if len(hp_periods) > 100 else hp_periods

    def mae_objective(oe_candidate: float) -> float:
        total_error = 0.0
        count = 0
        denom = oe_candidate + hlc
        if denom <= 0:
            return 1e9
        for p in sampled:
            t_in = p.get("indoor_temp")
            t_out = p.get("outdoor_temp")
            t_outlet = p.get("effective_temp", p.get("outlet_temp"))
            if t_in is None or t_out is None or t_outlet is None:
                continue
            if np.isnan(t_in) or np.isnan(t_out) or np.isnan(t_outlet):
                continue
            predicted = (oe_candidate * t_outlet + hlc * t_out) / denom
            total_error += abs(predicted - t_in)
            count += 1
        return total_error / count if count > 0 else 1e9

    bounds = ThermalParameterConfig.get_bounds("outlet_effectiveness")
    try:
        result = minimize_scalar(
            mae_objective,
            bounds=(bounds[0], bounds[1]),
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 200},
        )
        if result.success or result.fun < 5.0:
            refined = float(result.x)
            logging.info(
                "✅ Scipy-refined OE: %.4f kW/K (MAE=%.4f°C, "
                "analytical initial=%.4f, delta=%.4f)",
                refined, result.fun, oe_initial, refined - oe_initial,
            )
            return refined
        else:
            logging.warning(
                "⚠️ OE scipy refinement did not converge: %s", result.message
            )
            return None
    except Exception as exc:
        logging.warning("⚠️ OE scipy refinement failed: %s", exc)
        return None


def _calibrate_oe_analytical(
    stable_periods, hlc: float
) -> Optional[float]:
    """Estimate outlet_effectiveness from HP-only stable periods.

    Uses a two-stage approach:

    1. **Analytical initial guess** — per-window algebraic inversion of the
       equilibrium equation (no external sources)::

           T_indoor = (OE × T_outlet + HLC × T_outdoor) / (OE + HLC)
           → OE = HLC × (T_indoor − T_outdoor) / (T_outlet − T_indoor)

       The weighted median across qualifying windows provides a
       physics-anchored starting point.

    2. **Scipy refinement** — 1-D bounded optimization of OE (HLC locked)
       that minimizes MAE of the temperature-based equilibrium prediction
       vs. actual indoor temperature.  This is robust to per-sample noise
       that biases the algebraic inversion when the temperature drive
       is small.

    Parameters
    ----------
    stable_periods:
        List of period dicts from ``filter_stable_periods()``.
    hlc:
        Heat-loss coefficient [kW/K] from Step 1.

    Returns
    -------
    float or None
        Estimated outlet_effectiveness [kW/K], or *None* if insufficient data.
    """
    logging.info("=== STEP 2: OE ESTIMATION (analytical + scipy refinement) ===")

    if hlc <= 0:
        logging.warning("⚠️ OE calibration skipped — HLC ≤ 0")
        return None

    hp_periods = _filter_hp_only_periods(stable_periods)
    if not hp_periods:
        logging.warning("⚠️ No HP-only periods for OE calibration")
        return None

    oe_values = []
    weights = []

    for p in hp_periods:
        t_in = p.get("indoor_temp")
        t_out = p.get("outdoor_temp")
        # Prefer effective temperature (BT2+BT3)/2 when available
        t_outlet = p.get("effective_temp", p.get("outlet_temp"))

        if t_in is None or t_out is None or t_outlet is None:
            continue
        if np.isnan(t_in) or np.isnan(t_out) or np.isnan(t_outlet):
            continue

        # Quality gate: outlet must be meaningfully above indoor.
        # Raised from 2.0 to 3.0 — small drives make the algebraic
        # inversion (denominator = drive) extremely noise-sensitive.
        drive = t_outlet - t_in
        if drive < 3.0:
            continue

        delta_ti = t_in - t_out
        if delta_ti <= 0:
            continue  # No heating demand

        oe = hlc * delta_ti / drive

        bounds = ThermalParameterConfig.get_bounds("outlet_effectiveness")
        if bounds[0] <= oe <= bounds[1]:
            oe_values.append(oe)
            weights.append(drive)  # larger drive → more reliable estimate

    if len(oe_values) < 10:
        logging.warning(
            "⚠️ Insufficient HP-only periods for OE calibration: %d (need ≥10)",
            len(oe_values),
        )
        return None

    # Weighted median: sort by value, apply weight-based selection
    sorted_pairs = sorted(zip(oe_values, weights), key=lambda x: x[0])
    sorted_oe = [x[0] for x in sorted_pairs]
    sorted_w = [x[1] for x in sorted_pairs]
    cum_w = np.cumsum(sorted_w)
    total_w = cum_w[-1]
    median_idx = np.searchsorted(cum_w, total_w / 2.0)
    median_idx = min(median_idx, len(sorted_oe) - 1)
    oe_analytical = sorted_oe[median_idx]

    logging.info(
        "📐 Analytical OE estimate: %.4f kW/K "
        "(from %d HP-only periods, weighted median)",
        oe_analytical, len(oe_values),
    )

    # --- Stage 2: scipy refinement ---
    oe_refined = _refine_oe_scipy(hp_periods, hlc, oe_analytical)

    if oe_refined is not None:
        # Cross-validate: warn if analytical and refined differ substantially
        pct_diff = abs(oe_refined - oe_analytical) / max(oe_analytical, 0.01) * 100
        if pct_diff > 30:
            logging.warning(
                "⚠️ OE analytical (%.4f) and scipy-refined (%.4f) differ by %.0f%% "
                "— analytical estimate may be biased by noise or post-defrost periods",
                oe_analytical, oe_refined, pct_diff,
            )
        return oe_refined

    logging.info("Using analytical OE estimate (scipy refinement unavailable)")
    return oe_analytical


# ---------------------------------------------------------------------------
# Step 4-6: Residual heat source weights
# ---------------------------------------------------------------------------

def _residual_heat_source_weight(
    periods,
    source_key: str,
    hlc: float,
    oe: float,
    min_periods: int = 5,
    percentile: float = 50.0,
) -> Optional[float]:
    """Estimate a heat-source weight from equilibrium residuals.

    For periods where only the given source is active (HP + source, no
    others), rearrange the equilibrium equation to solve for the power
    contributed by the source::

        P_source = (T_indoor − T_outdoor) × (OE + HLC) − OE × (T_outlet − T_outdoor)

    For PV: weight = P_source / pv_power [kW/W]
    For FP/TV: weight = median(P_source) [kW] — these are per-event constants.

    Parameters
    ----------
    periods:
        Filtered period dicts (already restricted to relevant source on).
    source_key:
        ``"pv"``, ``"fp"``, or ``"tv"`` — selects divisor and units.
    hlc, oe:
        Fixed coefficients from earlier steps [kW/K each].
    min_periods:
        Minimum qualifying periods for a valid estimate.
    percentile:
        Percentile of the sample distribution to return (default median=50).

    Returns
    -------
    float or None
    """
    label = source_key.upper()
    logging.info("=== STEP: RESIDUAL %s WEIGHT ESTIMATION ===", label)

    values = []
    denom_total = oe + hlc
    if denom_total <= 0:
        logging.warning("⚠️ %s residual: OE+HLC ≤ 0, skipping", label)
        return None

    for p in periods:
        t_in = p.get("indoor_temp")
        t_out = p.get("outdoor_temp")
        t_outlet = p.get("effective_temp", p.get("outlet_temp"))

        if t_in is None or t_out is None or t_outlet is None:
            continue
        if np.isnan(t_in) or np.isnan(t_out) or np.isnan(t_outlet):
            continue

        # P_source = (T_in - T_out) × (OE+HLC) - OE × (T_outlet - T_out)
        p_source = (t_in - t_out) * denom_total - oe * (t_outlet - t_out)

        if source_key == "pv":
            pv = p.get("pv_power", 0.0)
            if pv < _MIN_PV_POWER_W:
                continue
            w = p_source / pv  # kW/W
            if _PV_WEIGHT_MIN_KW_PER_W <= w <= _PV_WEIGHT_MAX_KW_PER_W:
                values.append(w)
        else:
            # FP / TV: direct kW contribution; must be positive and bounded
            if 0.0 < p_source < _MAX_AUX_HEAT_SOURCE_KW:
                values.append(p_source)

    # IQR-based outlier rejection before taking the percentile.
    # Sensor noise or transient slab effects can produce extreme residuals
    # that bias the estimate even after the initial bounds filter.
    if len(values) >= 4:
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        if iqr > 0:
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            n_before = len(values)
            values = [v for v in values if lo <= v <= hi]
            n_removed = n_before - len(values)
            if n_removed:
                logging.debug(
                    "  %s residual: removed %d/%d outliers "
                    "(IQR fence [%.5f, %.5f])",
                    label, n_removed, n_before, lo, hi,
                )

    if len(values) < min_periods:
        logging.warning(
            "⚠️ Insufficient periods for %s weight estimation: %d (need %d)",
            label, len(values), min_periods,
        )
        return None

    estimate = float(np.percentile(values, percentile))
    logging.info(
        "✅ %s weight estimate: %.5f (P%.0f of %d samples)",
        label, estimate, percentile, len(values),
    )
    return estimate


def _filter_fp_only_periods(stable_periods):
    """HP + fireplace on, no PV (< _MIN_PV_POWER_W W), no TV."""
    return [
        p for p in stable_periods
        if p.get("fireplace_on", 0) > 0
        and p.get("pv_power", 0) < _MIN_PV_POWER_W
        and p.get("tv_on", 0) == 0
        and p.get("thermal_power_kw", 0) >= config.HEATING_MIN_THERMAL_POWER_KW
    ]


def _filter_tv_only_periods(stable_periods):
    """HP + TV on, no PV (< _MIN_PV_POWER_W W), no fireplace."""
    return [
        p for p in stable_periods
        if p.get("tv_on", 0) > 0
        and p.get("pv_power", 0) < _MIN_PV_POWER_W
        and p.get("fireplace_on", 0) == 0
        and p.get("thermal_power_kw", 0) >= config.HEATING_MIN_THERMAL_POWER_KW
    ]




# ---------------------------------------------------------------------------
# Step 7: Solar lag via cross-correlation
# ---------------------------------------------------------------------------

def _calibrate_solar_lag_xcorr(
    df, hlc: float, oe: float
) -> Optional[float]:
    """Estimate solar_lag_minutes from time-contiguous PV-active episodes.

    Cross-correlates PV power against the **rate of change** of the
    HP-equilibrium residual (``d(residual)/dt``) within each contiguous
    PV-active episode.  Using the rate of change rather than the residual
    level removes the slab-mass smoothing effect that otherwise pushes
    the correlation peak to artificially large lags.

    The weighted median of per-episode best-lags (weighted by peak
    correlation strength) is returned — more robust than the modal
    selection used previously.

    Parameters
    ----------
    df:
        Raw time-series DataFrame from ``fetch_historical_data_for_calibration``.
    hlc, oe:
        Heat-loss coefficient and outlet effectiveness [kW/K].

    Returns
    -------
    float or None
        Lag in minutes, or *None* if insufficient qualifying episodes.
    """
    logging.info("=== STEP 7: SOLAR LAG CROSS-CORRELATION ===")

    denom = oe + hlc
    if denom <= 0:
        logging.warning("⚠️ Solar lag: OE+HLC ≤ 0, skipping")
        return None

    if df is None or df.empty:
        return None

    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]

    for c in (pv_col, indoor_col, outdoor_col, outlet_col):
        if c not in df.columns:
            logging.info("Solar lag: column %s missing — skipping", c)
            return None

    df_s = df.sort_values("_time").reset_index(drop=True)
    pv_arr = df_s[pv_col].fillna(0.0).values.astype(float)
    t_in_arr = df_s[indoor_col].values.astype(float)
    t_out_arr = df_s[outdoor_col].values.astype(float)
    t_outlet_arr = df_s[outlet_col].values.astype(float)

    # HP-equilibrium residual for the full time-sorted series
    hp_eq = (oe * t_outlet_arr + hlc * t_out_arr) / denom
    residual_arr = t_in_arr - hp_eq

    # Use rate-of-change of residual: removes slab-mass delay that
    # would otherwise bias the lag estimate to very large values.
    d_residual = np.diff(residual_arr)

    # Physical solar lag is 10-60 min; slab delay is modeled separately.
    # Reduced from 36 (180 min) to 12 (60 min) to prevent slab-mass
    # confusion.
    max_lag_steps = 12   # 60 min in 5-min steps
    step_min = 5
    min_ep_len = _MIN_PV_SAMPLES_FOR_LAG + max_lag_steps

    pv_active = pv_arr > _MIN_PV_ACTIVE_FOR_LAG_W
    episode_lags: List[int] = []
    episode_corrs: List[float] = []

    # Segment into contiguous PV-active runs and compute per-episode xcorr
    i = 0
    n = len(pv_arr)
    while i < n:
        if not pv_active[i]:
            i += 1
            continue

        ep_start = i
        while i < n and pv_active[i]:
            i += 1
        ep_end = i  # exclusive

        if ep_end - ep_start < min_ep_len:
            continue

        # Use d(residual)/dt for the episode (one element shorter)
        ep_pv = pv_arr[ep_start:ep_end - 1]  # align with diff
        ep_dres = d_residual[ep_start:ep_end - 1]

        pv_std = float(ep_pv.std())
        dres_std = float(ep_dres.std())
        if pv_std < 1e-6 or dres_std < 1e-6:
            continue

        pv_norm = (ep_pv - ep_pv.mean()) / pv_std
        dres_norm = (ep_dres - ep_dres.mean()) / dres_std

        best_lag = 0
        best_corr = float(np.corrcoef(pv_norm, dres_norm)[0, 1])
        for lag in range(1, max_lag_steps + 1):
            corr = float(np.corrcoef(pv_norm[lag:], dres_norm[:-lag])[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        # Raised threshold from 0.1 to 0.3 — reject noisy episodes
        # that produce near-random correlation peaks at arbitrary lags.
        if best_corr >= 0.3:
            episode_lags.append(best_lag)
            episode_corrs.append(best_corr)

    if not episode_lags:
        logging.info(
            "Solar lag: no qualifying contiguous PV episodes found — "
            "using default"
        )
        return None

    # Weighted median by correlation strength (more robust than mode).
    sorted_pairs = sorted(zip(episode_lags, episode_corrs), key=lambda x: x[0])
    sorted_lags = [x[0] for x in sorted_pairs]
    sorted_w = [x[1] for x in sorted_pairs]
    cum_w = np.cumsum(sorted_w)
    total_w = cum_w[-1]
    median_idx = np.searchsorted(cum_w, total_w / 2.0)
    median_idx = min(median_idx, len(sorted_lags) - 1)
    best_lag_steps = sorted_lags[median_idx]
    best_lag_min = float(best_lag_steps * step_min)

    logging.info(
        "✅ Solar lag estimate: %.0f min (weighted median across %d PV episodes)",
        best_lag_min, len(episode_lags),
    )

    bounds = ThermalParameterConfig.get_bounds("solar_lag_minutes")
    best_lag_min = max(bounds[0], min(bounds[1], best_lag_min))
    return best_lag_min


# ---------------------------------------------------------------------------
# Step 9: Slab time constant via 1-D grid search
# ---------------------------------------------------------------------------

def _calibrate_slab_tau_grid_search(
    df, delta_t_floor: Optional[float] = None
) -> Optional[float]:
    """Estimate slab_time_constant_hours via a 1-D grid search.

    Uses the same physical model as ``calibrate_slab_time_constant()`` but
    replaces the scipy L-BFGS-B call with a uniform grid search over
    ``[0.1, 4.0]`` hours in 0.05 h steps.  The MSE function is well-behaved
    and unimodal for this parameter, so the grid gives the same result
    without requiring scipy.

    Parameters
    ----------
    df:
        Raw time-series DataFrame.
    delta_t_floor:
        Calibrated heat-pump loop delta-T floor [°C].  When provided the
        calibrated value is used; when *None* the config default is used.
        Pass the value from Step 8 so that the slab-tau estimate is not
        biased by an incorrect default.

    Returns
    -------
    float or None
    """
    logging.info("=== STEP 9: SLAB TIME CONSTANT GRID SEARCH ===")

    if df is None or df.empty:
        return None

    inlet_col = config.INLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    flow_col = config.FLOW_RATE_ENTITY_ID.split(".", 1)[-1]

    for c in (inlet_col, outlet_col, flow_col):
        if c not in df.columns:
            logging.info(
                "Column %s missing — skipping slab tau grid search", c
            )
            return None

    if delta_t_floor is None:
        delta_t_floor = ThermalParameterConfig.get_default("delta_t_floor")

    df_s = df.sort_values("_time").reset_index(drop=True)
    delta_t_vals = df_s[outlet_col] - df_s[inlet_col]
    thermal_power = (
        (df_s[flow_col] / 60.0)
        * config.SPECIFIC_HEAT_CAPACITY
        * delta_t_vals
    ).clip(lower=0.0)

    hp_on = thermal_power >= config.HEATING_MIN_THERMAL_POWER_KW
    hp_on_prev = hp_on.shift(1, fill_value=False)
    startups = hp_on & ~hp_on_prev

    startup_indices = startups[startups].index.tolist()
    if not startup_indices:
        logging.info("No HP startup events found for slab tau grid search")
        return None

    # Collect (t_hours, inlet_vals, target_vals) for each startup
    events = []
    for start_idx in startup_indices:
        end_idx = start_idx
        while end_idx + 1 < len(df_s) and hp_on.iloc[end_idx + 1]:
            end_idx += 1

        window = df_s.iloc[start_idx: end_idx + 1]
        if len(window) < 6:
            continue

        t_hours_all = (
            window["_time"] - window["_time"].iloc[0]
        ).dt.total_seconds().values / 3600.0
        use = t_hours_all <= 1.5
        window = window.iloc[: int(use.sum())]
        t_hours = t_hours_all[: len(window)]
        if len(window) < 6:
            continue

        inlet_vals = window[inlet_col].values.astype(float)
        outlet_vals_w = window[outlet_col].values.astype(float)
        target_vals = outlet_vals_w - delta_t_floor

        max_gap = float(np.max(target_vals - inlet_vals))
        if max_gap < 0.5:
            continue

        dt_steps = np.diff(t_hours)
        events.append((dt_steps, inlet_vals, target_vals))

    if not events:
        logging.info("No valid HP startup events for slab tau grid search")
        return None

    # 1-D grid search
    tau_grid = np.arange(0.1, 4.05, 0.05)
    best_tau = None
    best_mse = float("inf")

    for tau in tau_grid:
        total_mse = 0.0
        total_n = 0
        for dt_steps, inlet_vals, target_vals in events:
            pred = inlet_vals[0]
            for k, dt in enumerate(dt_steps):
                alpha = min(1.0, dt / tau)
                pred = pred + alpha * (target_vals[k] - pred)
                total_mse += (pred - inlet_vals[k + 1]) ** 2
                total_n += 1
        if total_n > 0:
            avg_mse = total_mse / total_n
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_tau = float(tau)

    if best_tau is None:
        return None

    # Quality gate: MSE < 0.5 °C²
    if best_mse > 0.5:
        logging.warning(
            "⚠️ Slab tau grid search: best MSE %.3f°C² too high — "
            "result may be unreliable", best_mse
        )

    result = max(0.2, min(4.0, best_tau))
    logging.info(
        "✅ Slab time constant (grid search): %.2fh (MSE=%.4f°C², "
        "%d startup events)",
        result, best_mse, len(events),
    )
    return result


# ---------------------------------------------------------------------------
# Step 12: Cloud factor exponent via log-OLS
# ---------------------------------------------------------------------------

def _calibrate_cloud_exponent_log_ols(
    df, pv_heat_weight: float
) -> Optional[float]:
    """Estimate cloud_factor_exponent via closed-form log-OLS.

    The cloud model is::

        effective_pv = pv × (1 - cloud/100)^exp

    Ideal approach: take ``ln(effective_pv / pv) = exp × ln(1 - cloud/100)``
    and regress. However, we do not have a direct measurement of
    ``effective_pv`` from the raw DataFrame — we only have raw PV production
    and cloud-cover fraction.

    Proxy relationship used instead: OLS of ``ln(pv)`` against
    ``ln(1 - cloud/100)`` over a diverse cloud-cover range.  The slope of
    this regression is the exponent, on the assumption that ``pv`` production
    (for a fixed sky brightness) is modulated by ``(1 - cloud/100)^exp``.
    The intercept absorbs irradiance-level variation; only the slope is used.

    This gives a good first-principles estimate of the exponent but requires
    periods with varying cloud cover and consistent solar angle — treat as a
    starting point that online learning refines at runtime.

    Returns
    -------
    float or None
    """
    logging.info("=== STEP 12: CLOUD FACTOR EXPONENT LOG-OLS ===")

    if pv_heat_weight <= 0:
        logging.warning("⚠️ Cloud exponent: pv_heat_weight ≤ 0, skipping")
        return None

    if df is None or df.empty:
        return None

    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    cloud_col = "cloud_cover_proxy"

    for c in (pv_col, cloud_col):
        if c not in df.columns:
            logging.info(
                "Column %s missing — skipping cloud exponent OLS", c
            )
            return None

    mask = (
        (df[pv_col] > 200)
        & df[cloud_col].between(10, 90)
        & df[pv_col].notna()
        & df[cloud_col].notna()
    )
    subset = df[mask]
    if len(subset) < 30:
        logging.info(
            "Insufficient cloud data (%d rows) for log-OLS", len(subset)
        )
        return None

    pv_vals = subset[pv_col].values.astype(float)
    cloud_vals = subset[cloud_col].values.astype(float)

    # x = ln(1 - cloud/100): the log of the clear-sky fraction
    # y = ln(pv) demeaned: removes irradiance-level baseline
    # The OLS slope estimates the cloud attenuation exponent.
    cloud_frac = cloud_vals / 100.0
    log_attenuation = np.log(np.clip(1.0 - cloud_frac, 1e-6, 1.0))
    log_pv = np.log(pv_vals)

    x = log_attenuation
    y = log_pv - np.mean(log_pv)
    x_centered = x - np.mean(x)
    denom = float(np.dot(x_centered, x_centered))
    if denom < 1e-8:
        logging.info("Cloud exponent: insufficient x-variance in log-OLS")
        return None
    exp_est = float(np.dot(x_centered, y)) / denom

    bounds = ThermalParameterConfig.get_bounds("cloud_factor_exponent")
    exp_est = max(bounds[0], min(bounds[1], exp_est))

    logging.info(
        "✅ Cloud factor exponent (log-OLS): %.3f (from %d rows)",
        exp_est, len(subset),
    )
    return exp_est


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def calibrate_thermal_model_physics(
    state_manager=None,
) -> Optional[ThermalEquilibriumModel]:
    """Calibrate all thermal parameters using physics-direct methods.

    This is the main entry point for the physics-based calibration path.
    It mirrors ``train_thermal_equilibrium_model()`` in its output format
    (saves to the same unified thermal state) but derives every parameter
    analytically rather than through scipy joint optimization.

    Parameters
    ----------
    state_manager:
        Optional state manager to persist calibrated parameters to.
        Defaults to the heating singleton when *None*.

    Returns
    -------
    ThermalEquilibriumModel or None
        Configured model on success, *None* on failure.
    """
    logging.info(
        "=== THERMAL EQUILIBRIUM MODEL CALIBRATION (PHYSICS-DIRECT) ==="
    )

    # --- Step 0: Backup existing calibration ---
    backup_existing_calibration(state_manager=state_manager)

    # Resolve the active state manager early so that persisted parameter
    # values from unified_thermal_state.json are available as warm-start
    # fallbacks for every step that fails to produce enough calibration data.
    _active_sm = (
        state_manager if state_manager is not None else get_thermal_state_manager()
    )
    _persisted: Dict[str, float] = _active_sm.state.get("baseline_parameters", {})

    def _state_fallback(key: str) -> float:
        """Return the persisted state-file value when available and valid;
        fall back to the hard-coded config default otherwise.

        Priority: calibration result > state-file value (within bounds) > config default.
        """
        val = _persisted.get(key)
        try:
            fval = float(val)  # type: ignore[arg-type]
            if not np.isnan(fval):
                # Validate against parameter bounds to reject corrupted values.
                try:
                    lo, hi = ThermalParameterConfig.get_bounds(key)
                    if lo <= fval <= hi:
                        return fval
                    logging.warning(
                        "⚠️ Persisted '%s'=%.4g is outside bounds [%.4g, %.4g]"
                        " — falling back to config default",
                        key, fval, lo, hi,
                    )
                except KeyError:
                    # No bounds defined for this key — accept any finite value.
                    return fval
        except (TypeError, ValueError):
            pass
        return ThermalParameterConfig.get_default(key)

    def _fallback_source(key: str) -> str:
        """Return 'persisted' when the state file has a valid, in-bounds value, else 'default'."""
        val = _persisted.get(key)
        try:
            fval = float(val)  # type: ignore[arg-type]
            if not np.isnan(fval):
                try:
                    lo, hi = ThermalParameterConfig.get_bounds(key)
                    if lo <= fval <= hi:
                        return "persisted"
                except KeyError:
                    return "persisted"
        except (TypeError, ValueError):
            pass
        return "default"

    # --- Fetch data ---
    logging.info("Fetching historical data...")
    df = fetch_historical_data_for_calibration(
        lookback_hours=config.TRAINING_LOOKBACK_HOURS
    )
    if df is None or df.empty:
        logging.error("❌ Failed to fetch historical data")
        return None
    logging.info(
        "✅ Retrieved %d samples (%.1f hours)", len(df), len(df) / _SAMPLES_PER_HOUR
    )

    # Warn if key columns have large NaN gaps — bfill/ffill imputation can
    # silently fill sensor-offline stretches and bias calibration parameters.
    _nan_gap_cols = (
        config.FLOW_RATE_ENTITY_ID.split(".", 1)[-1],
        config.INLET_TEMP_ENTITY_ID.split(".", 1)[-1],
        config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1],
    )
    _NAN_GAP_WARN_ROWS = 6  # > 30 min of consecutive NaN at 5-min resolution
    for _col in _nan_gap_cols:
        if _col not in df.columns:
            continue
        _nan_mask = df[_col].isna()
        if not _nan_mask.any():
            continue
        # Count consecutive NaN runs
        _run_len = (
            _nan_mask.groupby((~_nan_mask).cumsum()).transform("sum")
        )
        _max_gap = int(_run_len[_nan_mask].max()) if _nan_mask.any() else 0
        if _max_gap > _NAN_GAP_WARN_ROWS:
            logging.warning(
                "⚠️ Column '%s' has %d consecutive NaN rows (>%d min gap) "
                "— bfill/ffill imputation may bias calibrated parameters",
                _col, _max_gap, _max_gap * 5,
            )

    # Filter stable periods (shared with scipy path)
    logging.info("Filtering for stable thermal equilibrium periods...")
    stable_periods = filter_stable_periods(df)
    if len(stable_periods) < 20:
        logging.error(
            "❌ Insufficient stable periods: %d (need at least 20)",
            len(stable_periods),
        )
        return None
    logging.info("✅ Found %d stable periods for calibration", len(stable_periods))

    # --- Step 1: HLC ---
    logging.info("Step 1: HLC calibration via calibrate_hlc()...")
    hlc_result = calibrate_hlc()
    if hlc_result.get("success"):
        hlc = hlc_result["hlc_kw_per_k"]
        logging.info("✅ HLC = %.5f kW/K (R²=%.3f, n=%d)",
                     hlc, hlc_result.get("r2", 0), hlc_result.get("n_periods", 0))
    else:
        logging.warning(
            "⚠️ HLC calibration failed (%s) — using %s value",
            hlc_result.get("message"),
            _fallback_source("heat_loss_coefficient"),
        )
        hlc = _state_fallback("heat_loss_coefficient")

    # --- Step 2: OE ---
    logging.info("Step 2: Analytical OE estimation...")
    oe = _calibrate_oe_analytical(stable_periods, hlc)
    if oe is None:
        logging.warning("⚠️ OE calibration failed — using %s value",
                        _fallback_source("outlet_effectiveness"))
        oe = _state_fallback("outlet_effectiveness")

    # --- Step 3: τ_room (thermal time constant) ---
    # Priority: (1) transient calibration from heating sequences (most data),
    #           (2) cooling curves (HP-off periods ≥ 2h, often unavailable),
    #           (3) persisted/default fallback.
    logging.info("Step 3: Thermal time constant calibration...")
    tau = None
    tau_r2 = None

    # Primary: transient calibration (heating sequences, scipy L-BFGS-B)
    logging.info("  3a: Trying transient calibration (multi-step heating sequences)...")
    transient_seqs = filter_transient_periods(df)
    if transient_seqs and len(transient_seqs) >= 12:
        # Build a temporary model with current HLC/OE for tau optimization
        _tau_model = ThermalEquilibriumModel()
        _tau_model.heat_loss_coefficient = hlc
        _tau_model.outlet_effectiveness = oe
        tau_transient = calibrate_transient_parameters(_tau_model, transient_seqs)
        if tau_transient is not None:
            tau = tau_transient
            logging.info("✅ τ_room = %.2fh (transient calibration, %d sequences)",
                         tau, len(transient_seqs))
    else:
        logging.info("  Insufficient transient sequences: %d (need ≥12)",
                     len(transient_seqs) if transient_seqs else 0)

    # Fallback: cooling curves (HP-off periods)
    if tau is None:
        logging.info("  3b: Trying cooling time constant (HP-off periods)...")
        tau_cooling, tau_r2 = calculate_cooling_time_constant(df)
        if tau_cooling is not None:
            tau = tau_cooling
            logging.info("✅ τ_room = %.2fh (cooling curves, R²=%.3f)", tau, tau_r2)

    # Final fallback: persisted value
    if tau is None:
        logging.warning("⚠️ Both transient and cooling tau failed — using %s value",
                        _fallback_source("thermal_time_constant"))
        tau = _state_fallback("thermal_time_constant")

    # --- Step 4: pv_heat_weight ---
    logging.info("Step 4: PV heat weight (residual method)...")
    try:
        from .physics_calibration import _filter_pv_only_periods as _fpvo
    except ImportError:
        from physics_calibration import _filter_pv_only_periods as _fpvo  # type: ignore
    pv_periods = _fpvo(stable_periods, hlc=hlc, oe=oe)
    pv_weight = _residual_heat_source_weight(
        pv_periods, "pv", hlc, oe, min_periods=15, percentile=50.0
    )
    if pv_weight is None:
        logging.warning("⚠️ PV weight calibration failed — using %s value",
                        _fallback_source("pv_heat_weight"))
        pv_weight = _state_fallback("pv_heat_weight")

    # --- Step 5: fireplace_heat_weight ---
    logging.info("Step 5: Fireplace heat weight (residual method)...")
    fp_periods = _filter_fp_only_periods(stable_periods)
    fp_weight = _residual_heat_source_weight(
        fp_periods, "fp", hlc, oe, min_periods=5, percentile=50.0
    )
    if fp_weight is None:
        logging.warning("⚠️ Fireplace weight calibration failed — using %s value",
                        _fallback_source("fireplace_heat_weight"))
        fp_weight = _state_fallback("fireplace_heat_weight")

    # --- Step 6: tv_heat_weight ---
    logging.info("Step 6: TV heat weight (residual method)...")
    tv_periods = _filter_tv_only_periods(stable_periods)
    tv_weight = _residual_heat_source_weight(
        tv_periods, "tv", hlc, oe, min_periods=5, percentile=60.0
    )
    if tv_weight is None:
        logging.warning("⚠️ TV weight calibration failed — using %s value",
                        _fallback_source("tv_heat_weight"))
        tv_weight = _state_fallback("tv_heat_weight")

    # --- Step 7: solar_lag_minutes ---
    logging.info("Step 7: Solar lag via cross-correlation...")
    solar_lag = _calibrate_solar_lag_xcorr(df, hlc, oe)
    if solar_lag is None:
        logging.warning("⚠️ Solar lag calibration failed — using %s value",
                        _fallback_source("solar_lag_minutes"))
        solar_lag = _state_fallback("solar_lag_minutes")

    # --- Step 8: delta_t_floor ---
    logging.info("Step 8: delta_t_floor (P25 of outlet-inlet)...")
    delta_t_floor_val = calibrate_delta_t_floor(stable_periods)
    if delta_t_floor_val is None:
        logging.warning("⚠️ delta_t_floor calibration failed — using %s value",
                        _fallback_source("delta_t_floor"))
        delta_t_floor_val = _state_fallback("delta_t_floor")

    # --- Step 9: slab_time_constant_hours (grid search) ---
    # Pass the calibrated delta_t_floor so the slab model is not biased by
    # the config default when the actual value differs.
    logging.info("Step 9: Slab time constant (1-D grid search)...")
    slab_tau = _calibrate_slab_tau_grid_search(df, delta_t_floor=delta_t_floor_val)
    if slab_tau is None:
        logging.warning("⚠️ Slab tau grid search failed — using %s value",
                        _fallback_source("slab_time_constant_hours"))
        slab_tau = _state_fallback("slab_time_constant_hours")

    # --- Step 10: fp_decay_time_constant ---
    logging.info("Step 10: FP decay time constant (log-linear OLS)...")
    fp_decay_periods = filter_fp_decay_periods(df, hlc=hlc, oe=oe)
    fp_decay_tau = calibrate_fp_decay_tau(fp_decay_periods)
    if fp_decay_tau is None:
        logging.warning("⚠️ FP decay tau calibration failed — using %s value",
                        _fallback_source("fp_decay_time_constant"))
        fp_decay_tau = _state_fallback("fp_decay_time_constant")

    # --- Step 11: room_spread_delay_minutes ---
    logging.info("Step 11: Room spread delay (cross-correlation)...")
    fp_spread_periods = filter_fp_spread_periods(df)
    room_spread_delay = calibrate_room_spread_delay(fp_spread_periods)
    if room_spread_delay is None:
        logging.warning("⚠️ Room spread delay calibration failed — using %s value",
                        _fallback_source("room_spread_delay_minutes"))
        room_spread_delay = _state_fallback("room_spread_delay_minutes")

    # --- Step 12: cloud_factor_exponent (log-OLS, optional) ---
    cloud_exponent = _state_fallback("cloud_factor_exponent")
    if getattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False):
        logging.info("Step 12: Cloud factor exponent (log-OLS)...")
        cloud_exp = _calibrate_cloud_exponent_log_ols(df, pv_weight)
        if cloud_exp is not None:
            cloud_exponent = cloud_exp
        else:
            logging.warning(
                "⚠️ Cloud factor exponent calibration failed — using %s value",
                _fallback_source("cloud_factor_exponent"),
            )
    else:
        logging.info(
            "Step 12: Cloud factor exponent skipped "
            "(CLOUD_COVER_CORRECTION_ENABLED=false)"
        )

    # --- Step 13: solar_decay_tau_hours ---
    logging.info("Step 13: Solar decay tau (log-linear OLS)...")
    pv_decay_periods = filter_pv_decay_periods(df)
    solar_decay_tau = calibrate_solar_decay_tau(pv_decay_periods)
    if solar_decay_tau is None:
        logging.warning("⚠️ Solar decay tau calibration failed — using %s value",
                        _fallback_source("solar_decay_tau_hours"))
        solar_decay_tau = _state_fallback("solar_decay_tau_hours")

    # --- Assemble result ---
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
        "cloud_factor_exponent": cloud_exponent,
        "solar_decay_tau_hours": solar_decay_tau,
    }

    # --- Build the thermal model ---
    logging.info("Building thermal model with physics-direct parameters...")
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
    thermal_model.sync_heat_source_channels_from_model_state()

    # Apply channel-specific parameters to orchestrator channels
    channel_params: Dict[str, float] = {
        "delta_t_floor": delta_t_floor_val,
        "slab_time_constant_hours": slab_tau,
        "fp_decay_time_constant": fp_decay_tau,
        "room_spread_delay_minutes": room_spread_delay,
    }
    if cloud_exponent != ThermalParameterConfig.get_default("cloud_factor_exponent"):
        channel_params["cloud_factor_exponent"] = cloud_exponent
    if solar_decay_tau != ThermalParameterConfig.get_default("solar_decay_tau_hours"):
        channel_params["solar_decay_tau_hours"] = solar_decay_tau

    if thermal_model.orchestrator is not None and channel_params:
        _apply_channel_params(thermal_model.orchestrator, channel_params)

    # --- Log summary ---
    logging.info("\n=== PHYSICS-DIRECT CALIBRATED PARAMETERS ===")
    logging.info("  heat_loss_coefficient:  %.5f kW/K", hlc)
    logging.info("  outlet_effectiveness:   %.4f kW/K", oe)
    logging.info("  thermal_time_constant:  %.2f h", tau)
    logging.info("  pv_heat_weight:         %.6f kW/W", pv_weight)
    logging.info("  fireplace_heat_weight:  %.3f kW", fp_weight)
    logging.info("  tv_heat_weight:         %.3f kW", tv_weight)
    logging.info("  solar_lag_minutes:      %.1f min", solar_lag)
    logging.info("  delta_t_floor:          %.2f °C", delta_t_floor_val)
    logging.info("  slab_time_constant:     %.2f h", slab_tau)
    logging.info("  fp_decay_time_constant: %.2f h", fp_decay_tau)
    logging.info("  room_spread_delay:      %.0f min", room_spread_delay)
    logging.info("  cloud_factor_exponent:  %.3f", cloud_exponent)
    logging.info("  solar_decay_tau_hours:  %.2f h", solar_decay_tau)

    # --- Persist to unified thermal state ---
    logging.info("Saving physics-direct calibrated parameters to thermal state...")
    try:
        if state_manager is None:
            state_manager = get_thermal_state_manager()

        state_manager.set_calibrated_baseline(
            calibrated_params,
            calibration_cycles=len(stable_periods),
        )
        state_manager.update_learning_state(learning_confidence=3.0)
        thermal_model.sync_heat_source_channels_from_model_state(persist=True)

        logging.info(
            "✅ Physics-direct parameters saved to unified thermal state"
        )
        logging.info(
            "🔄 Restart ml_heating service to use calibrated thermal model"
        )
    except Exception as exc:
        logging.error("❌ Failed to save physics-direct parameters: %s", exc)
        try:
            save_state(
                thermal_learning_state={
                    "thermal_time_constant": tau,
                    "learning_confidence": 3.0,
                }
            )
            logging.warning("⚠️ Used fallback save method")
        except Exception as exc2:
            logging.error("❌ Fallback save also failed: %s", exc2)

    return thermal_model
