"""
HLC Calibration — Historical Heat Loss Coefficient Estimation

This module provides a historical calibration function that bootstraps the
building's Heat Loss Coefficient (HLC) from InfluxDB / HA history.

Concept
-------
At thermal equilibrium and with only the heat pump running, the steady-state
energy balance simplifies to:

    Q_hp ≈ HLC × (T_indoor − T_outdoor)

where Q_hp is the heat pump thermal power [kW] and HLC is the building heat
loss coefficient [kW/K].

The :func:`calibrate_hlc` function fetches historical sensor data from
InfluxDB (or HA), filters stable HP-only periods, and runs forced-through-origin
OLS regression to estimate HLC.  Calibration is triggered automatically on
startup when the HLC calibration flag file is present (written by ``main.py``).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from . import config
except ImportError:
    import config  # type: ignore

try:
    from .unified_thermal_state import get_thermal_state_manager
except ImportError:
    from unified_thermal_state import get_thermal_state_manager  # type: ignore

try:
    from .physics_calibration import fetch_historical_data_for_calibration
except ImportError:
    from physics_calibration import fetch_historical_data_for_calibration  # type: ignore



# ---------------------------------------------------------------------------
# Historical HLC Calibration
# ---------------------------------------------------------------------------

def calibrate_hlc(influx_service=None) -> Dict:
    """Calibrate HLC from historical sensor data (InfluxDB / HA).

    Fetches historical data for the configured lookback period, filters
    stable HP-only periods (same quality gates as the session learner),
    calculates thermal_power_kw from inlet/outlet/flow_rate, and runs
    forced-through-origin OLS regression to estimate HLC.

    The result is saved to the unified thermal state as a calibrated
    baseline value.

    Data fetching delegates to
    :func:`physics_calibration.fetch_historical_data_for_calibration`,
    which respects ``TRAINING_DATA_SOURCE`` ("influx", "ha_history",
    "auto") and performs HA history fallback/supplement in auto mode —
    identical to the strategy used for model calibration.

    Parameters
    ----------
    influx_service : InfluxService, optional
        Accepted for backward compatibility but no longer used.
        Data sourcing is handled by
        :func:`fetch_historical_data_for_calibration`.

    Returns
    -------
    dict
        Diagnostic results with keys: ``success``, ``hlc_kw_per_k``,
        ``r2``, ``n_periods``, ``date_range``, ``message``.
    """
    lookback_hours = getattr(config, "HLC_CALIBRATION_LOOKBACK_HOURS", 720)
    min_periods = getattr(config, "HLC_CALIBRATION_MIN_PERIODS", 20)
    window_size = getattr(config, "HLC_WINDOW_SIZE_ROWS", 12)
    # Guard against misconfiguration: window_size must be >= 1 to avoid
    # a ZeroDivisionError / ValueError from range(..., step=0).
    if window_size < 1:
        msg = (
            f"HLC_WINDOW_SIZE_ROWS must be >= 1, got {window_size}. "
            "Resetting to default (12 rows = 60 min)."
        )
        logger.warning("⚠️ HLC calibration: %s", msg)
        window_size = 12
        config.HLC_WINDOW_SIZE_ROWS = 12  # keep config consistent
    elif window_size < 4:
        # Values 1–3 (5–15 min) are technically valid but far too small to
        # approximate thermal equilibrium for typical buildings.  Log a
        # prominent warning so the user knows calibration quality may suffer.
        logger.warning(
            "⚠️ HLC calibration: HLC_WINDOW_SIZE_ROWS=%d (< 4 rows / 20 min) "
            "is very small. Thermal equilibrium approximation may be poor. "
            "Consider raising to at least 12 (60 min).",
            window_size,
        )
    min_flow = getattr(config, "HLC_MIN_FLOW_RATE_LPM", 0.5)
    min_thermal_power = getattr(config, "HEATING_MIN_THERMAL_POWER_KW", 0.5)
    use_intercept = getattr(config, "HLC_REGRESSION_INTERCEPT", False)

    logger.info(
        "🔬 HLC calibration: fetching %d hours of historical data...",
        lookback_hours,
    )

    # --- Fetch historical data ---
    # Delegate to the shared helper used by model calibration, which respects
    # TRAINING_DATA_SOURCE and performs HA history fallback/supplement in auto
    # mode — the same data-source strategy as physics_calibration.
    try:
        df = fetch_historical_data_for_calibration(lookback_hours=lookback_hours)
    except Exception as exc:
        msg = f"Failed to fetch historical data: {exc}"
        logger.error("❌ HLC calibration: %s", msg)
        return {"success": False, "message": msg}

    if df is None or df.empty:
        msg = "No historical data available for HLC calibration"
        logger.warning("⚠️ HLC calibration: %s", msg)
        return {"success": False, "message": msg}

    # --- Build column map from config entity IDs ---
    # Use the same short-name convention as physics_calibration and
    # influx_service.get_training_data(): entity_id.split(".", 1)[-1].
    # This correctly resolves non-English entity IDs (e.g. "rt_mittelwert")
    # without relying on keyword guessing.
    required_cols = {
        "indoor_temp", "outdoor_temp", "outlet_temp", "inlet_temp",
        "flow_rate",
    }
    col_map: Dict[str, str] = {}

    def _add_col(key: str, entity_attr: str) -> None:
        col_name = getattr(config, entity_attr, "").split(".", 1)[-1]
        if col_name and col_name in df.columns:
            col_map[key] = col_name

    _add_col("indoor_temp", "INDOOR_TEMP_ENTITY_ID")
    _add_col("outdoor_temp", "OUTDOOR_TEMP_ENTITY_ID")
    _add_col("outlet_temp", "ACTUAL_OUTLET_TEMP_ENTITY_ID")
    _add_col("inlet_temp", "INLET_TEMP_ENTITY_ID")
    _add_col("flow_rate", "FLOW_RATE_ENTITY_ID")
    _add_col("pv_power", "PV_POWER_ENTITY_ID")
    _add_col("fireplace", "FIREPLACE_STATUS_ENTITY_ID")
    _add_col("tv", "TV_STATUS_ENTITY_ID")
    _add_col("dhw", "DHW_STATUS_ENTITY_ID")
    _add_col("defrost", "DEFROST_STATUS_ENTITY_ID")
    _add_col("target_temp", "TARGET_INDOOR_TEMP_ENTITY_ID")

    missing = required_cols - set(col_map.keys())
    if missing:
        msg = f"Missing required columns in historical data: {missing}"
        logger.error("❌ HLC calibration: %s", msg)
        return {"success": False, "message": msg}

    # Fix 3 — When the target_temp column is absent, synthesise it from
    # the configured default target temperature.  This keeps the
    # indoor_far_from_target and low_heating_demand quality gates active
    # which materially improves HLC regression quality.
    if "target_temp" not in col_map:
        default_target = float(
            getattr(config, "HLC_DEFAULT_TARGET_TEMP", 22.6)
        )
        synth_col = "_hlc_synth_target_temp"
        df[synth_col] = default_target
        col_map["target_temp"] = synth_col
        logger.info(
            "ℹ️ HLC calibration: target_temp column not available — "
            "using default target temperature %.1f°C for quality gates.",
            default_target,
        )

    # --- Calculate thermal power and filter stable periods ---
    specific_heat = getattr(config, "SPECIFIC_HEAT_CAPACITY", 4.186)
    pv_max = getattr(config, "HLC_PV_MAX_W", 50.0)
    outdoor_min = getattr(config, "HLC_OUTDOOR_TEMP_MIN", -10.0)
    outdoor_max = getattr(config, "HLC_OUTDOOR_TEMP_MAX", 15.0)
    min_demand = getattr(config, "HLC_MIN_HEATING_DEMAND_K", 1.0)
    max_indoor_delta = getattr(config, "HLC_MAX_INDOOR_DELTA", 0.3)
    max_trend = getattr(config, "HLC_MAX_TREND", 0.2)

    periods_q = []  # thermal power per period [kW]
    periods_dt = []  # delta T per period [K]

    n_rows = len(df)
    rejected = {"total": 0, "reasons": {}}
    _has_time_col = "_time" in df.columns

    for start_idx in range(0, n_rows - window_size + 1, window_size):
        window = df.iloc[start_idx:start_idx + window_size]

        # Fix 4 — Reject windows that span a timestamp gap > 10 min.
        # After HA/InfluxDB concatenation the integer RangeIndex makes
        # consecutive rows look adjacent even when they are hours apart.
        # Checking the actual _time values catches these phantom windows.
        if _has_time_col:
            max_gap = window["_time"].diff().abs().max()
            if pd.notna(max_gap) and max_gap > pd.Timedelta("10min"):
                _reject(rejected, "time_gap_in_window")
                continue

        # Extract values
        try:
            outlet_vals = window[col_map["outlet_temp"]].dropna()
            inlet_vals = window[col_map["inlet_temp"]].dropna()
            flow_vals = window[col_map["flow_rate"]].dropna()
            indoor_vals = window[col_map["indoor_temp"]].dropna()
            outdoor_vals = window[col_map["outdoor_temp"]].dropna()
        except (KeyError, TypeError):
            continue

        if (len(outlet_vals) < 2 or len(inlet_vals) < 2
                or len(flow_vals) < 2 or len(indoor_vals) < 2
                or len(outdoor_vals) < 2):
            continue

        mean_outlet = outlet_vals.mean()
        mean_inlet = inlet_vals.mean()
        mean_flow = flow_vals.mean()
        mean_indoor = indoor_vals.mean()
        mean_outdoor = outdoor_vals.mean()

        # Fix 2 — Reject windows with insufficient flow before computing
        # thermal power.  This prevents forward-filled standby periods (where
        # the pump is off) from masquerading as active heating windows.
        if mean_flow < min_flow:
            _reject(rejected, "flow_too_low")
            continue

        # Thermal power: Q = (flow_rate / 60) × c_p × (outlet − inlet)
        delta_t_hp = mean_outlet - mean_inlet
        thermal_power_kw = (mean_flow / 60.0) * specific_heat * delta_t_hp

        # Fix 9 — Use a configurable minimum thermal power instead of a
        # simple > 0 check so that marginal / residual-heat windows are also
        # excluded.
        if thermal_power_kw < min_thermal_power:
            _reject(rejected, "thermal_power_too_low")
            continue

        # ΔT for HLC regression: indoor − outdoor
        delta_t = mean_indoor - mean_outdoor

        if delta_t <= 0:
            _reject(rejected, "negative_delta_t")
            continue

        # Outdoor range check
        if not (outdoor_min <= mean_outdoor <= outdoor_max):
            _reject(rejected, "outdoor_temp_range")
            continue

        # Target temp check (if available)
        if "target_temp" in col_map:
            target_vals = window[col_map["target_temp"]].dropna()
            if len(target_vals) >= 2:
                mean_target = target_vals.mean()
                if abs(mean_indoor - mean_target) > max_indoor_delta:
                    _reject(rejected, "indoor_far_from_target")
                    continue
                if mean_target - mean_outdoor < min_demand:
                    _reject(rejected, "low_heating_demand")
                    continue

        # Indoor stability: require < 0.3°C change within window
        indoor_range = indoor_vals.max() - indoor_vals.min()
        if indoor_range > max_indoor_delta * 2:
            _reject(rejected, "indoor_unstable")
            continue

        # Indoor trend check: first-to-last change within the window
        # must not exceed max_trend (same gate as session learner)
        indoor_trend = abs(indoor_vals.iloc[-1] - indoor_vals.iloc[0])
        if indoor_trend > max_trend:
            _reject(rejected, "indoor_trend_too_high")
            continue

        # PV check
        if "pv_power" in col_map:
            pv_vals = window[col_map["pv_power"]].dropna()
            if len(pv_vals) > 0 and pv_vals.mean() > pv_max:
                _reject(rejected, "pv_too_high")
                continue

        # Blocking checks
        for blocker_key in ("fireplace", "tv", "dhw", "defrost"):
            if blocker_key in col_map:
                blocker_vals = window[col_map[blocker_key]].dropna()
                if len(blocker_vals) > 0 and blocker_vals.max() > 0.5:
                    _reject(rejected, f"{blocker_key}_active")
                    break
        else:
            # All blockers passed — accept period
            periods_q.append(thermal_power_kw)
            periods_dt.append(delta_t)
            continue
        # Blocker was active — period already rejected by break above

    n_periods = len(periods_q)
    logger.info(
        "🔬 HLC calibration: %d valid periods from %d windows "
        "(rejected: %s)",
        n_periods, n_rows // window_size,
        {k: v for k, v in rejected.get("reasons", {}).items()},
    )

    if n_periods < min_periods:
        msg = (
            f"Only {n_periods} valid periods found, "
            f"need at least {min_periods}"
        )
        logger.warning("⚠️ HLC calibration: %s", msg)
        return {"success": False, "n_periods": n_periods, "message": msg}

    # --- OLS regression: HLC = Σ(Q × ΔT) / Σ(ΔT²) ---
    sum_qdt = sum(q * dt for q, dt in zip(periods_q, periods_dt))
    sum_dt2 = sum(dt * dt for dt in periods_dt)

    if sum_dt2 < 1e-6:
        msg = "Degenerate data: ΔT variance too small"
        logger.warning("⚠️ HLC calibration: %s", msg)
        return {"success": False, "message": msg}

    hlc = sum_qdt / sum_dt2

    # Sanity bounds — reject physically implausible values.
    # Typical residential HLC range is 0.03–1.0 kW/K; allow generous
    # bounds to cover unusual buildings but catch regression artefacts.
    HLC_MIN_PLAUSIBLE = 0.01   # kW/K
    HLC_MAX_PLAUSIBLE = 2.0    # kW/K
    if not (HLC_MIN_PLAUSIBLE <= hlc <= HLC_MAX_PLAUSIBLE):
        msg = (
            f"HLC estimate {hlc:.5f} kW/K outside plausible range "
            f"[{HLC_MIN_PLAUSIBLE}, {HLC_MAX_PLAUSIBLE}] — rejected"
        )
        logger.warning("⚠️ HLC calibration: %s", msg)
        return {"success": False, "hlc_kw_per_k": round(hlc, 5), "message": msg}

    # --- Fix 5 — Extended fit-quality diagnostics ---
    # Standard R² (relative to mean Q) — what was reported before.
    mean_q = sum(periods_q) / n_periods
    ss_res = sum((q - hlc * dt) ** 2 for q, dt in zip(periods_q, periods_dt))
    ss_tot = sum((q - mean_q) ** 2 for q in periods_q)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

    # FTO-R² (relative to zero) — appropriate measure for a forced-through-
    # origin model: fraction of variance in Q explained when the null is Q=0.
    ss_zero = sum(q * q for q in periods_q)
    r2_fto = 1.0 - ss_res / ss_zero if ss_zero > 1e-9 else 0.0

    # Pearson r between Q and ΔT — the most interpretable scatter indicator.
    mean_dt = sum(periods_dt) / n_periods
    cov_qdt = sum(
        (dt - mean_dt) * (q - mean_q)
        for dt, q in zip(periods_dt, periods_q)
    )
    std_dt = math.sqrt(sum((dt - mean_dt) ** 2 for dt in periods_dt))
    std_q = math.sqrt(sum((q - mean_q) ** 2 for q in periods_q))
    r_pearson = (
        cov_qdt / (std_dt * std_q)
        if (std_dt > 1e-9 and std_q > 1e-9)
        else 0.0
    )

    # Fix 1 — Use the _time column for date range (if present) so that the
    # logged range shows actual datetime strings instead of integer indices
    # (which appear after reset_index in fetch_historical_data_for_calibration).
    date_range = ""
    if "_time" in df.columns:
        try:
            date_range = f"{df['_time'].min()} — {df['_time'].max()}"
        except Exception:
            date_range = "unknown"
    elif hasattr(df.index, "min") and hasattr(df.index, "max"):
        try:
            date_range = f"{df.index.min()} — {df.index.max()}"
        except Exception:
            date_range = "unknown"

    logger.info(
        "✅ HLC calibration result: HLC = %.5f kW/K "
        "(R² = %.3f, FTO-R² = %.3f, r = %.3f, n = %d, range: %s)",
        hlc, r2, r2_fto, r_pearson, n_periods, date_range,
    )

    # Fix 8 — Optional with-intercept regression for contamination diagnosis.
    # Fits Q = HLC_i × ΔT + Q0; a large |Q0| flags non-zero baseline heat
    # (e.g. residual DHW heat, ffill-contaminated standby periods).
    if use_intercept and n_periods >= 3:
        n_i = n_periods
        sum_x = sum(periods_dt)
        sum_y = sum(periods_q)
        sum_xy = sum(x * y for x, y in zip(periods_dt, periods_q))
        sum_x2 = sum(x * x for x in periods_dt)
        denom_i = n_i * sum_x2 - sum_x ** 2
        if abs(denom_i) > 1e-9:
            hlc_intercept = (n_i * sum_xy - sum_x * sum_y) / denom_i
            q0 = (sum_y - hlc_intercept * sum_x) / n_i
            logger.info(
                "🔬 HLC with-intercept: HLC = %.5f kW/K, Q0 = %.4f kW "
                "(large |Q0| indicates contamination)",
                hlc_intercept, q0,
            )

    # --- Save to unified thermal state ---
    try:
        tsm = get_thermal_state_manager()
        tsm.set_calibrated_baseline(
            {"heat_loss_coefficient": hlc},
            calibration_cycles=n_periods,
        )
        logger.info("✅ HLC calibration: saved to unified thermal state")
    except Exception as exc:
        logger.error(
            "❌ HLC calibration: failed to save to thermal state — %s", exc
        )
        return {
            "success": False,
            "hlc_kw_per_k": round(hlc, 5),
            "r2": round(r2, 4),
            "r2_fto": round(r2_fto, 4),
            "r_pearson": round(r_pearson, 4),
            "n_periods": n_periods,
            "message": f"Calibration succeeded but save failed: {exc}",
        }

    return {
        "success": True,
        "hlc_kw_per_k": round(hlc, 5),
        "r2": round(r2, 4),
        "r2_fto": round(r2_fto, 4),
        "r_pearson": round(r_pearson, 4),
        "n_periods": n_periods,
        "date_range": date_range,
        "message": (
            f"HLC calibrated to {hlc:.5f} kW/K "
            f"(R²={r2:.3f}, FTO-R²={r2_fto:.3f}, r={r_pearson:.3f}, n={n_periods})"
        ),
    }


def _reject(rejected: Dict, reason: str) -> None:
    """Helper to track rejection counts."""
    rejected["total"] = rejected.get("total", 0) + 1
    reasons = rejected.setdefault("reasons", {})
    reasons[reason] = reasons.get(reason, 0) + 1
