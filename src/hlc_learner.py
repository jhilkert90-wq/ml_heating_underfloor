"""
HLC Learner — Online Heat Loss Coefficient Estimation

This module provides a lightweight, in-memory estimator for the building's
Heat Loss Coefficient (HLC) from validated live-cycle data.

Concept
-------
At thermal equilibrium and with only the heat pump running, the steady-state
energy balance simplifies to:

    Q_hp ≈ HLC × (T_indoor − T_outdoor)

where Q_hp is the heat pump thermal power [kW] and HLC is the building heat
loss coefficient [kW/K].

The learner accumulates 60-minute windows of cycle data and validates each
window for:
  - HP-only conditions (no fireplace, TV, significant PV)
  - No blocking states (DHW, defrost, disinfection, boost heater)
  - Thermal equilibrium (indoor temp stable, near target)
  - Sufficient outdoor-indoor ΔT (real heating demand)

A forced-through-origin ordinary least squares (OLS) regression over all
validated windows yields the HLC estimate:

    HLC = Σ(Q_i · ΔT_i) / Σ(ΔT_i²)

When enough validated windows exist the estimate can be applied to the
unified thermal state as an updated calibrated baseline value.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from . import config
except ImportError:
    import config  # type: ignore

try:
    from .unified_thermal_state import get_thermal_state_manager
except ImportError:
    from unified_thermal_state import get_thermal_state_manager  # type: ignore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HLCCycle:
    """A single 5-minute control cycle snapshot used for HLC learning."""
    timestamp: datetime
    thermal_power_kw: float       # HP thermal power [kW]
    indoor_temp: float            # Current indoor temperature [°C]
    outdoor_temp: float           # Current outdoor temperature [°C]
    target_temp: float            # Target indoor temperature [°C]
    indoor_temp_delta_60m: float  # 60-min indoor temp change [K]
    pv_now_electrical: float      # Raw (uncorrected) PV power [W]
    fireplace_on: float           # 0.0 / 1.0
    tv_on: float                  # 0.0 / 1.0
    dhw_heating: float            # 0.0 / 1.0
    defrosting: float             # 0.0 / 1.0
    dhw_boost_heater: float       # 0.0 / 1.0
    is_blocking: bool             # combined blocking flag

    @property
    def delta_t(self) -> float:
        """Indoor − Outdoor ΔT [K]."""
        return self.indoor_temp - self.outdoor_temp


@dataclass
class HLCWindow:
    """A validated 60-minute window used in the regression."""
    start_time: datetime
    end_time: datetime
    mean_thermal_power_kw: float
    mean_delta_t: float           # mean (T_indoor − T_outdoor)
    n_cycles: int
    outdoor_temp_mean: float
    indoor_temp_mean: float


# ---------------------------------------------------------------------------
# HLCLearner
# ---------------------------------------------------------------------------

class HLCLearner:
    """
    Online estimator for the building's Heat Loss Coefficient.

    Usage
    -----
    Instantiate once and call :meth:`push_cycle` every main control cycle.
    Windows are assembled automatically.  When the returned dict from
    ``push_cycle`` contains ``"window_validated": True`` a new window has
    been appended to the regression set.

    Call :meth:`estimate_hlc` at any time to obtain the current OLS estimate
    and call :meth:`apply_to_thermal_state` to persist it.

    Parameters are read from :mod:`src.config` at call time so that the
    usual env-var override mechanism works.
    """

    def __init__(self) -> None:
        self._current_window: List[HLCCycle] = []
        self._window_start: Optional[datetime] = None
        # Rolling store of validated windows (capped at HLC_MAX_WINDOWS)
        self._validated_windows: deque[HLCWindow] = deque()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_cycle(self, context: Dict) -> Dict:
        """Ingest one control-cycle snapshot.

        Parameters
        ----------
        context : dict
            Must contain at minimum the keys listed below.  Missing keys
            default to safe sentinel values so that the window is rejected.

        Required keys
        -------------
        timestamp, thermal_power_kw, indoor_temp, outdoor_temp, target_temp,
        indoor_temp_delta_60m, pv_now_electrical, fireplace_on, tv_on,
        dhw_heating, defrosting, dhw_boost_heater, is_blocking

        Returns
        -------
        dict with keys:
          - ``"window_complete"`` (bool): True when the window duration has
            elapsed, regardless of validation result.
          - ``"window_validated"`` (bool): True when the completed window
            passed all quality gates.
          - ``"reject_reason"`` (str | None): Human-readable rejection reason.
          - ``"validated_windows"`` (int): Total number of validated windows.
        """
        cycle = self._build_cycle(context)
        if cycle is None:
            return {
                "window_complete": False,
                "window_validated": False,
                "reject_reason": "missing required cycle data",
                "validated_windows": len(self._validated_windows),
            }

        if self._window_start is None:
            self._window_start = cycle.timestamp

        self._current_window.append(cycle)

        elapsed_minutes = (
            (cycle.timestamp - self._window_start).total_seconds() / 60.0
        )
        if elapsed_minutes < config.HLC_WINDOW_MINUTES:
            return {
                "window_complete": False,
                "window_validated": False,
                "reject_reason": None,
                "validated_windows": len(self._validated_windows),
            }

        # Window is complete — validate and reset
        window_cycles = list(self._current_window)
        window_start = self._window_start
        self._current_window.clear()
        self._window_start = None

        validated, reject_reason, window = self._validate_window(
            window_cycles, window_start, config
        )

        if validated and window is not None:
            self._validated_windows.append(window)
            while len(self._validated_windows) > config.HLC_MAX_WINDOWS:
                self._validated_windows.popleft()

        return {
            "window_complete": True,
            "window_validated": validated,
            "reject_reason": reject_reason,
            "validated_windows": len(self._validated_windows),
        }

    def estimate_hlc(self) -> Tuple[Optional[float], Dict]:
        """Run OLS regression and return the HLC estimate.

        Returns
        -------
        (hlc_estimate, stats_dict)
            ``hlc_estimate`` is ``None`` when fewer validated windows exist
            than ``config.HLC_MIN_WINDOWS``.

        ``stats_dict`` keys: n_windows, sum_qdt, sum_dt2, r2, mean_residual
        """
        windows = list(self._validated_windows)
        n = len(windows)
        stats: Dict = {"n_windows": n}

        if n < config.HLC_MIN_WINDOWS:
            stats["reject_reason"] = (
                f"only {n} validated windows, need {config.HLC_MIN_WINDOWS}"
            )
            return None, stats

        qs = [w.mean_thermal_power_kw for w in windows]
        dts = [w.mean_delta_t for w in windows]

        sum_qdt = sum(q * dt for q, dt in zip(qs, dts))
        sum_dt2 = sum(dt * dt for dt in dts)

        if sum_dt2 < 1e-6:
            stats["reject_reason"] = "degenerate: ΔT variance too small"
            return None, stats

        hlc = sum_qdt / sum_dt2

        # Coefficient of determination (forced-origin model)
        ss_res = sum((q - hlc * dt) ** 2 for q, dt in zip(qs, dts))
        ss_tot = sum((q - (sum(qs) / n)) ** 2 for q in qs)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

        mean_residual = (sum(q - hlc * dt for q, dt in zip(qs, dts)) / n)

        stats.update(
            {
                "sum_qdt": round(sum_qdt, 6),
                "sum_dt2": round(sum_dt2, 6),
                "r2": round(r2, 4),
                "mean_residual": round(mean_residual, 4),
                "hlc_kw_per_k": round(hlc, 5),
            }
        )
        return hlc, stats

    def apply_to_thermal_state(
        self, thermal_state_manager=None
    ) -> Tuple[bool, str]:
        """Estimate HLC and apply it to the unified thermal state baseline.

        Parameters
        ----------
        thermal_state_manager : ThermalStateManager, optional
            If omitted the singleton from :func:`get_thermal_state_manager`
            is used.

        Returns
        -------
        (success, message)
        """
        hlc_estimate, stats = self.estimate_hlc()
        if hlc_estimate is None:
            reason = stats.get("reject_reason", "unknown")
            return False, f"HLC estimation rejected: {reason}"

        if thermal_state_manager is None:
            thermal_state_manager = get_thermal_state_manager()

        current_params = thermal_state_manager.get_computed_parameters()
        current_hlc = current_params.get("heat_loss_coefficient", hlc_estimate)

        # Guard: cap maximum relative change per calibration run
        if current_hlc > 0:
            relative_change = abs(hlc_estimate - current_hlc) / current_hlc
            if relative_change > config.HLC_MAX_UPDATE_FRACTION:
                sign = 1.0 if hlc_estimate > current_hlc else -1.0
                capped = current_hlc * (1.0 + sign * config.HLC_MAX_UPDATE_FRACTION)
                logger.warning(
                    "HLC estimate %.5f kW/K would change current value "
                    "%.5f kW/K by %.1f%% — capping at %.5f kW/K",
                    hlc_estimate, current_hlc,
                    relative_change * 100, capped,
                )
                hlc_estimate = capped

        n_windows = stats["n_windows"]
        thermal_state_manager.set_calibrated_baseline(
            {"heat_loss_coefficient": hlc_estimate},
            calibration_cycles=n_windows,
        )
        msg = (
            f"HLC updated to {hlc_estimate:.5f} kW/K "
            f"(R²={stats.get('r2', 0):.3f}, n={n_windows})"
        )
        logger.info("✅ %s", msg)
        return True, msg

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def validated_window_count(self) -> int:
        """Number of validated windows currently stored."""
        return len(self._validated_windows)

    @property
    def current_window_cycle_count(self) -> int:
        """Number of cycles in the window currently being assembled."""
        return len(self._current_window)

    def get_validated_windows(self) -> List[HLCWindow]:
        """Return a copy of the validated window list."""
        return list(self._validated_windows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cycle(context: Dict) -> Optional[HLCCycle]:
        """Build an :class:`HLCCycle` from a raw context dict.

        Returns ``None`` if any required numeric field is missing or
        the thermal power is ``None``.
        """
        required = (
            "thermal_power_kw", "indoor_temp", "outdoor_temp",
            "target_temp",
        )
        for key in required:
            if context.get(key) is None:
                return None

        try:
            return HLCCycle(
                timestamp=context.get("timestamp", datetime.now()),
                thermal_power_kw=float(context["thermal_power_kw"]),
                indoor_temp=float(context["indoor_temp"]),
                outdoor_temp=float(context["outdoor_temp"]),
                target_temp=float(context["target_temp"]),
                indoor_temp_delta_60m=float(
                    context.get("indoor_temp_delta_60m", 0.0)
                ),
                pv_now_electrical=float(
                    context.get("pv_now_electrical", 0.0)
                ),
                fireplace_on=float(context.get("fireplace_on", 0.0)),
                tv_on=float(context.get("tv_on", 0.0)),
                dhw_heating=float(context.get("dhw_heating", 0.0)),
                defrosting=float(context.get("defrosting", 0.0)),
                dhw_boost_heater=float(context.get("dhw_boost_heater", 0.0)),
                is_blocking=bool(context.get("is_blocking", False)),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _validate_window(
        cycles: List[HLCCycle],
        window_start: datetime,
        config,
    ) -> Tuple[bool, Optional[str], Optional[HLCWindow]]:
        """Validate a completed window and return (ok, reason, window).

        Returns
        -------
        (True, None, HLCWindow) on success.
        (False, reason_str, None) on any validation failure.
        """
        if not cycles:
            return False, "empty window", None

        window_end = cycles[-1].timestamp

        # --- 1. Minimum cycle count ---
        n_total = len(cycles)
        valid_power = [
            c for c in cycles if c.thermal_power_kw > 0
        ]
        n_valid = len(valid_power)
        min_frac = getattr(config, "HLC_CYCLES_PER_WINDOW_MIN_FRAC", 0.8)
        if n_valid < max(1, int(n_total * min_frac)):
            return (
                False,
                f"only {n_valid}/{n_total} cycles have positive thermal_power_kw"
                f" (need ≥{min_frac:.0%})",
                None,
            )

        # Use only cycles with positive thermal power for statistics
        active = valid_power

        # --- 2. No external heat sources ---
        if any(c.fireplace_on > 0.5 for c in active):
            return False, "fireplace active in window", None
        if any(c.tv_on > 0.5 for c in active):
            return False, "TV active in window", None

        # --- 3. No blocking states ---
        if any(c.dhw_heating > 0.5 for c in active):
            return False, "DHW heating active in window", None
        if any(c.defrosting > 0.5 for c in active):
            return False, "defrost active in window", None
        if any(c.dhw_boost_heater > 0.5 for c in active):
            return False, "DHW boost heater active in window", None
        if any(c.is_blocking for c in active):
            return False, "blocking state active in window", None

        # --- 4. PV contribution must be negligible ---
        pv_max = getattr(config, "HLC_PV_MAX_W", 50.0)
        mean_pv = sum(c.pv_now_electrical for c in active) / len(active)
        if mean_pv > pv_max:
            return (
                False,
                f"mean PV {mean_pv:.0f} W exceeds threshold {pv_max:.0f} W",
                None,
            )

        # --- 5. Equilibrium check: indoor close to target ---
        mean_indoor = sum(c.indoor_temp for c in active) / len(active)
        mean_target = sum(c.target_temp for c in active) / len(active)
        max_delta = getattr(config, "HLC_MAX_INDOOR_DELTA", 0.3)
        if abs(mean_indoor - mean_target) > max_delta:
            return (
                False,
                f"|indoor {mean_indoor:.2f} − target {mean_target:.2f}| "
                f"= {abs(mean_indoor - mean_target):.3f} K > {max_delta} K",
                None,
            )

        # --- 6. Stability check: slow temperature trend ---
        max_trend = getattr(config, "HLC_MAX_TREND", 0.2)
        last_delta = abs(active[-1].indoor_temp_delta_60m)
        if last_delta > max_trend:
            return (
                False,
                f"indoor_temp_delta_60m {last_delta:.3f} K > {max_trend} K",
                None,
            )

        # --- 7. Outdoor temperature range ---
        mean_outdoor = sum(c.outdoor_temp for c in active) / len(active)
        t_min = getattr(config, "HLC_OUTDOOR_TEMP_MIN", -10.0)
        t_max = getattr(config, "HLC_OUTDOOR_TEMP_MAX", 15.0)
        if not (t_min <= mean_outdoor <= t_max):
            return (
                False,
                f"mean outdoor temp {mean_outdoor:.1f} °C outside range "
                f"[{t_min}, {t_max}] °C",
                None,
            )

        # --- 8. Minimum heating demand ---
        min_demand = getattr(config, "HLC_MIN_HEATING_DEMAND_K", 1.0)
        if mean_target - mean_outdoor < min_demand:
            return (
                False,
                f"T_target−T_outdoor = {mean_target - mean_outdoor:.2f} K "
                f"< {min_demand} K — not enough heating demand",
                None,
            )

        # --- 9. Compute window statistics ---
        mean_q = sum(c.thermal_power_kw for c in active) / len(active)
        mean_dt = sum(c.delta_t for c in active) / len(active)

        if mean_dt <= 0:
            return False, f"mean ΔT {mean_dt:.2f} K is not positive", None

        window = HLCWindow(
            start_time=window_start,
            end_time=window_end,
            mean_thermal_power_kw=round(mean_q, 4),
            mean_delta_t=round(mean_dt, 4),
            n_cycles=len(active),
            outdoor_temp_mean=round(mean_outdoor, 2),
            indoor_temp_mean=round(mean_indoor, 2),
        )
        return True, None, window
