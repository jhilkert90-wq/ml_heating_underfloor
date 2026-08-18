"""
Predictive Overheating Prevention for Cooling Mode.

Runs a passive (HP OFF) trajectory simulation using PV and outdoor
temperature forecasts.  If the simulation predicts indoor temperature
will exceed the cooling target within the configured horizon, the
system activates the heat pump *proactively* — before the room
actually overheats.

This is critical for underfloor cooling where slab thermal inertia
(~3 h time constant) means reactive cooling starts too late.
"""

import logging
from typing import Any, Dict, Optional

try:
    from . import config
except ImportError:
    import config  # type: ignore

logger = logging.getLogger(__name__)


class OverheatingPredictor:
    """Forecast-driven overheating risk assessment for cooling mode.

    Each cycle, ``predict_overheating_risk`` simulates the next
    ``PRE_COOL_HORIZON_HOURS`` with the HP *off* (outlet = inlet).
    If the trajectory exceeds the cooling target the result tells
    the caller to activate cooling now.
    """

    def predict_overheating_risk(
        self,
        current_indoor: float,
        target_cooling: float,
        features: Dict[str, Any],
        thermal_model: Any,
        climate_mode: str = "cooling",
    ) -> Dict[str, Any]:
        """Run a passive trajectory and assess overheating risk.

        Args:
            current_indoor: Current indoor temperature [°C].
            target_cooling: Cooling target temperature [°C].
            features: Feature dict from ``build_physics_features``.
                Must contain ``inlet_temp``, ``outdoor_temp``,
                ``pv_forecast_*h``, ``temp_forecast_*h`` keys.
            thermal_model: ``ThermalEquilibriumModel`` instance
                (cooling-mode model).
            climate_mode: Must be ``"cooling"`` — safety gate.

        Returns:
            Dict with keys:
                risk (bool): Overheating is predicted.
                peak_temp (float): Maximum predicted indoor temp [°C].
                peak_hour (float): Hours from now until the peak.
                hours_until_peak (float): Same as ``peak_hour``.
                should_cool_now (bool): Pre-cooling should activate
                    this cycle.
                reason (str): Human-readable explanation.
        """
        no_risk = {
            "risk": False,
            "peak_temp": current_indoor,
            "peak_hour": 0.0,
            "hours_until_peak": 0.0,
            "should_cool_now": False,
            "reason": "",
        }

        # ── Safety gate: only in cooling mode ────────────────────────
        if climate_mode != "cooling":
            no_risk["reason"] = "not in cooling mode"
            return no_risk

        if not getattr(config, "PRE_COOL_ENABLED", True):
            no_risk["reason"] = "PRE_COOL_ENABLED=false"
            return no_risk

        # ── Collect forecast arrays ──────────────────────────────────
        horizon = int(getattr(config, "PRE_COOL_HORIZON_HOURS", 12))
        inlet_temp = features.get("inlet_temp")
        if inlet_temp is None:
            no_risk["reason"] = "no inlet_temp in features"
            return no_risk

        outdoor_temp = float(features.get("outdoor_temp", 20.0))

        # Build outdoor forecast array: [current, +1h, +2h, ...]
        outdoor_forecast = [outdoor_temp]
        for h in range(1, horizon + 1):
            val = features.get(f"temp_forecast_{h}h", outdoor_temp)
            outdoor_forecast.append(float(val) if val is not None else outdoor_temp)

        # Build PV forecast array: [current, +1h, +2h, ...]
        # Use thermal-corrected values consistently (pv_forecast_{h}h)
        # to match the pv_now anchor.
        pv_now = float(features.get("pv_now", 0.0))
        pv_forecast = [pv_now]
        for h in range(1, horizon + 1):
            val = features.get(
                f"pv_forecast_{h}h", 0.0,
            )
            pv_forecast.append(float(val) if val is not None else 0.0)

        # Build cloud cover: average from per-hour forecasts
        cloud_values = []
        for h in range(1, horizon + 1):
            cc = features.get(f"cloud_cover_forecast_{h}h")
            if cc is not None:
                cloud_values.append(float(cc))
        avg_cloud_cover = (
            sum(cloud_values) / len(cloud_values)
            if cloud_values
            else 50.0
        )

        # ── Reactive check: room already above target ────────────────
        # Must run BEFORE guards — if the room is overheated, cooling
        # is mandatory regardless of PV/outdoor conditions.
        if current_indoor > target_cooling:
            # Still run trajectory for peak_temp info, but bypass guards
            pass  # fall through to trajectory simulation
        else:
            # ── Guard: minimum PV + outdoor thresholds ───────────────
            total_pv = sum(pv_forecast)
            peak_outdoor = max(outdoor_forecast)
            min_pv = float(
                getattr(config, "PRE_COOL_MIN_PV_FORECAST_W", 1000.0)
            )
            min_outdoor = float(
                getattr(config, "PRE_COOL_MIN_OUTDOOR_FORECAST_C", 22.0)
            )

            if total_pv < min_pv and peak_outdoor < min_outdoor:
                no_risk["reason"] = (
                    f"guards not met: total_pv={total_pv:.0f}W "
                    f"< {min_pv:.0f}W AND peak_outdoor="
                    f"{peak_outdoor:.1f}°C < {min_outdoor:.1f}°C"
                )
                logger.debug("☀️ Pre-cool: %s", no_risk["reason"])
                return no_risk

        # Compute aggregate PV/outdoor for result dict
        total_pv = sum(pv_forecast)
        peak_outdoor = max(outdoor_forecast)
        max_pv_forecast = max(pv_forecast) if pv_forecast else 0.0

        # ── Run passive trajectory (HP OFF → outlet = inlet) ─────────
        try:
            # PV history for solar lag initialization
            pv_history = features.get("pv_power_history")
            pv_input = pv_history if pv_history else pv_now

            trajectory_result = thermal_model.predict_thermal_trajectory(
                current_indoor=current_indoor,
                target_indoor=target_cooling,
                outlet_temp=float(inlet_temp),  # HP OFF: outlet = inlet
                outdoor_temp=outdoor_forecast,
                time_horizon_hours=float(horizon),
                time_step_minutes=int(
                    getattr(config, "CYCLE_INTERVAL_MINUTES", 10)
                ),
                pv_power=pv_input,
                pv_forecasts=pv_forecast,
                fireplace_on=float(features.get("fireplace_on", 0.0)),
                tv_on=float(features.get("tv_on", 0.0)),
                cloud_cover_pct=avg_cloud_cover,
                inlet_temp=float(inlet_temp),
                delta_t_floor=0.0,  # HP OFF: no delta_t
                indoor_temp_delta_60m=float(
                    features.get("indoor_temp_delta_60m", 0.0)
                ),
                climate_mode="cooling",
                solar_contribution_cap_kw=float(
                    getattr(config, "PRE_COOL_PASSIVE_SOLAR_CAP_KW", 1.5)
                ),
            )
        except Exception as exc:
            logger.warning("Pre-cool trajectory simulation failed: %s", exc)
            no_risk["reason"] = f"trajectory failed: {exc}"
            return no_risk

        if (
            not trajectory_result
            or "trajectory" not in trajectory_result
            or not trajectory_result["trajectory"]
        ):
            no_risk["reason"] = "trajectory returned empty result"
            return no_risk

        # ── Analyze trajectory ───────────────────────────────────────
        trajectory = trajectory_result["trajectory"]
        times = trajectory_result.get("times", [])

        peak_temp = max(trajectory)
        peak_idx = trajectory.index(peak_temp)
        peak_hour = times[peak_idx] if peak_idx < len(times) else 0.0

        plausible_peak_limit = self._compute_plausible_peak_limit(
            current_indoor=current_indoor,
            target_cooling=target_cooling,
            peak_outdoor=peak_outdoor,
            max_pv_forecast=max_pv_forecast,
        )
        if (
            current_indoor <= target_cooling
            and peak_temp > plausible_peak_limit
        ):
            no_risk.update(
                {
                    "peak_temp": peak_temp,
                    "peak_hour": peak_hour,
                    "hours_until_peak": peak_hour,
                }
            )
            no_risk["reason"] = (
                f"plausibility guard: predicted peak {peak_temp:.1f}°C "
                f"exceeds passive limit {plausible_peak_limit:.1f}°C"
            )
            logger.info("❄️ Pre-cool blocked: %s", no_risk["reason"])
            return no_risk

        trigger_margin = float(
            getattr(config, "PRE_COOL_TRIGGER_MARGIN_K", 0.5)
        )
        trigger_threshold = target_cooling + trigger_margin
        risk = peak_temp > trigger_threshold

        # ── Determine if cooling should start NOW ────────────────────
        lead_time = float(
            getattr(config, "PRE_COOL_LEAD_TIME_HOURS", 3.0)
        )
        should_cool_now = False
        reason_parts = []

        if current_indoor > target_cooling:
            # Room already above target → always cool (reactive)
            should_cool_now = True
            reason_parts.append(
                f"room {current_indoor:.1f}°C > target {target_cooling:.1f}°C"
            )
        elif risk and peak_hour <= lead_time:
            # Peak is within lead time → start pre-cooling
            should_cool_now = True
            reason_parts.append(
                f"predicted peak {peak_temp:.1f}°C in {peak_hour:.1f}h "
                f"(> {trigger_threshold:.1f}°C, within {lead_time:.0f}h lead)"
            )
        elif risk:
            reason_parts.append(
                f"predicted peak {peak_temp:.1f}°C in {peak_hour:.1f}h "
                f"(> {trigger_threshold:.1f}°C, but {peak_hour:.1f}h > "
                f"{lead_time:.0f}h lead — waiting)"
            )
        else:
            reason_parts.append(
                f"peak {peak_temp:.1f}°C ≤ threshold {trigger_threshold:.1f}°C"
            )

        reason = "; ".join(reason_parts)

        result = {
            "risk": risk,
            "peak_temp": peak_temp,
            "peak_hour": peak_hour,
            "hours_until_peak": peak_hour,
            "should_cool_now": should_cool_now,
            "reason": reason,
            "trajectory": trajectory,
            "trigger_threshold": trigger_threshold,
            "peak_outdoor": peak_outdoor,
            "total_pv_forecast": total_pv,
        }

        if should_cool_now:
            logger.info(
                "❄️ PRE-COOL ACTIVATED: %s | peak_outdoor=%.1f°C, "
                "total_pv=%.0fW",
                reason, peak_outdoor, total_pv,
            )
        elif risk:
            logger.info(
                "❄️ Pre-cool risk detected (waiting): %s", reason
            )
        else:
            logger.debug("❄️ Pre-cool: no risk — %s", reason)

        return result

    def _compute_plausible_peak_limit(
        self,
        *,
        current_indoor: float,
        target_cooling: float,
        peak_outdoor: float,
        max_pv_forecast: float,
    ) -> float:
        """Return an upper bound for passive indoor peaks."""
        base_limit = max(current_indoor, target_cooling, peak_outdoor)
        outdoor_allowance = float(
            getattr(config, "PRE_COOL_MAX_PEAK_ABOVE_OUTDOOR_K", 2.0)
        )
        pv_allowance_per_kw = float(
            getattr(config, "PRE_COOL_PEAK_ALLOWANCE_PER_KW_PV", 0.4)
        )
        pv_allowance_cap = float(
            getattr(config, "PRE_COOL_MAX_PV_PEAK_ALLOWANCE_K", 1.0)
        )
        pv_allowance = min(
            pv_allowance_cap,
            max(0.0, max_pv_forecast) / 1000.0 * pv_allowance_per_kw,
        )
        return base_limit + outdoor_allowance + pv_allowance
