"""Building characteristic curves derived from the learned thermal model.

Two curves are computed per climate mode, using only the learned physics
(heat loss coefficient and outlet effectiveness).  External heat sources
(PV, fireplace, TV) are deliberately ignored by the base curves:

* base outlet temperature [°C] needed to hold the target indoor
  temperature at a given outdoor temperature (raw, unclamped)::

      T_outlet = (T_target * (HLC + eff) - HLC * T_out) / eff

* building load [kW] = HLC [kW/K] * (T_target - T_out) for heating and
  HLC * (T_out - T_target) for cooling (raw, may be negative outside the
  mode's useful range).

Each curve is also approximated by a polynomial (degree 1-4).

The building-kW sensor also estimates the zero-HVAC outdoor balance
temperature from stable observations while hydronic flow and thermal power
are near zero.  It keeps up to 24 hours of observations and withholds the
estimate until at least three stable samples span two hours with low spread.
"""

from __future__ import annotations

import json
import logging
import math
import time
import warnings
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

HEATING_RANGE = (-20.0, 20.0)
COOLING_RANGE = (10.0, 40.0)
GRID_STEP = 1.0
MAX_ATTRIBUTE_BYTES = 14000  # HA recorder limit is 16384
REFRESH_SECONDS = 3600.0
OUTDOOR_REPUBLISH_DELTA = 0.5
BALANCE_ESTIMATE_WINDOW_SECONDS = 24 * 3600
BALANCE_ESTIMATE_MIN_SAMPLES = 3
BALANCE_ESTIMATE_MIN_WINDOW_SECONDS = 2 * 3600
BALANCE_ESTIMATE_MAX_INDOOR_DRIFT = 0.25
BALANCE_ESTIMATE_MAX_HVAC_POWER_KW = 0.1
BALANCE_ESTIMATE_MAX_FLOW_RATE = 0.1
BALANCE_ESTIMATE_MAX_GAIN_KW = 5.0
BALANCE_ESTIMATE_MAX_SPREAD_K = 2.0

BASE_OUTLET_ENTITY_ID = "sensor.ml_heating_base_outlet_curve"
BUILDING_KW_ENTITY_ID = "sensor.ml_heating_building_curve_kw"


def base_outlet_temp(
    outdoor_temp: float,
    target_temp: float,
    heat_loss_coefficient: float,
    outlet_effectiveness: float,
) -> Optional[float]:
    """Raw equilibrium outlet temperature without external heat sources."""
    if outlet_effectiveness <= 0:
        return None
    total = heat_loss_coefficient + outlet_effectiveness
    return (
        target_temp * total - heat_loss_coefficient * outdoor_temp
    ) / outlet_effectiveness


def building_load_kw(
    outdoor_temp: float,
    target_temp: float,
    heat_loss_coefficient: float,
    climate_mode: str = "heating",
) -> float:
    """Building load in kW (HLC x delta T); positive = demand in mode."""
    if climate_mode == "cooling":
        return heat_loss_coefficient * (outdoor_temp - target_temp)
    return heat_loss_coefficient * (target_temp - outdoor_temp)


def balance_outdoor_temp(
    target_temp: float, heat_loss_coefficient: float, gains_kw: float
) -> Optional[float]:
    """Estimate zero-HVAC outdoor temperature from non-HVAC gains."""
    try:
        target = float(target_temp)
        hlc = float(heat_loss_coefficient)
        gains = float(gains_kw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (target, hlc, gains)):
        return None
    if hlc <= 0 or gains < 0:
        return None
    return target - gains / hlc


class BalancePointEstimator:
    """Estimate average gains from stable observations while HVAC is off."""

    def __init__(self) -> None:
        self._samples: Dict[str, deque] = {
            "heating": deque(),
            "cooling": deque(),
        }

    def observe(
        self,
        mode: str,
        indoor_temp: float,
        outdoor_temp: float,
        heat_loss_coefficient: float,
        indoor_temp_delta_60m: float,
        thermal_power_kw: float,
        flow_rate: float,
        now: float,
    ) -> Optional[Dict[str, Any]]:
        """Add a conservative gain sample when conditions are suitable."""
        if mode not in self._samples:
            return None
        try:
            indoor, outdoor, hlc, drift, power, flow, timestamp = (
                float(value)
                for value in (
                    indoor_temp,
                    outdoor_temp,
                    heat_loss_coefficient,
                    indoor_temp_delta_60m,
                    thermal_power_kw,
                    flow_rate,
                    now,
                )
            )
        except (TypeError, ValueError):
            return None
        if not all(
            math.isfinite(value)
            for value in (indoor, outdoor, hlc, drift, power, flow, timestamp)
        ):
            return None
        if (
            hlc <= 0
            or abs(drift) > BALANCE_ESTIMATE_MAX_INDOOR_DRIFT
            or abs(power) > BALANCE_ESTIMATE_MAX_HVAC_POWER_KW
            or abs(flow) > BALANCE_ESTIMATE_MAX_FLOW_RATE
        ):
            return None

        indoor_outdoor_delta = indoor - outdoor
        gains_kw = hlc * indoor_outdoor_delta
        if not 0 <= gains_kw <= BALANCE_ESTIMATE_MAX_GAIN_KW:
            return None

        samples = self._samples[mode]
        cutoff = timestamp - BALANCE_ESTIMATE_WINDOW_SECONDS
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        if samples and timestamp <= samples[-1][0]:
            return self.get_estimate(mode, timestamp, hlc)
        samples.append((timestamp, indoor_outdoor_delta, hlc))
        return self.get_estimate(mode, timestamp, hlc)

    def get_estimate(
        self,
        mode: str,
        now: Optional[float] = None,
        heat_loss_coefficient: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return a stable mean gain estimate, or None until confidence gates pass."""
        if mode not in self._samples:
            return None
        timestamp = time.time() if now is None else float(now)
        samples = self._samples[mode]
        cutoff = timestamp - BALANCE_ESTIMATE_WINDOW_SECONDS
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        if (
            len(samples) < BALANCE_ESTIMATE_MIN_SAMPLES
            or samples[-1][0] - samples[0][0]
            < BALANCE_ESTIMATE_MIN_WINDOW_SECONDS
        ):
            return None
        try:
            hlc = float(
                heat_loss_coefficient
                if heat_loss_coefficient is not None
                else samples[-1][2]
            )
        except (TypeError, ValueError):
            return None
        if not math.isfinite(hlc) or hlc <= 0:
            return None
        mean_delta = sum(sample[1] for sample in samples) / len(samples)
        gains = mean_delta * hlc
        if not 0 <= gains <= BALANCE_ESTIMATE_MAX_GAIN_KW:
            return None
        spread_k = math.sqrt(
            sum((sample[1] - mean_delta) ** 2 for sample in samples)
            / len(samples)
        )
        if spread_k > BALANCE_ESTIMATE_MAX_SPREAD_K:
            return None
        return {
            "gain_kw": gains,
            "sample_count": len(samples),
            "window_minutes": int((samples[-1][0] - samples[0][0]) / 60),
            "uncertainty_k": spread_k,
        }


def outdoor_grid(climate_mode: str) -> np.ndarray:
    low, high = COOLING_RANGE if climate_mode == "cooling" else HEATING_RANGE
    return np.arange(low, high + GRID_STEP / 2, GRID_STEP)


def fit_polynomial(
    x: np.ndarray, y: np.ndarray, degree: int
) -> Dict[str, Any]:
    """Fit a polynomial and return coefficients (a4..a0) and fit errors."""
    degree = max(1, min(4, int(degree)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs = np.polyfit(x, y, degree)
    residual = y - np.polyval(coeffs, x)
    padded = [0.0] * (5 - len(coeffs)) + [float(c) for c in coeffs]
    return {
        "coeffs": padded,
        "degree": degree,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "max_error": float(np.max(np.abs(residual))),
    }


def _r(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _curve_attributes(
    prefix: str, x: np.ndarray, y: np.ndarray, degree: int
) -> Dict[str, Any]:
    fit = fit_polynomial(x, y, degree)
    attrs: Dict[str, Any] = {
        f"{prefix}_poly_coeffs": [float(f"{c:.8g}") for c in fit["coeffs"]],
        f"{prefix}_poly_degree": fit["degree"],
        f"{prefix}_fit_rmse": _r(fit["rmse"], 6),
        f"{prefix}_fit_max_error": _r(fit["max_error"], 6),
    }
    for power, coeff in zip(range(4, -1, -1), fit["coeffs"]):
        attrs[f"{prefix}_coef_x{power}"] = float(f"{coeff:.8g}")
    attrs[f"{prefix}_outdoor_temps"] = [_r(v, 1) for v in x]
    attrs[f"{prefix}_values"] = [_r(v, 2) for v in y]
    return attrs


def compute_curves(
    parameters_by_mode: Dict[str, Dict[str, Any]],
    targets: Dict[str, float],
    degree: int = 4,
    balance_estimates: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute attribute dicts for both sensors (all modes).

    ``targets`` maps climate mode -> active target indoor temperature.
    Returns ``{"outlet": {...}, "kw": {...}}``.
    """
    outlet_attrs: Dict[str, Any] = {}
    kw_attrs: Dict[str, Any] = {}
    for mode, target in targets.items():
        parameters = parameters_by_mode[mode]
        hlc = float(parameters["heat_loss_coefficient"])
        eff = float(parameters["outlet_effectiveness"])
        grid = outdoor_grid(mode)
        outlets = [base_outlet_temp(t, target, hlc, eff) for t in grid]
        if any(v is None for v in outlets):
            logging.debug("Base outlet curve skipped: no effectiveness")
        else:
            outlet_attrs[f"{mode}_target_indoor"] = _r(target, 2)
            outlet_attrs.update(
                _curve_attributes(mode, grid, np.array(outlets), degree)
            )
        loads = np.array(
            [building_load_kw(t, target, hlc, mode) for t in grid]
        )
        kw_attrs[f"{mode}_target_indoor"] = _r(target, 2)
        kw_attrs.update(_curve_attributes(mode, grid, loads, degree))
        # Retain the no-gains zero crossing separately from the empirical
        # balance estimate, which is only available after stable idle samples.
        kw_attrs[f"{mode}_no_gains_zero_load_outdoor_temp"] = _r(target, 2)
        estimate = (balance_estimates or {}).get(mode) or {}
        gain_kw = estimate.get("gain_kw")
        balance_temp = (
            balance_outdoor_temp(target, hlc, gain_kw)
            if gain_kw is not None
            else None
        )
        estimate_available = balance_temp is not None
        kw_attrs[f"{mode}_balance_outdoor_temp"] = (
            _r(balance_temp, 2) if estimate_available else None
        )
        kw_attrs[f"{mode}_balance_gain_kw"] = (
            _r(gain_kw, 3) if estimate_available else None
        )
        uncertainty = estimate.get("uncertainty_k")
        kw_attrs[f"{mode}_balance_uncertainty_k"] = (
            _r(uncertainty, 2)
            if estimate_available and uncertainty is not None
            else None
        )
        kw_attrs[f"{mode}_balance_estimate_status"] = (
            "available" if estimate_available else "unavailable"
        )
        kw_attrs[f"{mode}_balance_sample_count"] = (
            int(estimate.get("sample_count", 0)) if estimate_available else 0
        )
        kw_attrs[f"{mode}_balance_window_minutes"] = (
            int(estimate.get("window_minutes", 0)) if estimate_available else 0
        )
        kw_attrs[f"{mode}_balance_estimate_method"] = (
            "stable_hvac_off_observations; assumes negligible stored-heat drift"
            if estimate_available
            else "unavailable"
        )
        kw_attrs[f"{mode}_balance_min_sample_count"] = (
            BALANCE_ESTIMATE_MIN_SAMPLES
        )
        kw_attrs[f"{mode}_balance_min_window_minutes"] = int(
            BALANCE_ESTIMATE_MIN_WINDOW_SECONDS / 60
        )
        kw_attrs[f"{mode}_balance_max_indoor_drift_60m"] = (
            BALANCE_ESTIMATE_MAX_INDOOR_DRIFT
        )
        kw_attrs[f"{mode}_balance_max_hvac_power_kw"] = (
            BALANCE_ESTIMATE_MAX_HVAC_POWER_KW
        )
        kw_attrs[f"{mode}_balance_max_flow_rate"] = (
            BALANCE_ESTIMATE_MAX_FLOW_RATE
        )
        kw_attrs[f"{mode}_balance_max_inferred_gains_kw"] = (
            BALANCE_ESTIMATE_MAX_GAIN_KW
        )
        kw_attrs[f"{mode}_balance_max_uncertainty_k"] = (
            BALANCE_ESTIMATE_MAX_SPREAD_K
        )
        kw_attrs[f"{mode}_design_load_kw"] = _r(
            building_load_kw(float(grid[0] if mode == "heating" else grid[-1]),
                             target, hlc, mode), 3
        )
    return {"outlet": outlet_attrs, "kw": kw_attrs}


def _shrink_to_limit(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the raw point lists if attributes would exceed HA's limit."""
    if len(json.dumps(attrs, default=str)) <= MAX_ATTRIBUTE_BYTES:
        return attrs
    slim = {
        k: v for k, v in attrs.items()
        if not (k.endswith("_outdoor_temps") or k.endswith("_values"))
    }
    slim["curve_points_dropped"] = True
    return slim


def _parameter_signature(
    parameters_by_mode: Dict[str, Dict[str, Any]], targets: Dict[str, float]
):
    return (
        tuple(
            (
                mode,
                tuple(
                    sorted(
                        (k, round(float(v), 6))
                        for k, v in parameters.items()
                        if isinstance(v, (int, float))
                    )
                ),
                tuple(
                    sorted(
                        (str(k), round(float(v), 6))
                        for k, v in parameters.get(
                            "external_source_weights", {}
                        ).items()
                        if isinstance(v, (int, float))
                    )
                ),
            )
            for mode, parameters in sorted(parameters_by_mode.items())
        ),
        tuple(sorted((k, round(float(v), 2)) for k, v in targets.items())),
    )


class BuildingCurvePublisher:
    """Publishes both curve sensors when needed (parameter change/hourly)."""

    def __init__(self) -> None:
        self._last_signature = None
        self._last_time = 0.0
        self._last_outdoor: Optional[float] = None
        self.balance_estimator = BalancePointEstimator()

    def should_publish(self, signature, outdoor_temp, now: float) -> bool:
        if signature != self._last_signature:
            return True
        if now - self._last_time >= REFRESH_SECONDS:
            return True
        if (
            outdoor_temp is not None
            and (
                self._last_outdoor is None
                or abs(outdoor_temp - self._last_outdoor)
                >= OUTDOOR_REPUBLISH_DELTA
            )
        ):
            return True
        return False

    def publish(
        self,
        ha_client,
        thermal_models: Dict[str, Any],
        targets: Dict[str, float],
        climate_mode: str,
        outdoor_temp: Optional[float],
        degree: int = 4,
        now: Optional[float] = None,
        balance_observation: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Publish sensors; returns True when a publish happened."""
        from .ha_client import get_sensor_attributes
        from .shadow_mode import get_shadow_output_entity_id
        from . import config

        now = time.time() if now is None else now
        if not targets or climate_mode not in targets:
            return False
        try:
            t_out = float(outdoor_temp)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(t_out):
            return False
        parameters_by_mode = {}
        for mode in ("heating", "cooling"):
            thermal_model = thermal_models.get(mode)
            if thermal_model is None:
                return False
            parameters = dict(thermal_model._get_current_export_parameters())
            parameters["external_source_weights"] = dict(
                getattr(thermal_model, "external_source_weights", {}) or {}
            )
            parameters_by_mode[mode] = parameters

        if balance_observation:
            observation = dict(balance_observation)
            mode = observation.get("mode")
            if mode in parameters_by_mode:
                observation["heat_loss_coefficient"] = parameters_by_mode[
                    mode
                ]["heat_loss_coefficient"]
                self.balance_estimator.observe(now=now, **observation)
        balance_estimates = {
            mode: self.balance_estimator.get_estimate(
                mode,
                now,
                float(parameters_by_mode[mode]["heat_loss_coefficient"]),
            )
            for mode in ("heating", "cooling")
        }
        balance_signature = []
        for mode in ("heating", "cooling"):
            estimate = balance_estimates[mode]
            balance_temp = None
            if estimate is not None and mode in targets:
                balance_temp = balance_outdoor_temp(
                    targets[mode],
                    float(parameters_by_mode[mode]["heat_loss_coefficient"]),
                    estimate["gain_kw"],
                )
            balance_signature.append(
                (mode, round(balance_temp, 1) if balance_temp is not None else None)
            )
        signature = (
            _parameter_signature(parameters_by_mode, targets),
            climate_mode,
            int(degree),
            tuple(balance_signature),
        )
        if not self.should_publish(signature, outdoor_temp, now):
            return False

        curves = compute_curves(
            parameters_by_mode, targets, degree, balance_estimates
        )
        learned = {}
        for mode, parameters in parameters_by_mode.items():
            mode_params = {
                k: v for k, v in parameters.items()
                if isinstance(v, (int, float))
            }
            learned.update(
                {f"param_{mode}_{k}": _r(v, 6)
                 for k, v in mode_params.items()}
            )
            for src, weight in parameters["external_source_weights"].items():
                learned[f"param_{mode}_weight_{src}"] = _r(weight, 6)
        active_parameters = parameters_by_mode[climate_mode]
        active_params = {
            k: v for k, v in active_parameters.items()
            if isinstance(v, (int, float))
        }
        learned.update(
            {f"param_{k}": _r(v, 6) for k, v in active_params.items()}
        )
        for src, weight in active_parameters["external_source_weights"].items():
            learned[f"param_weight_{src}"] = _r(weight, 6)
        updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

        num_params = active_params
        hlc = float(num_params["heat_loss_coefficient"])
        eff = float(num_params["outlet_effectiveness"])
        target = targets[climate_mode]
        outlet_state = base_outlet_temp(t_out, target, hlc, eff)
        kw_state = building_load_kw(t_out, target, hlc, climate_mode)

        shadow = getattr(config, "SHADOW_MODE", False)
        all_succeeded = True
        for entity, state, key, digits in (
            (BASE_OUTLET_ENTITY_ID, outlet_state, "outlet", 1),
            (BUILDING_KW_ENTITY_ID, kw_state, "kw", 3),
        ):
            if state is None:
                all_succeeded = False
                continue
            entity_id = get_shadow_output_entity_id(
                entity, shadow_deployment=shadow
            )
            attrs = get_sensor_attributes(entity_id)
            attrs.update(curves[key])
            attrs.update(learned)
            attrs.update({
                "climate_mode": climate_mode,
                "current_outdoor_temp": _r(t_out, 2),
                "current_target_indoor": _r(target, 2),
                "last_updated": updated,
            })
            if ha_client.set_state(
                entity_id, state, _shrink_to_limit(attrs), round_digits=digits
            ) is not True:
                all_succeeded = False

        if not all_succeeded:
            return False

        self._last_signature = signature
        self._last_time = now
        self._last_outdoor = outdoor_temp
        return True
