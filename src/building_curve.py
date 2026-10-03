"""Building characteristic curves derived from the learned thermal model.

Two curves are computed per climate mode, using only the learned physics
(heat loss coefficient and outlet effectiveness).  External heat sources
(PV, fireplace, TV) are deliberately ignored:

* base outlet temperature [°C] needed to hold the target indoor
  temperature at a given outdoor temperature (raw, unclamped)::

      T_outlet = (T_target * (HLC + eff) - HLC * T_out) / eff

* building load [kW] = HLC [kW/K] * (T_target - T_out) for heating and
  HLC * (T_out - T_target) for cooling (raw, may be negative outside the
  mode's useful range).

Each curve is also approximated by a polynomial (degree 1-4).
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

HEATING_RANGE = (-20.0, 20.0)
COOLING_RANGE = (10.0, 40.0)
GRID_STEP = 1.0
MAX_ATTRIBUTE_BYTES = 14000  # HA recorder limit is 16384
REFRESH_SECONDS = 3600.0
OUTDOOR_REPUBLISH_DELTA = 0.5

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
    parameters: Dict[str, Any],
    targets: Dict[str, float],
    degree: int = 4,
) -> Dict[str, Dict[str, Any]]:
    """Compute attribute dicts for both sensors (all modes).

    ``targets`` maps climate mode -> active target indoor temperature.
    Returns ``{"outlet": {...}, "kw": {...}}``.
    """
    hlc = float(parameters["heat_loss_coefficient"])
    eff = float(parameters["outlet_effectiveness"])
    outlet_attrs: Dict[str, Any] = {}
    kw_attrs: Dict[str, Any] = {}
    for mode, target in targets.items():
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
        # Outdoor temperature where the load drops to zero == target
        kw_attrs[f"{mode}_balance_outdoor_temp"] = _r(target, 2)
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


def _parameter_signature(parameters: Dict[str, Any], targets: Dict[str, float]):
    return (
        tuple(sorted((k, round(float(v), 6)) for k, v in parameters.items()
                     if isinstance(v, (int, float)))),
        tuple(sorted((k, round(float(v), 2)) for k, v in targets.items())),
    )


class BuildingCurvePublisher:
    """Publishes both curve sensors when needed (parameter change/hourly)."""

    def __init__(self) -> None:
        self._last_signature = None
        self._last_time = 0.0
        self._last_outdoor: Optional[float] = None

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
        thermal_model,
        targets: Dict[str, float],
        climate_mode: str,
        outdoor_temp: Optional[float],
        degree: int = 4,
        now: Optional[float] = None,
    ) -> bool:
        """Publish sensors; returns True when a publish happened."""
        from .ha_client import get_sensor_attributes
        from .shadow_mode import get_shadow_output_entity_id
        from . import config

        now = time.time() if now is None else now
        parameters = dict(thermal_model._get_current_export_parameters())
        parameters["external_source_weights"] = dict(
            getattr(thermal_model, "external_source_weights", {}) or {}
        )
        num_params = {
            k: v for k, v in parameters.items()
            if isinstance(v, (int, float))
        }
        if not targets or climate_mode not in targets:
            return False
        signature = _parameter_signature(num_params, targets)
        if not self.should_publish(signature, outdoor_temp, now):
            return False

        curves = compute_curves(num_params, targets, degree)
        learned = {f"param_{k}": _r(v, 6) for k, v in num_params.items()}
        for src, weight in parameters["external_source_weights"].items():
            learned[f"param_weight_{src}"] = _r(weight, 6)
        updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

        hlc = float(num_params["heat_loss_coefficient"])
        eff = float(num_params["outlet_effectiveness"])
        target = targets[climate_mode]
        t_out = outdoor_temp if outdoor_temp is not None else 0.0
        outlet_state = base_outlet_temp(t_out, target, hlc, eff)
        kw_state = building_load_kw(t_out, target, hlc, climate_mode)

        shadow = getattr(config, "SHADOW_MODE", False)
        for entity, state, key, digits in (
            (BASE_OUTLET_ENTITY_ID, outlet_state, "outlet", 1),
            (BUILDING_KW_ENTITY_ID, kw_state, "kw", 3),
        ):
            if state is None:
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
            ha_client.set_state(
                entity_id, state, _shrink_to_limit(attrs), round_digits=digits
            )

        self._last_signature = signature
        self._last_time = now
        self._last_outdoor = outdoor_temp
        return True
