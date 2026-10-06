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

The building-kW sensor reports a balance outdoor temperature per mode from a
persisted rolling window (see ``building_gain_window``): a modelled value
from PV/fireplace/TV gains and a measured value from the energy balance.  Both
are always exported; ``building_balance_method`` selects the one published as
``*_balance_outdoor_temp``.  The stable HVAC-off estimate is kept as a
``hvac_off`` reference.
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

from .building_capacity import resolve_capacities
from .building_gain_window import (
    BalanceSettings,
    BuildingGainWindow,
    external_gain_kw,
)

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

    def __init__(self, settings: Optional[BalanceSettings] = None) -> None:
        self.settings = settings or BalanceSettings()
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
        flow_rate_available: bool = True,
        inlet_temp_available: bool = True,
        indoor_history_available: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Add a conservative gain sample when conditions are suitable."""
        if mode not in self._samples:
            return None
        if not (
            flow_rate_available
            and inlet_temp_available
            and indoor_history_available
        ):
            self.clear(mode)
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
            self.clear(mode)
            return None
        if not all(
            math.isfinite(value)
            for value in (indoor, outdoor, hlc, drift, power, flow, timestamp)
        ):
            self.clear(mode)
            return None
        settings = self.settings
        if (
            hlc <= 0
            or abs(drift) > settings.hvac_off_max_indoor_drift_60m
            or abs(power) > settings.hvac_off_max_hvac_power_kw
            or abs(flow) > settings.hvac_off_max_flow_rate
        ):
            self.clear(mode)
            return None

        indoor_outdoor_delta = indoor - outdoor
        gains_kw = hlc * indoor_outdoor_delta
        if not 0 <= gains_kw <= settings.hvac_off_max_gain_kw:
            self.clear(mode)
            return None

        samples = self._samples[mode]
        cutoff = timestamp - settings.window_seconds
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        if (
            samples
            and timestamp - samples[-1][0]
            > settings.hvac_off_max_gap_minutes * 60
        ):
            samples.clear()
        if samples and timestamp <= samples[-1][0]:
            return self.get_estimate(mode, timestamp, hlc)
        samples.append((timestamp, indoor_outdoor_delta, hlc))
        return self.get_estimate(mode, timestamp, hlc)

    def clear(self, mode: str) -> None:
        """Discard a mode's history after an HVAC-active or invalid interval."""
        if mode in self._samples:
            self._samples[mode].clear()

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
        settings = self.settings
        cutoff = timestamp - settings.window_seconds
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        if (
            len(samples) < settings.hvac_off_min_samples
            or samples[-1][0] - samples[0][0]
            < settings.hvac_off_min_window_hours * 3600
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
        if not 0 <= gains <= settings.hvac_off_max_gain_kw:
            return None
        spread_k = math.sqrt(
            sum((sample[1] - mean_delta) ** 2 for sample in samples)
            / len(samples)
        )
        if spread_k > settings.hvac_off_max_spread_k:
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


def _estimate_balance(
    estimate: Dict[str, Any], target: float, hlc: float
) -> tuple:
    """Return (balance outdoor temp, gain kW) for an estimate dict."""
    gain_kw = estimate.get("gain_kw")
    balance_temp = estimate.get("balance_outdoor_temp")
    if balance_temp is None and gain_kw is not None:
        balance_temp = balance_outdoor_temp(target, hlc, gain_kw)
    return balance_temp, gain_kw


def _select_balance(method: str, modelled, measured) -> tuple:
    """Return (estimate, method used) for the configured method."""
    if method == "modelled":
        return modelled, "modelled" if modelled else "unavailable"
    if method == "measured":
        return measured, "measured" if measured else "unavailable"
    if measured:
        return measured, "measured"
    if modelled:
        return modelled, "modelled"
    return None, "unavailable"


def _variant_attributes(
    attrs: Dict[str, Any],
    mode: str,
    suffix: str,
    estimate: Dict[str, Any],
    target: float,
    hlc: float,
) -> None:
    temp, gain = _estimate_balance(estimate, target, hlc)
    present = estimate.get("gain_kw") is not None
    attrs[f"{mode}_balance_outdoor_temp_{suffix}"] = (
        _r(temp, 2) if temp is not None else None
    )
    attrs[f"{mode}_balance_gain_kw_{suffix}"] = (
        _r(gain, 3) if present else None
    )
    attrs[f"{mode}_balance_status_{suffix}"] = (
        estimate.get("status", "full") if present else "unavailable"
    )
    attrs[f"{mode}_balance_window_minutes_{suffix}"] = (
        int(estimate.get("window_minutes", 0)) if present else 0
    )


def compute_curves(
    parameters_by_mode: Dict[str, Dict[str, Any]],
    targets: Dict[str, float],
    degree: int = 4,
    balance_estimates: Optional[Dict[str, Dict[str, Any]]] = None,
    hvac_off_estimates: Optional[Dict[str, Dict[str, Any]]] = None,
    modelled_estimates: Optional[Dict[str, Dict[str, Any]]] = None,
    measured_estimates: Optional[Dict[str, Dict[str, Any]]] = None,
    methods_used: Optional[Dict[str, str]] = None,
    capacities: Optional[Dict[str, Dict[str, Any]]] = None,
    settings: Optional[BalanceSettings] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute attribute dicts for both sensors (all modes).

    ``targets`` maps climate mode -> active target indoor temperature.
    ``balance_estimates`` is the estimate selected by the configured method;
    ``modelled_estimates`` and ``measured_estimates`` are always exported
    next to it, ``hvac_off_estimates`` is the legacy stable-idle reference.
    Returns ``{"outlet": {...}, "kw": {...}}``.
    """
    settings = settings or BalanceSettings()
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
        # Retain the no-gains zero crossing separately from the estimates.
        kw_attrs[f"{mode}_no_gains_zero_load_outdoor_temp"] = _r(target, 2)

        estimate = (balance_estimates or {}).get(mode) or {}
        balance_temp, gain_kw = _estimate_balance(estimate, target, hlc)
        available = balance_temp is not None
        uncertainty = estimate.get("uncertainty_k")
        kw_attrs[f"{mode}_balance_outdoor_temp"] = (
            _r(balance_temp, 2) if available else None
        )
        kw_attrs[f"{mode}_balance_gain_kw"] = (
            _r(gain_kw, 3) if available else None
        )
        kw_attrs[f"{mode}_balance_uncertainty_k"] = (
            _r(uncertainty, 2)
            if available and uncertainty is not None
            else None
        )
        kw_attrs[f"{mode}_balance_estimate_status"] = (
            estimate.get("status", "full") if available else "unavailable"
        )
        kw_attrs[f"{mode}_balance_sample_count"] = (
            int(estimate.get("sample_count", 0)) if available else 0
        )
        kw_attrs[f"{mode}_balance_window_minutes"] = (
            int(estimate.get("window_minutes", 0)) if available else 0
        )
        kw_attrs[f"{mode}_balance_coverage_fraction"] = (
            _r(estimate.get("coverage_fraction", 1.0), 3)
            if available
            else 0.0
        )
        used = (methods_used or {}).get(mode, "modelled")
        kw_attrs[f"{mode}_balance_method"] = settings.method
        kw_attrs[f"{mode}_balance_method_used"] = (
            used if available else "unavailable"
        )
        kw_attrs[f"{mode}_balance_estimate_method"] = (
            (
                "rolling_window_measured_energy_balance; includes hp heat "
                "and room/slab storage"
                if used == "measured"
                else "rolling_window_modelled_non_hp_gains; mode-specific "
                "model weights; hp heat and indoor overshoot excluded"
            )
            if available
            else "unavailable"
        )
        kw_attrs[f"{mode}_balance_window_hours"] = _r(settings.window_hours, 2)
        kw_attrs[f"{mode}_balance_min_provisional_hours"] = _r(
            settings.min_provisional_hours, 2
        )
        kw_attrs[f"{mode}_balance_full_coverage_fraction"] = (
            settings.full_coverage_fraction
        )
        kw_attrs[f"{mode}_balance_max_interval_minutes"] = (
            settings.max_interval_minutes
        )
        kw_attrs[f"{mode}_balance_storage_enabled"] = settings.storage_enabled
        kw_attrs[f"{mode}_balance_dhw_intervals"] = settings.dhw_intervals
        kw_attrs[f"{mode}_balance_capacity_mode"] = settings.capacity_mode
        capacity = (capacities or {}).get(mode) or {}
        kw_attrs[f"{mode}_balance_capacity_room_kwh_per_k"] = (
            _r(capacity["room"], 3) if capacity.get("room") else None
        )
        kw_attrs[f"{mode}_balance_capacity_slab_kwh_per_k"] = (
            _r(capacity["slab"], 3) if capacity.get("slab") else None
        )
        kw_attrs[f"{mode}_balance_capacity_source"] = capacity.get(
            "source", "unavailable"
        )
        _variant_attributes(
            kw_attrs,
            mode,
            "modelled",
            (modelled_estimates or {}).get(mode) or {},
            target,
            hlc,
        )
        _variant_attributes(
            kw_attrs,
            mode,
            "measured",
            (measured_estimates or {}).get(mode) or {},
            target,
            hlc,
        )

        reference = (hvac_off_estimates or {}).get(mode) or {}
        reference_temp, reference_gain = _estimate_balance(
            reference, target, hlc
        )
        reference_available = reference_temp is not None
        kw_attrs[f"{mode}_balance_hvac_off_outdoor_temp"] = (
            _r(reference_temp, 2) if reference_available else None
        )
        kw_attrs[f"{mode}_balance_hvac_off_gain_kw"] = (
            _r(reference_gain, 3) if reference_available else None
        )
        reference_uncertainty = reference.get("uncertainty_k")
        kw_attrs[f"{mode}_balance_hvac_off_uncertainty_k"] = (
            _r(reference_uncertainty, 2)
            if reference_available and reference_uncertainty is not None
            else None
        )
        kw_attrs[f"{mode}_balance_hvac_off_sample_count"] = (
            int(reference.get("sample_count", 0)) if reference_available else 0
        )
        kw_attrs[f"{mode}_balance_hvac_off_window_minutes"] = (
            int(reference.get("window_minutes", 0)) if reference_available else 0
        )
        kw_attrs[f"{mode}_balance_hvac_off_estimate_status"] = (
            "available" if reference_available else "unavailable"
        )
        kw_attrs[f"{mode}_balance_hvac_off_estimate_method"] = (
            "stable_hvac_off_observations; assumes negligible stored-heat drift"
            if reference_available
            else "unavailable"
        )
        kw_attrs[f"{mode}_balance_hvac_off_min_sample_count"] = (
            settings.hvac_off_min_samples
        )
        kw_attrs[f"{mode}_balance_hvac_off_min_window_minutes"] = int(
            settings.hvac_off_min_window_hours * 60
        )
        kw_attrs[f"{mode}_balance_hvac_off_max_sample_gap_minutes"] = int(
            settings.hvac_off_max_gap_minutes
        )
        kw_attrs[f"{mode}_balance_hvac_off_max_indoor_drift_60m"] = (
            settings.hvac_off_max_indoor_drift_60m
        )
        kw_attrs[f"{mode}_balance_hvac_off_max_hvac_power_kw"] = (
            settings.hvac_off_max_hvac_power_kw
        )
        kw_attrs[f"{mode}_balance_hvac_off_max_flow_rate"] = (
            settings.hvac_off_max_flow_rate
        )
        kw_attrs[f"{mode}_balance_hvac_off_max_inferred_gains_kw"] = (
            settings.hvac_off_max_gain_kw
        )
        kw_attrs[f"{mode}_balance_hvac_off_max_uncertainty_k"] = (
            settings.hvac_off_max_spread_k
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

    def __init__(
        self,
        gain_window: Optional[BuildingGainWindow] = None,
        settings: Optional[BalanceSettings] = None,
    ) -> None:
        self._last_signature = None
        self._last_time = 0.0
        self._last_outdoor: Optional[float] = None
        self.gain_window = gain_window or BuildingGainWindow(settings=settings)
        self.settings = self.gain_window.settings
        self.balance_estimator = BalancePointEstimator(self.settings)

    def record_sample(
        self, thermal_models: Dict[str, Any], sample: Dict[str, Any]
    ) -> None:
        """Store one raw measurement set with both modes' modelled gains."""
        stored = dict(sample)
        outdoor = sample.get("outdoor")
        pv = sample.get("pv")
        for mode in ("heating", "cooling"):
            stored[f"g_{mode}"] = None
            model = thermal_models.get(mode)
            if model is None or outdoor is None or pv is None:
                continue
            try:
                stored[f"g_{mode}"] = external_gain_kw(
                    model,
                    self.gain_window.recent_pv() + [max(0.0, float(pv))],
                    outdoor,
                    sample.get("fireplace") or 0.0,
                    sample.get("tv") or 0.0,
                )
            except Exception:
                logging.debug("Modelled gain unavailable.", exc_info=True)
        self.gain_window.add_sample(stored)

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
        invalidate_balance_mode: Optional[str] = None,
    ) -> bool:
        """Publish sensors; returns True when a publish happened."""
        from .ha_client import get_sensor_attributes
        from .shadow_mode import get_shadow_output_entity_id
        from . import config

        if invalidate_balance_mode is not None:
            self.balance_estimator.clear(invalidate_balance_mode)
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

        observed_mode = None
        observed_estimate = None
        if balance_observation:
            observation = dict(balance_observation)
            observed_mode = observation.get("mode")
            if observed_mode in parameters_by_mode:
                observation["heat_loss_coefficient"] = parameters_by_mode[
                    observed_mode
                ]["heat_loss_coefficient"]
                observed_estimate = self.balance_estimator.observe(
                    mode=observed_mode,
                    indoor_temp=observation.get("indoor_temp"),
                    outdoor_temp=observation.get("outdoor_temp"),
                    heat_loss_coefficient=observation.get(
                        "heat_loss_coefficient"
                    ),
                    indoor_temp_delta_60m=observation.get(
                        "indoor_temp_delta_60m"
                    ),
                    thermal_power_kw=observation.get("thermal_power_kw"),
                    flow_rate=observation.get("flow_rate"),
                    now=now,
                    flow_rate_available=observation.get(
                        "flow_rate_available", True
                    ),
                    inlet_temp_available=observation.get(
                        "inlet_temp_available", True
                    ),
                    indoor_history_available=observation.get(
                        "indoor_history_available", True
                    ),
                )
        hvac_off_estimates = {}
        for mode in ("heating", "cooling"):
            if mode == observed_mode and observed_estimate is not None:
                hvac_off_estimates[mode] = observed_estimate
            else:
                hvac_off_estimates[mode] = self.balance_estimator.get_estimate(
                    mode,
                    now,
                    float(parameters_by_mode[mode]["heat_loss_coefficient"]),
                )
        settings = self.settings
        median_flow = self.gain_window.median_flow(now)
        specific_heat = float(
            getattr(config, "SPECIFIC_HEAT_CAPACITY", 4.186)
        )
        capacities: Dict[str, Dict[str, Any]] = {}
        modelled_estimates: Dict[str, Optional[Dict[str, Any]]] = {}
        measured_estimates: Dict[str, Optional[Dict[str, Any]]] = {}
        for mode in ("heating", "cooling"):
            mode_parameters = parameters_by_mode[mode]
            hlc = float(mode_parameters["heat_loss_coefficient"])
            learned_slab = None
            if settings.capacity_mode == "learned":
                learned_slab = self.gain_window.learned_slab_capacity(
                    mode, now, mode_parameters.get("outlet_effectiveness")
                )
            capacities[mode] = resolve_capacities(
                settings.capacity_mode,
                mode_parameters,
                manual_room=settings.room_capacity_kwh_per_k,
                manual_slab=settings.slab_capacity_kwh_per_k,
                median_flow_l_min=median_flow,
                specific_heat_kj_per_kg_k=specific_heat,
                learned_slab=learned_slab,
            )
            modelled_estimates[mode] = self.gain_window.gain_estimate(
                mode, now, hlc
            )
            measured_estimates[mode] = self.gain_window.measured_estimate(
                mode,
                now,
                hlc,
                capacities[mode]["room"],
                capacities[mode]["slab"],
            )
        selected_estimates: Dict[str, Optional[Dict[str, Any]]] = {}
        methods_used: Dict[str, str] = {}
        balance_signature = []
        for mode in ("heating", "cooling"):
            hlc = float(parameters_by_mode[mode]["heat_loss_coefficient"])
            entry = [mode]
            for estimates in (
                hvac_off_estimates, modelled_estimates, measured_estimates
            ):
                estimate = estimates[mode]
                balance_temp = None
                if estimate is not None and mode in targets:
                    estimate = dict(estimate)
                    balance_temp = balance_outdoor_temp(
                        targets[mode], hlc, estimate["gain_kw"]
                    )
                    estimate["balance_outdoor_temp"] = balance_temp
                    estimates[mode] = estimate
                entry.append(
                    round(balance_temp * 2) / 2
                    if balance_temp is not None
                    else None
                )
            selected_estimates[mode], methods_used[mode] = _select_balance(
                settings.method,
                modelled_estimates[mode],
                measured_estimates[mode],
            )
            entry.append(methods_used[mode])
            entry.append((selected_estimates[mode] or {}).get("status"))
            balance_signature.append(tuple(entry))
        signature = (
            _parameter_signature(parameters_by_mode, targets),
            climate_mode,
            int(degree),
            tuple(balance_signature),
        )
        if not self.should_publish(signature, outdoor_temp, now):
            return False

        curves = compute_curves(
            parameters_by_mode,
            targets,
            degree,
            balance_estimates=selected_estimates,
            hvac_off_estimates=hvac_off_estimates,
            modelled_estimates=modelled_estimates,
            measured_estimates=measured_estimates,
            methods_used=methods_used,
            capacities=capacities,
            settings=settings,
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
