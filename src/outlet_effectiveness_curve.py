"""Helpers for outdoor-temperature-dependent outlet effectiveness."""

from __future__ import annotations

from typing import Any

import numpy as np


SUPPORT_TEMPERATURES = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
MIN_SAMPLES_PER_SUPPORT = 10
BIN_HALF_WIDTH = 2.5


def normalize_curve(curve: Any) -> list[dict[str, float]]:
    """Return sorted, finite curve support points or an empty list."""
    if not isinstance(curve, (list, tuple)):
        return []
    points = []
    for point in curve:
        if not isinstance(point, dict):
            continue
        try:
            temperature = float(point["outdoor_temp"])
            effectiveness = float(point["outlet_effectiveness"])
            sample_count = max(0.0, float(point.get("sample_count", 0)))
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(temperature) and np.isfinite(effectiveness):
            points.append(
                {
                    "outdoor_temp": temperature,
                    "outlet_effectiveness": effectiveness,
                    "sample_count": sample_count,
                }
            )
    return sorted(points, key=lambda point: point["outdoor_temp"])


def interpolate_curve(
    curve: Any, outdoor_temp: float, fallback: float
) -> float:
    """Interpolate OE at an outdoor temperature, falling back to scalar OE."""
    points = normalize_curve(curve)
    if not points:
        return float(fallback)
    try:
        temperature = float(outdoor_temp)
    except (TypeError, ValueError):
        return float(fallback)
    if not np.isfinite(temperature):
        return float(fallback)
    return float(
        np.interp(
            temperature,
            [point["outdoor_temp"] for point in points],
            [point["outlet_effectiveness"] for point in points],
        )
    )


def _weighted_isotonic_non_decreasing(
    values: list[float], weights: list[float]
) -> list[float]:
    """Project values onto a non-decreasing sequence using weighted PAVA."""
    blocks = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append(
            {
                "start": index,
                "end": index + 1,
                "weight": weight,
                "weighted_value": value * weight,
            }
        )
        while len(blocks) > 1:
            previous, current = blocks[-2:]
            previous_mean = previous["weighted_value"] / previous["weight"]
            current_mean = current["weighted_value"] / current["weight"]
            if previous_mean <= current_mean:
                break
            blocks[-2:] = [
                {
                    "start": previous["start"],
                    "end": current["end"],
                    "weight": previous["weight"] + current["weight"],
                    "weighted_value": (
                        previous["weighted_value"] + current["weighted_value"]
                    ),
                }
            ]

    fitted = [0.0] * len(values)
    for block in blocks:
        mean = block["weighted_value"] / block["weight"]
        for index in range(block["start"], block["end"]):
            fitted[index] = mean
    return fitted


def calibrate_curve(
    periods: list[dict],
    heat_loss_coefficient: float,
    scalar_oe: float,
    bounds: tuple[float, float],
    climate_mode: str = "heating",
) -> list[dict[str, float]]:
    """Estimate robust, shrunken OE support points from stable periods."""
    if heat_loss_coefficient <= 0:
        return []
    points = []
    for support_temp in SUPPORT_TEMPERATURES:
        estimates = []
        for period in periods:
            try:
                outdoor = float(period["outdoor_temp"])
                indoor = float(period["indoor_temp"])
                outlet_value = period.get("effective_temp")
                if outlet_value is None:
                    outlet_value = period["outlet_temp"]
                outlet = float(outlet_value)
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite([outdoor, indoor, outlet]).all():
                continue
            if abs(outdoor - support_temp) > BIN_HALF_WIDTH:
                continue
            if climate_mode == "cooling":
                drive = indoor - outlet
                numerator = outdoor - indoor
                if drive <= 1.0 or numerator <= 0:
                    continue
            else:
                drive = outlet - indoor
                numerator = indoor - outdoor
                if drive <= 1.0 or numerator <= 0:
                    continue
            estimate = heat_loss_coefficient * numerator / drive
            if bounds[0] <= estimate <= bounds[1]:
                estimates.append(estimate)

        count = len(estimates)
        if count >= MIN_SAMPLES_PER_SUPPORT:
            robust_estimate = float(np.median(estimates))
            confidence = count / (count + MIN_SAMPLES_PER_SUPPORT)
            calibrated_oe = (
                confidence * robust_estimate + (1.0 - confidence) * scalar_oe
            )
        else:
            calibrated_oe = scalar_oe
        points.append(
            {
                "outdoor_temp": float(support_temp),
                "outlet_effectiveness": float(
                    np.clip(calibrated_oe, bounds[0], bounds[1])
                ),
                "sample_count": float(count),
            }
        )

    # Smooth populated supports only; empty bins remain exactly at the scalar
    # prior rather than inheriting a neighboring estimate.
    values = [point["outlet_effectiveness"] for point in points]
    for index, point in enumerate(points):
        if point["sample_count"] < MIN_SAMPLES_PER_SUPPORT:
            continue
        neighbor_indexes = range(max(0, index - 1), min(len(values), index + 2))
        weighted_values = [
            values[i] * points[i]["sample_count"]
            for i in neighbor_indexes
            if points[i]["sample_count"] >= MIN_SAMPLES_PER_SUPPORT
        ]
        weights = [
            points[i]["sample_count"]
            for i in neighbor_indexes
            if points[i]["sample_count"] >= MIN_SAMPLES_PER_SUPPORT
        ]
        if weights:
            point["outlet_effectiveness"] = float(
                np.clip(sum(weighted_values) / sum(weights), bounds[0], bounds[1])
            )

    if climate_mode != "cooling":
        # For heating, constrain colder-weather OE to be no higher than
        # warmer-weather OE. The scalar shrinkage above contributes the same
        # pseudo-sample weight as MIN_SAMPLES_PER_SUPPORT.
        weights = [
            point["sample_count"] + MIN_SAMPLES_PER_SUPPORT for point in points
        ]
        fitted_values = _weighted_isotonic_non_decreasing(
            [point["outlet_effectiveness"] for point in points], weights
        )
        for point, value in zip(points, fitted_values):
            point["outlet_effectiveness"] = float(
                np.clip(value, bounds[0], bounds[1])
            )
    return points


def update_curve_at_temperature(
    curve: Any,
    outdoor_temp: float,
    adjustment: float,
    bounds: tuple[float, float],
) -> list[dict[str, float]]:
    """Apply a bounded, locally weighted learning delta to nearby knots."""
    points = normalize_curve(curve)
    if not points or not np.isfinite(outdoor_temp) or not np.isfinite(adjustment):
        return points
    temperatures = [point["outdoor_temp"] for point in points]
    nearest = int(np.argmin(np.abs(np.asarray(temperatures) - outdoor_temp)))
    for index, point in enumerate(points):
        distance = abs(index - nearest)
        weight = 1.0 if distance == 0 else (0.25 if distance == 1 else 0.0)
        if not weight:
            continue
        point["outlet_effectiveness"] = float(
            np.clip(
                point["outlet_effectiveness"] + adjustment * weight,
                bounds[0],
                bounds[1],
            )
        )
    return points
