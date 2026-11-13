"""Pure helpers that keep the WP load forecast stable.

Covers power sampling/averaging for the start-estimate window and slot 0,
blockade (DHW/defrost) hold of the last good simulation, and the 24 h sum
stabilisation. No Home Assistant access happens here.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, MutableMapping, Sequence

LOGGER = logging.getLogger(__name__)

POWER_HISTORY_KEY = "power_history"


def _utc(stamp: datetime) -> datetime:
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def record_power_sample(
    state: MutableMapping[str, Any],
    now: datetime,
    *,
    el_kw: float | None,
    th_kw: float | None,
    retention_minutes: float = 120.0,
) -> None:
    """Append one power sample and drop samples older than the retention."""
    now_utc = _utc(now)
    history = [
        item
        for item in state.get(POWER_HISTORY_KEY, [])
        if isinstance(item, Mapping) and "t" in item
    ]
    if el_kw is not None or th_kw is not None:
        history.append(
            {"t": now_utc.isoformat(), "el": el_kw, "th": th_kw}
        )
    cutoff = now_utc - timedelta(minutes=max(1.0, retention_minutes))
    kept = []
    for item in history:
        try:
            if datetime.fromisoformat(item["t"]) >= cutoff:
                kept.append(item)
        except (TypeError, ValueError):
            continue
    state[POWER_HISTORY_KEY] = kept


def windowed_mean_kw(
    state: Mapping[str, Any],
    now: datetime,
    *,
    key: str,
    window_minutes: float,
) -> float | None:
    """Mean of the recorded ``key`` ('el' or 'th') over the last window."""
    cutoff = _utc(now) - timedelta(minutes=max(1.0, window_minutes))
    values = []
    for item in state.get(POWER_HISTORY_KEY, []):
        try:
            if datetime.fromisoformat(item["t"]) < cutoff:
                continue
            value = item.get(key)
            if value is not None and math.isfinite(float(value)):
                values.append(max(0.0, float(value)))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def start_window_power_kw(
    *,
    instantaneous_kw: float | None,
    mean_kw: float | None,
    reference_kw: float,
    cap_factor: float,
    max_kw: float,
) -> tuple[float, str] | None:
    """Power for the start-estimate window: windowed mean, capped.

    The mean replaces a possibly defrost-distorted instantaneous value and is
    capped at ``cap_factor`` x the reference (baseline / simulated power).
    """
    value, source = (
        (mean_kw, "window_mean")
        if mean_kw is not None
        else (instantaneous_kw, "instantaneous")
    )
    if value is None or value <= 0.0:
        return None
    cap = min(max_kw, max(0.0, cap_factor) * reference_kw)
    if cap > 0.0 and value > cap:
        LOGGER.info(
            "Start-estimate window power capped %.2f -> %.2f kW (%.1fx reference %.2f kW)",
            value,
            cap,
            cap_factor,
            reference_kw,
        )
        return cap, f"{source}_capped"
    return min(max_kw, value), source


def blend_slot0_projection(
    *,
    measured_mean_kw: float | None,
    model_rate_kw: float,
    weight: float,
    cap_factor: float,
) -> float:
    """Mix the measured mean with the model rate; capped at the simulation."""
    if measured_mean_kw is None:
        return model_rate_kw
    weight = min(1.0, max(0.0, weight))
    mixed = weight * measured_mean_kw + (1.0 - weight) * model_rate_kw
    if model_rate_kw > 0.0 and cap_factor > 0.0:
        mixed = min(mixed, cap_factor * model_rate_kw)
    return max(0.0, mixed)


def shift_held_forecast(
    held_kwh: Sequence[float],
    *,
    first_hour_fraction_at_generation: float,
    hours_elapsed: int,
    remaining_hours: float,
) -> list[float]:
    """Re-index a held hourly forecast to the current hour.

    Slot 0 of the held list only covered the remainder of its hour; it is
    converted to an hourly rate before it is moved or rescaled.
    """
    if not held_kwh:
        return []
    frac = max(1e-6, min(1.0, first_hour_fraction_at_generation))
    rates = [float(held_kwh[0]) / frac] + [float(v) for v in held_kwh[1:]]
    shift = max(0, int(hours_elapsed))
    shifted = rates[shift:] or [rates[-1]]
    shifted += [shifted[-1]] * (len(held_kwh) - len(shifted))
    shifted[0] *= max(0.0, min(1.0, remaining_hours))
    return shifted


def stabilize_window_sum(
    heat_kwh: Sequence[float],
    baseline_kwh: Sequence[float],
    *,
    previous_sum: float | None,
    window_hours: int = 24,
    baseline_band: float = 0.15,
    smoothing_alpha: float = 0.5,
    hysteresis: float = 0.03,
    min_baseline_kwh: float = 1.0,
) -> tuple[list[float], dict[str, Any]]:
    """Bind the future window sum to the baseline and smooth it over runs.

    The simulation only distributes the energy over the window; slot 0 is
    left alone because it is reconciled with the counters later.
    """
    result = [float(v) for v in heat_kwh]
    indices = list(range(1, min(len(result), window_hours + 1)))
    info: dict[str, Any] = {"applied": False}
    if not indices:
        return result, info
    raw = sum(result[i] for i in indices)
    target = raw
    info.update(raw_sum=round(raw, 3), reason="unchanged")
    base = sum(max(0.0, float(baseline_kwh[i])) for i in indices if i < len(baseline_kwh))
    if base >= min_baseline_kwh and baseline_band >= 0.0:
        low, high = base * (1.0 - baseline_band), base * (1.0 + baseline_band)
        if target > high:
            target, info["reason"] = high, "baseline_upper_bound"
        elif target < low:
            target, info["reason"] = low, "baseline_lower_bound"
        info["baseline_sum"] = round(base, 3)
    if previous_sum is not None and previous_sum > 0.0:
        if abs(target - previous_sum) <= hysteresis * previous_sum:
            target, info["reason"] = previous_sum, "hysteresis_hold"
        else:
            alpha = min(1.0, max(0.0, smoothing_alpha))
            target = previous_sum + alpha * (target - previous_sum)
            info["reason"] = f"{info['reason']}+smoothed"
    if raw > 1e-9 and abs(target - raw) > 1e-6:
        factor = target / raw
        for i in indices:
            result[i] *= factor
        info["applied"] = True
        info["factor"] = round(factor, 4)
    elif raw <= 1e-9 and target > 1e-6:
        share = target / len(indices)
        for i in indices:
            result[i] = share
        info["applied"] = True
    info["final_sum"] = round(sum(result[i] for i in indices), 3)
    return result, info
