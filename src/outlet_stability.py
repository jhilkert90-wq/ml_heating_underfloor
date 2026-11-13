"""Stabilisation helpers for the outlet temperature search.

With an inert underfloor slab the 4 h trajectory end temperature is nearly
flat in the outlet temperature (sensitivity S = dT_end/dT_outlet of a few
hundredths K/K).  The zero crossing of the end error is then ill-conditioned
and tiny input noise moves it by several Kelvin.  These pure helpers
regularise such searches towards the learned physical equilibrium outlet
(``building_curve.base_outlet_temp``) and make the cycle-to-cycle limits
demand-aware.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

from .building_curve import base_outlet_temp


def has_heat_demand(
    current_indoor: float,
    target_indoor: float,
    trend_60m: Optional[float],
    deficit_k: float,
    trend_threshold: float,
) -> bool:
    """True when the room is below target or is falling towards it."""
    deficit = float(target_indoor) - float(current_indoor)
    if deficit > deficit_k:
        return True
    try:
        trend = float(trend_60m) if trend_60m is not None else 0.0
    except (TypeError, ValueError):
        trend = 0.0
    return trend < -trend_threshold and deficit > -deficit_k


def reference_outlet(
    outdoor_temp: float,
    target_indoor: float,
    heat_loss_coefficient: float,
    outlet_effectiveness: float,
    lower: float,
    upper: float,
) -> Optional[float]:
    """Learned base-curve outlet clamped to the active search bounds."""
    try:
        ref = base_outlet_temp(
            float(outdoor_temp),
            float(target_indoor),
            float(heat_loss_coefficient),
            float(outlet_effectiveness),
        )
    except (TypeError, ValueError):
        return None
    if ref is None or not math.isfinite(ref):
        return None
    return min(upper, max(lower, ref))


def reference_band(demand: bool, base_k: float, demand_extra_k: float) -> float:
    """Allowed deviation from the reference; wider with proven demand."""
    return base_k + (demand_extra_k if demand else 0.0)


def pick_regularized(
    candidates: Iterable[Dict[str, Any]],
    reference: float,
    lam: float,
    lower: float,
    upper: float,
) -> Optional[Dict[str, Any]]:
    """Best candidate by ``score + lam * (outlet - reference)^2``.

    Only candidates inside ``[lower, upper]`` are considered.  The raw
    ``score`` already measures comfort error in K, so ``lam`` is the comfort
    error (K) tolerated per K² of deviation from the reference.
    """
    best = None
    best_value = math.inf
    for cand in candidates:
        outlet = cand["outlet"]
        if outlet < lower - 1e-9 or outlet > upper + 1e-9:
            continue
        value = cand["score"] + lam * (outlet - reference) ** 2
        if value < best_value:
            best, best_value = cand, value
    return best


def slew_limit(
    max_change: float,
    calm_limit: float,
    demand: bool,
    target_jump: bool,
) -> float:
    """Per-cycle setpoint change limit; relaxed on demand or target jumps."""
    if demand or target_jump:
        return max_change
    return min(max_change, calm_limit)
