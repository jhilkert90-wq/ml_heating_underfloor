"""Thermal storage capacities [kWh/K] shared by the balance point and replay.

The thermal model relaxes the room towards
``T_eq = (OE*T_floor + HLC*T_out + G) / (OE + HLC)`` with ``thermal_time_constant``.
Outlet effectiveness is a conductance [kW/K], so the room capacity is
``tau * (OE + HLC)`` and not ``tau * HLC``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

CAPACITY_MODES = ("model", "manual", "learned")


def _positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def room_capacity_kwh_per_k(parameters: Mapping[str, Any]) -> Optional[float]:
    """Room node capacity ``tau * (OE + HLC)``; missing OE counts as 0."""
    tau = _positive(parameters.get("thermal_time_constant"))
    hlc = _positive(parameters.get("heat_loss_coefficient"))
    if tau is None or hlc is None:
        return None
    outlet_effectiveness = _positive(parameters.get("outlet_effectiveness")) or 0.0
    return tau * (hlc + outlet_effectiveness)


def slab_capacity_kwh_per_k(
    parameters: Mapping[str, Any],
    median_flow_l_min: Optional[float],
    specific_heat_kj_per_kg_k: float,
) -> Optional[float]:
    """Slab capacity ``m_dot * cp * tau_slab`` from the pump-on water flow."""
    tau_slab = _positive(parameters.get("slab_time_constant_hours"))
    flow = _positive(median_flow_l_min)
    specific_heat = _positive(specific_heat_kj_per_kg_k)
    if tau_slab is None or flow is None or specific_heat is None:
        return None
    return flow / 60.0 * specific_heat * tau_slab


def resolve_capacities(
    capacity_mode: str,
    parameters: Mapping[str, Any],
    *,
    manual_room: float,
    manual_slab: float,
    median_flow_l_min: Optional[float],
    specific_heat_kj_per_kg_k: float,
    learned_slab: Optional[float] = None,
) -> Dict[str, Any]:
    """Return ``{"room", "slab", "source"}``; falls back to the manual values."""
    if capacity_mode == "manual":
        return {"room": manual_room, "slab": manual_slab, "source": "manual"}

    room = room_capacity_kwh_per_k(parameters)
    room_source = "model"
    if room is None:
        room, room_source = manual_room, "manual_fallback"

    slab_source = "model"
    slab = None
    if capacity_mode == "learned" and _positive(learned_slab) is not None:
        slab, slab_source = float(learned_slab), "learned"
    if slab is None:
        slab = slab_capacity_kwh_per_k(
            parameters, median_flow_l_min, specific_heat_kj_per_kg_k
        )
        slab_source = "model"
    if slab is None:
        slab, slab_source = manual_slab, "manual_fallback"
    return {
        "room": room,
        "slab": slab,
        "source": f"room:{room_source},slab:{slab_source}",
    }
