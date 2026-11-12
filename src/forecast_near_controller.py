"""Near-range hand-over from the live controller to the WP load forecast.

The control cycle stores the outlet setpoint it actually chose. For a short
horizon the forecast uses that setpoint (power = scale * eff * (outlet - room))
instead of the idealised lookahead controller. ``scale`` is an EMA against the
measured thermal power so model bias is learned.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

SNAPSHOT_MAX_AGE_S = 30 * 60
SCALE_MIN, SCALE_MAX = 0.5, 2.0
SCALE_ALPHA = 0.1
MEASURED_RUNNING_KW = 0.5


def save_snapshot(
    wrapper: Any, *, outlet: float | None, indoor: float | None
) -> None:
    """Remember the last controller decision on the wrapper."""
    if outlet is None or indoor is None:
        wrapper._forecast_control_snapshot = None
        return
    wrapper._forecast_control_snapshot = {
        "outlet_c": float(outlet),
        "indoor_c": float(indoor),
        "time": time.time(),
    }


def update_scale(
    state: dict[str, Any],
    *,
    snapshot: Mapping[str, Any],
    eff: float,
    measured_kw: float | None,
) -> float:
    """Update and return the persisted power scale factor."""
    scale = float(state.get("near_controller_scale", 1.0))
    if measured_kw is not None and measured_kw > 0.2 and eff > 0:
        modelled = eff * max(0.0, snapshot["outlet_c"] - snapshot["indoor_c"])
        if modelled > 0.2:
            ratio = min(SCALE_MAX, max(SCALE_MIN, measured_kw / modelled))
            scale = (1 - SCALE_ALPHA) * scale + SCALE_ALPHA * ratio
    scale = min(SCALE_MAX, max(SCALE_MIN, scale))
    state["near_controller_scale"] = scale
    return scale


def near_parameters(
    wrapper: Any,
    state: dict[str, Any],
    *,
    eff: float,
    inlet: float | None,
    measured_kw: float | None,
    dhw_active: bool,
    delta_t_floor: float,
    near_hours: float,
    blend_hours: float,
    now_ts: float | None = None,
) -> dict[str, Any] | None:
    """Return simulation kwargs, or None when the snapshot is not usable."""
    snapshot = getattr(wrapper, "_forecast_control_snapshot", None)
    if not snapshot or dhw_active or inlet is None:
        return None
    now_ts = time.time() if now_ts is None else now_ts
    if now_ts - snapshot["time"] > SNAPSHOT_MAX_AGE_S:
        return None
    measured_running = measured_kw is not None and measured_kw >= MEASURED_RUNNING_KW
    if snapshot["outlet_c"] - inlet < delta_t_floor and not measured_running:
        return None
    if not math.isfinite(eff) or eff <= 0:
        return None
    scale = update_scale(
        state, snapshot=snapshot, eff=eff, measured_kw=measured_kw
    )
    return {
        "near_outlet_c": snapshot["outlet_c"],
        "near_eff": eff,
        "near_scale": scale,
        "near_hours": near_hours,
        "near_blend_hours": blend_hours,
    }
