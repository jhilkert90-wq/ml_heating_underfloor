"""Joint trajectory outlet selector for ``heating_correction_mode: trajectory``.

Every candidate outlet on a fixed grid is simulated over the full horizon and
scored by the squared comfort-band violation (K²h).  The cheapest candidate
wins; ties go to the lowest outlet while the room is above the band and to
the outlet nearest the last published value otherwise.
"""
import math
from typing import Callable, Optional, Sequence

COMFORT_BAND_K = 0.1


def comfort_cost(
    trajectory: Sequence[float], target: float, band: float, step_hours: float
) -> float:
    """Return the squared comfort-band violation of a trajectory in K²h."""
    total = 0.0
    for temp in trajectory:
        under = max(target - band - temp, 0.0)
        over = max(temp - target - band, 0.0)
        total += under * under + over * over
    return total * step_hours


def select_trajectory_outlet(
    simulate: Callable[[float], Sequence[float]],
    *,
    lower: float,
    upper: float,
    baseline: float,
    current_indoor: float,
    target: float,
    step_hours: float,
    band: float = COMFORT_BAND_K,
    resolution: float = 0.1,
) -> Optional[dict]:
    """Pick the outlet whose simulated trajectory violates the band least.

    Candidates whose simulation raises or yields no finite trajectory are
    skipped; ``None`` is returned when no candidate is usable.
    """
    if upper < lower:
        upper = lower
    baseline = min(upper, max(lower, baseline))
    candidates = {lower, upper, baseline}
    start = math.ceil(lower / resolution - 1e-9)
    stop = math.floor(upper / resolution + 1e-9)
    for index in range(start, stop + 1):
        candidates.add(round(index * resolution, 10))

    above_band = current_indoor > target + band
    best = None
    best_key = None
    for outlet in sorted(candidates):
        try:
            trajectory = [float(t) for t in simulate(outlet)]
        except Exception:
            continue
        if not trajectory or not all(math.isfinite(t) for t in trajectory):
            continue
        cost = comfort_cost(trajectory, target, band, step_hours)
        key = (
            round(cost, 9),
            outlet if above_band else abs(outlet - baseline),
            outlet,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = {
                "outlet": outlet,
                "cost": cost,
                "min_indoor": min(trajectory),
                "max_indoor": max(trajectory),
            }
    return best
