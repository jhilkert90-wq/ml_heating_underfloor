"""Optional reader for an external "heat pump start / runtime estimate" sensor.

The sensor is built in Home Assistant (see docs/HA_WP_START_ESTIMATE_GUIDE.md)
and only reports a result: state = minutes until start (0 = running,
unknown = no start expected) plus the attributes ``status``, ``mode``,
``method``, ``expected_runtime_min``, ``confidence`` and ``updated_at``.

The reader never raises and returns ``None`` whenever the estimate must not
be used, so callers behave exactly as without a sensor.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

LOGGER = logging.getLogger(__name__)

VALID_STATUS = ("idle", "starting_soon", "running", "blocked", "unknown")
VALID_METHODS = ("degree_minutes", "hysteresis", "learned")
_CIRCUIT_BREAKER_LIMIT = 5
_CIRCUIT_BREAKER_PAUSE_S = 600.0


@dataclass(frozen=True)
class HPStartEstimate:
    mode: str
    status: str
    method: str
    minutes_to_start: Optional[float]
    expected_runtime_min: Optional[float]
    runtime_unbounded: bool
    confidence: float
    age_min: float
    attributes: Mapping[str, Any]
    runtime_source: str = "attribute"

    @property
    def known_method(self) -> bool:
        return self.method in VALID_METHODS


_failures = 0
_paused_until = 0.0
_last_reason: dict[str, str] = {}


def last_fallback_reason(mode: str) -> str:
    """Reason why the last read for ``mode`` returned no estimate."""
    return _last_reason.get(mode, "")


def _reset_for_tests() -> None:
    global _failures, _paused_until
    _failures = 0
    _paused_until = 0.0
    _last_reason.clear()


def _parse_time(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _number(raw: Any) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _runtime_sensor_minutes(
    states: Mapping[str, Any], mode: str, cfg: Any
) -> Optional[float]:
    """Minutes from the optional separate runtime sensor, else ``None``.

    Unavailable, unknown or invalid values fall back to the start sensor's
    ``expected_runtime_min`` attribute. Freshness follows the start sensor
    because a state-only sensor does not update while its value is unchanged.
    """
    entity_id = getattr(
        cfg,
        "HP_START_ESTIMATE_HEATING_RUNTIME_ENTITY_ID"
        if mode == "heating"
        else "HP_START_ESTIMATE_COOLING_RUNTIME_ENTITY_ID",
        "",
    )
    if not entity_id:
        return None
    entity = states.get(entity_id)
    if not isinstance(entity, Mapping):
        return None
    value = _number(entity.get("state"))
    return value if value is not None and value >= 0 else None


def read_hp_start_estimate(
    states: Mapping[str, Any],
    mode: str,
    *,
    now: Optional[datetime] = None,
    cfg: Any = None,
) -> Optional[HPStartEstimate]:
    """Return a validated estimate for ``mode`` or ``None`` (fallback)."""
    global _failures, _paused_until
    if cfg is None:
        from . import config as cfg  # type: ignore[no-redef]

    def fallback(reason: str) -> None:
        _last_reason[mode] = reason
        return None

    try:
        if not getattr(cfg, "HP_START_ESTIMATE_ENABLED", False):
            return fallback("disabled")
        entity_id = getattr(
            cfg,
            "HP_START_ESTIMATE_HEATING_ENTITY_ID"
            if mode == "heating"
            else "HP_START_ESTIMATE_COOLING_ENTITY_ID",
            "",
        )
        if not entity_id:
            return fallback("not_configured")
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if now.timestamp() < _paused_until:
            return fallback("circuit_breaker")
        entity = states.get(entity_id)
        if not isinstance(entity, Mapping):
            return fallback("missing")
        attrs = entity.get("attributes")
        attrs = attrs if isinstance(attrs, Mapping) else {}
        if attrs.get("mode") != mode:
            return fallback("wrong_mode")
        status = str(attrs.get("status", "unknown"))
        if status not in VALID_STATUS:
            return fallback("invalid_status")
        updated = _parse_time(attrs.get("updated_at"))
        if updated is None:
            return fallback("no_timestamp")
        age_min = (now - updated.astimezone(timezone.utc)).total_seconds() / 60.0
        if age_min > float(getattr(cfg, "HP_START_ESTIMATE_MAX_AGE_MIN", 5.0)):
            return fallback("stale")
        confidence = _number(attrs.get("confidence"))
        if confidence is None or confidence < float(
            getattr(cfg, "HP_START_ESTIMATE_MIN_CONFIDENCE", 0.0)
        ):
            return fallback("low_confidence")
        raw_state = entity.get("state")
        minutes = _number(raw_state)
        if minutes is None and str(raw_state).lower() not in (
            "unknown", "unavailable", "none", ""
        ):
            return fallback("invalid_state")
        if minutes is not None and minutes < 0:
            return fallback("invalid_state")
        if minutes is None and status in ("starting_soon", "running"):
            return fallback("inconsistent")
        sensor_runtime = _runtime_sensor_minutes(states, mode, cfg)
        runtime = _number(attrs.get("expected_runtime_min"))
        runtime_source = "attribute"
        if sensor_runtime is not None:
            runtime = sensor_runtime
            runtime_source = "sensor"
        elif runtime is not None and runtime < 0:
            return fallback("invalid_runtime")
        _failures = 0
        _last_reason[mode] = ""
        return HPStartEstimate(
            mode=mode,
            status=status,
            method=str(attrs.get("method", "")),
            minutes_to_start=minutes,
            expected_runtime_min=runtime,
            runtime_unbounded=(
                bool(attrs.get("runtime_unbounded", False))
                and runtime_source == "attribute"
            ),
            runtime_source=runtime_source,
            confidence=confidence,
            age_min=age_min,
            attributes=dict(attrs),
        )
    except Exception as exc:  # reader must never break a control cycle
        _failures += 1
        if _failures >= _CIRCUIT_BREAKER_LIMIT:
            _paused_until = (
                (now or datetime.now(timezone.utc)).timestamp()
                + _CIRCUIT_BREAKER_PAUSE_S
            )
            LOGGER.warning(
                "HP start estimate reader paused after %d errors: %s",
                _failures, exc,
            )
            _failures = 0
        return fallback("error")


def trajectory_window(estimate: Optional[HPStartEstimate]) -> Optional[dict]:
    """Start delay/runtime [h] for the thermal trajectory, or None (heuristic).

    No expected start (idle, unknown minutes) means the HP contributes nothing
    in the horizon.
    """
    if estimate is None or not estimate.known_method:
        return None
    if estimate.status in ("blocked", "unknown"):
        return None
    runtime = estimate.expected_runtime_min
    runtime_h = runtime / 60.0 if runtime is not None else None
    if estimate.minutes_to_start is None:
        return {"delay_h": 1e6, "runtime_h": runtime_h}
    return {"delay_h": estimate.minutes_to_start / 60.0, "runtime_h": runtime_h}


def candidate_window(
    estimate: Optional[HPStartEstimate],
    window: Optional[dict],
    *,
    candidate_outlet: float,
    reference_outlet: Optional[float],
    mode: str,
) -> Optional[dict]:
    """Start window for one outlet candidate (degree-minutes method only).

    The degree-minute deficit grows with the calculated supply setpoint, which
    follows the outlet one to one. A candidate ``d`` K above (heating) or below
    (cooling) the reference outlet raises the deficit rate from ``need`` to
    ``need + d`` and shortens the delay proportionally. The sensor's delay is
    an upper bound; everything else keeps the sensor window unchanged.
    """
    if (
        estimate is None
        or not isinstance(window, dict)
        or reference_outlet is None
        or estimate.method != "degree_minutes"
        or estimate.status not in ("idle", "starting_soon")
        or not estimate.minutes_to_start
        or estimate.minutes_to_start <= 0
    ):
        return window
    required = _number(estimate.attributes.get("required_supply_c"))
    supply = _number(estimate.attributes.get("supply_actual_c"))
    if required is None or supply is None:
        return window
    need = (required - supply) if mode == "heating" else (supply - required)
    shift = (
        candidate_outlet - reference_outlet
        if mode == "heating"
        else reference_outlet - candidate_outlet
    )
    if need <= 0.05 or need + shift <= need:
        return window
    return {
        **window,
        "delay_h": min(window["delay_h"], window["delay_h"] * need / (need + shift)),
    }


def learning_weight(
    estimate: Optional[HPStartEstimate], cycle_minutes: float
) -> Optional[float]:
    """Weight [0, 1] of a cycle for HP learning; None = no sensor info.

    Blocked cycles (DHW/defrost) get 0; running cycles keep weight 1. For
    idle/starting_soon only the share after the expected start counts, so a
    cycle without a start inside it gets 0. ``unknown`` carries no information.
    """
    if estimate is None or not estimate.known_method:
        return None
    if estimate.status == "blocked":
        return 0.0
    if estimate.status == "unknown":
        return None
    if estimate.status == "running":
        return 1.0
    minutes = estimate.minutes_to_start
    if minutes is None or cycle_minutes <= 0:
        return 0.0
    return max(0.0, min(1.0, (cycle_minutes - minutes) / cycle_minutes))


def cost_skip_allowed(
    estimate: Optional[HPStartEstimate], horizon_min: float
) -> bool:
    """False if the sensor says the HP will not run within ``horizon_min``.

    Without usable sensor information the existing cost-mode skip applies.
    """
    if estimate is None or not estimate.known_method:
        return True
    if estimate.status in ("blocked", "unknown", "running"):
        return True
    minutes = estimate.minutes_to_start
    return minutes is not None and minutes <= horizon_min


def diagnostics(
    states: Mapping[str, Any], *, now: Optional[datetime] = None, cfg: Any = None
) -> dict[str, Any]:
    """Diagnostic attributes only; never changes control behaviour."""
    out: dict[str, Any] = {}
    for mode in ("heating", "cooling"):
        estimate = read_hp_start_estimate(states, mode, now=now, cfg=cfg)
        out[mode] = (
            {
                "start_estimate_source": "sensor",
                "status": estimate.status,
                "method": estimate.method,
                "minutes_to_start": estimate.minutes_to_start,
                "age_min": round(estimate.age_min, 1),
                "runtime_source": estimate.runtime_source,
            }
            if estimate
            else {
                "start_estimate_source": "fallback",
                "fallback_reason": last_fallback_reason(mode),
            }
        )
    return out
