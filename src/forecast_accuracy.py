"""Forecast-vs-actual accuracy tracking for the WP load forecast.

Predictions are stored per (target hour, lead time) with first write wins.
Once the hourly counter energy of that hour is known, the pair is moved to a
bounded history from which bias/MAE/MAPE and start-detection rates derive.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

LEADS = (1, 6, 24)
HISTORY_DAYS = 7
MISSED_PRED_MAX = 0.1
MISSED_ACTUAL_MIN = 0.2


def _section(state: dict[str, Any]) -> dict[str, Any]:
    section = state.setdefault("forecast_accuracy", {})
    section.setdefault("pending", {})
    section.setdefault("history", [])
    return section


def record_predictions(
    state: dict[str, Any],
    *,
    now: datetime,
    starts: Sequence[datetime],
    el_kwh: Sequence[float],
    th_kwh: Sequence[float],
) -> None:
    """Store the first prediction for each target hour and lead."""
    section = _section(state)
    pending = section["pending"]
    for lead in LEADS:
        if lead >= len(starts) or lead >= len(el_kwh) or lead >= len(th_kwh):
            continue
        hour = starts[lead].astimezone(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )
        key = f"{hour.isoformat()}|{lead}"
        pending.setdefault(
            key,
            {
                "hour_utc": hour.isoformat(),
                "lead": lead,
                "el": float(el_kwh[lead]),
                "th": float(th_kwh[lead]),
            },
        )
    LOGGER.debug("Forecast accuracy: %d pending predictions", len(pending))
    cutoff = now.astimezone(timezone.utc) - timedelta(hours=48)
    section["pending"] = {
        key: value
        for key, value in pending.items()
        if datetime.fromisoformat(value["hour_utc"]) >= cutoff
    }


def anchor_rejection_reason(anchor: Mapping[str, Any] | None) -> str | None:
    """Explain why an anchor cannot supply aligned accuracy counter readings."""
    if not isinstance(anchor, Mapping):
        return "missing anchor"
    try:
        boundary = datetime.fromisoformat(anchor["hour_utc"]).astimezone(timezone.utc)
        if boundary != boundary.replace(minute=0, second=0, microsecond=0):
            return "not a full-hour boundary"
        if anchor.get("coverage") != "counter_anchor":
            return "partial counter coverage"
        readings = anchor["readings"]
        reading_times = anchor["reading_times_utc"]
        from . import config

        tolerance = timedelta(
            minutes=getattr(config, "HEAT_FORECAST_ANCHOR_TOLERANCE_MINUTES", 5)
        )
        for name in ("heating_el", "heating_th", "defrost_el", "defrost_th"):
            value = readings.get(name)
            if value is None and name in ("defrost_el", "defrost_th"):
                continue
            if value is None or not math.isfinite(float(value)):
                return f"{name} counter missing or non-finite"
            stamp = datetime.fromisoformat(reading_times[name]).astimezone(timezone.utc)
            if abs(stamp - boundary) > tolerance:
                return f"{name} timestamp outside boundary tolerance"
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        return f"invalid anchor metadata: {exc}"
    return None


def actual_from_anchors(
    old: Mapping[str, Any] | None,
    new: Mapping[str, Any] | None,
    *,
    defrost_handling: str = "net",
    defrost_el_in_heating_counter: bool = True,
) -> tuple[datetime, float, float] | None:
    """Return aligned, defrost-normalized actuals for one hour."""
    if not isinstance(old, Mapping) or not isinstance(new, Mapping):
        LOGGER.debug("Forecast accuracy: waiting for adjacent hour anchors")
        return None
    try:
        for label, anchor in (("previous", old), ("current", new)):
            reason = anchor_rejection_reason(anchor)
            if reason:
                LOGGER.info(
                    "Forecast accuracy: %s anchor %s rejected: %s",
                    label, anchor.get("hour_utc"), reason,
                )
                return None
        old_hour = datetime.fromisoformat(old["hour_utc"])
        new_hour = datetime.fromisoformat(new["hour_utc"])
        if new_hour - old_hour != timedelta(hours=1):
            LOGGER.info("Forecast accuracy: rejected non-adjacent hour anchors")
            return None
        old_boundary = old_hour.astimezone(timezone.utc)
        readings = (old["readings"], new["readings"])

        def delta(name: str, *, optional: bool = False) -> float | None:
            start = readings[0].get(name)
            end = readings[1].get(name)
            if start is None or end is None:
                return 0.0 if optional and start is None and end is None else None
            start_value, end_value = float(start), float(end)
            if (
                not math.isfinite(start_value)
                or not math.isfinite(end_value)
                or end_value < start_value
            ):
                return None
            return end_value - start_value

        electricity = delta("heating_el")
        heat = delta("heating_th")
        defrost_electricity = delta("defrost_el", optional=True)
        defrost_heat = delta("defrost_th", optional=True)
        if None in (electricity, heat, defrost_electricity, defrost_heat):
            LOGGER.info(
                "Forecast accuracy: hour %s rejected: missing counter pair or counter reset",
                old["hour_utc"],
            )
            return None
        from .heat_forecast import cop_calibration_energies

        normalized_heat, normalized_electricity = cop_calibration_energies(
            heat_kwh=heat,
            electricity_kwh=electricity,
            defrost_heat_kwh=defrost_heat,
            defrost_electricity_kwh=defrost_electricity,
            defrost_el_in_heating_counter=defrost_el_in_heating_counter,
            defrost_handling=defrost_handling,
        )
        return old_boundary, normalized_electricity, normalized_heat
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        LOGGER.info("Forecast accuracy: invalid counter pair: %s", exc)
        return None


def record_actual(
    state: dict[str, Any], hour: datetime, el_kwh: float, th_kwh: float
) -> None:
    """Close all pending predictions of ``hour`` with the measured energy."""
    section = _section(state)
    hour_iso = hour.astimezone(timezone.utc).isoformat()
    for key in [k for k, v in section["pending"].items() if v["hour_utc"] == hour_iso]:
        item = section["pending"].pop(key)
        item["act_el"] = el_kwh
        item["act_th"] = th_kwh
        section["history"].append(item)
    LOGGER.info(
        "Forecast accuracy: hour %s closed, actual el=%.3f kWh th=%.3f kWh",
        hour_iso, el_kwh, th_kwh,
    )
    cutoff = hour.astimezone(timezone.utc) - timedelta(days=HISTORY_DAYS)
    section["history"] = [
        item
        for item in section["history"]
        if datetime.fromisoformat(item["hour_utc"]) >= cutoff
    ]


def _metrics(items: Sequence[Mapping[str, Any]], pred: str, act: str) -> dict[str, Any]:
    if not items:
        return {"n": 0}
    errors = [i[pred] - i[act] for i in items]
    significant = [i for i in items if i[act] > MISSED_ACTUAL_MIN]
    return {
        "n": len(items),
        "bias_kwh": round(sum(errors) / len(errors), 4),
        "mae_kwh": round(sum(abs(e) for e in errors) / len(errors), 4),
        "mape": (
            round(
                sum(abs(i[pred] - i[act]) / i[act] for i in significant)
                / len(significant),
                4,
            )
            if significant
            else None
        ),
        "missed_start_rate": (
            round(
                sum(1 for i in significant if i[pred] < MISSED_PRED_MAX)
                / len(significant),
                4,
            )
            if significant
            else None
        ),
        "false_start_rate": _false_rate(items, pred, act),
    }


def _false_rate(items: Sequence[Mapping[str, Any]], pred: str, act: str) -> float | None:
    predicted = [i for i in items if i[pred] > MISSED_ACTUAL_MIN]
    if not predicted:
        return None
    return round(
        sum(1 for i in predicted if i[act] < MISSED_PRED_MAX) / len(predicted), 4
    )


def summary(state: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    """Return per-lead metrics over 24 h and 7 d for the sensor attribute."""
    history = state.get("forecast_accuracy", {}).get("history", [])
    now_utc = now.astimezone(timezone.utc)
    result: dict[str, Any] = {}
    for lead in LEADS:
        rows = [h for h in history if h["lead"] == lead]
        windows = {}
        for label, delta in (("24h", timedelta(hours=24)), ("7d", timedelta(days=7))):
            recent = [
                h
                for h in rows
                if now_utc - datetime.fromisoformat(h["hour_utc"]) <= delta
            ]
            windows[label] = {
                "el": _metrics(recent, "el", "act_el"),
                "th": _metrics(recent, "th", "act_th"),
            }
        result[f"lead_{lead}h"] = windows
    return result


def correction_factor(
    state: Mapping[str, Any],
    now: datetime,
    *,
    window_hours: float = 36.0,
    limit: float = 0.2,
    min_samples: int = 12,
) -> float:
    """Multiplicative electricity correction = actual / forecast (lead 1 h).

    Predictions are stored uncorrected, so the factor never compounds. The
    result is bounded to ``1 +/- limit``; ``1.0`` without enough history.
    """
    history = state.get("forecast_accuracy", {}).get("history", [])
    cutoff = now.astimezone(timezone.utc) - timedelta(hours=window_hours)
    rows = []
    for item in history:
        try:
            if (
                item["lead"] == 1
                and datetime.fromisoformat(item["hour_utc"]) >= cutoff
                and item["el"] > 0
                and item["act_el"] >= 0
            ):
                rows.append(item)
        except (KeyError, TypeError, ValueError):
            continue
    if len(rows) < max(1, int(min_samples)):
        LOGGER.debug(
            "Forecast correction inactive: %d/%d usable hours", len(rows), min_samples
        )
        return 1.0
    predicted = sum(r["el"] for r in rows)
    actual = sum(r["act_el"] for r in rows)
    if predicted <= 0:
        return 1.0
    limit = max(0.0, float(limit))
    factor = min(1.0 + limit, max(1.0 - limit, actual / predicted))
    LOGGER.info(
        "Forecast correction factor %.3f from %d hours (actual %.2f / forecast %.2f kWh)",
        factor,
        len(rows),
        actual,
        predicted,
    )
    return factor
