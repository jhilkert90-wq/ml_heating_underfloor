"""Forecast-vs-actual accuracy tracking for the WP load forecast.

Predictions are stored per (target hour, lead time) with first write wins.
Once the hourly counter energy of that hour is known, the pair is moved to a
bounded history from which bias/MAE/MAPE and start-detection rates derive.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

LEADS = (1, 6, 24)
HISTORY_DAYS = 7
MISSED_PRED_MAX = 0.1
MISSED_ACTUAL_MIN = 0.2
ANCHOR_TOLERANCE = timedelta(minutes=20)


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
    cutoff = now.astimezone(timezone.utc) - timedelta(hours=48)
    section["pending"] = {
        key: value
        for key, value in pending.items()
        if datetime.fromisoformat(value["hour_utc"]) >= cutoff
    }


def actual_from_anchors(
    old: Mapping[str, Any] | None, new: Mapping[str, Any] | None
) -> tuple[datetime, float, float] | None:
    """Return (hour, el, th) when two consecutive hour anchors are exact."""
    if not isinstance(old, Mapping) or not isinstance(new, Mapping):
        return None
    try:
        old_hour = datetime.fromisoformat(old["hour_utc"])
        new_hour = datetime.fromisoformat(new["hour_utc"])
        if new_hour - old_hour != timedelta(hours=1):
            return None
        if old.get("coverage") != "counter_anchor" or new.get(
            "coverage"
        ) != "counter_anchor":
            return None
        for anchor, boundary in ((old, old_hour), (new, new_hour)):
            stamp = datetime.fromisoformat(anchor["reading_times_utc"]["heating_el"])
            if abs(stamp - boundary) > ANCHOR_TOLERANCE:
                return None
        values = []
        for name in ("heating_el", "heating_th"):
            start = old["readings"].get(name)
            end = new["readings"].get(name)
            if start is None or end is None or end < start:
                return None
            values.append(float(end) - float(start))
        return old_hour, values[0], values[1]
    except (KeyError, TypeError, ValueError, AttributeError):
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
