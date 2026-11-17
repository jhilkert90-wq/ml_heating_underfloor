"""Diagnostic dump for the WP load forecast accuracy pipeline.

Writes ``forecast_accuracy_debug.json`` next to the heat forecast state file
(the same folder as ``unified_thermal_state.json``) so an empty
``accuracy`` attribute can be diagnosed from the file alone: raw counter
samples per source, how each boundary reading was derived, why anchors were
rejected, and a rolling log of the last decisions. Never raises.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

FILE_NAME = "forecast_accuracy_debug.json"
MAX_EVENTS = 72
SAMPLE_HEAD = 4
SAMPLE_TAIL = 6


def debug_path(state_path: Path | str) -> Path:
    return Path(state_path).parent / FILE_NAME


def summarize_samples(
    samples: Sequence[tuple[datetime, float]] | None,
) -> dict[str, Any]:
    """Compact, readable view of one entity's raw history samples."""
    rows = sorted(
        (stamp.astimezone(timezone.utc), value) for stamp, value in samples or []
    )
    fmt = [[stamp.isoformat(), value] for stamp, value in rows]
    if len(fmt) > SAMPLE_HEAD + SAMPLE_TAIL:
        return {
            "n": len(fmt),
            "first": fmt[:SAMPLE_HEAD],
            "last": fmt[-SAMPLE_TAIL:],
        }
    return {"n": len(fmt), "samples": fmt}


def _anchor_view(anchor: Any, reason: str | None) -> dict[str, Any]:
    if not isinstance(anchor, Mapping):
        return {"present": False, "rejection": reason}
    return {
        "present": True,
        "hour_utc": anchor.get("hour_utc"),
        "coverage": anchor.get("coverage"),
        "rejection": reason,
        "readings": anchor.get("readings"),
        "reading_times_utc": anchor.get("reading_times_utc"),
        "reading_methods": anchor.get("reading_methods"),
    }


def _load(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_snapshot(
    state_path: Path | str,
    state: Mapping[str, Any],
    *,
    now: datetime,
    reason_fn: Any,
    fetch_debug: Mapping[str, Any] | None,
    outcome: str,
    summary_fn: Any = None,
) -> None:
    """Persist the current accuracy diagnostics; swallow every error."""
    try:
        path = debug_path(state_path)
        previous = _load(path)
        events = list(previous.get("events", []))[-(MAX_EVENTS - 1):]
        accuracy = state.get("forecast_accuracy", {}) or {}
        anchor = state.get("hour_anchor")
        prev_anchor = state.get("previous_hour_anchor")
        snapshot: dict[str, Any] = {
            "written_at_utc": now.astimezone(timezone.utc).isoformat(),
            "outcome": outcome,
            "hour_anchor": _anchor_view(anchor, reason_fn(anchor)),
            "previous_hour_anchor": _anchor_view(prev_anchor, reason_fn(prev_anchor)),
            "pending_predictions": len(accuracy.get("pending", {})),
            "history_rows": len(accuracy.get("history", [])),
            "history_hours": sorted(
                {h.get("hour_utc") for h in accuracy.get("history", [])}
            )[-12:],
            "fetch": dict(fetch_debug or {}),
        }
        if summary_fn is not None:
            snapshot["summary"] = summary_fn(state, now)
        events.append(
            {
                "at_utc": snapshot["written_at_utc"],
                "outcome": outcome,
                "anchor_hour": (anchor or {}).get("hour_utc")
                if isinstance(anchor, Mapping) else None,
                "anchor_rejection": snapshot["hour_anchor"]["rejection"],
                "previous_hour": (prev_anchor or {}).get("hour_utc")
                if isinstance(prev_anchor, Mapping) else None,
                "previous_rejection": snapshot["previous_hour_anchor"]["rejection"],
                "history_rows": snapshot["history_rows"],
            }
        )
        snapshot["events"] = events
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2, default=str, sort_keys=False)
        os.replace(tmp, path)
    except Exception as exc:  # diagnostics must never break control
        LOGGER.debug("Forecast accuracy debug dump failed: %s", exc)
