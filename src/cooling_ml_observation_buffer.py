"""
cooling_ml_observation_buffer.py
---------------------------------
Rolling labeled-observation store for the LGBM pre-cooling classifier.

Each observation is one cooling-mode cycle snapshot plus a boolean label
that is resolved PRE_COOL_HORIZON_HOURS later once the true indoor peak
is known:

  label = 1  if  max(indoor_temp[t : t + horizon_steps]) > cooling_target
  label = 0  otherwise

The buffer is persisted to JSON so that labels can be attached
asynchronously (at observation time +horizon), and the model-training code
can reload history across restarts.

Thread safety: all public methods hold a threading.Lock.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CoolingObservationBuffer:
    """
    Sliding-window buffer of labeled cooling observations.

    Lifecycle of a single entry
    ---------------------------
    1. ``push_pending(features, indoor_temp, cooling_target, timestamp)``
       → entry appended with ``label=None``
    2. After ``horizon_steps`` cycles have elapsed,
       ``resolve_labels(current_indoor)`` is called each cycle and
       fills in labels for mature pending entries.
    3. Once labeled, entries count towards ``n_labeled`` and eventually
       trigger a retrain when enough new ones have accumulated.
    4. When the buffer exceeds ``max_n``, the oldest labeled entries are evicted.
    """

    def __init__(
        self,
        path: str,
        max_n: int = 500,
        min_training_samples: int = 200,
        retrain_trigger_k: int = 50,
        horizon_steps: int = 72,  # PRE_COOL_HORIZON_HOURS * steps_per_hour
    ) -> None:
        self._path = path
        self._max_n = max_n
        self._min_training_samples = min_training_samples
        self._retrain_trigger_k = retrain_trigger_k
        self._horizon_steps = horizon_steps

        self._lock = threading.RLock()
        # List[dict] — each entry:
        #   features: dict[str, float]
        #   indoor_temp: float          (snapshot at push time)
        #   cooling_target: float       (snapshot at push time)
        #   timestamp: str (ISO-8601)
        #   label: 0 | 1 | None
        #   max_indoor_seen: float      (running max since push; used for label resolution)
        #   steps_elapsed: int          (cycles since push; counts up to horizon_steps)
        self._entries: list[dict[str, Any]] = []
        self._labeled_since_last_train: int = 0

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_pending(
        self,
        features: dict[str, float],
        indoor_temp: float,
        cooling_target: float,
        timestamp: str,
    ) -> None:
        """Add a new pending (unlabeled) observation."""
        entry: dict[str, Any] = {
            "features": dict(features),
            "indoor_temp": indoor_temp,
            "cooling_target": cooling_target,
            "timestamp": timestamp,
            "label": None,
            "max_indoor_seen": indoor_temp,
            "steps_elapsed": 0,
        }
        with self._lock:
            self._entries.append(entry)
            self._evict()

    def resolve_labels(self, current_indoor: float) -> int:
        """
        Call once per cycle with the current indoor temperature.
        Updates ``max_indoor_seen`` for all pending entries and assigns a
        label when an entry reaches ``horizon_steps``.

        Returns the number of entries that received a label this call.
        """
        newly_labeled = 0
        with self._lock:
            for entry in self._entries:
                if entry["label"] is not None:
                    continue
                entry["steps_elapsed"] += 1
                entry["max_indoor_seen"] = max(
                    entry["max_indoor_seen"], current_indoor
                )
                if entry["steps_elapsed"] >= self._horizon_steps:
                    label = int(
                        entry["max_indoor_seen"] > entry["cooling_target"]
                    )
                    entry["label"] = label
                    self._labeled_since_last_train += 1
                    newly_labeled += 1
        return newly_labeled

    def should_retrain(self) -> bool:
        """True when enough newly labeled samples have accumulated."""
        with self._lock:
            return (
                self.n_labeled >= self._min_training_samples
                and self._labeled_since_last_train >= self._retrain_trigger_k
            )

    def reset_retrain_counter(self) -> None:
        with self._lock:
            self._labeled_since_last_train = 0

    def get_labeled_data(self) -> tuple[list[dict[str, float]], list[int]]:
        """Return (feature_dicts, labels) for all labeled entries."""
        with self._lock:
            labeled = [e for e in self._entries if e["label"] is not None]
        feature_dicts = [e["features"] for e in labeled]
        labels = [int(e["label"]) for e in labeled]
        return feature_dicts, labels

    @property
    def n_labeled(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries if e["label"] is not None)

    @property
    def n_pending(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries if e["label"] is None)

    @property
    def n_total(self) -> int:
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        with self._lock:
            # Snapshot entries inside the lock so that _evict() (which can set
            # list slots to None before reassigning self._entries) cannot race
            # with json.dump() running outside the lock.
            payload = {
                "max_n": self._max_n,
                "min_training_samples": self._min_training_samples,
                "retrain_trigger_k": self._retrain_trigger_k,
                "horizon_steps": self._horizon_steps,
                "labeled_since_last_train": self._labeled_since_last_train,
                "entries": [_sanitize_for_json(dict(e)) for e in self._entries],
            }
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=_json_default)
            os.replace(tmp, self._path)
        except Exception:
            logger.exception("CoolingObservationBuffer: failed to save %s", self._path)

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self._entries = payload.get("entries", [])
            self._labeled_since_last_train = payload.get(
                "labeled_since_last_train", 0
            )
            # Honour saved horizon_steps so pending labels are consistent.
            self._horizon_steps = payload.get("horizon_steps", self._horizon_steps)
            logger.info(
                "CoolingObservationBuffer: loaded %d entries (%d labeled) from %s",
                len(self._entries),
                self.n_labeled,
                self._path,
            )
        except Exception:
            logger.exception(
                "CoolingObservationBuffer: could not load %s, starting fresh",
                self._path,
            )
            self._entries = []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evict(self) -> None:
        """Remove oldest entries when the buffer exceeds max_n.

        Preference: evict labeled entries first (their data has been used for
        the retrain counter).  If there are not enough labeled entries to
        bring the buffer back to ``max_n``, also evict the oldest pending
        entries to prevent unbounded growth.  Adjust
        ``_labeled_since_last_train`` for any labeled entries removed.
        """
        if len(self._entries) <= self._max_n:
            return
        to_remove = len(self._entries) - self._max_n

        # Prefer evicting labeled entries first
        labeled_indices = [
            i for i, e in enumerate(self._entries) if e["label"] is not None
        ]
        evict_labeled = labeled_indices[:to_remove]
        remaining = to_remove - len(evict_labeled)

        # If still over budget, also evict oldest pending entries
        if remaining > 0:
            pending_indices = [
                i for i, e in enumerate(self._entries) if e["label"] is None
            ]
            evict_pending = pending_indices[:remaining]
        else:
            evict_pending = []

        evict_set = set(evict_labeled) | set(evict_pending)
        # Decrement the retrain counter for labeled entries being removed
        self._labeled_since_last_train = max(
            0, self._labeled_since_last_train - len(evict_labeled)
        )
        self._entries = [e for i, e in enumerate(self._entries) if i not in evict_set]


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None for JSON-safe serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _json_default(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
