"""
heating_correction_ml_observation_buffer.py
---------------------------------------------
Rolling labeled-observation store for the LightGBM heating-correction regressor.

Each observation is one heating-mode cycle snapshot plus a float regression
label that is resolved ``label_horizon_steps`` later once the true future
indoor temperature is known:

    label[t] = -(T_indoor[t + N_steps] - T_target) / S_H

    S_H = (η / (η + U)) × (1 − exp(−H / τ_room))

A positive label means the outlet should have been raised (undershoot).
A negative label means the outlet should have been lowered (overshoot).

The buffer is persisted to JSON so that labels can be attached
asynchronously (at observation time + horizon), and the model-training code
can reload history across restarts.

Design choices (compared with CoolingObservationBuffer):
- Labels are floats (regression), not binary integers.
- ``resolve_labels()`` accepts the current S_H value computed by the caller
  so that the label reflects the *current* calibrated thermal parameters
  (answer 2b: recompute at resolve-time).
- Observations are collected for every heating-mode cycle regardless of
  HEATING_CORRECTION_MODE (answer 1: always collect).
- There is no cold-season gate in the buffer itself; the per-cycle push is
  unconditional on the climate side — calibration filtering remains in
  ``calibrate_heating_correction_ml()`` (answer 3: collect all heating).

Thread safety: all public methods hold a threading.RLock.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Maximum regression label magnitude (°C / S_H unit).
# Values beyond this are almost certainly noise or sensor error.
_LABEL_CLIP = 10.0


class HeatingCorrectionObservationBuffer:
    """
    Sliding-window buffer of labeled heating-correction observations.

    Lifecycle of a single entry
    ---------------------------
    1. ``push_pending(features, indoor_temp, heating_target, timestamp)``
       → entry appended with ``label=None``
    2. After ``horizon_steps`` cycles have elapsed,
       ``resolve_labels(current_indoor, s_h)`` assigns the regression label.
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
        horizon_steps: int = 24,  # HEATING_ML_LABEL_HORIZON_H * steps_per_hour
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
        #   heating_target: float       (snapshot at push time)
        #   timestamp: str (ISO-8601)
        #   label: float | None
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
        heating_target: float,
        timestamp: str,
    ) -> None:
        """Add a new pending (unlabeled) observation."""
        entry: dict[str, Any] = {
            "features": dict(features),
            "indoor_temp": indoor_temp,
            "heating_target": heating_target,
            "timestamp": timestamp,
            "label": None,
            "steps_elapsed": 0,
        }
        with self._lock:
            self._entries.append(entry)
            self._evict()

    def resolve_labels(self, current_indoor: float, s_h: float) -> int:
        """
        Call once per cycle with the current indoor temperature and the
        current sensitivity S_H computed from calibrated thermal parameters.

        Increments ``steps_elapsed`` for all pending entries.  When an entry
        reaches ``horizon_steps``, assigns:

            label = clip(-(current_indoor - heating_target) / s_h, ±_LABEL_CLIP)

        Returns the number of entries that received a label this call.

        Parameters
        ----------
        current_indoor:
            Measured indoor temperature at this cycle [°C].
        s_h:
            Physics sensitivity S_H = (η / (η+U)) × (1 − exp(−H/τ)).
            If ≤ 0 the label would be undefined; affected entries are
            skipped and will be resolved on the next cycle that has a
            valid s_h.
        """
        newly_labeled = 0
        with self._lock:
            for entry in self._entries:
                if entry["label"] is not None:
                    continue
                entry["steps_elapsed"] += 1
                if entry["steps_elapsed"] >= self._horizon_steps:
                    if s_h <= 0.0:
                        # S_H degenerate — defer labeling until params available
                        logger.debug(
                            "HeatingCorrectionObservationBuffer: s_h=%.4f ≤ 0 "
                            "— deferring label for entry at %s",
                            s_h,
                            entry.get("timestamp", "?"),
                        )
                        # Don't increment steps_elapsed beyond horizon so we
                        # retry next cycle without over-aging the entry.
                        entry["steps_elapsed"] = self._horizon_steps
                        continue
                    raw_label = -(current_indoor - entry["heating_target"]) / s_h
                    entry["label"] = float(
                        max(-_LABEL_CLIP, min(_LABEL_CLIP, raw_label))
                    )
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

    def get_labeled_data(self) -> tuple[list[dict[str, float]], list[float]]:
        """Return (feature_dicts, labels) for all labeled entries."""
        with self._lock:
            labeled = [e for e in self._entries if e["label"] is not None]
        feature_dicts = [e["features"] for e in labeled]
        labels = [float(e["label"]) for e in labeled]
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
            logger.exception(
                "HeatingCorrectionObservationBuffer: failed to save %s", self._path
            )

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
            # Honour saved horizon_steps so pending labels stay consistent.
            self._horizon_steps = payload.get("horizon_steps", self._horizon_steps)
            logger.info(
                "HeatingCorrectionObservationBuffer: loaded %d entries "
                "(%d labeled) from %s",
                len(self._entries),
                self.n_labeled,
                self._path,
            )
        except Exception:
            logger.exception(
                "HeatingCorrectionObservationBuffer: could not load %s, "
                "starting fresh",
                self._path,
            )
            self._entries = []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evict(self) -> None:
        """Remove oldest entries when the buffer exceeds max_n.

        Preference: evict labeled entries first (their data has been used for
        the retrain counter).  If there are not enough labeled entries to bring
        the buffer back to ``max_n``, also evict the oldest pending entries to
        prevent unbounded growth.  Adjusts ``_labeled_since_last_train`` for
        any labeled entries removed.
        """
        if len(self._entries) <= self._max_n:
            return
        to_remove = len(self._entries) - self._max_n

        labeled_indices = [
            i for i, e in enumerate(self._entries) if e["label"] is not None
        ]
        evict_labeled = labeled_indices[:to_remove]
        remaining = to_remove - len(evict_labeled)

        if remaining > 0:
            pending_indices = [
                i for i, e in enumerate(self._entries) if e["label"] is None
            ]
            evict_pending = pending_indices[:remaining]
        else:
            evict_pending = []

        evict_set = set(evict_labeled) | set(evict_pending)
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
