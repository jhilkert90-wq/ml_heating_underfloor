"""
Dashboard Data Service - Centralized Real Data Access

Reads real data from the unified thermal state JSON file and provides
structured data for all dashboard components. Eliminates simulated/demo
data by extracting actual learning metrics, parameters, and history
from the ML system's persisted state.
"""

import json
import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default state file locations (addon vs local development)
_STATE_FILE_CANDIDATES = [
    os.environ.get("UNIFIED_STATE_FILE", ""),
    "/config/ml_heating/unified_thermal_state.json",
    "/opt/ml_heating/unified_thermal_state.json",
    "/data/models/unified_thermal_state.json",
    os.path.join(os.path.dirname(os.path.dirname(__file__)),
                 "unified_thermal_state.json"),
]

_SHADOW_SUFFIX = "_shadow"


def _shadow_variant(path: str) -> str:
    """Return the shadow-mode variant of a state file path.

    Inserts ``_shadow`` before the file extension, e.g.
    ``unified_thermal_state.json`` → ``unified_thermal_state_shadow.json``.
    """
    if not path:
        return path
    root, ext = os.path.splitext(path)
    if not ext:
        return f"{path}{_SHADOW_SUFFIX}"
    return f"{root}{_SHADOW_SUFFIX}{ext}"


def _find_state_file() -> Optional[str]:
    """Locate the unified thermal state JSON file.

    For each candidate path the shadow-mode variant
    (e.g. ``*_shadow.json``) is checked first so that the dashboard
    automatically picks up shadow deployments.
    """
    for candidate in _STATE_FILE_CANDIDATES:
        if not candidate:
            continue
        # Shadow deployments write to a suffixed file – prefer it.
        shadow = _shadow_variant(candidate)
        if os.path.isfile(shadow):
            return shadow
        if os.path.isfile(candidate):
            return candidate
    return None


def load_thermal_state() -> Optional[Dict[str, Any]]:
    """Load the unified thermal state from JSON.

    Returns the full state dict, or ``None`` when the file cannot be found
    or parsed.
    """
    path = _find_state_file()
    if path is None:
        logger.warning("Unified thermal state file not found")
        return None
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load thermal state from %s: %s", path, exc)
        return None


# ------------------------------------------------------------------
# Metric helpers consumed by overview / performance components
# ------------------------------------------------------------------

def get_system_metrics() -> Dict[str, Any]:
    """Return key system metrics extracted from the real state file.

    Returns a dict with keys: confidence, mae, rmse, cycle_count,
    last_prediction, status.  Values default to 0/unknown when the
    state file is unavailable.
    """
    state = load_thermal_state()
    if state is None:
        return _empty_metrics()

    learning = state.get("learning_state", {})
    pred_metrics = state.get("prediction_metrics", {})
    accuracy = pred_metrics.get("accuracy_stats", {})
    recent = pred_metrics.get("recent_performance", {})
    operational = state.get("operational_state", {})
    metadata = state.get("metadata", {})

    cycle_count = learning.get("cycle_count", 0)
    confidence = learning.get("learning_confidence", 0.0)

    mae = accuracy.get("mae_all_time", recent.get("last_10_mae", 0.0))
    rmse = accuracy.get("rmse_all_time", 0.0)

    last_prediction = operational.get("last_prediction")
    if last_prediction is None:
        last_prediction = operational.get("last_final_temp", 0.0)
    last_prediction = last_prediction or 0.0

    # Derive a simple status from operational data
    last_run = operational.get("last_run_time")
    if last_run:
        try:
            run_dt = datetime.fromisoformat(str(last_run))
            age_seconds = (datetime.now() - run_dt).total_seconds()
            if age_seconds < 600:
                status = "active"
            elif age_seconds < 3600:
                status = "idle"
            else:
                status = "stale"
        except (ValueError, TypeError):
            status = "unknown"
    else:
        status = "unknown"

    return {
        "confidence": confidence,
        "mae": mae,
        "rmse": rmse,
        "cycle_count": cycle_count,
        "last_prediction": last_prediction,
        "status": status,
    }


def _empty_metrics() -> Dict[str, Any]:
    return {
        "confidence": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
        "cycle_count": 0,
        "last_prediction": 0.0,
        "status": "unavailable",
    }


# ------------------------------------------------------------------
# Prediction & parameter history helpers
# ------------------------------------------------------------------

def get_prediction_history() -> List[Dict[str, Any]]:
    """Return the prediction history list from the state file."""
    state = load_thermal_state()
    if state is None:
        return []
    return state.get("learning_state", {}).get("prediction_history", [])


def get_parameter_history() -> List[Dict[str, Any]]:
    """Return the parameter history list from the state file."""
    state = load_thermal_state()
    if state is None:
        return []
    return state.get("learning_state", {}).get("parameter_history", [])


# ------------------------------------------------------------------
# Baseline & learned parameters
# ------------------------------------------------------------------

def get_baseline_parameters() -> Dict[str, Any]:
    """Return the baseline thermal parameters dict."""
    state = load_thermal_state()
    if state is None:
        return {}
    return state.get("baseline_parameters", {})


def get_parameter_adjustments() -> Dict[str, float]:
    """Return current learning deltas."""
    state = load_thermal_state()
    if state is None:
        return {}
    return (
        state.get("learning_state", {}).get("parameter_adjustments", {})
    )


def get_effective_parameters() -> Dict[str, float]:
    """Compute effective parameters = baseline + deltas."""
    baseline = get_baseline_parameters()
    deltas = get_parameter_adjustments()
    effective: Dict[str, float] = {}
    for key, value in baseline.items():
        if not isinstance(value, (int, float)):
            continue
        delta_key = f"{key}_delta"
        delta = deltas.get(delta_key, 0.0)
        effective[key] = value + delta
    return effective


# ------------------------------------------------------------------
# Heat source channel helpers
# ------------------------------------------------------------------

def get_heat_source_channels() -> Dict[str, Any]:
    """Return heat-source channel data from the state file."""
    state = load_thermal_state()
    if state is None:
        return {}
    return (
        state.get("learning_state", {}).get("heat_source_channels", {})
    )


def get_channel_summary() -> List[Dict[str, Any]]:
    """Return a summary list of each heat-source channel."""
    channels = get_heat_source_channels()
    summary = []
    for name, data in channels.items():
        params = data.get("parameters", {})
        history = data.get("history", [])
        history_count = data.get("history_count", len(history))

        recent_errors = [
            abs(h.get("error", 0.0)) for h in history[-20:]
        ]
        avg_recent_error = (
            sum(recent_errors) / len(recent_errors)
            if recent_errors else 0.0
        )

        summary.append({
            "channel": name,
            "parameters": params,
            "history_count": history_count,
            "recent_avg_abs_error": round(avg_recent_error, 4),
        })
    return summary


# ------------------------------------------------------------------
# Metadata & operational state
# ------------------------------------------------------------------

def get_metadata() -> Dict[str, Any]:
    """Return metadata section from state file."""
    state = load_thermal_state()
    if state is None:
        return {}
    return state.get("metadata", {})


def get_operational_state() -> Dict[str, Any]:
    """Return operational state section."""
    state = load_thermal_state()
    if state is None:
        return {}
    return state.get("operational_state", {})


def get_state_file_info() -> Optional[Dict[str, Any]]:
    """Return file-system info about the state file (path, size, mtime)."""
    path = _find_state_file()
    if path is None:
        return None
    try:
        stat = os.stat(path)
        return {
            "path": path,
            "size_bytes": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 1),
            "last_modified": datetime.fromtimestamp(stat.st_mtime),
        }
    except OSError:
        return None
