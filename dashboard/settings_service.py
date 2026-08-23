"""
Supervisor API access for dashboard settings.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import requests
import yaml

from config_schema import load_settings_metadata


_LOCAL_OPTIONS_PATH = Path("/data/options.json")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / "ml_heating_underfloor" / "config.yaml"

#: Profile keys that contain nested dicts — excluded from flat settings metadata
#: but handled explicitly by the profiles component.
PROFILE_KEYS = frozenset({"heating_profile", "cooling_profile"})


class SettingsServiceError(RuntimeError):
    """Raised when dashboard settings cannot be fetched or saved."""


def _get_supervisor_base_url() -> str:
    return os.environ.get("SUPERVISOR_URL", "http://supervisor").rstrip("/")


@lru_cache(maxsize=1)
def _load_raw_config() -> dict:
    """Parse config.yaml once and cache the result."""
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))


def _get_all_config_option_keys() -> frozenset[str]:
    """Return every top-level option key from config.yaml, including profile keys."""
    return frozenset(_load_raw_config().get("options", {}).keys())


def get_default_options() -> dict[str, object]:
    return dict(load_settings_metadata().defaults)


def get_profile_defaults() -> dict[str, dict[str, object]]:
    """Return the default values for heating_profile and cooling_profile from config.yaml."""
    options = _load_raw_config().get("options", {})
    return {
        key: dict(options[key])
        for key in ("heating_profile", "cooling_profile")
        if key in options and isinstance(options[key], dict)
    }


def _sanitize_options(options: dict[str, object]) -> dict[str, object]:
    all_keys = _get_all_config_option_keys()
    return {key: value for key, value in options.items() if key in all_keys}


def load_local_options() -> dict[str, object]:
    options: dict[str, object] = {**get_default_options(), **get_profile_defaults()}
    try:
        file_options = json.loads(_LOCAL_OPTIONS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return options
    if not isinstance(file_options, dict):
        return options
    options.update(_sanitize_options(file_options))
    return options


def _get_headers() -> dict[str, str]:
    token = os.environ.get("SUPERVISOR_TOKEN", "").strip()
    if not token:
        raise SettingsServiceError("SUPERVISOR_TOKEN is not available.")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def fetch_addon_options() -> dict[str, object]:
    response = requests.get(
        f"{_get_supervisor_base_url()}/addons/self/options",
        headers=_get_headers(),
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", {})
    options = data.get("options", data)
    if not isinstance(options, dict):
        return {**get_default_options(), **get_profile_defaults()}
    merged: dict[str, object] = {**get_default_options(), **get_profile_defaults()}
    merged.update(_sanitize_options(options))
    return merged


def update_addon_options(options: dict[str, object]) -> None:
    sanitized_options = _sanitize_options(options)
    response = requests.post(
        f"{_get_supervisor_base_url()}/addons/self/options",
        headers=_get_headers(),
        json={"options": sanitized_options},
        timeout=10,
    )
    response.raise_for_status()
