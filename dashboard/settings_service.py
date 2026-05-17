"""
Supervisor API access for dashboard settings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from config_schema import load_settings_metadata


_LOCAL_OPTIONS_PATH = Path("/data/options.json")


class SettingsServiceError(RuntimeError):
    """Raised when dashboard settings cannot be fetched or saved."""


def _get_supervisor_base_url() -> str:
    return os.environ.get("SUPERVISOR_URL", "http://supervisor").rstrip("/")


def get_default_options() -> dict[str, object]:
    return dict(load_settings_metadata().defaults)


def _sanitize_options(options: dict[str, object]) -> dict[str, object]:
    defaults = get_default_options()
    return {key: value for key, value in options.items() if key in defaults}


def load_local_options() -> dict[str, object]:
    options = get_default_options()
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
        return get_default_options()
    merged = get_default_options()
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
