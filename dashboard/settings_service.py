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
_SUPERVISOR_BASE_URL = os.environ.get("SUPERVISOR_URL", "http://supervisor").rstrip("/")


class SettingsServiceError(RuntimeError):
    """Raised when dashboard settings cannot be fetched or saved."""


def get_default_options() -> dict[str, object]:
    return dict(load_settings_metadata().defaults)


def load_local_options() -> dict[str, object]:
    options = get_default_options()
    try:
        file_options = json.loads(_LOCAL_OPTIONS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return options
    options.update(file_options)
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
        f"{_SUPERVISOR_BASE_URL}/addons/self/options",
        headers=_get_headers(),
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", {})
    options = data.get("options", data)
    merged = get_default_options()
    if isinstance(options, dict):
        merged.update(options)
    return merged


def update_addon_options(options: dict[str, object]) -> None:
    response = requests.post(
        f"{_SUPERVISOR_BASE_URL}/addons/self/options",
        headers=_get_headers(),
        json={"options": options},
        timeout=10,
    )
    response.raise_for_status()
