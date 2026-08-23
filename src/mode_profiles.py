"""Mode profiles: per-mode overlay of feature-flag and tuning settings.

At startup, :func:`apply_profile` reads the ``heating_profile`` or
``cooling_profile`` block from ``/data/options.json`` (or a separate
``/data/config/mode_profiles.json`` as fallback) and patches the live
``config`` module globals in-process.

This means all downstream code that reads e.g. ``config.ELECTRICITY_PRICE_ENABLED``
automatically picks up the profile settings for the active mode without any
further changes to those modules.

Only keys listed in :data:`PROFILEABLE_KEYS` are eligible for override; unknown
keys are logged at DEBUG level and skipped so that a mis-typed profile entry
cannot corrupt unrelated settings.

Profile blocks in ``options.json`` use lower-case option names (as written in
``config.yaml``), which are **automatically upper-cased** before being matched
against the config module constants.

Example ``options.json`` fragment::

    "heating_profile": {
        "electricity_price_enabled": true,
        "pv_surplus_cheap_enabled": true,
        "price_target_offset": 0.3,
        "overshoot_detection_enabled": true
    },
    "cooling_profile": {
        "electricity_price_enabled": false,
        "pv_surplus_cheap_enabled": false,
        "pre_cool_enabled": true,
        "price_target_offset": 0.0
    }
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

# ---------------------------------------------------------------------------
# Source file search order
# ---------------------------------------------------------------------------

#: Locations searched in order.  The first file that contains at least one
#: recognised profile block wins.
_OPTIONS_PATHS: tuple[str, ...] = (
    "/data/options.json",
    "/data/config/mode_profiles.json",
)

# ---------------------------------------------------------------------------
# Registry of config constants that profiles may override
# ---------------------------------------------------------------------------

#: Boolean flags eligible for per-mode override.
PROFILEABLE_BOOLS: frozenset[str] = frozenset(
    {
        "ELECTRICITY_PRICE_ENABLED",
        "PV_SURPLUS_CHEAP_ENABLED",
        "PV_TRAJ_FORECAST_MODE_ENABLED",
        "PV_TRAJ_DISABLE_PRICE_IN_FORECAST_MODE",
        "PV_TRAJ_FORECAST_RESCUE_ENABLED",
        "PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION",
        "SOLAR_CORRECTION_ENABLED",
        "CLOUD_COVER_CORRECTION_ENABLED",
        "OVERSHOOT_DETECTION_ENABLED",
        "PRE_COOL_ENABLED",
    }
)

#: Numeric (float) constants eligible for per-mode override.
PROFILEABLE_FLOATS: frozenset[str] = frozenset(
    {
        "PRICE_TARGET_OFFSET",
        "PRICE_EXPENSIVE_OVERSHOOT",
    }
)

#: Numeric (int) constants eligible for per-mode override.
PROFILEABLE_INTS: frozenset[str] = frozenset(
    {
        "PRICE_CHEAP_PERCENTILE",
        "PRICE_EXPENSIVE_PERCENTILE",
    }
)

#: All constants eligible for per-mode override.
PROFILEABLE_KEYS: frozenset[str] = PROFILEABLE_BOOLS | PROFILEABLE_FLOATS | PROFILEABLE_INTS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_raw_profiles() -> dict[str, dict[str, Any]]:
    """Return a dict mapping mode names to raw profile dicts.

    Returned keys are ``"heating"`` and/or ``"cooling"``.  Returns ``{}``
    if no profile data is found in any of :data:`_OPTIONS_PATHS`.
    """
    for path in _OPTIONS_PATHS:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("⚠️ mode_profiles: could not read %s: %s", path, exc)
            continue

        result: dict[str, dict[str, Any]] = {}
        for profile_name in ("heating_profile", "cooling_profile"):
            raw = data.get(profile_name)
            if isinstance(raw, dict) and raw:
                mode_key = profile_name.removesuffix("_profile")  # "heating" / "cooling"
                result[mode_key] = raw
        if result:
            logging.debug(
                "📋 Mode profiles loaded from %s: %s", path, list(result)
            )
            return result

    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_profile(climate_mode: str) -> None:
    """Overlay the matching mode profile onto ``config`` module globals.

    Parameters
    ----------
    climate_mode:
        The currently active climate mode (``"heating"`` or ``"cooling"``).
        Any other value (e.g. ``"off"``) is a no-op.

    Only keys in :data:`PROFILEABLE_KEYS` are applied.  Profile entries that
    use lower-case names (as stored in ``options.json``) are automatically
    upper-cased before lookup.  Values are coerced to the canonical type
    (``bool`` or ``float``) before being set.  Unknown/disallowed keys are
    logged at DEBUG level and skipped.
    """
    if climate_mode not in ("heating", "cooling"):
        return

    from . import config as _cfg  # late import — avoids circular dependency

    profiles = _load_raw_profiles()
    profile = profiles.get(climate_mode)
    if not profile:
        logging.debug(
            "📋 No %s profile defined — using default config values", climate_mode
        )
        return

    applied: list[str] = []
    skipped: list[str] = []

    for raw_key, raw_value in profile.items():
        # Accept both lower-case (options.json style) and UPPER_CASE keys.
        key = raw_key.upper()

        if key not in PROFILEABLE_KEYS:
            skipped.append(raw_key)
            continue

        try:
            if key in PROFILEABLE_BOOLS:
                if isinstance(raw_value, bool):
                    value: Any = raw_value
                else:
                    value = str(raw_value).lower().strip() in ("true", "1", "yes")
            elif key in PROFILEABLE_INTS:
                value = int(raw_value)
            else:
                # PROFILEABLE_FLOATS
                value = float(raw_value)
        except (TypeError, ValueError) as exc:
            logging.warning(
                "⚠️ Profile: cannot coerce %s=%r to expected type — skipping (%s)",
                key,
                raw_value,
                exc,
            )
            continue

        setattr(_cfg, key, value)
        applied.append(f"{key}={value!r}")

    if skipped:
        logging.debug(
            "📋 Profile: ignored unknown/non-profileable keys: %s", skipped
        )

    if applied:
        mode_icon = "🔥" if climate_mode == "heating" else "❄️"
        logging.info(
            "%s %s profile applied (%d settings): %s",
            mode_icon,
            climate_mode.capitalize(),
            len(applied),
            ", ".join(applied),
        )
    else:
        logging.debug(
            "📋 %s profile found but contained no eligible keys", climate_mode
        )
