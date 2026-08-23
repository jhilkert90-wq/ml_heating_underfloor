"""
ML Heating Dashboard - Mode Profiles component.

Provides a UI for viewing and editing the heating_profile and cooling_profile
option blocks.  Each profile is a per-mode overlay that overrides specific
feature-flag and tuning settings while the HVAC operates in that mode.

Profile keys are optional (nullable): when a key is absent the add-on falls
back to its flat top-level value.  The UI expresses this via a per-field
"Override in this profile" checkbox — untick to remove the key from the
saved profile dict.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from settings_service import (
    SettingsServiceError,
    fetch_addon_options,
    get_profile_defaults,
    load_local_options,
    update_addon_options,
)


# ---------------------------------------------------------------------------
# Static metadata for the 14 profileable settings
# ---------------------------------------------------------------------------

_BOOL_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("electricity_price_enabled", "Electricity Price Enabled",
     "Enable Tibber electricity price optimisation in this mode."),
    ("pv_surplus_cheap_enabled", "PV Surplus Cheap Enabled",
     "Treat excess PV production as 'cheap' energy in this mode."),
    ("pv_traj_forecast_mode_enabled", "PV Trajectory Forecast Mode",
     "Use PV trajectory forecast to guide heating decisions."),
    ("pv_traj_disable_price_in_forecast_mode", "Disable Price in Forecast Mode",
     "Ignore price signal while PV trajectory forecast mode is active."),
    ("pv_traj_forecast_rescue_enabled", "PV Forecast Rescue",
     "Allow the trajectory rescue logic when forecast mode is active."),
    ("pv_traj_disable_overshoot_correction", "Disable Overshoot Correction",
     "Skip overshoot correction while PV trajectory forecast mode is active."),
    ("solar_correction_enabled", "Solar Correction Enabled",
     "Apply indoor solar heat-gain correction in this mode."),
    ("cloud_cover_correction_enabled", "Cloud Cover Correction Enabled",
     "Scale solar correction by cloud cover in this mode."),
    ("overshoot_detection_enabled", "Overshoot Detection Enabled",
     "Enable temperature overshoot detection and intervention in this mode."),
    ("pre_cool_enabled", "Pre-Cool Enabled",
     "Enable predictive pre-cooling before expected overheating in this mode."),
)

_FLOAT_FIELDS: tuple[tuple[str, str, str, float, float, float], ...] = (
    ("price_target_offset", "Price Target Offset",
     "Raise the indoor setpoint by this many °C during cheap-price windows.",
     0.0, 1.0, 0.05),
    ("price_expensive_overshoot", "Price Expensive Overshoot",
     "Allow the indoor temperature to exceed the setpoint by this °C during expensive windows.",
     0.0, 1.0, 0.05),
)

_INT_FIELDS: tuple[tuple[str, str, str, int, int], ...] = (
    ("price_cheap_percentile", "Cheap Price Percentile",
     "Prices below this percentile (of the daily distribution) are treated as cheap.",
     10, 50),
    ("price_expensive_percentile", "Expensive Price Percentile",
     "Prices above this percentile (of the daily distribution) are treated as expensive.",
     50, 90),
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_current_options() -> tuple[dict[str, Any], str]:
    try:
        return fetch_addon_options(), "Supervisor API"
    except Exception:
        return load_local_options(), "local fallback"


def _ensure_current_options() -> None:
    if "profiles_current_options" not in st.session_state:
        options, source = _load_current_options()
        st.session_state["profiles_current_options"] = options
        st.session_state["profiles_source"] = source


def _refresh_current_options() -> None:
    options, source = _load_current_options()
    st.session_state["profiles_current_options"] = options
    st.session_state["profiles_source"] = source
    st.session_state.pop("profiles_pending_heating", None)
    st.session_state.pop("profiles_pending_cooling", None)


def _render_profile_form(
    profile_key: str,
    current_profile: dict[str, Any],
    defaults: dict[str, Any],
    form_key: str,
) -> dict[str, Any] | None:
    """Render a form for one profile dict.  Returns the new profile dict on submit, else None."""
    with st.form(form_key):
        st.markdown(
            "Enable the **Override** checkbox to include a key in the profile. "
            "Un-tick it to remove the key (the add-on will fall back to the flat top-level value)."
        )

        new_profile: dict[str, Any] = {}

        # --- Bool fields ---
        st.markdown("**Feature Flags**")
        for key, label, description in _BOOL_FIELDS:
            is_overridden = key in current_profile
            default_val = defaults.get(key, False)
            current_val = current_profile.get(key, default_val)

            col_check, col_widget = st.columns([1, 3])
            with col_check:
                override = st.checkbox(
                    "Override",
                    value=is_overridden,
                    key=f"{form_key}_{key}_override",
                    help=f"Include `{key}` in this profile.",
                )
            with col_widget:
                widget_disabled = not override
                val = st.checkbox(
                    label,
                    value=bool(current_val),
                    disabled=widget_disabled,
                    help=description,
                    key=f"{form_key}_{key}_value",
                )
            if override:
                new_profile[key] = val

        st.divider()

        # --- Float fields ---
        st.markdown("**Numeric Tuning**")
        for key, label, description, min_v, max_v, step in _FLOAT_FIELDS:
            is_overridden = key in current_profile
            default_val = float(defaults.get(key, 0.0))
            current_val = float(current_profile.get(key, default_val))

            col_check, col_widget = st.columns([1, 3])
            with col_check:
                override = st.checkbox(
                    "Override",
                    value=is_overridden,
                    key=f"{form_key}_{key}_override",
                    help=f"Include `{key}` in this profile.",
                )
            with col_widget:
                val = st.number_input(
                    label,
                    min_value=float(min_v),
                    max_value=float(max_v),
                    value=current_val,
                    step=float(step),
                    disabled=not override,
                    format="%.2f",
                    help=description,
                    key=f"{form_key}_{key}_value",
                )
            if override:
                new_profile[key] = float(val)

        for key, label, description, min_v, max_v in _INT_FIELDS:
            is_overridden = key in current_profile
            default_val = int(defaults.get(key, min_v))
            current_val = int(current_profile.get(key, default_val))

            col_check, col_widget = st.columns([1, 3])
            with col_check:
                override = st.checkbox(
                    "Override",
                    value=is_overridden,
                    key=f"{form_key}_{key}_override",
                    help=f"Include `{key}` in this profile.",
                )
            with col_widget:
                val = st.number_input(
                    label,
                    min_value=int(min_v),
                    max_value=int(max_v),
                    value=current_val,
                    step=1,
                    disabled=not override,
                    help=description,
                    key=f"{form_key}_{key}_value",
                )
            if override:
                new_profile[key] = int(val)

        submitted = st.form_submit_button("💾 Save profile", type="primary")

    if submitted:
        return new_profile
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_profiles() -> None:
    """Render the Mode Profiles management page."""
    _ensure_current_options()
    current_options: dict[str, Any] = st.session_state["profiles_current_options"]
    profile_defaults = get_profile_defaults()

    st.subheader("🔀 Mode Profiles")
    st.caption(
        "Per-mode setting overrides applied at add-on startup.  "
        "Heating profile is active while the HVAC is in **heating** mode; "
        "cooling profile is active while in **cooling** mode."
    )

    col_reload, col_source = st.columns([1, 2])
    with col_reload:
        if st.button("🔄 Reload current values", width="stretch"):
            _refresh_current_options()
            st.toast("Profiles reloaded.", icon="🔄")
            st.rerun()
    with col_source:
        st.info(f"Source: {st.session_state['profiles_source']}")

    st.info(
        "⚠️ Profile changes take effect on the **next add-on restart**. "
        "A warm restart is triggered automatically when the HVAC switches modes."
    )

    tab_heat, tab_cool = st.tabs(["🔥 Heating Profile", "❄️ Cooling Profile"])

    with tab_heat:
        current_heating = dict(current_options.get("heating_profile") or {})
        defaults_heating = profile_defaults.get("heating_profile", {})
        result = _render_profile_form(
            "heating_profile",
            current_heating,
            defaults_heating,
            "profiles_form_heating",
        )
        if result is not None:
            updated = dict(current_options)
            updated["heating_profile"] = result
            try:
                update_addon_options(updated)
                st.session_state["profiles_current_options"] = updated
                st.toast("Heating profile saved.", icon="🔥")
                st.success(
                    "Heating profile saved. Restart the add-on (or wait for the "
                    "next mode transition) for the changes to take effect."
                )
                st.rerun()
            except SettingsServiceError as exc:
                st.toast(f"Save failed: {exc}", icon="⚠️")
                st.error(str(exc))
            except Exception as exc:
                st.toast(f"Save failed: {exc}", icon="⚠️")
                st.error(f"Save failed: {exc}")

    with tab_cool:
        current_cooling = dict(current_options.get("cooling_profile") or {})
        defaults_cooling = profile_defaults.get("cooling_profile", {})
        result = _render_profile_form(
            "cooling_profile",
            current_cooling,
            defaults_cooling,
            "profiles_form_cooling",
        )
        if result is not None:
            updated = dict(current_options)
            updated["cooling_profile"] = result
            try:
                update_addon_options(updated)
                st.session_state["profiles_current_options"] = updated
                st.toast("Cooling profile saved.", icon="❄️")
                st.success(
                    "Cooling profile saved. Restart the add-on (or wait for the "
                    "next mode transition) for the changes to take effect."
                )
                st.rerun()
            except SettingsServiceError as exc:
                st.toast(f"Save failed: {exc}", icon="⚠️")
                st.error(str(exc))
            except Exception as exc:
                st.toast(f"Save failed: {exc}", icon="⚠️")
                st.error(f"Save failed: {exc}")
