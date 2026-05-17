"""
ML Heating Dashboard - Settings component.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from config_schema import FieldMetadata, load_settings_metadata
from settings_service import (
    SettingsServiceError,
    fetch_addon_options,
    load_local_options,
    update_addon_options,
)


def _load_current_options() -> tuple[dict[str, Any], str]:
    try:
        return fetch_addon_options(), "Supervisor API"
    except Exception:
        return load_local_options(), "local fallback"


def _ensure_current_options() -> None:
    if "settings_current_options" not in st.session_state:
        options, source = _load_current_options()
        st.session_state["settings_current_options"] = options
        st.session_state["settings_source"] = source


def _refresh_current_options() -> None:
    options, source = _load_current_options()
    st.session_state["settings_current_options"] = options
    st.session_state["settings_source"] = source
    st.session_state.pop("settings_pending_options", None)
    st.session_state.pop("settings_pending_changes", None)


def _render_field(field: FieldMetadata, value: Any) -> Any:
    widget_key = f"settings_field_{field.key}"
    if field.widget_type == "bool":
        return st.checkbox(
            field.de_label,
            value=bool(value),
            help=field.description,
            key=widget_key,
        )
    if field.widget_type == "select":
        options = list(field.options)
        selected = value if value in options else field.default
        return st.selectbox(
            field.de_label,
            options,
            index=options.index(selected),
            help=field.description,
            key=widget_key,
        )
    if field.widget_type == "int":
        current = int(value if value is not None else field.default)
        return st.number_input(
            field.de_label,
            min_value=int(field.min_value),
            max_value=int(field.max_value),
            value=current,
            step=int(field.step or 1),
            help=field.description,
            key=widget_key,
        )
    if field.widget_type == "float":
        current = float(value if value is not None else field.default)
        return st.number_input(
            field.de_label,
            min_value=float(field.min_value),
            max_value=float(field.max_value),
            value=current,
            step=float(field.step or 0.1),
            help=field.description,
            format="%.4f",
            key=widget_key,
        )
    return st.text_input(
        field.de_label,
        value="" if value is None else str(value),
        help=field.description,
        type="password" if field.is_secret else "default",
        key=widget_key,
    )


def _build_diff(current_options: dict[str, Any], candidate_options: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate_options.items()
        if current_options.get(key) != value
    }


def render_settings() -> None:
    """Render grouped settings management page."""
    _ensure_current_options()
    metadata = load_settings_metadata()
    current_options = st.session_state["settings_current_options"]

    st.subheader("⚙️ Einstellungen")
    st.caption(
        "Gruppierte Add-on-Konfiguration mit deutschen Labels und englischen Tooltips."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🔄 Aktuelle Werte neu laden", width="stretch"):
            _refresh_current_options()
            st.toast("Einstellungen neu geladen.", icon="🔄")
            st.rerun()
    with col2:
        st.info(f"Quelle: {st.session_state['settings_source']}")

    with st.form("dashboard_settings_form"):
        candidate_options: dict[str, Any] = {}
        for group in metadata.groups.values():
            group_fields = [
                metadata.fields[key]
                for key in metadata.field_order
                if metadata.fields[key].group.slug == group.slug
            ]
            if not group_fields:
                continue
            with st.expander(group.title_de, expanded=group.expanded):
                for field in group_fields:
                    candidate_options[field.key] = _render_field(
                        field,
                        current_options.get(field.key, field.default),
                    )
        review_changes = st.form_submit_button("Änderungen prüfen", type="primary")

    if review_changes:
        changes = _build_diff(current_options, candidate_options)
        if not changes:
            st.info("Keine Änderungen erkannt.")
            st.session_state.pop("settings_pending_options", None)
            st.session_state.pop("settings_pending_changes", None)
        else:
            pending_options = dict(current_options)
            pending_options.update(changes)
            st.session_state["settings_pending_options"] = pending_options
            st.session_state["settings_pending_changes"] = changes

    pending_options = st.session_state.get("settings_pending_options")
    pending_changes = st.session_state.get("settings_pending_changes")
    if pending_options and pending_changes:
        st.warning("Bitte prüfen und bestätigen Sie die Änderungen vor dem Speichern.")
        for key, new_value in pending_changes.items():
            field = metadata.fields[key]
            st.write(
                f"**{field.de_label}**  \n"
                f"`{current_options.get(key)}` → `{new_value}`"
            )

        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            if st.button("✅ Änderungen speichern", type="primary", width="stretch"):
                try:
                    update_addon_options(pending_options)
                    st.session_state["settings_current_options"] = pending_options
                    st.session_state["settings_source"] = "Supervisor API"
                    st.session_state.pop("settings_pending_options", None)
                    st.session_state.pop("settings_pending_changes", None)
                    st.toast("Einstellungen gespeichert.", icon="✅")
                    st.success(
                        "Die Add-on-Optionen wurden gespeichert. Ein Neustart des Add-ons kann erforderlich sein."
                    )
                    st.rerun()
                except SettingsServiceError as exc:
                    st.toast(f"Speichern fehlgeschlagen: {exc}", icon="⚠️")
                    st.error(str(exc))
                except Exception as exc:
                    st.toast(f"Speichern fehlgeschlagen: {exc}", icon="⚠️")
                    st.error(f"Speichern fehlgeschlagen: {exc}")
        with cancel_col:
            if st.button("Abbrechen", width="stretch"):
                st.session_state.pop("settings_pending_options", None)
                st.session_state.pop("settings_pending_changes", None)
                st.rerun()
