"""
Tests for dashboard settings metadata and Supervisor API helpers.
"""

import os
import sys

import pytest


sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "dashboard")
)


from config_schema import GROUP_DEFINITIONS, load_settings_metadata
from settings_service import (
    SettingsServiceError,
    _sanitize_options,
    fetch_addon_options,
    get_default_options,
    update_addon_options,
)


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class TestSettingsMetadata:
    def test_metadata_covers_all_addon_options(self):
        metadata = load_settings_metadata()

        assert len(metadata.fields) == 208
        assert len(metadata.defaults) == 208
        assert len(metadata.field_order) == 208

    def test_expected_groups_and_translations_are_loaded(self):
        metadata = load_settings_metadata()

        assert metadata.fields["target_indoor_temp_entity"].group.slug == "core"
        assert metadata.fields["pv_power_entity"].group.slug == "solar"
        assert metadata.fields["heating_ml_blend_min_r2"].group.slug == "ml_heating"
        assert metadata.fields["trajectory_steps"].group.slug == "advanced"

        assert metadata.fields["target_indoor_temp_entity"].label.startswith("[Core]")
        assert metadata.fields["target_indoor_temp_entity"].de_label.startswith("[Kern]")
        assert metadata.fields["heating_ml_cv_enabled"].label
        assert metadata.fields["heating_ml_cv_enabled"].de_label

    def test_group_definitions_keep_expected_order(self):
        assert list(GROUP_DEFINITIONS) == [
            "core",
            "solar",
            "blocking",
            "ml",
            "safety",
            "cooling",
            "pre_cooling",
            "ml_pre_cooling",
            "ml_heating",
            "influxdb",
            "model",
            "dashboard",
            "dev",
            "advanced",
        ]


class TestSettingsService:
    def test_default_options_match_config_schema(self):
        defaults = get_default_options()

        assert defaults["debug"] is False
        assert defaults["dashboard_theme"] == "auto"
        assert defaults["trajectory_steps"] == 4

    def test_fetch_addon_options_uses_supervisor_api(self, monkeypatch):
        calls = {}

        def fake_get(url, headers=None, timeout=None):
            calls["url"] = url
            calls["headers"] = headers
            calls["timeout"] = timeout
            return DummyResponse({"data": {"options": {"debug": True}}})

        monkeypatch.setenv("SUPERVISOR_TOKEN", "test-token")
        monkeypatch.setattr("settings_service.requests.get", fake_get)

        options = fetch_addon_options()

        assert options["debug"] is True
        assert calls["url"].endswith("/addons/self/options")
        assert calls["headers"]["Authorization"] == "Bearer test-token"
        assert calls["timeout"] == 10

    def test_fetch_addon_options_ignores_unknown_keys(self, monkeypatch):
        def fake_get(url, headers=None, timeout=None):
            return DummyResponse(
                {"data": {"options": {"debug": True, "unknown_setting": "x"}}}
            )

        monkeypatch.setenv("SUPERVISOR_TOKEN", "test-token")
        monkeypatch.setattr("settings_service.requests.get", fake_get)

        options = fetch_addon_options()

        assert options["debug"] is True
        assert "unknown_setting" not in options

    def test_fetch_addon_options_requires_supervisor_token(self, monkeypatch):
        monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)

        with pytest.raises(SettingsServiceError):
            fetch_addon_options()

    def test_update_addon_options_posts_full_payload(self, monkeypatch):
        calls = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            calls["url"] = url
            calls["headers"] = headers
            calls["json"] = json
            calls["timeout"] = timeout
            return DummyResponse({"result": "ok"})

        monkeypatch.setenv("SUPERVISOR_TOKEN", "test-token")
        monkeypatch.setattr("settings_service.requests.post", fake_post)

        update_addon_options({"debug": True, "dashboard_theme": "dark"})

        assert calls["url"].endswith("/addons/self/options")
        assert calls["headers"]["Authorization"] == "Bearer test-token"
        assert calls["json"] == {
            "options": {"debug": True, "dashboard_theme": "dark"}
        }
        assert calls["timeout"] == 10

    def test_sanitize_options_drops_unknown_keys(self):
        sanitized = _sanitize_options(
            {"debug": True, "dashboard_theme": "dark", "unknown_setting": 123}
        )

        assert sanitized["debug"] is True
        assert sanitized["dashboard_theme"] == "dark"
        assert "unknown_setting" not in sanitized


class TestSettingsComponentHelpers:
    streamlit = pytest.importorskip("streamlit", reason="streamlit not installed")

    def test_coerce_bool_handles_string_values(self):
        from components.settings import _coerce_bool

        assert _coerce_bool("true") is True
        assert _coerce_bool("1") is True
        assert _coerce_bool("false") is False
        assert _coerce_bool("0") is False
