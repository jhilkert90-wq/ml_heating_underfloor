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
    def test_forecast_dhw_option_is_persistable_checkbox_with_help(self):
        field = load_settings_metadata().fields["heat_forecast_include_dhw"]
        assert field.widget_type == "bool"
        assert field.default is True
        assert "EOS Connect" in field.description
        description = field.description.lower()
        assert any(term in description for term in ("warmwasser", "hot-water"))
        assert any(term in description for term in ("start", "bereitung"))
        assert _sanitize_options({"heat_forecast_include_dhw": False}) == {
            "heat_forecast_include_dhw": False
        }

    def test_metadata_covers_all_addon_options(self):
        metadata = load_settings_metadata()

        _SKIPPED_KEYS = {"heating_profile", "cooling_profile"}
        assert len(metadata.fields) == len(metadata.defaults) - len(_SKIPPED_KEYS)
        assert len(metadata.field_order) == len(metadata.fields)
        assert set(metadata.defaults.keys()) - set(metadata.fields.keys()) == _SKIPPED_KEYS

    def test_expected_groups_and_translations_are_loaded(self):
        metadata = load_settings_metadata()

        assert metadata.fields["target_indoor_temp_entity"].group.slug == "core"
        assert metadata.fields["pv_power_entity"].group.slug == "heat_sources"
        assert metadata.fields["heating_ml_cv_enabled"].group.slug == "ml_heating"
        assert metadata.fields["trajectory_steps"].group.slug == "trajectory"
        assert metadata.fields["grace_period_max_minutes"].group.slug == "blocking"
        assert metadata.fields["grace_stagnation_window_minutes"].group.slug == "blocking"
        assert metadata.fields["grace_stagnation_min_drop_k"].group.slug == "blocking"
        assert metadata.fields["building_balance_method"].group.slug == "balance"
        assert metadata.fields["building_curve_enabled"].group.slug == "balance"
        assert metadata.fields["building_balance_method"].options == (
            "modelled",
            "measured",
            "measured_with_fallback",
        )

        assert metadata.fields["target_indoor_temp_entity"].label.startswith("[Core]")
        assert metadata.fields["heating_ml_cv_enabled"].label
        assert metadata.fields["cooling_physics_min_outdoor_rolling_24h_c"].label
        assert metadata.fields["cooling_physics_min_outdoor_rolling_24h_c"].description
        assert metadata.fields["outlet_effectiveness_mode"].widget_type == "select"
        assert metadata.fields["outlet_effectiveness_mode"].options == (
            "scalar",
            "curve",
        )

    def test_group_definitions_keep_expected_order(self):
        assert list(GROUP_DEFINITIONS) == [
            "core",
            "blocking",
            "safety",
            "learning",
            "thermal_model",
            "heat_sources",
            "trajectory",
            "functions",
            "cooling",
            "pre_cooling",
            "ml_pre_cooling",
            "ml_heating",
            "ml_cooling_correction",
            "hlc",
            "balance",
            "wp_forecast",
            "price_pv",
            "shadow",
            "outlet",
            "influxdb",
            "model",
            "dashboard",
            "dev",
            "advanced",
        ]


class TestSettingsService:
    def test_forecast_dhw_option_round_trips_via_supervisor(self, monkeypatch):
        stored = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            stored.update(json["options"])
            return DummyResponse({})

        def fake_get(url, headers=None, timeout=None):
            return DummyResponse({"data": {"options": stored}})

        monkeypatch.setenv("SUPERVISOR_TOKEN", "test-token")
        monkeypatch.setattr("settings_service.requests.post", fake_post)
        monkeypatch.setattr("settings_service.requests.get", fake_get)
        assert fetch_addon_options()["heat_forecast_include_dhw"] is True
        update_addon_options({"heat_forecast_include_dhw": False})
        assert fetch_addon_options()["heat_forecast_include_dhw"] is False

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

    def test_forecast_dhw_checkbox_displays_explanation(self, monkeypatch):
        from components import settings

        captions = []
        monkeypatch.setattr(settings.st, "checkbox", lambda *args, **kwargs: False)
        monkeypatch.setattr(settings.st, "caption", captions.append)
        field = load_settings_metadata().fields["heat_forecast_include_dhw"]
        assert settings._render_field(field, True) is False
        assert captions == [field.description]

    def test_coerce_bool_handles_string_values(self):
        from components.settings import _coerce_bool

        assert _coerce_bool("true") is True
        assert _coerce_bool("1") is True
        assert _coerce_bool("false") is False
        assert _coerce_bool("0") is False


def test_wp_forecast_options_have_own_dashboard_group():
    metadata = load_settings_metadata()
    assert "wp_forecast" in GROUP_DEFINITIONS
    for key in (
        "heat_forecast_enabled",
        "heat_forecast_near_controller_enabled",
        "heat_forecast_electricity_counter_entity",
        "cop_calibration_enabled",
        "hp_start_estimate_enabled",
        "hp_start_estimate_heating_entity_id",
        "hp_start_estimate_use_for_grace",
    ):
        assert metadata.fields[key].group.slug == "wp_forecast", key
    assert metadata.fields["price_cost_max_overshoot_k"].group.slug == "price_pv"
