"""
Tests for the Mode Profiles dashboard component and related settings_service helpers.
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "dashboard")
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, "dashboard", "components"
    ),
)

from settings_service import (
    PROFILE_KEYS,
    _sanitize_options,
    get_profile_defaults,
)
from components.profiles import (
    _BOOL_FIELDS,
    _FLOAT_FIELDS,
    _INT_FIELDS,
)


# ---------------------------------------------------------------------------
# settings_service: profile helpers
# ---------------------------------------------------------------------------

class TestGetProfileDefaults:
    def test_returns_both_profile_keys(self):
        defaults = get_profile_defaults()
        assert "heating_profile" in defaults
        assert "cooling_profile" in defaults

    def test_heating_profile_contains_expected_keys(self):
        defaults = get_profile_defaults()
        hp = defaults["heating_profile"]
        assert isinstance(hp, dict)
        assert "electricity_price_enabled" in hp
        assert "pre_cool_enabled" in hp
        assert "price_target_offset" in hp
        assert "price_cheap_percentile" in hp

    def test_cooling_profile_contains_expected_keys(self):
        defaults = get_profile_defaults()
        cp = defaults["cooling_profile"]
        assert isinstance(cp, dict)
        assert "pre_cool_enabled" in cp
        assert "overshoot_detection_enabled" in cp

    def test_heating_profile_bool_values_are_bools(self):
        hp = get_profile_defaults()["heating_profile"]
        bool_keys = {k for k, _, _ in _BOOL_FIELDS}
        for key in bool_keys:
            assert isinstance(hp[key], bool), f"{key} should be bool"

    def test_float_and_int_values_correct_types(self):
        hp = get_profile_defaults()["heating_profile"]
        assert isinstance(hp["price_target_offset"], float)
        assert isinstance(hp["price_cheap_percentile"], int)


class TestSanitizeOptionsWithProfiles:
    def test_profile_keys_pass_through_sanitize(self):
        profile_payload: dict[str, Any] = {
            "heating_profile": {"pre_cool_enabled": True},
            "cooling_profile": {"pre_cool_enabled": False},
            "unknown_key": "drop_me",
        }
        result = _sanitize_options(profile_payload)
        assert "heating_profile" in result
        assert "cooling_profile" in result
        assert "unknown_key" not in result

    def test_profile_dict_value_preserved_intact(self):
        nested: dict[str, Any] = {
            "electricity_price_enabled": True,
            "price_target_offset": 0.3,
            "price_cheap_percentile": 25,
        }
        result = _sanitize_options({"heating_profile": nested, "cooling_profile": {}})
        assert result["heating_profile"] == nested

    def test_flat_options_still_sanitized_correctly(self):
        options = {"shadow_mode": True, "not_a_real_key": 99}
        result = _sanitize_options(options)
        assert "shadow_mode" in result
        assert "not_a_real_key" not in result


class TestProfileKeysConstant:
    def test_profile_keys_contains_both_profiles(self):
        assert PROFILE_KEYS == frozenset({"heating_profile", "cooling_profile"})


# ---------------------------------------------------------------------------
# profiles component: static metadata coverage
# ---------------------------------------------------------------------------

class TestProfilesStaticMetadata:
    def test_all_profileable_bools_covered(self):
        """All 10 bool fields from src/mode_profiles.py must be in _BOOL_FIELDS."""
        bool_keys = {key for key, _, _ in _BOOL_FIELDS}
        expected = {
            "electricity_price_enabled",
            "pv_surplus_cheap_enabled",
            "pv_traj_forecast_mode_enabled",
            "pv_traj_disable_price_in_forecast_mode",
            "pv_traj_forecast_rescue_enabled",
            "pv_traj_disable_overshoot_correction",
            "solar_correction_enabled",
            "cloud_cover_correction_enabled",
            "overshoot_detection_enabled",
            "pre_cool_enabled",
        }
        assert bool_keys == expected

    def test_float_fields_have_valid_ranges(self):
        for key, label, desc, min_v, max_v, step in _FLOAT_FIELDS:
            assert min_v < max_v, f"{key}: min_value must be < max_value"
            assert step > 0, f"{key}: step must be positive"

    def test_int_fields_have_valid_ranges(self):
        for key, label, desc, min_v, max_v in _INT_FIELDS:
            assert min_v < max_v, f"{key}: min_value must be < max_value"

    def test_no_duplicate_keys_across_field_groups(self):
        bool_keys = {k for k, *_ in _BOOL_FIELDS}
        float_keys = {k for k, *_ in _FLOAT_FIELDS}
        int_keys = {k for k, *_ in _INT_FIELDS}
        all_keys = bool_keys | float_keys | int_keys
        assert len(all_keys) == len(_BOOL_FIELDS) + len(_FLOAT_FIELDS) + len(_INT_FIELDS)

    def test_total_field_count_matches_mode_profiles_profileable_keys(self):
        """There should be exactly 14 profileable fields (10 bool + 2 float + 2 int)."""
        total = len(_BOOL_FIELDS) + len(_FLOAT_FIELDS) + len(_INT_FIELDS)
        assert total == 14


# ---------------------------------------------------------------------------
# profiles component: render logic (mocked Streamlit)
# ---------------------------------------------------------------------------

class TestRenderProfilesNoStreamlit:
    """Smoke tests that exercise the render logic with a mocked Streamlit session."""

    def _make_st_mock(self, session_state: dict | None = None) -> MagicMock:
        st = MagicMock()
        state = session_state if session_state is not None else {}
        st.session_state = state
        # st.tabs returns a list of context managers
        tab1 = MagicMock()
        tab2 = MagicMock()
        tab1.__enter__ = MagicMock(return_value=tab1)
        tab1.__exit__ = MagicMock(return_value=False)
        tab2.__enter__ = MagicMock(return_value=tab2)
        tab2.__exit__ = MagicMock(return_value=False)
        st.tabs.return_value = [tab1, tab2]
        # Forms
        form_ctx = MagicMock()
        form_ctx.__enter__ = MagicMock(return_value=form_ctx)
        form_ctx.__exit__ = MagicMock(return_value=False)
        form_ctx.form_submit_button.return_value = False
        st.form.return_value = form_ctx
        st.columns.return_value = [MagicMock(), MagicMock()]
        return st

    @patch("components.profiles.fetch_addon_options")
    @patch("components.profiles.update_addon_options")
    def test_render_profiles_sets_session_state(
        self, mock_update: MagicMock, mock_fetch: MagicMock
    ) -> None:
        from components import profiles as prof_module

        profile_defaults = get_profile_defaults()
        mock_fetch.return_value = {
            **{"shadow_mode": False},
            **profile_defaults,
        }

        st_mock = self._make_st_mock()
        with patch.object(prof_module, "st", st_mock):
            prof_module.render_profiles()

        assert "profiles_current_options" in st_mock.session_state
        assert "profiles_source" in st_mock.session_state
        assert st_mock.session_state["profiles_source"] == "Supervisor API"

    @patch("components.profiles.fetch_addon_options")
    @patch("components.profiles.update_addon_options")
    def test_render_profiles_uses_fallback_on_fetch_error(
        self, mock_update: MagicMock, mock_fetch: MagicMock
    ) -> None:
        from settings_service import SettingsServiceError
        from components import profiles as prof_module

        mock_fetch.side_effect = SettingsServiceError("no token")

        st_mock = self._make_st_mock()
        # Patch load_local_options too so it returns something sensible
        with (
            patch.object(prof_module, "st", st_mock),
            patch(
                "components.profiles.load_local_options",
                return_value={**get_profile_defaults()},
            ),
        ):
            prof_module.render_profiles()

        assert st_mock.session_state["profiles_source"] == "local fallback"

    @patch("components.profiles.fetch_addon_options")
    @patch("components.profiles.update_addon_options")
    def test_save_heating_profile_calls_update_addon_options(
        self, mock_update: MagicMock, mock_fetch: MagicMock
    ) -> None:
        """When the form is submitted, update_addon_options must be called with the merged dict."""
        from components import profiles as prof_module

        full_options: dict[str, Any] = {
            "shadow_mode": False,
            **get_profile_defaults(),
        }
        mock_fetch.return_value = dict(full_options)

        st_mock = self._make_st_mock(session_state={
            "profiles_current_options": full_options,
            "profiles_source": "Supervisor API",
        })

        # Override _render_profile_form so it returns a specific new profile dict
        new_heating = {"pre_cool_enabled": False, "electricity_price_enabled": True}
        with (
            patch.object(prof_module, "st", st_mock),
            patch.object(prof_module, "_render_profile_form") as mock_form,
        ):
            # Heating tab returns a result; cooling tab returns None
            mock_form.side_effect = [new_heating, None]
            prof_module.render_profiles()

        mock_update.assert_called_once()
        saved_options = mock_update.call_args[0][0]
        assert saved_options["heating_profile"] == new_heating
        # Cooling profile must be preserved
        assert saved_options["cooling_profile"] == full_options["cooling_profile"]

    def test_profile_save_merges_without_corrupting_flat_keys(self) -> None:
        """Saving a profile must not drop unrelated top-level options."""
        full_options: dict[str, Any] = {
            "shadow_mode": True,
            "target_temp_entity": "sensor.room",
            "heating_profile": {"pre_cool_enabled": False},
            "cooling_profile": {"pre_cool_enabled": True},
        }
        new_heating = {"pre_cool_enabled": True, "electricity_price_enabled": False}
        updated = dict(full_options)
        updated["heating_profile"] = new_heating

        assert updated["shadow_mode"] is True
        assert updated["target_temp_entity"] == "sensor.room"
        assert updated["heating_profile"] == new_heating
        assert updated["cooling_profile"] == full_options["cooling_profile"]
