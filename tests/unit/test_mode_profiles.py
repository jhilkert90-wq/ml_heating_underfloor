"""Tests for src.mode_profiles."""

import io
import json
from unittest.mock import patch

from src import config
from src.mode_profiles import _load_raw_profiles, apply_profile


class TestApplyProfile:
    """apply_profile behavior."""

    @patch("src.mode_profiles._load_raw_profiles")
    def test_coerces_types_and_skips_unknown_keys(self, mock_load, monkeypatch):
        monkeypatch.setattr(config, "PRE_COOL_ENABLED", True)
        monkeypatch.setattr(config, "PRICE_TARGET_OFFSET", 0.0)
        monkeypatch.setattr(config, "PRICE_CHEAP_PERCENTILE", 5)
        if hasattr(config, "UNKNOWN_KEY"):
            delattr(config, "UNKNOWN_KEY")

        mock_load.return_value = {
            "heating": {
                "pre_cool_enabled": "false",
                "price_target_offset": "1.5",
                "price_cheap_percentile": "12",
                "unknown_key": "ignored",
            }
        }

        apply_profile("heating")

        assert config.PRE_COOL_ENABLED is False
        assert config.PRICE_TARGET_OFFSET == 1.5
        assert config.PRICE_CHEAP_PERCENTILE == 12
        assert not hasattr(config, "UNKNOWN_KEY")


class TestLoadRawProfiles:
    """_load_raw_profiles file search order."""

    @patch("src.mode_profiles.os.path.exists", side_effect=[True, True])
    def test_first_existing_profile_file_wins(self, mock_exists, monkeypatch):
        monkeypatch.setattr(
            "src.mode_profiles._OPTIONS_PATHS",
            ("/first.json", "/second.json"),
        )
        first = json.dumps({"heating_profile": {"pre_cool_enabled": False}})
        second = json.dumps({"heating_profile": {"pre_cool_enabled": True}})

        def _open_side_effect(path, encoding="utf-8"):
            if path == "/first.json":
                return io.StringIO(first)
            if path == "/second.json":
                return io.StringIO(second)
            raise FileNotFoundError(path)

        with patch("builtins.open", side_effect=_open_side_effect):
            profiles = _load_raw_profiles()

        assert profiles["heating"]["pre_cool_enabled"] is False
