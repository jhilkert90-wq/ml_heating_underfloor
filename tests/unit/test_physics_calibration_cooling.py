"""
Tests for the cooling physics calibration path (physics_calibration_cooling).
"""

from __future__ import annotations

import pytest
from datetime import timezone
from unittest.mock import patch


# ---------------------------------------------------------------------------
# TestCoolingPhysicsStartDate
# ---------------------------------------------------------------------------

class TestCoolingPhysicsStartDate:
    """Tests for COOLING_PHYSICS_CALIBRATION_START_DATE resolution in calibrate_cooling_physics."""

    def test_parse_helper_valid_date(self):
        from src.config import _parse_cooling_physics_start_date
        dt = _parse_cooling_physics_start_date("15.06.2021")
        assert dt is not None
        assert dt.year == 2021 and dt.month == 6 and dt.day == 15
        assert dt.tzinfo == timezone.utc

    def test_parse_helper_empty_string(self):
        from src.config import _parse_cooling_physics_start_date
        assert _parse_cooling_physics_start_date("") is None

    def test_parse_helper_invalid_format(self):
        from src.config import _parse_cooling_physics_start_date
        assert _parse_cooling_physics_start_date("2021-06-15") is None
        assert _parse_cooling_physics_start_date("not-a-date") is None

    def test_lookback_overridden_by_past_date(self):
        """calibrate_cooling_physics resolves lookback_hours from a past start date."""
        from src.physics_calibration_cooling import calibrate_cooling_physics
        from src.config import _parse_cooling_physics_start_date

        captured = {}

        def fake_fetch(lookback_hours, **kwargs):
            captured["lookback_hours"] = lookback_hours
            return None  # trigger early return

        with patch(
            "src.physics_calibration_cooling.fetch_historical_data_for_calibration",
            side_effect=fake_fetch,
        ), patch(
            "src.physics_calibration_cooling.config"
        ) as mock_cfg:
            mock_cfg.TRAINING_LOOKBACK_HOURS = 168
            mock_cfg.COOLING_PHYSICS_CALIBRATION_START_DATE = "01.06.2021"
            mock_cfg._parse_cooling_physics_start_date = _parse_cooling_physics_start_date
            calibrate_cooling_physics()

        assert "lookback_hours" in captured
        # Should be many thousands of hours since mid-2021
        assert captured["lookback_hours"] > 8760

    def test_empty_start_date_uses_default_double(self):
        """Empty COOLING_PHYSICS_CALIBRATION_START_DATE falls back to TRAINING_LOOKBACK_HOURS × 2."""
        from src.physics_calibration_cooling import calibrate_cooling_physics

        captured = {}

        def fake_fetch(lookback_hours, **kwargs):
            captured["lookback_hours"] = lookback_hours
            return None

        with patch(
            "src.physics_calibration_cooling.fetch_historical_data_for_calibration",
            side_effect=fake_fetch,
        ), patch(
            "src.physics_calibration_cooling.config"
        ) as mock_cfg:
            mock_cfg.TRAINING_LOOKBACK_HOURS = 300
            mock_cfg.COOLING_PHYSICS_CALIBRATION_START_DATE = ""
            calibrate_cooling_physics()

        assert captured.get("lookback_hours") == 600  # 300 × 2

    def test_future_date_uses_default(self):
        """Future COOLING_PHYSICS_CALIBRATION_START_DATE falls back to default, logs warning."""
        from src.physics_calibration_cooling import calibrate_cooling_physics
        from src.config import _parse_cooling_physics_start_date

        captured = {}

        def fake_fetch(lookback_hours, **kwargs):
            captured["lookback_hours"] = lookback_hours
            return None

        with patch(
            "src.physics_calibration_cooling.fetch_historical_data_for_calibration",
            side_effect=fake_fetch,
        ), patch(
            "src.physics_calibration_cooling.config"
        ) as mock_cfg:
            mock_cfg.TRAINING_LOOKBACK_HOURS = 168
            mock_cfg.COOLING_PHYSICS_CALIBRATION_START_DATE = "01.01.2099"
            mock_cfg._parse_cooling_physics_start_date = _parse_cooling_physics_start_date
            calibrate_cooling_physics()

        assert captured.get("lookback_hours") == 336  # 168 × 2

    def test_invalid_date_format_uses_default(self):
        """Invalid COOLING_PHYSICS_CALIBRATION_START_DATE falls back to default, logs warning."""
        from src.physics_calibration_cooling import calibrate_cooling_physics
        from src.config import _parse_cooling_physics_start_date

        captured = {}

        def fake_fetch(lookback_hours, **kwargs):
            captured["lookback_hours"] = lookback_hours
            return None

        with patch(
            "src.physics_calibration_cooling.fetch_historical_data_for_calibration",
            side_effect=fake_fetch,
        ), patch(
            "src.physics_calibration_cooling.config"
        ) as mock_cfg:
            mock_cfg.TRAINING_LOOKBACK_HOURS = 168
            mock_cfg.COOLING_PHYSICS_CALIBRATION_START_DATE = "2021/06/15"
            mock_cfg._parse_cooling_physics_start_date = _parse_cooling_physics_start_date
            calibrate_cooling_physics()

        assert captured.get("lookback_hours") == 336  # 168 × 2
