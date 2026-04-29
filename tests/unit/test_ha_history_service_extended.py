"""Extended tests for ha_history_service.py – _build_entity_map, edge cases."""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ha_history_service import (
    _build_entity_map,
    _ha_history_to_dataframe,
    _is_binary_entity,
    _parse_state_value,
    compute_cloud_proxy,
)


def _ts(minutes_ago: int) -> str:
    """ISO timestamp *minutes_ago* minutes in the past (UTC)."""
    return (
        datetime.now(tz=timezone.utc) - timedelta(minutes=minutes_ago)
    ).isoformat()


# ---------------------------------------------------------------------------
# _build_entity_map
# ---------------------------------------------------------------------------
class TestBuildEntityMap:
    def test_returns_dict(self):
        from src import config
        entity_map = _build_entity_map()
        assert isinstance(entity_map, dict)

    def test_all_values_are_short_names(self):
        """Short name should equal entity_id.split('.', 1)[-1]."""
        entity_map = _build_entity_map()
        for eid, short_name in entity_map.items():
            assert short_name == eid.split(".", 1)[-1]

    def test_required_entities_present(self):
        from src import config
        entity_map = _build_entity_map()
        # Some core entities that must always be present
        assert config.INDOOR_TEMP_ENTITY_ID in entity_map
        assert config.OUTDOOR_TEMP_ENTITY_ID in entity_map
        assert config.ACTUAL_OUTLET_TEMP_ENTITY_ID in entity_map

    def test_optional_living_room_included_when_configured(self):
        """If LIVING_ROOM_TEMP_ENTITY_ID is set, it appears in the map."""
        from src import config
        with patch.object(config, "LIVING_ROOM_TEMP_ENTITY_ID", "sensor.living_room"):
            entity_map = _build_entity_map()
            assert "sensor.living_room" in entity_map

    def test_optional_living_room_absent_when_none(self):
        from src import config
        original_living_room = getattr(config, "LIVING_ROOM_TEMP_ENTITY_ID", None)
        baseline_entity_map = _build_entity_map()

        with patch.object(config, "LIVING_ROOM_TEMP_ENTITY_ID", None):
            entity_map = _build_entity_map()

        if original_living_room:
            expected_entity_map = {
                eid: short_name
                for eid, short_name in baseline_entity_map.items()
                if eid != original_living_room
            }
            assert entity_map == expected_entity_map
            assert original_living_room not in entity_map
        else:
            assert entity_map == baseline_entity_map


# ---------------------------------------------------------------------------
# _parse_state_value – additional edge cases
# ---------------------------------------------------------------------------
class TestParseStateValueExtended:
    def test_empty_string_returns_nan(self):
        assert math.isnan(_parse_state_value("", False))

    def test_binary_true_string(self):
        assert _parse_state_value("true", True) == 1.0

    def test_binary_one_string(self):
        assert _parse_state_value("1", True) == 1.0

    def test_binary_false_returns_zero(self):
        assert _parse_state_value("False", True) == 0.0

    def test_non_numeric_non_binary_returns_nan(self):
        assert math.isnan(_parse_state_value("unavailable_state", False))


# ---------------------------------------------------------------------------
# _ha_history_to_dataframe – edge cases
# ---------------------------------------------------------------------------
class TestHaHistoryToDataframeEdgeCases:
    def test_empty_raw_histories_returns_empty_df(self):
        df = _ha_history_to_dataframe([], {}, [])
        assert df.empty

    def test_duplicate_timestamps_deduplicated(self):
        """Duplicate last_changed timestamps should be deduplicated (keep last)."""
        entity_ids = ["sensor.temp"]
        entity_map = {"sensor.temp": "temp"}
        ts = _ts(10)
        raw = [[
            {"last_changed": ts, "state": "20.0"},
            {"last_changed": ts, "state": "21.0"},  # same timestamp, different value
            {"last_changed": _ts(5), "state": "22.0"},
        ]]
        df = _ha_history_to_dataframe(raw, entity_map, entity_ids)
        assert not df.empty
        # Should not have duplicated rows for the same timestamp
        assert df["temp"].notna().any()

    def test_records_missing_last_changed_skipped(self):
        entity_ids = ["sensor.temp"]
        entity_map = {"sensor.temp": "temp"}
        raw = [[
            {"state": "20.0"},  # no last_changed key
            {"last_changed": _ts(5), "state": "21.0"},
        ]]
        df = _ha_history_to_dataframe(raw, entity_map, entity_ids)
        assert not df.empty

    def test_records_with_last_updated_fallback(self):
        """Records that lack last_changed but have last_updated should be accepted."""
        entity_ids = ["sensor.temp"]
        entity_map = {"sensor.temp": "temp"}
        raw = [[
            {"last_updated": _ts(10), "state": "20.0"},
            {"last_updated": _ts(5), "state": "21.0"},
        ]]
        df = _ha_history_to_dataframe(raw, entity_map, entity_ids)
        assert not df.empty
        assert "temp" in df.columns

    def test_invalid_timestamp_strings_skipped(self):
        entity_ids = ["sensor.temp"]
        entity_map = {"sensor.temp": "temp"}
        raw = [[
            {"last_changed": "not-a-timestamp", "state": "20.0"},
            {"last_changed": _ts(5), "state": "21.0"},
        ]]
        df = _ha_history_to_dataframe(raw, entity_map, entity_ids)
        # Should still produce a DataFrame from the valid record
        assert not df.empty

    def test_all_records_have_invalid_timestamps_returns_empty(self):
        entity_ids = ["sensor.temp"]
        entity_map = {"sensor.temp": "temp"}
        raw = [[
            {"last_changed": "INVALID", "state": "20.0"},
        ]]
        df = _ha_history_to_dataframe(raw, entity_map, entity_ids)
        assert df.empty

    def test_entity_map_lookup_falls_back_to_split(self):
        """When entity_id not in entity_map, short name is derived by split."""
        entity_ids = ["sensor.unknown_entity"]
        entity_map = {}  # not in map
        raw = [[
            {"last_changed": _ts(10), "state": "20.0"},
            {"last_changed": _ts(5), "state": "21.0"},
        ]]
        df = _ha_history_to_dataframe(raw, entity_map, entity_ids)
        assert "unknown_entity" in df.columns

    def test_binary_entity_values_are_0_or_1(self):
        entity_ids = ["binary_sensor.fireplace"]
        entity_map = {"binary_sensor.fireplace": "fireplace"}
        raw = [[
            {"last_changed": _ts(20), "state": "on"},
            {"last_changed": _ts(10), "state": "off"},
            {"last_changed": _ts(5), "state": "on"},
        ]]
        df = _ha_history_to_dataframe(raw, entity_map, entity_ids)
        assert not df.empty
        unique_vals = set(df["fireplace"].dropna().unique())
        assert unique_vals.issubset({0.0, 1.0})

    def test_more_entity_ids_than_histories_handles_gracefully(self):
        """If len(entity_ids) > len(raw_histories), extras are ignored."""
        entity_ids = ["sensor.a", "sensor.b"]
        entity_map = {"sensor.a": "a", "sensor.b": "b"}
        raw = [[{"last_changed": _ts(5), "state": "10.0"}]]  # only 1 history
        df = _ha_history_to_dataframe(raw, entity_map, entity_ids)
        # Only 'a' should be present
        assert "a" in df.columns


# ---------------------------------------------------------------------------
# compute_cloud_proxy – additional edge cases
# ---------------------------------------------------------------------------
class TestComputeCloudProxyExtended:
    def test_zero_peak_uses_series_max(self):
        """When peak_pv_watts=0, max of series is used as reference."""
        idx = pd.date_range("2026-04-04 12:00", periods=3, freq="5min", tz="UTC")
        pv = pd.Series([0.0, 4000.0, 8000.0], index=idx)
        cloud = compute_cloud_proxy(pv, idx, peak_pv_watts=0.0)
        # At peak (8000W), cloud should be ≈0%
        assert cloud.iloc[2] == pytest.approx(0.0, abs=1.0)

    def test_all_zero_pv_no_peak_returns_50(self):
        """All PV = 0 and no peak given → series max = 0 → fallback to 50%."""
        idx = pd.date_range("2026-04-04 12:00", periods=3, freq="5min", tz="UTC")
        pv = pd.Series([0.0] * 3, index=idx)
        cloud = compute_cloud_proxy(pv, idx, peak_pv_watts=0.0)
        assert (cloud == 50.0).all()

    def test_partial_cloud_cover(self):
        idx = pd.date_range("2026-04-04 12:00", periods=1, freq="5min", tz="UTC")
        pv = pd.Series([3000.0], index=idx)
        cloud = compute_cloud_proxy(pv, idx, peak_pv_watts=6000.0)
        # 50% cloud (1 - 3000/6000)*100
        assert cloud.iloc[0] == pytest.approx(50.0)

    def test_overcapacity_pv_clamped_to_zero_cloud(self):
        """PV slightly above stated peak → ratio > 1 → cloud = 0%."""
        idx = pd.date_range("2026-04-04 12:00", periods=1, freq="5min", tz="UTC")
        pv = pd.Series([9000.0], index=idx)
        cloud = compute_cloud_proxy(pv, idx, peak_pv_watts=8000.0)
        # (1 - 9000/8000) = negative → clipped to 0 → 0% cloud
        assert cloud.iloc[0] == pytest.approx(0.0)
