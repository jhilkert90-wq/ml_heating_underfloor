"""Tests for HA history service and data source selection."""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ha_history_service import (
    _ha_history_to_dataframe,
    _is_binary_entity,
    _parse_state_value,
    compute_cloud_proxy,
    get_training_data_from_ha,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ts(minutes_ago: int) -> str:
    """Return ISO-formatted UTC timestamp *minutes_ago* minutes in the past."""
    return (
        datetime.now(tz=timezone.utc) - timedelta(minutes=minutes_ago)
    ).isoformat()


def _make_raw_history(entity_ids, records_per_entity):
    """Build a minimal HA history JSON structure.

    ``records_per_entity`` is a list of lists; each inner list is a series
    of ``(minutes_ago, state_str)`` tuples.
    """
    raw = []
    for recs in records_per_entity:
        raw.append(
            [{"last_changed": _ts(m), "state": s} for m, s in recs]
        )
    return raw


# -----------------------------------------------------------------------
# ha_client.get_history_bulk tests
# -----------------------------------------------------------------------
class TestHAGetHistoryBulk:
    """Tests for HAClient.get_history_bulk."""

    def test_success(self):
        from src.ha_client import HAClient

        client = HAClient("http://fake:8123", "tok")
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = [[{"state": "21.5", "last_changed": _ts(10)}]]
        fake_resp.raise_for_status = MagicMock()

        with patch("src.ha_client.requests.get", return_value=fake_resp):
            result = client.get_history_bulk(
                ["sensor.temp"],
                datetime.now(tz=timezone.utc) - timedelta(hours=1),
            )
        assert result is not None
        assert len(result) == 1

    def test_timeout(self):
        from src.ha_client import HAClient
        import requests as req_lib

        client = HAClient("http://fake:8123", "tok")
        with patch(
            "src.ha_client.requests.get", side_effect=req_lib.Timeout("boom")
        ):
            result = client.get_history_bulk(
                ["sensor.temp"],
                datetime.now(tz=timezone.utc) - timedelta(hours=1),
            )
        assert result is None

    def test_auth_error(self):
        from src.ha_client import HAClient
        import requests as req_lib

        client = HAClient("http://fake:8123", "tok")
        resp = MagicMock()
        resp.status_code = 401
        resp.raise_for_status.side_effect = req_lib.HTTPError("401")
        with patch("src.ha_client.requests.get", return_value=resp):
            result = client.get_history_bulk(
                ["sensor.temp"],
                datetime.now(tz=timezone.utc) - timedelta(hours=1),
            )
        assert result is None


# -----------------------------------------------------------------------
# Parsing tests
# -----------------------------------------------------------------------
class TestHAHistoryParsing:
    """Tests for state value parsing and DataFrame conversion."""

    def test_parse_numeric(self):
        assert _parse_state_value("21.5", False) == 21.5

    def test_parse_binary_on_off(self):
        assert _parse_state_value("on", True) == 1.0
        assert _parse_state_value("off", True) == 0.0

    def test_parse_unavailable(self):
        assert math.isnan(_parse_state_value("unavailable", False))
        assert math.isnan(_parse_state_value("unknown", True))

    def test_is_binary_entity(self):
        assert _is_binary_entity("binary_sensor.foo")
        assert _is_binary_entity("input_boolean.bar")
        assert not _is_binary_entity("sensor.temp")

    def test_reindex_5min(self):
        """Irregular timestamps → regular 5-min grid."""
        entity_ids = ["sensor.temp"]
        entity_map = {"sensor.temp": "temp"}
        raw = [[
            {"last_changed": _ts(30), "state": "20.0"},
            {"last_changed": _ts(17), "state": "20.5"},
            {"last_changed": _ts(3), "state": "21.0"},
        ]]
        df = _ha_history_to_dataframe(raw, entity_map, entity_ids)
        assert not df.empty
        # All intervals should be 5-min
        diffs = df["_time"].diff().dropna().dt.total_seconds()
        assert (diffs == 300).all()

    def test_ffill_gaps(self):
        """30-min gap → forward-filled correctly (no NaN)."""
        entity_ids = ["sensor.temp"]
        entity_map = {"sensor.temp": "temp"}
        raw = [[
            {"last_changed": _ts(60), "state": "20.0"},
            {"last_changed": _ts(0), "state": "21.0"},
        ]]
        df = _ha_history_to_dataframe(raw, entity_map, entity_ids)
        assert not df["temp"].isna().any()

    def test_schema_matches_influx(self):
        """HA DataFrame columns use entity short-names like InfluxDB."""
        entity_ids = ["sensor.nibe_bt2_supply_temp_s1", "binary_sensor.warmwassermodus"]
        entity_map = {
            "sensor.nibe_bt2_supply_temp_s1": "nibe_bt2_supply_temp_s1",
            "binary_sensor.warmwassermodus": "warmwassermodus",
        }
        raw = [
            [{"last_changed": _ts(10), "state": "32.5"}],
            [{"last_changed": _ts(10), "state": "off"}],
        ]
        df = _ha_history_to_dataframe(raw, entity_map, entity_ids)
        assert "nibe_bt2_supply_temp_s1" in df.columns
        assert "warmwassermodus" in df.columns


# -----------------------------------------------------------------------
# Training data source fallback
# -----------------------------------------------------------------------
class TestTrainingDataSource:
    """Tests for get_training_data_from_ha and auto-fallback in calibration."""

    @patch("src.ha_history_service._build_entity_map")
    @patch("src.ha_history_service.HAClient")
    def test_ha_only(self, MockClient, mock_map):
        """config=ha_history → calls HA, produces DataFrame."""
        mock_map.return_value = {"sensor.temp": "temp"}

        mock_inst = MagicMock()
        mock_inst.get_history_bulk.return_value = [
            [{"last_changed": _ts(30), "state": "20.0"},
             {"last_changed": _ts(0), "state": "21.0"}]
        ]

        df = get_training_data_from_ha(lookback_hours=1, ha_client=mock_inst)
        assert not df.empty
        mock_inst.get_history_bulk.assert_called_once()

    @patch("src.physics_calibration.InfluxService")
    def test_auto_fallback(self, MockInflux):
        """InfluxDB fails → fetch_historical_data attempts HA history path."""
        import src.physics_calibration as pc

        # Make InfluxDB return empty → triggers fallback branch
        MockInflux.return_value.get_training_data.return_value = pd.DataFrame()

        with patch.object(pc.config, "TRAINING_DATA_SOURCE", "auto"), \
             patch.object(pc.config, "INFLUX_URL", "http://fake"), \
             patch.object(pc.config, "INFLUX_TOKEN", "tok"), \
             patch.object(pc.config, "INFLUX_ORG", "org"):

            # Patch the local import target inside the function
            fake_ha_df = pd.DataFrame({"_time": [datetime.now(tz=timezone.utc)], "temp": [20.0]})
            with patch.dict(
                "sys.modules",
                {"src.ha_history_service": MagicMock(get_training_data_from_ha=MagicMock(return_value=fake_ha_df))},
            ):
                result = pc.fetch_historical_data_for_calibration(lookback_hours=24)
                # InfluxDB was tried first
                MockInflux.return_value.get_training_data.assert_called_once()

    @patch("src.physics_calibration.InfluxService")
    def test_auto_supplements_missing_columns(self, MockInflux):
        """InfluxDB has data but missing columns → supplements from HA."""
        import src.physics_calibration as pc

        # InfluxDB returns data but WITHOUT fireplace column
        influx_df = pd.DataFrame({
            "_time": pd.date_range("2026-04-01", periods=10, freq="5min", tz="UTC"),
            pc.config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]: [20.0] * 10,
            pc.config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]: [30.0] * 10,
            pc.config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]: [30.0] * 10,
            pc.config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]: [5.0] * 10,
            pc.config.PV_POWER_ENTITY_ID.split(".", 1)[-1]: [1000.0] * 10,
            pc.config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]: [0] * 10,
        })
        MockInflux.return_value.get_training_data.return_value = influx_df

        # HA history has the fireplace column
        fp_col = pc.config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
        ha_df = pd.DataFrame({
            "_time": pd.date_range("2026-04-01", periods=10, freq="5min", tz="UTC"),
            fp_col: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        })

        with patch.object(pc.config, "TRAINING_DATA_SOURCE", "auto"), \
             patch.object(pc.config, "INFLUX_URL", "http://fake"), \
             patch.object(pc.config, "INFLUX_TOKEN", "tok"), \
             patch.object(pc.config, "INFLUX_ORG", "org"):
            mock_ha = MagicMock(get_training_data_from_ha=MagicMock(return_value=ha_df))
            with patch.dict("sys.modules", {"src.ha_history_service": mock_ha}):
                result = pc.fetch_historical_data_for_calibration(lookback_hours=24)
                assert result is not None
                assert fp_col in result.columns
                assert result[fp_col].sum() > 0  # FP data was merged


# -----------------------------------------------------------------------
# Cloud proxy
# -----------------------------------------------------------------------
class TestCloudProxy:

    def test_pv_midday(self):
        """PV=5000W with peak 8000W → cloud ≈ 37.5%."""
        idx = pd.date_range("2026-04-04 12:00", periods=5, freq="5min", tz="UTC")
        pv = pd.Series([5000.0] * 5, index=idx)
        cloud = compute_cloud_proxy(pv, idx, peak_pv_watts=8000.0)
        expected = (1 - 5000 / 8000) * 100  # 37.5
        assert abs(cloud.iloc[0] - expected) < 0.1

    def test_nighttime(self):
        """PV=0 at night → cloud = 50% (neutral default)."""
        idx = pd.date_range("2026-04-04 23:00", periods=5, freq="5min", tz="UTC")
        pv = pd.Series([0.0] * 5, index=idx)
        cloud = compute_cloud_proxy(pv, idx, peak_pv_watts=8000.0)
        assert (cloud == 50.0).all()


# -----------------------------------------------------------------------
# Per-entity time-coverage gap detection and HA supplement
# -----------------------------------------------------------------------
class TestPerEntityCoverageGap:
    """Tests for the per-entity InfluxDB coverage gap detection in
    fetch_historical_data_for_calibration."""

    def _make_complete_influx_df(self, start, periods):
        """Build a minimal valid InfluxDB-like DataFrame with all columns."""
        import src.physics_calibration as pc
        times = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
        return pd.DataFrame({
            "_time": times,
            pc.config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]: [20.0] * periods,
            pc.config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]: [30.0] * periods,
            pc.config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]: [30.0] * periods,
            pc.config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]: [5.0] * periods,
            pc.config.PV_POWER_ENTITY_ID.split(".", 1)[-1]: [500.0] * periods,
            pc.config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]: [0.0] * periods,
            # Optional columns — prevent column-supplement from triggering
            pc.config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]: [0.0] * periods,
            pc.config.INLET_TEMP_ENTITY_ID.split(".", 1)[-1]: [25.0] * periods,
            pc.config.FLOW_RATE_ENTITY_ID.split(".", 1)[-1]: [8.0] * periods,
            pc.config.POWER_CONSUMPTION_ENTITY_ID.split(".", 1)[-1]: [200.0] * periods,
            pc.config.DHW_STATUS_ENTITY_ID.split(".", 1)[-1]: [0.0] * periods,
        })

    @patch("src.physics_calibration.InfluxService")
    def test_no_gap_no_ha_call(self, MockInflux):
        """When InfluxDB covers the full lookback, HA is not called."""
        import src.physics_calibration as pc

        now = datetime.now(tz=timezone.utc)
        # 48h lookback, InfluxDB has data from 48h ago → no gap
        influx_df = self._make_complete_influx_df(
            now - timedelta(hours=48), periods=48 * 12,
        )
        MockInflux.return_value.get_training_data.return_value = influx_df

        ha_mock = MagicMock(return_value=pd.DataFrame())

        with patch.object(pc.config, "TRAINING_DATA_SOURCE", "auto"), \
             patch.object(pc.config, "INFLUX_URL", "http://fake"), \
             patch.object(pc.config, "INFLUX_TOKEN", "tok"), \
             patch.object(pc.config, "INFLUX_ORG", "org"):
            with patch.dict(
                "sys.modules",
                {"src.ha_history_service": MagicMock(
                    get_training_data_from_ha=ha_mock,
                )},
            ):
                result = pc.fetch_historical_data_for_calibration(
                    lookback_hours=48,
                )
                assert result is not None
                # HA should NOT have been called (no gap detected)
                ha_mock.assert_not_called()

    @patch("src.physics_calibration.InfluxService")
    def test_entity_gap_supplemented_from_ha(self, MockInflux):
        """Entity with NaN gap is filled from HA history data."""
        import src.physics_calibration as pc

        now = datetime.now(tz=timezone.utc)
        lookback = 48
        pv_col = pc.config.PV_POWER_ENTITY_ID.split(".", 1)[-1]

        # InfluxDB: full time range but PV column has NaN for first 24h
        n = lookback * 12
        influx_df = self._make_complete_influx_df(
            now - timedelta(hours=lookback), periods=n,
        )
        half = n // 2
        influx_df.loc[:half - 1, pv_col] = np.nan  # first 24h missing

        MockInflux.return_value.get_training_data.return_value = influx_df

        # HA has PV data for the full 48h
        ha_df = self._make_complete_influx_df(
            now - timedelta(hours=lookback), periods=n,
        )
        ha_df[pv_col] = 2000.0  # HA reports 2000W PV

        ha_mock = MagicMock(return_value=ha_df)

        with patch.object(pc.config, "TRAINING_DATA_SOURCE", "auto"), \
             patch.object(pc.config, "INFLUX_URL", "http://fake"), \
             patch.object(pc.config, "INFLUX_TOKEN", "tok"), \
             patch.object(pc.config, "INFLUX_ORG", "org"):
            with patch.dict(
                "sys.modules",
                {"src.ha_history_service": MagicMock(
                    get_training_data_from_ha=ha_mock,
                )},
            ):
                result = pc.fetch_historical_data_for_calibration(
                    lookback_hours=lookback,
                )
                assert result is not None
                # HA was called to supplement the gap
                ha_mock.assert_called_once()
                # PV column should now have values everywhere
                assert result[pv_col].notna().all()
                # First rows should have HA's 2000W value
                assert result[pv_col].iloc[0] == pytest.approx(2000.0)

    @patch("src.physics_calibration.InfluxService")
    def test_global_gap_extends_time_range(self, MockInflux):
        """When InfluxDB starts late, HA rows are prepended."""
        import src.physics_calibration as pc

        now = datetime.now(tz=timezone.utc)
        lookback = 48

        # InfluxDB only has last 24h (starts 24h ago, not 48h ago)
        influx_n = 24 * 12
        influx_df = self._make_complete_influx_df(
            now - timedelta(hours=24), periods=influx_n,
        )
        MockInflux.return_value.get_training_data.return_value = influx_df

        # HA has the full 48h
        ha_n = lookback * 12
        ha_df = self._make_complete_influx_df(
            now - timedelta(hours=lookback), periods=ha_n,
        )

        ha_mock = MagicMock(return_value=ha_df)

        with patch.object(pc.config, "TRAINING_DATA_SOURCE", "auto"), \
             patch.object(pc.config, "INFLUX_URL", "http://fake"), \
             patch.object(pc.config, "INFLUX_TOKEN", "tok"), \
             patch.object(pc.config, "INFLUX_ORG", "org"):
            with patch.dict(
                "sys.modules",
                {"src.ha_history_service": MagicMock(
                    get_training_data_from_ha=ha_mock,
                )},
            ):
                result = pc.fetch_historical_data_for_calibration(
                    lookback_hours=lookback,
                )
                assert result is not None
                # Result should cover more than the InfluxDB-only 24h
                actual_hours = (
                    (result["_time"].max() - result["_time"].min()
                     ).total_seconds() / 3600
                )
                assert actual_hours > 30  # significantly more than 24h

    @patch("src.physics_calibration.InfluxService")
    def test_no_supplement_when_source_influx(self, MockInflux):
        """source=influx → per-entity gap check is skipped."""
        import src.physics_calibration as pc

        now = datetime.now(tz=timezone.utc)
        pv_col = pc.config.PV_POWER_ENTITY_ID.split(".", 1)[-1]

        # InfluxDB: PV has NaN gap in first half
        n = 48 * 12
        influx_df = self._make_complete_influx_df(
            now - timedelta(hours=48), periods=n,
        )
        influx_df.loc[:n // 2 - 1, pv_col] = np.nan

        MockInflux.return_value.get_training_data.return_value = influx_df

        ha_mock = MagicMock(return_value=pd.DataFrame())

        with patch.object(pc.config, "TRAINING_DATA_SOURCE", "influx"), \
             patch.object(pc.config, "INFLUX_URL", "http://fake"), \
             patch.object(pc.config, "INFLUX_TOKEN", "tok"), \
             patch.object(pc.config, "INFLUX_ORG", "org"):
            with patch.dict(
                "sys.modules",
                {"src.ha_history_service": MagicMock(
                    get_training_data_from_ha=ha_mock,
                )},
            ):
                result = pc.fetch_historical_data_for_calibration(
                    lookback_hours=48,
                )
                assert result is not None
                # HA should NOT have been called (source=influx)
                ha_mock.assert_not_called()
