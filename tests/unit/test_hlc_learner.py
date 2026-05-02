"""
Unit tests for src/hlc_learner.py - Day-Level HLC + Historical Calibration.

Covers:
- HLCCycle dataclass construction and delta_t property
- Module-level _build_cycle() helper
- calibrate_hlc() historical calibration function
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest

from src.hlc_learner import HLCCycle, _build_cycle, calibrate_hlc


# ---------------------------------------------------------------------------
# HLCCycle
# ---------------------------------------------------------------------------

class TestHLCCycle:
    def test_delta_t(self):
        c = HLCCycle(
            timestamp=datetime.now(),
            thermal_power_kw=2.0,
            indoor_temp=21.5,
            outdoor_temp=5.5,
            target_temp=21.0,
            indoor_temp_delta_60m=0.0,
            pv_now_electrical=0.0,
            fireplace_on=0.0,
            tv_on=0.0,
            dhw_heating=0.0,
            defrosting=0.0,
            dhw_boost_heater=0.0,
            is_blocking=False,
        )
        assert c.delta_t == pytest.approx(16.0)

    def test_delta_t_negative_when_outdoor_warmer(self):
        c = HLCCycle(
            timestamp=datetime.now(),
            thermal_power_kw=0.0,
            indoor_temp=18.0,
            outdoor_temp=22.0,
            target_temp=21.0,
            indoor_temp_delta_60m=0.0,
            pv_now_electrical=0.0,
            fireplace_on=0.0,
            tv_on=0.0,
            dhw_heating=0.0,
            defrosting=0.0,
            dhw_boost_heater=0.0,
            is_blocking=False,
        )
        assert c.delta_t == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# _build_cycle
# ---------------------------------------------------------------------------

class TestBuildCycle:
    def _ctx(self, **overrides):
        base = {
            "timestamp": datetime(2026, 1, 10, 12, 0),
            "thermal_power_kw": 1.5,
            "indoor_temp": 21.0,
            "outdoor_temp": 5.0,
            "target_temp": 21.0,
            "indoor_temp_delta_60m": 0.0,
            "pv_now_electrical": 0.0,
            "fireplace_on": 0.0,
            "tv_on": 0.0,
            "dhw_heating": 0.0,
            "defrosting": 0.0,
            "dhw_boost_heater": 0.0,
            "is_blocking": False,
        }
        base.update(overrides)
        return base

    def test_valid_context_returns_cycle(self):
        cycle = _build_cycle(self._ctx())
        assert isinstance(cycle, HLCCycle)
        assert cycle.thermal_power_kw == 1.5
        assert cycle.delta_t == pytest.approx(16.0)

    def test_missing_thermal_power_returns_none(self):
        ctx = self._ctx()
        del ctx["thermal_power_kw"]
        assert _build_cycle(ctx) is None

    def test_none_thermal_power_returns_none(self):
        assert _build_cycle(self._ctx(thermal_power_kw=None)) is None

    def test_missing_indoor_temp_returns_none(self):
        ctx = self._ctx()
        del ctx["indoor_temp"]
        assert _build_cycle(ctx) is None

    def test_missing_outdoor_temp_returns_none(self):
        ctx = self._ctx()
        del ctx["outdoor_temp"]
        assert _build_cycle(ctx) is None


# ---------------------------------------------------------------------------
# calibrate_hlc
# ---------------------------------------------------------------------------

# Entity IDs used across calibrate_hlc tests (mimicking a non-English HA setup)
_INDOOR_ID = "sensor.rt_mittelwert"
_OUTDOOR_ID = "sensor.nibe_bt1_outdoor_temperature"
_OUTLET_ID = "sensor.nibe_bt50_supply_line_temperature"
_INLET_ID = "sensor.nibe_bt3_return_line_temperature"
_FLOW_ID = "sensor.nibe_flow_rate"
_PV_ID = "sensor.pv_power"
_FIREPLACE_ID = "binary_sensor.kamin_aktiv"
_TV_ID = "input_boolean.fernseher"
_DHW_ID = "binary_sensor.nibe_dhw_active"
_DEFROST_ID = "binary_sensor.nibe_defrost_active"
_TARGET_ID = "input_number.soll_rt"


def _apply_entity_ids(mock_config) -> None:
    """Set all entity ID attributes on a mock config object."""
    mock_config.INDOOR_TEMP_ENTITY_ID = _INDOOR_ID
    mock_config.OUTDOOR_TEMP_ENTITY_ID = _OUTDOOR_ID
    mock_config.ACTUAL_OUTLET_TEMP_ENTITY_ID = _OUTLET_ID
    mock_config.INLET_TEMP_ENTITY_ID = _INLET_ID
    mock_config.FLOW_RATE_ENTITY_ID = _FLOW_ID
    mock_config.PV_POWER_ENTITY_ID = _PV_ID
    mock_config.FIREPLACE_STATUS_ENTITY_ID = _FIREPLACE_ID
    mock_config.TV_STATUS_ENTITY_ID = _TV_ID
    mock_config.DHW_STATUS_ENTITY_ID = _DHW_ID
    mock_config.DEFROST_STATUS_ENTITY_ID = _DEFROST_ID
    mock_config.TARGET_INDOOR_TEMP_ENTITY_ID = _TARGET_ID


def _make_df(n_rows: int = 100, include_target: bool = True) -> pd.DataFrame:
    """Create a DataFrame with non-English column names (entity short IDs).

    The column names deliberately match the short form of the entity IDs set
    by ``_apply_entity_ids()`` — e.g. ``"rt_mittelwert"`` instead of
    ``"indoor_temp"`` — to exercise the config-based column lookup.
    """
    idx = pd.date_range("2026-01-01", periods=n_rows, freq="5min")
    data = {
        _INDOOR_ID.split(".", 1)[-1]: np.full(n_rows, 21.0),   # rt_mittelwert
        _OUTDOOR_ID.split(".", 1)[-1]: np.full(n_rows, 5.0),
        _OUTLET_ID.split(".", 1)[-1]: np.full(n_rows, 35.0),
        _INLET_ID.split(".", 1)[-1]: np.full(n_rows, 30.0),
        _FLOW_ID.split(".", 1)[-1]: np.full(n_rows, 10.0),
    }
    if include_target:
        data[_TARGET_ID.split(".", 1)[-1]] = np.full(n_rows, 21.0)
    return pd.DataFrame(data, index=idx)


class TestCalibrateHLC:
    @patch("src.hlc_learner.get_thermal_state_manager")
    @patch("src.hlc_learner.fetch_historical_data_for_calibration")
    @patch("src.hlc_learner.config")
    def test_successful_calibration(self, mock_config, mock_fetch, mock_tsm_fn):
        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 5
        mock_config.SPECIFIC_HEAT_CAPACITY = 4.186
        mock_config.HLC_PV_MAX_W = 50.0
        mock_config.HLC_MAX_INDOOR_DELTA = 0.3
        mock_config.HLC_OUTDOOR_TEMP_MIN = -10.0
        mock_config.HLC_OUTDOOR_TEMP_MAX = 15.0
        mock_config.HLC_MIN_HEATING_DEMAND_K = 1.0
        mock_config.HLC_MAX_TREND = 0.2
        _apply_entity_ids(mock_config)

        mock_tsm = MagicMock()
        mock_tsm_fn.return_value = mock_tsm
        mock_fetch.return_value = _make_df(100)

        result = calibrate_hlc()

        assert result["success"] is True
        assert result["hlc_kw_per_k"] > 0
        assert result["n_periods"] >= 5
        mock_tsm.set_calibrated_baseline.assert_called_once()

    @patch("src.hlc_learner.get_thermal_state_manager")
    @patch("src.hlc_learner.fetch_historical_data_for_calibration")
    @patch("src.hlc_learner.config")
    def test_non_english_column_names_succeeds(
        self, mock_config, mock_fetch, mock_tsm_fn
    ):
        """calibrate_hlc must succeed when column names are non-English entity
        short IDs (e.g. 'rt_mittelwert') that cannot be keyword-matched.
        This is the primary bug scenario from the log."""
        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 5
        mock_config.SPECIFIC_HEAT_CAPACITY = 4.186
        mock_config.HLC_PV_MAX_W = 50.0
        mock_config.HLC_MAX_INDOOR_DELTA = 0.3
        mock_config.HLC_OUTDOOR_TEMP_MIN = -10.0
        mock_config.HLC_OUTDOOR_TEMP_MAX = 15.0
        mock_config.HLC_MIN_HEATING_DEMAND_K = 1.0
        mock_config.HLC_MAX_TREND = 0.2
        _apply_entity_ids(mock_config)

        mock_tsm = MagicMock()
        mock_tsm_fn.return_value = mock_tsm
        # Column names are 'rt_mittelwert', 'nibe_bt1_outdoor_temperature', …
        mock_fetch.return_value = _make_df(100)

        result = calibrate_hlc()

        assert result["success"] is True, f"Failed: {result.get('message')}"
        assert result["hlc_kw_per_k"] > 0
        mock_fetch.assert_called_once_with(lookback_hours=720)

    @patch("src.hlc_learner.fetch_historical_data_for_calibration")
    @patch("src.hlc_learner.config")
    def test_no_data_returns_failure(self, mock_config, mock_fetch):
        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 5
        _apply_entity_ids(mock_config)

        mock_fetch.return_value = None

        result = calibrate_hlc()
        assert result["success"] is False

    @patch("src.hlc_learner.fetch_historical_data_for_calibration")
    @patch("src.hlc_learner.config")
    def test_empty_df_returns_failure(self, mock_config, mock_fetch):
        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 5
        _apply_entity_ids(mock_config)

        mock_fetch.return_value = pd.DataFrame()

        result = calibrate_hlc()
        assert result["success"] is False

    @patch("src.hlc_learner.fetch_historical_data_for_calibration")
    @patch("src.hlc_learner.config")
    def test_missing_columns_returns_failure(self, mock_config, mock_fetch):
        """When the DataFrame is missing required columns (identified by config
        entity IDs), calibration must fail with a descriptive message."""
        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 5
        _apply_entity_ids(mock_config)

        # DataFrame only has indoor + outdoor — outlet/inlet/flow_rate missing
        partial_df = pd.DataFrame({
            _INDOOR_ID.split(".", 1)[-1]: [21.0, 21.0, 21.0, 21.0],
            _OUTDOOR_ID.split(".", 1)[-1]: [5.0, 5.0, 5.0, 5.0],
        })
        mock_fetch.return_value = partial_df

        result = calibrate_hlc()
        assert result["success"] is False
        assert "Missing" in result["message"]

    @patch("src.hlc_learner.fetch_historical_data_for_calibration")
    @patch("src.hlc_learner.config")
    def test_too_few_periods_returns_failure(self, mock_config, mock_fetch):
        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 100
        mock_config.SPECIFIC_HEAT_CAPACITY = 4.186
        mock_config.HLC_PV_MAX_W = 50.0
        mock_config.HLC_MAX_INDOOR_DELTA = 0.3
        mock_config.HLC_OUTDOOR_TEMP_MIN = -10.0
        mock_config.HLC_OUTDOOR_TEMP_MAX = 15.0
        mock_config.HLC_MIN_HEATING_DEMAND_K = 1.0
        mock_config.HLC_MAX_TREND = 0.2
        _apply_entity_ids(mock_config)

        mock_fetch.return_value = _make_df(16)

        result = calibrate_hlc()
        assert result["success"] is False

    @patch("src.hlc_learner.fetch_historical_data_for_calibration")
    @patch("src.hlc_learner.config")
    def test_fetch_exception_returns_failure(self, mock_config, mock_fetch):
        """When fetch_historical_data_for_calibration raises, calibrate_hlc
        must catch it and return success=False with the error message."""
        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 5
        _apply_entity_ids(mock_config)

        mock_fetch.side_effect = ConnectionError("timeout")

        result = calibrate_hlc()
        assert result["success"] is False
        assert "timeout" in result["message"]

    @patch("src.hlc_learner.get_thermal_state_manager")
    @patch("src.hlc_learner.fetch_historical_data_for_calibration")
    @patch("src.hlc_learner.config")
    def test_ha_history_fallback_data_succeeds(
        self, mock_config, mock_fetch, mock_tsm_fn
    ):
        """calibrate_hlc succeeds when the data comes from HA history (the
        fetch helper returns valid data regardless of underlying source)."""
        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 5
        mock_config.SPECIFIC_HEAT_CAPACITY = 4.186
        mock_config.HLC_PV_MAX_W = 50.0
        mock_config.HLC_MAX_INDOOR_DELTA = 0.3
        mock_config.HLC_OUTDOOR_TEMP_MIN = -10.0
        mock_config.HLC_OUTDOOR_TEMP_MAX = 15.0
        mock_config.HLC_MIN_HEATING_DEMAND_K = 1.0
        mock_config.HLC_MAX_TREND = 0.2
        _apply_entity_ids(mock_config)

        mock_tsm = MagicMock()
        mock_tsm_fn.return_value = mock_tsm
        # Simulate data returned from the HA history fallback path —
        # same column-name convention, just a different data origin.
        mock_fetch.return_value = _make_df(100)

        result = calibrate_hlc()

        assert result["success"] is True
        assert result["hlc_kw_per_k"] > 0
        mock_tsm.set_calibrated_baseline.assert_called_once()

    @patch("src.hlc_learner.fetch_historical_data_for_calibration")
    @patch("src.hlc_learner.config")
    def test_backward_compat_influx_service_param_accepted(
        self, mock_config, mock_fetch
    ):
        """calibrate_hlc still accepts influx_service= for backward
        compatibility but ignores it (fetch helper manages data sourcing)."""
        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 5
        _apply_entity_ids(mock_config)

        mock_fetch.return_value = pd.DataFrame()
        dummy_influx = MagicMock()

        # Must not raise even though influx_service is supplied
        result = calibrate_hlc(influx_service=dummy_influx)
        assert result["success"] is False
        # The old influx mock should NOT have been called
        dummy_influx.get_training_data.assert_not_called()
