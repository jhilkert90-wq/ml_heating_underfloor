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

class TestCalibrateHLC:
    def _make_df(self, n_rows=100):
        """Create a DataFrame mimicking InfluxDB training data."""
        import pandas as pd
        import numpy as np

        idx = pd.date_range("2026-01-01", periods=n_rows, freq="5min")
        return pd.DataFrame(
            {
                "indoor_temp": np.full(n_rows, 21.0),
                "outdoor_temp": np.full(n_rows, 5.0),
                "outlet_temp": np.full(n_rows, 35.0),
                "inlet_temp": np.full(n_rows, 30.0),
                "flow_rate": np.full(n_rows, 10.0),
                "target_temp": np.full(n_rows, 21.0),
            },
            index=idx,
        )

    @patch("src.hlc_learner.get_thermal_state_manager")
    @patch("src.hlc_learner.config")
    def test_successful_calibration(self, mock_config, mock_tsm_fn):
        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 5
        mock_config.SPECIFIC_HEAT_CAPACITY = 4.186
        mock_config.HLC_PV_MAX_W = 50.0
        mock_config.HLC_MAX_INDOOR_DELTA = 0.3
        mock_config.HLC_OUTDOOR_TEMP_MIN = -10.0
        mock_config.HLC_OUTDOOR_TEMP_MAX = 15.0
        mock_config.HLC_MIN_HEATING_DEMAND_K = 1.0
        mock_config.HLC_MAX_TREND = 0.2

        mock_tsm = MagicMock()
        mock_tsm_fn.return_value = mock_tsm

        mock_influx = MagicMock()
        mock_influx.get_training_data.return_value = self._make_df(100)

        result = calibrate_hlc(influx_service=mock_influx)

        assert result["success"] is True
        assert result["hlc_kw_per_k"] > 0
        assert result["n_periods"] >= 5
        mock_tsm.set_calibrated_baseline.assert_called_once()

    @patch("src.hlc_learner.config")
    def test_no_data_returns_failure(self, mock_config):
        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 5

        mock_influx = MagicMock()
        mock_influx.get_training_data.return_value = None

        result = calibrate_hlc(influx_service=mock_influx)
        assert result["success"] is False

    @patch("src.hlc_learner.config")
    def test_empty_df_returns_failure(self, mock_config):
        import pandas as pd

        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 5

        mock_influx = MagicMock()
        mock_influx.get_training_data.return_value = pd.DataFrame()

        result = calibrate_hlc(influx_service=mock_influx)
        assert result["success"] is False

    @patch("src.hlc_learner.config")
    def test_missing_columns_returns_failure(self, mock_config):
        import pandas as pd

        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 5

        mock_influx = MagicMock()
        mock_influx.get_training_data.return_value = pd.DataFrame(
            {"indoor_temp": [21.0], "outdoor_temp": [5.0]}
        )

        result = calibrate_hlc(influx_service=mock_influx)
        assert result["success"] is False
        assert "Missing" in result["message"]

    @patch("src.hlc_learner.config")
    def test_too_few_periods_returns_failure(self, mock_config):
        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 100
        mock_config.SPECIFIC_HEAT_CAPACITY = 4.186
        mock_config.HLC_PV_MAX_W = 50.0
        mock_config.HLC_MAX_INDOOR_DELTA = 0.3
        mock_config.HLC_OUTDOOR_TEMP_MIN = -10.0
        mock_config.HLC_OUTDOOR_TEMP_MAX = 15.0
        mock_config.HLC_MIN_HEATING_DEMAND_K = 1.0
        mock_config.HLC_MAX_TREND = 0.2

        mock_influx = MagicMock()
        mock_influx.get_training_data.return_value = self._make_df(16)

        result = calibrate_hlc(influx_service=mock_influx)
        assert result["success"] is False

    @patch("src.hlc_learner.config")
    def test_influx_exception_returns_failure(self, mock_config):
        mock_config.HLC_CALIBRATION_LOOKBACK_HOURS = 720
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 5

        mock_influx = MagicMock()
        mock_influx.get_training_data.side_effect = ConnectionError("timeout")

        result = calibrate_hlc(influx_service=mock_influx)
        assert result["success"] is False
        assert "timeout" in result["message"]
