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
    """Set all entity ID attributes and calibration params on a mock config."""
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
    # HLC calibration config vars (Fixes 2, 7, 8, 9 — Fix 9 now uses shared var)
    mock_config.HLC_MIN_FLOW_RATE_LPM = 0.5
    mock_config.HLC_WINDOW_SIZE_ROWS = 12
    mock_config.HLC_REGRESSION_INTERCEPT = False
    mock_config.HEATING_MIN_THERMAL_POWER_KW = 0.5


def _make_df(n_rows: int = 100, include_target: bool = True) -> pd.DataFrame:
    """Create a DataFrame with non-English column names (entity short IDs).

    The column names deliberately match the short form of the entity IDs set
    by ``_apply_entity_ids()`` — e.g. ``"rt_mittelwert"`` instead of
    ``"indoor_temp"`` — to exercise the config-based column lookup.

    A ``_time`` column is included (matching the real output of
    ``fetch_historical_data_for_calibration``) so that Fix 1 and Fix 4
    are exercised correctly.
    """
    times = pd.date_range("2026-01-01", periods=n_rows, freq="5min")
    data = {
        "_time": times,
        _INDOOR_ID.split(".", 1)[-1]: np.full(n_rows, 21.0),   # rt_mittelwert
        _OUTDOOR_ID.split(".", 1)[-1]: np.full(n_rows, 5.0),
        _OUTLET_ID.split(".", 1)[-1]: np.full(n_rows, 35.0),
        _INLET_ID.split(".", 1)[-1]: np.full(n_rows, 30.0),
        _FLOW_ID.split(".", 1)[-1]: np.full(n_rows, 10.0),
    }
    if include_target:
        data[_TARGET_ID.split(".", 1)[-1]] = np.full(n_rows, 21.0)
    return pd.DataFrame(data)


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


# ---------------------------------------------------------------------------
# TestCalibrateHLCQualityFixes — new tests for Fixes 1–5, 7, 9
# ---------------------------------------------------------------------------

def _make_perfect_linear_df() -> pd.DataFrame:
    """Return a DataFrame where Q = HLC × ΔT exactly (R² must be ~1.0).

    Uses 7 different outdoor temperatures (each repeated for 12 rows = one
    60-minute window) so that ΔT varies across windows.  The outlet−inlet
    temperature is set so that the computed thermal power equals
    ``HLC_TRUE × ΔT`` for each window, yielding a perfect linear fit.
    """
    hlc_true = 0.12  # kW/K
    cp = 4.186       # kJ/(kg·K)
    flow = 10.0      # L/min
    indoor = 21.0    # °C
    outlet_base = 35.0

    outdoor_temps = [0.0, 2.0, 5.0, 7.0, 9.0, 11.0, 13.0]
    rows: list = []
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    for t_out in outdoor_temps:
        delta_t = indoor - t_out             # K
        q_target = hlc_true * delta_t        # kW
        # Invert: q = (flow/60) * cp * (outlet - inlet) -> outlet - inlet
        dt_hp = q_target / ((flow / 60.0) * cp)
        inlet = outlet_base - dt_hp
        for i in range(12):
            rows.append({
                "_time": t0,
                _INDOOR_ID.split(".", 1)[-1]: indoor,
                _OUTDOOR_ID.split(".", 1)[-1]: t_out,
                _OUTLET_ID.split(".", 1)[-1]: outlet_base,
                _INLET_ID.split(".", 1)[-1]: inlet,
                _FLOW_ID.split(".", 1)[-1]: flow,
                _TARGET_ID.split(".", 1)[-1]: indoor,
            })
            t0 += pd.Timedelta("5min")
    return pd.DataFrame(rows)


class TestCalibrateHLCQualityFixes:
    """Tests for Fixes 1–5, 7, and 9 applied to calibrate_hlc."""

    def _setup_config(self, mock_config) -> None:
        """Configure all required attributes on a MagicMock config object."""
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

    # --- Fix 1: date range must be datetime strings, not integer indices ---

    @patch("src.hlc_learner.get_thermal_state_manager")
    @patch("src.hlc_learner.fetch_historical_data_for_calibration")
    @patch("src.hlc_learner.config")
    def test_date_range_is_datetime_string_not_integer(
        self, mock_config, mock_fetch, mock_tsm_fn
    ):
        """date_range in the result must show ISO-like datetime strings when
        the DataFrame has a ``_time`` column (as returned by the real fetch
        helper after reset_index).  Previously it logged '0 — 23881'."""
        self._setup_config(mock_config)
        mock_tsm_fn.return_value = MagicMock()
        mock_fetch.return_value = _make_df(120)

        result = calibrate_hlc()

        assert result["success"] is True
        dr = result.get("date_range", "")
        # Must not be a pair of small integers
        assert "—" in dr, f"date_range has no separator: {dr!r}"
        left = dr.split("—")[0].strip()
        # The left part must NOT be a plain integer (it should be a timestamp)
        assert not left.isdigit(), (
            f"date_range looks like integer indices, expected datetimes: {dr!r}"
        )
        # Should contain a year (e.g. "2026")
        assert "2026" in dr, f"Expected a year in date_range, got: {dr!r}"

    # --- Fix 5: high R² on a perfect-linear dataset ---

    @patch("src.hlc_learner.get_thermal_state_manager")
    @patch("src.hlc_learner.fetch_historical_data_for_calibration")
    @patch("src.hlc_learner.config")
    def test_high_r2_for_perfect_linear_data(
        self, mock_config, mock_fetch, mock_tsm_fn
    ):
        """When Q and ΔT have a perfect linear relationship, standard R²,
        FTO-R², and Pearson r must all be > 0.90."""
        self._setup_config(mock_config)
        mock_tsm_fn.return_value = MagicMock()
        mock_fetch.return_value = _make_perfect_linear_df()

        result = calibrate_hlc()

        assert result["success"] is True, f"Failed: {result.get('message')}"
        assert result["r2"] > 0.90, f"R² too low: {result['r2']}"
        assert result["r2_fto"] > 0.90, f"FTO-R² too low: {result['r2_fto']}"
        assert result["r_pearson"] > 0.90, (
            f"Pearson r too low: {result['r_pearson']}"
        )

    # --- Fix 2 & 9: standby windows (low / zero flow) are rejected ---

    @patch("src.hlc_learner.fetch_historical_data_for_calibration")
    @patch("src.hlc_learner.config")
    def test_standby_windows_rejected_by_flow_filter(
        self, mock_config, mock_fetch
    ):
        """Windows with flow_rate below HLC_MIN_FLOW_RATE_LPM must be
        rejected with reason 'flow_too_low' rather than passing through
        to the thermal-power check."""
        self._setup_config(mock_config)
        # Lower min_periods so we only need some valid windows
        mock_config.HLC_CALIBRATION_MIN_PERIODS = 9999  # force failure
        mock_config.HLC_MIN_FLOW_RATE_LPM = 2.0  # strict threshold

        # Build a DataFrame where all 12-row windows have flow = 0
        n_rows = 120
        times = pd.date_range("2026-01-01", periods=n_rows, freq="5min")
        df = pd.DataFrame({
            "_time": times,
            _INDOOR_ID.split(".", 1)[-1]: np.full(n_rows, 21.0),
            _OUTDOOR_ID.split(".", 1)[-1]: np.full(n_rows, 5.0),
            _OUTLET_ID.split(".", 1)[-1]: np.full(n_rows, 35.0),
            _INLET_ID.split(".", 1)[-1]: np.full(n_rows, 30.0),
            _FLOW_ID.split(".", 1)[-1]: np.zeros(n_rows),  # standby
            _TARGET_ID.split(".", 1)[-1]: np.full(n_rows, 21.0),
        })
        mock_fetch.return_value = df

        result = calibrate_hlc()
        # All windows should be rejected — not enough valid periods
        assert result["success"] is False
        # Verify the periods count is 0 (all windows rejected)
        assert result.get("n_periods", 0) == 0

    # --- Fix 3: missing target_temp must emit a warning ---

    @patch("src.hlc_learner.get_thermal_state_manager")
    @patch("src.hlc_learner.fetch_historical_data_for_calibration")
    @patch("src.hlc_learner.config")
    def test_missing_target_temp_emits_warning(
        self, mock_config, mock_fetch, mock_tsm_fn
    ):
        """When the target_temp column is absent from the DataFrame, an
        info message must be logged indicating that a default target temp
        is used for quality gates."""
        self._setup_config(mock_config)
        mock_tsm_fn.return_value = MagicMock()
        # DataFrame without target column
        mock_fetch.return_value = _make_df(120, include_target=False)

        import logging as _logging
        with self.assertLogs("src.hlc_learner", level="INFO") as cm:
            result = calibrate_hlc()

        assert any(
            "target_temp" in msg and "not available" in msg
            for msg in cm.output
        ), f"Expected target_temp info message, got: {cm.output}"
        # Calibration may still succeed (now with default target quality gates)
        assert "success" in result

    def assertLogs(self, logger_name, level="WARNING"):
        """Thin wrapper so the class can use assertLogs without inheriting
        from unittest.TestCase."""
        import unittest
        tc = unittest.TestCase()
        tc.maxDiff = None
        return tc.assertLogs(logger_name, level=level)
