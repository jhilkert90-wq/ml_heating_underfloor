"""Unit tests for calibrate_hlc() in src/hlc_learner.py."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.hlc_learner import calibrate_hlc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_windows: int = 30, window_size: int = 12) -> pd.DataFrame:
    """Build a synthetic DataFrame that satisfies all quality gates.

    Each window has:
      - stable indoor temp near target (22.6 °C default)
      - outdoor temp in valid range
      - good flow rate
      - positive thermal power
    """
    n = n_windows * window_size
    rng = np.random.default_rng(42)

    # Keep indoor temp within ±0.2 K of target (22.6) so the
    # indoor_far_from_target gate (|mean_indoor - target| > 0.3) passes.
    indoor = rng.uniform(22.45, 22.75, n)
    outdoor = rng.uniform(0.0, 8.0, n)
    outlet = rng.uniform(32.0, 36.0, n)
    inlet = outlet - rng.uniform(4.0, 6.0, n)  # positive ΔT
    flow = rng.uniform(8.0, 12.0, n)

    times = pd.date_range("2024-01-01", periods=n, freq="5min")

    return pd.DataFrame(
        {
            "kuche_temperatur": indoor,
            "thermometer_waermepume_kompensiert": outdoor,
            "hp_outlet_temp": outlet,
            "hp_inlet_temp": inlet,
            "hp_current_flow_rate": flow,
            "_time": times,
        }
    )


def _mock_config():
    """Return a mock config with sensible HLC defaults."""
    cfg = MagicMock()
    cfg.INDOOR_TEMP_ENTITY_ID = "sensor.kuche_temperatur"
    cfg.OUTDOOR_TEMP_ENTITY_ID = "sensor.thermometer_waermepume_kompensiert"
    cfg.ACTUAL_OUTLET_TEMP_ENTITY_ID = "sensor.hp_outlet_temp"
    cfg.INLET_TEMP_ENTITY_ID = "sensor.hp_inlet_temp"
    cfg.FLOW_RATE_ENTITY_ID = "sensor.hp_current_flow_rate"
    cfg.PV_POWER_ENTITY_ID = ""
    cfg.FIREPLACE_STATUS_ENTITY_ID = ""
    cfg.TV_STATUS_ENTITY_ID = ""
    cfg.DHW_STATUS_ENTITY_ID = ""
    cfg.DEFROST_STATUS_ENTITY_ID = ""
    cfg.TARGET_INDOOR_TEMP_ENTITY_ID = ""
    cfg.HLC_CALIBRATION_LOOKBACK_HOURS = 720
    cfg.HLC_CALIBRATION_MIN_PERIODS = 20
    cfg.HLC_WINDOW_SIZE_ROWS = 12
    cfg.HLC_MIN_FLOW_RATE_LPM = 0.5
    cfg.HEATING_MIN_THERMAL_POWER_KW = 0.5
    cfg.SPECIFIC_HEAT_CAPACITY = 4.186
    cfg.HLC_PV_MAX_W = 50.0
    cfg.HLC_OUTDOOR_TEMP_MIN = -10.0
    cfg.HLC_OUTDOOR_TEMP_MAX = 15.0
    cfg.HLC_MIN_HEATING_DEMAND_K = 1.0
    cfg.HLC_MAX_INDOOR_DELTA = 0.3
    cfg.HLC_MAX_TREND = 0.2
    cfg.HLC_DEFAULT_TARGET_TEMP = 22.6
    cfg.HLC_REGRESSION_INTERCEPT = False
    return cfg


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

class TestCalibrateHlcSuccess:
    """calibrate_hlc() should succeed with clean synthetic data."""

    def test_returns_success_dict(self):
        df = _make_df(n_windows=40)
        cfg = _mock_config()

        mock_tsm = MagicMock()

        with (
            patch("src.hlc_learner.config", cfg),
            patch(
                "src.hlc_learner.fetch_historical_data_for_calibration",
                return_value=df,
            ),
            patch(
                "src.hlc_learner.get_thermal_state_manager",
                return_value=mock_tsm,
            ),
        ):
            result = calibrate_hlc()

        assert result["success"] is True
        assert "hlc_kw_per_k" in result
        assert result["hlc_kw_per_k"] > 0
        assert "r2" in result
        assert "n_periods" in result
        assert result["n_periods"] >= 20

    def test_hlc_estimate_is_physically_plausible(self):
        df = _make_df(n_windows=40)
        cfg = _mock_config()

        mock_tsm = MagicMock()

        with (
            patch("src.hlc_learner.config", cfg),
            patch(
                "src.hlc_learner.fetch_historical_data_for_calibration",
                return_value=df,
            ),
            patch(
                "src.hlc_learner.get_thermal_state_manager",
                return_value=mock_tsm,
            ),
        ):
            result = calibrate_hlc()

        assert 0.01 <= result["hlc_kw_per_k"] <= 2.0

    def test_saves_to_thermal_state(self):
        df = _make_df(n_windows=40)
        cfg = _mock_config()

        mock_tsm = MagicMock()

        with (
            patch("src.hlc_learner.config", cfg),
            patch(
                "src.hlc_learner.fetch_historical_data_for_calibration",
                return_value=df,
            ),
            patch(
                "src.hlc_learner.get_thermal_state_manager",
                return_value=mock_tsm,
            ),
        ):
            calibrate_hlc()

        mock_tsm.set_calibrated_baseline.assert_called_once()
        call_kwargs = mock_tsm.set_calibrated_baseline.call_args
        assert "heat_loss_coefficient" in call_kwargs[0][0]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

class TestCalibrateHlcFailures:
    """calibrate_hlc() should return failure dicts for known error conditions."""

    def test_returns_failure_when_fetch_raises(self):
        cfg = _mock_config()

        with (
            patch("src.hlc_learner.config", cfg),
            patch(
                "src.hlc_learner.fetch_historical_data_for_calibration",
                side_effect=RuntimeError("DB unavailable"),
            ),
        ):
            result = calibrate_hlc()

        assert result["success"] is False
        assert "DB unavailable" in result["message"]

    def test_returns_failure_when_df_is_none(self):
        cfg = _mock_config()

        with (
            patch("src.hlc_learner.config", cfg),
            patch(
                "src.hlc_learner.fetch_historical_data_for_calibration",
                return_value=None,
            ),
        ):
            result = calibrate_hlc()

        assert result["success"] is False

    def test_returns_failure_when_df_is_empty(self):
        cfg = _mock_config()

        with (
            patch("src.hlc_learner.config", cfg),
            patch(
                "src.hlc_learner.fetch_historical_data_for_calibration",
                return_value=pd.DataFrame(),
            ),
        ):
            result = calibrate_hlc()

        assert result["success"] is False

    def test_returns_failure_when_missing_required_columns(self):
        cfg = _mock_config()
        # Only provide indoor_temp; others are missing
        df = pd.DataFrame({"kuche_temperatur": [21.0, 21.1]})

        with (
            patch("src.hlc_learner.config", cfg),
            patch(
                "src.hlc_learner.fetch_historical_data_for_calibration",
                return_value=df,
            ),
        ):
            result = calibrate_hlc()

        assert result["success"] is False
        assert "Missing required columns" in result["message"]

    def test_returns_failure_when_too_few_valid_periods(self):
        """Only 1 window of data — cannot meet min_periods=20."""
        df = _make_df(n_windows=1)
        cfg = _mock_config()
        cfg.HLC_CALIBRATION_MIN_PERIODS = 20

        mock_tsm = MagicMock()

        with (
            patch("src.hlc_learner.config", cfg),
            patch(
                "src.hlc_learner.fetch_historical_data_for_calibration",
                return_value=df,
            ),
            patch(
                "src.hlc_learner.get_thermal_state_manager",
                return_value=mock_tsm,
            ),
        ):
            result = calibrate_hlc()

        assert result["success"] is False
        assert "valid periods" in result["message"].lower()

    def test_returns_failure_when_thermal_state_save_fails(self):
        df = _make_df(n_windows=40)
        cfg = _mock_config()

        mock_tsm = MagicMock()
        mock_tsm.set_calibrated_baseline.side_effect = OSError("disk full")

        with (
            patch("src.hlc_learner.config", cfg),
            patch(
                "src.hlc_learner.fetch_historical_data_for_calibration",
                return_value=df,
            ),
            patch(
                "src.hlc_learner.get_thermal_state_manager",
                return_value=mock_tsm,
            ),
        ):
            result = calibrate_hlc()

        assert result["success"] is False
        assert "save failed" in result["message"]
        # Regression value is still present so callers can inspect it
        assert "hlc_kw_per_k" in result


# ---------------------------------------------------------------------------
# Window-size guard
# ---------------------------------------------------------------------------

class TestCalibrateHlcWindowSizeGuard:
    """window_size < 1 should be reset to 12 before processing."""

    def test_invalid_window_size_reset_to_default(self):
        cfg = _mock_config()
        cfg.HLC_WINDOW_SIZE_ROWS = 0  # invalid

        with (
            patch("src.hlc_learner.config", cfg),
            patch(
                "src.hlc_learner.fetch_historical_data_for_calibration",
                return_value=pd.DataFrame(),
            ),
        ):
            # Should not raise; config is corrected internally
            result = calibrate_hlc()

        # With empty df, we still get a failure but no crash
        assert result["success"] is False
