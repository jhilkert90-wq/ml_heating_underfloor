"""
Tests for the cooling physics calibration path (physics_calibration_cooling).
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# TestParsePhysicsStartDate (direct, not just via alias)
# ---------------------------------------------------------------------------

class TestParsePhysicsStartDate:
    """Tests for the unified _parse_physics_start_date helper."""

    def test_valid_date(self):
        from src.config import _parse_physics_start_date
        dt = _parse_physics_start_date("01.03.2022")
        assert dt is not None
        assert dt.year == 2022 and dt.month == 3 and dt.day == 1
        assert dt.tzinfo == timezone.utc

    def test_empty_string(self):
        from src.config import _parse_physics_start_date
        assert _parse_physics_start_date("") is None

    def test_none_input(self):
        from src.config import _parse_physics_start_date
        assert _parse_physics_start_date(None) is None

    def test_whitespace_only(self):
        from src.config import _parse_physics_start_date
        assert _parse_physics_start_date("   ") is None

    def test_wrong_format_iso(self):
        from src.config import _parse_physics_start_date
        assert _parse_physics_start_date("2022-03-01") is None

    def test_wrong_format_slash(self):
        from src.config import _parse_physics_start_date
        assert _parse_physics_start_date("01/03/2022") is None

    def test_alias_is_same_function(self):
        from src.config import _parse_physics_start_date, _parse_cooling_physics_start_date
        assert _parse_cooling_physics_start_date is _parse_physics_start_date


# ---------------------------------------------------------------------------
# TestApplyOutdoorRollingFilter
# ---------------------------------------------------------------------------

class TestApplyOutdoorRollingFilter:
    """Tests for _apply_outdoor_rolling_filter."""

    def _make_df(self, n_rows=300, outdoor_temps=None):
        """Build a DataFrame with the required outdoor temp column."""
        if outdoor_temps is None:
            outdoor_temps = [20.0] * n_rows
        data = {
            "thermometer_waermepume_kompensiert": outdoor_temps[:n_rows],
        }
        return pd.DataFrame(data)

    def test_all_above_threshold(self):
        from src.physics_calibration_cooling import _apply_outdoor_rolling_filter
        df = self._make_df(n_rows=300, outdoor_temps=[20.0] * 300)
        with patch("src.physics_calibration_cooling.config") as mc:
            mc.OUTDOOR_TEMP_ENTITY_ID = "sensor.thermometer_waermepume_kompensiert"
            mc.COOLING_PHYSICS_MIN_OUTDOOR_ROLLING_24H_C = 16.0
            result = _apply_outdoor_rolling_filter(df)
        # First 144 rows (half-window min_periods) will be NaN and filtered out
        assert len(result) > 0
        assert len(result) <= 300

    def test_all_below_threshold(self):
        from src.physics_calibration_cooling import _apply_outdoor_rolling_filter
        df = self._make_df(n_rows=300, outdoor_temps=[10.0] * 300)
        with patch("src.physics_calibration_cooling.config") as mc:
            mc.OUTDOOR_TEMP_ENTITY_ID = "sensor.thermometer_waermepume_kompensiert"
            mc.COOLING_PHYSICS_MIN_OUTDOOR_ROLLING_24H_C = 16.0
            result = _apply_outdoor_rolling_filter(df)
        assert len(result) == 0

    def test_missing_column_returns_empty(self):
        from src.physics_calibration_cooling import _apply_outdoor_rolling_filter
        df = pd.DataFrame({"other_col": [20.0] * 10})
        with patch("src.physics_calibration_cooling.config") as mc:
            mc.OUTDOOR_TEMP_ENTITY_ID = "sensor.thermometer_waermepume_kompensiert"
            mc.COOLING_PHYSICS_MIN_OUTDOOR_ROLLING_24H_C = 16.0
            result = _apply_outdoor_rolling_filter(df)
        assert result.empty

    def test_empty_df_returns_empty(self):
        from src.physics_calibration_cooling import _apply_outdoor_rolling_filter
        df = pd.DataFrame({"thermometer_waermepume_kompensiert": []})
        with patch("src.physics_calibration_cooling.config") as mc:
            mc.OUTDOOR_TEMP_ENTITY_ID = "sensor.thermometer_waermepume_kompensiert"
            mc.COOLING_PHYSICS_MIN_OUTDOOR_ROLLING_24H_C = 16.0
            result = _apply_outdoor_rolling_filter(df)
        assert result.empty


# ---------------------------------------------------------------------------
# TestFilterStablePeriodsCooling
# ---------------------------------------------------------------------------

class TestFilterStablePeriodsCooling:
    """Tests for filter_stable_periods_cooling."""

    def _make_cooling_df(self, n_rows=50, indoor=24.0, outdoor=30.0, outlet=20.0):
        """Build a DataFrame simulating stable cooling conditions."""
        data = {
            "kuche_temperatur": [indoor] * n_rows,
            "thermometer_waermepume_kompensiert": [outdoor] * n_rows,
            "hp_outlet_temp": [outlet] * n_rows,
            "hp_inlet_temp": [outlet + 3.0] * n_rows,
            "pv_power": [0.0] * n_rows,
            "hp_current_flow_rate": [8.0] * n_rows,
        }
        return pd.DataFrame(data)

    def test_stable_cooling_returns_periods(self):
        from src.physics_calibration_cooling import filter_stable_periods_cooling
        df = self._make_cooling_df(n_rows=50, indoor=24.0, outdoor=30.0, outlet=20.0)
        with patch("src.physics_calibration_cooling.config") as mc:
            mc.INDOOR_TEMP_ENTITY_ID = "sensor.kuche_temperatur"
            mc.OUTDOOR_TEMP_ENTITY_ID = "sensor.thermometer_waermepume_kompensiert"
            mc.ACTUAL_OUTLET_TEMP_ENTITY_ID = "sensor.hp_outlet_temp"
            mc.INLET_TEMP_ENTITY_ID = "sensor.hp_inlet_temp"
            mc.PV_POWER_ENTITY_ID = "sensor.pv_power"
            mc.FLOW_RATE_ENTITY_ID = "sensor.hp_current_flow_rate"
            mc.STABILITY_TEMP_CHANGE_THRESHOLD = 0.2
            mc.MIN_STABLE_PERIOD_MINUTES = 20
            mc.SPECIFIC_HEAT_CAPACITY = 4.186
            result = filter_stable_periods_cooling(df)
        assert len(result) > 0
        # Check period dict structure
        p = result[0]
        assert "indoor_temp" in p
        assert "outdoor_temp" in p
        assert "outlet_temp" in p
        assert "thermal_power_kw" in p

    def test_heating_data_rejected(self):
        """Rows where outlet >> indoor (heating, not cooling) should be rejected."""
        from src.physics_calibration_cooling import filter_stable_periods_cooling
        # outlet > indoor + 1.0 → rejected
        df = self._make_cooling_df(n_rows=50, indoor=20.0, outdoor=5.0, outlet=35.0)
        with patch("src.physics_calibration_cooling.config") as mc:
            mc.INDOOR_TEMP_ENTITY_ID = "sensor.kuche_temperatur"
            mc.OUTDOOR_TEMP_ENTITY_ID = "sensor.thermometer_waermepume_kompensiert"
            mc.ACTUAL_OUTLET_TEMP_ENTITY_ID = "sensor.hp_outlet_temp"
            mc.INLET_TEMP_ENTITY_ID = "sensor.hp_inlet_temp"
            mc.PV_POWER_ENTITY_ID = "sensor.pv_power"
            mc.FLOW_RATE_ENTITY_ID = "sensor.hp_current_flow_rate"
            mc.STABILITY_TEMP_CHANGE_THRESHOLD = 0.2
            mc.MIN_STABLE_PERIOD_MINUTES = 20
            mc.SPECIFIC_HEAT_CAPACITY = 4.186
            result = filter_stable_periods_cooling(df)
        assert len(result) == 0

    def test_unstable_indoor_temps_rejected(self):
        """Large indoor temperature fluctuations should be rejected."""
        from src.physics_calibration_cooling import filter_stable_periods_cooling
        rng = np.random.default_rng(42)
        n = 50
        # Indoor temps with large range (> 0.2 in any window)
        indoor_temps = [20.0 + 2.0 * np.sin(i / 2.0) for i in range(n)]
        data = {
            "kuche_temperatur": indoor_temps,
            "thermometer_waermepume_kompensiert": [30.0] * n,
            "hp_outlet_temp": [18.0] * n,
            "hp_inlet_temp": [21.0] * n,
            "pv_power": [0.0] * n,
            "hp_current_flow_rate": [8.0] * n,
        }
        df = pd.DataFrame(data)
        with patch("src.physics_calibration_cooling.config") as mc:
            mc.INDOOR_TEMP_ENTITY_ID = "sensor.kuche_temperatur"
            mc.OUTDOOR_TEMP_ENTITY_ID = "sensor.thermometer_waermepume_kompensiert"
            mc.ACTUAL_OUTLET_TEMP_ENTITY_ID = "sensor.hp_outlet_temp"
            mc.INLET_TEMP_ENTITY_ID = "sensor.hp_inlet_temp"
            mc.PV_POWER_ENTITY_ID = "sensor.pv_power"
            mc.FLOW_RATE_ENTITY_ID = "sensor.hp_current_flow_rate"
            mc.STABILITY_TEMP_CHANGE_THRESHOLD = 0.2
            mc.MIN_STABLE_PERIOD_MINUTES = 20
            mc.SPECIFIC_HEAT_CAPACITY = 4.186
            result = filter_stable_periods_cooling(df)
        assert len(result) == 0

    def test_empty_df_returns_empty_list(self):
        from src.physics_calibration_cooling import filter_stable_periods_cooling
        df = pd.DataFrame()
        with patch("src.physics_calibration_cooling.config") as mc:
            mc.INDOOR_TEMP_ENTITY_ID = "sensor.kuche_temperatur"
            mc.OUTDOOR_TEMP_ENTITY_ID = "sensor.thermometer_waermepume_kompensiert"
            mc.ACTUAL_OUTLET_TEMP_ENTITY_ID = "sensor.hp_outlet_temp"
            mc.INLET_TEMP_ENTITY_ID = "sensor.hp_inlet_temp"
            mc.PV_POWER_ENTITY_ID = "sensor.pv_power"
            mc.FLOW_RATE_ENTITY_ID = "sensor.hp_current_flow_rate"
            mc.STABILITY_TEMP_CHANGE_THRESHOLD = 0.2
            mc.MIN_STABLE_PERIOD_MINUTES = 20
            mc.SPECIFIC_HEAT_CAPACITY = 4.186
            result = filter_stable_periods_cooling(df)
        assert result == []


# ---------------------------------------------------------------------------
# TestFilterCoolingActivePeriods
# ---------------------------------------------------------------------------

class TestFilterCoolingActivePeriods:
    """Tests for _filter_cooling_active_periods."""

    def test_selects_cooling_periods_no_pv(self):
        from src.physics_calibration_cooling import _filter_cooling_active_periods
        periods = [
            {"indoor_temp": 24.0, "outlet_temp": 20.0, "pv_power": 0.0},  # cooling, no PV
            {"indoor_temp": 24.0, "outlet_temp": 20.0, "pv_power": 500.0},  # PV too high
            {"indoor_temp": 20.0, "outlet_temp": 24.0, "pv_power": 0.0},  # heating, not cooling
        ]
        result = _filter_cooling_active_periods(periods)
        assert len(result) == 1
        assert result[0]["indoor_temp"] == 24.0 and result[0]["pv_power"] == 0.0

    def test_empty_list_returns_empty(self):
        from src.physics_calibration_cooling import _filter_cooling_active_periods
        assert _filter_cooling_active_periods([]) == []


# ---------------------------------------------------------------------------
# TestCalibrateOECooling
# ---------------------------------------------------------------------------

class TestCalibrateOECooling:
    """Tests for _calibrate_oe_cooling."""

    def _make_cooling_periods(self, n=30, hlc=0.12, oe=0.8):
        """Generate cooling periods consistent with OE = hlc*(T_out-T_in)/(T_in-T_outlet)."""
        rng = np.random.default_rng(42)
        periods = []
        for _ in range(n):
            t_outdoor = rng.uniform(28, 35)  # warm outdoor
            t_indoor = rng.uniform(23, 25)   # comfortable indoor
            # OE = hlc * (t_outdoor - t_indoor) / (t_indoor - t_outlet)
            # => t_indoor - t_outlet = hlc * (t_outdoor - t_indoor) / oe
            drive = hlc * (t_outdoor - t_indoor) / oe
            t_outlet = t_indoor - drive
            periods.append({
                "indoor_temp": float(t_indoor),
                "outdoor_temp": float(t_outdoor),
                "outlet_temp": float(t_outlet),
                "effective_temp": float(t_outlet),
                "pv_power": 0.0,
            })
        return periods

    def test_recovers_known_oe(self):
        from src.physics_calibration_cooling import _calibrate_oe_cooling
        true_hlc = 0.12
        true_oe = 0.8
        periods = self._make_cooling_periods(n=50, hlc=true_hlc, oe=true_oe)

        with patch("src.physics_calibration_cooling.ThermalParameterConfig") as tpc:
            tpc.get_cooling_bounds.return_value = (0.01, 5.0)
            result = _calibrate_oe_cooling(periods, true_hlc)

        assert result is not None
        assert abs(result - true_oe) / true_oe < 0.05, (
            f"OE estimate {result:.4f} deviates >5% from true {true_oe}"
        )

    def test_returns_none_for_zero_hlc(self):
        from src.physics_calibration_cooling import _calibrate_oe_cooling
        result = _calibrate_oe_cooling([], hlc=0.0)
        assert result is None

    def test_returns_none_for_negative_hlc(self):
        from src.physics_calibration_cooling import _calibrate_oe_cooling
        result = _calibrate_oe_cooling([], hlc=-1.0)
        assert result is None

    def test_returns_none_for_insufficient_periods(self):
        from src.physics_calibration_cooling import _calibrate_oe_cooling
        periods = self._make_cooling_periods(n=5, hlc=0.12, oe=0.8)
        with patch("src.physics_calibration_cooling.ThermalParameterConfig") as tpc:
            tpc.get_cooling_bounds.return_value = (0.01, 5.0)
            result = _calibrate_oe_cooling(periods, 0.12)
        assert result is None


# ---------------------------------------------------------------------------
# TestCalibrateHLCCooling
# ---------------------------------------------------------------------------

class TestCalibrateHLCCooling:
    """Tests for _calibrate_hlc_cooling."""

    def _make_hlc_periods(self, n=30, hlc=0.12):
        """Generate cooling-active periods with known HLC relationship."""
        rng = np.random.default_rng(42)
        periods = []
        for _ in range(n):
            t_outdoor = rng.uniform(28, 35)
            t_indoor = rng.uniform(22, 24)
            dt = t_outdoor - t_indoor
            # cooling power = HLC * dt (at equilibrium)
            q_cool = hlc * dt
            periods.append({
                "indoor_temp": float(t_indoor),
                "outdoor_temp": float(t_outdoor),
                "outlet_temp": float(t_indoor - 3.0),  # outlet < indoor
                "thermal_power_kw": float(-q_cool),  # negative for cooling
                "pv_power": 0.0,
            })
        return periods

    def test_recovers_known_hlc(self):
        from src.physics_calibration_cooling import _calibrate_hlc_cooling
        true_hlc = 0.12
        periods = self._make_hlc_periods(n=40, hlc=true_hlc)

        with patch("src.physics_calibration_cooling.config") as mc, \
             patch("src.physics_calibration_cooling.ThermalParameterConfig") as tpc:
            mc.COOLING_MIN_THERMAL_POWER_KW = -0.5
            tpc.get_cooling_bounds.return_value = (0.01, 2.0)
            result = _calibrate_hlc_cooling(periods)

        assert result is not None
        assert abs(result - true_hlc) / true_hlc < 0.10, (
            f"HLC estimate {result:.5f} deviates >10% from true {true_hlc}"
        )

    def test_returns_none_for_insufficient_periods(self):
        from src.physics_calibration_cooling import _calibrate_hlc_cooling
        periods = self._make_hlc_periods(n=5, hlc=0.12)
        with patch("src.physics_calibration_cooling.config") as mc:
            mc.COOLING_MIN_THERMAL_POWER_KW = -0.5
            result = _calibrate_hlc_cooling(periods)
        assert result is None

    def test_returns_none_for_empty_list(self):
        from src.physics_calibration_cooling import _calibrate_hlc_cooling
        with patch("src.physics_calibration_cooling.config") as mc:
            mc.COOLING_MIN_THERMAL_POWER_KW = -0.5
            result = _calibrate_hlc_cooling([])
        assert result is None


# ---------------------------------------------------------------------------
# TestSetCalibratedBaselinePersistsAllKeys
# ---------------------------------------------------------------------------

class TestSetCalibratedBaselinePersistsAllKeys:
    """Verify that set_calibrated_baseline persists cloud_factor_exponent and solar_decay_tau_hours."""

    def test_persists_cloud_and_solar_decay(self):
        from src.unified_thermal_state_cooling import CoolingThermalStateManager
        sm = CoolingThermalStateManager.__new__(CoolingThermalStateManager)
        sm.state = {
            "baseline_parameters": {},
            "learning_state": {
                "parameter_adjustments": {},
            },
        }
        sm.state_file = "/tmp/fake_cooling_state.json"
        sm.save_state = MagicMock()

        params = {
            "heat_loss_coefficient": 0.12,
            "outlet_effectiveness": 0.8,
            "thermal_time_constant": 5.0,
            "pv_heat_weight": 0.002,
            "fireplace_heat_weight": 0.0,
            "tv_heat_weight": 0.0,
            "solar_lag_minutes": 45.0,
            "delta_t_floor": 2.3,
            "slab_time_constant_hours": 1.5,
            "fp_decay_time_constant": 2.0,
            "room_spread_delay_minutes": 30.0,
            "cloud_factor_exponent": 1.2,
            "solar_decay_tau_hours": 3.5,
        }
        sm.set_calibrated_baseline(params, calibration_cycles=50)

        baseline = sm.state["baseline_parameters"]
        assert baseline["cloud_factor_exponent"] == 1.2
        assert baseline["solar_decay_tau_hours"] == 3.5
        assert baseline["source"] == "calibrated"


# ---------------------------------------------------------------------------
# TestCoolingPhysicsCLIDispatch
# ---------------------------------------------------------------------------

class TestCoolingPhysicsCLIDispatch:
    """Tests for --calibrate-cooling-physics CLI argument dispatch."""

    def test_cli_arg_calls_calibrate_cooling_physics(self):
        """The --calibrate-cooling-physics CLI arg should invoke calibrate_cooling_physics."""
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        with patch(
            "src.physics_calibration_cooling.calibrate_cooling_physics",
            return_value=mock_model,
        ) as mock_fn, patch("src.main.load_dotenv"), patch(
            "src.main.create_influx_service",
            return_value=MagicMock(),
        ), patch("sys.argv", [
            "main", "--calibrate-cooling-physics"
        ]):
            from src.main import main
            try:
                main()
            except SystemExit:
                pass

            mock_fn.assert_called_once()

    def test_tau_uses_actual_outlet_when_target_outlet_missing(self):
        """Cooling tau estimation aliases the fetched outlet column for reuse."""
        from src.physics_calibration_cooling import calibrate_cooling_physics
        from src import config

        actual_outlet_col = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
        target_outlet_col = config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
        df = pd.DataFrame(
            {
                "_time": pd.date_range("2026-01-01", periods=120, freq="5min"),
                actual_outlet_col: np.linspace(22.0, 18.0, 120),
            }
        )

        heating_sm = MagicMock()
        heating_sm.state = {"baseline_parameters": {"thermal_time_constant": 4.0}}
        cooling_sm = MagicMock()
        cooling_sm.state = {"baseline_parameters": {}, "learning_state": {}}

        with patch(
            "src.physics_calibration_cooling.get_thermal_state_manager",
            return_value=heating_sm,
        ), patch(
            "src.physics_calibration_cooling.get_cooling_state_manager",
            return_value=cooling_sm,
        ), patch(
            "src.physics_calibration_cooling.fetch_historical_data_for_calibration",
            return_value=df,
        ), patch(
            "src.physics_calibration_cooling._apply_outdoor_rolling_filter",
            return_value=df,
        ), patch(
            "src.physics_calibration_cooling.filter_stable_periods_cooling",
            return_value=[{}] * 20,
        ), patch(
            "src.physics_calibration_cooling._calibrate_hlc_cooling",
            return_value=0.12,
        ), patch(
            "src.physics_calibration_cooling._calibrate_oe_cooling",
            return_value=0.8,
        ), patch(
            "src.physics_calibration_cooling.calculate_cooling_time_constant",
            return_value=(None, 0.0),
        ) as mock_tau, patch(
            "src.physics_calibration_cooling._filter_cooling_pv_periods",
            return_value=[],
        ), patch(
            "src.physics_calibration_cooling._residual_heat_source_weight",
            return_value=0.0,
        ), patch(
            "src.physics_calibration_cooling._calibrate_solar_lag_xcorr",
            return_value=0.0,
        ), patch(
            "src.physics_calibration_cooling.calibrate_delta_t_floor",
            return_value=2.0,
        ), patch(
            "src.physics_calibration_cooling._calibrate_slab_tau_grid_search",
            return_value=None,
        ), patch(
            "src.physics_calibration_cooling.filter_pv_decay_periods",
            return_value=[],
        ), patch(
            "src.physics_calibration_cooling.calibrate_solar_decay_tau",
            return_value=3.0,
        ), patch(
            "src.physics_calibration_cooling.ThermalParameterConfig.get_cooling_default",
            side_effect=lambda key: {
                "thermal_time_constant": 4.0,
                "heat_loss_coefficient": 0.12,
                "outlet_effectiveness": 0.8,
                "pv_heat_weight": 0.0,
                "fireplace_heat_weight": 0.0,
                "tv_heat_weight": 0.0,
                "solar_lag_minutes": 0.0,
                "delta_t_floor": 2.0,
                "slab_time_constant_hours": 1.5,
                "fp_decay_time_constant": 2.0,
                "room_spread_delay_minutes": 30.0,
                "cloud_factor_exponent": 1.2,
                "solar_decay_tau_hours": 3.0,
            }[key],
        ), patch(
            "src.physics_calibration_cooling.ThermalParameterConfig.get_cooling_bounds",
            return_value=(0.0, 10.0),
        ):
            calibrate_cooling_physics()

        mock_tau.assert_called_once()
        tau_df = mock_tau.call_args[0][0]
        assert target_outlet_col in tau_df.columns
        assert tau_df[target_outlet_col].equals(tau_df[actual_outlet_col])


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
