"""Tests for fireplace batch calibration and remaining channel param calibration."""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.physics_calibration import (
    calibrate_delta_t_floor,
    calibrate_fp_decay_tau,
    calibrate_room_spread_delay,
    calibrate_slab_time_constant,
    calibrate_solar_decay_tau,
    filter_cloudy_pv_periods,
    filter_fp_decay_periods,
    filter_fp_spread_periods,
    filter_pv_decay_periods,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal 5-min DataFrame
# ---------------------------------------------------------------------------
def _make_df(rows, columns, start="2026-01-15 18:00"):
    """Build a calibration-style DataFrame with ``_time`` column."""
    idx = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    df = pd.DataFrame(rows, columns=columns)
    df.insert(0, "_time", idx)
    return df


# ===================================================================
# Phase 2 – Fireplace decay calibration
# ===================================================================
class TestFilterFpDecayPeriods:

    def _cols(self):
        from src import config
        return {
            "fp": config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1],
            "indoor": config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1],
            "outdoor": config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1],
            "outlet": config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1],
            "dhw": config.DHW_STATUS_ENTITY_ID.split(".", 1)[-1],
        }

    def test_basic(self):
        c = self._cols()
        # FP on for 4 steps (20min), then off for 24 steps (2h)
        # With hlc=0.15, oe=0.7: hp_eq = (0.7*30 + 0.15*2) / (0.7+0.15) = 25.06
        # Indoor excess above hp_eq baseline decays exponentially
        rows = []
        for i in range(4):
            rows.append({c["fp"]: 1, c["indoor"]: 26.0 + i * 0.1,
                         c["outdoor"]: 2.0, c["outlet"]: 30.0, c["dhw"]: 0})
        hp_eq = (0.7 * 30.0 + 0.15 * 2.0) / (0.7 + 0.15)  # ~25.06
        for i in range(30):
            # Simulate exponential decay of indoor excess above HP eq
            excess = 2.0 * math.exp(-i * 5 / 45.0)  # τ ≈ 45min
            rows.append({c["fp"]: 0, c["indoor"]: hp_eq + excess,
                         c["outdoor"]: 2.0, c["outlet"]: 30.0, c["dhw"]: 0})
        df = _make_df(rows, list(rows[0].keys()))
        periods = filter_fp_decay_periods(df, min_on_minutes=15, hlc=0.15, oe=0.7)
        assert len(periods) >= 1
        assert len(periods[0]["indoor_excess"]) >= 3

    def test_too_short(self):
        """FP on for 5min → excluded (needs ≥15min)."""
        c = self._cols()
        rows = []
        rows.append({c["fp"]: 1, c["indoor"]: 22.0, c["outdoor"]: 2.0, c["outlet"]: 30.0, c["dhw"]: 0})
        for i in range(30):
            rows.append({c["fp"]: 0, c["indoor"]: 20.0, c["outdoor"]: 2.0, c["outlet"]: 30.0, c["dhw"]: 0})
        df = _make_df(rows, list(rows[0].keys()))
        periods = filter_fp_decay_periods(df, min_on_minutes=15)
        assert len(periods) == 0

    def test_blocking_excluded(self):
        """DHW active during FP off → excluded."""
        c = self._cols()
        rows = []
        for i in range(4):
            rows.append({c["fp"]: 1, c["indoor"]: 22.0, c["outdoor"]: 2.0, c["outlet"]: 30.0, c["dhw"]: 0})
        for i in range(30):
            rows.append({c["fp"]: 0, c["indoor"]: 21.0, c["outdoor"]: 2.0, c["outlet"]: 30.0, c["dhw"]: 1})
        df = _make_df(rows, list(rows[0].keys()))
        periods = filter_fp_decay_periods(df, min_on_minutes=15)
        assert len(periods) == 0


class TestCalibrateFpDecayTau:

    def _make_decay_periods(self, true_tau_h, n_periods=8, noise=0.0):
        """Generate synthetic FP decay periods with known τ."""
        tau_min = true_tau_h * 60.0
        periods = []
        for _ in range(n_periods):
            A = np.random.uniform(1.0, 3.0)
            pts = []
            for k in range(20):
                t = k * 5  # minutes
                val = A * math.exp(-t / tau_min) + np.random.normal(0, noise)
                if val > 0.1:
                    pts.append((t, val))
            if len(pts) >= 3:
                periods.append({"indoor_excess": pts, "outdoor_temp": 2.0, "outlet_temp": 30.0})
        return periods

    def test_known_tau(self):
        """Synthetic τ=0.75h → calibrated ≈ 0.75 ± 0.15."""
        periods = self._make_decay_periods(0.75, n_periods=10)
        tau = calibrate_fp_decay_tau(periods)
        assert tau is not None
        assert abs(tau - 0.75) < 0.15

    def test_noisy(self):
        """Synthetic + noise → still converges within bounds."""
        periods = self._make_decay_periods(0.75, n_periods=15, noise=0.05)
        tau = calibrate_fp_decay_tau(periods)
        assert tau is not None
        assert 0.1 <= tau <= 6.0

    def test_insufficient(self):
        """Only 3 periods → returns None."""
        periods = self._make_decay_periods(0.75, n_periods=3)
        tau = calibrate_fp_decay_tau(periods)
        assert tau is None


# ===================================================================
# Phase 2 – Fireplace spread calibration
# ===================================================================
class TestFilterFpSpreadPeriods:

    def _cols(self):
        from src import config
        return {
            "fp": config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1],
            "indoor": config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1],
            "lr": config.LIVING_ROOM_TEMP_ENTITY_ID.split(".", 1)[-1],
        }

    def test_basic(self):
        c = self._cols()
        rows = []
        # 6 FP-on events of 25min each (5 steps), separated by 10 off steps
        for ev in range(6):
            for s in range(5):
                rows.append({c["fp"]: 1, c["indoor"]: 20.0 + s * 0.05, c["lr"]: 20.0 + s * 0.15})
            for s in range(10):
                rows.append({c["fp"]: 0, c["indoor"]: 20.25, c["lr"]: 20.75})
        df = _make_df(rows, list(rows[0].keys()))
        periods = filter_fp_spread_periods(df, min_on_minutes=20)
        assert len(periods) >= 5

    def test_no_living_room(self):
        """Living room column missing → returns empty."""
        from src import config
        c = {
            "fp": config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1],
            "indoor": config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1],
        }
        rows = [{c["fp"]: 1, c["indoor"]: 20.0}] * 10
        df = _make_df(rows, list(rows[0].keys()))
        periods = filter_fp_spread_periods(df)
        assert len(periods) == 0


class TestCalibrateRoomSpreadDelay:

    def test_known_delay(self):
        """Synthetic signals with 30min (6-step) lag → calibrated ≈ 30 ± 15."""
        rng = np.random.RandomState(42)
        periods = []
        for _ in range(12):
            n = 30
            lr = np.cumsum(rng.randn(n) * 0.3) + 20.0
            # avg lags living room by 6 steps
            avg = np.zeros(n)
            for i in range(n):
                src_idx = max(0, i - 6)
                avg[i] = lr[src_idx]
            periods.append({"living_room": lr, "indoor_avg": avg, "duration_steps": n})
        delay = calibrate_room_spread_delay(periods)
        assert delay is not None
        assert abs(delay - 30) <= 15  # 30min ± 15

    def test_no_lag(self):
        """Identical signals → delay ≈ 0."""
        rng = np.random.RandomState(123)
        periods = []
        for _ in range(8):
            sig = np.cumsum(rng.randn(15) * 0.1) + 20.0
            periods.append({"living_room": sig, "indoor_avg": sig.copy(), "duration_steps": 15})
        delay = calibrate_room_spread_delay(periods)
        assert delay is not None
        assert delay <= 10  # near zero


# ===================================================================
# Phase 5 – delta_t_floor
# ===================================================================
class TestCalibrateDeltaTFloor:

    def test_percentile10(self):
        """Synthetic outlet-inlet → correct P25."""
        periods = []
        for dt in np.linspace(1.5, 5.0, 50):
            periods.append({"outlet_temp": 30.0, "inlet_temp": 30.0 - dt, "thermal_power_kw": 2.0})
        result = calibrate_delta_t_floor(periods)
        assert result is not None
        expected_p25 = float(np.percentile(np.linspace(1.5, 5.0, 50), 25))
        assert abs(result - expected_p25) < 0.2

    def test_no_inlet(self):
        """No inlet_temp → returns None."""
        periods = [{"outlet_temp": 30.0, "thermal_power_kw": 2.0}] * 20
        result = calibrate_delta_t_floor(periods)
        assert result is None

    def test_low_flow_excluded(self):
        """thermal_power_kw < 0.5 → excluded."""
        periods = [{"outlet_temp": 30.0, "inlet_temp": 27.0, "thermal_power_kw": 0.4}] * 20
        result = calibrate_delta_t_floor(periods)
        assert result is None


# ===================================================================
# Phase 5 – cloud factor
# ===================================================================
class TestCalibrateCloudFactor:

    def test_known_exponent(self):
        """Filter returns periods with PV + cloud data."""
        from src import config
        pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
        indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
        rows = []
        for _ in range(40):
            rows.append({pv_col: 3000.0, "cloud_cover_proxy": 50.0, indoor_col: 21.0})
        df = _make_df(rows, list(rows[0].keys()))
        periods = filter_cloudy_pv_periods(df)
        assert len(periods) >= 30

    def test_insufficient_data(self):
        """< 30 periods → returns empty."""
        from src import config
        pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
        indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
        rows = [{pv_col: 3000.0, "cloud_cover_proxy": 50.0, indoor_col: 21.0}] * 10
        df = _make_df(rows, list(rows[0].keys()))
        periods = filter_cloudy_pv_periods(df)
        assert len(periods) == 0


# ===================================================================
# Phase 5 – solar decay tau
# ===================================================================
class TestFilterPvDecayPeriods:

    def test_basic(self):
        from src import config
        pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
        indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
        outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
        rows = []
        # PV high then drop
        rows.append({pv_col: 6000, indoor_col: 22.0, outdoor_col: 10.0})
        for i in range(30):
            excess = 2.0 * math.exp(-i * 5 / 30.0)
            rows.append({pv_col: 50, indoor_col: 10.0 + excess, outdoor_col: 10.0})
        df = _make_df(rows, list(rows[0].keys()))
        periods = filter_pv_decay_periods(df)
        assert len(periods) >= 1

    def test_insufficient(self):
        """< 10 transitions → returns None from calibrator."""
        periods = [{"indoor_excess": [(0, 1.0), (5, 0.8), (10, 0.6)]}] * 5
        tau = calibrate_solar_decay_tau(periods)
        assert tau is None


# ===================================================================
# Phase 5 – slab tau co-optimization placeholder
# ===================================================================
class TestSlabTauCoOptimization:

    def test_bounds(self):
        """Slab tau calibration is bounded [0.5, 3.0]h."""
        from src.thermal_config import ThermalParameterConfig
        bounds = ThermalParameterConfig.get_bounds("slab_time_constant_hours")
        assert bounds[0] == 0.5
        assert bounds[1] == 3.0

    def test_pump_on_approach_detection(self):
        """Slab tau detects pump-ON inlet approach toward outlet - delta_t_floor."""
        tau_true = 1.0  # hours
        delta_t_floor = 3.0
        outlet_steady = 35.0
        target = outlet_steady - delta_t_floor  # 32°C
        inlet_0 = 22.0  # cold slab at startup

        # Build 3 HP-startup events: HP-off then HP-on with inlet approach.
        segments = []
        for seg in range(3):
            # HP-off segment (6 samples): outlet ≈ inlet, power ≈ 0
            for i in range(6):
                t_offset = seg * 24 + i
                segments.append({
                    '_time': pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * t_offset),
                    'inlet': inlet_0,
                    'outlet': inlet_0 + 0.1,  # tiny delta → power ≈ 0
                    'flow': 12.2,
                    'indoor': 22.0,
                })
            # HP-on segment (18 samples = 90 min): outlet=35, inlet approaches target
            for i in range(18):
                t_offset = seg * 24 + 6 + i
                t_h = i * 5.0 / 60.0
                inlet_val = target + (inlet_0 - target) * math.exp(-t_h / tau_true)
                segments.append({
                    '_time': pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * t_offset),
                    'inlet': inlet_val,
                    'outlet': outlet_steady,
                    'flow': 12.2,
                    'indoor': 22.0,
                })

        df = pd.DataFrame(segments)

        from unittest.mock import patch
        with patch('src.physics_calibration.config') as mock_cfg:
            mock_cfg.INLET_TEMP_ENTITY_ID = "sensor.inlet"
            mock_cfg.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID = "sensor.outlet"
            mock_cfg.FLOW_RATE_ENTITY_ID = "sensor.flow"
            mock_cfg.INDOOR_TEMP_ENTITY_ID = "sensor.indoor"
            mock_cfg.SPECIFIC_HEAT_CAPACITY = 4.186
            result = calibrate_slab_time_constant(df, delta_t_floor=delta_t_floor)

        assert result is not None
        assert 0.5 < result < 2.0, f"Expected ~1.0h, got {result}"

    def test_no_startup_returns_none(self):
        """Returns None when HP is always on (no startup transitions)."""
        n = 30
        df = pd.DataFrame({
            '_time': pd.date_range("2024-01-01", periods=n, freq="5min"),
            'inlet': np.full(n, 30.0),
            'outlet': np.full(n, 35.0),    # always 5°C delta → HP on
            'flow': np.full(n, 12.2),
            'indoor': np.full(n, 22.0),
        })

        from unittest.mock import patch
        with patch('src.physics_calibration.config') as mock_cfg:
            mock_cfg.INLET_TEMP_ENTITY_ID = "sensor.inlet"
            mock_cfg.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID = "sensor.outlet"
            mock_cfg.FLOW_RATE_ENTITY_ID = "sensor.flow"
            mock_cfg.INDOOR_TEMP_ENTITY_ID = "sensor.indoor"
            mock_cfg.SPECIFIC_HEAT_CAPACITY = 4.186
            result = calibrate_slab_time_constant(df, delta_t_floor=3.0)

        assert result is None
