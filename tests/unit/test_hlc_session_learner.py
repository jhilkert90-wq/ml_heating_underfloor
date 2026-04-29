"""
Unit tests for HLCSessionLearner and DayRecord in src/hlc_learner.py.

Covers:
- DayRecord.avg_power_w == mean_thermal_power_kw * 1000
- Days without HP activity (all thermal_power_kw == 0) produce no DayRecord
- Days with fewer active cycles than HLC_SESSION_MIN_CYCLES are rejected
- Day rollover closes the previous day automatically
- load_day_records / _save_day_records round-trip
- estimate_hlc returns None when fewer than HLC_SESSION_MIN_DAYS records
- estimate_hlc OLS correctness with known values
- apply_to_thermal_state caps large updates via HLC_SESSION_MAX_UPDATE_FRACTION
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from src.hlc_learner import DayRecord, HLCSessionLearner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cycle_ctx(
    thermal_power_kw: float = 1.5,
    indoor_temp: float = 20.5,
    outdoor_temp: float = 5.0,
    target_temp: float = 20.5,
    indoor_temp_delta_60m: float = 0.0,
    pv_now_electrical: float = 0.0,
    fireplace_on: float = 0.0,
    tv_on: float = 0.0,
    dhw_heating: float = 0.0,
    defrosting: float = 0.0,
    dhw_boost_heater: float = 0.0,
    is_blocking: bool = False,
    timestamp: datetime | None = None,
) -> Dict:
    return {
        "timestamp": timestamp or datetime.now(),
        "thermal_power_kw": thermal_power_kw,
        "indoor_temp": indoor_temp,
        "outdoor_temp": outdoor_temp,
        "target_temp": target_temp,
        "indoor_temp_delta_60m": indoor_temp_delta_60m,
        "pv_now_electrical": pv_now_electrical,
        "fireplace_on": fireplace_on,
        "tv_on": tv_on,
        "dhw_heating": dhw_heating,
        "defrosting": defrosting,
        "dhw_boost_heater": dhw_boost_heater,
        "is_blocking": is_blocking,
    }


def _push_n_valid_cycles(learner: HLCSessionLearner, n: int = 10) -> None:
    """Push N valid HP cycles (positive power) into the learner without triggering rollover."""
    learner._day_cycles.clear()
    for _ in range(n):
        from src.hlc_learner import HLCLearner
        c = HLCLearner._build_cycle(_cycle_ctx())
        learner._day_cycles.append(c)


# ---------------------------------------------------------------------------
# Config patching helpers
# ---------------------------------------------------------------------------

_CONFIG_DEFAULTS = {
    "HLC_SESSION_MIN_CYCLES": 6,
    "HLC_SESSION_MAX_DAYS": 60,
    "HLC_SESSION_MIN_DAYS": 5,
    "HLC_SESSION_MAX_UPDATE_FRACTION": 0.3,
    "HLC_SESSION_FILE": "",
    "HLC_PV_MAX_W": 50.0,
    "HLC_MAX_INDOOR_DELTA": 0.3,
    "HLC_MAX_TREND": 0.2,
    "HLC_OUTDOOR_TEMP_MIN": -10.0,
    "HLC_OUTDOOR_TEMP_MAX": 15.0,
    "HLC_MIN_HEATING_DEMAND_K": 1.0,
}


def _patch_config(overrides: Dict | None = None):
    """Return a context-manager that patches src.hlc_learner.config."""
    cfg = dict(_CONFIG_DEFAULTS)
    if overrides:
        cfg.update(overrides)
    mock_cfg = MagicMock()
    for k, v in cfg.items():
        setattr(mock_cfg, k, v)
    return patch("src.hlc_learner.config", mock_cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDayRecord:
    def test_avg_power_w_equals_mean_kw_times_1000(self):
        record = DayRecord(
            date="2024-01-01",
            mean_thermal_power_kw=2.5,
            mean_delta_t=12.0,
            n_cycles=10,
            outdoor_temp_mean=5.0,
            indoor_temp_mean=20.0,
            avg_power_w=2500.0,
        )
        assert record.avg_power_w == pytest.approx(record.mean_thermal_power_kw * 1000)

    def test_avg_power_w_field_present(self):
        record = DayRecord(
            date="2024-01-15",
            mean_thermal_power_kw=1.8,
            mean_delta_t=10.0,
            n_cycles=8,
            outdoor_temp_mean=3.0,
            indoor_temp_mean=20.5,
            avg_power_w=1800.0,
        )
        assert hasattr(record, "avg_power_w")


class TestNoHPActivity:
    def test_no_hp_activity_produces_no_record(self, tmp_path):
        """Day with all zero thermal_power_kw → DayRecord not created."""
        with _patch_config({"HLC_SESSION_FILE": str(tmp_path / "sessions.json")}):
            learner = HLCSessionLearner()
            # Simulate a full day of zero-power cycles
            for _ in range(10):
                from src.hlc_learner import HLCLearner
                c = HLCLearner._build_cycle(_cycle_ctx(thermal_power_kw=0.0))
                learner._day_cycles.append(c)

            result = learner._close_day()
            assert result["day_closed"] is False
            assert result["day_validated"] is False
            assert len(learner._day_records) == 0

    def test_zero_power_cycles_do_not_count_as_active(self, tmp_path):
        with _patch_config({"HLC_SESSION_FILE": str(tmp_path / "sessions.json")}):
            learner = HLCSessionLearner()
            for _ in range(20):
                from src.hlc_learner import HLCLearner
                c = HLCLearner._build_cycle(_cycle_ctx(thermal_power_kw=0.0))
                learner._day_cycles.append(c)
            result = learner._close_day()
            # 20 cycles but none with power > 0 → treated as no HP activity
            assert result["day_closed"] is False


class TestBelowMinCycles:
    def test_fewer_than_min_cycles_rejected(self, tmp_path):
        """Day with only 2 active HP cycles is rejected when min is 6."""
        with _patch_config({
            "HLC_SESSION_MIN_CYCLES": 6,
            "HLC_SESSION_FILE": str(tmp_path / "sessions.json"),
        }):
            learner = HLCSessionLearner()
            for _ in range(2):
                from src.hlc_learner import HLCLearner
                c = HLCLearner._build_cycle(_cycle_ctx(thermal_power_kw=1.5))
                learner._day_cycles.append(c)

            result = learner._close_day()
            assert result["day_validated"] is False
            assert "2 active cycles" in result["reject_reason"]
            assert len(learner._day_records) == 0

    def test_exactly_min_cycles_accepted(self, tmp_path):
        """Day with exactly min cycles should pass cycle count gate."""
        min_cycles = 6
        with _patch_config({
            "HLC_SESSION_MIN_CYCLES": min_cycles,
            "HLC_SESSION_FILE": str(tmp_path / "sessions.json"),
        }):
            learner = HLCSessionLearner()
            for _ in range(min_cycles):
                from src.hlc_learner import HLCLearner
                c = HLCLearner._build_cycle(_cycle_ctx(
                    thermal_power_kw=1.5,
                    indoor_temp=20.5,
                    outdoor_temp=5.0,
                    target_temp=20.5,
                ))
                learner._day_cycles.append(c)

            result = learner._close_day()
            # May be accepted or rejected by quality gates — but NOT by cycle count
            if not result["day_validated"] and result["reject_reason"]:
                assert "active cycles" not in result["reject_reason"]


class TestDayRollover:
    def test_day_rollover_closes_previous_day(self, tmp_path):
        """Pushing cycles on a new date closes and validates the previous day."""
        with _patch_config({
            "HLC_SESSION_MIN_CYCLES": 3,
            "HLC_SESSION_FILE": str(tmp_path / "sessions.json"),
        }):
            learner = HLCSessionLearner()
            # Manually set to yesterday
            yesterday = (date.today() - timedelta(days=1)).isoformat()
            learner._today = yesterday

            # Add valid cycles for "yesterday"
            for _ in range(10):
                from src.hlc_learner import HLCLearner
                c = HLCLearner._build_cycle(_cycle_ctx(
                    thermal_power_kw=1.5,
                    indoor_temp=20.5,
                    outdoor_temp=5.0,
                    target_temp=20.5,
                ))
                learner._day_cycles.append(c)

            # Now push a cycle for today — triggers rollover
            result = learner.push_cycle(_cycle_ctx())

            # The day was closed (either validated or rejected by quality gates)
            assert result["day_closed"] is True

    def test_same_day_push_does_not_close(self, tmp_path):
        """Pushing cycles on the same day does not trigger day_closed."""
        with _patch_config({
            "HLC_SESSION_MIN_CYCLES": 3,
            "HLC_SESSION_FILE": str(tmp_path / "sessions.json"),
        }):
            learner = HLCSessionLearner()
            result = learner.push_cycle(_cycle_ctx())
            assert result["day_closed"] is False


class TestLoadSaveRoundtrip:
    def test_load_save_roundtrip(self, tmp_path):
        session_file = str(tmp_path / "sessions.json")
        with _patch_config({"HLC_SESSION_FILE": session_file}):
            learner1 = HLCSessionLearner()
            record = DayRecord(
                date="2024-02-01",
                mean_thermal_power_kw=1.8,
                mean_delta_t=12.0,
                n_cycles=10,
                outdoor_temp_mean=4.0,
                indoor_temp_mean=20.0,
                avg_power_w=1800.0,
            )
            learner1._day_records.append(record)
            learner1._save_day_records()

        with _patch_config({
            "HLC_SESSION_FILE": session_file,
            "HLC_SESSION_MAX_DAYS": 60,
        }):
            learner2 = HLCSessionLearner()
            count = learner2.load_day_records()

        assert count == 1
        loaded = learner2.get_day_records()
        assert loaded[0].date == "2024-02-01"
        assert loaded[0].avg_power_w == pytest.approx(1800.0)

    def test_load_missing_file_returns_zero(self, tmp_path):
        with _patch_config({"HLC_SESSION_FILE": str(tmp_path / "nonexistent.json")}):
            learner = HLCSessionLearner()
            count = learner.load_day_records()
        assert count == 0

    def test_saved_file_contains_avg_power_w(self, tmp_path):
        session_file = str(tmp_path / "sessions.json")
        with _patch_config({"HLC_SESSION_FILE": session_file}):
            learner = HLCSessionLearner()
            learner._day_records.append(DayRecord(
                date="2024-03-01",
                mean_thermal_power_kw=2.0,
                mean_delta_t=10.0,
                n_cycles=8,
                outdoor_temp_mean=3.0,
                indoor_temp_mean=20.0,
                avg_power_w=2000.0,
            ))
            learner._save_day_records()

        with open(session_file) as f:
            data = json.load(f)
        assert "avg_power_w" in data["day_records"][0]
        assert data["day_records"][0]["avg_power_w"] == pytest.approx(2000.0)


class TestEstimateHLC:
    def test_estimate_hlc_below_min_days_returns_none(self):
        with _patch_config({"HLC_SESSION_MIN_DAYS": 5}):
            learner = HLCSessionLearner()
            # Add only 3 records (below min of 5)
            for i in range(3):
                learner._day_records.append(DayRecord(
                    date=f"2024-01-{i+1:02d}",
                    mean_thermal_power_kw=1.5,
                    mean_delta_t=10.0 + i,
                    n_cycles=10,
                    outdoor_temp_mean=5.0,
                    indoor_temp_mean=20.0,
                    avg_power_w=1500.0,
                ))
            hlc, stats = learner.estimate_hlc()
        assert hlc is None
        assert "reject_reason" in stats

    def test_estimate_hlc_ols_correct(self):
        """Known Q and ΔT values: HLC = Σ(Q·ΔT)/Σ(ΔT²)."""
        # Q = HLC * ΔT with HLC = 0.12 kW/K
        hlc_true = 0.12
        dts = [10.0, 12.0, 8.0, 15.0, 11.0]
        qs = [hlc_true * dt for dt in dts]

        with _patch_config({"HLC_SESSION_MIN_DAYS": 5}):
            learner = HLCSessionLearner()
            for i, (q, dt) in enumerate(zip(qs, dts)):
                learner._day_records.append(DayRecord(
                    date=f"2024-01-{i+1:02d}",
                    mean_thermal_power_kw=q,
                    mean_delta_t=dt,
                    n_cycles=10,
                    outdoor_temp_mean=5.0,
                    indoor_temp_mean=20.0,
                    avg_power_w=q * 1000,
                ))
            hlc, stats = learner.estimate_hlc()

        assert hlc is not None
        assert hlc == pytest.approx(hlc_true, rel=1e-6)
        assert stats["r2"] == pytest.approx(1.0, abs=1e-4)

    def test_estimate_hlc_degenerate_zero_dt_returns_none(self):
        """When all ΔT values are near zero, estimate returns None."""
        with _patch_config({"HLC_SESSION_MIN_DAYS": 3}):
            learner = HLCSessionLearner()
            for i in range(3):
                learner._day_records.append(DayRecord(
                    date=f"2024-01-{i+1:02d}",
                    mean_thermal_power_kw=1.5,
                    mean_delta_t=0.0,  # degenerate
                    n_cycles=10,
                    outdoor_temp_mean=20.0,
                    indoor_temp_mean=20.0,
                    avg_power_w=1500.0,
                ))
            hlc, stats = learner.estimate_hlc()
        assert hlc is None


class TestApplyToThermalState:
    def test_caps_upward_update(self):
        """Large upward estimate is capped by HLC_SESSION_MAX_UPDATE_FRACTION."""
        with _patch_config({
            "HLC_SESSION_MIN_DAYS": 1,
            "HLC_SESSION_MAX_UPDATE_FRACTION": 0.3,
        }):
            learner = HLCSessionLearner()
            # One record: Q/ΔT gives a very high HLC
            learner._day_records.append(DayRecord(
                date="2024-01-01",
                mean_thermal_power_kw=10.0,  # very high
                mean_delta_t=5.0,
                n_cycles=10,
                outdoor_temp_mean=5.0,
                indoor_temp_mean=20.0,
                avg_power_w=10000.0,
            ))

            mock_tsm = MagicMock()
            mock_tsm.get_computed_parameters.return_value = {
                "heat_loss_coefficient": 0.1  # current: 0.1, estimate ~2.0
            }

            ok, msg = learner.apply_to_thermal_state(mock_tsm)

        assert ok is True
        # The applied value should be capped at 0.1 * 1.3 = 0.13
        call_args = mock_tsm.set_calibrated_baseline.call_args[0][0]
        applied_hlc = call_args["heat_loss_coefficient"]
        assert applied_hlc == pytest.approx(0.13, rel=1e-6)

    def test_caps_downward_update(self):
        """Large downward estimate is capped by HLC_SESSION_MAX_UPDATE_FRACTION."""
        with _patch_config({
            "HLC_SESSION_MIN_DAYS": 1,
            "HLC_SESSION_MAX_UPDATE_FRACTION": 0.3,
        }):
            learner = HLCSessionLearner()
            learner._day_records.append(DayRecord(
                date="2024-01-01",
                mean_thermal_power_kw=0.001,  # very low → tiny HLC estimate
                mean_delta_t=5.0,
                n_cycles=10,
                outdoor_temp_mean=5.0,
                indoor_temp_mean=20.0,
                avg_power_w=1.0,
            ))

            mock_tsm = MagicMock()
            mock_tsm.get_computed_parameters.return_value = {
                "heat_loss_coefficient": 0.5  # current: 0.5, estimate ~0.0002
            }

            ok, msg = learner.apply_to_thermal_state(mock_tsm)

        assert ok is True
        call_args = mock_tsm.set_calibrated_baseline.call_args[0][0]
        applied_hlc = call_args["heat_loss_coefficient"]
        # Capped at 0.5 * (1 - 0.3) = 0.35
        assert applied_hlc == pytest.approx(0.35, rel=1e-6)

    def test_within_cap_applies_exact_estimate(self):
        """Small change within cap is applied exactly."""
        with _patch_config({
            "HLC_SESSION_MIN_DAYS": 1,
            "HLC_SESSION_MAX_UPDATE_FRACTION": 0.3,
        }):
            learner = HLCSessionLearner()
            # HLC = 1.5 / 10 = 0.15 kW/K; current = 0.14
            learner._day_records.append(DayRecord(
                date="2024-01-01",
                mean_thermal_power_kw=1.5,
                mean_delta_t=10.0,
                n_cycles=10,
                outdoor_temp_mean=5.0,
                indoor_temp_mean=20.0,
                avg_power_w=1500.0,
            ))

            mock_tsm = MagicMock()
            mock_tsm.get_computed_parameters.return_value = {
                "heat_loss_coefficient": 0.14  # within 30% of 0.15
            }

            ok, msg = learner.apply_to_thermal_state(mock_tsm)

        assert ok is True
        call_args = mock_tsm.set_calibrated_baseline.call_args[0][0]
        applied_hlc = call_args["heat_loss_coefficient"]
        assert applied_hlc == pytest.approx(0.15, rel=1e-5)

    def test_returns_false_when_insufficient_records(self):
        with _patch_config({"HLC_SESSION_MIN_DAYS": 5}):
            learner = HLCSessionLearner()
            # Only 2 records
            for i in range(2):
                learner._day_records.append(DayRecord(
                    date=f"2024-01-{i+1:02d}",
                    mean_thermal_power_kw=1.5,
                    mean_delta_t=10.0,
                    n_cycles=10,
                    outdoor_temp_mean=5.0,
                    indoor_temp_mean=20.0,
                    avg_power_w=1500.0,
                ))
            ok, msg = learner.apply_to_thermal_state(MagicMock())
        assert ok is False
        assert "rejected" in msg.lower()
