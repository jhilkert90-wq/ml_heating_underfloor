"""
Unit tests for HLCSessionLearner and SessionRecord in src/hlc_learner.py.

Covers:
- SessionRecord fields (avg_power_w == mean_thermal_power_kw * 1000)
- PV-triggered session FSM: open at PV < threshold, close at PV >= threshold
- Session does NOT open when PV stays above threshold
- Session state survives JSON round-trip (restart survival)
- Per-cycle filtering: DHW, defrost, DHW-boost, blocking, TV filtered individually
- Whole-session reject: fireplace in any cycle rejects entire session
- estimate_hlc returns None when fewer than HLC_SESSION_MIN_SESSIONS records
- estimate_hlc OLS correctness with known values
- apply_to_thermal_state caps large updates via HLC_SESSION_MAX_UPDATE_FRACTION
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from src.hlc_learner import SessionRecord, HLCSessionLearner, _build_cycle



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cycle_ctx(
    thermal_power_kw: float = 1.5,
    indoor_temp: float = 20.5,
    outdoor_temp: float = 5.0,
    target_temp: float = 20.5,
    indoor_temp_delta_60m: float = 0.0,
    pv_now_electrical: float = 30.0,   # default: below 50 W threshold
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


# ---------------------------------------------------------------------------
# Config patching helpers
# ---------------------------------------------------------------------------

_CONFIG_DEFAULTS = {
    "HLC_SESSION_MIN_CYCLES": 6,
    "HLC_SESSION_MAX_SESSIONS": 120,
    "HLC_SESSION_MIN_SESSIONS": 10,
    "HLC_SESSION_MAX_UPDATE_FRACTION": 0.3,
    "HLC_SESSION_FILE": "",
    "HLC_PV_MAX_W": 50.0,
    "HLC_MAX_INDOOR_DELTA": 0.3,
    "HLC_MAX_TREND": 0.2,
    "HLC_OUTDOOR_TEMP_MIN": -10.0,
    "HLC_OUTDOOR_TEMP_MAX": 15.0,
    "HLC_MIN_HEATING_DEMAND_K": 1.0,
    "HEATING_MIN_THERMAL_POWER_KW": 0.5,
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
# SessionRecord dataclass tests
# ---------------------------------------------------------------------------

class TestSessionRecord:
    def test_avg_power_w_equals_mean_kw_times_1000(self):
        record = SessionRecord(
            session_start="2024-01-01T22:00:00",
            session_end="2024-01-02T06:00:00",
            duration_minutes=480.0,
            mean_thermal_power_kw=2.5,
            mean_delta_t=12.0,
            n_cycles=10,
            outdoor_temp_mean=5.0,
            indoor_temp_mean=20.0,
            avg_power_w=2500.0,
        )
        assert record.avg_power_w == pytest.approx(record.mean_thermal_power_kw * 1000)

    def test_all_fields_present(self):
        record = SessionRecord(
            session_start="2024-01-01T22:00:00",
            session_end="2024-01-02T06:00:00",
            duration_minutes=480.0,
            mean_thermal_power_kw=1.8,
            mean_delta_t=10.0,
            n_cycles=8,
            outdoor_temp_mean=3.0,
            indoor_temp_mean=20.5,
            avg_power_w=1800.0,
        )
        for f in ("session_start", "session_end", "duration_minutes",
                  "n_cycles", "outdoor_temp_mean", "indoor_temp_mean", "avg_power_w"):
            assert hasattr(record, f)


# ---------------------------------------------------------------------------
# PV-triggered FSM: session open / close
# ---------------------------------------------------------------------------

class TestPVTriggeredFSM:
    def test_session_opens_when_pv_drops_below_threshold(self, tmp_path):
        """push_cycle with PV=30W opens a new session."""
        with _patch_config({"HLC_SESSION_FILE": str(tmp_path / "s.json")}):
            learner = HLCSessionLearner()
            result = learner.push_cycle(_cycle_ctx(pv_now_electrical=30.0))
        assert learner._session_active is True
        assert result["session_closed"] is False

    def test_session_does_not_open_when_pv_above_threshold(self, tmp_path):
        """push_cycle with PV=70W (above threshold) does NOT open a session."""
        with _patch_config({"HLC_SESSION_FILE": str(tmp_path / "s.json")}):
            learner = HLCSessionLearner()
            result = learner.push_cycle(_cycle_ctx(pv_now_electrical=70.0))
        assert learner._session_active is False
        assert result["session_closed"] is False

    def test_session_does_not_open_at_exact_threshold(self, tmp_path):
        """push_cycle with PV == HLC_PV_MAX_W (50 W) does NOT open session."""
        with _patch_config({"HLC_SESSION_FILE": str(tmp_path / "s.json")}):
            learner = HLCSessionLearner()
            learner.push_cycle(_cycle_ctx(pv_now_electrical=50.0))
        assert learner._session_active is False

    def test_session_opens_just_below_threshold(self, tmp_path):
        """push_cycle with PV = 49.9 W opens a session."""
        with _patch_config({"HLC_SESSION_FILE": str(tmp_path / "s.json")}):
            learner = HLCSessionLearner()
            learner.push_cycle(_cycle_ctx(pv_now_electrical=49.9))
        assert learner._session_active is True

    def test_session_closes_when_pv_reaches_threshold(self, tmp_path):
        """Session closes when PV rises to exactly HLC_PV_MAX_W (50 W)."""
        with _patch_config({
            "HLC_SESSION_FILE": str(tmp_path / "s.json"),
            "HLC_SESSION_MIN_CYCLES": 2,
        }):
            learner = HLCSessionLearner()
            learner.push_cycle(_cycle_ctx(pv_now_electrical=30.0))
            assert learner._session_active is True
            for _ in range(4):
                learner.push_cycle(_cycle_ctx(pv_now_electrical=30.0))
            result = learner.push_cycle(_cycle_ctx(pv_now_electrical=50.0))
        assert result["session_closed"] is True
        assert learner._session_active is False

    def test_session_closes_above_threshold(self, tmp_path):
        """Session closes when PV rises above HLC_PV_MAX_W."""
        with _patch_config({
            "HLC_SESSION_FILE": str(tmp_path / "s.json"),
            "HLC_SESSION_MIN_CYCLES": 2,
        }):
            learner = HLCSessionLearner()
            learner.push_cycle(_cycle_ctx(pv_now_electrical=30.0))
            for _ in range(4):
                learner.push_cycle(_cycle_ctx(pv_now_electrical=30.0))
            result = learner.push_cycle(_cycle_ctx(pv_now_electrical=70.0))
        assert result["session_closed"] is True
        assert learner._session_active is False

    def test_opening_cycle_is_collected(self, tmp_path):
        """The cycle that opens the session IS collected."""
        with _patch_config({"HLC_SESSION_FILE": str(tmp_path / "s.json")}):
            learner = HLCSessionLearner()
            learner.push_cycle(_cycle_ctx(pv_now_electrical=30.0))
        assert len(learner._session_cycles) == 1

    def test_cycles_collected_during_active_session(self, tmp_path):
        """Multiple cycles during active session are all collected."""
        with _patch_config({"HLC_SESSION_FILE": str(tmp_path / "s.json")}):
            learner = HLCSessionLearner()
            for _ in range(5):
                learner.push_cycle(_cycle_ctx(pv_now_electrical=30.0))
        assert len(learner._session_cycles) == 5

    def test_trigger_cycle_not_in_session(self, tmp_path):
        """After close, session_cycles is cleared; trigger cycle not included."""
        with _patch_config({
            "HLC_SESSION_FILE": str(tmp_path / "s.json"),
            "HLC_SESSION_MIN_CYCLES": 1,
        }):
            learner = HLCSessionLearner()
            learner.push_cycle(_cycle_ctx(pv_now_electrical=30.0))  # opens + collects
            n_before_close = len(learner._session_cycles)
            learner.push_cycle(_cycle_ctx(pv_now_electrical=70.0))  # closes
        # After close, session is reset
        assert len(learner._session_cycles) == 0
        assert n_before_close == 1  # only the opening cycle was collected


# ---------------------------------------------------------------------------
# Per-cycle filtering: DHW, defrost, DHW-boost, blocking, TV
# ---------------------------------------------------------------------------

class TestPerCycleFiltering:
    def _inject_and_close(
        self, tmp_path, n_clean: int, n_blocked: int, block_kwarg: Dict
    ) -> Dict:
        """Inject mixed cycles into an active session and close it."""
        with _patch_config({
            "HLC_SESSION_FILE": str(tmp_path / "s.json"),
            "HLC_SESSION_MIN_CYCLES": n_clean,
        }):
            learner = HLCSessionLearner()
            learner._session_active = True
            learner._session_start = datetime.now() - timedelta(hours=1)
            for _ in range(n_clean):
                c = _build_cycle(_cycle_ctx())
                if c:
                    learner._session_cycles.append(c)
            for _ in range(n_blocked):
                c = _build_cycle(_cycle_ctx(**block_kwarg))
                if c:
                    learner._session_cycles.append(c)
            return learner._close_session()

    def test_dhw_cycles_filtered_session_still_validates(self, tmp_path):
        result = self._inject_and_close(tmp_path, 8, 2, {"dhw_heating": 1.0})
        assert result["session_validated"] is True

    def test_defrost_cycles_filtered_session_still_validates(self, tmp_path):
        result = self._inject_and_close(tmp_path, 8, 2, {"defrosting": 1.0})
        assert result["session_validated"] is True

    def test_dhw_boost_cycles_filtered_session_still_validates(self, tmp_path):
        result = self._inject_and_close(tmp_path, 8, 2, {"dhw_boost_heater": 1.0})
        assert result["session_validated"] is True

    def test_blocking_cycles_filtered_session_still_validates(self, tmp_path):
        result = self._inject_and_close(tmp_path, 8, 2, {"is_blocking": True})
        assert result["session_validated"] is True

    def test_tv_cycles_filtered_session_still_validates(self, tmp_path):
        result = self._inject_and_close(tmp_path, 8, 2, {"tv_on": 1.0})
        assert result["session_validated"] is True

    def test_filtered_count_used_for_min_cycles_gate(self, tmp_path):
        """After filtering, only clean cycles count toward HLC_SESSION_MIN_CYCLES."""
        # 4 clean + 6 DHW = 10 total; but only 4 clean → below min_cycles=6
        with _patch_config({
            "HLC_SESSION_FILE": str(tmp_path / "s.json"),
            "HLC_SESSION_MIN_CYCLES": 6,
        }):
            learner = HLCSessionLearner()
            learner._session_active = True
            learner._session_start = datetime.now() - timedelta(hours=1)
            for _ in range(4):
                c = _build_cycle(_cycle_ctx())
                if c:
                    learner._session_cycles.append(c)
            for _ in range(6):
                c = _build_cycle(_cycle_ctx(dhw_heating=1.0))
                if c:
                    learner._session_cycles.append(c)
            result = learner._close_session()
        assert result["session_validated"] is False
        assert result["reject_reason"] is not None
        assert "clean cycles" in result["reject_reason"]

    def test_all_cycles_blocked_session_not_closed(self, tmp_path):
        """Session with no clean cycles (all blocked) yields session_closed=False."""
        with _patch_config({
            "HLC_SESSION_FILE": str(tmp_path / "s.json"),
            "HLC_SESSION_MIN_CYCLES": 1,
        }):
            learner = HLCSessionLearner()
            learner._session_active = True
            learner._session_start = datetime.now() - timedelta(hours=1)
            for _ in range(5):
                c = _build_cycle(_cycle_ctx(dhw_heating=1.0))
                if c:
                    learner._session_cycles.append(c)
            result = learner._close_session()
        assert result["session_closed"] is False


# ---------------------------------------------------------------------------
# Whole-session reject: fireplace
# ---------------------------------------------------------------------------

class TestFireplaceWholeSessionReject:
    def test_fireplace_in_one_cycle_rejects_entire_session(self, tmp_path):
        """Fireplace active in even ONE cycle → whole session rejected."""
        with _patch_config({
            "HLC_SESSION_FILE": str(tmp_path / "s.json"),
            "HLC_SESSION_MIN_CYCLES": 2,
        }):
            learner = HLCSessionLearner()
            learner._session_active = True
            learner._session_start = datetime.now() - timedelta(hours=1)
            for _ in range(9):
                c = _build_cycle(_cycle_ctx())
                if c:
                    learner._session_cycles.append(c)
            fp_cycle = _build_cycle(_cycle_ctx(fireplace_on=1.0))
            if fp_cycle:
                learner._session_cycles.append(fp_cycle)
            result = learner._close_session()
        assert result["session_validated"] is False
        assert "fireplace" in result["reject_reason"]

    def test_no_fireplace_passes_fireplace_gate(self, tmp_path):
        """Session without fireplace is not rejected by the fireplace gate."""
        with _patch_config({
            "HLC_SESSION_FILE": str(tmp_path / "s.json"),
            "HLC_SESSION_MIN_CYCLES": 6,
        }):
            learner = HLCSessionLearner()
            learner._session_active = True
            learner._session_start = datetime.now() - timedelta(hours=1)
            for _ in range(8):
                c = _build_cycle(_cycle_ctx(fireplace_on=0.0))
                if c:
                    learner._session_cycles.append(c)
            result = learner._close_session()
        assert "fireplace" not in (result["reject_reason"] or "")


# ---------------------------------------------------------------------------
# Empty session handling
# ---------------------------------------------------------------------------

class TestEmptySession:
    def test_empty_session_not_closed(self, tmp_path):
        """Session with zero cycles → session_closed=False."""
        with _patch_config({"HLC_SESSION_FILE": str(tmp_path / "s.json")}):
            learner = HLCSessionLearner()
            learner._session_active = True
            result = learner._close_session()
        assert result["session_closed"] is False


# ---------------------------------------------------------------------------
# Session state persistence (JSON round-trip)
# ---------------------------------------------------------------------------

class TestSessionStatePersistence:
    def test_session_records_round_trip(self, tmp_path):
        """Saved session records are reloaded correctly."""
        session_file = str(tmp_path / "sessions.json")
        with _patch_config({
            "HLC_SESSION_FILE": session_file,
            "HLC_SESSION_MAX_SESSIONS": 120,
        }):
            learner1 = HLCSessionLearner()
            record = SessionRecord(
                session_start="2024-02-01T22:00:00",
                session_end="2024-02-02T06:00:00",
                duration_minutes=480.0,
                mean_thermal_power_kw=1.8,
                mean_delta_t=12.0,
                n_cycles=10,
                outdoor_temp_mean=4.0,
                indoor_temp_mean=20.0,
                avg_power_w=1800.0,
            )
            learner1._session_records.append(record)
            learner1._save_session_records()

        with _patch_config({
            "HLC_SESSION_FILE": session_file,
            "HLC_SESSION_MAX_SESSIONS": 120,
        }):
            learner2 = HLCSessionLearner()
            count = learner2.load_session_records()

        assert count == 1
        loaded = learner2.get_session_records()
        assert loaded[0].session_start == "2024-02-01T22:00:00"
        assert loaded[0].avg_power_w == pytest.approx(1800.0)

    def test_session_active_state_persists(self, tmp_path):
        """Active session flag and start time survive a save/load round-trip."""
        session_file = str(tmp_path / "sessions.json")
        start_time = datetime(2024, 1, 15, 23, 0, 0)
        with _patch_config({
            "HLC_SESSION_FILE": session_file,
            "HLC_SESSION_MAX_SESSIONS": 120,
        }):
            learner1 = HLCSessionLearner()
            learner1.push_cycle(_cycle_ctx(timestamp=start_time, pv_now_electrical=30.0))

        with _patch_config({
            "HLC_SESSION_FILE": session_file,
            "HLC_SESSION_MAX_SESSIONS": 120,
        }):
            learner2 = HLCSessionLearner()
            learner2.load_session_records()

        assert learner2._session_active is True
        assert learner2._session_start == start_time
        assert len(learner2._session_cycles) == 1

    def test_inactive_session_state_persists(self, tmp_path):
        """Inactive session is correctly restored on load."""
        session_file = str(tmp_path / "sessions.json")
        with _patch_config({
            "HLC_SESSION_FILE": session_file,
            "HLC_SESSION_MAX_SESSIONS": 120,
        }):
            learner1 = HLCSessionLearner()
            learner1._session_active = False
            learner1._session_start = None
            learner1._save_session_records()

        with _patch_config({
            "HLC_SESSION_FILE": session_file,
            "HLC_SESSION_MAX_SESSIONS": 120,
        }):
            learner2 = HLCSessionLearner()
            learner2.load_session_records()

        assert learner2._session_active is False
        assert learner2._session_start is None

    def test_cold_start_returns_zero(self, tmp_path):
        """Cold start (no file) returns 0 and creates the file."""
        import os
        session_file = str(tmp_path / "nonexistent.json")
        with _patch_config({"HLC_SESSION_FILE": session_file}):
            learner = HLCSessionLearner()
            count = learner.load_session_records()
        assert count == 0
        assert os.path.isfile(session_file)

    def test_saved_json_structure(self, tmp_path):
        """Saved file contains session_records, session_active, session_start keys."""
        session_file = str(tmp_path / "sessions.json")
        with _patch_config({"HLC_SESSION_FILE": session_file}):
            learner = HLCSessionLearner()
            learner._session_records.append(SessionRecord(
                session_start="2024-03-01T22:00:00",
                session_end="2024-03-02T06:00:00",
                duration_minutes=480.0,
                mean_thermal_power_kw=2.0,
                mean_delta_t=10.0,
                n_cycles=8,
                outdoor_temp_mean=3.0,
                indoor_temp_mean=20.0,
                avg_power_w=2000.0,
            ))
            learner._save_session_records()

        with open(session_file) as f:
            data = json.load(f)
        assert "session_records" in data
        assert "session_active" in data
        assert "session_start" in data
        assert data["session_records"][0]["avg_power_w"] == pytest.approx(2000.0)

    def test_closed_session_persists_inactive_state(self, tmp_path):
        """After a real close, persisted state must no longer mark the session active."""
        session_file = str(tmp_path / "sessions.json")
        start = datetime(2024, 1, 15, 22, 0, 0)
        end = datetime(2024, 1, 16, 6, 0, 0)
        with _patch_config({
            "HLC_SESSION_FILE": session_file,
            "HLC_SESSION_MIN_CYCLES": 1,
            "HLC_SESSION_MAX_SESSIONS": 120,
        }):
            learner = HLCSessionLearner()
            learner.push_cycle(_cycle_ctx(timestamp=start, pv_now_electrical=30.0))
            learner.push_cycle(_cycle_ctx(timestamp=end, pv_now_electrical=70.0))

        with open(session_file, encoding="utf-8") as f:
            data = json.load(f)

        assert data["session_active"] is False
        assert data["session_start"] is None
        assert len(data["session_records"]) == 1

    def test_active_session_cycles_survive_restart(self, tmp_path):
        """An in-progress session must reload with its collected cycles intact."""
        session_file = str(tmp_path / "sessions.json")
        start = datetime(2024, 1, 15, 22, 0, 0)
        mid = datetime(2024, 1, 15, 22, 10, 0)
        end = datetime(2024, 1, 16, 6, 0, 0)

        with _patch_config({
            "HLC_SESSION_FILE": session_file,
            "HLC_SESSION_MIN_CYCLES": 2,
            "HLC_SESSION_MAX_SESSIONS": 120,
        }):
            learner1 = HLCSessionLearner()
            learner1.push_cycle(_cycle_ctx(timestamp=start, pv_now_electrical=30.0))
            learner1.push_cycle(_cycle_ctx(timestamp=mid, pv_now_electrical=30.0))

        with _patch_config({
            "HLC_SESSION_FILE": session_file,
            "HLC_SESSION_MIN_CYCLES": 2,
            "HLC_SESSION_MAX_SESSIONS": 120,
        }):
            learner2 = HLCSessionLearner()
            learner2.load_session_records()
            assert learner2._session_active is True
            assert learner2._session_start == start
            assert len(learner2._session_cycles) == 2
            result = learner2.push_cycle(_cycle_ctx(timestamp=end, pv_now_electrical=70.0))

        assert result["session_closed"] is True
        assert result["session_validated"] is True
        assert result["session_records"] == 1

    def test_close_uses_trigger_cycle_timestamp(self, tmp_path):
        """Stored session_end and duration come from the closing trigger cycle timestamp."""
        session_file = str(tmp_path / "sessions.json")
        start = datetime(2024, 1, 15, 22, 0, 0)
        mid = datetime(2024, 1, 15, 23, 0, 0)
        end = datetime(2024, 1, 16, 6, 0, 0)

        with _patch_config({
            "HLC_SESSION_FILE": session_file,
            "HLC_SESSION_MIN_CYCLES": 2,
            "HLC_SESSION_MAX_SESSIONS": 120,
        }):
            learner = HLCSessionLearner()
            learner.push_cycle(_cycle_ctx(timestamp=start, pv_now_electrical=30.0))
            learner.push_cycle(_cycle_ctx(timestamp=mid, pv_now_electrical=30.0))
            learner.push_cycle(_cycle_ctx(timestamp=end, pv_now_electrical=70.0))

        with open(session_file, encoding="utf-8") as f:
            data = json.load(f)

        stored = data["session_records"][0]
        assert stored["session_start"] == start.isoformat()
        assert stored["session_end"] == end.isoformat()
        assert stored["duration_minutes"] == pytest.approx(480.0)

    def test_legacy_day_records_are_migrated(self, tmp_path):
        """Existing day_records files are migrated so historical HLC data is not lost."""
        session_file = str(tmp_path / "sessions.json")
        legacy_payload = {
            "day_records": [
                {
                    "date": "2024-02-01",
                    "mean_thermal_power_kw": 1.8,
                    "mean_delta_t": 12.0,
                    "n_cycles": 10,
                    "outdoor_temp_mean": 4.0,
                    "indoor_temp_mean": 20.0,
                    "avg_power_w": 1800.0,
                }
            ]
        }
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(legacy_payload, f)

        with _patch_config({
            "HLC_SESSION_FILE": session_file,
            "HLC_SESSION_MAX_SESSIONS": 120,
        }):
            learner = HLCSessionLearner()
            count = learner.load_session_records()

        assert count == 1
        loaded = learner.get_session_records()
        assert loaded[0].session_start == "2024-02-01T00:00:00"
        assert loaded[0].session_end == "2024-02-01T23:59:59"
        assert loaded[0].duration_minutes == pytest.approx(1440.0)
        assert loaded[0].mean_thermal_power_kw == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# estimate_hlc
# ---------------------------------------------------------------------------

class TestEstimateHLC:
    def _make_record(self, i: int, q: float, dt: float) -> SessionRecord:
        return SessionRecord(
            session_start=f"2024-01-{i+1:02d}T22:00:00",
            session_end=f"2024-01-{i+2:02d}T06:00:00",
            duration_minutes=480.0,
            mean_thermal_power_kw=q,
            mean_delta_t=dt,
            n_cycles=10,
            outdoor_temp_mean=5.0,
            indoor_temp_mean=20.0,
            avg_power_w=q * 1000,
        )

    def test_below_min_sessions_returns_none(self):
        with _patch_config({"HLC_SESSION_MIN_SESSIONS": 10}):
            learner = HLCSessionLearner()
            for i in range(5):
                learner._session_records.append(self._make_record(i, 1.5, 10.0 + i))
            hlc, stats = learner.estimate_hlc()
        assert hlc is None
        assert "reject_reason" in stats
        assert "session" in stats["reject_reason"]

    def test_ols_correct(self):
        """Verify HLC = Σ(Q·ΔT)/Σ(ΔT²) with known values."""
        hlc_true = 0.12
        dts = [10.0, 12.0, 8.0, 15.0, 11.0, 9.0, 13.0, 14.0, 7.0, 10.0]
        qs = [hlc_true * dt for dt in dts]
        with _patch_config({"HLC_SESSION_MIN_SESSIONS": 10}):
            learner = HLCSessionLearner()
            for i, (q, dt) in enumerate(zip(qs, dts)):
                learner._session_records.append(self._make_record(i, q, dt))
            hlc, stats = learner.estimate_hlc()
        assert hlc is not None
        assert hlc == pytest.approx(hlc_true, rel=1e-6)
        assert stats["r2"] == pytest.approx(1.0, abs=1e-4)

    def test_degenerate_zero_dt_returns_none(self):
        with _patch_config({"HLC_SESSION_MIN_SESSIONS": 3}):
            learner = HLCSessionLearner()
            for i in range(3):
                learner._session_records.append(self._make_record(i, 1.5, 0.0))
            hlc, stats = learner.estimate_hlc()
        assert hlc is None

    def test_stats_includes_n_sessions(self):
        with _patch_config({"HLC_SESSION_MIN_SESSIONS": 10}):
            learner = HLCSessionLearner()
            _, stats = learner.estimate_hlc()
        assert "n_sessions" in stats


# ---------------------------------------------------------------------------
# apply_to_thermal_state
# ---------------------------------------------------------------------------

class TestApplyToThermalState:
    def _add_session_record(self, learner: HLCSessionLearner, i: int, q: float, dt: float):
        learner._session_records.append(SessionRecord(
            session_start=f"2024-01-{i+1:02d}T22:00:00",
            session_end=f"2024-01-{i+2:02d}T06:00:00",
            duration_minutes=480.0,
            mean_thermal_power_kw=q,
            mean_delta_t=dt,
            n_cycles=10,
            outdoor_temp_mean=5.0,
            indoor_temp_mean=20.0,
            avg_power_w=q * 1000,
        ))

    def test_caps_upward_update(self):
        """Large upward estimate capped at current × (1 + MAX_UPDATE_FRACTION)."""
        with _patch_config({
            "HLC_SESSION_MIN_SESSIONS": 1,
            "HLC_SESSION_MAX_UPDATE_FRACTION": 0.3,
        }):
            learner = HLCSessionLearner()
            self._add_session_record(learner, 0, 10.0, 5.0)  # HLC ≈ 2.0
            mock_tsm = MagicMock()
            mock_tsm.get_computed_parameters.return_value = {"heat_loss_coefficient": 0.1}
            ok, _ = learner.apply_to_thermal_state(mock_tsm)
        assert ok is True
        applied = mock_tsm.set_calibrated_baseline.call_args[0][0]["heat_loss_coefficient"]
        assert applied == pytest.approx(0.13, rel=1e-6)  # 0.1 * 1.3

    def test_caps_downward_update(self):
        """Large downward estimate capped at current × (1 - MAX_UPDATE_FRACTION)."""
        with _patch_config({
            "HLC_SESSION_MIN_SESSIONS": 1,
            "HLC_SESSION_MAX_UPDATE_FRACTION": 0.3,
        }):
            learner = HLCSessionLearner()
            self._add_session_record(learner, 0, 0.001, 5.0)  # HLC ≈ 0.0002
            mock_tsm = MagicMock()
            mock_tsm.get_computed_parameters.return_value = {"heat_loss_coefficient": 0.5}
            ok, _ = learner.apply_to_thermal_state(mock_tsm)
        assert ok is True
        applied = mock_tsm.set_calibrated_baseline.call_args[0][0]["heat_loss_coefficient"]
        assert applied == pytest.approx(0.35, rel=1e-6)  # 0.5 * 0.7

    def test_small_change_applied_exactly(self):
        """Change within cap is applied without modification."""
        with _patch_config({
            "HLC_SESSION_MIN_SESSIONS": 1,
            "HLC_SESSION_MAX_UPDATE_FRACTION": 0.3,
        }):
            learner = HLCSessionLearner()
            self._add_session_record(learner, 0, 1.5, 10.0)  # HLC = 0.15
            mock_tsm = MagicMock()
            mock_tsm.get_computed_parameters.return_value = {"heat_loss_coefficient": 0.14}
            ok, _ = learner.apply_to_thermal_state(mock_tsm)
        assert ok is True
        applied = mock_tsm.set_calibrated_baseline.call_args[0][0]["heat_loss_coefficient"]
        assert applied == pytest.approx(0.15, rel=1e-5)

    def test_returns_false_when_insufficient_sessions(self):
        with _patch_config({"HLC_SESSION_MIN_SESSIONS": 10}):
            learner = HLCSessionLearner()
            for i in range(3):
                self._add_session_record(learner, i, 1.5, 10.0)
            ok, msg = learner.apply_to_thermal_state(MagicMock())
        assert ok is False
        assert "rejected" in msg.lower()


# ---------------------------------------------------------------------------
# FSM integration
# ---------------------------------------------------------------------------

class TestFSMIntegration:
    def test_full_night_session_validates(self, tmp_path):
        """10 cycles at PV=30W, then PV=70W closes and validates the session."""
        with _patch_config({
            "HLC_SESSION_FILE": str(tmp_path / "s.json"),
            "HLC_SESSION_MIN_CYCLES": 6,
        }):
            learner = HLCSessionLearner()
            for _ in range(10):
                learner.push_cycle(_cycle_ctx(pv_now_electrical=30.0))
            result = learner.push_cycle(_cycle_ctx(pv_now_electrical=70.0))
        assert result["session_closed"] is True
        assert result["session_validated"] is True

    def test_new_session_opens_after_close(self, tmp_path):
        """After a session closes, a new one opens immediately on next low-PV push."""
        with _patch_config({
            "HLC_SESSION_FILE": str(tmp_path / "s.json"),
            "HLC_SESSION_MIN_CYCLES": 2,
        }):
            learner = HLCSessionLearner()
            for _ in range(5):
                learner.push_cycle(_cycle_ctx(pv_now_electrical=30.0))
            learner.push_cycle(_cycle_ctx(pv_now_electrical=70.0))  # close
            assert learner._session_active is False
            learner.push_cycle(_cycle_ctx(pv_now_electrical=30.0))  # new open
        assert learner._session_active is True

    def test_session_count_increments_after_validation(self, tmp_path):
        """session_records count in result dict increments on validated session."""
        with _patch_config({
            "HLC_SESSION_FILE": str(tmp_path / "s.json"),
            "HLC_SESSION_MIN_CYCLES": 3,
        }):
            learner = HLCSessionLearner()
            for _ in range(8):
                learner.push_cycle(_cycle_ctx(pv_now_electrical=30.0))
            result = learner.push_cycle(_cycle_ctx(pv_now_electrical=70.0))
        if result["session_validated"]:
            assert result["session_records"] == 1

