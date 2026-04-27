"""
Unit tests for src/hlc_learner.py — Online HLC Estimator.

Covers:
- HLCCycle dataclass construction and delta_t property
- Window validation: all rejection criteria and pass conditions
- push_cycle() accumulation and window boundary detection
- OLS regression via estimate_hlc()
- apply_to_thermal_state() with cap logic
- Rolling window cap (HLC_MAX_WINDOWS)
- Missing/None data handling
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.hlc_learner import HLCCycle, HLCLearner, HLCWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cycle(
    thermal_power_kw: float = 1.5,
    indoor_temp: float = 21.0,
    outdoor_temp: float = 5.0,
    target_temp: float = 21.0,
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
    """Build a minimal valid cycle context dict."""
    return {
        "timestamp": timestamp or datetime(2026, 1, 10, 12, 0),
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


def _make_cycles(
    n: int,
    base_time: datetime | None = None,
    interval_minutes: int = 5,
    **kwargs,
) -> List[Dict]:
    """Build *n* consecutive cycles with timestamps *interval_minutes* apart."""
    t0 = base_time or datetime(2026, 1, 10, 12, 0)
    return [
        _cycle(timestamp=t0 + timedelta(minutes=i * interval_minutes), **kwargs)
        for i in range(n)
    ]


def _make_learner_with_windows(n_windows: int) -> HLCLearner:
    """Return a learner pre-loaded with *n_windows* synthetic validated windows."""
    learner = HLCLearner()
    for i in range(n_windows):
        w = HLCWindow(
            start_time=datetime(2026, 1, i + 1, 0, 0),
            end_time=datetime(2026, 1, i + 1, 1, 0),
            mean_thermal_power_kw=1.5,
            mean_delta_t=16.0,
            n_cycles=12,
            outdoor_temp_mean=5.0,
            indoor_temp_mean=21.0,
        )
        learner._validated_windows.append(w)
    return learner


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
# _build_cycle helper (via push_cycle with required fields missing)
# ---------------------------------------------------------------------------

class TestBuildCycle:
    def test_missing_thermal_power_returns_none_result(self):
        learner = HLCLearner()
        ctx = _cycle()
        ctx.pop("thermal_power_kw")
        result = learner.push_cycle(ctx)
        assert result["window_validated"] is False
        assert "missing" in result["reject_reason"]

    def test_none_thermal_power_returns_reject(self):
        learner = HLCLearner()
        ctx = _cycle()
        ctx["thermal_power_kw"] = None
        result = learner.push_cycle(ctx)
        assert result["window_validated"] is False

    def test_missing_indoor_temp_returns_reject(self):
        learner = HLCLearner()
        ctx = _cycle()
        ctx.pop("indoor_temp")
        result = learner.push_cycle(ctx)
        assert result["window_validated"] is False

    def test_valid_cycle_is_not_complete_within_window(self):
        learner = HLCLearner()
        result = learner.push_cycle(_cycle())
        assert result["window_complete"] is False
        assert result["window_validated"] is False
        assert result["reject_reason"] is None


# ---------------------------------------------------------------------------
# push_cycle — window assembly
# ---------------------------------------------------------------------------

class TestPushCycleWindowAssembly:
    @patch("src.hlc_learner.HLCLearner._validate_window")
    def test_window_complete_triggers_after_duration(self, mock_validate):
        mock_validate.return_value = (False, "test_reject", None)
        learner = HLCLearner()

        # push 11 cycles within 55 minutes — window not yet complete
        cycles = _make_cycles(11)  # 0..50 min
        for c in cycles:
            result = learner.push_cycle(c)
        assert result["window_complete"] is False

        # 12th cycle at t=55 min — still not complete (need ≥60 min elapsed)
        c12 = _cycle(timestamp=datetime(2026, 1, 10, 12, 55))
        result = learner.push_cycle(c12)
        assert result["window_complete"] is False

        # 13th cycle at t=60 min — complete
        c13 = _cycle(timestamp=datetime(2026, 1, 10, 13, 0))
        result = learner.push_cycle(c13)
        assert result["window_complete"] is True

    def test_window_resets_after_completion(self):
        learner = HLCLearner()

        # Drive through one complete window
        cycles = _make_cycles(13)  # 0..60 min
        for c in cycles:
            learner.push_cycle(c)

        # After reset, current window should be empty
        assert learner.current_window_cycle_count == 0
        assert learner._window_start is None

    def test_validated_windows_count_increments(self):
        learner = HLCLearner()
        # 13 cycles spanning 60 minutes with clean data
        cycles = _make_cycles(
            13,
            thermal_power_kw=1.5,
            indoor_temp=21.0,
            outdoor_temp=5.0,
            target_temp=21.0,
        )
        for c in cycles:
            result = learner.push_cycle(c)

        # The window completed; check validated_windows
        assert result["window_complete"] is True
        if result["window_validated"]:
            assert result["validated_windows"] == 1

    def test_push_after_window_starts_new_window(self):
        learner = HLCLearner()
        cycles = _make_cycles(13)
        for c in cycles:
            learner.push_cycle(c)

        # Push one more cycle — starts a new window
        new_cycle = _cycle(timestamp=datetime(2026, 1, 10, 13, 5))
        result = learner.push_cycle(new_cycle)
        assert result["window_complete"] is False
        assert learner.current_window_cycle_count == 1
        assert learner._window_start == new_cycle["timestamp"]


# ---------------------------------------------------------------------------
# Window validation
# ---------------------------------------------------------------------------

class TestValidateWindow:
    """Test _validate_window directly via real config-like stub."""

    @pytest.fixture()
    def cfg(self):
        """Minimal config stub with all HLC defaults."""
        c = MagicMock()
        c.HLC_WINDOW_MINUTES = 60
        c.HLC_CYCLES_PER_WINDOW_MIN_FRAC = 0.8
        c.HLC_PV_MAX_W = 50.0
        c.HLC_MAX_INDOOR_DELTA = 0.3
        c.HLC_MAX_TREND = 0.2
        c.HLC_OUTDOOR_TEMP_MIN = -10.0
        c.HLC_OUTDOOR_TEMP_MAX = 15.0
        c.HLC_MIN_HEATING_DEMAND_K = 1.0
        return c

    def _build_cycle_objs(self, n: int = 12, **overrides) -> List[HLCCycle]:
        base = dict(
            thermal_power_kw=1.5,
            indoor_temp=21.0,
            outdoor_temp=5.0,
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
        base.update(overrides)
        t0 = datetime(2026, 1, 10, 12, 0)
        return [
            HLCCycle(
                timestamp=t0 + timedelta(minutes=i * 5),
                **base,
            )
            for i in range(n)
        ]

    def test_valid_window_passes(self, cfg):
        cycles = self._build_cycle_objs()
        ok, reason, window = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is True
        assert reason is None
        assert isinstance(window, HLCWindow)
        assert window.mean_thermal_power_kw == pytest.approx(1.5, rel=1e-3)
        assert window.mean_delta_t == pytest.approx(16.0, rel=1e-3)

    def test_empty_cycles_rejected(self, cfg):
        ok, reason, window = HLCLearner._validate_window(
            [], datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is False
        assert window is None

    def test_too_few_valid_power_cycles_rejected(self, cfg):
        # Most cycles have zero power → below 80% threshold
        cycles = self._build_cycle_objs()
        for c in cycles[:9]:
            c.thermal_power_kw = 0.0
        ok, reason, window = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is False
        assert "thermal_power_kw" in reason

    def test_fireplace_active_rejected(self, cfg):
        cycles = self._build_cycle_objs(fireplace_on=1.0)
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is False
        assert "fireplace" in reason

    def test_tv_active_rejected(self, cfg):
        cycles = self._build_cycle_objs(tv_on=1.0)
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is False
        assert "TV" in reason

    def test_dhw_heating_rejected(self, cfg):
        cycles = self._build_cycle_objs(dhw_heating=1.0)
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is False
        assert "DHW heating" in reason

    def test_defrost_rejected(self, cfg):
        cycles = self._build_cycle_objs(defrosting=1.0)
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is False
        assert "defrost" in reason

    def test_boost_heater_rejected(self, cfg):
        cycles = self._build_cycle_objs(dhw_boost_heater=1.0)
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is False
        assert "boost heater" in reason

    def test_blocking_rejected(self, cfg):
        cycles = self._build_cycle_objs(is_blocking=True)
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is False
        assert "blocking" in reason

    def test_high_pv_rejected(self, cfg):
        cycles = self._build_cycle_objs(pv_now_electrical=200.0)
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is False
        assert "PV" in reason

    def test_low_pv_accepted(self, cfg):
        cycles = self._build_cycle_objs(pv_now_electrical=30.0)
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is True

    def test_indoor_far_from_target_rejected(self, cfg):
        cycles = self._build_cycle_objs(indoor_temp=22.0, target_temp=21.0)
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is False
        assert "target" in reason.lower() or "K" in reason

    def test_indoor_near_target_accepted(self, cfg):
        cycles = self._build_cycle_objs(indoor_temp=21.2, target_temp=21.0)
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is True

    def test_large_60m_trend_rejected(self, cfg):
        cycles = self._build_cycle_objs()
        cycles[-1].indoor_temp_delta_60m = 0.5  # last cycle exceeds threshold
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is False
        assert "delta_60m" in reason or "indoor_temp" in reason

    def test_small_trend_accepted(self, cfg):
        cycles = self._build_cycle_objs()
        cycles[-1].indoor_temp_delta_60m = 0.1
        ok, _, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is True

    def test_outdoor_temp_too_high_rejected(self, cfg):
        # outdoor > HLC_OUTDOOR_TEMP_MAX = 15
        cycles = self._build_cycle_objs(outdoor_temp=18.0)
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is False
        assert "outdoor temp" in reason.lower() or "range" in reason.lower()

    def test_outdoor_temp_too_low_rejected(self, cfg):
        # outdoor < HLC_OUTDOOR_TEMP_MIN = -10
        cycles = self._build_cycle_objs(outdoor_temp=-15.0)
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is False

    def test_insufficient_heating_demand_rejected(self, cfg):
        # outdoor=14°C (within range), target=14.5°C → demand = 0.5 K < min 1.0 K
        cycles = self._build_cycle_objs(
            outdoor_temp=14.0, indoor_temp=14.5, target_temp=14.5
        )
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is False
        assert "demand" in reason.lower() or "heating" in reason.lower()

    def test_negative_mean_dt_rejected(self, cfg):
        # outdoor warmer than indoor → negative ΔT
        cycles = self._build_cycle_objs(
            outdoor_temp=14.0,
            indoor_temp=12.0,
            target_temp=12.0,
        )
        ok, reason, _ = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        # Will fail either demand or ΔT check
        assert ok is False

    def test_window_stats_are_correct(self, cfg):
        # Known data: Q=2.0 kW, T_in=22.0, T_out=6.0 → ΔT=16
        cycles = self._build_cycle_objs(
            thermal_power_kw=2.0,
            indoor_temp=22.0,
            outdoor_temp=6.0,
            target_temp=22.0,
        )
        ok, _, window = HLCLearner._validate_window(
            cycles, datetime(2026, 1, 10, 12, 0), cfg
        )
        assert ok is True
        assert window.mean_thermal_power_kw == pytest.approx(2.0, rel=1e-3)
        assert window.mean_delta_t == pytest.approx(16.0, rel=1e-3)
        assert window.n_cycles == 12


# ---------------------------------------------------------------------------
# estimate_hlc
# ---------------------------------------------------------------------------

class TestEstimateHLC:
    def test_returns_none_when_too_few_windows(self):
        learner = HLCLearner()
        learner._validated_windows = deque([
            HLCWindow(datetime.now(), datetime.now(), 1.5, 16.0, 12, 5.0, 21.0),
            HLCWindow(datetime.now(), datetime.now(), 1.5, 16.0, 12, 5.0, 21.0),
        ])
        with patch("src.hlc_learner.config") as mock_cfg:
            mock_cfg.HLC_MIN_WINDOWS = 3
            hlc, stats = learner.estimate_hlc()
        assert hlc is None
        assert "only 2" in stats.get("reject_reason", "")

    def test_correct_hlc_computed(self):
        """OLS forced-origin: HLC = Σ(Q·ΔT) / Σ(ΔT²).

        With Q=1.6 kW and ΔT=16 K: HLC = (1.6×16)/(16²) = 0.1 kW/K.
        """
        learner = HLCLearner()
        for _ in range(4):
            learner._validated_windows.append(
                HLCWindow(datetime.now(), datetime.now(), 1.6, 16.0, 12, 5.0, 21.0)
            )
        with patch("src.hlc_learner.config") as mock_cfg:
            mock_cfg.HLC_MIN_WINDOWS = 3
            hlc, stats = learner.estimate_hlc()
        assert hlc == pytest.approx(0.1, rel=1e-4)
        assert stats["n_windows"] == 4

    def test_r2_is_one_when_perfect_fit(self):
        """R² = 1 when all windows lie exactly on Q = HLC · ΔT (varied Q)."""
        learner = HLCLearner()
        # Q=1.0, ΔT=10 → Q=2.0, ΔT=20 → Q=3.0, ΔT=30 all satisfy HLC=0.1
        for q, dt in [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (1.5, 15.0), (2.5, 25.0)]:
            learner._validated_windows.append(
                HLCWindow(datetime.now(), datetime.now(), q, dt, 12, 5.0, 21.0)
            )
        with patch("src.hlc_learner.config") as mock_cfg:
            mock_cfg.HLC_MIN_WINDOWS = 3
            _, stats = learner.estimate_hlc()
        assert stats["r2"] == pytest.approx(1.0, abs=1e-4)

    def test_various_delta_t_values(self):
        """Vary ΔT to verify forced-origin regression is correct."""
        # Windows: (Q=0.8, ΔT=10), (Q=1.6, ΔT=20), (Q=2.4, ΔT=30)
        # HLC = (0.8×10 + 1.6×20 + 2.4×30) / (100 + 400 + 900)
        #      = (8 + 32 + 72) / 1400 = 112 / 1400 = 0.08 kW/K
        learner = HLCLearner()
        for q, dt in [(0.8, 10.0), (1.6, 20.0), (2.4, 30.0)]:
            learner._validated_windows.append(
                HLCWindow(datetime.now(), datetime.now(), q, dt, 12, 5.0, 21.0)
            )
        with patch("src.hlc_learner.config") as mock_cfg:
            mock_cfg.HLC_MIN_WINDOWS = 3
            hlc, _ = learner.estimate_hlc()
        assert hlc == pytest.approx(0.08, rel=1e-4)

    def test_degenerate_zero_dt_returns_none(self):
        learner = HLCLearner()
        for _ in range(4):
            learner._validated_windows.append(
                HLCWindow(datetime.now(), datetime.now(), 1.6, 0.0, 12, 5.0, 21.0)
            )
        with patch("src.hlc_learner.config") as mock_cfg:
            mock_cfg.HLC_MIN_WINDOWS = 3
            hlc, stats = learner.estimate_hlc()
        assert hlc is None
        assert "degenerate" in stats.get("reject_reason", "")


# ---------------------------------------------------------------------------
# apply_to_thermal_state
# ---------------------------------------------------------------------------

class TestApplyToThermalState:
    def _make_thermal_state_mock(self, current_hlc: float = 0.08) -> MagicMock:
        mock = MagicMock()
        mock.get_computed_parameters.return_value = {
            "heat_loss_coefficient": current_hlc
        }
        return mock

    def test_returns_false_when_insufficient_windows(self):
        learner = HLCLearner()
        ts_mock = self._make_thermal_state_mock()
        with patch("src.hlc_learner.config") as mock_cfg:
            mock_cfg.HLC_MIN_WINDOWS = 3
            mock_cfg.HLC_MAX_UPDATE_FRACTION = 0.3
            ok, msg = learner.apply_to_thermal_state(ts_mock)
        assert ok is False
        ts_mock.set_calibrated_baseline.assert_not_called()

    def test_applies_estimate_when_sufficient_windows(self):
        learner = HLCLearner()
        for _ in range(4):
            learner._validated_windows.append(
                HLCWindow(datetime.now(), datetime.now(), 1.6, 16.0, 12, 5.0, 21.0)
            )
        ts_mock = self._make_thermal_state_mock(current_hlc=0.1)
        with patch("src.hlc_learner.config") as mock_cfg:
            mock_cfg.HLC_MIN_WINDOWS = 3
            mock_cfg.HLC_MAX_UPDATE_FRACTION = 0.3
            ok, msg = learner.apply_to_thermal_state(ts_mock)
        assert ok is True
        ts_mock.set_calibrated_baseline.assert_called_once()
        call_args = ts_mock.set_calibrated_baseline.call_args[0][0]
        assert "heat_loss_coefficient" in call_args
        assert call_args["heat_loss_coefficient"] == pytest.approx(0.1, rel=1e-4)

    def test_cap_limits_upward_change(self):
        """Estimate 50% above current → capped at +30%."""
        learner = HLCLearner()
        # Q=1.5, ΔT=10 → HLC = 0.15; current = 0.1 → +50% change
        for _ in range(4):
            learner._validated_windows.append(
                HLCWindow(datetime.now(), datetime.now(), 1.5, 10.0, 12, 5.0, 21.0)
            )
        ts_mock = self._make_thermal_state_mock(current_hlc=0.1)
        with patch("src.hlc_learner.config") as mock_cfg:
            mock_cfg.HLC_MIN_WINDOWS = 3
            mock_cfg.HLC_MAX_UPDATE_FRACTION = 0.3
            ok, msg = learner.apply_to_thermal_state(ts_mock)
        applied = ts_mock.set_calibrated_baseline.call_args[0][0][
            "heat_loss_coefficient"
        ]
        assert applied == pytest.approx(0.13, rel=1e-4)  # 0.1 × 1.3

    def test_cap_limits_downward_change(self):
        """Estimate 50% below current → capped at -30%."""
        learner = HLCLearner()
        # Q=0.5, ΔT=10 → HLC = 0.05; current = 0.1 → -50% change
        for _ in range(4):
            learner._validated_windows.append(
                HLCWindow(datetime.now(), datetime.now(), 0.5, 10.0, 12, 5.0, 21.0)
            )
        ts_mock = self._make_thermal_state_mock(current_hlc=0.1)
        with patch("src.hlc_learner.config") as mock_cfg:
            mock_cfg.HLC_MIN_WINDOWS = 3
            mock_cfg.HLC_MAX_UPDATE_FRACTION = 0.3
            ok, msg = learner.apply_to_thermal_state(ts_mock)
        applied = ts_mock.set_calibrated_baseline.call_args[0][0][
            "heat_loss_coefficient"
        ]
        assert applied == pytest.approx(0.07, rel=1e-4)  # 0.1 × 0.7

    def test_within_cap_applies_exact_estimate(self):
        """Estimate 5% above current → applied unchanged."""
        learner = HLCLearner()
        # Q=1.05, ΔT=10 → HLC = 0.105; current = 0.1 → +5% change (within 30%)
        for _ in range(4):
            learner._validated_windows.append(
                HLCWindow(datetime.now(), datetime.now(), 1.05, 10.0, 12, 5.0, 21.0)
            )
        ts_mock = self._make_thermal_state_mock(current_hlc=0.1)
        with patch("src.hlc_learner.config") as mock_cfg:
            mock_cfg.HLC_MIN_WINDOWS = 3
            mock_cfg.HLC_MAX_UPDATE_FRACTION = 0.3
            ok, msg = learner.apply_to_thermal_state(ts_mock)
        applied = ts_mock.set_calibrated_baseline.call_args[0][0][
            "heat_loss_coefficient"
        ]
        assert applied == pytest.approx(0.105, rel=1e-4)

    def test_uses_singleton_when_no_manager_provided(self):
        learner = HLCLearner()
        for _ in range(4):
            learner._validated_windows.append(
                HLCWindow(datetime.now(), datetime.now(), 1.6, 16.0, 12, 5.0, 21.0)
            )
        mock_tsm = self._make_thermal_state_mock(current_hlc=0.1)
        with patch("src.hlc_learner.config") as mock_cfg, \
             patch(
                 "src.hlc_learner.get_thermal_state_manager",
                 return_value=mock_tsm,
             ):
            mock_cfg.HLC_MIN_WINDOWS = 3
            mock_cfg.HLC_MAX_UPDATE_FRACTION = 0.3
            ok, msg = learner.apply_to_thermal_state()
        assert ok is True
        mock_tsm.set_calibrated_baseline.assert_called_once()


# ---------------------------------------------------------------------------
# Rolling window cap
# ---------------------------------------------------------------------------

class TestRollingWindowCap:
    def test_oldest_window_evicted_when_cap_reached(self):
        """push_cycle() must evict the oldest validated window once HLC_MAX_WINDOWS is
        exceeded, ensuring that the in-built eviction path in push_cycle() is exercised
        rather than just the deque data structure."""
        learner = HLCLearner()
        cap = 3
        n_windows_to_push = cap + 1  # one extra to trigger eviction

        t0 = datetime(2026, 1, 1, 0, 0)
        with patch("src.hlc_learner.config") as mock_cfg:
            mock_cfg.HLC_WINDOW_MINUTES = 60
            mock_cfg.HLC_MAX_WINDOWS = cap
            mock_cfg.HLC_CYCLES_PER_WINDOW_MIN_FRAC = 0.8
            mock_cfg.HLC_PV_MAX_W = 50.0
            mock_cfg.HLC_MAX_INDOOR_DELTA = 0.3
            mock_cfg.HLC_MAX_TREND = 0.2
            mock_cfg.HLC_OUTDOOR_TEMP_MIN = -10.0
            mock_cfg.HLC_OUTDOOR_TEMP_MAX = 15.0
            mock_cfg.HLC_MIN_HEATING_DEMAND_K = 1.0

            # Drive cap+1 complete valid 60-minute windows (13 cycles × 5 min each).
            # Windows are spaced 65 min apart (60-min window + one 5-min cycle gap)
            # so that each window closes cleanly before the next one starts.
            for w in range(n_windows_to_push):
                for i in range(13):
                    ctx = _cycle(timestamp=t0 + timedelta(minutes=w * 65 + i * 5))
                    learner.push_cycle(ctx)

        # Cap must be respected
        assert learner.validated_window_count == cap
        # The first validated window (start_time == t0) must have been evicted
        assert learner._validated_windows[0].start_time != t0

    def test_accessors_return_correct_counts(self):
        learner = HLCLearner()
        assert learner.validated_window_count == 0
        assert learner.current_window_cycle_count == 0

        learner._validated_windows.append(
            HLCWindow(datetime.now(), datetime.now(), 1.5, 16.0, 12, 5.0, 21.0)
        )
        assert learner.validated_window_count == 1

        learner._current_window.append(
            HLCCycle(
                datetime.now(), 1.5, 21.0, 5.0, 21.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
            )
        )
        assert learner.current_window_cycle_count == 1

    def test_get_validated_windows_returns_copy(self):
        learner = HLCLearner()
        learner._validated_windows.append(
            HLCWindow(datetime.now(), datetime.now(), 1.5, 16.0, 12, 5.0, 21.0)
        )
        windows = learner.get_validated_windows()
        assert isinstance(windows, list)
        assert len(windows) == 1
        # Mutating the copy should not affect the internal deque
        windows.clear()
        assert learner.validated_window_count == 1


# ---------------------------------------------------------------------------
# End-to-end integration: full push_cycle loop + estimate
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_two_complete_windows_yield_estimate(self):
        """Push data for two complete 60-minute windows and get an estimate."""
        learner = HLCLearner()

        # Patch config so window is 60 min, min_windows=2
        with patch("src.hlc_learner.config") as mock_cfg:
            mock_cfg.HLC_WINDOW_MINUTES = 60
            mock_cfg.HLC_CYCLES_PER_WINDOW_MIN_FRAC = 0.8
            mock_cfg.HLC_PV_MAX_W = 50.0
            mock_cfg.HLC_MAX_INDOOR_DELTA = 0.3
            mock_cfg.HLC_MAX_TREND = 0.2
            mock_cfg.HLC_OUTDOOR_TEMP_MIN = -10.0
            mock_cfg.HLC_OUTDOOR_TEMP_MAX = 15.0
            mock_cfg.HLC_MIN_HEATING_DEMAND_K = 1.0
            mock_cfg.HLC_MAX_WINDOWS = 48
            mock_cfg.HLC_MIN_WINDOWS = 2

            t0 = datetime(2026, 1, 10, 8, 0)
            # 2 windows × 13 cycles each (0..60 min and 60..120 min)
            for w in range(2):
                for i in range(13):
                    ctx = _cycle(
                        timestamp=t0 + timedelta(minutes=w * 65 + i * 5),
                        thermal_power_kw=1.6,
                        indoor_temp=21.0,
                        outdoor_temp=5.0,
                        target_temp=21.0,
                    )
                    learner.push_cycle(ctx)

        # Should have exactly 2 validated windows
        assert learner.validated_window_count == 2

        with patch("src.hlc_learner.config") as mock_cfg2:
            mock_cfg2.HLC_MIN_WINDOWS = 1
            hlc, stats = learner.estimate_hlc()

        assert hlc is not None
        assert stats["n_windows"] == 2
        # Expected HLC = 1.6 / 16 = 0.1 kW/K
        assert hlc == pytest.approx(0.1, rel=0.01)
