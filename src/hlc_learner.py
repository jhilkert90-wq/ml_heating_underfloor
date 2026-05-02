"""
HLC Learner — PV-Triggered Heat Loss Coefficient Estimation

This module provides a persistent, PV-triggered estimator for the building's
Heat Loss Coefficient (HLC) from validated live-cycle data, plus a
historical calibration function that bootstraps HLC from InfluxDB / HA
history.

Concept
-------
At thermal equilibrium and with only the heat pump running, the steady-state
energy balance simplifies to:

    Q_hp ≈ HLC × (T_indoor − T_outdoor)

where Q_hp is the heat pump thermal power [kW] and HLC is the building heat
loss coefficient [kW/K].

The :class:`HLCSessionLearner` accumulates per-cycle data for each PV-night
session. A session opens when PV power drops below the configured threshold
and closes when PV rises back to or above it. Validated sessions are stored
as :class:`SessionRecord` entries in a rolling JSON file. Forced-through-origin
OLS regression over the stored session records yields an HLC estimate that
survives process restarts.

The :func:`calibrate_hlc` function fetches historical sensor data from
InfluxDB (or HA), filters stable HP-only periods, and runs the same OLS
regression to bootstrap HLC on first deployment or on-demand from the
dashboard.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from . import config
except ImportError:
    import config  # type: ignore

try:
    from .unified_thermal_state import get_thermal_state_manager
except ImportError:
    from unified_thermal_state import get_thermal_state_manager  # type: ignore

try:
    from .physics_calibration import fetch_historical_data_for_calibration
except ImportError:
    from physics_calibration import fetch_historical_data_for_calibration  # type: ignore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HLCCycle:
    """A single 5-minute control cycle snapshot used for HLC learning."""
    timestamp: datetime
    thermal_power_kw: float       # HP thermal power [kW]
    indoor_temp: float            # Current indoor temperature [°C]
    outdoor_temp: float           # Current outdoor temperature [°C]
    target_temp: float            # Target indoor temperature [°C]
    indoor_temp_delta_60m: float  # 60-min indoor temp change [K]
    pv_now_electrical: float      # Raw (uncorrected) PV power [W]
    fireplace_on: float           # 0.0 / 1.0
    tv_on: float                  # 0.0 / 1.0
    dhw_heating: float            # 0.0 / 1.0
    defrosting: float             # 0.0 / 1.0
    dhw_boost_heater: float       # 0.0 / 1.0
    is_blocking: bool             # combined blocking flag

    @property
    def delta_t(self) -> float:
        """Indoor − Outdoor ΔT [K]."""
        return self.indoor_temp - self.outdoor_temp


@dataclass
class SessionRecord:
    """A validated PV-night session record used in the session OLS regression."""
    session_start: str           # ISO datetime "YYYY-MM-DDTHH:MM:SS"
    session_end: str             # ISO datetime "YYYY-MM-DDTHH:MM:SS"
    duration_minutes: float      # Session duration [min]
    mean_thermal_power_kw: float # Mean HP thermal power for active (filtered) cycles [kW]
    mean_delta_t: float          # Mean (T_indoor − T_outdoor) [K]
    n_cycles: int                # Number of active filtered HP cycles in session
    outdoor_temp_mean: float     # Mean outdoor temperature [°C]
    indoor_temp_mean: float      # Mean indoor temperature [°C]
    avg_power_w: float           # mean_thermal_power_kw × 1000 [W] (energy/time)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_cycle(context: Dict) -> Optional[HLCCycle]:
    """Build an :class:`HLCCycle` from a raw context dict.

    Returns ``None`` if any required numeric field is missing or
    the thermal power is ``None``.
    """
    required = (
        "thermal_power_kw", "indoor_temp", "outdoor_temp",
        "target_temp",
    )
    for key in required:
        if context.get(key) is None:
            return None

    try:
        return HLCCycle(
            timestamp=context.get("timestamp", datetime.now()),
            thermal_power_kw=float(context["thermal_power_kw"]),
            indoor_temp=float(context["indoor_temp"]),
            outdoor_temp=float(context["outdoor_temp"]),
            target_temp=float(context["target_temp"]),
            indoor_temp_delta_60m=float(
                context.get("indoor_temp_delta_60m", 0.0)
            ),
            pv_now_electrical=float(
                context.get("pv_now_electrical", 0.0)
            ),
            fireplace_on=float(context.get("fireplace_on", 0.0)),
            tv_on=float(context.get("tv_on", 0.0)),
            dhw_heating=float(context.get("dhw_heating", 0.0)),
            defrosting=float(context.get("defrosting", 0.0)),
            dhw_boost_heater=float(context.get("dhw_boost_heater", 0.0)),
            is_blocking=bool(context.get("is_blocking", False)),
        )
    except (TypeError, ValueError):
        return None


def _serialize_cycle(cycle: HLCCycle) -> Dict:
    """Convert a cycle to a JSON-serializable dict."""
    return {
        "timestamp": cycle.timestamp.isoformat(),
        "thermal_power_kw": cycle.thermal_power_kw,
        "indoor_temp": cycle.indoor_temp,
        "outdoor_temp": cycle.outdoor_temp,
        "target_temp": cycle.target_temp,
        "indoor_temp_delta_60m": cycle.indoor_temp_delta_60m,
        "pv_now_electrical": cycle.pv_now_electrical,
        "fireplace_on": cycle.fireplace_on,
        "tv_on": cycle.tv_on,
        "dhw_heating": cycle.dhw_heating,
        "defrosting": cycle.defrosting,
        "dhw_boost_heater": cycle.dhw_boost_heater,
        "is_blocking": cycle.is_blocking,
    }


def _deserialize_cycle(payload: Dict) -> Optional[HLCCycle]:
    """Restore a cycle from persisted JSON payload."""
    raw = dict(payload)
    timestamp = raw.get("timestamp")
    if isinstance(timestamp, str):
        try:
            raw["timestamp"] = datetime.fromisoformat(timestamp)
        except ValueError:
            return None
    return _build_cycle(raw)


def _migrate_day_record(payload: Dict) -> Optional[SessionRecord]:
    """Convert a legacy DayRecord-shaped payload into a SessionRecord."""
    date_raw = payload.get("date")
    if not date_raw:
        return None

    try:
        session_start = datetime.fromisoformat(f"{date_raw}T00:00:00")
        mean_q = float(payload["mean_thermal_power_kw"])
        mean_dt = float(payload["mean_delta_t"])
        n_cycles = int(float(payload["n_cycles"]))
        outdoor_mean = float(payload["outdoor_temp_mean"])
        indoor_mean = float(payload["indoor_temp_mean"])
        avg_power_w = float(payload.get("avg_power_w", mean_q * 1000.0))
    except (KeyError, TypeError, ValueError):
        return None

    session_end = session_start + timedelta(hours=23, minutes=59, seconds=59)
    return SessionRecord(
        session_start=session_start.isoformat(),
        session_end=session_end.isoformat(),
        duration_minutes=1440.0,
        mean_thermal_power_kw=mean_q,
        mean_delta_t=mean_dt,
        n_cycles=n_cycles,
        outdoor_temp_mean=outdoor_mean,
        indoor_temp_mean=indoor_mean,
        avg_power_w=avg_power_w,
    )


# ---------------------------------------------------------------------------
# HLCSessionLearner
# ---------------------------------------------------------------------------

class HLCSessionLearner:
    """
    PV-triggered persistent HLC session learner.

    A session opens when ``pv_now_electrical < config.HLC_PV_MAX_W`` (50 W)
    and closes when ``pv_now_electrical >= config.HLC_PV_MAX_W``.  Cycles
    collected during the session are filtered individually: cycles where DHW,
    defrost, DHW-boost, blocking, or TV are active are discarded; only
    fireplace triggers a whole-session reject.  OLS regression over the
    stored :class:`SessionRecord` s yields a rolling HLC estimate that
    survives process restarts.

    Parameters are read from :mod:`src.config` at call time.
    """

    def __init__(self) -> None:
        self._session_active: bool = False
        self._session_start: Optional[datetime] = None
        self._session_cycles: List[HLCCycle] = []
        self._session_records: deque[SessionRecord] = deque()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_cycle(self, context: Dict) -> Dict:
        """Ingest one control-cycle snapshot.

        Implements a single-threshold PV FSM:
        - Opens a session when ``pv_now < HLC_PV_MAX_W`` and no session is active.
        - Closes and evaluates the session when ``pv_now >= HLC_PV_MAX_W`` and a
          session is active.  The trigger cycle (first daytime cycle) is not
          collected.
        - Collects cycles into the active session.

        Returns
        -------
        dict with keys:
          - ``"session_closed"`` (bool)
          - ``"session_validated"`` (bool)
          - ``"reject_reason"`` (str | None)
          - ``"session_records"`` (int): total stored session records
        """
        cycle = _build_cycle(context)

        base: Dict = {
            "session_closed": False,
            "session_validated": False,
            "reject_reason": None,
            "session_records": len(self._session_records),
        }

        if cycle is None:
            base["reject_reason"] = "missing required cycle data"
            return base

        pv_now = cycle.pv_now_electrical
        pv_max = config.HLC_PV_MAX_W

        if not self._session_active and pv_now < pv_max:
            # Open a new session — this cycle is the first one collected
            self._session_active = True
            self._session_start = cycle.timestamp
            self._session_cycles = []
            logger.debug(
                "📡 HLC session: opened (PV=%.0f W < %.0f W threshold)",
                pv_now,
                pv_max,
            )

        elif self._session_active and pv_now >= pv_max:
            # Close and evaluate — trigger cycle belongs to daytime, not collected
            close_result = self._close_session(session_end=cycle.timestamp)
            self._session_active = False
            self._session_start = None
            self._session_cycles = []
            self._save_session_records()
            close_result["session_records"] = len(self._session_records)
            return close_result

        if self._session_active:
            self._session_cycles.append(cycle)
            self._save_session_records()

        base["session_records"] = len(self._session_records)
        return base

    def load_session_records(self) -> int:
        """Load persisted session records from :attr:`config.HLC_SESSION_FILE`.

        Restores ``_session_active`` and ``_session_start`` so an in-progress
        session survives a container restart.  On cold start (file does not
        exist), an empty stub file is created so the user has immediate
        confirmation that the learner is active.

        Returns
        -------
        int
            Number of records loaded (0 on cold start).
        """
        session_file = config.HLC_SESSION_FILE
        if not os.path.isfile(session_file):
            logger.info(
                "📡 HLC session: cold start — creating empty session file %s",
                session_file,
            )
            self._save_session_records()
            return 0
        try:
            with open(session_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            normalized = False
            records_raw = data.get("session_records")
            if records_raw is None and "day_records" in data:
                migrated_records = []
                for legacy_payload in data.get("day_records", []):
                    migrated = _migrate_day_record(legacy_payload)
                    if migrated is not None:
                        migrated_records.append(migrated)
                records = migrated_records
                normalized = True
                logger.info(
                    "📡 HLC session: migrated %d legacy day records from %s",
                    len(records),
                    session_file,
                )
            else:
                records = [SessionRecord(**r) for r in (records_raw or [])]

            self._session_records = deque(records)
            # Trim to current cap
            while len(self._session_records) > config.HLC_SESSION_MAX_SESSIONS:
                self._session_records.popleft()
                normalized = True

            restored_cycles = []
            for payload in data.get("session_cycles", []):
                cycle = _deserialize_cycle(payload)
                if cycle is not None:
                    restored_cycles.append(cycle)
                else:
                    normalized = True

            # Restore in-progress session state (survives container restart)
            self._session_active = bool(data.get("session_active", False))
            session_start_raw = data.get("session_start")
            self._session_start = None

            if self._session_active and session_start_raw:
                try:
                    self._session_start = datetime.fromisoformat(session_start_raw)
                except (ValueError, TypeError):
                    normalized = True
                    self._session_start = None

            if self._session_active and self._session_start is None and restored_cycles:
                self._session_start = restored_cycles[0].timestamp
                normalized = True

            if self._session_active and not restored_cycles:
                logger.warning(
                    "HLC session learner: ignoring persisted active flag without cycles in %s",
                    session_file,
                )
                self._session_active = False
                self._session_start = None
                normalized = True

            if self._session_active:
                self._session_cycles = restored_cycles
            else:
                if restored_cycles or session_start_raw:
                    normalized = True
                self._session_cycles = []
                self._session_start = None

            if normalized:
                self._save_session_records()

            logger.debug(
                "HLC session learner: loaded %d session records from %s "
                "(session_active=%s, session_cycles=%d)",
                len(self._session_records),
                session_file,
                self._session_active,
                len(self._session_cycles),
            )
            return len(self._session_records)
        except Exception as exc:
            logger.warning(
                "HLC session learner: failed to load %s — %s", session_file, exc
            )
            return 0

    def estimate_hlc(self) -> Tuple[Optional[float], Dict]:
        """Run forced-through-origin OLS over stored session records.

        Returns
        -------
        (hlc_estimate, stats_dict)
            ``hlc_estimate`` is ``None`` when fewer session records exist than
            ``config.HLC_SESSION_MIN_SESSIONS``.

        ``stats_dict`` keys: n_sessions, sum_qdt, sum_dt2, r2, mean_residual
        """
        records = list(self._session_records)
        n = len(records)
        stats: Dict = {"n_sessions": n}

        if n < config.HLC_SESSION_MIN_SESSIONS:
            stats["reject_reason"] = (
                f"only {n} session records, need {config.HLC_SESSION_MIN_SESSIONS}"
            )
            return None, stats

        qs = [r.mean_thermal_power_kw for r in records]
        dts = [r.mean_delta_t for r in records]

        sum_qdt = sum(q * dt for q, dt in zip(qs, dts))
        sum_dt2 = sum(dt * dt for dt in dts)

        if sum_dt2 < 1e-6:
            stats["reject_reason"] = "degenerate: ΔT variance too small"
            return None, stats

        hlc = sum_qdt / sum_dt2

        ss_res = sum((q - hlc * dt) ** 2 for q, dt in zip(qs, dts))
        ss_tot = sum((q - (sum(qs) / n)) ** 2 for q in qs)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

        mean_residual = sum(q - hlc * dt for q, dt in zip(qs, dts)) / n

        stats.update(
            {
                "sum_qdt": round(sum_qdt, 6),
                "sum_dt2": round(sum_dt2, 6),
                "r2": round(r2, 4),
                "mean_residual": round(mean_residual, 4),
                "hlc_kw_per_k": round(hlc, 5),
            }
        )
        return hlc, stats

    def apply_to_thermal_state(
        self, thermal_state_manager=None
    ) -> Tuple[bool, str]:
        """Estimate HLC from session records and apply it to the thermal state.

        Parameters
        ----------
        thermal_state_manager : ThermalStateManager, optional
            If omitted the singleton from :func:`get_thermal_state_manager`
            is used.

        Returns
        -------
        (success, message)
        """
        hlc_estimate, stats = self.estimate_hlc()
        if hlc_estimate is None:
            reason = stats.get("reject_reason", "unknown")
            return False, f"HLC session estimation rejected: {reason}"

        if thermal_state_manager is None:
            thermal_state_manager = get_thermal_state_manager()

        current_params = thermal_state_manager.get_computed_parameters()
        current_hlc = current_params.get("heat_loss_coefficient", hlc_estimate)

        if current_hlc > 0:
            relative_change = abs(hlc_estimate - current_hlc) / current_hlc
            if relative_change > config.HLC_SESSION_MAX_UPDATE_FRACTION:
                sign = 1.0 if hlc_estimate > current_hlc else -1.0
                capped = current_hlc * (
                    1.0 + sign * config.HLC_SESSION_MAX_UPDATE_FRACTION
                )
                logger.warning(
                    "HLC session estimate %.5f kW/K would change current value "
                    "%.5f kW/K by %.1f%% — capping at %.5f kW/K",
                    hlc_estimate,
                    current_hlc,
                    relative_change * 100,
                    capped,
                )
                hlc_estimate = capped

        n_sessions = stats["n_sessions"]
        thermal_state_manager.set_calibrated_baseline(
            {"heat_loss_coefficient": hlc_estimate},
            calibration_cycles=n_sessions,
        )
        msg = (
            f"HLC updated to {hlc_estimate:.5f} kW/K "
            f"(R²={stats.get('r2', 0):.3f}, n_sessions={n_sessions})"
        )
        logger.info("✅ HLC session: %s", msg)
        return True, msg

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def session_record_count(self) -> int:
        """Number of validated session records currently stored."""
        return len(self._session_records)

    def get_session_records(self) -> List[SessionRecord]:
        """Return a copy of the stored session records."""
        return list(self._session_records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close_session(self, session_end: Optional[datetime] = None) -> Dict:
        """Validate and store a :class:`SessionRecord` for the completed PV-night session.

        Called automatically when PV rises above threshold.

        Returns
        -------
        dict with ``"session_closed"``, ``"session_validated"``, ``"reject_reason"``
        keys.
        """
        all_cycles = list(self._session_cycles)

        base = {
            "session_closed": True,
            "session_validated": False,
            "reject_reason": None,
            "session_records": len(self._session_records),
        }

        if not all_cycles:
            base["session_closed"] = False
            return base

        if session_end is None:
            session_end = all_cycles[-1].timestamp

        session_start_dt = self._session_start or all_cycles[0].timestamp
        duration_minutes = max(
            0.0,
            (session_end - session_start_dt).total_seconds() / 60.0,
        )

        # --- Tier 1: Whole-session reject ---
        # Fireplace distorts the room heat balance for the entire session.
        if any(c.fireplace_on > 0.5 for c in all_cycles):
            base["reject_reason"] = "fireplace active during session"
            return base

        # --- Tier 2: Per-cycle filter ---
        # DHW, defrost, DHW-boost, blocking, and TV are removed individually;
        # the session itself is NOT rejected.
        active = [
            c for c in all_cycles
            if c.thermal_power_kw > 0
            and c.tv_on <= 0.5
            and c.dhw_heating <= 0.5
            and c.defrosting <= 0.5
            and c.dhw_boost_heater <= 0.5
            and not c.is_blocking
        ]

        if len(active) == 0:
            base["session_closed"] = False
            return base

        # --- Session-level gates on filtered cycles ---
        if len(active) < config.HLC_SESSION_MIN_CYCLES:
            base["reject_reason"] = (
                f"only {len(active)} clean cycles after filtering "
                f"(need {config.HLC_SESSION_MIN_CYCLES})"
            )
            return base

        mean_indoor = sum(c.indoor_temp for c in active) / len(active)
        mean_target = sum(c.target_temp for c in active) / len(active)
        max_delta = getattr(config, "HLC_MAX_INDOOR_DELTA", 0.3)
        if abs(mean_indoor - mean_target) > max_delta:
            base["reject_reason"] = (
                f"|indoor {mean_indoor:.2f} − target {mean_target:.2f}| "
                f"= {abs(mean_indoor - mean_target):.3f} K > {max_delta} K"
            )
            return base

        max_trend = getattr(config, "HLC_MAX_TREND", 0.2)
        last_delta = abs(active[-1].indoor_temp_delta_60m)
        if last_delta > max_trend:
            base["reject_reason"] = (
                f"indoor_temp_delta_60m {last_delta:.3f} K > {max_trend} K"
            )
            return base

        mean_outdoor = sum(c.outdoor_temp for c in active) / len(active)
        t_min = getattr(config, "HLC_OUTDOOR_TEMP_MIN", -10.0)
        t_max = getattr(config, "HLC_OUTDOOR_TEMP_MAX", 15.0)
        if not (t_min <= mean_outdoor <= t_max):
            base["reject_reason"] = (
                f"mean outdoor temp {mean_outdoor:.1f} °C outside range "
                f"[{t_min}, {t_max}] °C"
            )
            return base

        min_demand = getattr(config, "HLC_MIN_HEATING_DEMAND_K", 1.0)
        if mean_target - mean_outdoor < min_demand:
            base["reject_reason"] = (
                f"T_target−T_outdoor = {mean_target - mean_outdoor:.2f} K "
                f"< {min_demand} K — not enough heating demand"
            )
            return base

        mean_q = sum(c.thermal_power_kw for c in active) / len(active)
        mean_dt = sum(c.delta_t for c in active) / len(active)

        if mean_dt <= 0:
            base["reject_reason"] = f"mean ΔT {mean_dt:.2f} K is not positive"
            return base

        session_start_str = session_start_dt.isoformat()

        record = SessionRecord(
            session_start=session_start_str,
            session_end=session_end.isoformat(),
            duration_minutes=round(duration_minutes, 1),
            mean_thermal_power_kw=round(mean_q, 4),
            mean_delta_t=round(mean_dt, 4),
            n_cycles=len(active),
            outdoor_temp_mean=round(mean_outdoor, 2),
            indoor_temp_mean=round(mean_indoor, 2),
            avg_power_w=round(mean_q * 1000.0, 2),
        )
        self._session_records.append(record)
        while len(self._session_records) > config.HLC_SESSION_MAX_SESSIONS:
            self._session_records.popleft()

        base["session_validated"] = True
        base["session_records"] = len(self._session_records)
        logger.info(
            "📡 HLC session: validated and stored "
            "(n_active=%d, n_total=%d, duration=%.0f min, "
            "mean_q=%.3f kW, mean_dt=%.2f K)",
            record.n_cycles,
            len(all_cycles),
            record.duration_minutes,
            record.mean_thermal_power_kw,
            record.mean_delta_t,
        )
        return base

    def _save_session_records(self) -> None:
        """Atomically persist session records and state to :attr:`config.HLC_SESSION_FILE`."""
        session_file = config.HLC_SESSION_FILE
        try:
            dir_path = os.path.dirname(session_file) or "."
            os.makedirs(dir_path, exist_ok=True)
            payload = {
                "session_records": [asdict(r) for r in self._session_records],
                "session_active": self._session_active,
                "session_start": (
                    self._session_start.isoformat()
                    if self._session_start else None
                ),
                "session_cycles": [
                    _serialize_cycle(cycle) for cycle in self._session_cycles
                ],
            }
            tmp_path = None
            with tempfile.NamedTemporaryFile(
                "w", dir=dir_path, delete=False, suffix=".tmp", encoding="utf-8"
            ) as tmp_f:
                tmp_path = tmp_f.name
                json.dump(payload, tmp_f, indent=2)
            os.replace(tmp_path, session_file)
            tmp_path = None  # successfully renamed; nothing to clean up
            logger.debug(
                "💾 HLC session: saved %d session records to %s",
                len(self._session_records),
                session_file,
            )
        except Exception as exc:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            logger.error("❌ HLC session: failed to save %s — %s", session_file, exc)


# ---------------------------------------------------------------------------
# Historical HLC Calibration
# ---------------------------------------------------------------------------

def calibrate_hlc(influx_service=None) -> Dict:
    """Calibrate HLC from historical sensor data (InfluxDB / HA).

    Fetches historical data for the configured lookback period, filters
    stable HP-only periods (same quality gates as the session learner),
    calculates thermal_power_kw from inlet/outlet/flow_rate, and runs
    forced-through-origin OLS regression to estimate HLC.

    The result is saved to the unified thermal state as a calibrated
    baseline value.

    Data fetching delegates to
    :func:`physics_calibration.fetch_historical_data_for_calibration`,
    which respects ``TRAINING_DATA_SOURCE`` ("influx", "ha_history",
    "auto") and performs HA history fallback/supplement in auto mode —
    identical to the strategy used for model calibration.

    Parameters
    ----------
    influx_service : InfluxService, optional
        Accepted for backward compatibility but no longer used.
        Data sourcing is handled by
        :func:`fetch_historical_data_for_calibration`.

    Returns
    -------
    dict
        Diagnostic results with keys: ``success``, ``hlc_kw_per_k``,
        ``r2``, ``n_periods``, ``date_range``, ``message``.
    """
    lookback_hours = getattr(config, "HLC_CALIBRATION_LOOKBACK_HOURS", 720)
    min_periods = getattr(config, "HLC_CALIBRATION_MIN_PERIODS", 20)

    logger.info(
        "🔬 HLC calibration: fetching %d hours of historical data...",
        lookback_hours,
    )

    # --- Fetch historical data ---
    # Delegate to the shared helper used by model calibration, which respects
    # TRAINING_DATA_SOURCE and performs HA history fallback/supplement in auto
    # mode — the same data-source strategy as physics_calibration.
    try:
        df = fetch_historical_data_for_calibration(lookback_hours=lookback_hours)
    except Exception as exc:
        msg = f"Failed to fetch historical data: {exc}"
        logger.error("❌ HLC calibration: %s", msg)
        return {"success": False, "message": msg}

    if df is None or df.empty:
        msg = "No historical data available for HLC calibration"
        logger.warning("⚠️ HLC calibration: %s", msg)
        return {"success": False, "message": msg}

    # --- Build column map from config entity IDs ---
    # Use the same short-name convention as physics_calibration and
    # influx_service.get_training_data(): entity_id.split(".", 1)[-1].
    # This correctly resolves non-English entity IDs (e.g. "rt_mittelwert")
    # without relying on keyword guessing.
    required_cols = {
        "indoor_temp", "outdoor_temp", "outlet_temp", "inlet_temp",
        "flow_rate",
    }
    col_map: Dict[str, str] = {}

    def _add_col(key: str, entity_attr: str) -> None:
        col_name = getattr(config, entity_attr, "").split(".", 1)[-1]
        if col_name and col_name in df.columns:
            col_map[key] = col_name

    _add_col("indoor_temp", "INDOOR_TEMP_ENTITY_ID")
    _add_col("outdoor_temp", "OUTDOOR_TEMP_ENTITY_ID")
    _add_col("outlet_temp", "ACTUAL_OUTLET_TEMP_ENTITY_ID")
    _add_col("inlet_temp", "INLET_TEMP_ENTITY_ID")
    _add_col("flow_rate", "FLOW_RATE_ENTITY_ID")
    _add_col("pv_power", "PV_POWER_ENTITY_ID")
    _add_col("fireplace", "FIREPLACE_STATUS_ENTITY_ID")
    _add_col("tv", "TV_STATUS_ENTITY_ID")
    _add_col("dhw", "DHW_STATUS_ENTITY_ID")
    _add_col("defrost", "DEFROST_STATUS_ENTITY_ID")
    _add_col("target_temp", "TARGET_INDOOR_TEMP_ENTITY_ID")

    missing = required_cols - set(col_map.keys())
    if missing:
        msg = f"Missing required columns in historical data: {missing}"
        logger.error("❌ HLC calibration: %s", msg)
        return {"success": False, "message": msg}

    # --- Calculate thermal power and filter stable periods ---
    specific_heat = getattr(config, "SPECIFIC_HEAT_CAPACITY", 4.186)
    pv_max = getattr(config, "HLC_PV_MAX_W", 50.0)
    outdoor_min = getattr(config, "HLC_OUTDOOR_TEMP_MIN", -10.0)
    outdoor_max = getattr(config, "HLC_OUTDOOR_TEMP_MAX", 15.0)
    min_demand = getattr(config, "HLC_MIN_HEATING_DEMAND_K", 1.0)
    max_indoor_delta = getattr(config, "HLC_MAX_INDOOR_DELTA", 0.3)
    max_trend = getattr(config, "HLC_MAX_TREND", 0.2)

    periods_q = []  # thermal power per period [kW]
    periods_dt = []  # delta T per period [K]

    # Use 20-minute windows (4 × 5-min rows)
    window_size = 4
    n_rows = len(df)
    rejected = {"total": 0, "reasons": {}}

    for start_idx in range(0, n_rows - window_size + 1, window_size):
        window = df.iloc[start_idx:start_idx + window_size]

        # Extract values
        try:
            outlet_vals = window[col_map["outlet_temp"]].dropna()
            inlet_vals = window[col_map["inlet_temp"]].dropna()
            flow_vals = window[col_map["flow_rate"]].dropna()
            indoor_vals = window[col_map["indoor_temp"]].dropna()
            outdoor_vals = window[col_map["outdoor_temp"]].dropna()
        except (KeyError, TypeError):
            continue

        if (len(outlet_vals) < 2 or len(inlet_vals) < 2
                or len(flow_vals) < 2 or len(indoor_vals) < 2
                or len(outdoor_vals) < 2):
            continue

        mean_outlet = outlet_vals.mean()
        mean_inlet = inlet_vals.mean()
        mean_flow = flow_vals.mean()
        mean_indoor = indoor_vals.mean()
        mean_outdoor = outdoor_vals.mean()

        # Thermal power: Q = (flow_rate / 60) × c_p × (outlet − inlet)
        delta_t_hp = mean_outlet - mean_inlet
        thermal_power_kw = (mean_flow / 60.0) * specific_heat * delta_t_hp

        if thermal_power_kw <= 0:
            _reject(rejected, "no_thermal_power")
            continue

        # ΔT for HLC regression: indoor − outdoor
        delta_t = mean_indoor - mean_outdoor

        if delta_t <= 0:
            _reject(rejected, "negative_delta_t")
            continue

        # Outdoor range check
        if not (outdoor_min <= mean_outdoor <= outdoor_max):
            _reject(rejected, "outdoor_temp_range")
            continue

        # Target temp check (if available)
        if "target_temp" in col_map:
            target_vals = window[col_map["target_temp"]].dropna()
            if len(target_vals) >= 2:
                mean_target = target_vals.mean()
                if abs(mean_indoor - mean_target) > max_indoor_delta:
                    _reject(rejected, "indoor_far_from_target")
                    continue
                if mean_target - mean_outdoor < min_demand:
                    _reject(rejected, "low_heating_demand")
                    continue

        # Indoor stability: require < 0.3°C change within window
        indoor_range = indoor_vals.max() - indoor_vals.min()
        if indoor_range > max_indoor_delta * 2:
            _reject(rejected, "indoor_unstable")
            continue

        # Indoor trend check: first-to-last change within the window
        # must not exceed max_trend (same gate as session learner)
        indoor_trend = abs(indoor_vals.iloc[-1] - indoor_vals.iloc[0])
        if indoor_trend > max_trend:
            _reject(rejected, "indoor_trend_too_high")
            continue

        # PV check
        if "pv_power" in col_map:
            pv_vals = window[col_map["pv_power"]].dropna()
            if len(pv_vals) > 0 and pv_vals.mean() > pv_max:
                _reject(rejected, "pv_too_high")
                continue

        # Blocking checks
        for blocker_key in ("fireplace", "tv", "dhw", "defrost"):
            if blocker_key in col_map:
                blocker_vals = window[col_map[blocker_key]].dropna()
                if len(blocker_vals) > 0 and blocker_vals.max() > 0.5:
                    _reject(rejected, f"{blocker_key}_active")
                    break
        else:
            # All blockers passed — accept period
            periods_q.append(thermal_power_kw)
            periods_dt.append(delta_t)
            continue
        # Blocker was active — period already rejected by break above

    n_periods = len(periods_q)
    logger.info(
        "🔬 HLC calibration: %d valid periods from %d windows "
        "(rejected: %s)",
        n_periods, n_rows // window_size,
        {k: v for k, v in rejected.get("reasons", {}).items()},
    )

    if n_periods < min_periods:
        msg = (
            f"Only {n_periods} valid periods found, "
            f"need at least {min_periods}"
        )
        logger.warning("⚠️ HLC calibration: %s", msg)
        return {"success": False, "n_periods": n_periods, "message": msg}

    # --- OLS regression: HLC = Σ(Q × ΔT) / Σ(ΔT²) ---
    sum_qdt = sum(q * dt for q, dt in zip(periods_q, periods_dt))
    sum_dt2 = sum(dt * dt for dt in periods_dt)

    if sum_dt2 < 1e-6:
        msg = "Degenerate data: ΔT variance too small"
        logger.warning("⚠️ HLC calibration: %s", msg)
        return {"success": False, "message": msg}

    hlc = sum_qdt / sum_dt2

    # Sanity bounds — reject physically implausible values.
    # Typical residential HLC range is 0.03–1.0 kW/K; allow generous
    # bounds to cover unusual buildings but catch regression artefacts.
    HLC_MIN_PLAUSIBLE = 0.01   # kW/K
    HLC_MAX_PLAUSIBLE = 2.0    # kW/K
    if not (HLC_MIN_PLAUSIBLE <= hlc <= HLC_MAX_PLAUSIBLE):
        msg = (
            f"HLC estimate {hlc:.5f} kW/K outside plausible range "
            f"[{HLC_MIN_PLAUSIBLE}, {HLC_MAX_PLAUSIBLE}] — rejected"
        )
        logger.warning("⚠️ HLC calibration: %s", msg)
        return {"success": False, "hlc_kw_per_k": round(hlc, 5), "message": msg}

    # R² (coefficient of determination)
    mean_q = sum(periods_q) / n_periods
    ss_res = sum((q - hlc * dt) ** 2 for q, dt in zip(periods_q, periods_dt))
    ss_tot = sum((q - mean_q) ** 2 for q in periods_q)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

    # Date range
    date_range = ""
    if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
        try:
            date_range = f"{df.index.min()} — {df.index.max()}"
        except Exception:
            date_range = "unknown"

    logger.info(
        "✅ HLC calibration result: HLC = %.5f kW/K "
        "(R² = %.3f, n = %d, range: %s)",
        hlc, r2, n_periods, date_range,
    )

    # --- Save to unified thermal state ---
    try:
        tsm = get_thermal_state_manager()
        tsm.set_calibrated_baseline(
            {"heat_loss_coefficient": hlc},
            calibration_cycles=n_periods,
        )
        logger.info("✅ HLC calibration: saved to unified thermal state")
    except Exception as exc:
        logger.error(
            "❌ HLC calibration: failed to save to thermal state — %s", exc
        )
        return {
            "success": False,
            "hlc_kw_per_k": round(hlc, 5),
            "r2": round(r2, 4),
            "n_periods": n_periods,
            "message": f"Calibration succeeded but save failed: {exc}",
        }

    return {
        "success": True,
        "hlc_kw_per_k": round(hlc, 5),
        "r2": round(r2, 4),
        "n_periods": n_periods,
        "date_range": date_range,
        "message": (
            f"HLC calibrated to {hlc:.5f} kW/K "
            f"(R²={r2:.3f}, n={n_periods})"
        ),
    }


def _reject(rejected: Dict, reason: str) -> None:
    """Helper to track rejection counts."""
    rejected["total"] = rejected.get("total", 0) + 1
    reasons = rejected.setdefault("reasons", {})
    reasons[reason] = reasons.get(reason, 0) + 1
