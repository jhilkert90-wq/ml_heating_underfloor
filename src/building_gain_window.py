"""Persisted rolling window behind the building balance-point estimates.

Every control cycle (any route) stores one raw measurement set plus the
modelled non-HP gains [kW] for *both* climate modes.  Two estimates use it:

* modelled: time-weighted mean of PV/fireplace/TV gains with each mode's own
  model weights.  Heat-pump heat never enters it.
* measured: energy balance ``G = HLC*(Ti-To) + dU/dt - P_floor`` with room and
  slab storage.  Thermal power is signed, so defrost and cooling (return >
  flow) draw energy from the slab.  DHW-type intervals send no heat to the
  floor (diverter valve), which the settings can change.
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

PV_HISTORY_MAX_SAMPLES = 36
MODES = ("heating", "cooling")
METHODS = ("modelled", "measured", "measured_with_fallback")
CAPACITY_MODES = ("model", "manual", "learned")
DHW_POLICIES = ("zero_floor_power", "exclude")
PUMP_ON_MIN_FLOW_L_MIN = 1.0
LEARN_MIN_INTERVALS = 30
LEARN_MIN_RATE_ENERGY = 1.0
LEARN_CAPACITY_BOUNDS = (0.5, 50.0)


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number(cfg: Any, name: str, default: float, low: float, high: float) -> float:
    value = _finite(getattr(cfg, name, default))
    return default if value is None else min(high, max(low, value))


def _choice(cfg: Any, name: str, default: str, allowed: Tuple[str, ...]) -> str:
    value = getattr(cfg, name, default)
    return value if value in allowed else default


@dataclass(frozen=True)
class BalanceSettings:
    """Balance-point settings (dashboard options, see ``config.py``)."""

    method: str = "measured_with_fallback"
    window_hours: float = 24.0
    min_provisional_hours: float = 6.0
    full_coverage_fraction: float = 0.9
    max_interval_minutes: float = 30.0
    storage_enabled: bool = True
    capacity_mode: str = "model"
    room_capacity_kwh_per_k: float = 5.0
    slab_capacity_kwh_per_k: float = 4.0
    dhw_intervals: str = "zero_floor_power"
    hvac_off_min_samples: int = 3
    hvac_off_min_window_hours: float = 2.0
    hvac_off_max_gap_minutes: float = 60.0
    hvac_off_max_indoor_drift_60m: float = 0.25
    hvac_off_max_hvac_power_kw: float = 0.1
    hvac_off_max_flow_rate: float = 0.1
    hvac_off_max_gain_kw: float = 5.0
    hvac_off_max_spread_k: float = 2.0

    @classmethod
    def from_config(cls, cfg: Any) -> "BalanceSettings":
        defaults = cls()
        prefix = "BUILDING_BALANCE_"
        return cls(
            method=_choice(cfg, prefix + "METHOD", defaults.method, METHODS),
            window_hours=_number(
                cfg, prefix + "WINDOW_HOURS", defaults.window_hours, 6, 72
            ),
            min_provisional_hours=_number(
                cfg,
                prefix + "MIN_PROVISIONAL_HOURS",
                defaults.min_provisional_hours,
                1,
                24,
            ),
            full_coverage_fraction=_number(
                cfg,
                prefix + "FULL_COVERAGE_FRACTION",
                defaults.full_coverage_fraction,
                0.5,
                1.0,
            ),
            max_interval_minutes=_number(
                cfg,
                prefix + "MAX_INTERVAL_MINUTES",
                defaults.max_interval_minutes,
                10,
                120,
            ),
            storage_enabled=bool(
                getattr(
                    cfg, prefix + "STORAGE_ENABLED", defaults.storage_enabled
                )
            ),
            capacity_mode=_choice(
                cfg,
                prefix + "CAPACITY_MODE",
                defaults.capacity_mode,
                CAPACITY_MODES,
            ),
            room_capacity_kwh_per_k=_number(
                cfg,
                prefix + "ROOM_CAPACITY_KWH_PER_K",
                defaults.room_capacity_kwh_per_k,
                0.5,
                100,
            ),
            slab_capacity_kwh_per_k=_number(
                cfg,
                prefix + "SLAB_CAPACITY_KWH_PER_K",
                defaults.slab_capacity_kwh_per_k,
                0.5,
                100,
            ),
            dhw_intervals=_choice(
                cfg,
                prefix + "DHW_INTERVALS",
                defaults.dhw_intervals,
                DHW_POLICIES,
            ),
            hvac_off_min_samples=int(
                _number(
                    cfg,
                    prefix + "HVAC_OFF_MIN_SAMPLES",
                    defaults.hvac_off_min_samples,
                    2,
                    50,
                )
            ),
            hvac_off_min_window_hours=_number(
                cfg,
                prefix + "HVAC_OFF_MIN_WINDOW_HOURS",
                defaults.hvac_off_min_window_hours,
                0.5,
                24,
            ),
            hvac_off_max_gap_minutes=_number(
                cfg,
                prefix + "HVAC_OFF_MAX_GAP_MINUTES",
                defaults.hvac_off_max_gap_minutes,
                10,
                240,
            ),
            hvac_off_max_indoor_drift_60m=_number(
                cfg,
                prefix + "HVAC_OFF_MAX_INDOOR_DRIFT_60M",
                defaults.hvac_off_max_indoor_drift_60m,
                0.05,
                2,
            ),
            hvac_off_max_hvac_power_kw=_number(
                cfg,
                prefix + "HVAC_OFF_MAX_HVAC_POWER_KW",
                defaults.hvac_off_max_hvac_power_kw,
                0,
                2,
            ),
            hvac_off_max_flow_rate=_number(
                cfg,
                prefix + "HVAC_OFF_MAX_FLOW_RATE",
                defaults.hvac_off_max_flow_rate,
                0,
                5,
            ),
            hvac_off_max_gain_kw=_number(
                cfg,
                prefix + "HVAC_OFF_MAX_GAIN_KW",
                defaults.hvac_off_max_gain_kw,
                0.5,
                20,
            ),
            hvac_off_max_spread_k=_number(
                cfg,
                prefix + "HVAC_OFF_MAX_SPREAD_K",
                defaults.hvac_off_max_spread_k,
                0.5,
                10,
            ),
        )

    @property
    def window_seconds(self) -> float:
        return self.window_hours * 3600.0

    @property
    def min_provisional_seconds(self) -> float:
        return self.min_provisional_hours * 3600.0

    @property
    def max_interval_seconds(self) -> float:
        return self.max_interval_minutes * 60.0


def external_gain_kw(
    model: Any,
    pv_history: List[float],
    outdoor_temp: float,
    fireplace_on: float,
    tv_on: float,
    outlet_temp: Optional[float] = None,
    indoor_temp: Optional[float] = None,
) -> float:
    """Modelled non-HP heat gain [kW] using the model's own source physics."""
    outdoor = float(outdoor_temp)
    equilibrium = model.predict_equilibrium_temperature(
        outlet_temp=float(outdoor if outlet_temp is None else outlet_temp),
        outdoor_temp=outdoor,
        current_indoor=float(outdoor if indoor_temp is None else indoor_temp),
        pv_power=pv_history,
        fireplace_on=float(fireplace_on),
        tv_on=float(tv_on),
        thermal_power=0.0,
        _suppress_logging=True,
    )
    return (float(equilibrium) - outdoor) * float(model.heat_loss_coefficient)


def _clean_sample(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    timestamp = _finite(raw.get("t"))
    if timestamp is None:
        return None
    mode = raw.get("mode")
    return {
        "t": timestamp,
        "outdoor": _finite(raw.get("outdoor")),
        "indoor": _finite(raw.get("indoor")),
        "pv": _finite(raw.get("pv")),
        "fireplace": _finite(raw.get("fireplace")),
        "tv": _finite(raw.get("tv")),
        "thermal_power_kw": _finite(raw.get("thermal_power_kw")),
        "return_temp": _finite(raw.get("return_temp")),
        "outlet_temp": _finite(raw.get("outlet_temp")),
        "flow": _finite(raw.get("flow")),
        "mode": mode if mode in MODES else None,
        "dhw_active": bool(raw.get("dhw_active", False)),
        "g_heating": _finite(raw.get("g_heating")),
        "g_cooling": _finite(raw.get("g_cooling")),
    }


class BuildingGainWindow:
    """Time-weighted rolling window; persisted when a path is supplied."""

    def __init__(
        self,
        path: Optional[str] = None,
        settings: Optional[BalanceSettings] = None,
    ) -> None:
        self._path = path
        self._loaded = path is None
        self._samples: List[Dict[str, Any]] = []
        self.settings = settings or BalanceSettings()

    # -- persistence -----------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(self._path, "r", encoding="utf-8") as handle:
                stored = json.load(handle).get("samples", [])
        except (OSError, ValueError, AttributeError):
            return
        samples = [_clean_sample(item) for item in stored if isinstance(item, dict)]
        self._samples = sorted(
            (item for item in samples if item is not None),
            key=lambda item: item["t"],
        )

    def _save(self) -> None:
        if not self._path:
            return
        directory = os.path.dirname(self._path) or "."
        if not os.path.isdir(directory):
            return
        try:
            handle, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
            with os.fdopen(handle, "w", encoding="utf-8") as tmp:
                json.dump({"version": 2, "samples": self._samples}, tmp)
            os.replace(tmp_path, self._path)
        except OSError:
            logging.debug("Could not persist building gain window.", exc_info=True)

    # -- recording -------------------------------------------------------

    def recent_pv(self) -> List[float]:
        """Latest contiguous raw PV values (oldest first) for the lag filter."""
        self._ensure_loaded()
        history: List[float] = []
        newer_t: Optional[float] = None
        for sample in reversed(self._samples):
            if (
                newer_t is not None
                and newer_t - sample["t"] > self.settings.max_interval_seconds
            ):
                break
            history.append(sample["pv"] if sample["pv"] is not None else 0.0)
            newer_t = sample["t"]
            if len(history) >= PV_HISTORY_MAX_SAMPLES - 1:
                break
        return list(reversed(history))

    def add_sample(self, raw: Dict[str, Any]) -> None:
        self._ensure_loaded()
        sample = _clean_sample(raw)
        if sample is None:
            return
        if self._samples and sample["t"] <= self._samples[-1]["t"]:
            return
        self._samples.append(sample)
        cutoff = (
            sample["t"]
            - self.settings.window_seconds
            - self.settings.max_interval_seconds
        )
        while self._samples and self._samples[0]["t"] < cutoff:
            self._samples.pop(0)
        self._save()

    # -- shared helpers --------------------------------------------------

    def _weights(self, now: float) -> Iterator[Tuple[Dict[str, Any], float]]:
        start = now - self.settings.window_seconds
        for index, sample in enumerate(self._samples):
            following = (
                self._samples[index + 1]["t"]
                if index + 1 < len(self._samples)
                else now
            )
            end = min(
                following, sample["t"] + self.settings.max_interval_seconds, now
            )
            begin = max(sample["t"], start)
            if end > begin:
                yield sample, end - begin

    def _status(self, covered_seconds: float) -> Optional[str]:
        settings = self.settings
        if covered_seconds <= 0 or covered_seconds < settings.min_provisional_seconds:
            return None
        fraction = covered_seconds / settings.window_seconds
        return "full" if fraction >= settings.full_coverage_fraction else "provisional"

    def _floor_view(self) -> List[Tuple[Dict[str, Any], Optional[float], Optional[float]]]:
        """Per sample: (sample, floor return temp, floor power).

        During DHW-type intervals the diverter valve sends no heat to the
        floor, so power is 0 and the last floor-loop return temp is held.
        """
        zero_power = self.settings.dhw_intervals == "zero_floor_power"
        last_return: Optional[float] = None
        view = []
        for sample in self._samples:
            if sample["dhw_active"]:
                view.append((sample, last_return, 0.0 if zero_power else None))
                continue
            if sample["return_temp"] is not None:
                last_return = sample["return_temp"]
            view.append((sample, sample["return_temp"], sample["thermal_power_kw"]))
        return view

    def _valid_pairs(self, mode: str, now: float) -> Iterator[Tuple[Any, Any, float]]:
        """Consecutive samples of ``mode`` close enough to form an interval."""
        start = now - self.settings.window_seconds
        view = self._floor_view()
        for first, second in zip(view, view[1:]):
            duration = second[0]["t"] - first[0]["t"]
            if not 0 < duration <= self.settings.max_interval_seconds:
                continue
            if second[0]["t"] < start:
                continue
            if first[0]["mode"] != mode or second[0]["mode"] != mode:
                continue
            yield first, second, duration

    def median_flow(self, now: float) -> Optional[float]:
        """Median pump-on water flow [l/min] inside the window."""
        self._ensure_loaded()
        start = now - self.settings.window_seconds
        flows = [
            sample["flow"]
            for sample in self._samples
            if sample["t"] >= start
            and sample["flow"] is not None
            and sample["flow"] >= PUMP_ON_MIN_FLOW_L_MIN
            and not sample["dhw_active"]
        ]
        return statistics.median(flows) if flows else None

    # -- estimation ------------------------------------------------------

    def gain_estimate(
        self, mode: str, now: float, heat_loss_coefficient: float
    ) -> Optional[Dict[str, Any]]:
        """Time-weighted mean modelled gain with the mode's own weights."""
        self._ensure_loaded()
        hlc = _finite(heat_loss_coefficient)
        if mode not in MODES or hlc is None or hlc <= 0:
            return None
        key = f"g_{mode}"
        covered = total = total_sq = 0.0
        count = 0
        for sample, weight in self._weights(now):
            gain = sample[key]
            if gain is None:
                continue
            covered += weight
            total += weight * gain
            total_sq += weight * gain * gain
            count += 1
        status = self._status(covered)
        if status is None:
            return None
        mean = total / covered
        spread = math.sqrt(max(0.0, total_sq / covered - mean * mean))
        return {
            "gain_kw": mean,
            "uncertainty_k": spread / hlc,
            "sample_count": count,
            "window_minutes": int(covered / 60),
            "coverage_fraction": min(1.0, covered / self.settings.window_seconds),
            "status": status,
        }

    def measured_estimate(
        self,
        mode: str,
        now: float,
        heat_loss_coefficient: float,
        room_capacity: Optional[float],
        slab_capacity: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """Gain from ``HLC*(Ti-To) + (C_room*dTi + C_slab*dT_return)/dt - P``.

        Only intervals attributed to ``mode`` count.  The storage terms
        telescope over contiguous intervals, so energy stored by daytime
        preheating and released at night is not mistaken for a gain.
        """
        self._ensure_loaded()
        hlc = _finite(heat_loss_coefficient)
        if mode not in MODES or hlc is None or hlc <= 0:
            return None
        storage = self.settings.storage_enabled
        if storage and (room_capacity is None or slab_capacity is None):
            return None
        covered = total = 0.0
        count = 0
        for first_view, second_view, duration in self._valid_pairs(mode, now):
            first, return_first, power = first_view
            second, return_second, _ = second_view
            if None in (first["indoor"], second["indoor"], first["outdoor"], power):
                continue
            gain = hlc * (first["indoor"] - first["outdoor"]) - power
            if storage:
                if return_first is None or return_second is None:
                    continue
                gain += (
                    room_capacity * (second["indoor"] - first["indoor"])
                    + slab_capacity * (return_second - return_first)
                ) / (duration / 3600.0)
            covered += duration
            total += duration * gain
            count += 1
        status = self._status(covered)
        if status is None:
            return None
        return {
            "gain_kw": total / covered,
            "sample_count": count,
            "window_minutes": int(covered / 60),
            "coverage_fraction": min(1.0, covered / self.settings.window_seconds),
            "status": status,
        }

    def learned_slab_capacity(
        self, mode: str, now: float, outlet_effectiveness: Optional[float]
    ) -> Optional[float]:
        """Experimental: slab storage rate ``P - OE*(T_eff - Ti)`` vs dT_return/dt."""
        self._ensure_loaded()
        oe = _finite(outlet_effectiveness)
        if oe is None or oe <= 0:
            return None
        numerator = denominator = 0.0
        count = 0
        for first_view, second_view, duration in self._valid_pairs(mode, now):
            first, return_first, power = first_view
            second, return_second, _ = second_view
            if first["dhw_active"] or second["dhw_active"]:
                continue
            needed = (
                return_first, return_second, power, first["outlet_temp"],
                first["indoor"], first["flow"],
            )
            if any(value is None for value in needed):
                continue
            if first["flow"] < PUMP_ON_MIN_FLOW_L_MIN:
                continue
            rate = (return_second - return_first) / (duration / 3600.0)
            effective_temp = (first["outlet_temp"] + return_first) / 2.0
            storage_rate = power - oe * (effective_temp - first["indoor"])
            numerator += storage_rate * rate
            denominator += rate * rate
            count += 1
        if count < LEARN_MIN_INTERVALS or denominator < LEARN_MIN_RATE_ENERGY:
            return None
        capacity = numerator / denominator
        low, high = LEARN_CAPACITY_BOUNDS
        return capacity if low <= capacity <= high else None
