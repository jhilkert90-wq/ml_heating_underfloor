"""Pure helpers for the opt-in hourly heat-pump load forecast."""

from __future__ import annotations

import math
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")
LOGGER = logging.getLogger(__name__)


class ForecastStateStore:
    """Load and atomically persist compact forecast learning state."""

    VERSION = 1

    def __init__(self, path: str):
        self.path = Path(path)

    @staticmethod
    def empty() -> dict[str, Any]:
        return {
            "schema_version": ForecastStateStore.VERSION,
            "counter_snapshots": {},
            "cop_bins": {},
            "daily_energy": {},
            "defrost_counter_started_at": None,
            "defrost_el_in_heating_counter": "unknown",
            "defrost_inclusion_evidence": {"included": 0, "excluded": 0},
            "defrost_heat_fraction_live": None,
            "defrost_el_fraction_live": None,
            "defrost_heat_fraction_power": None,
            "defrost_el_fraction_power": None,
            "defrost_heat_counter_kwh": 0.0,
            "defrost_heat_reference_kwh": 0.0,
            "defrost_el_counter_kwh": 0.0,
            "defrost_el_reference_kwh": 0.0,
            "last_backfill_at": None,
        }

    def load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("schema_version") != self.VERSION:
                raise ValueError("Unsupported forecast-state schema")
            state = self.empty()
            state.update(data)
            for key in ("counter_snapshots", "cop_bins", "daily_energy"):
                if not isinstance(state.get(key), dict):
                    raise ValueError(f"Invalid forecast state field: {key}")
            return state
        except FileNotFoundError:
            return self.empty()
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring invalid heat-forecast state %s: %s", self.path, exc)
            return self.empty()

    def save(self, state: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(state)
        payload["schema_version"] = self.VERSION
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            if self.path.exists():
                shutil.copy2(self.path, self.path.with_suffix(self.path.suffix + ".bak"))
            os.replace(temp_name, self.path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise


def _entity_value(states: Mapping[str, Any], entity_id: str) -> float | None:
    if not entity_id:
        return None
    entity = states.get(entity_id)
    if not isinstance(entity, Mapping):
        return None
    raw = entity.get("state")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _defrost_fraction(
    state: Mapping[str, Any],
    *,
    forecast_key: str,
    power_key: str,
    default: float,
) -> float:
    value = state.get(forecast_key)
    if value is None:
        value = state.get(power_key)
    return float(default if value is None else value)


def update_counter_snapshot(
    state: dict[str, Any],
    key: str,
    value_kwh: float | None,
    now: datetime,
    *,
    max_delta_kwh: float,
) -> tuple[float | None, float | None]:
    """Update a persisted meter sample and return its delta and elapsed hours."""
    if value_kwh is None or not math.isfinite(value_kwh) or value_kwh < 0:
        return None, None
    snapshots = state.setdefault("counter_snapshots", {})
    previous = snapshots.get(key)
    current_time = now.astimezone(timezone.utc)
    delta = elapsed = None
    if isinstance(previous, Mapping):
        try:
            previous_time = datetime.fromisoformat(previous["time_utc"])
            elapsed = (current_time - previous_time).total_seconds() / 3600.0
            delta = compute_counter_delta(
                previous.get("value_kwh"),
                value_kwh,
                elapsed_hours=elapsed,
                max_delta_kwh=max_delta_kwh,
            )
        except (KeyError, TypeError, ValueError):
            delta = elapsed = None
    snapshots[key] = {
        "value_kwh": float(value_kwh),
        "time_utc": current_time.isoformat(),
    }
    return delta, elapsed


def add_cop_observation(
    state: dict[str, Any],
    *,
    outdoor_c: float,
    heat_kwh: float,
    electric_kwh: float,
    elapsed_hours: float,
    timestamp: datetime,
    half_life_days: float,
) -> None:
    """Update decayed sufficient statistics in an integer-degree outdoor bin."""
    if (
        not all(
            math.isfinite(v)
            for v in (outdoor_c, heat_kwh, electric_kwh, elapsed_hours)
        )
        or heat_kwh <= 0
        or electric_kwh <= 0
        or elapsed_hours <= 0
    ):
        return
    bins = state.setdefault("cop_bins", {})
    key = str(math.floor(outdoor_c))
    bucket = bins.setdefault(
        key,
        {
            "heat_kwh": 0.0,
            "electric_kwh": 0.0,
            "temp_energy_kwh": 0.0,
            "hours": 0.0,
            "updated_at": timestamp.astimezone(timezone.utc).isoformat(),
        },
    )
    try:
        updated_at = datetime.fromisoformat(bucket["updated_at"])
        age_hours = max(
            0.0,
            (timestamp.astimezone(timezone.utc) - updated_at).total_seconds()
            / 3600.0,
        )
    except (KeyError, TypeError, ValueError):
        age_hours = 0.0
    decay = 0.5 ** (age_hours / max(1.0, half_life_days * 24.0))
    for field in ("heat_kwh", "electric_kwh", "temp_energy_kwh", "hours"):
        bucket[field] = float(bucket.get(field, 0.0)) * decay
    bucket["heat_kwh"] += heat_kwh
    bucket["electric_kwh"] += electric_kwh
    bucket["temp_energy_kwh"] += outdoor_c * electric_kwh
    bucket["hours"] += elapsed_hours
    bucket["updated_at"] = timestamp.astimezone(timezone.utc).isoformat()


def infer_defrost_counter_inclusion(
    *,
    heating_counter_kwh: float,
    defrost_counter_kwh: float,
    power_integral_kwh: float,
    tolerance_kwh: float = 0.05,
) -> bool | None:
    """Compare meter deltas with the power integral during a defrost interval."""
    values = (heating_counter_kwh, defrost_counter_kwh, power_integral_kwh)
    if any(not math.isfinite(value) or value < 0 for value in values):
        return None
    if defrost_counter_kwh < 0.02 or power_integral_kwh < 0.02:
        return None
    included_error = abs(power_integral_kwh - heating_counter_kwh)
    excluded_error = abs(
        power_integral_kwh - heating_counter_kwh - defrost_counter_kwh
    )
    threshold = max(tolerance_kwh, power_integral_kwh * 0.2)
    if abs(included_error - excluded_error) <= threshold:
        return None
    return included_error < excluded_error


def thermal_pv_gain_kw(
    pv_power_w: float,
    *,
    weight_kw_per_w: float,
    correction_factor: float = 1.0,
    cap_kw: float = 3.0,
) -> float:
    """Apply the thermal model's PV weight and saturation to electrical PV."""
    values = (pv_power_w, weight_kw_per_w, correction_factor, cap_kw)
    if any(not math.isfinite(float(value)) for value in values) or cap_kw <= 0:
        raise ValueError("PV inputs and cap must be finite, with a positive cap")
    raw_gain = max(0.0, pv_power_w) * max(0.0, weight_kw_per_w)
    raw_gain *= max(0.0, correction_factor)
    return cap_kw * math.tanh(raw_gain / cap_kw)


def estimate_defrost_power_history(
    *,
    electrical_power: Sequence[tuple[datetime, float]],
    defrost_state: Sequence[tuple[datetime, float]],
    thermal_power: Sequence[tuple[datetime, float]],
    heating_electricity_kwh: float | None,
    heating_heat_kwh: float | None,
) -> dict[str, Any]:
    """Estimate defrost event energy and fractions from minute-resolution sensors."""
    power = {
        stamp.astimezone(timezone.utc).replace(second=0, microsecond=0): float(value)
        for stamp, value in electrical_power
        if math.isfinite(float(value)) and float(value) >= 0
    }
    active = {
        stamp.astimezone(timezone.utc).replace(second=0, microsecond=0): float(value) > 0
        for stamp, value in defrost_state
        if math.isfinite(float(value))
    }
    thermal = {
        stamp.astimezone(timezone.utc).replace(second=0, microsecond=0): float(value)
        for stamp, value in thermal_power
        if math.isfinite(float(value))
    }
    minutes = sorted(set(power) & set(active))
    event_electricity: list[float] = []
    event_heat: list[float] = []
    current_electricity = current_heat = 0.0
    active_minutes = thermal_minutes = 0
    total_active_minutes = total_thermal_minutes = 0
    previous_minute: datetime | None = None
    in_event = False
    for minute in minutes:
        is_active = active[minute]
        contiguous = (
            previous_minute is not None
            and (minute - previous_minute).total_seconds() <= 120
        )
        if not is_active or (in_event and not contiguous):
            if in_event:
                event_electricity.append(current_electricity)
                event_heat.append(current_heat)
            current_electricity = current_heat = 0.0
            active_minutes = thermal_minutes = 0
            in_event = False
        if is_active:
            if not in_event:
                current_electricity = current_heat = 0.0
                in_event = True
            current_electricity += power[minute] / 60_000.0
            active_minutes += 1
            total_active_minutes += 1
            if minute in thermal:
                current_heat += max(0.0, -thermal[minute]) / 60_000.0
                thermal_minutes += 1
                total_thermal_minutes += 1
        previous_minute = minute
    if in_event:
        event_electricity.append(current_electricity)
        event_heat.append(current_heat)

    def fraction(numerator: float, denominator: float | None) -> float | None:
        if denominator is None or not math.isfinite(denominator) or denominator <= 0:
            return None
        return min(0.5, max(0.0, numerator / denominator))

    return {
        "event_count": len(event_electricity),
        "mean_electricity_kwh_per_event": (
            sum(event_electricity) / len(event_electricity)
            if event_electricity
            else None
        ),
        "mean_heat_kwh_per_event": (
            sum(event_heat) / len(event_heat) if event_heat else None
        ),
        "mean_electrical_power_kw": (
            sum(event_electricity) * 60.0 / total_active_minutes
            if total_active_minutes > 0
            else None
        ),
        "mean_thermal_defrost_power_kw": (
            sum(event_heat) * 60.0 / total_thermal_minutes
            if total_thermal_minutes > 0
            else None
        ),
        "electricity_fraction": fraction(
            sum(event_electricity), heating_electricity_kwh
        ),
        "heat_fraction": fraction(sum(event_heat), heating_heat_kwh),
        "total_defrost_electricity_kwh": sum(event_electricity),
        "total_defrost_heat_kwh": sum(event_heat),
    }


def cop_curve_from_state(
    state: Mapping[str, Any],
    *,
    now: datetime | None = None,
    half_life_days: float = 45.0,
    start_intercept: float,
    start_slope: float,
    min_hours_per_bin: float,
    min_value: float,
    max_value: float,
) -> dict[str, Any]:
    """Return a learned energy-weighted COP line or an explicit start curve."""
    now = now or datetime.now(timezone.utc)
    samples = []
    for bucket in state.get("cop_bins", {}).values():
        if not isinstance(bucket, Mapping):
            continue
        try:
            updated_at = datetime.fromisoformat(bucket["updated_at"])
            age_hours = max(
                0.0,
                (
                    now.astimezone(timezone.utc)
                    - updated_at.astimezone(timezone.utc)
                ).total_seconds()
                / 3600.0,
            )
        except (KeyError, TypeError, ValueError):
            age_hours = 0.0
        decay = 0.5 ** (age_hours / max(1.0, half_life_days * 24.0))
        electric = float(bucket.get("electric_kwh", 0.0)) * decay
        heat = float(bucket.get("heat_kwh", 0.0)) * decay
        hours = float(bucket.get("hours", 0.0)) * decay
        if electric <= 0 or heat <= 0 or hours < min_hours_per_bin:
            continue
        samples.append(
            {
                "outdoor_c": float(bucket.get("temp_energy_kwh", 0.0)) / electric,
                "heat_kwh": heat,
                "electric_kwh": electric,
                "weight": decay,
            }
        )
    fit = fit_energy_weighted_cop(samples, cop_min=min_value, cop_max=max_value)
    if len(samples) < 2:
        return {
            "intercept": start_intercept,
            "slope": start_slope,
            "samples": fit["samples"],
            "calibrated": False,
            "temp_range_c": fit["temp_range_c"],
        }
    fit["calibrated"] = True
    return fit


def hourly_electrical_forecast(
    *,
    starts: Sequence[datetime],
    heat_th_kwh: Sequence[float],
    outdoor_c: Sequence[float],
    cop_curve: Mapping[str, float],
    dhw_el_kwh: Sequence[float],
    standby_el_kwh: Sequence[float],
    defrost_handling: str = "net",
    defrost_heat_fraction: float = 0.08,
    defrost_el_fraction: float = 0.06,
) -> dict[str, Any]:
    """Convert thermal heating to electrical energy and form EOS profile rows."""
    arrays = (heat_th_kwh, outdoor_c, dhw_el_kwh, standby_el_kwh)
    if any(len(values) != len(starts) for values in arrays):
        raise ValueError("All forecast arrays must match the timestamp horizon")
    heat_el, defrost_el, total_el = [], [], []
    for heat, outdoor, dhw, standby in zip(
        heat_th_kwh, outdoor_c, dhw_el_kwh, standby_el_kwh
    ):
        cop = float(cop_curve["intercept"]) + float(cop_curve["slope"]) * outdoor
        cop = min(10.0, max(1.5, cop))
        defrost_heat = heat * defrost_heat_fraction if outdoor <= 5.0 else 0.0
        if defrost_handling == "separate":
            hp_electric = (heat + defrost_heat) / cop
            defrost_electric = (
                hp_electric * defrost_el_fraction / max(1e-9, 1.0 - defrost_el_fraction)
                if defrost_heat > 0
                else 0.0
            )
        else:
            hp_electric = heat / cop
            defrost_electric = 0.0
        heat_el.append(hp_electric)
        defrost_el.append(defrost_electric)
        total_el.append(hp_electric + defrost_electric + dhw + standby)
    profile = [
        {"start": start.isoformat(), "value": round(float(value), 4)}
        for start, value in zip(starts, total_el)
    ]
    return {
        "forecast": profile,
        "forecast_heat_th_kwh": [round(float(v), 4) for v in heat_th_kwh],
        "forecast_heat_el_kwh": [round(v, 4) for v in heat_el],
        "forecast_defrost_el_kwh": [round(v, 4) for v in defrost_el],
        "forecast_dhw_el_kwh": [round(float(v), 4) for v in dhw_el_kwh],
        "forecast_standby_el_kwh": [
            round(float(v), 4) for v in standby_el_kwh
        ],
        "forecast_total_el_kwh": [round(v, 4) for v in total_el],
    }


class HeatForecastEngine:
    """Build and publish a read-only hourly heating/DHW/standby profile."""

    def __init__(self, state_path: str):
        self.store = ForecastStateStore(state_path)
        self.state = self.store.load()
        self._backfill_attempted = False

    @staticmethod
    def _number(ha_client: Any, states: Mapping[str, Any], entity_id: str) -> float | None:
        if not entity_id:
            return None
        try:
            value = ha_client.get_state(entity_id, states)
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _record_interval(
        self,
        *,
        now: datetime,
        states: Mapping[str, Any],
        outdoor_c: float,
    ) -> dict[str, float | None]:
        from . import config

        counter_entities = {
            "heating_el": config.HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID,
            "heating_th": config.HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID,
            "dhw_el": config.HEAT_FORECAST_DHW_COUNTER_ENTITY_ID,
            "standby_el": config.HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID,
            "defrost_el": config.HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID,
            "defrost_th": config.HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID,
        }
        max_deltas = {
            "heating_th": 10.0,
            "heating_el": 3.0,
            "dhw_el": 3.0,
            "standby_el": 1.0,
            "defrost_el": 3.0,
            "defrost_th": 10.0,
        }
        deltas: dict[str, float | None] = {}
        elapsed: dict[str, float | None] = {}
        readings: dict[str, float | None] = {}
        prior_snapshots = dict(self.state.get("counter_snapshots", {}))
        for name, entity_id in counter_entities.items():
            reading = _entity_value(states, entity_id)
            readings[name] = reading
            delta, duration = update_counter_snapshot(
                self.state,
                name,
                reading,
                now,
                max_delta_kwh=max_deltas[name],
            )
            deltas[name] = delta
            elapsed[name] = duration

        power_reading = _entity_value(states, config.POWER_CONSUMPTION_ENTITY_ID)
        power_delta = None
        power_snapshot = self.state.get("power_snapshot")
        if power_reading is not None:
            if isinstance(power_snapshot, Mapping):
                try:
                    prior_time = datetime.fromisoformat(power_snapshot["time_utc"])
                    power_hours = (
                        now.astimezone(timezone.utc) - prior_time
                    ).total_seconds() / 3600.0
                    prior_power = float(power_snapshot["power_w"])
                    if (
                        0 < power_hours <= 2.0
                        and math.isfinite(prior_power)
                        and prior_power >= 0
                    ):
                        power_delta = (
                            (prior_power + power_reading) / 2_000.0
                        ) * power_hours
                except (KeyError, TypeError, ValueError):
                    power_delta = None
            self.state["power_snapshot"] = {
                "power_w": power_reading,
                "time_utc": now.astimezone(timezone.utc).isoformat(),
            }

        if (
            config.COP_DEFROST_EL_IN_HEATING_COUNTER == "auto"
            and deltas.get("heating_el") is not None
            and deltas.get("defrost_el") is not None
            and power_delta is not None
        ):
            inferred = infer_defrost_counter_inclusion(
                heating_counter_kwh=float(deltas["heating_el"]),
                defrost_counter_kwh=float(deltas["defrost_el"]),
                power_integral_kwh=power_delta,
            )
            if inferred is not None:
                evidence = self.state.setdefault(
                    "defrost_inclusion_evidence",
                    {"included": 0, "excluded": 0},
                )
                key = "included" if inferred else "excluded"
                evidence[key] = int(evidence.get(key, 0)) + 1
                if max(evidence.values()) >= 3:
                    result = "true" if evidence["included"] > evidence["excluded"] else "false"
                    if evidence["included"] != evidence["excluded"]:
                        self.state["defrost_el_in_heating_counter"] = result
                        self.state["defrost_el_in_heating_counter_source"] = (
                            "power_integral_diagnostic"
                        )

        heater_interval = elapsed.get("heating_th")
        if (
            (config.COP_CALIBRATION_ENABLED or config.HEAT_FORECAST_ENABLED)
            and deltas.get("heating_th") is not None
            and deltas.get("heating_el") is not None
            and heater_interval is not None
            and abs(float(elapsed.get("heating_el") or 0) - heater_interval) < 0.2
        ):
            heat = float(deltas["heating_th"])
            electricity = float(deltas["heating_el"])
            defrost_heat = deltas.get("defrost_th")
            defrost_electricity = deltas.get("defrost_el")
            live_heat = defrost_heat is not None
            live_electricity = defrost_electricity is not None
            defrost_heat = float(defrost_heat or 0.0)
            defrost_electricity = float(defrost_electricity or 0.0)
            el_in_counter = (
                config.COP_DEFROST_EL_IN_HEATING_COUNTER == "true"
                or (
                    config.COP_DEFROST_EL_IN_HEATING_COUNTER == "auto"
                    and self.state.get("defrost_el_in_heating_counter") != "false"
                )
            )

            cold_run_hours = float(self.state.get("cold_run_hours", 0.0))
            if outdoor_c <= 5.0 and electricity >= 0.05:
                cold_run_hours += heater_interval
                if live_heat and heat > 0:
                    self.state["defrost_heat_counter_kwh"] = (
                        float(self.state.get("defrost_heat_counter_kwh", 0.0))
                        + defrost_heat
                    )
                    self.state["defrost_heat_reference_kwh"] = (
                        float(self.state.get("defrost_heat_reference_kwh", 0.0))
                        + heat
                    )
                    self.state["defrost_heat_fraction_live"] = min(
                        0.5,
                        self.state["defrost_heat_counter_kwh"]
                        / self.state["defrost_heat_reference_kwh"],
                    )
                if live_electricity and electricity > 0:
                    electrical_denominator = (
                        electricity
                        if el_in_counter
                        else electricity + defrost_electricity
                    )
                    self.state["defrost_el_counter_kwh"] = (
                        float(self.state.get("defrost_el_counter_kwh", 0.0))
                        + defrost_electricity
                    )
                    self.state["defrost_el_reference_kwh"] = (
                        float(self.state.get("defrost_el_reference_kwh", 0.0))
                        + electrical_denominator
                    )
                    if self.state["defrost_el_reference_kwh"] > 0:
                        self.state["defrost_el_fraction_live"] = min(
                            0.5,
                            self.state["defrost_el_counter_kwh"]
                            / self.state["defrost_el_reference_kwh"],
                        )
            self.state["cold_run_hours"] = cold_run_hours
            blend = min(
                1.0,
                cold_run_hours
                / max(1.0, config.COP_DEFROST_BLEND_COLD_RUN_HOURS),
            )
            heat_blend = (
                blend
                if float(self.state.get("defrost_heat_reference_kwh", 0.0)) > 0
                else 0.0
            )
            el_blend = (
                blend
                if float(self.state.get("defrost_el_reference_kwh", 0.0)) > 0
                else 0.0
            )
            heat_fraction_start = float(
                config.DEFROST_DEFAULT_HEAT_FRACTION
                if self.state.get("defrost_heat_fraction_power") is None
                else self.state["defrost_heat_fraction_power"]
            )
            el_fraction_start = float(
                config.DEFROST_DEFAULT_EL_FRACTION
                if self.state.get("defrost_el_fraction_power") is None
                else self.state["defrost_el_fraction_power"]
            )
            if outdoor_c <= 5.0:
                defrost_heat = (
                    defrost_heat * heat_blend
                    + heat * heat_fraction_start * (1.0 - heat_blend)
                    if live_heat
                    else heat * heat_fraction_start
                )

            needs_explicit_defrost_electricity = (
                config.COP_DEFROST_HANDLING == "separate" and el_in_counter
            ) or (config.COP_DEFROST_HANDLING == "net" and not el_in_counter)
            if outdoor_c <= 5.0 and needs_explicit_defrost_electricity:
                defrost_electricity = (
                    defrost_electricity * el_blend
                    + electricity * el_fraction_start * (1.0 - el_blend)
                    if live_electricity
                    else electricity * el_fraction_start
                )
            heat_net, electric_total = net_cop_energies(
                heat_kwh=heat,
                electricity_kwh=electricity,
                defrost_heat_kwh=defrost_heat,
                defrost_electricity_kwh=defrost_electricity,
                defrost_el_in_heating_counter=(
                    True
                    if config.COP_DEFROST_HANDLING == "separate"
                    else el_in_counter
                ),
            )
            if config.COP_DEFROST_HANDLING == "separate" and el_in_counter:
                electric_total = max(0.0, electricity - defrost_electricity)
            heat_fraction_live = self.state.get("defrost_heat_fraction_live")
            el_fraction_live = self.state.get("defrost_el_fraction_live")
            self.state["defrost_heat_fraction_forecast"] = (
                heat_fraction_start * (1.0 - heat_blend)
                + float(heat_fraction_live) * heat_blend
                if heat_fraction_live is not None
                else heat_fraction_start
            )
            self.state["defrost_el_fraction_forecast"] = (
                el_fraction_start * (1.0 - el_blend)
                + float(el_fraction_live) * el_blend
                if el_fraction_live is not None
                else el_fraction_start
            )
            self.state["defrost_heat_blend"] = heat_blend
            self.state["defrost_el_blend"] = el_blend
            if config.COP_CALIBRATION_ENABLED:
                add_cop_observation(
                    self.state,
                    outdoor_c=outdoor_c,
                    heat_kwh=heat_net,
                    electric_kwh=electric_total,
                    elapsed_hours=heater_interval,
                    timestamp=now,
                    half_life_days=config.COP_HALF_LIFE_DAYS,
                )
            self.state["defrost_blend"] = min(heat_blend, el_blend)
            if readings.get("defrost_el") is not None:
                self.state.setdefault("defrost_counter_started_at", now.isoformat())

        self._record_daily_energy(now, deltas, elapsed)
        hour_key = now.astimezone(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        ).isoformat()
        hour_anchor = self.state.get("hour_anchor")
        if not isinstance(hour_anchor, dict) or hour_anchor.get("hour_utc") != hour_key:
            hour_start = datetime.fromisoformat(hour_key)
            now_utc = now.astimezone(timezone.utc)
            energy_so_far: dict[str, float] = {}
            energy_coverage: dict[str, bool] = {}
            for name in (
                "heating_el",
                "heating_th",
                "dhw_el",
                "standby_el",
                "defrost_el",
                "defrost_th",
            ):
                delta = deltas.get(name)
                duration = elapsed.get(name)
                previous = prior_snapshots.get(name)
                covered_energy = 0.0
                valid = False
                if delta is not None and duration is not None and isinstance(previous, Mapping):
                    try:
                        previous_time = datetime.fromisoformat(
                            previous["time_utc"]
                        ).astimezone(timezone.utc)
                        interval_start = now_utc - timedelta(hours=float(duration))
                        interval_start = max(interval_start, previous_time)
                        covered_hours = max(
                            0.0,
                            (now_utc - max(interval_start, hour_start)).total_seconds()
                            / 3600.0,
                        )
                        covered_energy = float(delta) * min(
                            1.0, covered_hours / float(duration)
                        )
                        valid = True
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):
                        valid = False
                energy_so_far[name] = covered_energy
                energy_coverage[name] = valid
            self.state["hour_anchor"] = {
                "hour_utc": hour_key,
                "sample_time_utc": now_utc.isoformat(),
                "readings": readings,
                "energy_so_far": energy_so_far,
                "energy_coverage": energy_coverage,
                "coverage": "partial",
            }
        return deltas

    def _record_daily_energy(
        self,
        now: datetime,
        deltas: Mapping[str, float | None],
        elapsed: Mapping[str, float | None],
    ) -> None:
        local_now = now.astimezone(BERLIN)
        date_key = local_now.date().isoformat()
        hour = local_now.hour
        days = self.state.setdefault("daily_energy", {})
        day = days.setdefault(
            date_key,
            {
                "dhw": [0.0] * 24,
                "standby": [0.0] * 24,
                "dhw_coverage_h": 0.0,
                "standby_coverage_h": 0.0,
            },
        )
        for source, field in (("dhw_el", "dhw"), ("standby_el", "standby")):
            delta = deltas.get(source)
            if delta is not None:
                day[field][hour] += float(delta)
            duration = elapsed.get(source)
            if (
                delta is not None
                and duration is not None
                and 0 < float(duration) <= 2.0
            ):
                coverage_key = f"{field}_coverage_h"
                day[coverage_key] = min(
                    24.0, float(day.get(coverage_key, 0.0)) + float(duration)
                )
        cutoff = (local_now.date() - timedelta(days=14)).isoformat()
        self.state["daily_energy"] = {
            key: value for key, value in days.items() if key >= cutoff
        }

    def _backfill(
        self,
        *,
        now: datetime,
        influx_service: Any,
        outdoor_entity_id: str,
    ) -> None:
        from . import config

        if self._backfill_attempted or influx_service is None:
            return
        if self.state.get("last_backfill_at"):
            self._backfill_defrost_power_history(
                now=now,
                influx_service=influx_service,
            )
            self._backfill_attempted = True
            return
        ids = [
            config.HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID,
            config.HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID,
            config.HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID,
            config.HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID,
            config.HEAT_FORECAST_DHW_COUNTER_ENTITY_ID,
            config.HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID,
            outdoor_entity_id,
        ]
        ids = list(dict.fromkeys(entity for entity in ids if entity))
        try:
            history = influx_service.fetch_counter_history(
                ids, now - timedelta(days=90), now
            )
        except Exception:
            LOGGER.warning("Heat forecast counter history backfill failed", exc_info=True)
            return
        self._backfill_attempted = True
        self._backfill_daily_energy(history, now)
        self._backfill_defrost_power_history(
            now=now,
            influx_service=influx_service,
            counter_history=history,
        )
        heat_id = config.HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID
        el_id = config.HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID
        outdoor_id = outdoor_entity_id
        if not config.COP_CALIBRATION_ENABLED or not all(
            history.get(entity) for entity in (heat_id, el_id, outdoor_id)
        ):
            self.state["last_backfill_at"] = now.astimezone(timezone.utc).isoformat()
            return
        indexed = {
            entity: {timestamp.replace(minute=0, second=0, microsecond=0): value
                     for timestamp, value in entries}
            for entity, entries in history.items()
        }
        times = sorted(set(indexed[heat_id]) & set(indexed[el_id]) & set(indexed[outdoor_id]))
        for previous, current in zip(times, times[1:]):
            elapsed = (current - previous).total_seconds() / 3600.0
            heat = compute_counter_delta(
                indexed[heat_id].get(previous),
                indexed[heat_id].get(current),
                elapsed_hours=elapsed,
                max_delta_kwh=10.0,
            )
            electric = compute_counter_delta(
                indexed[el_id].get(previous),
                indexed[el_id].get(current),
                elapsed_hours=elapsed,
                max_delta_kwh=3.0,
            )
            if heat is None or electric is None or elapsed > 2.0:
                continue
            outdoor = indexed[outdoor_id][current]
            defrost_heat_value = indexed.get(
                config.HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID, {}
            )
            defrost_heat = compute_counter_delta(
                defrost_heat_value.get(previous),
                defrost_heat_value.get(current),
                elapsed_hours=elapsed,
                max_delta_kwh=10.0,
            )
            if defrost_heat is None and outdoor <= 5.0:
                defrost_heat = heat * config.DEFROST_DEFAULT_HEAT_FRACTION
            defrost_electricity_value = indexed.get(
                config.HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID, {}
            )
            defrost_electricity = compute_counter_delta(
                defrost_electricity_value.get(previous),
                defrost_electricity_value.get(current),
                elapsed_hours=elapsed,
                max_delta_kwh=3.0,
            ) or 0.0
            el_in_counter = (
                config.COP_DEFROST_EL_IN_HEATING_COUNTER != "false"
                if config.COP_DEFROST_EL_IN_HEATING_COUNTER != "auto"
                else self.state.get("defrost_el_in_heating_counter") != "false"
            )
            if config.COP_DEFROST_HANDLING == "separate" and el_in_counter:
                electric_for_cop = max(0.0, electric - defrost_electricity)
            else:
                electric_for_cop = electric
            heat_net, total_electric = net_cop_energies(
                heat_kwh=heat,
                electricity_kwh=electric_for_cop,
                defrost_heat_kwh=float(defrost_heat or 0.0),
                defrost_electricity_kwh=defrost_electricity,
                defrost_el_in_heating_counter=(
                    True
                    if config.COP_DEFROST_HANDLING == "separate"
                    else el_in_counter
                ),
            )
            add_cop_observation(
                self.state,
                outdoor_c=outdoor,
                heat_kwh=heat_net,
                electric_kwh=total_electric,
                elapsed_hours=elapsed,
                timestamp=current,
                half_life_days=config.COP_HALF_LIFE_DAYS,
            )
        self.state["last_backfill_at"] = now.astimezone(timezone.utc).isoformat()
        LOGGER.info(
            "Heat forecast history backfill processed %d hourly intervals",
            max(0, len(times) - 1),
        )

    def _backfill_defrost_power_history(
        self,
        *,
        now: datetime,
        influx_service: Any,
        counter_history: Mapping[str, Sequence[tuple[datetime, float]]] | None = None,
    ) -> None:
        from . import config

        if self.state.get("defrost_power_history_attempted_at"):
            return
        fetch_minute_history = getattr(influx_service, "fetch_minute_history", None)
        if not callable(fetch_minute_history):
            return

        start = now - timedelta(days=10)
        counter_ids = [
            entity
            for entity in (
                config.HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID,
                config.HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID,
            )
            if entity
        ]
        if counter_history is None:
            try:
                counter_history = influx_service.fetch_counter_history(
                    counter_ids, start, now
                )
            except Exception:
                LOGGER.warning(
                    "Could not fetch counters for defrost power-history initialization",
                    exc_info=True,
                )
                counter_history = {}

        power_entities = [
            entity
            for entity in (
                config.POWER_CONSUMPTION_ENTITY_ID,
                getattr(config, "THERMAL_POWER_ENTITY_ID", ""),
            )
            if entity
        ]
        try:
            power_history = fetch_minute_history(
                power_entities, start, now, agg_fn="mean"
            )
            defrost_history = fetch_minute_history(
                [config.DEFROST_STATUS_ENTITY_ID], start, now, agg_fn="max"
            )
        except Exception:
            LOGGER.warning(
                "Defrost power-history initialization failed", exc_info=True
            )
            return

        def recent_total(entity_id: str, max_delta_kwh: float) -> float:
            samples = sorted(
                counter_history.get(entity_id, []), key=lambda sample: sample[0]
            )
            total = 0.0
            for (previous_time, previous), (current_time, current) in zip(
                samples, samples[1:]
            ):
                elapsed = (current_time - previous_time).total_seconds() / 3600.0
                if elapsed <= 0 or elapsed > 2.0:
                    continue
                delta = compute_counter_delta(
                    previous,
                    current,
                    elapsed_hours=elapsed,
                    max_delta_kwh=max_delta_kwh,
                )
                if delta is None:
                    continue
                interval_start = current_time - timedelta(hours=elapsed)
                overlap = max(
                    0.0,
                    (
                        current_time
                        - max(interval_start, start)
                    ).total_seconds()
                    / 3600.0,
                )
                total += delta * min(1.0, overlap / elapsed)
            return total

        heat_entity = config.HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID
        electricity_entity = config.HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID
        estimate = estimate_defrost_power_history(
            electrical_power=power_history.get(
                config.POWER_CONSUMPTION_ENTITY_ID, []
            ),
            defrost_state=defrost_history.get(
                config.DEFROST_STATUS_ENTITY_ID, []
            ),
            thermal_power=power_history.get(
                getattr(config, "THERMAL_POWER_ENTITY_ID", ""), []
            ),
            heating_electricity_kwh=recent_total(electricity_entity, 3.0),
            heating_heat_kwh=recent_total(heat_entity, 10.0),
        )
        if (
            config.COP_DEFROST_EL_IN_HEATING_COUNTER == "false"
            and estimate["electricity_fraction"] is not None
        ):
            total_heat_electricity = recent_total(electricity_entity, 3.0)
            total_heat_electricity += estimate["total_defrost_electricity_kwh"]
            if total_heat_electricity > 0:
                estimate["electricity_fraction"] = min(
                    0.5,
                    estimate["total_defrost_electricity_kwh"]
                    / total_heat_electricity,
                )
        self.state["defrost_power_history_attempted_at"] = now.astimezone(
            timezone.utc
        ).isoformat()
        self.state["defrost_power_history"] = estimate
        if estimate["event_count"] > 0:
            if estimate["heat_fraction"] is not None:
                self.state["defrost_heat_fraction_power"] = estimate["heat_fraction"]
            if estimate["electricity_fraction"] is not None:
                self.state["defrost_el_fraction_power"] = estimate[
                    "electricity_fraction"
                ]
            self.state["defrost_power_history_at"] = now.astimezone(
                timezone.utc
            ).isoformat()
            LOGGER.info(
                "Initialized defrost estimates from %d minute-history events "
                "(heat %.4f kWh/event, electricity %.4f kWh/event)",
                estimate["event_count"],
                estimate["mean_heat_kwh_per_event"] or 0.0,
                estimate["mean_electricity_kwh_per_event"] or 0.0,
            )

    def _backfill_daily_energy(
        self,
        history: Mapping[str, Sequence[tuple[datetime, float]]],
        now: datetime,
    ) -> None:
        from . import config

        days: dict[str, dict[str, Any]] = {}
        counter_map = (
            (config.HEAT_FORECAST_DHW_COUNTER_ENTITY_ID, "dhw"),
            (config.HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID, "standby"),
        )
        for entity_id, field in counter_map:
            samples = history.get(entity_id, []) if entity_id else []
            ordered = sorted(samples, key=lambda item: item[0])
            for (previous_time, previous), (current_time, current) in zip(
                ordered, ordered[1:]
            ):
                elapsed = (current_time - previous_time).total_seconds() / 3600.0
                delta = compute_counter_delta(
                    previous,
                    current,
                    elapsed_hours=elapsed,
                    max_delta_kwh=3.0 if field == "dhw" else 1.0,
                )
                if delta is None or elapsed > 2.0:
                    continue
                local = current_time.astimezone(BERLIN)
                day = days.setdefault(
                    local.date().isoformat(),
                    {
                        "dhw": [0.0] * 24,
                        "standby": [0.0] * 24,
                        "dhw_coverage_h": 0.0,
                        "standby_coverage_h": 0.0,
                    },
                )
                day[field][local.hour] += delta
                coverage_key = f"{field}_coverage_h"
                day[coverage_key] = min(
                    24.0, day[coverage_key] + elapsed
                )
        if days:
            cutoff = (now.astimezone(BERLIN).date() - timedelta(days=14)).isoformat()
            self.state["daily_energy"] = {
                key: value for key, value in days.items() if key >= cutoff
            }

    def _dhw_and_standby_profiles(
        self, starts: Sequence[datetime]
    ) -> tuple[list[float], list[float], dict[str, Any]]:
        from statistics import median
        from . import config

        days = self.state.get("daily_energy", {})
        complete_dhw = [
            (key, value)
            for key, value in sorted(days.items())
            if float(value.get("dhw_coverage_h", 0.0)) >= 21.6
        ][-7:]
        complete_standby = [
            (key, value)
            for key, value in sorted(days.items())
            if float(value.get("standby_coverage_h", 0.0)) >= 21.6
        ][-7:]
        dhw_totals = [sum(value["dhw"]) for _, value in complete_dhw]
        dhw_day = median(dhw_totals) if dhw_totals else 1.1
        dhw_profiles = [value["dhw"] for _, value in complete_dhw]
        if dhw_profiles:
            hourly_shape = [
                sum(profile[hour] for profile in dhw_profiles) / len(dhw_profiles)
                for hour in range(24)
            ]
            shape_total = sum(hourly_shape)
            hourly_shape = (
                [value / shape_total for value in hourly_shape]
                if shape_total > 0
                else [1.0 / 24.0] * 24
            )
        else:
            hourly_shape = [0.0] * 24
            for hour in range(8, 13):
                hourly_shape[hour] = 1.0 / 5.0
        dhw_remaining = max(0.0, dhw_day)
        dhw = []
        for index, start in enumerate(starts):
            value = dhw_day * hourly_shape[start.hour]
            if start.date() == starts[0].date() and index == 0:
                consumed = sum(
                    sum(day["dhw"])
                    for key, day in days.items()
                    if key == start.date().isoformat()
                )
                dhw_remaining = max(0.0, dhw_day - consumed)
                value = min(value, dhw_remaining)
                dhw_remaining -= value
            dhw.append(max(0.0, value))

        standby_values = [
            sum(value["standby"]) / 24.0
            for _, value in complete_standby
        ]
        standby_hour = (
            sum(standby_values) / len(standby_values)
            if standby_values
            else config.STANDBY_DEFAULT_KW
        )
        return dhw, [standby_hour] * len(starts), {
            "dhw_learned_days": len(complete_dhw),
            "dhw_daily_median_kwh": dhw_day,
            "standby_learned_days": len(complete_standby),
            "standby_mean_kw": standby_hour,
        }

    def update_and_publish(
        self,
        *,
        ha_client: Any,
        states: Mapping[str, Any],
        wrapper: Any,
        influx_service: Any = None,
        now: datetime | None = None,
    ) -> bool:
        """Refresh learning and publish one complete forecast, isolated from control."""
        from . import config
        from .building_capacity import resolve_capacities
        from .ha_client import get_sensor_attributes
        from .physics_features import parse_hourly_pv_forecast
        from .pv_trajectory import get_forecast_trajectory_state

        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        model = getattr(
            wrapper, "_heating_thermal_model", wrapper.thermal_model
        )
        outdoor_now = self._number(ha_client, states, config.OUTDOOR_TEMP_ENTITY_ID)
        indoor_now = self._number(ha_client, states, config.INDOOR_TEMP_ENTITY_ID)
        target_now = self._number(
            ha_client, states, config.TARGET_INDOOR_TEMP_ENTITY_ID
        )
        outdoor_now = 10.0 if outdoor_now is None else outdoor_now
        indoor_now = target_now if indoor_now is None else indoor_now
        target_now = 22.6 if target_now is None else target_now
        mode = config.get_climate_mode(
            ha_client.get_state(config.HEATING_STATUS_ENTITY_ID, states)
        )
        self._backfill(
            now=now,
            influx_service=influx_service,
            outdoor_entity_id=config.OUTDOOR_TEMP_ENTITY_ID,
        )
        deltas = self._record_interval(
            now=now, states=states, outdoor_c=outdoor_now
        )
        if not config.HEAT_FORECAST_ENABLED:
            self.store.save(self.state)
            return True

        horizon = max(1, min(48, int(config.HEAT_FORECAST_HORIZON_HOURS)))
        starts = forecast_hour_starts(now, horizon)
        temp_future = ha_client.get_hourly_forecast_strict(horizon - 1)
        pv_future = parse_hourly_pv_forecast(
            states,
            horizon - 1,
            now=starts[0].astimezone(timezone.utc),
        )
        outdoor = [outdoor_now]
        pv_now = self._number(ha_client, states, config.PV_POWER_ENTITY_ID)
        pv_electrical = [max(0.0, pv_now or 0.0)]
        temp_flags = [{"source": "current", "padded": False}]
        pv_flags = [{"source": "current", "padded": False}]
        last_temp = outdoor_now
        for index in range(1, horizon):
            raw_temp = temp_future[index - 1] if index - 1 < len(temp_future) else None
            if raw_temp is None:
                temp_flags.append({"source": "current_fallback", "padded": True})
                outdoor.append(last_temp)
            else:
                temp_flags.append({"source": "weather_forecast", "padded": False})
                last_temp = float(raw_temp)
                outdoor.append(last_temp)
            if index - 1 < len(pv_future):
                pv_electrical.append(max(0.0, float(pv_future[index - 1])))
                pv_flags.append({"source": "pv_forecast", "padded": False})
            else:
                pv_electrical.append(0.0)
                pv_flags.append({"source": "missing_zero", "padded": True})

        correction = 1.0
        if getattr(config, "SOLAR_CORRECTION_ENABLED", False):
            correction_value = self._number(
                ha_client, states, config.SOLAR_CORRECTION_ENTITY_ID
            )
            correction_percent = (
                config.SOLAR_CORRECTION_DEFAULT_PERCENT
                if correction_value is None
                else correction_value
            )
            correction = max(0.0, min(1.0, correction_percent / 100.0))
        solar_weight = float(
            getattr(model, "external_source_weights", {}).get(
                "pv", config.PV_HEAT_WEIGHT
            )
        )
        from .thermal_constants import PhysicsConstants

        solar_cap = float(PhysicsConstants.MAX_SOLAR_CONTRIBUTION)
        solar_tau = max(0.0, float(getattr(model, "solar_decay_tau_hours", 0.5)))
        pv_effective_power = []
        prior_power = pv_electrical[0]
        for power in pv_electrical:
            if power >= prior_power or solar_tau <= 0:
                effective = power
            else:
                effective = power + (prior_power - power) * math.exp(
                    -1.0 / solar_tau
                )
            pv_effective_power.append(max(0.0, effective))
            prior_power = effective
        pv_gain = [
            thermal_pv_gain_kw(
                power,
                weight_kw_per_w=solar_weight,
                correction_factor=correction,
                cap_kw=solar_cap,
            )
            for power in pv_effective_power
        ]

        fireplace_on = bool(
            ha_client.get_state(config.FIREPLACE_STATUS_ENTITY_ID, states, is_binary=True)
        )
        channel = None
        orchestrator = getattr(model, "orchestrator", None)
        if orchestrator is not None:
            channel = getattr(orchestrator, "channels", {}).get("fireplace")
        fireplace_kw = float(
            getattr(channel, "fp_heat_output_kw", config.FIREPLACE_HEAT_WEIGHT)
        )
        tau = max(
            0.1,
            float(
                getattr(
                    channel,
                    "fp_decay_time_constant",
                    config.FP_DECAY_TIME_CONSTANT,
                )
            ),
        )
        last_on = self.state.get("fireplace_last_on_utc")
        if fireplace_on:
            self.state["fireplace_last_on_utc"] = now.astimezone(
                timezone.utc
            ).isoformat()
            last_on = self.state["fireplace_last_on_utc"]
        if fireplace_on:
            hours_off = 0.0
        elif last_on:
            try:
                hours_off = max(
                    0.0,
                    (now - datetime.fromisoformat(last_on)).total_seconds() / 3600.0,
                )
            except (TypeError, ValueError):
                hours_off = 100.0
        else:
            hours_off = 100.0

        def fireplace_profile(duration_h: float) -> list[float]:
            profile = []
            for hour in range(horizon):
                if fireplace_on and hour < duration_h:
                    profile.append(fireplace_kw)
                elif fireplace_on and hour >= duration_h:
                    profile.append(
                        fireplace_kw
                        * math.exp(-(hour - duration_h) / tau)
                    )
                else:
                    profile.append(
                        fireplace_kw * math.exp(-(hours_off + hour) / tau)
                    )
            return profile

        main_duration = max(
            0.0, float(config.HEAT_FORECAST_FIREPLACE_DURATION_H)
        )
        fireplace = fireplace_profile(main_duration)
        price_offsets, price_sources = self._target_offsets(
            starts=starts,
            pv_electrical=pv_electrical,
            ha_client=ha_client,
        )
        targets = [target_now + offset for offset in price_offsets]

        flow = self._number(ha_client, states, config.FLOW_RATE_ENTITY_ID)
        capacity_params = {
            "thermal_time_constant": getattr(model, "thermal_time_constant", None),
            "heat_loss_coefficient": getattr(model, "heat_loss_coefficient", None),
            "outlet_effectiveness": getattr(model, "outlet_effectiveness", None),
            "slab_time_constant_hours": getattr(
                model, "slab_time_constant_hours", config.SLAB_TIME_CONSTANT_HOURS
            ),
        }
        capacities = resolve_capacities(
            getattr(config, "BUILDING_BALANCE_CAPACITY_MODE", "model"),
            capacity_params,
            manual_room=5.8,
            manual_slab=3.9,
            median_flow_l_min=flow,
            specific_heat_kj_per_kg_k=config.SPECIFIC_HEAT_CAPACITY,
        )
        inlet = self._number(ha_client, states, config.INLET_TEMP_ENTITY_ID)
        outlet = self._number(
            ha_client, states, config.ACTUAL_OUTLET_TEMP_ENTITY_ID
        )
        slab_start = (
            (inlet + outlet) / 2.0
            if inlet is not None and outlet is not None
            else indoor_now
        )
        heat_loss = max(
            0.01, float(getattr(model, "heat_loss_coefficient", 0.131))
        )
        slab_conductance = max(
            0.01, float(getattr(model, "outlet_effectiveness", 0.2))
        )
        heat_result = simulate_heating_energy(
            outdoor_c=outdoor,
            pv_gain_kw=pv_gain,
            fireplace_kw=fireplace,
            target_c=targets,
            room_c=indoor_now,
            slab_c=slab_start,
            room_capacity_kwh_k=float(capacities["room"]),
            slab_capacity_kwh_k=float(capacities["slab"]),
            heat_loss_kw_k=heat_loss,
            slab_to_room_kw_k=slab_conductance,
            max_hp_kw=config.HEAT_FORECAST_MAX_HEATING_KW,
            step_minutes=config.HEAT_FORECAST_STEP_MINUTES,
            base_gain_kw=config.HEAT_FORECAST_BASE_GAIN_KW,
        )
        if mode != "heating":
            heat_result["heat_kwh"] = [0.0] * horizon

        baseline = [
            max(
                0.0,
                heat_loss * (targets[index] - outdoor[index])
                - pv_gain[index]
                - fireplace[index]
                - config.HEAT_FORECAST_BASE_GAIN_KW,
            )
            if mode == "heating"
            else 0.0
            for index in range(horizon)
        ]
        dhw, standby, auxiliary_stats = self._dhw_and_standby_profiles(starts)
        curve = cop_curve_from_state(
            self.state,
            now=now,
            half_life_days=config.COP_HALF_LIFE_DAYS,
            start_intercept=config.COP_START_INTERCEPT,
            start_slope=config.COP_START_SLOPE,
            min_hours_per_bin=config.COP_MIN_HOURS_PER_BIN,
            min_value=config.COP_MIN_VALUE,
            max_value=config.COP_MAX_VALUE,
        )
        defrost_heat_fraction = _defrost_fraction(
            self.state,
            forecast_key="defrost_heat_fraction_forecast",
            power_key="defrost_heat_fraction_power",
            default=config.DEFROST_DEFAULT_HEAT_FRACTION,
        )
        defrost_el_fraction = _defrost_fraction(
            self.state,
            forecast_key="defrost_el_fraction_forecast",
            power_key="defrost_el_fraction_power",
            default=config.DEFROST_DEFAULT_EL_FRACTION,
        )
        electrical = hourly_electrical_forecast(
            starts=starts,
            heat_th_kwh=heat_result["heat_kwh"],
            outdoor_c=outdoor,
            cop_curve=curve,
            dhw_el_kwh=dhw,
            standby_el_kwh=standby,
            defrost_handling=config.COP_DEFROST_HANDLING,
            defrost_heat_fraction=defrost_heat_fraction,
            defrost_el_fraction=defrost_el_fraction,
        )
        self._correct_current_slot(
            now=now,
            states=states,
            ha_client=ha_client,
            deltas=deltas,
            electrical=electrical,
            heat_result=heat_result,
            outdoor=outdoor,
            remaining_hours=max(
                0.0,
                1.0
                - (
                    now.minute * 60
                    + now.second
                    + now.microsecond / 1_000_000.0
                )
                / 3600.0,
            ),
        )

        scenarios = {}
        if fireplace_on:
            for duration in (3.0, 4.0):
                scenario_heat = simulate_heating_energy(
                    outdoor_c=outdoor,
                    pv_gain_kw=pv_gain,
                    fireplace_kw=fireplace_profile(duration),
                    target_c=targets,
                    room_c=indoor_now,
                    slab_c=slab_start,
                    room_capacity_kwh_k=float(capacities["room"]),
                    slab_capacity_kwh_k=float(capacities["slab"]),
                    heat_loss_kw_k=heat_loss,
                    slab_to_room_kw_k=slab_conductance,
                    max_hp_kw=config.HEAT_FORECAST_MAX_HEATING_KW,
                    step_minutes=config.HEAT_FORECAST_STEP_MINUTES,
                    base_gain_kw=config.HEAT_FORECAST_BASE_GAIN_KW,
                )
                scenario_electric = hourly_electrical_forecast(
                    starts=starts,
                    heat_th_kwh=scenario_heat["heat_kwh"],
                    outdoor_c=outdoor,
                    cop_curve=curve,
                    dhw_el_kwh=dhw,
                    standby_el_kwh=standby,
                    defrost_handling=config.COP_DEFROST_HANDLING,
                    defrost_heat_fraction=defrost_heat_fraction,
                    defrost_el_fraction=defrost_el_fraction,
                )
                scenarios[f"{int(duration)}h"] = {
                    "forecast_heat_th_kwh": scenario_heat["heat_kwh"],
                    "forecast_total_el_kwh": scenario_electric[
                        "forecast_total_el_kwh"
                    ],
                }

        per_hour = [
            {
                "temperature_source": temp_flags[index]["source"],
                "pv_source": pv_flags[index]["source"],
                "price_source": price_sources[index],
                "padded": bool(
                    temp_flags[index]["padded"] or pv_flags[index]["padded"]
                ),
                "cop_extrapolated": bool(
                    curve["calibrated"]
                    and curve.get("temp_range_c")
                    and (
                        outdoor[index] < curve["temp_range_c"][0]
                        or outdoor[index] > curve["temp_range_c"][1]
                    )
                ),
                "mode_flags": {
                    "climate_mode": mode,
                    "fireplace": fireplace_on,
                },
            }
            for index in range(horizon)
        ]
        attrs = {
            **get_sensor_attributes(config.HEAT_FORECAST_SENSOR_ENTITY_ID),
            **electrical,
            "forecast_baseline_th_kwh": [round(value, 4) for value in baseline],
            "forecast_fireplace_scenarios": scenarios,
            "generated_at": now.astimezone(timezone.utc).isoformat(),
            "horizon_hours": horizon,
            "scenario": "configured_fireplace_duration",
            "includes": ["heating", "dhw", "standby"],
            "climate_mode_assumed": mode,
            "cooling_forecast": False,
            "cop_source": "learned_energy_counters" if curve["calibrated"] else "configured_start_curve",
            "cop_calibrated": bool(curve["calibrated"]),
            "cop_curve": curve,
            "defrost_handling": config.COP_DEFROST_HANDLING,
            "defrost_mode": (
                "default"
                if (
                    self.state.get("defrost_heat_fraction_live") is None
                    and self.state.get("defrost_el_fraction_live") is None
                    and self.state.get("defrost_heat_fraction_power") is None
                    and self.state.get("defrost_el_fraction_power") is None
                )
                else (
                    "power_history"
                    if (
                        self.state.get("defrost_heat_fraction_live") is None
                        and self.state.get("defrost_el_fraction_live") is None
                    )
                    else (
                        "blended"
                        if float(self.state.get("defrost_blend", 0.0)) < 1.0
                        else "live"
                    )
                )
            ),
            "defrost_el_in_heating_counter": self.state.get(
                "defrost_el_in_heating_counter", config.COP_DEFROST_EL_IN_HEATING_COUNTER
            ),
            "defrost_el_in_heating_counter_source": (
                self.state.get("defrost_el_in_heating_counter_source")
                or (
                    "configured"
                    if config.COP_DEFROST_EL_IN_HEATING_COUNTER != "auto"
                    else "auto_assumed_included"
                )
            ),
            "defrost_blend": float(self.state.get("defrost_blend", 0.0)),
            "defrost_heat_blend": float(
                self.state.get("defrost_heat_blend", 0.0)
            ),
            "defrost_el_blend": float(
                self.state.get("defrost_el_blend", 0.0)
            ),
            "defrost_power_history": self.state.get(
                "defrost_power_history", {}
            ),
            "defrost_heat_fraction": defrost_heat_fraction,
            "defrost_el_fraction": defrost_el_fraction,
            "learned": bool(curve["calibrated"]),
            **auxiliary_stats,
            "data_quality": per_hour,
            "current_hour_estimate": self.state.get(
                "current_hour_estimate", "counter_plus_power_projection"
            ),
        }
        if config.COP_DEFROST_HANDLING != "separate":
            attrs.pop("forecast_defrost_el_kwh", None)
        attrs["unit_of_measurement"] = "kWh"
        attrs["device_class"] = "energy"
        attrs["state_class"] = "measurement"
        attrs["friendly_name"] = "ML Heating WP Load Forecast"
        self.store.save(self.state)
        return bool(
            ha_client.set_state(
                config.HEAT_FORECAST_SENSOR_ENTITY_ID,
                float(electrical["forecast_total_el_kwh"][0]),
                attrs,
                round_digits=4,
            )
        )

    def _target_offsets(
        self,
        *,
        starts: Sequence[datetime],
        pv_electrical: Sequence[float],
        ha_client: Any,
    ) -> tuple[list[float], list[str]]:
        from . import config
        from .pv_trajectory import get_forecast_trajectory_state

        offsets = [0.0] * len(starts)
        sources = ["none"] * len(starts)
        if not any(
            (
                config.ELECTRICITY_PRICE_ENABLED,
                config.PRICE_COST_MODE_ENABLED,
                config.PV_TRAJ_FORECAST_MODE_ENABLED,
                config.PV_SURPLUS_CHEAP_ENABLED,
            )
        ):
            return offsets, sources
        optimizer = None
        today_prices = None
        if config.ELECTRICITY_PRICE_ENABLED or config.PRICE_COST_MODE_ENABLED:
            try:
                from .price_optimizer import get_price_optimizer

                optimizer = get_price_optimizer()
                optimizer.refresh_prices_if_needed(ha_client)
                today_prices = optimizer.get_today_prices(starts[0])
            except Exception as exc:
                LOGGER.warning("Could not refresh forecast prices: %s", exc)
                optimizer = None

        prev_price = None
        for index, start in enumerate(starts):
            price = None
            source = "none"
            offset = 0.0
            cost_applied = False
            if optimizer is not None:
                try:
                    price = optimizer.get_current_price(start)
                    source = "tibber" if price is not None else "none"
                    if (
                        price is None
                        and config.HEAT_FORECAST_REUSE_PRICES
                        and prev_price is not None
                    ):
                        price = prev_price
                        source = "reused"
                    if price is not None:
                        prev_price = price
                    if config.PRICE_COST_MODE_ENABLED and price is not None:
                        cost = optimizer.compute_cost_target_offset(
                            list(pv_electrical[index:]),
                            float(config.HP_EXPECTED_POWER_FALLBACK_KW),
                            now=start,
                        )
                        if cost is not None:
                            offset = float(cost.get("offset", 0.0))
                            cost_applied = True
                    if (
                        not cost_applied
                        and config.ELECTRICITY_PRICE_ENABLED
                        and price is not None
                        and today_prices
                    ):
                        level = optimizer.classify_price(price, today_prices)
                        offset = float(optimizer.get_target_offset(level))
                except Exception as exc:
                    LOGGER.warning(
                        "Could not resolve forecast price at %s: %s",
                        start.isoformat(),
                        exc,
                    )

            state = get_forecast_trajectory_state(
                float(pv_electrical[index]),
                list(pv_electrical[index + 1 :]),
            )
            if (
                not cost_applied
                and state.active
                and config.PV_TRAJ_FORECAST_MODE_ENABLED
            ):
                offset = max(
                    offset,
                    float(getattr(config, "PV_TRAJ_HEATING_TARGET_OFFSET", 0.0)),
                )
                source = "pv_forecast" if source == "none" else source
            elif (
                not cost_applied
                and config.PV_SURPLUS_CHEAP_ENABLED
                and pv_electrical[index] >= config.PV_SURPLUS_CHEAP_THRESHOLD_W
            ):
                offset = max(offset, config.PRICE_TARGET_OFFSET)
                source = "pv_forecast" if source == "none" else source
            offsets[index] = offset
            sources[index] = source
        return offsets, sources

    def _correct_current_slot(
        self,
        *,
        now: datetime,
        states: Mapping[str, Any],
        ha_client: Any,
        deltas: Mapping[str, float | None],
        electrical: dict[str, Any],
        heat_result: dict[str, Any],
        outdoor: Sequence[float],
        remaining_hours: float,
    ) -> None:
        from . import config

        anchor = self.state.get("hour_anchor")
        readings = anchor.get("readings", {}) if isinstance(anchor, dict) else {}
        initial_energy = (
            anchor.get("energy_so_far", {}) if isinstance(anchor, dict) else {}
        )
        energy_coverage = (
            anchor.get("energy_coverage", {}) if isinstance(anchor, dict) else {}
        )
        heat_el_so_far = float(initial_energy.get("heating_el", 0.0))
        heat_el_so_far = float(initial_energy.get("heating_el", 0.0))
        heat_th_so_far = float(initial_energy.get("heating_th", 0.0))
        dhw_so_far = float(initial_energy.get("dhw_el", 0.0))
        standby_so_far = float(initial_energy.get("standby_el", 0.0))
        defrost_el_so_far = float(initial_energy.get("defrost_el", 0.0))
        defrost_th_so_far = float(initial_energy.get("defrost_th", 0.0))
        self.state["current_hour_estimate"] = "degraded_missing_counter_anchor"
        heating_counter_delta_valid = bool(energy_coverage.get("heating_el", False))
        if isinstance(anchor, dict):
            try:
                anchor_time = datetime.fromisoformat(
                    anchor.get("sample_time_utc", anchor["hour_utc"])
                )
                elapsed = (
                    now.astimezone(timezone.utc) - anchor_time
                ).total_seconds() / 3600.0
                if elapsed <= 0:
                    elapsed = 0.0
                for key, field in (
                    ("heating_el", "heating_el"),
                    ("heating_th", "heating_th"),
                    ("dhw_el", "dhw_el"),
                    ("standby_el", "standby_el"),
                    ("defrost_el", "defrost_el"),
                    ("defrost_th", "defrost_th"),
                ):
                    if elapsed <= 0:
                        continue
                    current = _entity_value(
                        states,
                        {
                            "heating_el": config.HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID,
                            "heating_th": config.HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID,
                            "dhw_el": config.HEAT_FORECAST_DHW_COUNTER_ENTITY_ID,
                            "standby_el": config.HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID,
                            "defrost_el": config.HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID,
                            "defrost_th": config.HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID,
                        }[key],
                    )
                    past = compute_counter_delta(
                        readings.get(key),
                        current,
                        elapsed_hours=max(1e-6, elapsed),
                        max_gap_hours=1.1,
                        max_delta_kwh=10.0,
                    )
                    if past is not None:
                        if field == "heating_el":
                            heat_el_so_far += past
                            heating_counter_delta_valid = True
                        elif field == "heating_th":
                            heat_th_so_far += past
                        elif field == "dhw_el":
                            dhw_so_far += past
                        elif field == "standby_el":
                            standby_so_far += past
                        elif field == "defrost_el":
                            defrost_el_so_far += past
                        elif field == "defrost_th":
                            defrost_th_so_far += past
                if heating_counter_delta_valid:
                    self.state["current_hour_estimate"] = "counter_only_pending_power"
            except (KeyError, TypeError, ValueError):
                self.state["current_hour_estimate"] = "degraded_missing_counter_anchor"
        from . import config

        defrost_el_in_counter = (
            config.COP_DEFROST_EL_IN_HEATING_COUNTER == "true"
            or (
                config.COP_DEFROST_EL_IN_HEATING_COUNTER == "auto"
                and self.state.get("defrost_el_in_heating_counter") != "false"
            )
        )
        heat_el_component_so_far = heat_el_so_far
        defrost_el_component_so_far = defrost_el_so_far
        if config.COP_DEFROST_HANDLING == "separate":
            if defrost_el_in_counter:
                heat_el_component_so_far = max(
                    0.0, heat_el_so_far - defrost_el_so_far
                )
            total_so_far = (
                heat_el_so_far
                + (0.0 if defrost_el_in_counter else defrost_el_so_far)
                + dhw_so_far
                + standby_so_far
            )
            defrost_el_component_so_far += (
                float(electrical["forecast_defrost_el_kwh"][0])
                * remaining_hours
            )
            electrical["forecast_defrost_el_kwh"][0] = round(
                defrost_el_component_so_far, 4
            )
        else:
            if not defrost_el_in_counter:
                heat_el_component_so_far += defrost_el_so_far
            total_so_far = heat_el_component_so_far + dhw_so_far + standby_so_far
        current_power_w = self._number(
            ha_client, states, config.POWER_CONSUMPTION_ENTITY_ID
        )
        if current_power_w is None:
            self.state["current_hour_estimate"] = "degraded_counter_only"
            projected = electrical["forecast_total_el_kwh"][0] * remaining_hours
        else:
            projected = max(0.0, current_power_w) / 1000.0 * remaining_hours
            if self.state["current_hour_estimate"] != "degraded_missing_counter_anchor":
                self.state["current_hour_estimate"] = "counter_plus_power_projection"
        electrical["forecast_total_el_kwh"][0] = round(total_so_far + projected, 4)
        electrical["forecast"][0]["value"] = round(total_so_far + projected, 4)
        electrical["forecast_heat_th_kwh"][0] = round(
            max(0.0, heat_th_so_far - defrost_th_so_far)
            + float(heat_result["heat_kwh"][0]) * remaining_hours,
            4,
        )
        electrical["forecast_heat_el_kwh"][0] = round(
            heat_el_component_so_far
            + float(electrical["forecast_heat_el_kwh"][0]) * remaining_hours,
            4,
        )
        electrical["forecast_dhw_el_kwh"][0] = round(
            dhw_so_far
            + float(electrical["forecast_dhw_el_kwh"][0]) * remaining_hours,
            4,
        )
        electrical["forecast_standby_el_kwh"][0] = round(
            standby_so_far
            + float(electrical["forecast_standby_el_kwh"][0]) * remaining_hours,
            4,
        )


_forecast_engine: HeatForecastEngine | None = None


def publish_heat_forecast_async(
    *,
    ha_client: Any,
    states: Mapping[str, Any],
    wrapper: Any,
    influx_service: Any = None,
) -> None:
    """Run forecast updates in a single background worker, never on control path."""
    from concurrent.futures import ThreadPoolExecutor
    from . import config

    if not (config.HEAT_FORECAST_ENABLED or config.COP_CALIBRATION_ENABLED):
        return
    global _forecast_engine, _forecast_executor, _forecast_future
    if _forecast_executor is None:
        _forecast_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="heat-forecast"
        )
    if _forecast_future is not None and not _forecast_future.done():
        LOGGER.debug("Heat forecast worker is still running; skipping this cycle")
        return
    if _forecast_future is not None:
        try:
            _forecast_future.result()
        except Exception:
            LOGGER.exception("Heat forecast worker failed")
    if _forecast_engine is None:
        _forecast_engine = HeatForecastEngine(config.HEAT_FORECAST_STATE_FILE)
    _forecast_future = _forecast_executor.submit(
        _forecast_engine.update_and_publish,
        ha_client=ha_client,
        states=dict(states),
        wrapper=wrapper,
        influx_service=influx_service,
    )


_forecast_executor = None
_forecast_future = None


def compute_counter_delta(
    previous_kwh: float | None,
    current_kwh: float | None,
    *,
    elapsed_hours: float,
    max_gap_hours: float = 2.0,
    max_delta_kwh: float = 10.0,
) -> float | None:
    """Return a reset-safe counter delta, or ``None`` for unusable samples."""
    try:
        current = float(current_kwh)
        elapsed = float(elapsed_hours)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(current) or current < 0 or elapsed <= 0:
        return None
    if not math.isfinite(elapsed) or elapsed > max_gap_hours:
        return None
    if previous_kwh is None:
        return None
    try:
        previous = float(previous_kwh)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(previous) or previous < 0:
        return None
    delta = current - previous if current >= previous else current
    if delta > max_delta_kwh:
        return None
    return delta


def net_cop_energies(
    *,
    heat_kwh: float,
    electricity_kwh: float,
    defrost_heat_kwh: float,
    defrost_electricity_kwh: float,
    defrost_el_in_heating_counter: bool,
) -> tuple[float, float]:
    """Return delivered heat and total electricity for the net COP."""
    values = (heat_kwh, electricity_kwh, defrost_heat_kwh, defrost_electricity_kwh)
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
        raise ValueError("Energy values must be finite and non-negative")
    heat_net = max(0.0, float(heat_kwh) - float(defrost_heat_kwh))
    electricity_total = float(electricity_kwh)
    if not defrost_el_in_heating_counter:
        electricity_total += float(defrost_electricity_kwh)
    return heat_net, electricity_total


def fit_energy_weighted_cop(
    samples: Sequence[Mapping[str, float]],
    *,
    cop_min: float = 1.5,
    cop_max: float = 10.0,
    max_abs_slope: float = 0.5,
) -> dict[str, Any]:
    """Fit ``COP = intercept + slope * outdoor_temperature`` weighted by kWh."""
    points: list[tuple[float, float, float]] = []
    for sample in samples:
        try:
            outdoor = float(sample["outdoor_c"])
            heat = float(sample["heat_kwh"])
            electric = float(sample["electric_kwh"])
            age_weight = float(sample.get("weight", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not all(math.isfinite(v) for v in (outdoor, heat, electric, age_weight))
            or heat <= 0
            or electric <= 0
            or age_weight <= 0
        ):
            continue
        points.append((outdoor, heat / electric, electric * age_weight))

    if not points:
        return {"intercept": None, "slope": None, "samples": 0, "temp_range_c": None}

    weight_sum = sum(weight for _, _, weight in points)
    mean_x = sum(x * weight for x, _, weight in points) / weight_sum
    mean_y = sum(y * weight for _, y, weight in points) / weight_sum
    variance = sum(weight * (x - mean_x) ** 2 for x, _, weight in points)
    covariance = sum(
        weight * (x - mean_x) * (y - mean_y) for x, y, weight in points
    )
    slope = covariance / variance if variance > 1e-12 else 0.0
    slope = min(max_abs_slope, max(-max_abs_slope, slope))
    intercept = mean_y - slope * mean_x

    observed_min = min(y for _, y, _ in points)
    observed_max = max(y for _, y, _ in points)
    intercept = min(cop_max, max(cop_min, intercept))
    return {
        "intercept": intercept,
        "slope": slope,
        "samples": len(points),
        "temp_range_c": [
            min(x for x, _, _ in points),
            max(x for x, _, _ in points),
        ],
        "observed_cop_range": [observed_min, observed_max],
    }


def forecast_hour_starts(now: datetime, hours: int) -> list[datetime]:
    """Return hour-aligned local timestamps for a UTC-ordered forecast."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    hour = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return [
        (hour + timedelta(hours=index)).astimezone(BERLIN)
        for index in range(max(0, int(hours)))
    ]


def simulate_heating_energy(
    *,
    outdoor_c: Sequence[float],
    pv_gain_kw: Sequence[float],
    fireplace_kw: Sequence[float],
    target_c: Sequence[float],
    room_c: float,
    slab_c: float,
    room_capacity_kwh_k: float,
    slab_capacity_kwh_k: float,
    heat_loss_kw_k: float,
    slab_to_room_kw_k: float,
    max_hp_kw: float = 8.0,
    step_minutes: int = 15,
    target_band_k: float = 0.1,
    base_gain_kw: float = 0.0,
) -> dict[str, Any]:
    """Simulate a two-node room/slab balance and meet the lower comfort band."""
    length = len(outdoor_c)
    if not length or any(
        len(series) != length
        for series in (pv_gain_kw, fireplace_kw, target_c)
    ):
        raise ValueError("Forecast inputs must be non-empty and have equal lengths")
    positive = (
        room_capacity_kwh_k,
        slab_capacity_kwh_k,
        heat_loss_kw_k,
        slab_to_room_kw_k,
        max_hp_kw,
    )
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in positive):
        raise ValueError("Capacities and limits must be finite and positive")
    if step_minutes <= 0 or 60 % step_minutes:
        raise ValueError("step_minutes must be a positive divisor of 60")

    dt = step_minutes / 60.0
    steps_per_hour = 60 // step_minutes
    room = float(room_c)
    slab = float(slab_c)
    heat_kwh: list[float] = []
    hourly_max_kw: list[float] = []

    def advance(q_hp: float, outdoor: float, gains: float) -> tuple[float, float]:
        slab_midpoint = slab + dt * q_hp / (2.0 * slab_capacity_kwh_k)
        transfer_kw = slab_to_room_kw_k * (slab_midpoint - room)
        next_room = room + dt * (
            transfer_kw + gains - heat_loss_kw_k * (room - outdoor)
        ) / room_capacity_kwh_k
        next_slab = slab + dt * (q_hp - transfer_kw) / slab_capacity_kwh_k
        return next_room, next_slab

    for hour in range(length):
        energy = 0.0
        max_kw_this_hour = 0.0
        for _ in range(steps_per_hour):
            target_floor = float(target_c[hour]) - target_band_k
            gains = float(pv_gain_kw[hour]) + float(fireplace_kw[hour]) + base_gain_kw

            def projected_room(q_hp: float) -> float:
                next_room, _ = advance(q_hp, float(outdoor_c[hour]), gains)
                return next_room

            if projected_room(0.0) >= target_floor:
                q_hp = 0.0
            elif projected_room(max_hp_kw) < target_floor:
                q_hp = max_hp_kw
            else:
                lower, upper = 0.0, max_hp_kw
                for _iteration in range(24):
                    middle = (lower + upper) / 2.0
                    if projected_room(middle) >= target_floor:
                        upper = middle
                    else:
                        lower = middle
                q_hp = upper

            room, slab = advance(q_hp, float(outdoor_c[hour]), gains)
            energy += q_hp * dt
            max_kw_this_hour = max(max_kw_this_hour, q_hp)
        heat_kwh.append(energy)
        hourly_max_kw.append(max_kw_this_hour)

    return {
        "heat_kwh": heat_kwh,
        "max_hp_kw": max(hourly_max_kw, default=0.0),
        "hourly_max_hp_kw": hourly_max_kw,
        "room_end_c": room,
        "slab_end_c": slab,
    }
