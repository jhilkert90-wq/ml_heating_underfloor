"""Pure helpers for the opt-in hourly heat-pump load forecast."""

from __future__ import annotations

import math
import json
import logging
import os
import shutil
import tempfile
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")
LOGGER = logging.getLogger(__name__)


class ForecastStateStore:
    """Load and atomically persist compact forecast learning state."""

    VERSION = 2

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
            "backfill_last_attempt_at": None,
            "backfill_attempt_count": 0,
            "backfill_last_error": None,
            "defrost_power_history_last_attempt_at": None,
            "defrost_power_history_attempt_count": 0,
            "defrost_power_history_last_error": None,
            "calibration_source": "defaults",
            "calibration_hours": 0.0,
        }

    def load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Unsupported forecast-state schema")
            version = data.get("schema_version", 1)
            if version not in (1, self.VERSION):
                raise ValueError("Unsupported forecast-state schema")
            state = self.empty()
            state.update(data)
            if version == 1:
                state["schema_version"] = self.VERSION
                if not state["cop_bins"]:
                    state["last_backfill_at"] = None
                power_history = state.get("defrost_power_history", {})
                if (
                    not isinstance(power_history, Mapping)
                    or int(power_history.get("event_count", 0)) < 5
                ):
                    state["defrost_power_history_attempted_at"] = None
            for key in ("counter_snapshots", "cop_bins", "daily_energy"):
                if not isinstance(state.get(key), dict):
                    raise ValueError(f"Invalid forecast state field: {key}")
            return state
        except FileNotFoundError:
            return self._load_backup()
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring invalid heat-forecast state %s: %s", self.path, exc)
            return self._load_backup()

    def _load_backup(self) -> dict[str, Any]:
        backup = self.path.with_suffix(self.path.suffix + ".bak")
        try:
            data = json.loads(backup.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("schema_version") not in (
                1,
                self.VERSION,
            ):
                raise ValueError("Unsupported forecast-state backup schema")
            state = self.empty()
            state.update(data)
            if data.get("schema_version") == 1:
                if not state["cop_bins"]:
                    state["last_backfill_at"] = None
                power_history = state.get("defrost_power_history", {})
                if (
                    not isinstance(power_history, Mapping)
                    or int(power_history.get("event_count", 0)) < 5
                ):
                    state["defrost_power_history_attempted_at"] = None
            state["schema_version"] = self.VERSION
            for key in ("counter_snapshots", "cop_bins", "daily_energy"):
                if not isinstance(state.get(key), dict):
                    raise ValueError(f"Invalid forecast state field: {key}")
            LOGGER.warning("Recovered heat-forecast state from backup %s", backup)
            return state
        except FileNotFoundError:
            return self.empty()
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring invalid heat-forecast backup %s: %s", backup, exc)
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


def _entity_timestamp(
    states: Mapping[str, Any], entity_id: str, fallback: datetime
) -> datetime:
    entity = states.get(entity_id)
    if isinstance(entity, Mapping):
        for field in ("last_updated", "last_changed"):
            raw = entity.get(field)
            if raw:
                try:
                    value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
                except ValueError:
                    continue
    return fallback


def _history_value(raw: Any) -> float | None:
    if isinstance(raw, str):
        state = raw.strip().lower()
        if state in ("on", "true"):
            return 1.0
        if state in ("off", "false"):
            return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _aligned_history_sample(
    samples: Sequence[tuple[datetime, float]],
    timestamp: datetime,
    *,
    tolerance: timedelta = timedelta(minutes=1),
) -> tuple[datetime, float] | None:
    target = timestamp.astimezone(timezone.utc)
    nearest = min(
        samples,
        key=lambda sample: abs(
            sample[0].astimezone(timezone.utc) - target
        ),
        default=None,
    )
    if nearest is None or abs(nearest[0].astimezone(timezone.utc) - target) > tolerance:
        return None
    return nearest


def _time_blocks(
    start: datetime, stop: datetime, *, days: int
) -> list[datetime]:
    result = []
    cursor = start
    while cursor < stop:
        result.append(cursor)
        cursor += timedelta(days=days)
    return result


def _fetch_influx_history_in_blocks(
    influx_service: Any,
    method_name: str,
    entity_ids: Sequence[str],
    start: datetime,
    stop: datetime,
    *,
    block_days: int,
    timeout_seconds: int,
    agg_fn: str | None = None,
) -> dict[str, list[tuple[datetime, float]]]:
    result: dict[str, list[tuple[datetime, float]]] = {
        entity: [] for entity in entity_ids
    }
    method = getattr(influx_service, method_name, None)
    if not callable(method):
        return result
    tasks = [
        (entity, block_start, min(stop, block_start + timedelta(days=block_days)))
        for entity in entity_ids
        for block_start in _time_blocks(start, stop, days=block_days)
    ]

    def fetch(task):
        entity, block_start, block_end = task
        kwargs = {"timeout_seconds": timeout_seconds}
        if agg_fn is not None:
            kwargs["agg_fn"] = agg_fn
        try:
            return entity, method([entity], block_start, block_end, **kwargs).get(
                entity, []
            )
        except Exception as exc:
            LOGGER.info(
                "Influx forecast history failed for %s: %s", entity, exc
            )
            return entity, []

    if len(tasks) == 1:
        fetched = [fetch(tasks[0])]
    else:
        with ThreadPoolExecutor(
            max_workers=min(8, len(tasks)), thread_name_prefix="forecast-history"
        ) as executor:
            fetched = list(executor.map(fetch, tasks))
    for entity, samples in fetched:
        result[entity].extend(samples)
    return {
        entity: _merge_samples([], samples)
        for entity, samples in result.items()
    }


def _merge_samples(
    first: Sequence[tuple[datetime, float]],
    second: Sequence[tuple[datetime, float]],
) -> list[tuple[datetime, float]]:
    merged = {stamp.astimezone(timezone.utc): value for stamp, value in first}
    merged.update(
        {stamp.astimezone(timezone.utc): value for stamp, value in second}
    )
    return sorted(merged.items())


def _fetch_ha_history(
    ha_client: Any,
    entity_ids: Sequence[str],
    start: datetime,
    stop: datetime,
) -> dict[str, list[tuple[datetime, float]]]:
    history: dict[str, list[tuple[datetime, float]]] = {
        entity: [] for entity in entity_ids
    }
    if not entity_ids or not callable(getattr(ha_client, "get_history_bulk", None)):
        return history

    def fetch(block_start: datetime):
        block_end = min(stop, block_start + timedelta(days=3))
        try:
            response = ha_client.get_history_bulk(
                list(entity_ids), block_start, block_end
            )
        except Exception as exc:
            LOGGER.info("Home Assistant forecast history failed: %s", exc)
            return {}
        block_history = {entity: [] for entity in entity_ids}
        if not isinstance(response, list):
            return block_history
        for records in response:
            if not isinstance(records, list):
                continue
            entity_id = next(
                (
                    record.get("entity_id")
                    for record in records
                    if isinstance(record, Mapping) and record.get("entity_id")
                ),
                None,
            )
            if entity_id not in block_history:
                continue
            for record in records if isinstance(records, list) else ():
                if not isinstance(record, Mapping):
                    continue
                try:
                    value = _history_value(record.get("state"))
                    if value is None:
                        continue
                    raw_time = record.get("last_changed") or record.get("last_updated")
                    stamp = datetime.fromisoformat(
                        str(raw_time).replace("Z", "+00:00")
                    )
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=timezone.utc)
                    block_history[entity_id].append(
                        (stamp.astimezone(timezone.utc), value)
                    )
                except (TypeError, ValueError):
                    continue
        return block_history

    blocks = _time_blocks(start, stop, days=3)
    if len(blocks) > 1:
        with ThreadPoolExecutor(
            max_workers=min(4, len(blocks)), thread_name_prefix="forecast-ha-history"
        ) as executor:
            block_histories = list(executor.map(fetch, blocks))
    else:
        block_histories = [fetch(blocks[0])] if blocks else []
    for block_history in block_histories:
        for entity_id, samples in block_history.items():
            history[entity_id].extend(samples)
    for entity_id, samples in history.items():
        history[entity_id] = _merge_samples([], samples)
    return history


def _calibration_hours(
    history: Mapping[str, Sequence[tuple[datetime, float]]],
    heat_id: str,
    electricity_id: str,
    outdoor_id: str,
) -> float:
    if not all(history.get(entity) for entity in (heat_id, electricity_id, outdoor_id)):
        return 0.0
    heat = sorted(history[heat_id], key=lambda sample: sample[0])
    electricity = history[electricity_id]
    outdoor = history[outdoor_id]
    total_hours = 0.0
    for (previous_time, _), (current_time, _) in zip(heat, heat[1:]):
        elapsed = (current_time - previous_time).total_seconds() / 3600.0
        if not 0 < elapsed <= 2.0:
            continue
        if (
            _aligned_history_sample(electricity, previous_time) is None
            or _aligned_history_sample(electricity, current_time) is None
            or _aligned_history_sample(outdoor, current_time) is None
        ):
            continue
        total_hours += elapsed
    return total_hours


def _retry_ready(state: Mapping[str, Any], prefix: str, now: datetime) -> bool:
    attempt = state.get(f"{prefix}_last_attempt_at")
    if not attempt:
        return True
    try:
        last_attempt = datetime.fromisoformat(str(attempt))
        elapsed = (now.astimezone(timezone.utc) - last_attempt).total_seconds()
    except (TypeError, ValueError):
        return True
    count = int(state.get(f"{prefix}_attempt_count", 0))
    delays = (5, 15, 60, 360, 360)
    delay_minutes = delays[min(max(0, count - 1), len(delays) - 1)]
    return elapsed >= delay_minutes * 60


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


def _add_daily_energy_interval(
    days: dict[str, dict[str, Any]],
    *,
    previous_time: datetime,
    current_time: datetime,
    delta: float,
    field: str,
) -> None:
    """Allocate interval energy and coverage to its local-hour buckets."""
    if previous_time.tzinfo is None:
        previous_time = previous_time.replace(tzinfo=timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    start = previous_time.astimezone(timezone.utc)
    stop = current_time.astimezone(timezone.utc)
    total_hours = (stop - start).total_seconds() / 3600.0
    if total_hours <= 0:
        return
    cursor = start
    while cursor < stop:
        next_hour = cursor.replace(minute=0, second=0, microsecond=0) + timedelta(
            hours=1
        )
        segment_stop = min(stop, next_hour)
        segment_hours = (segment_stop - cursor).total_seconds() / 3600.0
        local = (cursor + (segment_stop - cursor) / 2).astimezone(BERLIN)
        day = days.setdefault(
            local.date().isoformat(),
            {
                "dhw": [0.0] * 24,
                "standby": [0.0] * 24,
                "dhw_coverage_h": 0.0,
                "standby_coverage_h": 0.0,
            },
        )
        day[field][local.hour] += delta * segment_hours / total_hours
        coverage_key = f"{field}_coverage_h"
        day[coverage_key] = min(
            24.0, float(day.get(coverage_key, 0.0)) + segment_hours
        )
        cursor = segment_stop


def _smooth_heating_energy(values: Sequence[float]) -> list[float]:
    """Spread isolated thermal-energy peaks across their six-hour block."""
    result = [max(0.0, float(value)) for value in values]
    for start in range(0, len(result), 6):
        stop = min(start + 6, len(result))
        block = result[start:stop]
        if len(block) < 3:
            continue
        average = sum(block) / len(block)
        if max(block) >= max(average * 1.8, average + 1.0):
            result[start:stop] = [average] * len(block)
    return result


def _smooth_future_heating_energy(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    return [float(values[0]), *_smooth_heating_energy(values[1:])]


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


def _resample_history_to_minutes(
    histories: Sequence[Sequence[tuple[datetime, float]]],
    max_carry_minutes: Sequence[int],
) -> list[list[tuple[datetime, float]]]:
    if not histories or len(histories) != len(max_carry_minutes):
        return [[] for _ in histories]
    normalized = [
        sorted(
            (
                (stamp.astimezone(timezone.utc), float(value))
                for stamp, value in samples
                if math.isfinite(float(value))
            ),
            key=lambda sample: sample[0],
        )
        for samples in histories
    ]
    power_samples = normalized[0]
    if not power_samples:
        return [[] for _ in histories]
    first_sample = min(
        power_samples[0][0],
        *(samples[0][0] for samples in normalized[1:] if samples),
    )
    grid_start = first_sample.replace(second=0, microsecond=0)
    if first_sample > grid_start:
        grid_start += timedelta(minutes=1)
    grid_end = power_samples[-1][0].replace(second=0, microsecond=0)
    if grid_end < grid_start:
        return [[] for _ in histories]
    result: list[list[tuple[datetime, float]]] = [[] for _ in histories]
    indices = [0] * len(normalized)
    latest: list[tuple[datetime, float] | None] = [None] * len(normalized)
    minute = grid_start
    while minute <= grid_end:
        for index, samples in enumerate(normalized):
            while indices[index] < len(samples) and samples[indices[index]][0] <= minute:
                latest[index] = samples[indices[index]]
                indices[index] += 1
            sample = latest[index]
            if (
                sample is not None
                and (minute - sample[0]).total_seconds()
                <= max_carry_minutes[index] * 60
            ):
                result[index].append((minute, sample[1]))
        minute += timedelta(minutes=1)
    return result


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
    active = sorted(
        (
            stamp.astimezone(timezone.utc).replace(second=0, microsecond=0),
            float(value) > 0,
        )
        for stamp, value in defrost_state
        if math.isfinite(float(value))
    )
    thermal = {
        stamp.astimezone(timezone.utc).replace(second=0, microsecond=0): float(value)
        for stamp, value in thermal_power
        if math.isfinite(float(value))
    }
    minutes = sorted(power)
    event_electricity: list[float] = []
    event_heat: list[float] = []
    current_electricity = current_heat = 0.0
    thermal_minutes = 0
    total_active_minutes = total_thermal_minutes = 0
    previous_minute: datetime | None = None
    in_event = False
    state_index = 0
    known_active: bool | None = None

    def finish_event() -> None:
        nonlocal current_electricity, current_heat, thermal_minutes, in_event
        if in_event:
            event_electricity.append(current_electricity)
            event_heat.append(current_heat)
        current_electricity = current_heat = 0.0
        thermal_minutes = 0
        in_event = False

    for minute in minutes:
        while state_index < len(active) and active[state_index][0] <= minute:
            known_active = active[state_index][1]
            state_index += 1
        contiguous = (
            previous_minute is not None
            and (minute - previous_minute).total_seconds() <= 120
        )
        if not contiguous or known_active is None or not known_active:
            finish_event()
        if known_active:
            if not in_event:
                current_electricity = current_heat = 0.0
                in_event = True
            current_electricity += power[minute] / 60_000.0
            total_active_minutes += 1
            if minute in thermal:
                current_heat += max(0.0, -thermal[minute]) / 60_000.0
                thermal_minutes += 1
                total_thermal_minutes += 1
        previous_minute = minute
    finish_event()

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
        "heat_fraction": (
            fraction(sum(event_heat), heating_heat_kwh)
            if total_thermal_minutes > 0
            else None
        ),
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
                "outdoor_c": (
                    float(bucket.get("temp_energy_kwh", 0.0)) * decay / electric
                ),
                "heat_kwh": heat,
                "electric_kwh": electric,
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
    cop_min: float = 1.5,
    cop_max: float = 10.0,
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
        cop = min(cop_max, max(cop_min, cop))
        if defrost_handling == "separate":
            hp_electric = heat / cop
            defrost_electric = (
                hp_electric * defrost_el_fraction / max(1e-9, 1.0 - defrost_el_fraction)
                if outdoor <= 5.0
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
        self._price_optimizer = None

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
        interval_starts: dict[str, datetime | None] = {}
        interval_ends: dict[str, datetime | None] = {}
        prior_snapshots = dict(self.state.get("counter_snapshots", {}))
        for name, entity_id in counter_entities.items():
            reading = _entity_value(states, entity_id)
            readings[name] = reading
            current_time = _entity_timestamp(states, entity_id, now).astimezone(
                timezone.utc
            )
            delta, duration = update_counter_snapshot(
                self.state,
                name,
                reading,
                current_time,
                max_delta_kwh=max_deltas[name],
            )
            deltas[name] = delta
            elapsed[name] = duration
            previous = prior_snapshots.get(name)
            try:
                interval_starts[name] = (
                    datetime.fromisoformat(previous["time_utc"]).astimezone(
                        timezone.utc
                    )
                    if isinstance(previous, Mapping)
                    else None
                )
            except (KeyError, TypeError, ValueError):
                interval_starts[name] = None
            interval_ends[name] = current_time if reading is not None else None

        def intervals_aligned(first: str, second: str) -> bool:
            first_start, second_start = interval_starts.get(first), interval_starts.get(second)
            first_end, second_end = interval_ends.get(first), interval_ends.get(second)
            return bool(
                first_start is not None
                and second_start is not None
                and first_end is not None
                and second_end is not None
                and abs((first_start - second_start).total_seconds()) <= 60
                and abs((first_end - second_end).total_seconds()) <= 60
            )

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
        elif config.COP_DEFROST_EL_IN_HEATING_COUNTER in ("true", "false"):
            self.state["defrost_el_in_heating_counter"] = (
                config.COP_DEFROST_EL_IN_HEATING_COUNTER
            )
            self.state["defrost_el_in_heating_counter_source"] = "configured"

        heater_interval = elapsed.get("heating_th")
        electricity_interval = elapsed.get("heating_el")
        if (
            (config.COP_CALIBRATION_ENABLED or config.HEAT_FORECAST_ENABLED)
            and deltas.get("heating_th") is not None
            and deltas.get("heating_el") is not None
            and intervals_aligned("heating_th", "heating_el")
            and heater_interval is not None
            and electricity_interval is not None
            and abs(float(electricity_interval) - float(heater_interval))
            <= 1.0 / 60.0
        ):
            heat = float(deltas["heating_th"])
            electricity = float(deltas["heating_el"])
            defrost_heat = deltas.get("defrost_th")
            defrost_electricity = deltas.get("defrost_el")
            if not intervals_aligned("heating_th", "defrost_th"):
                defrost_heat = None
            if not intervals_aligned("heating_el", "defrost_el"):
                defrost_electricity = None
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
            heat_net, electric_total = cop_calibration_energies(
                heat_kwh=heat,
                electricity_kwh=electricity,
                defrost_heat_kwh=defrost_heat,
                defrost_electricity_kwh=defrost_electricity,
                defrost_el_in_heating_counter=(
                    el_in_counter
                ),
                defrost_handling=config.COP_DEFROST_HANDLING,
            )
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

        self._record_daily_energy(now, deltas, elapsed, prior_snapshots)
        hour_key = now.astimezone(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        ).isoformat()
        hour_anchor = self.state.get("hour_anchor")
        if not isinstance(hour_anchor, dict) or hour_anchor.get("hour_utc") != hour_key:
            hour_start = datetime.fromisoformat(hour_key)
            now_utc = now.astimezone(timezone.utc)
            energy_so_far: dict[str, float] = {}
            energy_coverage: dict[str, bool] = {}
            energy_coverage_hours: dict[str, float] = {}
            reading_times: dict[str, str] = {}
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
                covered_hours = 0.0
                snapshot = self.state.get("counter_snapshots", {}).get(name, {})
                if delta is not None and duration is not None and isinstance(previous, Mapping):
                    try:
                        previous_time = datetime.fromisoformat(
                            previous["time_utc"]
                        ).astimezone(timezone.utc)
                        interval_end = datetime.fromisoformat(
                            snapshot["time_utc"]
                        ).astimezone(timezone.utc)
                        interval_start = previous_time
                        actual_duration = (
                            interval_end - interval_start
                        ).total_seconds() / 3600.0
                        covered_hours = max(
                            0.0,
                            (
                                min(interval_end, now_utc)
                                - max(interval_start, hour_start)
                            ).total_seconds()
                            / 3600.0,
                        )
                        if actual_duration > 0:
                            covered_energy = float(delta) * min(
                                1.0, covered_hours / actual_duration
                            )
                        valid = (
                            interval_end == hour_start
                            or previous_time == hour_start
                        )
                        reading_times[name] = interval_end.isoformat()
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):
                        valid = False
                if name not in reading_times:
                    reading_times[name] = snapshot.get(
                        "time_utc", now_utc.isoformat()
                    )
                energy_so_far[name] = covered_energy
                energy_coverage[name] = valid
                energy_coverage_hours[name] = covered_hours
            usable_counter_anchor = False
            if readings.get("heating_el") is not None:
                try:
                    anchor_time = datetime.fromisoformat(
                        reading_times["heating_el"]
                    ).astimezone(timezone.utc)
                    usable_counter_anchor = (
                        hour_start - timedelta(hours=1)
                        <= anchor_time
                        <= hour_start
                        or (
                            hour_start < anchor_time <= now_utc
                            and energy_coverage.get("heating_el", False)
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    LOGGER.info("Ignoring malformed heating counter anchor timestamp")
            self.state["hour_anchor"] = {
                "hour_utc": hour_key,
                "sample_time_utc": now_utc.isoformat(),
                "reading_times_utc": reading_times,
                "readings": readings,
                "energy_so_far": energy_so_far,
                "energy_coverage_hours": energy_coverage_hours,
                "energy_coverage": energy_coverage,
                "coverage": (
                    "counter_anchor" if usable_counter_anchor else "partial"
                ),
            }
        return deltas

    def _record_daily_energy(
        self,
        now: datetime,
        deltas: Mapping[str, float | None],
        elapsed: Mapping[str, float | None],
        prior_snapshots: Mapping[str, Any],
    ) -> None:
        local_now = now.astimezone(BERLIN)
        days = self.state.setdefault("daily_energy", {})
        for source, field in (("dhw_el", "dhw"), ("standby_el", "standby")):
            delta = deltas.get(source)
            duration = elapsed.get(source)
            if (
                delta is not None
                and duration is not None
                and 0 < float(duration) <= 2.0
            ):
                previous = prior_snapshots.get(source)
                try:
                    interval_start = datetime.fromisoformat(
                        previous["time_utc"]
                    )
                except (KeyError, TypeError, ValueError):
                    interval_start = now - timedelta(hours=float(duration))
                current = self.state.get("counter_snapshots", {}).get(source, {})
                try:
                    interval_end = datetime.fromisoformat(current["time_utc"])
                except (KeyError, TypeError, ValueError):
                    interval_end = now
                _add_daily_energy_interval(
                    days,
                    previous_time=interval_start,
                    current_time=interval_end,
                    delta=float(delta),
                    field=field,
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
        ha_client: Any = None,
    ) -> None:
        from . import config

        if not _retry_ready(self.state, "backfill", now):
            return
        if self.state.get("last_backfill_at"):
            self._refresh_hour_anchor(
                now=now,
                influx_service=influx_service,
                ha_client=ha_client,
            )
            self._backfill_defrost_power_history(
                now=now,
                influx_service=influx_service,
                ha_client=ha_client,
            )
            self._backfill_attempted = True
            return
        self.state["backfill_last_attempt_at"] = now.astimezone(
            timezone.utc
        ).isoformat()
        self.state["backfill_attempt_count"] = int(
            self.state.get("backfill_attempt_count", 0)
        ) + 1
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
        start = now - timedelta(days=90)
        history: dict[str, list[tuple[datetime, float]]] = {
            entity: [] for entity in ids
        }
        source = "defaults"
        if influx_service is not None:
            history = _fetch_influx_history_in_blocks(
                influx_service,
                "fetch_counter_history",
                ids,
                start,
                now,
                block_days=14,
                timeout_seconds=config.HEAT_FORECAST_INFLUX_TIMEOUT_SECONDS,
            )
            if any(history.values()):
                source = "influx"
        heat_id = config.HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID
        el_id = config.HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID
        outdoor_id = outdoor_entity_id
        min_hours = int(getattr(config, "HEAT_FORECAST_MIN_CALIBRATION_HOURS", 48))
        current_calibration_hours = _calibration_hours(
            history, heat_id, el_id, outdoor_id
        )
        needs_history_fallback = current_calibration_hours < min_hours or any(
            len(history.get(entity, [])) < 2 for entity in ids
        )
        if needs_history_fallback and ha_client is not None:
            ha_history = _fetch_ha_history(
                ha_client,
                ids,
                max(start, now - timedelta(days=10)),
                now,
            )
            has_influx_history = any(history.values())
            added_ha_history = False
            for entity in ids:
                influx_samples = history.get(entity, [])
                ha_samples = ha_history.get(entity, [])
                history[entity] = _merge_samples(influx_samples, ha_samples)
                added_ha_history = added_ha_history or (
                    len(history[entity]) > len(influx_samples)
                )
            fallback_hours = _calibration_hours(history, heat_id, el_id, outdoor_id)
            if added_ha_history:
                source = "mixed" if has_influx_history else "homeassistant"
            current_calibration_hours = fallback_hours

        if not any(history.values()):
            self.state["backfill_last_error"] = (
                "No counter history from InfluxDB or Home Assistant"
            )
            self.state["calibration_source"] = "defaults"
            self.store.save(self.state)
            return
        self.state["backfill_last_error"] = None
        self._backfill_daily_energy(history, now)
        self._backfill_defrost_power_history(
            now=now,
            influx_service=influx_service,
            ha_client=ha_client,
            counter_history=history,
        )
        hour_start = now.astimezone(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )
        anchor_readings = {}
        anchor_coverage = {}
        anchor_times = {}
        for name, entity_id in (
            ("heating_el", el_id),
            ("heating_th", heat_id),
            ("dhw_el", config.HEAT_FORECAST_DHW_COUNTER_ENTITY_ID),
            ("standby_el", config.HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID),
            ("defrost_el", config.HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID),
            ("defrost_th", config.HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID),
        ):
            samples = history.get(entity_id, []) if entity_id else []
            candidates = [
                (stamp.astimezone(timezone.utc), value)
                for stamp, value in samples
                if stamp.astimezone(timezone.utc) <= hour_start
            ]
            sample = max(candidates, default=None, key=lambda item: item[0])
            if sample and hour_start - sample[0] <= timedelta(hours=1):
                anchor_readings[name] = sample[1]
                anchor_times[name] = sample[0].isoformat()
                anchor_coverage[name] = sample[0] == hour_start
            else:
                anchor_readings[name] = None
                anchor_times[name] = hour_start.isoformat()
                anchor_coverage[name] = False
        if anchor_readings.get("heating_el") is not None:
            self.state["hour_anchor"] = {
                "hour_utc": hour_start.isoformat(),
                "sample_time_utc": hour_start.isoformat(),
                "reading_times_utc": anchor_times,
                "readings": anchor_readings,
                "energy_so_far": {key: 0.0 for key in anchor_readings},
                "energy_coverage_hours": {key: 0.0 for key in anchor_readings},
                "energy_coverage": anchor_coverage,
                "coverage": (
                    "counter_anchor"
                    if anchor_readings.get("heating_el") is not None
                    else "partial"
                ),
            }
        if not config.COP_CALIBRATION_ENABLED:
            self.state["last_backfill_at"] = now.astimezone(
                timezone.utc
            ).isoformat()
            self.state["backfill_last_error"] = None
            self.state["calibration_source"] = source
            self.state["calibration_hours"] = current_calibration_hours
            self.store.save(self.state)
            return
        if current_calibration_hours < min_hours:
            self.state["backfill_last_error"] = (
                f"Only {current_calibration_hours:.1f} calibration hours available"
            )
            self.state["calibration_source"] = source if current_calibration_hours else "defaults"
            self.state["calibration_hours"] = current_calibration_hours
            self.store.save(self.state)
            return
        indexed = {
            entity: sorted(entries, key=lambda item: item[0])
            for entity, entries in history.items()
        }
        heat_samples = indexed.get(heat_id, [])
        processed_intervals = 0
        for (previous, previous_heat), (current, current_heat) in zip(
            heat_samples, heat_samples[1:]
        ):
            elapsed = (current - previous).total_seconds() / 3600.0
            previous_electric = _aligned_history_sample(indexed.get(el_id, []), previous)
            current_electric = _aligned_history_sample(indexed.get(el_id, []), current)
            outdoor_sample = _aligned_history_sample(
                indexed.get(outdoor_id, []), current
            )
            if (
                previous_electric is None
                or current_electric is None
                or outdoor_sample is None
            ):
                continue
            electric_elapsed = (
                current_electric[0] - previous_electric[0]
            ).total_seconds() / 3600.0
            if (
                abs((previous_electric[0] - previous).total_seconds()) > 60
                or abs((current_electric[0] - current).total_seconds()) > 60
                or abs(electric_elapsed - elapsed) > 1.0 / 60.0
            ):
                continue
            heat = compute_counter_delta(
                previous_heat,
                current_heat,
                elapsed_hours=elapsed,
                max_delta_kwh=10.0,
            )
            electric = compute_counter_delta(
                previous_electric[1],
                current_electric[1],
                elapsed_hours=electric_elapsed,
                max_delta_kwh=3.0,
            )
            if heat is None or electric is None or elapsed > 2.0:
                continue
            outdoor = outdoor_sample[1]
            defrost_heat_value = indexed.get(
                config.HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID, {}
            )
            defrost_heat_previous = _aligned_history_sample(defrost_heat_value, previous)
            defrost_heat_current = _aligned_history_sample(defrost_heat_value, current)
            defrost_heat = None
            if defrost_heat_previous is not None and defrost_heat_current is not None:
                defrost_heat_elapsed = (
                    defrost_heat_current[0] - defrost_heat_previous[0]
                ).total_seconds() / 3600.0
                if (
                    abs((defrost_heat_previous[0] - previous).total_seconds()) <= 60
                    and abs((defrost_heat_current[0] - current).total_seconds()) <= 60
                    and abs(defrost_heat_elapsed - elapsed) <= 1.0 / 60.0
                ):
                    defrost_heat = compute_counter_delta(
                        defrost_heat_previous[1],
                        defrost_heat_current[1],
                        elapsed_hours=defrost_heat_elapsed,
                        max_delta_kwh=10.0,
                    )
            if defrost_heat is None and outdoor <= 5.0:
                defrost_heat = heat * config.DEFROST_DEFAULT_HEAT_FRACTION
            defrost_electricity_value = indexed.get(
                config.HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID, {}
            )
            defrost_electricity_previous = _aligned_history_sample(
                defrost_electricity_value, previous
            )
            defrost_electricity_current = _aligned_history_sample(
                defrost_electricity_value, current
            )
            defrost_electricity = 0.0
            if (
                defrost_electricity_previous is not None
                and defrost_electricity_current is not None
            ):
                defrost_electricity_elapsed = (
                    defrost_electricity_current[0]
                    - defrost_electricity_previous[0]
                ).total_seconds() / 3600.0
                if (
                    abs(
                        (
                            defrost_electricity_previous[0] - previous
                        ).total_seconds()
                    )
                    <= 60
                    and abs(
                        (
                            defrost_electricity_current[0] - current
                        ).total_seconds()
                    )
                    <= 60
                    and abs(defrost_electricity_elapsed - elapsed) <= 1.0 / 60.0
                ):
                    defrost_electricity = compute_counter_delta(
                        defrost_electricity_previous[1],
                        defrost_electricity_current[1],
                        elapsed_hours=defrost_electricity_elapsed,
                        max_delta_kwh=3.0,
                    ) or 0.0
            el_in_counter = (
                config.COP_DEFROST_EL_IN_HEATING_COUNTER == "true"
                if config.COP_DEFROST_EL_IN_HEATING_COUNTER != "auto"
                else self.state.get("defrost_el_in_heating_counter") != "false"
            )
            heat_net, total_electric = cop_calibration_energies(
                heat_kwh=heat,
                electricity_kwh=electric,
                defrost_heat_kwh=float(defrost_heat or 0.0),
                defrost_electricity_kwh=defrost_electricity,
                defrost_el_in_heating_counter=el_in_counter,
                defrost_handling=config.COP_DEFROST_HANDLING,
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
            if heat_net > 0 and total_electric > 0:
                processed_intervals += 1
        if processed_intervals == 0:
            self.state["backfill_last_error"] = (
                "No plausible aligned heating-counter intervals"
            )
            self.state["calibration_source"] = "defaults"
            self.state["calibration_hours"] = 0.0
            self.store.save(self.state)
            return
        self.state["last_backfill_at"] = now.astimezone(timezone.utc).isoformat()
        self.state["backfill_last_error"] = None
        self.state["calibration_source"] = source
        self.state["calibration_hours"] = current_calibration_hours
        self.store.save(self.state)
        LOGGER.info(
            "Heat forecast history backfill processed %d hourly intervals",
            processed_intervals,
        )

    def _refresh_hour_anchor(
        self,
        *,
        now: datetime,
        influx_service: Any,
        ha_client: Any,
    ) -> None:
        from . import config

        hour_start = now.astimezone(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )
        hour_key = hour_start.isoformat()
        anchor = self.state.get("hour_anchor")
        if (
            isinstance(anchor, dict)
            and anchor.get("hour_utc") == hour_key
            and anchor.get("coverage") == "counter_anchor"
        ):
            return
        if self.state.get("hour_anchor_refresh_hour") == hour_key:
            return
        self.state["hour_anchor_refresh_hour"] = hour_key

        counters = (
            ("heating_el", config.HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID),
            ("heating_th", config.HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID),
            ("dhw_el", config.HEAT_FORECAST_DHW_COUNTER_ENTITY_ID),
            ("standby_el", config.HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID),
            ("defrost_el", config.HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID),
            ("defrost_th", config.HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID),
        )
        entity_ids = list(dict.fromkeys(entity for _, entity in counters if entity))
        history = {entity: [] for entity in entity_ids}
        start = hour_start - timedelta(hours=1)
        if influx_service is not None:
            history = _fetch_influx_history_in_blocks(
                influx_service,
                "fetch_counter_history",
                entity_ids,
                start,
                now,
                block_days=14,
                timeout_seconds=config.HEAT_FORECAST_INFLUX_TIMEOUT_SECONDS,
            )

        def latest_anchor(
            samples: Sequence[tuple[datetime, float]],
        ) -> tuple[datetime, float] | None:
            eligible = [
                (stamp.astimezone(timezone.utc), value)
                for stamp, value in samples
                if stamp.astimezone(timezone.utc) <= hour_start
                and hour_start - stamp.astimezone(timezone.utc)
                <= timedelta(hours=1)
            ]
            return max(eligible, default=None, key=lambda item: item[0])

        heating_entity = config.HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID
        if not latest_anchor(history.get(heating_entity, [])) and ha_client is not None:
            ha_history = _fetch_ha_history(ha_client, entity_ids, start, now)
            history = {
                entity: _merge_samples(
                    history.get(entity, []), ha_history.get(entity, [])
                )
                for entity in entity_ids
            }

        readings: dict[str, float | None] = {}
        reading_times: dict[str, str] = {}
        energy_coverage: dict[str, bool] = {}
        for name, entity_id in counters:
            sample = latest_anchor(history.get(entity_id, [])) if entity_id else None
            readings[name] = sample[1] if sample else None
            reading_times[name] = sample[0].isoformat() if sample else hour_key
            energy_coverage[name] = bool(sample and sample[0] == hour_start)

        if readings.get("heating_el") is not None:
            self.state["hour_anchor"] = {
                "hour_utc": hour_key,
                "sample_time_utc": hour_key,
                "reading_times_utc": reading_times,
                "readings": readings,
                "energy_so_far": {key: 0.0 for key in readings},
                "energy_coverage_hours": {key: 0.0 for key in readings},
                "energy_coverage": energy_coverage,
                "coverage": "counter_anchor",
            }
        elif not isinstance(anchor, dict) or anchor.get("hour_utc") != hour_key:
            self.state["hour_anchor"] = {
                "hour_utc": hour_key,
                "sample_time_utc": now.astimezone(timezone.utc).isoformat(),
                "reading_times_utc": reading_times,
                "readings": readings,
                "energy_so_far": {key: 0.0 for key in readings},
                "energy_coverage_hours": {key: 0.0 for key in readings},
                "energy_coverage": energy_coverage,
                "coverage": "partial",
            }

    def _backfill_defrost_power_history(
        self,
        *,
        now: datetime,
        influx_service: Any,
        ha_client: Any = None,
        counter_history: Mapping[str, Sequence[tuple[datetime, float]]] | None = None,
    ) -> None:
        from . import config

        if self.state.get("defrost_power_history_attempted_at"):
            return
        if not _retry_ready(self.state, "defrost_power_history", now):
            return
        self.state["defrost_power_history_last_attempt_at"] = now.astimezone(
            timezone.utc
        ).isoformat()
        self.state["defrost_power_history_attempt_count"] = int(
            self.state.get("defrost_power_history_attempt_count", 0)
        ) + 1
        fetch_minute_history = getattr(influx_service, "fetch_minute_history", None)

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
                    counter_ids,
                    start,
                    now,
                    timeout_seconds=config.HEAT_FORECAST_INFLUX_TIMEOUT_SECONDS,
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
        power_history: dict[str, list[tuple[datetime, float]]] = {
            entity: [] for entity in power_entities
        }
        defrost_history: dict[str, list[tuple[datetime, float]]] = {
            config.DEFROST_STATUS_ENTITY_ID: []
        }
        if callable(fetch_minute_history):
            power_history = _fetch_influx_history_in_blocks(
                influx_service,
                "fetch_minute_history",
                power_entities,
                start,
                now,
                block_days=3,
                timeout_seconds=config.HEAT_FORECAST_INFLUX_TIMEOUT_SECONDS,
                agg_fn="mean",
            )
            defrost_history = _fetch_influx_history_in_blocks(
                influx_service,
                "fetch_minute_history",
                [config.DEFROST_STATUS_ENTITY_ID],
                start,
                now,
                block_days=3,
                timeout_seconds=config.HEAT_FORECAST_INFLUX_TIMEOUT_SECONDS,
                agg_fn="max",
            )
        used_ha_history = False
        if (not any(power_history.values()) or not defrost_history.get(
            config.DEFROST_STATUS_ENTITY_ID
        )) and ha_client is not None:
            fallback = _fetch_ha_history(
                ha_client, power_entities + [config.DEFROST_STATUS_ENTITY_ID], start, now
            )
            used_ha_history = True
            for entity in power_entities:
                if not power_history.get(entity):
                    power_history[entity] = fallback.get(entity, [])
            if not defrost_history.get(config.DEFROST_STATUS_ENTITY_ID):
                defrost_history[config.DEFROST_STATUS_ENTITY_ID] = fallback.get(
                    config.DEFROST_STATUS_ENTITY_ID, []
                )
        if used_ha_history:
            electrical, status, thermal = _resample_history_to_minutes(
                [
                    power_history.get(config.POWER_CONSUMPTION_ENTITY_ID, []),
                    defrost_history.get(config.DEFROST_STATUS_ENTITY_ID, []),
                    power_history.get(getattr(config, "THERMAL_POWER_ENTITY_ID", ""), []),
                ],
                (10, 30, 10),
            )
            power_history[config.POWER_CONSUMPTION_ENTITY_ID] = electrical
            defrost_history[config.DEFROST_STATUS_ENTITY_ID] = status
            thermal_entity = getattr(config, "THERMAL_POWER_ENTITY_ID", "")
            if thermal_entity:
                power_history[thermal_entity] = thermal

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
        if estimate["event_count"] < 5:
            self.state["defrost_power_history_last_error"] = (
                f"Only {estimate['event_count']} plausible defrost events"
            )
            self.state["defrost_power_history"] = estimate
            self.store.save(self.state)
            return
        self.state["defrost_power_history_last_error"] = None
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
        observation_entities = (
            config.HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID,
            config.HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID,
            config.HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID,
            config.HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID,
            config.HEAT_FORECAST_DHW_COUNTER_ENTITY_ID,
            config.HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID,
        )
        timeline = sorted(
            {
                stamp.astimezone(timezone.utc)
                for entity_id in observation_entities
                if entity_id
                for stamp, _ in history.get(entity_id, [])
            }
        )
        for entity_id, field in counter_map:
            if not entity_id:
                continue
            ordered = sorted(
                (
                    (stamp.astimezone(timezone.utc), value)
                    for stamp, value in history.get(entity_id, [])
                ),
                key=lambda item: item[0],
            )
            sample_times = [stamp for stamp, _ in ordered]
            measured_intervals: list[tuple[datetime, datetime]] = []
            for (previous_time, previous), (current_time, current) in zip(
                ordered, ordered[1:]
            ):
                elapsed = (current_time - previous_time).total_seconds() / 3600.0
                if elapsed <= 0 or elapsed > 2.0:
                    continue
                delta = compute_counter_delta(
                    previous,
                    current,
                    elapsed_hours=elapsed,
                    max_delta_kwh=3.0 if field == "dhw" else 1.0,
                )
                if delta is None:
                    continue
                measured_intervals.append((previous_time, current_time))
                _add_daily_energy_interval(
                    days,
                    previous_time=previous_time,
                    current_time=current_time,
                    delta=delta,
                    field=field,
                )
            measured_interval_ends = [
                interval_end for _, interval_end in measured_intervals
            ]
            for previous_time, current_time in zip(timeline, timeline[1:]):
                elapsed = (current_time - previous_time).total_seconds() / 3600.0
                if elapsed <= 0 or elapsed > 2.0:
                    continue
                previous_index = bisect_right(sample_times, previous_time) - 1
                current_index = bisect_left(sample_times, current_time)
                if previous_index < 0 or current_index >= len(ordered):
                    continue
                measured_index = bisect_left(measured_interval_ends, current_time)
                if (
                    measured_index < len(measured_intervals)
                    and measured_intervals[measured_index][0] <= previous_time
                ):
                    continue
                if ordered[previous_index][1] != ordered[current_index][1]:
                    continue
                _add_daily_energy_interval(
                    days,
                    previous_time=previous_time,
                    current_time=current_time,
                    delta=0.0,
                    field=field,
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
        current_date = starts[0].date()
        consumed_today = sum(
            sum(day["dhw"])
            for key, day in days.items()
            if key == current_date.isoformat()
        )
        dhw_remaining = max(0.0, dhw_day - consumed_today)
        dhw = []
        for start in starts:
            value = dhw_day * hourly_shape[start.hour]
            if start.date() == current_date:
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
        target_now = 22.6 if target_now is None else target_now
        indoor_now = target_now if indoor_now is None else indoor_now
        mode = config.get_climate_mode(
            ha_client.get_state(config.HEATING_STATUS_ENTITY_ID, states)
        )
        deltas = self._record_interval(
            now=now, states=states, outdoor_c=outdoor_now
        )
        if not config.HEAT_FORECAST_ENABLED:
            self._backfill(
                now=now,
                influx_service=influx_service,
                outdoor_entity_id=config.OUTDOOR_TEMP_ENTITY_ID,
                ha_client=ha_client,
            )
            self.store.save(self.state)
            return True

        horizon = max(1, min(48, int(config.HEAT_FORECAST_HORIZON_HOURS)))
        starts = forecast_hour_starts(now, horizon)
        remaining_hours = max(
            0.0,
            1.0
            - (
                now.minute * 60
                + now.second
                + now.microsecond / 1_000_000.0
            )
            / 3600.0,
        )
        temp_future = ha_client.get_hourly_forecast_strict(
            horizon - 1, starts=starts[1:]
        )
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
            elapsed_h = 0.0
            for hour in range(horizon):
                slot_h = remaining_hours if hour == 0 else 1.0
                if fireplace_on:
                    burn_h = max(0.0, min(slot_h, duration_h - elapsed_h))
                    decay_start = elapsed_h + burn_h
                    decay_h = max(0.0, slot_h - burn_h)
                    decay_energy = fireplace_kw * tau * (
                        math.exp(-max(0.0, decay_start - duration_h) / tau)
                        - math.exp(-max(0.0, decay_start + decay_h - duration_h) / tau)
                    )
                    energy = fireplace_kw * burn_h + decay_energy
                    profile.append(energy / slot_h if slot_h > 0 else 0.0)
                else:
                    profile.append(
                        fireplace_kw
                        * math.exp(-(hours_off + elapsed_h) / tau)
                    )
                elapsed_h += slot_h
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
            first_hour_fraction=remaining_hours,
        )
        if mode != "heating":
            heat_result["heat_kwh"] = [0.0] * horizon
        else:
            heat_result["heat_kwh"] = _smooth_future_heating_energy(
                heat_result["heat_kwh"]
            )

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
            cop_min=config.COP_MIN_VALUE,
            cop_max=config.COP_MAX_VALUE,
        )
        self._correct_current_slot(
            now=now,
            states=states,
            ha_client=ha_client,
            deltas=deltas,
            electrical=electrical,
            heat_result=heat_result,
            outdoor=outdoor,
            remaining_hours=remaining_hours,
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
                    first_hour_fraction=remaining_hours,
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
                    cop_min=config.COP_MIN_VALUE,
                    cop_max=config.COP_MAX_VALUE,
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
                "current_hour_estimate", "degraded_partial"
            ),
            "calibration_source": self.state.get(
                "calibration_source", "defaults"
            ),
            "calibration_hours": float(
                self.state.get("calibration_hours", 0.0)
            ),
            "prices_reused_hours": int(
                getattr(self, "_prices_reused_hours", 0)
            ),
            "price_source_per_hour": list(
                getattr(self, "_price_source_per_hour", ["none"] * horizon)
            ),
        }
        degraded_reasons = []
        if attrs["calibration_source"] == "defaults":
            degraded_reasons.append("calibration_history_unavailable")
        elif attrs["calibration_hours"] < int(
            getattr(config, "HEAT_FORECAST_MIN_CALIBRATION_HOURS", 48)
        ):
            degraded_reasons.append("insufficient_calibration_history")
        if attrs["current_hour_estimate"] in (
            "degraded_partial",
            "degraded_missing_counter_anchor",
            "degraded_counter_only",
        ):
            degraded_reasons.append("current_hour_counter_anchor_unavailable")
        if any(source == "none" for source in attrs["price_source_per_hour"]):
            degraded_reasons.append("price_data_unavailable")
        attrs["degraded_reasons"] = degraded_reasons
        if config.COP_DEFROST_HANDLING != "separate":
            attrs.pop("forecast_defrost_el_kwh", None)
        attrs["unit_of_measurement"] = "kWh"
        attrs["device_class"] = "energy"
        attrs["state_class"] = "measurement"
        attrs["friendly_name"] = "ML Heating WP Load Forecast"
        self.store.save(self.state)
        published = bool(
            ha_client.set_state(
                config.HEAT_FORECAST_SENSOR_ENTITY_ID,
                float(electrical["forecast_total_el_kwh"][0]),
                attrs,
                round_digits=4,
            )
        )
        self._backfill(
            now=now,
            influx_service=influx_service,
            outdoor_entity_id=config.OUTDOOR_TEMP_ENTITY_ID,
            ha_client=ha_client,
        )
        return published

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
        self._prices_reused_hours = 0
        self._price_source_per_hour = ["none"] * len(starts)
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
        hourly_prices = []
        if config.ELECTRICITY_PRICE_ENABLED or config.PRICE_COST_MODE_ENABLED:
            try:
                from .price_optimizer import PriceOptimizer

                if self._price_optimizer is None:
                    self._price_optimizer = PriceOptimizer()
                optimizer = self._price_optimizer
                optimizer.refresh_prices_if_needed(ha_client)
                hourly_prices = optimizer.get_hourly_grid_prices(
                    len(starts), starts[0]
                )
            except Exception as exc:
                LOGGER.warning("Could not refresh forecast prices: %s", exc)
                optimizer = None

        today_profile: dict[int, float] = {}
        if optimizer is not None:
            today = optimizer._to_local(starts[0]).date()
            today_profile = {
                optimizer._to_local(stamp).hour: price
                for stamp, price in optimizer._price_entries
                if optimizer._to_local(stamp).date() == today
            }
        daily_prices: dict[Any, list[float]] = {}
        price_sources = ["none"] * len(starts)
        reused_hours = 0
        missing_price_dates = set()
        resolved_prices: list[float | None] = []
        resolved_sources: list[str] = []
        synthetic_entries = []
        for index, start in enumerate(starts):
            price = hourly_prices[index] if index < len(hourly_prices) else None
            source = "actual" if price is not None else "none"
            local_start = optimizer._to_local(start) if optimizer is not None else start
            if (
                optimizer is not None
                and price is None
                and config.HEAT_FORECAST_REUSE_PRICES
                and local_start.date() > optimizer._to_local(starts[0]).date()
            ):
                price = today_profile.get(local_start.hour)
                if price is not None:
                    source = "reused_today"
                    reused_hours += 1
                    synthetic_entries.append(
                        (
                            local_start.replace(minute=0, second=0, microsecond=0),
                            price,
                        )
                    )
            resolved_prices.append(price)
            resolved_sources.append(source)
        cost_price_entries = (
            sorted(list(optimizer._price_entries) + synthetic_entries, key=lambda item: item[0])
            if optimizer is not None
            else []
        )
        for index, start in enumerate(starts):
            price = resolved_prices[index]
            source = resolved_sources[index]
            offset = 0.0
            cost_applied = False
            if optimizer is not None:
                try:
                    if config.PRICE_COST_MODE_ENABLED and price is not None:
                        original_entries = optimizer._price_entries
                        optimizer._price_entries = cost_price_entries
                        try:
                            cost = optimizer.compute_cost_target_offset(
                                list(pv_electrical[index:]),
                                float(config.HP_EXPECTED_POWER_FALLBACK_KW)
                                + max(
                                    0.0,
                                    float(
                                        config.PRICE_COST_HOUSEHOLD_POWER_OFFSET_KW
                                    ),
                                ),
                                now=start,
                            )
                        finally:
                            optimizer._price_entries = original_entries
                        if cost is not None:
                            offset = float(cost.get("offset", 0.0))
                            cost_applied = True
                    if (
                        not cost_applied
                        and config.ELECTRICITY_PRICE_ENABLED
                        and price is not None
                    ):
                        local_date = optimizer._to_local(start).date()
                        if local_date not in daily_prices:
                            daily_prices[local_date] = (
                                optimizer.get_today_prices(start)
                                or (
                                    optimizer.get_today_prices(starts[0])
                                    if source == "reused_today"
                                    else []
                                )
                            )
                        level = optimizer.classify_price(
                            price, daily_prices[local_date]
                        )
                        if daily_prices[local_date]:
                            offset = float(optimizer.get_target_offset(level))
                        else:
                            source = "none"
                            price = None
                            missing_price_dates.add(local_date)
                except Exception as exc:
                    LOGGER.warning(
                        "Could not resolve forecast price at %s: %s",
                        start.isoformat(),
                        exc,
                    )

            price_source = source
            state = get_forecast_trajectory_state(
                float(pv_electrical[index]),
                list(pv_electrical[index + 1 :]),
            )
            suppress_price = (
                not cost_applied
                and state.active
                and config.PV_TRAJ_FORECAST_MODE_ENABLED
                and not config.PRICE_COST_MODE_ENABLED
                and getattr(
                    config, "PV_TRAJ_DISABLE_PRICE_IN_FORECAST_MODE", True
                )
            )
            if suppress_price:
                offset = 0.0
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
            price_sources[index] = (
                "reused_today"
                if price is not None and price_source == "reused_today"
                else "actual"
                if price is not None
                else "none"
            )
        if missing_price_dates:
            LOGGER.info(
                "Forecast has no daily price profile for %s; using neutral offsets",
                ", ".join(sorted(day.isoformat() for day in missing_price_dates)),
            )
        self._prices_reused_hours = reused_hours
        self._price_source_per_hour = price_sources
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
        historical_counter_anchor = (
            isinstance(anchor, dict) and anchor.get("coverage") == "counter_anchor"
        )
        readings = anchor.get("readings", {}) if isinstance(anchor, dict) else {}
        initial_energy = (
            anchor.get("energy_so_far", {}) if isinstance(anchor, dict) else {}
        )
        energy_coverage = (
            anchor.get("energy_coverage", {}) if isinstance(anchor, dict) else {}
        )
        coverage_hours = (
            anchor.get("energy_coverage_hours", {}) if isinstance(anchor, dict) else {}
        )
        reading_times = (
            anchor.get("reading_times_utc", {}) if isinstance(anchor, dict) else {}
        )
        heat_el_so_far = float(initial_energy.get("heating_el", 0.0))
        heat_th_so_far = float(initial_energy.get("heating_th", 0.0))
        dhw_so_far = float(initial_energy.get("dhw_el", 0.0))
        standby_so_far = float(initial_energy.get("standby_el", 0.0))
        defrost_el_so_far = float(initial_energy.get("defrost_el", 0.0))
        defrost_th_so_far = float(initial_energy.get("defrost_th", 0.0))
        self.state["current_hour_estimate"] = "degraded_partial"
        heating_counter_delta_valid = bool(energy_coverage.get("heating_el", False))
        hour_start = None
        if isinstance(anchor, dict):
            try:
                hour_start = datetime.fromisoformat(anchor["hour_utc"]).astimezone(
                    timezone.utc
                )
                entity_ids = {
                    "heating_el": config.HEAT_FORECAST_ELECTRICITY_COUNTER_ENTITY_ID,
                    "heating_th": config.HEAT_FORECAST_THERMAL_COUNTER_ENTITY_ID,
                    "dhw_el": config.HEAT_FORECAST_DHW_COUNTER_ENTITY_ID,
                    "standby_el": config.HEAT_FORECAST_STANDBY_COUNTER_ENTITY_ID,
                    "defrost_el": config.HEAT_FORECAST_DEFROST_ELECTRICITY_COUNTER_ENTITY_ID,
                    "defrost_th": config.HEAT_FORECAST_DEFROST_THERMAL_COUNTER_ENTITY_ID,
                }
                for key, field in (
                    ("heating_el", "heating_el"),
                    ("heating_th", "heating_th"),
                    ("dhw_el", "dhw_el"),
                    ("standby_el", "standby_el"),
                    ("defrost_el", "defrost_el"),
                    ("defrost_th", "defrost_th"),
                ):
                    start_raw = reading_times.get(key) or anchor.get(
                        "sample_time_utc", anchor["hour_utc"]
                    )
                    interval_start = datetime.fromisoformat(start_raw).astimezone(
                        timezone.utc
                    )
                    current = _entity_value(
                        states, entity_ids[key]
                    )
                    interval_end = _entity_timestamp(
                        states, entity_ids[key], now
                    ).astimezone(timezone.utc)
                    elapsed = (
                        interval_end - interval_start
                    ).total_seconds() / 3600.0
                    if elapsed <= 0:
                        continue
                    past = compute_counter_delta(
                        readings.get(key),
                        current,
                        elapsed_hours=max(1e-6, elapsed),
                        max_gap_hours=1.1,
                        max_delta_kwh=(
                            10.0 if key in ("heating_th", "defrost_th") else 3.0
                        ),
                    )
                    if past is not None:
                        covered = max(
                            0.0,
                            (
                                min(interval_end, now.astimezone(timezone.utc))
                                - max(interval_start, hour_start)
                            ).total_seconds()
                            / 3600.0,
                        )
                        past *= min(1.0, covered / elapsed)
                        coverage_hours[key] = min(
                            1.0,
                            float(coverage_hours.get(key, 0.0)) + covered,
                        )
                        if field == "heating_el":
                            heat_el_so_far += past
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
                    self.state["current_hour_estimate"] = "counter_anchor"
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
                defrost_el_component_so_far = min(
                    defrost_el_component_so_far, heat_el_so_far
                )
                heat_el_component_so_far = max(
                    0.0, heat_el_so_far - defrost_el_so_far
                )
        else:
            if not defrost_el_in_counter:
                heat_el_component_so_far += defrost_el_so_far
        future_fraction = max(0.0, min(1.0, remaining_hours))
        heat_full = float(electrical["forecast_heat_el_kwh"][0])
        defrost_full = (
            float(electrical["forecast_defrost_el_kwh"][0])
            if config.COP_DEFROST_HANDLING == "separate"
            else 0.0
        )
        dhw_full = float(electrical["forecast_dhw_el_kwh"][0])
        standby_full = float(electrical["forecast_standby_el_kwh"][0])
        heat_remainder = max(0.0, heat_full)
        defrost_remainder = max(0.0, defrost_full)
        dhw_remainder = max(0.0, dhw_full) * future_fraction
        standby_remainder = max(0.0, standby_full) * future_fraction
        current_power_w = self._number(
            ha_client, states, config.POWER_CONSUMPTION_ENTITY_ID
        )
        if current_power_w is None:
            self.state["current_hour_estimate"] = (
                "counter_anchor"
                if historical_counter_anchor or heating_counter_delta_valid
                else "degraded_partial"
            )
        else:
            projected = max(0.0, current_power_w) / 1000.0 * future_fraction
            standby_remainder = min(projected, standby_remainder)
            active_projection = max(0.0, projected - standby_remainder)
            dhw_state = states.get(config.DHW_STATUS_ENTITY_ID)
            dhw_state = (
                dhw_state.get("state")
                if isinstance(dhw_state, Mapping)
                else dhw_state
            )
            dhw_active = (
                dhw_state is True
                or str(dhw_state).strip().lower() in ("on", "true", "1")
            )
            defrost_entity = getattr(config, "DEFROST_STATUS_ENTITY_ID", "")
            defrost_state = states.get(defrost_entity) if defrost_entity else None
            defrost_state = (
                defrost_state.get("state")
                if isinstance(defrost_state, Mapping)
                else defrost_state
            )
            defrost_active = (
                defrost_state is True
                or str(defrost_state).strip().lower() in ("on", "true", "1")
            )
            if dhw_active:
                dhw_remainder = active_projection
                heat_remainder = 0.0
                defrost_remainder = 0.0
            elif defrost_active and config.COP_DEFROST_HANDLING == "separate":
                dhw_remainder = 0.0
                heat_remainder = 0.0
                defrost_remainder = active_projection
            else:
                dhw_remainder = 0.0
                modeled_active = heat_remainder + defrost_remainder
                if modeled_active > 0:
                    heat_remainder = active_projection * heat_remainder / modeled_active
                    defrost_remainder = active_projection * defrost_remainder / modeled_active
                else:
                    heat_remainder = active_projection
                    defrost_remainder = 0.0
            if historical_counter_anchor:
                self.state["current_hour_estimate"] = "counter_anchor"
            elif self.state["current_hour_estimate"] not in (
                "degraded_partial",
                "degraded_missing_counter_anchor",
            ):
                self.state["current_hour_estimate"] = "counter_plus_power_projection"
        heat_total = round(heat_el_component_so_far + heat_remainder, 4)
        dhw_total = round(dhw_so_far + dhw_remainder, 4)
        standby_total = round(standby_so_far + standby_remainder, 4)
        electrical["forecast_heat_el_kwh"][0] = heat_total
        electrical["forecast_dhw_el_kwh"][0] = dhw_total
        electrical["forecast_standby_el_kwh"][0] = standby_total
        component_totals = [heat_total, dhw_total, standby_total]
        if config.COP_DEFROST_HANDLING == "separate":
            defrost_total = round(
                defrost_el_component_so_far + defrost_remainder, 4
            )
            electrical["forecast_defrost_el_kwh"][0] = defrost_total
            component_totals.append(defrost_total)
        total = round(sum(component_totals), 4)
        electrical["forecast_total_el_kwh"][0] = total
        electrical["forecast"][0]["value"] = total
        electrical["forecast_heat_th_kwh"][0] = round(
            max(0.0, heat_th_so_far - defrost_th_so_far)
            + float(heat_result["heat_kwh"][0]),
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


def cop_calibration_energies(
    *,
    heat_kwh: float,
    electricity_kwh: float,
    defrost_heat_kwh: float,
    defrost_electricity_kwh: float,
    defrost_el_in_heating_counter: bool,
    defrost_handling: str,
) -> tuple[float, float]:
    """Return one of the valid net or separate COP energy pairings."""
    heat_net = max(0.0, float(heat_kwh) - float(defrost_heat_kwh))
    electricity = float(electricity_kwh)
    defrost_electricity = float(defrost_electricity_kwh)
    if not all(
        math.isfinite(value) and value >= 0
        for value in (heat_kwh, electricity, defrost_heat_kwh, defrost_electricity)
    ):
        raise ValueError("Energy values must be finite and non-negative")
    if defrost_handling == "separate":
        electricity_total = (
            max(0.0, electricity - defrost_electricity)
            if defrost_el_in_heating_counter
            else electricity
        )
    elif defrost_handling == "net":
        electricity_total = (
            electricity
            if defrost_el_in_heating_counter
            else electricity + defrost_electricity
        )
    else:
        raise ValueError("Defrost handling must be 'net' or 'separate'")
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
    first_hour_fraction: float = 1.0,
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
    if (
        not math.isfinite(first_hour_fraction)
        or not 0.0 <= first_hour_fraction <= 1.0
    ):
        raise ValueError("first_hour_fraction must be between zero and one")

    step_hours = step_minutes / 60.0
    room = float(room_c)
    slab = float(slab_c)
    heat_kwh: list[float] = []
    hourly_max_kw: list[float] = []

    def advance(
        q_hp: float, outdoor: float, gains: float, dt_hours: float
    ) -> tuple[float, float]:
        slab_midpoint = slab + dt_hours * q_hp / (2.0 * slab_capacity_kwh_k)
        transfer_kw = slab_to_room_kw_k * (slab_midpoint - room)
        next_room = room + dt_hours * (
            transfer_kw + gains - heat_loss_kw_k * (room - outdoor)
        ) / room_capacity_kwh_k
        next_slab = slab + dt_hours * (q_hp - transfer_kw) / slab_capacity_kwh_k
        return next_room, next_slab

    for hour in range(length):
        energy = 0.0
        max_kw_this_hour = 0.0
        remaining = first_hour_fraction if hour == 0 else 1.0
        while remaining > 1e-12:
            dt = min(step_hours, remaining)
            target_floor = float(target_c[hour]) - target_band_k
            gains = float(pv_gain_kw[hour]) + float(fireplace_kw[hour]) + base_gain_kw

            def projected_room(q_hp: float) -> float:
                next_room, _ = advance(q_hp, float(outdoor_c[hour]), gains, dt)
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

            room, slab = advance(q_hp, float(outdoor_c[hour]), gains, dt)
            energy += q_hp * dt
            max_kw_this_hour = max(max_kw_this_hour, q_hp)
            remaining -= dt
        heat_kwh.append(energy)
        hourly_max_kw.append(max_kw_this_hour)

    return {
        "heat_kwh": heat_kwh,
        "max_hp_kw": max(hourly_max_kw, default=0.0),
        "hourly_max_hp_kw": hourly_max_kw,
        "room_end_c": room,
        "slab_end_c": slab,
    }
