"""Read-only historical target-hold energy counterfactual."""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_service import _find_cooling_state_file, _find_state_file
from src import config
from src.building_gain_window import external_gain_kw
from src.counterfactual_replay import (
    CLIMATE_MODE_ALIASES,
    MAX_VALID_COP,
    MIN_VALID_COP,
    replay_target_hold,
)
from src.ha_client import create_ha_client
from src.thermal_equilibrium_model import ThermalEquilibriumModel


_HISTORY_TAIL_HOURS = 24
_HISTORY_INTERVAL_MINUTES = 5
_MAX_PERIOD_DAYS = 14
_LOG = logging.getLogger(__name__)
_STATE_ALIASES = CLIMATE_MODE_ALIASES
_NUMERIC_ENTITIES = {
    "indoor_temp": config.INDOOR_TEMP_ENTITY_ID,
    "outdoor_temp": config.OUTDOOR_TEMP_ENTITY_ID,
    "target_heating": config.TARGET_INDOOR_TEMP_ENTITY_ID,
    "target_cooling": (
        getattr(config, "TARGET_INDOOR_TEMP_COOLING_ENTITY_ID", "")
        or config.TARGET_INDOOR_TEMP_ENTITY_ID
    ),
    "outlet_temp": config.ACTUAL_OUTLET_TEMP_ENTITY_ID,
    "inlet_temp": config.INLET_TEMP_ENTITY_ID,
    "flow_rate": config.FLOW_RATE_ENTITY_ID,
    "electrical_power_w": config.POWER_CONSUMPTION_ENTITY_ID,
    "pv_power": config.PV_POWER_ENTITY_ID,
    "fireplace_on": config.FIREPLACE_STATUS_ENTITY_ID,
    "tv_on": config.TV_STATUS_ENTITY_ID,
}


def _parse_number(state: Any) -> float:
    try:
        value = float(state)
        return value if math.isfinite(value) else float("nan")
    except (TypeError, ValueError):
        if str(state).strip().lower() in {"on", "true"}:
            return 1.0
        if str(state).strip().lower() in {"off", "false"}:
            return 0.0
        return float("nan")


def _parse_history_timestamp(record: dict[str, Any]) -> pd.Timestamp | None:
    timestamp = record.get("last_changed") or record.get("last_updated")
    if not timestamp:
        return None
    try:
        stamp = pd.Timestamp(timestamp)
        return (
            stamp.tz_localize("UTC")
            if stamp.tzinfo is None
            else stamp.tz_convert("UTC")
        )
    except (TypeError, ValueError):
        return None


def _build_history_frame(
    raw: list[list[dict[str, Any]]],
    entity_ids: list[str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    if len(raw) != len(entity_ids):
        return pd.DataFrame()

    interval = f"{_HISTORY_INTERVAL_MINUTES}min"
    start_utc = pd.Timestamp(start).tz_convert("UTC").floor(interval)
    end_utc = pd.Timestamp(end).tz_convert("UTC").ceil(interval)
    index = pd.date_range(start_utc, end_utc, freq=interval, tz="UTC")
    by_entity: dict[str, list[str]] = {}
    for name, entity_id in _NUMERIC_ENTITIES.items():
        if entity_id:
            by_entity.setdefault(entity_id, []).append(name)
    mode_entity = config.HEATING_STATUS_ENTITY_ID

    histories = dict(zip(entity_ids, raw))
    columns: dict[str, pd.Series] = {}
    for entity_id, names in by_entity.items():
        events: dict[pd.Timestamp, Any] = {}
        for record in histories.get(entity_id, []):
            stamp = _parse_history_timestamp(record)
            if stamp is None:
                continue
            value = _parse_number(record.get("state"))
            events[stamp] = value if np.isfinite(value) else "__invalid__"
        series = pd.Series(events, dtype=object).sort_index()
        aligned = series.reindex(index.union(series.index)).sort_index().ffill()
        aligned = aligned.reindex(index).replace("__invalid__", np.nan).astype(float)
        for name in names:
            columns[name] = aligned

    mode_events: dict[pd.Timestamp, str] = {}
    for record in histories.get(mode_entity, []):
        stamp = _parse_history_timestamp(record)
        if stamp is None:
            continue
        mode_events[stamp] = _STATE_ALIASES.get(
            str(record.get("state", "")).strip().lower(), ""
        )
    mode_series = pd.Series(mode_events, dtype=object).sort_index()
    mode_aligned = mode_series.reindex(index.union(mode_series.index)).sort_index().ffill()
    columns["mode"] = mode_aligned.reindex(index)
    frame = pd.DataFrame(columns, index=index)
    frame.index.name = "_time"
    frame.reset_index(inplace=True)
    frame["target_temp"] = np.where(
        frame["mode"] == "cooling",
        frame["target_cooling"],
        frame["target_heating"],
    )
    frame["thermal_power_kw"] = (
        (frame["flow_rate"] / 60.0)
        * config.SPECIFIC_HEAT_CAPACITY
        * (frame["outlet_temp"] - frame["inlet_temp"])
    )
    return frame


def _read_state(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as state_file:
            state = json.load(state_file)
        baseline = state.get("baseline_parameters", {})
        adjustments = state.get("learning_state", {}).get(
            "parameter_adjustments", {}
        )
        effective = {
            key: value + adjustments.get(f"{key}_delta", 0.0)
            for key, value in baseline.items()
            if isinstance(value, (int, float))
        }
        channels = state.get("learning_state", {}).get("heat_source_channels", {})
        for channel, parameter in (
            ("pv", "pv_heat_weight"),
            ("fireplace", "fireplace_heat_weight"),
            ("tv", "tv_heat_weight"),
        ):
            channel_parameters = channels.get(channel, {}).get("parameters", {})
            if parameter in channel_parameters:
                channel_value = channel_parameters[parameter]
                if isinstance(channel_value, (int, float)):
                    effective[parameter] = channel_value
        return effective
    except (OSError, ValueError, TypeError) as exc:
        _LOG.warning(
            "Unable to read counterfactual thermal state from %s: %s", path, exc
        )
        return {}


def _make_model(parameters: dict[str, Any]) -> ThermalEquilibriumModel:
    model = ThermalEquilibriumModel()
    model.orchestrator = None
    for attribute in (
        "thermal_time_constant",
        "heat_loss_coefficient",
        "outlet_effectiveness",
        "slab_time_constant_hours",
        "solar_lag_minutes",
    ):
        if attribute in parameters:
            setattr(model, attribute, float(parameters[attribute]))
    model.external_source_weights = {
        "pv": float(
            parameters.get("pv_heat_weight", model.external_source_weights["pv"])
        ),
        "fireplace": float(
            parameters.get(
                "fireplace_heat_weight", model.external_source_weights["fireplace"]
            )
        ),
        "tv": float(
            parameters.get("tv_heat_weight", model.external_source_weights["tv"])
        ),
    }
    return model


def _history_entity_ids() -> list[str]:
    entities = list(dict.fromkeys(
        [entity_id for entity_id in _NUMERIC_ENTITIES.values() if entity_id]
        + [config.HEATING_STATUS_ENTITY_ID]
    ))
    return entities


def _fetch_history(start: datetime, end: datetime) -> pd.DataFrame:
    client = create_ha_client()
    entities = _history_entity_ids()
    raw = client.get_history_bulk(entities, start, end)
    if raw is None:
        return pd.DataFrame()
    return _build_history_frame(raw, entities, start, end)


def _external_gain_kw(
    *,
    model: ThermalEquilibriumModel,
    pv_values: list[float],
    position: int,
    outlet_temp: float,
    outdoor_temp: float,
    indoor_temp: float,
    fireplace_on: float,
    tv_on: float,
) -> float:
    history_step = max(
        1, int(round(config.HISTORY_STEP_MINUTES / _HISTORY_INTERVAL_MINUTES))
    )
    pv_history = list(reversed(pv_values[position::-history_step]))
    return external_gain_kw(
        model,
        pv_history,
        outdoor_temp,
        fireplace_on,
        tv_on,
        outlet_temp=outlet_temp,
        indoor_temp=indoor_temp,
    )


def _validate_history(history: pd.DataFrame) -> str | None:
    if history.empty:
        return "Home Assistant returned no usable history."
    required_history = tuple(_NUMERIC_ENTITIES) + ("mode",)
    missing_inputs = [
        column for column in required_history if column not in history
    ]
    if missing_inputs:
        return "Required history is missing: " + ", ".join(missing_inputs)
    invalid_rows = history.loc[:, required_history].isna().any(axis=1)
    for column in required_history:
        if column != "mode":
            invalid_rows |= ~np.isfinite(
                pd.to_numeric(history[column], errors="coerce")
            )
    if invalid_rows.any():
        return (
            "Synchronized sensor history is incomplete or invalid "
            f"({int(invalid_rows.sum())} intervals)."
        )
    return None


def _has_valid_model_parameters(parameters: dict[str, Any]) -> bool:
    for key in ("heat_loss_coefficient", "thermal_time_constant"):
        try:
            if key not in parameters or not np.isfinite(float(parameters[key])):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _resolve_mode_parameters(
    history: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
    heating_parameters = _read_state(_find_state_file())
    cooling_parameters = _read_state(_find_cooling_state_file())
    has_cooling_history = history["mode"].isin(["cooling", "cool"]).any()
    if not _has_valid_model_parameters(heating_parameters):
        return None, "Heating model parameters are unavailable."
    if not has_cooling_history and not _has_valid_model_parameters(
        cooling_parameters
    ):
        cooling_parameters = heating_parameters
    mode_parameters = {
        "heating": heating_parameters,
        "cooling": cooling_parameters,
    }
    for mode, parameters in mode_parameters.items():
        if not _has_valid_model_parameters(parameters):
            return None, f"{mode.title()} model parameters are unavailable."
    return mode_parameters, None


def _compute_external_gains(
    history: pd.DataFrame,
    mode_parameters: dict[str, dict[str, Any]],
) -> list[float]:
    models = {mode: _make_model(parameters) for mode, parameters in mode_parameters.items()}
    pv_values = history["pv_power"].astype(float).tolist()
    external_gains = []
    last_mode = "heating"
    external_gain_columns = (
        "mode",
        "outlet_temp",
        "outdoor_temp",
        "indoor_temp",
        "fireplace_on",
        "tv_on",
    )
    for position, values in enumerate(
        history.loc[:, external_gain_columns].itertuples(index=False, name=None)
    ):
        (
            raw_mode,
            outlet_temp,
            outdoor_temp,
            indoor_temp,
            fireplace_on,
            tv_on,
        ) = values
        mode = CLIMATE_MODE_ALIASES.get(str(raw_mode).strip().lower(), "")
        if mode == "heating":
            last_mode = "heating"
        elif mode == "cooling":
            last_mode = "cooling"
        model = models.get(last_mode, models["heating"])
        external_gains.append(
            _external_gain_kw(
                model=model,
                pv_values=pv_values,
                position=position,
                outlet_temp=outlet_temp,
                outdoor_temp=outdoor_temp,
                indoor_temp=indoor_temp,
                fireplace_on=fireplace_on,
                tv_on=tv_on,
            )
        )
    return external_gains


def _trim_and_resum(
    result: dict[str, Any], selected_start: datetime, end: datetime
) -> dict[str, Any]:
    if not result.get("complete"):
        return result
    intervals = result["intervals"]
    intervals = intervals.loc[
        (intervals["_time"] >= pd.Timestamp(selected_start))
        & (intervals["_time"] < pd.Timestamp(end))
    ].reset_index(drop=True)
    if intervals.empty:
        return {
            "complete": False,
            "reason": "No history was available in the selected date range.",
        }
    result["intervals"] = intervals
    for result_key, interval_key in (
        ("actual_thermal_kwh", "actual_thermal_energy_kwh"),
        ("counterfactual_thermal_kwh", "counterfactual_thermal_energy_kwh"),
        ("actual_electrical_kwh", "actual_electrical_energy_kwh"),
        ("counterfactual_electrical_kwh", "counterfactual_electrical_energy_kwh"),
    ):
        result[result_key] = float(intervals[interval_key].sum())
    result["thermal_energy_difference_kwh"] = (
        result["actual_thermal_kwh"] - result["counterfactual_thermal_kwh"]
    )
    result["electrical_energy_difference_kwh"] = (
        result["actual_electrical_kwh"] - result["counterfactual_electrical_kwh"]
    )
    result["sample_count"] = len(intervals)
    result["measured_cop_count"] = int(intervals["cop_source"].eq("measured").sum())
    result["estimated_cop_count"] = int(
        intervals["cop_source"].eq("estimated fallback").sum()
    )
    return result


def _run_counterfactual(
    start_date: date, end_date: date, fallback_cop: float
) -> dict[str, Any]:
    if start_date > end_date:
        return {"complete": False, "reason": "Start date must not be after end date."}
    if (end_date - start_date).days > _MAX_PERIOD_DAYS:
        return {
            "complete": False,
            "reason": f"Select a period of {_MAX_PERIOD_DAYS} days or less.",
        }
    selected_start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    selected_end = datetime.combine(
        end_date + timedelta(days=1), time.min, tzinfo=timezone.utc
    )
    end = min(selected_end, datetime.now(timezone.utc))
    if end <= selected_start:
        return {
            "complete": False,
            "reason": "The selected period is in the future; no history can be fetched.",
        }
    fetch_start = selected_start - timedelta(hours=_HISTORY_TAIL_HOURS)
    history = _fetch_history(fetch_start, end)
    validation_error = _validate_history(history)
    if validation_error:
        return {"complete": False, "reason": validation_error}

    history.reset_index(drop=True, inplace=True)
    mode_parameters, parameter_error = _resolve_mode_parameters(history)
    if parameter_error:
        return {"complete": False, "reason": parameter_error}
    history["external_gain_kw"] = _compute_external_gains(history, mode_parameters)

    result = replay_target_hold(
        history,
        parameters_by_mode=mode_parameters,
        specific_heat_capacity=float(config.SPECIFIC_HEAT_CAPACITY),
        fallback_cop=fallback_cop,
    )
    return _trim_and_resum(result, selected_start, end)


def render_counterfactual() -> None:
    st.header("🧪 Target-Hold Energy Replay")
    st.warning(
        "Experimental, read-only estimate. It does not change live control or "
        "learning. Effective thermal storage is approximated from the model's "
        "heat-loss coefficient and thermal time constant; slab temperature is "
        "not directly measured by this replay."
    )
    st.caption(
        "Uses recorded targets, indoor/outdoor temperatures, climate mode, "
        "PV/internal heat sources, hydronic sensors, and electrical power. "
        "Missing or invalid history makes the estimate incomplete."
    )

    today = datetime.now(timezone.utc).date()
    selected_dates = st.date_input(
        "Historical period (UTC)",
        value=(today - timedelta(days=1), today),
        max_value=today,
        key="counterfactual_period",
    )
    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.info("Select both a start and end date to calculate a replay.")
        return
    start_date, end_date = selected_dates
    fallback_cop = st.number_input(
        "Fallback COP when a valid measured COP is unavailable",
        min_value=MIN_VALID_COP,
        max_value=MAX_VALID_COP,
        value=3.0,
        step=0.1,
        key="counterfactual_fallback_cop",
    )
    if start_date > end_date:
        st.error("Start date must not be after end date.")
        return
    if (end_date - start_date).days > _MAX_PERIOD_DAYS:
        st.error(f"Select a period of {_MAX_PERIOD_DAYS} days or less.")
        return

    if not st.button("Calculate target-hold replay", type="primary"):
        return
    with st.spinner("Fetching synchronized Home Assistant history and replaying..."):
        try:
            result = _run_counterfactual(start_date, end_date, fallback_cop)
        except Exception:
            _LOG.exception("Target-hold replay failed unexpectedly.")
            st.error("Replay could not be calculated due to an unexpected error.")
            return

    if not result.get("complete"):
        st.error(f"Estimate incomplete: {result.get('reason', 'Unknown issue')}")
        if result.get("missing_count"):
            st.caption(f"Invalid/missing intervals: {result['missing_count']}")
        return

    intervals = result["intervals"]
    if intervals.empty:
        st.error("No intervals were available inside the selected period.")
        return

    st.caption(
        f"Coverage: {result['sample_count']} synchronized intervals. "
        f"COP: {result['cop_estimation_method']}."
    )
    st.caption(
        "Effective heat capacity uses mode-specific values: "
        + ", ".join(
            f"{mode}: {capacity:.3f} kWh/K"
            for mode, capacity in result[
                "effective_heat_capacity_kwh_per_k_by_mode"
            ].items()
        )
        + "."
    )
    columns = st.columns(3)
    columns[0].metric(
        "Counterfactual thermal energy",
        f"{result['counterfactual_thermal_kwh']:.2f} kWh",
    )
    columns[1].metric(
        "Counterfactual electrical energy",
        f"{result['counterfactual_electrical_kwh']:.2f} kWh",
    )
    columns[2].metric(
        "Actual − counterfactual electricity",
        f"{result['electrical_energy_difference_kwh']:+.2f} kWh",
        help="A positive result means recorded electrical energy exceeded the estimate; it is not a guaranteed savings figure.",
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=intervals["_time"],
            y=intervals["actual_indoor_temp"],
            mode="lines",
            name="Actual indoor temperature",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=intervals["_time"],
            y=intervals["target_temp"],
            mode="lines",
            name="Historical target",
            line={"dash": "dash"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=intervals["_time"],
            y=intervals["counterfactual_indoor_temp"],
            mode="lines",
            name="Target-hold replay",
        )
    )
    figure.update_layout(
        title="Actual and counterfactual indoor temperature",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        height=450,
    )
    st.plotly_chart(figure, width="stretch")
    st.dataframe(
        intervals[
            [
                "_time",
                "climate_mode",
                "actual_indoor_temp",
                "target_temp",
                "counterfactual_indoor_temp",
                "actual_thermal_energy_kwh",
                "counterfactual_thermal_energy_kwh",
                "actual_electrical_energy_kwh",
                "counterfactual_electrical_energy_kwh",
                "cop_source",
            ]
        ],
        width="stretch",
    )
