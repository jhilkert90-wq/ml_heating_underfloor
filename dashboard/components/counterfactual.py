"""Read-only historical target-hold energy counterfactual."""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_service import _find_cooling_state_file, _find_state_file
from src import config
from src.counterfactual_replay import replay_target_hold
from src.ha_client import create_ha_client
from src.thermal_equilibrium_model import ThermalEquilibriumModel


_HISTORY_TAIL_HOURS = 24
_STATE_ALIASES = {
    "heat": "heating",
    "heating": "heating",
    "cool": "cooling",
    "cooling": "cooling",
    "off": "off",
    "idle": "off",
}
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


def _build_history_frame(
    raw: list[list[dict[str, Any]]],
    entity_ids: list[str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    if len(raw) != len(entity_ids):
        return pd.DataFrame()

    start_utc = pd.Timestamp(start).tz_convert("UTC").floor("5min")
    end_utc = pd.Timestamp(end).tz_convert("UTC").ceil("5min")
    index = pd.date_range(start_utc, end_utc, freq="5min", tz="UTC")
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
            timestamp = record.get("last_changed") or record.get("last_updated")
            if not timestamp:
                continue
            try:
                stamp = pd.Timestamp(timestamp)
                stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
            except (TypeError, ValueError):
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
        timestamp = record.get("last_changed") or record.get("last_updated")
        if not timestamp:
            continue
        try:
            stamp = pd.Timestamp(timestamp)
            stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        except (TypeError, ValueError):
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
        import json

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
                return_value = channel_parameters[parameter]
                if isinstance(return_value, (int, float)):
                    effective[parameter] = return_value
        return effective
    except (OSError, ValueError, TypeError):
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
        "pv": float(parameters.get("pv_heat_weight", model.pv_heat_weight)),
        "fireplace": float(
            parameters.get("fireplace_heat_weight", model.fireplace_heat_weight)
        ),
        "tv": float(parameters.get("tv_heat_weight", model.tv_heat_weight)),
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
    row: pd.Series,
    model: ThermalEquilibriumModel,
    pv_values: list[float],
    position: int,
) -> float:
    interval_minutes = 5
    history_step = max(1, int(round(config.HISTORY_STEP_MINUTES / interval_minutes)))
    pv_history = list(reversed(pv_values[position::-history_step]))
    equilibrium = model.predict_equilibrium_temperature(
        outlet_temp=float(row["outlet_temp"]),
        outdoor_temp=float(row["outdoor_temp"]),
        current_indoor=float(row["indoor_temp"]),
        pv_power=pv_history,
        fireplace_on=float(row["fireplace_on"]),
        tv_on=float(row["tv_on"]),
        thermal_power=0.0,
        _suppress_logging=True,
    )
    return (float(equilibrium) - float(row["outdoor_temp"])) * float(
        model.heat_loss_coefficient
    )


def _run_counterfactual(start_date: date, end_date: date, fallback_cop: float) -> dict[str, Any]:
    selected_start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    selected_end = datetime.combine(
        end_date + timedelta(days=1), time.min, tzinfo=timezone.utc
    )
    end = min(selected_end, datetime.now(timezone.utc))
    fetch_start = selected_start - timedelta(hours=_HISTORY_TAIL_HOURS)
    history = _fetch_history(fetch_start, end)
    if history.empty:
        return {"complete": False, "reason": "Home Assistant returned no usable history."}
    required_history = (
        "indoor_temp",
        "outdoor_temp",
        "target_heating",
        "target_cooling",
        "outlet_temp",
        "inlet_temp",
        "flow_rate",
        "electrical_power_w",
        "pv_power",
        "fireplace_on",
        "tv_on",
        "mode",
    )
    missing_inputs = [
        column for column in required_history if column not in history
    ]
    if missing_inputs:
        return {
            "complete": False,
            "reason": "Required history is missing: " + ", ".join(missing_inputs),
        }
    invalid_rows = history.loc[:, required_history].isna().any(axis=1)
    for column in required_history:
        if column != "mode":
            invalid_rows |= ~np.isfinite(
                pd.to_numeric(history[column], errors="coerce")
            )
    if invalid_rows.any():
        return {
            "complete": False,
            "reason": (
                "Synchronized sensor history is incomplete or invalid "
                f"({int(invalid_rows.sum())} intervals)."
            ),
        }

    history.reset_index(drop=True, inplace=True)
    heating_parameters = _read_state(_find_state_file())
    cooling_parameters = _read_state(_find_cooling_state_file())
    has_cooling_history = history["mode"].isin(["cooling", "cool"]).any()
    if not cooling_parameters and not has_cooling_history:
        cooling_parameters = heating_parameters
    mode_parameters = {
        "heating": heating_parameters,
        "cooling": cooling_parameters,
    }
    for mode, parameters in mode_parameters.items():
        if mode == "cooling" and not has_cooling_history and not parameters:
            continue
        for key in ("heat_loss_coefficient", "thermal_time_constant"):
            if key not in parameters or not np.isfinite(float(parameters[key])):
                return {
                    "complete": False,
                    "reason": f"{mode.title()} model parameter {key} is unavailable.",
                }

    models = {mode: _make_model(parameters) for mode, parameters in mode_parameters.items()}
    pv_values = history["pv_power"].astype(float).tolist()
    external_gains = []
    last_mode = "heating"
    for position, row in history.iterrows():
        mode = str(row["mode"]).strip().lower()
        if mode in {"heat", "heating"}:
            last_mode = "heating"
        elif mode in {"cool", "cooling"}:
            last_mode = "cooling"
        model = models.get(last_mode, models["heating"])
        external_gains.append(
            _external_gain_kw(row, model, pv_values, position)
        )
    history["external_gain_kw"] = external_gains

    result = replay_target_hold(
        history,
        parameters_by_mode=mode_parameters,
        specific_heat_capacity=float(config.SPECIFIC_HEAT_CAPACITY),
        fallback_cop=fallback_cop,
    )
    if result.get("complete"):
        intervals = result["intervals"]
        intervals = intervals.loc[
            intervals["_time"] >= pd.Timestamp(selected_start)
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
            (
                "counterfactual_electrical_kwh",
                "counterfactual_electrical_energy_kwh",
            ),
        ):
            result[result_key] = float(intervals[interval_key].sum())
        result["thermal_energy_difference_kwh"] = (
            result["actual_thermal_kwh"] - result["counterfactual_thermal_kwh"]
        )
        result["electrical_energy_difference_kwh"] = (
            result["actual_electrical_kwh"]
            - result["counterfactual_electrical_kwh"]
        )
        result["sample_count"] = len(intervals)
        result["measured_cop_count"] = int(
            intervals["cop_source"].eq("measured").sum()
        )
        result["estimated_cop_count"] = int(
            intervals["cop_source"].eq("estimated fallback").sum()
        )
    return result


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
    start_date, end_date = st.date_input(
        "Historical period (UTC)",
        value=(today - timedelta(days=1), today),
        max_value=today,
        key="counterfactual_period",
    )
    fallback_cop = st.number_input(
        "Fallback COP when a valid measured COP is unavailable",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
        key="counterfactual_fallback_cop",
    )
    if start_date > end_date:
        st.error("Start date must not be after end date.")
        return
    if (end_date - start_date).days > 14:
        st.error("Select a period of 14 days or less.")
        return

    if not st.button("Calculate target-hold replay", type="primary"):
        return
    with st.spinner("Fetching synchronized Home Assistant history and replaying..."):
        try:
            result = _run_counterfactual(start_date, end_date, fallback_cop)
        except Exception as exc:
            st.error(f"Replay could not be calculated: {exc}")
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
        use_container_width=True,
    )
