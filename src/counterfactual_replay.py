"""Informational replay of historical target-hold thermal energy demand."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .building_capacity import room_capacity_kwh_per_k


REQUIRED_COLUMNS = (
    "_time",
    "indoor_temp",
    "target_temp",
    "outdoor_temp",
    "external_gain_kw",
    "mode",
    "thermal_power_kw",
    "electrical_power_w",
)
MAX_HISTORY_GAP_HOURS = 0.25
MIN_COP_ELECTRICAL_POWER_KW = 0.1
MIN_VALID_COP = 1.0
MAX_VALID_COP = 10.0
CLIMATE_MODE_ALIASES = {
    "auto": "heating",
    "heat": "heating",
    "heating": "heating",
    "cool": "cooling",
    "cooling": "cooling",
    "off": "off",
    "idle": "off",
}


def _incomplete(reason: str, missing_count: int = 0) -> Dict[str, Any]:
    return {
        "complete": False,
        "reason": reason,
        "missing_count": missing_count,
        "intervals": pd.DataFrame(),
    }


def replay_target_hold(
    history: pd.DataFrame,
    *,
    heat_loss_coefficient: float | None = None,
    thermal_time_constant_hours: float | None = None,
    parameters_by_mode: Dict[str, Dict[str, float]] | None = None,
    specific_heat_capacity: float,
    fallback_cop: float = 3.0,
) -> Dict[str, Any]:
    """Estimate required HP energy to follow historical targets.

    A one-node RC model carries the initial indoor temperature forward and
    estimates effective heat capacity as tau x (outlet effectiveness + HLC).
    This captures a room already above target without pretending the temperature
    was actually at target. It is an informational approximation, not a
    measured slab-temperature model or a live-control input.
    """
    if not isinstance(history, pd.DataFrame) or history.empty:
        return _incomplete("No historical rows were supplied.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in history]
    if missing_columns:
        return _incomplete(
            "Required history is missing: " + ", ".join(missing_columns)
        )

    try:
        specific_heat = float(specific_heat_capacity)
        estimated_cop = float(fallback_cop)
    except (TypeError, ValueError):
        return _incomplete("Model parameters or fallback COP are invalid.")
    if (
        not all(np.isfinite(value) for value in (specific_heat, estimated_cop))
        or specific_heat <= 0
        or not MIN_VALID_COP <= estimated_cop <= MAX_VALID_COP
    ):
        return _incomplete("Model parameters or fallback COP are outside valid bounds.")

    if parameters_by_mode is None:
        parameters_by_mode = {
            "heating": {
                "heat_loss_coefficient": heat_loss_coefficient,
                "thermal_time_constant": thermal_time_constant_hours,
            },
            "cooling": {
                "heat_loss_coefficient": heat_loss_coefficient,
                "thermal_time_constant": thermal_time_constant_hours,
            },
        }
    mode_parameters = {}
    capacity_by_mode = {}
    try:
        for mode in ("heating", "cooling"):
            parameters = parameters_by_mode[mode]
            mode_hlc = float(parameters["heat_loss_coefficient"])
            mode_tau = float(parameters["thermal_time_constant"])
            if (
                not np.isfinite(mode_hlc)
                or not np.isfinite(mode_tau)
                or mode_hlc <= 0
                or mode_tau <= 0
            ):
                return _incomplete(f"Invalid {mode} thermal parameters.")
            capacity = room_capacity_kwh_per_k(
                {
                    "thermal_time_constant": mode_tau,
                    "heat_loss_coefficient": mode_hlc,
                    "outlet_effectiveness": parameters.get(
                        "outlet_effectiveness"
                    ),
                }
            )
            capacity_by_mode[mode] = capacity
            # Direct power input relaxes with C/HLC, not with the model tau.
            mode_parameters[mode] = (mode_hlc, capacity / mode_hlc)
    except (KeyError, TypeError, ValueError):
        return _incomplete("Heating and cooling thermal parameters are required.")

    frame = history.loc[:, REQUIRED_COLUMNS].copy()
    frame["_time"] = pd.to_datetime(frame["_time"], utc=True, errors="coerce")
    numeric_columns = (
        "indoor_temp",
        "target_temp",
        "outdoor_temp",
        "external_gain_kw",
        "thermal_power_kw",
        "electrical_power_w",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    missing_mask = frame["_time"].isna()
    for column in numeric_columns:
        missing_mask |= ~np.isfinite(frame[column])
    frame["mode"] = frame["mode"].astype(str).str.strip().str.lower()
    missing_mask |= ~frame["mode"].isin(
        {"heating", "heat", "auto", "cooling", "cool", "off", "idle"}
    )
    if missing_mask.any():
        return _incomplete(
            "Historical values are missing, invalid, or have an unknown climate mode.",
            int(missing_mask.sum()),
        )

    frame.sort_values("_time", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    if frame["_time"].duplicated().any():
        return _incomplete("History contains duplicate timestamps.")
    elapsed_hours = frame["_time"].diff().dt.total_seconds().div(3600.0)
    interval_hours = float(elapsed_hours.dropna().median())
    if (
        not np.isfinite(interval_hours)
        or interval_hours <= 0
        or (elapsed_hours.dropna() > MAX_HISTORY_GAP_HOURS).any()
    ):
        max_gap_minutes = MAX_HISTORY_GAP_HOURS * 60
        return _incomplete(
            "History must have at least two ordered samples with no gap over "
            f"{max_gap_minutes:g} minutes."
        )

    frame["mode"] = frame["mode"].map(CLIMATE_MODE_ALIASES)
    parameter_modes = []
    last_active_mode = "heating"
    for mode in frame["mode"]:
        if mode in mode_parameters:
            last_active_mode = mode
        parameter_modes.append(
            mode if mode in mode_parameters else last_active_mode
        )

    interval_durations = elapsed_hours.to_numpy(dtype=float, copy=True)
    interval_durations[0] = interval_hours
    mode_time_constants = np.array(
        [mode_parameters[mode][1] for mode in parameter_modes], dtype=float
    )
    decay_values = np.exp(-interval_durations / mode_time_constants)
    if not np.all(decay_values < 1.0):
        return _incomplete("Thermal replay interval is too short to resolve.")

    effective_capacity_by_mode = capacity_by_mode
    simulated_indoor = float(frame.loc[0, "indoor_temp"])
    interval_rows = []
    actual_thermal_kwh = 0.0
    counterfactual_thermal_kwh = 0.0
    actual_electrical_kwh = 0.0
    counterfactual_electrical_kwh = 0.0
    measured_cop_count = 0
    estimated_cop_count = 0

    replay_rows = frame.loc[:, REQUIRED_COLUMNS].itertuples(
        index=False, name=None
    )
    for index, (
        timestamp,
        indoor_temp,
        target_temp,
        outdoor_temp,
        external_gain,
        mode,
        thermal_power,
        electrical_power,
    ) in enumerate(replay_rows):
        dt_hours = (
            interval_hours
            if index == 0
            else float(elapsed_hours.iloc[index])
        )
        hlc, tau = mode_parameters[parameter_modes[index]]
        decay = float(decay_values[index])
        target = float(target_temp)
        outdoor = float(outdoor_temp)
        external_gain = float(external_gain)
        desired_equilibrium = (
            target - simulated_indoor * decay
        ) / (1.0 - decay)
        signed_hp_power = (
            hlc * (desired_equilibrium - outdoor) - external_gain
        )
        if mode == "heating":
            hp_power_kw = max(0.0, signed_hp_power)
        elif mode == "cooling":
            hp_power_kw = min(0.0, signed_hp_power)
        else:
            hp_power_kw = 0.0

        passive_equilibrium = outdoor + (
            hp_power_kw + external_gain
        ) / hlc
        simulated_indoor = passive_equilibrium + (
            simulated_indoor - passive_equilibrium
        ) * decay

        measured_thermal_power = abs(float(thermal_power))
        electrical_power_kw = max(0.0, float(electrical_power)) / 1000.0
        measured_cop = (
            measured_thermal_power / electrical_power_kw
            if electrical_power_kw > MIN_COP_ELECTRICAL_POWER_KW
            else 0.0
        )
        if (
            np.isfinite(measured_cop)
            and MIN_VALID_COP <= measured_cop <= MAX_VALID_COP
        ):
            cop = measured_cop
            cop_source = "measured"
            measured_cop_count += 1
        else:
            cop = estimated_cop
            cop_source = "estimated fallback"
            estimated_cop_count += 1

        actual_interval_thermal = measured_thermal_power * dt_hours
        counterfactual_interval_thermal = abs(hp_power_kw) * dt_hours
        actual_interval_electrical = electrical_power_kw * dt_hours
        counterfactual_interval_electrical = (
            counterfactual_interval_thermal / cop
        )
        actual_thermal_kwh += actual_interval_thermal
        counterfactual_thermal_kwh += counterfactual_interval_thermal
        actual_electrical_kwh += actual_interval_electrical
        counterfactual_electrical_kwh += counterfactual_interval_electrical

        interval_rows.append(
            {
                "_time": timestamp,
                "actual_indoor_temp": float(indoor_temp),
                "target_temp": target,
                "counterfactual_indoor_temp": simulated_indoor,
                "climate_mode": mode,
                "actual_thermal_power_kw": measured_thermal_power,
                "counterfactual_thermal_power_kw": abs(hp_power_kw),
                "actual_thermal_energy_kwh": actual_interval_thermal,
                "counterfactual_thermal_energy_kwh": (
                    counterfactual_interval_thermal
                ),
                "actual_electrical_energy_kwh": actual_interval_electrical,
                "counterfactual_electrical_energy_kwh": (
                    counterfactual_interval_electrical
                ),
                "cop": cop,
                "cop_source": cop_source,
            }
        )

    intervals = pd.DataFrame(interval_rows)
    return {
        "complete": True,
        "reason": "",
        "missing_count": 0,
        "intervals": intervals,
        "sample_count": len(intervals),
        "measured_cop_count": measured_cop_count,
        "estimated_cop_count": estimated_cop_count,
        "cop_estimation_method": (
            f"Measured interval COP when valid; otherwise fallback COP "
            f"{estimated_cop:.2f}"
        ),
        "effective_heat_capacity_kwh_per_k_by_mode": (
            effective_capacity_by_mode
        ),
        "actual_thermal_kwh": actual_thermal_kwh,
        "counterfactual_thermal_kwh": counterfactual_thermal_kwh,
        "thermal_energy_difference_kwh": (
            actual_thermal_kwh - counterfactual_thermal_kwh
        ),
        "actual_electrical_kwh": actual_electrical_kwh,
        "counterfactual_electrical_kwh": counterfactual_electrical_kwh,
        "electrical_energy_difference_kwh": (
            actual_electrical_kwh - counterfactual_electrical_kwh
        ),
    }
