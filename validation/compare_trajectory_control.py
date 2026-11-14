"""Conditional, one-cycle heating-control comparison; never a live controller.

Run from the repository root with ``python -m validation.compare_trajectory_control``.
The profile below is local to this offline process, not a change to config.py.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Callable
from unittest.mock import patch

from src import config
from src.cycle_context import CycleContext
from src.cycle_routes import (
    step_ema_smoothing,
    step_gradual_control,
    step_setpoint_hold,
)
from src.model_wrapper import EnhancedModelWrapper
from src.shadow_mode import get_effective_unified_state_file
from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src.unified_thermal_state import ThermalStateManager


OFFLINE_PROFILE = {
    "CYCLE_INTERVAL_MINUTES": 10,
    "HISTORY_STEP_MINUTES": 10,
    "TRAJECTORY_STEPS": 4,
    "TRAJECTORY_PREDICTION_ENABLED": True,
    "TRAJECTORY_SEARCH_ERROR_MODE": "early_comfort_plus_horizon",
    "TRAJECTORY_SEARCH_EARLY_STEPS": 3,
    "HEATING_CORRECTION_MODE": "physics",
    "ENABLE_HEAT_SOURCE_CHANNELS": True,
    "OUTLET_EFFECTIVENESS_MODE": "scalar",
    "OUTLET_REFERENCE_ENABLED": True,
    "OUTLET_REFERENCE_ALWAYS_ACTIVE": True,
    "OUTLET_REFERENCE_BAND_K": 2.0,
    "OUTLET_REFERENCE_DEMAND_EXTRA_K": 4.0,
    "OUTLET_REFERENCE_LAMBDA": 0.01,
    "OUTLET_SEARCH_MIN_SENSITIVITY": 0.1,
    "OUTLET_DEMAND_DEFICIT_K": 0.3,
    "OUTLET_DEMAND_TREND_K_PER_H": 0.1,
    "MAX_TEMP_CHANGE_PER_CYCLE": 2.0,
    "OUTLET_CALM_SLEW_K": 1.0,
    "OUTLET_SMOOTHING_ALPHA": 0.8,
    "OUTLET_SMOOTHING_BYPASS": 2.0,
    "SETPOINT_HOLD_ENABLED": True,
    "MIN_SETPOINT_HOLD_CYCLES": 4,
    "MAX_SETPOINT_HOLD_CYCLES": 20,
    "OUTLET_HYSTERESIS_K": 0.3,
    "SMART_ROUNDING_ENABLED": False,
    "PV_TRAJ_FORECAST_MODE_ENABLED": False,
    "PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION": False,
    "PRICE_COST_DISABLE_OVERSHOOT_CORRECTION": False,
    "PRICE_EXPENSIVE_OVERSHOOT": 0.2,
    "TREND_DECAY_TAU_HOURS": 1.5,
    "FD_NEWTON_MIN_SENSITIVITY": 0.1,
    "CLOUD_COVER_CORRECTION_ENABLED": True,
    "CLOUD_CORRECTION_MIN_FACTOR": 0.1,
}

PARAMETER_NAMES = (
    "thermal_time_constant", "heat_loss_coefficient", "outlet_effectiveness",
    "slab_time_constant_hours", "delta_t_floor", "solar_lag_minutes",
    "pv_heat_weight", "fireplace_heat_weight", "tv_heat_weight",
    "fp_heat_output_kw", "fp_decay_time_constant", "room_spread_delay_minutes",
    "cloud_factor_exponent", "solar_decay_tau_hours",
)


@dataclass
class ReplayCycle:
    timestamp: str
    features: dict[str, Any]
    parameters: dict[str, float]
    parameter_source: str
    outdoor_forecast: list[float]
    pv_forecast: list[float]
    actual_indoor: float
    target_adjusted: float
    baseline: float
    lower: float
    upper: float
    logged_raw: float
    logged_published: float
    hp_status: str
    hp_window: dict[str, float | None] | None
    hold_remaining: int
    price_level: str
    logged_trajectory: list[float]
    logged_trajectory_outlet: float | None


def comfort_metrics(
    trajectory: list[float], target: float, band: float, step_hours: float,
) -> dict[str, float]:
    if not trajectory or not all(math.isfinite(t) for t in trajectory):
        raise ValueError("A finite, nonempty trajectory is required")
    if (
        not all(math.isfinite(v) for v in (target, band, step_hours))
        or band < 0 or step_hours <= 0
    ):
        raise ValueError("Invalid comfort band, target or time step")
    under = [max(target - band - t, 0.0) for t in trajectory]
    over = [max(t - target - band, 0.0) for t in trajectory]
    return {
        "squared_violation_k2h": sum(u * u + o * o for u, o in zip(under, over))
        * step_hours,
        "undershoot_kh": sum(under) * step_hours,
        "overshoot_kh": sum(over) * step_hours,
        "hours_below_band": sum(u > 1e-12 for u in under) * step_hours,
        "hours_above_band": sum(o > 1e-12 for o in over) * step_hours,
        "min_indoor": min(trajectory),
        "max_indoor": max(trajectory),
    }


def optimize_trajectory(
    simulate: Callable[[float], list[float]], *,
    lower: float, upper: float, baseline: float,
    current_indoor: float, target: float, comfort_band: float,
    step_hours: float, resolution: float = 0.1,
) -> dict[str, float]:
    if (
        not all(math.isfinite(v) for v in
                (lower, upper, baseline, current_indoor, resolution))
        or lower > upper or resolution <= 0
    ):
        raise ValueError("Invalid candidate bounds or resolution")
    candidates = {lower, upper, min(upper, max(lower, baseline))}
    for index in range(math.ceil(lower / resolution), math.floor(upper / resolution) + 1):
        candidates.add(round(index * resolution, 10))
    best: dict[str, float] | None = None
    best_key: tuple[float, float, float] | None = None
    for outlet in sorted(candidates):
        metrics = comfort_metrics(simulate(outlet), target, comfort_band, step_hours)
        key = (
            metrics["squared_violation_k2h"],
            outlet if current_indoor > target + comfort_band else abs(outlet - baseline),
            outlet,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = {"outlet": outlet, **metrics}
    assert best is not None
    return best


def _match(pattern: str, text: str, field: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text))
    if not matches:
        raise ValueError(f"missing {field}")
    return matches[-1]


def _payload(text: str, entity: str) -> dict[str, Any]:
    match = _match(
        rf"Setting HA state for sensor\.{entity}: (\{{[^\n]*\}})",
        text, entity,
    )
    return ast.literal_eval(match[1])["attributes"]


def _parse_cycle(text: str) -> ReplayCycle:
    features = _payload(text, "ml_heating_features")
    required_features = (
        "indoor_temp_lag_30m", "outdoor_temp", "inlet_temp", "outlet_temp",
        "target_temp", "delta_t", "indoor_temp_delta_60m", "pv_now",
        "pv_power_history", "fireplace_on", "tv_on",
    )
    for name in required_features:
        value = features[name]
        values = value if isinstance(value, list) else [value]
        if not values or not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f"invalid feature {name}")
    source = "learning"
    if "Setting HA state for sensor.ml_heating_learning:" in text:
        parameters = _payload(text, "ml_heating_learning")
        if parameters.get("outlet_effectiveness_mode", "scalar") != "scalar":
            raise ValueError("offline profile supports scalar effectiveness only")
        parameters = {name: float(parameters[name]) for name in PARAMETER_NAMES}
    else:
        source = "same_cycle_curve_rounded_6dp"
        curve = _payload(text, "ml_heating_base_outlet_curve")
        parameters = {
            name: float(curve[f"param_heating_{name}"]) for name in PARAMETER_NAMES
        }
    if not all(math.isfinite(v) for v in parameters.values()):
        raise ValueError("invalid thermal parameters")

    prediction = _match(
        r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),[^\n]*Prediction: Current ([\d.]+)"
        r".*?\(adjusted ([\d.]+).*?Calculated outlet: ([\d.]+)",
        text, "prediction",
    )
    decision_text = text[:prediction.start()]
    bounds = _match(
        r"Using search bounds: outlet_min=([\d.]+).*?outlet_max=([\d.]+)",
        text, "search bounds",
    )
    forecast = _match(
        r"Trajectory prediction using forecast arrays: outdoor=(\[[^\n]*?\]), PV=(\[[^\n]*?\])",
        text, "forecast arrays",
    )
    baseline = float(_match(r"Gradual control: baseline=([\d.]+)", text, "baseline")[1])
    start = _match(
        r"HP start estimate \(heating\): status=(\w+) method=(\w+) "
        r"start_in=(None|[\d.]+) min runtime=(None|[\d.]+) min",
        decision_text, "HP start window",
    )
    if start[2] != "degree_minutes" or start[1] not in {"idle", "running", "starting_soon"}:
        raise ValueError(f"unsupported HP window {start[1]}/{start[2]}")
    window = {
        "delay_h": 1e6 if start[3] == "None" else float(start[3]) / 60,
        "runtime_h": None if start[4] == "None" else float(start[4]) / 60,
    }
    hold = re.search(r"Setpoint hold: keeping [\d.]+.*?for (\d+) more cycle", text)
    state = _payload(text, "ml_heating_state")
    price = (
        _payload(text, "ml_heating_price_level")["price_level"]
        if "Setting HA state for sensor.ml_heating_price_level:" in text
        else _match(
            r"Price (cheap|normal|expensive): target",
            decision_text, "effective price level",
        )[1]
    )
    trajectory_outlet = re.findall(r"Trajectory DEBUG: outlet=([\d.]+)", decision_text)
    return ReplayCycle(
        timestamp=prediction[1], features=features,
        parameters=parameters, parameter_source=source,
        outdoor_forecast=ast.literal_eval(forecast[1]),
        pv_forecast=ast.literal_eval(forecast[2]),
        actual_indoor=float(prediction[2]), target_adjusted=float(prediction[3]),
        baseline=baseline, lower=float(bounds[1]), upper=float(bounds[2]),
        logged_raw=float(state["suggested_temp"]),
        logged_published=float(state["final_temp"]),
        hp_status=start[1], hp_window=window,
        hold_remaining=int(hold[1]) if hold else 0,
        price_level=price,
        logged_trajectory=[
            float(value) for value in re.findall(
                r" - DEBUG -\s+[\d.]+h: ([\d.]+)\u00b0C", text,
            )
        ],
        logged_trajectory_outlet=float(trajectory_outlet[-1]) if trajectory_outlet else None,
    )


def parse_log(text: str) -> tuple[list[ReplayCycle], list[str]]:
    signatures = set(re.findall(
        r"Binary search features: .*?horizon=([\d.]+)h, objective=(\w+), early_steps=(\d+)",
        text,
    ))
    if signatures and signatures != {("4.0", "early_comfort_plus_horizon", "3")}:
        raise ValueError(f"Log does not match the offline search profile: {signatures}")
    starts = list(re.finditer(r"^.* - DEBUG - [^\n]*CYCLE \d+ START:[^\n]*$", text, re.M))
    cycles = []
    excluded = []
    if not starts:
        raise ValueError("No control-cycle boundaries found")
    if text[:starts[0].start()].strip():
        excluded.append("Initial partial cycle: no CYCLE START boundary")
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[start.start():end]
        try:
            cycles.append(_parse_cycle(block))
        except (ValueError, KeyError, SyntaxError) as error:
            excluded.append(f"{start[0][:19]}: {error}")
    return cycles, excluded


class _OfflineWrapper(EnhancedModelWrapper):
    """Inject recorded inputs without constructing live clients or singletons."""

    def __init__(
        self, model: ThermalEquilibriumModel, cycle: ReplayCycle,
        window: dict[str, float | None] | None,
    ):
        self.thermal_model = model
        self._climate_mode = "heating"
        self._current_features = cycle.features
        self._current_indoor = cycle.features["indoor_temp_lag_30m"]
        self._price_info = {"price_level": cycle.price_level}
        self._avg_cloud_cover = 50.0
        self._hp_start_window = window
        self._hp_start_estimate = None  # Deliberately freeze the conditional window.
        self._fireplace_last_on_time = None
        self._fireplace_on_since = None
        self.cycle_aligned_forecast = {}
        self.cycle = cycle

    def _get_forecast_conditions(self, outdoor_temp, pv_power, thermal_features):
        return (
            self.cycle.features["outdoor_temp"], self.cycle.features["pv_now"],
            self.cycle.outdoor_forecast, self.cycle.pv_forecast,
        )

    def _log_multi_horizon_predictions(self, **kwargs):
        logging.debug("Offline comparison omits telemetry-only multi-horizon logging")


def _publishable(cycle: ReplayCycle, outlet: float) -> float:
    ctx = CycleContext(
        climate_mode="heating", actual_outlet_temp=cycle.features["outlet_temp"],
        actual_indoor=cycle.actual_indoor, prediction_indoor_temp=cycle.actual_indoor,
        target_indoor_temp=cycle.features["target_temp"],
        final_temp=outlet, features_dict=cycle.features,
        state={
            "last_final_temp": cycle.baseline,
            "last_target_indoor_temp": cycle.features["target_temp"],
            "setpoint_hold_cycles_remaining": cycle.hold_remaining,
        },
    )
    step_gradual_control(ctx)
    step_ema_smoothing(ctx)
    step_setpoint_hold(ctx)
    assert ctx.final_temp is not None
    return float(ctx.final_temp)


def compare_cycles(
    cycles: list[ReplayCycle], *, comfort_band: float,
) -> list[dict[str, Any]]:
    rows = []
    with patch.multiple(config, **OFFLINE_PROFILE), TemporaryDirectory() as temporary:
        manager = ThermalStateManager(
            state_file=str(Path(temporary) / "offline_thermal_state.json"),
        )
        model = ThermalEquilibriumModel(state_manager=manager)
        if model.orchestrator is None:
            raise RuntimeError("The offline comparison requires heat-source channels")
        for cycle in cycles:
            model.orchestrator.sync_from_model_parameters(cycle.parameters)
            model._sync_model_from_orchestrator()
            lower = max(cycle.lower, cycle.features["inlet_temp"] - 1.0)
            upper = min(cycle.upper, cycle.features["inlet_temp"] + 5.0)
            scenarios: list[tuple[str, dict[str, float | None] | None]] = [
                ("recorded_window", cycle.hp_window),
            ]
            delay = cycle.hp_window["delay_h"] if cycle.hp_window else None
            if cycle.hp_status == "idle" and delay is not None and delay >= 1e6:
                scenarios = [
                    ("idle_no_start", cycle.hp_window),
                    ("idle_immediate_start", {"delay_h": 0.0, "runtime_h": None}),
                ]
            for scenario, window in scenarios:
                with patch.multiple(
                    config, CLAMP_MIN_ABS=cycle.lower, CLAMP_MAX_ABS=cycle.upper,
                ):
                    wrapper = _OfflineWrapper(model, cycle, window)
                    thermal_features = {
                        "pv_power": cycle.features["pv_now"],
                        "pv_power_history": cycle.features["pv_power_history"],
                        "fireplace_on": cycle.features["fireplace_on"],
                        "tv_on": cycle.features["tv_on"],
                    }
                    started = perf_counter()
                    current = wrapper._calculate_required_outlet_temp(
                        current_indoor=cycle.features["indoor_temp_lag_30m"],
                        target_indoor=cycle.target_adjusted,
                        outdoor_temp=cycle.features["outdoor_temp"],
                        thermal_features=thermal_features,
                        inlet_temp=cycle.features["inlet_temp"],
                    )
                    current_ms = (perf_counter() - started) * 1000
                    entry = getattr(wrapper, "_trajectory_simulator", None)
                    if entry is None:
                        raise RuntimeError(f"No trajectory simulator at {cycle.timestamp}")
                    simulate = entry[1]

                    started = perf_counter()
                    alternative = optimize_trajectory(
                        simulate, lower=lower, upper=upper, baseline=cycle.baseline,
                        current_indoor=cycle.features["indoor_temp_lag_30m"],
                        target=cycle.target_adjusted, comfort_band=comfort_band,
                        step_hours=1 / 6,
                    )
                    alternative_ms = (perf_counter() - started) * 1000
                    row: dict[str, Any] = {
                        "timestamp": cycle.timestamp, "scenario": scenario,
                        "parameter_source": cycle.parameter_source,
                        "indoor_search": cycle.features["indoor_temp_lag_30m"],
                        "indoor_control": cycle.actual_indoor,
                        "target_adjusted": cycle.target_adjusted,
                        "baseline": cycle.baseline, "lower": lower, "upper": upper,
                        "hp_delay_h": window["delay_h"] if window else None,
                        "hp_runtime_h": window["runtime_h"] if window else None,
                        "logged_raw": cycle.logged_raw,
                        "logged_published": cycle.logged_published,
                        "current_raw": current, "alternative_raw": alternative["outlet"],
                        "current_publishable": _publishable(cycle, current),
                        "alternative_publishable": _publishable(cycle, alternative["outlet"]),
                        "current_ms": current_ms, "alternative_ms": alternative_ms,
                    }
                    for method in (
                        "logged_published", "current_raw", "alternative_raw",
                        "current_publishable", "alternative_publishable",
                    ):
                        metrics = comfort_metrics(
                            simulate(row[method]), cycle.target_adjusted, comfort_band, 1 / 6,
                        )
                        row.update({f"{method}_{name}": value for name, value in metrics.items()})
                    row["logged_trajectory_outlet"] = cycle.logged_trajectory_outlet
                    row["logged_trajectory_max_error_k"] = None
                    if (
                        scenario != "idle_immediate_start"
                        and cycle.logged_trajectory_outlet is not None
                    ):
                        reconstructed = simulate(cycle.logged_trajectory_outlet)
                        if len(reconstructed) == len(cycle.logged_trajectory):
                            row["logged_trajectory_max_error_k"] = max(
                                abs(a - b)
                                for a, b in zip(reconstructed, cycle.logged_trajectory)
                            )
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--output", type=Path,
        help="Output directory (default: directory of the effective unified state file)",
    )
    parser.add_argument("--comfort-band", type=float, default=0.1)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(get_effective_unified_state_file()).parent
    args.output.mkdir(parents=True, exist_ok=True)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                args.output / "trajectory_control_comparison.log",
                mode="w", encoding="utf-8",
            ),
            console,
        ],
        force=True,
    )
    log_bytes = args.log.read_bytes()
    cycles, excluded = parse_log(log_bytes.decode("utf-8"))
    if not cycles:
        raise ValueError(f"No reconstructable cycles: {excluded}")
    rows = compare_cycles(cycles, comfort_band=args.comfort_band)
    summary: dict[str, Any] = {
        "input_sha256": hashlib.sha256(log_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comfort_band_k": args.comfort_band,
        "optimizer": {
            "objective": "sum(dt_h * (below_band_K**2 + above_band_K**2))",
            "candidate_resolution_k": 0.1,
            "additional_candidates": "Exact bounds and clamped recorded baseline",
            "ties": "Lower outlet above comfort band, otherwise nearest baseline",
            "post_correction": "None",
        },
        "profile": OFFLINE_PROFILE, "cycles": len(cycles), "rows": len(rows),
        "excluded": excluded,
        "limitations": [
            "Conditional one-cycle comparison, not an exact replay or live shadow controller.",
            "Each cycle resets to logged observed temperatures and logged prior outlet.",
            "Missing GM/supply attributes: all sensor windows frozen across candidates.",
            "Unknown idle start: no-start and immediate-start/unlimited-runtime scenarios.",
            "No new model parameters or learning; same-cycle logged channel parameters used.",
            "Search lower bounds read at log precision (0.1 K), then inlet-relative bound applied.",
            "Trajectory cross-check uses its logged verification outlet (0.1 K), not post-Newton output.",
            "Cloud cover assumed 50%; complete original configuration/hold history unavailable.",
            "Hold count from explicit hold log, otherwise assumed zero; last target assumed unchanged.",
            "Profile combines visible log settings with explicitly listed offline assumptions.",
            "No HA publishing: publishable values stop after original gradual/EMA/hold steps.",
            "Each score holds its candidate constant over 4 h, not a closed-loop rollout.",
            "Overlapping horizons must not be summed as measured comfort or energy savings.",
            "Input log has overshoot only; real undershoot performance remains unvalidated.",
        ],
        "scenarios": {},
    }
    groups = {
        scenario: [row for row in rows if row["scenario"] == scenario]
        for scenario in sorted({row["scenario"] for row in rows})
    }
    groups["historical_windows_combined"] = [
        row for row in rows if row["scenario"] != "idle_immediate_start"
    ]
    for scenario, group in groups.items():
        summary["scenarios"][scenario] = {
            "cycles": len(group),
            "raw_different_cycles": sum(abs(r["current_raw"] - r["alternative_raw"]) > 0.05 for r in group),
            "publishable_different_cycles": sum(abs(r["current_publishable"] - r["alternative_publishable"]) > 0.05 for r in group),
            "mean_current_ms": sum(r["current_ms"] for r in group) / len(group),
            "mean_alternative_ms": sum(r["alternative_ms"] for r in group) / len(group),
            "mean_current_raw_cost_k2h": sum(r["current_raw_squared_violation_k2h"] for r in group) / len(group),
            "mean_alternative_raw_cost_k2h": sum(r["alternative_raw_squared_violation_k2h"] for r in group) / len(group),
            "mean_current_publishable_cost_k2h": sum(r["current_publishable_squared_violation_k2h"] for r in group) / len(group),
            "mean_alternative_publishable_cost_k2h": sum(r["alternative_publishable_squared_violation_k2h"] for r in group) / len(group),
            "publishable_increases_above_baseline": sum(
                r[method] > r["baseline"] + 1e-9 for r in group
                for method in ("current_publishable", "alternative_publishable")
            ),
            "max_logged_trajectory_error_k": max(
                (r["logged_trajectory_max_error_k"] for r in group
                 if r["logged_trajectory_max_error_k"] is not None),
                default=None,
            ),
        }
    with (args.output / "trajectory_control_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    serialized = json.dumps(summary, indent=2, allow_nan=False)
    (args.output / "trajectory_control_summary.json").write_text(serialized + "\n", encoding="utf-8")
    logging.info(
        "Offline comparison complete: %d cycles, %d scenario rows; reports in %s",
        len(cycles), len(rows), args.output,
    )
    print(serialized)


if __name__ == "__main__":
    main()
