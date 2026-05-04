#!/usr/bin/env python3
"""
Epsilon Sensitivity Analysis for Finite-Difference Gradient Calculation.

For each learnable parameter, sweeps epsilon from 0.1% to 50% of the default
value and measures the resulting ΔT = trajectory[-1](+ε) − trajectory[-1](−ε).

Goal: find epsilon values that produce ΔT ≈ 0.1–0.3°C under typical heating
conditions, staying in the linear regime of the finite-difference approximation.

Also checks linearity: gradient at ε, ε/2, ε/4 should agree within ~10%.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src import config
from src.thermal_constants import PhysicsConstants
from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src.thermal_config import ThermalParameterConfig

# Reference conditions (typical winter heating)
REF_INDOOR = 21.0
REF_OUTDOOR = 5.0
REF_OUTLET = 30.0
REF_HORIZON_H = 4.0
REF_STEP_MIN = 10  # matches CYCLE_INTERVAL_MINUTES default

# Learnable parameters and the epsilon currently configured in the runtime.
PARAMS = {
    "thermal_time_constant": {
        "current_eps": PhysicsConstants.THERMAL_TIME_CONSTANT_EPSILON,
    },
    "heat_loss_coefficient": {
        "current_eps": PhysicsConstants.HEAT_LOSS_COEFFICIENT_EPSILON,
    },
    "outlet_effectiveness": {
        "current_eps": PhysicsConstants.OUTLET_EFFECTIVENESS_EPSILON,
    },
    "pv_heat_weight": {
        "current_eps": PhysicsConstants.PV_HEAT_WEIGHT_EPSILON,
    },
    "tv_heat_weight": {
        "current_eps": PhysicsConstants.TV_HEAT_WEIGHT_EPSILON,
    },
    "solar_lag_minutes": {
        "current_eps": PhysicsConstants.SOLAR_LAG_EPSILON,
    },
    "slab_time_constant_hours": {
        "current_eps": PhysicsConstants.SLAB_TIME_CONSTANT_EPSILON,
    },
}


class _CalibrationStateManager:
    """Minimal state manager for isolated model construction."""

    def get_current_parameters(self):
        return {}

    def get_heat_source_channel_state(self):
        return None


def create_model():
    """Create an isolated ThermalEquilibriumModel with default parameters."""
    os.environ.setdefault("HA_URL", "http://dummy:8123")
    os.environ.setdefault("HA_TOKEN", "dummy_token")
    defaults = ThermalParameterConfig.DEFAULTS
    bounds = ThermalParameterConfig.BOUNDS

    original_channel_mode = config.ENABLE_HEAT_SOURCE_CHANNELS
    config.ENABLE_HEAT_SOURCE_CHANNELS = False
    try:
        model = ThermalEquilibriumModel(
            state_manager=_CalibrationStateManager()
        )
    finally:
        config.ENABLE_HEAT_SOURCE_CHANNELS = original_channel_mode

    model.orchestrator = None

    model.thermal_time_constant = defaults["thermal_time_constant"]
    model.heat_loss_coefficient = defaults["heat_loss_coefficient"]
    model._baseline_heat_loss_coefficient = defaults["heat_loss_coefficient"]
    model.outlet_effectiveness = defaults["outlet_effectiveness"]
    model.solar_lag_minutes = defaults["solar_lag_minutes"]
    model.slab_time_constant_hours = defaults["slab_time_constant_hours"]
    model.external_source_weights = {
        "pv": defaults["pv_heat_weight"],
        "fireplace": defaults["fireplace_heat_weight"],
        "tv": defaults["tv_heat_weight"],
    }

    model.prediction_horizon_hours = REF_HORIZON_H
    model.delta_t_floor = defaults["delta_t_floor"]
    model.cloud_factor_exponent = defaults["cloud_factor_exponent"]
    model.solar_decay_tau_hours = defaults.get("solar_decay_tau_hours", 1.0)

    # Bounds
    model.thermal_time_constant_bounds = bounds["thermal_time_constant"]
    model.heat_loss_coefficient_bounds = bounds["heat_loss_coefficient"]
    model.outlet_effectiveness_bounds = bounds["outlet_effectiveness"]
    model.pv_heat_weight_bounds = bounds["pv_heat_weight"]
    model.tv_heat_weight_bounds = bounds["tv_heat_weight"]
    model.solar_lag_minutes_bounds = bounds["solar_lag_minutes"]
    model.slab_time_constant_bounds = bounds["slab_time_constant_hours"]

    # Additional attributes needed by predict_thermal_trajectory
    model.safety_margin = 0.2
    model._pv_power_history = []
    model._solar_contribution_history = []
    model._fireplace_decay_kw = 0.0
    model.fp_heat_output_kw = defaults.get("fp_heat_output_kw", 3.0)
    model.fp_decay_time_constant = defaults.get("fp_decay_time_constant", 3.9)
    model.fp_room_spread_delay_minutes = defaults.get(
        "room_spread_delay_minutes", 18.0
    )

    model.momentum_decay_rate = PhysicsConstants.MOMENTUM_DECAY_RATE
    model.learning_rate = PhysicsConstants.DEFAULT_LEARNING_RATE
    model.learning_confidence = PhysicsConstants.INITIAL_LEARNING_CONFIDENCE
    model.adaptive_learning_enabled = True

    return model


def _reference_forecast(base_value, deltas):
    return [base_value + delta for delta in deltas[: config.TRAJECTORY_STEPS]]


def build_reference_prediction(parameter_name, climate_mode="heating"):
    """Build a representative stored prediction record for the learner."""
    pv_history = [
        0.0, 80.0, 120.0, 180.0, 260.0, 340.0, 430.0, 520.0, 640.0,
        780.0, 920.0, 1040.0, 900.0, 760.0, 620.0, 500.0, 420.0, 360.0,
    ]
    base_context = {
        "outlet_temp": 30.0,
        "outdoor_temp": REF_OUTDOOR,
        "current_indoor": REF_INDOOR,
        "pv_power": pv_history[-1],
        "pv_power_history": pv_history,
        "pv_forecast": _reference_forecast(520.0, [180.0, 320.0, 140.0, -60.0]),
        "outdoor_forecast": _reference_forecast(REF_OUTDOOR, [-1.0, -2.0, -2.0, -1.0]),
        "fireplace_on": 0,
        "tv_on": 0,
        "avg_cloud_cover": 35.0,
        "inlet_temp": 27.5,
        "delta_t": 2.5,
        "thermal_power": 1.8,
        "auxiliary_heat": 0.0,
        "climate_mode": climate_mode,
        "indoor_temp_delta_60m": 0.0,
    }

    if parameter_name == "tv_heat_weight":
        base_context["tv_on"] = 1
    if parameter_name == "solar_lag_minutes":
        base_context["pv_forecast"] = _reference_forecast(
            650.0, [450.0, 300.0, -50.0, -300.0]
        )
        base_context["pv_power_history"] = [
            0.0, 0.0, 40.0, 120.0, 260.0, 420.0, 650.0, 910.0, 1120.0,
            980.0, 760.0, 540.0, 380.0, 260.0, 190.0, 150.0, 120.0, 90.0,
        ]
        base_context["pv_power"] = base_context["pv_power_history"][-1]
    if parameter_name == "slab_time_constant_hours":
        base_context["outlet_temp"] = 33.0
        base_context["inlet_temp"] = 28.0
        base_context["delta_t"] = 3.5
        base_context["thermal_power"] = 2.6
    if climate_mode == "cooling":
        base_context.update(
            {
                "outlet_temp": 19.0,
                "outdoor_temp": 29.0,
                "current_indoor": 24.0,
                "outdoor_forecast": _reference_forecast(29.0, [1.0, 2.0, 1.0, 0.0]),
                "inlet_temp": 22.0,
                "delta_t": -2.5,
                "thermal_power": -1.3,
                "climate_mode": "cooling",
            }
        )

    return {
        "error": 1.0,
        "timestamp": "2026-05-04T12:00:00",
        "context": base_context,
    }


def compute_runtime_gradient(model, param_name, epsilon, prediction=None):
    """Compute the learner's actual gradient for one stored prediction."""
    prediction = prediction or build_reference_prediction(param_name)
    return model._calculate_parameter_gradient(
        param_name, epsilon, [prediction]
    )


def compute_signal_delta(model, param_name, epsilon, prediction=None):
    """Convert the runtime gradient back to the implied finite-difference ΔT."""
    gradient = compute_runtime_gradient(model, param_name, epsilon, prediction)
    return abs(gradient * 2 * epsilon)


def compute_gradient(model, param_name, epsilon):
    """Compute finite-difference gradient dT/dparam."""
    return compute_runtime_gradient(model, param_name, epsilon)


def check_linearity(model, param_name, epsilon, prediction=None):
    """Check linearity: gradient at ε, ε/2, ε/4 should agree within ~10%."""
    g1 = compute_runtime_gradient(model, param_name, epsilon, prediction)
    g2 = compute_runtime_gradient(model, param_name, epsilon / 2, prediction)
    g4 = compute_runtime_gradient(model, param_name, epsilon / 4, prediction)

    if abs(g1) < 1e-12:
        return g1, g2, g4, float("inf"), float("inf")

    dev_2 = abs(g2 - g1) / abs(g1) * 100
    dev_4 = abs(g4 - g1) / abs(g1) * 100
    return g1, g2, g4, dev_2, dev_4


def find_target_epsilon(model, param_name, target_dt=0.2):
    """Search for an epsilon that produces a near-target runtime ΔT signal."""
    default_val = getattr(model, param_name)
    ratios = np.logspace(-4, np.log10(0.5), 36)
    candidates = []
    prediction = build_reference_prediction(param_name)

    for ratio in ratios:
        epsilon = max(abs(default_val) * ratio, 1e-8)
        signal = compute_signal_delta(model, param_name, epsilon, prediction)
        g1, g2, g4, dev2, dev4 = check_linearity(
            model, param_name, epsilon, prediction
        )
        candidates.append(
            {
                "epsilon": epsilon,
                "signal": signal,
                "gradient": g1,
                "dev2": dev2,
                "dev4": dev4,
                "linear": dev2 < 15 and dev4 < 15,
            }
        )

    valid = [
        candidate
        for candidate in candidates
        if candidate["linear"] and 0.05 <= candidate["signal"] <= 0.5
    ]
    pool = valid or candidates
    best = min(pool, key=lambda item: abs(item["signal"] - target_dt))
    best["reachable"] = bool(valid)
    return best


def run_calibration():
    """Run the full calibration sweep and return structured results."""
    model = create_model()
    results = []
    sweeps = {}

    for param, info in PARAMS.items():
        prediction = build_reference_prediction(param)
        current_eps = info["current_eps"]
        default_val = getattr(model, param)
        current_signal = compute_signal_delta(
            model, param, current_eps, prediction
        )
        g1, g2, g4, dev2, dev4 = check_linearity(
            model, param, current_eps, prediction
        )
        recommendation = find_target_epsilon(model, param)
        sweep_rows = []
        for ratio in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            epsilon = max(abs(default_val) * ratio, 1e-8)
            sweep_rows.append(
                {
                    "ratio": ratio,
                    "epsilon": epsilon,
                    "signal": compute_signal_delta(
                        model, param, epsilon, prediction
                    ),
                }
            )
        sweeps[param] = sweep_rows
        results.append(
            {
                "parameter": param,
                "default": default_val,
                "current_epsilon": current_eps,
                "current_signal": current_signal,
                "current_gradient": g1,
                "current_linear": dev2 < 15 and dev4 < 15,
                "recommended_epsilon": recommendation["epsilon"],
                "recommended_signal": recommendation["signal"],
                "recommended_linear": recommendation["linear"],
                "target_reachable": recommendation["reachable"],
            }
        )

    return {"results": results, "sweeps": sweeps}


def main():
    calibration = run_calibration()
    model = create_model()

    print("=" * 100)
    print("EPSILON SENSITIVITY ANALYSIS — Finite-Difference Gradient Calibration")
    print("=" * 100)
    print(f"\nReference conditions: indoor={REF_INDOOR}°C, outdoor={REF_OUTDOOR}°C, "
          f"outlet={REF_OUTLET}°C, horizon={REF_HORIZON_H}h, step={REF_STEP_MIN}min, "
          f"PV history+forecast enabled")
    print()

    # === Part 1: Current epsilon analysis ===
    print("─" * 100)
    print("PART 1: Current Epsilon Values — Runtime Replay Signal Analysis")
    print("─" * 100)
    header = f"{'Parameter':<30} {'Default':>10} {'ε':>10} {'ε/Def%':>8} {'ΔT(°C)':>10} {'Gradient':>12}"
    print(header)
    print("─" * len(header))

    for row in calibration["results"]:
        rel_pct = (
            row["current_epsilon"] / abs(row["default"]) * 100
            if row["default"] != 0
            else float("inf")
        )
        print(
            f"{row['parameter']:<30} {row['default']:>10.4f} "
            f"{row['current_epsilon']:>10.4f} {rel_pct:>7.1f}% "
            f"{row['current_signal']:>10.6f} {row['current_gradient']:>12.6f}"
        )

    # === Part 2: Linearity check at current epsilon ===
    print()
    print("─" * 100)
    print("PART 2: Linearity Check (gradient at ε vs ε/2 vs ε/4)")
    print("─" * 100)
    header2 = f"{'Parameter':<30} {'g(ε)':>12} {'g(ε/2)':>12} {'g(ε/4)':>12} {'dev(ε/2)%':>10} {'dev(ε/4)%':>10} {'Linear?':>8}"
    print(header2)
    print("─" * len(header2))

    for row in calibration["results"]:
        param = row["parameter"]
        eps = row["current_epsilon"]
        g1, g2, g4, dev2, dev4 = check_linearity(
            model, param, eps, build_reference_prediction(param)
        )
        linear = "YES" if dev2 < 10 and dev4 < 10 else "NO"
        print(f"{param:<30} {g1:>12.6f} {g2:>12.6f} {g4:>12.6f} {dev2:>9.1f}% {dev4:>9.1f}% {linear:>8}")

    # === Part 3: Epsilon sweep ===
    print()
    print("─" * 100)
    print("PART 3: Epsilon Sweep (ΔT at various ε/default ratios)")
    print("─" * 100)
    sweep_ratios = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    for row in calibration["results"]:
        param = row["parameter"]
        default_val = row["default"]
        print(f"\n  {param} (default={default_val:.6f}):")
        print(f"    {'ε/Def%':>8} {'ε':>12} {'ΔT(°C)':>12} {'|ΔT|':>10} {'In target?':>12}")
        for sweep in calibration["sweeps"][param]:
            signal = sweep["signal"]
            in_target = "<<< YES" if 0.05 <= signal <= 0.5 else ""
            print(
                f"    {sweep['ratio']*100:>7.1f}% {sweep['epsilon']:>12.6f} "
                f"{signal:>12.6f} {signal:>10.6f} {in_target:>12}"
            )

    # === Part 4: Recommended epsilon (binary search for ΔT≈0.2°C) ===
    print()
    print("─" * 100)
    print("PART 4: Recommended Epsilon (target ΔT ≈ 0.2°C)")
    print("─" * 100)
    header4 = f"{'Parameter':<30} {'Current ε':>12} {'Recommended ε':>14} {'Achieved ΔT':>12} {'ε/Def%':>8} {'Linearity':>10}"
    print(header4)
    print("─" * len(header4))

    recommendations = {}
    for row in calibration["results"]:
        rel_pct = (
            row["recommended_epsilon"] / abs(row["default"]) * 100
            if row["default"] != 0
            else 0
        )
        linear = "LINEAR" if row["recommended_linear"] else "NONLIN"
        target_state = "OK" if row["target_reachable"] else "LOW-SIGNAL"
        print(
            f"{row['parameter']:<30} {row['current_epsilon']:>12.6f} "
            f"{row['recommended_epsilon']:>14.6f} {row['recommended_signal']:>12.6f} "
            f"{rel_pct:>7.1f}% {linear + '/' + target_state:>10}"
        )
        recommendations[row["parameter"]] = {
            "epsilon": row["recommended_epsilon"],
            "delta_t": row["recommended_signal"],
            "rel_pct": rel_pct,
            "linear": linear,
            "reachable": row["target_reachable"],
        }

    # === Part 5: Summary constants for PhysicsConstants ===
    print()
    print("─" * 100)
    print("PART 5: Proposed PhysicsConstants Values")
    print("─" * 100)
    print()
    for param, rec in recommendations.items():
        const_name = {
            "solar_lag_minutes": "SOLAR_LAG_EPSILON",
            "slab_time_constant_hours": "SLAB_TIME_CONSTANT_EPSILON",
        }.get(param, param.upper() + "_EPSILON")
        note = "target reached" if rec["reachable"] else "best reachable"
        print(
            f"    {const_name} = {rec['epsilon']:.6f}  "
            f"# ΔT≈{rec['delta_t']:.3f}°C, {rec['rel_pct']:.1f}% of default, "
            f"{rec['linear']}, {note}"
        )


if __name__ == "__main__":
    main()
