"""Offline comparison of physics-direct vs scipy calibration paths.

Runs both calibration paths on the saved stable_periods.json and prints
a side-by-side comparison of all calibrated parameters.

Usage
-----
    python test_calibration_compare.py [--hlc VALUE]

When ``--hlc`` is given, both paths lock HLC to that value.
Otherwise each path computes its own HLC from the stable periods.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Ensure the src/ package is importable
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Suppress noisy logs during model init
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import numpy as np

from src import config
from src.thermal_config import ThermalParameterConfig
from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src.physics_calibration import (
    _filter_hp_only_periods,
    _filter_pv_only_periods,
    calculate_direct_heat_loss,
    optimize_thermal_parameters,
    calibrate_delta_t_floor,
)
from src.physics_calibration_direct import (
    _calibrate_oe_analytical,
    _residual_heat_source_weight,
    _filter_fp_only_periods as direct_filter_fp,
    _filter_tv_only_periods as direct_filter_tv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_stable_periods(path: str):
    with open(path) as f:
        txt = f.read()
    txt = txt.replace("Infinity", "1e9").replace("-Infinity", "-1e9").replace("NaN", "null")
    periods = json.loads(txt)
    periods = [p for p in periods if p is not None]
    return periods


def separator(title: str):
    print(f"\n{'-'*70}")
    print(f"  {title}")
    print(f"{'-'*70}")


def _filter_fp_only(periods):
    return direct_filter_fp(periods)


def _filter_tv_only(periods):
    return direct_filter_tv(periods)


# ---------------------------------------------------------------------------
# Run physics-direct calibration (Step 2 onwards)
# ---------------------------------------------------------------------------

def run_physics_direct(periods, hlc_override=None):
    """Run the physics-direct calibration path on stable periods."""
    separator("PHYSICS-DIRECT CALIBRATION")

    # Step 1: HLC
    if hlc_override is not None:
        hlc = hlc_override
        print(f"  HLC (locked): {hlc:.5f} kW/K")
    else:
        hlc_direct = calculate_direct_heat_loss(periods)
        if hlc_direct:
            hlc = hlc_direct
            print(f"  HLC (direct P/ΔT): {hlc:.5f} kW/K")
        else:
            hlc = ThermalParameterConfig.get_default("heat_loss_coefficient")
            print(f"  HLC (default): {hlc:.5f} kW/K")

    # Step 2: OE
    oe = _calibrate_oe_analytical(periods, hlc)
    if oe is None:
        oe = ThermalParameterConfig.get_default("outlet_effectiveness")
    print(f"  OE: {oe:.5f} kW/K")

    # Step 4: PV weight
    hp_periods = _filter_hp_only_periods(periods)
    pv_periods = _filter_pv_only_periods(periods, hlc=hlc, oe=oe)
    pv_weight = _residual_heat_source_weight(
        pv_periods, "pv", hlc, oe, min_periods=5, percentile=50.0
    )
    if pv_weight is None:
        pv_weight = ThermalParameterConfig.get_default("pv_heat_weight")
    print(f"  PV weight: {pv_weight:.6f} kW/W")

    # Step 5: FP weight
    fp_periods = _filter_fp_only(periods)
    fp_weight = _residual_heat_source_weight(
        fp_periods, "fp", hlc, oe, min_periods=5, percentile=50.0
    )
    if fp_weight is None:
        fp_weight = ThermalParameterConfig.get_default("fireplace_heat_weight")
    print(f"  FP weight: {fp_weight:.5f} kW")

    # Step 6: TV weight
    tv_periods = _filter_tv_only(periods)
    tv_weight = _residual_heat_source_weight(
        tv_periods, "tv", hlc, oe, min_periods=5, percentile=60.0
    )
    if tv_weight is None:
        tv_weight = ThermalParameterConfig.get_default("tv_heat_weight")
    print(f"  TV weight: {tv_weight:.5f} kW")

    # Step 8: delta_t_floor
    dt_floor_result = calibrate_delta_t_floor(periods)
    dt_floor = dt_floor_result if dt_floor_result else 2.3
    print(f"  delta_t_floor: {dt_floor:.2f} °C")

    return {
        "heat_loss_coefficient": hlc,
        "outlet_effectiveness": oe,
        "pv_heat_weight": pv_weight,
        "fireplace_heat_weight": fp_weight,
        "tv_heat_weight": tv_weight,
        "delta_t_floor": dt_floor,
    }


# ---------------------------------------------------------------------------
# Run scipy calibration path
# ---------------------------------------------------------------------------

def run_scipy(periods, hlc_override=None):
    """Run the scipy multi-pass calibration path on stable periods."""
    separator("SCIPY MULTI-PASS CALIBRATION")

    # If HLC override, inject it via calculate_direct_heat_loss override
    if hlc_override is not None:
        # Monkey-patch calculate_direct_heat_loss to return the override
        import src.physics_calibration as pc
        original_cdhl = pc.calculate_direct_heat_loss
        pc.calculate_direct_heat_loss = lambda _: hlc_override
        print(f"  HLC (locked): {hlc_override:.5f} kW/K")

    result = optimize_thermal_parameters(periods, df=None, state_manager=None)

    if hlc_override is not None:
        pc.calculate_direct_heat_loss = original_cdhl

    if result is None:
        print("  ❌ Scipy optimization failed!")
        return None

    # Also get delta_t_floor
    dt_floor_result = calibrate_delta_t_floor(periods)
    dt_floor = dt_floor_result if dt_floor_result else 2.3

    params = {
        "heat_loss_coefficient": result["heat_loss_coefficient"],
        "outlet_effectiveness": result["outlet_effectiveness"],
        "pv_heat_weight": result["pv_heat_weight"],
        "fireplace_heat_weight": result["fireplace_heat_weight"],
        "tv_heat_weight": result.get("tv_heat_weight",
                                      ThermalParameterConfig.get_default("tv_heat_weight")),
        "delta_t_floor": dt_floor,
    }
    for k, v in params.items():
        print(f"  {k}: {v}")

    return params


# ---------------------------------------------------------------------------
# MAE evaluation
# ---------------------------------------------------------------------------

def evaluate_mae(periods, params, label=""):
    """Evaluate MAE of equilibrium predictions with given parameters."""
    model = ThermalEquilibriumModel()
    model.heat_loss_coefficient = params["heat_loss_coefficient"]
    model.outlet_effectiveness = params["outlet_effectiveness"]
    model.external_source_weights["pv"] = params["pv_heat_weight"]
    model.external_source_weights["fireplace"] = params["fireplace_heat_weight"]
    model.external_source_weights["tv"] = params["tv_heat_weight"]

    errors = []
    for p in periods:
        t_in = p.get("indoor_temp")
        if t_in is None or np.isnan(t_in):
            continue
        pv_input = p.get("pv_power_history", p.get("pv_power", 0))
        try:
            predicted = model.predict_equilibrium_temperature(
                outlet_temp=p.get("effective_temp", p.get("outlet_temp")),
                outdoor_temp=p.get("outdoor_temp"),
                current_indoor=t_in,
                pv_power=pv_input,
                fireplace_on=p.get("fireplace_on", 0),
                tv_on=p.get("tv_on", 0),
                thermal_power=None,
                _suppress_logging=True,
                cloud_cover_pct=0.0,
            )
            err = predicted - t_in
            if abs(err) < 50:
                errors.append(err)
        except Exception:
            continue

    errors = np.array(errors)
    if len(errors) == 0:
        return {}
    return {
        "label": label,
        "mae": np.mean(np.abs(errors)),
        "bias": np.mean(errors),
        "rmse": np.sqrt(np.mean(errors**2)),
        "n": len(errors),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare calibration paths")
    parser.add_argument("--hlc", type=float, default=None,
                        help="Lock HLC to this value [kW/K]")
    args = parser.parse_args()

    periods_path = os.path.join(ROOT, "Logs", "stable_periods.json")
    if not os.path.exists(periods_path):
        print(f"ERROR: {periods_path} not found")
        sys.exit(1)

    periods = load_stable_periods(periods_path)
    print(f"\n{'='*70}")
    print(f"  OFFLINE CALIBRATION COMPARISON")
    print(f"{'='*70}")
    print(f"  Stable periods: {len(periods)}")
    if args.hlc:
        print(f"  HLC locked to: {args.hlc:.5f} kW/K")

    # Collect environment-loaded previous values for reference
    prev = {}
    for key in ["heat_loss_coefficient", "outlet_effectiveness", "pv_heat_weight",
                 "fireplace_heat_weight", "tv_heat_weight", "delta_t_floor"]:
        prev[key] = getattr(config, key.upper(), ThermalParameterConfig.get_default(key))

    # ------- Run physics-direct -------
    phys = run_physics_direct(periods, hlc_override=args.hlc)

    # ------- Run scipy -------
    scipy_params = run_scipy(periods, hlc_override=args.hlc)

    # ------- MAE evaluation -------
    separator("MAE EVALUATION (full model, all periods)")

    hp_periods = _filter_hp_only_periods(periods)

    results = []
    if phys:
        for subset_name, subset in [("ALL", periods), ("HP-only", hp_periods)]:
            r = evaluate_mae(subset, phys, f"Physics-direct ({subset_name})")
            if r:
                results.append(r)
    if scipy_params:
        for subset_name, subset in [("ALL", periods), ("HP-only", hp_periods)]:
            r = evaluate_mae(subset, scipy_params, f"Scipy ({subset_name})")
            if r:
                results.append(r)
    if prev:
        for subset_name, subset in [("ALL", periods), ("HP-only", hp_periods)]:
            r = evaluate_mae(subset, prev, f"Previous/env ({subset_name})")
            if r:
                results.append(r)

    if results:
        print(f"\n  {'Label':<35} {'MAE':>8} {'Bias':>8} {'RMSE':>8} {'N':>6}")
        print(f"  {'-'*35} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
        for r in results:
            print(f"  {r['label']:<35} {r['mae']:8.4f} {r['bias']:+8.4f} {r['rmse']:8.4f} {r['n']:6d}")

    # ------- Side-by-side comparison -------
    separator("PARAMETER COMPARISON")

    param_keys = [
        ("heat_loss_coefficient", "HLC [kW/K]", 5),
        ("outlet_effectiveness", "OE [kW/K]", 5),
        ("pv_heat_weight", "PV weight [kW/W]", 6),
        ("fireplace_heat_weight", "FP weight [kW]", 5),
        ("tv_heat_weight", "TV weight [kW]", 5),
        ("delta_t_floor", "delta_t_floor [°C]", 2),
    ]

    defaults = {}
    for key, _, _ in param_keys:
        defaults[key] = ThermalParameterConfig.get_default(key)

    print(f"\n  {'Parameter':<25} {'Physics':>10} {'Scipy':>10} {'Previous':>10} {'Default':>10}")
    print(f"  {'-'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for key, label, dec in param_keys:
        fmt = f"{{:.{dec}f}}"
        p_val = fmt.format(phys[key]) if phys else "N/A"
        s_val = fmt.format(scipy_params[key]) if scipy_params else "N/A"
        pr_val = fmt.format(prev.get(key, 0))
        d_val = fmt.format(defaults[key])
        print(f"  {label:<25} {p_val:>10} {s_val:>10} {pr_val:>10} {d_val:>10}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
