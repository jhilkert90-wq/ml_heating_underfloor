#!/usr/bin/env python3
"""
Offline test of physics-direct calibration using stable_periods.json.

Runs all calibration steps that work on stable_periods data (no InfluxDB
or Home Assistant required).  Steps that require the raw time-series
DataFrame (solar_lag, slab_tau, transient tau, fp_decay, room_spread,
solar_decay) are skipped — they need live data.

Usage:
    python test_calibration_offline.py [--hlc 0.11864]

The HLC value from the last calibration run is used by default.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import argparse
from pathlib import Path

import numpy as np

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- Ensure src/ is importable as package ---
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def load_stable_periods(path: str) -> list:
    """Load stable_periods.json, handling Infinity values."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # JSON doesn't support Infinity — replace with a large number
    text = text.replace("Infinity", "1e9")
    text = text.replace("-Infinity", "-1e9")
    periods = json.loads(text)
    logging.info("Loaded %d stable periods from %s", len(periods), path)
    return periods


def main():
    parser = argparse.ArgumentParser(description="Offline physics-direct calibration test")
    parser.add_argument("--hlc", type=float, default=0.11864,
                        help="HLC value [kW/K] from last calibration (default: 0.11864)")
    parser.add_argument("--periods-file", type=str,
                        default=str(Path(__file__).parent / "Logs" / "stable_periods.json"),
                        help="Path to stable_periods.json")
    args = parser.parse_args()

    hlc = args.hlc

    # --- Load data ---
    stable_periods = load_stable_periods(args.periods_file)

    # --- Import calibration functions ---
    from src.physics_calibration_direct import (
        _calibrate_oe_analytical,
        _refine_oe_scipy,
        _residual_heat_source_weight,
        _filter_fp_only_periods,
        _filter_tv_only_periods,
    )
    from src.physics_calibration import (
        _filter_hp_only_periods,
        _filter_pv_only_periods,
        calibrate_delta_t_floor,
    )
    from src.thermal_config import ThermalParameterConfig

    print("\n" + "=" * 70)
    print("  OFFLINE PHYSICS-DIRECT CALIBRATION TEST")
    print("=" * 70)
    print(f"\n  Input: {len(stable_periods)} stable periods")
    print(f"  HLC (locked from prior calibration): {hlc:.5f} kW/K")

    # --- Analyze data coverage ---
    hp_only = _filter_hp_only_periods(stable_periods)
    fp_only = _filter_fp_only_periods(stable_periods)
    tv_only = _filter_tv_only_periods(stable_periods)

    # Count drives for the analytical OE
    drives = []
    for p in hp_only:
        t_in = p.get("indoor_temp")
        t_eff = p.get("effective_temp", p.get("outlet_temp"))
        if t_in is not None and t_eff is not None:
            d = t_eff - t_in
            if not math.isnan(d):
                drives.append(d)

    print(f"\n  --- Data Coverage ---")
    print(f"  HP-only periods:     {len(hp_only)}")
    print(f"  FP-only periods:     {len(fp_only)}")
    print(f"  TV-only periods:     {len(tv_only)}")
    if drives:
        print(f"  Drive (eff-indoor):  min={min(drives):.1f}, "
              f"median={np.median(drives):.1f}, max={max(drives):.1f} °C")
        print(f"  Drives ≥ 2°C:        {sum(1 for d in drives if d >= 2.0)}")
        print(f"  Drives ≥ 3°C:        {sum(1 for d in drives if d >= 3.0)}")

    # =========================================================================
    # STEP 2: OE calibration (analytical + scipy refinement)
    # =========================================================================
    print("\n" + "-" * 70)
    print("  STEP 2: OE CALIBRATION")
    print("-" * 70)

    oe = _calibrate_oe_analytical(stable_periods, hlc)

    if oe is not None:
        # Also show what the pure analytical value was (before scipy)
        # by re-running just the analytical part
        oe_values_debug = []
        for p in hp_only:
            t_in = p.get("indoor_temp")
            t_out = p.get("outdoor_temp")
            t_eff = p.get("effective_temp", p.get("outlet_temp"))
            if t_in is None or t_out is None or t_eff is None:
                continue
            if np.isnan(t_in) or np.isnan(t_out) or np.isnan(t_eff):
                continue
            drive = t_eff - t_in
            if drive < 3.0:
                continue
            delta_ti = t_in - t_out
            if delta_ti <= 0:
                continue
            oe_val = hlc * delta_ti / drive
            bounds = ThermalParameterConfig.get_bounds("outlet_effectiveness")
            if bounds[0] <= oe_val <= bounds[1]:
                oe_values_debug.append(oe_val)

        if oe_values_debug:
            print(f"\n  Analytical OE distribution ({len(oe_values_debug)} samples):")
            print(f"    P10 = {np.percentile(oe_values_debug, 10):.4f}")
            print(f"    P25 = {np.percentile(oe_values_debug, 25):.4f}")
            print(f"    P50 = {np.percentile(oe_values_debug, 50):.4f}")
            print(f"    P75 = {np.percentile(oe_values_debug, 75):.4f}")
            print(f"    P90 = {np.percentile(oe_values_debug, 90):.4f}")
            print(f"    Mean = {np.mean(oe_values_debug):.4f}")

        print(f"\n  >>> FINAL OE = {oe:.4f} kW/K")
        print(f"      (expected ~0.95, previous scipy result = 0.9527)")
    else:
        print("  >>> OE calibration FAILED — insufficient data")
        oe = ThermalParameterConfig.get_default("outlet_effectiveness")
        print(f"  >>> Using default: {oe:.4f}")

    # =========================================================================
    # STEP 4: PV heat weight
    # =========================================================================
    print("\n" + "-" * 70)
    print("  STEP 4: PV HEAT WEIGHT")
    print("-" * 70)

    pv_periods = _filter_pv_only_periods(stable_periods, hlc=hlc, oe=oe)
    print(f"  PV-active periods (after blind filter): {len(pv_periods)}")

    pv_weight = _residual_heat_source_weight(
        pv_periods, "pv", hlc, oe, min_periods=15, percentile=50.0
    )
    if pv_weight is not None:
        print(f"\n  >>> PV weight = {pv_weight:.6f} kW/W")
        print(f"      (previous = 0.000321, default = 0.002070)")
    else:
        print("  >>> PV weight calibration FAILED")

    # =========================================================================
    # STEP 5: Fireplace heat weight
    # =========================================================================
    print("\n" + "-" * 70)
    print("  STEP 5: FIREPLACE HEAT WEIGHT")
    print("-" * 70)

    print(f"  FP-only periods: {len(fp_only)}")
    fp_weight = _residual_heat_source_weight(
        fp_only, "fp", hlc, oe, min_periods=5, percentile=50.0
    )
    if fp_weight is not None:
        print(f"\n  >>> FP weight = {fp_weight:.3f} kW")
        print(f"      (previous = 0.381, default = 0.387)")
    else:
        print("  >>> FP weight calibration FAILED")

    # =========================================================================
    # STEP 6: TV heat weight
    # =========================================================================
    print("\n" + "-" * 70)
    print("  STEP 6: TV HEAT WEIGHT")
    print("-" * 70)

    print(f"  TV-only periods: {len(tv_only)}")
    tv_weight = _residual_heat_source_weight(
        tv_only, "tv", hlc, oe, min_periods=5, percentile=60.0
    )
    if tv_weight is not None:
        print(f"\n  >>> TV weight = {tv_weight:.3f} kW")
        print(f"      (previous = 0.531, default = 0.350)")
    else:
        print("  >>> TV weight calibration FAILED")

    # =========================================================================
    # STEP 8: delta_t_floor
    # =========================================================================
    print("\n" + "-" * 70)
    print("  STEP 8: DELTA_T_FLOOR")
    print("-" * 70)

    delta_t_floor = calibrate_delta_t_floor(stable_periods)
    if delta_t_floor is not None:
        print(f"\n  >>> delta_t_floor = {delta_t_floor:.2f} °C")
        print(f"      (previous = 2.04, default = 2.30)")
    else:
        print("  >>> delta_t_floor calibration FAILED")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("  CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"  {'Parameter':<30} {'New':>10} {'Previous':>10} {'Default':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    results = [
        ("HLC [kW/K]", hlc, 0.11864, 0.12452),
        ("OE [kW/K]", oe, 0.9527, 0.9527),
        ("PV weight [kW/W]", pv_weight, 0.000321, 0.002070),
        ("FP weight [kW]", fp_weight, 0.381, 0.387),
        ("TV weight [kW]", tv_weight, 0.531, 0.350),
        ("delta_t_floor [°C]", delta_t_floor, 2.04, 2.30),
    ]

    for name, new, prev, default in results:
        new_str = f"{new:.5f}" if new is not None else "FAILED"
        print(f"  {name:<30} {new_str:>10} {prev:>10.5f} {default:>10.5f}")

    print()
    print("  Steps NOT tested (require raw time-series DataFrame):")
    print("    - Step 3: thermal_time_constant (transient/cooling)")
    print("    - Step 7: solar_lag_minutes (PV xcorr)")
    print("    - Step 9: slab_time_constant_hours (grid search)")
    print("    - Step 10: fp_decay_time_constant")
    print("    - Step 11: room_spread_delay_minutes")
    print("    - Step 12: cloud_factor_exponent")
    print("    - Step 13: solar_decay_tau_hours")
    print("=" * 70)


if __name__ == "__main__":
    main()
