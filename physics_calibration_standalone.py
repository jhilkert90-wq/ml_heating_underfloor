#!/usr/bin/env python3
"""
Standalone physics calibration script for laptop execution.

Usage:
    # 1. On the Pi / Docker container, export calibration data:
    #    docker exec -it ml_heating_underfloor \
    #        python3 -m src.main --calibrate-physics-export-only
    #
    # 2. Copy the exported files to your laptop (scp / docker cp):
    #    docker cp ml_heating_underfloor:/opt/ml_heating/calibration_data.csv .
    #    docker cp ml_heating_underfloor:/opt/ml_heating/calibration_config.json .
    #    docker cp ml_heating_underfloor:/opt/ml_heating/unified_thermal_state.json .
    #
    # 3. Run calibration locally:
    #    python physics_calibration_standalone.py \
    #        --data calibration_data.csv \
    #        --config calibration_config.json \
    #        [--state unified_thermal_state.json] \
    #        [--output calibrated_params.json]
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ``import src.*`` works.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def _patch_config(config_json: dict):
    """Overwrite ``src.config`` module attributes with exported values."""
    import src.config as config

    for key, value in config_json.items():
        setattr(config, key, value)

    # Ensure data source is never "influx"/"ha_history" — we already
    # have the data in CSV, so skip any remote fetching.
    config.TRAINING_DATA_SOURCE = "influx"


def main():
    parser = argparse.ArgumentParser(
        description="Run physics calibration from exported CSV data.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to calibration_data.csv exported from Docker.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to calibration_config.json exported from Docker.",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Path to unified_thermal_state.json (optional).",
    )
    parser.add_argument(
        "--output",
        default="calibrated_params.json",
        help="Output path for calibrated parameters JSON (default: calibrated_params.json).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )

    # ------------------------------------------------------------------
    # 1. Load exported config and patch ``src.config``
    # ------------------------------------------------------------------
    logging.info("Loading config from %s", args.config)
    with open(args.config) as f:
        config_json = json.load(f)
    _patch_config(config_json)

    # ------------------------------------------------------------------
    # 2. Load exported CSV into DataFrame
    # ------------------------------------------------------------------
    logging.info("Loading calibration data from %s", args.data)
    df = pd.read_csv(args.data)
    if "_time" in df.columns:
        df["_time"] = pd.to_datetime(df["_time"], utc=True, format="ISO8601")
    logging.info(
        "Loaded %d rows × %d cols (%.1f hours)",
        len(df), len(df.columns), len(df) / 12,
    )

    # ------------------------------------------------------------------
    # 3. Import calibration functions (after config is patched)
    # ------------------------------------------------------------------
    from src.physics_calibration import (
        calibrate_cloud_factor,
        calibrate_delta_t_floor,
        calibrate_fp_decay_tau,
        calibrate_room_spread_delay,
        calibrate_slab_time_constant,
        calibrate_solar_decay_tau,
        calibrate_transient_parameters,
        calculate_cooling_time_constant,
        filter_cloudy_pv_periods,
        filter_fp_decay_periods,
        filter_fp_spread_periods,
        filter_pv_decay_periods,
        filter_stable_periods,
        filter_transient_periods,
        optimize_thermal_parameters,
    )
    from src.thermal_config import ThermalParameterConfig
    from src.thermal_equilibrium_model import ThermalEquilibriumModel
    import src.config as config

    # ------------------------------------------------------------------
    # 4. Run the calibration pipeline (mirrors train_thermal_equilibrium_model)
    # ------------------------------------------------------------------
    logging.info("=== STANDALONE CALIBRATION START ===")

    # Step 2: Filter stable periods
    logging.info("Step 2: Filtering for stable thermal equilibrium periods...")
    stable_periods = filter_stable_periods(df)
    if len(stable_periods) < 50:
        logging.error(
            "Insufficient stable periods: %d (need >= 50)", len(stable_periods),
        )
        sys.exit(1)
    logging.info("Found %d stable periods", len(stable_periods))

    # Step 3: Optimize thermal parameters
    logging.info("Step 3: Optimizing thermal parameters (scipy)...")
    optimized_params = optimize_thermal_parameters(stable_periods, df=df)
    if not optimized_params or not optimized_params.get("optimization_success"):
        logging.error("Parameter optimization failed")
        sys.exit(1)
    logging.info("Optimization MAE: %.4f°C", optimized_params["mae"])

    # Step 3b: Transient calibration (thermal_time_constant)
    logging.info("Step 3b: Transient calibration (thermal_time_constant)...")
    temp_model = ThermalEquilibriumModel()
    temp_model.heat_loss_coefficient = optimized_params["heat_loss_coefficient"]
    temp_model.outlet_effectiveness = optimized_params["outlet_effectiveness"]
    temp_model.external_source_weights["pv"] = optimized_params.get(
        "pv_heat_weight", ThermalParameterConfig.get_default("pv_heat_weight"),
    )
    temp_model.external_source_weights["fireplace"] = optimized_params.get(
        "fireplace_heat_weight",
        ThermalParameterConfig.get_default("fireplace_heat_weight"),
    )
    temp_model.external_source_weights["tv"] = optimized_params.get(
        "tv_heat_weight", ThermalParameterConfig.get_default("tv_heat_weight"),
    )
    temp_model.sync_heat_source_channels_from_model_state()

    transient_calibration_succeeded = False
    transient_samples = filter_transient_periods(df)
    if transient_samples:
        best_tau = calibrate_transient_parameters(temp_model, transient_samples)
        if best_tau:
            optimized_params["thermal_time_constant"] = best_tau
            transient_calibration_succeeded = True
            logging.info("Optimized thermal_time_constant: %.2fh", best_tau)
    else:
        logging.warning("No transient periods found")

    # Step 3c: Cooling time constant validation
    logging.info("Step 3c: Cooling time constant validation...")
    cooling_tau, cooling_r2 = calculate_cooling_time_constant(df)
    if cooling_tau:
        logging.info(
            "Cooling tau: %.2fh (R²=%.2f), Heating tau: %.2fh",
            cooling_tau, cooling_r2, optimized_params["thermal_time_constant"],
        )
        default_tau = ThermalParameterConfig.get_default("thermal_time_constant")
        if (
            not transient_calibration_succeeded
            and abs(optimized_params["thermal_time_constant"] - default_tau) < 0.1
            and cooling_r2 > 0.9
            and cooling_tau < 48.0
        ):
            optimized_params["thermal_time_constant"] = cooling_tau
            logging.info("Using cooling tau as thermal_time_constant")

    # Step 3d: Channel-specific calibrations
    logging.info("Step 3d: Channel-specific calibrations...")
    channel_params = {}

    fp_decay_periods = filter_fp_decay_periods(
        df,
        hlc=optimized_params["heat_loss_coefficient"],
        oe=optimized_params["outlet_effectiveness"],
    )
    fp_tau = calibrate_fp_decay_tau(fp_decay_periods)
    if fp_tau is not None:
        channel_params["fp_decay_time_constant"] = fp_tau

    fp_spread_periods = filter_fp_spread_periods(df)
    fp_spread_delay = calibrate_room_spread_delay(fp_spread_periods)
    if fp_spread_delay is not None:
        channel_params["room_spread_delay_minutes"] = fp_spread_delay

    dt_floor = calibrate_delta_t_floor(stable_periods)
    if dt_floor is not None:
        channel_params["delta_t_floor"] = dt_floor

    if getattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False):
        cloudy_periods = filter_cloudy_pv_periods(df)
        pv_w = optimized_params.get(
            "pv_heat_weight",
            ThermalParameterConfig.get_default("pv_heat_weight"),
        )
        cloud_exp = calibrate_cloud_factor(cloudy_periods, pv_w)
        if cloud_exp is not None:
            channel_params["cloud_factor_exponent"] = cloud_exp
    else:
        logging.info("Skipping cloud_factor_exponent (CLOUD_COVER_CORRECTION_ENABLED=false)")

    pv_decay_periods = filter_pv_decay_periods(df)
    solar_tau = calibrate_solar_decay_tau(pv_decay_periods)
    if solar_tau is not None:
        channel_params["solar_decay_tau_hours"] = solar_tau

    # Step 3e: Slab time constant calibration
    logging.info("Step 3e: Slab time constant calibration...")
    slab_tau = calibrate_slab_time_constant(df)
    if slab_tau is not None:
        channel_params["slab_time_constant_hours"] = slab_tau
        logging.info("Calibrated slab_time_constant_hours = %.2fh", slab_tau)

    if channel_params:
        logging.info(
            "Calibrated %d channel params: %s",
            len(channel_params),
            ", ".join(f"{k}={v:.3f}" for k, v in channel_params.items()),
        )

    # ------------------------------------------------------------------
    # 5. Assemble final calibrated parameters
    # ------------------------------------------------------------------
    calibrated_params = {
        "thermal_time_constant": optimized_params["thermal_time_constant"],
        "heat_loss_coefficient": optimized_params["heat_loss_coefficient"],
        "outlet_effectiveness": optimized_params["outlet_effectiveness"],
        "pv_heat_weight": optimized_params.get(
            "pv_heat_weight",
            ThermalParameterConfig.get_default("pv_heat_weight"),
        ),
        "fireplace_heat_weight": optimized_params.get(
            "fireplace_heat_weight",
            ThermalParameterConfig.get_default("fireplace_heat_weight"),
        ),
        "tv_heat_weight": optimized_params.get(
            "tv_heat_weight",
            ThermalParameterConfig.get_default("tv_heat_weight"),
        ),
        "solar_lag_minutes": optimized_params.get(
            "solar_lag_minutes",
            ThermalParameterConfig.get_default("solar_lag_minutes"),
        ),
        "slab_time_constant_hours": channel_params.get(
            "slab_time_constant_hours",
            ThermalParameterConfig.get_default("slab_time_constant_hours"),
        ),
        "optimization_mae": optimized_params["mae"],
        "stable_periods_count": len(stable_periods),
    }
    calibrated_params.update(channel_params)

    # ------------------------------------------------------------------
    # 6. Write output
    # ------------------------------------------------------------------
    with open(args.output, "w") as f:
        json.dump(calibrated_params, f, indent=2, default=float)
    logging.info("=== CALIBRATION COMPLETE ===")
    logging.info("Output written to %s", args.output)

    logging.info("\n=== CALIBRATED PARAMETERS ===")
    for k, v in calibrated_params.items():
        if isinstance(v, float):
            logging.info("  %s: %.6f", k, v)
        else:
            logging.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
