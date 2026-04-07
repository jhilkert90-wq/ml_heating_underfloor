"""
Physics Model Calibration for ML Heating Controller

This module provides calibration functionality for the realistic physics model
using historical target temperature data and actual house behavior.
"""

import logging
import json
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None
    logging.warning("scipy not available - optimization will be disabled")

# Support both package-relative and direct import for notebooks/scripts
try:
    from . import config
    from .thermal_equilibrium_model import ThermalEquilibriumModel
    from .state_manager import save_state
    from .influx_service import InfluxService
    from .thermal_config import ThermalParameterConfig
    from .unified_thermal_state import get_thermal_state_manager
except ImportError:
    # Direct import fallback for standalone execution
    import config
    from thermal_equilibrium_model import ThermalEquilibriumModel
    from state_manager import save_state
    from influx_service import InfluxService
    from thermal_config import ThermalParameterConfig
    from unified_thermal_state import get_thermal_state_manager


def train_thermal_equilibrium_model():
    """Train the Thermal Equilibrium Model with historical data for optimal
    thermal parameters using scipy optimization"""

    logging.info(
        "=== THERMAL EQUILIBRIUM MODEL TRAINING (SCIPY OPTIMIZATION) ==="
    )

    # Step 0: Backup existing calibration
    backup_existing_calibration()

    # Step 1: Fetch historical data
    logging.info("Step 1: Fetching historical data...")
    df = fetch_historical_data_for_calibration(
        lookback_hours=config.TRAINING_LOOKBACK_HOURS
    )

    if df is None or df.empty:
        logging.error("❌ Failed to fetch historical data")
        return None

    logging.info(f"✅ Retrieved {len(df)} samples ({len(df)/12:.1f} hours)")

    # Step 2: Filter for stable periods
    logging.info("Step 2: Filtering for stable thermal equilibrium periods...")
    stable_periods = filter_stable_periods(df)

    if len(stable_periods) < 50:
        logging.error(
            f"❌ Insufficient stable periods: {len(stable_periods)} "
            "(need at least 50)"
        )
        return None

    logging.info(
        f"✅ Found {len(stable_periods)} stable periods for calibration"
    )

    # Step 3: Optimize thermal parameters using scipy
    logging.info(
        "Step 3: Optimizing thermal parameters using scipy.optimize..."
    )
    optimized_params = optimize_thermal_parameters(stable_periods, df=df)

    if not optimized_params or not optimized_params.get(
        'optimization_success'
    ):
        logging.error("❌ Parameter optimization failed")
        return None

    logging.info(
        f"✅ Optimization completed - MAE: {optimized_params['mae']:.4f}°C"
    )

    # Step 3b: Optimize dynamic parameters (transient)
    logging.info("Step 3b: Optimizing dynamic parameters (thermal_time_constant)...")

    # Create a temporary model with optimized static params for transient calibration
    temp_model = ThermalEquilibriumModel()
    temp_model.heat_loss_coefficient = optimized_params['heat_loss_coefficient']
    temp_model.outlet_effectiveness = optimized_params['outlet_effectiveness']

    # Apply weights
    temp_model.external_source_weights['pv'] = optimized_params.get(
        'pv_heat_weight', ThermalParameterConfig.get_default('pv_heat_weight')
    )
    temp_model.external_source_weights['fireplace'] = optimized_params.get(
        'fireplace_heat_weight',
        ThermalParameterConfig.get_default('fireplace_heat_weight')
    )
    temp_model.external_source_weights['tv'] = optimized_params.get(
        'tv_heat_weight', ThermalParameterConfig.get_default('tv_heat_weight')
    )
    temp_model.sync_heat_source_channels_from_model_state()

    transient_calibration_succeeded = False
    transient_samples = filter_transient_periods(df)
    if transient_samples:
        best_tau = calibrate_transient_parameters(temp_model, transient_samples)
        if best_tau:
            optimized_params['thermal_time_constant'] = best_tau
            transient_calibration_succeeded = True
            logging.info(f"✅ Optimized thermal_time_constant: {best_tau:.2f}h")
    else:
        logging.warning("⚠️ No transient periods found for dynamic calibration")

    # Step 3c: Validate with cooling time constant (DHW periods)
    logging.info("Step 3c: Validating with cooling time constant...")
    cooling_tau, cooling_r2 = calculate_cooling_time_constant(df)

    # Maximum physically plausible cooling time constant for a house.
    # DHW periods are often too short to produce a reliable estimate — when
    # the house barely cools, log-linear regression returns a spuriously huge
    # tau. Cap at 48h to protect against that.
    MAX_PLAUSIBLE_COOLING_TAU = 48.0

    if cooling_tau:
        heating_tau = optimized_params['thermal_time_constant']
        diff_pct = abs(heating_tau - cooling_tau) / heating_tau * 100

        logging.info(
            f"Cooling time constant: {cooling_tau:.2f}h (R²={cooling_r2:.2f})"
        )
        logging.info(f"Heating time constant: {heating_tau:.2f}h")
        logging.info(f"Difference: {diff_pct:.1f}%")

        if diff_pct > 30.0:
            logging.warning(
                "⚠️ Significant difference between heating and cooling "
                "time constants (>30%)"
            )
            logging.warning(
                "This may indicate unmodeled heat gains or sensor issues."
            )

        # Only override with cooling tau if:
        # 1. Transient calibration did NOT succeed (heating tau is still the
        #    unchanged default — meaning we have no better estimate), AND
        # 2. The cooling tau is physically plausible (< MAX_PLAUSIBLE_COOLING_TAU),
        #    AND
        # 3. The cooling R² is high (reliable fit).
        # If transient calibration succeeded we trust its result even when
        # it happens to match the default value.
        default_tau = ThermalParameterConfig.get_default('thermal_time_constant')
        if (
            not transient_calibration_succeeded
            and abs(heating_tau - default_tau) < 0.1
            and cooling_r2 > 0.9
            and cooling_tau < MAX_PLAUSIBLE_COOLING_TAU
        ):
            logging.info(
                f"Transient calibration unavailable; using cooling tau "
                f"{cooling_tau:.2f}h (R²={cooling_r2:.2f})"
            )
            optimized_params['thermal_time_constant'] = cooling_tau
        elif cooling_tau >= MAX_PLAUSIBLE_COOLING_TAU:
            logging.warning(
                "⚠️ Cooling tau %.2fh exceeds physical limit (%.0fh) — "
                "DHW periods too short for reliable estimate, ignoring.",
                cooling_tau, MAX_PLAUSIBLE_COOLING_TAU,
            )

    # Step 3d: Calibrate channel-specific parameters
    # These parameters live in the heat source channels and cannot be
    # optimized from stable-period equilibrium data — they require
    # dedicated analysis of specific operating conditions.
    logging.info(
        "Step 3d: Calibrating channel-specific parameters "
        "(FP decay/spread, delta_t_floor, cloud exponent, solar decay)..."
    )
    channel_params = {}

    # --- Fireplace decay time constant ---
    fp_decay_periods = filter_fp_decay_periods(
        df,
        hlc=optimized_params.get('heat_loss_coefficient'),
        oe=optimized_params.get('outlet_effectiveness'),
    )
    fp_tau = calibrate_fp_decay_tau(fp_decay_periods)
    if fp_tau is not None:
        channel_params["fp_decay_time_constant"] = fp_tau

    # --- Fireplace room spread delay ---
    fp_spread_periods = filter_fp_spread_periods(df)
    fp_spread_delay = calibrate_room_spread_delay(fp_spread_periods)
    if fp_spread_delay is not None:
        channel_params["room_spread_delay_minutes"] = fp_spread_delay

    # --- delta_t_floor ---
    dt_floor = calibrate_delta_t_floor(stable_periods)
    if dt_floor is not None:
        channel_params["delta_t_floor"] = dt_floor

    # --- slab_time_constant_hours ---
    slab_tau = calibrate_slab_time_constant(df)
    if slab_tau is not None:
        channel_params["slab_time_constant_hours"] = slab_tau

    # --- Cloud factor exponent (only when cloud correction is enabled) ---
    if getattr(config, 'CLOUD_COVER_CORRECTION_ENABLED', False):
        cloudy_periods = filter_cloudy_pv_periods(df)
        pv_weight_for_cloud = optimized_params.get(
            "pv_heat_weight",
            ThermalParameterConfig.get_default("pv_heat_weight"),
        )
        cloud_exp = calibrate_cloud_factor(cloudy_periods, pv_weight_for_cloud)
        if cloud_exp is not None:
            channel_params["cloud_factor_exponent"] = cloud_exp
    else:
        logging.info(
            "Skipping cloud_factor_exponent calibration"
            " (CLOUD_COVER_CORRECTION_ENABLED=false)"
        )

    # --- Solar decay tau ---
    pv_decay_periods = filter_pv_decay_periods(df)
    solar_tau = calibrate_solar_decay_tau(pv_decay_periods)
    if solar_tau is not None:
        channel_params["solar_decay_tau_hours"] = solar_tau

    if channel_params:
        logging.info(
            "✅ Calibrated %d channel parameters: %s",
            len(channel_params),
            ", ".join(f"{k}={v:.3f}" for k, v in channel_params.items()),
        )
    else:
        logging.warning(
            "⚠️ No channel-specific parameters calibrated "
            "(insufficient data for all)"
        )

    # Step 4: Create thermal model with optimized parameters
    logging.info("Step 4: Creating thermal model with optimized parameters...")
    thermal_model = ThermalEquilibriumModel()

    # Apply optimized parameters to thermal model
    thermal_model.thermal_time_constant = optimized_params[
        'thermal_time_constant'
    ]
    thermal_model.heat_loss_coefficient = optimized_params[
        'heat_loss_coefficient'
    ]
    thermal_model.outlet_effectiveness = optimized_params[
        'outlet_effectiveness'
    ]

    # Apply heat source weights and lag
    pv_weight = optimized_params.get(
        'pv_heat_weight',
        ThermalParameterConfig.get_default('pv_heat_weight')
    )
    fireplace_weight = optimized_params.get(
        'fireplace_heat_weight',
        ThermalParameterConfig.get_default('fireplace_heat_weight')
    )
    tv_weight = optimized_params.get(
        'tv_heat_weight',
        ThermalParameterConfig.get_default('tv_heat_weight')
    )
    solar_lag = optimized_params.get(
        'solar_lag_minutes',
        ThermalParameterConfig.get_default('solar_lag_minutes')
    )

    thermal_model.external_source_weights['pv'] = pv_weight
    thermal_model.external_source_weights['fireplace'] = fireplace_weight
    thermal_model.external_source_weights['tv'] = tv_weight
    thermal_model.solar_lag_minutes = solar_lag
    thermal_model.sync_heat_source_channels_from_model_state()

    if thermal_model.orchestrator is not None:
        fireplace_channel = thermal_model.orchestrator.channels.get(
            'fireplace'
        )
        if fireplace_channel is not None:
            logging.info(
                "🔥 Channel sync seeded fp_heat_output_kw from calibrated "
                "fireplace_heat_weight: %.2f",
                fireplace_channel.fp_heat_output_kw,
            )

        # Apply channel-specific calibrated parameters (Step 3d results)
        if channel_params:
            _apply_channel_params(thermal_model.orchestrator, channel_params)
            logging.info(
                "✅ Applied %d channel-specific calibrated parameters",
                len(channel_params),
            )
        else:
            logging.info(
                "ℹ️ No channel-specific parameters to apply "
                "(all fell below data thresholds)"
            )

    # Set reasonable learning confidence based on optimization success
    thermal_model.learning_confidence = 3.0  # High confidence from scipy

    logging.info("\n=== OPTIMIZED THERMAL PARAMETERS ===")
    logging.info(
        f"Thermal time constant: {thermal_model.thermal_time_constant:.2f}h"
    )
    logging.info(
        f"Heat loss coefficient: {thermal_model.heat_loss_coefficient:.3f}"
    )
    logging.info(
        f"Outlet effectiveness: {thermal_model.outlet_effectiveness:.3f}"
    )
    logging.info(
        "PV heat weight: "
        f"{thermal_model.external_source_weights.get('pv', 0):.4f}"
    )
    logging.info(
        "Fireplace heat weight: "
        f"{thermal_model.external_source_weights.get('fireplace', 0):.2f}"
    )
    logging.info(
        "TV heat weight: "
        f"{thermal_model.external_source_weights.get('tv', 0):.2f}"
    )
    logging.info(f"Solar lag: {thermal_model.solar_lag_minutes:.1f} min")
    logging.info(f"Optimization MAE: {optimized_params['mae']:.4f}°C")
    logging.info(
        f"Learning confidence: {thermal_model.learning_confidence:.3f}"
    )

    # Step 5: Save thermal learning state to unified thermal state manager
    logging.info(
        "Step 5: Saving calibrated parameters to unified thermal state..."
    )
    try:
        state_manager = get_thermal_state_manager()

        # Use optimized parameters as calibrated baseline
        calibrated_params = {
            'thermal_time_constant': optimized_params['thermal_time_constant'],
            'heat_loss_coefficient': optimized_params['heat_loss_coefficient'],
            'outlet_effectiveness': optimized_params['outlet_effectiveness'],
            'pv_heat_weight': pv_weight,
            'fireplace_heat_weight': fireplace_weight,
            'tv_heat_weight': tv_weight,
            'solar_lag_minutes': solar_lag,
            # slab_time_constant_hours: prefer calibrated value from
            # channel_params (step response), else fall back to runtime value.
            'slab_time_constant_hours': channel_params.get(
                'slab_time_constant_hours',
                getattr(
                    thermal_model,
                    'slab_time_constant_hours',
                    ThermalParameterConfig.get_default('slab_time_constant_hours'),
                ),
            ),
        }

        # Merge channel-specific calibrated parameters into baseline
        calibrated_params.update(channel_params)

        # Set as calibrated baseline
        state_manager.set_calibrated_baseline(
            calibrated_params, calibration_cycles=len(stable_periods)
        )

        # Explicitly set confidence to 3.0 after calibration
        state_manager.update_learning_state(learning_confidence=3.0)
        thermal_model.sync_heat_source_channels_from_model_state(
            persist=True
        )

        logging.info(
            "✅ Calibrated parameters (scipy-optimized) saved to unified "
            "thermal state"
        )
        logging.info(
            "✅ Parameters will be automatically loaded on next restart"
        )
        logging.info(
            "🔄 Restart ml_heating service to use calibrated thermal model"
        )

    except Exception as e:
        logging.error(f"❌ Failed to save calibrated parameters: {e}")
        # Fallback to old method
        thermal_learning_state = {
            'thermal_time_constant': thermal_model.thermal_time_constant,
            'learning_confidence': thermal_model.learning_confidence,
        }
        save_state(thermal_learning_state=thermal_learning_state)
        logging.warning(
            "⚠️ Used fallback save method - parameters may not persist"
        )

    return thermal_model


def validate_thermal_model():
    """Validate thermal equilibrium model behavior across temperature ranges"""

    logging.info("=== THERMAL MODEL VALIDATION ===")

    try:
        # Import centralized thermal configuration

        # Initialize thermal model
        thermal_model = ThermalEquilibriumModel()

        logging.info("Testing thermal equilibrium physics compliance:")
        print("\nOUTLET TEMP → EQUILIBRIUM TEMP")
        print("=" * 35)

        # Test monotonicity
        monotonic_check = []
        outdoor_temp_test = 5.0  # Test outdoor temperature
        for outlet_temp in [25, 30, 35, 40, 45, 50, 55, 60]:
            equilibrium = thermal_model.predict_equilibrium_temperature(
                outlet_temp=outlet_temp,
                outdoor_temp=outdoor_temp_test,
                current_indoor=21.0,  # Test indoor temperature
                pv_power=0,
                fireplace_on=0,
                tv_on=0,
                _suppress_logging=True,
                cloud_cover_pct=0.0,
            )
            monotonic_check.append(equilibrium)
            print(f"{outlet_temp:3d}°C       → {equilibrium:.2f}°C")

        is_monotonic = all(monotonic_check[i] <= monotonic_check[i+1]
                           for i in range(len(monotonic_check)-1))

        print(f"\n{'✅' if is_monotonic else '❌'} Physics compliance: "
              f"{'PASSED' if is_monotonic else 'FAILED'}")
        print(f"Range: {min(monotonic_check):.2f}°C to "
              f"{max(monotonic_check):.2f}°C")

        # Test parameter bounds
        logging.info("\n=== THERMAL PARAMETER BOUNDS TEST ===")
        params_ok = True

        thermal_bounds = ThermalParameterConfig.get_bounds(
            'thermal_time_constant'
        )
        if not (thermal_bounds[0] <=
                thermal_model.thermal_time_constant <= thermal_bounds[1]):
            logging.error(
                "Thermal time constant out of bounds: "
                f"{thermal_model.thermal_time_constant:.2f}h "
                f"(bounds: {thermal_bounds})"
            )
            params_ok = False

        if params_ok:
            logging.info("✅ All thermal parameters within physical bounds")
        else:
            logging.error("❌ Some thermal parameters out of bounds")

        # Test adaptive learning system
        logging.info("\n=== ADAPTIVE LEARNING SYSTEM TEST ===")

        test_context = {
            'outlet_temp': 45.0,
            'outdoor_temp': 5.0,
            'pv_power': 0,
            'fireplace_on': 0,
            'tv_on': 0,
            'current_indoor': 21.0,
            'cloud_cover_pct': 50.0,
        }

        initial_confidence = thermal_model.learning_confidence

        # Simulate good predictions
        for _ in range(10):
            predicted = thermal_model.predict_equilibrium_temperature(
                **test_context,
                _suppress_logging=True
            )
            actual = predicted + 0.1  # Small error
            thermal_model.update_prediction_feedback(
                predicted_temp=predicted,
                actual_temp=actual,
                prediction_context=test_context
            )

        final_confidence = thermal_model.learning_confidence
        learning_works = final_confidence != initial_confidence

        logging.info(f"Initial confidence: {initial_confidence:.3f}")
        logging.info(f"Final confidence: {final_confidence:.3f}")
        logging.info(
            f"{'✅' if learning_works else '❌'} "
            "Adaptive learning: "
            f"{'WORKING' if learning_works else 'NOT WORKING'}"
        )

        overall_success = is_monotonic and params_ok and learning_works
        logging.info(
            f"\n{'✅' if overall_success else '❌'} "
            f"Overall validation: {'PASSED' if overall_success else 'FAILED'}"
        )

        return overall_success

    except Exception as e:
        logging.error("Thermal model validation error: %s", e, exc_info=True)
        return False


def fetch_historical_data_for_calibration(lookback_hours=672):
    """Fetch historical data for calibration.

    Respects ``config.TRAINING_DATA_SOURCE``:
    * ``"influx"``    – only InfluxDB
    * ``"ha_history"`` – only HA REST API history
    * ``"auto"``       – try InfluxDB first; if data is empty **or** missing
                         important columns, supplement / fall back to HA history
    """
    logging.info(f"=== FETCHING {lookback_hours} HOURS OF HISTORICAL DATA ===")

    source = getattr(config, "TRAINING_DATA_SOURCE", "auto").lower()

    df = pd.DataFrame()

    # --- InfluxDB path ---
    if source in ("influx", "auto"):
        try:
            influx = InfluxService(
                url=config.INFLUX_URL,
                token=config.INFLUX_TOKEN,
                org=config.INFLUX_ORG,
            )
            df = influx.get_training_data(lookback_hours=lookback_hours)
        except Exception as exc:
            logging.warning("InfluxDB fetch failed: %s", exc)
            df = pd.DataFrame()

        if not df.empty:
            logging.info(
                "✅ Fetched %d samples (%.1f hours) from InfluxDB",
                len(df), len(df) / 12,
            )

    # Columns we expect for full-featured calibration
    required_columns = [
        config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1],
        config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1],
        config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1],
        config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1],
        config.PV_POWER_ENTITY_ID.split(".", 1)[-1],
        config.TV_STATUS_ENTITY_ID.split(".", 1)[-1],
    ]
    important_optional_columns = [
        config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1],
        config.INLET_TEMP_ENTITY_ID.split(".", 1)[-1],
        config.FLOW_RATE_ENTITY_ID.split(".", 1)[-1],
        config.POWER_CONSUMPTION_ENTITY_ID.split(".", 1)[-1],
        config.DHW_STATUS_ENTITY_ID.split(".", 1)[-1],
    ]

    # --- HA history: full fallback (empty primary) or column supplement ---
    def _get_ha_df():
        """Lazy-load HA history DataFrame."""
        try:
            from .ha_history_service import get_training_data_from_ha
        except ImportError:
            from ha_history_service import get_training_data_from_ha  # type: ignore
        return get_training_data_from_ha(lookback_hours=lookback_hours)

    if df.empty and source in ("ha_history", "auto"):
        # Full fallback
        logging.info(
            "Fetching calibration data from HA history API "
            "(source=%s) …", source,
        )
        df = _get_ha_df()
        if not df.empty:
            logging.info(
                "✅ Fetched %d samples (%.1f hours) from HA history",
                len(df), len(df) / 12,
            )
    elif not df.empty and source == "auto":
        # InfluxDB returned data – check for missing columns we can fill
        all_wanted = required_columns + important_optional_columns
        missing_in_influx = [c for c in all_wanted if c not in df.columns]
        if missing_in_influx:
            logging.info(
                "InfluxDB data missing %d columns (%s) — "
                "attempting to supplement from HA history …",
                len(missing_in_influx),
                ", ".join(missing_in_influx),
            )
            ha_df = _get_ha_df()
            if not ha_df.empty:
                # Merge missing columns by nearest timestamp
                for col in missing_in_influx:
                    if col in ha_df.columns:
                        # Align on _time with merge_asof (±5 min tolerance)
                        supplement = ha_df[["_time", col]].dropna(
                            subset=[col]
                        )
                        if supplement.empty:
                            continue
                        df = pd.merge_asof(
                            df.sort_values("_time"),
                            supplement.sort_values("_time"),
                            on="_time",
                            tolerance=pd.Timedelta("5min"),
                            direction="nearest",
                        )
                        if col in df.columns and df[col].notna().sum() > 0:
                            logging.info(
                                "  ✅ Supplemented '%s' from HA history "
                                "(%d values)",
                                col, df[col].notna().sum(),
                            )
                        else:
                            logging.warning(
                                "  ⚠️  Column '%s' still empty after "
                                "HA supplement", col,
                            )
            else:
                logging.warning(
                    "⚠️  HA history returned no data — "
                    "proceeding without missing columns"
                )

    # ------------------------------------------------------------------
    # Per-entity time-coverage check (auto mode, InfluxDB data present)
    # ------------------------------------------------------------------
    # InfluxDB may not retain data for the full lookback window, or
    # individual entities may have started recording later.  Detect
    # per-entity gaps and supplement from HA where possible.
    # ------------------------------------------------------------------
    _META_COLS = frozenset({
        "_time", "result", "table", "_start", "_stop", "_measurement",
    })

    if not df.empty and source == "auto":
        expected_start = datetime.now(tz=timezone.utc) - timedelta(
            hours=lookback_hours,
        )
        data_cols = [c for c in df.columns if c not in _META_COLS]
        entities_with_gaps: dict = {}

        for col in data_cols:
            valid_mask = df[col].notna()
            if not valid_mask.any():
                continue  # entirely missing → handled by column check
            first_valid_time = df.loc[valid_mask.idxmax(), "_time"]
            gap_hours = (
                (first_valid_time - expected_start).total_seconds() / 3600
            )
            if gap_hours > 1:
                entities_with_gaps[col] = {
                    "first_valid_time": first_valid_time,
                    "gap_hours": gap_hours,
                }

        if entities_with_gaps:
            logging.info(
                "InfluxDB per-entity coverage gaps detected for %d "
                "entities (requested %dh):",
                len(entities_with_gaps),
                lookback_hours,
            )
            for col, info in entities_with_gaps.items():
                logging.info(
                    "  %s: missing first %.1fh (data starts %s)",
                    col, info["gap_hours"], info["first_valid_time"],
                )

            ha_df = _get_ha_df()

            if not ha_df.empty:
                # Extend the time grid if InfluxDB doesn't cover the
                # full lookback (all entities start late).
                influx_start = df["_time"].min()
                if influx_start > expected_start + timedelta(hours=1):
                    ha_early = ha_df[ha_df["_time"] < influx_start].copy()
                    if not ha_early.empty:
                        # Keep only _time + entity columns present in df
                        keep_cols = ["_time"] + [
                            c for c in df.columns
                            if c in ha_early.columns and c != "_time"
                        ]
                        ha_early = ha_early[[
                            c for c in keep_cols if c in ha_early.columns
                        ]]
                        df = pd.concat(
                            [ha_early, df], ignore_index=True,
                        )
                        df.sort_values("_time", inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        logging.info(
                            "  ✅ Extended time range by %.1fh from HA "
                            "(%d rows prepended)",
                            (influx_start - ha_early["_time"].min()
                             ).total_seconds() / 3600,
                            len(ha_early),
                        )

                # Fill per-entity NaN gaps from HA data
                for col, info in entities_with_gaps.items():
                    if col not in ha_df.columns:
                        logging.warning(
                            "  ⚠️  '%s' not available in HA history",
                            col,
                        )
                        continue
                    null_mask = df[col].isna()
                    if not null_mask.any():
                        continue
                    gap_rows = df.loc[null_mask, ["_time"]].copy()
                    ha_col = ha_df[["_time", col]].dropna(subset=[col])
                    if ha_col.empty:
                        logging.warning(
                            "  ⚠️  HA has no data for '%s'", col,
                        )
                        continue
                    filled = pd.merge_asof(
                        gap_rows.sort_values("_time"),
                        ha_col.sort_values("_time"),
                        on="_time",
                        tolerance=pd.Timedelta("5min"),
                        direction="nearest",
                    )
                    n_filled = filled[col].notna().sum()
                    df.loc[null_mask, col] = filled[col].values
                    logging.info(
                        "  ✅ Supplemented '%s' with %d HA values "
                        "(%.1fh gap)",
                        col, n_filled, info["gap_hours"],
                    )
            else:
                logging.warning(
                    "⚠️  HA history returned no data — "
                    "proceeding with incomplete entity coverage"
                )

    # ------------------------------------------------------------------
    # Final gap-filling pass
    # ------------------------------------------------------------------
    if not df.empty:
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    if df.empty:
        logging.error("❌ No training data available from any source")
        return None

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logging.error(f"❌ Missing required columns: {missing_cols}")
        return None

    # Log optional columns that are still absent
    fireplace_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    if fireplace_col not in df.columns:
        logging.info(
            f"⚠️  Optional fireplace column '{fireplace_col}' not found "
            "- will use 0"
        )

    logging.info("✅ All required columns present")
    return df


def main():
    """Main function to run training and validation."""
    logging.basicConfig(
        level=logging.INFO, format='%(levelname)s - %(message)s'
    )

    try:
        print("🚀 Starting thermal equilibrium model training...")
        thermal_model = train_thermal_equilibrium_model()

        if thermal_model:
            print("✅ Thermal training completed successfully!")

            print("\n🧪 Running thermal model validation...")
            validation_passed = validate_thermal_model()

            if validation_passed:
                print("✅ Thermal model validation PASSED!")
            else:
                print("❌ Thermal model validation FAILED!")

        else:
            print("❌ Thermal training failed!")
            return False

        return True

    except Exception as e:
        logging.error("Main execution error: %s", e, exc_info=True)
        return False


def filter_transient_periods(df, sequence_length_steps=12, min_temp_change=0.3):
    """
    Filter for transient periods and return multi-step sequences for tau calibration.

    WHY MULTI-STEP SEQUENCES:
    The thermal time constant tau determines how fast the house approaches its
    equilibrium temperature.  A single 5-minute step produces an approach factor
    of only 1-exp(-5min/tau):

        tau = 4h  -> approach = 2.1%  -> delta_T = 0.10 degC for 5 degC gap
        tau = 6h  -> approach = 1.4%  -> delta_T = 0.07 degC  (sensor noise!)
        tau = 16h -> approach = 0.5%  -> delta_T = 0.025 degC (completely hidden)

    Any tau > 5h is invisible in a single 5-minute step and the optimizer lands
    wherever it likes inside [5h, 100h] -- explaining the spurious 16.5h result.

    A 1-hour sequence at tau=4h gives approach=22%, tau=16h gives 6%.  The
    difference is large enough for scipy to converge to the right answer.

    Returns a list of sequences, each a list of consecutive 5-min rows spanning
    sequence_length_steps steps (~1 h by default: 12 * 5 min).
    Only sequences with a sustained indoor temperature change >= min_temp_change
    over the window are included (quasi-stable windows carry zero information
    about tau and bias the optimizer toward very large values).
    """
    logging.info("=== FILTERING FOR TRANSIENT PERIODS (multi-step sequences) ===")

    if df is None or df.empty:
        return []

    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    fireplace_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    tv_col = config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]
    inlet_col = config.INLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    actual_outlet_col = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]

    df = df.sort_values('_time').reset_index(drop=True)

    try:
        time_diffs = df['_time'].diff().dt.total_seconds() / 60.0
    except Exception as e:
        logging.warning("Could not calculate time differences: %s", e)
        return []

    sequences = []
    n = len(df)

    for start in range(0, n - sequence_length_steps):
        # Verify every step in the window is consecutive (4-11 min apart)
        ok = True
        for k in range(1, sequence_length_steps + 1):
            td = time_diffs.iloc[start + k]
            if np.isnan(td) or not (4.0 <= td <= 11.0):
                ok = False
                break
        if not ok:
            continue

        # Extract the window
        window = df.iloc[start: start + sequence_length_steps + 1]
        indoor_vals = window[indoor_col].values
        outlet_vals = window[outlet_col].values
        outdoor_vals = window[outdoor_col].values
        inlet_vals = (
            window[inlet_col].values
            if inlet_col in window.columns else None
        )
        actual_outlet_vals = (
            window[actual_outlet_col].values
            if actual_outlet_col in window.columns else None
        )

        if np.any(np.isnan(indoor_vals)) or np.any(np.isnan(outlet_vals)):
            continue

        # Require a meaningful temperature change over the whole window.
        # Without this, quasi-stable windows (delta~0) dominate and the
        # optimizer converges to tau->infinity (any large tau predicts ~0
        # change per step, matching quasi-stable observations perfectly).
        total_change = abs(indoor_vals[-1] - indoor_vals[0])
        if total_change < min_temp_change:
            continue

        # Build per-step records for the objective function
        steps = []
        for k in range(sequence_length_steps):
            t_diff_h = time_diffs.iloc[start + k + 1] / 60.0
            # Compute effective temp: (BT2 + BT3) / 2 when available
            if (actual_outlet_vals is not None and inlet_vals is not None
                    and not np.isnan(actual_outlet_vals[k])
                    and not np.isnan(inlet_vals[k])):
                eff_temp = (
                    float(np.asarray(actual_outlet_vals[k]).item()) + float(np.asarray(inlet_vals[k]).item())
                ) / 2.0
            else:
                eff_temp = float(np.asarray(outlet_vals[k]).item())

            steps.append({
                'current_indoor': float(np.asarray(indoor_vals[k]).item()),
                'next_indoor': float(np.asarray(indoor_vals[k + 1]).item()),
                'outlet_temp': float(np.asarray(outlet_vals[k]).item()),
                'effective_temp': eff_temp,
                'outdoor_temp': float(np.asarray(outdoor_vals[k]).item()),
                'pv_power': float(window.iloc[k].get(pv_col, 0) or 0),
                'fireplace_on': float(window.iloc[k].get(fireplace_col, 0) or 0),
                'tv_on': float(window.iloc[k].get(tv_col, 0) or 0),
                'time_step_hours': t_diff_h,
            })

        sequences.append({
            'steps': steps,
            'total_change': total_change,
            'start_indoor': float(np.asarray(indoor_vals[0]).item()),
            'end_indoor': float(np.asarray(indoor_vals[-1]).item()),
        })

    # Deduplicate overlapping windows: keep only every nth sequence so no two
    # sequences share more than half their steps.
    stride = max(1, sequence_length_steps // 2)
    sequences = sequences[::stride]

    logging.info(
        "Found %d transient sequences (len=%d steps, min_change=%.2f degC)",
        len(sequences), sequence_length_steps, min_temp_change
    )
    return sequences


def calculate_cooling_time_constant(df):
    """
    Estimate thermal time constant from cooling periods (HP-off).
    Detects HP-off via thermal power < 0.5 kW (outlet-inlet delta is small),
    falls back to DHW periods.
    Uses log-linear regression on (T_indoor - T_outdoor).
    Returns (tau, r_squared).
    """
    logging.info("=== CALCULATING COOLING TIME CONSTANT ===")

    if df is None or df.empty:
        return None, 0.0

    # Identify columns
    dhw_col = config.DHW_STATUS_ENTITY_ID.split(".", 1)[-1]
    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    flow_col = config.FLOW_RATE_ENTITY_ID.split(".", 1)[-1]
    inlet_col = config.INLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]

    # Work on a copy to avoid modifying the original dataframe
    df_cooling = df.copy()
    df_cooling = df_cooling.sort_values('_time')

    # Compute thermal_power_kw for HP-off detection
    can_compute_power = (
        flow_col in df_cooling.columns
        and inlet_col in df_cooling.columns
        and outlet_col in df_cooling.columns
    )
    if can_compute_power:
        delta_t = df_cooling[outlet_col] - df_cooling[inlet_col]
        df_cooling['_thermal_power_kw'] = (
            (df_cooling[flow_col] / 60.0)
            * config.SPECIFIC_HEAT_CAPACITY
            * delta_t
        ).clip(lower=0.0)
        logging.info("Using thermal-power based HP-off detection (power < 0.5 kW)")
        hp_off = df_cooling['_thermal_power_kw'] < 0.5
        df_cooling['cooling_group'] = (hp_off != hp_off.shift()).cumsum()
        group_col = 'cooling_group'
        filter_val = True  # select groups where hp_off is True
    elif dhw_col in df_cooling.columns:
        logging.info("Thermal power unavailable, falling back to DHW-based detection")
        df_cooling['dhw_group'] = (
            df_cooling[dhw_col] != df_cooling[dhw_col].shift()
        ).cumsum()
        group_col = 'dhw_group'
        filter_val = None  # will check dhw > 0
    else:
        logging.warning(
            "⚠️ Neither thermal power nor DHW column found - cannot estimate cooling constant"
        )
        return None, 0.0

    cooling_taus = []

    for _, group in df_cooling.groupby(group_col):
        # Filter: only HP-off groups (power-based) or DHW-on groups (DHW-based)
        if filter_val is True:
            if not (group['_thermal_power_kw'].iloc[0] < 0.5):
                continue
        else:
            if group[dhw_col].iloc[0] == 0:
                continue  # Not a DHW period

        if len(group) < 3:  # Need some points (assuming 5 min intervals)
            continue

        # Calculate time in hours from start of block
        times = (
            group['_time'] - group['_time'].iloc[0]
        ).dt.total_seconds() / 3600.0

        # Calculate log diff
        # T(t) - T_out = Delta * exp(-t/tau)
        # ln(T(t) - T_out) = ln(Delta) - t/tau
        # y = c + m*x
        # m = -1/tau => tau = -1/m

        # Use average outdoor temp for the block to simplify
        avg_outdoor = group[outdoor_col].mean()
        temp_diffs = group[indoor_col] - avg_outdoor

        # Filter for valid log inputs
        # Ensure indoor is at least 2 degrees above outdoor
        valid_indices = temp_diffs > 2.0
        if valid_indices.sum() < 3:
            continue

        y = np.log(temp_diffs[valid_indices])
        x = times[valid_indices]

        if len(x) < 3:
            continue

        # Linear regression
        # slope = cov(x, y) / var(x)
        # intercept = mean(y) - slope * mean(x)

        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)

        denominator = (n * sum_xx - sum_x * sum_x)
        if denominator == 0:
            continue

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Calculate R^2
        mean_y = np.mean(y)
        ss_tot = np.sum((y - mean_y)**2)
        ss_res = np.sum((y - (slope * x + (sum_y - slope * sum_x)/n))**2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Expecting cooling, so slope should be negative
        if slope >= -0.0005:  # Too flat or heating up
            continue

        tau = -1.0 / slope

        # Sanity check tau (0.5h to 72h).
        # DHW periods are typically short (20-60 min); the temperature drop
        # is tiny. A near-flat cooling curve produces a slope close to 0,
        # which maps to tau → ∞. Cap at 72h so such periods are discarded.
        if 0.5 < tau < 72.0 and r_squared > 0.6:
            cooling_taus.append({
                'tau': tau,
                'r2': r_squared,
                'duration': times.max()
            })

    if not cooling_taus:
        logging.info("No valid cooling periods found")
        return None, 0.0

    # Weighted average by R^2 * duration
    total_weight = sum(item['r2'] * item['duration'] for item in cooling_taus)
    weighted_tau = sum(
        item['tau'] * item['r2'] * item['duration'] for item in cooling_taus
    ) / total_weight

    avg_r2 = sum(item['r2'] for item in cooling_taus) / len(cooling_taus)

    logging.info(
        f"✅ Estimated cooling time constant: {weighted_tau:.2f}h "
        f"(from {len(cooling_taus)} periods, avg R²={avg_r2:.2f})"
    )

    return weighted_tau, avg_r2


def calibrate_transient_parameters(thermal_model, transient_sequences):
    """
    Calibrate thermal time constant from multi-step transient sequences.

    Each sequence in transient_sequences is a dict with a 'steps' list of
    consecutive 5-minute rows (see filter_transient_periods).  The objective
    simulates the full sequence with a candidate tau, rolling the indoor
    temperature step-by-step, and measures MSE against observed temperatures.

    Using multi-step rollouts (default ~1 h) means the cumulative approach
    factor differs significantly between candidate tau values:
        tau=4h  -> 1-exp(-1/4)  = 22%  cumulative approach over 1 h
        tau=8h  -> 1-exp(-1/8)  = 12%
        tau=16h -> 1-exp(-1/16) = 6%
    This gives scipy enough signal to converge to the physically correct tau.
    With single 5-minute steps the differences are < sensor noise and the
    optimizer is free to wander, producing spurious values like 16.5h.
    """
    logging.info("=== CALIBRATING TRANSIENT PARAMETERS (multi-step) ===")

    if not transient_sequences:
        logging.warning("No transient sequences available")
        return None

    if minimize is None:
        logging.error("scipy.optimize.minimize not available")
        return None

    def objective(params):
        tau = params[0]
        if tau <= 0:
            return 1e9
        total_mse = 0.0
        count = 0

        for seq in transient_sequences:
            steps = seq['steps']
            temp = steps[0]['current_indoor']

            for s in steps:
                eq_temp = thermal_model.predict_equilibrium_temperature(
                    outlet_temp=s.get('effective_temp', s['outlet_temp']),
                    outdoor_temp=s['outdoor_temp'],
                    current_indoor=temp,
                    pv_power=s['pv_power'],
                    fireplace_on=s['fireplace_on'],
                    tv_on=s['tv_on'],
                    thermal_power=None,   # force temperature-based formula
                    _suppress_logging=True,
                    cloud_cover_pct=0.0,
                )
                dt = s['time_step_hours']
                approach = 1.0 - np.exp(-dt / tau)
                temp = temp + (eq_temp - temp) * approach

                total_mse += (temp - s['next_indoor']) ** 2
                count += 1

        return total_mse / count if count > 0 else 1e9

    # Start from a physically reasonable guess (not the previous calibrated
    # value, which may be the wrong 16.5h).
    initial_tau = 4.0
    # Physically: 2h (lightweight partition) to 12h (well-insulated house).
    # Tighter bounds prevent the optimizer drifting to unrealistic values.
    bounds = [(2.0, 12.0)]

    logging.info(
        "Calibrating tau from %d sequences (initial guess: %.1fh)",
        len(transient_sequences), initial_tau
    )

    try:
        result = minimize(
            objective,
            [initial_tau],
            bounds=bounds,
            method='L-BFGS-B',
            options={'ftol': 1e-6, 'gtol': 1e-5, 'maxiter': 200},
        )

        if result.success or result.fun < 0.05:  # accept near-convergence too
            best_tau = float(result.x[0])
            logging.info(
                "Optimized thermal_time_constant: %.2fh (MSE: %.6f, success=%s)",
                best_tau, result.fun, result.success
            )
            # Sanity-check: warn if result is near a bound (may indicate poor fit)
            if best_tau < 2.5 or best_tau > 11.0:
                logging.warning(
                    "tau=%.2fh is near a calibration bound - verify data quality",
                    best_tau
                )
            return best_tau
        else:
            logging.error("Transient optimization failed: %s", result.message)
            return None
    except Exception as e:
        logging.error("Transient optimization error: %s", e)
        return None


def filter_stable_periods(df, temp_change_threshold=0.2, min_duration=20):
    """Filter for stable periods with blocking state detection."""
    logging.info(
        "=== FILTERING FOR STABLE PERIODS WITH BLOCKING STATE DETECTION ==="
    )

    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    target_outlet_col = config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    fireplace_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    tv_col = config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]

    # New sensors
    flow_rate_col = config.FLOW_RATE_ENTITY_ID.split(".", 1)[-1]
    inlet_col = config.INLET_TEMP_ENTITY_ID.split(".", 1)[-1]

    dhw_col = config.DHW_STATUS_ENTITY_ID.split(".", 1)[-1]
    defrost_col = config.DEFROST_STATUS_ENTITY_ID.split(".", 1)[-1]
    disinfect_col = config.DISINFECTION_STATUS_ENTITY_ID.split(".", 1)[-1]
    boost_col = config.DHW_BOOST_HEATER_STATUS_ENTITY_ID.split(".", 1)[-1]

    grace_period_minutes = config.GRACE_PERIOD_MAX_MINUTES
    grace_period_samples = int(grace_period_minutes) // 5

    # Pre-compute minutes since last defrost for each row.
    # Used later to exclude post-defrost slab-recovery periods from
    # OE / HLC calibration (the HP re-heats the slab before the room
    # reaches true steady-state, which biases OE downward).
    if defrost_col in df.columns:
        defrost_mask = df[defrost_col].fillna(0).astype(bool)
        # Build an integer index of the last defrost event per row.
        # Where defrost is active, record the row position; elsewhere NaN.
        # Forward-fill gives the position of the most recent defrost row.
        defrost_positions = pd.Series(np.where(defrost_mask, np.arange(len(df)), np.nan))
        last_defrost_pos = defrost_positions.ffill()
        samples_since = np.arange(len(df)) - last_defrost_pos.values
        # 5 min per sample
        df['_minutes_since_defrost'] = samples_since * 5
        # Rows before any defrost event → inf (never affected)
        df['_minutes_since_defrost'] = df['_minutes_since_defrost'].fillna(
            float('inf')
        )
    else:
        df['_minutes_since_defrost'] = float('inf')

    stable_periods = []
    window_size = int(min_duration) // 5

    logging.info(
        f"Looking for periods with <{temp_change_threshold}°C change"
        f" over {min_duration} min"
    )
    logging.info(
        f"Using {grace_period_minutes}min grace periods after blocking states"
    )

    filter_stats = {
        'total_checked': 0, 'missing_data': 0, 'temp_unstable': 0,
        'blocking_active': 0, 'grace_period': 0, 'fireplace_changed': 0,
        'outlet_unstable': 0, 'low_flow': 0, 'passed': 0
    }

    for i in range(window_size + grace_period_samples, len(df) - window_size):
        filter_stats['total_checked'] += 1

        window_start = i - window_size // 2
        window_end = i + window_size // 2
        window = df.iloc[window_start:window_end]

        grace_start = i - grace_period_samples
        grace_end = i + window_size // 2
        grace_window = df.iloc[grace_start:grace_end]

        indoor_temps = window[indoor_col].dropna()
        if len(indoor_temps) < window_size * 0.8:
            filter_stats['missing_data'] += 1
            continue

        temp_range = indoor_temps.max() - indoor_temps.min()
        temp_std = indoor_temps.std()

        if (temp_range > temp_change_threshold or
                temp_std > temp_change_threshold / 2):
            filter_stats['temp_unstable'] += 1
            continue

        blocking_detected, _ = check_blocking_states(
            grace_window, dhw_col, defrost_col, disinfect_col, boost_col
        )

        if blocking_detected:
            current_window_blocking, _ = check_blocking_states(
                window, dhw_col, defrost_col, disinfect_col, boost_col
            )
            if current_window_blocking:
                filter_stats['blocking_active'] += 1
            else:
                filter_stats['grace_period'] += 1
            continue

        if fireplace_col in window.columns:
            if window[fireplace_col].nunique() > 1:
                filter_stats['fireplace_changed'] += 1
                continue

        if outlet_col in window.columns:
            outlet_temps = window[outlet_col].dropna()
            if len(outlet_temps) >= window_size * 0.8:
                if outlet_temps.std() > 2.0:
                    filter_stats['outlet_unstable'] += 1
                    continue

        # Filter by flow rate if available (ensure pump is running)
        if flow_rate_col in window.columns:
            flow_rates = window[flow_rate_col].dropna()
            if not flow_rates.empty:
                avg_flow = flow_rates.mean()
                # If flow is too low (< 10 L/h), heat pump is likely off/idle
                if avg_flow < 10.0:
                    filter_stats['low_flow'] = (
                        filter_stats.get('low_flow', 0) + 1
                    )
                    continue

        center_row = df.iloc[i]

        # Calculate thermal power if available
        thermal_power_kw = 0.0
        if (inlet_col in df.columns and flow_rate_col in df.columns and
                outlet_col in df.columns):
            inlet_val = center_row.get(inlet_col)
            flow_val = center_row.get(flow_rate_col)
            outlet_val = center_row.get(outlet_col)

            if (inlet_val is not None and flow_val is not None and
                    outlet_val is not None):
                delta_t = outlet_val - inlet_val
                # Q = m * c * dt (flow in L/min -> L/s: / 60)
                thermal_power_kw = (
                    (flow_val / 60.0) *
                    config.SPECIFIC_HEAT_CAPACITY *
                    delta_t
                )

        # Extract PV history for solar lag calculation (max 3 hours = ~36 samples)
        pv_history = []
        if pv_col in df.columns:
            hist_start = max(0, i - 36)
            pv_history = df[pv_col].iloc[hist_start: i + 1].tolist()

        period = {
            'indoor_temp': center_row[indoor_col],
            'outlet_temp': center_row.get(target_outlet_col, center_row[outlet_col]),
            'outdoor_temp': center_row[outdoor_col],
            'pv_power': center_row.get(pv_col, 0.0),
            'pv_power_history': pv_history,
            'fireplace_on': center_row.get(fireplace_col, 0.0),
            'tv_on': center_row.get(tv_col, 0.0),
            'thermal_power_kw': thermal_power_kw,
            'minutes_since_defrost': center_row.get(
                '_minutes_since_defrost', float('inf')
            ),
            'timestamp': center_row['_time'],
            'stability_score': 1.0 / (temp_std + 0.01),
            'outlet_stability': 1.0 / (outlet_temps.std() + 0.01)
            if outlet_col in window.columns else 1.0
        }

        # Compute effective heating temperature matching runtime workflow:
        # When pump is running (guaranteed by flow_rate filter above),
        # effective = (BT2 + BT3) / 2 = mean water temperature in floor loop.
        actual_outlet_val = (
            center_row.get(outlet_col) if outlet_col in df.columns else None
        )
        actual_inlet_val = (
            center_row.get(inlet_col) if inlet_col in df.columns else None
        )
        if (actual_outlet_val is not None and actual_inlet_val is not None
                and not np.isnan(actual_outlet_val)
                and not np.isnan(actual_inlet_val)):
            period['inlet_temp'] = float(actual_inlet_val)
            period['effective_temp'] = (
                float(actual_outlet_val) + float(actual_inlet_val)
            ) / 2.0
        else:
            period['effective_temp'] = period['outlet_temp']

        stable_periods.append(period)
        filter_stats['passed'] += 1

    log_filtering_stats(filter_stats)

    logging.info(
        f"✅ Found {len(stable_periods)} stable periods with blocking "
        "state filtering"
    )

    with open("/opt/ml_heating/stable_periods.json", "w") as f:
        json.dump(stable_periods, f, indent=2, default=str)

    return stable_periods


def check_blocking_states(df, dhw_col, defrost_col, disinfect_col, boost_col):
    """Check for blocking states in a DataFrame."""
    blocking_detected = False
    blocking_reasons = []
    if dhw_col in df.columns and df[dhw_col].sum() > 0:
        blocking_detected = True
        blocking_reasons.append('dhw_heating')
    if defrost_col in df.columns and df[defrost_col].sum() > 0:
        blocking_detected = True
        blocking_reasons.append('defrosting')
    if disinfect_col in df.columns and df[disinfect_col].sum() > 0:
        blocking_detected = True
        blocking_reasons.append('disinfection')
    if boost_col in df.columns and df[boost_col].sum() > 0:
        blocking_detected = True
        blocking_reasons.append('boost_heater')
    return blocking_detected, blocking_reasons


def log_filtering_stats(stats):
    """Log statistics from the filtering process."""
    logging.info("=== BLOCKING STATE FILTERING RESULTS ===")
    logging.info(f"Total periods checked: {stats['total_checked']}")
    logging.info(f"Stable periods found: {stats['passed']}")
    logging.info("Filter exclusions:")
    logging.info(f"  Missing data: {stats['missing_data']}")
    logging.info(f"  Temperature unstable: {stats['temp_unstable']}")
    logging.info(f"  Blocking states active: {stats['blocking_active']}")
    logging.info(f"  Grace period recovery: {stats['grace_period']}")
    logging.info(f"  Fireplace state changes: {stats['fireplace_changed']}")
    logging.info(f"  Outlet temperature unstable: {stats['outlet_unstable']}")
    logging.info(f"  Low flow rate: {stats.get('low_flow', 0)}")

    retention = (stats['passed'] / stats['total_checked']) * 100 \
        if stats['total_checked'] > 0 else 0
    logging.info(f"Data retention rate: {retention:.1f}%")


def debug_thermal_predictions(stable_periods, sample_size=5):
    """Debug thermal model predictions on sample data."""
    logging.info("=== DEBUGGING THERMAL PREDICTIONS ===")

    test_model = ThermalEquilibriumModel()

    logging.info("Testing thermal model on sample periods:")
    for i, period in enumerate(stable_periods[:sample_size]):
        predicted_temp = test_model.predict_equilibrium_temperature(
            outlet_temp=period.get('effective_temp', period['outlet_temp']),
            outdoor_temp=period['outdoor_temp'],
            current_indoor=period.get(
                'indoor_temp', period['outdoor_temp'] + 10.0
            ),
            pv_power=period['pv_power'],
            fireplace_on=period['fireplace_on'],
            tv_on=period['tv_on'],
            thermal_power=None,  # Force temperature-based path for consistency
            _suppress_logging=True,
            cloud_cover_pct=0.0,
        )

        actual_temp = period['indoor_temp']
        error = abs(predicted_temp - actual_temp)

        logging.info(f"Sample {i+1}:")
        logging.info(
            f"  Outlet: {period['outlet_temp']:.1f}°C, "
            f"Outdoor: {period['outdoor_temp']:.1f}°C"
        )
        logging.info(
            "  PV: %.1fW, Fireplace: %.0f, "
            "TV: %.0f",
            period['pv_power'], period['fireplace_on'], period['tv_on']
        )
        logging.info(
            f"  Predicted: {predicted_temp:.1f}°C, "
            f"Actual: {actual_temp:.1f}°C"
        )
        logging.info(f"  Error: {error:.1f}°C")
        logging.info("")


def calculate_direct_heat_loss(stable_periods):
    """
    Calculate heat loss coefficient directly from periods with known thermal power
    and minimal external heat sources.

    U = P_thermal / (T_indoor - T_outdoor)
    """
    logging.info("=== CALCULATING DIRECT HEAT LOSS COEFFICIENT ===")

    u_values = []

    grace = config.DEFROST_RECOVERY_GRACE_MINUTES
    n_defrost_excluded = 0

    for p in stable_periods:
        # Check if we have thermal power data
        if 'thermal_power_kw' not in p or not p['thermal_power_kw']:
            continue

        # Filter for pure heating periods (minimal external sources)
        # Allow small PV/TV but ensure they are minor compared to heating
        if (p.get('pv_power', 0) > 100 or
                p.get('fireplace_on', 0) > 0 or
                p.get('tv_on', 0) > 0):
            continue

        # Ensure significant temperature difference and heating power
        # to avoid division by zero or noise amplification
        delta_t = p['indoor_temp'] - p['outdoor_temp']
        if delta_t < 5.0 or p['thermal_power_kw'] < 0.5:
            continue

        # Exclude post-defrost slab recovery and outlet ≤ inlet periods
        if (p.get('minutes_since_defrost', float('inf')) < grace
                or p.get('effective_temp', p.get('outlet_temp', 1))
                    <= p.get('inlet_temp', 0)):
            n_defrost_excluded += 1
            continue

        # Calculate U = P / dT
        u = p['thermal_power_kw'] / delta_t

        # Sanity check (0.05 to 1.0 kW/K are reasonable values for houses)
        if 0.05 <= u <= 1.0:
            u_values.append(u)

    if not u_values:
        logging.warning(
            "⚠️ No suitable periods found for direct heat loss calculation"
        )
        return None

    avg_u = sum(u_values) / len(u_values)
    std_u = (sum((x - avg_u) ** 2 for x in u_values) / len(u_values)) ** 0.5

    logging.info(f"✅ Calculated direct heat loss from {len(u_values)} periods")
    logging.info(f"  Mean U: {avg_u:.4f} kW/K")
    logging.info(f"  Std Dev: {std_u:.4f}")
    if n_defrost_excluded:
        logging.info(
            f"  Defrost-recovery excluded: {n_defrost_excluded} periods "
            f"(grace={grace} min)"
        )

    return avg_u


def _filter_hp_only_periods(stable_periods):
    """Return stable periods where only the heat pump is actively heating.

    Criteria:
    - PV < 100 W, fireplace off, TV off
    - HP delivering meaningful thermal power (≥ 0.5 kW)
    - Not in post-defrost slab recovery (outlet > inlet, grace elapsed)

    Without the power threshold HP-off periods (pump recirculating at
    ~19 W standby) pull OE to unrealistically low values.  Without the
    defrost grace, slab-recovery periods (HP re-heating slab after
    defrost stole heat) bias OE downward in cold weather.
    """
    grace = config.DEFROST_RECOVERY_GRACE_MINUTES
    filtered = [
        p for p in stable_periods
        if p.get('pv_power', 0) < 100
        and p.get('fireplace_on', 0) == 0
        and p.get('tv_on', 0) == 0
        and p.get('thermal_power_kw', 0) >= 0.5
        and p.get('minutes_since_defrost', float('inf')) >= grace
        and p.get('effective_temp', p.get('outlet_temp', 1))
            > p.get('inlet_temp', 0)
    ]
    n_defrost = sum(
        1 for p in stable_periods
        if p.get('thermal_power_kw', 0) >= 0.5
        and p.get('pv_power', 0) < 100
        and p.get('fireplace_on', 0) == 0
        and p.get('tv_on', 0) == 0
        and (p.get('minutes_since_defrost', float('inf')) < grace
             or p.get('effective_temp', p.get('outlet_temp', 1))
                <= p.get('inlet_temp', 0))
    )
    if n_defrost:
        logging.info(
            "  Defrost-recovery filter excluded %d HP-only periods "
            "(grace=%d min)", n_defrost, grace
        )
    return filtered


def _filter_hp_tv_periods(stable_periods):
    """Return stable periods where only HP (+ optionally TV) is active.

    Criteria: PV < 100 W, fireplace off. TV is allowed.
    Used for Pass 1 when TV weight is frozen to its config value.
    """
    return [
        p for p in stable_periods
        if p.get('pv_power', 0) < 100
        and p.get('fireplace_on', 0) == 0
    ]


def _filter_pv_only_periods(stable_periods, hlc=None, oe=None):
    """Return stable periods where PV is the only external source.

    Criteria: PV > 100 W, fireplace off, TV off.
    If *hlc* and *oe* are provided (from Pass 1), uses a residual-based
    blind detection heuristic: only keeps periods where actual indoor temp
    exceeds the HP-only equilibrium prediction, indicating PV is actively
    heating the room (blinds open).  Periods at or below the HP-only
    prediction are excluded (blinds likely closed, PV blocked).
    """
    all_pv = [
        p for p in stable_periods
        if p.get('pv_power', 0) > 100
        and p.get('fireplace_on', 0) == 0
        and p.get('tv_on', 0) == 0
    ]
    if hlc is not None and oe is not None and (hlc + oe) > 0:
        filtered = []
        excluded = 0
        for p in all_pv:
            outlet = p.get('effective_temp', p.get('outlet_temp', 25.0))
            outdoor = p.get('outdoor_temp', 10.0)
            hp_eq = (oe * outlet + hlc * outdoor) / (oe + hlc)
            actual = p.get('indoor_temp', 20.0)
            if actual > hp_eq + 0.1:
                filtered.append(p)
            else:
                excluded += 1
        if excluded > 0:
            logging.info(
                "PV filter: excluded %d/%d periods where indoor ≤ HP equilibrium"
                " (blinds likely closed)",
                excluded, len(all_pv),
            )
    else:
        filtered = all_pv
    return filtered


def _filter_pv_periods_from_df(df, hlc=None, oe=None):
    """Extract ALL rows with PV > 100 W from the raw DataFrame as period dicts.

    Unlike ``_filter_pv_only_periods`` this does NOT require stability
    filtering, capturing the strong PV signal visible during temperature
    changes.  Applies the same residual blind-detection heuristic when
    *hlc* and *oe* are available.
    """
    if df is None or df.empty:
        return []

    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    fireplace_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    tv_col = config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]
    inlet_col = config.INLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    actual_outlet_col = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]

    required = [indoor_col, outlet_col, outdoor_col, pv_col]
    for c in required:
        if c not in df.columns:
            logging.info(
                "Column %s missing — cannot build PV periods from df", c,
            )
            return []

    # Basic filter: PV > 100 W, fireplace off, no NaN in critical columns
    mask = (df[pv_col] > 100)
    if fireplace_col in df.columns:
        mask = mask & (df[fireplace_col] == 0)
    mask = (
        mask
        & df[indoor_col].notna()
        & df[outlet_col].notna()
        & df[outdoor_col].notna()
    )

    pv_df = df.loc[mask].copy()
    if pv_df.empty:
        return []

    # Effective temp: (BT2 + BT3) / 2 when both are available
    has_actual = actual_outlet_col and actual_outlet_col in pv_df.columns
    has_inlet = inlet_col in pv_df.columns
    if has_actual and has_inlet:
        both_ok = pv_df[actual_outlet_col].notna() & pv_df[inlet_col].notna()
        pv_df['_eff'] = pv_df[outlet_col].astype(float)
        pv_df.loc[both_ok, '_eff'] = (
            pv_df.loc[both_ok, actual_outlet_col].astype(float)
            + pv_df.loc[both_ok, inlet_col].astype(float)
        ) / 2.0
    else:
        pv_df['_eff'] = pv_df[outlet_col].astype(float)

    # Residual blind detection (same heuristic as _filter_pv_only_periods)
    excluded = 0
    if hlc is not None and oe is not None and (hlc + oe) > 0:
        hp_eq = (oe * pv_df['_eff'] + hlc * pv_df[outdoor_col]) / (oe + hlc)
        blind_mask = pv_df[indoor_col] <= hp_eq + 0.1
        excluded = int(blind_mask.sum())
        pv_df = pv_df[~blind_mask]

    if excluded > 0:
        logging.info(
            "PV df filter: excluded %d/%d rows where indoor ≤ HP equilibrium"
            " (blinds likely closed)",
            excluded, excluded + len(pv_df),
        )

    if pv_df.empty:
        return []

    # Sort by time and build positional index for PV history lookback
    pv_df = pv_df.sort_values('_time').reset_index(drop=True)

    # Pre-extract full PV column from the ORIGINAL df (sorted) for history
    # lookback.  The original df index lets us look back 36 samples (~3h).
    df_sorted = df.sort_values('_time').reset_index(drop=True)
    pv_all = df_sorted[pv_col].values.astype(float)
    # Map each pv_df row back to its position in the original sorted df
    # via the _time column for efficient lookback.
    orig_times = df_sorted['_time'].values
    pv_df_times = pv_df['_time'].values

    # Build period dicts compatible with _run_optimization_pass
    has_tv = tv_col in pv_df.columns
    periods = []
    search_start = 0  # sliding window for bisect-like search
    for idx in range(len(pv_df)):
        row = pv_df.iloc[idx]
        row_time = pv_df_times[idx]

        # Find position in original df for PV history lookback
        # (linear scan from last position — both are sorted by _time)
        while (search_start < len(orig_times) - 1
               and orig_times[search_start] < row_time):
            search_start += 1
        orig_idx = search_start

        # PV history: up to 36 samples lookback (3 hours at 5-min steps)
        hist_start = max(0, orig_idx - 36)
        pv_history = pv_all[hist_start:orig_idx + 1].tolist()

        periods.append({
            'indoor_temp': float(row[indoor_col]),
            'outlet_temp': float(row[outlet_col]),
            'outdoor_temp': float(row[outdoor_col]),
            'pv_power': float(row[pv_col]),
            'pv_power_history': pv_history,
            'fireplace_on': 0,
            'tv_on': float(row.get(tv_col, 0) or 0) if has_tv else 0,
            'effective_temp': float(row['_eff']),
        })

    logging.info(
        "PV df filter: %d periods from raw DataFrame (PV > 100W, FP off)",
        len(periods),
    )
    return periods


def _run_optimization_pass(
    param_names, param_values, param_bounds,
    periods, current_params, frozen_params=None,
    pass_label="",
):
    """Run a single L-BFGS-B optimisation pass.

    Returns a dict ``{param_name: optimised_value, ..., 'mae': float}``
    on success, or *None* on failure.
    """
    # Scaling factors for better convergence
    scaling_factors = []
    for name in param_names:
        if name == 'solar_lag_minutes':
            scaling_factors.append(100.0)
        elif name == 'pv_heat_weight':
            scaling_factors.append(0.001)
        else:
            scaling_factors.append(1.0)

    scaled_values = [v / s for v, s in zip(param_values, scaling_factors)]
    scaled_bounds = [
        (b[0] / s, b[1] / s) for b, s in zip(param_bounds, scaling_factors)
    ]

    # Reset call counter for each pass
    calculate_mae_for_params.call_count = 0

    def objective(scaled_params):
        real_params = [p * s for p, s in zip(scaled_params, scaling_factors)]
        return calculate_mae_for_params(
            real_params, param_names, periods, current_params,
            frozen_params=frozen_params,
        )

    label = f" ({pass_label})" if pass_label else ""
    logging.info(
        "Starting optimisation%s with %d periods, %d params: %s",
        label, len(periods), len(param_names), param_names,
    )

    try:
        result = minimize(
            objective,
            x0=scaled_values,
            bounds=scaled_bounds,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-6, 'disp': True, 'iprint': 2},
        )
        # Unscale
        result.x = result.x * np.array(scaling_factors)

        log_optimization_results(result, param_names, param_values)

        if result.success:
            out = {name: result.x[i] for i, name in enumerate(param_names)}
            out['mae'] = result.fun
            logging.info("✅ Pass%s MAE: %.4f°C", label, result.fun)
            return out

        logging.error("❌ Optimisation%s failed: %s", label, result.message)
        return None

    except Exception as e:
        logging.error("❌ Optimisation%s error: %s", label, e)
        return None


def optimize_thermal_parameters(stable_periods, df=None):
    """Multi-parameter optimization with data availability checks."""
    logging.info(
        "=== MULTI-PARAMETER OPTIMIZATION WITH DATA AVAILABILITY CHECKS ==="
    )

    if minimize is None:
        logging.error("❌ scipy not available - cannot optimize parameters")
        return None

    debug_thermal_predictions(stable_periods)

    logging.info("=== CHECKING DATA AVAILABILITY ===")

    total_periods = len(stable_periods)
    data_stats = {
        'pv_power': sum(
            1 for p in stable_periods if p.get('pv_power', 0) > 0
        ),
        'fireplace_on': sum(
            1 for p in stable_periods if p.get('fireplace_on', 0) > 0
        ),
        'tv_on': sum(1 for p in stable_periods if p.get('tv_on', 0) > 0)
    }

    for source, count in data_stats.items():
        percentage = (count / total_periods) * 100
        logging.info(
            f"  {source}: {count}/{total_periods} periods ({percentage:.1f}%)"
        )

    # Log effective temperature availability
    eff_count = sum(1 for p in stable_periods if 'inlet_temp' in p)
    logging.info(
        f"  effective_temp: {eff_count}/{total_periods} periods "
        f"({eff_count/total_periods*100:.0f}%) — using (BT2+BT3)/2"
    )

    # Try to calculate direct heat loss first
    direct_u = calculate_direct_heat_loss(stable_periods)

    initial_heat_loss = ThermalParameterConfig.get_default(
        'heat_loss_coefficient'
    )
    if direct_u is not None:
        initial_heat_loss = direct_u
        logging.info(
            f"Using calculated heat loss {direct_u:.4f} as initial guess"
        )

    current_params = {
        'thermal_time_constant':
            ThermalParameterConfig.get_default('thermal_time_constant'),
        'heat_loss_coefficient': initial_heat_loss,
        'outlet_effectiveness':
            ThermalParameterConfig.get_default('outlet_effectiveness'),
        'pv_heat_weight':
            ThermalParameterConfig.get_default('pv_heat_weight'),
        'fireplace_heat_weight':
            ThermalParameterConfig.get_default('fireplace_heat_weight'),
        'tv_heat_weight': ThermalParameterConfig.get_default('tv_heat_weight'),
        'solar_lag_minutes':
            ThermalParameterConfig.get_default('solar_lag_minutes')
    }

    logging.getLogger().setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Pass 1: OE on HP-only periods (HLC fixed from energy balance)
    # When direct_u is available, HLC is frozen at the physics-anchored
    # value to break the HLC/OE degeneracy.  Falls back to co-optimising
    # HLC+OE if the direct calculation failed.
    # ------------------------------------------------------------------
    fix_hlc = direct_u is not None
    pass_label = "Pass 1 OE-only" if fix_hlc else "Pass 1 HLC+OE"
    logging.info("=== %s (HP-only periods) ===", pass_label)

    hp_periods = _filter_hp_only_periods(stable_periods)
    MIN_HP_PERIODS = 10
    if len(hp_periods) < MIN_HP_PERIODS:
        logging.warning(
            "⚠️ Only %d HP-only periods (need %d) — "
            "falling back to all periods for Pass 1",
            len(hp_periods), MIN_HP_PERIODS,
        )
        hp_periods = stable_periods

    excluded_params = []  # kept for build_optimization_params signature
    param_names, param_values, param_bounds = build_optimization_params(
        current_params, excluded_params,
        heat_loss_center=direct_u,
        fix_hlc=fix_hlc,
    )

    # Freeze all heat-source weights to zero so they cannot contaminate
    # the OE estimate.  When fix_hlc, also freeze HLC at the direct value.
    frozen_pass1 = {
        'pv_heat_weight': 0.0,
        'fireplace_heat_weight': 0.0,
        'tv_heat_weight': 0.0,
        'solar_lag_minutes': 0.0,
    }
    if fix_hlc:
        frozen_pass1['heat_loss_coefficient'] = direct_u

    pass1 = _run_optimization_pass(
        param_names, param_values, param_bounds,
        hp_periods, current_params, frozen_params=frozen_pass1,
        pass_label=pass_label,
    )
    if pass1 is None:
        logging.error("❌ %s failed", pass_label)
        return None

    hlc = direct_u if fix_hlc else pass1['heat_loss_coefficient']
    oe = pass1['outlet_effectiveness']
    pass1_mae = pass1['mae']
    logging.info(
        "✅ Pass 1 result: HLC=%.4f%s, OE=%.4f  (MAE %.4f°C)",
        hlc, " (fixed)" if fix_hlc else "", oe, pass1_mae,
    )

    # ------------------------------------------------------------------
    # Pass 2: PV weight on PV-only periods (HLC/OE/solar_lag locked)
    # solar_lag_minutes is fixed at its config default — cross-correlation
    # is unreliable for UFH systems where the slab buffers the PV→room
    # signal, and slab_time_constant_hours models the delay physics.
    # When a raw DataFrame is available, ALL PV rows are used (not just
    # stability-filtered ones) to capture the strong PV signal that the
    # stability filter excludes.
    # ------------------------------------------------------------------
    if df is not None:
        pv_periods = _filter_pv_periods_from_df(df, hlc=hlc, oe=oe)
        if len(pv_periods) < 5:
            logging.info(
                "Only %d PV periods from df — falling back to stable-period filter",
                len(pv_periods),
            )
            pv_periods = _filter_pv_only_periods(stable_periods, hlc=hlc, oe=oe)
    else:
        pv_periods = _filter_pv_only_periods(stable_periods, hlc=hlc, oe=oe)
    MIN_PV_PERIODS = 5
    pv_weight = current_params['pv_heat_weight']
    solar_lag = current_params['solar_lag_minutes']  # kept at config default

    if len(pv_periods) >= MIN_PV_PERIODS:
        frozen_pass2 = {
            'heat_loss_coefficient': hlc,
            'outlet_effectiveness': oe,
            'fireplace_heat_weight': 0.0,
            'tv_heat_weight': current_params['tv_heat_weight'],
            'solar_lag_minutes': solar_lag,
        }
        logging.info(
            "=== PASS 2: PV weight only (%d PV-only periods, solar_lag frozen=%.0f) ===",
            len(pv_periods), solar_lag,
        )
        pv_names, pv_values, pv_bounds = _build_pv_params(current_params)

        pass2 = _run_optimization_pass(
            pv_names, pv_values, pv_bounds,
            pv_periods, current_params, frozen_params=frozen_pass2,
            pass_label="Pass 2 PV",
        )
        if pass2 is not None:
            pv_weight = pass2['pv_heat_weight']
            logging.info(
                "✅ Pass 2 result: pv_weight=%.5f, solar_lag=%.1f min (frozen)",
                pv_weight, solar_lag,
            )
        else:
            logging.warning(
                "⚠️ Pass 2 (PV) failed — using default pv_weight"
            )
    else:
        logging.info(
            "⚠️ Only %d PV-only periods (need %d) — "
            "skipping Pass 2, using default PV params",
            len(pv_periods), MIN_PV_PERIODS,
        )

    # ------------------------------------------------------------------
    # Pass 3: FP weight on FP-active periods (HLC/OE/PV locked)
    # ------------------------------------------------------------------
    fp_periods = [
        p for p in stable_periods
        if p.get('fireplace_on', 0) > 0
        and p.get('pv_power', 0) < 100  # avoid PV contamination
    ]
    MIN_FP_PERIODS = 5
    fp_weight = current_params['fireplace_heat_weight']

    if len(fp_periods) >= MIN_FP_PERIODS:
        logging.info(
            "=== PASS 3: FP weight (%d FP-active periods) ===",
            len(fp_periods),
        )
        fp_names = ['fireplace_heat_weight']
        fp_values = [current_params['fireplace_heat_weight']]
        fp_bounds = [(0.1, 5.0)]

        frozen_pass3 = {
            'heat_loss_coefficient': hlc,
            'outlet_effectiveness': oe,
            'pv_heat_weight': pv_weight,
            'solar_lag_minutes': solar_lag,
            'tv_heat_weight': current_params['tv_heat_weight'],
        }

        pass3 = _run_optimization_pass(
            fp_names, fp_values, fp_bounds,
            fp_periods, current_params, frozen_params=frozen_pass3,
            pass_label="Pass 3 FP",
        )
        if pass3 is not None:
            fp_weight = pass3['fireplace_heat_weight']
            logging.info(
                "✅ Pass 3 result: fireplace_heat_weight=%.4f",
                fp_weight,
            )
        else:
            logging.warning(
                "⚠️ Pass 3 (FP) failed — using default fireplace_heat_weight"
            )
    else:
        logging.info(
            "⚠️ Only %d FP-active periods (need %d) — "
            "skipping Pass 3, using default FP weight",
            len(fp_periods), MIN_FP_PERIODS,
        )

    # ------------------------------------------------------------------
    # Merge results
    # ------------------------------------------------------------------
    optimized_params = dict(current_params)
    optimized_params['heat_loss_coefficient'] = hlc
    optimized_params['outlet_effectiveness'] = oe
    optimized_params['pv_heat_weight'] = pv_weight
    optimized_params['solar_lag_minutes'] = solar_lag
    optimized_params['fireplace_heat_weight'] = fp_weight
    optimized_params['mae'] = pass1_mae
    optimized_params['optimization_success'] = True
    optimized_params['excluded_parameters'] = []

    log_optimized_parameters(optimized_params, current_params, [])

    return optimized_params


def build_optimization_params(
    current_params, excluded_params, heat_loss_center=None,
    fix_hlc=False,
):
    """Build lists of parameters for optimization.

    When *fix_hlc* is True **and** *heat_loss_center* is provided, HLC is
    excluded from the optimisation list (it will be frozen via
    ``frozen_params`` instead).  This breaks the HLC/OE degeneracy by
    anchoring HLC from the energy-balance calculation.
    """
    param_names = []
    param_values = []
    param_bounds = []

    core_params = [
        'heat_loss_coefficient',
        'outlet_effectiveness'
    ]

    # Note: thermal_time_constant cannot be calibrated using stable (equilibrium)
    # periods because the time-dependent exponential term decays to zero.
    # It requires dynamic/transient data for calibration.

    for p in core_params:
        # When fix_hlc is requested and we have a physics anchor, skip HLC
        if p == 'heat_loss_coefficient' and fix_hlc and heat_loss_center is not None:
            logging.info(
                "  Fixing heat_loss_coefficient = %.4f from energy balance"
                " (not optimised)",
                heat_loss_center,
            )
            continue

        param_names.append(p)
        param_values.append(current_params[p])

        if p == 'heat_loss_coefficient' and heat_loss_center is not None:
            # Constrain to +/- 10% of calculated value (HLC and OE are
            # degenerate in the temp-based model; anchor HLC tightly)
            lower = heat_loss_center * 0.9
            upper = heat_loss_center * 1.1
            # But keep within absolute physical bounds
            abs_bounds = ThermalParameterConfig.get_bounds(p)
            final_lower = max(lower, abs_bounds[0])
            final_upper = min(upper, abs_bounds[1])
            param_bounds.append((final_lower, final_upper))
            logging.info(
                f"  Constraining heat_loss_coefficient to {final_lower:.4f}-"
                f"{final_upper:.4f} based on direct calculation"
            )
        elif p == 'heat_loss_coefficient':
            # Use strict bounds from config to ensure physical realism
            default_bounds = ThermalParameterConfig.get_bounds(p)
            param_bounds.append(default_bounds)
            logging.info(
                f"  Using strict bounds for heat_loss_coefficient: "
                f"{default_bounds[0]}-{default_bounds[1]}"
            )
        else:
            param_bounds.append(ThermalParameterConfig.get_bounds(p))

    # Heat-source weights are NO LONGER co-optimised with HLC/OE.
    # PV is calibrated in an isolated Pass 2 (_build_pv_params).
    # FP weight comes from Step 3d channel calibration.
    # TV weight is fixed at default.

    return param_names, param_values, param_bounds


def _calibrate_solar_lag_crosscorr(df):
    """Estimate solar_lag_minutes via cross-correlation of PV vs d(indoor)/dt.

    Returns the lag in minutes that maximises the correlation, or None if
    insufficient data.  The search range is 0–180 min in 5-min steps.
    """
    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]

    if df is None or pv_col not in df.columns or indoor_col not in df.columns:
        return None

    df_s = df.sort_values('_time').reset_index(drop=True)
    pv = df_s[pv_col].values.astype(float)
    indoor = df_s[indoor_col].values.astype(float)

    # Indoor temp rate of change (forward diff, per 5-min step)
    d_indoor = np.diff(indoor)
    pv_trimmed = pv[:-1]  # align lengths

    if len(d_indoor) < 72:  # need at least 6 hours of data
        return None

    max_lag_steps = 36  # 180 min / 5 min
    best_lag = 0
    best_corr = -np.inf

    for lag in range(max_lag_steps + 1):
        if lag == 0:
            x = pv_trimmed
            y = d_indoor
        else:
            x = pv_trimmed[:-lag]
            y = d_indoor[lag:]

        if len(x) < 36:
            break

        # Pearson correlation (handle constant arrays)
        x_std = np.std(x)
        y_std = np.std(y)
        if x_std < 1e-10 or y_std < 1e-10:
            continue

        corr = np.corrcoef(x, y)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    lag_minutes = best_lag * 5.0
    logging.info(
        "Solar lag cross-correlation: best_lag=%.0f min (r=%.3f)",
        lag_minutes, best_corr,
    )
    if best_corr < 0.05:
        logging.warning("Solar lag correlation too weak (r=%.3f) — ignoring", best_corr)
        return None

    return lag_minutes


def _build_pv_params(current_params):
    """Build parameter lists for the PV-only optimisation pass.

    solar_lag_minutes is frozen (not calibrated) — the slab time constant
    models the thermal delay physics more accurately for UFH systems.
    """
    param_names = ['pv_heat_weight']
    param_values = [current_params['pv_heat_weight']]
    param_bounds = [(0.0001, 0.005)]
    return param_names, param_values, param_bounds


def calculate_mae_for_params(
    params, param_names, stable_periods, current_params,
    frozen_params=None,
):
    """Calculate MAE for a given set of parameters.

    ``frozen_params`` is an optional dict of parameter-name → value that
    take precedence over both *param_dict* (the values being optimised)
    and *current_params* (the defaults).  This is used by the isolated-pass
    optimiser to lock parameters that must not change in a given pass
    (e.g. pv/fp/tv weights set to 0 during the HLC+OE pass).
    """
    total_error = 0.0
    valid_predictions = 0

    param_dict = dict(zip(param_names, params))
    if frozen_params:
        param_dict.update(frozen_params)

    debug_str = ", ".join(
        [f"{name}={val:.4f}" for name, val in param_dict.items()]
    )
    if not hasattr(calculate_mae_for_params, 'call_count'):
        calculate_mae_for_params.call_count = 1
    else:
        calculate_mae_for_params.call_count += 1

    if calculate_mae_for_params.call_count % 10 == 1:
        logging.debug("Testing params: %s", debug_str)

    test_periods = stable_periods[::5]

    # Suppress INFO logs from ThermalEquilibriumModel initialization
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.WARNING)

    # Create model once and update params (avoids re-init per period)
    test_model = ThermalEquilibriumModel()
    test_model.thermal_time_constant = param_dict.get(
        'thermal_time_constant',
        current_params['thermal_time_constant']
    )
    test_model.heat_loss_coefficient = param_dict[
        'heat_loss_coefficient'
    ]
    test_model.outlet_effectiveness = param_dict[
        'outlet_effectiveness'
    ]

    test_model.external_source_weights['pv'] = param_dict.get(
        'pv_heat_weight', current_params['pv_heat_weight']
    )
    test_model.external_source_weights['fireplace'] = param_dict.get(
        'fireplace_heat_weight',
        current_params['fireplace_heat_weight']
    )
    test_model.external_source_weights['tv'] = param_dict.get(
        'tv_heat_weight', current_params['tv_heat_weight']
    )

    test_model.solar_lag_minutes = param_dict.get(
        'solar_lag_minutes', current_params['solar_lag_minutes']
    )
    test_model.sync_heat_source_channels_from_model_state()

    for period in test_periods:
        try:

            # FORCE TEMPERATURE-BASED PHYSICS:
            # We intentionally pass thermal_power=None here even if available.
            # Why?
            # 1. If we use thermal_power, the model uses the energy balance equation
            #    which ignores 'outlet_effectiveness'. This causes outlet_effectiveness
            #    calibration to stagnate (zero gradient).
            # 2. The primary goal of this calibration is to tune the parameters
            #    (heat_loss_coefficient, outlet_effectiveness) so that the
            #    TEMPERATURE-BASED model (used for control planning) accurately
            #    predicts equilibrium.
            # 3. We want the controller's "Required Outlet Temp" calculation to be
            #    accurate, and that calculation relies on the relationship between
            #    outlet_temp and equilibrium established by these two parameters.

            # Use PV history if available for lag calculation
            pv_input = period.get('pv_power_history', period['pv_power'])

            predicted_temp = test_model.predict_equilibrium_temperature(
                outlet_temp=period.get('effective_temp', period['outlet_temp']),
                outdoor_temp=period['outdoor_temp'],
                current_indoor=period.get(
                    'indoor_temp', period['outdoor_temp'] + 10.0
                ),
                pv_power=pv_input,
                fireplace_on=period['fireplace_on'],
                tv_on=period['tv_on'],
                thermal_power=None,  # Force temperature-based path
                _suppress_logging=True,
                cloud_cover_pct=0.0,
            )

            error = abs(predicted_temp - period['indoor_temp'])

            if error > 50.0:
                if calculate_mae_for_params.call_count <= 1 and valid_predictions == 0:
                    logging.debug(
                        f"Skipping large error: {error:.1f} "
                        f"(Pred: {predicted_temp:.1f}, "
                        f"Actual: {period['indoor_temp']:.1f})"
                    )
                continue

            total_error += error
            valid_predictions += 1

        except Exception as e:
            if calculate_mae_for_params.call_count <= 1:
                logging.debug(f"Prediction exception: {e}")
            continue

    root_logger.setLevel(original_level)

    if valid_predictions == 0:
        if calculate_mae_for_params.call_count <= 1:
            logging.warning("No valid predictions found in MAE calculation!")
        return 1000.0

    mae = total_error / valid_predictions

    if calculate_mae_for_params.call_count % 10 == 1:
        logging.debug("MAE for %s: %.4f", debug_str, mae)

    return mae


def log_optimization_results(result, param_names, param_values):
    """Log the results of the optimization."""
    logging.info("🔍 SCIPY OPTIMIZATION RESULTS:")
    logging.info(f"  Success: {result.success}")
    logging.info(f"  Message: {result.message}")
    logging.info(f"  Function evaluations: {result.nfev}")
    logging.info(
        f"  Iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}"
    )
    logging.info(f"  Final function value: {result.fun:.6f}")

    logging.info("  Parameter changes:")
    for i, param_name in enumerate(param_names):
        initial_val = param_values[i]
        final_val = result.x[i]
        change = final_val - initial_val
        logging.info(
            f"    {param_name}: {initial_val:.6f} → {final_val:.6f} "
            f"(Δ{change:+.6f})"
        )


def build_optimized_params(
    result, current_params, param_names, excluded_params
):
    """Build the dictionary of optimized parameters."""
    optimized_params = dict(current_params)

    for i, param_name in enumerate(param_names):
        optimized_params[param_name] = result.x[i]

    optimized_params['mae'] = result.fun
    optimized_params['optimization_success'] = True
    optimized_params['excluded_parameters'] = excluded_params
    return optimized_params


def log_optimized_parameters(
    optimized_params, current_params, excluded_params
):
    """Log the final optimized parameters."""
    logging.info("✅ Optimization completed successfully!")
    logging.info("Optimized parameters:")
    for param, value in optimized_params.items():
        if param not in ['mae', 'optimization_success', 'excluded_parameters']:
            old_value = current_params[param]
            if param in excluded_params:
                logging.info(f"  {param}: {value:.4f} (FIXED - no data)")
            else:
                change_pct = (
                    ((value - old_value) / old_value) * 100 if old_value else 0
                )
                logging.info(
                    f"  {param}: {value:.4f} "
                    f"(was {old_value:.4f}, {change_pct:+.1f}%)"
                )

    logging.info(f"Final MAE: {optimized_params['mae']:.4f}°C")


# ===================================================================
# Apply calibrated channel parameters to orchestrator
# ===================================================================

def _apply_channel_params(orchestrator, channel_params: dict) -> None:
    """Apply calibrated channel-specific parameters to the orchestrator."""
    fp_ch = orchestrator.channels.get("fireplace")
    hp_ch = orchestrator.channels.get("heat_pump")
    pv_ch = orchestrator.channels.get("pv")

    if fp_ch is not None:
        if "fp_decay_time_constant" in channel_params:
            fp_ch.fp_decay_time_constant = channel_params["fp_decay_time_constant"]
            logging.info(
                "🔥 Applied fp_decay_time_constant = %.2fh",
                fp_ch.fp_decay_time_constant,
            )
        if "room_spread_delay_minutes" in channel_params:
            fp_ch.room_spread_delay_minutes = channel_params["room_spread_delay_minutes"]
            logging.info(
                "🔥 Applied room_spread_delay_minutes = %.0fmin",
                fp_ch.room_spread_delay_minutes,
            )

    if hp_ch is not None:
        if "delta_t_floor" in channel_params:
            hp_ch.delta_t_floor = channel_params["delta_t_floor"]
            logging.info(
                "🔧 Applied delta_t_floor = %.2f°C",
                hp_ch.delta_t_floor,
            )

    if pv_ch is not None:
        if "cloud_factor_exponent" in channel_params:
            pv_ch.cloud_factor_exponent = channel_params["cloud_factor_exponent"]
            logging.info(
                "☁️ Applied cloud_factor_exponent = %.2f",
                pv_ch.cloud_factor_exponent,
            )
        if "solar_decay_tau_hours" in channel_params:
            pv_ch.solar_decay_tau_hours = channel_params["solar_decay_tau_hours"]
            logging.info(
                "☀️ Applied solar_decay_tau_hours = %.2fh",
                pv_ch.solar_decay_tau_hours,
            )


# ===================================================================
# Phase 2 – Fireplace batch calibration
# ===================================================================

def filter_fp_decay_periods(df, min_on_minutes=15, post_off_minutes=120,
                            hlc=None, oe=None):
    """Find fireplace on→off transitions and extract post-off decay curves.

    Returns a list of dicts, each with:
      * ``indoor_excess`` – list of (minutes_since_off, excess_temp) tuples
      * ``outdoor_temp``  – mean outdoor temp during the decay window
      * ``outlet_temp``   – mean outlet temp during the decay window
    Only includes transitions where the FP was on for ≥ *min_on_minutes*.

    If *hlc* and *oe* are supplied the baseline used for computing indoor
    excess is the HP-only equilibrium ``(oe*outlet + hlc*outdoor)/(oe+hlc)``
    instead of the raw outdoor temperature. This isolates the FP-only
    contribution for more accurate decay fitting.
    """
    logging.info("=== FILTERING FOR FIREPLACE DECAY PERIODS ===")
    if df is None or df.empty:
        return []

    fp_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    dhw_col = config.DHW_STATUS_ENTITY_ID.split(".", 1)[-1]

    for c in (fp_col, indoor_col, outdoor_col):
        if c not in df.columns:
            logging.warning("Missing column %s for FP decay filtering", c)
            return []

    df = df.sort_values("_time").reset_index(drop=True)
    n = len(df)
    step_min = 5  # 5-minute resolution

    min_on_steps = max(1, min_on_minutes // step_min)
    post_off_steps = post_off_minutes // step_min

    periods = []
    i = 0
    while i < n - post_off_steps:
        # Look for FP on → off transition
        if df[fp_col].iloc[i] > 0 and (i + 1 < n) and df[fp_col].iloc[i + 1] <= 0:
            # Count how long FP was on before this point
            on_count = 0
            j = i
            while j >= 0 and df[fp_col].iloc[j] > 0:
                on_count += 1
                j -= 1
            if on_count < min_on_steps:
                i += 1
                continue

            off_start = i + 1
            off_end = min(off_start + post_off_steps, n)
            window = df.iloc[off_start:off_end]

            # Skip if blocking active during decay
            if dhw_col in df.columns and window[dhw_col].sum() > 0:
                i = off_end
                continue

            # Skip if FP comes back on during window
            if window[fp_col].sum() > 0:
                # Find where FP comes back on and truncate
                first_on = window[fp_col].gt(0).idxmax()
                window = df.iloc[off_start:first_on]
                if len(window) < 4:
                    i += 1
                    continue

            indoor_vals = window[indoor_col].values
            outdoor_mean = window[outdoor_col].mean()
            outlet_mean = window[outlet_col].mean() if outlet_col in window.columns else 25.0

            # Compute indoor excess above baseline
            if hlc is not None and oe is not None and (hlc + oe) > 0:
                baseline = (oe * outlet_mean + hlc * outdoor_mean) / (oe + hlc)
            else:
                baseline = outdoor_mean
            excess = [(k * step_min, float(indoor_vals[k]) - baseline)
                      for k in range(len(indoor_vals))
                      if (float(indoor_vals[k]) - baseline) > 0.1]

            if len(excess) >= 3:
                periods.append({
                    "indoor_excess": excess,
                    "outdoor_temp": outdoor_mean,
                    "outlet_temp": outlet_mean,
                })
            i = off_end
        else:
            i += 1

    logging.info("Found %d fireplace decay periods", len(periods))
    return periods


def calibrate_fp_decay_tau(decay_periods):
    """Calibrate fp_decay_time_constant from post-off exponential decay.

    Uses log-linear regression: ln(T_excess) = ln(A) - t/τ.
    Returns τ in hours, or None if insufficient data.
    """
    if len(decay_periods) < 5:
        logging.warning(
            "Insufficient FP decay periods: %d (need ≥5)", len(decay_periods)
        )
        return None

    taus = []
    for period in decay_periods:
        pts = period["indoor_excess"]
        if len(pts) < 3:
            continue
        t_vals = np.array([p[0] for p in pts], dtype=float)  # minutes
        y_vals = np.array([p[1] for p in pts], dtype=float)

        valid = y_vals > 0.05
        if valid.sum() < 3:
            continue

        t_vals = t_vals[valid]
        y_vals = y_vals[valid]
        log_y = np.log(y_vals)

        # Linear regression
        n_pts = len(t_vals)
        sx = t_vals.sum()
        sy = log_y.sum()
        sxy = (t_vals * log_y).sum()
        sxx = (t_vals * t_vals).sum()
        denom = n_pts * sxx - sx * sx
        if abs(denom) < 1e-10:
            continue
        slope = (n_pts * sxy - sx * sy) / denom

        if slope >= -0.001:
            continue  # Not decaying

        tau_min = -1.0 / slope  # in minutes
        tau_h = tau_min / 60.0

        if 0.1 <= tau_h <= 6.0:
            taus.append(tau_h)

    if not taus:
        logging.warning("No valid FP decay fits")
        return None

    result = float(np.median(taus))
    logging.info(
        "Calibrated fp_decay_time_constant = %.2fh (from %d fits)",
        result, len(taus),
    )
    return result


def calibrate_slab_time_constant(df, delta_t_floor=None):
    """Calibrate slab_time_constant_hours from pump-ON inlet response.

    When the heat pump starts, the outlet (supply) temperature rises and
    the slab (measured at inlet/return) follows with time constant tau_slab:

        inlet(t)  →  outlet - delta_t_floor   (exponential approach)

    This matches the trajectory model usage:
        alpha = min(1.0, dt / slab_tau)
        t_slab += alpha * ((outlet - delta_t_floor) - t_slab)

    HP startup is detected via thermal_power transitioning from < 0.5 kW
    to >= 0.5 kW.  For each event we fit the first 90 min of HP-ON data.

    Returns tau in hours, or None if insufficient data.
    """
    logging.info("=== CALIBRATING SLAB TIME CONSTANT ===")
    if df is None or df.empty:
        return None

    inlet_col = config.INLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    outlet_col = config.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID.split(".", 1)[-1]
    flow_col = config.FLOW_RATE_ENTITY_ID.split(".", 1)[-1]
    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]

    for c in (inlet_col, outlet_col, flow_col, indoor_col):
        if c not in df.columns:
            logging.info("Column %s missing — skipping slab tau calibration", c)
            return None

    if delta_t_floor is None:
        delta_t_floor = ThermalParameterConfig.get_default('delta_t_floor')

    df_s = df.sort_values('_time').reset_index(drop=True)

    # Compute thermal power for HP state detection
    delta_t = df_s[outlet_col] - df_s[inlet_col]
    thermal_power = (
        (df_s[flow_col] / 60.0)
        * config.SPECIFIC_HEAT_CAPACITY
        * delta_t
    ).clip(lower=0.0)

    # Detect HP startup transitions (rising edges: off → on)
    hp_on = thermal_power >= 0.5
    hp_on_prev = hp_on.shift(1, fill_value=False)
    startups = hp_on & ~hp_on_prev

    slab_taus = []
    n_startups = 0

    for start_idx in startups[startups].index:
        n_startups += 1

        # Find contiguous HP-ON window from this startup
        end_idx = start_idx
        while end_idx + 1 < len(df_s) and hp_on.iloc[end_idx + 1]:
            end_idx += 1

        window = df_s.iloc[start_idx:end_idx + 1]
        if len(window) < 6:  # Need ≥ 30 min of HP-ON data
            continue

        # Use first 90 min only
        t_hours_all = (
            window['_time'] - window['_time'].iloc[0]
        ).dt.total_seconds().values / 3600.0
        use = t_hours_all <= 1.5
        window = window.iloc[:int(use.sum())]
        t_hours = t_hours_all[:len(window)]

        if len(window) < 6:
            continue

        inlet_vals = window[inlet_col].values.astype(float)
        outlet_vals = window[outlet_col].values.astype(float)

        # Check minimum gap: inlet must be meaningfully below
        # the target (outlet - delta_t_floor) somewhere in the window
        target_vals = outlet_vals - delta_t_floor
        max_gap = float(np.max(target_vals - inlet_vals))
        if max_gap < 0.5:
            continue

        # Timestep-based MSE fitting using the model's own formula:
        #   pred_{k+1} = pred_k + min(1, dt/tau) * (outlet_k - dtf - pred_k)
        # This handles ramping outlet naturally.
        dt_steps = np.diff(t_hours)  # time steps in hours

        def _slab_mse(params):
            tau = params[0]
            pred = inlet_vals[0]
            mse = 0.0
            for k in range(len(dt_steps)):
                alpha = min(1.0, dt_steps[k] / tau)
                pred = pred + alpha * (target_vals[k] - pred)
                mse += (pred - inlet_vals[k + 1]) ** 2
            return mse / len(dt_steps)

        if minimize is None:
            continue

        try:
            result = minimize(
                _slab_mse, [1.0],
                bounds=[(0.1, 6.0)],
                method='L-BFGS-B',
                options={'ftol': 1e-8, 'maxiter': 100},
            )
        except Exception:
            continue

        if not (result.success or result.fun < 0.5):
            continue

        best_tau = float(result.x[0])
        best_mse = float(result.fun)

        # Quality gate: MSE < 0.5 °C² (≈ 0.7°C RMS)
        if best_mse > 0.5:
            continue

        if 0.1 < best_tau < 6.0:
            slab_taus.append(best_tau)

    logging.info(
        "Slab tau: checked %d HP-startup events, %d valid fits",
        n_startups, len(slab_taus),
    )

    if not slab_taus:
        logging.info("No valid slab approach periods found")
        return None

    result = float(np.median(slab_taus))
    result = max(0.2, min(4.0, result))
    logging.info(
        "Calibrated slab_time_constant_hours = %.2fh (median of %d events)",
        result, len(slab_taus),
    )
    return result


def filter_fp_spread_periods(df, min_on_minutes=20):
    """Find FP events where living_room_temp AND indoor_temp are available.

    Returns list of dicts with ``living_room`` and ``indoor_avg`` time
    series arrays suitable for cross-correlation.
    """
    logging.info("=== FILTERING FOR FIREPLACE SPREAD PERIODS ===")
    if df is None or df.empty:
        return []

    fp_col = config.FIREPLACE_STATUS_ENTITY_ID.split(".", 1)[-1]
    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    lr_col = getattr(config, "LIVING_ROOM_TEMP_ENTITY_ID", "").split(".", 1)[-1]

    if not lr_col or lr_col not in df.columns:
        logging.info("Living room temp not available — skipping spread calibration")
        return []

    for c in (fp_col, indoor_col):
        if c not in df.columns:
            return []

    df = df.sort_values("_time").reset_index(drop=True)
    step_min = 5
    min_steps = max(1, min_on_minutes // step_min)

    periods = []
    in_event = False
    event_start = 0

    for i in range(len(df)):
        fp_on = df[fp_col].iloc[i] > 0
        if fp_on and not in_event:
            event_start = i
            in_event = True
        elif not fp_on and in_event:
            event_len = i - event_start
            if event_len >= min_steps:
                window = df.iloc[event_start:i]
                lr_vals = window[lr_col].values
                avg_vals = window[indoor_col].values
                if not (np.isnan(lr_vals).any() or np.isnan(avg_vals).any()):
                    periods.append({
                        "living_room": lr_vals.astype(float),
                        "indoor_avg": avg_vals.astype(float),
                        "duration_steps": event_len,
                    })
            in_event = False

    logging.info("Found %d FP spread periods", len(periods))
    return periods


def calibrate_room_spread_delay(spread_periods):
    """Calibrate room_spread_delay_minutes via cross-correlation.

    Returns delay in minutes, or None if insufficient data.
    """
    if len(spread_periods) < 5:
        logging.warning(
            "Insufficient FP spread periods: %d (need ≥5)", len(spread_periods)
        )
        return None

    step_min = 5
    delays = []
    for period in spread_periods:
        lr = period["living_room"]
        avg = period["indoor_avg"]
        if len(lr) < 4:
            continue
        # Differentiate to get rate of change
        lr_diff = np.diff(lr)
        avg_diff = np.diff(avg)
        if np.std(lr_diff) < 1e-8 or np.std(avg_diff) < 1e-8:
            continue
        # Normalized cross-correlation
        corr = np.correlate(
            (lr_diff - lr_diff.mean()) / (np.std(lr_diff) * len(lr_diff)),
            (avg_diff - avg_diff.mean()) / np.std(avg_diff),
            mode="full",
        )
        mid = len(lr_diff) - 1
        # Only look at positive lags (living room leads)
        positive_lags = corr[mid:]
        if len(positive_lags) == 0:
            continue
        peak_idx = np.argmax(positive_lags)
        delay_min = peak_idx * step_min
        if 0 <= delay_min <= 180:
            delays.append(delay_min)

    if not delays:
        logging.warning("No valid FP spread delay estimates")
        return None

    result = float(np.median(delays))
    logging.info(
        "Calibrated room_spread_delay_minutes = %.0fmin (from %d estimates)",
        result, len(delays),
    )
    return result


def calibrate_delta_t_floor(stable_periods):
    """Calibrate delta_t_floor as percentile-10 of outlet-inlet during HP-on.

    Returns delta_t_floor in °C, or None if insufficient data.
    Filters out periods with low flow rate (< 5 L/min) where delta_t
    is dominated by standstill heat loss and is not representative.
    """
    logging.info("=== CALIBRATING DELTA_T_FLOOR ===")
    deltas = []
    for p in stable_periods:
        inlet = p.get("inlet_temp")
        outlet = p.get("outlet_temp")
        thermal_power = p.get("thermal_power_kw", 0)
        if inlet is None or outlet is None:
            continue
        if not isinstance(inlet, (int, float)) or not isinstance(outlet, (int, float)):
            continue
        dt = outlet - inlet
        if dt > 1.0 and thermal_power > 0.5:
            deltas.append(dt)

    if len(deltas) < 10:
        logging.warning("Insufficient data for delta_t_floor: %d", len(deltas))
        return None

    result = float(np.percentile(deltas, 25))
    result = max(1.0, min(10.0, result))
    logging.info(
        "Calibrated delta_t_floor = %.2f°C (P25 of %d samples)",
        result, len(deltas),
    )
    return result


def filter_cloudy_pv_periods(df, min_pv=200, cloud_col="cloud_cover_proxy",
                             min_cloud=20, max_cloud=80):
    """Extract periods with PV > min_pv and variable cloud cover."""
    if df is None or df.empty:
        return []

    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]

    if pv_col not in df.columns or cloud_col not in df.columns:
        return []

    mask = (
        (df[pv_col] > min_pv)
        & (df[cloud_col] >= min_cloud)
        & (df[cloud_col] <= max_cloud)
    )
    subset = df[mask]
    if len(subset) < 30:
        return []

    periods = []
    for _, row in subset.iterrows():
        periods.append({
            "pv_power": float(row[pv_col]),
            "cloud_pct": float(row[cloud_col]),
            "indoor_temp": float(row.get(indoor_col, 20.0)),
        })
    return periods


def calibrate_cloud_factor(cloudy_periods, pv_heat_weight):
    """Fit cloud_factor_exponent: effective_pv = pv × (1-cloud/100)^exp.

    Returns exponent, or None if insufficient data.
    """
    if len(cloudy_periods) < 30 or pv_heat_weight <= 0:
        return None

    if minimize is None:
        return None

    def objective(params):
        exp = params[0]
        if exp <= 0:
            return 1e9
        errors = []
        for p in cloudy_periods:
            cloud_frac = p["cloud_pct"] / 100.0
            effective = p["pv_power"] * (1.0 - cloud_frac) ** exp
            predicted_contribution = effective * pv_heat_weight
            # We don't have a perfect target; use residual variance
            errors.append(predicted_contribution)
        # Minimize variance of contributions (well-calibrated exponent
        # should reduce scatter)
        arr = np.array(errors)
        return float(np.std(arr))

    try:
        result = minimize(objective, [1.0], bounds=[(0.1, 3.0)], method="L-BFGS-B")
        if result.success:
            exp = float(result.x[0])
            logging.info("Calibrated cloud_factor_exponent = %.2f", exp)
            return exp
    except Exception as exc:
        logging.warning("Cloud factor calibration failed: %s", exc)

    return None


def filter_pv_decay_periods(df, pv_high=500, pv_low=100, post_steps=24,
                            crossing_window=6):
    """Find transitions where PV drops from >pv_high to <pv_low.

    Uses a sliding window (``crossing_window`` steps, default 6 = 30 min)
    to detect gradual sunset transitions, not just single-step sharp drops.

    Returns list of dicts with ``indoor_excess`` (minutes, temp) tuples.
    """
    if df is None or df.empty:
        return []

    pv_col = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
    indoor_col = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
    outdoor_col = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]

    for c in (pv_col, indoor_col, outdoor_col):
        if c not in df.columns:
            return []

    df = df.sort_values("_time").reset_index(drop=True)
    n = len(df)
    step_min = 5
    periods = []

    i = crossing_window
    while i < n - post_steps:
        # Check if PV was above threshold in any of the preceding window steps
        # and is now below threshold
        curr_pv = df[pv_col].iloc[i]
        if curr_pv < pv_low:
            window_before = df[pv_col].iloc[max(0, i - crossing_window): i]
            had_high_pv = (window_before > pv_high).any()
            if had_high_pv:
                post_window = df.iloc[i: i + post_steps]
                outdoor_mean = post_window[outdoor_col].mean()
                baseline = outdoor_mean
                indoor_vals = post_window[indoor_col].values
                excess = [
                    (k * step_min, float(indoor_vals[k]) - baseline)
                    for k in range(len(indoor_vals))
                    if float(indoor_vals[k]) - baseline > 0.1
                ]
                if len(excess) >= 3:
                    periods.append({"indoor_excess": excess})
                i += post_steps
                continue
        i += 1

    logging.info("Found %d PV decay periods", len(periods))
    return periods


def calibrate_solar_decay_tau(decay_periods):
    """Calibrate solar_decay_tau_hours from PV drop residual curves.

    Same log-linear approach as FP decay. Returns τ in hours or None.
    """
    if len(decay_periods) < 10:
        logging.warning(
            "Insufficient PV decay periods: %d (need ≥10)", len(decay_periods)
        )
        return None

    taus = []
    for period in decay_periods:
        pts = period["indoor_excess"]
        if len(pts) < 3:
            continue
        t_vals = np.array([p[0] for p in pts], dtype=float)
        y_vals = np.array([p[1] for p in pts], dtype=float)
        valid = y_vals > 0.05
        if valid.sum() < 3:
            continue
        t_vals = t_vals[valid]
        log_y = np.log(y_vals[valid])

        n_pts = len(t_vals)
        sx = t_vals.sum()
        sy = log_y.sum()
        sxy = (t_vals * log_y).sum()
        sxx = (t_vals * t_vals).sum()
        denom = n_pts * sxx - sx * sx
        if abs(denom) < 1e-10:
            continue
        slope = (n_pts * sxy - sx * sy) / denom
        if slope >= -0.001:
            continue
        tau_h = (-1.0 / slope) / 60.0
        if 0.1 <= tau_h <= 3.0:
            taus.append(tau_h)

    if not taus:
        return None

    result = float(np.median(taus))
    logging.info(
        "Calibrated solar_decay_tau_hours = %.2fh (from %d fits)",
        result, len(taus),
    )
    return result


def backup_existing_calibration():
    """Create a backup of the existing thermal calibration."""
    logging.info("Creating backup of existing thermal calibration...")

    try:
        from datetime import datetime

        state_manager = get_thermal_state_manager()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"pre_calibration_{timestamp}.json"

        success, result = state_manager.create_backup(backup_name)

        if success:
            logging.info(f"✅ Backup created: {result}")
            return result
        else:
            logging.warning(f"Failed to create calibration backup: {result}")
            return None

    except Exception as e:
        logging.warning(f"Failed to create calibration backup: {e}")
        return None


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
