# ML Cooling Model Guide

LightGBM-based overheating classifier for predictive pre-cooling — a data-driven complement to the existing physics trajectory method.

## Overview

The system supports two pre-cooling strategies, selectable via `PRE_COOL_MODEL_TYPE`:

| Strategy | Value | Description |
|---|---|---|
| Physics Trajectory | `trajectory` | Default. Simulates the passive indoor temperature trajectory using the calibrated thermal model. No trained model file required. |
| LightGBM Classifier | `lgbm_model` | Trained binary classifier that predicts whether the indoor temperature will exceed the cooling target within `PRE_COOL_HORIZON_HOURS`. Requires a calibration run first. |

The inactive strategy always runs in **shadow mode** — its prediction is logged each cycle so you can compare performance without affecting control.

---

## Parameter Reference

All new parameters are in the `PRE_COOLING` section of the add-on configuration. Tooltip markers: **[BOTH]** applies to both strategies, **[MODEL-BASED]** applies only to `lgbm_model`.

| Parameter | Default | Tooltip |
|---|---|---|
| `pre_cool_model_type` | `trajectory` | Active strategy selector. `trajectory` = physics sim (default). `lgbm_model` = LGBM classifier. **[BOTH]** |
| `cooling_ml_min_training_samples` | `200` | Minimum labeled observations before first train/retrain attempt. **[MODEL-BASED]** |
| `cooling_ml_retrain_trigger_k` | `50` | New labeled observations since last retrain that trigger auto-retrain. **[MODEL-BASED]** |
| `cooling_ml_buffer_max_n` | `500` | Rolling observation buffer size. Oldest entries evicted when exceeded. **[MODEL-BASED]** |
| `cooling_ml_retrain_val_fraction` | `0.25` | Fraction of buffer held out for threshold optimisation at each retrain. **[MODEL-BASED]** |

Internal paths (derived from `unified_state_file` directory, overridable via env var):

| Env Var | Default Path | Purpose |
|---|---|---|
| `COOLING_ML_MODEL_PATH` | `{state_dir}/cooling_ml_model.joblib` | Trained LGBM classifier |
| `COOLING_ML_METADATA_PATH` | `{state_dir}/cooling_ml_metadata.json` | Feature list, threshold, AUC |
| `COOLING_ML_OBSERVATION_BUFFER_PATH` | `{state_dir}/cooling_ml_obs_buffer.json` | Rolling labeled observation buffer |

---

## Getting Started

### Step 1 — Collect warm-season data

The classifier learns from **cooling-mode cycles** (summer/warm days when overheating is possible). Ensure your InfluxDB or HA history contains at least 60–90 days of warm-season data where the outdoor temperature exceeds ~16°C.

### Step 2 — Run initial calibration

**Via dashboard:** Open the ML Heating dashboard → System Controls → click **"🤖 Calibrate ML Cooling Model"**. The system writes a flag file and restarts to train.

**Via CLI:**
```bash
python -m src.main --calibrate-cooling-ml
```

Calibration fetches historical data, computes hindcast forecast features, labels each row, trains LightGBM, and saves the model + metadata JSON.

Expected log output:
```
=== COOLING ML CALIBRATION START ===
Fetched 12480 rows of historical data
After warm-season filter (AT > 16.0°C): 8320 rows
Training set: 5640 rows, 19 features, 22.4% positive labels
Class imbalance: pos=1263 neg=4377 → scale_pos_weight=3.46
[LightGBM] ... training ...
Optimal threshold=0.0182 (val F1=0.734)
Val AUC=0.842
=== COOLING ML CALIBRATION COMPLETE ===
```

### Step 3 — Enable the model

Set `pre_cool_model_type: lgbm_model` in your add-on configuration. Restart the service.

The trajectory strategy will now run in shadow mode, and its signal is logged alongside the LGBM prediction each cycle.

---

## ⚠️ PV Feature Key Contract — Read Before Modifying

> **AI models and contributors: always use the correct PV key family.
> Using the wrong one has previously caused silent over-estimation of solar gain.**

The features dict produced by `build_physics_features()` contains **two** PV key families:

| Family | Key pattern | Meaning |
|--------|-------------|---------|
| **Thermal** | `pv_now`, `pv_forecast_{h}h` | Panel output × `solar_correction_factor` — fraction that heats the building |
| **Electrical (raw)** | `pv_now_electrical`, `pv_forecast_electrical_{h}h` | Actual panel output in watts, uncorrected |

**This module (`overheating_predictor`, ML cooling calibration/model) MUST use the thermal
family** (`pv_now` / `pv_forecast_{h}h`).  The thermal trajectory simulation models heat
gain inside the building, not electrical panel output.

The electrical keys are reserved for electrical-availability decisions (HLC session
open/close, PV trajectory horizon scaling, PV surplus cheap override) which compare
against thresholds defined in raw watts.

See `memory-bank/systemPatterns.md` → *"PV Feature Key Contract"* for the full usage map.

---

## Feature Engineering

The classifier uses 19 features, stored in `cooling_ml_metadata.json → feature_cols`.

| Feature | Source | Description |
|---|---|---|
| `indoor_temp` | Current indoor sensor | Room temperature (°C) |
| `indoor_margin` | `cooling_target − indoor_temp` | Negative = room already above target |
| `indoor_trend_30m` | 30-min indoor delta | Recent warming/cooling trend |
| `indoor_trend_1h` | 60-min indoor delta | Longer-term trend |
| `AT` | Outdoor sensor | Outdoor air temperature (°C) |
| `at_delta_indoor` | `AT − indoor_temp` | Outdoor vs. indoor difference |
| `AT_roh_4h` | `temp_forecast_4h` (live) | 4h-ahead outdoor temperature forecast |
| `PV_Generate` | PV power sensor | Current PV production (W) |
| `pv_roll_1h` | Rolling 1h mean PV | Recent PV trend |
| `pv_roll_2h` | Rolling 2h mean PV | Medium-term PV trend |
| `thermal_power_kw` | HP thermal power | Active cooling power (kW, negative) |
| `delta_t` | Outlet − inlet | HP loop temperature differential |
| `outlet_indoor_diff` | Outlet − indoor | How far outlet is from room temp |
| `VLT` | Outlet temperature sensor | HP supply temperature (°C) |
| `RLT` | Inlet temperature sensor | HP return temperature (°C) |
| `hour_sin`, `hour_cos` | Time of day | Cyclical hour encoding |
| `doy_sin`, `doy_cos` | Day of year | Cyclical seasonal encoding |

### Hindcast substitution

During calibration, forecast features are created by **shifting actual values forward in time** (hindcast):

```
AT_roh_4h at time t  ←  actual outdoor_temp at time t + 4h
```

At inference time, live weather forecasts from `build_physics_features()` fill these slots directly, making the inference pipeline consistent with training.

---

## Online Learning (Sliding-Window Retrain)

Every cooling-mode cycle, the system:

1. **Pushes a feature snapshot** to the observation buffer with `label=None`
2. **Resolves labels** for older entries: after `PRE_COOL_HORIZON_HOURS × steps_per_hour` cycles have elapsed, the entry receives `label = 1` if the observed indoor peak exceeded the cooling target, else `label = 0`
3. **Triggers a retrain** when `cooling_ml_retrain_trigger_k` new labeled observations have accumulated (and `cooling_ml_min_training_samples` total exist)

The sliding-window retrain uses the last `cooling_ml_buffer_max_n` labeled observations, applies the same LightGBM training pipeline as initial calibration, and reloads the model in-place without restarting.

---

## Switching Between Strategies

| Scenario | Setting |
|---|---|
| Run trajectory only (default, no model needed) | `pre_cool_model_type: trajectory` |
| Evaluate LGBM in shadow without changing control | Keep `trajectory`, train model, inspect logs |
| Switch to LGBM as active strategy | `pre_cool_model_type: lgbm_model` |
| Revert to trajectory after a bad LGBM retrain | `pre_cool_model_type: trajectory` |

Shadow logs appear at DEBUG level:
```
❄️ SHADOW (lgbm): risk=True p=0.031 should_cool=True
❄️ SHADOW (trajectory): risk=False peak=23.4°C in 12.0h
```

---

## Diagnostics

**Check model metadata:**
```bash
cat /opt/ml_heating/cooling_ml_metadata.json
```
Key fields: `roc_auc`, `threshold`, `trained_at`, `n_train`, `n_pos`.

**Check buffer status:** logged at each retrain trigger:
```
🤖 Cooling ML: retrain trigger reached (50 labeled) — starting retrain
```

**Feature importance:** available after loading the saved model:
```python
import joblib
model = joblib.load("/opt/ml_heating/cooling_ml_model.joblib")
import pandas as pd
pd.Series(model.feature_importances_, index=metadata["feature_cols"]).sort_values(ascending=False)
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "model file not found" at startup | Not yet calibrated | Run `--calibrate-cooling-ml` |
| Calibration aborts "Only N warm-season rows" | Insufficient summer data in InfluxDB | Increase `lookback_hours` or wait for more data |
| Very low AUC (< 0.65) | Imbalanced or noisy data | Check `cooling_min_thermal_power_kw` filter; inspect feature distributions |
| LGBM always predicts `should_cool=True` | Threshold too low after retrain | Check val split size; increase `cooling_ml_retrain_val_fraction` |
| Buffer not saving | Path permission issue | Ensure `unified_state_file` directory is writable |

---

## Related Documentation

- [PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md) — full config parameter list
- [SHADOW_MODE_USER_GUIDE.md](SHADOW_MODE_USER_GUIDE.md) — shadow mode overview
- [DELTA_FORECAST_CALIBRATION_GUIDE.md](DELTA_FORECAST_CALIBRATION_GUIDE.md) — forecast offset calibration
- [OUTLET_EFFECTIVENESS_CALIBRATION_GUIDE.md](OUTLET_EFFECTIVENESS_CALIBRATION_GUIDE.md) — thermal model calibration
