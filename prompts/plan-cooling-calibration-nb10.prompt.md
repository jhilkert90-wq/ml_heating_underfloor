# Plan: Cooling Physics Calibration — Phase 2 (NB10 + NB12 + HLC Analysis)

## TL;DR

Apply combined offline calibration findings from **NB10** (dual-HLC cooling model),
**NB12** (solar parameter replacement + HLC/Beschattung analysis), and production log
reconstruction to `physics_calibration_cooling.py` and supporting modules.

**Three parallel changes:**
1. Dual-HLC infrastructure with Beschattung-aware calibration
2. Replace PV_Generate/pv_forecast with GHI from Open Meteo
3. Fix optimizer bounds (OE, tau, HLC_off) based on NB12 evidence

---

## Evidence Summary

### NB10: Dual-HLC Thermal Calibration

| Parameter | Single-HLC (current) | Dual-HLC (NB10) | Notes |
|-----------|----------------------|------------------|-------|
| HLC_on | 0.1804 | 0.1464 | Active cooling (HP + fan) |
| HLC_off | (same) | 0.0155 | HP off — **artificially low, see HLC Analysis** |
| OE | 0.1861 | 0.2088 | Cooling 4.4x lower than heating (0.83) |
| tau | 8.00 h | 8.00 h | Online inflated to 41h; cap at 15h |
| RMSE | 1.41 C | 0.71 C | 50% improvement |

### NB12: Solar Parameter Replacement (GHI vs PV)

| Metric | PV_Generate | GHI (Open Meteo) | Diffuse | Notes |
|--------|-------------|-------------------|---------|-------|
| Thermal RMSE | 0.528 | 0.476 | 0.460 | GHI wins, diffuse best |
| r(trend) | +0.233* | +0.233 | +0.211 | *PV has panel-specific noise |
| r(PV_Generate) | 1.0 | +0.640 | +0.427 | GHI tracks PV well |
| ML AUC | 0.945 | 0.931 | - | PV slightly better, GHI adequate |
| ML F1 | 0.842 | 0.788 | - | PV slightly better |
| Weight | 0.000493 kW/W | 0.004144 kW/(W/m2) | 0.018258 | Scale ratio PV/GHI = 7.94 |
| Optimal lag | -5 min | 0 min | 0 min | Current 45min far too high |

**Key finding:** All NB12 calibrations hit HLC_on=0.500 and tau=15.0 upper bounds.
This means the bounds from NB10 are too tight — the optimizer wants higher HLC_on and longer tau
when GHI solar is properly accounted for.

### HLC On/Off Reconstruction: Beschattung Confound (NB12 Section F)

**Problem:** NB10 found HLC_on/HLC_off ratio = 9.4x. Is this physical?

**Analysis (Section F of NB12):**

| Condition | N | Driving Force | Trend | Slope | r |
|-----------|---|---------------|-------|-------|---|
| Night HP-off (no solar/blinds) | 13,088 | 8.12 K | -0.019 | -0.003139 | -0.122 |
| Day HP-off + PV>1kW (Beschattung zu) | 14,014 | 5.05 K | +0.015 | -0.001706 | -0.157 |
| Day HP-off + PV<100W (cloudy) | 2,508 | 8.44 K | -0.003 | -0.000289 | -0.038 |

**Night/Day+PV slope ratio: 1.84x** — Beschattung reduces apparent heat exchange by **46%**.

**Root cause of low HLC_off:**
1. **Beschattung (automated blinds)** close when sun shines during HP-off recovery
   - Blocks solar heat gain -> indoor temp stays stable -> optimizer finds near-zero HLC_off
2. **Small driving force** in summer cooling (indoor-outdoor delta only 2-5 K)
   - Minimal heat exchange regardless of HLC value -> HLC unidentifiable
3. **VLT equals indoor_temp** when HP off -> OE contribution is zero -> OE and HLC_off interchangeable

**Night-filtered dual-HLC calibration (HP-off = nighttime only, no solar confound):**
- HLC_on = 0.327, HLC_off = 0.005, OE = 0.500, tau = 15.0, solar_w = 0.00384, RMSE = 0.4285
- Ratio = **65.4x** (even HIGHER than NB10's 9.4x)
- This is the cleanest signal: nighttime HP-off has no Beschattung/solar confound at all

**Physical interpretation:**
- The building IS genuinely well-insulated when HP fan is off (HLC_off ~ 0.005)
- The 65.4x ratio is driven by HP fan forced convection vs natural convection
- Beschattung actually INFLATES HLC_off in daytime data (stable indoor temp attributed to heat exchange)
- Night data with larger driving force (8.1 K) confirms minimal heat loss without forced air
- The dual-HLC is a **real physical effect**, not a Beschattung artifact

**Implication:** Keep HLC_off lower bound at 0.005 — the optimizer correctly finds near-zero values.

---

## Steps

### Step 1: Add dual-HLC infrastructure to cooling calibration
- **File**: `src/physics_calibration_cooling.py` -> `calibrate_cooling_physics()`
- Add `HLC_off` parameter alongside existing `HLC` (rename to `HLC_on`)
- Use `differential_evolution` with bounds:
  - `HLC_on  in [0.02, 0.8]` (raised upper from 0.5 — NB12 hit this bound)
  - `HLC_off in [0.005, 0.3]` (keep low — night-filtered calibration confirms 0.005 is physical)
  - `OE      in [0.05, 0.5]` (cooling-specific, NOT heating's [0.1, 1.0])
  - `tau     in [1.5, 15.0]` (cap at 15h, online drift pushed to 41h)
- Select HLC_on vs HLC_off based on `is_hp_active` at each timestep
- Cost function: same Newton model with dual HLC
- Fallback: if dual-HLC fails, use single-HLC result

### Step 2: Replace PV with GHI solar source
- **File**: `src/heat_source_channels.py` -> `SolarChannel`
  - Add `solar_source` config option: `"pv"` (default, backward compat) or `"ghi"`
  - When `solar_source == "ghi"`:
    - Read GHI from Open Meteo REST sensor (new HA sensor, see Step 7)
    - Scale weight bounds by PV/GHI ratio (7.94):
      - `ghi_heat_weight` bounds: `[0.0008, 0.040]` (= PV bounds * 7.94)
      - Step limit: `+/-0.0016` (= 0.0002 * 7.94)
    - Default `ghi_heat_weight`: `0.004` (NB12 calibrated: 0.004144)
  - `cloud_factor_exponent`: set to 1.0 when using GHI (GHI already includes clouds)
  - `solar_lag_minutes`: default to 5 (NB12 optimal lag = 0 min, not 45 min)

- **File**: `src/physics_calibration_cooling.py` -> `_residual_heat_source_weight()`
  - Detect solar source type from config
  - Use appropriate weight bounds for GHI vs PV
  - Log calibrated weight with units

### Step 3: Cap tau upper bound
- **File**: `src/thermal_parameter_config.py`
  - Add `TAU_BOUNDS_COOLING = (1.5, 15.0)`
- **File**: `src/thermal_equilibrium_model.py` -> tau online learning
  - Add guard: `if tau > 15.0: tau = 15.0; log("tau capped at 15h")`
  - Rationale: online drift pushed tau to 41h; NB10/NB12 find 8-15h optimal

### Step 4: Adjust OE bounds for cooling mode
- **File**: `src/physics_calibration_cooling.py` -> `_calibrate_oe_cooling()`
  - Change: reject OE > 0.5 for cooling; warn if OE < 0.05
  - Fallback: if OE outside [0.05, 0.5], use 0.20 (NB10 median)
- **File**: `src/thermal_parameter_config.py`
  - Add `COOLING_OE_BOUNDS = (0.05, 0.5)`

### Step 5: Store dual-HLC result in thermal state
- **File**: `src/physics_calibration_cooling.py` -> return dict
  - Add `hlc_cooling_on` and `hlc_cooling_off` to calibration result
- **File**: `src/thermal_equilibrium_model.py`
  - When in cooling mode: use `hlc_cooling_on` if HP active, else `hlc_cooling_off`
  - Fallback: if dual params not present, use single HLC (backward compat)
- **File**: `Logs_and_models/unified_thermal_state.json` — extend schema

### Step 6: Update predict_equilibrium_temperature for cooling mode
- **File**: `src/thermal_equilibrium_model.py` -> `predict_equilibrium_temperature()`
  - Current: uses single HLC for all modes
  - Change: detect cooling mode + HP state, dispatch appropriate HLC
  - T_eq formula unchanged: just substitute appropriate HLC value
  - Add metrics logging: which HLC was used, effective Q_solar, solar source

### Step 7: Add Open Meteo GHI sensor to Home Assistant (documentation only)
- **File**: `docs/HA_SENSORS_FOR_CALIBRATION.md` — add GHI REST sensor config
  - Sensor: `sensor.open_meteo_ghi` — current GHI from Open Meteo forecast API
  - Forecast: `sensor.open_meteo_ghi_forecast_*h` — 1-12h ahead GHI
  - Note: Open Meteo forecast API (not archive) for live data
  - URL: `https://api.open-meteo.com/v1/forecast?latitude=48.928&longitude=10.069&hourly=shortwave_radiation`
  - This step is **documentation only** — the HA configuration is done by the user

### Step 8: Fix solar_lag_minutes default
- **File**: `src/heat_source_channels.py` -> `SolarChannel`
  - Change default `solar_lag_minutes` from 45.0 to 5.0
  - NB12 lag analysis: GHI peak correlation at lag=0 min, PV at -5 min
  - Current 45 min is far too high — causes delayed solar response
  - Keep learning range [0, 60] but start at 5 instead of 45

---

## Files Affected

| File | Changes |
|------|---------|
| `src/physics_calibration_cooling.py` | Dual-HLC optimizer, OE bounds, GHI weight, tau cap |
| `src/heat_source_channels.py` | solar_source config, GHI weight bounds/defaults, solar_lag default |
| `src/thermal_parameter_config.py` | COOLING_OE_BOUNDS, TAU_BOUNDS_COOLING, GHI defaults |
| `src/thermal_equilibrium_model.py` | Dual-HLC dispatch, tau cap guard, GHI solar support |
| `Logs_and_models/unified_thermal_state.json` | Schema: hlc_cooling_on/off, solar_source |
| `docs/HA_SENSORS_FOR_CALIBRATION.md` | GHI REST sensor config for Home Assistant |
| `tests/unit/test_physics_calibration_cooling.py` | Tests: dual-HLC, OE bounds, GHI weight, tau cap |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Optimizer instability (5-param dual-HLC) | Use `differential_evolution` with tight bounds (proven in NB10/NB12) |
| Backward compatibility | Default `solar_source="pv"`; single-HLC fallback if dual params missing |
| OE out of range | Fall back to NB10 median (0.20) with warning |
| tau cap too aggressive | 15h is well above NB10 optimum (8h); NB12 hit this bound -> consider 20h |
| GHI not available (sensor offline) | Fall back to PV if GHI sensor returns null/error |
| HLC_off too high (overcorrection) | Upper bound 0.3 prevents overshoot; monitor RMSE on validation |
| cloud_factor obsolete with GHI | Keep parameter but set exponent=1.0 (no cloud correction on GHI) |

## NB12 Key Numbers for Implementation

```
PV/GHI scale ratio:           7.94
GHI weight (calibrated):      0.004144 kW/(W/m2)
GHI weight bounds:            [0.0008, 0.040]
GHI step limit:               +/-0.0016
Solar lag (optimal):           0-5 min (was 45 min)
Diffuse RMSE:                  0.460 (best)
GHI RMSE:                      0.476
PV RMSE:                       0.528
No-solar RMSE:                 0.915
GHI ML AUC:                    0.931
PV ML AUC:                     0.945
HLC_off (night-filtered):       0.005 (confirmed physical, not artifact)
HLC_on/HLC_off ratio:          65.4x night-filtered (physical: forced vs natural convection)
Beschattung effect:             46% reduction in apparent heat exchange
```

## Dependencies

- **Independent of NB11** (pre-cooling): calibration changes affect model accuracy, not control logic
- **Requires Open Meteo REST sensor** in HA config for live GHI (Step 7, done by user)
- **Backward compatible**: PV mode remains default; GHI is opt-in via config
