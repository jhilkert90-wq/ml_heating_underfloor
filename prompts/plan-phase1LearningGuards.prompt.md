# Plan: Phase 1 Learning Guards + Slab Pump-OFF Fix

## TL;DR

Stop gradient descent contamination from fireplace/PV/pump-OFF states and fix HLC drift — all without `ENABLE_HEAT_SOURCE_CHANNELS`. Deploys as a minimal, safe patch to the existing learning loop.

---

## Problem Summary

- **HLC drifted -15%** (0.146→0.124): model underestimates heat loss by ~0.5 kW → primary night-time undershoot cause
- **OE drifted +0.7%**: minor but compounds with HLC → +0.46°C equilibrium over-prediction
- **Gradient learning pump-OFF bug**: `context["outlet_temp"]` = Sollwert always > inlet → slab model never simulates pump-OFF in gradient trajectory → wrong parameter updates
- **No fireplace learning guard**: fireplace heat contaminates OE↑ and HLC↓
- **Cold weather protection gap (0°C to -5°C)**: common overnight temps have no HLC-reduction damping

---

## Steps

### Step 1: Fireplace learning guard
- **File**: `src/thermal_equilibrium_model.py` → `_adapt_parameters_from_recent_errors()` ~L860
- When ANY recent_prediction has `context.fireplace_on > 0`: skip OE, HLC, τ, slab_τ gradients
- Still allow pv_heat_weight, tv_heat_weight, fireplace_heat_weight to learn
- Log: "🔥 Skipping HP parameter learning: fireplace active in recent window"

### Step 2: Tag prediction records
- **File**: `src/thermal_equilibrium_model.py` → `update_prediction_feedback()` ~L773
- Add `"fireplace_active": bool(prediction_context.get("fireplace_on", 0))` to prediction_record
- Add `"pump_off": bool(context.get("delta_t", 999) < 1.0)` to prediction_record

### Step 3: Filter contaminated records in gradient calculation
- **File**: `src/thermal_equilibrium_model.py` → `_calculate_parameter_gradient()` ~L1310
- For HP params (OE, HLC, τ, slab_τ): skip records where `pred.get("fireplace_active", False)`
- For HP params: also skip records where `pred.get("pump_off", False)`
- For source weights (pv_weight, fp_weight, tv_weight): use all records

### Step 4: Pump-OFF fix in gradient trajectory simulation
- **File**: `src/thermal_equilibrium_model.py` → `_calculate_parameter_gradient()` ~L1335
- Before calling `predict_thermal_trajectory()`:
  - Check `context.get("delta_t", 999) < 1.0` OR `context.get("thermal_power_kw", 999) < 0.2`
  - If pump was off: pass `outlet_temp=context["inlet_temp"]` instead of `context["outlet_temp"]`
  - This forces slab model into Pump-OFF branch (inlet ≤ inlet → t_effective = t_slab)
  - Otherwise: keep using `context["outlet_temp"]` (Sollwert) — correct "what if" simulation
- **Rationale**: Binary search correctly uses Sollwert as candidate. But gradient learning re-uses the Sollwert from the LAST cycle — if pump was off, the Sollwert was irrelevant, and simulating Pump-ON with it produces wrong gradients.

### Step 5: High-PV learning dampening
- **File**: `src/thermal_equilibrium_model.py` → `_adapt_parameters_from_recent_errors()` ~L940
- When avg PV across recent_predictions > 500W: apply 0.3× dampening to OE/HLC updates
- Stacks with existing cold weather / indoor trend protection dampening

### Step 6: Extend cold weather protection range
- **File**: `src/thermal_constants.py` → `COLD_WEATHER_PROTECTION_THRESHOLD`
- Change from -5.0°C to 2.0°C (covers common overnight range 0..2°C)
- Keep `EXTREME_COLD` at -7°C (0.01 damping)
- Alternative: make configurable via env var

### Step 7: HLC drift safety floor
- **File**: `src/thermal_equilibrium_model.py` → `_adapt_parameters_from_recent_errors()`
- After HLC update: enforce `HLC >= baseline × 0.85` (max -15% drift from calibrated value)
- Current delta of -0.0218 (-15%) would be at the limit
- Log warning when floor is hit

---

## Relevant Files

| File | Steps | Changes |
|------|-------|---------|
| `src/thermal_equilibrium_model.py` | 1-5, 7 | Guards, filters, pump-off fix, HLC floor |
| `src/thermal_constants.py` | 6 | Cold weather threshold |
| `src/config.py` | 6 alt | Optional env var for threshold |
| `src/main.py` | — | No changes (already provides inlet_temp, delta_t, thermal_power_kw) |

---

## Verification

1. **Unit test**: FP active → OE/HLC/τ/slab_τ unchanged, pv_weight can still learn
2. **Unit test**: pump_off records → skipped for HP gradients
3. **Unit test**: pump_off → gradient simulation uses inlet_temp, not Sollwert
4. **Unit test**: HLC cannot drift below 85% of baseline
5. **Existing**: `test_learning_stability.py` — all 9 tests still pass
6. **Manual**: deploy, monitor parameter_history for stable HLC/OE overnight

---

## Analysis Notes

- Slab model structurally correct: `(outlet + slab) / 2` used everywhere needed
- Binary search pump ON/OFF logic correct — simulates "what if outlet = X"
- Bug is gradient learning: `context["outlet_temp"]` = Sollwert, always > inlet → always Pump-ON
- Combined HLC (-15%) + OE (+0.7%) drift → +0.46°C equilibrium over-prediction
- Cold weather protection gap at 0..−5°C left HLC unprotected during typical overnight conditions

---

## Post-Deploy Actions

- Consider manually resetting `heat_loss_coefficient_delta` to 0.0 after deploy
- Monitor parameter_history for 2+ weeks
- If fireplace is infrequent, Phase 1 guard alone may suffice — Phase 2-4 optional
