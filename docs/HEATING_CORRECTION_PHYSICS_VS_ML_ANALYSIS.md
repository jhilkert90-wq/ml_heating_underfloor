# Heating Correction: Physics-Based vs ML-Based Analysis

_Generated 2026-05-15. Parameters sourced directly from `src/thermal_config.py` DEFAULTS._

---

## 1. Actual Default Parameters (thermal_config.py)

| Parameter | Symbol | Value | Unit |
|---|---|---|---|
| `outlet_effectiveness` | η | **0.830** | kW/K |
| `heat_loss_coefficient` | U | **0.124** | kW/K |
| `thermal_time_constant` | τ_room | **4.39** | h |
| `slab_time_constant_hours` | τ_slab | **3.19** | h |
| `delta_t_floor` | δT | **2.3** | °C |
| Cycle interval | Δt | 10 | min |
| Default trajectory horizon | H | 4 | h |

---

## 2. Derived Physics Quantities

### Equilibrium ratio

```
η / (η + U) = 0.830 / (0.830 + 0.124) = 0.830 / 0.954 = 0.870
```

87% of the indoor equilibrium temperature is driven by the outlet temperature; only 13% by
the outdoor temperature. This reflects the dominant role of the underfloor heating loop
relative to fabric heat loss.

### 4-hour horizon sensitivity (S_H)

This is the key quantity for physics-based correction: how much does a 1 °C change in outlet
temperature shift indoor temperature after H hours?

```
S_H = [η/(η+U)] × [1 − exp(−H/τ_room)]
    = 0.870 × [1 − exp(−4/4.39)]
    = 0.870 × 0.598
    = 0.5202  K_indoor / K_outlet
```

A 1 °C increase in outlet temperature moves indoor temperature by **0.52 °C** over the 4-hour
horizon.

### Single 10-minute cycle sensitivity (S_Δt)

```
S_Δt = 0.870 × [1 − exp(−0.167/4.39)] = 0.870 × 0.0373 = 0.0324  K/K
```

Each 1 °C change in outlet produces only **0.032 °C per 10-minute cycle**. Consequently:

- To close a 0.3 K undershoot in one cycle alone would require +9.3 °C outlet — physically
  impossible in most operating conditions.
- Multiple cycles are always required, which is the physical reason behind the slab
  equilibration delay.

### Slab equilibration time

```
τ_slab / Δt = 3.19 h / (10/60 h) ≈ 19 cycles ≈ 3.2 h
```

The slab needs roughly 19 cycles (3.2 hours) to equilibrate to a new outlet temperature.

---

## 3. Calculated Examples

### Example 1 — Undershoot ε = 0.3 K

**Setup:** T_outdoor = 3 °C, T_target = 21 °C, T_current = 20.7 °C, outlet = 25 °C

At the current outlet temperature the equilibrium is:

```
T_eq = (η × 25 + U × 3) / (η + U) = (0.83×25 + 0.124×3) / 0.954 = 22.14 °C
```

The room will eventually reach 22.14 °C, so the **baseline already exceeds the target** —
no structural correction is needed; only timing matters.

| Approach | Formula | Correction | Corrected outlet | T_peak (10 h) |
|---|---|---|---|---|
| Physics | ε / S_H = 0.3 / 0.5202 | **+0.577 °C** | 25.58 °C | 22.45 °C |
| Current code | 0.3 × 1.325 × 3.0 × 1.094 | **+1.305 °C** | 26.30 °C | 23.01 °C |
| No correction | — | 0 °C | 25.00 °C | 21.99 °C |

Current code over-corrects by a factor of **2.26×** relative to the physics optimum.
The equilibrium under the code's correction is 23.27 °C — a future overshoot of ~2 °C
that then requires a negative correction next cycle, which can create sustained oscillation.

Code factors:
```python
base_scale       = min(1.0 / outlet_effectiveness * 1.1, 2.5)  # min(1/0.83 * 1.1, 2.5) = 1.325
aggression_factor = min(1.0 + initial_deviation / slab_tau, 2.0)  # min(1 + 0.3/3.19, 2.0) = 1.094
urgency_multiplier = 1.0 + 2.0 * (time_pressure ** 2)  # 1.0 + 2.0 * 1.0**2 = 3.0
correction       = temp_error * physics_scale * urgency_multiplier * aggression_factor
#                = 0.3 * 1.325 * 3.0 * 1.094 = 1.305 °C
```

### Example 2 — Small undershoot ε = 0.15 K

| Approach | Correction | Corrected outlet |
|---|---|---|
| Physics | 0.15 / 0.5202 = **+0.288 °C** | 25.29 °C |
| Current code | 0.15 × 1.325 × 3.0 × 1.047 = **+0.624 °C** | 25.62 °C |
| Ratio | **2.17× too aggressive** | — |

### Example 3 — Overshoot ε = −0.4 K

**Setup:** T_outdoor = 5 °C, T_target = 21 °C, T_current = 21 °C, outlet = 35 °C,
trajectory max predicted = 21.4 °C

| Approach | Formula | Correction | Corrected outlet |
|---|---|---|---|
| Physics | −0.4 / S_H = −0.4 / 0.5202 | **−0.769 °C** | 34.23 °C |
| Current code | −0.4 × 1.325 × 3.0 × 0.3135 | **−0.499 °C** | 34.50 °C |
| Ratio | **0.65× too gentle** | — | — |

For overshoot the pattern **reverses**: the code under-corrects because
`overshoot_dampening = 1/τ_slab = 1/3.19 = 0.313` is much smaller than the physics value.
The room continues to overshoot after the correction is applied.

### Downstream trajectory comparison (Example 1)

Simulated indoor temperature for each approach after 10 hours:

| Approach | T_eq | T @ 1 h | T @ 4 h | T_peak |
|---|---|---|---|---|
| Physics (+0.58 °C) | 22.65 °C | 21.10 °C | 21.86 °C | 22.45 °C |
| Code (+1.30 °C) | 23.27 °C | 21.22 °C | 22.24 °C | 23.01 °C |
| Baseline (no corr) | 22.14 °C | 20.99 °C | 21.56 °C | 21.99 °C |

The code's correction causes the room to overshoot the target by ~2 °C over the next
few hours, whereas the physics correction limits overshoot to ~1.5 °C.

---

## 4. Sensitivity Table — S_H across parameter space

How the required outlet correction (for ε = 0.3 K) varies with horizon and time constant:

| Horizon H | τ_room | S_H | ΔT_outlet needed |
|---|---|---|---|
| 1 h | 4.39 h | 0.177 | 1.69 °C |
| 2 h | 4.39 h | 0.318 | 0.94 °C |
| **4 h** | **4.39 h** | **0.520** | **0.58 °C** |
| 6 h | 4.39 h | 0.648 | 0.46 °C |
| 4 h | 3.00 h | 0.641 | 0.47 °C |
| 4 h | 6.00 h | 0.423 | 0.71 °C |
| 4 h | 8.00 h | 0.342 | 0.88 °C |

**Key insight:** Larger horizon or smaller τ_room → larger S_H → smaller correction needed.
The current code ignores S_H entirely, applying a fixed multiplicative formula regardless
of the actual physical sensitivity.

---

## 5. Physics-Based MPC Schedule

For the standard undershoot scenario (ε = 0.3 K, outlet = 25 °C):

| Stage | Duration | Outlet setpoint | Purpose |
|---|---|---|---|
| **Charging** | Cycles 1–19 (~3.2 h) | 25.58 °C (+0.58 °C) | Slab equilibrates, room approaches target |
| **Maintenance** | Cycle 20+ | 25.00 °C | Holds at equilibrium |

This 2-stage schedule is the minimum-overshoot profile that respects the slab time constant.
The equilibrium outlet for exact maintenance is:

```
T_outlet_maint = (T_target × (η+U) − U × T_outdoor) / η
               = (21 × 0.954 − 0.124 × 3) / 0.83
               = 23.69 °C  (clamped to 25 °C minimum in heating mode)
```

Simulated indoor trajectory (first-order, no slab model):

| t | T_indoor |
|---|---|
| 0.2 h | 20.77 °C |
| 0.5 h | 20.91 °C |
| **0.67 h** | **~21.00 °C ← target reached** |
| 1.0 h | 21.10 °C |
| 2.0 h | 21.41 °C |
| 4.0 h | 21.79 °C |
| 8.0 h | 22.00 °C |

---

## 6. The Fundamental Problem with the Current Code

The current `_calculate_physics_based_correction()` in `model_wrapper.py` uses the formula:

```python
base_scale         = min(1.0 / outlet_effectiveness * 1.1, 2.5)  # = 1.325
physics_scale      = base_scale
aggression_factor  = min(1.0 + initial_deviation / slab_tau, 2.0)  # ~1.0–2.0
time_pressure      = self._calculate_time_pressure(trajectory, cycle_hours)  # 0.0–1.0
urgency_multiplier = 1.0 + 2.0 * (time_pressure ** 2)  # = 3.0 at max time-pressure

# Undershoot (temp_error > 0):
correction = temp_error * physics_scale * urgency_multiplier * aggression_factor

# Overshoot (temp_error < 0):
overshoot_dampening = 1.0 / max(slab_tau, 1.0)  # = 0.313 for slab_tau=3.19
correction = temp_error * physics_scale * urgency_multiplier * overshoot_dampening
```

**Problems:**

1. **`urgency_multiplier` = 3.0 inflates every correction by 3×.** The urgency factor was designed for
   time-pressure situations but is always 3.0 when the target hasn't been reached yet.

2. **Asymmetric treatment:** undershoot multiplies by `aggression_factor` (~1.0–2.0),
   overshoot multiplies by `overshoot_dampening = 1.0 / max(slab_tau, 1.0) = 0.313`. Physics has no such
   asymmetry — S_H is the same in both directions.

3. **No horizon-awareness:** The correction doesn't depend on the prediction horizon H.
   A 4-hour horizon requires half the outlet change that a 2-hour horizon would need.

4. **No equilibrium awareness:** The baseline outlet already produces an equilibrium above
   the target in many operating conditions (as shown in Example 1). Applying a full
   urgency-scaled correction on top causes systematic overshoot.

**The correct physics-based correction is simply:**

```
ΔT_outlet = ε / S_H
           = (T_target − T_predicted_H) / {[η/(η+U)] × [1 − exp(−H/τ_room)]}
```

This is a single Newton step that exactly compensates the predicted error at horizon H.

---

## 7. ML-Based Correction: Design

Analogous to `CoolingMLModel` / `cooling_ml_calibration.py`, a heating correction ML model
would be a **LightGBM regression model** predicting the required outlet temperature correction.

### Training label construction

```python
# For each timestep t in history:
correction_needed[t] = T_indoor_actual[t + N_cycles] - T_target[t]
# Label: ΔT_outlet that would have zeroed this error
delta_outlet_label[t] = -correction_needed[t] / S_H_estimated
```

This is analogous to the cooling ML's hindcast label (did indoor temp exceed
`cooling_target` within `lead_time_h`?), but as a regression target instead of binary.

### Proposed feature vector

| Feature | Source key | Description |
|---|---|---|
| `indoor_temp` | `physics["indoor_temp"]` | Current indoor temperature |
| `indoor_margin` | `target - indoor_temp` | Undershoot (positive) or overshoot (negative) |
| `indoor_trend_30m` | `physics["indoor_temp_delta_30m"]` | 30-min indoor temperature trend |
| `indoor_trend_1h` | `physics["indoor_temp_delta_60m"]` | 60-min indoor temperature trend |
| `outlet_temp` | `physics["outlet_temp"]` | Current VLT |
| `inlet_temp` | `physics["inlet_temp"]` | Current RLT (slab state proxy) |
| `delta_t_floor` | `physics["delta_t"]` | BT2 − BT3 (HP load indicator) |
| `AT` | `physics["outdoor_temp"]` | Outdoor temperature |
| `AT_roh_1h` | `physics["temp_forecast_1h"]` | 1-h outdoor forecast |
| `AT_roh_2h` | `physics["temp_forecast_2h"]` | 2-h outdoor forecast |
| `AT_roh_4h` | `physics["temp_forecast_4h"]` | 4-h outdoor forecast |
| `trajectory_error_4h` | trajectory output | T_predicted_4h − T_target |
| `hour_sin` | `physics["hour_sin"]` | Time-of-day cyclical encoding |
| `hour_cos` | `physics["hour_cos"]` | Time-of-day cyclical encoding |
| `doy_sin` | `datetime.now()` | Day-of-year cyclical encoding |
| `doy_cos` | `datetime.now()` | Day-of-year cyclical encoding |

### Proposed module structure

```
src/heating_correction_ml_model.py        # Inference  (mirrors cooling_ml_model.py)
src/heating_correction_ml_calibration.py  # Training   (mirrors cooling_ml_calibration.py)
```

Training is triggered via `--calibrate-heating-correction-ml` (CLI) or a flag file,
exactly like `--calibrate-cooling-ml`.

---

## 8. Physics-Based vs ML-Based Correction: Comparison

| Aspect | Physics-Based (`ε / S_H`) | ML-Based (LightGBM regressor) |
|---|---|---|
| **Core formula** | `ΔT = ε / S_H` — 1 formula, 2 calibrated params | Nonlinear function of ~16 features learned from data |
| **Data requirements** | None — uses already-calibrated η, U, τ_room | ~90 days of heating history (same as cooling ML) |
| **Cold start** | ✅ Works on day 1 | ❌ Needs data collection phase |
| **Interpretability** | ✅ High — every number has physical meaning | ❌ Black box |
| **Systematic bias** | ✅ None — Newton step is exact for the linear model | ✅ None — regressor trained to minimise residuals |
| **Unmeasured effects** | ❌ Occupancy, drafts, window opening not captured | ✅ Captures correlations with time-of-day, season |
| **Partial-load η variation** | ❌ Assumes constant η | ✅ Learns actual η(outlet, load) from data |
| **Slab state** | ❌ Only via τ_slab time constant | ✅ `delta_t_floor` and `inlet_temp` are direct features |
| **Recalibration** | Automatic — uses current `baseline_parameters` | Manual — periodic `--calibrate-heating-correction-ml` |
| **Overcorrection risk** | Low — single Newton step is proportional | Medium — can overfit noisy training data |
| **Accuracy ceiling** | Bounded by physics model accuracy (~5–10% from η variation) | Potentially higher if training data is rich |
| **When physics params are wrong** | Error scales with calibration error | Robust — doesn't rely on calibrated params |
| **Computational cost** | Negligible (3 multiplications) | Moderate (tree inference; fast in practice) |

### When to use which approach

**Use physics correction when:**
- The system has been recently re-calibrated (fresh `baseline_parameters`)
- Cold-start conditions (< 90 days of data)
- Interpretability is important for debugging
- Thermal behaviour is stable and well-characterised

**Use ML correction when:**
- The house has strong unmeasured heat sources (solar gain through windows, occupancy)
- Calibrated η/U are known to have systematic errors (mixed zone use)
- The `calibrate_cooling_ml` pipeline is already in place
- You want the correction to adapt automatically to seasonal η changes

### Recommended blended approach

Run both in parallel and weight by ML model confidence (out-of-sample R²):

```python
w = ml_model.r2_score  # 0.0 → 1.0
delta_outlet = (1 - w) * delta_physics + w * delta_ml
```

This mirrors the `PRE_COOL_MODEL_TYPE` soft-switch between `OverheatingPredictor` and
`CoolingMLModel`, but as a continuous blend rather than a hard switch. If the ML model
is undertrained (low R²), the system falls back to pure physics automatically.

---

## 9. Summary of Findings

1. **The current code over-corrects undershoot by ~2.2× and under-corrects overshoot by
   ~0.65×**, primarily because `urgency_multiplier = 3.0` is always applied to undershoot and
   `overshoot_dampening = 1.0 / max(slab_tau, 1.0) = 0.313` is always applied to overshoot.

2. **The correct physics correction is `ΔT = ε / S_H`**, where
   `S_H = 0.870 × [1 − exp(−H/τ_room)] = 0.5202` at the 4-hour horizon with
   `τ_room = 4.39 h`.

3. **A single Newton step of +0.577 °C** suffices for a 0.3 K undershoot at a 25 °C
   outlet. The code currently applies +1.305 °C, pushing equilibrium to 23.27 °C and
   causing a ~2 °C future overshoot.

4. **An ML-based correction** (LightGBM regressor, modelled after `CoolingMLModel`)
   can capture effects that the physics model ignores (unmeasured heat sources, seasonal
   η variation, slab non-linearity) once sufficient historical data is available, but
   should be blended with physics on a confidence-weighted basis.
