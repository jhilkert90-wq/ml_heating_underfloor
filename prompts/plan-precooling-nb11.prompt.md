# Plan: Adapt Pre-Cooling with NB11 Cycle Analysis Results

## TL;DR

Apply findings from **notebook 11** (cooling cycle analysis) to improve the pre-cooling trigger, trajectory prediction, and recovery gating. Addresses LGBM threshold being too low (always triggers), Newton correction being too large (median −0.9°C suggests OE error), and RECOVERY gate consuming 63% of cycles.

---

## Evidence Summary (from notebooks/analysis/11_cooling_cycle_analysis.ipynb)

| Finding | Current | NB11 Result | Action |
|---------|---------|-------------|--------|
| LGBM threshold | 0.345 | min output = 0.417 → always triggers | Raise to 0.55-0.65 or shadow-mode |
| LGBM std | — | bimodal, std=0.23 | Poor discrimination |
| Trajectory | — | rejects 12% of cycles | More discriminative than LGBM |
| Newton correction | hardcoded | median = −0.9°C, applied 53% | Should shrink after OE fix (Phase 2) |
| Gate distribution | — | 63% RECOVERY, 37% RUNNING | Too much idle time |
| Prediction MAE | — | 0.036°C | Prediction is accurate |
| Threshold sweet spot | — | 0.60-0.65 | Best discrimination range |

### Key Insights
- **LGBM always fires** because threshold (0.345) is below minimum output (0.417) — it never says "don't cool"
- **Trajectory simulation** is more valuable: rejects 12% of cycles based on physics (Newton model)
- **Newton correction −0.9°C** compensates for OE being too low in cooling mode → should reduce after Phase 2 OE fix
- **RECOVERY gate 63%** means the system spends more time recovering than actively cooling — consider shorter RECOVERY period or adaptive duration
- **Threshold sensitivity** (Phase G): 0.60-0.65 is the sweet spot for meaningful LGBM discrimination

---

## Steps

### Step 1: Raise LGBM trigger threshold
- **File**: `src/overheating_predictor.py`
- Current: `trigger_threshold = target + 0.5` (margin)
- LGBM threshold: 0.345 (hardcoded or from metadata)
- Change: raise LGBM probability threshold to 0.55 (conservative) or 0.60 (NB11 optimal)
- **Alternative**: shadow-mode LGBM (log predictions but don't act) while trajectory-only drives decisions
- Add config parameter: `lgbm_trigger_threshold` (default 0.55, range [0.3, 0.8])

### Step 2: Make trajectory margin configurable
- **File**: `src/overheating_predictor.py`
- Current: `trigger_threshold = target + 0.5` — hardcoded 0.5°C margin
- Change: make margin a config parameter `trajectory_margin_celsius` (default 0.5, range [0.2, 1.0])
- NB11 shows prediction MAE = 0.036°C → 0.5°C margin is very conservative
- Consider: reduce to 0.3°C to trigger earlier, catching more overshoot events

### Step 3: Monitor Newton correction magnitude
- **File**: `src/thermal_equilibrium_model.py` or `src/overheating_predictor.py`
- NB11: median Newton correction = −0.9°C, applied 53% of cycles
- After Phase 2 (OE fix): expect Newton correction to shrink
- Add metric: `cooling_newton_correction_mean` to HA sensor attributes
- Add alert: if |Newton correction| > 1.5°C sustained, log warning about OE miscalibration
- **DO NOT** change correction logic now — wait for Phase 2 OE fix to take effect first

### Step 4: Adaptive RECOVERY gate duration
- **File**: `src/overheating_predictor.py` or `src/main.py` (control loop)
- Current: RECOVERY phase has fixed or implicit duration → 63% of cycles
- Change: make RECOVERY duration adaptive:
  - After HP-OFF: check indoor_temp trend every 5 min
  - If `indoor_trend_30m > -0.05°C/h` for 15 min → exit RECOVERY (temp stabilized)
  - Maximum RECOVERY duration: 30 min (configurable)
- Goal: reduce RECOVERY from 63% to ~40% of cycle time

### Step 5: LGBM shadow-mode logging
- **File**: `src/overheating_predictor.py`
- Before changing the threshold, deploy shadow mode:
  - Always compute LGBM prediction
  - Always compute trajectory prediction
  - Log both with `would_trigger` flag to HA sensor attributes
  - Let trajectory alone drive decisions for 2 weeks
- After shadow period: compare which predictions match actual overshoot → choose best threshold
- Add config: `lgbm_mode` ∈ {`active`, `shadow`, `disabled`} (default: `shadow`)

### Step 6: PV guard gate adjustment
- **File**: `src/overheating_predictor.py`
- Current: `total PV > 1000W OR peak outdoor > 22°C`
- NB11 shows PV features are mid-tier in LGBM importance
- After NB12 (GHI replacement): change PV guard to GHI-based threshold
- Placeholder: keep PV guard as-is, mark with `TODO: replace with GHI after NB12`

---

## Files Affected

| File | Changes |
|------|---------|
| `src/overheating_predictor.py` | LGBM threshold, trajectory margin, PV guard, shadow-mode |
| `src/config.py` | New config params: lgbm_trigger_threshold, trajectory_margin, lgbm_mode |
| `src/thermal_equilibrium_model.py` | Newton correction monitoring metric |
| `src/main.py` | RECOVERY gate adaptive duration |
| `tests/unit/test_overheating_predictor.py` | Tests for new thresholds, shadow-mode |

---

## Implementation Order

1. **Step 5 (shadow-mode)** first — zero risk, collects data
2. **Step 1 (threshold)** — apply after 1-2 weeks of shadow data
3. **Step 2 (margin)** — simple config change
4. **Step 4 (RECOVERY)** — most complex, implement after Steps 1-2 stabilize
5. **Step 3 (Newton monitoring)** — passive monitoring, deploy anytime
6. **Step 6 (PV guard)** — wait for NB12 results

## Dependencies

- **Depends on Phase 2** (NB10 calibration): Newton correction should shrink after OE fix → Step 3 validates this
- **Depends on NB12** (solar replacement): Step 6 (PV guard) needs GHI threshold
- **Independent of Phase 1** (learning guards): these are control-logic changes, not learning changes

## Risks & Mitigations

- **Raising LGBM threshold too high**: could miss real overshoot events → shadow-mode first (Step 5)
- **Shorter RECOVERY**: could cause rapid on/off cycling → add minimum cycle time guard (already exists?)
- **Trajectory-only**: if Newton model has drift, trajectory alone may be unreliable → keep LGBM as fallback, never fully disable
