# Plan: Decomposed Heat-Source Learning Architecture

## TL;DR

The current thermal model uses a **single set of learned parameters** (HLC, OE, τ, etc.) that get contaminated when uncontrollable heat sources (fireplace, solar) are active — the gradient descent cannot distinguish which source caused a prediction error, so it wrongly adjusts heat pump parameters. **Recommended approach**: Phase 1 (quick guard) + Phase 2-4 (full channel decomposition, optional based on monitoring).

---

## Problem Analysis

### Current Architecture (Single Model)

```
T_eq = (OE × T_outlet + HLC × T_outdoor + Q_ext) / (OE + HLC)
Q_ext = pv_heat + fireplace_heat + tv_heat
```

ALL 8 parameters learned from a single gradient descent every cycle.

### Contamination Evidence (unified_thermal_state.json)

- `outlet_effectiveness_delta: +0.057` — model thinks HP is 6% more effective than reality
- `heat_loss_coefficient_delta: -0.004` — model thinks house is better insulated
- Both caused by fireplace heat misattributed to heat pump
- **Zero fireplace-specific guards exist** in gradient code

### The equilibrium formula itself is correct

The problem isn't the physics — it's that the **learned parameters** (OE, HLC) are trained on data contaminated by unmodeled heat.

### Each Heat Source Has Different Dynamics

| Source | Controllable | Response Time | Decay After Off | Buffer |
|--------|-------------|---------------|-----------------|--------|
| Heat Pump (FBH) | ✅ Yes | Hours (slab τ ≈ 1h) | Very slow (estrich τ ≈ 4h) | High |
| Solar (PV) | ❌ No | ~45 min lag | Immediate (no thermal mass) | None |
| Fireplace | ❌ No | Fast (convection) | Medium (body radiates ~30-60min) | Medium |

---

## Phase 1: Learning Guards (Quick Win — Deploy First)

**Goal**: Stop gradient descent from running when uncontrollable sources are active, preventing further HP parameter corruption.

### Steps

1. **Fireplace learning guard** in `thermal_equilibrium_model.py` `update_prediction_feedback()` ~L665
   - When `prediction_context.get("fireplace_on", 0) > 0`: skip OE, HLC, τ gradient updates
   - Still allow pv_heat_weight and fireplace_heat_weight to learn
   - Log: "Skipping HP parameter learning: fireplace active"

2. **High-PV learning dampening** in same method
   - When `prediction_context.get("pv_power", 0) > 500W`: apply 0.3× dampening to OE/HLC gradients
   - Rationale: Solar adds heat that gets misattributed to HP; dampening prevents over-fitting

3. **Tag prediction_history entries** with `fireplace_active` flag ~L713
   - Store `fireplace_active: bool` in each prediction record
   - Allows gradient calculation to skip contaminated records

4. **Filter contaminated records in gradient calculation** in `_calculate_parameter_gradient()` ~L1310
   - For HP parameters (OE, HLC, τ, slab_τ): skip recent_predictions where `fireplace_active == True`
   - For source weights (pv_heat_weight, fireplace_heat_weight): use all records

### Files
- `src/thermal_equilibrium_model.py` — guards in `update_prediction_feedback()`, filter in `_calculate_parameter_gradient()`
- `src/model_wrapper.py` — `learn_from_prediction_feedback()` L1622

### Tests
- `tests/unit/test_learning_isolation.py` — verify HP params unchanged when FP active
- Existing `test_learning_stability.py` — all 9 tests still pass

---

## Phase 2: Heat Source Channel Architecture

**Goal**: Decompose total indoor heat into 3 independent channels, each with its own learned parameters and temporal dynamics.

### Architecture

```
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │  HP Channel   │   │ Solar Channel │   │  FP Channel   │
  │  (controlled) │   │ (forecast)   │   │ (observed)    │
  │               │   │               │   │               │
  │ OE, HLC, τ,  │   │ pv_weight,   │   │ fp_weight,    │
  │ slab_τ, Δt   │   │ solar_lag,   │   │ fp_decay_τ,   │
  │               │   │ cloud_factor │   │ room_spread_τ │
  │ Decay: slab   │   │ Decay: none  │   │ Decay: exp    │
  │ model (slow)  │   │ (immediate)  │   │ (medium)      │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                   │                   │
         └───────────┬───────┘───────────────────┘
                     ▼
              Q_total = Q_hp + Q_solar + Q_fireplace
              T_eq = T_outdoor + Q_total / HLC
```

### Steps

5. **Create `HeatSourceChannel` base class** in new `src/heat_source_channels.py`
   - Abstract interface: `estimate_heat_contribution(context) → float (kW)`
   - Abstract: `estimate_decay_contribution(time_since_off, context) → float (kW)`
   - Abstract: `get_learnable_parameters() → Dict[str, float]`
   - Abstract: `apply_gradient_update(gradients, learning_rate)`

6. **Implement `HeatPumpChannel`** (wraps existing slab model)
   - Parameters: outlet_effectiveness, slab_time_constant, delta_t_floor
   - Heat contribution: `OE × (outlet_temp - indoor_temp)`
   - Decay: Existing slab Euler model (pump-off → passive cooling toward indoor)
   - Learning: Only from cycles where fireplace=OFF AND pv < 100W

7. **Implement `SolarChannel`** (wraps existing PV logic)
   - Parameters: pv_heat_weight, solar_lag_minutes, cloud_factor_exponent
   - Heat contribution: `effective_pv × pv_weight × cloud_factor`
   - Decay: None (immediate drop when sun sets) — key for evening forecast scenario
   - Learning: Only from daytime cycles (PV > 0)
   - Forecast-aware: Uses pv_forecast array to predict future contribution

8. **Implement `FireplaceChannel`** (wraps adaptive_fireplace_learning)
   - Parameters: fp_heat_output_kw, fp_decay_time_constant, room_spread_delay_minutes
   - Heat contribution: `fp_heat_output_kw × effectiveness(learned_from_differential)`
   - Decay: Exponential decay after fireplace off (`fp_decay_τ` ~ 30-60 min)
   - Room spread: Living room heat → house average with delay (~20-40 min)
   - Learning: Only from fireplace-active sessions (uses living_room_temp differential)

9. **Create `HeatSourceOrchestrator`** in `src/heat_source_channels.py`
   - Combines all 3 channels: `Q_total = Q_hp(ctx) + Q_solar(ctx) + Q_fp(ctx)`
   - Routes learning updates to correct channel based on active sources
   - Provides `predict_future_heat(horizon_hours, forecasts) → array[Q_total_per_step]`
   - Manages channel state persistence (extend unified_thermal_state.json)

10. **Integrate orchestrator into `ThermalEquilibriumModel`**
    - Replace `external_thermal_power = pv + fireplace + tv` with `orchestrator.total_heat(context)`
    - In `predict_thermal_trajectory()`: Use `orchestrator.predict_future_heat()` per step
    - In `predict_equilibrium_temperature()`: Use orchestrator for energy-based mode
    - Keep temperature-based fallback as-is

11. **Integrate orchestrator into binary search** in `model_wrapper.py`
    - `_calculate_required_outlet_temp()`: asks "what outlet_temp gives target, given solar+fireplace from orchestrator?"
    - Solar forecast naturally handles evening transition: PV dropping → Q_solar drops → binary search increases outlet_temp ahead of time

### Files
- `src/heat_source_channels.py` — **NEW**: base class + 3 implementations + orchestrator
- `src/thermal_equilibrium_model.py` — integrate orchestrator into equilibrium + trajectory
- `src/model_wrapper.py` — integrate into binary search
- `src/config.py` — per-channel config vars + bounds
- `ml_heating_data/unified_thermal_state.json` — extend with per-channel state

### Tests
- `tests/unit/test_heat_source_channels.py` — **NEW**: channel unit tests
- Each channel: heat estimate, decay, parameter bounds

---

## Phase 3: Channel-Isolated Learning

**Goal**: Each channel learns from its own data, preventing cross-contamination.

### Steps

12. **Channel-isolated gradient descent**
    - HP channel gradients: Only from cycles where FP=OFF, PV<100W (night/overcast)
    - Solar channel gradients: From daytime cycles, error after subtracting HP contribution
    - Fireplace channel gradients: From FP-active cycles, error after subtracting HP+solar
    - Key: Each channel uses OTHER channels' current estimates as fixed when computing its own gradient

13. **Attribution algorithm** in orchestrator:
    ```python
    error = actual_temp - predicted_temp
    Q_hp = hp_channel.estimate(context)
    Q_solar = solar_channel.estimate(context)
    Q_fp = fp_channel.estimate(context)

    if only HP active:
        hp_channel.learn(error)
    elif HP + solar active:
        hp_learns = error × Q_hp / (Q_hp + Q_solar)   # proportional
        solar_learns = error × Q_solar / (Q_hp + Q_solar)
    elif all 3 active:
        # proportional attribution across all channels
    ```

14. **Per-channel prediction history** in unified_thermal_state.json
    - Each channel tracks its own prediction quality
    - Enables monitoring which channel is accurate

### Files
- `src/heat_source_channels.py` — learning methods per channel
- `src/thermal_equilibrium_model.py` — delegate gradient routing
- `ml_heating_data/unified_thermal_state.json` — per-channel history

### Tests
- `tests/unit/test_learning_isolation.py` — **NEW**: FP ON → HP params unchanged
- Contamination regression: 100 cycles FP ON/OFF, OE/HLC drift < 2%

---

## Phase 4: Evening Solar Transition (Key Scenario)

**Goal**: Proactively increase heat pump output before sunset so indoor temp stays stable.

### Scenario
> Nachts läuft Wärmepumpe, schaltet um 7 Uhr ab/reduziert (Sonne scheint). Indoor stabil durch Blinds. Estrich kühlt aus. Abends geht Sonne unter → sofort keine Wärme mehr. Skript muss rechtzeitig die WP-Leistung erhöhen, damit Estrichtemperatur wieder passend zur Outdoor-Temp ist.

### Steps

15. **Solar forecast integration in trajectory prediction**
    - Already partially implemented: `pv_forecast` array in trajectory steps
    - Enhancement: Solar channel provides `Q_solar_forecast[t]` per trajectory step
    - When forecast shows PV → 0 in 2h, solar contribution drops to 0 immediately (no decay)
    - Binary search automatically compensates: needs higher outlet_temp

16. **Slab pre-heating verification**
    - When binary search sees solar dropping, it increases outlet_temp NOW
    - Slab model: higher outlet → slab stores more heat → inlet_temp rises
    - When sun sets, slab has thermal reserve to bridge the gap
    - Verify with specific test case

17. **Solar transition test scenario**
    - Test: sunny afternoon (PV=3000W) → evening (PV drops to 0 over 2h)
    - Verify: required_outlet_temp increases 1-2h before sunset
    - Verify: indoor_temp stays within ±0.2°C of target through transition

### Files
- `src/heat_source_channels.py` — `SolarChannel.predict_future_contribution()`
- `src/thermal_equilibrium_model.py` — trajectory uses channel forecasts
- `tests/unit/test_solar_transition.py` — **NEW**

---

## All Relevant Files

| File | Status | Purpose |
|------|--------|---------|
| `src/thermal_equilibrium_model.py` | Modify | Equilibrium formula, gradients, trajectory |
| `src/model_wrapper.py` | Modify | Binary search, learning entry point |
| `src/adaptive_fireplace_learning.py` | Wrap | FP session tracking → FireplaceChannel |
| `src/heat_source_channels.py` | **NEW** | 3 channel implementations + orchestrator |
| `src/config.py` | Modify | Per-channel config vars + bounds |
| `src/main.py` | Minor | prediction_context wiring (already has needed data) |
| `src/multi_heat_source_physics.py` | Unchanged | Dashboard only, not in control loop |
| `ml_heating_data/unified_thermal_state.json` | Extend | Per-channel state |
| `tests/unit/test_heat_source_channels.py` | **NEW** | Channel unit tests |
| `tests/unit/test_learning_isolation.py` | **NEW** | Cross-contamination regression |
| `tests/unit/test_solar_transition.py` | **NEW** | Evening scenario tests |
| `CHANGELOG.md` | Update | Document new feature |

---

## Verification Plan

1. **Unit tests per channel**: heat estimate, decay after off, parameter bounds
2. **Learning isolation**: FP ON → HP params do NOT change; only FP channel learns
3. **Solar transition**: outlet increases proactively before sunset; indoor ±0.2°C
4. **Contamination regression**: 100 simulated cycles with FP ON/OFF; OE/HLC drift < 2%
5. **Backward compatibility**: `ENABLE_HEAT_SOURCE_CHANNELS=False` → identical to current
6. **Existing tests**: all 50 passing tests still pass
7. **Manual validation**: deploy, check HA sensors, verify parameter_history shows no drift

---

## CHANGELOG Entry

```markdown
## [Unreleased]

### Added
- Heat source channel architecture: 3 independent learning models for heat pump, solar, and fireplace
- Learning guards: fireplace-active and high-PV dampening prevent HP parameter contamination
- Solar transition forecasting: proactive outlet temp increase before sunset
- Per-channel prediction history and learning metrics
- New tests: test_heat_source_channels, test_learning_isolation, test_solar_transition

### Fixed
- Gradient descent contamination: fireplace heat no longer misattributed to outlet effectiveness
- HLC/OE parameter drift during fireplace sessions eliminated
```

---

## Decisions & Recommendations

1. **Deploy Phase 1 first** — stops parameter corruption immediately, minimal code change
2. **Monitor Phase 1 for 2 weeks** — if fireplace is infrequent, guard alone may suffice
3. **Phase 2-4 optional** — full decomposition is architecturally cleaner but adds complexity
4. **TV stays simple** — ~0.5 kW, additive term in HP channel, not worth own channel
5. **Config flag** `ENABLE_HEAT_SOURCE_CHANNELS=False` default for safe rollout
6. **Parallel architecture**: new orchestrator wraps existing model (no refactor risk)
7. **Evening solar transition already partially works** via pv_forecast — Phase 1 guard lets pv_weight converge correctly over daytime cycles
