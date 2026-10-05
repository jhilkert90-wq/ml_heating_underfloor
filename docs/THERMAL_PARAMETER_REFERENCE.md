# Thermal Parameter Reference

> A comprehensive guide to every calibrated thermal parameter in the ML Heating system.
> Each parameter is explained in depth with physical meaning, formulas, real-world examples,
> and practical guidance on what happens when values change.

## Overview

The ML Heating system uses **13 core thermal parameters** that together describe how your house gains and loses heat. These parameters are stored in `calibrated_params.json` and managed centrally by `ThermalParameterConfig` in `src/thermal_config.py`.

```json
{
    "thermal_time_constant": 4.390554703745845,
    "equilibrium_ratio": 0.17,
    "total_conductance": 0.8,
    "heat_loss_coefficient": 0.1245214561975565,
    "outlet_effectiveness": 0.9526723072021629,
    "solar_lag_minutes": 45.0,
    "slab_time_constant_hours": 3.19,
    "pv_heat_weight": 0.0020704649305198215,
    "fireplace_heat_weight": 0.387,
    "tv_heat_weight": 0.35,
    "delta_t_floor": 2.3,
    "fp_decay_time_constant": 3.9144707244638868,
    "room_spread_delay_minutes": 18.0
}
```

### How Parameters Work Together

The system models indoor temperature as an **energy balance** problem. At every moment, your house gains heat from the heat pump, solar radiation, the fireplace, and electronics — while simultaneously losing heat to the outdoor environment. The equilibrium temperature is the point where gains and losses balance:

```
T_equilibrium = (U_outlet × T_outlet + U_loss × T_outdoor + Q_external)
                / (U_outlet + U_loss)
```

Where:
- `U_outlet` = `outlet_effectiveness` in kW/K
- `U_loss` = `heat_loss_coefficient` in kW/K
- `Q_external` = PV, fireplace, and TV heat contribution in kW

This is the model's temperature-based approximation; the trajectory also
models slab dynamics. Where measured heat-pump thermal power is supplied, the
energy-based equilibrium path uses `T_outdoor + (Q_thermal + Q_external) /
U_loss` instead.

The time constants (`thermal_time_constant`, `slab_time_constant_hours`)
control how quickly the house approaches equilibrium, and the lag parameters
(`solar_lag_minutes`, `room_spread_delay_minutes`) model transport delays.

### Calibration & Learning

Parameters are established through a two-stage process:

1. **Initial Calibration** (`--calibrate-physics`): Fits parameters to historical data from InfluxDB using scipy optimization over stable periods
2. **Continuous Online Learning**: Gradient-based updates after every heating cycle, adjusting parameters based on prediction vs actual temperature

Each parameter has bounds enforced by `ThermalParameterConfig.BOUNDS` to prevent physically impossible values.

---

## Parameter Deep-Dive

---

### 1. `thermal_time_constant`

| Property | Value |
|----------|-------|
| **Default (heating)** | 4.39 hours |
| **Default (cooling)** | 3.0 hours |
| **Bounds** | 3.0 – 100.0 hours |
| **Unit** | hours |
| **Source files** | `thermal_config.py`, `thermal_equilibrium_model.py`, `physics_calibration.py` |

#### What It Means

The thermal time constant (τ) represents how long it takes your house to respond to heating changes. Specifically, it is the time required for the indoor temperature to reach approximately 63% of the way from its current value to the new equilibrium after a step change in heating.

Think of it like filling a bathtub — τ measures how slowly the tub fills. A large house with thick concrete walls and heavy furniture (large thermal mass) has a big τ; a lightweight wooden cabin responds quickly and has a small τ.

#### Physical Interpretation

This is a **first-order RC time constant** from thermal circuit theory:

```
τ = R × C
```

Where:
- R = thermal resistance of the building envelope (how well insulated)
- C = thermal capacitance (mass of walls, floors, furniture storing heat)

The indoor temperature approaches equilibrium exponentially:

```
T(t) = T_eq + (T_initial - T_eq) × exp(-t / τ)
```

#### Example

Suppose your indoor temperature is 19°C and the equilibrium (where the heat pump will eventually bring it) is 22°C, with τ = 4.39 hours:

| Time elapsed | Indoor temperature | % of way to equilibrium |
|-------------|-------------------|-------------------------|
| 0 hours | 19.0°C | 0% |
| 2 hours | 19.9°C | 37% |
| 4.39 hours (1τ) | 20.9°C | 63% |
| 8.78 hours (2τ) | 21.6°C | 86% |
| 13.17 hours (3τ) | 21.9°C | 95% |

#### What Happens If It's Bigger?

**Larger τ (e.g., 8+ hours)**:
- The model expects the house to respond **very slowly** to heating changes
- Outlet temperature changes have a delayed, gradual effect on predictions
- Appropriate for: heavy masonry buildings, thick concrete slab floors, large open-plan homes
- Risk if too large: model becomes sluggish, slow to detect actual temperature changes

#### What Happens If It's Smaller?

**Smaller τ (e.g., 2 hours)**:
- The model expects **rapid** indoor temperature response
- Quick heating = quick predictions
- Appropriate for: lightweight construction, small rooms, radiant ceiling panels
- Risk if too small: model over-reacts to small fluctuations, causes outlet temperature oscillation

#### How It's Learned

- **Initial calibration**: Fitted from transient heating curves via `calibrate_transient_periods()` in `physics_calibration.py`
- **Online learning**: Gradient descent with ε = 1.0 hour perturbation; clamped to ±0.2 hours per cycle
- **Convergence**: Typically stabilizes within 50–100 learning cycles

---

### 2. `equilibrium_ratio`

| Property | Value |
|----------|-------|
| **Default (heating)** | 0.17 |
| **Default (cooling)** | 0.20 |
| **Bounds** | 0.1 – 0.9 |
| **Unit** | dimensionless |
| **Source files** | `thermal_config.py`, `unified_thermal_state.py` |

#### What It Means

The equilibrium ratio was originally intended to represent the proportion of indoor-to-outdoor thermal coupling at steady state. A value of 0.17 means that at equilibrium, the indoor temperature is positioned 17% of the way between the heat pump output and outdoor temperature.

#### Current Status: Legacy Parameter

> ⚠️ **This parameter is defined and persisted but not actively used in the current prediction formulas.** It has been superseded by the more physically meaningful `heat_loss_coefficient` and `outlet_effectiveness` parameters that together determine the energy balance.

The parameter is retained in the state file for backward compatibility and is exported in the learning sensor attributes. It is **not calibrated** and does not affect model predictions.

#### Original Physical Interpretation

```
T_indoor = T_outdoor + equilibrium_ratio × (T_outlet - T_outdoor) + external_heat
```

A higher value would mean stronger coupling to the heat pump (more heating effect), while a lower value means the house equilibrium sits closer to outdoor temperature.

#### What Happens If It Changes?

Since the parameter is not used in current formulas, changing it has **no effect on model behavior**. It may be removed in a future refactoring.

---

### 3. `total_conductance`

| Property | Value |
|----------|-------|
| **Default (heating)** | 0.8 1/hour |
| **Default (cooling)** | 0.6 1/hour |
| **Bounds** | 0.1 – 0.8 1/hour |
| **Unit** | 1/hour |
| **Source files** | `thermal_config.py`, `thermal_equilibrium_model.py` |

#### What It Means

#### Current Status: Legacy Parameter

The persisted `total_conductance` value is not used by current prediction or
curve calculations. In the temperature-based equilibrium calculation, a
local denominator with the same name is computed as
`heat_loss_coefficient + effective_outlet_effectiveness`; that derived value
has units kW/K and is distinct from this legacy stored parameter. The current
energy-based balance path divides thermal power by `heat_loss_coefficient`.

---

### 4. `heat_loss_coefficient`

| Property | Value |
|----------|-------|
| **Default (heating)** | 0.1245 kW/K |
| **Default (cooling)** | 0.12 kW/K |
| **Bounds** | 0.01 – 1.2 kW/K (heating); 0.01 – 1.0 kW/K (cooling) |
| **Unit** | kW/K |
| **Source files** | `thermal_config.py`, `thermal_equilibrium_model.py`, `physics_calibration.py`, `heat_source_channels.py` |

#### What It Means

The heat loss coefficient (U_loss) represents the rate at which your building loses heat to the outdoor environment. It captures everything: wall insulation, window quality, air leakage, roof insulation, and ground losses — all rolled into a single number.

In building physics terms, this is analogous to the **overall UA-value** of your building normalized per degree of temperature difference per hour.

#### Physical Interpretation

At steady state with no heat sources except the outdoor environment:

```
Heat loss rate = U_loss × (T_indoor - T_outdoor)
```

If `U_loss = 0.1245 kW/K` and the indoor-outdoor difference is 20 K:
```
Heat loss rate = 0.1245 × 20 = 2.49 kW
```

The corresponding indoor temperature drop depends on the building's effective
thermal mass; the heat-loss coefficient alone does not determine a °C/hour rate.

#### Estimated Outdoor Balance Temperature

The building-curve sensor estimates the outdoor temperature at which no heating
or cooling is needed to hold the indoor target. It estimates average net
non-HVAC gains from idle observations with near-zero hydronic flow and thermal
power, and little indoor-temperature drift:

```
estimated gains (kW) = heat_loss_coefficient × (indoor temperature - outdoor temperature)
balance outdoor temperature = indoor target - average estimated gains / heat_loss_coefficient
```

The estimate is mode-specific, uses up to 24 hours of qualifying observations,
and is withheld until at least three samples span two hours and their inferred
indoor-outdoor temperature deltas have no more than 2°C standard deviation.
Solar, occupant, appliance, and fireplace effects are represented only insofar
as they appear in these measurements; learned source weights are not treated as
measured kW.
Published attributes show the acceptance gates: ≤0.25°C indoor drift per hour,
≤0.1 kW HVAC power, ≤0.1 flow, 0–5 kW inferred gains, and ≤2°C standard
deviation. The estimate assumes stored slab heat is not materially changing
during the observation window; room-temperature stability alone cannot verify
that assumption.
When no sufficiently stable estimate is available, `*_balance_outdoor_temp`
is `null`; `*_no_gains_zero_load_outdoor_temp` is the simpler target-based
crossing and does not include internal or solar gains.

#### Example: Comparing Houses

| House type | Typical U_loss | Description |
|-----------|---------------|-------------|
| Passive house | 0.02 – 0.05 | Exceptional insulation, minimal losses |
| Well-insulated modern | 0.05 – 0.15 | Good insulation, double/triple glazing |
| Average house (default) | 0.10 – 0.20 | Standard insulation, some air leaks |
| Poorly insulated | 0.20 – 0.50 | Old building, single glazing, drafts |
| Very poorly insulated | 0.50 – 1.00 | Minimal insulation, significant air leakage |

The default value of **0.1245** suggests a well-insulated modern house.

#### What Happens If It's Bigger?

**Larger U_loss** (e.g., 0.30):
- Model predicts **more heat loss to outdoors** — indoor temperature drops faster without heating
- Heat pump needs to work harder (higher outlet temperature) to maintain target
- Outdoor temperature swings have a **larger effect** on indoor comfort
- In the equilibrium formula, the outdoor temperature gets more weight:
  ```
  More U_loss → equilibrium pulled toward outdoor temperature
  ```

**Practical consequence**: The model requests higher outlet temperatures, especially on cold days. If the value is too high, it will overshoot by running the heat pump too hard.

#### What Happens If It's Smaller?

**Smaller U_loss** (e.g., 0.03):
- Model predicts excellent insulation — **minimal outdoor influence**
- Heat pump can use lower outlet temperatures
- The model may under-predict heat losses on cold, windy days
- Risk: if too small, the house will be under-heated when outdoor temperature drops sharply

**Practical consequence**: The model underestimates the heat pump requirement, leading to slow temperature recovery after cold snaps.

#### How It's Learned

- **Initial calibration**: Energy balance over stable periods: `U_loss = P_thermal / (T_eq - T_outdoor)`, then refined by scipy optimization
- **Online learning**: Finite-difference gradient with ε = 0.01; clamped to ±0.01 per cycle
- **Channel isolation**: Only learns during clean HP-only cycles (no fireplace/PV active)

---

### 5. `outlet_effectiveness`

| Property | Value |
|----------|-------|
| **Default (heating)** | 0.83 kW/K |
| **Default (cooling)** | 0.20 kW/K |
| **Bounds** | 0.3 – 2.0 kW/K (heating); 0.05 – 2.0 kW/K (cooling) |
| **Unit** | kW/K |
| **Source files** | `thermal_config.py`, `thermal_equilibrium_model.py`, `physics_calibration.py`, `heat_source_channels.py` |

#### What It Means

Outlet effectiveness (`U_outlet`) is the learned thermal conductance between the effective floor-loop temperature and the room. It converts a temperature difference into a heat rate; it is not a dimensionless percentage.

#### Physical Interpretation

The heat pump contributes heating proportional to the temperature difference between outlet water and indoor air:

```
Q_heat_pump (kW) = U_outlet (kW/K) × (T_outlet - T_indoor) (K)
```

This captures:
- Floor surface area and thermal conductivity (Estrich/screed)
- Water flow rate through the UFH loops
- Heat transfer coefficient between slab surface and room air
- Piping layout density and distribution uniformity

#### Example

With effective floor temperature at 30°C, indoor at 22°C, and
`U_outlet = 0.953 kW/K`:
```
Q = 0.953 × (30 - 22) = 7.62 kW
```

If U_outlet were only 0.5:
```
Q = 0.5 × (30 - 22) = 4.0 kW
```

The lower effectiveness means you'd need a higher outlet temperature (or longer run time) to achieve the same heating.

#### What Happens If It's Bigger?

**Larger U_outlet** (e.g., 1.5):
- Each degree of outlet-indoor difference delivers **more heating**
- The model predicts it can reach target temperature with **lower outlet temperatures**
- Excellent for efficient operation — less energy consumption
- If too high: model under-predicts required outlet, leading to cold rooms

**Practical consequence**: Lower outlet temperatures requested → more efficient heat pump operation (higher COP). But if overestimated, the house may not reach the target temperature.

#### What Happens If It's Smaller?

**Smaller U_outlet** (e.g., 0.4):
- Heat transfer is poor — higher outlet temperatures needed for same effect
- Model requests aggressive outlet temperatures
- Can indicate: poor flow rate, sludged pipes, insufficient floor area, wrong pipe spacing
- If too low: model over-heats by requesting unnecessarily high outlet temperatures

**Practical consequence**: Higher outlet temperatures → lower COP → higher energy bills. Fixing the actual heat transfer issue (purge air, adjust flow, etc.) is better than accepting a low learned value.

#### How It's Learned

- **Initial calibration**: Fitted alongside `heat_loss_coefficient` via scipy optimization, minimizing prediction error over stable periods
- **Online learning**: Gradient descent with ε = 0.005; clamped to ±0.005 per cycle
- **Channel isolation**: Learned only by the Heat Pump channel during clean cycles

---

### 6. `solar_lag_minutes`

| Property | Value |
|----------|-------|
| **Default** | 45.0 minutes |
| **Bounds** | 0.0 – 180.0 minutes |
| **Unit** | minutes |
| **Source files** | `thermal_config.py`, `thermal_equilibrium_model.py`, `heat_source_channels.py` |

#### What It Means

Solar lag represents the **time delay** between solar radiation arriving at your house (measured via PV power output) and the resulting indoor temperature increase. When the sun shines on your house, it takes time for solar energy to pass through windows, warm surfaces, and eventually heat the air your temperature sensor measures.

#### Physical Interpretation

The delay is caused by:
1. **Window-to-surface transfer** (~5–10 min): Sunlight hits floors/walls through windows
2. **Surface-to-air transfer** (~15–20 min): Heated surfaces warm the surrounding air via convection
3. **Air mixing** (~10–15 min): Warm air circulates through rooms to reach the temperature sensor
4. **Sensor response** (~5 min): Temperature sensor thermal lag

Total: roughly 35–50 minutes for a typical house, which aligns with the 45-minute default.

#### Example

It's noon, the sun comes out, and PV production jumps from 0 to 5,000W:

| Time | PV Power | Indoor temp effect | What's happening |
|------|----------|-------------------|-----------------|
| 12:00 | 5,000W | None yet | Sunlight entering through windows |
| 12:15 | 5,000W | +0.05°C | Floors/walls absorbing radiation |
| 12:30 | 5,000W | +0.15°C | Surfaces warming air |
| 12:45 | 5,000W | +0.30°C | Full effect reaching sensor |
| 13:00 | 5,000W | +0.40°C | Steady-state solar gain established |

#### What Happens If It's Bigger?

**Larger lag** (e.g., 120 minutes):
- Model delays the solar contribution by 2 hours
- Appropriate for: heavy thermal mass buildings, indirect sun exposure, north-facing windows
- Risk if too large: model ignores solar gains that are already warming the house, causing overheating by not reducing outlet temperature soon enough

#### What Happens If It's Smaller?

**Smaller lag** (e.g., 10 minutes):
- Model applies solar gains nearly immediately
- Appropriate for: lightweight buildings, large south-facing windows, direct sun on sensor area
- Risk if too small: model reacts to PV fluctuations (clouds passing) too aggressively, causing outlet temperature oscillation

#### How It's Learned

- **Starting point**: 45 minutes (default)
- **Online learning**: Solar channel gradient during daytime; `solar_lag_minutes_delta = -avg_error × (rising_pv / threshold)`
- **Constraints**: ±5 minutes per cycle; total range 0–180 minutes

---

### 7. `slab_time_constant_hours`

| Property | Value |
|----------|-------|
| **Default (heating)** | 3.19 hours |
| **Default (cooling)** | 0.8 hours |
| **Bounds (heating)** | 0.5 – 6.0 hours |
| **Bounds (cooling)** | 0.3 – 2.5 hours |
| **Unit** | hours |
| **Source files** | `thermal_config.py`, `thermal_equilibrium_model.py`, `heat_source_channels.py` |

#### What It Means

The slab time constant describes the thermal inertia of your underfloor heating (UFH) screed/slab. After the heat pump turns off, the slab continues radiating stored heat into the room. This parameter controls how long that "residual heating" lasts.

The default of 3.19 hours was **data-fitted from a 180 m² floor** with forced-convection water circuits through standard Estrich (screed).

#### Physical Interpretation

After the heat pump shuts off, the slab temperature decays exponentially:

```
T_slab(t) = T_room + (T_slab_initial - T_room) × exp(-t / τ_slab)
```

Where τ_slab = 3.19 hours means 63% of stored heat is released in the first ~3 hours.

The heat contribution from this cooling slab is:

```python
# From heat_source_channels.py
alpha = 1 - exp(-time_since_off / slab_time_constant)
T_slab = T_inlet + alpha * (T_indoor - T_inlet)
Q_decay (kW) = outlet_effectiveness (kW/K) * max(0, T_slab - T_indoor) (K)
```

#### Example: Heat Pump Turns Off at 14:00

Initial slab temperature: 28°C, room temperature: 22°C, τ_slab = 3.19 hours:

| Time | Hours since off | Slab temp | Residual heating |
|------|----------------|-----------|-----------------|
| 14:00 | 0 | 28.0°C | 5.72 kW |
| 15:00 | 1.0 | 26.2°C | 4.00 kW |
| 17:00 | 3.0 | 23.4°C | 1.33 kW |
| 17:12 | 3.19 (1τ) | 23.2°C | 1.14 kW (37% remaining) |
| 20:24 | 6.38 (2τ) | 22.4°C | 0.38 kW (14% remaining) |

#### What Happens If It's Bigger?

**Larger τ_slab** (e.g., 5 hours):
- Slab retains heat **much longer** after pump stops
- Longer "free" residual heating — potentially more efficient
- Model knows it can turn off the pump earlier before reaching target
- Appropriate for: thick screed (>7 cm), high-density concrete, heated thermal mass walls
- Risk if too large: model underestimates cooling rate, turns off pump too early → temperature undershoots

#### What Happens If It's Smaller?

**Smaller τ_slab** (e.g., 1 hour):
- Slab cools quickly — residual heating ends fast
- Model knows it needs to keep the pump running longer
- Appropriate for: thin screed, dry construction, small pipe spacing, low-mass floors
- Risk if too small: model runs the pump longer than necessary → wasted energy

**Cooling mode** uses 0.8 hours because cold water through a warm slab exchanges heat faster than hot water through a cool slab.

#### How It's Learned

- **Initial calibration**: Data-fitted from cold-start step-response analysis
- **Online learning**: Finite-difference gradient with ε = 0.1 hours; clamped to ±0.05 hours per cycle
- **Active learning gate**: Only learns when `inlet_temp != outlet_cmd` (active heating/cooling phase)

---

### 8. `pv_heat_weight`

| Property | Value |
|----------|-------|
| **Default (heating)** | 0.002 kW/W |
| **Default (cooling)** | 0.002 kW/W |
| **Bounds** | 0.00001 – 0.005 kW/W |
| **Unit** | kW per PV watt |
| **Source files** | `thermal_config.py`, `thermal_equilibrium_model.py`, `heat_source_channels.py` |

#### What It Means

PV heat weight maps electrical PV output (W) to an estimated solar thermal
contribution (kW). PV output is a proxy for solar irradiance; it is not a
direct measurement of heat entering through windows.

A value of `0.002 kW/W` maps 1,000 W of PV output to 2 kW of estimated heat
before cloud correction and saturation. The resulting equilibrium-temperature
effect also depends on total conductance.

#### Physical Interpretation

The heat contribution from solar is:

```python
# From heat_source_channels.py
Q_pv = PV_power × pv_heat_weight × cloud_factor
```

This is added to the equilibrium calculation as external thermal power in kW.
The cloud factor damps the contribution on cloudy days.

#### Example

On a sunny afternoon with 6,000W PV production and cloud_factor = 0.9:

```
Q_pv = 6000 × 0.002 × 0.9 = 10.8 kW (before saturation)
```

This is the model's estimated solar heat input; its equilibrium effect is
`Q_pv / (U_outlet + U_loss)`, not a direct 10.8°C temperature rise.

On a cloudy day with only 1,000W PV and cloud_factor = 0.4:

```
Q_pv = 1000 × 0.002 × 0.4 = 0.8 kW
```

Much less significant — the heat pump carries most of the load.

#### What Happens If It's Bigger?

**Larger weight** (e.g., 0.004):
- Model predicts **strong solar heating** — aggressively reduces outlet temperature during sun
- Appropriate for: large south-facing windows, sun room, ground-floor conservatory
- Risk if too large: model over-reduces outlet temperature on sunny days → rooms get cold when clouds arrive

#### What Happens If It's Smaller?

**Smaller weight** (e.g., 0.0005):
- Model predicts **minimal solar effect** — largely ignores PV for heating decisions
- Appropriate for: north-facing rooms, small windows, shaded house
- Risk if too small: model ignores real solar gains → overheating on sunny days

**Cooling mode** uses the mode's learned weight because solar heat works
*against* cooling — it is modeled as a load the cooling system must overcome.

#### How It's Learned

- **Initial calibration**: Estimated from energy balance or default
- **Online learning**: Solar channel gradient during daytime; **zero-PV guard** suppresses learning at night
- **Constraints**: ±0.0002 kW/W per cycle
- **Special protection**: No learning when PV = 0W (no signal to learn from)

---

### 9. `fireplace_heat_weight`

| Property | Value |
|----------|-------|
| **Default (heating)** | 0.387 kW |
| **Default (cooling)** | 1.0 kW |
| **Bounds** | 0.01 – 6.0 kW |
| **Unit** | kW |
| **Source files** | `thermal_config.py`, `thermal_equilibrium_model.py`, `heat_source_channels.py` |

#### What It Means

In the temperature-based model, fireplace heat weight represents an estimated
thermal power contribution while the fireplace is on. The heat-source channel
uses the separately bounded `fp_heat_output_kw` parameter for its output.

A value of `0.387 kW` means the model adds 0.387 kW while the fireplace is on;
the resulting indoor-temperature effect depends on total conductance.

#### Physical Interpretation

```python
# From thermal_equilibrium_model.py
heat_from_fireplace (kW) = fireplace_on × fireplace_heat_weight (kW)
                           + fireplace_decay_kw (kW)
```

The total fireplace contribution has two components:
1. **Active heating** (this parameter): Estimated heat while burning
2. **Residual decay** (`fp_decay_time_constant`): Heat lingering after shutdown

#### Example: Evening Fireplace Session

Fireplace lit at 18:00, burns until 22:00:

| Time | Fireplace | Active contribution | Decay contribution | Total |
|------|-----------|--------------------|--------------------|-------|
| 18:00 | ON | +0.39 kW | 0 kW | +0.39 kW |
| 19:00 | ON | +0.39 kW | 0 kW | +0.39 kW |
| 22:00 | OFF | 0 kW | +0.39 kW (exp decay starts) | +0.39 kW |
| 23:00 | OFF | 0 kW | +0.30 kW | +0.30 kW |
| 01:00 | OFF | 0 kW | +0.17 kW | +0.17 kW |
| 04:00 | OFF | 0 kW | +0.05 kW | +0.05 kW |

#### What Happens If It's Bigger?

**Larger weight** (e.g., 2.0):
- Model predicts strong fireplace heating — significantly reduces outlet temperature when fireplace is on
- Appropriate for: large wood-burning stove in small room, open fireplace with convection fan
- Risk if too large: model over-reduces heat pump output → rooms may get cold after fireplace goes out

#### What Happens If It's Smaller?

**Smaller weight** (e.g., 0.05):
- Model treats fireplace as negligible — barely adjusts outlet temperature
- Appropriate for: decorative fireplace, small flame, fireplace far from temperature sensor
- Risk if too small: model ignores real fireplace heating → overshoots during fire evenings

#### How It's Learned

- **Limited automatic learning**: Binary input (on/off) lacks gradient signal for continuous optimization
- **Adaptive fireplace learning module** (`adaptive_fireplace_learning.py`) can refine the value when heat source channel mode is disabled
- **Best practice**: Set based on observation during commissioning — light the fireplace, note temperature rise after 1–2 hours

---

### 10. `tv_heat_weight`

| Property | Value |
|----------|-------|
| **Default** | 0.35 kW |
| **Bounds** | 0.05 – 1.5 kW |
| **Unit** | kW |
| **Source files** | `thermal_config.py`, `thermal_equilibrium_model.py`, `heat_source_channels.py` |

#### What It Means

TV heat weight represents the estimated thermal power from electronics and
appliances when they are detected as active. It is a binary (on/off) source
term, not a direct temperature rise.

A value of `0.35 kW` means the model adds 350 W of estimated heat while the
tracked TV/electronics entity is active.

#### Physical Interpretation

```python
# From heat_source_channels.py
Q_tv (kW) = tv_on × tv_heat_weight (kW)
```

This captures heat from:
- Television sets (plasma/LCD, 100–300W typical)
- AV receivers, gaming consoles
- Computers, monitors
- Lighting in the vicinity
- Any appliance tracked by the binary entity

#### Example

A 65" OLED TV and associated electronics tracked as a source:
```
tv_on × 0.35 kW = 0.35 kW estimated thermal contribution
```

This is a modest but measurable effect. In a well-insulated house where the heat pump maintains ±0.2°C accuracy, a 0.35°C un-modeled disturbance would be significant.

#### What Happens If It's Bigger?

**Larger weight** (e.g., 1.0):
- Model predicts significant internal gains from electronics
- Would reduce outlet temperature when TV is on
- Appropriate for: home theater room with multiple high-power devices, server room
- Risk if too large: outlet drops too much when TV is on → rooms get cold

#### What Happens If It's Smaller?

**Smaller weight** (e.g., 0.05):
- Electronics heat is treated as negligible
- No meaningful outlet adjustment when TV turns on/off
- Appropriate for: energy-efficient devices, large room volume, TV far from sensor
- Risk if too small: model ignores real internal gains → slight overheating when electronics are on

#### How It's Learned

- **Online learning**: TV channel gradient descent during TV-on periods
- **Constraints**: ±0.05 °C per cycle (conservative to avoid over-fitting)
- **Minor source**: Learning rate intentionally slow — TV heat is a small correction

---

### 11. `delta_t_floor`

| Property | Value |
|----------|-------|
| **Default (heating)** | 2.3 °C |
| **Default (cooling)** | 2.0 °C |
| **Bounds** | 0.0 – 10.0 °C |
| **Unit** | °C |
| **Source files** | `thermal_config.py`, `thermal_equilibrium_model.py`, `heat_source_channels.py` |

#### What It Means

Delta-T floor is the **minimum temperature difference** between the heat pump outlet and return (inlet) that indicates the heat pump is actively heating. Below this threshold, the temperature spread is too small for meaningful heat transfer through the underfloor loops.

The default of 2.3°C means: if `T_outlet - T_inlet < 2.3°C`, the system considers the heat pump effectively off (no useful heating).

#### Physical Interpretation

In a UFH system, heat transfer is driven by the temperature difference across the slab:

```
Q_heat = flow_rate × specific_heat × (T_outlet - T_inlet)
```

When the heat pump is off but the circulator still runs (or residual heat exists), the delta-T may be very small (< 1°C). The `delta_t_floor` provides a threshold:

- **Above delta_t_floor**: Heat pump is actively heating → use normal slab model
- **Below delta_t_floor**: Heat pump is off or passive → use passive slab model

#### Example: Binary Search HP-Off Fix

When the heat pump is off (measured delta-T = 0.5°C < 2.3°C), the binary search substitutes the `delta_t_floor`:

```python
# From model_wrapper.py
if _dtf < 1.0:
    _dtf = self.thermal_model._resolve_delta_t_floor(_dtf)  # → ~2.3°C
```

Without this substitution, all outlet temperature candidates produce identical passive-slab predictions, making the binary search unable to find a useful temperature → it defaults to the maximum 35°C (a spike).

#### What Happens If It's Bigger?

**Larger delta_t_floor** (e.g., 5.0°C):
- System requires a **larger outlet-inlet spread** before considering the HP as "on"
- More conservative — only counts strong heating as active
- The simulated HP-on mode in binary search uses a larger assumed delta-T
- May lead to higher outlet temperature requests (model thinks it needs more spread)
- Appropriate for: systems with high flow rates where small delta-T is normal

#### What Happens If It's Smaller?

**Smaller delta_t_floor** (e.g., 0.5°C):
- Even tiny temperature differences count as "active heating"
- May false-trigger the active slab model when the HP is actually off
- Appropriate for: low-flow systems where even small delta-T represents real heating
- Risk: passive heat and circulation artifacts trigger false HP-on detection

#### How It's Learned

- **Primarily a commissioning parameter**: Based on UFH system physical design
- **Adaptive learning available**: ±0.2°C per cycle (heat pump channel)
- **Rarely changes**: Determined by heat pump and piping characteristics, not building thermal mass

---

### 12. `fp_decay_time_constant`

| Property | Value |
|----------|-------|
| **Default (heating)** | 3.91 hours |
| **Default (cooling)** | 2.0 hours |
| **Bounds** | 0.1 – 5.0 hours |
| **Unit** | hours |
| **Source files** | `thermal_config.py`, `thermal_equilibrium_model.py`, `heat_source_channels.py` |

#### What It Means

The fireplace decay time constant controls how long **residual heat** persists after the fireplace is turned off. When a fireplace goes out, the hot masonry, flue, and heated surrounding surfaces continue radiating heat for hours. This parameter models that exponential cooling process.

#### Physical Interpretation

After the fireplace shuts off at time t₀:

```python
# From heat_source_channels.py
Q_decay(t) = Q_peak × exp(-(t - t₀) / fp_decay_time_constant)
```

With τ = 3.91 hours, the residual heat decays to:
- **63% released** after 3.91 hours (1τ)
- **86% released** after 7.82 hours (2τ)
- **95% released** after 11.73 hours (3τ)

#### Example: Fireplace Evening Decay

Fireplace off at 22:00 with full heat output:

| Time | Hours since off | Remaining heat | Room effect |
|------|----------------|---------------|-------------|
| 22:00 | 0 | 100% | +0.39°C |
| 23:00 | 1.0 | 77.4% | +0.30°C |
| 00:00 | 2.0 | 59.9% | +0.23°C |
| 02:00 | 4.0 (≈1τ) | 35.9% | +0.14°C |
| 06:00 | 8.0 (≈2τ) | 12.9% | +0.05°C |
| 10:00 | 12.0 (≈3τ) | 4.6% | +0.02°C |

This matches real-world observation: after a fireplace evening, you can still feel warmth the next morning from the residual heat in masonry and surrounding walls.

#### What Happens If It's Bigger?

**Larger τ_fp** (e.g., 5 hours):
- Model predicts fireplace heat lingers **much longer** after shutdown
- Heat pump outlet stays lower longer after fireplace evening
- Appropriate for: massive stone fireplace, closed stove with water jacket, tiled masonry oven (Kachelofen)
- Risk if too large: model under-heats the next morning, thinking fireplace residual is still warming

#### What Happens If It's Smaller?

**Smaller τ_fp** (e.g., 0.5 hours):
- Model predicts fireplace heat disappears quickly after shutdown
- Heat pump ramps up sooner after fireplace goes out
- Appropriate for: gas fireplace, small decorative fire, open fireplace (heat escapes up chimney)
- Risk if too small: model over-heats right after fireplace goes out (doesn't account for lingering warmth)

#### How It's Learned

- **Online learning**: Fireplace channel gradient; ±0.1 hours per cycle
- **Learning gate**: Only active during fireplace-on or recent-off periods
- **Gradient direction**: Positive prediction error (model under-predicts) → increase τ (heat decays slower than expected)

---

### 13. `room_spread_delay_minutes`

| Property | Value |
|----------|-------|
| **Default (heating)** | 18.0 minutes |
| **Default (cooling)** | 30.0 minutes |
| **Bounds** | 0.0 – 180.0 minutes |
| **Unit** | minutes |
| **Source files** | `thermal_config.py`, `thermal_equilibrium_model.py`, `heat_source_channels.py` |

#### What It Means

Room spread delay represents the **transport lag** for heat from a localized source (like a fireplace in the living room) to spread throughout the house and reach the temperature sensor. The fireplace heats its immediate room first; the average house temperature (measured elsewhere) responds with a delay.

The default of 18 minutes suggests a ~200–300 m² building with moderate air circulation.

#### Physical Interpretation

Heat from the living room spreads through:
1. **Natural convection** — warm air rises and flows through doorways
2. **Forced circulation** — HVAC fans, open ventilation paths
3. **Conduction through walls** — slow but continuous heat transfer between rooms

The delay creates a phase lag:

```
T_house_average(t) ≈ T_living_room(t - room_spread_delay)
```

#### Example: Fireplace Lights at 18:00

With 18-minute delay:

| Time | Living room | House average (sensor) | Delay effect |
|------|------------|----------------------|-------------|
| 18:00 | Fireplace lit, +1°C | No change yet | 18 min lag |
| 18:10 | +1.5°C | +0.3°C starting | Warmth spreading |
| 18:18 | +2.0°C | +1.0°C | Full effect arrives |
| 18:30 | +2.2°C | +1.5°C | Synchronized |

Without modeling this delay, the system would see the fireplace turn on but no immediate temperature change — it might wrongly conclude the fireplace has minimal effect and reduce the heat weight.

#### What Happens If It's Bigger?

**Larger delay** (e.g., 60 minutes):
- Model expects a **long time** before fireplace heat reaches the sensor
- Appropriate for: large house, fireplace far from sensor, closed doors between rooms
- Risk if too large: model ignores immediate fireplace heating at the sensor → poor predictions during fire evenings

#### What Happens If It's Smaller?

**Smaller delay** (e.g., 5 minutes):
- Model expects near-immediate house-wide heating from the fireplace
- Appropriate for: open-plan layout, sensor near fireplace, small apartment
- Risk if too small: model reacts to fireplace before heat actually spreads → premature outlet reduction

**Cooling mode** uses 30 minutes because passive convection (without heating-driven air movement) is slower.

#### How It's Learned

- **Primarily a commissioning parameter**: Set based on building layout
- **Not actively adapted** in the current code
- **Best practice**: Time how long it takes for the house average temperature to respond after lighting the fireplace

---

## Parameter Interaction Summary

### How Parameters Affect Each Other

The parameters don't work in isolation — they form an interconnected system:

```
                    ┌────────────────────────────┐
                    │     Equilibrium Formula     │
                    │                              │
  outlet_effectiveness ──► U_outlet ──┐            │
                                      ├─► U_total ─┤
  heat_loss_coefficient ──► U_loss ───┘            │
                                                    │
  pv_heat_weight ──────────► Q_pv ──┐              │
  fireplace_heat_weight ──► Q_fp ───┼─► Q_external │
  tv_heat_weight ──────────► Q_tv ──┘              │
                                                    │
                    │  T_eq = (U_outlet×T_out +    │
                    │          U_loss×T_outdoor +   │
                    │          Q_external)           │
                    │         / U_total              │
                    └────────────────────────────────┘
                                  │
                    ┌─────────────▼──────────────────┐
                    │    Time Response                │
                    │                                  │
                    │  thermal_time_constant ──► τ     │
                    │  slab_time_constant ──► τ_slab   │
                    │  solar_lag_minutes ──► lag_pv     │
                    │  room_spread_delay ──► lag_fp     │
                    │  fp_decay_time_constant ──► τ_fp  │
                    │                                  │
                    │  T(t) → T_eq exponentially       │
                    │  with time constants and lags    │
                    └──────────────────────────────────┘
```

### Key Relationships

| If this increases... | Then this happens... |
|---------------------|---------------------|
| `outlet_effectiveness` ↑ | Lower outlet temp needed → more efficient |
| `heat_loss_coefficient` ↑ | Higher outlet temp needed → more energy |
| `pv_heat_weight` ↑ | Lower outlet during sun → savings |
| `thermal_time_constant` ↑ | Slower response → needs earlier action |
| `slab_time_constant_hours` ↑ | More residual heating → pump can stop earlier |
| `fireplace_heat_weight` ↑ | Lower outlet during fire → avoid overshoot |

### Cooling Mode Differences

In cooling mode, the physics reverses: the heat pump **removes** heat, and external sources (solar, fireplace) work **against** cooling:

| Parameter | Heating mode | Cooling mode | Why different |
|-----------|-------------|-------------|---------------|
| `thermal_time_constant` | 4.39 h | 3.0 h | Cooling response faster |
| `slab_time_constant_hours` | 3.19 h | 0.8 h | Cold water ↔ warm slab exchanges faster |
| `outlet_effectiveness` | 0.953 | 0.90 | Cooling slightly less effective |
| `pv_heat_weight` | 0.00207 | 0.0003 | Solar adds load in cooling |
| `delta_t_floor` | 2.3°C | 2.0°C | Narrower operating band |
| `outlet_temp range` | 0–35°C | 18–24°C | Much narrower for cooling |

---

## Practical Tuning Guide

### When to Manually Adjust Parameters

The adaptive learning system should converge to correct values over 50–200 cycles (1–4 days). However, if you notice persistent issues:

| Symptom | Likely parameter | Adjustment |
|---------|-----------------|-----------|
| House heats too slowly | `thermal_time_constant` too large | Reduce toward 3.0 |
| House overheats in sun | `pv_heat_weight` too small | Increase toward 0.003 |
| Outlet always too high | `outlet_effectiveness` too low | Increase toward 1.0 |
| Outlet always too low | `heat_loss_coefficient` too low | Increase toward 0.2 |
| Temperature drops fast after HP off | `slab_time_constant_hours` too small | Increase toward 4.0 |
| Overshoot during fireplace | `fireplace_heat_weight` too small | Increase toward 1.0 |

### Resetting to Defaults

To reset all parameters to defaults, delete `unified_thermal_state.json` (or `unified_thermal_state_cooling.json` for cooling) and restart. The system will re-initialize with defaults from `thermal_config.py` and begin learning again.

### Running Initial Calibration

For the best starting point, run initial calibration on historical data:

```bash
python3 -m src.main --calibrate-physics
```

This requires at least `TRAINING_LOOKBACK_HOURS` (default: 168 hours / 7 days) of historical data in InfluxDB and will optimize `heat_loss_coefficient` and `outlet_effectiveness` over stable periods, then fit `thermal_time_constant` from transient curves.
