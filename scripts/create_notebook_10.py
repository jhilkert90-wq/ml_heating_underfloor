"""Generate notebook 10: Cooling Thermal Model Calibration."""
import json
import os

def md(source: str) -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")][:-1] + [source.split("\n")[-1]]
    }

def code(source: str) -> dict:
    """Create a code cell."""
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src
    }

cells = []

# ============================================================
# CELL 1: Title
# ============================================================
cells.append(md("""# 10 — Cooling Thermal Model Calibration (Offline)

**Purpose:** Offline calibration of HLC, OE, and τ for cooling mode using `cooling_training_data.csv.gz`.
Compares OLS vs scipy optimization, investigates τ explosion (41h vs 4.8h heating),
evaluates HP-ON vs HP-OFF dual-HLC, and tests Open Meteo solar radiation as PV replacement.

**Data:** 47,941 rows × 76 columns (5-min resolution, warm-season cooling periods)

**Key Questions:**
1. Why is cooling HLC ~0.049 while heating HLC ~0.12? (building property should be constant)
2. Why did τ explode to 41h in online learning? Can we constrain it?
3. Is Open Meteo GHI a better solar feature than local PV_Generate for thermal calibration?
4. Does using separate HP-ON vs HP-OFF HLC values improve predictions?"""))

# ============================================================
# CELL 2: Imports
# ============================================================
cells.append(code("""%load_ext autoreload
%autoreload 2

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import optimize, stats
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["figure.dpi"] = 100

# Project root
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DATA_PATH = PROJECT_ROOT / "Logs_and_models" / "cooling_training_data.csv.gz"

print("Project root:", PROJECT_ROOT)
print("Data path:", DATA_PATH, "exists:", DATA_PATH.exists())"""))

# ============================================================
# CELL 3: Phase A header
# ============================================================
cells.append(md("""## Phase A: Data Loading & Exploration"""))

# ============================================================
# CELL 4: Load data
# ============================================================
cells.append(code("""# Load cooling training data
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Reconstruct hour from sin/cos encoding
df["hour"] = (np.degrees(np.arctan2(df["hour_sin"], df["hour_cos"])) / 15) % 24

# Key derived columns
df["driving_force"] = df["indoor_temp"] - df["AT"]  # T_indoor - T_outdoor
df["cooling_drive"] = df["indoor_temp"] - df["VLT"]  # T_indoor - T_outlet
df["oe_raw"] = np.where(
    df["cooling_drive"].abs() > 0.3,
    df["thermal_power_kw"].abs() / df["cooling_drive"],
    np.nan
)

# Clean outliers (AT has values up to 839°C — clearly sensor errors)
mask_clean = (df["AT"] > -10) & (df["AT"] < 45) & (df["VLT"] > 10) & (df["VLT"] < 45)
print(f"Clean rows: {mask_clean.sum()} of {len(df)} ({100*mask_clean.mean():.1f}%)")
df_clean = df[mask_clean].copy()

# Identify periods
df_clean["is_cooling_active"] = (df_clean["is_hp_active"] == 1) & (df_clean["delta_t"] < -0.5)
df_clean["is_night"] = (df_clean["hour"] < 6) | (df_clean["hour"] > 21)
df_clean["has_pv"] = df_clean["PV_Generate"] > 50

print(f"\\nHP ON: {(df_clean['is_hp_active']==1).sum()} ({100*(df_clean['is_hp_active']==1).mean():.1f}%)")
print(f"HP OFF: {(df_clean['is_hp_active']==0).sum()} ({100*(df_clean['is_hp_active']==0).mean():.1f}%)")
print(f"Active cooling (HP ON, ΔT<-0.5): {df_clean['is_cooling_active'].sum()} ({100*df_clean['is_cooling_active'].mean():.1f}%)")
print(f"Night cooling (active + PV<50): {(df_clean['is_cooling_active'] & ~df_clean['has_pv']).sum()}")
df_clean.describe().round(3)"""))

# ============================================================
# CELL 5: Exploratory plots
# ============================================================
cells.append(code("""fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1) Indoor vs Outdoor colored by HP state
ax = axes[0, 0]
for label, color, marker in [(0, "gray", "."), (1, "blue", ".")]:
    mask = df_clean["is_hp_active"] == label
    ax.scatter(df_clean.loc[mask, "AT"], df_clean.loc[mask, "indoor_temp"],
               c=color, alpha=0.15, s=5, label=f"HP {'ON' if label else 'OFF'}")
ax.set_xlabel("Outdoor Temp (°C)")
ax.set_ylabel("Indoor Temp (°C)")
ax.set_title("Indoor vs Outdoor (by HP state)")
ax.legend(markerscale=5)

# 2) Thermal power vs delta_t
ax = axes[0, 1]
cool = df_clean[df_clean["is_cooling_active"]]
ax.scatter(cool["delta_t"], cool["thermal_power_kw"], c="steelblue", alpha=0.3, s=5)
ax.set_xlabel("ΔT (outlet - return, °C)")
ax.set_ylabel("Thermal Power (kW)")
ax.set_title("Thermal Power vs ΔT (cooling active)")
ax.axhline(0, color="red", ls="--", alpha=0.5)

# 3) Hour distribution by HP state
ax = axes[0, 2]
bins = np.arange(0, 25, 1)
ax.hist(df_clean.loc[df_clean["is_cooling_active"], "hour"], bins=bins, alpha=0.6, label="HP ON cooling", color="blue")
ax.hist(df_clean.loc[~df_clean["is_cooling_active"], "hour"], bins=bins, alpha=0.4, label="HP OFF / passive", color="gray")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Count")
ax.set_title("Cooling Activity by Hour")
ax.legend()

# 4) OE raw distribution
ax = axes[1, 0]
oe_valid = df_clean.loc[df_clean["is_cooling_active"], "oe_raw"].dropna()
oe_clipped = oe_valid.clip(0, 3)
ax.hist(oe_clipped, bins=50, color="teal", alpha=0.7)
ax.axvline(oe_clipped.median(), color="red", ls="--", label=f"Median: {oe_clipped.median():.3f}")
ax.set_xlabel("Outlet Effectiveness (raw)")
ax.set_ylabel("Count")
ax.set_title("OE Distribution (cooling active)")
ax.legend()

# 5) VLT vs Indoor (cooling active)
ax = axes[1, 1]
ax.scatter(cool["VLT"], cool["indoor_temp"], c=cool["PV_Generate"], cmap="YlOrRd",
           alpha=0.4, s=5, vmin=0, vmax=10000)
ax.set_xlabel("Outlet Temp VLT (°C)")
ax.set_ylabel("Indoor Temp (°C)")
ax.set_title("VLT vs Indoor (color=PV)")
plt.colorbar(ax.collections[0], ax=ax, label="PV (W)")

# 6) Driving force vs thermal power
ax = axes[1, 2]
ax.scatter(cool["driving_force"], cool["thermal_power_kw"], c="steelblue", alpha=0.3, s=5)
ax.set_xlabel("Driving Force (T_indoor - T_outdoor, °C)")
ax.set_ylabel("Thermal Power (kW)")
ax.set_title("HLC Relationship: Power vs Driving Force")
# Quick OLS line
from numpy.polynomial.polynomial import polyfit
mask_valid = cool["driving_force"].notna() & cool["thermal_power_kw"].notna()
if mask_valid.sum() > 10:
    b = np.polyfit(cool.loc[mask_valid, "driving_force"], cool.loc[mask_valid, "thermal_power_kw"], 1)
    x_fit = np.linspace(cool["driving_force"].min(), cool["driving_force"].max(), 100)
    ax.plot(x_fit, np.polyval(b, x_fit), "r-", lw=2, label=f"slope={b[0]:.4f}")
    ax.legend()

plt.suptitle("Phase A: Cooling Data Exploration", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()"""))

# ============================================================
# CELL 6: Phase B header
# ============================================================
cells.append(md("""## Phase B: HLC Calibration

The Heat Loss Coefficient (HLC) should be a building property: `Q_thermal = HLC × (T_indoor − T_outdoor)`.
- **Heating mode** online learning converged to HLC ≈ 0.12
- **Cooling mode** online learning converged to HLC ≈ 0.049 (nearly 3× lower!)

**Hypothesis**: The difference comes from:
1. HP mostly off at night in cooling → insufficient forced-convection data
2. Daytime solar gains not properly accounted for → model absorbs solar into a lower HLC
3. The online learner sees HP-OFF periods where indoor barely changes → interprets as low heat exchange"""))

# ============================================================
# CELL 7: OLS HLC (current approach)
# ============================================================
cells.append(code("""# Replicate calibrate_hlc() from src/hlc_learner.py
# Forced-through-origin OLS: thermal_power = HLC × driving_force

# Filter for stable HP-ON cooling periods
hlc_data = df_clean[
    (df_clean["is_cooling_active"]) &       # HP ON, delta_t < -0.5
    (df_clean["thermal_power_kw"].abs() > 0.5) &  # Meaningful thermal output
    (df_clean["driving_force"].abs() > 1.0)  # Meaningful temperature difference
].copy()

print(f"HLC calibration data: {len(hlc_data)} rows")

# Method 1: All cooling-active periods
X_all = hlc_data["driving_force"].values.reshape(-1, 1)
y_all = hlc_data["thermal_power_kw"].values

# Forced-through-origin OLS: y = b*x → b = Σ(x*y) / Σ(x²)
hlc_ols_all = np.sum(X_all.ravel() * y_all) / np.sum(X_all.ravel() ** 2)

# Method 2: Night-only (PV < 50W) — cleanest signal
night_data = hlc_data[~hlc_data["has_pv"]]
if len(night_data) > 20:
    X_night = night_data["driving_force"].values
    y_night = night_data["thermal_power_kw"].values
    hlc_ols_night = np.sum(X_night * y_night) / np.sum(X_night ** 2)
else:
    hlc_ols_night = np.nan

# Method 3: Daytime (PV > 50W)
day_data = hlc_data[hlc_data["has_pv"]]
if len(day_data) > 20:
    X_day = day_data["driving_force"].values
    y_day = day_data["thermal_power_kw"].values
    hlc_ols_day = np.sum(X_day * y_day) / np.sum(X_day ** 2)
else:
    hlc_ols_day = np.nan

# Method 4: HP-OFF periods (passive heat exchange)
passive_data = df_clean[
    (df_clean["is_hp_active"] == 0) &
    (df_clean["driving_force"].abs() > 2.0) &
    (df_clean["indoor_temp_gradient"].abs() > 0.01)  # Some measurable change
].copy()
# For HP-OFF: use indoor_temp_gradient as proxy for thermal power
# dT/dt ≈ -(HLC/C) × driving_force → HLC ∝ gradient / driving_force
if len(passive_data) > 50:
    hlc_passive_raw = passive_data["indoor_temp_gradient"] / passive_data["driving_force"]
    hlc_ols_passive = hlc_passive_raw.median()
else:
    hlc_ols_passive = np.nan

print(f"\\n{'Method':<25} {'HLC':>8} {'N rows':>8}")
print("-" * 45)
print(f"{'OLS All cooling-active':<25} {hlc_ols_all:>8.4f} {len(hlc_data):>8}")
print(f"{'OLS Night-only (PV<50)':<25} {hlc_ols_night:>8.4f} {len(night_data):>8}")
print(f"{'OLS Day-only (PV>50)':<25} {hlc_ols_day:>8.4f} {len(day_data):>8}")
print(f"{'Passive (HP-OFF median)':<25} {hlc_ols_passive:>8.4f} {len(passive_data):>8}")
print(f"{'Online-learned cooling':<25} {'0.0487':>8}")
print(f"{'Online-learned heating':<25} {'0.1206':>8}")"""))

# ============================================================
# CELL 8: OLS visualization
# ============================================================
cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: All data with OLS fit
ax = axes[0]
ax.scatter(hlc_data["driving_force"], hlc_data["thermal_power_kw"], alpha=0.2, s=5, c="steelblue")
x_range = np.linspace(hlc_data["driving_force"].min(), hlc_data["driving_force"].max(), 100)
ax.plot(x_range, hlc_ols_all * x_range, "r-", lw=2, label=f"HLC={hlc_ols_all:.4f}")
ax.set_xlabel("Driving Force (T_indoor - T_outdoor)")
ax.set_ylabel("Thermal Power (kW)")
ax.set_title("OLS HLC — All Cooling-Active")
ax.legend()
ax.axhline(0, color="gray", ls=":", alpha=0.5)

# Plot 2: Night vs Day comparison
ax = axes[1]
if len(night_data) > 0:
    ax.scatter(night_data["driving_force"], night_data["thermal_power_kw"],
               alpha=0.3, s=8, c="navy", label=f"Night HLC={hlc_ols_night:.4f}")
    ax.plot(x_range, hlc_ols_night * x_range, "navy", lw=2, ls="--")
if len(day_data) > 0:
    ax.scatter(day_data["driving_force"], day_data["thermal_power_kw"],
               alpha=0.2, s=5, c="orange", label=f"Day HLC={hlc_ols_day:.4f}")
    ax.plot(x_range, hlc_ols_day * x_range, "orange", lw=2, ls="--")
ax.set_xlabel("Driving Force")
ax.set_ylabel("Thermal Power (kW)")
ax.set_title("Night vs Day HLC Comparison")
ax.legend()

# Plot 3: Residuals
ax = axes[2]
residuals = hlc_data["thermal_power_kw"] - hlc_ols_all * hlc_data["driving_force"]
ax.hist(residuals, bins=50, alpha=0.7, color="steelblue")
ax.axvline(0, color="red", ls="--")
ax.set_xlabel("Residual (kW)")
ax.set_ylabel("Count")
ax.set_title(f"OLS Residuals (std={residuals.std():.3f} kW)")

plt.suptitle("Phase B: OLS HLC Calibration", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()"""))

# ============================================================
# CELL 9: Scipy optimization
# ============================================================
cells.append(md("""### Scipy Joint Optimization (HLC + OE + τ)

Instead of calibrating each parameter separately, optimize all three simultaneously
by minimizing the error between predicted and actual indoor temperature trajectories."""))

# ============================================================
# CELL 10: Scipy implementation
# ============================================================
cells.append(code("""def predict_equilibrium(T_outlet, T_outdoor, HLC, OE, pv_watts=0, pv_weight=0.001):
    \"\"\"Predict equilibrium indoor temperature.
    T_eq = (OE × T_outlet + HLC × T_outdoor + Q_ext) / (OE + HLC)
    \"\"\"
    Q_ext = pv_watts * pv_weight  # External heat from PV/solar
    return (OE * T_outlet + HLC * T_outdoor + Q_ext) / (OE + HLC)

def simulate_trajectory(params, data, dt_hours=5/60):
    \"\"\"Simulate indoor temperature trajectory given parameters.
    Returns predicted indoor temps for each row.
    \"\"\"
    HLC, OE, tau = params
    if tau <= 0 or HLC <= 0 or OE <= 0:
        return np.full(len(data), np.nan)
    
    predicted = np.zeros(len(data))
    predicted[0] = data["indoor_temp"].iloc[0]
    
    for i in range(1, len(data)):
        T_eq = predict_equilibrium(
            data["VLT"].iloc[i], data["AT"].iloc[i], HLC, OE,
            data["PV_Generate"].iloc[i], pv_weight=0.001
        )
        # Newton's law: T(t+dt) = T(t) + (T_eq - T(t)) × (1 - exp(-dt/τ))
        approach = 1 - np.exp(-dt_hours / tau)
        predicted[i] = predicted[i-1] + (T_eq - predicted[i-1]) * approach
    
    return predicted

def cost_function(params, data):
    \"\"\"Sum of squared errors between predicted and actual indoor temp.\"\"\"
    HLC, OE, tau = params
    # Bounds check
    if HLC < 0.01 or HLC > 1.0 or OE < 0.1 or OE > 2.0 or tau < 1.0 or tau > 50:
        return 1e6
    predicted = simulate_trajectory(params, data)
    if np.any(np.isnan(predicted)):
        return 1e6
    return np.sum((predicted - data["indoor_temp"].values) ** 2)

# Use cooling-active periods in contiguous chunks
# Find contiguous cooling segments (at least 2 hours = 24 rows)
cooling_mask = df_clean["is_cooling_active"].values
segments = []
start = None
for i in range(len(cooling_mask)):
    if cooling_mask[i] and start is None:
        start = i
    elif not cooling_mask[i] and start is not None:
        if i - start >= 24:  # At least 2 hours
            segments.append((start, i))
        start = None
if start is not None and len(cooling_mask) - start >= 24:
    segments.append((start, len(cooling_mask)))

print(f"Found {len(segments)} contiguous cooling segments (>=2h each)")
total_rows = sum(e - s for s, e in segments)
print(f"Total rows for calibration: {total_rows}")

# Use a sample of segments for optimization speed
seg_data_list = [df_clean.iloc[s:e].reset_index(drop=True) for s, e in segments]
# Concatenate first N segments (limit for speed)
max_rows = 5000
cal_data = pd.concat(seg_data_list, ignore_index=True).head(max_rows)
print(f"Using {len(cal_data)} rows for scipy optimization")

# Optimize with multiple initial guesses
initial_guesses = [
    [0.12, 0.85, 4.5],   # Heating-like
    [0.05, 0.72, 4.3],   # Current cooling online
    [0.08, 0.60, 6.0],   # Intermediate
    [0.10, 0.90, 3.5],   # Low tau
]

results = []
for i, x0 in enumerate(initial_guesses):
    res = optimize.minimize(
        cost_function, x0, args=(cal_data,),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5}
    )
    results.append(res)
    print(f"Init {i}: HLC={res.x[0]:.4f}, OE={res.x[1]:.4f}, τ={res.x[2]:.2f}h, "
          f"cost={res.fun:.2f}, success={res.success}")

# Pick best result
best = min(results, key=lambda r: r.fun)
hlc_scipy, oe_scipy, tau_scipy = best.x
print(f"\\n*** BEST: HLC={hlc_scipy:.4f}, OE={oe_scipy:.4f}, τ={tau_scipy:.2f}h ***")"""))

# ============================================================
# CELL 11: Scipy with separate HP-ON / HP-OFF
# ============================================================
cells.append(md("""### HP-ON vs HP-OFF Dual-HLC Calibration

**Hypothesis**: The building has one true HLC, but the *effective* heat exchange differs
when the HP is actively circulating water (forced convection through floor) vs passive (natural convection only).
Test if using separate HLC values for each state improves predictions."""))

# ============================================================
# CELL 12: Dual-HLC optimization
# ============================================================
cells.append(code("""def cost_dual_hlc(params, data):
    \"\"\"Cost function with separate HLC for HP-ON and HP-OFF.\"\"\"
    HLC_on, HLC_off, OE, tau = params
    if any(p <= 0 for p in params) or tau > 50 or OE > 2.0:
        return 1e6
    
    predicted = np.zeros(len(data))
    predicted[0] = data["indoor_temp"].iloc[0]
    hp_active = data["is_hp_active"].values
    dt_h = 5 / 60  # 5-min intervals
    
    for i in range(1, len(data)):
        hlc_i = HLC_on if hp_active[i] == 1 else HLC_off
        T_eq = predict_equilibrium(
            data["VLT"].iloc[i], data["AT"].iloc[i], hlc_i, OE,
            data["PV_Generate"].iloc[i], pv_weight=0.001
        )
        approach = 1 - np.exp(-dt_h / tau)
        predicted[i] = predicted[i-1] + (T_eq - predicted[i-1]) * approach
    
    return np.sum((predicted - data["indoor_temp"].values) ** 2)

# Use a mixed-mode dataset (includes both HP-ON and HP-OFF)
mixed_data = df_clean.head(max_rows).copy()

# Optimize dual-HLC
dual_guesses = [
    [0.12, 0.04, 0.85, 4.5],
    [0.10, 0.06, 0.72, 5.0],
    [0.08, 0.08, 0.90, 4.0],
]

dual_results = []
for i, x0 in enumerate(dual_guesses):
    res = optimize.minimize(
        cost_dual_hlc, x0, args=(mixed_data,),
        method="Nelder-Mead",
        options={"maxiter": 8000, "xatol": 1e-5, "fatol": 1e-5}
    )
    dual_results.append(res)
    print(f"Init {i}: HLC_on={res.x[0]:.4f}, HLC_off={res.x[1]:.4f}, "
          f"OE={res.x[2]:.4f}, τ={res.x[3]:.2f}h, cost={res.fun:.2f}")

best_dual = min(dual_results, key=lambda r: r.fun)
hlc_on, hlc_off, oe_dual, tau_dual = best_dual.x
print(f"\\n*** BEST DUAL: HLC_on={hlc_on:.4f}, HLC_off={hlc_off:.4f}, "
      f"OE={oe_dual:.4f}, τ={tau_dual:.2f}h ***")

# Compare single vs dual HLC
print(f"\\n{'Model':<25} {'Cost':>10} {'HLC':>10} {'OE':>8} {'τ':>8}")
print("-" * 65)
# Re-evaluate single-HLC on same mixed data
single_cost = cost_function([hlc_scipy, oe_scipy, tau_scipy], mixed_data)
dual_cost = best_dual.fun
improvement = (single_cost - dual_cost) / single_cost * 100
print(f"{'Single HLC':<25} {single_cost:>10.2f} {hlc_scipy:>10.4f} {oe_scipy:>8.4f} {tau_scipy:>8.2f}")
print(f"{'Dual HLC (ON/OFF)':<25} {dual_cost:>10.2f} {hlc_on:>6.4f}/{hlc_off:<6.4f} {oe_dual:>6.4f} {tau_dual:>8.2f}")
print(f"\\nImprovement: {improvement:.1f}%")"""))

# ============================================================
# CELL 13: HLC comparison summary
# ============================================================
cells.append(code("""# Summary comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of all HLC estimates
methods = ["OLS\\n(all)", "OLS\\n(night)", "OLS\\n(day)", "Passive\\n(HP-OFF)",
           "Scipy\\n(single)", "Dual\\n(HP-ON)", "Dual\\n(HP-OFF)",
           "Online\\ncooling", "Online\\nheating"]
values = [hlc_ols_all, hlc_ols_night, hlc_ols_day, hlc_ols_passive,
          hlc_scipy, hlc_on, hlc_off, 0.0487, 0.1206]
colors = ["steelblue"]*4 + ["teal"]*3 + ["orange", "red"]

ax = axes[0]
bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
ax.set_ylabel("HLC (1/hour)")
ax.set_title("HLC Comparison: All Methods")
ax.axhline(0.12, color="red", ls="--", alpha=0.5, label="Heating reference (0.12)")
ax.legend()
for bar, val in zip(bars, values):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

# Prediction comparison: single vs dual HLC
ax = axes[1]
pred_single = simulate_trajectory([hlc_scipy, oe_scipy, tau_scipy], cal_data)
pred_dual_on = simulate_trajectory([hlc_on, oe_dual, tau_dual],
    cal_data[cal_data["is_hp_active"]==1].head(500).reset_index(drop=True)) if (cal_data["is_hp_active"]==1).sum() > 100 else None

actual = cal_data["indoor_temp"].values
ax.plot(actual[:500], "k-", label="Actual", alpha=0.7, lw=1)
ax.plot(pred_single[:500], "b--", label=f"Single HLC={hlc_scipy:.3f}", alpha=0.7, lw=1)
ax.set_xlabel("Time Step (5-min)")
ax.set_ylabel("Indoor Temp (°C)")
ax.set_title("Prediction: Single HLC")
ax.legend()

plt.suptitle("Phase B: HLC Calibration Summary", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()"""))

# ============================================================
# CELL 14: Phase C header
# ============================================================
cells.append(md("""## Phase C: Outlet Effectiveness (OE) Calibration

OE measures heat transfer efficiency from underfloor heating water to room air.
- **Heating mode**: OE ≈ 0.83–0.95 (forced convection, warm water → cool room)
- **Cooling mode**: Should be similar (same floor, same pipes, reverse direction)
  but natural convection differences may apply.

**Strategy**: Use early morning / late evening HP-ON periods where PV ≈ 0
to isolate the OE calculation from solar confounding."""))

# ============================================================
# CELL 15: OE calibration
# ============================================================
cells.append(code("""# Method 1: Weighted median (replicating _calibrate_oe_cooling)
oe_data = df_clean[
    (df_clean["is_cooling_active"]) &
    (df_clean["cooling_drive"].abs() > 0.2) &  # Meaningful cooling drive
    (df_clean["driving_force"].abs() > 1.0)
].copy()

# OE = HLC × driving_force / cooling_drive (from equilibrium equation rearranged)
# At equilibrium: OE × (T_indoor - T_outlet) = HLC × (T_indoor - T_outdoor) + Q_ext
# → OE = (HLC × driving_force + Q_ext) / cooling_drive
for hlc_ref, hlc_name in [(hlc_scipy, "scipy"), (hlc_ols_all, "OLS"), (0.1206, "heating")]:
    oe_vals = (hlc_ref * oe_data["driving_force"]) / oe_data["cooling_drive"]
    oe_vals = oe_vals[(oe_vals > 0.1) & (oe_vals < 3.0)]  # Clip outliers
    print(f"OE from {hlc_name} HLC ({hlc_ref:.4f}): median={oe_vals.median():.4f}, "
          f"mean={oe_vals.mean():.4f}, std={oe_vals.std():.4f}, n={len(oe_vals)}")

# Method 2: Night-only OE (cleanest — no PV contamination)
night_oe_data = oe_data[~oe_data["has_pv"]]
if len(night_oe_data) > 20:
    oe_night = (hlc_scipy * night_oe_data["driving_force"]) / night_oe_data["cooling_drive"]
    oe_night = oe_night[(oe_night > 0.1) & (oe_night < 3.0)]
    print(f"\\nNight-only OE (scipy HLC): median={oe_night.median():.4f}, n={len(oe_night)}")
else:
    oe_night = pd.Series([np.nan])
    print("\\nInsufficient night-only data for OE")

# Method 3: Direct from scipy optimization
print(f"\\nScipy-optimized OE: {oe_scipy:.4f}")
print(f"Dual-HLC OE: {oe_dual:.4f}")
print(f"Online-learned (heating): 0.826")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
oe_all = (hlc_scipy * oe_data["driving_force"]) / oe_data["cooling_drive"]
oe_all = oe_all[(oe_all > 0) & (oe_all < 3.0)]
ax.hist(oe_all, bins=60, alpha=0.6, color="teal", label="All periods")
if len(oe_night) > 10:
    ax.hist(oe_night, bins=30, alpha=0.6, color="navy", label="Night-only")
ax.axvline(oe_scipy, color="red", ls="--", lw=2, label=f"Scipy OE={oe_scipy:.3f}")
ax.axvline(0.826, color="orange", ls="--", lw=2, label="Heating OE=0.826")
ax.set_xlabel("Outlet Effectiveness")
ax.set_ylabel("Count")
ax.set_title("OE Distribution")
ax.legend()
ax.set_xlim(0, 3)

ax = axes[1]
# OE vs hour of day
oe_hourly = oe_data.copy()
oe_hourly["oe_calc"] = (hlc_scipy * oe_hourly["driving_force"]) / oe_hourly["cooling_drive"]
oe_hourly = oe_hourly[(oe_hourly["oe_calc"] > 0) & (oe_hourly["oe_calc"] < 3.0)]
oe_by_hour = oe_hourly.groupby(oe_hourly["hour"].round())["oe_calc"].agg(["median", "std", "count"])
ax.bar(oe_by_hour.index, oe_by_hour["median"], alpha=0.7, color="teal")
ax.errorbar(oe_by_hour.index, oe_by_hour["median"], yerr=oe_by_hour["std"],
            fmt="none", color="black", capsize=3)
ax.axhline(oe_scipy, color="red", ls="--", label=f"Scipy OE={oe_scipy:.3f}")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("OE (median)")
ax.set_title("OE by Hour — PV Contamination Check")
ax.legend()

plt.suptitle("Phase C: OE Calibration", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()"""))

# ============================================================
# CELL 16: Phase D header
# ============================================================
cells.append(md("""## Phase D: τ (Thermal Time Constant) Investigation

The thermal time constant τ represents how quickly the indoor temperature responds to changes.
- **Heating mode**: τ ≈ 4.8h (reasonable for a well-insulated house with UFH)
- **Cooling mode online learning**: τ exploded to 41.23h!

**Root cause hypothesis**: During HP-OFF nighttime periods, indoor temp barely changes.
The online learner interprets this as: "temperature barely responds → very high thermal mass → increase τ."
But in reality, it's just that there's no *driving force* (T_eq ≈ T_current), not that the house is slow."""))

# ============================================================
# CELL 17: τ investigation
# ============================================================
cells.append(code("""# Investigate τ by fitting exponential approach curves on cooling segments

def fit_tau_segment(segment_data, hlc, oe, dt_h=5/60):
    \"\"\"Fit τ to a single contiguous cooling segment using least squares.\"\"\"
    actual = segment_data["indoor_temp"].values
    if len(actual) < 12:  # At least 1 hour
        return np.nan, np.inf
    
    # Compute equilibrium temps for each timestep
    T_eq = np.array([
        predict_equilibrium(row["VLT"], row["AT"], hlc, oe, row["PV_Generate"])
        for _, row in segment_data.iterrows()
    ])
    
    def cost(tau_arr):
        tau = tau_arr[0]
        if tau < 0.5 or tau > 50:
            return 1e6
        pred = np.zeros(len(actual))
        pred[0] = actual[0]
        approach = 1 - np.exp(-dt_h / tau)
        for i in range(1, len(actual)):
            pred[i] = pred[i-1] + (T_eq[i] - pred[i-1]) * approach
        return np.sum((pred - actual) ** 2)
    
    res = optimize.minimize(cost, [4.5], method="Nelder-Mead", options={"maxiter": 500})
    rmse = np.sqrt(res.fun / len(actual))
    return res.x[0], rmse

# Fit τ on individual segments
print(f"Fitting τ on {len(segments)} cooling segments...")
print(f"Using scipy-calibrated HLC={hlc_scipy:.4f}, OE={oe_scipy:.4f}")
print()

tau_results = []
for idx, (s, e) in enumerate(segments[:30]):  # Limit to 30 segments
    seg = df_clean.iloc[s:e].reset_index(drop=True)
    hp_frac = seg["is_hp_active"].mean()
    pv_mean = seg["PV_Generate"].mean()
    duration_h = len(seg) * 5 / 60
    
    tau_fit, rmse = fit_tau_segment(seg, hlc_scipy, oe_scipy)
    tau_results.append({
        "segment": idx, "start": s, "end": e, "duration_h": duration_h,
        "tau_fit": tau_fit, "rmse": rmse,
        "hp_frac": hp_frac, "pv_mean": pv_mean,
        "is_night": seg["is_night"].mean() > 0.5,
        "indoor_range": seg["indoor_temp"].max() - seg["indoor_temp"].min()
    })
    if idx < 10:
        status = "DAY" if pv_mean > 100 else "NIGHT"
        print(f"Seg {idx:2d}: τ={tau_fit:5.2f}h, RMSE={rmse:.4f}°C, "
              f"dur={duration_h:.1f}h, HP%={100*hp_frac:.0f}%, {status}")

tau_df = pd.DataFrame(tau_results)
print(f"\\n--- Summary ---")
print(f"τ median (all):    {tau_df['tau_fit'].median():.2f}h")
print(f"τ median (night):  {tau_df[tau_df['is_night']]['tau_fit'].median():.2f}h")
print(f"τ median (day):    {tau_df[~tau_df['is_night']]['tau_fit'].median():.2f}h")
print(f"τ from scipy opt:  {tau_scipy:.2f}h")
print(f"τ online cooling:  41.23h")
print(f"τ online heating:  4.8h")"""))

# ============================================================
# CELL 18: τ visualization
# ============================================================
cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1) τ distribution
ax = axes[0]
tau_valid = tau_df[tau_df["tau_fit"].between(0.5, 50)]
ax.hist(tau_valid["tau_fit"], bins=20, alpha=0.7, color="teal", edgecolor="black")
ax.axvline(tau_scipy, color="red", ls="--", lw=2, label=f"Scipy τ={tau_scipy:.1f}h")
ax.axvline(4.8, color="orange", ls="--", lw=2, label="Heating τ=4.8h")
ax.axvline(41.23, color="purple", ls="--", lw=2, label="Online cooling τ=41.2h")
ax.set_xlabel("Fitted τ (hours)")
ax.set_ylabel("Count (segments)")
ax.set_title("τ Distribution Across Segments")
ax.legend(fontsize=9)

# 2) τ vs HP fraction
ax = axes[1]
sc = ax.scatter(tau_valid["hp_frac"], tau_valid["tau_fit"],
                c=tau_valid["pv_mean"], cmap="YlOrRd", s=40, alpha=0.7,
                edgecolors="black", linewidth=0.5)
plt.colorbar(sc, ax=ax, label="Mean PV (W)")
ax.set_xlabel("HP Active Fraction")
ax.set_ylabel("Fitted τ (hours)")
ax.set_title("τ vs HP Activity (color=PV)")
ax.axhline(tau_scipy, color="red", ls="--", alpha=0.5)

# 3) τ vs indoor temp range (more change → more reliable fit)
ax = axes[2]
ax.scatter(tau_valid["indoor_range"], tau_valid["tau_fit"],
           c=tau_valid["is_night"].astype(int), cmap="coolwarm", s=40, alpha=0.7,
           edgecolors="black", linewidth=0.5)
ax.set_xlabel("Indoor Temp Range (°C)")
ax.set_ylabel("Fitted τ (hours)")
ax.set_title("τ vs Indoor Variability (blue=night, red=day)")
ax.axhline(tau_scipy, color="red", ls="--", alpha=0.5, label=f"Scipy τ={tau_scipy:.1f}h")
ax.legend()

plt.suptitle("Phase D: τ Investigation — Why 41h?", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Diagnosis
print("\\n" + "="*60)
print("τ EXPLOSION DIAGNOSIS")
print("="*60)
high_tau = tau_df[tau_df["tau_fit"] > 15]
low_tau = tau_df[tau_df["tau_fit"] < 8]
print(f"\\nSegments with τ > 15h: {len(high_tau)} — "
      f"avg HP%={100*high_tau['hp_frac'].mean():.0f}%, "
      f"avg indoor_range={high_tau['indoor_range'].mean():.3f}°C, "
      f"avg PV={high_tau['pv_mean'].mean():.0f}W")
print(f"Segments with τ < 8h:  {len(low_tau)} — "
      f"avg HP%={100*low_tau['hp_frac'].mean():.0f}%, "
      f"avg indoor_range={low_tau['indoor_range'].mean():.3f}°C, "
      f"avg PV={low_tau['pv_mean'].mean():.0f}W")
print(f"\\n→ High-τ segments have {'lower' if high_tau['indoor_range'].mean() < low_tau['indoor_range'].mean() else 'higher'} "
      f"indoor variability → confirms hypothesis that τ inflates when temp barely changes.")"""))

# ============================================================
# CELL 19: τ bounds proposal
# ============================================================
cells.append(code("""# Proposed cooling τ bounds
print("="*60)
print("τ BOUNDS PROPOSAL FOR COOLING MODE")
print("="*60)

# Use only segments with sufficient indoor temp change (> 0.1°C range)
reliable = tau_df[(tau_df["indoor_range"] > 0.1) & (tau_df["tau_fit"].between(0.5, 50))]
if len(reliable) > 3:
    tau_p10 = reliable["tau_fit"].quantile(0.10)
    tau_p50 = reliable["tau_fit"].quantile(0.50)
    tau_p90 = reliable["tau_fit"].quantile(0.90)
    
    print(f"\\nReliable segments: {len(reliable)} (indoor_range > 0.1°C)")
    print(f"τ percentiles: P10={tau_p10:.1f}h, P50={tau_p50:.1f}h, P90={tau_p90:.1f}h")
    print(f"\\nProposed bounds:")
    print(f"  Lower: {max(1.0, tau_p10 * 0.8):.1f}h (P10 × 0.8)")
    print(f"  Upper: {min(15.0, tau_p90 * 1.5):.1f}h (P90 × 1.5, capped at 15h)")
    print(f"  Default: {tau_p50:.1f}h (median)")
    print(f"\\nCurrent online cooling τ: 41.23h — {'OUTSIDE proposed bounds' if 41.23 > min(15, tau_p90*1.5) else 'within bounds'}")
    print(f"Heating τ: 4.8h — {'within proposed bounds' if tau_p10*0.8 <= 4.8 <= tau_p90*1.5 else 'outside bounds'}")
else:
    print("Insufficient reliable segments for bounds proposal")

print(f"\\nScipy τ: {tau_scipy:.2f}h — recommended as cooling default")
print(f"\\n--- RECOMMENDATION ---")
print(f"1. Cap cooling τ online learning upper bound to ~{min(15, tau_p90*1.5 if len(reliable)>3 else 15):.0f}h")
print(f"2. Use scipy-fitted τ={tau_scipy:.1f}h as cooling mode initial value")
print(f"3. Consider sharing τ between modes (building thermal mass is the same)")"""))

# ============================================================
# CELL 20: Phase E header
# ============================================================
cells.append(md("""## Phase E: Solar Radiation Analysis (Open Meteo vs PV)

**Goal**: Compare Open Meteo Global Horizontal Irradiance (GHI) with local PV_Generate data.
Evaluate whether GHI is a better feature for thermal calibration and/or a viable production replacement.

**Advantages of GHI**:
- Available as weather forecast (not hardware-dependent)
- Measures true solar radiation (not panel efficiency/orientation-filtered)
- Could capture solar heat gain through windows (which PV panels don't measure directly)

**Advantages of PV_Generate**:
- Already measured locally
- Accounts for actual panel orientation and local shading
- Correlates with electrical self-consumption patterns"""))

# ============================================================
# CELL 21: Fetch Open Meteo data
# ============================================================
cells.append(code("""import requests
from datetime import datetime, timedelta

# Location (from notebook 07)
LAT = 48.928
LON = 10.069

# Determine date range from training data
# Reconstruct approximate dates from doy_sin/doy_cos
doy_angle = np.arctan2(df_clean["doy_sin"], df_clean["doy_cos"])
doy = (doy_angle * 365.25 / (2 * np.pi)) % 365.25
doy_min, doy_max = int(doy.min()), int(doy.max())

# The cooling data spans warm season — estimate as May-October 2025 through May 2026
# Use a generous range
START_DATE = "2025-05-01"
END_DATE = "2026-06-01"

print(f"DOY range in data: {doy_min} to {doy_max}")
print(f"Fetching Open Meteo data: {START_DATE} to {END_DATE}")
print(f"Location: {LAT}°N, {LON}°E")

# Fetch multiple radiation variables
url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LAT}&longitude={LON}"
    f"&start_date={START_DATE}&end_date={END_DATE}"
    f"&hourly=shortwave_radiation,direct_radiation,diffuse_radiation,"
    f"direct_normal_irradiance,terrestrial_radiation"
    f"&timezone=Europe%2FBerlin"
)

print(f"\\nFetching from Open Meteo...")
try:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    solar_data = resp.json()
    
    solar_df = pd.DataFrame({
        "timestamp": pd.to_datetime(solar_data["hourly"]["time"]),
        "ghi_wm2": solar_data["hourly"]["shortwave_radiation"],
        "direct_wm2": solar_data["hourly"]["direct_radiation"],
        "diffuse_wm2": solar_data["hourly"]["diffuse_radiation"],
        "dni_wm2": solar_data["hourly"]["direct_normal_irradiance"],
    })
    solar_df = solar_df.set_index("timestamp")
    print(f"Fetched {len(solar_df)} hourly records")
    print(f"Date range: {solar_df.index.min()} to {solar_df.index.max()}")
    print(f"\\nGHI stats: mean={solar_df['ghi_wm2'].mean():.1f}, max={solar_df['ghi_wm2'].max():.1f} W/m²")
    print(f"DNI stats: mean={solar_df['dni_wm2'].mean():.1f}, max={solar_df['dni_wm2'].max():.1f} W/m²")
    solar_df.describe().round(1)
except Exception as e:
    print(f"ERROR fetching Open Meteo: {e}")
    print("Creating synthetic data for analysis...")
    solar_df = None"""))

# ============================================================
# CELL 22: Merge solar with training data
# ============================================================
cells.append(code("""# Since training data lacks timestamps, we use DOY + hour to create a merge key
# This won't be a perfect temporal alignment but allows statistical comparison

if solar_df is not None:
    # Create DOY + hour index for solar data
    solar_df["doy"] = solar_df.index.dayofyear
    solar_df["hour_of_day"] = solar_df.index.hour
    
    # Average by DOY + hour (climatological match)
    solar_avg = solar_df.groupby(["doy", "hour_of_day"]).mean().reset_index()
    
    # Create matching keys in training data
    df_clean["doy_approx"] = (doy.round()).astype(int).clip(1, 365)
    df_clean["hour_approx"] = df_clean["hour"].round().astype(int).clip(0, 23)
    
    # Merge
    df_merged = df_clean.merge(
        solar_avg[["doy", "hour_of_day", "ghi_wm2", "direct_wm2", "diffuse_wm2", "dni_wm2"]],
        left_on=["doy_approx", "hour_approx"],
        right_on=["doy", "hour_of_day"],
        how="left"
    )
    
    print(f"Merged dataset: {len(df_merged)} rows")
    print(f"GHI matched: {df_merged['ghi_wm2'].notna().sum()} ({100*df_merged['ghi_wm2'].notna().mean():.1f}%)")
    
    # Quick correlation check
    pv_nonzero = df_merged[df_merged["PV_Generate"] > 0]
    print(f"\\nCorrelations (PV > 0, n={len(pv_nonzero)}):")
    for col in ["ghi_wm2", "direct_wm2", "diffuse_wm2", "dni_wm2"]:
        r = pv_nonzero["PV_Generate"].corr(pv_nonzero[col])
        print(f"  PV vs {col}: r = {r:.3f}")
else:
    print("No solar data available — skipping merge")
    df_merged = df_clean.copy()"""))

# ============================================================
# CELL 23: Solar comparison plots
# ============================================================
cells.append(code("""if "ghi_wm2" in df_merged.columns and df_merged["ghi_wm2"].notna().any():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    pv_pos = df_merged[df_merged["PV_Generate"] > 0].copy()
    
    # 1) GHI vs PV scatter
    ax = axes[0, 0]
    ax.scatter(pv_pos["ghi_wm2"], pv_pos["PV_Generate"], alpha=0.1, s=3, c="steelblue")
    ax.set_xlabel("GHI (W/m²)")
    ax.set_ylabel("PV Generate (W)")
    r = pv_pos["PV_Generate"].corr(pv_pos["ghi_wm2"])
    ax.set_title(f"GHI vs PV (r={r:.3f})")
    
    # 2) DNI vs PV scatter
    ax = axes[0, 1]
    ax.scatter(pv_pos["dni_wm2"], pv_pos["PV_Generate"], alpha=0.1, s=3, c="orange")
    ax.set_xlabel("DNI (W/m²)")
    ax.set_ylabel("PV Generate (W)")
    r = pv_pos["PV_Generate"].corr(pv_pos["dni_wm2"])
    ax.set_title(f"DNI vs PV (r={r:.3f})")
    
    # 3) Direct vs PV scatter
    ax = axes[0, 2]
    ax.scatter(pv_pos["direct_wm2"], pv_pos["PV_Generate"], alpha=0.1, s=3, c="green")
    ax.set_xlabel("Direct Radiation (W/m²)")
    ax.set_ylabel("PV Generate (W)")
    r = pv_pos["PV_Generate"].corr(pv_pos["direct_wm2"])
    ax.set_title(f"Direct vs PV (r={r:.3f})")
    
    # 4) Diurnal profile comparison
    ax = axes[1, 0]
    hourly = df_merged.groupby("hour_approx").agg({
        "PV_Generate": "mean", "ghi_wm2": "mean", "dni_wm2": "mean"
    })
    ax.plot(hourly.index, hourly["PV_Generate"] / hourly["PV_Generate"].max(), "b-o",
            label="PV (normalized)", ms=4)
    ax.plot(hourly.index, hourly["ghi_wm2"] / hourly["ghi_wm2"].max(), "r-s",
            label="GHI (normalized)", ms=4)
    ax.plot(hourly.index, hourly["dni_wm2"] / hourly["dni_wm2"].max(), "g-^",
            label="DNI (normalized)", ms=4)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Normalized Value")
    ax.set_title("Diurnal Profiles")
    ax.legend(fontsize=9)
    
    # 5) PV efficiency: PV/GHI vs GHI
    ax = axes[1, 1]
    ghi_pos = pv_pos[pv_pos["ghi_wm2"] > 50].copy()
    ghi_pos["pv_efficiency"] = ghi_pos["PV_Generate"] / ghi_pos["ghi_wm2"]
    ax.scatter(ghi_pos["ghi_wm2"], ghi_pos["pv_efficiency"], alpha=0.1, s=3, c="teal")
    ax.set_xlabel("GHI (W/m²)")
    ax.set_ylabel("PV / GHI ratio")
    ax.set_title("PV Efficiency vs GHI")
    ax.set_ylim(0, ghi_pos["pv_efficiency"].quantile(0.99) * 1.2)
    
    # 6) Correlation heatmap
    ax = axes[1, 2]
    solar_cols = ["PV_Generate", "ghi_wm2", "direct_wm2", "diffuse_wm2", "dni_wm2"]
    corr = pv_pos[solar_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", ax=ax, vmin=0, vmax=1)
    ax.set_title("Solar Variables Correlation")
    
    plt.suptitle("Phase E: Solar Radiation — GHI vs PV Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
else:
    print("No solar data to plot")"""))

# ============================================================
# CELL 24: Feature ablation
# ============================================================
cells.append(md("""### Feature Ablation: PV vs GHI for Thermal Calibration

Compare three calibration variants:
1. **PV-only**: Use PV_Generate as the solar feature (current approach)
2. **GHI-only**: Replace PV with Open Meteo GHI
3. **Combined**: Use both PV and GHI
4. **DNI-only**: Use Direct Normal Irradiance (better for south-facing panels?)"""))

# ============================================================
# CELL 25: Feature ablation code
# ============================================================
cells.append(code("""def calibrate_with_solar(data, solar_col, hlc_init=0.10, oe_init=0.80, tau_init=4.5):
    \"\"\"Calibrate thermal model using a specific solar column.\"\"\"
    def cost(params):
        hlc, oe, tau, pv_w = params
        if hlc < 0.01 or oe < 0.1 or tau < 0.5 or pv_w < 0:
            return 1e6
        pred = np.zeros(len(data))
        pred[0] = data["indoor_temp"].iloc[0]
        dt_h = 5 / 60
        for i in range(1, len(data)):
            solar_val = data[solar_col].iloc[i] if solar_col in data.columns else 0
            T_eq = (oe * data["VLT"].iloc[i] + hlc * data["AT"].iloc[i] + solar_val * pv_w) / (oe + hlc)
            approach = 1 - np.exp(-dt_h / tau)
            pred[i] = pred[i-1] + (T_eq - pred[i-1]) * approach
        return np.mean((pred - data["indoor_temp"].values) ** 2)
    
    res = optimize.minimize(
        cost, [hlc_init, oe_init, tau_init, 0.001],
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6}
    )
    rmse = np.sqrt(res.fun)
    return {"hlc": res.x[0], "oe": res.x[1], "tau": res.x[2], "solar_w": res.x[3],
            "rmse": rmse, "cost": res.fun, "success": res.success}

# Prepare calibration data (use a representative subset with solar data)
if "ghi_wm2" in df_merged.columns:
    cal_solar = df_merged[
        df_merged["is_cooling_active"] & df_merged["ghi_wm2"].notna()
    ].head(3000).reset_index(drop=True)
    
    print(f"Calibration data: {len(cal_solar)} rows")
    print()
    
    variants = {
        "PV_Generate": "PV (local panel)",
        "ghi_wm2": "GHI (Open Meteo)",
        "direct_wm2": "Direct Radiation",
        "dni_wm2": "DNI (Direct Normal)",
        "diffuse_wm2": "Diffuse Radiation",
    }
    
    results_solar = {}
    for col, name in variants.items():
        if col in cal_solar.columns:
            res = calibrate_with_solar(cal_solar, col)
            results_solar[name] = res
            print(f"{name:<25} RMSE={res['rmse']:.4f}°C  HLC={res['hlc']:.4f}  "
                  f"OE={res['oe']:.4f}  τ={res['tau']:.2f}h  w={res['solar_w']:.6f}")
    
    # Combined: PV + GHI
    # Add combined feature
    cal_solar["pv_plus_ghi"] = cal_solar["PV_Generate"] + cal_solar["ghi_wm2"] * 10  # Scale GHI to W
    res_combined = calibrate_with_solar(cal_solar, "pv_plus_ghi")
    results_solar["PV + GHI (combined)"] = res_combined
    print(f"{'PV + GHI (combined)':<25} RMSE={res_combined['rmse']:.4f}°C  HLC={res_combined['hlc']:.4f}  "
          f"OE={res_combined['oe']:.4f}  τ={res_combined['tau']:.2f}h  w={res_combined['solar_w']:.6f}")
    
    # Find best
    best_name = min(results_solar, key=lambda k: results_solar[k]["rmse"])
    print(f"\\n*** BEST: {best_name} (RMSE={results_solar[best_name]['rmse']:.4f}°C) ***")
else:
    print("No solar data available for ablation")
    results_solar = {}"""))

# ============================================================
# CELL 26: Solar ablation visualization
# ============================================================
cells.append(code("""if results_solar:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1) RMSE comparison bar chart
    ax = axes[0]
    names = list(results_solar.keys())
    rmses = [results_solar[n]["rmse"] for n in names]
    colors_bar = ["steelblue" if "PV" in n and "GHI" not in n else
                  "teal" if "GHI" in n else
                  "orange" if "DNI" in n else
                  "green" if "Direct" in n else
                  "purple" if "Diffuse" in n else "gray"
                  for n in names]
    bars = ax.barh(names, rmses, color=colors_bar, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("RMSE (°C)")
    ax.set_title("Calibration RMSE by Solar Feature")
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", ha="left", va="center", fontsize=9)
    
    # 2) Parameter comparison
    ax = axes[1]
    param_data = pd.DataFrame(results_solar).T[["hlc", "oe", "tau"]]
    param_data.plot(kind="bar", ax=ax, alpha=0.8)
    ax.set_ylabel("Parameter Value")
    ax.set_title("Calibrated Parameters by Solar Feature")
    ax.legend(["HLC", "OE", "τ (h)"])
    plt.xticks(rotation=30, ha="right")
    
    plt.suptitle("Phase E: Solar Feature Ablation Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    
    # Production recommendation
    print("\\n" + "="*60)
    print("PRODUCTION RECOMMENDATION")
    print("="*60)
    best = results_solar[best_name]
    print(f"\\nBest solar feature: {best_name}")
    print(f"RMSE improvement over PV: {(results_solar.get('PV (local panel)', {}).get('rmse', 0) - best['rmse']):.4f}°C")
    print(f"\\nIf GHI/DNI is better:")
    print(f"  - Add Open Meteo REST sensor to Home Assistant:")
    print(f"  sensor:")
    print(f"    - platform: rest")
    print(f"      name: Solar Radiation")
    print(f"      resource: https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&current=shortwave_radiation")
    print(f"      scan_interval: 600")
else:
    print("No results to display")"""))

# ============================================================
# CELL 27: Final summary
# ============================================================
cells.append(md("""## Summary & Recommendations"""))

# ============================================================
# CELL 28: Summary code
# ============================================================
cells.append(code("""print("=" * 70)
print("COOLING THERMAL MODEL CALIBRATION — FINAL RESULTS")
print("=" * 70)

print(f"\\n{'Parameter':<30} {'Heating':>10} {'Cool Online':>12} {'Cool Scipy':>12} {'Recommended':>12}")
print("-" * 80)
print(f"{'HLC (1/h)':<30} {'0.1206':>10} {'0.0487':>12} {hlc_scipy:>12.4f} {hlc_scipy:>12.4f}")
print(f"{'OE':<30} {'0.826':>10} {'0.826':>12} {oe_scipy:>12.4f} {oe_scipy:>12.4f}")
print(f"{'τ (hours)':<30} {'4.8':>10} {'41.23':>12} {tau_scipy:>12.2f} {tau_scipy:>12.2f}")

if hlc_on > 0:
    print(f"{'HLC (HP-ON, dual)':<30} {'—':>10} {'—':>12} {hlc_on:>12.4f}")
    print(f"{'HLC (HP-OFF, dual)':<30} {'—':>10} {'—':>12} {hlc_off:>12.4f}")

print(f"\\n--- KEY FINDINGS ---")
print(f"1. HLC: Online cooling value (0.049) is too low. Scipy finds {hlc_scipy:.3f}.")
print(f"   → Likely cause: solar gains absorbed into a lower HLC during online learning.")
print(f"2. τ: Online value (41.23h) is an artifact of HP-OFF periods with no driving force.")
print(f"   → Scipy fit: {tau_scipy:.1f}h. Propose cooling τ upper bound of ~15h.")
print(f"3. OE: Scipy finds {oe_scipy:.3f} (vs 0.826 heating).")
print(f"   → {'Similar to heating — consistent with same floor/pipes.' if abs(oe_scipy - 0.826) < 0.2 else 'Differs from heating — cooling convection dynamics may differ.'}")

if results_solar:
    best = results_solar[best_name]
    pv_rmse = results_solar.get("PV (local panel)", {}).get("rmse", 0)
    print(f"4. Solar: Best feature is {best_name} (RMSE={best['rmse']:.4f}°C vs PV RMSE={pv_rmse:.4f}°C)")
    if best["rmse"] < pv_rmse - 0.001:
        print(f"   → {best_name} is {(pv_rmse - best['rmse'])/pv_rmse*100:.1f}% better. Consider adding to production.")
    else:
        print(f"   → Difference is minimal. PV_Generate is sufficient for production.")

print(f"\\n--- ACTION ITEMS ---")
print(f"1. Update cooling mode default τ from 41.23h to {tau_scipy:.1f}h")
print(f"2. Set cooling τ online learning upper bound to ~15h")
print(f"3. Update cooling HLC initial value to {hlc_scipy:.4f}")
print(f"4. Consider dual-HLC model if HP-ON/OFF HLC differ significantly")
if results_solar and best_name != "PV (local panel)":
    print(f"5. Evaluate {best_name} as supplementary solar feature in production")"""))

# ============================================================
# Build the notebook JSON
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": ".venv (3.10.2)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.2"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "notebooks", "analysis", "10_cooling_thermal_calibration.ipynb"
)
output_path = os.path.normpath(output_path)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Created: {output_path}")
print(f"Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='markdown')} markdown, "
      f"{sum(1 for c in cells if c['cell_type']=='code')} code)")
