"""
Nighttime HLC Calculation — Single Building Envelope HLC

Physical reasoning:
- HLC is a BUILDING ENVELOPE property — it doesn't change when the HP turns on/off
- HP cooling mode only slows overheating via cold outlet water (captured by OE)
- At night: HP off, PV=0 -> pure envelope cooling: dT/dt ~ -HLC/C * (T_indoor - T_outdoor)
- This is the cleanest signal to measure the true building HLC

Three methods:
A) Linear regression: indoor_trend vs driving_force -> slope = -HLC/C
B) Newton cooling on contiguous nighttime segments
C) Scipy optimize single-HLC model on nighttime data + validate on full data
"""
import sys, os, warnings, calendar
import numpy as np
import pandas as pd
from scipy import optimize, stats
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DATA_PATH = PROJECT_ROOT / "Logs_and_models" / "cooling_training_data.csv.gz"

# --- Data Loading ---
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["hour"] = (np.degrees(np.arctan2(df["hour_sin"], df["hour_cos"])) / 15) % 24
df["driving_force"] = df["indoor_temp"] - df["AT"]
mask_clean = (df["AT"] > -10) & (df["AT"] < 45) & (df["VLT"] > 10) & (df["VLT"] < 45)
df = df[mask_clean].copy()
print(f"Clean data: {len(df)} rows")

# --- Fetch Open Meteo GHI ---
import requests, urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LAT, LON = 48.928, 10.069
chunks = []
print("Fetching Open Meteo GHI...")
y, m = 2025, 5
while (y, m) <= (2026, 5):
    last_day = calendar.monthrange(y, m)[1]
    s_str = f"{y}-{m:02d}-01"
    e_str = f"{y}-{m:02d}-{last_day:02d}"
    try:
        url = (f"https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={LAT}&longitude={LON}&start_date={s_str}&end_date={e_str}"
               f"&hourly=shortwave_radiation&timezone=Europe%2FBerlin")
        resp = requests.get(url, timeout=60, verify=False)
        resp.raise_for_status()
        data = resp.json()
        hourly = data["hourly"]
        chunk_df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"]),
            "ghi_wm2": hourly["shortwave_radiation"]
        })
        chunks.append(chunk_df)
    except Exception as e:
        print(f"  {s_str}: FAILED ({str(e)[:60]})")
    m += 1
    if m > 12:
        m, y = 1, y + 1

solar_df = pd.concat(chunks, ignore_index=True).set_index("timestamp").sort_index()
solar_df = solar_df[~solar_df.index.duplicated(keep="first")]
solar_df["doy"] = solar_df.index.dayofyear
solar_df["hour_of_day"] = solar_df.index.hour
solar_avg = solar_df.groupby(["doy", "hour_of_day"])["ghi_wm2"].mean().reset_index()

# Merge GHI into training data
doy_angle = np.arctan2(df["doy_sin"], df["doy_cos"])
doy = (doy_angle * 365.25 / (2 * np.pi)) % 365.25
df["doy_approx"] = doy.round().astype(int).clip(1, 365)
df["hour_approx"] = df["hour"].round().astype(int).clip(0, 23)
df = df.merge(solar_avg, left_on=["doy_approx", "hour_approx"],
              right_on=["doy", "hour_of_day"], how="left")
print(f"Data ready: {len(df)} rows, GHI matched: {df['ghi_wm2'].notna().sum()}")

# --- Nighttime Filter ---
night_hp_off = df[(df["is_hp_active"] == 0) &
                  (df["PV_Generate"] < 50) &
                  ((df["hour"] < 6) | (df["hour"] > 21))].copy()
print(f"\nNighttime HP-off data: {len(night_hp_off)} rows")
print(f"  Mean driving force: {night_hp_off['driving_force'].mean():.2f} K")
print(f"  Mean indoor_temp: {night_hp_off['indoor_temp'].mean():.2f} C")
print(f"  Mean outdoor: {night_hp_off['AT'].mean():.2f} C")

# =========================================================================
print("\n" + "=" * 75)
print("METHOD A: LINEAR REGRESSION (indoor_trend vs driving_force)")
print("=" * 75)
# =========================================================================

valid = night_hp_off[["indoor_trend_30m", "driving_force"]].dropna()
valid = valid[valid["driving_force"].abs() > 0.5]
sl, intercept, r, p, se = stats.linregress(valid["driving_force"], valid["indoor_trend_30m"])
print(f"  N = {len(valid)}")
print(f"  slope = {sl:.6f} (K/10min per K driving force)")
print(f"  r = {r:.4f}, p = {p:.2e}")
print(f"  Interpretation: indoor_trend_30m = {sl:.6f} * driving_force + {intercept:.6f}")
print(f"  --> Negative slope means indoor cools when indoor > outdoor (expected)")
print(f"  --> slope magnitude ~ HLC/C (heat loss per unit driving force per time)")

# =========================================================================
print("\n" + "=" * 75)
print("METHOD B: NEWTON COOLING ON CONTIGUOUS NIGHTTIME SEGMENTS")
print("=" * 75)
# =========================================================================

# Find contiguous nighttime HP-off segments (at least 30 min = 6 steps)
night_hp_off_sorted = night_hp_off.sort_index()
min_segment_len = 6  # 30 min at 5-min resolution

segments = []
current_start = None
current_indices = []

for i in range(len(night_hp_off_sorted)):
    idx = night_hp_off_sorted.index[i]
    if current_start is None:
        current_start = idx
        current_indices = [idx]
    else:
        # Check if consecutive (within 2 index positions = 10 min gap tolerance)
        if idx - current_indices[-1] <= 2:
            current_indices.append(idx)
        else:
            if len(current_indices) >= min_segment_len:
                segments.append(current_indices)
            current_start = idx
            current_indices = [idx]

if len(current_indices) >= min_segment_len:
    segments.append(current_indices)

print(f"  Found {len(segments)} contiguous nighttime segments (>= 30 min)")
seg_lengths = [len(s) * 5 for s in segments]
print(f"  Segment lengths: min={min(seg_lengths)} min, max={max(seg_lengths)} min, "
      f"median={np.median(seg_lengths):.0f} min")

# Fit Newton cooling to each segment: T(t) = T_eq + (T0 - T_eq) * exp(-t/tau)
# At night with HP off: T_eq ~ T_outdoor (if HLC dominates) or weighted
# Simpler: just measure dT/dt and correlate with (T_indoor - T_outdoor)

tau_estimates = []
hlc_c_estimates = []  # HLC/C ratio

for seg_idx, seg_indices in enumerate(segments):
    seg_data = night_hp_off_sorted.loc[seg_indices]
    if len(seg_data) < min_segment_len:
        continue

    T_indoor = seg_data["indoor_temp"].values
    T_outdoor = seg_data["AT"].values
    dt_h = 5 / 60  # hours

    # Method: dT/dt = -(1/tau_env) * (T_indoor - T_outdoor_mean)
    # where tau_env = C / HLC
    T_out_mean = T_outdoor.mean()
    driving = T_indoor - T_out_mean

    if driving.std() < 0.1 or len(driving) < 4:
        continue

    # Fit exponential decay: T_indoor(t) = T_out_mean + A * exp(-t/tau)
    t_hours = np.arange(len(T_indoor)) * dt_h
    delta_T = T_indoor - T_out_mean

    if delta_T[0] < 0.5:  # Need meaningful initial difference
        continue

    try:
        # Log-linear fit: ln(T - T_eq) = ln(A) - t/tau
        log_delta = np.log(np.maximum(delta_T, 0.01))
        slope_log, intercept_log, r_log, _, _ = stats.linregress(t_hours, log_delta)
        if slope_log >= 0:  # Not cooling
            continue
        tau_est = -1.0 / slope_log
        if 1.0 < tau_est < 100.0:  # Reasonable range
            tau_estimates.append(tau_est)
            hlc_c_estimates.append(1.0 / tau_est)  # HLC/C = 1/tau_env
    except Exception:
        continue

if tau_estimates:
    tau_arr = np.array(tau_estimates)
    hlc_c_arr = np.array(hlc_c_estimates)
    print(f"\n  Fitted {len(tau_estimates)} segments successfully")
    print(f"  tau_envelope (C/HLC): median={np.median(tau_arr):.1f} h, "
          f"mean={np.mean(tau_arr):.1f} h, std={np.std(tau_arr):.1f} h")
    print(f"  HLC/C ratio: median={np.median(hlc_c_arr):.4f}, "
          f"mean={np.mean(hlc_c_arr):.4f}")
    print(f"  --> This is the pure envelope time constant (no HP, no solar)")
    print(f"  --> For tau_system ~ 8h and OE ~ 0.2: tau_env = C/HLC, tau_sys = C/(HLC+OE)")
    print(f"  --> If tau_env >> tau_sys: HLC << OE (well insulated, HP dominates cooling)")

    # Estimate HLC from tau_envelope
    # tau_env = C / HLC => HLC = C / tau_env
    # We need C (thermal capacitance). From tau_system = C / (HLC + OE):
    # If tau_sys = 8h, OE = 0.2, then C = tau_sys * (HLC + OE)
    # Substituting: tau_env = tau_sys * (HLC + OE) / HLC = tau_sys * (1 + OE/HLC)
    # So: tau_env / tau_sys = 1 + OE/HLC => HLC = OE / (tau_env/tau_sys - 1)
    tau_sys = 8.0  # From NB10
    OE_est = 0.20  # From NB10
    tau_env_median = np.median(tau_arr)
    if tau_env_median > tau_sys:
        hlc_from_tau = OE_est / (tau_env_median / tau_sys - 1)
        print(f"\n  HLC estimate from tau_env/tau_sys ratio:")
        print(f"    tau_env = {tau_env_median:.1f} h, tau_sys = {tau_sys} h, OE = {OE_est}")
        print(f"    HLC = OE / (tau_env/tau_sys - 1) = {hlc_from_tau:.4f}")
else:
    print("  No segments successfully fitted")

# =========================================================================
print("\n" + "=" * 75)
print("METHOD C: SCIPY OPTIMIZE — SINGLE-HLC ON NIGHTTIME DATA")
print("=" * 75)
# =========================================================================

def _precompute_arrays(data, solar_col):
    """Precompute numpy arrays from dataframe (call once, pass to cost functions)."""
    vlt = data["VLT"].values.astype(np.float64)
    at = data["AT"].values.astype(np.float64)
    indoor = data["indoor_temp"].values.astype(np.float64)
    solar = data[solar_col].fillna(0).values.astype(np.float64) if solar_col in data.columns else np.zeros(len(data))
    hp_active = data["is_hp_active"].values.astype(np.float64) if "is_hp_active" in data.columns else np.zeros(len(data))
    return vlt, at, indoor, solar, hp_active

def _simulate_single_hlc(HLC, OE, tau, solar_w, vlt, at, indoor0, solar_vals, n):
    """Simulate Newton cooling with single HLC — vectorized T_eq, sequential recurrence."""
    Q_ext = solar_vals * solar_w
    Q_ext = 3.0 * np.tanh(Q_ext / 3.0)
    T_eq_all = (OE * vlt + HLC * at + Q_ext) / (OE + HLC)
    approach = 1 - np.exp(-(5.0/60.0) / tau)
    pred = np.empty(n)
    pred[0] = indoor0
    for i in range(1, n):
        pred[i] = pred[i-1] + (T_eq_all[i] - pred[i-1]) * approach
    return pred

def cost_single_hlc(params, arrays):
    """Single-HLC model — same HLC for HP-on and HP-off."""
    HLC, OE, tau, solar_w = params
    if HLC < 0.005 or OE < 0.01 or tau < 0.5 or solar_w < 0:
        return 1e6
    vlt, at, indoor, solar, _ = arrays
    n = len(indoor)
    pred = _simulate_single_hlc(HLC, OE, tau, solar_w, vlt, at, indoor[0], solar, n)
    if np.any(np.isnan(pred)):
        return 1e6
    return np.mean((pred - indoor) ** 2)

# C.1: Calibrate on nighttime-only data (HP off, PV off)
max_night_rows = 3000
cal_night = night_hp_off.head(max_night_rows).copy()
print(f"\nC.1: Nighttime-only calibration ({len(cal_night)} rows, all HP-off)")

# Bounds: HLC, OE, tau, solar_w (solar_w should be ~0 at night but keep for consistency)
bounds_night = [(0.005, 0.5), (0.01, 1.0), (1.0, 50.0), (0.0, 0.0)]  # Fix solar_w=0 at night

arrays_night = _precompute_arrays(cal_night, "ghi_wm2")
print("  Optimizing single-HLC on nighttime data (no solar)...")
res_night = optimize.differential_evolution(
    cost_single_hlc, bounds_night,
    args=(arrays_night,),
    maxiter=500, seed=42, tol=1e-6
)
rmse_night = np.sqrt(res_night.fun)
p = res_night.x
print(f"  HLC = {p[0]:.4f}, OE = {p[1]:.4f}, tau = {p[2]:.2f} h")
print(f"  RMSE = {rmse_night:.4f} C")
print(f"  Note: At night HP-off, VLT ~ indoor_temp, so OE effect is near zero")
print(f"        --> HLC and tau are the main drivers of fit quality")

HLC_NIGHTTIME = p[0]
TAU_NIGHTTIME = p[2]

# =========================================================================
print("\n" + "=" * 75)
print("METHOD C.2: VALIDATE — USE NIGHTTIME HLC ON FULL DATASET")
print("=" * 75)
# =========================================================================

# Now calibrate on mixed data (HP-on + HP-off, day + night) but with SINGLE HLC
# Only optimize OE, tau, solar_w — FIX HLC to nighttime value
max_rows = 5000
cal_full = df.head(max_rows).copy()
print(f"\nFull dataset: {len(cal_full)} rows")
print(f"  HP-ON: {cal_full['is_hp_active'].sum()}, HP-OFF: {(~cal_full['is_hp_active'].astype(bool)).sum()}")

# PV/GHI scale
pv_mean = cal_full["PV_Generate"].mean()
ghi_mean = cal_full["ghi_wm2"].dropna().mean()
scale_ratio = pv_mean / (ghi_mean + 1e-8)

# C.2a: Single-HLC fully free (optimize HLC + OE + tau + solar_w on all data)
print("\nC.2a: Single-HLC FREE on all data...")
bounds_free = [(0.005, 0.5), (0.01, 1.0), (1.0, 30.0), (0.0, 0.01 * scale_ratio)]
arrays_full = _precompute_arrays(cal_full, "ghi_wm2")
res_free = optimize.differential_evolution(
    cost_single_hlc, bounds_free,
    args=(arrays_full,),
    maxiter=500, seed=42, tol=1e-6
)
rmse_free = np.sqrt(res_free.fun)
pf = res_free.x
print(f"  HLC = {pf[0]:.4f}, OE = {pf[1]:.4f}, tau = {pf[2]:.2f} h, solar_w = {pf[3]:.6f}")
print(f"  RMSE = {rmse_free:.4f} C")

# C.2b: Single-HLC FIXED to nighttime value (optimize only OE, tau, solar_w)
def cost_fixed_hlc(params, arrays, hlc_fixed):
    OE, tau, solar_w = params
    return cost_single_hlc([hlc_fixed, OE, tau, solar_w], arrays)

print(f"\nC.2b: Single-HLC FIXED to nighttime value ({HLC_NIGHTTIME:.4f})...")
bounds_fixed = [(0.01, 1.0), (1.0, 30.0), (0.0, 0.01 * scale_ratio)]
res_fixed = optimize.differential_evolution(
    cost_fixed_hlc, bounds_fixed,
    args=(arrays_full, HLC_NIGHTTIME),
    maxiter=500, seed=42, tol=1e-6
)
rmse_fixed = np.sqrt(res_fixed.fun)
px = res_fixed.x
print(f"  HLC = {HLC_NIGHTTIME:.4f} (fixed), OE = {px[0]:.4f}, tau = {px[1]:.2f} h, solar_w = {px[2]:.6f}")
print(f"  RMSE = {rmse_fixed:.4f} C")

# C.2c: Dual-HLC reference (same as NB10/NB12 approach)
def cost_dual_hlc(params, arrays):
    HLC_on, HLC_off, OE, tau, solar_w = params
    if HLC_on < 0.005 or HLC_off < 0.001 or OE < 0.01 or tau < 0.5 or solar_w < 0:
        return 1e6
    vlt, at, indoor, solar, hp_active = arrays
    n = len(indoor)
    Q_ext = solar * solar_w
    Q_ext = 3.0 * np.tanh(Q_ext / 3.0)
    hlc_arr = np.where(hp_active == 1, HLC_on, HLC_off)
    T_eq_all = (OE * vlt + hlc_arr * at + Q_ext) / (OE + hlc_arr)
    approach = 1 - np.exp(-(5.0/60.0) / tau)
    pred = np.empty(n)
    pred[0] = indoor[0]
    for i in range(1, n):
        pred[i] = pred[i-1] + (T_eq_all[i] - pred[i-1]) * approach
    if np.any(np.isnan(pred)):
        return 1e6
    return np.mean((pred - indoor) ** 2)

print(f"\nC.2c: Dual-HLC reference on all data...")
bounds_dual = [(0.005, 0.8), (0.005, 0.3), (0.01, 1.0), (1.0, 30.0), (0.0, 0.01 * scale_ratio)]
res_dual = optimize.differential_evolution(
    cost_dual_hlc, bounds_dual,
    args=(arrays_full,),
    maxiter=500, seed=42, tol=1e-6
)
rmse_dual = np.sqrt(res_dual.fun)
pd_ = res_dual.x
print(f"  HLC_on = {pd_[0]:.4f}, HLC_off = {pd_[1]:.4f}, OE = {pd_[2]:.4f}, "
      f"tau = {pd_[3]:.2f} h, solar_w = {pd_[4]:.6f}")
print(f"  RMSE = {rmse_dual:.4f} C")
print(f"  HLC_on/HLC_off ratio: {pd_[0]/pd_[1]:.1f}x")

# C.2d: No-solar baseline (single HLC, no solar)
print(f"\nC.2d: Single-HLC no solar (baseline)...")
bounds_nosolar = [(0.005, 0.5), (0.01, 1.0), (1.0, 30.0), (0.0, 0.0)]
res_nosolar = optimize.differential_evolution(
    cost_single_hlc, bounds_nosolar,
    args=(arrays_full,),
    maxiter=500, seed=42, tol=1e-6
)
rmse_nosolar = np.sqrt(res_nosolar.fun)
pn = res_nosolar.x
print(f"  HLC = {pn[0]:.4f}, OE = {pn[1]:.4f}, tau = {pn[2]:.2f} h")
print(f"  RMSE = {rmse_nosolar:.4f} C")

# =========================================================================
print("\n" + "=" * 75)
print("COMPARISON TABLE")
print("=" * 75)
# =========================================================================

print(f"\n{'Model':<35s} {'RMSE':>8s} {'HLC':>8s} {'HLC_off':>8s} {'OE':>8s} {'tau':>6s} {'solar_w':>10s}")
print("-" * 95)
print(f"{'Night-only (HP-off, no solar)':<35s} {rmse_night:>8.4f} {HLC_NIGHTTIME:>8.4f} {'':>8s} {p[1]:>8.4f} {p[2]:>6.1f} {'0':>10s}")
print(f"{'Single-HLC free (all data+GHI)':<35s} {rmse_free:>8.4f} {pf[0]:>8.4f} {'':>8s} {pf[1]:>8.4f} {pf[2]:>6.1f} {pf[3]:>10.6f}")
print(f"{'Single-HLC fixed night (all+GHI)':<35s} {rmse_fixed:>8.4f} {HLC_NIGHTTIME:>8.4f} {'':>8s} {px[0]:>8.4f} {px[1]:>6.1f} {px[2]:>10.6f}")
print(f"{'Dual-HLC (all data+GHI)':<35s} {rmse_dual:>8.4f} {pd_[0]:>8.4f} {pd_[1]:>8.4f} {pd_[2]:>8.4f} {pd_[3]:>6.1f} {pd_[4]:>10.6f}")
print(f"{'Single-HLC no solar (baseline)':<35s} {rmse_nosolar:>8.4f} {pn[0]:>8.4f} {'':>8s} {pn[1]:>8.4f} {pn[2]:>6.1f} {'0':>10s}")

print(f"\nKey comparisons:")
print(f"  Single-HLC free vs Dual-HLC:    RMSE diff = {rmse_free - rmse_dual:+.4f} C")
print(f"  Single-HLC fixed vs Dual-HLC:   RMSE diff = {rmse_fixed - rmse_dual:+.4f} C")
print(f"  Single-HLC free vs no-solar:    RMSE diff = {rmse_free - rmse_nosolar:+.4f} C (solar gain)")

if rmse_free - rmse_dual < 0.05:
    print(f"\n  --> Single-HLC is within 0.05 C of Dual-HLC")
    print(f"  --> Dual-HLC improvement was likely optimizer artifact, not physics")
    print(f"  --> USE SINGLE HLC = {pf[0]:.4f} (or nighttime: {HLC_NIGHTTIME:.4f})")
else:
    print(f"\n  --> Dual-HLC gives meaningfully better RMSE ({rmse_dual:.4f} vs {rmse_free:.4f})")
    print(f"  --> HP fan MAY genuinely change convective coupling")
    print(f"  --> Consider: model fan effect as OE boost, not second HLC")

print(f"\n  Nighttime HLC = {HLC_NIGHTTIME:.4f}")
print(f"  All-data free HLC = {pf[0]:.4f}")
print(f"  Ratio: {pf[0]/HLC_NIGHTTIME:.2f}x")
