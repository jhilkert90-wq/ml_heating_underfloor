"""Section F analysis: HLC on/off with Beschattung confound for NB12."""
import sys, os, warnings
import numpy as np
import pandas as pd
from scipy import optimize, stats
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"c:\Users\ZOJHILK\OneDrive - Carl Zeiss AG\Dokumente\Heizung\PI4\ml_heating\ml_heating_underfloor")
DATA_PATH = PROJECT_ROOT / "Logs_and_models" / "cooling_training_data.csv.gz"

# Load data
df = pd.read_csv(DATA_PATH)
df["hour"] = (np.degrees(np.arctan2(df["hour_sin"], df["hour_cos"])) / 15) % 24
df["driving_force"] = df["indoor_temp"] - df["AT"]
mask_clean = (df["AT"] > -10) & (df["AT"] < 45) & (df["VLT"] > 10) & (df["VLT"] < 45)
df_clean = df[mask_clean].copy()
df_clean["is_cooling_active"] = (df_clean["is_hp_active"] == 1) & (df_clean["delta_t"] < -0.5)
doy_angle = np.arctan2(df_clean["doy_sin"], df_clean["doy_cos"])
doy = (doy_angle * 365.25 / (2 * np.pi)) % 365.25
df_clean["doy_approx"] = doy.round().astype(int).clip(1, 365)
df_clean["hour_approx"] = df_clean["hour"].round().astype(int).clip(0, 23)

# Load cached Open Meteo (use saved if available, else skip)
import requests, urllib3, calendar
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LAT, LON = 48.928, 10.069
HOURLY_VARS = "shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,global_tilted_irradiance_instant,sunshine_duration"

chunks = []
month_starts = []
y, m = 2025, 5
while (y, m) <= (2026, 5):
    month_starts.append((y, m))
    m += 1
    if m > 12:
        m, y = 1, y + 1

print("Fetching Open Meteo...", flush=True)
for y, m in month_starts:
    last_day = calendar.monthrange(y, m)[1]
    s_str, e_str = f"{y}-{m:02d}-01", f"{y}-{m:02d}-{last_day:02d}"
    try:
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={s_str}&end_date={e_str}&hourly={HOURLY_VARS}&timezone=Europe%2FBerlin"
        resp = requests.get(url, timeout=60, verify=False)
        resp.raise_for_status()
        data = resp.json()
        hourly = data["hourly"]
        chunk_df = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
        for var in ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance"]:
            if var in hourly:
                chunk_df[var] = hourly[var]
        for var in ["global_tilted_irradiance_instant", "sunshine_duration"]:
            if var in hourly and hourly[var] is not None:
                chunk_df[var] = hourly[var]
        chunks.append(chunk_df)
    except Exception as e:
        print(f"  {s_str}: FAILED ({str(e)[:60]})")

solar_df = pd.concat(chunks, ignore_index=True).set_index("timestamp").sort_index()
solar_df = solar_df[~solar_df.index.duplicated(keep="first")]
rename_map = {"shortwave_radiation": "ghi_wm2", "direct_radiation": "direct_wm2",
              "diffuse_radiation": "diffuse_wm2", "direct_normal_irradiance": "dni_wm2"}
if "global_tilted_irradiance_instant" in solar_df.columns:
    rename_map["global_tilted_irradiance_instant"] = "gti_wm2"
solar_df = solar_df.rename(columns=rename_map)
solar_df["doy"] = solar_df.index.dayofyear
solar_df["hour_of_day"] = solar_df.index.hour
solar_cols = [c for c in solar_df.columns if c not in ["doy", "hour_of_day"]]
solar_avg = solar_df.groupby(["doy", "hour_of_day"])[solar_cols].mean().reset_index()

df_merged = df_clean.merge(solar_avg, left_on=["doy_approx", "hour_approx"],
                           right_on=["doy", "hour_of_day"], how="left")
print(f"Data ready: {len(df_merged)} rows", flush=True)

# ====================================================================
# F.1: HP-off thermal drift by condition
# ====================================================================
print("\n" + "=" * 75)
print("F.1: HP-OFF THERMAL DRIFT BY CONDITION")
print("=" * 75 + "\n")

df_f = df_merged.copy()
df_f["driving_force"] = df_f["indoor_temp"] - df_f["AT"]
df_f["daytime"] = (df_f["hour"] >= 8) & (df_f["hour"] <= 20)

hp_off = df_f[df_f["is_hp_active"] == 0].copy()
night = hp_off[~hp_off["daytime"]]
day_pv_high = hp_off[hp_off["daytime"] & (hp_off["PV_Generate"] > 1000)]
day_pv_low = hp_off[hp_off["daytime"] & (hp_off["PV_Generate"] <= 100)]

conditions = {
    "Night HP-off (no solar/blinds)": night,
    "Day HP-off + PV>1kW (Beschattung zu)": day_pv_high,
    "Day HP-off + PV<100W (cloudy)": day_pv_low,
}

print(f"{'Condition':<42s} {'N':>6s} {'dF':>8s} {'trend':>10s} {'slope':>10s} {'r':>6s}")
print("-" * 85)

slopes_dict = {}
for label, sub in conditions.items():
    if len(sub) < 100:
        print(f"{label:<42s} {len(sub):>6d}  (insufficient)")
        continue
    valid = sub[["indoor_trend_30m", "driving_force"]].dropna()
    valid = valid[valid["driving_force"].abs() > 0.5]
    sl, intercept, r, p, se = stats.linregress(valid["driving_force"], valid["indoor_trend_30m"])
    slopes_dict[label] = {"slope": sl, "r": r, "n": len(valid),
                          "mean_df": valid["driving_force"].mean(),
                          "mean_trend": valid["indoor_trend_30m"].mean()}
    print(f"{label:<42s} {len(valid):>6d} {valid['driving_force'].mean():>8.2f} "
          f"{valid['indoor_trend_30m'].mean():>10.4f} {sl:>10.6f} {r:>6.3f}")

print()
if "Night HP-off (no solar/blinds)" in slopes_dict and "Day HP-off + PV>1kW (Beschattung zu)" in slopes_dict:
    night_sl = abs(slopes_dict["Night HP-off (no solar/blinds)"]["slope"])
    day_sl = abs(slopes_dict["Day HP-off + PV>1kW (Beschattung zu)"]["slope"])
    print(f"Night/Day+PV slope ratio: {night_sl/day_sl:.2f}x")
    print(f"  --> Beschattung reduces apparent heat exchange by {100*(1-day_sl/night_sl):.0f}%")

# ====================================================================
# F.3: Re-calibrate dual-HLC using nighttime-only HP-off data
# ====================================================================
print("\n" + "=" * 75)
print("F.3: RE-CALIBRATE DUAL-HLC (NIGHTTIME HP-OFF ONLY)")
print("=" * 75 + "\n")

def predict_equilibrium(T_outlet, T_outdoor, HLC, OE, solar_val=0, solar_w=0):
    Q_ext = solar_val * solar_w
    if abs(Q_ext) > 0:
        Q_ext = 3.0 * np.tanh(Q_ext / 3.0)
    return (OE * T_outlet + HLC * T_outdoor + Q_ext) / (OE + HLC)

def cost_dual_hlc_solar(params, data, solar_col):
    HLC_on, HLC_off, OE, tau, solar_w = params
    if HLC_on < 0.01 or HLC_off < 0.001 or OE < 0.05 or tau < 1.0 or solar_w < 0:
        return 1e6
    pred = np.zeros(len(data))
    pred[0] = data["indoor_temp"].iloc[0]
    hp_active = data["is_hp_active"].values
    solar_vals = data[solar_col].fillna(0).values if solar_col in data.columns else np.zeros(len(data))
    dt_h = 5 / 60
    for i in range(1, len(data)):
        hlc_i = HLC_on if hp_active[i] == 1 else HLC_off
        T_eq = predict_equilibrium(data["VLT"].iloc[i], data["AT"].iloc[i], hlc_i, OE,
                                   solar_vals[i], solar_w)
        approach = 1 - np.exp(-dt_h / tau)
        pred[i] = pred[i-1] + (T_eq - pred[i-1]) * approach
    if np.any(np.isnan(pred)):
        return 1e6
    return np.mean((pred - data["indoor_temp"].values) ** 2)

# Build calibration set: HP-on (all) + HP-off (nighttime only)
night_off_mask = (df_merged["is_hp_active"] == 0) & ((df_merged["hour"] < 6) | (df_merged["hour"] > 21))
hp_on_mask = df_merged["is_hp_active"] == 1
cal_night = df_merged[hp_on_mask | night_off_mask].head(5000).copy()

print(f"Night-filtered calibration: {len(cal_night)} rows")
print(f"  HP-ON: {cal_night['is_hp_active'].sum()}")
print(f"  HP-OFF (night only): {(~cal_night['is_hp_active'].astype(bool)).sum()}")

pv_mean = cal_night["PV_Generate"].mean()
ghi_mean = cal_night["ghi_wm2"].dropna().mean()
scale_ratio = pv_mean / (ghi_mean + 1e-8)

w_bounds = (0.0, 0.01 * scale_ratio)
bounds_night = [(0.02, 0.5), (0.005, 0.3), (0.05, 0.5), (1.5, 15.0), w_bounds]

print("\nCalibrating dual-HLC (nighttime HP-off only) with GHI...", flush=True)
res_night = optimize.differential_evolution(
    cost_dual_hlc_solar, bounds_night,
    args=(cal_night, "ghi_wm2"),
    maxiter=500, seed=42, tol=1e-6
)
rmse_night = np.sqrt(res_night.fun)
p = res_night.x

print(f"\n{'Parameter':<15s} {'NB10 (all data)':<18s} {'Night-filtered':<18s} {'Ratio':<10s}")
print("-" * 65)
nb10_params = {"HLC_on": 0.1464, "HLC_off": 0.0155, "OE": 0.2088, "tau": 8.00}
night_params = {"HLC_on": p[0], "HLC_off": p[1], "OE": p[2], "tau": p[3]}
for key in ["HLC_on", "HLC_off", "OE", "tau"]:
    r = night_params[key] / nb10_params[key] if nb10_params[key] > 0 else 0
    print(f"{key:<15s} {nb10_params[key]:<18.4f} {night_params[key]:<18.4f} {r:<10.2f}")
print(f"{'solar_w':<15s} {'N/A':<18s} {p[4]:<18.6f}")
print(f"{'RMSE':<15s} {'0.71':<18s} {rmse_night:<18.4f}")

hlc_ratio_night = night_params["HLC_on"] / night_params["HLC_off"]
print(f"\nHLC_on/HLC_off ratio: {hlc_ratio_night:.1f}x (NB10: 9.4x)")

# Also calibrate with ALL data but wider HLC_off bounds
print("\n\nCalibrating with ALL data (reference, same bounds)...", flush=True)
cal_all = df_merged.head(5000).copy()
res_all = optimize.differential_evolution(
    cost_dual_hlc_solar, bounds_night,
    args=(cal_all, "ghi_wm2"),
    maxiter=500, seed=42, tol=1e-6
)
rmse_all = np.sqrt(res_all.fun)
pa = res_all.x
print(f"All-data: HLC_on={pa[0]:.4f}, HLC_off={pa[1]:.4f}, OE={pa[2]:.4f}, "
      f"tau={pa[3]:.2f}, w={pa[4]:.6f}, RMSE={rmse_all:.4f}")
print(f"All-data ratio: {pa[0]/pa[1]:.1f}x")

# ====================================================================
# F.4: Physical interpretation
# ====================================================================
print("\n" + "=" * 75)
print("F.4: PHYSICAL INTERPRETATION")
print("=" * 75 + "\n")

print(f"Night-filtered: HLC_on={night_params['HLC_on']:.4f}, HLC_off={night_params['HLC_off']:.4f}, ratio={hlc_ratio_night:.1f}x")
print(f"All-data:       HLC_on={pa[0]:.4f}, HLC_off={pa[1]:.4f}, ratio={pa[0]/pa[1]:.1f}x")
print()

if hlc_ratio_night < 5:
    print("FINDING: Night-filtered ratio < 5x")
    print("  --> Beschattung WAS inflating the HLC difference")
    print("  --> HLC_off is higher than NB10 estimated (building not as isolated as thought)")
    print("  --> Phase 2: tighten HLC_off lower bound, add Beschattung correction")
elif hlc_ratio_night > 8:
    print("FINDING: Night-filtered ratio still > 8x")
    print("  --> Dual-HLC genuinely physical (HP ventilation dominates)")
    print("  --> Phase 2: keep dual-HLC as-is")
else:
    print(f"FINDING: Night-filtered ratio = {hlc_ratio_night:.1f}x (moderate)")
    print("  --> Mix of Beschattung artifact + real ventilation effect")
    print("  --> A ratio of 3-5x is physically plausible (forced vs natural convection)")
    print("  --> Phase 2: keep dual-HLC but tighten HLC_off bounds")

# Key numbers for Phase 2 prompt
print("\n--- VALUES FOR PHASE 2 PROMPT ---")
print(f"Recommended HLC_on bounds: [0.02, 0.5]")
print(f"Recommended HLC_off bounds: [0.01, 0.3] (raise lower from 0.005)")
print(f"Recommended OE bounds (cooling): [0.05, 0.5]")
print(f"Recommended tau bounds: [1.5, 15.0]")
print(f"GHI weight (calibrated): {p[4]:.6f}")
print(f"GHI weight bounds: [0.0, {0.01 * scale_ratio:.4f}]")
print(f"Scale ratio PV/GHI: {scale_ratio:.2f}")
