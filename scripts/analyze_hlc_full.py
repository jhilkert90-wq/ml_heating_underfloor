"""Deep analysis: HP-off HLC with Beschattung confound."""
import pandas as pd, numpy as np
from scipy import stats

df = pd.read_csv('Logs_and_models/cooling_training_data.csv.gz')
hour = (np.degrees(np.arctan2(df['hour_sin'], df['hour_cos'])) / 15) % 24
df['hour'] = hour
df['driving_force'] = df['indoor_temp'] - df['AT']

hp_off = df[df['is_hp_active']==0].copy()
hp_off['daytime'] = (hp_off['hour'] >= 8) & (hp_off['hour'] <= 20)

night = hp_off[~hp_off['daytime']]
day_hpv = hp_off[hp_off['daytime'] & (hp_off['PV_Generate'] > 1000)]
day_nopv = hp_off[hp_off['daytime'] & (hp_off['PV_Generate'] <= 100)]

print("=== HP-OFF: Nighttime vs Daytime (Beschattung effect) ===\n")
for label, sub in [("Night (21-8h, no solar)", night),
                    ("Day + PV>1000W (Beschattung active)", day_hpv),
                    ("Day + PV<100W (cloudy, no Beschattung)", day_nopv)]:
    if len(sub) < 50:
        print(f"{label}: too few rows ({len(sub)})")
        continue
    df_force = sub['driving_force']
    trend = sub['indoor_trend_30m']
    valid = sub[['indoor_trend_30m', 'driving_force']].dropna()
    if len(valid) > 50:
        slope, intercept, r, p, se = stats.linregress(valid['driving_force'], valid['indoor_trend_30m'])
    else:
        slope, r, p = 0, 0, 1
    
    print(f"{label}:")
    print(f"  n={len(sub)}, driving_force={df_force.mean():.2f}+/-{df_force.std():.2f}")
    print(f"  indoor_trend_30m={trend.mean():.4f}+/-{trend.std():.4f}")
    print(f"  PV_Generate: mean={sub['PV_Generate'].mean():.0f}W")
    print(f"  Slope(trend vs force): {slope:.6f} (r={r:.3f})")
    print()

print("=== KEY QUESTION: Is HLC_off=0.016 real or Beschattung artifact? ===\n")

# Nighttime HP-off with sufficient driving force
night_valid = night[night['driving_force'].abs() > 2]
slope_night, _, r_night, _, _ = stats.linregress(
    night_valid['driving_force'], night_valid['indoor_trend_30m'])

# HP-on with actual cooling delta
hp_on = df[(df['is_hp_active']==1) & (df['delta_t'] < -0.5)].copy()
hp_on_valid = hp_on[hp_on['driving_force'].abs() > 2]

if len(hp_on_valid) > 100:
    slope_on, _, r_on, _, _ = stats.linregress(
        hp_on_valid['driving_force'], hp_on_valid['indoor_trend_30m'])
    
    print(f"Night HP-off slope: {slope_night:.6f} (r={r_night:.3f})")
    print(f"Active HP-on slope (delta_t<-0.5): {slope_on:.6f} (r={r_on:.3f})")
    if abs(slope_night) > 0:
        print(f"Ratio on/off: {abs(slope_on/slope_night):.1f}x")
    print()
    
    # Daytime HP-off with PV (Beschattung active)
    day_valid = day_hpv[day_hpv['driving_force'].abs() > 2]
    slope_day, _, r_day, _, _ = stats.linregress(
        day_valid['driving_force'], day_valid['indoor_trend_30m'])
    print(f"Day+PV HP-off slope: {slope_day:.6f} (r={r_day:.3f})")
    if abs(slope_day) > 0:
        print(f"Night/Day+PV ratio: {abs(slope_night/slope_day):.2f}x")
    print()
    
    print("NB10 calibration:")
    print(f"  HLC_on = 0.146, HLC_off = 0.016, ratio = {0.146/0.016:.1f}x")
    print()

# Also estimate HLC from temperature change rate during clear nighttime windows
print("=== EFFECTIVE HLC ESTIMATION FROM NIGHTTIME COOLING ===\n")
# During night HP-off: dT_indoor/dt = -HLC/C * (T_indoor - T_outdoor) + Q_internal/C
# Where Q_internal is occupancy/appliance heat (small, ~0.3-0.5 kW for sleeping house)
# At equilibrium: HLC * dT = Q_internal
# Using slope: slope = -HLC/C => HLC = -slope * C

# Typical German house thermal mass: C = 50-100 kWh/K = 180-360 MJ/K
# Let's use C = 70 kWh/K (modern well-insulated house, ~200m2)
C_building = 70  # kWh/K
print(f"Assuming C_building = {C_building} kWh/K (typical German house)")
print(f"  slope_night = {slope_night:.6f} degC/10min per degC driving force")
print(f"  = {slope_night * 6:.6f} degC/h per degC")
hlc_night = abs(slope_night * 6 * C_building)
print(f"  HLC_night = |slope| * 6 * C = {hlc_night:.3f} kW/K")
print()
print(f"  NB10 HLC_off = 0.016 kW/K")
print(f"  Night-derived HLC = {hlc_night:.3f} kW/K")
print(f"  Night HLC / NB10 HLC_off = {hlc_night/0.016:.1f}x")
print()

# Also check: what's the typical night cooling rate?
night_cooling = night[night['driving_force'] > 3]
if len(night_cooling) > 100:
    mean_trend = night_cooling['indoor_trend_30m'].mean()
    mean_force = night_cooling['driving_force'].mean()
    print(f"Night HP-off (dF>3): avg trend={mean_trend:.4f} degC/10min, avg dF={mean_force:.1f} degC")
    print(f"  = {mean_trend*6:.3f} degC/h cooling rate")
    print(f"  With dF={mean_force:.1f}: HLC_eff = |trend*6| * C / dF = {abs(mean_trend*6)*C_building/mean_force:.3f} kW/K")

print("\n=== CONCLUSION ===")
print()
print("Two confounders make HLC_off appear low:")
print("  1. Beschattung blocks solar gain during daytime HP-off")
print("     -> temp stays stable even with moderate HLC")
print("  2. Small driving force in summer (indoor-outdoor ~2-5 degC)")
print("     -> minimal heat exchange regardless of HLC")
print("  3. HP-on ventilation increases effective HLC (physical)")
print()
print("True building HLC (envelope) is likely between HLC_off and HLC_on")
print("The dual-HLC model needs Beschattung-aware calibration")
