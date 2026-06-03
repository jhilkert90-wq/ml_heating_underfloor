"""Deep analysis: HP-off HLC with Beschattung confound."""
import pandas as pd, numpy as np

df = pd.read_csv('Logs_and_models/cooling_training_data.csv.gz')
hour = (np.degrees(np.arctan2(df['hour_sin'], df['hour_cos'])) / 15) % 24
df['hour'] = hour

hp_off = df[df['is_hp_active']==0].copy()
hp_off['daytime'] = (hp_off['hour'] >= 8) & (hp_off['hour'] <= 20)
hp_off['driving_force'] = hp_off['indoor_temp'] - hp_off['AT']

# Nighttime HP-off: no solar, no Beschattung → pure envelope HLC
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
    gradient = sub['indoor_temp_gradient']
    
    # Effective HLC from energy balance: dT/dt ≈ HLC * (T_outdoor - T_indoor) / C
    # But we can estimate relative HLC from slope of trend vs driving force
    valid = sub[['indoor_trend_30m', 'driving_force']].dropna()
    if len(valid) > 50:
        slope, intercept, r, p, se = __import__('scipy').stats.linregress(
            valid['driving_force'], valid['indoor_trend_30m'])
    else:
        slope, r = 0, 0
    
    print(f"{label}:")
    print(f"  n={len(sub)}, driving_force={df_force.mean():.2f}+/-{df_force.std():.2f}")
    print(f"  indoor_trend_30m={trend.mean():.4f}+/-{trend.std():.4f}")
    print(f"  gradient={gradient.mean():.4f}+/-{gradient.std():.4f}")
    print(f"  PV_Generate: mean={sub['PV_Generate'].mean():.0f}W")
    print(f"  VLT-indoor: {(sub['VLT']-sub['indoor_temp']).mean():.3f}")
    print(f"  Slope(trend vs force): {slope:.6f} (r={r:.3f}, p={p:.4f})")
    print(f"  → Interpretation: slope ∝ HLC/C, higher = more heat exchange")
    print()

print("=== KEY QUESTION: Is HLC_off=0.016 real or Beschattung artifact? ===")
print()

# Nighttime HP-off: estimate HLC from cooling rate
# Newton cooling: dT/dt = -(T_in - T_out) / tau_eff
# HLC_off = C / tau_eff (where C = thermal capacity)
night_valid = night[night['driving_force'].abs() > 2]  # Need sufficient driving force
if len(night_valid) > 100:
    from scipy import stats
    slope_night, _, r_night, p_night, _ = stats.linregress(
        night_valid['driving_force'], night_valid['indoor_trend_30m'])
    
    # Compare to HP-on cooling rate  
    hp_on = df[(df['is_hp_active']==1) & (df['delta_t'] < -0.5)].copy()
    hp_on['driving_force'] = hp_on['indoor_temp'] - hp_on['AT']
    hp_on_valid = hp_on[hp_on['driving_force'].abs() > 2]
    if len(hp_on_valid) > 100:
        slope_on, _, r_on, _, _ = stats.linregress(
            hp_on_valid['driving_force'], hp_on_valid['indoor_trend_30m'])
        
        print(f"Night HP-off slope: {slope_night:.6f} (r={r_night:.3f})")
        print(f"Active HP-on slope: {slope_on:.6f} (r={r_on:.3f})")
        print(f"Ratio on/off: {slope_on/slope_night:.1f}x")
        print()
        print(f"NB10 HLC_on/HLC_off ratio: {0.146/0.016:.1f}x")
        print(f"Log-derived ratio: {slope_on/slope_night:.1f}x")
        print()
        
        # Daytime HP-off with PV (Beschattung active)  
        if len(day_hpv) > 100:
            slope_day_pv, _, r_day_pv, _, _ = stats.linregress(
                day_hpv['driving_force'], day_hpv['indoor_trend_30m'])
            print(f"Day+PV HP-off slope: {slope_day_pv:.6f} (r={r_day_pv:.3f})")
            print(f"Night/Day+PV ratio: {slope_night/slope_day_pv:.2f}x")
            print("  → If ratio >> 1: Beschattung is masking HLC (reducing apparent heat exchange)")
            print("  → If ratio ≈ 1: HLC_off is genuinely low regardless of Beschattung")
