"""Quick analysis of HP-on vs HP-off thermal stats for HLC investigation."""
import pandas as pd, numpy as np

df = pd.read_csv('Logs_and_models/cooling_training_data.csv.gz')

# Check for solar/correction columns
solar_cols = [c for c in df.columns if 'solar' in c.lower() or 'correction' in c.lower() or 'factor' in c.lower()]
print("Solar/correction columns:", solar_cols)

hp_on = df[df['is_hp_active']==1]
hp_off = df[df['is_hp_active']==0]

for label, subset in [("HP-ON", hp_on), ("HP-OFF", hp_off)]:
    driving = subset['indoor_temp'] - subset['AT']
    print(f"\n{label}: {len(subset)} rows ({100*len(subset)/len(df):.1f}%)")
    print(f"  indoor_temp: {subset['indoor_temp'].mean():.2f} +/- {subset['indoor_temp'].std():.2f}")
    print(f"  AT (outdoor): {subset['AT'].mean():.2f} +/- {subset['AT'].std():.2f}")
    print(f"  delta_t: {subset['delta_t'].mean():.2f} (cooling active delta)")
    print(f"  PV_Generate: {subset['PV_Generate'].mean():.0f}W, median={subset['PV_Generate'].median():.0f}W")
    print(f"  driving_force: {driving.mean():.2f} +/- {driving.std():.2f}")
    print(f"  indoor_trend_30m: {subset['indoor_trend_30m'].mean():.4f}")
    print(f"  indoor_temp_gradient: {subset['indoor_temp_gradient'].mean():.4f}")
    
    # Daytime vs nighttime
    hour = (np.degrees(np.arctan2(subset['hour_sin'], subset['hour_cos'])) / 15) % 24
    daytime = (hour >= 8) & (hour <= 20)
    print(f"  daytime (8-20h): {daytime.sum()} ({100*daytime.mean():.1f}%)")
    print(f"  nighttime: {(~daytime).sum()} ({100*(~daytime).mean():.1f}%)")
    
    # PV during HP-off
    if label == "HP-OFF":
        high_pv = subset['PV_Generate'] > 1000
        print(f"  PV>1000W: {high_pv.sum()} ({100*high_pv.mean():.1f}%)")
        print(f"    trend when PV>1000: {subset.loc[high_pv, 'indoor_trend_30m'].mean():.4f}")
        print(f"    trend when PV<1000: {subset.loc[~high_pv, 'indoor_trend_30m'].mean():.4f}")

# VLT (outlet) during HP-off — is it near indoor (no flow) or still different?
print("\n=== OUTLET TEMPERATURE ===")
print(f"HP-ON: VLT={hp_on['VLT'].mean():.1f}, inlet-VLT diff={hp_on['delta_t'].mean():.2f}")
print(f"HP-OFF: VLT={hp_off['VLT'].mean():.1f}, inlet-VLT diff={hp_off['delta_t'].mean():.2f}")

# Key: when HP is off, outlet approaches indoor → OE contribution shrinks
outlet_indoor_on = hp_on['VLT'] - hp_on['indoor_temp']
outlet_indoor_off = hp_off['VLT'] - hp_off['indoor_temp']
print(f"\nVLT - indoor_temp:")
print(f"  HP-ON: {outlet_indoor_on.mean():.2f} (outlet below indoor → cooling)")
print(f"  HP-OFF: {outlet_indoor_off.mean():.2f} (outlet near indoor → minimal effect)")
