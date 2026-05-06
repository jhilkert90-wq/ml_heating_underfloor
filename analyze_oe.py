"""Quick analysis: why does OE converge to 0.81 instead of 0.95?"""
import json
import numpy as np

with open("Logs/stable_periods.json") as f:
    txt = f.read().replace("Infinity", "1e9").replace("-Infinity", "-1e9").replace("NaN", "null")
    periods = json.loads(txt)

# HP-only filter (same as _filter_hp_only_periods in physics_calibration.py)
MIN_THERMAL_POWER = 0.3  # config.HEATING_MIN_THERMAL_POWER_KW default
DEFROST_GRACE = 45  # config.DEFROST_RECOVERY_GRACE_MINUTES default
hp = []
for p in periods:
    pv = p.get("pv_power", 0)
    if pv is None:
        pv = 0
    if pv >= 100:
        continue
    if p.get("fireplace_on", 0) != 0:
        continue
    if p.get("tv_on", 0) != 0:
        continue
    tp = p.get("thermal_power_kw", 0)
    if tp is None or tp < MIN_THERMAL_POWER:
        continue
    msd = p.get("minutes_since_defrost", float("inf"))
    if msd is None:
        msd = float("inf")
    if msd < DEFROST_GRACE:
        continue
    eff = p.get("effective_temp", p.get("outlet_temp", 1))
    inl = p.get("inlet_temp", 0)
    if eff <= inl:
        continue
    hp.append(p)

print(f"HP-only periods: {len(hp)}")
print(f"Sample fields: {list(hp[0].keys())[:20]}")

# Check what effective_temp looks like
hlc = 0.11864
drives = []
for p in hp:
    t_in = p.get("indoor_temp")
    t_eff = p.get("effective_temp", p.get("outlet_temp"))
    if t_in is not None and t_eff is not None:
        if not np.isnan(t_in) and not np.isnan(t_eff):
            drives.append(t_eff - t_in)

drives = np.array(drives)
print(f"\nDrive distribution (eff_temp - indoor):")
print(f"  P10={np.percentile(drives,10):.2f}, P25={np.percentile(drives,25):.2f}, "
      f"P50={np.percentile(drives,50):.2f}, P75={np.percentile(drives,75):.2f}, P90={np.percentile(drives,90):.2f}")

# MAE at different OE values
print("\nMAE comparison at different OE values (drive >= 3°C):")
for oe_test in [0.50, 0.60, 0.70, 0.81, 0.90, 0.95, 1.00, 1.10]:
    errors = []
    for p in hp:
        t_in = p.get("indoor_temp")
        t_out = p.get("outdoor_temp")
        t_eff = p.get("effective_temp", p.get("outlet_temp"))
        if t_in is None or t_out is None or t_eff is None:
            continue
        if np.isnan(t_in) or np.isnan(t_out) or np.isnan(t_eff):
            continue
        drive = t_eff - t_in
        if drive < 3.0:
            continue
        denom = oe_test + hlc
        predicted = (oe_test * t_eff + hlc * t_out) / denom
        errors.append(predicted - t_in)
    errors = np.array(errors)
    print(f"  OE={oe_test:.2f}: MAE={np.mean(np.abs(errors)):.4f}°C, "
          f"bias={np.mean(errors):+.4f}°C, RMSE={np.sqrt(np.mean(errors**2)):.4f}°C (N={len(errors)})")

# Same but for ALL drive values (no minimum)
print("\nMAE comparison at different OE values (ALL drives):")
for oe_test in [0.50, 0.60, 0.70, 0.81, 0.90, 0.95, 1.00, 1.10]:
    errors = []
    for p in hp:
        t_in = p.get("indoor_temp")
        t_out = p.get("outdoor_temp")
        t_eff = p.get("effective_temp", p.get("outlet_temp"))
        if t_in is None or t_out is None or t_eff is None:
            continue
        if np.isnan(t_in) or np.isnan(t_out) or np.isnan(t_eff):
            continue
        drive = t_eff - t_in
        if drive < 0.5:  # tiny filter to avoid division issues
            continue
        denom = oe_test + hlc
        predicted = (oe_test * t_eff + hlc * t_out) / denom
        errors.append(predicted - t_in)
    errors = np.array(errors)
    print(f"  OE={oe_test:.2f}: MAE={np.mean(np.abs(errors)):.4f}°C, "
          f"bias={np.mean(errors):+.4f}°C, RMSE={np.sqrt(np.mean(errors**2)):.4f}°C (N={len(errors)})")

# Show sample data points to understand the data
print("\nSample HP-only periods with drive >= 3°C:")
count = 0
for p in hp:
    t_in = p.get("indoor_temp")
    t_out = p.get("outdoor_temp")
    t_eff = p.get("effective_temp", p.get("outlet_temp"))
    if t_in is None or t_out is None or t_eff is None:
        continue
    drive = t_eff - t_in
    if drive >= 3.0:
        bt2 = p.get("inlet_temp", "?")
        bt3 = p.get("outlet_temp", "?")
        print(f"  indoor={t_in:.1f}, outdoor={t_out:.1f}, eff_temp={t_eff:.1f}, "
              f"inlet={bt2}, outlet={bt3}, drive={drive:.1f}")
        count += 1
        if count >= 8:
            break

# Check if effective_temp is being computed as (inlet+outlet)/2
print("\nVerify effective_temp calculation:")
count = 0
for p in hp:
    bt2 = p.get("inlet_temp")
    bt3 = p.get("outlet_temp")
    t_eff = p.get("effective_temp")
    if bt2 is not None and bt3 is not None and t_eff is not None:
        calc = (bt2 + bt3) / 2
        diff = abs(t_eff - calc)
        if count < 5:
            print(f"  inlet={bt2:.1f}, outlet={bt3:.1f}, eff_calc={calc:.1f}, eff_stored={t_eff:.1f}, diff={diff:.3f}")
        count += 1
if count > 0:
    print(f"  Checked {count} periods")
