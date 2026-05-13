"""Implement all planned overhaul changes to nb03 and nb04.

Phase 1  – Fix IndexError: ffill sparse columns, fix tail removal, add row guard
Phase 2  – Feature overhaul: remove Helligkeit/Kuehlung_Soll/Pth_H,
           add thermal_power_kw/delta_t/outlet_indoor_diff + ADDON comments
Phase 3  – Add Section 11: SGDClassifier online learning demo
NB04     – Replace Helligkeit slider with new HP thermal state sliders
"""

import json
from pathlib import Path
import copy

ROOT = Path(__file__).parent.parent
NB03 = ROOT / "notebooks/analysis/03_overheating_ml_training.ipynb"
NB04 = ROOT / "notebooks/analysis/04_overheating_ml_interactive.ipynb"


# ── Helper: convert multiline string → notebook source line list ──────────────
def to_src(text: str) -> list:
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        result.append(line + "\n" if i < len(lines) - 1 else line)
    if result and result[-1] == "":
        result.pop()
    return result


def code_cell(text: str, cell_id: str = "") -> dict:
    c = {"cell_type": "code", "execution_count": None,
         "metadata": {}, "outputs": [], "source": to_src(text)}
    if cell_id:
        c["id"] = cell_id
    return c


def md_cell(text: str, cell_id: str = "") -> dict:
    c = {"cell_type": "markdown", "metadata": {}, "source": to_src(text)}
    if cell_id:
        c["id"] = cell_id
    return c


# ═══════════════════════════════════════════════════════════════════════════════
# NEW CELL CONTENT — NB03
# ═══════════════════════════════════════════════════════════════════════════════

RESAMPLE_FFILL = """\
# ── Resample both to 5-min grid and merge ────────────────────────────────────
# Use numeric columns only for resampling, then merge
sensor_5m = sensor_raw.resample("5min").mean(numeric_only=True)
pv_5m     = pv_raw[["PV_Generate", "PV_Consume", "PV_Export", "PV_Import", "PV_Batteriestatus"]].resample("5min").mean(numeric_only=True)

df = sensor_5m.join(pv_5m, how="left")

# ── Forward-fill sparse columns ───────────────────────────────────────────────
# AT_roh_Nh: the HA add-on fetches hourly weather forecasts from an API.
# After 5-min resampling, 11 of every 12 rows are NaN.
# ffill(limit=24) carries the last valid value forward up to 2 h (safe because
# the next API call overwrites within 1 h).
# NOTE PRODUCTION: weather API runs every control cycle → no ffill needed there.
at_forecast_cols = [c for c in df.columns if c.startswith("AT_roh_")]
if at_forecast_cols:
    df[at_forecast_cols] = df[at_forecast_cols].ffill(limit=24)
    print(f"  ffilled {len(at_forecast_cols)} AT forecast columns (limit=24 steps)")

# VLT / RLT / Vol / Pth_PC / Pel: only logged when the HP actively runs.
# ffill(limit=6) carries the last reading forward ≤ 30 min between HP cycles.
# NOTE PRODUCTION: sensors are polled every 5 min → no ffill needed there.
for col in ["VLT", "RLT", "Vol", "Pth_PC", "Pel"]:
    if col in df.columns:
        df[col] = df[col].ffill(limit=6).fillna(0.0)

print(f"Merged DataFrame: {len(df):,} rows × {df.shape[1]} columns")
print(f"Date range       : {df.index.min()} → {df.index.max()}")

# Show missing-value overview for key columns
key_cols = ["AT", "Pth_H", "Pth_PC", "VLT", "RLT", "Vol",
            "RT_Flur_OG", "RT_Bad_OG", "RT_WZ", "RT_Flur_EG", "PV_Generate"]
missing = df[[c for c in key_cols if c in df.columns]].isna().mean() * 100
print("\\nMissing % for key columns (after ffill):")
print(missing.round(2).to_string())\
"""

CORE_FEATURES = """\
# ── Core derived features ─────────────────────────────────────────────────────
# NOTE: All features below must be computable from inputs available in the
# HA add-on at prediction time. Comments marked "# ADDON: ..." show the exact
# production feature key / formula used in src/physics_features.py.

df_feat = df_valid.copy()

# Distance to overheating threshold
df_feat["indoor_margin"] = df_feat["indoor_temp"] - OVERHEAT_THRESHOLD_C

# 30-min indoor trend  (6 × 5-min steps)
# ADDON: indoor_temp - features["indoor_temp_lag_30m"]
df_feat["indoor_trend_30m"] = df_feat["indoor_temp"] - df_feat["indoor_temp"].shift(6)

# 1h indoor trend  (12 × 5-min steps)
# ADDON: features["indoor_temp_delta_60m"]
df_feat["indoor_trend_1h"] = df_feat["indoor_temp"] - df_feat["indoor_temp"].shift(12)

# Outdoor – indoor differential
# ADDON: features["temp_diff_indoor_outdoor"]
df_feat["at_delta_indoor"] = df_feat["AT"] - df_feat["indoor_temp"]

# PV rolling means  (1h and 2h)
# ADDON: derived from features["pv_power_history"] (rolling window in physics_features.py)
df_feat["pv_roll_1h"] = df_feat["PV_Generate"].rolling(12, min_periods=6).mean()
df_feat["pv_roll_2h"] = df_feat["PV_Generate"].rolling(24, min_periods=12).mean()

# ── HP thermal state features ─────────────────────────────────────────────────
# Passive cooling power from NIBE system.
# Pth_PC sign: NEGATIVE when cooling is active (heat removed from house), ≈0 when idle.
# Verify on first run: print(df_feat["Pth_PC"].describe())
# ADDON: features["thermal_power_kw"] = (Vol_L_min / 60) * c_p * (VLT - RLT)
#         automatically negative during passive cooling because RLT > VLT then.
if "Pth_PC" in df_feat.columns:
    df_feat["thermal_power_kw"] = df_feat["Pth_PC"] / 1000.0
else:
    df_feat["thermal_power_kw"] = 0.0
    print("  ⚠ Pth_PC not found — thermal_power_kw set to 0")

# Supply – return temperature delta (negative during passive cooling since RLT > VLT)
# ADDON: features["delta_t"] = outlet_temp - inlet_temp
if "VLT" in df_feat.columns and "RLT" in df_feat.columns:
    df_feat["delta_t"] = df_feat["VLT"] - df_feat["RLT"]
else:
    df_feat["delta_t"] = 0.0
    print("  ⚠ VLT/RLT not found — delta_t set to 0")

# Driving force: how much warmer the floor loop supply is than the room
# ADDON: features["outlet_indoor_diff"] = outlet_temp - indoor_temp
if "VLT" in df_feat.columns:
    df_feat["outlet_indoor_diff"] = df_feat["VLT"] - df_feat["indoor_temp"]
else:
    df_feat["outlet_indoor_diff"] = 0.0

# Cyclical time-of-day encoding
hour_frac = df_feat.index.hour + df_feat.index.minute / 60.0
df_feat["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
df_feat["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)

# Cyclical day-of-year encoding
doy = df_feat.index.dayofyear
df_feat["doy_sin"] = np.sin(2 * np.pi * doy / 365)
df_feat["doy_cos"] = np.cos(2 * np.pi * doy / 365)

print("Feature engineering done")

# FUTURE — incremental production labels:
# Call sgd_online.partial_fit() once per day after generating delayed labels:
#   label_t = int(max(indoor_temp[t : t+96_steps]) > OVERHEAT_THRESHOLD_C)
# The add-on can log predictions + timestamps to InfluxDB and retrieve
# outcomes the next morning for daily model updates.\
"""

FEATURE_COLS_CELL = """\
# ── Outdoor forecast columns  (AT_roh_1h … AT_roh_8h) ────────────────────────
# ADDON: features["temp_forecast_Nh"] from live weather API — no ffill needed.
forecast_cols_at = [f"AT_roh_{h}h" for h in range(1, 9) if f"AT_roh_{h}h" in df_feat.columns]
print(f"Outdoor forecast columns found: {forecast_cols_at}")

# ── Feature list — only inputs available in the HA add-on ────────────────────
# Removed: Helligkeit      — no brightness sensor in the add-on
# Removed: Kuehlung_Soll   — config constant (23.1 °C), not a live sensor reading
# Removed: Pth_H           — heating power ≈ 0 W on summer cooling days (zero signal)
FEATURE_COLS = (
    # ── Indoor state ─────────────────────────────────────────────────────────
    ["indoor_temp",           # ADDON: computed from sensor.rt_mittelwert
     "indoor_margin",         # ADDON: indoor_temp - OVERHEAT_THRESHOLD_C
     "indoor_trend_30m",      # ADDON: indoor_temp - features["indoor_temp_lag_30m"]
     "indoor_trend_1h"]       # ADDON: features["indoor_temp_delta_60m"]
    # ── Outdoor ──────────────────────────────────────────────────────────────
    + ["AT",                  # ADDON: features["outdoor_temp"]
       "at_delta_indoor"]     # ADDON: features["temp_diff_indoor_outdoor"]
    # ── Outdoor temperature forecasts (1–8 h) ────────────────────────────────
    + forecast_cols_at        # ADDON: features["temp_forecast_Nh"]
    # ── Solar / PV ───────────────────────────────────────────────────────────
    + ["PV_Generate",         # ADDON: features["pv_now"]
       "pv_roll_1h",          # ADDON: rolling mean of pv_power_history (1 h)
       "pv_roll_2h"]          # ADDON: rolling mean of pv_power_history (2 h)
    # ── HP thermal state ─────────────────────────────────────────────────────
    + (["thermal_power_kw"]   if "thermal_power_kw"   in df_feat.columns else [])
                              # ADDON: features["thermal_power_kw"] = (Vol/60)*c_p*(VLT-RLT)
    + (["delta_t"]            if "delta_t"            in df_feat.columns else [])
                              # ADDON: features["delta_t"] = VLT - RLT
    + (["outlet_indoor_diff"] if "outlet_indoor_diff"  in df_feat.columns else [])
                              # ADDON: features["outlet_indoor_diff"] = VLT - indoor_temp
    + (["VLT"]                if "VLT"                in df_feat.columns else [])
                              # ADDON: features["outlet_temp"]
    + (["RLT"]                if "RLT"                in df_feat.columns else [])
                              # ADDON: features["inlet_temp"]
    # ── Cyclical time ────────────────────────────────────────────────────────
    + ["hour_sin", "hour_cos",   # ADDON: computed from current timestamp
       "doy_sin",  "doy_cos"]    # ADDON: computed from current timestamp
)

print(f"\\nTotal features: {len(FEATURE_COLS)}")
for f in FEATURE_COLS:
    print(f"  {f}")\
"""

LABEL_TAIL_CELL = """\
def make_label(series: pd.Series, horizon_steps: int, threshold: float) -> pd.Series:
    \"\"\"Return 1 if max(series[t : t+horizon_steps]) > threshold.

    Reverse-rolling trick: reversing turns pandas look-back into
    look-forward. Tail rows (< horizon_steps) become NaN.
    \"\"\"
    # Reverse-rolling: gives max(series[t:t+h]) without shift
    return (
        series[::-1]
        .rolling(horizon_steps, min_periods=horizon_steps)
        .max()[::-1] > threshold
    ).astype("Int8")  # nullable Int8: 0/1 with NaN for tail rows


df_feat["label_8h"]  = make_label(df_feat["indoor_temp"], LABEL_HORIZON_STEPS, OVERHEAT_THRESHOLD_C)
for h in EXTRA_HORIZONS_H:
    df_feat[f"label_{h}h"] = make_label(df_feat["indoor_temp"], h * STEPS_PER_HOUR, OVERHEAT_THRESHOLD_C)

# Drop rows where the 8h label cannot be computed (end of each contiguous day block).
# NOTE: iloc[:-N] trims exactly N rows from the physical tail but under-trims
# when day-gap boundaries exist inside df_feat.  notna() filter is exact.
# TRAINING-ONLY: labels don't exist in production; this block never runs live.
df_feat = df_feat[df_feat["label_8h"].notna()]

print("Label class balance:")
for col in ["label_8h"] + [f"label_{h}h" for h in EXTRA_HORIZONS_H]:
    n_pos = (df_feat[col] == 1).sum()
    n_tot = df_feat[col].notna().sum()
    print(f"  {col:12s}: {n_pos:5,} positive  /  {n_tot:6,} total  ({n_pos/n_tot*100:.1f}%)")\
"""

SPLIT_CELL = """\
# ── NaN diagnostic — identify which column drives row loss ────────────────────
pre_drop_nan = df_feat[FEATURE_COLS + ["label_8h"]].isna().mean() * 100
problematic  = pre_drop_nan[pre_drop_nan > 1.0]
if len(problematic):
    print("Columns with > 1% NaN (will reduce model_df size):")
    print(problematic.round(1).to_string())
    print()
else:
    print("All feature columns have ≤ 1% NaN ✓")

# ── Keep only rows with all features and label available ──────────────────────
model_df = df_feat[FEATURE_COLS + ["label_8h"]].dropna()
print(f"\\nRows with complete features + label: {len(model_df):,}")

assert len(model_df) >= 1000, (
    f"Only {len(model_df)} complete rows — too few to train.  "
    "Check NaN diagnostic above; usually AT_roh_Nh columns need the ffill block in Section 2.")

X = model_df[FEATURE_COLS].values
y = model_df["label_8h"].astype(int).values

# ── Time-ordered 75 / 25 split (no shuffle!) ─────────────────────────────────
split_idx = int(len(model_df) * TRAIN_FRACTION)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

assert 0 < split_idx < len(model_df), (
    f"Bad split_idx={split_idx}. Increase dataset size or adjust TRAIN_FRACTION.")

train_start = model_df.index[0]
train_end   = model_df.index[split_idx - 1]
test_start  = model_df.index[split_idx]
test_end    = model_df.index[-1]

print(f"\\nTraining  : {train_start.date()} → {train_end.date()}  ({len(X_train):,} rows)")
print(f"Test      : {test_start.date()} → {test_end.date()}  ({len(X_test):,} rows)")
print(f"Train pos : {y_train.mean()*100:.1f}%   |   Test pos: {y_test.mean()*100:.1f}%")\
"""

METADATA_EXPORT = """\
# ── Save models ───────────────────────────────────────────────────────────────
rf_path   = os.path.join(MODELS_DIR, "overheating_predictor_rf.joblib")
lgbm_path = os.path.join(MODELS_DIR, "overheating_predictor_lgbm.joblib")

joblib.dump(rf,         rf_path)
joblib.dump(lgbm_model, lgbm_path)
print(f"Saved: {rf_path}")
print(f"Saved: {lgbm_path}")

# ── Save metadata ─────────────────────────────────────────────────────────────
best_model_name = "rf" if rf_auc >= lgbm_auc else "lgbm"

metadata = {
    "created_at"            : datetime.utcnow().isoformat() + "Z",
    "overheat_threshold_c"  : OVERHEAT_THRESHOLD_C,
    "label_horizon_h"       : LABEL_HORIZON_H,
    "train_date_range"      : [str(train_start.date()), str(train_end.date())],
    "test_date_range"       : [str(test_start.date()), str(test_end.date())],
    "feature_cols"          : FEATURE_COLS,
    "best_model"            : best_model_name,
    "rf": {
        "roc_auc"      : round(rf_auc, 5),
        "avg_precision": round(rf_ap, 5),
        "threshold"    : round(rf_threshold, 5),
        "path"         : rf_path,
    },
    "lgbm": {
        "model_type"   : boost_name,
        "roc_auc"      : round(lgbm_auc, 5),
        "avg_precision": round(lgbm_ap, 5),
        "threshold"    : round(lgbm_threshold, 5),
        "path"         : lgbm_path,
    },
    "feature_mapping_for_live_system": {
        # ── Indoor state ─────────────────────────────────────────────────────
        "indoor_temp"        : "computed from sensor.rt_mittelwert (median of room sensors)",
        "indoor_margin"      : "indoor_temp - OVERHEAT_THRESHOLD_C",
        "indoor_trend_30m"   : "indoor_temp - features['indoor_temp_lag_30m']",
        "indoor_trend_1h"    : "features['indoor_temp_delta_60m']",
        # ── Outdoor ──────────────────────────────────────────────────────────
        "AT"                 : "features['outdoor_temp']",
        "at_delta_indoor"    : "features['temp_diff_indoor_outdoor']",
        # ── Outdoor forecasts ─────────────────────────────────────────────────
        "AT_roh_1h"          : "features['temp_forecast_1h']  (hourly weather API)",
        "AT_roh_2h"          : "features['temp_forecast_2h']",
        "AT_roh_3h"          : "features['temp_forecast_3h']",
        "AT_roh_4h"          : "features['temp_forecast_4h']",
        "AT_roh_5h"          : "features['temp_forecast_5h']",
        "AT_roh_6h"          : "features['temp_forecast_6h']",
        "AT_roh_7h"          : "features['temp_forecast_7h']",
        "AT_roh_8h"          : "features['temp_forecast_8h']",
        # ── Solar / PV ───────────────────────────────────────────────────────
        "PV_Generate"        : "features['pv_now']",
        "pv_roll_1h"         : "rolling mean of pv_power_history over last 1 h",
        "pv_roll_2h"         : "rolling mean of pv_power_history over last 2 h",
        # ── HP thermal state ─────────────────────────────────────────────────
        "thermal_power_kw"   : "features['thermal_power_kw']  = (Vol/60)*c_p*(VLT-RLT); "
                               "NEGATIVE when cooling, ~0 when idle",
        "delta_t"            : "features['delta_t']  = VLT - RLT; "
                               "negative during passive cooling (RLT > VLT)",
        "outlet_indoor_diff" : "features['outlet_indoor_diff']  = VLT - indoor_temp",
        "VLT"                : "features['outlet_temp']  (Vorlauftemperatur / supply)",
        "RLT"                : "features['inlet_temp']   (Rücklauftemperatur / return)",
        # ── Cyclical time ────────────────────────────────────────────────────
        "hour_sin"           : "sin(2*pi*hour/24)  — computed from current timestamp",
        "hour_cos"           : "cos(2*pi*hour/24)",
        "doy_sin"            : "sin(2*pi*doy/365)  — computed from current timestamp",
        "doy_cos"            : "cos(2*pi*doy/365)",
    },
}

meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
with open(meta_path, "w") as fh:
    json.dump(metadata, fh, indent=2)
print(f"Saved: {meta_path}")

print("\\n── Model Card ──────────────────────────────────────────────")
print(f"  Overheat threshold : {OVERHEAT_THRESHOLD_C} °C")
print(f"  Prediction horizon : {LABEL_HORIZON_H} h")
print(f"  Training rows      : {len(X_train):,}")
print(f"  Test rows          : {len(X_test):,}")
print(f"  RandomForest       : ROC-AUC={rf_auc:.4f}  threshold={rf_threshold:.3f}")
print(f"  {boost_name:16s}: ROC-AUC={lgbm_auc:.4f}  threshold={lgbm_threshold:.3f}")
print(f"  Best model         : {best_model_name.upper()}")
print(f"  Features ({len(FEATURE_COLS)})      : {', '.join(FEATURE_COLS)}")\
"""

SEC11_MD = """\
## 11. Online Learning — SGDClassifier with `partial_fit()`

`SGDClassifier(loss="log_loss")` supports **incremental updates** via `partial_fit()`.
It can be retrained in-place on each day's new data without reloading the full history,
making it suitable for the Pi 4 add-on.

**Production update loop (suggested daily cron):**
1. Retrieve yesterday's prediction timestamps from InfluxDB.
2. Generate labels: `label_t = int(max(indoor_temp[t : t+8h]) > 23.1 °C)`.
3. Call `sgd.partial_fit(X_new, y_new, classes=[0, 1])`.
4. Persist with `joblib.dump(sgd, "overheating_predictor_sgd_online.joblib")`.

**Upgrade path:** `river` library — `ARFClassifier` for true concept-drift-adaptive stream
learning (no batch retraining at all). Install with `pip install river`.\
"""

SEC11_TRAIN = """\
from sklearn.linear_model import SGDClassifier

# ── 11a: Initial offline training (same train set as RF / LightGBM) ──────────
print("Training SGDClassifier (log-loss = logistic regression with SGD)…")
sgd = SGDClassifier(
    loss="log_loss",
    class_weight="balanced",
    max_iter=1000,
    tol=1e-4,
    random_state=42,
    n_jobs=-1,
)
sgd.fit(X_train, y_train)

sgd_prob_test = sgd.predict_proba(X_test)[:, 1]
sgd_auc = roc_auc_score(y_test, sgd_prob_test)
sgd_ap  = average_precision_score(y_test, sgd_prob_test)

print(f"  SGD (initial)  — ROC-AUC: {sgd_auc:.4f}   Avg Precision: {sgd_ap:.4f}")
print(f"  RandomForest   — ROC-AUC: {rf_auc:.4f}   Avg Precision: {rf_ap:.4f}")
print(f"  {boost_name:14s} — ROC-AUC: {lgbm_auc:.4f}   Avg Precision: {lgbm_ap:.4f}")
print()
print("Note: SGD is a linear model — lower initial AUC than tree models is expected.")
print("Its advantage is O(1) memory incremental updates via partial_fit().")\
"""

SEC11_SIM = """\
# ── 11b: Simulate production — weekly partial_fit() updates ──────────────────
# Walk forward over the test set in 1-week chunks.
# Each iteration: evaluate first, then update with that week's ground truth.
STEPS_PER_WEEK = STEPS_PER_HOUR * 24 * 7

# Clone fresh starting point so we don't contaminate the static sgd above
sgd_online = SGDClassifier(
    loss="log_loss",
    class_weight="balanced",
    max_iter=1000,
    tol=1e-4,
    random_state=42,
    n_jobs=-1,
)
sgd_online.fit(X_train, y_train)   # warm-start on training data

weekly_aucs   = []
n_test        = len(X_test)
chunk_start   = 0

while chunk_start < n_test:
    chunk_end = min(chunk_start + STEPS_PER_WEEK, n_test)
    X_chunk   = X_test[chunk_start:chunk_end]
    y_chunk   = y_test[chunk_start:chunk_end]

    # Evaluate on this week *before* seeing the labels
    if len(np.unique(y_chunk)) > 1:        # need both classes for AUC
        prob_chunk = sgd_online.predict_proba(X_chunk)[:, 1]
        weekly_aucs.append(roc_auc_score(y_chunk, prob_chunk))

    # Incremental update with this week's ground-truth labels
    sgd_online.partial_fit(X_chunk, y_chunk, classes=[0, 1])
    chunk_start = chunk_end

print(f"Production simulation: {len(weekly_aucs)} evaluation weeks")
print(f"AUC range : {min(weekly_aucs):.4f} – {max(weekly_aucs):.4f}")
print(f"Mean AUC  : {np.mean(weekly_aucs):.4f} ± {np.std(weekly_aucs):.4f}")\
"""

SEC11_PLOT = """\
# ── 11c: Plot AUC progression ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

weeks = range(1, len(weekly_aucs) + 1)
ax.plot(weeks, weekly_aucs, marker="o", linewidth=1.8, color="darkorchid",
        label="SGD online (after weekly partial_fit)")
ax.axhline(rf_auc,   linestyle="--", color="steelblue", alpha=0.7,
           label=f"RF static baseline  (AUC={rf_auc:.3f})")
ax.axhline(lgbm_auc, linestyle="--", color="seagreen",  alpha=0.7,
           label=f"{boost_name} static baseline  (AUC={lgbm_auc:.3f})")
ax.axhline(sgd_auc,  linestyle=":",  color="grey",      alpha=0.7,
           label=f"SGD initial — no updates  (AUC={sgd_auc:.3f})")

ax.set_xlabel("Production week (test-set chronological order)")
ax.set_ylabel("ROC-AUC on that week's data")
ax.set_title("Online learning: SGD AUC progression with weekly partial_fit()")
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()\
"""

SEC11_EXPORT = """\
# ── 11d: Export online model + update metadata ────────────────────────────────
sgd_threshold = select_threshold(y_test, sgd_online.predict_proba(X_test)[:, 1])
sgd_path      = os.path.join(MODELS_DIR, "overheating_predictor_sgd_online.joblib")
joblib.dump(sgd_online, sgd_path)
print(f"Saved: {sgd_path}")

meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
with open(meta_path) as fh:
    metadata = json.load(fh)

metadata["sgd_online"] = {
    "model_type"      : "SGDClassifier(loss=log_loss) — incremental / online",
    "roc_auc_initial" : round(float(sgd_auc), 5),
    "roc_auc_final"   : round(float(np.mean(weekly_aucs[-3:])), 5),
    "threshold"       : round(float(sgd_threshold), 5),
    "path"            : sgd_path,
    "production_note" : (
        "Call partial_fit(X_new, y_new, classes=[0,1]) once per day. "
        "Generate label 8 h after prediction: "
        "int(max(indoor_temp[t : t+96_steps]) > OVERHEAT_THRESHOLD_C). "
        "Upgrade path: river.ensemble.ARFClassifier for concept-drift adaptation."
    ),
}
with open(meta_path, "w") as fh:
    json.dump(metadata, fh, indent=2)
print(f"Updated: {meta_path}")

print(f"\\nSGD online model summary:")
print(f"  Initial AUC (static eval)  : {sgd_auc:.4f}")
print(f"  Final AUC  (last 3 weeks)  : {np.mean(weekly_aucs[-3:]):.4f}")
print(f"  Threshold                  : {sgd_threshold:.3f}")\
"""

# ═══════════════════════════════════════════════════════════════════════════════
# NB03 APPLY
# ═══════════════════════════════════════════════════════════════════════════════

def apply_nb03(nb: dict) -> int:
    fixes = 0
    cells = nb["cells"]

    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        src_str = "".join(cell["source"])

        # ── 1. Resample/merge cell → add ffill block ─────────────────────────
        if ('sensor_5m = sensor_raw.resample("5min")' in src_str
                and "ffill" not in src_str):
            cell["source"] = to_src(RESAMPLE_FFILL)
            fixes += 1
            print("  [NB03-1] Resample cell: ffill block added")

        # ── 2. Core feature engineering cell → add HP thermal features ───────
        elif ("df_feat[\"indoor_margin\"]" in src_str
              and "thermal_power_kw" not in src_str):
            cell["source"] = to_src(CORE_FEATURES)
            fixes += 1
            print("  [NB03-2] Core features cell: thermal_power_kw/delta_t/outlet_indoor_diff added")

        # ── 3. FEATURE_COLS cell → remove Helligkeit/Kuehlung_Soll, add new ──
        elif ("FEATURE_COLS = (" in src_str
              and ("Kuehlung_Soll" in src_str or "Helligkeit" in src_str)):
            cell["source"] = to_src(FEATURE_COLS_CELL)
            fixes += 1
            print("  [NB03-3] FEATURE_COLS cell: Helligkeit/Kuehlung_Soll removed; new features + ADDON comments")

        # ── 4. Label + tail removal → replace iloc with notna ────────────────
        elif ("iloc[:-LABEL_HORIZON_STEPS]" in src_str):
            cell["source"] = to_src(LABEL_TAIL_CELL)
            fixes += 1
            print("  [NB03-4] Label cell: iloc tail removed → notna() filter")

        # ── 5. Section 6 split cell → add NaN diagnostic + assert guard ──────
        elif ("model_df = df_feat[FEATURE_COLS" in src_str
              and "assert len(model_df)" not in src_str):
            cell["source"] = to_src(SPLIT_CELL)
            fixes += 1
            print("  [NB03-5] Split cell: NaN diagnostic + assert guard added")

        # ── 6. Metadata export → update feature_mapping ──────────────────────
        elif ("feature_mapping_for_live_system" in src_str
              and "Helligkeit" in src_str):
            cell["source"] = to_src(METADATA_EXPORT)
            fixes += 1
            print("  [NB03-6] Metadata cell: feature_mapping updated")

    # ── 7. Add imports for SGDClassifier in the imports cell ─────────────────
    for cell in cells:
        src_str = "".join(cell["source"])
        if ("from sklearn.ensemble import RandomForestClassifier" in src_str
                and "SGDClassifier" not in src_str):
            # Insert SGDClassifier into the sklearn imports line
            new_src = []
            for line in cell["source"]:
                new_src.append(line)
                if "from sklearn.ensemble import RandomForestClassifier" in line:
                    new_src.append("from sklearn.linear_model import SGDClassifier\n")
            cell["source"] = new_src
            fixes += 1
            print("  [NB03-7] Imports cell: SGDClassifier added")
            break

    # ── 8. Append Section 11 after last existing cell ────────────────────────
    last_src = "".join(nb["cells"][-1]["source"])
    if "Section 11" not in "".join(
            "".join(c["source"]) for c in nb["cells"]
    ):
        nb["cells"].extend([
            md_cell(SEC11_MD, "sec11-md"),
            code_cell(SEC11_TRAIN, "sec11-train"),
            code_cell(SEC11_SIM,   "sec11-sim"),
            code_cell(SEC11_PLOT,  "sec11-plot"),
            code_cell(SEC11_EXPORT,"sec11-export"),
        ])
        fixes += 1
        print("  [NB03-8] Section 11 (online learning) added (5 cells)")

    return fixes


# ═══════════════════════════════════════════════════════════════════════════════
# NB04 APPLY — replace Helligkeit slider with new HP thermal state sliders
# ═══════════════════════════════════════════════════════════════════════════════

def apply_nb04(nb: dict) -> int:
    fixes = 0
    for cell in nb["cells"]:
        src_str = "".join(cell["source"])
        if "sl_hell" not in src_str or "Helligkeit" not in src_str:
            continue

        new_src = []
        for line in cell["source"]:
            # Remove the Helligkeit slider definition
            if "sl_hell" in line and "Helligkeit" in line and "make_slider" in line:
                # Replace with three new sliders
                new_src.append(
                    '    sl_thermal_pw  = make_slider("thermal_power_kw",   "Thermal power (kW)",    0.05)'
                    ' if "thermal_power_kw"   in FEATURE_COLS else None\n'
                )
                new_src.append(
                    '    sl_delta_t     = make_slider("delta_t",             "Supply-Return ΔT (K)",  0.1)'
                    '  if "delta_t"            in FEATURE_COLS else None\n'
                )
                new_src.append(
                    '    sl_outlet_diff = make_slider("outlet_indoor_diff",  "Outlet-Indoor diff (K)",0.1)'
                    '  if "outlet_indoor_diff"  in FEATURE_COLS else None\n'
                )
                continue

            # Update the all_sliders list to replace sl_hell
            if "sl_hell," in line:
                line = line.replace("sl_hell,", "sl_thermal_pw, sl_delta_t, sl_outlet_diff,")
            elif "sl_hell]" in line:
                line = line.replace("sl_hell]", "sl_thermal_pw, sl_delta_t, sl_outlet_diff]")

            # Update feats dict in on_predict_click: replace sl_hell block
            if 'if sl_hell:' in line and 'Helligkeit' in line:
                new_src.append('        if sl_thermal_pw:  feats["thermal_power_kw"]   = sl_thermal_pw.value\n')
                new_src.append('        if sl_delta_t:     feats["delta_t"]             = sl_delta_t.value\n')
                new_src.append('        if sl_outlet_diff: feats["outlet_indoor_diff"]  = sl_outlet_diff.value\n')
                continue

            new_src.append(line)

        cell["source"] = new_src
        fixes += 1
        print("  [NB04-1] Helligkeit slider → thermal_power_kw / delta_t / outlet_indoor_diff")
    return fixes


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Loading {NB03.name} …")
    with open(NB03, encoding="utf-8") as f:
        nb03 = json.load(f)
    n = apply_nb03(nb03)
    with open(NB03, "w", encoding="utf-8") as f:
        json.dump(nb03, f, indent=1, ensure_ascii=False)
    print(f"  → {n} change(s) applied, saved.\n")

    print(f"Loading {NB04.name} …")
    with open(NB04, encoding="utf-8") as f:
        nb04 = json.load(f)
    n = apply_nb04(nb04)
    with open(NB04, "w", encoding="utf-8") as f:
        json.dump(nb04, f, indent=1, ensure_ascii=False)
    print(f"  → {n} change(s) applied, saved.\n")

    print("All done.")
