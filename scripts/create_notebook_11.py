"""Generate notebook 11: Cooling Cycle Analysis from Production Logs."""
import json
import os

def md(source: str) -> dict:
    lines = source.split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]
    }

def code(source: str) -> dict:
    lines = source.split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]
    }

cells = []

# ============================================================
# CELL 1: Title
# ============================================================
cells.append(md("""# 11 — Cooling Cycle Analysis (Production Logs)

**Purpose:** Parse 3 production log files cycle-by-cycle. Compare ML-based (LGBM)
vs trajectory-based (shadow) pre-cooling precision. Analyse cooling effectiveness,
binary search convergence, cycle gate behavior, and parameter evolution.

**Logs analyzed:**
1. `2026-05-31T15-43-28` — Recovery phase, 7 cycles
2. `2026-06-01T10-19-29` — Active cooling, higher demand
3. `2026-06-01T19-58-36` — Dynamic control, τ explosion visible

**Key Question:** Is trajectory-based pre-cooling more reliable and easier to handle
than the LGBM classifier? The LGBM probability appears near-constant (~0.42)."""))

# ============================================================
# CELL 2: Imports
# ============================================================
cells.append(code("""import sys, os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (16, 6)
plt.rcParams["figure.dpi"] = 100

PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))
LOG_DIR = PROJECT_ROOT / "Logs_and_models"

LOG_FILES = sorted(LOG_DIR.glob("74f4b7ef_ml_heating_underfloor_2026-0*.log"))
print(f"Found {len(LOG_FILES)} log files:")
for f in LOG_FILES:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name} ({size_kb:.0f} KB)")"""))

# ============================================================
# CELL 3: Log parser
# ============================================================
cells.append(md("""## Phase A: Log Parsing

Extract structured cycle data from raw log text using regex patterns."""))

# ============================================================
# CELL 4: Parser code (uses variable to avoid triple-quote conflicts)
# ============================================================
_parser_code = (
'def parse_cooling_cycles(log_path):\n'
'    """Parse a single log file and extract cooling cycle data."""\n'
'    with open(log_path, "r", encoding="utf-8", errors="replace") as f:\n'
'        text = f.read()\n'
'    \n'
'    cycles = []\n'
'    current = {}\n'
'    \n'
'    for line in text.split("\\n"):\n'
'        # Cycle start\n'
'        m = re.search(r"CYCLE (\\d+) START: (\\d{2}:\\d{2}:\\d{2})", line)\n'
'        if m:\n'
'            if current.get("cycle_num"):\n'
'                cycles.append(current)\n'
'            current = {"cycle_num": int(m.group(1)), "start_time": m.group(2)}\n'
'            continue\n'
'        \n'
'        # Cooling mode detection\n'
'        if "Cooling mode active" in line or "COOLING MODE" in line:\n'
'            current["is_cooling"] = True\n'
'        \n'
'        # Cooling target\n'
'        m = re.search(r"cooling target entity: ([\\d.]+)", line)\n'
'        if m:\n'
'            current["target_temp"] = float(m.group(1))\n'
'        \n'
'        # LGBM pre-cool\n'
'        m = re.search(r"LGBM PRE-COOL (\\w+).*?p=([\\d.]+).*?thr=([\\d.]+).*?max=([\\d.]+)", line)\n'
'        if m:\n'
'            current["lgbm_activated"] = m.group(1) == "ACTIVATED"\n'
'            current["lgbm_prob"] = float(m.group(2))\n'
'            current["lgbm_threshold"] = float(m.group(3))\n'
'            current["lgbm_peak_temp"] = float(m.group(4))\n'
'        \n'
'        # Trajectory (shadow)\n'
'        m = re.search(r"SHADOW.*?risk=(\\w+)\\s+peak=([\\d.]+).*?in ([\\d.]+)h", line)\n'
'        if m:\n'
'            current["traj_risk"] = m.group(1) == "True"\n'
'            current["traj_peak"] = float(m.group(2))\n'
'            current["traj_peak_hours"] = float(m.group(3))\n'
'        \n'
'        # PRE-COOL activation (trajectory)\n'
'        m = re.search(r"PRE-COOL (\\w+):.*?room ([\\d.]+).*?target ([\\d.]+).*?peak_outdoor=([\\d.]+)", line)\n'
'        if m and "LGBM" not in line:\n'
'            current["precool_activated"] = m.group(1) == "ACTIVATED"\n'
'            current["room_temp"] = float(m.group(2))\n'
'            current["precool_target"] = float(m.group(3))\n'
'        \n'
'        # Binary search\n'
'        m = re.search(r"Binary search start: target=([\\d.]+).*?current=([\\d.]+).*?range=([\\d.]+)-([\\d.]+)", line)\n'
'        if m:\n'
'            current["bs_target"] = float(m.group(1))\n'
'            current["bs_current"] = float(m.group(2))\n'
'            current["bs_range_min"] = float(m.group(3))\n'
'            current["bs_range_max"] = float(m.group(4))\n'
'        \n'
'        m = re.search(r"Binary search converged after (\\d+) iterations: ([\\d.]+).*?predicted.*?([\\d.]+).*?error: ([+-]?[\\d.]+)", line)\n'
'        if m:\n'
'            current["bs_iterations"] = int(m.group(1))\n'
'            current["bs_outlet"] = float(m.group(2))\n'
'            current["bs_predicted"] = float(m.group(3))\n'
'            current["bs_error"] = float(m.group(4))\n'
'        \n'
'        # Newton correction\n'
'        m = re.search(r"\\[Newton\\].*?outlet ([\\d.]+)", line)\n'
'        if m:\n'
'            current["newton_outlet"] = float(m.group(1))\n'
'        m = re.search(r"\\[Newton\\].*?T=([+-]?[\\d.]+)", line)\n'
'        if m:\n'
'            current["newton_delta"] = float(m.group(1))\n'
'        \n'
'        # Cooling cycle gate\n'
'        m = re.search(r"Cooling cycle gate: (\\w+)", line)\n'
'        if m:\n'
'            current["gate_state"] = m.group(1)\n'
'        \n'
'        # Equilibrium physics\n'
'        m = re.search(r"Equilibrium physics:.*?heat_loss_coeff=([\\d.]+).*?outlet_eff=([\\d.]+).*?equilibrium=([\\d.]+)", line)\n'
'        if m:\n'
'            current["eq_hlc"] = float(m.group(1))\n'
'            current["eq_oe"] = float(m.group(2))\n'
'            current["eq_temp"] = float(m.group(3))\n'
'        \n'
'        # Model wrapper params\n'
'        m = re.search(r"Wrapper params: U=([\\d.]+), eff=([\\d.]+), tau=([\\d.]+)", line)\n'
'        if m:\n'
'            current["param_hlc"] = float(m.group(1))\n'
'            current["param_oe"] = float(m.group(2))\n'
'            current["param_tau"] = float(m.group(3))\n'
'        \n'
'        # Final outlet (from HA sensor)\n'
'        m = re.search(r"sensor\\.ml_vorlauftemperatur.*?\'state\': \'([\\d.]+)\'", line)\n'
'        if m:\n'
'            current["final_outlet"] = float(m.group(1))\n'
'        \n'
'        # Prediction output\n'
'        m = re.search(r"Prediction:.*?Current ([\\d.]+).*?Target ([\\d.]+).*?outlet: ([\\d.]+)", line)\n'
'        if m:\n'
'            current["pred_current"] = float(m.group(1))\n'
'            current["pred_target"] = float(m.group(2))\n'
'            current["pred_outlet"] = float(m.group(3))\n'
'        \n'
'        # Prediction feedback\n'
'        m = re.search(r"Prediction feedback: error=([\\d.]+).*?confidence=([\\d.]+)", line)\n'
'        if m:\n'
'            current["feedback_error"] = float(m.group(1))\n'
'            current["feedback_confidence"] = float(m.group(2))\n'
'        \n'
'        # Added prediction (previous cycle validation)\n'
'        m = re.search(r"Added prediction: pred=([\\d.]+), actual=([\\d.]+), error=([\\d.]+)", line)\n'
'        if m:\n'
'            current["prev_predicted"] = float(m.group(1))\n'
'            current["prev_actual"] = float(m.group(2))\n'
'            current["prev_error"] = float(m.group(3))\n'
'        \n'
'        # Channel learning\n'
'        m = re.search(r"heat_pump parameter update.*?delta_t_floor: [\\d.]+.([\\d.]+).*?heat_loss_coefficient: [\\d.]+.([\\d.]+).*?outlet_effectiveness: [\\d.]+.([\\d.]+).*?thermal_time_constant: [\\d.]+.([\\d.]+)", line)\n'
'        if m:\n'
'            current["learn_dtf"] = float(m.group(1))\n'
'            current["learn_hlc"] = float(m.group(2))\n'
'            current["learn_oe"] = float(m.group(3))\n'
'            current["learn_tau"] = float(m.group(4))\n'
'        \n'
'        # MAE/RMSE\n'
'        m = re.search(r"sensor\\.ml_model_mae.*?\'state\': \'([\\d.]+)\'", line)\n'
'        if m:\n'
'            current["mae"] = float(m.group(1))\n'
'        m = re.search(r"sensor\\.ml_model_rmse.*?\'state\': \'([\\d.]+)\'", line)\n'
'        if m:\n'
'            current["rmse"] = float(m.group(1))\n'
'        \n'
'        # Outdoor temp from features\n'
'        m = re.search(r"\'outdoor_temp\': ([\\d.]+)", line)\n'
'        if m and "sensor.ml_heating_features" in line:\n'
'            current["feat_outdoor"] = float(m.group(1))\n'
'        \n'
'        # Cycle end\n'
'        m = re.search(r"CYCLE \\d+ END.*?duration: ([\\d.]+)s", line)\n'
'        if m:\n'
'            current["duration_s"] = float(m.group(1))\n'
'    \n'
'    # Don\'t forget last cycle\n'
'    if current.get("cycle_num"):\n'
'        cycles.append(current)\n'
'    \n'
'    return [c for c in cycles if c.get("is_cooling")]\n'
'\n'
'# Parse all log files\n'
'all_cycles = []\n'
'for log_path in LOG_FILES:\n'
'    cycles = parse_cooling_cycles(log_path)\n'
'    for c in cycles:\n'
'        c["log_file"] = log_path.name\n'
'    all_cycles.extend(cycles)\n'
'    print(f"{log_path.name}: {len(cycles)} cooling cycles")\n'
'\n'
'df_cycles = pd.DataFrame(all_cycles)\n'
'print(f"\\nTotal cooling cycles: {len(df_cycles)}")\n'
'print(f"Columns: {list(df_cycles.columns)}")\n'
'avail = [c for c in ["cycle_num", "log_file", "gate_state", "bs_iterations",\n'
'         "final_outlet", "lgbm_prob", "traj_risk", "feedback_error"] if c in df_cycles.columns]\n'
'df_cycles[avail].head(20)'
)

cells.append(code(_parser_code))

# ============================================================
# CELL 5: Cycle dashboard
# ============================================================
cells.append(md("""## Phase B: Cycle-by-Cycle Dashboard"""))

# ============================================================
# CELL 6: Dashboard plots
# ============================================================
cells.append(code("""fig, axes = plt.subplots(3, 2, figsize=(18, 14))
x = range(len(df_cycles))
labels = [f"C{int(r['cycle_num'])}" for _, r in df_cycles.iterrows()]

# Color by log file
log_colors = {}
for i, lf in enumerate(df_cycles["log_file"].unique()):
    log_colors[lf] = ["#1f77b4", "#ff7f0e", "#2ca02c"][i % 3]
colors = [log_colors[lf] for lf in df_cycles["log_file"]]

# 1) Indoor temp vs target
ax = axes[0, 0]
if "pred_current" in df_cycles.columns:
    ax.plot(x, df_cycles["pred_current"], "ko-", ms=5, label="Current Indoor")
if "pred_target" in df_cycles.columns:
    ax.plot(x, df_cycles["pred_target"], "r--", lw=2, label="Target")
if "bs_target" in df_cycles.columns:
    ax.plot(x, df_cycles["bs_target"], "g:", lw=2, label="BS Target (pre-cool shifted)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Indoor Temp vs Target")
ax.legend(fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=7)

# 2) Outlet temps
ax = axes[0, 1]
if "bs_outlet" in df_cycles.columns:
    ax.plot(x, df_cycles["bs_outlet"], "b^-", ms=5, label="Binary Search")
if "newton_outlet" in df_cycles.columns:
    ax.plot(x, df_cycles["newton_outlet"], "rs-", ms=5, label="After Newton")
if "final_outlet" in df_cycles.columns:
    ax.plot(x, df_cycles["final_outlet"], "gD-", ms=6, label="Final (HA)")
ax.set_ylabel("Outlet Temp (°C)")
ax.set_title("Outlet Temp Pipeline")
ax.legend(fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=7)

# 3) Prediction error
ax = axes[1, 0]
if "feedback_error" in df_cycles.columns:
    bars = ax.bar(x, df_cycles["feedback_error"], color=colors, alpha=0.7, edgecolor="black", lw=0.5)
    ax.axhline(0.05, color="green", ls="--", alpha=0.5, label="Target (0.05°C)")
    ax.set_ylabel("Prediction Error (°C)")
    ax.set_title("Prediction Feedback Error per Cycle")
    ax.legend(fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=7)

# 4) Binary search iterations
ax = axes[1, 1]
if "bs_iterations" in df_cycles.columns:
    ax.bar(x, df_cycles["bs_iterations"], color=colors, alpha=0.7, edgecolor="black", lw=0.5)
    ax.set_ylabel("Iterations")
    ax.set_title("Binary Search Iterations")
    ax.axhline(1, color="green", ls="--", alpha=0.5, label="Ideal (1)")
    ax.legend(fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=7)

# 5) Gate state
ax = axes[2, 0]
if "gate_state" in df_cycles.columns:
    gate_map = {"RUNNING": 1, "RECOVERY": 0}
    gate_vals = df_cycles["gate_state"].map(gate_map).fillna(0.5)
    ax.bar(x, gate_vals, color=["green" if v == 1 else "orange" for v in gate_vals],
           alpha=0.7, edgecolor="black", lw=0.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["RECOVERY", "RUNNING"])
    ax.set_title("Cooling Cycle Gate State")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=7)

# 6) Parameter evolution
ax = axes[2, 1]
if "learn_hlc" in df_cycles.columns:
    ax2 = ax.twinx()
    ln1 = ax.plot(x, df_cycles["learn_hlc"], "b-o", ms=4, label="HLC")
    ln2 = ax.plot(x, df_cycles["learn_oe"], "g-s", ms=4, label="OE")
    ln3 = ax2.plot(x, df_cycles["learn_tau"], "r-^", ms=4, label="τ (h)")
    ax.set_ylabel("HLC / OE")
    ax2.set_ylabel("τ (hours)", color="red")
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=9, loc="upper left")
    ax.set_title("Parameter Evolution")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=7)

plt.suptitle("Cooling Cycle Dashboard — All Logs", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()"""))

# ============================================================
# CELL 7: ML vs Trajectory header
# ============================================================
cells.append(md("""## Phase C: ML (LGBM) vs Trajectory Pre-Cooling Comparison

**LGBM Classifier**: Binary overheating prediction with probability output.
- Threshold: 0.345
- If p > threshold → activate pre-cooling

**Trajectory (Shadow)**: Physics-based simulation of indoor temp with HP OFF.
- Simulates 12h forward using calibrated thermal model
- Reports: risk (bool), peak_temp, peak_hour

**Key observation from logs**: LGBM probability is near-constant (~0.42) across all conditions.
Does it discriminate at all?"""))

# ============================================================
# CELL 8: Comparison table
# ============================================================
cells.append(code("""# Build comparison table
comparison_cols = [
    "cycle_num", "log_file",
    "lgbm_prob", "lgbm_threshold", "lgbm_activated",
    "traj_risk", "traj_peak", "traj_peak_hours",
    "pred_current", "pred_target", "bs_target",
    "final_outlet", "feedback_error", "gate_state"
]

available_cols = [c for c in comparison_cols if c in df_cycles.columns]
comparison = df_cycles[available_cols].copy()

print("=== ML vs Trajectory Pre-Cooling Comparison ===\n")
print(comparison.to_string(index=False))

# Discrimination analysis
print("\n\n=== LGBM Discrimination Analysis ===")
if "lgbm_prob" in df_cycles.columns:
    probs = df_cycles["lgbm_prob"].dropna()
    print(f"LGBM probability range: {probs.min():.3f} — {probs.max():.3f}")
    print(f"LGBM probability std:   {probs.std():.4f}")
    print(f"LGBM probability mean:  {probs.mean():.3f}")
    print(f"Unique values: {probs.nunique()}")
    print(f"Always activated: {(probs > 0.345).all()}")
    
    if probs.std() < 0.05:
        print("\n⚠️ LGBM probability has VERY LOW variance — near-constant output!")
        print("   The classifier is not discriminating between conditions.")
        print("   This means it always predicts the same risk regardless of weather/state.")

print("\n=== Trajectory Discrimination Analysis ===")
if "traj_risk" in df_cycles.columns:
    risks = df_cycles["traj_risk"].dropna()
    peaks = df_cycles["traj_peak"].dropna()
    print(f"Trajectory risk True:  {risks.sum()} / {len(risks)} cycles")
    print(f"Trajectory risk False: {(~risks).sum()} / {len(risks)} cycles")
    print(f"Peak temp range: {peaks.min():.1f} — {peaks.max():.1f}°C")
    print(f"Peak temp std:   {peaks.std():.3f}°C")
    
    if risks.nunique() > 1:
        print("\n✅ Trajectory DOES discriminate — risk changes with conditions.")
    else:
        print("\n⚠️ Trajectory is also constant in this dataset.")"""))

# ============================================================
# CELL 9: Comparison plots
# ============================================================
cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1) LGBM probability distribution
ax = axes[0, 0]
if "lgbm_prob" in df_cycles.columns:
    probs = df_cycles["lgbm_prob"].dropna()
    ax.hist(probs, bins=max(5, len(probs)//3), alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(0.345, color="red", ls="--", lw=2, label=f"Threshold (0.345)")
    ax.set_xlabel("LGBM Probability")
    ax.set_ylabel("Count")
    ax.set_title(f"LGBM Probability Distribution (std={probs.std():.4f})")
    ax.legend()
else:
    ax.text(0.5, 0.5, "No LGBM data", transform=ax.transAxes, ha="center")

# 2) Trajectory peak vs actual indoor
ax = axes[0, 1]
if "traj_peak" in df_cycles.columns and "pred_current" in df_cycles.columns:
    ax.scatter(df_cycles["pred_current"], df_cycles["traj_peak"],
               c=df_cycles["traj_risk"].astype(float), cmap="RdYlGn_r",
               s=60, edgecolors="black", linewidth=0.5, zorder=3)
    lims = [min(df_cycles["pred_current"].min(), df_cycles["traj_peak"].min()) - 0.2,
            max(df_cycles["pred_current"].max(), df_cycles["traj_peak"].max()) + 0.2]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("Current Indoor (°C)")
    ax.set_ylabel("Trajectory Peak Prediction (°C)")
    ax.set_title("Trajectory: Current vs Predicted Peak")
    ax.legend()
else:
    ax.text(0.5, 0.5, "No trajectory data", transform=ax.transAxes, ha="center")

# 3) LGBM probability over cycles (should vary!)
ax = axes[1, 0]
if "lgbm_prob" in df_cycles.columns:
    x = range(len(df_cycles))
    ax.plot(x, df_cycles["lgbm_prob"], "bo-", ms=5, label="LGBM prob")
    ax.axhline(0.345, color="red", ls="--", alpha=0.7, label="Threshold")
    if "traj_peak" in df_cycles.columns and "pred_target" in df_cycles.columns:
        # Normalize traj peak as proxy for "risk level"
        traj_excess = df_cycles["traj_peak"] - df_cycles["pred_target"].fillna(23.0)
        ax2 = ax.twinx()
        ax2.plot(x, traj_excess, "g^-", ms=5, label="Traj excess (°C)")
        ax2.set_ylabel("Traj Peak - Target (°C)", color="green")
        ax2.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Cycle Index")
    ax.set_ylabel("LGBM Probability")
    ax.set_title("LGBM Prob vs Trajectory Excess Over Cycles")
    ax.legend(loc="upper left", fontsize=9)

# 4) Summary comparison metrics
ax = axes[1, 1]
ax.axis("off")
summary_text = "ML vs Trajectory Summary\n" + "=" * 35 + "\n\n"

if "lgbm_prob" in df_cycles.columns:
    p = df_cycles["lgbm_prob"].dropna()
    summary_text += f"LGBM Probability:\n"
    summary_text += f"  Range: {p.min():.3f} - {p.max():.3f}\n"
    summary_text += f"  Std:   {p.std():.4f}\n"
    summary_text += f"  Always triggers: {(p > 0.345).all()}\n\n"

if "traj_risk" in df_cycles.columns:
    r = df_cycles["traj_risk"].dropna()
    pk = df_cycles["traj_peak"].dropna()
    summary_text += f"Trajectory:\n"
    summary_text += f"  Risk True:  {r.sum()}/{len(r)}\n"
    summary_text += f"  Peak range: {pk.min():.1f}-{pk.max():.1f}°C\n"
    summary_text += f"  Peak std:   {pk.std():.3f}°C\n\n"

if "feedback_error" in df_cycles.columns:
    e = df_cycles["feedback_error"].dropna()
    summary_text += f"Overall Accuracy:\n"
    summary_text += f"  MAE: {e.mean():.4f}°C\n"
    summary_text += f"  Max error: {e.max():.4f}°C\n"

summary_text += f"\nVerdict: {'Trajectory' if df_cycles.get('lgbm_prob', pd.Series()).std() < 0.05 else 'LGBM'}\n"
summary_text += f"is more discriminative."
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.suptitle("Phase C: ML vs Trajectory Pre-Cooling", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()"""))

# ============================================================
# CELL 10: Binary search analysis
# ============================================================
cells.append(md("""## Phase D: Binary Search Convergence & Newton Correction Analysis"""))

# ============================================================
# CELL 11: Binary search & Newton
# ============================================================
cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1) Iterations by log file
ax = axes[0]
if "bs_iterations" in df_cycles.columns:
    for lf in df_cycles["log_file"].unique():
        mask = df_cycles["log_file"] == lf
        iters = df_cycles.loc[mask, "bs_iterations"].dropna()
        short_name = lf.split("_")[-1].replace(".log", "")[:10]
        ax.bar(range(len(iters)), iters, alpha=0.6, label=short_name)
    ax.set_xlabel("Cycle (within log)")
    ax.set_ylabel("Iterations")
    ax.set_title("Binary Search Iterations by Log")
    ax.legend(fontsize=9)

# 2) BS outlet vs Newton outlet
ax = axes[1]
if "bs_outlet" in df_cycles.columns and "newton_outlet" in df_cycles.columns:
    ax.scatter(df_cycles["bs_outlet"], df_cycles["newton_outlet"],
               c=df_cycles["bs_iterations"], cmap="viridis", s=60,
               edgecolors="black", linewidth=0.5)
    lims = [17, 25]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlabel("Binary Search Outlet (°C)")
    ax.set_ylabel("Newton-Corrected Outlet (°C)")
    ax.set_title("BS vs Newton (color=iterations)")
    plt.colorbar(ax.collections[0], ax=ax, label="BS Iterations")

# 3) Newton correction magnitude
ax = axes[2]
if "newton_delta" in df_cycles.columns:
    deltas = df_cycles["newton_delta"].dropna()
    ax.hist(deltas, bins=15, alpha=0.7, color="teal", edgecolor="black")
    ax.axvline(deltas.median(), color="red", ls="--", label=f"Median: {deltas.median():.2f}°C")
    ax.set_xlabel("Newton ΔT Correction (°C)")
    ax.set_ylabel("Count")
    ax.set_title("Newton Correction Distribution")
    ax.legend()

plt.suptitle("Phase D: Binary Search & Newton Correction", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Root cause analysis for oscillation
if "bs_iterations" in df_cycles.columns:
    print("\n=== Binary Search Convergence Analysis ===")
    iters = df_cycles["bs_iterations"].dropna()
    print(f"Mean iterations: {iters.mean():.1f}")
    print(f"1-iteration (perfect): {(iters == 1).sum()}/{len(iters)}")
    print(f"High iterations (>5): {(iters > 5).sum()}/{len(iters)}")
    
    if "pred_current" in df_cycles.columns:
        # Correlate iterations with temperature difference
        temp_diff = (df_cycles["pred_current"] - df_cycles["pred_target"]).abs()
        corr = temp_diff.corr(df_cycles["bs_iterations"])
        print(f"\nCorrelation (|T_current - T_target| vs iterations): {corr:.3f}")
        print("Higher temp diff → more iterations = harder to converge" if corr > 0.3 else
              "Iterations not strongly correlated with temp difference")"""))

# ============================================================
# CELL 12: Cycle gate analysis
# ============================================================
cells.append(md("""## Phase E: Cooling Cycle Gate Analysis

The cycle gate prevents HP short-cycling by enforcing RECOVERY periods after RUNNING.
- **RUNNING**: HP actively cooling
- **RECOVERY**: HP off, waiting for slab to absorb room heat"""))

# ============================================================
# CELL 13: Gate analysis code
# ============================================================
cells.append(code("""if "gate_state" in df_cycles.columns:
    print("=== Cooling Cycle Gate Transitions ===\n")
    
    gate_counts = df_cycles["gate_state"].value_counts()
    print(f"RUNNING:  {gate_counts.get('RUNNING', 0)} cycles")
    print(f"RECOVERY: {gate_counts.get('RECOVERY', 0)} cycles")
    
    # Transitions
    transitions = []
    for i in range(1, len(df_cycles)):
        prev = df_cycles.iloc[i-1].get("gate_state", "")
        curr = df_cycles.iloc[i].get("gate_state", "")
        if prev and curr and prev != curr:
            transitions.append(f"C{int(df_cycles.iloc[i-1]['cycle_num'])} {prev} → C{int(df_cycles.iloc[i]['cycle_num'])} {curr}")
    
    print(f"\nTransitions detected: {len(transitions)}")
    for t in transitions:
        print(f"  {t}")
    
    # Gate state impact on prediction error
    if "feedback_error" in df_cycles.columns:
        for state in ["RUNNING", "RECOVERY"]:
            mask = df_cycles["gate_state"] == state
            if mask.sum() > 0:
                errors = df_cycles.loc[mask, "feedback_error"].dropna()
                print(f"\n{state}: MAE={errors.mean():.4f}°C, max={errors.max():.4f}°C, n={len(errors)}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(14, 4))
    x = range(len(df_cycles))
    gate_colors = {"RUNNING": "green", "RECOVERY": "orange"}
    for i, row in df_cycles.iterrows():
        gs = row.get("gate_state", "")
        ax.barh(0, 1, left=i, color=gate_colors.get(gs, "gray"), edgecolor="white", lw=0.5)
    ax.set_xlim(-0.5, len(df_cycles) - 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Cycle Index")
    ax.set_title("Cooling Cycle Gate Timeline (Green=RUNNING, Orange=RECOVERY)")
    
    # Add cycle numbers
    for i, row in df_cycles.iterrows():
        ax.text(i + 0.5, 0, f"C{int(row['cycle_num'])}", ha="center", va="center",
                fontsize=7, rotation=90)
    plt.tight_layout()
    plt.show()
else:
    print("No gate state data available")"""))

# ============================================================
# CELL 14: Recommendations
# ============================================================
cells.append(md("""## Phase F: Findings & Improvement Recommendations"""))

# ============================================================
# CELL 15: Final analysis
# ============================================================
_final_analysis = '''print("=" * 70)
print("COOLING CYCLE ANALYSIS -- FINDINGS & RECOMMENDATIONS")
print("=" * 70)

print("\\n1. ML (LGBM) vs TRAJECTORY PRE-COOLING")
print("-" * 40)

if "lgbm_prob" in df_cycles.columns:
    p_std = df_cycles["lgbm_prob"].dropna().std()
    print(f"  LGBM probability std: {p_std:.4f}")
    if p_std < 0.05:
        print("  WARNING: LGBM is NOT discriminating -- probability is near-constant.")
        print("  -> The classifier always outputs ~0.42 regardless of conditions.")
        print("  -> This is equivalent to a constant 'always cool' rule.")
        print("")
        print("  ROOT CAUSE: Likely trained on imbalanced data where cooling")
        print("  was always needed (warm season only), so it learned no boundary.")

if "traj_risk" in df_cycles.columns:
    r = df_cycles["traj_risk"].dropna()
    if r.nunique() > 1:
        print(f"\\n  Trajectory DOES discriminate (risk: {r.sum()}/{len(r)} True)")
        print("  -> It reacts to changing conditions (outdoor temp, PV, time of day)")
        print("  -> Physically interpretable -- you can trace why it triggered")
    else:
        print(f"\\n  Trajectory is constant in this dataset ({r.iloc[0]})")

print("\\n  RECOMMENDATION: Use trajectory as PRIMARY pre-cooling trigger.")
print("  Demote LGBM to shadow-mode for benchmarking only.")
print("  If LGBM is to be useful, retrain with:")
print("    - More diverse data (include shoulder seasons)")
print("    - Stricter positive labeling (higher overshoot threshold)")
print("    - Feature engineering focused on discrimination (rate of change)")

print("\\n2. BINARY SEARCH CONVERGENCE")
print("-" * 30)
if "bs_iterations" in df_cycles.columns:
    iters = df_cycles["bs_iterations"].dropna()
    print(f"  Mean: {iters.mean():.1f}, Range: {iters.min():.0f}-{iters.max():.0f}")
    if iters.max() > 5:
        print("  WARNING: Some cycles need >5 iterations -- binary search oscillating.")
        print("  ROOT CAUSE: When equilibrium temp is near the target,")
        print("  small outlet changes cause large predicted changes -> oscillation.")
        print("")
        print("  RECOMMENDATION:")
        print("  - Increase convergence tolerance from 0.05C to 0.1C")
        print("  - Or: use golden-section search instead of binary search")

print("\\n3. NEWTON CORRECTION")
print("-" * 22)
if "newton_delta" in df_cycles.columns:
    nd = df_cycles["newton_delta"].dropna()
    print(f"  Applied in {len(nd)}/{len(df_cycles)} cycles")
    print(f"  Mean correction: {nd.mean():.3f}C")
    print("  Always overrides ML in cooling mode (HEATING_CORRECTION_MODE='ml' ignored)")
    print("")
    print("  ASSESSMENT: Newton correction is the RIGHT approach for cooling.")
    print("  The thermal model is physics-based and directly applicable.")
    print("  ML correction was designed for heating-specific patterns.")

print("\\n4. CYCLE GATE BEHAVIOR")
print("-" * 24)
if "gate_state" in df_cycles.columns:
    gc = df_cycles["gate_state"].value_counts()
    print(f"  RUNNING: {gc.get('RUNNING', 0)}, RECOVERY: {gc.get('RECOVERY', 0)}")
    if gc.get("RECOVERY", 0) > gc.get("RUNNING", 0):
        print("  WARNING: More RECOVERY than RUNNING -- system spending too much time waiting.")
        print("  -> Consider relaxing RECOVERY->RUNNING transition thresholds.")
    else:
        print("  OK: Mostly RUNNING -- HP actively cooling as expected.")

print("\\n5. OVERALL COOLING EFFECTIVENESS")
print("-" * 34)
if "feedback_error" in df_cycles.columns:
    e = df_cycles["feedback_error"].dropna()
    print(f"  Prediction MAE:  {e.mean():.4f}C")
    print(f"  Prediction max:  {e.max():.4f}C")
    if e.mean() < 0.1:
        print("  Excellent prediction accuracy (<0.1C)")
    elif e.mean() < 0.2:
        print("  Good prediction accuracy (<0.2C)")
    else:
        print("  Prediction accuracy could be improved")

print("\\n" + "=" * 70)'''

cells.append(code(_final_analysis))

# ============================================================
# Build notebook JSON
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
    "notebooks", "analysis", "11_cooling_cycle_analysis.ipynb"
)
output_path = os.path.normpath(output_path)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Created: {output_path}")
print(f"Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='markdown')} markdown, "
      f"{sum(1 for c in cells if c['cell_type']=='code')} code)")
