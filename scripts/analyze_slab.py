"""
Analysis script: Indoor temp + heat_loss_coefficient over time
Also analyses slab model behaviour from parameter_history.
"""
import json
import sys
from datetime import datetime

data_path = "../ml_heating_data/unified_thermal_state.json"
with open(data_path) as f:
    d = json.load(f)

ph = d["learning_state"]["parameter_history"]
pred = d["learning_state"]["prediction_history"]

print("=== PARAMETER HISTORY SAMPLE (every 50 records) ===")
print("slab_time_constant_hours in records:", "slab_time_constant_hours" in ph[-1])
print("gradient keys:", list(ph[-1].get("gradients", {}).keys()))
for i in range(0, len(ph), 50):
    r = ph[i]
    tau = r.get("slab_time_constant_hours", "N/A")
    sg = r.get("gradients", {}).get("slab_time_constant", "N/A")
    print("  [%3d] %s  hlc=%.5f  tau=%s  slab_grad=%s" % (
        i, r["timestamp"][:16], r["heat_loss_coefficient"], tau, sg))
r = ph[-1]
tau = r.get("slab_time_constant_hours", "N/A")
sg = r.get("gradients", {}).get("slab_time_constant", "N/A")
print("  [499] %s  hlc=%.5f  tau=%s  slab_grad=%s" % (
    r["timestamp"][:16], r["heat_loss_coefficient"], tau, sg))

print("\n=== PREDICTION HISTORY (actual indoor temp) ===")
actuals = [(p["timestamp"][:16], p["actual"], p["predicted"]) for p in pred]
print("  First:", actuals[0])
print("  Last:", actuals[-1])
print("  Count:", len(actuals))

# Try to produce matplotlib chart
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    # --- Data extraction ---
    hlc_times = [datetime.fromisoformat(r["timestamp"]) for r in ph]
    hlc_vals = [r["heat_loss_coefficient"] for r in ph]
    tau_vals = [r.get("slab_time_constant_hours", None) for r in ph]
    slab_grads = [r.get("gradients", {}).get("slab_time_constant", 0.0) for r in ph]

    pred_times = [datetime.fromisoformat(p["timestamp"]) for p in pred]
    actual_vals = [p["actual"] for p in pred]
    predicted_vals = [p["predicted"] for p in pred]
    errors = [p["error"] for p in pred]

    # --- Common time range: full parameter_history window ---
    t_min = min(hlc_times[0], pred_times[0])
    t_max = max(hlc_times[-1], pred_times[-1])

    # --- Figure: 3 subplots sharing x-axis ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("ML Heating – Slab Model & Heat Loss Analysis\n"
                 "(Mar 14–20, 2026)", fontsize=13, fontweight="bold")

    # --- Panel 1: Indoor temp (predicted vs actual) + trajectory correction events ---
    ax1 = axes[0]
    ax1.plot(pred_times, actual_vals, color="#e07b39", lw=1.5, label="Actual indoor °C")
    ax1.plot(pred_times, predicted_vals, color="#4c8be0", lw=1.0, alpha=0.7,
             ls="--", label="Predicted indoor °C")
    target = 22.6
    ax1.axhline(target, color="gray", lw=0.8, ls=":", label=f"Target {target}°C")
    # Mark undershoot events (actual < target - 0.05)
    under = [(t, v) for t, v in zip(pred_times, actual_vals) if v < target - 0.05]
    if under:
        ax1.scatter([u[0] for u in under], [u[1] for u in under],
                    color="red", s=20, zorder=5, label="Undershoot events")
    ax1.set_ylabel("Indoor Temperature (°C)")
    ax1.set_title("Indoor Temperature (prediction history)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: heat_loss_coefficient over full parameter_history ---
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    ax2.plot(hlc_times, hlc_vals, color="#2ca02c", lw=1.5, label="heat_loss_coeff (U)")
    ax2.axhline(0.13003, color="#2ca02c", lw=0.8, ls=":", alpha=0.5,
                label="Calibration baseline (0.130)")
    ax2.set_ylabel("Heat Loss Coefficient U (1/h)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    # Overlay outlet_effectiveness
    eff_vals = [r["outlet_effectiveness"] for r in ph]
    ax2_twin.plot(hlc_times, eff_vals, color="#9467bd", lw=1.0, alpha=0.7,
                  ls="--", label="outlet_eff")
    ax2_twin.axhline(0.486322, color="#9467bd", lw=0.8, ls=":", alpha=0.4)
    ax2_twin.set_ylabel("Outlet Effectiveness", color="#9467bd")
    ax2_twin.tick_params(axis="y", labelcolor="#9467bd")

    ax2.set_title("Learned Parameters: U (heat loss) + outlet effectiveness (Mar 14–20)")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    # Indoor temp reference on secondary axis
    ax2_temp = ax2.twinx()
    ax2_temp.spines["right"].set_position(("outward", 55))
    ax2_temp.plot(pred_times, actual_vals, color="#e07b39", lw=1.0, alpha=0.45,
                 ls="-", label="Indoor °C (ref)")
    ax2_temp.axhline(target, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax2_temp.set_ylabel("Indoor Temp (°C)", color="#e07b39", fontsize=7)
    ax2_temp.tick_params(axis="y", labelcolor="#e07b39", labelsize=7)
    lines3, labels3 = ax2_temp.get_legend_handles_labels()
    ax2.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Annotation: calibration baseline
    ax2.annotate("Calibrated baseline\nU=0.130", xy=(hlc_times[0], 0.13003),
                 xytext=(hlc_times[len(hlc_times)//4], 0.135),
                 fontsize=7, color="#2ca02c",
                 arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.7))
    ax2.annotate("Current effective\nU≈0.109", xy=(hlc_times[-1], hlc_vals[-1]),
                 xytext=(hlc_times[len(hlc_times)*3//4], 0.108),
                 fontsize=7, color="#2ca02c",
                 arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.7))

    # --- Panel 3: slab gradient over time (where identifiable) ---
    ax3 = axes[2]
    # τ_slab values (only available after our feature branch)
    tau_available = [(t, v) for t, v in zip(hlc_times, tau_vals) if v is not None]
    slab_grad_abs = [abs(g) for g in slab_grads]

    ax3.fill_between(hlc_times, slab_grad_abs, color="#d62728", alpha=0.3,
                     label="|slab gradient| (τ identifiability)")
    ax3.set_ylabel("|Slab gradient|", color="#d62728")
    ax3.tick_params(axis="y", labelcolor="#d62728")
    ax3.set_title("Slab τ gradient magnitude (≈0 = equilibrium, system not identifiable)")

    if tau_available:
        ax3_twin = ax3.twinx()
        ax3_twin.plot([t for t, _ in tau_available], [v for _, v in tau_available],
                      color="#8c564b", lw=1.5, label="τ_slab (h)")
        ax3_twin.set_ylabel("τ_slab (hours)", color="#8c564b")
        ax3_twin.tick_params(axis="y", labelcolor="#8c564b")
        ax3_twin.set_ylim(0, 4)
        ax3_twin.axhline(1.0, color="#8c564b", lw=0.8, ls=":", alpha=0.5,
                         label="τ=1.0h default")
        ax3_twin.legend(fontsize=8, loc="upper right")

    # Indoor temp reference
    ax3_temp = ax3.twinx()
    if tau_available:
        ax3_temp.spines["right"].set_position(("outward", 55))
    ax3_temp.plot(pred_times, actual_vals, color="#e07b39", lw=1.0, alpha=0.45,
                 ls="-", label="Indoor \u00b0C (ref)")
    ax3_temp.axhline(target, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax3_temp.set_ylabel("Indoor Temp (\u00b0C)", color="#e07b39", fontsize=7)
    ax3_temp.tick_params(axis="y", labelcolor="#e07b39", labelsize=7)

    lines_a, labels_a = ax3.get_legend_handles_labels()
    lines_b, labels_b = ax3_temp.get_legend_handles_labels()
    ax3.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="upper left")
    ax3.grid(True, alpha=0.3)

    # --- Shared x-axis formatting ---
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=12))
    for ax in axes:
        ax.set_xlim(t_min, t_max)
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    plt.tight_layout()
    out = "../ml_heating_data/slab_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {out}")

except ImportError as e:
    print(f"\nmatplotlib not available: {e}")
    print("Printing ASCII summary instead:")
    print("\nHLC trend (sampled):")
    for i in range(0, len(ph), 25):
        r = ph[i]
        bar = "#" * int(r["heat_loss_coefficient"] * 200)
        print("  %s  U=%.4f  %s" % (r["timestamp"][:13], r["heat_loss_coefficient"], bar))
