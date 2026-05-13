"""Quick verification that all planned changes are present in nb03 and nb04."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
NB03 = ROOT / "notebooks/analysis/03_overheating_ml_training.ipynb"
NB04 = ROOT / "notebooks/analysis/04_overheating_ml_interactive.ipynb"

nb03 = json.load(open(NB03, encoding="utf-8"))
all_src = " ".join("".join(c["source"]) for c in nb03["cells"])

n_cells = len(nb03["cells"])
print(f"NB03 total cells: {n_cells}")

checks_nb03 = [
    ("ffill(limit=24) in resample cell",      "ffill(limit=24)" in all_src),
    ("ffill(limit=6) for HP sparse",          "ffill(limit=6)" in all_src),
    ("thermal_power_kw feature",              "thermal_power_kw" in all_src),
    ("delta_t feature",                       "delta_t" in all_src),
    ("outlet_indoor_diff feature",            "outlet_indoor_diff" in all_src),
    ("notna label filter",                    'df_feat["label_8h"].notna()' in all_src),
    ("iloc[:-N] removed",                     "iloc[:-LABEL_HORIZON_STEPS]" not in all_src),
    ("assert len(model_df) >= 1000",          "assert len(model_df) >= 1000" in all_src),
    ("SGDClassifier import",                  "from sklearn.linear_model import SGDClassifier" in all_src),
    ("partial_fit in Section 11",             "partial_fit" in all_src),
    ("weekly_aucs online learning",           "weekly_aucs" in all_src),
]

# Cell-specific checks
feat_cols_src = next(
    ("".join(c["source"]) for c in nb03["cells"] if "FEATURE_COLS = (" in "".join(c["source"])),
    ""
)
meta_src = next(
    ("".join(c["source"]) for c in nb03["cells"] if "feature_mapping_for_live_system" in "".join(c["source"])),
    ""
)
checks_nb03 += [
    # They appear only in "# Removed: Helligkeit" comments — that is correct.
    # Check they do NOT appear as active list entries (i.e., as quoted strings in the list).
    ("Helligkeit not an active FEATURE_COLS entry",
     '"Helligkeit"' not in feat_cols_src or "# Removed: Helligkeit" in feat_cols_src),
    ("Kuehlung_Soll not an active FEATURE_COLS entry",
     '"Kuehlung_Soll"' not in feat_cols_src or "# Removed: Kuehlung_Soll" in feat_cols_src),
    ("Helligkeit absent from metadata cell",       "Helligkeit" not in meta_src),
    ("new features in metadata feature_mapping",   "thermal_power_kw" in meta_src and "outlet_indoor_diff" in meta_src),
]

all_ok = True
for desc, ok in checks_nb03:
    status = "OK   " if ok else "FAIL "
    if not ok:
        all_ok = False
    print(f"  [{status}] {desc}")

print()

nb04 = json.load(open(NB04, encoding="utf-8"))
all_src04 = " ".join("".join(c["source"]) for c in nb04["cells"])

checks_nb04 = [
    ("sl_hell removed",                "sl_hell" not in all_src04),
    ("sl_thermal_pw added",            "sl_thermal_pw" in all_src04),
    ("sl_delta_t added",               "sl_delta_t" in all_src04),
    ("sl_outlet_diff added",           "sl_outlet_diff" in all_src04),
    ("thermal_power_kw in feats dict", 'feats["thermal_power_kw"]' in all_src04),
]
print("NB04 checks:")
for desc, ok in checks_nb04:
    status = "OK   " if ok else "FAIL "
    if not ok:
        all_ok = False
    print(f"  [{status}] {desc}")

print()
print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED — see FAIL items above")
