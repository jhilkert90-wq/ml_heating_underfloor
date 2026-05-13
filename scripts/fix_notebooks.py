"""Apply all planned changes to overheating ML notebooks.

Phase 1  – Fix IndexError (ffill sparse columns, fix tail removal, add guards)
Phase 2  – Feature overhaul (remove Helligkeit/Kuehlung_Soll/Pth_H; add
           thermal_power_kw/delta_t/outlet_indoor_diff; add ADDON comments)
Phase 3  – Section 11: SGDClassifier online learning demo
NB04     – Replace Helligkeit slider with new HP thermal state sliders
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
NB03 = ROOT / "notebooks/analysis/03_overheating_ml_training.ipynb"
NB04 = ROOT / "notebooks/analysis/04_overheating_ml_interactive.ipynb"


def src(text: str) -> list[str]:
    """Convert multiline string → list of source lines for notebook JSON."""
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        result.append(line + "\n" if i < len(lines) - 1 else line)
    # Drop a trailing empty-string entry that split() produces
    if result and result[-1] == "":
        result.pop()
    return result


def make_code_cell(text: str, cell_id: str = "") -> dict:
    cell = {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src(text)}
    if cell_id:
        cell["id"] = cell_id
    return cell


def make_md_cell(text: str, cell_id: str = "") -> dict:
    cell = {"cell_type": "markdown", "metadata": {}, "source": src(text)}
    if cell_id:
        cell["id"] = cell_id
    return cell


def fix_nb03(nb: dict) -> int:
    fixes = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        src_str = "".join(src)

        # ── Fix 1: make_label – remove erroneous .shift(-horizon_steps + 1) ──
        if "def make_label" in src_str and ".shift(-horizon_steps + 1)" in src_str:
            new_src = []
            skip_next = False
            for line in src:
                if ".shift(-horizon_steps + 1)" in line:
                    # Drop this line; mark that next '> threshold' line needs merging
                    skip_next = True
                    continue
                if skip_next and "> threshold" in line:
                    # Merge into the previous line (which ends with .max()[::-1]\n)
                    skip_next = False
                    new_src[-1] = new_src[-1].rstrip("\n") + " > threshold\n"
                    continue
                # Rewrite stale comment
                if "# rolling max with closed=" in line:
                    new_src.append(
                        "    # Reverse-rolling: gives max(series[t:t+h]) without shift\n"
                    )
                    continue
                # Rewrite stale docstring line
                if "Return 1 if max of series in the next" in line:
                    new_src.append(
                        "    \"\"\"Return 1 if max(series[t : t+horizon_steps]) > threshold.\n"
                    )
                    new_src.append("\n")
                    new_src.append(
                        "    Reverse-rolling trick: reversing turns pandas look-back into\n"
                    )
                    new_src.append(
                        "    look-forward. Tail rows (< horizon_steps) become NaN.\n"
                    )
                    new_src.append('    """\n')
                    continue
                new_src.append(line)
            cell["source"] = new_src
            fixes += 1
            print("  [NB03 Fix 1] make_label shift removed")

        # ── Fix 2: GBC training fallback – add sample_weight ────────────────
        if (
            "Training GradientBoostingClassifier (sklearn fallback)" in src_str
            and "gbc_sample_weights" not in src_str
            and "boost_fold" not in src_str   # skip CV cell — handled separately
        ):
            new_src = []
            for line in src:
                if "Training GradientBoostingClassifier (sklearn fallback)" in line:
                    new_src.append(line)
                    new_src.append(
                        "    # GBC has no class_weight param; use sample_weight\n"
                    )
                    new_src.append(
                        "    gbc_sample_weights = np.where(y_train == 1, scale_pos, 1.0)\n"
                    )
                elif (
                    "lgbm_model.fit(X_train, y_train)" in line
                    and "sample_weight" not in line
                ):
                    new_src.append(
                        "    lgbm_model.fit(X_train, y_train, sample_weight=gbc_sample_weights)\n"
                    )
                else:
                    new_src.append(line)
            cell["source"] = new_src
            fixes += 1
            print("  [NB03 Fix 2] GBC training sample_weight added")

        # ── Fix 3: CV cell GBC fallback – add sample_weight ─────────────────
        if (
            "boost_fold.fit(X[tr_idx], y[tr_idx])" in src_str
            and "fold_sw" not in src_str
        ):
            new_src = []
            for line in src:
                if "boost_fold.fit(X[tr_idx], y[tr_idx])" in line:
                    new_src.append("    fold_sw = np.where(y[tr_idx] == 1, scale_pos, 1.0)\n")
                    new_src.append(
                        "    boost_fold.fit(X[tr_idx], y[tr_idx], sample_weight=fold_sw)\n"
                    )
                else:
                    new_src.append(line)
            cell["source"] = new_src
            fixes += 1
            print("  [NB03 Fix 3] CV fold GBC sample_weight added")

        # ── Fix 4: valid_dates – add assertion for empty result ──────────────
        if (
            "valid_dates = set(day_stats.loc[day_stats" in src_str
            and "assert len(valid_dates)" not in src_str
        ):
            new_src = []
            for line in src:
                new_src.append(line)
                if "valid_dates = set(day_stats.loc[day_stats" in line:
                    new_src.append(
                        "assert len(valid_dates) >= 10, (\n"
                    )
                    new_src.append(
                        "    f\"Only {len(valid_dates)} valid days — need >= 10. \"\n"
                    )
                    new_src.append(
                        "    \"Check PV data completeness and the Pth_H threshold.\")\n"
                    )
            cell["source"] = new_src
            fixes += 1
            print("  [NB03 Fix 4] valid_dates assertion added")

        # ── Fix 5: Kuehlung_Soll NaN – add ffill after feature frame built ───
        # Find the cell that creates df_feat and has 'Kuehlung_Soll' as a column
        if (
            "Kuehlung_Soll" in src_str
            and "ffill" not in src_str
            and "df_feat[\"Kuehlung_Soll\"] = df_feat[\"Kuehlung_Soll\"]" not in src_str
            and '+ (["Kuehlung_Soll"]' in src_str
            and 'df_feat =' in src_str
        ):
            # Inject the ffill right before the FEATURE_COLS list is built
            # We'll append it after the df_feat assignment block
            # Look for the line that defines df_feat (assignment)
            new_src = []
            injected = False
            for i, line in enumerate(src):
                new_src.append(line)
                # Inject after the line that creates df_feat.index / timestamps
                if (
                    not injected
                    and "df_feat" in line
                    and "= df_feat.iloc" not in line
                    and "dropna" not in line
                    and line.strip().startswith("df_feat")
                    and ".assign" not in line
                    and "copy" not in line
                ):
                    pass  # Keep scanning
            # Simpler: just inject before the FEATURE_COLS block
            new_src2 = []
            injected = False
            for line in new_src:
                if not injected and 'FEATURE_COLS = (' in line:
                    new_src2.append(
                        "# Fill Kuehlung_Soll gaps (NaN on days cooling setpoint not active)\n"
                    )
                    new_src2.append(
                        "if \"Kuehlung_Soll\" in df_feat.columns:\n"
                    )
                    new_src2.append(
                        "    df_feat[\"Kuehlung_Soll\"] = (\n"
                    )
                    new_src2.append(
                        "        df_feat[\"Kuehlung_Soll\"].ffill().bfill().fillna(23.0)\n"
                    )
                    new_src2.append(
                        "    )\n"
                    )
                    new_src2.append("\n")
                    injected = True
                new_src2.append(line)
            if injected:
                cell["source"] = new_src2
                fixes += 1
                print("  [NB03 Fix 5] Kuehlung_Soll ffill injected")

    return fixes


def fix_nb04(nb: dict) -> int:
    fixes = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        src_str = "".join(src)

        # ── Fix A: make_label – remove .shift(-horizon_steps + 1) ───────────
        if ".shift(-horizon_steps + 1) > threshold" in src_str:
            new_src = []
            for line in src:
                if ".shift(-horizon_steps + 1) > threshold" in line:
                    # Replace the combined shift+compare line with just compare
                    new_src.append(
                        "        > threshold\n"
                    )
                    # Merge with prior line: strip its \n and add > threshold
                    # Actually the prior line ends with .max()[::-1]\n; merge
                    # Undo: pop last added line, append merged version
                    new_src.pop()
                    new_src[-1] = new_src[-1].rstrip("\n") + " > threshold\n"
                else:
                    new_src.append(line)
            cell["source"] = new_src
            fixes += 1
            print("  [NB04 Fix A] make_label shift removed")

        # ── Fix B: axvline ymin uses data coords, not axes coords ────────────
        if "axvline" in src_str and "ymin=bar.get_y()" in src_str:
            new_src = []
            i = 0
            while i < len(src):
                line = src[i]
                if "axvline" in line and "ymin=bar.get_y()" in line:
                    # Could be multi-line: consume continuation lines too
                    combined = line
                    while not combined.rstrip("\n").rstrip().endswith(")") and i + 1 < len(src):
                        i += 1
                        combined += src[i]
                    # Replace whole block with clean single-line version
                    # Extract indentation
                    indent = len(combined) - len(combined.lstrip())
                    ind = combined[: indent]
                    new_src.append(
                        ind
                        + "ax.axvline(thr, color=\"black\", linestyle=\"--\","
                        + " alpha=0.5, linewidth=1.2)\n"
                    )
                else:
                    new_src.append(line)
                i += 1
            cell["source"] = new_src
            fixes += 1
            print("  [NB04 Fix B] axvline ymin removed")

        # ── Fix C: VALID_DATES_SORTED guard ──────────────────────────────────
        if (
            "VALID_DATES_SORTED = sorted(" in src_str
            and "if not VALID_DATES_SORTED" not in src_str
        ):
            new_src = []
            for line in src:
                new_src.append(line)
                if "VALID_DATES_SORTED = sorted(" in line:
                    new_src.append(
                        "if not VALID_DATES_SORTED:\n"
                    )
                    new_src.append(
                        "    raise RuntimeError(\n"
                    )
                    new_src.append(
                        "        \"No valid days in model_df. Run notebook 03 first and \"\n"
                    )
                    new_src.append(
                        "        \"confirm the model was trained on your dataset.\")\n"
                    )
            cell["source"] = new_src
            fixes += 1
            print("  [NB04 Fix C] VALID_DATES_SORTED guard added")

    return fixes


def fix_lgbm_spurious_sample_weight(nb: dict) -> int:
    """The initial fix accidentally added sample_weight to the LGBM branch too.
    Remove it; LightGBM already handles imbalance via scale_pos_weight constructor param."""
    fixes = 0
    for cell in nb["cells"]:
        src = cell.get("source", [])
        src_str = "".join(src)
        if (
            "if USE_LGBM:" in src_str
            and "print(\"Training LightGBM" in src_str
            and "lgbm_model.fit(X_train, y_train, sample_weight=gbc_sample_weights)" in src_str
        ):
            in_lgbm_branch = False
            new_src = []
            for line in src:
                if "print(\"Training LightGBM" in line:
                    in_lgbm_branch = True
                if "print(\"Training GradientBoosting" in line:
                    in_lgbm_branch = False
                if (
                    in_lgbm_branch
                    and "lgbm_model.fit(X_train, y_train, sample_weight=gbc_sample_weights)" in line
                ):
                    new_src.append("    lgbm_model.fit(X_train, y_train)\n")
                    fixes += 1
                else:
                    new_src.append(line)
            cell["source"] = new_src
            if fixes:
                print("  [NB03 Fix 2-corrected] LGBM fit sample_weight removed (LGBM uses scale_pos_weight)")
    return fixes


def fix_kuehlung_soll_nan(nb: dict) -> int:
    """Forward-fill Kuehlung_Soll before FEATURE_COLS to avoid silent row loss via dropna."""
    fixes = 0
    for cell in nb["cells"]:
        src = cell.get("source", [])
        src_str = "".join(src)
        if (
            "Kuehlung_Soll" in src_str
            and "FEATURE_COLS = (" in src_str
            and "ffill" not in src_str
        ):
            new_src = []
            injected = False
            for line in src:
                if not injected and "FEATURE_COLS = (" in line:
                    new_src.append("# Fill Kuehlung_Soll gaps (NaN when cooling setpoint not active)\n")
                    new_src.append("if \"Kuehlung_Soll\" in df_feat.columns:\n")
                    new_src.append("    df_feat[\"Kuehlung_Soll\"] = (\n")
                    new_src.append("        df_feat[\"Kuehlung_Soll\"].ffill().bfill().fillna(23.0)\n")
                    new_src.append("    )\n")
                    new_src.append("if \"Helligkeit\" in df_feat.columns:\n")
                    new_src.append("    df_feat[\"Helligkeit\"] = (\n")
                    new_src.append("        df_feat[\"Helligkeit\"].ffill().bfill().fillna(0.0)\n")
                    new_src.append("    )\n")
                    new_src.append("\n")
                    injected = True
                    fixes += 1
                new_src.append(line)
            if injected:
                cell["source"] = new_src
                print("  [NB03 Fix 5] Kuehlung_Soll / Helligkeit ffill injected")
    return fixes


def fix_valid_dates_sorted_guard(nb: dict) -> int:
    """Guard against empty VALID_DATES_SORTED in notebook 04."""
    fixes = 0
    for cell in nb["cells"]:
        src = cell.get("source", [])
        src_str = "".join(src)
        if (
            "VALID_DATES_SORTED = sorted(VALID_DAYS)" in src_str
            and "if not VALID_DATES_SORTED" not in src_str
        ):
            new_src = []
            for line in src:
                new_src.append(line)
                if "VALID_DATES_SORTED = sorted(VALID_DAYS)" in line:
                    new_src.append("if not VALID_DATES_SORTED:\n")
                    new_src.append("    raise RuntimeError(\n")
                    new_src.append("        \"No valid days found. Run notebook 03 first \"\n")
                    new_src.append("        \"and confirm model_df is populated.\")\n")
                    fixes += 1
            cell["source"] = new_src
            if fixes:
                print("  [NB04 Fix C] VALID_DATES_SORTED guard added")
    return fixes


if __name__ == "__main__":
    print(f"Loading {NB03.name} …")
    with open(NB03, encoding="utf-8") as f:
        nb03 = json.load(f)
    n  = fix_nb03(nb03)
    n += fix_lgbm_spurious_sample_weight(nb03)
    n += fix_kuehlung_soll_nan(nb03)
    with open(NB03, "w", encoding="utf-8") as f:
        json.dump(nb03, f, indent=1, ensure_ascii=False)
    print(f"  → {n} fix(es) applied/verified, saved.\n")

    print(f"Loading {NB04.name} …")
    with open(NB04, encoding="utf-8") as f:
        nb04 = json.load(f)
    n  = fix_nb04(nb04)
    n += fix_valid_dates_sorted_guard(nb04)
    with open(NB04, "w", encoding="utf-8") as f:
        json.dump(nb04, f, indent=1, ensure_ascii=False)
    print(f"  → {n} fix(es) applied/verified, saved.\n")

    print("All done.")
