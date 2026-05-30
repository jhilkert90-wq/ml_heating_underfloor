# Code Review Plan: Cooling ML Dual-Output Pipeline
*Based on: `cooling_ml_dual_output.md` — 6-phase implementation*

---

## 0 · Pre-Implementation Audit (before first commit)

Run this once before any code is written. Goal: no duplicate implementations, no stale dead code.

### Copilot Prompt — Pre-Implementation Audit
```
Before implementing cooling_ml_dual_output.md, perform a duplicate audit across:

Files in scope:
- src/cooling_ml_model.py
- src/cooling_ml_calibration.py
- src/cycle_routes.py
- src/config.py
- config.yaml
- translations/en.yaml

For each of the following, check whether a version already exists anywhere in the codebase:
1. Any pandas DataFrame wrapping of feature vectors before predict()
2. Any LGBMRegressor training or loading
3. Any `predicted_delta`, `predicted_max`, or `peak_temp` fields in the result dict
4. Any proportional/variable pre-cooling offset logic (anything other than fixed 0.5K)
5. Any `PRE_COOL_DUAL_OUTPUT_STRATEGY`, `PRE_COOL_PROPORTIONAL` config constants
6. Any `cooling_ml_regressor.joblib` load path

Report: symbol name, file, line number, and whether it conflicts or can be reused.
```

---

## Phase 1 Review — Fix sklearn Feature-Name Warnings

**Risk:** Low | **Files:** `cooling_ml_model.py`, `cooling_ml_calibration.py`

### Checklist
- [ ] `cooling_ml_model.py` ~L584: numpy array replaced with `pd.DataFrame([vec], columns=self._feature_cols)`
- [ ] Column order in the DataFrame exactly matches `self._feature_cols` — no reordering
- [ ] `cooling_ml_calibration.py` `_cross_validate_threshold()`: all `X_fold` slices are wrapped before `predict_proba()`
- [ ] `_optimise_threshold_fbeta` and any other callers of `predict_proba` / `predict` on raw arrays are also fixed
- [ ] No new import added twice (`import pandas as pd` deduplication check)
- [ ] sklearn `FutureWarning` / `UserWarning` about feature names no longer appears when model runs in isolation

### Copilot Prompt — Phase 1 Review
```
Review the following changes in src/cooling_ml_model.py and src/cooling_ml_calibration.py.
Scope: only uncommitted git changes (git diff HEAD).

Focus areas:
1. Every call to model.predict_proba() and model.predict() — confirm all receive a pandas
   DataFrame, not a numpy array, with the correct column list from self._feature_cols.
2. Check that the DataFrame column order matches the original feature order exactly.
3. Verify no sklearn FutureWarning or UserWarning about feature names remains possible.
4. Check that no other callers of the model in these two files still pass raw numpy arrays.
5. Confirm no duplicate `import pandas as pd` was introduced.

Report any remaining numpy array paths as blocking issues.
```

### Verification
```bash
# Must produce zero matches after fix
grep -n "np.array(vec" src/cooling_ml_model.py
grep -n "predict_proba(X_fold)" src/cooling_ml_calibration.py

# Run unit tests — must pass clean
python -m pytest tests/unit/test_cooling_ml_model.py tests/unit/test_cooling_ml_calibration.py -v
```

---

## Phase 2 Review — Regression Model in Calibration Pipeline

**Risk:** Medium | **Files:** `cooling_ml_calibration.py`

### Checklist
- [ ] `delta_indoor_8h` computed as `max_indoor_8h - indoor_temp` using rolling max — same window as notebook
- [ ] `LGBMRegressor` uses `objective='regression_l1'` (MAE, consistent with heating model convention)
- [ ] Hyperparameter structure mirrors the classifier branch — no silent divergence
- [ ] F1 threshold on `indoor_temp + predicted_delta > cooling_target` uses the **same** beta as classifier
- [ ] Regressor saved to `cooling_ml_regressor.joblib` with `joblib.dump()`
- [ ] Metadata JSON extended with all four new fields: `regression_threshold`, `regression_mae`, `regression_auc`, `model_approach: "dual"`
- [ ] `tests/unit/test_cooling_ml_calibration.py`: metadata key assertions updated for new fields
- [ ] Calibration does NOT break if the regression branch fails (try/except or guard)

### Copilot Prompt — Phase 2 Review
```
Review the regression training branch added to src/cooling_ml_calibration.py.
Scope: only uncommitted git changes (git diff HEAD).

Focus areas:
1. Verify delta_indoor_8h = max_indoor_8h - indoor_temp uses the same rolling window as in the notebook.
2. Confirm LGBMRegressor objective is 'regression_l1' and hyperparams are consistent with the classifier.
3. Check that the F1-threshold optimization on the regressor uses the same beta parameter as the classifier.
4. Verify cooling_ml_regressor.joblib is written via joblib.dump() in the same output directory as the classifier.
5. Confirm all four new metadata JSON fields are present: regression_threshold, regression_mae, regression_auc, model_approach.
6. Check that test_cooling_ml_calibration.py metadata assertions cover the new keys.
7. Identify any data leakage risk: is max_indoor_8h computed after the train/val split?

Report any data leakage, missing metadata field, or divergence from the classifier branch as blocking.
```

### Verification
```bash
# After calibration run, both files must exist
ls -lh cooling_ml_classifier.joblib cooling_ml_regressor.joblib

# Metadata must contain regression keys
python -c "import json; d=json.load(open('cooling_ml_metadata.json')); \
  assert all(k in d for k in ['regression_threshold','regression_mae','regression_auc','model_approach']), \
  'Missing metadata keys'; print('OK', d['model_approach'])"

# AUC must be >= 0.95 (notebook result: 0.9586)
python -m pytest tests/unit/test_cooling_ml_calibration.py -v
```

---

## Phase 3 Review — Regression Inference in CoolingMLModel

**Risk:** Medium | **Files:** `cooling_ml_model.py`

### Checklist
- [ ] `load()`: `cooling_ml_regressor.joblib` loaded into `self._reg_model`; `self._reg_threshold` populated from metadata
- [ ] Graceful fallback: if regressor file missing, `self._reg_model = None` and classifier-only mode works unchanged
- [ ] `predict_overheating_risk()`: `delta_pred = self._reg_model.predict(X)[0]` only called when `self._reg_model is not None`
- [ ] Result dict extended with `predicted_delta`, `predicted_max_temp` (not replacing, extending)
- [ ] Old `peak_temp_proxy` logic replaced by `peak_temp = current_indoor + delta_pred` — no dead code left
- [ ] `reg_risk` computed as `predicted_max > self._reg_threshold` and returned in result dict
- [ ] `tests/unit/test_cooling_ml_model.py` updated: test with regressor present AND absent (fallback path)

### Copilot Prompt — Phase 3 Review
```
Review changes to src/cooling_ml_model.py for Phase 3 (regression inference).
Scope: only uncommitted git changes (git diff HEAD).

Focus areas:
1. Confirm load() handles missing cooling_ml_regressor.joblib gracefully without raising an exception.
2. Verify predict_overheating_risk() guards every reg_model call with `if self._reg_model is not None`.
3. Check that result dict keys added (predicted_delta, predicted_max_temp, reg_risk) do not collide
   with existing keys — list all existing keys in the result dict for comparison.
4. Confirm the old peak_temp_proxy pattern is fully removed (no residual dead code).
5. Check that the DataFrame wrapping from Phase 1 is also applied to the regressor's predict() call.
6. Verify test coverage: one test with reg_model loaded, one without (fallback).

Report any unguarded reg_model access or key collision as blocking.
```

### Verification
```bash
# Fallback path: rename regressor, run predict, restore
mv cooling_ml_regressor.joblib cooling_ml_regressor.joblib.bak
python -c "from src.cooling_ml_model import CoolingMLModel; m=CoolingMLModel(); m.load(); \
  print('fallback OK, reg_model:', m._reg_model)"
mv cooling_ml_regressor.joblib.bak cooling_ml_regressor.joblib

python -m pytest tests/unit/test_cooling_ml_model.py -v -k "regressor or dual or fallback"
```

---

## Phase 4 Review — Proportional Pre-Cooling Intensity

**Risk:** Medium-High (behavioral change) | **Files:** `cycle_routes.py`

### Checklist
- [ ] Fixed `0.5K` offset is fully removed — no silent fallback to 0.5K if regression unavailable
  - **If `_reg_model` is None:** define fallback behavior explicitly (e.g. use `PRE_COOL_MIN_OFFSET_K`)
- [ ] Formula: `overshoot = predicted_max - cooling_target`, `offset = clip(overshoot × gain, min, max)`
- [ ] `clip` uses `PRE_COOL_MIN_OFFSET_K` and `PRE_COOL_MAX_OFFSET_K` from config — no hardcoded 0.2/1.0
- [ ] Gain factor sourced from `PRE_COOL_OVERSHOOT_GAIN` config — no hardcoded 0.7
- [ ] `cooling_target` used here is the same value as the threshold used in regression training (≈22.99°C)
- [ ] Log line includes `predicted_max`, `overshoot`, and `cooling_intensity` (offset applied)
- [ ] State persistence: `pre_cool_predicted_max` and `pre_cool_offset_k` persisted
- [ ] Edge case: `overshoot ≤ 0` (no predicted overshoot) — offset still at minimum, not zero or negative

### Copilot Prompt — Phase 4 Review
```
Review changes to src/cycle_routes.py for Phase 4 (proportional pre-cooling intensity).
Scope: only uncommitted git changes (git diff HEAD).

Focus areas:
1. Verify the fixed 0.5K offset is fully removed. Confirm there is no code path that silently
   falls back to 0.5K — if reg_model is None, the fallback behavior must be explicit and documented.
2. Check the offset formula: overshoot = predicted_max - cooling_target, offset = clip(overshoot × gain, min, max).
   All three parameters (gain, min, max) must come from config, not hardcoded.
3. Verify cooling_target is the same threshold value used during regression training (regression_threshold
   from metadata), not a different constant.
4. Confirm the log message includes predicted_max, overshoot, and the final offset_k applied.
5. Check state persistence: pre_cool_predicted_max and pre_cool_offset_k are written.
6. Verify the edge case: overshoot ≤ 0 produces offset = PRE_COOL_MIN_OFFSET_K, not 0 or negative.

Report any hardcoded offset, unguarded division, or missing fallback as blocking.
```

### Verification
```bash
# Integration test: trigger pre-cooling with a mock high predicted_max
# offset should be > 0.2K and < 1.0K, proportional to overshoot
python -m pytest tests/ -v -k "pre_cool and proportional"

# Manual sanity: overshoot=0.5°C, gain=0.7 → offset=0.35K (within [0.2, 1.0])
python -c "
gain, lo, hi = 0.7, 0.2, 1.0
for overshoot in [-0.5, 0.0, 0.3, 0.5, 1.0, 2.0]:
    offset = max(lo, min(hi, overshoot * gain))
    print(f'overshoot={overshoot:+.1f}°C → offset={offset:.2f}K')
"
```

---

## Phase 5 Review — Dual-Output Strategy + Config + HA Selector

**Risk:** Low | **Files:** `cooling_ml_model.py`, `config.py`, `config.yaml`, `translations/en.yaml`

### Checklist
- [ ] `PRE_COOL_DUAL_OUTPUT_STRATEGY` read from config with default `"classifier_gate"`
- [ ] `"classifier_gate"` logic: `should_cool` = `classifier_gate AND (reg_risk OR reactive)`
- [ ] `"either_triggers"` logic: `should_cool` = `classifier_gate OR reg_risk`
- [ ] Disagreement logging: when classifier and regressor differ → `"shadow warning"` logged regardless of strategy
- [ ] `config.py`: all 5 new constants present with correct types and defaults
- [ ] `config.yaml`: `pre_cool_dual_output_strategy` uses `list(classifier_gate|either_triggers)` pattern
- [ ] `translations/en.yaml`: description matches exactly what the plan specifies (both strategies explained)
- [ ] Invalid strategy value → raises `ValueError` with clear message, not silent default
- [ ] `tests/unit/test_cooling_ml_model.py`: one test per strategy

### Copilot Prompt — Phase 5 Review
```
Review changes for Phase 5 across src/cooling_ml_model.py, src/config.py, config.yaml,
and translations/en.yaml.
Scope: only uncommitted git changes (git diff HEAD).

Focus areas:
1. Verify both strategy branches ("classifier_gate" and "either_triggers") implement the logic
   exactly as specified in the plan. Write out the boolean expression for each.
2. Confirm disagreement logging fires when classifier and regressor produce different outputs,
   regardless of which strategy is active.
3. Check config.py for all 5 new constants (PRE_COOL_PROPORTIONAL, PRE_COOL_MIN_OFFSET_K,
   PRE_COOL_MAX_OFFSET_K, PRE_COOL_OVERSHOOT_GAIN, PRE_COOL_DUAL_OUTPUT_STRATEGY) with correct types.
4. Check config.yaml uses the list() pattern consistent with pre_cool_model_type.
5. Verify an invalid strategy string raises ValueError (not silently falls back).
6. Confirm translations/en.yaml description covers both strategies and mentions "proportional cooling intensity".

Report any strategy logic error, missing config constant, or silent invalid-input handling as blocking.
```

### Verification
```bash
grep -n "PRE_COOL_" src/config.py | sort
grep -n "pre_cool_dual_output_strategy" config.yaml translations/en.yaml

python -m pytest tests/unit/test_cooling_ml_model.py -v -k "strategy or dual_output"
```

---

## Phase 6 Review — Improved Calibration Diagnostics

**Risk:** Low | **Files:** `cooling_ml_calibration.py`

### Checklist
- [ ] Isotonic threshold shift logged: `"Isotonic threshold shift: raw=%.4f → calibrated=%.4f (Δ=%.4f)"`
- [ ] Large-shift warning fired if `abs(Δ) / raw > 0.5`: `"⚠️ Large isotonic threshold shift: %.1f%%"`
- [ ] F1 / Precision / Recall / predicted_pos_rate / true_pos_rate all logged after threshold optimization
- [ ] Log format: `"F1=%.4f | Precision=%.4f | Recall=%.4f | Predicted pos rate=%.1f%% (true=%.1f%%)"`
- [ ] Diagnostic logging does not modify any threshold value — read-only observation
- [ ] All new log lines use the same logger as the rest of calibration (no print() statements)

### Copilot Prompt — Phase 6 Review
```
Review diagnostic logging changes in src/cooling_ml_calibration.py for Phase 6.
Scope: only uncommitted git changes (git diff HEAD).

Focus areas:
1. Confirm isotonic threshold shift is logged with the exact format: raw → calibrated → delta.
2. Verify the large-shift warning threshold is 50% (abs(delta)/raw > 0.5) and uses ⚠️ prefix.
3. Check that F1, Precision, Recall, predicted_pos_rate, and true_pos_rate are all computed
   and logged after threshold optimization — not just one or two of them.
4. Confirm all new log statements use the existing module logger, not print().
5. Verify none of the logging code has any side effect on threshold values (pure observation).
6. Check there is no duplicate computation of these metrics if they were already computed elsewhere.

Report any print() usage, missing metric, or side-effecting log code as blocking.
```

### Verification
```bash
# No print() should be present in calibration logging additions
git diff HEAD src/cooling_ml_calibration.py | grep "^+" | grep "print("

# Calibration run log should contain new diagnostic lines
python -m src.cooling_ml_calibration 2>&1 | grep -E "Isotonic|threshold shift|F1=|Precision="
```

---

## Cross-Cutting Concerns

Review these after all phases are complete, on the full diff.

### Copilot Prompt — Cross-Cutting Review
```
Perform a cross-cutting review of all uncommitted changes (git diff HEAD) across:
src/cooling_ml_model.py, src/cooling_ml_calibration.py, src/cycle_routes.py, src/config.py.

Check:

BACKWARD COMPATIBILITY
1. Does the system behave identically to the current behavior if cooling_ml_regressor.joblib is absent?
2. Does PRE_COOL_PROPORTIONAL=False restore the fixed-offset behavior?
3. Are all new config keys optional with safe defaults?

ERROR HANDLING
4. What happens if the regressor produces a NaN or negative delta_pred? Is it clamped or guarded?
5. What happens if cooling_ml_metadata.json is missing the new regression keys?
6. Is joblib.load() for the regressor wrapped in a try/except?

LOGGING CONSISTENCY
7. Are all new log messages at appropriate levels (DEBUG vs INFO vs WARNING)?
8. Do log messages follow the existing format style (no inconsistent prefixes)?

SCOPE CREEP
9. Are there any changes to heating model, dashboard, or OverheatingPredictor? (must be 0)
10. Any Optuna HPO, SHAP, or regression-only mode code sneaked in? (must be 0)

TYPE SAFETY
11. Are all new config values type-checked or validated on startup?
12. Are float config values validated for sensible ranges (gain ∈ (0,1], min < max)?

Report each finding with file and line number. Classify as: BLOCKING / WARNING / SUGGESTION.
```

---

## Final Integration Checklist

Run after all 6 phases are implemented and reviewed.

| # | Verification step | Command / Check |
|---|---|---|
| 1 | All unit tests pass | `python -m pytest tests/ -v` |
| 2 | sklearn warnings absent | `python -m src.cooling_ml_model 2>&1 \| grep -i "feature names"` → 0 matches |
| 3 | Both joblib files produced | `ls cooling_ml_*.joblib` → 2 files |
| 4 | Regression AUC ≥ 0.95 on holdout | Check calibration log |
| 5 | Pre-cooling offset varies | Log shows different `offset_k` across different runs |
| 6 | Isotonic shift logged | `grep "Isotonic threshold shift"` in calibration output |
| 7 | Fallback mode works | Rename regressor.joblib → system runs classifier-only without error |
| 8 | Invalid strategy → ValueError | `PRE_COOL_DUAL_OUTPUT_STRATEGY="invalid"` → startup error |
| 9 | No changes outside scope | `git diff HEAD -- src/heating*.py` → empty |
| 10 | No print() in new code | `git diff HEAD \| grep "^+" \| grep "print("` → 0 matches |

---

## Review Order Recommendation

```
Phase 1  →  Phase 6  →  Phase 2  →  Phase 3  →  Phase 4  →  Phase 5  →  Cross-Cutting
(quick)     (quick)     (core)      (core)       (risky)     (config)     (full diff)
```

Do Phase 6 before Phase 2 to catch logging gaps early in the calibration file while the diff is still small.
Review Phase 4 with a product mindset, not just code — it changes runtime behavior in production.
