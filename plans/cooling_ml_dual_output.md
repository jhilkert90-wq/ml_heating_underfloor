# Plan: Implement Notebook Findings into Production Cooling ML Pipeline

## TL;DR
Apply 09_ notebook insights to production: (1) fix sklearn feature-name warnings, (2) add regression
model alongside classifier for dual-output, (3) implement proportional pre-cooling intensity based on
predicted overshoot, (4) improve threshold/calibration logging. Six phases, grouped by dependency.

## Context
- Notebook proved: regression (AUC=0.9586 Optuna-tuned) > classifier (AUC=0.9434), F1 threshold balanced
- Dual-output (P×Δ): classifier gate × regression overshoot = F1=0.935, precision=93.9%, recall=93.1%
- HA_LOG has 17× sklearn feature-name warnings (numpy array passed to named-feature model)
- Pre-cooling is currently binary: fixed 0.5K target offset regardless of predicted overshoot magnitude
- F1 calibration code already done (beta=2→1), but production hasn't been re-calibrated yet
- **R² investigation COMPLETED (2026-05-31)**: R²=0.36 is expected for delta_indoor_8h (std=0.177).
  Switching to max_indoor_8h gives R²=0.73 but HURTS AUC (0.952→0.898) and F1 (0.934→0.875).
  Decision: **keep delta_indoor_8h** — AUC/F1 are the metrics that matter for pre-cooling.

## Steps

### Phase 1: Fix sklearn feature-name warnings (quick win)
**Depends on: nothing | Risk: low**

1. **`src/cooling_ml_model.py` ~L584**: Change inference from numpy array to pandas DataFrame:
   - Current: `X = np.array(vec, dtype=float).reshape(1, -1)`
   - New: `X = pd.DataFrame([vec], columns=self._feature_cols)`
   - This silences all 17 runtime warnings

2. **`src/cooling_ml_calibration.py` ~L966-1000**: In `_cross_validate_threshold()`, wrap CV fold arrays:
   - Current: `proba_fold = model.predict_proba(X_fold)[:, 1]`
   - Fix: Create DataFrame with feature columns for `X_fold` before predict
   - Also affects `_optimise_threshold_fbeta` calls that use raw arrays

### Phase 2: Add regression model to calibration pipeline
**Depends on: Phase 1 | Risk: medium**

3. **`src/cooling_ml_calibration.py`**: After the existing classifier training (Step 10-11), add a
   regression branch that trains a LGBMRegressor on `delta_indoor_8h`:
   - Compute `delta_indoor_8h = max_indoor_8h - indoor_temp` (rolling max, same as notebook)
   - Train with `objective='regression_l1'`, same hyperparams structure
   - F1-optimize a temperature threshold on `indoor_temp + predicted_delta > cooling_target`
   - Save as separate joblib: `cooling_ml_regressor.joblib`

4. **`src/cooling_ml_calibration.py`**: Extend metadata JSON to include regression fields:
   - `"regression_threshold"`: temperature threshold (≈22.99°C from notebook)
   - `"regression_mae"`: MAE on validation (≈0.083°C)
   - `"regression_auc"`: AUC when used as classifier (≈0.958)
   - `"model_approach"`: "dual" (classifier + regressor)

### Phase 3: Add regression inference to CoolingMLModel
**Depends on: Phase 2 | Risk: medium**

5. **`src/cooling_ml_model.py`**: Add regression model loading in `load()`:
   - Load `cooling_ml_regressor.joblib` alongside classifier
   - Store as `self._reg_model`, `self._reg_threshold`
   - Graceful fallback: if regressor doesn't exist, classifier-only mode

6. **`src/cooling_ml_model.py`**: Extend `predict_overheating_risk()` with regression prediction:
   - After classifier probability: `delta_pred = self._reg_model.predict(X)[0]`
   - `predicted_max = current_indoor + delta_pred`
   - `reg_risk = predicted_max > self._reg_threshold`
   - Return both in result dict: `lgbm_proba`, `predicted_delta`, `predicted_max_temp`

7. **`src/cooling_ml_model.py`**: Add `peak_temp` from regression (replaces current proxy):
   - Current: `peak_temp_proxy = max(current_indoor, trigger_threshold + 0.1) if risk else current_indoor`
   - New: `peak_temp = current_indoor + delta_pred` (actual predicted max temperature)

### Phase 4: Proportional pre-cooling intensity
**Depends on: Phase 3 | Risk: medium-high (behavioral change)**

8. **`src/cycle_routes.py` ~L770-785**: Replace fixed offset with proportional intensity:
   - Current: `ctx.target_indoor_temp = ctx.prediction_indoor_temp - _offset` (always 0.5K)
   - New: Scale offset by predicted overshoot magnitude from regression:
     - `overshoot = predicted_max - cooling_target`
     - `offset = clip(overshoot × 0.7, min=0.2, max=1.0)` (proportional, bounded)
   - This uses regression's continuous output for graduated response

9. **`src/cycle_routes.py`**: Log the proportional offset and predicted max:
   - Add `predicted_max` and `cooling_intensity` to the pre-cool log message
   - Add to state persistence: `pre_cool_predicted_max`, `pre_cool_offset_k`

### Phase 5: Dual-output decision logic + HA Dashboard selector
**Depends on: Phase 3 and 4 | Risk: low**

10. **`src/cooling_ml_model.py`**: Implement dual-output decision with configurable strategy:
    - Read `PRE_COOL_DUAL_OUTPUT_STRATEGY` from config (default: "classifier_gate")
    - `"classifier_gate"`: `should_cool_now` = classifier_gate AND (regression_risk OR reactive) — conservative
    - `"either_triggers"`: `should_cool_now` = classifier_gate OR regression_risk — aggressive, catches more events
    - When models disagree → log shadow warning regardless of strategy
    - Regression always provides intensity (proportional offset) when activated

11. **`src/config.py`**: Add new config options:
    - `PRE_COOL_PROPORTIONAL: bool` (default True) — enable proportional intensity
    - `PRE_COOL_MIN_OFFSET_K: float` (default 0.2) — minimum pre-cool offset
    - `PRE_COOL_MAX_OFFSET_K: float` (default 1.0) — maximum pre-cool offset
    - `PRE_COOL_OVERSHOOT_GAIN: float` (default 0.7) — gain factor for offset scaling
    - `PRE_COOL_DUAL_OUTPUT_STRATEGY: str` (default "classifier_gate") — "classifier_gate" or "either_triggers"

12. **`config.yaml`**: Add HA dropdown for dual-output strategy:
    - `pre_cool_dual_output_strategy: "list(classifier_gate|either_triggers)"`
    - This follows the existing `list()` pattern used by `pre_cool_model_type`

13. **`translations/en.yaml`**: Add tooltip text:
    - `pre_cool_dual_output_strategy:`
      - `name: '[ML Pre-Cooling] Dual-Output Strategy'`
      - `description: "Controls how classifier and regression model combine decisions.
        'classifier_gate' (recommended) = only pre-cools when the F1-balanced classifier confirms risk —
        conservative, fewer false triggers. 'either_triggers' = pre-cools when either model detects risk —
        catches more overheating events but may trigger unnecessary pre-cooling. Both modes use the
        regression model's predicted overshoot for proportional cooling intensity. [BOTH]"`

### Phase 6: Improved calibration diagnostics
**Depends on: Phase 1 | Risk: low**

14. **`src/cooling_ml_calibration.py`**: Add diagnostic logging for isotonic calibration:
    - Log threshold shift: `"Isotonic threshold shift: raw=%.4f → calibrated=%.4f (Δ=%.4f)"`
    - Log if shift is >50% (suspicious): `"⚠️ Large isotonic threshold shift: %.1f%%"`
    - This addresses the HA_LOG gap where isotonic compression went unlogged

15. **`src/cooling_ml_calibration.py`**: Log F1 metrics alongside F-beta:
    - After threshold optimization, also compute and log precision/recall/predicted_pos_rate
    - `"F1=%.4f | Precision=%.4f | Recall=%.4f | Predicted pos rate=%.1f%% (true=%.1f%%)"`

## Relevant Files
- `src/cooling_ml_model.py` — L223-240 (build_feature_vector), L485-530 (load), L535-670 (predict)
- `src/cooling_ml_calibration.py` — L785-920 (threshold+isotonic), L940-1020 (optimize functions)
- `src/cycle_routes.py` — L640-800 (step_pre_cooling)
- `src/config.py` — L650-680 (PRE_COOL_ constants)
- `tests/unit/test_cooling_ml_calibration.py` — metadata key assertions
- `tests/unit/test_cooling_ml_model.py` — inference test updates

## Verification
1. All existing tests pass after Phase 1 (feature name fix)
2. Calibration produces both classifier + regressor joblib files (Phase 2)
3. sklearn feature-name warning disappears from HA_LOG (Phase 1)
4. Regression AUC matches notebook: ~0.95+ on holdout (Phase 2)
5. Pre-cooling offset varies by predicted overshoot, not fixed 0.5K (Phase 4)
6. Isotonic threshold shift is logged in calibration output (Phase 6)
7. Run full test suite after each phase

## Decisions
- **Keep delta_indoor_8h** as regression target: R²=0.36 is expected (low label variance, std=0.177). Switching to max_indoor_8h gives R²=0.73 but degrades AUC and F1. The R² metric is misleading here.
- **Dual-output over pure regression**: Classifier provides interpretable on/off gate with calibrated probability; regression provides continuous intensity. Neither alone is sufficient.
- **Proportional scaling bounded [0.2, 1.0]K**: Floor prevents negligible actions, ceiling prevents over-correction. Gain=0.7 means 1°C predicted overshoot → 0.7K target offset.
- **Backward compatible**: If regression model file doesn't exist, falls back to classifier-only (current behavior). Config flag controls proportional mode.
- **Scope**: Cooling ML pipeline only. No changes to heating model, dashboard, or OverheatingPredictor.
- **Out of scope**: Optuna HPO in production (already handled by calibration's existing hyperparams), SHAP integration, regression-only mode.
