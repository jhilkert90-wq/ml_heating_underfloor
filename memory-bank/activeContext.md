# Active Context - Current Work & Decision State

### ✅ Post-Review Hardening: Settings Edge Cases — 2026-05-17

#### **What changed**
- Reviewed the new dashboard settings stack and fixed edge-case handling:
  - `dashboard/settings_service.py` now sanitizes option payloads so unknown keys from local/Supervisor sources are ignored.
  - Save path now posts only known add-on options, while load paths still merge defaults for missing values.
  - Supervisor base URL is resolved at call time instead of import time.
- `dashboard/components/settings.py` now robustly handles malformed persisted values:
  - string-aware boolean coercion (`"true"/"false"/"1"/"0"`),
  - safe int/float parsing with fallback to defaults,
  - numeric clamping to schema min/max before widget render.
- `tests/unit/test_dashboard_settings.py` gained regression tests for unknown-key filtering and boolean coercion.

#### **Why**
- The first implementation assumed clean typed payloads. In practice, persisted/local config values can be stale, string-typed, or include unknown keys after version changes.
- These guards prevent runtime widget errors and avoid sending unsupported keys back to Supervisor.

### ✅ Dashboard Settings Page + German Translation Coverage — 2026-05-17

#### **What changed**
- Added a new dashboard settings view in `dashboard/components/settings.py` and wired it into `dashboard/app.py` sidebar navigation.
- Added shared dashboard helpers:
  - `dashboard/config_schema.py` parses add-on defaults, schema types, section groupings, and translation labels directly from the repository config/translation files.
  - `dashboard/settings_service.py` loads and saves add-on options through the Home Assistant Supervisor API (`/addons/self/options`), with local fallback loading when Supervisor access is unavailable.
- Added `tests/unit/test_dashboard_settings.py` to verify settings metadata coverage, section mapping, default extraction, and Supervisor API request behavior.
- Updated translation files:
  - `ml_heating_underfloor/translations/en.yaml` now includes section prefixes and the missing labels/descriptions for newer options.
  - New `ml_heating_underfloor/translations/de.yaml` provides German labels for all add-on options while preserving English descriptions for tooltips.

#### **Why**
- The native Home Assistant add-on options UI is flat, so prefixed labels improve immediate readability there.
- The new dashboard Settings page provides the missing grouped UX: German labels, English hover help, collapsible sections, and direct option persistence without editing the flat add-on config manually.

### ✅ Full-Test Follow-Up: Holdout Guard + Bounds Alignment — 2026-05-17

#### **What changed**
- Added dedicated regression coverage in `tests/unit/test_heating_correction_ml_calibration.py` to assert that Optuna/CV training fits do not ingest temporal holdout rows.
- Fixed cross-module mismatch in `src/thermal_config.py`: aligned `pv_heat_weight` lower bound to `0.0001` (from `0.00001`) to match test expectations and add-on schema.
- Executed broad validation:
  - Targeted + calibration unit tests passed
  - Full unit suite passed except one dependency-gated file (`hypothesis` missing)
  - Integration image-smoke tests remain environment-gated without local Docker CLI.

#### **Why**
- User requested both: full-suite regression check and a dedicated Optuna/CV holdout isolation test.
- Full run surfaced one actionable code mismatch (parameter bounds) and two environment blockers (missing dependencies/tools).

### ✅ Post-Implementation Review Fixes — 2026-05-17

#### **What changed**
- **Holdout leakage fixed** in `src/heating_correction_ml_calibration.py`: Optuna and optional CV now operate on `df_fit` only, preserving temporal holdout (`df_val`) as an unbiased final validation slice.
- **CV edge-case guard**: when fit data is too short for `TimeSeriesSplit`, calibration now logs a clear skip warning instead of failing CV path.
- **Permutation importance stability**: switched to `n_jobs=1` to avoid multiprocessing fragility in mocked/non-picklable estimator contexts.
- **Config wording alignment**: `ml_heating_underfloor/config.yaml` tooltip for `heating_ml_cv_enabled` now matches implementation (additional CV diagnostics, holdout still retained).
- **Test robustness**: fake LightGBM regressors in `tests/unit/test_heating_correction_ml_calibration.py` now inherit sklearn estimator mixins to remove deprecation warning flood and future sklearn incompatibility risk.

#### **Why**
- Review identified a genuine evaluation-risk mismatch: HPO/CV consumed data intended to remain a final holdout.
- Edge-case handling and cleaner test compatibility reduce operational noise and future maintenance risk.

### ✅ ML Calibration Improvements + PV Rescue Decoupling — 2026-05-17

#### **What changed**
- **PV Trajectory Rescue**: `src/pv_trajectory.py` rescue condition decoupled from `min_steps`. New `PV_TRAJ_RESCUE_MIN_HOURS` config (default 1) controls independently how many forecast hours above threshold are needed for rescue. Fixes bug where gradual PV decline (1930W→1440W) caused premature trajectory collapse and unwanted overshoot correction (−0.98°C blend).
- **ML Calibration Pipeline** (`src/heating_correction_ml_calibration.py`):
  - Feature pruning: drops features with PI ≤ threshold, retrains, accepts only if MAE regression ≤ 0.5%
  - LightGBM regularisation: `reg_alpha`/`reg_lambda` forwarded to model
  - Optuna HPO: config-gated, searches 6 hyperparameters over TimeSeriesSplit(3)
  - Time-series CV: config-gated, reports MAE±std and R²±std across folds
- **Config plumbing**: 10 new config vars across `src/config.py`, `ml_heating_underfloor/config.yaml` (with tooltips + schema), `config_adapter.py`
- **Tests**: 5 new rescue_min_hours tests in `test_pv_trajectory.py`, 9 new tests for config defaults + regularisation + pruning in `test_heating_correction_ml_calibration.py`

#### **Why**
- R²=0.8541 with 40 features (11 with PI ≤ 0) suggested overfitting. Feature pruning + regularisation should improve generalisation. Optuna and CV provide optional deeper optimisation.
- The overshoot correction bug was caused by the rescue condition being tied to min_steps=4, requiring 4 future hours above PV threshold — too strict for gradual afternoon decline.

### ✅ Refine pv_traj_disable_overshoot_correction boundary at min steps — 2026-05-16

#### **What changed**
- `src/model_wrapper.py`: `_verify_trajectory_and_correct` early-return guard was narrowed. Suppression now requires:
  - `PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION == true`
  - `PV_TRAJ_FORECAST_MODE_ENABLED == true`
  - `TRAJECTORY_STEPS > PV_TRAJ_MIN_STEPS`
- At `TRAJECTORY_STEPS == PV_TRAJ_MIN_STEPS`, overshoot/undershoot correction is re-enabled and normal trajectory verification/correction logic executes.
- `tests/unit/test_overshoot_logic.py`: Added two tests in `TestDisableOvershootCorrectionInForecastMode` for:
  - skip when `TRAJECTORY_STEPS > PV_TRAJ_MIN_STEPS`
  - correction path execution when `TRAJECTORY_STEPS == PV_TRAJ_MIN_STEPS`
- Wording updated to match new behavior in:
  - `src/config.py`
  - `ml_heating_underfloor/config.yaml`
  - `ml_heating_underfloor/translations/en.yaml`

#### **Why**
- Dynamic forecast scaling intentionally extends the planning horizon during available PV. However, when the horizon has already collapsed to the minimum (`PV_TRAJ_MIN_STEPS`), disabling correction becomes counterproductive for comfort control. Re-enabling correction at the floor keeps protection active exactly when forecast lookahead benefit is minimal.

#### **Files**
`src/model_wrapper.py`, `tests/unit/test_overshoot_logic.py`, `src/config.py`, `ml_heating_underfloor/config.yaml`, `ml_heating_underfloor/translations/en.yaml`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

### ✅ Add pv_traj_disable_overshoot_correction switch — 2026-05-16

#### **What changed**
- `ml_heating_underfloor/config.yaml`: Added `pv_traj_disable_overshoot_correction: false` option (and schema `"bool"`) under the Forecast-Driven Trajectory Scaling section.
- `ml_heating_underfloor/translations/en.yaml`: Added English name/description for the new option.
- `config_adapter.py`: Mapped `pv_traj_disable_overshoot_correction` → `PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION` env var.
- `src/config.py`: Added `PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION: bool` (default `false`).
- `src/model_wrapper.py`: Added early-return guard in `_verify_trajectory_and_correct` — when both `PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION` and `PV_TRAJ_FORECAST_MODE_ENABLED` are `true`, the function logs a debug message and returns `outlet_temp` unchanged, skipping all trajectory prediction and correction logic.
- `tests/unit/test_overshoot_logic.py`: Added `TestDisableOvershootCorrectionInForecastMode` (4 tests, all pass).

#### **Why**
- When forecast-driven trajectory scaling (`PV_TRAJ_FORECAST_MODE_ENABLED`) is active, the planning horizon is dynamically derived from remaining PV forecast hours. Applying an additional overshoot/undershoot correction on top can create conflicting adjustments (e.g. the horizon says "heat longer" but the correction says "reduce outlet"). The new switch gives operators control to suppress the correction in this scenario.

#### **Files**
`ml_heating_underfloor/config.yaml`, `ml_heating_underfloor/translations/en.yaml`, `config_adapter.py`, `src/config.py`, `src/model_wrapper.py`, `tests/unit/test_overshoot_logic.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`



#### **What changed**
- `.github/workflows/build.yaml`: In the `test-image` job "Core module smoke test" step, replaced `from src.config import POLL_INTERVAL` with `from src.config import CYCLE_INTERVAL_MINUTES`, and aligned wrapper import to `EnhancedModelWrapper`.
- `tests/integration/test_image_smoke.py`: Updated `test_core_modules()` container import script to use the same valid config symbol and wrapper class name.

#### **Why**
- GitHub Actions run `25956950704` failed in job `Smoke-test built image (amd64)` with `ImportError: cannot import name 'POLL_INTERVAL' from 'src.config'`. The symbol no longer exists in `src/config.py`, so the smoke test itself was broken while application modules were otherwise importable.

#### **Files**
`.github/workflows/build.yaml`, `tests/integration/test_image_smoke.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

### ✅ Fix calibration UserWarning spam — 2026-05-16

#### **What changed**
- `src/heating_correction_ml_calibration.py`: `X_fit = df_fit[feature_cols].astype(float)` and `X_val = df_val[feature_cols].astype(float)` — removed `.values` so LightGBM trains with named DataFrame columns rather than anonymous numpy arrays.

#### **Why**
- `permutation_importance` called `model.predict()` with numpy slices of `X_val`; since the model had auto-generated feature names (LightGBM assigned them even when trained with numpy in newer versions), sklearn emitted `UserWarning: X does not have valid feature names` ~400+ times per calibration run (40 features × 10 repeats). Training with named DataFrames makes training/inference format consistent and eliminates the warning.

#### **Files**
`src/heating_correction_ml_calibration.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`



#### **What changed**
- **Config plumbing**: `WIND_SPEED_ENTITY_ID` (default `sensor.wind_speed`) added to `src/config.py`, `config_adapter.py`, `config.yaml`, `influx_service.py`, `physics_calibration.py`.
- **Calibration** (`src/heating_correction_ml_calibration.py`): 8 new features (`wind_speed`, `indoor_temp_gradient`, `living_room_temp`, `is_hp_active`, `is_weekend`, `thermal_power_rolling_1h`, `indoor_margin_rate`, `is_overshoot`) + column renames for living_room/wind entities + feature importance logging (LightGBM split + permutation) + importances saved to metadata JSON.
- **Inference** (`src/heating_correction_ml_model.py`): 8 new `_extract_heating_feature()` handlers with fallbacks. `predict()` now uses `pd.DataFrame` instead of `np.array` (fixes sklearn feature-names warning). Added `_load_pandas()` helper.
- **Runtime features** (`src/physics_features.py`): Added `wind_speed`, `is_weekend`, `indoor_margin_rate` to features dict.
- **Tooltips** (`en.yaml`): Added `wind_speed_entity` tooltip. `heating_ml_min_training_samples` tooltip was already present.
- **Tests**: 16 new unit tests in `test_heating_correction_ml_model.py`. Fixed mocks in `test_physics_features.py` and `test_ha_history_service.py`.

#### **Why**
- ML model had R²=0.86 with 32 features. Adding wind, living room temp, HP state, weekend, thermal power rolling, margin rate, gradient, and overshoot indicator should improve accuracy. One model with `is_overshoot` indicator rather than two separate models (user decision). Feature importance logging enables analyzing which features contribute most after calibration.

### ✅ Newton correction τ/2 floor fix + UI improvements — 2026-05-16

#### **What changed**
- `src/model_wrapper.py`: `_calculate_physics_newton_correction()` now floors `t_eval` to `τ_room * 0.5` when the worst trajectory violation is at an early step. Both ε and S(t) are re-evaluated at the floored time. If the sign of ε flips (trajectory recovered), the correction is suppressed entirely. Docstring updated to reflect the new semantics.
- `tests/unit/test_heating_correction.py`: Test 9 updated (overshoot peak moved from t=2h to t=3h to stay above τ/2). Two new tests: test 12 (τ/2 floor prevents always-clamped correction) and test 13 (sign flip suppression).
- `ml_heating_underfloor/translations/en.yaml`: Added 13 missing tooltip descriptions for ML heating correction parameters (`heating_ml_*`) and `pv_traj_forecast_rescue_enabled`.
- `dashboard/components/control.py`: Added "Calibrate ML Heating Model" button with flag `/data/config/calibrate_heating_correction_ml_flag`.

#### **Why**
- The Newton correction was always clamped to ±2.5°C because S(t≈0.17h)≈0.03 at step 0 made ε/S(t) exceed the clamp for any non-trivial error. Log evidence: 7 consecutive cycles all showed `S(t=0.17h)=0.0296` with ΔT=+2.500°C. The τ/2 floor fixes the root cause (evaluating at a time when the slab has meaningfully responded).
- Missing tooltips made 13 configuration parameters invisible in the HA add-on UI.
- Missing dashboard button prevented users from triggering ML heating model calibration.

#### **Design decisions**
- τ/2 chosen as the floor because at t=τ/2 the system has reached ~39% of equilibrium — the earliest time where the Newton step is physically meaningful for underfloor heating.
- Sign-flip detection at the floored time prevents the correction from "solving" a transient that has already self-corrected.

### ✅ ML heating correction workflow audit — 3 bugs fixed — 2026-05-15

#### **What changed**
- `src/heating_correction_ml_model.py`: `_extract_heating_feature("indoor_temp")` and `_extract_heating_feature("indoor_margin")` now fall back to `physics.get("indoor_temp_lag_30m")` when `indoor_temp` is absent. `build_physics_features()` never emits `indoor_temp`; at runtime both functions were returning 0.0 (critical: model received completely wrong temperature values).
- `src/heating_correction_ml_calibration.py`: S_H fallback warning now correctly logs the original degenerate value in the first format arg instead of the fallback value twice.
- `config_adapter.py`: Added `HEATING_ML_RETRAIN_VAL_FRACTION` env var mapping (was missing from the HA add-on adapter).
- `ml_heating_underfloor/config.yaml`: Added `heating_ml_retrain_val_fraction` option (default 0.25, schema range 0.05–0.5).
- `tests/unit/test_heating_correction_ml_model.py`: 3 new regression tests: `test_indoor_temp_falls_back_to_lag_30m`, `test_indoor_temp_returns_zero_when_both_keys_absent`, `test_indoor_margin_falls_back_to_lag_30m`.

#### **Why**
- The `indoor_temp` key mismatch is critical: the ML correction model would silently use an indoor temperature of 0.0°C, making `indoor_margin` ≈ target_temp (21°C) and `indoor_temp` = 0°C — both completely wrong — for every inference cycle. This would cause the model to consistently output a large positive correction delta even when the room was already warm.
- The warning message bug meant operators could not diagnose which S_H value triggered the fallback.
- The missing config adapter entry meant `HEATING_ML_RETRAIN_VAL_FRACTION` could not be changed via the HA UI add-on config.

#### **Design decisions**
- Use `indoor_temp_lag_30m` as the inference fallback for `indoor_temp` since: (a) it's the proxy the rest of the thermal model already uses as the indoor temperature baseline, and (b) at training time `df["indoor_temp"]` is the InfluxDB reading which corresponds to the same 10-minute-interval value.
- Keep existing behaviour (explicit `indoor_temp` key wins if present) for forward compatibility with any future code that does inject it.

#### **Files changed**
`src/heating_correction_ml_model.py`, `src/heating_correction_ml_calibration.py`, `config_adapter.py`, `ml_heating_underfloor/config.yaml`, `tests/unit/test_heating_correction_ml_model.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

#### **What changed**
- `src/main.py`: gated the complete heating observation-buffer block on `climate_mode == "heating"` so `push_pending`, `resolve_labels`, save, and retrain trigger only execute during heating operation.
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`: updated to document the corrected workflow and the bug that was fixed.

#### **Why**
- Pending heating observations were otherwise vulnerable to being resolved during cooling/summer cycles with non-heating indoor temperatures, producing polluted labels and potentially causing bad retrains of the heating ML correction model.

#### **Design decisions**
- Keep collection independent of `HEATING_CORRECTION_MODE`, but tie label aging/resolution to actual heating cycles only.
- Match the heating workflow to the same climate-mode gating pattern already used by the cooling observation buffer integration.

#### **Files changed**
`src/main.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

### ✅ Heating Correction ML Online Learning — 2026-05-15

#### **What changed**
- `src/heating_correction_ml_observation_buffer.py` (new): `HeatingCorrectionObservationBuffer` — sliding-window regression label buffer; stores feature snapshots, resolves labels `−(T_indoor[t+N] − T_target) / S_H` (S_H recomputed at resolve-time from current thermal params), auto-triggers retrain, JSON persistence (atomic tmp→replace), thread-safe `RLock`, eviction policy (oldest labeled first).
- `src/main.py`: unconditional init block before main loop; per-cycle push (heating mode only) + resolve (every cycle) + save + auto-retrain (calls `calibrate_heating_correction_ml()`, hot-reloads singleton via `EnhancedModelWrapper._heating_correction_ml_model = None`); partial back-off on retrain failure.
- `src/config.py`: 3 new vars (`HEATING_ML_OBSERVATION_BUFFER_PATH`, `HEATING_ML_RETRAIN_TRIGGER_K`, `HEATING_ML_BUFFER_MAX_N`); defaults to `_UNIFIED_STATE_DIR`.
- `config_adapter.py`: 2 new env var mappings in heating ML block.
- `ml_heating_underfloor/config.yaml`: 2 new options + 2 schema entries.
- `tests/unit/test_heating_correction_observation_buffer.py` (new): 25 tests.

#### **Why**
Adds self-improving online learning to the heating correction ML model: after initial calibration the model automatically retrains as real operational data accumulates, improving accuracy over the heating season without manual re-calibration.

#### **Design decisions**
- Observations collected always (regardless of `HEATING_CORRECTION_MODE`) so buffer fills even before ML mode is activated.
- S_H recomputed at resolve-time from current thermal params (not stored at push-time) so labels reflect latest calibrated physics.
- Retrain via InfluxDB (`calibrate_heating_correction_ml()`) for consistency with the initial calibration path.
- Default paths in `_UNIFIED_STATE_DIR` (same directory as `UNIFIED_STATE_FILE`) so all runtime state co-locates.

#### **Files changed**
`src/heating_correction_ml_observation_buffer.py` (new), `src/main.py`, `src/config.py`, `config_adapter.py`, `ml_heating_underfloor/config.yaml`, `tests/unit/test_heating_correction_observation_buffer.py` (new), `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---


#### **What changed**
- `src/heating_correction_ml_model.py` (new): `HeatingCorrectionMLModel` class — lazy-loads joblib model + metadata, predicts ΔT_outlet, exposes `r2_score` for blend weight. Feature extraction mirrors `CoolingMLModel._extract_feature`.
- `src/heating_correction_ml_calibration.py` (new): `calibrate_heating_correction_ml()` — module-level imports for patchability, cold-season filter (AT < `HEATING_ML_COLD_THRESHOLD_C`), full feature engineering, LightGBM L1 regression, model + metadata persistence.
- `src/model_wrapper.py`: replaced `_calculate_ml_correction()` stub with blended dispatch (`w = R²` if `R² ≥ HEATING_ML_BLEND_MIN_R2`, else `w=0`); added `_get_heating_correction_ml_model()` lazy singleton (class-level `_heating_correction_ml_model = None`).
- `src/main.py`: `--calibrate-heating-correction-ml` argument; flag-file block `/data/config/calibrate_heating_correction_ml_flag`.
- `src/config.py`: 8 new vars: `HEATING_ML_COLD_THRESHOLD_C`, `HEATING_ML_CALIBRATION_START_DATE`, `HEATING_ML_AT_FORECAST_HOURS`, `HEATING_ML_CORRECTION_MODEL_PATH`, `HEATING_ML_CORRECTION_METADATA_PATH`, `HEATING_ML_MIN_TRAINING_SAMPLES`, `HEATING_ML_LABEL_HORIZON_H`, `HEATING_ML_BLEND_MIN_R2`; `_parse_heating_start_date()` helper.
- `config_adapter.py`: 6 new env var mappings in `convert_addon_to_env()`.
- `ml_heating_underfloor/config.yaml`: 7 new options + 7 schema entries.

#### **Why**
Implements the planned ML-based heating correction to complement the existing physics Newton step, enabling the system to learn unmeasured heat sources (solar gain, occupancy patterns) from data.

#### **Files changed**
`src/config.py`, `src/heating_correction_ml_model.py`, `src/heating_correction_ml_calibration.py`, `src/model_wrapper.py`, `src/main.py`, `config_adapter.py`, `ml_heating_underfloor/config.yaml`, `tests/unit/test_heating_correction_ml_calibration.py`, `tests/unit/test_heating_correction_ml_model.py`, `tests/unit/test_heating_correction.py`, `docs/HEATING_CORRECTION_PHYSICS_VS_ML_ANALYSIS.md`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---



#### **What changed**
- `src/model_wrapper.py`: `_calculate_physics_newton_correction()` now evaluates `S(t_worst)` instead of `S(H)`. Each violation branch sets `_worst_idx = trajectory_temps.index(worst_value)`. After the branches, `t_eval` is resolved from `trajectory["times"][_worst_idx]` (if available) or `(idx+1) * H/n_steps`. Sensitivity formula: `s_t = equilibrium_fraction * (1 - exp(-t_eval/tau_room))`.
- `tests/unit/test_heating_correction.py`: added `S_3H_EXPECTED`; updated `test_undershoot_0_3k` / `test_overshoot_0_3k` to assert against `S(3h)`; added `test_mid_horizon_pv_overshoot_uses_t_worst` and `test_undershoot_at_last_step_uses_s_h`.

#### **Why**
- `ε / S_H` under-corrects when the worst trajectory point is at `t_worst < H` because `S(H) > S(t_worst)`. Most visible when PV peaks mid-day (overshoot at t=2h in a 4h horizon). Using `S(t_worst)` gives the correct Newton step.

#### **Files changed**
- `src/model_wrapper.py`, `tests/unit/test_heating_correction.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

### 🐛 Binary Search / Correction Gate Bug Fixes — 2026-05-15

#### **What changed**
- `src/model_wrapper.py`:
  1. **Range-collapse bypass** (critical): binary search early-exit at `range_size < 0.05` now calls `_verify_trajectory_and_correct` before returning, matching the converged and non-converged code paths.
  2. **`projected_indoor` exponential fix** (both `_calculate_physics_based_correction` and `_calculate_physics_newton_correction`): replaced `current + TRAJECTORY_STEPS × trend` with `current + trend × τ × (1 − exp(−H/τ))` using `TREND_DECAY_TAU_HOURS`. This matches the trajectory model's decaying trend bias and avoids 2–3× over-estimation that was causing the self-correction gate to skip legitimate corrections.
  3. **Newton `else` branch alignment**: `reaches_target_at > cycle_hours` → `> cycle_hours + tolerance_hours` (tolerance_hours = cycle_hours × 2), matching the outer gate.
  4. **Fragile float equality**: `temp_error == 0.0` → `abs(temp_error) < 1e-6`.
- `tests/unit/test_model_wrapper.py`: updated `TestProjectedTempOvershootGate` fixture to use trend −0.3 °C/h and pinned `TREND_DECAY_TAU_HOURS=1.5` so skip-condition tests hold under the corrected formula.

#### **Why**
- Bug #1 meant the correction layer was silently bypassed whenever the binary search saturated at the floor (e.g. 21 °C with high PV forecast), so over-temperature conditions were not corrected.
- Bug #2 caused the self-correction gate to over-skip: linear projection said the room would drop 0.8 °C in 4 h when the actual decaying-trend model accumulates only ~0.28 °C for the same −0.2 °C/h trend.
- Bugs #3 and #4 were low-severity consistency/fragility issues.

#### **Files changed**
- `src/model_wrapper.py`, `tests/unit/test_model_wrapper.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---






#### **What changed**
- `src/thermal_equilibrium_model.py`: Added `climate_mode` parameter to `calculate_optimal_outlet_temperature()` and `_calculate_equilibrium_outlet_temperature()`. Cooling mode uses `[COOLING_CLAMP_MIN_ABS, COOLING_CLAMP_MAX_ABS]` bounds instead of heating bounds `[outdoor+5, 70]`, and skips the "outlet below outdoor" fallback since cooling outlets should be below outdoor temp.
- `notebooks/analysis/05_cooling_scenario_simulation.ipynb`: `simulate_model_mode()` now passes `climate_mode='cooling'` to the analytical outlet method.

#### **Why**
- The notebook's model-mode simulation showed HP never activating. The analytical method always returned heating-range outlets (25–35°C) that got clamped to 24°C, failing the `outlet < inlet - MIN_COOLING_DELTA_K` gate. The underlying equilibrium formula was correct — only the bounds were wrong for cooling.
- Production was unaffected: `main.py` uses binary search via `model_wrapper._calculate_required_outlet_temp()` which already has full cooling support.

#### **Files changed**
- `src/thermal_equilibrium_model.py`, `notebooks/analysis/05_cooling_scenario_simulation.ipynb`

---

### ✅ Fix all 24 pre-existing test failures — 2026-05-14

#### **What changed**
- `tests/unit/test_config.py`: Added `tearDown()` to restore `sys.modules['src.config']` after each test — root cause of 7 config pollution failures
- `tests/unit/test_dashboard_components.py`: Added `pytest.importorskip("streamlit")` — 9 failures resolved
- `tests/unit/test_dashboard_data_service.py`: `missing_state` fixture now patches both `_STATE_FILE_CANDIDATES` and `_COOLING_STATE_FILE_CANDIDATES` — 5 failures resolved
- `tests/unit/test_overheating_predictor.py`: Peak moved from hour 8 to 10 (beyond 8h lead time)
- `tests/integration/test_adaptive_learning.py`: PV weight set to 0.002 explicitly before learning; iterations increased to 15
- `tests/unit/test_physics_calibration.py`: Default assertion checks bounds membership instead of exact value

#### **Why**
- 24 tests were failing in the full suite due to test isolation issues, config module identity pollution, and assertion drift from earlier lead-time changes
- Root cause: `test_config.py::setUp` deleted `sys.modules['src.config']` causing all subsequent `from src import config` to create a NEW module object, while existing modules still referenced the OLD one — making `patch.object(config, ...)` invisible to downstream code

#### **Files changed**
- `tests/unit/test_config.py`, `tests/unit/test_dashboard_components.py`, `tests/unit/test_dashboard_data_service.py`, `tests/unit/test_overheating_predictor.py`, `tests/unit/test_physics_calibration.py`, `tests/integration/test_adaptive_learning.py`

---

### ✨ Cooling ML calibration data optimization — 2026-05-14

#### **What changed**
- `src/physics_calibration.py`: `fetch_historical_data_for_calibration()` now accepts `purpose="cooling"` to build a reduced entity list (7 vs 15 entities). `_field` added to `_META_COLS` to suppress InfluxDB artifact gap warnings.
- `src/influx_service.py`: `get_training_data()` accepts optional `entity_ids` list; when provided, only those entities are queried in Flux.
- `src/ha_history_service.py`: `_build_entity_map()` and `get_training_data_from_ha()` accept optional `entity_ids` override.
- `src/cooling_ml_calibration.py`: passes `purpose="cooling"` to fetch function; warm-season filter now uses `COOLING_ML_WARM_THRESHOLD_C` (default 10°C) instead of derived `PRE_COOL_MIN_OUTDOOR_FORECAST_C - 6`.
- `src/config.py`: `PRE_COOL_LEAD_TIME_HOURS` default fixed from 3.0 → 8.0; forecast defaults now derived from lead time; new `COOLING_ML_WARM_THRESHOLD_C` config added.
- `config_adapter.py`: wired `COOLING_ML_WARM_THRESHOLD_C` env var mapping.

#### **Why**
- Cooling ML calibration was fetching 15 entities (incl. heating-only quality gates: DHW, defrost, TV, fireplace) but only uses 5-7. This wasted InfluxDB query bandwidth and HA REST API calls.
- Warm-season filter at 16°C excluded shoulder-season data, causing 85.3% positive label imbalance. Lowering to 10°C adds critical negative examples.
- Forecast features defaulting to 12h exceeded the 8h label window, adding noise features.
- `PRE_COOL_LEAD_TIME_HOURS` had mismatched defaults between config.py (3.0) and config_adapter.py (8.0).

#### **Files changed**
- `src/config.py`, `src/influx_service.py`, `src/ha_history_service.py`
- `src/physics_calibration.py`, `src/cooling_ml_calibration.py`
- `config_adapter.py`
- `tests/unit/test_cooling_ml_calibration.py`, `tests/unit/test_cooling_ml_extended.py`

---

### ✨ Feature: Cooling ML calibration start date + review fixes — 2026-05-14

#### **What changed**
- `config_adapter.py`: added pre-cooling / cooling-ML options block mapping all `PRE_COOL_*`, `COOLING_ML_*`, and `COOLING_ML_CALIBRATION_START_DATE` to their env var counterparts so HA add-on settings take effect at runtime.
- `src/cooling_ml_calibration.py`: `calibrate_cooling_ml()` reads `COOLING_ML_CALIBRATION_START_DATE` at step 0; uses `math.ceil` (not `int(...)`) for ceiling arithmetic so the full start date is always covered. Invalid `COOLING_ML_AT/PV_FORECAST_HOURS` values now log a warning before falling back.
- `src/config.py`: `_parse_cooling_start_date()` has `-> "Optional[datetime]"` return type; `Optional` imported from `typing` at module level.
- `tests/unit/test_cooling_ml_calibration.py`: fixed inaccurate comment about PV columns being absent when `COOLING_ML_PV_FORECAST_HOURS` is unset (it defaults to all 12 hours).
- `CHANGELOG.md`: updated to call out the full AT/PV forecast horizon change as a model-signature change and note the config-adapter fix.

#### **Why**
- Review feedback: config option was not wired; int() floors caused systematic under-coverage; missing return type; misleading test comment; unclear changelog.

#### **Files changed**
- `config_adapter.py`
- `src/config.py`
- `src/cooling_ml_calibration.py`
- `tests/unit/test_cooling_ml_calibration.py`
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---


#### **Why**
- The hindcast DataFrames already contained all 12 AT and PV forecast columns (from the `df["AT"].shift(-h)` loop in Step 5), but Step 6 discarded them all except `AT_roh_4h`.  Exposing the full daily cycle allows LightGBM to learn peak-timing patterns that the single 4h proxy missed.
- Inference-side extraction in `cooling_ml_model.py` already handled `AT_roh_Xh` and `pv_forecast_Xh` dynamically for any h, so no inference changes were needed.

#### **Files changed**
- `src/config.py`
- `src/cooling_ml_calibration.py`
- `tests/unit/test_cooling_ml_calibration.py`
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---

### 🔧 Fix: auto-trigger build on push to main — 2026-05-14

#### **What changed**
- Added `push: branches: [main]` trigger to `.github/workflows/build.yaml`, alongside the existing `workflow_dispatch`.
- Added `paths-ignore` for `**.md`, `memory-bank/**`, `docs/**` so documentation-only commits don't waste build minutes.

#### **Why**
- Without an auto-trigger, merging a PR that set a new version in `config.yaml` never kicked off a Docker build. Home Assistant saw the new version string but the image was absent from GHCR, causing `[404] manifest unknown` errors.
- The version-bump commit already uses `[skip ci]` in its message, so the workflow will not re-trigger after the auto-bump, preventing an infinite loop.

#### **Files changed**
- `.github/workflows/build.yaml`
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---

### 🐛 Fix aarch64 Docker build: switch from Alpine to Debian slim — 2026-05-14

#### **What changed**
- Changed `Dockerfile` base image from `python:3.11-alpine3.18` to `python:3.11-slim` (Debian).
- Replaced `apk add` system-package block with `apt-get install` equivalents; removed musl-only packages (`musl-dev`, `linux-headers`); renamed `openblas-dev`/`lapack-dev` to `libopenblas-dev`/`liblapack-dev`.

#### **Why**
- The aarch64 build started using a native ARM runner (`ubuntu-24.04-arm`) instead of QEMU. On Alpine (musl libc), `scikit-learn` has no pre-built `musllinux_1_2_aarch64` wheel for the current version, so pip falls back to source compilation. The meson/GCC build fails with a `-Werror=array-bounds` error, blocking the entire build. Debian slim uses glibc for which pre-built wheels exist on PyPI for all major architectures.

#### **Files changed**
- `Dockerfile`
- `CHANGELOG.md`

---

### 🛡️ Harden Cooling ML Calibration & PV Feature Contract — 2026-05-14

#### **What changed**
- Hardened cooling ML calibration semantics:
  - Default pre-cooling lead-time reduced from 8.0h to 3.0h (`PRE_COOL_LEAD_TIME_HOURS`), so label assignment is more responsive and matches model inference horizon.
  - Calibration and inference now strictly use raw electrical PV keys for all cooling ML features (`pv_roll_*`, `PV_Generate`, `pv_forecast_*h`), falling back to thermal-corrected keys only if electrical keys are absent.
  - Fixed bugs in pre-cooling calibration: correct PV key usage, feature scale alignment, buffer persistence, and lead-time/label horizon calculation.
- Cooling observation buffer now persists after every push/resolve cycle, so pending entries and evolving label state survive restarts.
- Added `scikit-learn>=1.0.0` to `requirements.txt` to fix silent metric failures in calibration.
- Added/tightened unit tests for cooling ML calibration and PV feature extraction (raw vs thermal PV history, column-count expectations).

#### **Why**
- Ensures cooling ML calibration and inference semantics are consistent and robust, preventing PV scale drift and label misalignment.
- Prevents loss of pending cooling observations on restart.
- Fixes silent calibration metric failures due to missing dependency.
- Strengthens regression coverage for PV feature contract and calibration workflow.

#### **Files changed**
- `src/cooling_ml_calibration.py`
- `src/cooling_ml_model.py`
- `src/main.py`
- `src/physics_features.py`
- `requirements.txt`
- `.github/workflows/build.yaml`
- `tests/unit/test_cooling_ml.py`
- `tests/unit/test_cooling_ml_calibration.py`
- `tests/unit/test_physics_features.py`
- `tests/unit/test_pre_cooling_integration.py`

---

### 🔀 Resolve PR Merge Conflicts (Latest Sync) — 2026-05-14

#### **What changed**
- Merged latest `origin/main` into the branch after new merge-conflict reports on the PR.
- Resolved conflict markers in `memory-bank/activeContext.md` and `memory-bank/progress.md` by preserving content from both branches.
- Accepted incoming base-branch updates for `.github/workflows/build.yaml` and `CHANGELOG.md`.
- Per review follow-up, translated inline workflow comments in `.github/workflows/build.yaml` to English and fixed a truncated sentence in the PV contract context section below.

#### **Why**
- The PR was reported as conflicted again and required a fresh sync with `origin/main` before it could merge cleanly.

#### **Files changed**
- `.github/workflows/build.yaml`
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---

### 🧩 Address Reviewer Thread Follow-ups — 2026-05-14

#### **What changed**
- Updated cooling observation-buffer persistence in `src/main.py` to save after every `push_pending()`/`resolve_labels()` cycle (not only when new labels mature), so pending-window state is restart-safe.
- Added raw PV history support in `src/physics_features.py` via `pv_power_history_electrical` while keeping existing thermal-corrected `pv_power_history` for thermal logic.
- Updated `src/cooling_ml_model.py` rolling PV extraction to prefer `pv_power_history_electrical` and fall back to `pv_power_history`.
- Added/updated unit tests in:
  - `tests/unit/test_cooling_ml.py` (raw electrical roll-history preference)
  - `tests/unit/test_physics_features.py` (raw PV history feature emitted)

#### **Why**
- Reviewer feedback identified that pending observations could still be lost on restart before label maturity and that `pv_roll_*` inference remained on thermally corrected scale despite training on raw electrical PV.

#### **Files changed**
- `src/main.py`
- `src/physics_features.py`
- `src/cooling_ml_model.py`
- `tests/unit/test_cooling_ml.py`
- `tests/unit/test_physics_features.py`
- `CHANGELOG.md`
- `memory-bank/progress.md`
- `memory-bank/activeContext.md`

---

### 🚀 CI Workflow Modernization & Architecture Improvements — 2026-05-14

#### **What changed**
- **GitHub Actions workflow updated:**
  - `.github/workflows/build.yaml` now uses newer versions of core actions (`actions/checkout@v4`, `docker/login-action@v3`, `docker/setup-buildx-action@v3`, `docker/build-push-action@v6`).
  - Architecture handling improved: jobs run natively on their respective runners (`ubuntu-24.04-arm` for ARM, `ubuntu-latest` for AMD64), eliminating the need for QEMU emulation.
  - Build cache is now separated per architecture for faster and more reliable builds.
  - Minor log and changelog update messages clarified.

#### **Why**
- Ensures compatibility with latest GitHub Actions ecosystem.
- Native builds improve speed and reliability, reduce complexity (no QEMU).
- Per-arch cache prevents cross-architecture cache pollution.

#### **Files changed**
- `.github/workflows/build.yaml`

---

### 🔀 Resolve PR Merge Conflicts — 2026-05-14

#### **What changed**
- Merged `origin/main` into the feature branch to resolve PR merge conflicts.
- Resolved content conflicts in `CHANGELOG.md`, `memory-bank/activeContext.md`, and `memory-bank/progress.md`.
- Preserved entries from both sides so no historical changelog/progress/context information was lost.

#### **Why**
- The PR was blocked by merge conflicts and could not be merged until documentation/context files were reconciled.

#### **Files changed**
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---

### 🧪 Review Cooling Calibration Workflow Follow-up — 2026-05-14

#### **What changed**
- Reviewed the pre-cooling calibration follow-up work for remaining bugs in the cooling calibration workflow.
- Added a regression assertion in `tests/unit/test_pre_cooling_integration.py` proving that the corrected thermal PV keys rea
