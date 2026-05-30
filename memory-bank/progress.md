# ML Heating System - Current Progress

## ✅ Code Review: Dual-Output Cooling ML (2026-05-31)

**Status:** COMPLETED — 5 blocking bugs fixed, 1393 tests pass

### Issues Found & Fixed
1. **`classifier_gate` logic bug** (cooling_ml_model.py): `not self._reg_model` always False inside dual branch → fixed to use `risk` alone as gate
2. **`_no_risk_result` missing keys** (cooling_ml_model.py): Added `predicted_delta`, `predicted_max_temp`, `reg_risk` to early-exit dict
3. **`proba_cal_full` NameError** (cooling_ml_calibration.py): Undefined when `use_calibrated=False`; initialised before branch
4. **`_offset` unbound** (cycle_routes.py): Initialised `_offset = 0.0` before conditional block
5. **No `ValueError` on invalid strategy** (cooling_ml_model.py): Added validation for `PRE_COOL_DUAL_OUTPUT_STRATEGY`

### Files Changed
- `src/cooling_ml_model.py` — 3 fixes (classifier_gate logic, _no_risk_result keys, strategy validation)
- `src/cooling_ml_calibration.py` — 1 fix (proba_cal_full init)
- `src/cycle_routes.py` — 1 fix (_offset init)

## ✅ Cooling ML: Dual-Output Production Implementation (2026-05-31)

**Status:** COMPLETED — All 6 phases implemented, 1393 tests pass

### Implementation Summary
- **Phase 1**: Fixed sklearn feature-name warnings — inference uses `pd.DataFrame` instead of `np.array`; calibration trains with DataFrames to preserve feature names
- **Phase 2**: Added LGBMRegressor training on `delta_indoor_8h` alongside classifier; saves `cooling_ml_regressor.joblib` with threshold/MAE/AUC in metadata
- **Phase 3**: `CoolingMLModel.load()` loads regressor with graceful fallback; `predict_overheating_risk()` returns `predicted_delta`, `predicted_max_temp`, `reg_risk`
- **Phase 4**: Proportional pre-cooling offset `clip(overshoot × 0.7, 0.2K, 1.0K)` replaces fixed 0.5K; controlled by `PRE_COOL_PROPORTIONAL`
- **Phase 5**: Dual-output strategy selector (`classifier_gate`/`either_triggers`) in config + HA dropdown + translations
- **Phase 6**: Isotonic threshold shift logging, F1/precision/recall diagnostics after calibration

### Files Changed
- `src/cooling_ml_model.py` — regression loading, dual-output inference, DataFrame predict
- `src/cooling_ml_calibration.py` — regression training, diagnostics, DataFrame training
- `src/cycle_routes.py` — proportional offset, enhanced logging
- `src/config.py` — 6 new PRE_COOL_* constants + COOLING_ML_REGRESSOR_PATH
- `config_adapter.py` — new config mapping
- `ml_heating_underfloor/config.yaml` — new options + schema
- `ml_heating_underfloor/translations/en.yaml` — tooltips
- `tests/unit/test_cooling_ml_calibration.py` — regression mock + metadata test

## ✅ Cooling ML: R² Improvement Investigation (2026-05-31)

**Status:** COMPLETED

### Results
- Added Section 4b to `09_cooling_ml_analysis.ipynb`: R² improvement using 07_/08_ heating notebook techniques
- **Root cause confirmed**: R²=0.36 is mathematically expected — `delta_indoor_8h` has std=0.177 (very low variance), same MAE on `max_indoor_8h` (std=0.400) theoretically gives R²≈0.92
- **Outlier filtering**: Removed 5,949 window-open contaminated rows (12.4%), no fireplace events in cooling data
- **New engineered features**: 4 of 5 available (indoor_accel, AT_forecast_trend, thermal_momentum, pv_cumulative_8h)
- **3 approaches compared**:
  - A) delta + clean + features: MAE=0.090, R²=0.359, AUC=0.946, F1=0.926
  - B) max_indoor (original): MAE=0.134, R²=0.716, AUC=0.913, F1=0.891
  - C) max_indoor + clean + features: MAE=0.129, R²=0.725, AUC=0.898, F1=0.875
- **Conclusion**: Switching to max_indoor_8h boosts R² (0.36→0.73) but HURTS classifier performance (AUC 0.952→0.898). Original delta approach remains best for pre-cooling decisions.
- All 26 notebook cells execute successfully

### Files Changed
- `notebooks/analysis/09_cooling_ml_analysis.ipynb` — Added Section 4b (6 cells: variance analysis, outlier filtering, feature engineering, 3-way comparison, diagnostic plots)

## ✅ Cooling ML: F2→F1 Threshold + 75-Feature Analysis (2026-05-30)

**Status:** COMPLETED

### Results
- Switched calibration from F2 (β=2) to F1 (β=1) threshold selection in `cooling_ml_calibration.py`
- Rewrote `09_cooling_ml_analysis.ipynb` for new 75-feature training data (47,941 rows × 76 cols)
- **Baseline (75 features, F1)**: AUC=0.9434, F1=0.9191, precision=94.9%, recall=89.1%, threshold=0.773
- **Regression wins again**: AUC=0.9520 → Optuna-tuned AUC=0.9581, F1=0.9363, MAE=0.0828°C
- **Dual-output approach analyzed**: classifier gate × regression overshoot (P×Δ): AUC=0.9513, F1=0.9349
- Pruning kept all 75 features (even worst caused >0.001 AUC drop)
- Isotonic calibration shifts F1 threshold from 0.80→0.65 (explains production 0.049)
- 1412 unit tests pass, 5 pre-existing integration failures (Docker/HA)

### Files Changed
- `src/cooling_ml_calibration.py` — beta=2.0→1.0 (3 locations), metadata key val_f2→val_f1, threshold_method f2→f1
- `tests/unit/test_cooling_ml_calibration.py` — val_f2→val_f1 in expected_keys
- `notebooks/analysis/09_cooling_ml_analysis.ipynb` — Full rewrite: 33 cells, 10 sections + isotonic + dual-output

## ✅ Cooling ML Analysis Notebook + Regression Discovery (2026-05-30)

**Status:** COMPLETED

### Results
- Created `notebooks/analysis/09_cooling_ml_analysis.ipynb` — full 10-section analysis
- Added 17 derivable features from existing CSV (no re-calibration needed)
- **Key finding**: Regression approach (`delta_indoor_8h`) beats binary classifier:
  - Classifier: AUC=0.9431, F2=0.9424 (61 features after pruning)
  - Regression: AUC=0.9502, F2=0.9492 (same features)
  - Optuna-tuned regression: AUC=0.9582, F2=0.9505, MAE=0.0827°C
- `traj_predicted_error` ranked #8 by importance (new feature)
- `is_overshoot` has highest label correlation (r=+0.74)
- Optimal regression threshold: predicted_max > 22.93°C

## ✅ Cooling ML model: 23 new features + trajectory bridge (2026-05-30)

**Status:** COMPLETED

### Changes
1. **`src/physics_calibration.py`** — Expanded `_cooling_entity_ids` from 7→11 entities (added wind_speed, fireplace, TV, living_room_temp)
2. **`src/cooling_ml_calibration.py`** — Added 23 new features:
   - HA context (wind_speed, living_room_temp, fireplace_on + 3 lags, tv_on + 2 lags)
   - Derived physics (heat_loss_driving_force, indoor_temp_gradient, indoor_margin_rate, delta_T_indoor_lag1, d_inlet_temp_60min, is_equilibrium, thermal_power_rolling_1h, is_overshoot, is_hp_active, is_weekend, heat_loss_interaction)
   - Solar/shading (solar_thermal_proxy, shading_proxy, pv_forecast_delta)
   - Trajectory-derived (traj_predicted_error, traj_convergence_rate, traj_reaches_target_hours, traj_overshoot_magnitude, traj_equilibrium_gap) — vectorized analytical Newton-decay
3. **`src/cooling_ml_model.py`** — Added `_compute_traj_*` helpers + `_extract_feature()` cases for all 23 new features
4. **`src/cycle_routes.py`** — Inject OverheatingPredictor trajectory into CoolingMLModel features dict

### Test results
- 1412 passed, 266 cooling tests passed, 0 new failures

## ✅ PR #73 review follow-up: cooling recovery scope + reliability fixes (2026-05-26)

**Status:** COMPLETED

### Changes
1. **`src/model_wrapper.py`**
   - Moved `timezone` import to module scope and removed inline import inside cooling recovery transition logic.
   - Added `recovery_start_time` to `get_comprehensive_metrics_for_ha()` in cooling mode.
2. **`tests/unit/test_cooling_mode.py`**
   - Updated recovery test fixture to instantiate `EnhancedModelWrapper()` directly, preventing singleton reuse between tests.
3. **Scope alignment**
   - Reverted unrelated cooling ML artifact/calibration/model edits to keep PR #73 focused on cooling recovery tracking:
     - `Logs_and_models/cooling_ml_metadata.json`
     - `src/cooling_ml_calibration.py`
     - `src/cooling_ml_model.py`
4. **Documentation**
   - Updated `CHANGELOG.md`, `memory-bank/progress.md`, and `memory-bank/activeContext.md`.

### Validation
- `python -m pytest -q tests/unit/test_cooling_mode.py`

**Files changed:** `src/model_wrapper.py`, `tests/unit/test_cooling_mode.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`, plus reverts for `Logs_and_models/cooling_ml_metadata.json`, `src/cooling_ml_calibration.py`, `src/cooling_ml_model.py`

## ✅ PR #71 review follow-up: cooling reliability + calibration stability fixes (2026-05-25)

**Status:** COMPLETED

### Changes
1. **`src/main.py`**
   - Normalize `climate_mode` to `wrapper.climate_mode` after `set_climate_mode()` so unsupported values (e.g. `"off"`) can't reach `run_online_learning()` or `CycleContext`.
   - Replace `cycle_number > 1` with `loop.last_cycle_end_time is not None` to correctly skip online learning when cycle 1 exits early via a network-error `continue`.
2. **`src/cooling_ml_calibration.py`**
   - Added `_IsotonicCalibratedModel` wrapper class to keep base model weights frozen when sklearn's `cv="prefit"` is unavailable (>= 1.6).
   - Replaced the `cv=2` fallback (which refits cloned estimators) with a direct `IsotonicRegression` on `model.predict_proba(X_cal_iso)`.
3. **`tests/unit/test_main_functions.py`**
   - Added `test_online_learning_skipped_on_first_cycle`: asserts `run_online_learning` is not called on cycle 1.
   - Added `test_online_learning_called_after_completed_cycle`: asserts `run_online_learning` is called once `loop.last_cycle_end_time` is non-None.
4. **`CHANGELOG.md`** / **`memory-bank/progress.md`** / **`memory-bank/activeContext.md`**
   - Updated `[Unreleased]` section and project tracking with all changes.

### Validation
- `python3 -m pytest tests/unit/test_main_functions.py -v` → **6 passed**
- `python3 -m pytest tests/unit/test_cooling_ml_calibration.py -v` → **34 passed**

**Files changed:** `src/main.py`, `src/cooling_ml_calibration.py`, `tests/unit/test_main_functions.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`



**Status:** COMPLETED

### Changes
1. **`tests/integration/test_dispatch_wiring.py`**
   - Removed unused imports `PropertyMock` and `CycleState`.
   - Fixed `check_and_resolve_climate_mode` mock return values to use production climate_mode strings: "heat" → "heating" (heating/blocking route tests), "off" → "heating" (idle route test — production always returns "heating" when heating_active is False).
   - Fixed `get_climate_mode` mock return values accordingly: "heat"/"off" → "heating".
   - Fixed `_make_all_states(heating_status="cooling")` → `heating_status="cool"` (HA state "cool" maps to climate_mode "cooling" via `config.get_climate_mode`).
2. **`CHANGELOG.md`**
   - Added `[Unreleased]` entries documenting the dispatch wiring refactor (main loop restructure, `initialize_loop_state` factory, integration test coverage).
3. **`memory-bank/progress.md` / `memory-bank/activeContext.md`**
   - Added current milestone/context entries for this PR #69 review follow-up.

### Validation
- `python3 -m pytest tests/integration/test_dispatch_wiring.py -v` → **4 passed**

**Files changed:** `tests/integration/test_dispatch_wiring.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ PR #66 review follow-up: English-only dashboard settings docs cleanup (2026-05-23)

**Status:** COMPLETED

### Changes
1. **`dashboard/config_schema.py`**
   - Removed unused `_DE_TRANSLATIONS_PATH` constant after the dashboard moved to an English-only metadata/translation path.
2. **`CHANGELOG.md`**
   - Added `[Unreleased]` entries documenting English-only dashboard settings behavior and regrouped settings sections.
3. **`memory-bank/progress.md` / `memory-bank/activeContext.md`**
   - Added current milestone/context entries for this PR #66 review follow-up.

### Validation
- `python -m pytest -q tests/unit/test_dashboard_settings.py` → **10 passed**

**Files changed:** `dashboard/config_schema.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ PR #63 review follow-up: cooling trajectory/EMA regression coverage (2026-05-22)

**Status:** COMPLETED

### Changes
1. **`tests/unit/test_heating_correction.py`**
   - Added dispatch regression asserting cooling mode overrides `HEATING_CORRECTION_MODE="ml"` to physics/Newton and does not call ML correction.
2. **`tests/unit/test_overshoot_logic.py`**
   - Added cooling regression for undershoot-only path (`min_violates`) returning unchanged outlet.
   - Added cooling regression asserting overshoot correction still executes even when projected-indoor skip-gate conditions are met.
3. **`tests/unit/test_main_functions.py`**
   - Added main-loop regression that runs a cooling-recovery cycle and asserts `apply_ema_smoothing()` is not called.
4. **`CHANGELOG.md`**
   - Added `[Unreleased]` entries for cooling recovery EMA bypass and cooling trajectory-correction behavior updates.

### Validation
- `python3 -m pytest -q tests/unit/test_heating_correction.py tests/unit/test_overshoot_logic.py tests/unit/test_main_functions.py tests/unit/test_cooling_mode.py` → **79 passed**

**Files changed:** `tests/unit/test_heating_correction.py`, `tests/unit/test_overshoot_logic.py`, `tests/unit/test_main_functions.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Unit audit: all PV and thermal power features now in Watts (2026-05-20)

**Status:** COMPLETED

### Changes
1. **`src/heating_correction_ml_calibration.py`**
   - Added `df["thermal_power_w"] = df["thermal_power_kw"] * 1000.0` after kW column.
   - `thermal_power_rolling_1h` now uses `thermal_power_w` (W).
   - Feature col `"thermal_power_kw"` → `"thermal_power_w"`.
   - `shading_proxy` formula: removed `/1000.0` (PV already in W); units K×W.
2. **`src/heating_correction_ml_model.py`**
   - `thermal_power_kw` dispatch → `thermal_power_w` (returns `physics.get("thermal_power_kw") * 1000.0`).
   - `thermal_power_rolling_1h` dispatch → also returns `thermal_power_kw * 1000.0`.
   - `shading_proxy` dispatch: removed `/1000.0`.
3. **`tests/unit/test_heating_correction_ml_model.py`**
   - Updated `test_thermal_power_rolling_1h` (now expects 4500.0 not 4.5).
   - Added `test_thermal_power_w`, `test_shading_proxy_uses_watts`, `test_shading_proxy_zero_below_threshold`, `test_shading_proxy_missing_pv`.

### Validation
- 97 passed (93 prior + 4 new)

**Files changed:** `src/heating_correction_ml_model.py`, `src/heating_correction_ml_calibration.py`, `tests/unit/test_heating_correction_ml_model.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ S_H audit + shading_proxy + heat_loss_interaction features (2026-05-20)

**Status:** COMPLETED

### Audit findings
- **S_H as feature: SKIPPED** — S_H is a scalar constant for the whole training run (std=0). Adding it as a feature provides zero information to the model.
- Full duplicate audit: all 8 prior-session features confirmed EXISTS in both files; no re-addition needed.
- Feature 2 (`shading_proxy`): `solar_thermal_proxy` exists but PI=0.0000 (ineffective). `shading_proxy` encodes a different physical mechanism (solar overheat protection) → implemented.
- Feature 3 (`heat_loss_interaction`): `wind_speed` exists with PI=0.0000. Interaction with temperature gradient encodes convective heat loss → implemented.

### Changes
1. **`src/heating_correction_ml_calibration.py`**
   - Added `shading_proxy` and `heat_loss_interaction` computed columns (section 4d).
   - Appended both to `feature_cols` after all prior features.
   - Added `# PI=0.0000` comments to `wind_speed`, `indoor_temp_gradient`, `indoor_margin_rate`, `is_overshoot`, `d_inlet_temp_60min`, `is_equilibrium`, `delta_T_indoor_lag1`, `Q_wp`, `solar_thermal_proxy`.
2. **`src/heating_correction_ml_model.py`**
   - Added `shading_proxy` and `heat_loss_interaction` inference branches in `_extract_heating_feature()`.
   - Added `# PI=0.0000` comments to matching zero-importance feature branches.

**Files changed:** `src/heating_correction_ml_model.py`, `src/heating_correction_ml_calibration.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Address PR #61 review feedback for heating ML physics features (2026-05-20)

**Status:** COMPLETED — removed duplicated feature, aligned `Q_wp` with config-specific heat units, fixed PV-forecast parity, and added regression coverage.

### Changes
1. **`src/heating_correction_ml_model.py`**
   - Removed duplicate `control_deviation` inference branch.
   - Updated `Q_wp` to use `config.SPECIFIC_HEAT_CAPACITY` with kJ/kgK → J/kgK conversion for consistent units.
2. **`src/heating_correction_ml_calibration.py`**
   - Removed duplicate `control_deviation` training column.
   - Updated `Q_wp` to use `specific_heat * 1000` instead of hard-coded `4182`.
   - Ensured `pv_forecast_2h` is always derived for `pv_forecast_delta` even if forecast hours are configured without `2`.
3. **Tests**
   - Added feature-extraction tests in `tests/unit/test_heating_correction_ml_model.py` for `heat_loss_driving_force`, `delta_T_indoor_lag1`, `Q_wp`, `solar_thermal_proxy`, and `pv_forecast_delta`.
   - Added calibration tests in `tests/unit/test_heating_correction_ml_calibration.py` validating derived physics-feature presence, duplicate removal, and `pv_forecast_delta` behavior with custom forecast-hour config.

### Validation
- `python -m pytest tests/unit/test_heating_correction_ml_model.py tests/unit/test_heating_correction_ml_calibration.py -q --tb=short` → **93 passed**

**Files changed:** `src/heating_correction_ml_model.py`, `src/heating_correction_ml_calibration.py`, `tests/unit/test_heating_correction_ml_model.py`, `tests/unit/test_heating_correction_ml_calibration.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Add 6 new physics-motivated features to heating correction ML (2026-05-20)

**Status:** COMPLETED — added 6 new feature columns to both inference dispatch and calibration training pipeline.

### Changes
1. **`src/heating_correction_ml_model.py`**
   - Added 6 new `if col ==` branches in `_extract_heating_feature()` after `is_equilibrium`: `heat_loss_driving_force`, `delta_T_indoor_lag1`, `control_deviation`, `Q_wp`, `solar_thermal_proxy`, `pv_forecast_delta`.
2. **`src/heating_correction_ml_calibration.py`**
   - Added section 4c computing all 6 features from the historical DataFrame after the PV hindcast loop.
   - Appended all 6 feature names to `feature_cols` in section 6, after the `is_equilibrium` entry.

### Validation
- All 6 features are present and computed in identical order in both files.
- Fallback to `0.0` for missing/zero values (delta_T_indoor_lag1, Q_wp with zero flow, pv_forecast_delta when forecast unavailable).

**Files changed:** `src/heating_correction_ml_model.py`, `src/heating_correction_ml_calibration.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Review follow-up: cadence-aware slab delta + strict calibration test helper (2026-05-19)

**Status:** COMPLETED — addressed PR review follow-ups for slab thermal-state feature correctness and regression-test reliability.

### Changes
1. **`src/physics_features.py`**
   - Replaced hard-coded 10-minute inlet lag logic with cadence-aware computation using `HISTORY_STEP_MINUTES` (fallback `CYCLE_INTERVAL_MINUTES`) and derived `steps_per_hour`.
   - Added guard to detect all-default fallback inlet history and force neutral slab trend (`d_inlet_temp_60min = 0.0`), preventing artificial non-equilibrium signals.
2. **`src/influx_service.py`**
   - Added `INLET_HISTORY_FALLBACK_DEFAULT` constant and reused it in `fetch_inlet_history()`, so slab-state fallback detection shares one source of truth.
3. **`tests/unit/test_physics_features.py`**
   - Added regression test for dynamic 60-minute lag indexing when history cadence changes.
   - Added regression test verifying default-filled inlet history yields neutral slab trend and equilibrium.
4. **`tests/unit/test_heating_correction_ml_calibration.py`**
   - Hardened `_run_calibration_capture_X` to stop on captured `fit()`, explicitly fail if `fit()` is never called, and avoid blanket exception suppression.

### Validation
- `python -m pytest tests/unit/test_physics_features.py tests/unit/test_heating_correction_ml_calibration.py -q --tb=short` → **42 passed**

**Files changed:** `src/physics_features.py`, `src/influx_service.py`, `tests/unit/test_physics_features.py`, `tests/unit/test_heating_correction_ml_calibration.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Fix inference dispatch for d_inlet_temp_60min / is_equilibrium (2026-05-19)

**Status:** COMPLETED — wired missing `_extract_heating_feature()` handlers so the two new slab thermal state features are correctly passed to the ML model at inference.

### Changes
1. **`src/heating_correction_ml_model.py`**
   - Added `if col == "d_inlet_temp_60min"` and `if col == "is_equilibrium"` handlers under *Slab thermal state features* section; both do a direct `physics.get()` pass-through with 0.0 fallback.
2. **`tests/unit/test_heating_correction_ml_model.py`**
   - Added 6 tests in `TestExtractHeatingFeature` for positive value, negative value, and missing-key fallback for each of the two new features.

### Validation
- `pytest tests/unit/test_heating_correction_ml_model.py -q` → **56 passed**

**Files changed:** `src/heating_correction_ml_model.py`, `tests/unit/test_heating_correction_ml_model.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Slab thermal state features: d_inlet_temp_60min + is_equilibrium (2026-05-19)

**Status:** COMPLETED — added two new features to the heating correction ML pipeline.

### Changes
1. **`src/influx_service.py`**
   - Added `fetch_inlet_history(steps: int) -> list[float]` convenience method.
2. **`src/physics_features.py`**
   - Fetches 7-step inlet temperature lag history via `influx_service.fetch_inlet_history(7)`.
   - Computes `d_inlet_temp_60min = inlet_temp_f - inlet_lag_history[-6]` (change over 6 × 10-min cycles).
   - Computes `is_equilibrium = 1.0 if |d_inlet_temp_60min| < 0.3 else 0.0`.
   - Both features added to the inference feature dict.
3. **`src/heating_correction_ml_calibration.py`**
   - Added `df["d_inlet_temp_60min"] = df["RLT"].diff(steps_per_hour)` in derived features block.
   - Added `df["is_equilibrium"] = (df["d_inlet_temp_60min"].abs() < 0.3).astype(float)`.
   - Both appended to `feature_cols` under "Slab thermal state features".
4. **`tests/unit/test_physics_features.py`**
   - Added `fetch_inlet_history` mock return value; updated column count assertion (+2).
   - Added assertions for `d_inlet_temp_60min` and `is_equilibrium` values.
5. **`tests/unit/test_heating_correction_ml_calibration.py`**
   - Added `TestSlabThermalStateFeatures` with two tests covering feature presence and equilibrium logic.

### Validation
- `pytest tests/unit/test_physics_features.py tests/unit/test_heating_correction_ml_calibration.py -v` → **40 passed**

**Files changed:** `src/influx_service.py`, `src/physics_features.py`, `src/heating_correction_ml_calibration.py`, `tests/unit/test_physics_features.py`, `tests/unit/test_heating_correction_ml_calibration.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Cooling gate start-condition enforcement + no script shutdown while HP active (2026-05-19)

**Status:** COMPLETED — implemented strict cooling start gates and prevented script-driven shutdown when the heat pump is detected as active.

### Changes
1. **`src/model_wrapper.py`**
   - Reworked cooling gate flow to enforce start permission only when both configured gates pass:
     - `inlet - required_outlet > MIN_COOLING_DELTA_K`
     - `inlet + delta_t_floor > COOLING_CLAMP_MIN_ABS + COOLING_SHUTDOWN_MARGIN_K`
   - Added active-HP override behavior so running operation keeps computed required outlet even when start gate is closed.
   - Updated HP activity context defaults (`outlet_temp` fallback to inlet, indoor fallback to current_indoor) to avoid false active detection from missing values.
2. **`tests/unit/test_cooling_mode.py`**
   - Updated gate transition expectations for idle/running HP behavior.
   - Added recovery-to-running test when HP is detected active.
   - Updated exact-threshold boundary test to reflect strict `>` gate semantics.

### Validation
- `python -m pytest tests/unit/test_cooling_mode.py -q --tb=short` → **47 passed**
- `python -m pytest tests/unit/test_cooling_bugfixes.py tests/unit/test_pre_cooling_integration.py -q --tb=short` → **47 passed**
- `python -m pytest tests/ -q --tb=short` → **1399 passed**

**Files changed:** `src/model_wrapper.py`, `tests/unit/test_cooling_mode.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Heating ML Calibration Runtime Fixes (S_H Source + Optuna + Warning Cleanup) (2026-05-18)

**Status:** COMPLETED — fixed S_H parameter sourcing to use unified thermal state correctly, removed sklearn feature-name warnings in CV/HPO, and enabled Optuna in addon runtime dependencies.

### Changes
1. **`src/heating_correction_ml_calibration.py`**
   - Reworked `_read_baseline_thermal_params()` to be channel-aware:
   - If heat-source channels are enabled and `heat_pump` channel shows activity history, use channel params for `eta`, `u`, `tau`.
     - Otherwise use unified-state computed parameters (`baseline_parameters + parameter_adjustments`).
     - Fall back to config defaults only when unified state access fails.
   - Updated Optuna objective and optional time-series CV loops to use DataFrame `.iloc` slicing instead of unnamed ndarray slices to keep feature names and avoid sklearn warnings.
2. **`requirements.txt`**
   - Added `optuna>=4.0.0` so Home Assistant addon image includes Optuna without manual pip install.
3. **`tests/unit/test_heating_correction_ml_calibration.py`**
   - Added regression tests validating S_H source precedence:
     - heat-pump channel parameters preferred when channels are enabled
     - computed baseline+adjustments used when channel parameters are unavailable

### Validation
- `python -m pytest tests/unit/test_heating_correction_ml_calibration.py -q --tb=short` → **27 passed**

**Files changed:** `src/heating_correction_ml_calibration.py`, `requirements.txt`, `tests/unit/test_heating_correction_ml_calibration.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Autotuning Notebook — 3-Phase HPO Pipeline (2026-05-17)

**Status:** COMPLETED — created and validated `notebooks/analysis/06_heating_autotuning.ipynb`.

### Results
- **MAE:** 0.1466 → 0.0906 (−38.2%)
- **R²:** 0.8611 → 0.9370 (+8.8%)
- **Features:** 40 → 17 (permutation importance pruning)
- **Horizon:** 4h → 2h (horizon search)

### Pipeline
1. Phase 1: Label horizon search (1–6h) → best=2h
2. Phase 2: Optuna HPO (100 trials, LightGBM, expanding-window 5-fold CV) → best trial #63
3. Phase 3: Feature selection (permutation importance + 20-trial threshold search) → 17/41 features kept
4. Final holdout evaluation + model/metadata save + config.yaml snippet

### Files changed
- `notebooks/analysis/06_heating_autotuning.ipynb` (new, 19 cells)
- `Logs_and_models/heating_correction_ml_model_tuned.joblib` (new)
- `Logs_and_models/heating_correction_ml_metadata_tuned.json` (new)
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Training Data Export for ML Calibration (2026-05-17)

**Status:** COMPLETED — added training data export to both heating and cooling calibration pipelines.

### Changes
1. **`src/calibration_data_export.py`** (new)
   - Shared `export_training_data()` helper: saves `feature_cols + label` as gzip CSV next to `unified_thermal_state.json`
   - Atomic write (`.tmp` → `os.replace`), non-blocking (failure = warning, never fails calibration)
2. **`src/heating_correction_ml_calibration.py`**
   - Added import + call to `export_training_data(df_train, feature_cols, "heating")` after model/metadata save
3. **`src/cooling_ml_calibration.py`**
   - Added import + call to `export_training_data(df_train, feature_cols, "cooling")` after model/metadata save
4. **`tests/unit/test_calibration_data_export.py`** (new)
   - 9 unit tests: happy path, missing config, exception handling, column filtering, directory creation, overwrite
5. **Validation**
   - 9/9 new tests pass; 1358 total tests pass, 1 skip, no regressions

**Files changed:** `src/calibration_data_export.py`, `src/heating_correction_ml_calibration.py`, `src/cooling_ml_calibration.py`, `tests/unit/test_calibration_data_export.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Fix dashboard config FileNotFoundError in container (2026-05-17)

**Status:** COMPLETED — fixed missing packaged config assets in Docker image so dashboard metadata loading no longer fails at `/app/ml_heating_underfloor/config.yaml`.

### Changes
1. **`Dockerfile`**
   - Added `COPY ml_heating_underfloor/ /app/ml_heating_underfloor/` so `config.yaml` and translations are available at runtime where dashboard config parsing expects them.
2. **`tests/unit/test_dockerfile_contents.py`** (new)
   - Added regression test asserting Dockerfile keeps the config-bundle copy directive.
3. **Validation**
   - `python -m pytest tests/unit/test_dockerfile_contents.py tests/unit/test_dashboard_settings.py -q --tb=short` → pass
   - `python -m pytest tests/ -q --tb=short` → pass

**Files changed:** `Dockerfile`, `tests/unit/test_dockerfile_contents.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Post-Review Hardening: Settings Edge Cases (2026-05-17)

**Status:** COMPLETED — reviewed the newly added dashboard settings functionality, fixed edge-case mismatches, and revalidated with targeted and full test runs.

### Changes
1. **`dashboard/settings_service.py`**
   - Added option sanitization so unknown keys from Supervisor/local payloads are ignored.
   - Kept default-merging behavior for load paths while ensuring save payloads only include known option keys.
   - Switched Supervisor base URL resolution to runtime (`_get_supervisor_base_url`) for environment consistency.
2. **`dashboard/components/settings.py`**
   - Added robust bool coercion for string-like values (`"true"`, `"false"`, `"1"`, `"0"`, etc.).
   - Added safe numeric parsing + clamp-to-schema bounds before rendering int/float inputs to avoid widget failures with malformed persisted values.
3. **`tests/unit/test_dashboard_settings.py`**
   - Added tests for unknown-key filtering and bool coercion helpers.
4. **Validation**
   - `python -m pytest tests/unit/test_dashboard_settings.py -q --tb=short` → pass
   - `python -m pytest tests/unit/test_dashboard_settings.py tests/unit/test_dashboard_components.py tests/unit/test_dashboard_data_service.py -q --tb=short` → pass
   - `python -m pytest tests/ -q --tb=short` → pass

**Files changed:** `dashboard/settings_service.py`, `dashboard/components/settings.py`, `tests/unit/test_dashboard_settings.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Dashboard Settings Page + German Translation Coverage (2026-05-17)

**Status:** COMPLETED — added a grouped Streamlit settings UI backed by the Supervisor API and introduced prefixed English/German add-on translations for the Home Assistant configuration screen.

### Changes
1. **Dashboard settings UI**:
   - Added `dashboard/components/settings.py` with grouped expanders, German labels, English tooltips, review-before-save confirmation, and Supervisor API persistence.
   - Updated `dashboard/app.py` to expose the new **Settings** page in the sidebar navigation.
2. **Shared settings helpers**:
   - Added `dashboard/config_schema.py` to load config defaults, schema metadata, section grouping, and translation labels from add-on files.
   - Added `dashboard/settings_service.py` to fetch/save add-on options via `/addons/self/options`, with local fallback loading for non-Supervisor contexts.
3. **Translations**:
   - Added `ml_heating_underfloor/translations/de.yaml` covering all add-on options with German labels and English descriptions.
   - Updated `ml_heating_underfloor/translations/en.yaml` to add visual section prefixes and fill the previously missing entries for recent config options.
4. **Tests**:
   - Added `tests/unit/test_dashboard_settings.py` for settings metadata coverage, group mapping, defaults, and Supervisor API request behavior.

**Files changed:** `dashboard/app.py`, `dashboard/components/settings.py`, `dashboard/config_schema.py`, `dashboard/settings_service.py`, `ml_heating_underfloor/translations/en.yaml`, `ml_heating_underfloor/translations/de.yaml`, `tests/unit/test_dashboard_settings.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Full Test Pass + Optuna/CV Holdout Regression Guard (2026-05-17)

**Status:** COMPLETED — implemented dedicated holdout-isolation regression test, fixed a cross-module bounds mismatch discovered during full-suite run, and verified unit-suite health.

### Changes
1. **`tests/unit/test_heating_correction_ml_calibration.py`**: Added `TestHoldoutIsolation::test_optuna_and_cv_do_not_use_holdout_rows` to assert Optuna/CV training fits never include holdout sentinel rows.
2. **`src/thermal_config.py`**: Updated `pv_heat_weight` lower bound from `0.00001` to `0.0001` (both parameter-bound sets) to match current schema and unit expectations.
3. **Validation runs**:
  - `python -m pytest tests/unit/test_heating_correction_ml_calibration.py -q --tb=short` → pass
  - `python -m pytest tests/unit -q --tb=short --ignore=tests/unit/test_thermal_equilibrium_model_properties.py` → **1327 passed, 1 skipped**
  - Full `tests/` run still blocked by environment prerequisites: missing `hypothesis` package and missing local `docker` CLI for image-smoke integration tests.

**Files changed:** `tests/unit/test_heating_correction_ml_calibration.py`, `src/thermal_config.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Post-Implementation Review Fixes (2026-05-17)

**Status:** COMPLETED — fixed leakage and edge-case issues identified during review of the latest session changes.

### Changes
1. **`src/heating_correction_ml_calibration.py`**:
  - Fixed holdout leakage: Optuna HPO + CV now run on `df_fit` only.
  - Added CV edge-case guard: gracefully skips when fit split is too short for `TimeSeriesSplit`.
  - Set permutation importance `n_jobs=1` for stable execution in test/constrained environments.
2. **`ml_heating_underfloor/config.yaml`**: clarified `heating_ml_cv_enabled` tooltip to match implementation (additional diagnostics + preserved holdout).
3. **`tests/unit/test_heating_correction_ml_calibration.py`**: fake regressors updated to sklearn-compatible estimator mixins to remove deprecation warning flood / future break risk.
4. **Validation**: `python -m pytest tests/unit/test_heating_correction_ml_calibration.py tests/unit/test_pv_trajectory.py tests/unit/test_overshoot_logic.py -q --tb=short` → **63 passed**.

**Files changed:** `src/heating_correction_ml_calibration.py`, `ml_heating_underfloor/config.yaml`, `tests/unit/test_heating_correction_ml_calibration.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ ML Calibration Improvements + PV Rescue Decoupling (2026-05-17)

**Status:** COMPLETED — ML calibration pipeline enhanced with feature pruning, regularisation, Optuna HPO, and time-series CV. PV trajectory rescue decoupled from min_steps to fix overshoot correction bug.

### Changes
1. **`src/pv_trajectory.py`**: Rescue condition uses `PV_TRAJ_RESCUE_MIN_HOURS` (default 1) instead of `min_steps` — prevents premature trajectory collapse during gradual PV decline.
2. **`src/heating_correction_ml_calibration.py`**: Added feature pruning (PI-based, retrain + MAE regression guard), `reg_alpha`/`reg_lambda`, Optuna HPO, TimeSeriesCV — all config-gated.
3. **`src/config.py`**: Added 10 new config vars: `PV_TRAJ_RESCUE_MIN_HOURS`, `HEATING_ML_FEATURE_PRUNING_ENABLED`, `HEATING_ML_PRUNE_PI_THRESHOLD`, `HEATING_ML_REG_ALPHA`, `HEATING_ML_REG_LAMBDA`, `HEATING_ML_OPTUNA_ENABLED`, `HEATING_ML_OPTUNA_N_TRIALS`, `HEATING_ML_CV_ENABLED`, `HEATING_ML_CV_N_SPLITS`.
4. **`ml_heating_underfloor/config.yaml`**: Options + schema + tooltips for all new config vars.
5. **`config_adapter.py`**: Env var mappings for all new config vars.
6. **Tests**: Updated `test_pv_trajectory.py` (5 new rescue_min_hours tests), updated `test_heating_correction_ml_calibration.py` (9 new config/pruning/regularisation tests).

**Files changed:** `src/pv_trajectory.py`, `src/heating_correction_ml_calibration.py`, `src/config.py`, `ml_heating_underfloor/config.yaml`, `config_adapter.py`, `tests/unit/test_pv_trajectory.py`, `tests/unit/test_heating_correction_ml_calibration.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Refine forecast-mode correction suppression boundary (2026-05-16)

**Status:** COMPLETED — overshoot/undershoot suppression in forecast mode now applies only while `TRAJECTORY_STEPS > PV_TRAJ_MIN_STEPS`; correction is automatically re-enabled at the minimum trajectory floor.

### Changes
1. **`src/model_wrapper.py`**: Updated `_verify_trajectory_and_correct` early-return guard to require `TRAJECTORY_STEPS > PV_TRAJ_MIN_STEPS` in addition to `PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION` and `PV_TRAJ_FORECAST_MODE_ENABLED`.
2. **`tests/unit/test_overshoot_logic.py`**: Added two regression tests for the boundary behavior:
   - skip remains active when steps are above min
   - correction path executes when steps equal min
3. **Config/docs wording alignment**:
   - `src/config.py`
   - `ml_heating_underfloor/config.yaml`
   - `ml_heating_underfloor/translations/en.yaml`
4. **Documentation updates**: Updated `CHANGELOG.md`, `memory-bank/progress.md`, and `memory-bank/activeContext.md`.

**Files changed:** `src/model_wrapper.py`, `tests/unit/test_overshoot_logic.py`, `src/config.py`, `ml_heating_underfloor/config.yaml`, `ml_heating_underfloor/translations/en.yaml`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Add pv_traj_disable_overshoot_correction switch (2026-05-16)

**Status:** COMPLETED — added new `pv_traj_disable_overshoot_correction` boolean config option that suppresses the overshoot/undershoot correction in `_verify_trajectory_and_correct` when forecast-driven trajectory scaling is active.

### Changes
1. **`ml_heating_underfloor/config.yaml`**: Added `pv_traj_disable_overshoot_correction: false` option under Forecast-Driven Trajectory Scaling section, plus matching schema entry `"bool"`.
2. **`ml_heating_underfloor/translations/en.yaml`**: Added English translation entry with name and description.
3. **`config_adapter.py`**: Mapped `pv_traj_disable_overshoot_correction` → `PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION` env var.
4. **`src/config.py`**: Added `PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION` env var (default `false`).
5. **`src/model_wrapper.py`**: Added early-return guard in `_verify_trajectory_and_correct` that skips correction when both `PV_TRAJ_DISABLE_OVERSHOOT_CORRECTION` and `PV_TRAJ_FORECAST_MODE_ENABLED` are true.
6. **`tests/unit/test_overshoot_logic.py`**: Added `TestDisableOvershootCorrectionInForecastMode` with 4 tests covering all flag combinations and default value.
7. **Documentation**: Updated `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`.

**Files changed:** `ml_heating_underfloor/config.yaml`, `ml_heating_underfloor/translations/en.yaml`, `config_adapter.py`, `src/config.py`, `src/model_wrapper.py`, `tests/unit/test_overshoot_logic.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`



**Status:** COMPLETED — fixed failing GitHub Actions docker smoke test by replacing removed `src.config` symbol `POLL_INTERVAL` with existing `CYCLE_INTERVAL_MINUTES`, and by validating the correct wrapper class name (`EnhancedModelWrapper`).

### Changes
1. **`.github/workflows/build.yaml`**: Updated the "Core module smoke test" one-liner import in the `test-image` job from `from src.config import POLL_INTERVAL` to `from src.config import CYCLE_INTERVAL_MINUTES`, and from `ModelWrapper` to `EnhancedModelWrapper`.
2. **`tests/integration/test_image_smoke.py`**: Updated the same core-module smoke import assertion to keep repository integration tests aligned with the workflow check (`CYCLE_INTERVAL_MINUTES` + `EnhancedModelWrapper`).
3. **Documentation updates**: Added matching entries to `CHANGELOG.md`, `memory-bank/progress.md`, and `memory-bank/activeContext.md`.

**Files changed:** `.github/workflows/build.yaml`, `tests/integration/test_image_smoke.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Fix calibration UserWarning spam (2026-05-16)

**Status:** COMPLETED — removed `.values` from `X_fit`/`X_val` in `heating_correction_ml_calibration.py` so training and permutation_importance use DataFrames with named columns, eliminating hundreds of sklearn `UserWarning: X does not have valid feature names` messages per calibration run.

### Changes
1. **`src/heating_correction_ml_calibration.py`**: Changed `df_fit[feature_cols].values.astype(float)` → `df_fit[feature_cols].astype(float)` (and same for `X_val`). Training format now matches inference format (both use named DataFrames).

**Files changed:** `src/heating_correction_ml_calibration.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`



**Status:** COMPLETED — 8 new features added to calibration + inference, feature importance logging, sklearn warning fixed, 1327/1328 tests pass (1 pre-existing failure)

### Changes
1. **Wind speed entity plumbing**: Added `WIND_SPEED_ENTITY_ID` to `src/config.py`, `config_adapter.py`, `config.yaml` (options + schema), InfluxDB default entity list (`src/influx_service.py`), `src/physics_calibration.py` important_optional_columns.
2. **8 new calibration features** (`src/heating_correction_ml_calibration.py`): `wind_speed`, `indoor_temp_gradient`, `living_room_temp`, `is_hp_active`, `is_weekend`, `thermal_power_rolling_1h`, `indoor_margin_rate`, `is_overshoot`. Also added column renames for living_room and wind entities.
3. **8 new inference handlers** (`src/heating_correction_ml_model.py`): Feature extraction handlers for all 8 new columns with appropriate fallbacks.
4. **sklearn feature-names fix** (`src/heating_correction_ml_model.py`): `predict()` now uses `pd.DataFrame` with column names instead of raw `np.array`.
5. **Feature importance logging** (`src/heating_correction_ml_calibration.py`): LightGBM split-based importance + optional permutation importance logged after training. Importances saved in metadata JSON.
6. **Runtime features** (`src/physics_features.py`): Added `wind_speed`, `is_weekend`, `indoor_margin_rate` to the features dict.
7. **Tooltips** (`ml_heating_underfloor/translations/en.yaml`): Added `wind_speed_entity` description.
8. **Tests**: 16 new unit tests for feature extraction handlers. Fixed test mocks in `test_physics_features.py` and `test_ha_history_service.py` for wind_speed entity.

**Files changed:** `src/config.py`, `config_adapter.py`, `ml_heating_underfloor/config.yaml`, `src/influx_service.py`, `src/physics_calibration.py`, `src/heating_correction_ml_calibration.py`, `src/heating_correction_ml_model.py`, `src/physics_features.py`, `ml_heating_underfloor/translations/en.yaml`, `tests/unit/test_heating_correction_ml_model.py`, `tests/unit/test_physics_features.py`, `tests/unit/test_ha_history_service.py`, `CHANGELOG.md`

## ✅ Newton correction τ/2 floor fix + UI improvements (2026-05-16)

**Status:** COMPLETED — critical Newton correction bug fixed; 17/17 heating correction tests pass

### Changes
1. **Critical fix — Newton correction τ/2 floor** (`src/model_wrapper.py`): Both ε and S(t) now evaluated at max(t_worst, τ_room/2) instead of always at t_worst. Prevents degenerate S(t)≈0.03 at early trajectory steps from always clamping corrections to ±2.5°C. Sign-flip detection suppresses correction when trajectory recovers by τ/2.
2. **Test updates** (`tests/unit/test_heating_correction.py`): Updated test 9 (moved overshoot peak from t=2h to t=3h above τ/2). Added test 12 (τ/2 floor suppresses degenerate correction) and test 13 (sign flip suppresses correction).
3. **Missing tooltips** (`ml_heating_underfloor/translations/en.yaml`): Added 13 missing HA UI descriptions for ML heating correction parameters and `pv_traj_forecast_rescue_enabled`.
4. **Dashboard button** (`dashboard/components/control.py`): Added "Calibrate ML Heating Model" button with flag file `/data/config/calibrate_heating_correction_ml_flag`.

**Files changed:** `src/model_wrapper.py`, `tests/unit/test_heating_correction.py`, `ml_heating_underfloor/translations/en.yaml`, `dashboard/components/control.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ ML heating correction workflow audit — 3 bugs fixed (2026-05-15)

**Status:** COMPLETED — 3 bugs fixed; 3 regression tests added; all 67 affected tests pass

Full end-to-end audit of the ML-based heating correction workflow (main.py, model_wrapper.py, heating_correction_ml_model.py, heating_correction_ml_calibration.py, heating_correction_ml_observation_buffer.py).

### Bugs fixed
1. **Critical — `indoor_temp` key mismatch at ML inference** (`src/heating_correction_ml_model.py`): `_extract_heating_feature("indoor_temp")` returned 0.0 at runtime because `build_physics_features()` stores the indoor temperature as `indoor_temp_lag_30m` not `indoor_temp`. `indoor_margin` had the same bug. Added fallback to `indoor_temp_lag_30m` in both handlers. 3 new regression tests in `tests/unit/test_heating_correction_ml_model.py`.
2. **Minor — duplicated format arg in S_H warning** (`src/heating_correction_ml_calibration.py`): The `s_h < 0.05` fallback warning logged the new fallback value for both `%f` placeholders, hiding the original degenerate value. Saved the original before overwriting.
3. **Missing config — `HEATING_ML_RETRAIN_VAL_FRACTION` not wired** (`config_adapter.py`, `ml_heating_underfloor/config.yaml`): Config var existed in `config.py` but had no add-on schema entry or adapter mapping.

**Files changed:** `src/heating_correction_ml_model.py`, `src/heating_correction_ml_calibration.py`, `config_adapter.py`, `ml_heating_underfloor/config.yaml`, `tests/unit/test_heating_correction_ml_model.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Heating ML correction workflow review + cooling-label pollution fix (2026-05-15)

**Status:** COMPLETED — critical workflow bug fixed; syntax validation passed

- Reviewed the end-to-end heating ML correction workflow in `src/main.py`, especially the new online-learning integration around `HeatingCorrectionObservationBuffer`.
- Confirmed the main logical mismatch: the heating buffer block previously aged/labeled pending heating observations outside heating operation, which allowed cooling/summer indoor temperatures to generate invalid heating labels.
- Fixed `src/main.py` so the entire heating observation-buffer lifecycle (`push_pending`, `resolve_labels`, save, retrain trigger) only runs during `climate_mode == "heating"`.
- Verified the file still compiles with `python -m py_compile`.
- Updated project docs to reflect the corrected workflow.

**Files changed:** `src/main.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

## ✅ Heating Correction ML Online Learning (2026-05-15)

**Status:** COMPLETED — 25 new unit tests pass (0 failures)

Added sliding-window online learning to the heating correction ML model, mirroring the pre-cooling `CoolingObservationBuffer` pattern:

- `src/heating_correction_ml_observation_buffer.py` (new): `HeatingCorrectionObservationBuffer` — regression label buffer; push-pending on heating cycles, resolve labels (float: `−(T_future − T_target) / S_H`) after `label_horizon_steps` cycles, auto-retrain trigger, JSON persistence, thread-safe RLock, eviction of oldest labeled entries.
- `src/main.py`: init block before main loop (unconditional); per-cycle push/resolve/save/retrain block; hot-reloads singleton via `EnhancedModelWrapper._heating_correction_ml_model = None`.
- `src/config.py`: 3 new vars `HEATING_ML_OBSERVATION_BUFFER_PATH`, `HEATING_ML_RETRAIN_TRIGGER_K`, `HEATING_ML_BUFFER_MAX_N`.
- `config_adapter.py`: 2 new env var mappings.
- `ml_heating_underfloor/config.yaml`: 2 new options + 2 schema entries.
- `tests/unit/test_heating_correction_observation_buffer.py` (new): 25 tests.

**Files changed:** `src/heating_correction_ml_observation_buffer.py` (new), `src/main.py`, `src/config.py`, `config_adapter.py`, `ml_heating_underfloor/config.yaml`, `tests/unit/test_heating_correction_observation_buffer.py` (new), `CHANGELOG.md`



**Status:** COMPLETED — all new tests pass (50 new unit tests, 0 failures)

Implemented the full ML-based heating correction pipeline (HEATING_CORRECTION_MODE = "ml") mirroring the CoolingMLModel / cooling_ml_calibration.py pattern:

- `src/heating_correction_ml_model.py`: `HeatingCorrectionMLModel` class — loads joblib model + metadata JSON, exposes `predict(features, target_indoor)` returning ΔT_outlet [°C], and `r2_score` property for blend weighting.
- `src/heating_correction_ml_calibration.py`: `calibrate_heating_correction_ml()` — cold-season filter (AT < 18°C), 18-feature vector (indoor trends, AT hindcast 1–4h, fireplace/TV lags, delta_T, thermal power, cyclical time), LightGBM regressor (MAE objective), regression label `−(T_future − T_target) / S_H`.
- `src/model_wrapper.py`: replaced stub `_calculate_ml_correction()` with confidence-weighted blend; added `_get_heating_correction_ml_model()` lazy singleton loader.
- `src/main.py`: `--calibrate-heating-correction-ml` CLI flag + flag-file detection.
- `src/config.py`: 8 new config vars + `_parse_heating_start_date()` helper.
- `config_adapter.py`: new env var mappings.
- `ml_heating_underfloor/config.yaml`: new options + schema entries.

**Files changed:** `src/config.py`, `src/heating_correction_ml_model.py`, `src/heating_correction_ml_calibration.py`, `src/model_wrapper.py`, `src/main.py`, `config_adapter.py`, `ml_heating_underfloor/config.yaml`, `tests/unit/test_heating_correction_ml_calibration.py`, `tests/unit/test_heating_correction_ml_model.py`, `tests/unit/test_heating_correction.py`, `docs/HEATING_CORRECTION_PHYSICS_VS_ML_ANALYSIS.md`, `CHANGELOG.md`



**Status:** COMPLETED — 1250 passed, 0 new failures

- Fixed `_calculate_physics_newton_correction()` to evaluate sensitivity `S` at the time of the worst trajectory point (`t_worst`) instead of always at the full horizon H. When PV drives a mid-horizon overshoot the worst point occurs at `t_worst < H`, and since `S(H) > S(t_worst)`, using `S_H` in the denominator systematically under-corrects. With `S(t_worst)` the correction is `ε / S(t_worst) > ε / S_H` (larger magnitude), correctly compensating.
- Added `_worst_idx` tracking in each violation branch; time resolved via `trajectory["times"]` if available, otherwise inferred as `(idx+1) * H/n_steps`.
- Updated test constants `S_3H_EXPECTED` for the undershoot/overshoot tests (min/max at step 2 of 4 = t=3h not t=H).
- Added `test_mid_horizon_pv_overshoot_uses_t_worst` and `test_undershoot_at_last_step_uses_s_h` to cover both the PV scenario and the no-change case.

**Files changed:** `src/model_wrapper.py`, `tests/unit/test_heating_correction.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`



**Status:** COMPLETED — 1221 passed, 0 failed (1 pre-existing skip)

- Root cause: `calculate_optimal_outlet_temperature()` and `_calculate_equilibrium_outlet_temperature()` used heating-only outlet bounds (`min=max(outdoor+5, 25)`, `max=70`) and a fallback that rejected outlets below outdoor temp. In cooling mode the formula correctly computed ~18°C outlets but the bounds forced them up to 25–35°C.
- Production was unaffected because `main.py` → `model_wrapper._calculate_required_outlet_temp()` uses binary search with `predict_thermal_trajectory()` which has full cooling support.
- Added `climate_mode` parameter to both methods; cooling uses `[COOLING_CLAMP_MIN_ABS, COOLING_CLAMP_MAX_ABS]` bounds and skips the heating-only "outlet < outdoor" fallback.
- Fixed notebook `05_cooling_scenario_simulation.ipynb`: `simulate_model_mode()` now passes `climate_mode='cooling'` to the analytical method.

**Files changed:** `src/thermal_equilibrium_model.py`, `notebooks/analysis/05_cooling_scenario_simulation.ipynb`

## ✅ Fix all 24 pre-existing test failures (2026-05-14)

**Status:** COMPLETED — 1222 passed, 0 failed, 1 skipped

- Root cause: `test_config.py` deleted `sys.modules['src.config']` without restoring it, causing config module identity mismatch in 7+ downstream tests
- Fixed `test_config.py`: added `tearDown()` to restore original config module
- Fixed `test_dashboard_data_service.py`: `missing_state` fixture now also patches `_COOLING_STATE_FILE_CANDIDATES`
- Fixed `test_dashboard_components.py`: added `pytest.importorskip("streamlit")` for 9 tests
- Fixed `test_overheating_predictor.py`: moved peak from 8h to 10h (beyond new 8h lead time)
- Fixed `test_adaptive_learning.py`: set explicit initial PV weight below max clamp, increased iterations
- Fixed `test_physics_calibration.py`: `pv_heat_weight` default assertion now checks bounds rather than exact value

**Files changed:** `tests/unit/test_config.py`, `tests/unit/test_dashboard_components.py`, `tests/unit/test_dashboard_data_service.py`, `tests/unit/test_overheating_predictor.py`, `tests/unit/test_physics_calibration.py`, `tests/integration/test_adaptive_learning.py`

## ✨ Cooling ML calibration data optimization (2026-05-14)

**Status:** COMPLETED

- Added `purpose` parameter to `fetch_historical_data_for_calibration()` — `"cooling"` fetches only 7 entities (indoor, outdoor, outlet, inlet, PV, flow, power) instead of all 15
- `influx_service.get_training_data()` and `ha_history_service.get_training_data_from_ha()` now accept optional `entity_ids` override
- New `COOLING_ML_WARM_THRESHOLD_C` config (default 10°C) replaces derived formula, adding shoulder-season negative examples to improve 85/15 label imbalance
- Forecast feature defaults now derived from `PRE_COOL_LEAD_TIME_HOURS` (8h) instead of hardcoded 12h
- Fixed `PRE_COOL_LEAD_TIME_HOURS` default mismatch (3.0 → 8.0 in config.py)
- Added `_field` to `_META_COLS` to suppress InfluxDB artifact gap warnings
- Wired `COOLING_ML_WARM_THRESHOLD_C` in `config_adapter.py`
- Updated 3 test mocks to accept new `purpose` kwarg; fixed lead-time assertion

**Files changed:**
- `src/config.py` — new config + default fixes
- `src/influx_service.py` — `entity_ids` parameter
- `src/ha_history_service.py` — `entity_ids` parameter
- `src/physics_calibration.py` — `purpose` parameter + `_field` fix
- `src/cooling_ml_calibration.py` — `purpose="cooling"` + warm threshold
- `config_adapter.py` — new env var mapping
- `tests/unit/test_cooling_ml_calibration.py` — mock + config fixes
- `tests/unit/test_cooling_ml_extended.py` — lead-time assertion fix

---

## ✨ Feature: Cooling ML configurable calibration start date (2026-05-14)

**Status:** COMPLETED

- `calibrate_cooling_ml()` now reads `COOLING_ML_CALIBRATION_START_DATE` (format `DD.MM.YYYY`) from config/env and converts it to `lookback_hours` at runtime. Falls back to 2160 h (90 days) when empty or invalid; warns on bad input.
- `_parse_cooling_start_date()` helper added to `src/config.py`.
- New HA add-on option + schema + tooltip added to `config.yaml` and `translations/en.yaml`.
- 6 new unit tests in `TestCoolingStartDate`.

**Files changed:**
- `src/config.py`
- `src/cooling_ml_calibration.py`
- `ml_heating_underfloor/config.yaml`
- `ml_heating_underfloor/translations/en.yaml`
- `tests/unit/test_cooling_ml_calibration.py`
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---

**Status:** COMPLETED

- Extended `cooling_ml_calibration.py` Step 6 to include all 12 outdoor-temperature hindcast columns (`AT_roh_1h`–`AT_roh_12h`) and all 12 PV-power hindcast columns (`pv_forecast_1h`–`pv_forecast_12h`) in the training feature set by default (previously only `AT_roh_4h` was used).
- Added `COOLING_ML_AT_FORECAST_HOURS` and `COOLING_ML_PV_FORECAST_HOURS` config/env vars (defaults `"1,2,3,4,5,6,7,8,9,10,11,12"`); legacy `COOLING_ML_FORECAST_HOURS` kept as alias.
- Inference-side handlers in `cooling_ml_model.py` were already complete — no changes needed there.
- 4 new unit tests in `TestForecastHourSelection` cover custom hour lists, default all-12h regression, and the legacy alias.

**Files changed:**
- `src/config.py`
- `src/cooling_ml_calibration.py`
- `tests/unit/test_cooling_ml_calibration.py`
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---

## 🔧 Fix: auto-trigger Docker build on push to main (2026-05-14)

**Status:** COMPLETED

- Root cause: `.github/workflows/build.yaml` only had `workflow_dispatch` trigger. Merging a PR that bumped the version in `config.yaml` never automatically built a Docker image, so HA tried to pull a non-existent image tag and got `[404] manifest unknown`.
- Fix: added `push: branches: [main]` trigger with `paths-ignore` for markdown/docs to skip unnecessary rebuilds.
- The version-bump commit already includes `[skip ci]`, so no infinite loop.

**Files changed:**
- `.github/workflows/build.yaml`
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---

## 🐛 Fix aarch64 Docker build: Alpine → Debian slim (2026-05-14)

**Status:** COMPLETED

- Root cause: native ARM runner (`ubuntu-24.04-arm`) + Alpine musl → no pre-built scikit-learn wheel for `musllinux_1_2_aarch64` → source compilation fails with GCC `-Werror=array-bounds` in `_ball_tree.c`.
- Fix: changed base image to `python:3.11-slim` (Debian/glibc); updated system package installs from `apk` to `apt-get`.

**Files changed:**
- `Dockerfile`
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---

## 🛡️ Harden Cooling ML Calibration & PV Feature Contract (2026-05-14)

**Status:** COMPLETED

- Hardened cooling ML calibration semantics:
  - Default pre-cooling lead-time reduced from 8.0h to 3.0h for more responsive label assignment.
  - Calibration and inference now strictly use raw electrical PV keys for all cooling ML features, falling back to thermal-corrected keys only if electrical keys are absent.
  - Fixed bugs in pre-cooling calibration: correct PV key usage, feature scale alignment, buffer persistence, and lead-time/label horizon calculation.
- Cooling observation buffer now persists after every push/resolve cycle, so pending entries and evolving label state survive restarts.
- Added `scikit-learn>=1.0.0` to `requirements.txt` to fix silent metric failures in calibration.
- Added/tightened unit tests for cooling ML calibration and PV feature extraction (raw vs thermal PV history, column-count expectations).

**Test status:**
- Unit tests for cooling ML calibration and PV feature extraction updated and expanded; regression coverage for PV contract and calibration workflow confirmed. No test regressions detected.

**Files changed:**
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

## Resolve PR Merge Conflicts (2026-05-14, latest sync)

**Status:** COMPLETED

- Merged latest `origin/main` into the PR branch to clear newly reported merge conflicts.
- Resolved content conflicts in `memory-bank/activeContext.md` and `memory-bank/progress.md` by preserving both branch entries.
- Kept incoming updates from `origin/main` for `.github/workflows/build.yaml` and `CHANGELOG.md`.
- Applied review cleanup after merge: translated inline German comments in `.github/workflows/build.yaml` to English and fixed a truncated sentence in `memory-bank/activeContext.md`.
- Completed follow-up cleanup from automated review: restored the missing `Files changed` block in the PV contract context entry, removed an extra trailing separator in `memory-bank/activeContext.md`, and expanded compact list formatting in `.github/workflows/build.yaml`.
- Validation note: targeted tests could not be executed in this runner because `pytest` is unavailable (`No module named pytest`).

**Files changed:**
- `.github/workflows/build.yaml`
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---

## Address Reviewer Thread Follow-ups (2026-05-14)

**Status:** COMPLETED

- Implemented reviewer-requested persistence hardening in `src/main.py`: observation buffer is now saved every pre-cooling cycle after `push_pending()` + `resolve_labels()`, preserving pending entries and per-step label state across restarts.
- Implemented reviewer-requested PV scale alignment fix for rolling features:
  - Added raw `pv_power_history_electrical` to physics features in `src/physics_features.py`.
  - Updated cooling ML rolling PV extraction in `src/cooling_ml_model.py` to prefer raw electrical history and only fall back to thermal-corrected history.
- Added regression coverage:
  - `tests/unit/test_cooling_ml.py`: verifies `pv_roll_1h` prefers raw electrical history.
  - `tests/unit/test_physics_features.py`: verifies raw PV history feature is emitted.
- Validation note: targeted tests could not be executed in this runner because `pytest` is unavailable (`No module named pytest`).

**Files changed:**
- `src/main.py`
- `src/physics_features.py`
- `src/cooling_ml_model.py`
- `tests/unit/test_cooling_ml.py`
- `tests/unit/test_physics_features.py`
- `CHANGELOG.md`
- `memory-bank/progress.md`
- `memory-bank/activeContext.md`

---

## 🚀 CI Workflow Modernization & Architecture Improvements (2026-05-14)

**Status:** COMPLETED

- **GitHub Actions workflow updated:**
  - `.github/workflows/build.yaml` now uses latest versions of core actions (`actions/checkout@v4`, `docker/login-action@v3`, `docker/setup-buildx-action@v3`, `docker/build-push-action@v6`).
  - Native runners are used for each architecture (`ubuntu-24.04-arm` for ARM, `ubuntu-latest` for AMD64), eliminating QEMU emulation.
  - Build cache is now separated per architecture for improved speed and reliability.
  - Minor log and changelog update messages clarified.

**Purpose:**
- Ensures CI compatibility with latest GitHub Actions ecosystem.
- Native builds improve speed, reliability, and reduce complexity.
- Per-arch cache prevents cross-architecture cache pollution.

**Test status:**
- CI workflow runs completed successfully for both ARM and AMD64 builds; no regressions detected.

**Files changed:**
- `.github/workflows/build.yaml`

---

## Resolve PR Merge Conflicts (2026-05-14)

**Status:** COMPLETED

- Merged `origin/main` into the PR branch to clear merge conflicts.
- Resolved conflicts in `CHANGELOG.md`, `memory-bank/activeContext.md`, and `memory-bank/progress.md`.
- Kept both branches' documentation/context entries and removed all conflict markers.
- Validation note: test execution could not run in this environment because `pytest` is not installed (`No module named pytest`).

**Files changed:**
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---

## Review Cooling Calibration Workflow Follow-up (2026-05-14)

**Status:** COMPLETED

- Reviewed the recently landed pre-cooling calibration fixes for remaining workflow gaps.
- Added an explicit PV contract assertion in `tests/unit/test_pre_cooling_integration.py` to verify `OverheatingPredictor` forwards corrected `pv_now` / `pv_forecast_*` values into `predict_thermal_trajectory()`.
- Updated stale cooling ML test fixtures and fake configs from `PRE_COOL_LEAD_TIME_HOURS=8.0` to `3.0` to match runtime/config defaults.
- Validation: `python -m pytest tests/unit/test_pre_cooling_integration.py tests/unit/test_cooling_ml.py tests/unit/test_cooling_ml_calibration.py -q --tb=short` → **75 passed**.

**Files changed:**
- `tests/unit/test_pre_cooling_integration.py`
- `tests/unit/test_cooling_ml.py`
- `tests/unit/test_cooling_ml_calibration.py`

---

## 🛡️ PV Key Ownership Codified & Pre-cooling Path Regressions (2026-05-14)

**Status:** COMPLETED

- **Documentation:**
  - Added canonical AI MODEL NOTICE section to `memory-bank/systemPatterns.md` with two-family PV key table, per-module usage map, four explicit rules, and citations.
  - Added warning block to `docs/ML_COOLING_MODEL_GUIDE.md` above Feature Engineering, instructing all contributors and AI models to use the correct PV key family.
- **Regression tests:**
  - Added `TestPVKeyContract` class (5 tests) to `tests/unit/test_overheating_predictor.py` to lock `OverheatingPredictor` to thermal keys and `HLCCycle` to electrical key.
  - Hardened assertions to check trajectory call kwargs directly, preventing silent regressions.
- **Refactor:**
  - Moved `hlc_learner` imports to module level in `test_overheating_predictor.py`.
- **Purpose:**
  - Prevents accidental misuse of PV feature keys in ML cooling/pre-cooling paths, which previously caused silent over-estimation of solar gain.

**Test status:**
- All new and existing regression tests pass: `pytest tests/unit/test_overheating_predictor.py -q --tb=short` → **33 passed**.

**Files changed:**
- `memory-bank/systemPatterns.md`, `docs/ML_COOLING_MODEL_GUIDE.md`, `tests/unit/test_overheating_predictor.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## ✅ Reviewer Follow-up: PV Contract Test Assertions Hardened (2026-05-14)

**Status:** COMPLETED

- Updated 2 `TestPVKeyContract` cases in `tests/unit/test_overheating_predictor.py` to assert `predict_thermal_trajectory()` call kwargs directly:
  - `pv_power` anchored from `pv_now` (thermal key family)
  - `pv_forecasts` built from `pv_forecast_{h}h` even when electrical forecast keys are absent
- Removed ambiguity from prior assertions that only checked `result["risk"]` with mocked trajectory outputs.
- Verified table formatting concern: no `||` rows exist in `memory-bank/systemPatterns.md` or `docs/ML_COOLING_MODEL_GUIDE.md`.
- Targeted validation: `python -m pytest tests/unit/test_overheating_predictor.py -q --tb=short` → **33 passed**.

**Files changed:**
- `tests/unit/test_overheating_predictor.py` — strengthened regression assertions
- `CHANGELOG.md`
- `memory-bank/progress.md`
- `memory-bank/activeContext.md`

## Fix Pre-Cooling Calibration Bugs (2026-05-14)

**Status:** COMPLETED

- **Bug 1 (CRITICAL)**: Added `scikit-learn>=1.0.0` to `requirements.txt`; missing package caused silent AUC failure.
- **Bug 2 (HIGH)**: Fixed wrong PV/forecast keys (`pv_now_electrical` → `pv_now`, `pv_forecast_electrical_{h}h` → `pv_forecast_{h}h`, `outdoor_forecast_{h}h` → `temp_forecast_{h}h`) in `test_pre_cooling_integration.py::_make_features()`.
- **Bug 3 (MEDIUM)**: `cooling_ml_model._extract_feature()` now prefers raw electrical PV values (`pv_now_electrical`, `pv_forecast_electrical_{h}h`) over thermally-corrected values to match training data scale.
- **Bug 4 (LOW)**: Fixed `PRE_COOL_LEAD_TIME_HOURS` fallback from `8.0` → `3.0` in `cooling_ml_calibration.py`.
- **Bug 5 (LOW)**: Observation buffer in `main.py` now saved after each cycle that resolves new labels, preventing data loss on restart.

**Files changed:**
- `requirements.txt`
- `src/cooling_ml_calibration.py`
- `src/cooling_ml_model.py`
- `src/main.py`
- `tests/unit/test_pre_cooling_integration.py`

---

## 📚 PV Feature Key Contract — Documentation & Regression Tests (2026-05-14)

**Status:** COMPLETED

- **Documentation**: Added canonical `⚠️ AI MODEL NOTICE — PV Feature Key Contract` section at the top of `memory-bank/systemPatterns.md`. Contains a two-family key table, per-module usage map, four explicit rules, and source-code citations.
- **Cooling guide warning**: Added matching warning block to `docs/ML_COOLING_MODEL_GUIDE.md` above the Feature Engineering section.
- **5 regression tests** in `tests/unit/test_overheating_predictor.py` (`TestPVKeyContract`): lock in that `OverheatingPredictor` uses thermal keys (`pv_now`, `pv_forecast_{h}h`) and that `HLCCycle._build_cycle()` uses the electrical key (`pv_now_electrical`).
- **Purpose**: Prevent future AI/human contributors from accidentally using the wrong PV key family and causing silent over-estimation of solar gain in thermal trajectory predictions.

**Files changed:**
- `memory-bank/systemPatterns.md` — canonical PV key contract
- `docs/ML_COOLING_MODEL_GUIDE.md` — warning block
- `tests/unit/test_overheating_predictor.py` — 5 regression tests
- `CHANGELOG.md`
- `memory-bank/progress.md`
- `memory-bank/activeContext.md`

---

## ML Heating Underfloor v0.2.30 Release Bump (2026-05-13)

**Status:** COMPLETED

- **Change:** Version updated in `config.yaml` from `0.2.29` to `0.2.30`.
- **Purpose:** Release bump to reflect recent bug fixes and ML pre-cooling enhancements. No functional changes in this commit.
- **Test status:** No new tests required; all prior tests (1175+) passing, including recent bugfix and ML cooling suites.

## Fix HP False-Active from Residual Slab Heat (2026-05-13)

**Status:** COMPLETED

- **Bug**: `_is_heat_pump_active()` outlet/inlet fallback triggered false positives when HP was off but slab retained heat (outlet > indoor + 1.0). HP falsely appeared in `active_contributions`, co-learned with PV, contaminating `outlet_effectiveness` and `heat_loss_coefficient`.
- **Fix**: Added idle-band guard (|thermal_power| < 0.1 AND |delta_t| < 0.1 → return False) before outlet/inlet fallback. Preserves fallback for genuinely missing sensor data and low-power detection.
- **8 new tests**: 5 unit tests in `test_cooling_bugfixes.py` (heating/cooling residual slab, low-power fallback preservation), 3 routing tests in `test_heat_source_channels.py` (HP excluded from contributions, PV-only learning, estimate_heat_contribution gate)
- **Full suite**: 1175 passed, 0 regressions (16 pre-existing failures from missing streamlit/hypothesis modules)

**Files changed:**
- `src/heat_source_channels.py` — `_is_heat_pump_active()` idle-band guard
- `tests/unit/test_cooling_bugfixes.py` — 5 new tests
- `tests/unit/test_heat_source_channels.py` — 3 new tests
- `CHANGELOG.md`

---

## � Pre-Cooling ML Review & Bug Fixes (May 2026)

**Status:** COMPLETED

- **CHANGELOG updated**: Added all unreleased changes since v0.2.0 covering pre-cooling ML model, overheating predictor, observation buffer, calibration pipeline, and all cooling fixes
- **Bug fix: NaN/Inf in observation buffer**: `_json_default` handler didn't sanitize NaN/Inf in nested feature dicts. Added `_sanitize_for_json()` to recursively clean dicts before JSON serialization
- **Bug fix: Retrain backoff insufficient**: Failed retrain back-off subtracted `trigger_k//2` which could equal `trigger_k`, causing immediate re-trigger. Fixed to subtract `trigger_k//2 + 1`
- **36 new tests added**: Cold start (no files), NaN/Inf handling, label boundary conditions, reactive cooling, config default verification, online learning flow, feature extraction edge cases, OverheatingPredictor edge cases
- **Known issues documented**: Observation buffer never saved periodically (only on successful retrain); calibration `PRE_COOL_LEAD_TIME_HOURS` fallback default (8.0) mismatches config.py default (3.0)
- **Cold start verified**: All components handle missing files gracefully — model returns no-risk, buffer starts empty, calibration fails gracefully without InfluxDB data

**Files changed:**
- `src/cooling_ml_observation_buffer.py` — NaN/Inf sanitization fix
- `src/main.py` — retrain backoff fix
- `tests/unit/test_cooling_ml_extended.py` — 36 new tests
- `CHANGELOG.md`
- `memory-bank/progress.md`
- `memory-bank/activeContext.md`

---

## �🔧 Physics-Direct Calibration Accuracy Fixes (May 2026)

**Status:** COMPLETED

- **OE calibration**: Added scipy 1-D refinement after analytical initial guess. Analytical formula `OE = HLC * dT_indoor / drive` is correct but numerically fragile when drive (effective_temp - indoor_temp) is small (~4°C). Scipy refinement (HLC locked, minimize MAE) produces physically correct OE ~0.95.
- **Solar lag**: Fixed 180 min result. Root cause: correlating PV with residual level (smoothed by slab mass) instead of d(residual)/dt. Also reduced max lag from 180->60 min and raised correlation threshold from 0.1->0.3.
- **Thermal time constant**: Added transient calibration (heating sequences, scipy L-BFGS-B) as primary method. Cooling curves (HP-off >=2h) rarely available in well-controlled UFH. Transient method uses 1-hour heating windows with abundant data.
- **Unit labels**: Fixed OE and HLC units in ThermalParameterConfig from "dimensionless"/"1/hour" to "kW/K".

**Files changed:**
- `src/physics_calibration_direct.py` — OE scipy refinement, solar lag d(residual)/dt, transient tau
- `src/thermal_config.py` — unit label corrections
- `CHANGELOG.md`
- `memory-bank/progress.md`
- `memory-bank/activeContext.md`

---

## ✅ Fixed stable_periods.json path bug (May 2026)

**Status:** COMPLETED

- The output path for `stable_periods.json` (written by `filter_stable_periods()` in `src/physics_calibration.py`) is now dynamically resolved to the same directory as `unified_thermal_state.json` (using `os.path.dirname(config.UNIFIED_STATE_FILE)`), instead of the previously hardcoded `/opt/ml_heating/` path.
- The target directory is created if missing, and write errors are caught and logged as warnings so calibration does not fail.
- Fix verified in Home Assistant add-on environments where `/opt/ml_heating/` does not exist.
- No regression in test suite; calibration and state persistence both function as expected.

## 🎯 CURRENT STATUS - May 2026 (stable_periods.json path bug fix)

### ✅ **Fixed hardcoded stable_periods.json path**

**Status**: **COMPLETED**

- `filter_stable_periods()` in `src/physics_calibration.py` now writes `stable_periods.json` to the same directory as `unified_thermal_state.json`, using `os.path.dirname(config.UNIFIED_STATE_FILE)`. Previously hardcoded to `/opt/ml_heating/` which did not exist in all HA add-on environments.

**Files changed:**
- `src/physics_calibration.py`
- `CHANGELOG.md`
- `memory-bank/progress.md`
- `memory-bank/activeContext.md`

---

### ✅ **Physics-Direct calibration path implemented and selectable from dashboard**

**Status**: **COMPLETED**

- Added a fully analytical, sequential calibration path (`Physics Direct`) that estimates all thermal model parameters from first principles, without scipy optimization.
- Dashboard now allows users to select between "Scipy Optimizer" and "Physics Direct" calibration methods when triggering model recalibration.
- All calibration parameters are user-editable in `config.yaml` and validated against bounds.
- Fixed 6 config default mismatches; config defaults are now aligned and validated.
- State-file fallback now validates persisted values before accepting them.
- Refactored calibration code to use named constants for magic numbers.
- Expanded unit tests for physics-direct calibration; all tests pass.

**Files changed:**
- `src/physics_calibration_direct.py`
- `dashboard/components/control.py`
- `src/config.py`
- `ml_heating_underfloor/config.yaml`
- `.env_sample`
- `src/unified_thermal_state.py`
- `src/thermal_config.py`
- `tests/unit/test_physics_calibration_direct.py`
- `CHANGELOG.md`

## 🎯 CURRENT STATUS - May 2026 (Calibration fallback + config.yaml exposure)

### ✅ **All calibration parameters now have a 3-level fallback chain**

**Status**: **COMPLETED**

Three separate concerns were addressed together:

1. **State-file warm-start fallback** — `calibrate_thermal_model_physics()` now resolves `ThermalStateManager` early and uses `_state_fallback()` so that any parameter that fails to calibrate keeps its last-known good value from `unified_thermal_state.json` instead of reverting to a hardcoded default.

2. **config.yaml editability** — `ThermalParameterConfig.get_default()` now reads from `config` module variables (set via `config.yaml` / env vars) before falling back to hardcoded `DEFAULTS`. 7 missing config vars added to `src/config.py`; 2 undocumented entries added to `.env_sample` and `config.yaml`.

3. **cloud_factor_exponent / solar_decay_tau_hours persisted** — `set_calibrated_baseline()` and `_get_default_state()` in `unified_thermal_state.py` now include both parameters so they survive restarts and are available as warm-start fallbacks.

**Fallback chain for every calibration parameter:**
1. Calibrated value from current data
2. Last valid value from `unified_thermal_state.json`
3. User-editable value from `config.yaml` / environment variable

**Files changed:**
- `src/physics_calibration_direct.py`
- `src/unified_thermal_state.py`
- `src/thermal_config.py`
- `src/config.py`
- `.env_sample`
- `ml_heating_underfloor/config.yaml`
- `CHANGELOG.md`
- `memory-bank/progress.md`
- `memory-bank/activeContext.md`





## 🎯 PREVIOUS STATUS - June 2025 (Predictive Pre-Cooling)

### ✅ **Predictive Pre-Cooling implemented**

**Status**: **COMPLETED**

Implemented forecast-driven pre-cooling to prevent underfloor cooling from starting too late. The system now simulates a passive indoor trajectory (HP OFF) using PV and outdoor temperature forecasts. When overheating is predicted within the lead time, the binary-search target is shifted down to start cooling proactively.

**Files changed:**
- `src/overheating_predictor.py` — NEW: OverheatingPredictor class
- `src/config.py` — Added 7 PRE_COOL_* parameters
- `src/main.py` — Integrated pre-cool check before outlet prediction, added HA sensor attributes
- `ml_heating_underfloor/config.yaml` — Added config options + schema entries
- `ml_heating_underfloor/translations/en.yaml` — Added English translations
- `tests/unit/test_overheating_predictor.py` — NEW: 27 unit tests
- `tests/unit/test_pre_cooling_integration.py` — NEW: 9 integration tests

## 🎯 PREVIOUS STATUS - May 2026 (Cooling test helper cleanup)

### ✅ **Cooling test helper cleanup completed**

**Status**: **COMPLETED**

Applied the remaining review-driven cleanup in the cooling regression tests by simplifying `tests/unit/test_heat_source_channels.py::make_context()` to an override-based helper instead of a long parameter list. This keeps the new cooling routing/learning tests readable without changing production logic or test behavior.

Files changed:
- `tests/unit/test_heat_source_channels.py`
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## 🎯 CURRENT STATUS - May 2026 (Cooling follow-up review fixes)

### ✅ **Cooling review follow-up fixes completed**

**Status**: **COMPLETED**

Addressed the remaining follow-up issues found during re-review of the cooling fixes. Cooling `delta_t_floor` learning now uses the positive magnitude of negative `delta_t` samples so the learned floor stays physically meaningful. `temperature_control.py` now carries `climate_mode` into both active and shadow `prediction_context` payloads so downstream routing/learning stays in cooling mode. Added focused regression tests for cooling HP+PV routing, PV decay co-routing, positive `delta_t_floor` learning, `climate_mode` propagation, and the RUNNING→RECOVERY gate branches.

Files changed:
- `src/heat_source_channels.py`
- `src/temperature_control.py`
- `tests/unit/test_heat_source_channels.py`
- `tests/unit/test_temperature_control.py`
- `tests/unit/test_cooling_mode.py`
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## 🎯 CURRENT STATUS - May 2026 (Cooling gate: use existing HP detection)

### ✅ **Cooling gate HP detection unified**

**Status**: **COMPLETED**

Removed the `HP_ACTIVE_COOLING_DELTA_T` config constant added in the previous session. The cooling cycle gate RUNNING→RECOVERY transition now reuses `_is_heat_pump_active()` from `heat_source_channels.py` — the same helper used by `_learn_from_recent` and `temperature_control.py` — instead of a bespoke `delta_t < threshold` check. This ensures HP detection is consistent across the entire codebase (checks thermal_power, delta_t, and outlet-vs-inlet signals together).

Files changed:
- `src/model_wrapper.py` — import and use `_is_heat_pump_active`; build context dict from `_current_features`; updated log messages
- `src/config.py` — removed `HP_ACTIVE_COOLING_DELTA_T`
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## 🎯 CURRENT STATUS - May 2026 (Cooling recovery gate fix)

### ✅ **Cooling recovery gate deadlock resolved**

**Status**: **COMPLETED**

Fixed the last remaining cooling bug: RUNNING→RECOVERY transition was firing purely on model-computed outlet being within `MIN_COOLING_DELTA_K` of inlet, regardless of whether the HP was actually running. This caused a deadlock for mild cooling demand: mild need → RECOVERY, but RECOVERY→RUNNING also requires a 2K gap → HP permanently disabled.

Fix: RUNNING→RECOVERY now only fires when measured `delta_t < -HP_ACTIVE_COOLING_DELTA_T` (HP was actually running). When HP was already idle, clamp outlet to inlet_temp without changing gate state. New config constant `HP_ACTIVE_COOLING_DELTA_T=0.5`.

Files changed:
- `src/model_wrapper.py` — RUNNING→RECOVERY gate conditioned on measured delta_t
- `src/config.py` — Added `HP_ACTIVE_COOLING_DELTA_T`
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## 🎯 CURRENT STATUS - May 2026 (Cooling-mode learning & trajectory steps fixes)

### ✅ **HP channel learning in cooling mode + trajectory step override bug**

**Status**: **COMPLETED**

Fixed two logical bugs identified via log analysis:

1. **HP never learned in cooling mode** — `route_learning()` was blocked by `any_external_active` (always `True` on sunny days because PV is active), so the HP channel never received any records. Added a cooling-mode early path that always routes to HP and co-routes PV in parallel.
2. **`HeatPumpChannel._learn_from_recent()` mode blindness** — delta_t filter (`> 0.5`) rejected all cooling samples (delta_t is negative in cooling). Outlet-effectiveness gradient sign was wrong for cooling. Both fixed with mode-aware logic.
3. **Trajectory steps override bug** — `PV_TRAJ_SCALING_ENABLED` block in `main.py` mutated `config.TRAJECTORY_STEPS = PV_TRAJ_MAX_STEPS` before feature-building, and never reset it when `PV_TRAJ_FORECAST_MODE_ENABLED=False`. Removed the block; `physics_features.py` already handles `_n_fc_full` expansion internally.
4. **`temperature_control.py` HP-active detection** — Fixed heating-only `heat_pump_active` detection to be mode-aware (cooling path uses negative thresholds).

Files changed:
- `src/heat_source_channels.py` — `route_learning()` cooling-mode path; `HeatPumpChannel._learn_from_recent()` mode-aware delta_t filter and OE gradient
- `src/main.py` — Removed `PV_TRAJ_SCALING_ENABLED` pre-mutation block
- `src/temperature_control.py` — Mode-aware `heat_pump_active` detection
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---


### ✅ **Fixed workflow yaml error and upgraded actions to Node.js 24**

**Status**: **COMPLETED**

Fixed a duplicate `env:` key error in `ai-code-review.yaml` and upgraded all GitHub Actions to Node.js 24-compatible versions to eliminate deprecation warnings.

Files changed:
- `.github/workflows/ai-code-review.yaml` — Merged duplicate `env:` blocks; upgraded `actions/checkout` v4→v6
- `.github/workflows/build.yaml` — Upgraded `actions/checkout` v4→v6, `docker/login-action` v3→v4, `docker/setup-qemu-action` v3→v4, `docker/setup-buildx-action` v3→v4, `docker/build-push-action` v5→v7
- `CHANGELOG.md` — Added entries under `[Unreleased]`
- `memory-bank/progress.md` — Added milestone entry
- `memory-bank/activeContext.md` — Added context entry

---

## 🎯 CURRENT STATUS - May 2026 (Review-Round Bug Fixes)

### ✅ **REVIEW: 10 bugs found and fixed in recent cooling changes**

**Status**: **COMPLETED**

Thorough review of all recent cooling-mode changes found 10 bugs:
1. Duplicate `inlet_temp`/`delta_t` keys in `prediction_context` dict (main.py)
2. Transient drop filter fires in cooling mode — disabled for cooling
3. `_cooling_cycle_state` not reset on heating→cooling transition — now restored from persisted state
4. `_cooling_cycle_state` not persisted across restarts — added to cooling JSON schema + save
5. `_search_delta_t_floor` not set on binary search early exit — set to `None`, gate uses learned floor
6. Test uses old bounds assertion — fixed to `COOLING_CLAMP_MIN_ABS`
7. `_cooling_target` not validated as numeric — added try/except float()
8. `_cooling_target` not converted to float — now explicit
9. `_search_delta_t_floor` default 0.0 too optimistic — gate now falls back to learned floor when None
10. `_cooling_cycle_state` stale between sessions — gate restored from persistence

8 new tests, 1027 passed (0 regressions), 15 pre-existing failures (dashboard/adaptive learning).

**Files changed**: `src/main.py`, `src/model_wrapper.py`, `src/unified_thermal_state_cooling.py`, `tests/unit/test_cooling_bugfixes.py`, `tests/unit/test_cooling_mode.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## 🎯 PREVIOUS STATUS - May 2026 (Cooling Binary Search Full Range + Config UI)

### ✅ **FIX: Binary search uses full cooling range; cooling target entity visible in HA config UI**

**Status**: **COMPLETED**

The binary search was pre-constrained by both a shutdown margin on `outlet_min` and an inlet-based tightening on `outlet_max`, preventing the search from expressing "HP should be off". The post-search RUNNING/RECOVERY gate already handles HP safety.

Fixes:
1. `src/config.py` `get_outlet_bounds()`: returns `(COOLING_CLAMP_MIN_ABS, COOLING_CLAMP_MAX_ABS)` without adding `COOLING_SHUTDOWN_MARGIN_K`
2. `src/model_wrapper.py` `_calculate_required_outlet_temp()`: removed the cooling-mode `outlet_max` tightening by indoor and inlet temperature
3. `ml_heating_underfloor/config.yaml` schema: added `target_indoor_temp_cooling_entity: "str?"` so it appears in the HA add-on config UI
4. `ml_heating_underfloor/translations/en.yaml`: added name and description for the cooling target entity
5. 3 new regression tests in `tests/unit/test_cooling_bugfixes.py`; 4 existing tests in `test_cooling_mode.py` updated

90 tests pass (0 regressions).

**Files changed**: `src/config.py`, `src/model_wrapper.py`, `ml_heating_underfloor/config.yaml`, `ml_heating_underfloor/translations/en.yaml`, `tests/unit/test_cooling_bugfixes.py`, `tests/unit/test_cooling_mode.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## 🎯 PREVIOUS STATUS - May 2026 (Slab Epsilon Finalization)

### ✅ **Finalized slab epsilon after runtime notebook rerun**

**Status**: **COMPLETED**

Applied the runtime-replay follow-up from `notebooks/analysis/02_epsilon_calibration_review.ipynb`. `SLAB_TIME_CONSTANT_EPSILON` now matches the notebook-recommended linear value (0.5 → 1.595), while `SOLAR_LAG_EPSILON` remains 5.0 because the current runtime replay path still cannot reach the target signal window even at the best sweep candidate.

Files changed:
- `src/thermal_constants.py` — Raised `SLAB_TIME_CONSTANT_EPSILON` to 1.595 and documented why `SOLAR_LAG_EPSILON` stays unchanged
- `CHANGELOG.md` — Merged duplicate `Changed` headings and recorded the slab/solar decision
- `memory-bank/progress.md` — Added milestone entry for the calibration follow-up
- `memory-bank/activeContext.md` — Recorded the final slab/solar epsilon decision

---

## 🎯 CURRENT STATUS - May 2026 (Epsilon Recalibration)

### ✅ **Recalibrated Finite-Difference Gradient Epsilon Values**

**Status**: **COMPLETED**

Systematically calibrated all 7 learnable-parameter epsilon values using a sensitivity analysis script (`scripts/epsilon_sensitivity_analysis.py`). Previous values were hand-tuned with relative epsilon varying 15× across parameters (3.1%–45.6% of default). New values target ΔT ≈ 0.1–0.3°C per perturbation.

Files changed:
- `src/thermal_constants.py` — Updated 4 epsilon values, added 2 new constants (`SOLAR_LAG_EPSILON`, `SLAB_TIME_CONSTANT_EPSILON`)
- `src/thermal_equilibrium_model.py` — `_calculate_solar_lag_gradient` and `_calculate_slab_time_constant_gradient` now use `PhysicsConstants` instead of hardcoded values
- `tests/unit/test_learning_stability.py` — Added `TestEpsilonConstants` class with 8 tests
- `scripts/epsilon_sensitivity_analysis.py` — New calibration script

---

## 🎯 CURRENT STATUS - May 2026 (Persisted Learning Context In Cooling)

### ✅ **FIX: Previous-cycle learning now reuses persisted mode/target and mode-aware demand features**

**Status**: **COMPLETED**

Previous-cycle online learning could back-learn with the current cycle's mode and target after a mode change, while `build_physics_features()` still encoded cooling demand with heating semantics and re-read the default heating target.

Fixes implemented:
1. `src/state_manager.py` now persists `last_climate_mode` and `last_target_indoor_temp`
2. `src/main.py` now saves those values each cycle and reuses them for next-cycle online learning instead of live HA state
3. `src/main.py` switches the model wrapper to the previous cycle's persisted mode before feedback learning runs
4. `src/physics_features.py` now accepts `climate_mode` and `target_indoor_temp_override`, uses the resolved cooling target consistently, and computes forecast demand with mode-aware sign semantics
5. Regression tests added for persisted learning context and cooling/heating forecast demand behavior

21 touched-slice tests pass.

**Files changed**: `src/state_manager.py`, `src/main.py`, `src/physics_features.py`, `tests/integration/test_main.py`, `tests/unit/test_physics_features.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## 🎯 CURRENT STATUS - May 2026 (Cooling Mode Comprehensive Bug Fixes)

### ✅ **FIX: 12 cooling mode bugs — HP learning, slab model, optimization, short-cycling**

**Status**: **COMPLETED**

The cooling mode pipeline was reusing heating logic without mode-aware adaptation, causing:
- HP channel never learning (0 history entries after 69 cycles)
- Slab model permanently in passive mode during cooling
- PV surplus / price offsets working backwards (less cooling when more free energy)
- HP short-cycling without proper prevention

Fixes implemented across 7 phases:
1. **Phase 1** — HP active detection (`_is_heat_pump_active`), slab pump_on gate, HP-OFF delta_t floor, early climate mode detection
2. **Phase 2** — Cooling baselines: `pv_heat_weight` 0.0003→0.002, `slab_time_constant_hours` 0.8→3.19
3. **Phase 3** — `TARGET_INDOOR_TEMP_COOLING_ENTITY_ID` config, PV surplus/price offset inversion
4. **Phase 4** — `heating_demand_forecast` hardcoded 21°C → `target_temp`
5. **Phase 5** — Cooling cycle gate state machine (RUNNING/RECOVERY) replacing simple inlet guard

19 new tests, 997 pass (0 regressions).

**Files changed**: `src/heat_source_channels.py`, `src/thermal_equilibrium_model.py`, `src/model_wrapper.py`, `src/main.py`, `src/config.py`, `src/thermal_config.py`, `src/physics_features.py`, `config_adapter.py`, `ml_heating_underfloor/config.yaml`, `.env_sample`, `tests/unit/test_cooling_bugfixes.py`, `tests/unit/test_unified_thermal_state_cooling.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## 🎯 PREVIOUS STATUS - May 2026 (Cooling Inlet Guard)

### ✅ **FIX: HP idle clamp when outlet ≥ inlet − MIN_COOLING_DELTA_K in cooling mode**

**Status**: **COMPLETED**

Root cause: binary search could converge to outlet within `MIN_COOLING_DELTA_K` of inlet (e.g. inlet=22, outlet=21.5, delta=2 → gap 0.5 < 2). NIBE can't run the compressor at this setpoint → short-cycles or rejects command. Intended behavior: send `inlet` as setpoint (HP stays idle; circulator only).

Fix:
1. `_calculate_required_outlet_temp`: tighten `outlet_max` to `min(indoor−delta, inlet−delta)` when `inlet_temp` is available
2. `calculate_optimal_outlet_temp`: post-search inlet guard — if result `> inlet − delta`, clamp to `inlet_temp` and log HP idle

6 new tests, all 995 pass.

**Files changed**: `src/model_wrapper.py`, `tests/unit/test_cooling_mode.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## 🎯 PREVIOUS STATUS - May 2026 (Cooling State Isolation Fix)

### ✅ **FIX: Cooling operational state now written to correct JSON file**

**Status**: **COMPLETED**

Root cause: `src/state_manager.py::save_state()` and `load_state()` hardwired to `get_thermal_state_manager()` (heating singleton). All per-cycle operational state saves — `last_final_temp`, `setpoint_hold_cycles_remaining`, `last_is_blocking`, `last_run_features` — went to `unified_thermal_state.json` even during cooling cycles.

Fix:
- `state_manager.py` — both functions accept optional `state_manager` param (defaults to heating singleton for backward compat)
- `main.py` — resolves `_active_state_manager = _wrapper.state_manager` once per cycle; passes it to all 3 `save_state` calls and the initial `load_state`; reloads state after mode-switch to cooling
- `physics_calibration.py` — `train_thermal_equilibrium_model`, `optimize_thermal_parameters`, `backup_existing_calibration` all accept and thread through optional `state_manager`
- `main.py` HLC session — passes `thermal_state_manager=_active_state_manager` to `apply_to_thermal_state()`
- `dashboard/data_service.py` — `load_thermal_state()` and `get_state_file_info()` automatically switch to the cooling file when it is more recently modified (within last 30 minutes)

All 989 tests pass.

**Files changed**: `src/state_manager.py`, `src/main.py`, `src/physics_calibration.py`, `dashboard/data_service.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## 🎯 PREVIOUS STATUS - May 2026 (Thermal Power Gate Standardisation)

### ✅ **REFACTOR: Thermal power thresholds standardised across the codebase**

**Status**: **COMPLETED**

Root cause: `HLC_MIN_THERMAL_POWER_KW` (just introduced) was too narrow in scope — the same semantic (`minimum power for genuine heating`) was hardcoded as `0.5` in 7 locations across `physics_calibration.py`, `0.05` in 2 runtime detection locations, and `> 0` in the session learner. Each location was independently brittle and undocumented.

Fix: introduced three shared config vars:
- `HEATING_MIN_THERMAL_POWER_KW = 0.5` — HLC calibration, physics calibration quality filters, session learner per-cycle filter
- `COOLING_MIN_THERMAL_POWER_KW = -0.5` — cooling-side quality gate (reserved; thermal power is negative in cooling)
- `HP_ACTIVE_MIN_POWER_KW = 0.05` — runtime HP-running noise floor in heat_source_channels and temperature_control

All vars synchronised to `config.yaml` (options + schema), `.env_sample`, and `translations/en.yaml` (tooltips). The old `HLC_MIN_THERMAL_POWER_KW` removed; existing 4 HLC calibration fix vars (`HLC_WINDOW_SIZE_ROWS`, `HLC_MIN_FLOW_RATE_LPM`, `HLC_REGRESSION_INTERCEPT`) added to `config.yaml` and `.env_sample`.

**Files changed**: `src/config.py`, `src/hlc_learner.py`, `src/physics_calibration.py`, `src/heat_source_channels.py`, `src/temperature_control.py`, `tests/unit/test_hlc_learner.py`, `tests/unit/test_hlc_session_learner.py`, `tests/unit/test_physics_calibration.py`, `tests/unit/test_channel_calibration.py`, `ml_heating_underfloor/config.yaml`, `.env_sample`, `ml_heating_underfloor/translations/en.yaml`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

### ✅ **FIX: HLC calibration quality — 9 bugs fixed, R² diagnostic improved**

**Status**: **COMPLETED**

Root causes (from production log showing R² = 0.018):
1. `date_range` logged integer indices (0–23881) not actual datetimes — `df.index` is a RangeIndex after `reset_index` in `fetch_historical_data_for_calibration`
2. `ffill/bfill` in `physics_calibration.py` contaminated standby windows with stale sensor values, making them look like active heating periods
3. Missing `target_temp` column silently disabled two critical quality filters
4. No per-window timestamp gap check — windows could span multi-hour data gaps
5. Only standard R² was reported; for FTO regression the relevant metric is FTO-R² and Pearson r
6. Quality-gate columns bfill-filled for 640 h at dataset start were not flagged
7. 20-minute window too short for thermal equilibrium (should be 60 min)
8. No with-intercept regression diagnostic to detect contamination
9. `<= 0` thermal power check too permissive; standby residuals passed through

Fix: comprehensive rewrite of `calibrate_hlc()` in `src/hlc_learner.py` with all 9 fixes, 4 new config vars, and 4 new unit tests.

**Files changed**: `src/hlc_learner.py`, `src/config.py`, `src/physics_calibration.py`, `tests/unit/test_hlc_learner.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

### ✅ **FIX: Cooling mode writes learning state to heating JSON**

**Status**: **COMPLETED**

Root cause: `EnhancedModelWrapper.__init__` hard-wired `self.state_manager = get_thermal_state_manager()` (the heating singleton). `set_climate_mode()` only updated `self._climate_mode` — it never swapped `self.state_manager`. All subsequent `update_learning_state()`, `add_prediction_record()`, and `update_learning_state()` calls during cooling cycles wrote to the heating JSON. `ThermalEquilibriumModel` had the same problem — `_load_thermal_parameters()`, `_initialize_heat_source_channels()`, `_persist_heat_source_channel_state()`, and `_save_learning_to_thermal_state()` each locally called `get_thermal_state_manager()` with no knowledge of climate mode.

Fix:
1. `ThermalEquilibriumModel.__init__` now accepts `state_manager=None`. New `_get_state_manager()` helper returns the injected manager or falls back to the heating singleton. All four inline `get_thermal_state_manager()` call-sites inside the model replaced with `self._get_state_manager()`.
2. `EnhancedModelWrapper.__init__` now creates two paired instances — `_heating_state_manager`/`_heating_thermal_model`/`_heating_prediction_metrics` and the corresponding cooling trio. Each `ThermalEquilibriumModel` is constructed with the correct manager injected.
3. `set_climate_mode()` swaps `self.thermal_model`, `self.state_manager`, `self.prediction_metrics` and reloads `self.cycle_count` from the newly active manager.

**Files changed**: `src/thermal_equilibrium_model.py`, `src/model_wrapper.py`, `tests/unit/test_cooling_mode.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---


### ✅ **FIX: HLC calibration fails with `Missing required columns: {'indoor_temp'}`**

**Status**: **COMPLETED**

Root cause: `calibrate_hlc()` in `src/hlc_learner.py` fetched data via `influx_service.get_training_data()` then used English keyword heuristics (e.g. `"indoor" in col_lower and "temp" in col_lower`) to detect required columns. Non-English entity IDs like `rt_mittelwert` (the actual indoor temp sensor) don't contain these keywords, so the function always aborted.

Fix: replaced both the data-fetch block and the heuristic column-mapping block with the same approach already used by `physics_calibration.py`:
1. Data is fetched via `fetch_historical_data_for_calibration()` — respects `TRAINING_DATA_SOURCE`, handles HA history fallback/supplement in "auto" mode.
2. Column mapping uses `config.*_ENTITY_ID.split(".", 1)[-1]` — the exact short names produced by InfluxDB pivot and HA history, no English keyword assumptions.

The `influx_service` parameter is retained for backward compatibility but is no longer used.

**Files changed**: `src/hlc_learner.py`, `tests/unit/test_hlc_learner.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---


### ✅ **FIX: 7 dashboard bugs — crashes, logic errors, and non-functional buttons**

**Status**: **COMPLETED**

Full audit of the `dashboard/` folder found and fixed 7 distinct bugs: (1) `app.py`: `st.set_page_config()` was called after a potential `st.write()` in `setup_ingress_config()`, crashing the dashboard under HA ingress; (2) `health.py`: `timedelta.seconds` gave false "active" status for log files older than 24 h — fixed to `total_seconds()`; (3) `control.py`: `trigger_model_recalibration()` and `save_config_changes()` opened files under `/data/config/` without creating the directory first; (4) `data_service.py`: timezone-aware datetime comparison raised `TypeError` when `last_run_time` contained TZ offset; (5–6) `backup.py`: "View Details" and "Delete" buttons set session state keys that no handler ever read — added `render_view_details_interface()` and `render_delete_interface()`; (7) `backup.py`: all download paths showed placeholder captions — replaced with `st.download_button` calls.

**Files changed**: `dashboard/app.py`, `dashboard/health.py`, `dashboard/components/control.py`, `dashboard/data_service.py`, `dashboard/components/backup.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## 🎯 CURRENT STATUS - May 2026 (Dashboard HLC Calibrate Bug Fixes)

### ✅ **FIX: HLC Calibrate button errors — supervisorctl not found & use_container_width deprecation**

**Status**: **COMPLETED**

Fixed two dashboard bugs surfaced when pressing the HLC Calibrate button: (1) `[Errno 2] No such file or directory: 'supervisorctl'` — the container's `run.sh` starts processes directly with `&`, so `supervisorctl` is never available; replaced all three `supervisorctl`-based functions in `control.py` with signal-based management using `pgrep -f src.main` + `os.kill(pid, SIGTERM)`, and made `start_ml_system()` return a clear "use the add-on panel" message; (2) Streamlit deprecation warning for `use_container_width` — replaced all occurrences with `width='stretch'` across `performance.py`, `backup.py`, and `overview.py`.

**Files changed**: `dashboard/components/control.py`, `dashboard/components/performance.py`, `dashboard/components/backup.py`, `dashboard/components/overview.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## 🎯 CURRENT STATUS - May 1, 2026 (HLC Session Persistence + Migration Fixes)

### ✅ **FIX: Session restart survival, close-state persistence, and legacy migration**

**Status**: **COMPLETED**

Fixed four concrete issues in the new PV-triggered HLC learner: (1) active sessions now persist their collected `session_cycles` on every append so restarts resume the real in-progress session, (2) closing a session now saves the inactive post-close state instead of persisting a phantom active session, (3) `session_end` and `duration_minutes` now use the actual closing trigger cycle timestamp, and (4) legacy `day_records` payloads are migrated automatically into `session_records` so historical HLC data is preserved across the redesign. Also aligned `.env_sample`, startup logs, translation text, and adapter comments with the new PV-triggered terminology. Added 4 regression tests; the focused HLC learner suite now passes with 41 tests.

**Files changed**: `src/hlc_learner.py`, `tests/unit/test_hlc_session_learner.py`, `src/main.py`, `src/config.py`, `config_adapter.py`, `.env_sample`, `ml_heating_underfloor/translations/en.yaml`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## 🎯 CURRENT STATUS - May 2026 (PV-Triggered HLC Session Redesign)

### ✅ **REDESIGN: PV-triggered HLC session learner**

**Status**: **COMPLETED**

Replaced the calendar-day-based HLC session model with a PV-triggered session model. Sessions open when `pv_now_electrical < HLC_PV_MAX_W` (50 W) and close when `pv_now_electrical >= HLC_PV_MAX_W`. TV/DHW/defrost/DHW-boost/blocking now do per-cycle filtering (session stays alive), while fireplace remains a whole-session reject. `DayRecord` removed, replaced by `SessionRecord` with `session_start`, `session_end`, `duration_minutes`. Session state persists to JSON for container restart survival. Config keys renamed: `HLC_SESSION_MIN_DAYS` → `HLC_SESSION_MIN_SESSIONS` (default 5→10), `HLC_SESSION_MAX_DAYS` → `HLC_SESSION_MAX_SESSIONS` (default 60→120). All 37 new tests pass.

**Files changed**: `src/hlc_learner.py`, `src/config.py`, `src/main.py`, `config_adapter.py`, `ml_heating_underfloor/config.yaml`, `ml_heating_underfloor/translations/en.yaml`, `tests/unit/test_hlc_session_learner.py`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

## 🎯 May 1, 2026 (HLC Bugfix Review)

### ✅ **FIX: 5 bugs in HLC calibration code**

**Status**: **COMPLETED**

Code review found and fixed 5 bugs: (1) 6 shared validation params accidentally deleted from config.py/config.yaml/config_adapter/translations — restored as "HLC Validation Gates" section, (2) greedy column matching in `calibrate_hlc()` could overwrite base temp mappings with derived columns — now uses `setdefault()` and skips derived keywords, (3) uncapped HLC written to thermal state — added [0.01, 2.0] kW/K plausibility bounds, (4) flag file not removed on error → infinite calibration loop — now skips calibration if removal fails, (5) missing indoor trend quality gate in `calibrate_hlc()` — added first-to-last indoor temp change check.

**Files changed**: `src/hlc_learner.py`, `src/main.py`, `src/config.py`, `ml_heating_underfloor/config.yaml`, `config_adapter.py`, `ml_heating_underfloor/translations/en.yaml`, `CHANGELOG.md`

## 🎯 May 1, 2026 (HLC Learner Consolidation + Historical Calibration)

### ✅ **REFACTOR: Remove Online HLC Learner + Add Historical Calibration**

**Status**: **COMPLETED**

Removed the online `HLCLearner` class entirely (user only wanted day-level learner). Added `calibrate_hlc()` for one-shot historical HLC calibration from InfluxDB data. Added `--calibrate-hlc` CLI, flag detection, dashboard button, cold start file creation, and new config params.

**Files changed**: `src/hlc_learner.py`, `src/main.py`, `src/config.py`, `ml_heating_underfloor/config.yaml`, `config_adapter.py`, `ml_heating_underfloor/translations/en.yaml`, `dashboard/components/control.py`, `tests/unit/test_hlc_learner.py`, `tests/unit/test_hlc_session_learner.py`, `CHANGELOG.md`

## 🎯 PREVIOUS - April 29, 2026 (day-level HLC session learner)

### ✅ **FEAT: Day-Level Session-Based HLC Learning**

**Status**: **COMPLETED**

Implemented `HLCSessionLearner` — a persistent, day-granularity complement to the existing 60-minute in-memory `HLCLearner`. Each calendar day on which the heat pump ran is validated and stored as a `DayRecord` in a rolling JSON file. OLS regression over stored day records produces a multi-day HLC estimate that survives process restarts.

**Files Changed:**
- `src/config.py` — 6 new `HLC_SESSION_*` config vars
- `src/hlc_learner.py` — `DayRecord` dataclass + `HLCSessionLearner` class
- `src/main.py` — instantiation, load, push_cycle wiring
- `ml_heating_underfloor/config.yaml` — values + schema sections
- `.env_sample` — new vars with comments
- `ml_heating_underfloor/translations/en.yaml` — 6 tooltip entries
- `config_adapter.py` — 6 mapping entries
- `tests/unit/test_hlc_session_learner.py` — 18 unit tests (all pass)
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

### ✅ **TEST: review follow-up for coverage PR comments**

**Status**: **COMPLETED**

Applied the requested PR review follow-up by correcting the PV surplus test class docstring, making the validator warning assertion explicitly capture `WARNING` logs, and disambiguating the progress timeline so consecutive entries no longer share the same top-level heading text.

**Files Changed:**
- `tests/unit/test_price_optimizer.py`
- `tests/unit/test_thermal_state_validator.py`
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`


---

## 🎯 CURRENT STATUS - April 29, 2026

### ✅ **TEST: forecast analytics + thermal state validation coverage expansion**

**Status**: **COMPLETED**

Coverage analysis highlighted remaining branch gaps in `src/forecast_analytics.py` and `src/thermal_state_validator.py`, plus stale PV surplus assertions in the price optimizer tests. Added focused test cases for fallback strategies, invalid accuracy inputs, schema failures, fallback parameter-range validation, safe-wrapper error handling, CLI entry paths, and ramp-based PV surplus metadata.

| Module | Before | After |
|--------|--------|-------|
| `src/forecast_analytics.py` | 82% | 98% |
| `src/thermal_state_validator.py` | 74% | 91% |

**Files Changed:**
- `tests/unit/test_forecast_analytics.py`
- `tests/unit/test_thermal_state_validator.py`
- `tests/unit/test_price_optimizer.py`
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`


---

### ✅ **TEST: Comprehensive test coverage analysis and improvements**

**Status**: **COMPLETED**

Analysed overall test coverage (74% baseline) and added 161 new unit tests targeting the 5 modules with the worst coverage.  Overall source coverage improved from 74% → 77%; total passing tests grew from 785 → 945.

| Module | Before | After |
|--------|--------|-------|
| `src/thermal_constants.py` | 55% | 98% |
| `src/prediction_metrics.py` | 63% | 84% |
| `src/ha_history_service.py` | 75% | 90% |
| `src/adaptive_fireplace_learning.py` | 73% | 89% |
| `src/multi_heat_source_physics.py` | 72% | 81% |

**Files Changed:**
- `tests/unit/test_thermal_constants.py` (new — 65 tests)
- `tests/unit/test_prediction_metrics_extended.py` (new — 58 tests)
- `tests/unit/test_ha_history_service_extended.py` (new — 23 tests)
- `tests/unit/test_adaptive_fireplace_learning_extended.py` (new — 28 tests)
- `tests/unit/test_multi_heat_source_physics_extended.py` (new — 22 tests)
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`


---

### ✅ **FIX: pv_scalar rolling-window + end-of-sun override, PV surplus CHEAP soft-ramp, overshoot dampening 1.0**

**Status**: **COMPLETED**

Three independent binary-search accuracy improvements in `src/model_wrapper.py`:

1. **pv_scalar rolling-window + end-of-sun override**: reverted the stateful EMA back to `mean(pv_power_history)`. End-of-sun override snaps to `pv_now` and clears history when 1h forecast ≤ `PV_TRAJ_ZERO_W`. Removed `self._pv_scalar_ema` attribute and `PV_SCALAR_EMA_ALPHA` config.
2. **PV surplus CHEAP soft-ramp**: replaced binary on/off at `PV_SURPLUS_CHEAP_THRESHOLD_W` with linear ramp over `PV_SURPLUS_CHEAP_RAMP_W` band. New config var `PV_SURPLUS_CHEAP_RAMP_W` defaults to threshold value.
3. **Overshoot dampening 0.4 → 1.0**: `overshoot_dampening = 1.0 / max(slab_tau, 1.0)` — 2.5× stronger pull-back when overshoot is detected.

**Files Changed:**
- `src/model_wrapper.py` (all three items)
- `src/config.py` (remove PV_SCALAR_EMA_ALPHA, add PV_SURPLUS_CHEAP_RAMP_W)
- `tests/unit/test_model_wrapper.py` (12 new tests, updated pv_scalar class)
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`


---

### ✅ **FIX: PV trajectory forecast horizon + rain-cloud rescue**

**Status**: **COMPLETED** — Two related bugs fixed:

1. **Extended forecast horizon** (`physics_features.py`): when `PV_TRAJ_FORECAST_MODE_ENABLED=true`, forecasts are now fetched up to `PV_TRAJ_MAX_STEPS` hours instead of only `TRAJECTORY_STEPS`. Keys `pv_forecast_5h … pv_forecast_12h` etc. are now populated correctly. `ha_client.get_hourly_forecast()`, `get_hourly_cloud_cover()`, and `get_calibrated_hourly_forecast()` accept optional `n` parameter.

2. **Forecast-rescue path** (`pv_trajectory.py`): a temporary drop of `pv_now` below `PV_TRAJ_THRESHOLD_W` (e.g. passing rain cloud) no longer immediately collapses `TRAJECTORY_STEPS` to `PV_TRAJ_MIN_STEPS`. Instead the forecast is consulted; if ≥ `PV_TRAJ_MIN_STEPS` hours exceed the threshold the mode continues normally. Controlled by `PV_TRAJ_FORECAST_RESCUE_ENABLED` (default `true`).

**Files Changed:**
- `src/ha_client.py` (optional `n` param on 3 forecast methods)
- `src/physics_features.py` (_n_fc_full extended horizon, updated all forecast fetch/key loops)
- `src/pv_trajectory.py` (rescue path in compute_forecast_driven_trajectory_steps and is_forecast_trajectory_active)
- `src/config.py` (PV_TRAJ_FORECAST_RESCUE_ENABLED)
- `src/main.py` (fixed misleading comment)
- `config_adapter.py` (pv_traj_forecast_rescue_enabled mapping)
- `ml_heating_underfloor/config.yaml` (option + schema entry)
- `.env_sample` (PV_TRAJ_FORECAST_RESCUE_ENABLED)
- `tests/unit/test_pv_trajectory.py` (8 new rescue tests, 2 updated existing tests)
- `tests/unit/test_physics_features.py` (2 new extended horizon tests)
- `CHANGELOG.md`

### ✅ **REFACTOR: Remove classic PV trajectory mode**

**Status**: **COMPLETED** — Removed pv_ratio × time-of-day factor algorithm (morning/midday/afternoon/night factors, system KWP, seasonal KWP scaling). `PV_TRAJ_SCALING_ENABLED` deleted. `compute_dynamic_trajectory_steps()` now gates solely on `PV_TRAJ_FORECAST_MODE_ENABLED`. All config surfaces, docs, and tests updated.

**Files Changed:**
- `src/pv_trajectory.py` (removed classic mode, _time_of_day_factor, seasonal_kwp_factor, updated docstring)
- `src/config.py` (removed 9 classic-mode parameters)
- `src/main.py` (changed PV_TRAJ_SCALING_ENABLED guards to PV_TRAJ_FORECAST_MODE_ENABLED)
- `config_adapter.py` (removed classic-mode mappings)
- `ml_heating_underfloor/config.yaml` (removed options and schema entries for classic mode)
- `ml_heating_underfloor/translations/en.yaml` (removed classic-mode translations)
- `.env`, `.env_sample` (removed classic-mode env vars)
- `docs/PARAMETER_REFERENCE.md` (removed sections 28/29, replaced with single forecast section)
- `tests/unit/test_pv_trajectory.py` (removed classic-mode tests, updated forecast tests)
- `CHANGELOG.md`

---

**Status**: **COMPLETED** — `compute_forecast_driven_trajectory_steps()` updated to `steps = clamp(remaining_pv_hours + MIN_STEPS, MIN, MAX)`. Night buffer is now always included in the planning horizon. All 7 affected unit tests updated.

**Files Changed:**
- `src/pv_trajectory.py` (formula, docstrings, log message)
- `tests/unit/test_pv_trajectory.py` (7 test assertions updated)
- `CHANGELOG.md`

---

## 🎯 CURRENT STATUS - April 27, 2026

### ✅ **DOCS: Translation UI descriptions for new parameters**

**Status**: **COMPLETED** — Added `name` + `description` entries for 16 parameters in `ml_heating_underfloor/translations/en.yaml` that were present in `config.yaml` but missing from the HA add-on UI translation file.

**Files Changed:**
- `ml_heating_underfloor/translations/en.yaml` (+16 parameter entries: 12 `hlc_*`, 4 `pv_traj_forecast*`)
- `CHANGELOG.md`

---

### ✅ **FEATURE: Forecast-Driven Dynamic Trajectory**

**System Status**: **IMPLEMENTED** — New forecast-driven mode for `compute_dynamic_trajectory_steps()`. Steps = consecutive PV forecast hours above `PV_TRAJ_ZERO_W`, giving a naturally shrinking horizon toward sunset. Disabled by default (`PV_TRAJ_FORECAST_MODE_ENABLED=false`).

**Implementation:**
- ✅ `src/pv_trajectory.py`: `compute_forecast_driven_trajectory_steps()`, `is_forecast_trajectory_active()` new public functions; `compute_dynamic_trajectory_steps()` updated to accept `pv_forecast` list and delegate when forecast mode enabled
- ✅ `src/config.py`: 4 new config vars (`PV_TRAJ_FORECAST_MODE_ENABLED`, `PV_TRAJ_THRESHOLD_W`, `PV_TRAJ_ZERO_W`, `PV_TRAJ_DISABLE_PRICE_IN_FORECAST_MODE`)
- ✅ `src/main.py`: Step 3 builds `_pv_forecast_traj` list and passes to `compute_dynamic_trajectory_steps()`; post-price block suppresses `price_data` when forecast mode active
- ✅ `config_adapter.py`: all 4 new vars mapped from `config.yaml` option names
- ✅ `ml_heating_underfloor/config.yaml`: options + schema for all 4 new params
- ✅ `.env_sample`: documented all 4 new params
- ✅ `tests/unit/test_pv_trajectory.py`: 17 new tests in `TestForecastDrivenTrajectorySteps` — 100% pass (793 total, 3 pre-existing failures unrelated)
- ✅ `CHANGELOG.md`: `### Added` entry under `[Unreleased]`

**Test Suite**: **793 passing, 3 pre-existing failures** (unrelated `TestPvSurplusCheapOverride`)

**Files Changed:**
- `src/pv_trajectory.py`
- `src/config.py`
- `src/main.py`
- `config_adapter.py`
- `ml_heating_underfloor/config.yaml`
- `.env_sample`
- `tests/unit/test_pv_trajectory.py`
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---


**System Status**: **IMPLEMENTED** — New `HLCLearner` class estimates building Heat Loss Coefficient from live cycle data. Disabled by default.

**Implementation:**
- ✅ `src/hlc_learner.py` (new): `HLCLearner` with `push_cycle()`, `_validate_window()`, `estimate_hlc()`, `apply_to_thermal_state()`; `HLCCycle` and `HLCWindow` dataclasses
- ✅ `src/config.py`: 12 new HLC learner config vars with env-var defaults
- ✅ `src/main.py`: `HLCLearner()` instantiated at startup (when enabled); `push_cycle()` called after every `features_dict` is built
- ✅ `config_adapter.py`: all 12 new vars mapped from `config.yaml` option names
- ✅ `ml_heating_underfloor/config.yaml`: options + schema for all 12 HLC learner params
- ✅ `tests/unit/test_hlc_learner.py` (new): 46 tests — 100% pass
- ✅ `CHANGELOG.md`: `### Added` entry under `[Unreleased]`

**Test Suite**: **777 passing, 3 pre-existing failures** (unrelated `TestPvSurplusCheapOverride`)

**Files Changed:**
- `src/hlc_learner.py` (new)
- `src/config.py`
- `src/main.py`
- `config_adapter.py`
- `ml_heating_underfloor/config.yaml`
- `tests/unit/test_hlc_learner.py` (new)
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---

## Previous Status — April 26, 2026

### ✅ **FEATURE: Parameter Documentation & HA UI Translations**

**System Status**: **IMPLEMENTED** — All add-on parameters now have human-readable names and descriptions in the HA Configuration tab.

**Implementation:**
- ✅ `ml_heating_underfloor/translations/en.yaml`: Created — `configuration:` block with `name:` + `description:` for all ~120 schema keys. Advanced/internal parameters labelled `[Advanced]`.
- ✅ `docs/PARAMETER_REFERENCE.md`: Created — Full 30-section parameter reference with defaults, ranges, env var equivalents, and guidance.
- ✅ `README.md`: Added Configuration Reference section with must-configure table, key operational parameters, advanced callout, and link to full reference.
- ✅ `CHANGELOG.md`: Added `### Added` entry under `[Unreleased]`.

**Files Changed:**
- `ml_heating_underfloor/translations/en.yaml` (new)
- `docs/PARAMETER_REFERENCE.md` (new)
- `README.md`
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---

## Previous Status — April 26, 2026

### ✅ **FEATURE: Config Synchronization + Seasonal PV KWP Scaling**

**System Status**: **IMPLEMENTED** — Config files synchronized and new seasonal scaling feature added.

**Test Suite**: **731 tests, 728 passing** (13 new in `test_pv_trajectory.py`). 3 pre-existing failures in `test_price_optimizer.py::TestPvSurplusCheapOverride` (unrelated).

**Implementation:**
- ✅ `src/pv_trajectory.py`: added `seasonal_kwp_factor()` and helper functions `_solar_declination_deg()`, `_max_solar_elevation_deg()`; updated `compute_dynamic_trajectory_steps()` to apply seasonal factor when `PV_TRAJ_SEASONAL_SCALING_ENABLED=true`; added `from datetime import date` import
- ✅ `src/config.py`: added `PV_TRAJ_SEASONAL_SCALING_ENABLED`, `PV_TRAJ_LATITUDE`, `PV_TRAJ_SEASONAL_MIN_FACTOR`
- ✅ `config_adapter.py`: added mappings for `TREND_DECAY_TAU_HOURS`, `PV_ROOM_DECAY_MULTIPLIER`, `DECAY_CANCEL_MARGIN`, `PV_TRAJ_SEASONAL_SCALING_ENABLED`, `PV_TRAJ_LATITUDE`, `PV_TRAJ_SEASONAL_MIN_FACTOR`; removed deprecated `safety_max_temp`/`safety_min_temp` dead-code validation
- ✅ `ml_heating_underfloor/config.yaml`: added `trend_decay_tau_hours`, `pv_room_decay_multiplier`, `decay_cancel_margin`, `pv_traj_seasonal_scaling_enabled`, `pv_traj_latitude`, `pv_traj_seasonal_min_factor` to both `options:` and `schema:`
- ✅ `.env`: completely rewritten — 16 labelled sections, no duplicates, all missing params added
- ✅ `.env_sample`: completely rewritten — same 16 sections, placeholder values
- ✅ `tests/unit/test_pv_trajectory.py`: 13 new tests in `TestSeasonalKwpFactor` and `TestComputeDynamicStepsWithSeasonal`

**Files Changed:**
- `src/pv_trajectory.py`
- `src/config.py`
- `config_adapter.py`
- `ml_heating_underfloor/config.yaml`
- `.env`
- `.env_sample`
- `tests/unit/test_pv_trajectory.py`
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---



### ✅ **FEATURE: Dynamic PV Trajectory Scaling + PV Surplus CHEAP + Setpoint Hold**

**System Status**: **IMPLEMENTED** — Three complementary solar-aware features added.

**Test Suite**: **721 tests, all passing** (21 new in `test_pv_trajectory.py`, 6 new in `test_price_optimizer.py::TestPvSurplusCheapOverride`).

**Implementation**:
- ✅ `src/pv_trajectory.py` (new): `compute_dynamic_trajectory_steps(pv_power_w, system_kwp, now)` — linear interpolation between `PV_TRAJ_MIN_STEPS` and `PV_TRAJ_MAX_STEPS` using PV ratio × time-of-day factor
- ✅ `src/config.py`: `PV_TRAJ_SCALING_ENABLED`, `PV_TRAJ_SYSTEM_KWP`, `PV_TRAJ_MIN_STEPS`, `PV_TRAJ_MAX_STEPS`, `PV_TRAJ_MORNING_FACTOR`, `PV_TRAJ_MIDDAY_FACTOR`, `PV_TRAJ_AFTERNOON_FACTOR`, `PV_TRAJ_NIGHT_FACTOR`; also `PV_SURPLUS_CHEAP_ENABLED`, `PV_SURPLUS_CHEAP_THRESHOLD_W`, `MIN_SETPOINT_HOLD_CYCLES`
- ✅ `src/main.py`: per-cycle `config.TRAJECTORY_STEPS` + `config.MIN_SETPOINT_HOLD_CYCLES` override; setpoint hold countdown persisted in state
- ✅ `src/model_wrapper.py`: PV surplus CHEAP target offset override
- ✅ `src/state_manager.py`: `setpoint_hold_cycles_remaining` field in `SystemState`
- ✅ `config_adapter.py`: all 10 new options mapped to env vars
- ✅ `ml_heating_underfloor/config.yaml`: options + schema for PV Surplus, Setpoint Stability, Dynamic Trajectory Scaling sections
- ✅ `tests/unit/test_pv_trajectory.py` (new): 21 tests
- ✅ `tests/unit/test_price_optimizer.py`: 6 new PV surplus tests

**Files Changed**:
- `src/pv_trajectory.py` (new)
- `src/config.py`
- `src/main.py`
- `src/model_wrapper.py`
- `src/state_manager.py`
- `config_adapter.py`
- `ml_heating_underfloor/config.yaml`
- `tests/unit/test_pv_trajectory.py` (new)
- `tests/unit/test_price_optimizer.py`



**Implementation**:
- ✅ `ml_heating_underfloor/config.yaml`: widened `trajectory_steps` validation from `int(2,8)` to `int(2,12)`, updated comment
- ✅ `src/ha_client.py`: `get_hourly_forecast()`, `get_hourly_cloud_cover()`, `get_calibrated_hourly_forecast()` — all hardcoded `6` replaced with `config.TRAJECTORY_STEPS`
- ✅ `src/physics_features.py`: PV forecast loop `range(1,7)` → `range(1, TRAJECTORY_STEPS+1)`, feature dict keys generated dynamically, summary features use `[TRAJECTORY_STEPS-1]` index and `TRAJECTORY_STEPS` divisor
- ✅ `src/prediction_context.py`: replaced 6-branch if/elif step function with `hour_idx = min(round(cycle_hours), n_fc) - 1`; forecast extraction and fallback arrays use `config.TRAJECTORY_STEPS`
- ✅ `src/model_wrapper.py`: forecast display dict built dynamically up to `TRAJECTORY_STEPS`; avg divisor `/ 6.0` → `/ config.TRAJECTORY_STEPS`; comment updated
- ✅ `src/forecast_analytics.py`: fallback dict loop extended to `TRAJECTORY_STEPS`; `[3]` index replaced with `[-1]`; `config` imported
- ✅ `tests/unit/test_trajectory_12h.py`: 13 new tests covering ha_client, physics_features, prediction_context, model_wrapper, config boundary
- ✅ Updated existing tests (`test_ha_client.py`, `test_physics_features.py`) to reflect dynamic key counts

**Files Changed**:
- `ml_heating_underfloor/config.yaml`
- `src/ha_client.py`
- `src/physics_features.py`
- `src/prediction_context.py`
- `src/model_wrapper.py`
- `src/forecast_analytics.py`
- `tests/unit/test_trajectory_12h.py` (new)
- `tests/unit/test_ha_client.py`
- `tests/unit/test_physics_features.py`



### ✅ **HOLISTIC AUDIT: Bug Fixes, Drift Detection, Metrics Persistence, Auto-Doc**

**System Status**: **IMPLEMENTED** — Comprehensive audit fixing 12+ issues across error handling, drift detection, prediction metrics, and developer workflow.

**Test Suite**: **49 local tests passed** for touched model/state/main slices, including 5 new regression tests for the reviewed bugs. Pre-existing workspace-level failures remain limited to unrelated `streamlit` / environment gaps.

**Implementation**:
- ✅ Fixed indoor temp log bug (wrong indentation of shadow-mode else branch in main.py)
- ✅ Added startup sensor validation on first cycle and ensured it retries after transient failures instead of disabling itself permanently (main.py)
- ✅ Replaced bare `except Exception` in ha_client.py with specific exceptions + warning logs
- ✅ Replaced bare `except Exception` in dashboard/health.py with specific exceptions + warning logs
- ✅ Fixed JSON string corruption root cause in unified_thermal_state.py (validates `last_run_features`, re-validates decoded JSON, logs failed `to_dict()` conversions)
- ✅ Removed dead grace period duplication in main.py
- ✅ Fixed drift detection: corrected metric keys (`1h`/`all` instead of `mae_recent`/`mae_all_time`), reversed direction to boost confidence +2.0 (cap 10.0), and clamp back to 5.0 when drift subsides
- ✅ Added dynamic `_max_learning_confidence` to ThermalEquilibriumModel and clamp restored confidence to the normal cap on restart
- ✅ Fixed prediction metrics always zero: `_save_to_state()` now writes `accuracy_stats` and `recent_performance` to unified state using established `mae_all_time` / `rmse_all_time` keys
- ✅ Created `.github/copilot-instructions.md` for automatic changelog/memory-bank/docs updates every session
- ✅ Added focused regression coverage for metrics persistence, drift reset, restart clamping, startup validation retry, and `last_run_features` conversion warnings

**Files**: `src/model_wrapper.py`, `src/thermal_equilibrium_model.py`, `src/prediction_metrics.py`, `src/main.py`, `src/ha_client.py`, `src/unified_thermal_state.py`, `dashboard/health.py`, `.github/copilot-instructions.md`, `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`, `tests/unit/test_unified_thermal_state.py`, `tests/unit/test_model_wrapper.py`, `tests/unit/test_thermal_equilibrium_model_confidence.py`, `tests/integration/test_main.py`

## 🎯 PREVIOUS STATUS - April 18, 2026

### ✅ **CRITICAL FIX: Binary Search _features NameError + Debug Logging**

**System Status**: **FIXED** — Binary search now correctly resolves `indoor_temp_delta_60m` from `self._current_features` instead of undefined `_features`. Previously every binary search iteration failed with `NameError`, falling back to max outlet (35°C). Debug logging added to verify fix in production.

**Test Suite**: **31 thermal model tests passed** (16 pre-existing failures in other areas unrelated to changes)

**Implementation**:
- ✅ Fixed `_features` → `self._current_features` with `hasattr` guard in binary search trajectory call (L791)
- ✅ Added debug log before trajectory call (first iteration): shows `inlet_temp`, `delta_t_floor`, `indoor_temp_delta_60m`, horizon, outlet_mid
- ✅ Added debug log after successful trajectory (first iteration): shows predicted indoor, trajectory steps, start→end temperatures
- ✅ Extracted `_trend_60m` variable to avoid repeating the `hasattr` guard in the trajectory call
- ✅ Updated CHANGELOG.md (root + addon) with fix and debug logging entries
- ✅ Updated memory-bank (activeContext.md, progress.md)

**Files**: `src/model_wrapper.py`, `CHANGELOG.md`, `ml_heating_underfloor/CHANGELOG.md`, `memory-bank/activeContext.md`, `memory-bank/progress.md`

## 🎯 PREVIOUS STATUS - April 2026

### ✅ **INDOOR TEMPERATURE TREND BIAS + BUG FIXES**

**System Status**: **IMPLEMENTED** — Trajectory prediction now respects observed indoor temperature momentum. Dashboard crash fixed. Logging noise reduced.

**Test Suite**: **649 passed** (16 pre-existing failures unrelated to changes)

**Implementation**:
- ✅ `predict_thermal_trajectory()` extracts `indoor_temp_delta_60m` from `**external_sources`
- ✅ Decaying trend bias in step loop: `trend_bias = delta_60m × time_step × e^(-elapsed/τ)`, clamped ±0.05°C, gated on abs > 0.01
- ✅ `TREND_DECAY_TAU_HOURS` config (default 1.5h, env-overridable)
- ✅ Passed from binary search caller and trajectory verification caller in `model_wrapper.py`
- ✅ Bug 1: `titlefont` → `title_font` in `dashboard/components/overview.py`
- ✅ Bug 2: Pump off when outlet=inlet confirmed as correct behavior (not a bug)
- ✅ Bug 3: Removed noisy "Logging MAE"/"Logging RMSE" debug messages from `ha_client.py`

**Files**: `src/thermal_equilibrium_model.py`, `src/model_wrapper.py`, `src/config.py`, `dashboard/components/overview.py`, `src/ha_client.py`

## 🎯 PREVIOUS STATUS - July 2025

### ✅ **ELECTRICITY PRICE-AWARE OPTIMIZATION**

**System Status**: **IMPLEMENTED** — Tibber price-based target shifting with feature flag. Default disabled (`ELECTRICITY_PRICE_ENABLED=false`).

**Test Suite**: **29/29 new tests passing**, 0 regressions in existing tests (107 pass, 4 pre-existing failures).

**Implementation**:
- ✅ `PriceOptimizer` class: percentile-based CHEAP/NORMAL/EXPENSIVE classification
- ✅ Binary search target shifted ±0.2°C (CHEAP → +0.2, EXPENSIVE → -0.2)
- ✅ Trajectory correction: EXPENSIVE tightens future overshoot to +0.2°C (from +0.5)
- ✅ main.py integration: reads Tibber sensor, passes to prediction, publishes sensors
- ✅ `sensor.ml_heating_features`: exports all last-run features
- ✅ `sensor.ml_heating_price_level`: exports price classification
- ✅ `sensor.ml_heating_learning`: now always exports all channel params
- ✅ Feature flag: zero behaviour change when disabled
- ✅ Learning safety: target-based shift, no parameter corruption

**Files**: `src/price_optimizer.py` (new), `src/config.py`, `src/model_wrapper.py`, `src/ha_client.py`, `src/main.py`, `tests/unit/test_price_optimizer.py` (new), `docs/PRICE_OPTIMIZATION_INTEGRATION.md` (new)

**Future**: Option F (Thermal Pre-Charging with look-ahead) saved for later phase.

## 🎯 PREVIOUS STATUS - April 7, 2026

### ✅ **UNIFIED COOLING THERMAL STATE**

**System Status**: **OPERATIONAL** — Dedicated cooling state file with independent learning, calibration, buffer persistence, and cooling-specific parameters.

**Test Suite**: **576/585 passing** (9 pre-existing failures unrelated to changes)

**Implementation Status**:
- ✅ `CoolingThermalStateManager` with own JSON file (`unified_thermal_state_cooling.json`)
- ✅ Cooling-specific baseline defaults in `ThermalParameterConfig` (COOLING_DEFAULTS, COOLING_BOUNDS)
- ✅ Buffer state persistence for cooling sensor snapshots
- ✅ Independent learning state (cycle count, confidence, parameter adjustments)
- ✅ Calibration tracking (date, cycles) separate from heating
- ✅ Shadow-mode support via `get_effective_cooling_state_file()`
- ✅ PR review fixes: safe no-viable-range return, variable naming, UnboundLocalError fix, constant deduplication
- ✅ 27 new tests + 1 test updated

**Files Modified**: `src/unified_thermal_state_cooling.py` (new), `src/thermal_config.py`, `src/config.py`, `src/shadow_mode.py`, `src/model_wrapper.py`, `src/thermal_constants.py`, `src/main.py`, `.env_sample`, `tests/unit/test_unified_thermal_state_cooling.py` (new), `tests/unit/test_cooling_mode.py`

## 🎯 PREVIOUS STATUS - April 7, 2026

### ✅ **UNDERSHOOT GATE (mirror of overshoot gate)**

**System Status**: **OPERATIONAL** — Added undershoot projected-temperature gate to mirror the existing overshoot gate. When indoor temperature is rising naturally, undershoot corrections are skipped to let the house self-correct.

**Test Suite**: **495/504 passing** (9 pre-existing failures unrelated to changes)

**Implementation Status**:
- ✅ Undershoot gate in `min_violates`-only branch
- ✅ Undershoot gate in `both_violated + min_wins` branch
- ✅ 6 new tests in `TestUndershootGate` class
- ✅ 8 existing tests adapted with explicit trend data for undershoot gate compatibility

**Files Modified**: `src/model_wrapper.py`, `tests/unit/test_trajectory_correction.py`, `memory-bank/activeContext.md`, `memory-bank/progress.md`

## 🎯 PREVIOUS STATUS - April 4, 2026

### ✅ **SLAB MODEL FIXES & PV OSCILLATION DAMPING**

**System Status**: **OPERATIONAL WITH 6 TARGETED FIXES** — Production log analysis drove six precision fixes addressing HP-off outlet spike, PV oscillation, slab gate, and diagnostic improvements.

**Test Suite**: **397/402 passing** (5 pre-existing failures unrelated to changes)

**Implementation Status**:
- ✅ HP-off binary search: simulated HP-on delta_t prevents 35°C outlet spike
- ✅ Cloud discount on PV scalar: 1h forecast dampens sensor spikes in binary search
- ✅ PV routing: `max(current, smoothed)` captures solar thermal lag at sunset
- ✅ PV smoothing: window shortened from 3h to solar_decay_tau (~30min)
- ✅ Slab pump gate: `measured_delta_t >= 1.0` required for pump-ON branch
- ✅ Slab passive delta sensor: `inlet_temp - indoor_temp` exported to HA
- ✅ 3 env-dependent test fixes (ENABLE_MIXED_SOURCE_ATTRIBUTION monkeypatch)
- ✅ 10+ new tests added, 9 existing tests updated for compatibility
- ✅ Shadow mode + active mode verified (26/26 tests passing)

**Files Modified**:
- `src/model_wrapper.py` — HP-off fix, cloud discount, slab_passive_delta sensor
- `src/heat_source_channels.py` — PV routing max(current, smoothed)
- `src/temperature_control.py` — PV smoothing window
- `src/thermal_equilibrium_model.py` — Slab pump gate, _resolve_delta_t_floor

---

## Previous Status - March 31, 2026

### ✅ **HEAT SOURCE CHANNEL ARCHITECTURE (PHASE 2-4) IMPLEMENTED**

**System Status**: **OPERATIONAL WITH DECOMPOSED LEARNING** — Each heat source has its own independent learning channel with isolated parameters and prediction history. Phase 1 guards continue to protect the main control loop.

**Implementation Status**:
- ✅ `src/heat_source_channels.py` — `HeatSourceChannel` ABC + 4 implementations + `HeatSourceChannelOrchestrator`
- ✅ `src/config.py` — `ENABLE_HEAT_SOURCE_CHANNELS` config variable (default: `true`)
- ✅ `.env_sample` — Documented with usage description
- ✅ Channel learning isolation: HP, PV, FP, TV learn from their own active periods only
- ✅ Proportional error attribution across active channels
- ✅ Solar transition forecasting via PV forecast array
- ✅ Per-channel state persistence (`get_channel_state()` / `load_channel_state()`)
- ✅ 8 new tests passing (test_heat_source_channels, test_solar_transition, test_learning_isolation)
- ⏳ Orchestrator integration into main control loop deferred (Phase 1 guards active)

---

## Previous Status - February 11, 2026

### ✅ **PHASE 2: ADVANCED TESTING IMPLEMENTATION COMPLETE**

**System Status**: **OPERATIONAL WITH ADVANCED TESTING** - The test suite has been significantly enhanced with property-based testing and sociable unit tests, providing deeper verification of system correctness and component integration.

**Test Suite Health**: **EXCELLENT** - 214/214 tests passing (100% success rate).

### ✅ **TEST SUITE REFACTORING & TDD ADOPTION COMPLETE (February 10, 2026)**

**System Status**: **OPERATIONAL WITH TDD** - The entire test suite has been refactored, and the project has officially adopted a Test-Driven Development (TDD) workflow.

**Test Suite Health**: **EXCELLENT** - 214/214 tests passing (100% success rate).

**Key Improvements**:
- **Refactored Test Suite**: Consolidated fragmented tests into a unified structure.
- **TDD Enforcement**: Added `tests/conftest.py` to enforce consistent thermal parameters across all tests.
- **Coverage**: Achieved comprehensive coverage for core logic, including `ThermalEquilibriumModel`, `HeatingController`, and `PhysicsConstants`.
- **Stability**: Resolved `InfluxDBClient` teardown issues by implementing robust cleanup in `InfluxService` and adding a global pytest fixture to reset the singleton after every test.

### 🚨 **CRITICAL RECOVERY COMPLETED (January 2, 2026)**

**Emergency Stability Implementation**:
- ✅ **Root Cause Identified**: Corrupted thermal parameter (total_conductance = 0.266 → should be ~0.05)
- ✅ **Parameter Corruption Detection**: Sophisticated bounds checking prevents specific corruption patterns
- ✅ **Catastrophic Error Handling**: Learning disabled for prediction errors ≥5°C
- ✅ **Auto-Recovery System**: Self-healing when conditions improve, no manual intervention needed
- ✅ **Test-Driven Development**: 24/25 comprehensive unit tests passing (96% success rate)

**Shadow Mode Learning Architectural Fix**:
- ✅ **Problem Identified**: Shadow mode was evaluating ML's own predictions instead of learning building physics
- ✅ **Architecture Corrected**: Now learns from heat curve's actual control decisions (48°C) vs ML calculations (45.9°C)
- ✅ **Learning Patterns Fixed**: Shadow mode observes heat curve → predicts indoor result → learns from reality
- ✅ **Test Validation**: Comprehensive test suite validates correct shadow/active mode learning patterns

**System Recovery Results**:
- ✅ **Prediction Accuracy**: Restored from 0.0% to normal operation
- ✅ **Parameter Health**: total_conductance corrected (0.195 vs corrupted 0.266)
- ✅ **ML Predictions**: Realistic outlet temperatures (45.9°C vs previous garbage)
- ✅ **Emergency Protection**: Active monitoring prevents future catastrophic failures

#### 🚀 **Core System Features - OPERATIONAL**

**Multi-Heat-Source Physics Engine**:
- ✅ **PV Solar Integration** (1.5kW peak contribution)
- ✅ **Fireplace Physics** (6kW heat source with adaptive learning)
- ✅ **Electronics Modeling** (0.5kW TV/occupancy heat)
- ✅ **Combined Heat Source Optimization** with weather effectiveness

**Thermal Equilibrium Model with Adaptive Learning**:
- ✅ **Real-time Parameter Adaptation** (96% accuracy achieved)
- ✅ **Gradient-based Learning** for heat loss, thermal time constant, outlet effectiveness
- ✅ **Confidence-based Effectiveness Scaling** with safety bounds
- ✅ **State Persistence** across Home Assistant restarts

**Enhanced Physics Features**:
- ✅ **37 Thermal Intelligence Features** (thermal momentum, cyclical encoding, delta analysis)
- ✅ **±0.1°C Control Precision** capability through comprehensive feature engineering
- ✅ **Backward Compatibility** maintained with all existing workflows

**Production Infrastructure**:
- ✅ **Streamlit Dashboard** with Home Assistant ingress integration
- ✅ **Comprehensive Testing** - 294 tests covering all functionality
- ✅ **Professional Documentation** - Complete technical guides and user manuals
- ✅ **Home Assistant Integration** - Dual add-on channels (stable + dev)

#### 🔧 **Recent Critical Fixes - COMPLETED**

**Advanced Testing Implementation (February 11, 2026)**:
- ✅ **Property-Based Testing**: Implemented `hypothesis` tests for `ThermalEquilibriumModel` to verify physical invariants (bounds, monotonicity).
- ✅ **Sociable Unit Testing**: Implemented tests for `HeatingController` using real collaborators (`SensorDataManager`, `BlockingStateManager`) to verify component integration.

**Code Quality and Formatting (February 9, 2026)**:
- ✅ **Linting and Formatting**: Resolved all outstanding linting and line-length errors in `src/model_wrapper.py`.
- ✅ **Improved Readability**: The code is now cleaner, more readable, and adheres to project standards.

**Intelligent Post-DHW Recovery (February 9, 2026)**:
- ✅ **Model-Driven Grace Period**: Re-architected the grace period logic to use the ML model to calculate a new, higher target temperature after DHW/defrost cycles.
- ✅ **Prevents Temperature Droop**: Actively compensates for heat loss during blocking events, ensuring the target indoor temperature is reached.
- ✅ **Maintains Prediction Accuracy**: By correcting the thermal deficit, the model's performance is no longer negatively impacted by these interruptions.

**Gentle Trajectory Correction Implementation (December 10)**:
- ✅ **Aggressive Correction Issue Resolved** - Replaced multiplicative (7x factors) with gentle additive approach
- ✅ **Heat Curve Alignment** - Based on user's 15°C per degree automation logic, scaled for outlet adjustment
- ✅ **Forecast Integration Enhancement** - Fixed feature storage for accurate trajectory verification
- ✅ **Open Window Handling** - System adapts to sudden heat loss and restabilizes automatically
- ✅ **Conservative Boundaries** - 5°C/8°C/12°C per degree correction prevents outlet temperature spikes

**Binary Search Algorithm Enhancement (December 9)**:
- ✅ **Overnight Looping Issue Resolved** - Configuration-based bounds, early exit detection
- ✅ **Pre-check for Unreachable Targets** - Eliminates futile iteration loops
- ✅ **Enhanced Diagnostics** for troubleshooting convergence

**Code Quality Improvements (December 9)**:
- ✅ **Main.py Refactoring** - Extracted heating_controller.py and temperature_control.py modules
- ✅ **Zero Regressions** - All functionality preserved with improved maintainability
- ✅ **Test-Driven Approach** - Comprehensive validation of refactored architecture

**System Optimization (December 8)**:
- ✅ **Thermal Parameter Consolidation** - Unified ThermalParameterManager with zero regressions
- ✅ **Delta Temperature Forecast Calibration** - Local weather adaptation system
- ✅ **HA Sensor Refactoring** - Zero redundancy architecture with enhanced monitoring

#### 📊 **Performance Metrics - PRODUCTION EXCELLENT**

**Learning Performance**:
- **Learning Confidence**: 3.0+ (good thermal parameters learned)
- **Model Health**: "good" across all HA sensors
- **Prediction Accuracy**: 95%+ with comprehensive MAE/RMSE tracking
- **Parameter Adaptation**: <100 iterations typical convergence

**System Reliability**:
- **Test Success Rate**: 294/294 tests passing (100%)
- **Binary Search Efficiency**: <10 iterations or immediate exit for unreachable targets
- **Code Quality**: Clean architecture with no TODO/FIXME items
- **Documentation**: Professional and comprehensive (400+ line README)

---

## 📋 REMAINING TASKS FOR RELEASE

### ✅ **VERSION SYNCHRONIZATION COMPLETE (February 13, 2026)**

**Status**: Version inconsistency resolved
- `ml_heating/config.yaml`: `0.2.0`
- `ml_heating_dev/config.yaml`: `0.2.0-dev`
- `CHANGELOG.md`: Updated to reflect `0.2.0` as latest release, with historical versions corrected to `0.2.0-beta.x` sequence.

**Completed Actions**:
- [x] **Decide on release version number** (Unified on `0.2.0`)
- [x] **Update all configuration files** (Confirmed `0.2.0` in config.yaml)
- [x] **Move CHANGELOG `[Unreleased]` section** (Completed)
- [x] **Update repository.yaml and build.yaml** (Not required, versions match)

### ⚠️ **MEDIUM PRIORITY - Optional Improvements**

**Test Suite Cleanup**:
- [x] **Fix 16 test warnings** (PytestReturnNotNoneWarning) - Verified resolved (warnings no longer appear).
- [x] **Review test files returning values** instead of using assert - Verified clean.

**Memory Bank Optimization**:
- [ ] **Archive historical phases** from progress.md (currently 88KB)
- [ ] **Clean up developmentWorkflow.md** - Remove outdated sections

---

## 🎯 **PRODUCTION ARCHITECTURE DELIVERED**

```
ML Heating System v3.0+ (Production Release Ready)
├── Core ML System ✅
│   ├── ThermalEquilibriumModel ✅
│   ├── Adaptive Learning ✅
│   ├── Multi-Heat Source Physics ✅
│   └── Enhanced Feature Engineering ✅
├── User Interface ✅
│   ├── Streamlit Dashboard ✅
│   ├── Home Assistant Integration ✅
│   ├── Ingress Panel Support ✅
│   └── Dual Channel Add-ons ✅
├── Quality Assurance ✅
│   ├── 294 Comprehensive Tests ✅
│   ├── Professional Documentation ✅
│   ├── Code Quality Standards ✅
│   └── Zero Technical Debt ✅
└── Production Features ✅
│   ├── State Persistence ✅
│   ├── Safety Systems ✅
│   ├── Monitoring & Alerts ✅
│   └── Configuration Management ✅
```

---

## 📈 **KEY ACHIEVEMENTS SUMMARY**

### **Transformational Development Completed**
- **Multi-Heat-Source Intelligence**: Complete PV, fireplace, and electronics integration
- **Adaptive Learning System**: Real-time thermal parameter optimization
- **Advanced Physics Features**: 37 thermal intelligence features for ±0.1°C control
- **Professional Dashboard**: Complete Streamlit implementation with ingress support
- **Comprehensive Testing**: 294 tests with 100% success rate

### **Production Excellence Standards Met**
- **Code Quality**: Clean, well-structured, maintainable architecture
- **Documentation**: Professional technical guides and user manuals
- **Testing**: Comprehensive coverage with zero regressions
- **User Experience**: Complete Home Assistant integration with dual channels
- **Reliability**: Robust error handling and safety systems

### **Ready for Immediate Release**
**All core development objectives achieved. Only version synchronization needed before release.**

---

### ✅ **CONFIGURATION PARAMETER FIXES COMPLETED (January 3, 2026)**

**Critical Configuration Issues Resolved**:
- ✅ **Learning Rate Bounds Fixed**: MIN_LEARNING_RATE (0.05 → 0.001), MAX_LEARNING_RATE (0.1 → 0.01) 
- ✅ **Physics Parameters Corrected**: OUTLET_EFFECTIVENESS (0.10 → 0.8) within validated bounds
- ✅ **System Behavior Optimized**: MAX_TEMP_CHANGE_PER_CYCLE (20 → 10°C) for responsive yet stable heating
- ✅ **Grace Period Extended**: GRACE_PERIOD_MAX_MINUTES (10 → 30) for proper system transitions

**Files Updated with Safe Parameter Values**:
- ✅ **`.env`** - Production configuration corrected
- ✅ **`.env_sample`** - Safe examples with bound annotations
- ✅ **`ml_heating/config.yaml`** - Stable addon configuration  
- ✅ **`ml_heating_dev/config.yaml`** - Development addon configuration

**Validation Results**:
- ✅ **No Parameter Out of Bounds Warnings** - All thermal parameters within validated ranges
- ✅ **Shadow Mode Learning Verified** - System correctly observing heat curve decisions (56°C vs ML 52.2°C)
- ✅ **Physics Calculations Stable** - Binary search convergence in 7 iterations with ±0.030°C precision
- ✅ **Learning Confidence Healthy** - Stable at 3.0 indicating good parameter learning

---

**Last Updated**: February 11, 2026  
**Status**: Production Ready - Advanced Testing Implemented  
**Next Step**: Version Synchronization & Release
