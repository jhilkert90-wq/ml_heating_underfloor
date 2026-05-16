# Changelog - ML Heating Underfloor

## [0.2.45] - 2026-05-16

### Added
- **`pv_traj_disable_overshoot_correction` config switch**: New boolean option (default `false`) in `config.yaml` that, when enabled together with `pv_traj_forecast_mode_enabled: true`, skips the overshoot/undershoot outlet-temperature correction inside `_verify_trajectory_and_correct`. The forecast-driven trajectory scaling already adapts the planning horizon to remaining solar hours; the new switch prevents conflicting adjustments when both mechanisms are active simultaneously.

### Fixed
- CI docker smoke test now imports `CYCLE_INTERVAL_MINUTES` from `src.config` instead of removed `POLL_INTERVAL`, and validates `EnhancedModelWrapper` (current class name) in the "Core module smoke test" step in `.github/workflows/build.yaml`.
- Integration smoke test expectation in `tests/integration/test_image_smoke.py` now matches the same valid config import and wrapper class name to prevent false CI failures.

## [0.2.43] - 2026-05-16

### Fixed
- **Calibration `UserWarning` spam**: `heating_correction_ml_calibration.py` now passes DataFrames (with named columns) to `LGBMRegressor.fit()` and `permutation_importance()` instead of stripping names via `.values`. This eliminates hundreds of `UserWarning: X does not have valid feature names` messages that appeared during every permutation-importance run, and ensures training format is consistent with inference (`HeatingCorrectionMLModel.predict()` already used a named DataFrame).

## [0.2.41] - 2026-05-15

### Fixed
- **Critical — Newton correction always clamped to ±2.5°C**: `_calculate_physics_newton_correction()` evaluated S(t) at t_worst which was typically step 0 (t≈0.17h) for underfloor heating. At this early time S(t)≈0.03, causing ε/S(t) to always exceed the ±2.5°C clamp. Added τ/2 floor: both ε and S(t) are now evaluated at max(t_worst, τ_room/2), preventing degenerate corrections from the thermal-inertia transient. If the trajectory has recovered by τ/2 (sign flip), the correction is suppressed entirely.

### Added
- **ML Heating correction parameter tooltips**: Added 13 missing Home Assistant UI descriptions for `heating_ml_*` parameters and `pv_traj_forecast_rescue_enabled` in `en.yaml`.
- **Dashboard "Calibrate ML Heating Model" button**: Added calibration trigger button to the Streamlit control page, mirroring the existing ML Cooling button pattern.
- **Newton correction τ/2 floor tests**: Two new unit tests — `test_tau_half_floor_suppresses_degenerate_correction` (test 12) and `test_tau_half_floor_sign_flip_suppresses_correction` (test 13).

## [0.2.40] - 2026-05-15

### Fixed
- **Critical — Newton correction always clamped to ±2.5°C**: `_calculate_physics_newton_correction()` evaluated S(t) at t_worst which was typically step 0 (t≈0.17h) for underfloor heating. At this early time S(t)≈0.03, causing ε/S(t) to always exceed the ±2.5°C clamp. Added τ/2 floor: both ε and S(t) are now evaluated at max(t_worst, τ_room/2), preventing degenerate corrections from the thermal-inertia transient. If the trajectory has recovered by τ/2 (sign flip), the correction is suppressed entirely.

### Added
- **ML Heating correction parameter tooltips**: Added 13 missing Home Assistant UI descriptions for `heating_ml_*` parameters and `pv_traj_forecast_rescue_enabled` in `en.yaml`.
- **Dashboard "Calibrate ML Heating Model" button**: Added calibration trigger button to the Streamlit control page, mirroring the existing ML Cooling button pattern.
- **Newton correction τ/2 floor tests**: Two new unit tests — `test_tau_half_floor_suppresses_degenerate_correction` (test 12) and `test_tau_half_floor_sign_flip_suppresses_correction` (test 13).

## [0.2.38] - 2026-05-15

### Added
- **Heating Correction ML: Online Learning** (`HeatingCorrectionObservationBuffer`): Mirrors the pre-cooling sliding-window observation buffer pattern for the LightGBM heating regressor.
  - `src/heating_correction_ml_observation_buffer.py` — new `HeatingCorrectionObservationBuffer` class; stores heating-cycle feature snapshots, resolves regression labels `−(T_indoor[t+N] − T_target) / S_H` after `label_horizon_steps` cycles, auto-triggers retrain via `calibrate_heating_correction_ml()` when `n_labeled ≥ min_training_samples AND labeled_since_last_train ≥ retrain_trigger_k`; JSON persistence with atomic tmp→replace writes
  - Per-cycle integration in `src/main.py`: `push_pending`, `resolve_labels`, and auto-retrain all run on heating cycles only; successful retrains hot-reload by resetting `EnhancedModelWrapper._heating_correction_ml_model = None`
  - Buffer collects observations regardless of `HEATING_CORRECTION_MODE` so data accumulates even before ML mode is activated
  - S_H recomputed at resolve-time from current calibrated thermal parameters (`_compute_s_h` / `_read_baseline_thermal_params`)
- **New config vars**: `HEATING_ML_OBSERVATION_BUFFER_PATH` (default: `_UNIFIED_STATE_DIR/heating_correction_ml_obs_buffer.json`), `HEATING_ML_RETRAIN_TRIGGER_K` (default `50`), `HEATING_ML_BUFFER_MAX_N` (default `500`)
- 25 new unit tests in `tests/unit/test_heating_correction_observation_buffer.py`


  - `src/heating_correction_ml_model.py` — inference class `HeatingCorrectionMLModel`; loads joblib model + metadata, exposes `predict(features, target_indoor)` and `r2_score`
  - `src/heating_correction_ml_calibration.py` — one-shot training: cold-season filter (AT < 18 °C), feature vector with AT hindcast, PV hindcast, dynamic fireplace/TV lags, regression label `−(T_future − T_target) / S_H`
  - Blended dispatch in `model_wrapper._calculate_ml_correction()`: confidence-weighted blend `w = R²` (clamped, with `HEATING_ML_BLEND_MIN_R2` minimum threshold)
  - `--calibrate-heating-correction-ml` CLI flag and `/data/config/calibrate_heating_correction_ml_flag` flag file
- **Heating ML feature expansion**:
  - PV hindcast features (`pv_forecast_1h`–`pv_forecast_Nh`) controlled by `HEATING_ML_PV_FORECAST_HOURS` (default `"1,2,3,4"`)
  - PV instantaneous and rolling features (`PV_Generate`, `pv_roll_1h`, `pv_roll_2h`) at both training and inference time; prefers `pv_now_electrical`/`pv_forecast_electrical_Xh` keys to match training scale
  - Dynamic fireplace lag windows (`fireplace_lag_1h`, `fireplace_lag_2h`, …) controlled by `HEATING_ML_FIREPLACE_LAG_HOURS` (default `"1,2"`)
  - Dynamic TV lag windows (`tv_lag_30m`, `tv_lag_1h`, …) controlled by `HEATING_ML_TV_LAG_HOURS` (default `"0.5,1"`)
  - Regex-based feature extraction in inference model handles any `fireplace_lag_Xh/m`, `tv_lag_Xh/m`, `pv_forecast_Xh`, `AT_roh_Xh` pattern without hardcoding individual names
- **New config vars**: `HEATING_ML_COLD_THRESHOLD_C`, `HEATING_ML_CALIBRATION_START_DATE`, `HEATING_ML_AT_FORECAST_HOURS`, `HEATING_ML_PV_FORECAST_HOURS`, `HEATING_ML_FIREPLACE_LAG_HOURS`, `HEATING_ML_TV_LAG_HOURS`, `HEATING_ML_CORRECTION_MODEL_PATH`, `HEATING_ML_CORRECTION_METADATA_PATH`, `HEATING_ML_MIN_TRAINING_SAMPLES`, `HEATING_ML_LABEL_HORIZON_H`, `HEATING_ML_BLEND_MIN_R2`
- `_parse_heating_start_date()` helper in `config.py` (mirrors `_parse_cooling_start_date`)
- 60 new unit tests across 3 test files (calibration, inference, blend dispatch)
- Documentation: section 7 of `docs/HEATING_CORRECTION_PHYSICS_VS_ML_ANALYSIS.md` updated with final feature list, label construction, and blend formula

### Fixed
- **Heating Correction ML: `indoor_temp` key mismatch at inference time** (`src/heating_correction_ml_model.py`): `_extract_heating_feature("indoor_temp")` and `_extract_heating_feature("indoor_margin")` returned 0.0 at runtime because `build_physics_features()` stores the indoor temperature under `indoor_temp_lag_30m` (not `indoor_temp`). Added fallback so inference correctly reads `indoor_temp_lag_30m` when `indoor_temp` is absent.  3 new regression tests added in `tests/unit/test_heating_correction_ml_model.py`.
- **Heating Correction ML calibration: duplicated warning format arg** (`src/heating_correction_ml_calibration.py`): The `S_H from persisted params` warning message logged the fallback `s_h` value for both format args, hiding the original degenerate value. Fixed by saving the original before overwriting.
- **Config adapter / config.yaml missing `HEATING_ML_RETRAIN_VAL_FRACTION`** (`config_adapter.py`, `ml_heating_underfloor/config.yaml`): The validation-split fraction for the heating ML regressor was defined in `config.py` but not wired to the HA add-on config schema. Added `heating_ml_retrain_val_fraction` option (default 0.25, range 0.05–0.5).

## [0.2.37] - 2026-05-15

### Fixed
- **Binary search range-collapse bypass** (`model_wrapper.py`): when the binary search range collapsed to < 0.05 °C (e.g. saturating at 21 °C min outlet due to high PV forecast), the early-exit path returned immediately without running `_verify_trajectory_and_correct`. The correction layer is now called before returning in this path, matching the converged and non-converged paths.
- **`projected_indoor` linear over-estimation** (`model_wrapper.py`, both `_calculate_physics_based_correction` and `_calculate_physics_newton_correction`): the self-correction gate computed `projected_indoor = current + TRAJECTORY_STEPS × trend` (linear). At H=4 h this overestimated the room's natural travel by 2–3× compared to the actual trajectory, causing legitimate overshoot/undershoot corrections to be skipped. Replaced with the exponential-decay integral `current + trend × τ × (1 − exp(−H/τ))` using `TREND_DECAY_TAU_HOURS` (default 1.5 h), matching exactly how the trajectory model accumulates the trend bias.
- **Newton `else` branch threshold inconsistency** (`model_wrapper.py`, `_calculate_physics_newton_correction`): the internal fallback branch used `reaches_target_at > cycle_hours` (1× cycle) while the outer gate filters at `cycle_hours + tolerance_hours` (3× cycle). Aligned the Newton branch to `cycle_hours + tolerance_hours` for consistency.
- **Fragile exact float equality** (`model_wrapper.py`, `_calculate_physics_newton_correction`): replaced `if temp_error == 0.0` with `if abs(temp_error) < 1e-6` to avoid potential floating-point precision issues.
- **Newton uses `S(H)` instead of `S(t_worst)`** (`model_wrapper.py`, `_calculate_physics_newton_correction`): sensitivity was always evaluated at the full horizon H via `S_H = [η/(η+U)] × [1 − exp(−H/τ_room)]`, but ε was measured at the worst trajectory point which may occur at `t_worst < H`. Since `S(H) > S(t_worst)`, dividing by `S_H` under-corrects systematically — worst when PV drives a mid-horizon overshoot (solar gain peaks early in the day). Fixed by looking up the time-step index of the worst point and evaluating `S(t_worst)` there. The trajectory `times` array is used when present; otherwise the step size is inferred as `H / n_steps`.

### Added
- **Physics Newton-Step Heating Correction** (`_calculate_physics_newton_correction()` in `model_wrapper.py`): implements `ΔT_outlet = ε / S(t_worst)` where `S(t) = [η/(η+U)] × [1 − exp(−t/τ_room)]` is evaluated at the time of the worst trajectory violation. Symmetric for under- and overshoot, corrects PV-driven mid-horizon errors, and ~2× more accurate than the legacy formula after calibration. Shares all boundary-violation guards and clamp logic with the existing method.
- **ML Correction Stub** (`_calculate_ml_correction()` in `model_wrapper.py`): placeholder that warns and falls back to the Newton step, ready to be replaced with a LightGBM regressor once sufficient historical data is available.
- **`HEATING_CORRECTION_MODE` config variable** (`src/config.py`): selects the active correction algorithm at runtime. Accepted values: `"legacy"` (default), `"physics"`, `"ml"`.
- **Home Assistant dropdown selector** (`ml_heating_underfloor/config.yaml`): `heating_correction_mode: "list(legacy|physics|ml)"` renders as a dropdown in the HA add-on UI — no text entry required.
- **config_adapter wiring** (`config_adapter.py`): `heating_correction_mode` option is now mapped to `HEATING_CORRECTION_MODE` env var in `convert_addon_to_env()`.
- **Translation entry** (`ml_heating_underfloor/translations/en.yaml`): descriptive label and tooltip for the new dropdown.
- **11 unit tests** (`tests/unit/test_heating_correction.py`): cover Newton undershoot/overshoot accuracy (±0.01°C tolerance), degenerate-S_H fallback, clamp guard, mode dispatch for all three modes, and config_adapter mapping.

## [0.2.36] - 2026-05-14

### Fixed
- **`calculate_optimal_outlet_temperature` cooling mode support** — Method was heating-only (bounds `[outdoor+5, 70]`), always returning outlet ≥25°C in cooling scenarios. Added `climate_mode` parameter; cooling mode uses `[COOLING_CLAMP_MIN_ABS, COOLING_CLAMP_MAX_ABS]` bounds and skips the "outlet below outdoor" fallback. Also fixed `_calculate_equilibrium_outlet_temperature` with same cooling bounds. Production unaffected (uses binary search via `model_wrapper`), but analytical method now works for notebooks and offline simulation.

### Changed
- **Cooling ML calibration: dedicated fetch path** — `fetch_historical_data_for_calibration()` accepts `purpose="cooling"` to fetch only 7 entities instead of 15, reducing InfluxDB and HA API load by ~50%
- **Warm-season filter decoupled** — New `COOLING_ML_WARM_THRESHOLD_C` config (default 10°C) replaces the derived `PRE_COOL_MIN_OUTDOOR_FORECAST_C - 6` formula, including shoulder-season data for better label balance
- **Forecast defaults aligned to lead time** — `COOLING_ML_AT_FORECAST_HOURS` and `COOLING_ML_PV_FORECAST_HOURS` now default to `1..PRE_COOL_LEAD_TIME_HOURS` (8h) instead of hardcoded 12h

### Fixed
- **`PRE_COOL_LEAD_TIME_HOURS` default mismatch** — `config.py` default changed from 3.0 to 8.0 to match `config_adapter.py`
- **`_field` gap-detection artifact** — InfluxDB pivot metadata column `_field` no longer triggers misleading coverage-gap warnings
- **24 pre-existing test failures resolved** — Fixed config module identity pollution (`test_config.py` deleting `sys.modules['src.config']` without restore), dashboard data-service isolation (missing `_COOLING_STATE_FILE_CANDIDATES` patch), streamlit import skip for dashboard component tests, overheating predictor peak-hour assertion aligned to 8h lead time, PV weight adaptive-learning test robustness, and TDD-fixture-sensitive default assertion

## [0.2.35] - 2026-05-14

### Added
- **Cooling ML: Configurable calibration start date** — `calibrate_cooling_ml()` now reads `COOLING_ML_CALIBRATION_START_DATE` (format `DD.MM.YYYY`) from config/env and converts it to `lookback_hours` automatically, allowing the training window to be pinned to a specific seasonal start date instead of a fixed relative offset. Falls back to the default 2160 h (90 days) when the field is empty or invalid; logs a warning on bad input. The computed lookback uses ceiling arithmetic so the full start date is always included in the training window.
- New config var `COOLING_ML_CALIBRATION_START_DATE` (default `""`) and helper `_parse_cooling_start_date() → Optional[datetime]` in `src/config.py`.
- New HA add-on option `cooling_ml_calibration_start_date` in `ml_heating_underfloor/config.yaml` (schema type `str?`) with UI tooltip in `ml_heating_underfloor/translations/en.yaml`. The option is now fully wired into `config_adapter.py::convert_addon_to_env()` so it takes effect at runtime.
- 6 new unit tests in `TestCoolingStartDate` covering valid date, empty fallback, invalid string warning, and `_parse_cooling_start_date` edge cases.
- **Cooling ML: Full forecast horizon features (⚠ model signature change)** — `cooling_ml_calibration.py` now includes all 12 outdoor-temperature hindcast columns (`AT_roh_1h`–`AT_roh_12h`) and all 12 PV-power hindcast columns (`pv_forecast_1h`–`pv_forecast_12h`) in the training feature set by default. This changes the default model signature; any previously saved model trained with fewer features must be retrained after upgrading. The exact feature list is saved to the model metadata JSON so inference stays consistent.
- New config vars `COOLING_ML_AT_FORECAST_HOURS` and `COOLING_ML_PV_FORECAST_HOURS` (comma-separated hour lists, default `"1,2,3,4,5,6,7,8,9,10,11,12"`) to control which forecast horizons are included; the legacy `COOLING_ML_FORECAST_HOURS` env var remains as a backward-compatible alias for `COOLING_ML_AT_FORECAST_HOURS`. Invalid values now log a warning and fall back to the full 12-hour list.

## [0.2.34] - 2026-05-14

### Fixed
- Build workflow now triggers automatically on every push to `main` (in addition to manual `workflow_dispatch`), preventing Docker images from being missing after PR merges. A `paths-ignore` filter skips rebuilds for documentation-only commits.

## [0.2.33] - 2026-05-14

### Fixed
- Switch Docker base image from `python:3.11-alpine3.18` to `python:3.11-slim` to fix aarch64 build: Alpine uses musl libc which has no pre-built scikit-learn wheels for aarch64, causing source compilation to fail with a GCC error on the native ARM runner.

## [0.2.32] - 2026-05-14

### Changed
- **Cooling ML calibration lead-time semantics**: Reduced default `PRE_COOL_LEAD_TIME_HOURS` from 8.0 to 3.0 for more responsive pre-cooling label assignment, aligning model training and inference semantics.
- **Cooling ML PV feature contract hardening**: All cooling ML PV features (`pv_roll_*`, `PV_Generate`, `pv_forecast_*h`) now strictly prefer raw electrical keys (`pv_now_electrical`, `pv_power_history_electrical`, `pv_forecast_electrical_*h`) and only fall back to thermal-corrected keys if electrical keys are absent, ensuring feature scale matches training data.

### Fixed
- **Pre-cooling calibration bugs**: Fixed bugs in pre-cooling calibration including:
  - Use of correct PV keys for feature extraction and rolling PV history.
  - Alignment of feature scale between training and inference.
  - Buffer persistence: cooling observation buffer now saves after every push/resolve cycle, preserving pending entries and evolving label state across restarts.
  - Lead-time and label horizon calculation now matches intended semantics.
- **Missing `scikit-learn` dependency**: Added `scikit-learn>=1.0.0` to `requirements.txt` to fix silent failures in calibration metrics.

### Added
- **Test coverage for cooling calibration workflow**: Added and clarified unit tests for cooling ML calibration and PV feature extraction, including tighter assertions for raw vs thermal PV history and column-count expectations.

### Changed
- **CI: GitHub Actions workflow versions and architecture**: Updated `.github/workflows/build.yaml` to use newer versions of GitHub Actions (`actions/checkout@v4`, `docker/login-action@v3`, `docker/setup-buildx-action@v3`, `docker/build-push-action@v6`) and improved architecture handling. Native runners are now used per architecture (no QEMU required), and build cache is separated per arch for improved reliability and performance.

### Added
- **PV Feature Key Contract documentation**: Added a canonical, highly-visible `⚠️ AI MODEL NOTICE — PV Feature Key Contract` section at the top of `memory-bank/systemPatterns.md` that maps every consumer of `pv_now_electrical` / `pv_forecast_electrical_*` (electrical, raw) vs `pv_now` / `pv_forecast_{h}h` (thermal, corrected). Includes a table, explicit rules, anti-patterns, and source-code citations.
- **ML Cooling Guide PV warning**: Added a `⚠️ PV Feature Key Contract` warning block to `docs/ML_COOLING_MODEL_GUIDE.md` clarifying that `OverheatingPredictor` and ML cooling paths must use thermal keys.
- **PV key contract regression tests** (`tests/unit/test_overheating_predictor.py`): 5 new tests in `TestPVKeyContract` that:
  - Confirm `OverheatingPredictor` reads `pv_now` (thermal) for its guard check and works without `pv_now_electrical`
  - Confirm trajectory PV forecast is built from `pv_forecast_{h}h` (thermal) even when electrical keys are absent
  - Confirm `pv_now_electrical` alone (with `pv_now=0`) cannot pass the guard
  - Confirm `HLCCycle._build_cycle()` reads `pv_now_electrical` (not `pv_now`) from context
  - Confirm `_build_cycle()` defaults `pv_now_electrical` to 0.0 when the key is absent
- **Reviewer-follow-up regression hardening**: Strengthened 2 `TestPVKeyContract` tests to assert `predict_thermal_trajectory()` call kwargs (`pv_power`, `pv_forecasts`) directly, preventing false positives from mocked trajectory outputs.

### Fixed
- **Merge conflict resolution (latest sync)**: Resolved newly introduced conflicts while merging latest `origin/main` into the PR branch, reconciling `memory-bank/activeContext.md` and `memory-bank/progress.md` and preserving entries from both branches.
- **Post-merge cleanup**: Repaired a truncated sentence in `memory-bank/activeContext.md` and normalized non-English inline comments in `.github/workflows/build.yaml` to English for consistency.
- **Post-merge review cleanup**: Restored the missing `Files changed` block in the PV contract context entry, removed an extra trailing separator in `memory-bank/activeContext.md`, and reformatted `new_addon_lines` in `.github/workflows/build.yaml` for readability.
- **[HIGH] Cooling observation buffer durability gap before label maturity**: Pre-cooling loop now persists the observation buffer after every `push_pending()`/`resolve_labels()` cycle, so pending samples and evolving label state survive restarts even before labels mature.
- **[HIGH] Remaining PV roll scale drift in cooling ML inference**: Added raw `pv_power_history_electrical` to physics features and updated cooling ML `pv_roll_*` extraction to prefer the raw electrical history used during training.
- **Merge conflict resolution for PR branch**: Resolved conflicts against `origin/main` in `CHANGELOG.md`, `memory-bank/activeContext.md`, and `memory-bank/progress.md` by preserving entries from both branches and removing conflict markers.
- **[CRITICAL] Missing `scikit-learn` dependency**: Added `scikit-learn>=1.0.0` to `requirements.txt`; its absence caused `roc_auc_score` import to silently fail, writing `null` AUC to model metadata.
- **[HIGH] Wrong PV/forecast feature keys in pre-cooling integration tests**: `_make_features()` in `test_pre_cooling_integration.py` used `pv_now_electrical`, `pv_forecast_electrical_{h}h`, and `outdoor_forecast_{h}h` — none of which `OverheatingPredictor` reads. Fixed to `pv_now`, `pv_forecast_{h}h`, and `temp_forecast_{h}h` so the PV guard is actually exercised.
- **[HIGH] Missing PV contract assertion in pre-cooling integration tests**: Added an explicit assertion that `OverheatingPredictor` passes the corrected PV values into `predict_thermal_trajectory()` as `pv_power` and `pv_forecasts`, so PV-guard regressions are caught.
- **[MEDIUM] Training/inference PV scale mismatch for LGBM cooling model**: `_extract_feature()` in `cooling_ml_model.py` used thermally-corrected `pv_now`/`pv_forecast_{h}h` at inference while training used raw electrical watts. Now prefers `pv_now_electrical`/`pv_forecast_electrical_{h}h` with graceful fallback to corrected values.
- **[LOW] Incorrect `PRE_COOL_LEAD_TIME_HOURS` fallback in calibration**: Hardcoded default was `8.0` in `cooling_ml_calibration.py`; corrected to `3.0` to match `config.py`.
- **[LOW] Cooling observation buffer not persisted between cycles**: Buffer was only saved on successful retrain. Now saved whenever `resolve_labels()` returns newly-labeled observations, preventing data loss on restart.
- **[LOW] Stale cooling ML test defaults**: Updated cooling ML test fixtures and fake configs to use the runtime default `PRE_COOL_LEAD_TIME_HOURS=3.0` instead of the old `8.0`, keeping calibration tests aligned with production behavior.

## [0.2.31] - 2026-05-13

### Fixed
- **HP false-active detection from residual slab heat**: `_is_heat_pump_active()` outlet/inlet temperature fallback now suppressed when both `thermal_power` and `delta_t` are near zero (< 0.1). Prevents HP from co-learning with PV via mixed-source attribution when HP is off but floor slab retains residual warmth, which contaminated `outlet_effectiveness` and `heat_loss_coefficient` parameters.

### Added
- **Predictive Pre-Cooling with ML Model**: LightGBM-based overheating classifier (`CoolingMLModel`) as drop-in alternative to trajectory-based `OverheatingPredictor`, selectable via `PRE_COOL_MODEL_TYPE` config
- **Online Learning Observation Buffer**: `CoolingObservationBuffer` with sliding-window labeled-observation store, automatic label resolution after horizon steps, and auto-retrain trigger
- **ML Cooling Calibration Pipeline**: `calibrate_cooling_ml` one-shot training from InfluxDB historical data with hindcast substitution for forecast features, LightGBM with class weighting, and F1-optimized threshold tuning
- **Overheating Predictor**: Physics-based trajectory simulation for passive (HP OFF) overheating risk forecasting with PV and outdoor temperature forecasts
- **Shadow Mode for Pre-Cooling**: Active/shadow dual-strategy logging — trajectory and LGBM models run simultaneously, inactive strategy logs as shadow
- **Epsilon Sensitivity Analysis Script**: Automated sensitivity analysis for cooling mode parameters
- **Offline Calibration Comparison Scripts**: Tools for comparing physics-direct calibration approaches
- **Pre-cooling configuration parameters**: `PRE_COOL_ENABLED`, `PRE_COOL_MODEL_TYPE`, `PRE_COOL_TRIGGER_MARGIN_K`, `PRE_COOL_HORIZON_HOURS`, `PRE_COOL_LEAD_TIME_HOURS`, `PRE_COOL_TARGET_OFFSET_K`, `PRE_COOL_MIN_PV_FORECAST_W`, `PRE_COOL_MIN_OUTDOOR_FORECAST_C`, and `COOLING_ML_*` parameters

### Changed
- **Physics-Direct Calibration Accuracy**: Enhanced calibration with 7 algorithmic and quality fixes, 3-level parameter fallback (#43)
- **Stable Periods Path**: Write `stable_periods.json` to `UNIFIED_STATE_FILE` directory instead of hardcoded path (#44)

### Fixed
- **NaN/Inf serialization in observation buffer**: `CoolingObservationBuffer.save()` now recursively sanitizes NaN/Inf values in feature dicts to `null` before JSON serialization, preventing data corruption
- **Retrain backoff loop**: Failed cooling ML retrain no longer immediately re-triggers — back-off now subtracts `trigger_k//2 + 1` instead of `trigger_k//2`
- **HP Channel Learning Blocked by PV**: Fixed PV blocking heat pump channel learning; trajectory steps override; recovery gate deadlock (#41)
- **Cooling Mode Bug Fixes**: Review-round fixes including HP idle guard clamping outlet to inlet when gap < `MIN_COOLING_DELTA_K` (#39)
- **CI/CD**: Resolved Node.js 20 deprecation and Copilot API token rejection in workflows (#42); fixed duplicate `env` key and upgraded actions to Node.js 24 (#40)

## [0.2.30] - 2026-05-13

### Added
- **Extended Unit Tests for ML Pre-Cooling**: Comprehensive unit tests for ML pre-cooling modules, including cold start scenarios, observation buffer edge cases (NaN/Inf handling, label resolution), CoolingMLModel inference edge cases, OverheatingPredictor missing forecast/reactive logic, calibration label logic, online learning retrain flow, and configuration default consistency checks.
- **Baseline Model State JSON**: Added baseline `model_metadata.json` for ML cooling model calibration state and parameters.

### Fixed
- **HP false-active detection from residual slab heat**: `_is_heat_pump_active()` outlet/inlet temperature fallback now suppressed when both `thermal_power` and `delta_t` are near zero (< 0.1). Prevents HP from co-learning with PV via mixed-source attribution when HP is off but floor slab retains residual warmth, which contaminated `outlet_effectiveness` and `heat_loss_coefficient` parameters.

### Added
- **Predictive Pre-Cooling with ML Model**: LightGBM-based overheating classifier (`CoolingMLModel`) as drop-in alternative to trajectory-based `OverheatingPredictor`, selectable via `PRE_COOL_MODEL_TYPE` config
- **Online Learning Observation Buffer**: `CoolingObservationBuffer` with sliding-window labeled-observation store, automatic label resolution after horizon steps, and auto-retrain trigger
- **ML Cooling Calibration Pipeline**: `calibrate_cooling_ml` one-shot training from InfluxDB historical data with hindcast substitution for forecast features, LightGBM with class weighting, and F1-optimized threshold tuning
- **Overheating Predictor**: Physics-based trajectory simulation for passive (HP OFF) overheating risk forecasting with PV and outdoor temperature forecasts
- **Shadow Mode for Pre-Cooling**: Active/shadow dual-strategy logging — trajectory and LGBM models run simultaneously, inactive strategy logs as shadow
- **Epsilon Sensitivity Analysis Script**: Automated sensitivity analysis for cooling mode parameters
- **Offline Calibration Comparison Scripts**: Tools for comparing physics-direct calibration approaches
- **Pre-cooling configuration parameters**: `PRE_COOL_ENABLED`, `PRE_COOL_MODEL_TYPE`, `PRE_COOL_TRIGGER_MARGIN_K`, `PRE_COOL_HORIZON_HOURS`, `PRE_COOL_LEAD_TIME_HOURS`, `PRE_COOL_TARGET_OFFSET_K`, `PRE_COOL_MIN_PV_FORECAST_W`, `PRE_COOL_MIN_OUTDOOR_FORECAST_C`, and `COOLING_ML_*` parameters

### Changed
- **Physics-Direct Calibration Accuracy**: Enhanced calibration with 7 algorithmic and quality fixes, 3-level parameter fallback (#43)
- **Stable Periods Path**: Write `stable_periods.json` to `UNIFIED_STATE_FILE` directory instead of hardcoded path (#44)

### Fixed
- **NaN/Inf serialization in observation buffer**: `CoolingObservationBuffer.save()` now recursively sanitizes NaN/Inf values in feature dicts to `null` before JSON serialization, preventing data corruption
- **Retrain backoff loop**: Failed cooling ML retrain no longer immediately re-triggers — back-off now subtracts `trigger_k//2 + 1` instead of `trigger_k//2`
- **HP Channel Learning Blocked by PV**: Fixed PV blocking heat pump channel learning; trajectory steps override; recovery gate deadlock (#41)
- **Cooling Mode Bug Fixes**: Review-round fixes including HP idle guard clamping outlet to inlet when gap < `MIN_COOLING_DELTA_K` (#39)
- **CI/CD**: Resolved Node.js 20 deprecation and Copilot API token rejection in workflows (#42); fixed duplicate `env` key and upgraded actions to Node.js 24 (#40)

## [0.2.29] - 2026-05-06

### Added
- **Scipy optimization failure logging**: Added detailed logging and error handling for scipy optimizer failures during OE and transient parameter calibration, improving robustness and traceability.
- **Enhanced documentation**: Improved inline documentation and comments for calibration routines, clarifying algorithmic steps and parameter meanings.

### Changed
- **Solar lag calibration**: `_calibrate_solar_lag_xcorr()` now correlates PV with the rate of change of residuals (d(residual)/dt), reducing maximum lag from 36 to 12 steps and increasing correlation threshold from 0.1 to 0.3, addressing slab-mass delay bias and improving lag accuracy.
- **OE calibration**: `_calibrate_oe_analytical()` now uses a two-stage approach: analytical weighted-median OE as initial guess, followed by scipy `minimize_scalar` refinement, increasing outlet effectiveness (OE) accuracy from ~0.72 to ~0.95.
- **Thermal time constant calibration**: Calibration now prioritizes transient parameter estimation using `calibrate_transient_parameters()` and `filter_transient_periods()`, falling back to cooling curve analysis only if necessary.
- **Unit labels**: Corrected unit labels for `outlet_effectiveness` and `heat_loss_coefficient` in `ThermalParameterConfig` from "dimensionless" and "1/hour" to "kW/K".

### Fixed
- **OE calibration accuracy**: Added scipy 1-D refinement pass using full `ThermalEquilibriumModel.predict_equilibrium_temperature()` — matches the scipy path's objective function exactly. Removed the `drive >= 3°C` filter that was discarding 68% of HP-only periods and biasing OE downward; now uses all HP-only periods (drive > 0) with drive-weighted median for the analytical initial guess
- **HLC calibration quality gates**: When `target_temp` sensor is unavailable, HLC calibration now synthesises a constant column from `HLC_DEFAULT_TARGET_TEMP` (default 22.6°C) instead of silently disabling the `indoor_far_from_target` and `low_heating_demand` filters. This was causing HLC=0.119 (R²=-0.06) instead of correct ~0.133, which cascaded to OE=0.81 instead of ~0.92
- **Solar lag calibration**: Fixed `_calibrate_solar_lag_xcorr()` producing 180 min (upper bound) instead of correct ~40 min. Three changes: (1) cross-correlate PV with d(residual)/dt instead of residual level to remove slab-mass smoothing, (2) reduce max lag from 180->60 min since slab delay is modeled separately, (3) raise correlation threshold from 0.1->0.3 and use weighted median instead of brittle mode
- **Thermal time constant calibration**: Added transient calibration from heating sequences as primary method (was only trying cooling curves which require HP-off >=2h, rarely available in well-controlled UFH). Falls back to cooling curves, then persisted value
- **OE/HLC unit labels**: Corrected `outlet_effectiveness` and `heat_loss_coefficient` units in `ThermalParameterConfig` from "dimensionless"/"1/hour" to "kW/K" -- both are thermal conductances added in the equilibrium equation

## [0.2.28] - 2026-05-06

### Fixed
- `stable_periods.json` is now saved to the same directory as `unified_thermal_state.json` (derived from `config.UNIFIED_STATE_FILE`) instead of the hardcoded `/opt/ml_heating/` path, fixing a `FileNotFoundError` when running in Home Assistant add-on environments.

## [0.2.27] - 2026-05-05

### Added
- **Physics-Direct calibration path** — Added a fully analytical, sequential calibration method (`Physics Direct`) that estimates all thermal model parameters from first principles, without relying on scipy optimization. This path is selectable from the dashboard and exposes all parameters for user editing in `config.yaml`.
- **Calibration method selector in dashboard** — The dashboard now allows users to choose between "Scipy Optimizer" (default) and "Physics Direct" calibration methods when triggering model recalibration.

### Changed
- **Calibration method config option** — Added `CALIBRATION_METHOD` to `src/config.py` and `config.yaml`, allowing explicit selection of calibration path via config/environment.
- **Config schema validation** — Updated `config.yaml` schema to include bounds for `cloud_factor_exponent` and `solar_decay_tau_hours`.
- **Magic numbers refactored** — Extracted magic numbers as constants in calibration code; improved comments for cloud exponent logic.

### Fixed
- **Config default mismatches** — Fixed 6 mismatches between `src/config.py` defaults and `ThermalParameterConfig.DEFAULTS` (PV, fireplace, TV weights, thermal time constant, slab tau, total conductance). Now config defaults are aligned and validated.
- **State-file bounds validation** — Persisted calibration parameters are now validated against bounds before being accepted as fallback, preventing corrupted values from overriding config defaults.
- **Test coverage** — Updated and expanded unit tests for physics-direct calibration, including TV weight and solar lag xcorr edge cases; all tests pass.

### Fixed
- **Solar lag xcorr: non-contiguous data bug** — `_calibrate_solar_lag_xcorr()` in `src/physics_calibration_direct.py` previously operated on `stable_periods` (non-adjacent 20-min windows scattered across weeks), making lag shifts meaningless. Rewrote to operate on the raw time-sorted DataFrame, computing xcorr within each **contiguous PV-active episode** and returning the modal best-lag across episodes. Added 3 new edge-case tests.
- **Slab tau step-ordering dependency** — `_calibrate_slab_tau_grid_search()` used the config default for `delta_t_floor` even though the calibrated value was available from the immediately preceding step 8. Added `delta_t_floor` parameter; `calibrate_thermal_model_physics()` now passes the step-8 calibrated value so the slab-tau estimate is not biased by an incorrect default.
- **OE docstring incorrect** — Corrected the docstring in `_calibrate_oe_analytical()`: weight description now consistently uses the term `drive = T_outlet − T_indoor` to avoid confusion with the reciprocal.
- **IQR outlier rejection in residual weight estimator** — `_residual_heat_source_weight()` now applies a 1.5-IQR fence to the collected sample distribution before taking the percentile, preventing extreme residuals from transient slab effects or sensor spikes from biasing the estimate.
- **bfill NaN gap warning** — `calibrate_thermal_model_physics()` now logs a `⚠️` warning for each key column (flow_rate, inlet_temp, target_outlet_temp) that has more than 6 consecutive NaN rows (>30 min gap) after imputation, making silent bfill contamination visible.
- **Tau calibration duration gate** — `calculate_cooling_time_constant()` in `src/physics_calibration.py` now requires HP-off blocks to span at least 2 hours before including their tau estimate in the weighted average, preventing short blocks (typical in underfloor systems) from systematically underestimating the room thermal time constant.

### Added
- **State-file fallback for all calibration parameters** — `calibrate_thermal_model_physics()` now resolves the active `ThermalStateManager` early and uses a `_state_fallback()` helper. When a calibration step cannot produce a value (e.g. no fireplace-active rows in the dataset), it uses the previously persisted value from `unified_thermal_state.json` instead of the hardcoded default, preserving any prior calibration result. If the state file also has no valid value, the `config.yaml`-editable default is used as the final fallback.
- **`cloud_factor_exponent` and `solar_decay_tau_hours` persisted to state file** — `set_calibrated_baseline()` in `unified_thermal_state.py` and `_get_default_state()` now include these two parameters so they survive restarts and are available as warm-start fallbacks on the next calibration run.

### Changed
- **PV `min_periods` raised from 5 to 15** — `calibrate_thermal_model_physics()` now requires at least 15 qualifying stable periods for PV heat-weight estimation, reducing sensitivity to occupancy-correlated noise when PV-active days are few.
- **All calibration parameters now editable in `config.yaml`** — `ThermalParameterConfig.get_default()` now reads from `config` module variables first (which are sourced from `config.yaml` / environment variables) before falling back to the hardcoded `DEFAULTS` dict. Added 7 new config vars to `src/config.py` that were previously missing: `HEAT_LOSS_COEFFICIENT`, `OUTLET_EFFECTIVENESS`, `DELTA_T_FLOOR`, `FP_DECAY_TIME_CONSTANT`, `ROOM_SPREAD_DELAY_MINUTES`, `CLOUD_FACTOR_EXPONENT`, `SOLAR_DECAY_TAU_HOURS`. Both `.env_sample` and `config.yaml` are updated with the two previously undocumented entries (`cloud_factor_exponent`, `solar_decay_tau_hours`).


### Added
- **Physics-Direct Calibration Path** (`src/physics_calibration_direct.py`): new `calibrate_thermal_model_physics()` function that estimates every thermal parameter analytically — no scipy optimizer, no MAE fitting.  Sequential decoupling derives each parameter after locking previous ones:
  1. HLC via `calibrate_hlc()` (OLS flow-meter regression)
  2. OE via per-window algebra from HP-only stable periods
  3. τ_room from log-linear OLS on HP-off cooling curves
  4. `pv_heat_weight` via residual energy balance on PV-on periods
  5. `fireplace_heat_weight` via residual energy balance on FP-on periods
  6. `tv_heat_weight` via residual energy balance on TV-on periods
  7. `solar_lag_minutes` via PV ↔ indoor-residual cross-correlation
  8. `delta_t_floor` via P25 percentile of (outlet − inlet)
  9. `slab_time_constant_hours` via 1-D grid search over [0.1, 4.0 h] (replaces scipy dependency)
  10. `fp_decay_time_constant` via existing log-linear OLS
  11. `room_spread_delay_minutes` via existing cross-correlation
  12. `cloud_factor_exponent` via log-OLS (when `CLOUD_COVER_CORRECTION_ENABLED`)
  13. `solar_decay_tau_hours` via existing log-linear OLS
- `CALIBRATION_METHOD` config variable (`"scipy"` default, or `"physics"`): selects the calibration path system-wide.
- `--calibrate-physics-direct` CLI flag to `src/main.py`: runs the physics-direct path explicitly, exits after calibration.
- Flag-file support in `src/main.py`: writing `/data/config/calibrate_physics_direct_flag` triggers the physics-direct path on next startup (same pattern as the existing HLC flag).
- Dashboard calibration method radio toggle in `dashboard/components/control.py`: "Scipy Optimizer" vs "Physics Direct" — selecting Physics Direct writes the `calibrate_physics_direct_flag`.
- 19 new unit tests in `tests/unit/test_physics_calibration_direct.py` covering all new analytical estimators.

### Changed
- `train_thermal_equilibrium_model()` in `src/physics_calibration.py` now accepts a `method` parameter (`"scipy"` or `"physics"`).  Existing callers are unaffected (default remains `"scipy"`).  When `CALIBRATION_METHOD="physics"` is set in config and `method="scipy"` is passed (the default), the config value takes precedence.
- `--calibrate-physics` CLI flag now respects `CALIBRATION_METHOD` from config rather than always running scipy.

## [0.2.26] - 2026-05-05

### Added
- **Predictive Pre-Cooling**: Forecast-driven overheating prevention that simulates passive indoor trajectory using PV and outdoor temperature forecasts. Starts cooling before the room overheats by shifting the binary-search target temperature down. Only active in cooling mode.
- New `OverheatingPredictor` class (`src/overheating_predictor.py`) with configurable guard thresholds, horizon, and lead time
- 7 new configuration parameters (`PRE_COOL_*`) for fine-tuning pre-cooling behavior
- Pre-cooling state persistence and HA sensor attributes (`pre_cool_active`, `pre_cool_peak_temp`, `pre_cool_peak_hour`)
- 37 unit tests covering predictor logic, mode isolation, guard thresholds, and integration scenarios

### Fixed
- **CI: Node.js 20 deprecation** — upgraded `actions/checkout@v4` to `@v6` in `auto-docs.yaml`
- **CI: AI API token error** — switched `auto-docs.yaml` and `ai-code-review.yaml` from `https://api.githubcopilot.com` to `https://models.inference.ai.azure.com` (GitHub Models); server-to-server tokens are not supported on the Copilot endpoint
- **Pre-cooling guard bypass for reactive cooling**: Room already above target now bypasses PV/outdoor guards — cooling activates even at night with no PV
- **PV forecast consistency**: Predictor now uses thermal-corrected `pv_forecast_{h}h` values consistently (was mixing raw electrical with thermal-corrected anchor)
- **Cloud cover key mismatch**: Predictor now reads per-hour `cloud_cover_forecast_{h}h` keys from features_dict (was reading nonexistent `avg_cloud_cover` key, always falling back to 50%)

## [0.2.25] - 2026-05-05

### Changed
- **Cooling test helper cleanup**: Refactored `tests/unit/test_heat_source_channels.py::make_context()` into an override-based helper so the new cooling regression coverage stays readable without a long multi-parameter test helper signature.

### Fixed
- **Cooling follow-up review fixes**: `HeatPumpChannel._learn_from_recent()` now learns cooling `delta_t_floor` from the positive magnitude of negative `delta_t` samples, so the learned floor does not collapse toward zero. `temperature_control.py` now includes `climate_mode` in both active and shadow `prediction_context` payloads so downstream channel routing keeps using cooling-specific logic during learning. Added regression tests for cooling HP+PV routing, `climate_mode` propagation, positive `delta_t_floor` learning, and RUNNING→RECOVERY gate behavior.
- **HP channel never learns in cooling mode**: `route_learning()` now short-circuits for cooling mode before the PV-isolation block, always routing a record to the HP channel. PV co-learns in parallel when active. Previously, sunny cooling days had PV always active, so `any_external_active` was always `True` and HP history stayed permanently empty.
- **HeatPumpChannel `_learn_from_recent()` mode-aware fixes**: Delta-t filter now uses `< -0.5` for cooling (was `> 0.5`, rejecting all cooling samples). Outlet-effectiveness gradient sign is negated for cooling (`-avg_error`) to correctly drive OE upward when the model under-predicts cooling. Added mode label to the self-learned debug log.
- **Trajectory steps ignored when `PV_TRAJ_SCALING_ENABLED=true` and forecast mode disabled**: Removed the `PV_TRAJ_SCALING_ENABLED` block that pre-mutated `config.TRAJECTORY_STEPS` to `PV_TRAJ_MAX_STEPS` before feature-building. `physics_features.py` already expands the forecast horizon via `_n_fc_full = max(TRAJECTORY_STEPS, PV_TRAJ_MAX_STEPS)` when `PV_TRAJ_FORECAST_MODE_ENABLED=true`, making the pre-mutation redundant when forecast mode is on and harmful (TRAJECTORY_STEPS never reset) when forecast mode is off.
- **`temperature_control.py` HP-active detection**: Made `heat_pump_active` inference mode-aware — cooling path checks `thermal_power_kw <= COOLING_MIN_THERMAL_POWER_KW` or `delta_t < -0.5`; heating path retains the original `>= HP_ACTIVE_MIN_POWER_KW` / `delta_t > 0.5` logic.
- **Cooling recovery gate deadlock resolved**: RUNNING→RECOVERY transition now uses `_is_heat_pump_active()` (the shared HP detection helper from `heat_source_channels`) instead of a dedicated `delta_t < -HP_ACTIVE_COOLING_DELTA_T` check. This reuses the same logic (thermal_power, delta_t, outlet-vs-inlet) as learning and temperature_control, removing the need for the separate `HP_ACTIVE_COOLING_DELTA_T` config constant. When the HP was already idle and the model wants outlet close to inlet (mild cooling demand), the gate stays in RUNNING and simply clamps outlet to inlet_temp. This prevents a deadlock where mild cooling need → RECOVERY, then RECOVERY→RUNNING also requires a 2K gap that is never met, leaving the HP permanently disabled.

### Added
- `HP_ACTIVE_COOLING_DELTA_T` config constant (default `0.5` K) — threshold for the cooling cycle gate to determine whether the HP was actively cooling in the previous cycle.

### Removed
- `HP_ACTIVE_COOLING_DELTA_T` config constant — replaced by reusing `_is_heat_pump_active()` from `heat_source_channels`, which already combines thermal_power, delta_t, and outlet-vs-inlet signals consistently with learning and temperature_control.

## [0.2.24] - 2026-05-04

### Changed
- **Runtime-replay epsilon tuning**: Finalized finite-difference epsilon calibration using sensitivity analysis and the calibration review notebook so each runtime-reachable parameter produces ΔT ≈ 0.1–0.3°C while remaining in the linear regime
  - `THERMAL_TIME_CONSTANT_EPSILON`: 2.0 → 0.2
  - `HEAT_LOSS_COEFFICIENT_EPSILON`: 0.005 → 0.008
  - `OUTLET_EFFECTIVENESS_EPSILON`: 0.05 → 0.1
  - `TV_HEAT_WEIGHT_EPSILON`: 0.05 → 0.1
  - `SLAB_TIME_CONSTANT_EPSILON`: 0.5 → 1.595
  - `SOLAR_LAG_EPSILON`: kept at 5.0 because the current runtime replay path cannot reach the target signal window (best sweep at 22.5 only yields ΔT ≈ 0.0097°C)
- **Consolidated epsilon constants**: Added `SOLAR_LAG_EPSILON` and `SLAB_TIME_CONSTANT_EPSILON` to `PhysicsConstants`; wrapper methods now reference constants instead of hardcoded values
- `_is_heat_pump_active()` signature: now reads `climate_mode` from context dict
- `_resolve_delta_t_floor()`: accepts optional `climate_mode` parameter for mode-aware behavior
- `predict_thermal_trajectory()`: accepts `climate_mode` kwarg, propagated through binary search
- `COOLING_DEFAULTS`: `pv_heat_weight` 0.0003 → 0.002, `slab_time_constant_hours` 0.8 → 3.19
- `COOLING_BOUNDS`: `slab_time_constant_hours` upper bound 2.5 → 8.0
- Cooling inlet guard replaced with full cooling cycle gate state machine
- `main.py` learning context: removed inline `heat_pump_active` calculation, uses `_is_heat_pump_active()` via `climate_mode` in context

### Added
- **Cooling cycle gate** (Bug 11): State machine with `RUNNING`/`RECOVERY` states prevents HP short-cycling in cooling mode using gradient-based transitions with existing `cooling_shutdown_margin_k` parameter
- **`TARGET_INDOOR_TEMP_COOLING_ENTITY_ID`** (Bug 5): Separate target temperature entity for cooling mode in `config.py`, `config.yaml`, `.env_sample`, and `config_adapter.py`
- **Early climate mode detection** (Bug 3): Climate mode determined before learning step so learning context uses correct mode
- Comprehensive cooling bugfix test suite (`test_cooling_bugfixes.py`) with 19 tests

### Fixed
- **Previous-cycle learning context after mode changes**: Persist `last_climate_mode` and `last_target_indoor_temp`, reuse them during next-cycle online learning, and switch the wrapper to the previous cycle's mode before feedback learning runs
- **Cooling forecast demand semantics in feature building**: `build_physics_features()` now accepts climate mode plus a resolved target override, uses the cooling target consistently, and computes forecast demand as `forecast - target` in cooling mode
- **HP active detection in cooling** (Bug 1): `_is_heat_pump_active()` now mode-aware — uses `HEATING_MIN_THERMAL_POWER_KW` (0.5) for heating, `COOLING_MIN_THERMAL_POWER_KW` (-0.5) for cooling; delta_t and outlet checks inverted for cooling
- **Slab model pump_on gate** (Bug 1b): Slab pump-on detection in `thermal_equilibrium_model.py` now works for cooling (outlet < t_slab, delta_t <= -1.0)
- **HP-OFF delta_t floor substitution** (Bug 1c): In cooling mode, HP-off detected when delta_t > -1.0 (not < 1.0), simulated delta_t is negative for binary search
- **PV surplus offset inverted in cooling** (Bug 6): High PV now lowers target (more cooling) instead of raising it
- **Price offset inverted in cooling** (Bug 7): Cheap electricity now lowers target in cooling mode
- **`heating_demand_forecast` hardcoded 21°C** (Bug 8/9): Replaced with actual `target_temp_f` from sensor data
- **Cooling `pv_heat_weight` 7× too low** (Bug 4): Set to 0.002 (same as heating — building property)
- **Cooling `slab_time_constant_hours` wrong** (Bug 2): Set to 3.19h (same as heating — same slab mass)
- **HP channel never learns in cooling**: All fixes combined enable HP learning in cooling mode (previously 0 history entries after 69 cycles)

## [0.2.23] - 2026-05-04

### Fixed
- Duplicate `env:` key in `.github/workflows/ai-code-review.yaml` (merged `PR_TITLE`/`PR_BODY` into the existing step `env:` block)

### Changed
- Updated GitHub Actions to Node.js 24-compatible versions: `actions/checkout` v4→v6, `docker/login-action` v3→v4, `docker/setup-qemu-action` v3→v4, `docker/setup-buildx-action` v3→v4, `docker/build-push-action` v5→v7

### Changed
- **Runtime-replay epsilon tuning**: Finalized finite-difference epsilon calibration using sensitivity analysis and the calibration review notebook so each runtime-reachable parameter produces ΔT ≈ 0.1–0.3°C while remaining in the linear regime
  - `THERMAL_TIME_CONSTANT_EPSILON`: 2.0 → 0.2
  - `HEAT_LOSS_COEFFICIENT_EPSILON`: 0.005 → 0.008
  - `OUTLET_EFFECTIVENESS_EPSILON`: 0.05 → 0.1
  - `TV_HEAT_WEIGHT_EPSILON`: 0.05 → 0.1
  - `SLAB_TIME_CONSTANT_EPSILON`: 0.5 → 1.595
  - `SOLAR_LAG_EPSILON`: kept at 5.0 because the current runtime replay path cannot reach the target signal window (best sweep at 22.5 only yields ΔT ≈ 0.0097°C)
- **Consolidated epsilon constants**: Added `SOLAR_LAG_EPSILON` and `SLAB_TIME_CONSTANT_EPSILON` to `PhysicsConstants`; wrapper methods now reference constants instead of hardcoded values
- `_is_heat_pump_active()` signature: now reads `climate_mode` from context dict
- `_resolve_delta_t_floor()`: accepts optional `climate_mode` parameter for mode-aware behavior
- `predict_thermal_trajectory()`: accepts `climate_mode` kwarg, propagated through binary search
- `COOLING_DEFAULTS`: `pv_heat_weight` 0.0003 → 0.002, `slab_time_constant_hours` 0.8 → 3.19
- `COOLING_BOUNDS`: `slab_time_constant_hours` upper bound 2.5 → 8.0
- Cooling inlet guard replaced with full cooling cycle gate state machine
- `main.py` learning context: removed inline `heat_pump_active` calculation, uses `_is_heat_pump_active()` via `climate_mode` in context

### Added
- **Cooling cycle gate** (Bug 11): State machine with `RUNNING`/`RECOVERY` states prevents HP short-cycling in cooling mode using gradient-based transitions with existing `cooling_shutdown_margin_k` parameter
- **`TARGET_INDOOR_TEMP_COOLING_ENTITY_ID`** (Bug 5): Separate target temperature entity for cooling mode in `config.py`, `config.yaml`, `.env_sample`, and `config_adapter.py`
- **Early climate mode detection** (Bug 3): Climate mode determined before learning step so learning context uses correct mode
- Comprehensive cooling bugfix test suite (`test_cooling_bugfixes.py`) with 19 tests

### Fixed
- **Previous-cycle learning context after mode changes**: Persist `last_climate_mode` and `last_target_indoor_temp`, reuse them during next-cycle online learning, and switch the wrapper to the previous cycle's mode before feedback learning runs
- **Cooling forecast demand semantics in feature building**: `build_physics_features()` now accepts climate mode plus a resolved target override, uses the cooling target consistently, and computes forecast demand as `forecast - target` in cooling mode
- **HP active detection in cooling** (Bug 1): `_is_heat_pump_active()` now mode-aware — uses `HEATING_MIN_THERMAL_POWER_KW` (0.5) for heating, `COOLING_MIN_THERMAL_POWER_KW` (-0.5) for cooling; delta_t and outlet checks inverted for cooling
- **Slab model pump_on gate** (Bug 1b): Slab pump-on detection in `thermal_equilibrium_model.py` now works for cooling (outlet < t_slab, delta_t <= -1.0)
- **HP-OFF delta_t floor substitution** (Bug 1c): In cooling mode, HP-off detected when delta_t > -1.0 (not < 1.0), simulated delta_t is negative for binary search
- **PV surplus offset inverted in cooling** (Bug 6): High PV now lowers target (more cooling) instead of raising it
- **Price offset inverted in cooling** (Bug 7): Cheap electricity now lowers target in cooling mode
- **`heating_demand_forecast` hardcoded 21°C** (Bug 8/9): Replaced with actual `target_temp_f` from sensor data
- **Cooling `pv_heat_weight` 7× too low** (Bug 4): Set to 0.002 (same as heating — building property)
- **Cooling `slab_time_constant_hours` wrong** (Bug 2): Set to 3.19h (same as heating — same slab mass)
- **HP channel never learns in cooling**: All fixes combined enable HP learning in cooling mode (previously 0 history entries after 69 cycles)

## [0.2.22] - 2026-05-03

### Fixed
- **Cooling inlet guard** — in cooling mode, when the binary search converges to an outlet temperature within `MIN_COOLING_DELTA_K` of the actual inlet (return-water) temperature, the NIBE compressor cannot operate — the gap is too small. Previously the impossible setpoint was sent as-is, causing the heat pump to short-cycle or reject the command.
  - `src/model_wrapper.py` (`_calculate_required_outlet_temp`): the binary search upper bound is now tightened with `inlet − MIN_COOLING_DELTA_K` when `inlet_temp` is available in features, in addition to the existing `indoor − MIN_COOLING_DELTA_K` proxy.
  - `src/model_wrapper.py` (`calculate_optimal_outlet_temp`): new post-search **inlet guard** — when the result is `> inlet − MIN_COOLING_DELTA_K`, the outlet is clamped to `inlet_temp` so the compressor stays idle (circulator only). The HP idle signal is `outlet = inlet` (supply water at return temperature = no compressor work needed).
  - 6 new tests in `tests/unit/test_cooling_mode.py` covering: gap too small → clamp, gap sufficient → pass-through, exact boundary → pass-through, heating mode not affected, inlet unavailable → pass-through, binary search bound tightened by inlet.
- **State file reload on both heating↔cooling transitions** — previously operational state (`last_final_temp`, `setpoint_hold_cycles_remaining`, etc.) was only reloaded from the new file when switching TO cooling. Switching FROM cooling BACK TO heating left `state` stale (still pointing at the cooling file's data).
  - `src/main.py`: captures `_prev_state_manager` before `set_climate_mode()` and reloads `state` whenever `_active_state_manager is not _prev_state_manager`, covering both directions of the transition.
- **HLC calibration `window_size` validation** — if `HLC_WINDOW_SIZE_ROWS` was misconfigured to 0 or a negative value, `calibrate_hlc()` would raise a `ValueError` from `range()` (step must not be zero). Added an explicit guard that resets the value to the default (12) and logs a warning.
  - `src/hlc_learner.py`: clamp `window_size` to `max(1, window_size)` with a warning log.
- **"Sliding window" wording corrected** — `calibrate_hlc()` processes data in non-overlapping blocks (stride == window size), not a sliding window. The misleading description has been updated in:
  - `src/config.py`: comment for `HLC_WINDOW_SIZE_ROWS`
  - `ml_heating_underfloor/config.yaml`: inline comment for `hlc_window_size_rows`
  - `ml_heating_underfloor/translations/en.yaml`: UI description for `hlc_window_size_rows`
- **Cooling state isolation** — operational state, blocking state, and end-of-cycle saves were always written to the heating `unified_thermal_state.json` even during cooling-mode cycles, because `state_manager.save_state()` / `load_state()` were hardwired to the heating singleton (`get_thermal_state_manager()`).
  - `src/state_manager.py`: `load_state()` and `save_state()` now accept an optional `state_manager` parameter (defaults to the heating singleton so all existing callers are unaffected).
  - `src/main.py`: At the top of each cycle `_active_state_manager` is resolved from the wrapper's current mode manager; all three `save_state()` call sites pass `state_manager=_active_state_manager`. When cooling mode is detected, operational state is immediately reloaded from the cooling file.
  - `src/physics_calibration.py`: `train_thermal_equilibrium_model()`, `optimize_thermal_parameters()`, and `backup_existing_calibration()` each accept an optional `state_manager` parameter so calibration results can be directed to the correct file.
  - `src/main.py`: `HLCSessionLearner.apply_to_thermal_state()` call now passes `thermal_state_manager=_active_state_manager` so HLC updates go to the active mode's file.
  - `dashboard/data_service.py`: `load_thermal_state()` and `get_state_file_info()` automatically switch to the cooling state file when it is detected as more recently modified than the heating file (cooling mode active heuristic). New helpers `_find_cooling_state_file()` and `_is_cooling_mode_active()` encapsulate the detection logic.

### Added
- **HLC calibration quality improvements** — 9 targeted fixes to `calibrate_hlc()` in `src/hlc_learner.py`:
  - **Fix 1**: Date range in log and return dict now shows actual datetime strings (uses `_time` column) instead of integer indices after `reset_index`
  - **Fix 2**: New `HLC_MIN_FLOW_RATE_LPM` config gate (default 0.5 L/min) rejects low/zero-flow windows before thermal-power computation, preventing forward-filled standby rows from passing quality filters
  - **Fix 3**: Explicit warning when `target_temp` column is absent (indoor_far_from_target and low_heating_demand quality gates disabled)
  - **Fix 4**: Per-window timestamp-continuity check rejects windows that straddle a `> 10 min` time gap in the `_time` column
  - **Fix 5**: Three fit-quality metrics now logged and returned: standard R², FTO-R² (forced-through-origin), and Pearson r between Q and ΔT
  - **Fix 6**: `physics_calibration.py` warns when quality-gate columns (`defrost`, `tv`, `dhw`, `fireplace`) are bfill-filled for > 24 h at the dataset start — the corresponding rejection filter is silently inactive for that period
  - **Fix 7**: New `HLC_WINDOW_SIZE_ROWS` config var (default 12 = 60 min) for the calibration window size, replacing the hardcoded 20-minute (4-row) window
  - **Fix 8**: Optional with-intercept regression diagnostic via `HLC_REGRESSION_INTERCEPT=true` — fits `Q = HLC×ΔT + Q0` and logs Q0 as a contamination indicator
  - **Fix 9**: `HLC_MIN_THERMAL_POWER_KW` promoted to `HEATING_MIN_THERMAL_POWER_KW` (default 0.5 kW), shared across HLC calibration, physics calibration, and session learner
- **Thermal power gate standardisation** — replaced all hardcoded `0.5 kW` and `> 0` thermal-power comparisons with shared config vars:
  - `HEATING_MIN_THERMAL_POWER_KW = 0.5` — calibration quality gate applied in `hlc_learner.py` (calibration + session learner) and `physics_calibration.py` (7 call sites: slab tau, HP-off detection, direct heat loss, `_filter_hp_only_periods`, delta-T floor)
  - `COOLING_MIN_THERMAL_POWER_KW = -0.5` — reserved for cooling-side calibration quality gates (thermal power is negative in cooling mode)
  - `HP_ACTIVE_MIN_POWER_KW = 0.05` — runtime HP-running noise-floor check in `heat_source_channels._is_heat_pump_active()` and `temperature_control._perform_learning()`
- New config vars: `HLC_WINDOW_SIZE_ROWS`, `HLC_MIN_FLOW_RATE_LPM`, `HLC_REGRESSION_INTERCEPT`, `HEATING_MIN_THERMAL_POWER_KW`, `COOLING_MIN_THERMAL_POWER_KW`, `HP_ACTIVE_MIN_POWER_KW`
- All new vars synchronised in `ml_heating_underfloor/config.yaml` (options + schema), `.env_sample`, and `ml_heating_underfloor/translations/en.yaml` (tooltips)
- Return dict of `calibrate_hlc` now includes `r2_fto` and `r_pearson` fields
- 4 new unit tests in `tests/unit/test_hlc_learner.py` covering: datetime date_range, high-R² on perfect linear data, standby window rejection, missing target_temp warning

## [0.2.21] - 2026-05-02

### Fixed
- **Cooling-mode state contamination**: `EnhancedModelWrapper` and `ThermalEquilibriumModel` always used the heating-mode state manager (`unified_thermal_state.json`), so prediction records, learning parameter adjustments, and channel states written during cooling cycles contaminated the heating file. Each mode now uses its own state manager: `_heating_state_manager` (heating JSON) and `_cooling_state_manager` (cooling JSON). `set_climate_mode()` swaps the active `thermal_model`, `state_manager`, `prediction_metrics`, and reloads `cycle_count` from the correct file. `ThermalEquilibriumModel.__init__` now accepts an optional `state_manager` injection; the new `_get_state_manager()` helper returns the injected manager or falls back to the heating singleton, replacing all four inline `get_thermal_state_manager()` call-sites inside the model.
- **HLC calibration `Missing required columns: {'indoor_temp'}`**: `calibrate_hlc()` in `hlc_learner.py` previously used keyword-based heuristics (e.g. "indoor" + "temp") to identify DataFrame columns, which silently failed for non-English entity IDs such as `rt_mittelwert`. The function now uses the shared `fetch_historical_data_for_calibration()` helper from `physics_calibration.py` (same strategy as model calibration), which respects `TRAINING_DATA_SOURCE` and performs HA history fallback/supplement in auto mode. Column identification uses `config.*_ENTITY_ID.split(".", 1)[-1]` — the exact short names produced by InfluxDB and HA history, with no language assumptions.

## [0.2.20] - 2026-05-02

### Fixed
- Dashboard: replaced deprecated `use_container_width` parameter with `width='stretch'` in all `st.plotly_chart()` and `st.dataframe()` calls (`performance.py`, `backup.py`, `overview.py`)
- Dashboard: replaced non-existent `supervisorctl` subprocess calls in `control.py` with signal-based process management (`os.kill(pid, SIGTERM)`) so the HLC Calibrate, Restart, and Stop buttons work correctly in the containerised environment; `start_ml_system()` now returns an informative message instead of raising a `FileNotFoundError`
- `app.py`: moved `st.set_page_config()` to be the first Streamlit command in `main()`; removed a `st.write()` call from `setup_ingress_config()` that violated this requirement and caused a crash when running under HA ingress
- `health.py`: replaced `timedelta.seconds` with `timedelta.total_seconds()` in `check_ml_system()` so log files older than 24 h no longer falsely appear as "active"
- `control.py`: added `os.makedirs('/data/config', exist_ok=True)` in `trigger_model_recalibration()` and `save_config_changes()` to prevent `FileNotFoundError` when the config directory does not yet exist
- `data_service.py`: use `datetime.now(timezone.utc)` when `last_run_time` carries timezone info and `datetime.now()` when it is naive, so the age calculation is always correct regardless of UTC offset; added two regression tests for UTC-offset timestamps
- `control.py`: replaced `pgrep -f src.main` subprocess call with a direct `/proc` filesystem scan so that the `procps` package is not required in the Alpine container
- `control.py`: updated `stop_ml_system()` docstring and UI message to accurately describe that SIGTERM to the ML backend triggers the whole add-on to restart via `run.sh wait -n`
- `backup.py`: replaced all placeholder download stubs with real `st.download_button` calls for state file, backup ZIP, and export JSON downloads; state file download now resolves the path via `_find_state_file()` (honours `UNIFIED_STATE_FILE` env-var and `*_shadow` variants) instead of a hardcoded `/data/models` path
- `backup.py`: added `render_view_details_interface()` function and wired it into `render_backup()` so the "View Details" button actually renders backup metadata
- `backup.py`: added `render_delete_interface()` function and wired it into `render_backup()` so the "Delete" button presents a confirmation dialog and physically removes the backup file

## [0.2.19] - 2026-05-01

### Added
- **Historical HLC Calibration**: New `calibrate_hlc()` function fetches historical data from InfluxDB, filters stable HP-only periods, and runs OLS regression to estimate building heat loss coefficient
- **`--calibrate-hlc` CLI argument**: One-shot HLC calibration from historical data and exit
- **HLC calibration flag detection**: Main loop detects `/data/config/hlc_calibrate_flag` at startup and runs calibration automatically
- **Dashboard "Calibrate HLC" button**: Writes flag file and restarts the ML system to trigger HLC calibration
- **Cold start file creation**: `HLCSessionLearner.load_day_records()` now creates an empty stub file when no session file exists
- **HLC calibration config params**: `HLC_CALIBRATION_LOOKBACK_HOURS` (default 720) and `HLC_CALIBRATION_MIN_PERIODS` (default 20)

### Removed
- **Online HLC Learner**: Removed `HLCLearner` class and `HLCWindow` dataclass — replaced by day-level session learner and historical calibration
- **Online HLC config params**: Removed `HLC_LEARNER_ENABLED`, `HLC_WINDOW_MINUTES`, `HLC_CYCLES_PER_WINDOW_MIN_FRAC`, `HLC_MIN_WINDOWS`, `HLC_MAX_WINDOWS`, `HLC_MAX_UPDATE_FRACTION` from config, config.yaml, config_adapter, and translations

### Changed
- **`_build_cycle()` extracted as module-level function**: Previously `HLCLearner._build_cycle()` static method, now shared by `HLCSessionLearner`
- **main.py HLC push_cycle simplified**: Single session learner block instead of dual online + session blocks
- **Tests rewritten**: `test_hlc_learner.py` now tests `_build_cycle()`, `calibrate_hlc()`, and `HLCCycle` instead of removed classes

### Fixed
- **Shared HLC validation params restored**: 6 validation gate params (`HLC_PV_MAX_W`, `HLC_MAX_INDOOR_DELTA`, `HLC_MAX_TREND`, `HLC_OUTDOOR_TEMP_MIN`, `HLC_OUTDOOR_TEMP_MAX`, `HLC_MIN_HEATING_DEMAND_K`) were accidentally removed with the online learner but are used by `_close_day()` and `calibrate_hlc()` via `getattr()` — now restored under "HLC Validation Gates" section
- **Greedy column name matching in `calibrate_hlc()`**: Derived columns (e.g. `indoor_temp_delta_60m`) could overwrite base temperature mappings; now uses `setdefault()` and skips columns containing delta/lag/diff/trend/gradient/forecast
- **Uncapped HLC written to thermal state**: `calibrate_hlc()` now rejects estimates outside [0.01, 2.0] kW/K to prevent physically implausible values from corrupting the model
- **Flag file removal race**: If `/data/config/hlc_calibrate_flag` cannot be removed, calibration is skipped with an error log instead of running on every restart
- **Missing indoor trend gate in `calibrate_hlc()`**: Added first-to-last indoor temperature change check (max_trend) to match the session learner's quality gates

## [0.2.18] - 2026-05-01

### Added
- **Historical HLC Calibration**: New `calibrate_hlc()` function fetches historical data from InfluxDB, filters stable HP-only periods, and runs OLS regression to estimate building heat loss coefficient
- **`--calibrate-hlc` CLI argument**: One-shot HLC calibration from historical data and exit
- **HLC calibration flag detection**: Main loop detects `/data/config/hlc_calibrate_flag` at startup and runs calibration automatically
- **Dashboard "Calibrate HLC" button**: Writes flag file and restarts the ML system to trigger HLC calibration
- **Cold start file creation**: `HLCSessionLearner.load_day_records()` now creates an empty stub file when no session file exists
- **HLC calibration config params**: `HLC_CALIBRATION_LOOKBACK_HOURS` (default 720) and `HLC_CALIBRATION_MIN_PERIODS` (default 20)

### Removed
- **Online HLC Learner**: Removed `HLCLearner` class and `HLCWindow` dataclass — replaced by day-level session learner and historical calibration
- **Online HLC config params**: Removed `HLC_LEARNER_ENABLED`, `HLC_WINDOW_MINUTES`, `HLC_CYCLES_PER_WINDOW_MIN_FRAC`, `HLC_MIN_WINDOWS`, `HLC_MAX_WINDOWS`, `HLC_MAX_UPDATE_FRACTION` from config, config.yaml, config_adapter, and translations

### Changed
- **`_build_cycle()` extracted as module-level function**: Previously `HLCLearner._build_cycle()` static method, now shared by `HLCSessionLearner`
- **main.py HLC push_cycle simplified**: Single session learner block instead of dual online + session blocks
- **Tests rewritten**: `test_hlc_learner.py` now tests `_build_cycle()`, `calibrate_hlc()`, and `HLCCycle` instead of removed classes

### Fixed
- **Shared HLC validation params restored**: 6 validation gate params (`HLC_PV_MAX_W`, `HLC_MAX_INDOOR_DELTA`, `HLC_MAX_TREND`, `HLC_OUTDOOR_TEMP_MIN`, `HLC_OUTDOOR_TEMP_MAX`, `HLC_MIN_HEATING_DEMAND_K`) were accidentally removed with the online learner but are used by `_close_day()` and `calibrate_hlc()` via `getattr()` — now restored under "HLC Validation Gates" section
- **Greedy column name matching in `calibrate_hlc()`**: Derived columns (e.g. `indoor_temp_delta_60m`) could overwrite base temperature mappings; now uses `setdefault()` and skips columns containing delta/lag/diff/trend/gradient/forecast
- **Uncapped HLC written to thermal state**: `calibrate_hlc()` now rejects estimates outside [0.01, 2.0] kW/K to prevent physically implausible values from corrupting the model
- **Flag file removal race**: If `/data/config/hlc_calibrate_flag` cannot be removed, calibration is skipped with an error log instead of running on every restart
- **Missing indoor trend gate in `calibrate_hlc()`**: Added first-to-last indoor temperature change check (max_trend) to match the session learner's quality gates

## [0.2.17] - 2026-04-30

### Added
- **Day-Level HLC Session Learner** (`HLCSessionLearner` class in `src/hlc_learner.py`): accumulates HP cycles per calendar day, validates each day against the same quality gates as the existing 60-min HLC learner, and persists validated `DayRecord`s to a rolling JSON file. OLS regression over stored day records produces a multi-day HLC estimate that survives process restarts.
- `DayRecord` dataclass with fields: `date`, `mean_thermal_power_kw`, `mean_delta_t`, `n_cycles`, `outdoor_temp_mean`, `indoor_temp_mean`, `avg_power_w` (mean thermal power in Watts = mean_thermal_power_kw × 1000).
- 6 new configuration variables: `HLC_SESSION_ENABLED`, `HLC_SESSION_FILE`, `HLC_SESSION_MIN_CYCLES`, `HLC_SESSION_MAX_DAYS`, `HLC_SESSION_MIN_DAYS`, `HLC_SESSION_MAX_UPDATE_FRACTION`. `HLC_SESSION_FILE` defaults to the same directory as `UNIFIED_STATE_FILE`.
- Tooltip descriptions for all 6 new config parameters in `ml_heating_underfloor/translations/en.yaml`.
- 18 unit tests in `tests/unit/test_hlc_session_learner.py` covering `DayRecord`, no-HP-activity guard, minimum-cycle rejection, day rollover, load/save round-trip, OLS correctness, and apply-to-thermal-state capping.

## [0.2.16] - 2026-04-29

### Added
- Comprehensive test coverage improvements: 161 new unit tests across 5 modules
  - `tests/unit/test_thermal_constants.py`: `ThermalUnits`, `ThermalParameterValidator`, and convenience function coverage (55% → 98%)
  - `tests/unit/test_prediction_metrics_extended.py`: file I/O, 24h window methods, simplified accuracy breakdown, state-manager integration (63% → 84%)
  - `tests/unit/test_ha_history_service_extended.py`: `_build_entity_map`, duplicate timestamp handling, edge cases (75% → 90%)
  - `tests/unit/test_adaptive_fireplace_learning_extended.py`: `get_enhanced_fireplace_features`, `get_learning_summary`, integration helper (73% → 89%)
  - `tests/unit/test_multi_heat_source_physics_extended.py`: `enhance_physics_features_with_heat_sources`, `_encode_heat_source` (72% → 81%)
- Overall test count increased from 785 to 945 passing tests; overall source coverage improved from 74% to 77%

### Changed
- **`pv_scalar` calculation in binary search** (`src/model_wrapper.py`): reverted the stateful EMA back to the rolling-window average (`mean(pv_power_history)`). End-of-sun override: when `pv_forecast_electrical_1h` (fallback `pv_forecast_1h`) ≤ `PV_TRAJ_ZERO_W`, the rolling window is cleared and `pv_scalar` snaps to `pv_now` so the binary search plans without stale high averages. No new stateful attribute; `PV_SCALAR_EMA_ALPHA` removed.
- **PV surplus CHEAP offset** (`src/model_wrapper.py`): replaced binary on/off at `PV_SURPLUS_CHEAP_THRESHOLD_W` with a linear soft-ramp over a configurable `PV_SURPLUS_CHEAP_RAMP_W` band below the threshold. In the ramp zone `[threshold - ramp_w, threshold]` the offset scales from 0 to `PRICE_TARGET_OFFSET`; below the ramp floor offset is 0; at or above threshold full offset applies. The `new_adjusted > target_adjusted` guard is preserved.
- **Overshoot dampening numerator** (`src/model_wrapper.py`): increased from `0.4` to `1.0` in `overshoot_dampening = 1.0 / max(slab_tau, 1.0)`. Pull-back correction is 2.5× stronger when overshoot is detected. The `/ slab_tau` denominator still protects slow slabs from oscillation.

### Added
- `PV_SURPLUS_CHEAP_RAMP_W` config variable (default equals `PV_SURPLUS_CHEAP_THRESHOLD_W`, in `src/config.py`) — controls the blend-zone width for the soft-ramp CHEAP offset.

### Removed
- `PV_SCALAR_EMA_ALPHA` config variable (`src/config.py`) — EMA replaced by rolling-window average.

## [0.2.15] - 2026-04-28

### Added
- `PV_TRAJ_FORECAST_RESCUE_ENABLED` config flag (default `true`): when a passing rain cloud drops `pv_now` below `PV_TRAJ_THRESHOLD_W`, the forecast-driven trajectory mode stays active if at least `PV_TRAJ_MIN_STEPS` forecast hours still exceed the threshold — preventing an abrupt collapse of the pre-heat plan.

### Fixed
- `physics_features.py` now fetches forecasts up to `PV_TRAJ_MAX_STEPS` hours (instead of only `TRAJECTORY_STEPS` hours) when `PV_TRAJ_FORECAST_MODE_ENABLED=true`. Previously `pv_forecast_5h … pv_forecast_12h` were always 0.0 W (= "night"), artificially capping the planning horizon at `TRAJECTORY_STEPS` even when the forecast showed many remaining solar hours.

## [0.2.14] - 2026-04-28

### Removed
- Classic PV trajectory scaling mode (pv_ratio × time-of-day factor formula) including morning/midday/afternoon/night factors, system KWP normalisation, and seasonal KWP scaling. The forecast-driven mode is now the only available trajectory scaling algorithm.
- `PV_TRAJ_SCALING_ENABLED` config flag — no longer required. Forecast-driven trajectory is enabled directly via `PV_TRAJ_FORECAST_MODE_ENABLED`.
- Config parameters: `PV_TRAJ_SYSTEM_KWP`, `PV_TRAJ_MORNING_FACTOR`, `PV_TRAJ_MIDDAY_FACTOR`, `PV_TRAJ_AFTERNOON_FACTOR`, `PV_TRAJ_NIGHT_FACTOR`, `PV_TRAJ_SEASONAL_SCALING_ENABLED`, `PV_TRAJ_LATITUDE`, `PV_TRAJ_SEASONAL_MIN_FACTOR`.

### Changed
- `compute_forecast_driven_trajectory_steps()` now adds `PV_TRAJ_MIN_STEPS` to the count of remaining solar hours before clamping (`steps = clamp(remaining_pv_hours + MIN_STEPS, MIN, MAX)`). This reserves the minimum trajectory window for the post-sunset period, giving a fuller planning horizon (e.g. 9 solar hours + 4 MIN_STEPS → 13 → clamped to MAX 12).
- `compute_dynamic_trajectory_steps()` now gates on `PV_TRAJ_FORECAST_MODE_ENABLED` directly (was previously gated by `PV_TRAJ_SCALING_ENABLED` with an inner check for `PV_TRAJ_FORECAST_MODE_ENABLED`). When disabled it returns the static `TRAJECTORY_STEPS` value unchanged.
- Documentation (PARAMETER_REFERENCE.md, config.yaml help texts, .env_sample, translations/en.yaml) updated to reflect removal of classic mode parameters.

## [0.2.13] - 2026-04-28

### Added
- **Forecast-Driven Dynamic Trajectory Mode**: New `PV_TRAJ_FORECAST_MODE_ENABLED` option that replaces the `pv_ratio × tod_factor` formula with a forecast-driven algorithm. When enabled, trajectory steps equal the number of consecutive forecast hours with PV above `PV_TRAJ_ZERO_W` (50 W default), giving a long planning horizon in the morning that shrinks naturally toward sunset without any time-of-day factor or kWp normalisation. Requires `PV_TRAJ_SCALING_ENABLED=true`.
- `PV_TRAJ_THRESHOLD_W` (default 3000 W): minimum current PV to activate forecast mode.
- `PV_TRAJ_ZERO_W` (default 50 W): PV threshold below which a forecast slot counts as night.
- `PV_TRAJ_DISABLE_PRICE_IN_FORECAST_MODE` (default true): suppress electricity price target offset while forecast trajectory is active.
- `compute_forecast_driven_trajectory_steps()` and `is_forecast_trajectory_active()` public functions in `src/pv_trajectory.py`.
- 17 new unit tests in `TestForecastDrivenTrajectorySteps` covering all activation/deactivation paths, night mode, step clamping, and delegation from `compute_dynamic_trajectory_steps`.
- UI descriptions (`name` + `description`) added to `translations/en.yaml` for all 16 previously undocumented parameters: 12 Online HLC Learner params (`hlc_*`) and 4 Forecast-Driven Trajectory params (`pv_traj_forecast_mode_enabled`, `pv_traj_threshold_w`, `pv_traj_zero_w`, `pv_traj_disable_price_in_forecast_mode`).

## [0.2.12] - 2026-04-27

### Added
- **Online HLC Learner** (`src/hlc_learner.py`): new `HLCLearner` class that accumulates validated 60-minute windows of live HP-only, near-equilibrium cycle data and runs OLS regression (Q_hp = HLC × ΔT) to estimate the building's Heat Loss Coefficient in kW/K. Disabled by default (`HLC_LEARNER_ENABLED=false`). When enabled it pushes data from `main.py` every control cycle and can apply the resulting estimate to the unified thermal state baseline.
- Twelve new config variables for the HLC learner: `HLC_LEARNER_ENABLED`, `HLC_WINDOW_MINUTES`, `HLC_CYCLES_PER_WINDOW_MIN_FRAC`, `HLC_PV_MAX_W`, `HLC_MAX_INDOOR_DELTA`, `HLC_MAX_TREND`, `HLC_OUTDOOR_TEMP_MIN`, `HLC_OUTDOOR_TEMP_MAX`, `HLC_MIN_HEATING_DEMAND_K`, `HLC_MIN_WINDOWS`, `HLC_MAX_WINDOWS`, `HLC_MAX_UPDATE_FRACTION`.
- 46 unit tests in `tests/unit/test_hlc_learner.py` covering window validation (all rejection paths), OLS regression accuracy, cap logic, rolling-window eviction, and end-to-end push/estimate flow.

## [0.2.11] - 2026-04-26

### Added
- **Parameter documentation**: Added `ml_heating_underfloor/translations/en.yaml` with human-readable names and descriptions for all ~120 add-on configuration parameters, displayed in the Home Assistant Configuration tab. Added `docs/PARAMETER_REFERENCE.md` with the full parameter reference (all 30 sections, defaults, ranges, and guidance). Updated `README.md` with a new Configuration Reference section linking to the full reference.
- **Seasonal PV KWP Scaling**: New `seasonal_kwp_factor()` function in `src/pv_trajectory.py` scales the effective PV peak by the ratio of today's maximum solar elevation to the summer-solstice maximum. This normalises PV production so that a clear winter day (full output for the season) correctly maps to `pv_ratio=1.0`, giving the trajectory optimizer a full planning horizon even in winter.
- New config vars `PV_TRAJ_SEASONAL_SCALING_ENABLED` (default `false`), `PV_TRAJ_LATITUDE` (default `51.0`), and `PV_TRAJ_SEASONAL_MIN_FACTOR` (default `0.1`) in `src/config.py`, `ml_heating_underfloor/config.yaml`, `config_adapter.py`, `.env`, and `.env_sample`.
- 13 new unit tests in `TestSeasonalKwpFactor` and `TestComputeDynamicStepsWithSeasonal` in `tests/unit/test_pv_trajectory.py`.

### Changed
- **Config synchronization**: `.env` and `.env_sample` reorganised into 16 labelled sections aligned with `config.yaml` section headings. All duplicate parameter blocks removed from `.env`. Missing params added to all three config files: `TREND_DECAY_TAU_HOURS`, `PV_ROOM_DECAY_MULTIPLIER`, `DECAY_CANCEL_MARGIN`, `OUTLET_SMOOTHING_ALPHA`, `OUTLET_SMOOTHING_BYPASS`, `MIN_SETPOINT_HOLD_CYCLES`, `DEFROST_RECOVERY_GRACE_MINUTES`, `TRAINING_DATA_SOURCE`, all `PV_TRAJ_*` trajectory scaling params, `UNIFIED_STATE_FILE_COOLING`.
- `ml_heating_underfloor/config.yaml`: added `trend_decay_tau_hours`, `pv_room_decay_multiplier`, `decay_cancel_margin`, seasonal trajectory scaling options, and their schema entries.
- `config_adapter.py`: added mappings for `TREND_DECAY_TAU_HOURS`, `PV_ROOM_DECAY_MULTIPLIER`, `DECAY_CANCEL_MARGIN`, and the three seasonal scaling vars. Removed deprecated `safety_max_temp`/`safety_min_temp` validation dead code.

### Removed
- **`ELECTRICITY_PRICE_ENTITY_ID`** config var removed from all config surfaces (`src/config.py`, `.env`, `.env_sample`, `ml_heating_underfloor/config.yaml`, `config_adapter.py`). Prices are fetched exclusively via the `tibber.get_prices` HA service call through `PriceOptimizer.refresh_prices_if_needed()` — a sensor entity is not needed or polled.
- **`HAClient.get_electricity_price()`** method removed from `src/ha_client.py`. This method was already marked deprecated, was never called by any production code path, and relied on `ELECTRICITY_PRICE_ENTITY_ID`.

## [0.2.10] - 2026-04-25

### Added
- Extended trajectory horizon from 6 to up to 12 hours: `TRAJECTORY_STEPS` env var now accepted up to 12 (previously 8 via HA addon validation)
- `src/ha_client.py`: `get_hourly_forecast()`, `get_hourly_cloud_cover()`, and `get_calibrated_hourly_forecast()` now fetch up to `TRAJECTORY_STEPS` hourly slots from the HA weather API instead of hard-coding 6
- `src/physics_features.py`: `temp_forecast_{h}h`, `pv_forecast_{h}h`, and `cloud_cover_forecast_{h}h` feature keys are now generated dynamically up to `TRAJECTORY_STEPS` via a loop (previously hard-coded 1h–6h)
- `src/prediction_context.py`: forecast and fallback arrays are now `TRAJECTORY_STEPS` elements long; the cycle-aligned slot selection uses a general formula `min(round(cycle_hours), TRAJECTORY_STEPS) - 1` replacing the previous 6-branch if/elif ladder
- `src/model_wrapper.py`: forecast display dict for multi-horizon outlet-temp predictions is now built dynamically up to `TRAJECTORY_STEPS` steps; average divisor updated accordingly
- `src/forecast_analytics.py`: fallback strategy dict in `get_forecast_fallback_strategy()` and trend computation in `calculate_thermal_forecast_impact()` respect `TRAJECTORY_STEPS`; `[3]` hard-codes replaced with `[-1]`
- 13 new unit tests in `tests/unit/test_trajectory_12h.py` covering every pipeline layer at 12-hour horizon
- PV surplus CHEAP override (`PV_SURPLUS_CHEAP_ENABLED`, `PV_SURPLUS_CHEAP_THRESHOLD_W`): when current PV ≥ threshold the binary-search target is raised by `+PRICE_TARGET_OFFSET`, treating solar surplus identically to a cheap Tibber period
- Minimum setpoint hold (`MIN_SETPOINT_HOLD_CYCLES`): once a setpoint is emitted it is held for at least this many cycles before the optimizer may produce a new value; `setpoint_hold_cycles_remaining` persisted in `SystemState`
- Dynamic trajectory scaling (`PV_TRAJ_SCALING_ENABLED`): new module `src/pv_trajectory.py` with `compute_dynamic_trajectory_steps()` — each cycle, `TRAJECTORY_STEPS` and `MIN_SETPOINT_HOLD_CYCLES` are overridden based on actual PV power relative to `PV_TRAJ_SYSTEM_KWP` and time-of-day factors (morning 0.5 / midday 1.0 / afternoon 0.75 / night 0.0); more solar → longer horizon → bolder pre-heating commitment
- 6 new tests in `TestPvSurplusCheapOverride`, 21 new tests in `tests/unit/test_pv_trajectory.py` covering all time windows, boundary cases, 15 kWp example, and misconfiguration handling

### Changed
- `ml_heating_underfloor/config.yaml`: `trajectory_steps` validation widened from `int(2,8)` to `int(2,12)`; inline comment updated; new option groups for PV Surplus Optimization, Setpoint Stability, and Dynamic Trajectory Scaling

## [0.2.9] - 2026-04-24

### Added
- Startup sensor validation on first cycle in main loop — validates HA sensor availability before processing
- Prediction drift detection (`_check_prediction_drift`) in model_wrapper.py — detects sustained MAE degradation over 50 cycles and boosts learning confidence (+2.0, cap 10.0) to accelerate re-adaptation
- Dynamic confidence cap (`_max_learning_confidence`) on ThermalEquilibriumModel — allows drift-boosted confidence up to 10.0, normal cap 5.0
- Model health computation (`_compute_model_health`) with improvement-aware downgrade logic
- Prediction metrics persistence — `_save_to_state()` now writes `accuracy_stats` (MAE/RMSE per window) and `recent_performance` (last 10 predictions) to unified thermal state
- `.github/copilot-instructions.md` — project-wide Copilot instructions ensuring changelog, memory-bank, and docs are updated automatically every session

### Fixed
- Indoor temperature log bug — shadow-mode comparison `else` branch was at wrong indentation, producing misleading "indoor temp unavailable" log messages
- Bare `except Exception` blocks in ha_client.py — replaced with specific `except (requests.RequestException, KeyError, ValueError)` with warning logs
- Dashboard health error masking — replaced bare `except Exception` in health.py with specific `except OSError` / `except (json.JSONDecodeError, OSError)` with warning logs
- JSON string corruption in unified_thermal_state.py — `update_operational_state()` now validates `last_run_features` at write time, re-validates decoded JSON values, and logs failed `to_dict()` normalization attempts
- Grace period duplication — removed dead second `if is_grace_period:` block in main.py (first block does `continue`, second was unreachable)
- Drift detection metric keys — fixed `mae_recent`/`mae_all_time` to correct `metrics['1h']['mae']`/`metrics['all']['mae']` (method was non-functional, never fired)
- Drift detection direction — reversed from reducing confidence (slowing learning) to boosting confidence by +2.0 (accelerating re-adaptation)
- Prediction metrics persistence schema — restored established `mae_all_time` / `rmse_all_time` keys so unified state readers and HA export consume live values again
- Learning confidence reset path — boosted confidence is now clamped back to 5.0 when drift subsides and on restart, preventing stale drift-only boosts from persisting
- Startup sensor validation retry — transient validation failures no longer permanently disable the one-time startup check

### Changed
- Learning confidence clamp uses dynamic `_max_learning_confidence` attribute (default 5.0) instead of hardcoded 5.0, allowing drift detection to temporarily raise cap to 10.0
- Unified state defaults now include RMSE window fields and `last_10_count` so persisted prediction metrics match the live write schema

## [0.2.8] - 2026-04-23

### Added
- **Indoor Temperature Trend Bias in Trajectory Prediction**: `predict_thermal_trajectory()` now incorporates `indoor_temp_delta_60m` as a decaying momentum bias. Observed indoor temperature trend (°C over last 60 min) captures unmeasured heat sources (solar through windows, body heat, appliances, thermal mass) that the physics model cannot see. The bias uses exponential decay controlled by `TREND_DECAY_TAU_HOURS` (default 1.5h) so near-future predictions strongly reflect observed momentum while far-future predictions rely on physics.
  - New config variable `TREND_DECAY_TAU_HOURS` (default `1.5`, env-overridable) controls decay time constant.
  - Trend bias is clamped to ±0.05°C per step and gated on `abs(trend) > 0.01` to prevent floating-point noise.
  - Passed from both binary search optimization and trajectory verification callers in `model_wrapper.py`.
- **Binary search diagnostic logging**: Added debug logs on first iteration of binary search to show resolved `inlet_temp`, `delta_t_floor`, `indoor_temp_delta_60m`, optimization horizon, and trajectory result (steps, start→end temperatures). Helps verify the `_features` fix is working correctly in production.
- **Electricity Price-Aware Optimization**: Tibber-integrated price classification that shifts the binary search target temperature based on current electricity price relative to today's distribution.
  - `PriceOptimizer` class with percentile-based classification (CHEAP/NORMAL/EXPENSIVE) using daily price arrays from Tibber sensor.
  - CHEAP → target +0.2°C (heat more), EXPENSIVE → target −0.2°C (heat less), NORMAL → unchanged. Convergence precision stays at ±0.01°C.
  - Trajectory correction: EXPENSIVE tightens future overshoot threshold from +0.5°C to +0.2°C, preventing unnecessary heating during expensive hours.
  - Feature flag `ELECTRICITY_PRICE_ENABLED` (default: `false`) — zero behaviour change until explicitly enabled.
  - New module: `src/price_optimizer.py`.
  - New config variables: `ELECTRICITY_PRICE_ENTITY_ID`, `PRICE_CHEAP_PERCENTILE`, `PRICE_EXPENSIVE_PERCENTILE`, `PRICE_TARGET_OFFSET`, `PRICE_EXPENSIVE_OVERSHOOT`.
- **`sensor.ml_heating_features`**: New HA sensor exporting all last-run features as attributes for debugging and diagnostics.
- **`sensor.ml_heating_price_level`**: New HA sensor showing current price classification, thresholds, and target offset.
- **Enhanced `sensor.ml_heating_learning`**: Now exports ALL learnable parameters unconditionally (previously gated behind `ENABLE_HEAT_SOURCE_CHANNELS`), plus per-channel diagnostics (`ch_{name}_history_count`, `ch_{name}_last_error`).
- **29 unit tests** for price optimizer: classification, offsets, trajectory thresholds, feature flag, integration with binary search, singleton, edge cases.
- **Heat Source Channel Architecture (Phase 2-4)**: Decomposed heat-source learning with independent channels for heat pump, solar/PV, fireplace, and TV/electronics. Each channel has its own learnable parameters and prediction history, preventing cross-contamination of learned parameters.
  - `HeatSourceChannel` abstract base class with `estimate_heat_contribution()`, `estimate_decay_contribution()`, `get_learnable_parameters()`, and `apply_gradient_update()` methods. Channels self-learn via `_learn_from_recent()` triggered on each `record_learning()` call.
  - `HeatPumpChannel`: wraps existing slab model (outlet effectiveness, slab time constant, delta-T floor).
  - `SolarChannel`: forecast-aware PV heat estimation with cloud factor, solar lag, solar decay τ (0.5 h default for sun-warmed surface residual heat), and `predict_future_contribution()` with decay-smoothed evening transitions.
  - `FireplaceChannel`: exponential decay model after fireplace off (τ ~ 45 min) with room spread delay. **Learns independently** via gradient descent from prediction errors — no dependency on `adaptive_fireplace_learning.py`.
  - `TVChannel`: simple additive heat source for TV/electronics (~0.25 kW).
  - `HeatSourceChannelOrchestrator`: routes learning updates to correct channel, combines all channels for total heat prediction, proportional error attribution across active channels.
- **`ENABLE_HEAT_SOURCE_CHANNELS` config variable**: Enable/disable decomposed heat-source learning (default: `true`). Existing Phase 1 guards (fireplace, PV, pump-OFF) remain active independently.
- **Channel-isolated gradient descent (Phase 3)**: HP channel learns only from clean cycles (no fireplace, low PV); solar channel only from PV > 500 W; fireplace channel only when fireplace active.
- **Solar transition forecasting (Phase 4)**: `SolarChannel.predict_future_contribution()` uses PV forecast array to predict future solar heat per 10-min step, with exponential decay smoothing when PV drops (solar_decay_tau_hours). Enables proactive outlet temperature increase before sunset.
- **Orchestrator integration (Steps 10-11)**: `ThermalEquilibriumModel` initializes orchestrator when `ENABLE_HEAT_SOURCE_CHANNELS` is true and routes learning through it in `update_prediction_feedback()`.
- **New module**: `src/heat_source_channels.py` — 4 channel implementations + orchestrator.
- **Comprehensive scenario tests**: Evening (Step 17: PV 3000→0 W), morning (PV 0→3000 W with slab residual heat), solar decay τ, fireplace independent learning, and orchestrator integration — 36 tests total.
- **UFH Slab (Estrich) Thermal Model**: First-order lag between commanded outlet temperature and effective heating temperature: `T_slab(t+Δt) = T_slab(t) + Δt/τ_slab · (T_cmd − T_slab)`. `T_slab(0)` is initialised from `inlet_temp` (Rücklauf = current slab state). This prevents the trajectory model from applying a cold outlet command instantly to the room, which caused spurious `+15°C` corrections in cycles with PV-recovery paths.
- **`slab_time_constant_hours` as learnable parameter**: New adaptive parameter (default 1.0 h, bounds 0.25–4.0 h) using the same finite-difference gradient framework as all other parameters (`_calculate_parameter_gradient`). Gradient is non-zero only when `inlet_temp ≠ outlet_cmd` (transient phases), zero at equilibrium — correct physics.
- **`SLAB_TIME_CONSTANT_HOURS` config variable**: Overridable via environment variable, default `1.0`.
- **Test suite** (`tests/unit/test_slab_model.py`): 6 test classes covering slab dynamics (buffering, monotonicity, backward-compat), gradient observability (non-zero at disequilibrium, zero at equilibrium), parameter update/clipping, persistence of both new delta keys, and config bounds.
- **`slab_passive_delta` sensor**: New diagnostic metric (`inlet_temp - indoor_temp`) exported to HA. Positive = slab warmer than room (passive heating available), negative = slab absorbing heat. Visible in thermal features and HA sensor attributes.
- **`_search_delta_t_floor`**: Internal variable ensuring both binary search pre-check, loop, and trajectory verification use the same (potentially simulated) delta_t value per cycle.
- **`PV_CALIBRATION_INDOOR_CEILING` config variable**: Indoor temperature ceiling (default 23.0°C) for filtering blind-contaminated PV calibration periods. When automated blinds close (rooms ≥ 22.9°C), real solar heating drops 70–90% while the roof PV sensor still reads high — this causes the optimizer to push `pv_heat_weight` to its lower bound. Periods with `indoor_temp >= ceiling` are now excluded from PV Pass 2 calibration.
- **`LIVING_ROOM_TEMP_ENTITY_ID` in InfluxDB query**: Living room temperature sensor now included in the Flux query entity filter, ensuring indoor temperature data is available for calibration.

### Changed
- `predict_thermal_trajectory`: new optional `inlet_temp` parameter; when provided the slab model is active; when `None` the existing behaviour is preserved (backward-compatible).
- `_calculate_parameter_gradient`: passes `inlet_temp` from prediction context to both `+ε` and `−ε` trajectory evaluations.
- `unified_thermal_state.py`: `parameter_adjustments` default dict and `set_calibrated_baseline` reset dict extended with `solar_lag_minutes_delta` and `slab_time_constant_delta`; `update_learning_state` now accepts any key (no longer silently drops unknown delta keys).
- `physics_calibration.py`: `calibrated_params` now includes `slab_time_constant_hours` (preserved from current runtime value, not re-optimised — stable-period data cannot identify slab dynamics).
- `physics_calibration.py`: `_filter_pv_only_periods()` filters periods with `indoor_temp >= PV_CALIBRATION_INDOOR_CEILING` before PV Pass 2 scipy optimization.
- `physics_calibration.py`: `filter_pv_decay_periods()` uses 6-step sliding-window crossing detection instead of single-step sharp-drop filter.
- `heat_source_channels.py`: `SolarChannel._learn_from_recent()` and `apply_gradient_update()` skip `cloud_factor_exponent` updates when `CLOUD_COVER_CORRECTION_ENABLED=false`.
- `thermal_config.py`: `pv_heat_weight` default 0.0005 → 0.0002, bounds (0.0005, 0.005) → (0.0001, 0.005).
- `influx_service.py`: Flux query entity filter now includes `LIVING_ROOM_TEMP_ENTITY_ID`.
- `config.py`: Removed duplicate `LIVING_ROOM_TEMP_ENTITY_ID` definition.
- `inlet_temp` is now included in the `prediction_context` dict stored in `prediction_history`, enabling the slab gradient to be reconstructed from historical records.

### Fixed
- **Critical: Binary search `_features` NameError causing 35°C fallback**: `predict_thermal_trajectory` failed every binary search iteration with `name '_features' is not defined`, causing silent fallback to max outlet temperature (35°C). The gradual temperature control then capped/smoothed this down, masking the root cause but producing suboptimal heating decisions. Fixed by replacing bare `_features.get(...)` with `self._current_features.get(...)` using the safe `hasattr` guard pattern used elsewhere.
- **Dashboard `titlefont` crash**: Replaced deprecated Plotly `titlefont` → `title_font` in `dashboard/components/overview.py` confidence/error dual-axis chart. Plotly 6.x removed `titlefont`, causing `ValueError` on every dashboard load.
- **Noisy "Logging MAE"/"Logging RMSE" debug messages**: Removed unnecessary `logging.debug("Logging MAE")` and `logging.debug("Logging RMSE")` in `ha_client.py` that produced low-value noise every 10-minute cycle. The actual HA state updates and their results already provide sufficient logging.
- **Test `test_learning_isolation`**: Fixed `test_hp_params_update_when_no_contamination` to provide enough prediction feedback records (≥ `RECENT_ERRORS_WINDOW`) and use pump-ON context (`delta_t=5.0`) so gradient adaptation is actually triggered.
- **`solar_lag_minutes_delta` persistence**: `solar_lag_minutes` learning updates were accumulated in-memory but never persisted across restarts. Both `solar_lag_minutes_delta` and the new `slab_time_constant_delta` are now written to `unified_thermal_state.json` via `_save_learning_to_thermal_state`.
- **Control Stability:** Fixed "Deadbeat Control" oscillation by decoupling the control interval (30m) from the optimization horizon (4h). This prevents excessive outlet temperature spikes when correcting small deviations.
- **HP-off outlet spike (35°C)**: When heat pump is off (`delta_t < 1.0`), the binary search now simulates "HP on" using the learned `delta_t_floor` (~2.55°C) from the HP channel. Previously, all outlet candidates produced identical slab-passive trajectories → "unreachable" → outlet spiked to max 35°C pointlessly. The simulated HP-on delta_t lets candidates differentiate so the binary search converges to a sensible setpoint that tells NIBE when to start heating.
- **PV routing at sunset**: `_is_pv_active()` now uses `max(pv_power_current, pv_power_smoothed)` against `PV_LEARNING_THRESHOLD` (default 50, in watts). This captures solar thermal lag where smoothed PV stays high after instantaneous PV drops.
- **PV smoothing window**: Shortened from 3h (18 readings) to `solar_decay_tau` (~30min, 3 readings) in `temperature_control.py`. The old 3h window included stale morning PV values in the afternoon.
- **Slab pump-on gate**: Pump-ON branch now requires `measured_delta_t >= 1.0` in addition to `outlet_temp > t_slab`. Prevents slab model from entering active heating when HP is actually off (delta_t ≈ 0 but outlet reads higher than inlet due to stale setpoint).
- **Cloud discount on PV scalar**: Applied 1h cloud forecast discount to the PV scalar in `_extract_thermal_features()` before it enters the binary search. Raw sensor spikes during brief sun breaks (e.g. 4kW) no longer cause the binary search to snap outlet to 18°C, preventing 6am–11am outlet oscillation (21.8–24.9°C).
- **Blind-contaminated PV calibration data**: Root cause fix for `pv_heat_weight` stuck at lower bound. Automated blinds close when rooms > 22.9°C → real solar heating drops 70–90% → roof PV sensor still reads ~2000W → optimizer sees high PV with flat indoor temp → pushes weight to lower bound. `_filter_pv_only_periods()` now excludes periods where `indoor_temp >= PV_CALIBRATION_INDOOR_CEILING` (default 23.0°C).
- **`pv_heat_weight` bounds**: Default lowered from 0.0005 → 0.0002, lower bound from 0.0005 → 0.0001. Previous default = lower bound prevented the optimizer from exploring below the initial value.
- **`cloud_factor_exponent` learning when disabled**: Online learning (`SolarChannel._learn_from_recent()`) and batch calibration (`calibrate_cloud_factor()`) now gated behind `CLOUD_COVER_CORRECTION_ENABLED`. Previously both ran unconditionally — when the flag was `false`, the prediction path returned 1.0 but gradients still updated the exponent, causing parameter drift without feedback.
- **`calibrate_delta_t_floor` sensitivity**: Raised minimum delta_t threshold from 0.5 → 1.0°C and minimum calibration result from 0.5 → 1.0°C. Prevents floor from converging to sub-1°C values where HP is effectively off.
- **PV decay period detection**: Replaced single-step sharp-drop filter with 6-step (30 min) sliding-window crossing detection in `filter_pv_decay_periods()`. Old method required an exact single-step PV drop below threshold, missing gradual sunset transitions. New method detects when a 6-reading window crosses from above to below the PV threshold.
- **`cloud_cover_pct` calibration default**: Changed all 4 hardcoded fallback values from 50.0 → 0.0. When cloud cover data is unavailable, assuming clear sky (0%) is physically correct — the calibration should learn the actual heating at the measured PV power, not discount it by an assumed 50% cloud cover.

## [0.2.7] - 2026-04-18

### Added
- **Indoor Temperature Trend Bias in Trajectory Prediction**: `predict_thermal_trajectory()` now incorporates `indoor_temp_delta_60m` as a decaying momentum bias. Observed indoor temperature trend (°C over last 60 min) captures unmeasured heat sources (solar through windows, body heat, appliances, thermal mass) that the physics model cannot see. The bias uses exponential decay controlled by `TREND_DECAY_TAU_HOURS` (default 1.5h) so near-future predictions strongly reflect observed momentum while far-future predictions rely on physics.
  - New config variable `TREND_DECAY_TAU_HOURS` (default `1.5`, env-overridable) controls decay time constant.
  - Trend bias is clamped to ±0.05°C per step and gated on `abs(trend) > 0.01` to prevent floating-point noise.
  - Passed from both binary search optimization and trajectory verification callers in `model_wrapper.py`.
- **Binary search diagnostic logging**: Added debug logs on first iteration of binary search to show resolved `inlet_temp`, `delta_t_floor`, `indoor_temp_delta_60m`, optimization horizon, and trajectory result (steps, start→end temperatures). Helps verify the `_features` fix is working correctly in production.

### Fixed
- **Critical: Binary search `_features` NameError causing 35°C fallback**: `predict_thermal_trajectory` failed every binary search iteration with `name '_features' is not defined`, causing silent fallback to max outlet temperature (35°C). The gradual temperature control then capped/smoothed this down, masking the root cause but producing suboptimal heating decisions. Fixed by replacing bare `_features.get(...)` with `self._current_features.get(...)` using the safe `hasattr` guard pattern used elsewhere.
- **Dashboard `titlefont` crash**: Replaced deprecated Plotly `titlefont` → `title_font` in `dashboard/components/overview.py` confidence/error dual-axis chart. Plotly 6.x removed `titlefont`, causing `ValueError` on every dashboard load.
- **Noisy "Logging MAE"/"Logging RMSE" debug messages**: Removed unnecessary `logging.debug("Logging MAE")` and `logging.debug("Logging RMSE")` in `ha_client.py` that produced low-value noise every 10-minute cycle. The actual HA state updates and their results already provide sufficient logging.

### Added
- **Electricity Price-Aware Optimization**: Tibber-integrated price classification that shifts the binary search target temperature based on current electricity price relative to today's distribution.
  - `PriceOptimizer` class with percentile-based classification (CHEAP/NORMAL/EXPENSIVE) using daily price arrays from Tibber sensor.
  - CHEAP → target +0.2°C (heat more), EXPENSIVE → target −0.2°C (heat less), NORMAL → unchanged. Convergence precision stays at ±0.01°C.
  - Trajectory correction: EXPENSIVE tightens future overshoot threshold from +0.5°C to +0.2°C, preventing unnecessary heating during expensive hours.
  - Feature flag `ELECTRICITY_PRICE_ENABLED` (default: `false`) — zero behaviour change until explicitly enabled.
  - New module: `src/price_optimizer.py`.
  - New config variables: `ELECTRICITY_PRICE_ENTITY_ID`, `PRICE_CHEAP_PERCENTILE`, `PRICE_EXPENSIVE_PERCENTILE`, `PRICE_TARGET_OFFSET`, `PRICE_EXPENSIVE_OVERSHOOT`.
- **`sensor.ml_heating_features`**: New HA sensor exporting all last-run features as attributes for debugging and diagnostics.
- **`sensor.ml_heating_price_level`**: New HA sensor showing current price classification, thresholds, and target offset.
- **Enhanced `sensor.ml_heating_learning`**: Now exports ALL learnable parameters unconditionally (previously gated behind `ENABLE_HEAT_SOURCE_CHANNELS`), plus per-channel diagnostics (`ch_{name}_history_count`, `ch_{name}_last_error`).
- **29 unit tests** for price optimizer: classification, offsets, trajectory thresholds, feature flag, integration with binary search, singleton, edge cases.

### Added
- **Heat Source Channel Architecture (Phase 2-4)**: Decomposed heat-source learning with independent channels for heat pump, solar/PV, fireplace, and TV/electronics. Each channel has its own learnable parameters and prediction history, preventing cross-contamination of learned parameters.
  - `HeatSourceChannel` abstract base class with `estimate_heat_contribution()`, `estimate_decay_contribution()`, `get_learnable_parameters()`, and `apply_gradient_update()` methods. Channels self-learn via `_learn_from_recent()` triggered on each `record_learning()` call.
  - `HeatPumpChannel`: wraps existing slab model (outlet effectiveness, slab time constant, delta-T floor).
  - `SolarChannel`: forecast-aware PV heat estimation with cloud factor, solar lag, solar decay τ (0.5 h default for sun-warmed surface residual heat), and `predict_future_contribution()` with decay-smoothed evening transitions.
  - `FireplaceChannel`: exponential decay model after fireplace off (τ ~ 45 min) with room spread delay. **Learns independently** via gradient descent from prediction errors — no dependency on `adaptive_fireplace_learning.py`.
  - `TVChannel`: simple additive heat source for TV/electronics (~0.25 kW).
  - `HeatSourceChannelOrchestrator`: routes learning updates to correct channel, combines all channels for total heat prediction, proportional error attribution across active channels.
- **`ENABLE_HEAT_SOURCE_CHANNELS` config variable**: Enable/disable decomposed heat-source learning (default: `true`). Existing Phase 1 guards (fireplace, PV, pump-OFF) remain active independently.
- **Channel-isolated gradient descent (Phase 3)**: HP channel learns only from clean cycles (no fireplace, low PV); solar channel only from PV > 500 W; fireplace channel only when fireplace active.
- **Solar transition forecasting (Phase 4)**: `SolarChannel.predict_future_contribution()` uses PV forecast array to predict future solar heat per 10-min step, with exponential decay smoothing when PV drops (solar_decay_tau_hours). Enables proactive outlet temperature increase before sunset.
- **Orchestrator integration (Steps 10-11)**: `ThermalEquilibriumModel` initializes orchestrator when `ENABLE_HEAT_SOURCE_CHANNELS` is true and routes learning through it in `update_prediction_feedback()`.
- **New module**: `src/heat_source_channels.py` — 4 channel implementations + orchestrator.
- **Comprehensive scenario tests**: Evening (Step 17: PV 3000→0 W), morning (PV 0→3000 W with slab residual heat), solar decay τ, fireplace independent learning, and orchestrator integration — 36 tests total.

### Fixed
- **Test `test_learning_isolation`**: Fixed `test_hp_params_update_when_no_contamination` to provide enough prediction feedback records (≥ `RECENT_ERRORS_WINDOW`) and use pump-ON context (`delta_t=5.0`) so gradient adaptation is actually triggered.

- **UFH Slab (Estrich) Thermal Model**: First-order lag between commanded outlet temperature and effective heating temperature: `T_slab(t+Δt) = T_slab(t) + Δt/τ_slab · (T_cmd − T_slab)`. `T_slab(0)` is initialised from `inlet_temp` (Rücklauf = current slab state). This prevents the trajectory model from applying a cold outlet command instantly to the room, which caused spurious `+15°C` corrections in cycles with PV-recovery paths.
- **`slab_time_constant_hours` as learnable parameter**: New adaptive parameter (default 1.0 h, bounds 0.25–4.0 h) using the same finite-difference gradient framework as all other parameters (`_calculate_parameter_gradient`). Gradient is non-zero only when `inlet_temp ≠ outlet_cmd` (transient phases), zero at equilibrium — correct physics.
- **`solar_lag_minutes_delta` persistence fix**: `solar_lag_minutes` learning updates were accumulated in-memory but never persisted across restarts. Both `solar_lag_minutes_delta` and the new `slab_time_constant_delta` are now written to `unified_thermal_state.json` via `_save_learning_to_thermal_state`.
- **`inlet_temp` in prediction context**: `inlet_temp` (Rücklauf) is now included in the `prediction_context` dict stored in `prediction_history`, enabling the slab gradient to be reconstructed from historical records.
- **`SLAB_TIME_CONSTANT_HOURS` config variable**: Overridable via environment variable, default `1.0`.
- **Test suite** (`tests/unit/test_slab_model.py`): 6 test classes covering slab dynamics (buffering, monotonicity, backward-compat), gradient observability (non-zero at disequilibrium, zero at equilibrium), parameter update/clipping, persistence of both new delta keys, and config bounds.

### Changed
- `predict_thermal_trajectory`: new optional `inlet_temp` parameter; when provided the slab model is active; when `None` the existing behaviour is preserved (backward-compatible).
- `_calculate_parameter_gradient`: passes `inlet_temp` from prediction context to both `+ε` and `−ε` trajectory evaluations.
- `unified_thermal_state.py`: `parameter_adjustments` default dict and `set_calibrated_baseline` reset dict extended with `solar_lag_minutes_delta` and `slab_time_constant_delta`; `update_learning_state` now accepts any key (no longer silently drops unknown delta keys).
- `physics_calibration.py`: `calibrated_params` now includes `slab_time_constant_hours` (preserved from current runtime value, not re-optimised — stable-period data cannot identify slab dynamics).
- `physics_calibration.py`: `_filter_pv_only_periods()` filters periods with `indoor_temp >= PV_CALIBRATION_INDOOR_CEILING` before PV Pass 2 scipy optimization.
- `physics_calibration.py`: `filter_pv_decay_periods()` uses 6-step sliding-window crossing detection instead of single-step sharp-drop filter.
- `heat_source_channels.py`: `SolarChannel._learn_from_recent()` and `apply_gradient_update()` skip `cloud_factor_exponent` updates when `CLOUD_COVER_CORRECTION_ENABLED=false`.
- `thermal_config.py`: `pv_heat_weight` default 0.0005 → 0.0002, bounds (0.0005, 0.005) → (0.0001, 0.005).
- `influx_service.py`: Flux query entity filter now includes `LIVING_ROOM_TEMP_ENTITY_ID`.
- `config.py`: Removed duplicate `LIVING_ROOM_TEMP_ENTITY_ID` definition.

### Fixed
- **Control Stability:** Fixed "Deadbeat Control" oscillation by decoupling the control interval (30m) from the optimization horizon (4h). This prevents excessive outlet temperature spikes when correcting small deviations.
- **HP-off outlet spike (35°C)**: When heat pump is off (`delta_t < 1.0`), the binary search now simulates "HP on" using the learned `delta_t_floor` (~2.55°C) from the HP channel. Previously, all outlet candidates produced identical slab-passive trajectories → "unreachable" → outlet spiked to max 35°C pointlessly. The simulated HP-on delta_t lets candidates differentiate so the binary search converges to a sensible setpoint that tells NIBE when to start heating.
- **PV routing at sunset**: `_is_pv_active()` now uses `max(pv_power_current, pv_power_smoothed)` against the 500W threshold. This captures solar thermal lag where smoothed PV stays high after instantaneous PV drops.
- **PV smoothing window**: Shortened from 3h (18 readings) to `solar_decay_tau` (~30min, 3 readings) in `temperature_control.py`. The old 3h window included stale morning PV values in the afternoon.
- **Slab pump-on gate**: Pump-ON branch now requires `measured_delta_t >= 1.0` in addition to `outlet_temp > t_slab`. Prevents slab model from entering active heating when HP is actually off (delta_t ≈ 0 but outlet reads higher than inlet due to stale setpoint).
- **Cloud discount on PV scalar**: Applied 1h cloud forecast discount to the PV scalar in `_extract_thermal_features()` before it enters the binary search. Raw sensor spikes during brief sun breaks (e.g. 4kW) no longer cause the binary search to snap outlet to 18°C, preventing 6am–11am outlet oscillation (21.8–24.9°C).
- **Blind-contaminated PV calibration data**: Root cause fix for `pv_heat_weight` stuck at lower bound. Automated blinds close when rooms > 22.9°C → real solar heating drops 70–90% → roof PV sensor still reads ~2000W → optimizer sees high PV with flat indoor temp → pushes weight to lower bound. `_filter_pv_only_periods()` now excludes periods where `indoor_temp >= PV_CALIBRATION_INDOOR_CEILING` (default 23.0°C).
- **`pv_heat_weight` bounds**: Default lowered from 0.0005 → 0.0002, lower bound from 0.0005 → 0.0001. Previous default = lower bound prevented the optimizer from exploring below the initial value.
- **`cloud_factor_exponent` learning when disabled**: Online learning (`SolarChannel._learn_from_recent()`) and batch calibration (`calibrate_cloud_factor()`) now gated behind `CLOUD_COVER_CORRECTION_ENABLED`. Previously both ran unconditionally — when the flag was `false`, the prediction path returned 1.0 but gradients still updated the exponent, causing parameter drift without feedback.
- **`calibrate_delta_t_floor` sensitivity**: Raised minimum delta_t threshold from 0.5 → 1.0°C and minimum calibration result from 0.5 → 1.0°C. Prevents floor from converging to sub-1°C values where HP is effectively off.
- **PV decay period detection**: Replaced single-step sharp-drop filter with 6-step (30 min) sliding-window crossing detection in `filter_pv_decay_periods()`. Old method required an exact single-step PV drop below threshold, missing gradual sunset transitions. New method detects when a 6-reading window crosses from above to below the PV threshold.
- **`cloud_cover_pct` calibration default**: Changed all 4 hardcoded fallback values from 50.0 → 0.0. When cloud cover data is unavailable, assuming clear sky (0%) is physically correct — the calibration should learn the actual heating at the measured PV power, not discount it by an assumed 50% cloud cover.

### Added
- **`slab_passive_delta` sensor**: New diagnostic metric (`inlet_temp - indoor_temp`) exported to HA. Positive = slab warmer than room (passive heating available), negative = slab absorbing heat. Visible in thermal features and HA sensor attributes.
- **`_search_delta_t_floor`**: Internal variable ensuring both binary search pre-check, loop, and trajectory verification use the same (potentially simulated) delta_t value per cycle.
- **`PV_CALIBRATION_INDOOR_CEILING` config variable**: Indoor temperature ceiling (default 23.0°C) for filtering blind-contaminated PV calibration periods. When automated blinds close (rooms ≥ 22.9°C), real solar heating drops 70–90% while the roof PV sensor still reads high — this causes the optimizer to push `pv_heat_weight` to its lower bound. Periods with `indoor_temp >= ceiling` are now excluded from PV Pass 2 calibration.
- **`LIVING_ROOM_TEMP_ENTITY_ID` in InfluxDB query**: Living room temperature sensor now included in the Flux query entity filter, ensuring indoor temperature data is available for calibration.

### Technical Achievements

## [0.2.6] - 2026-04-18

### Added
- **Indoor Temperature Trend Bias in Trajectory Prediction**: `predict_thermal_trajectory()` now incorporates `indoor_temp_delta_60m` as a decaying momentum bias. Observed indoor temperature trend (°C over last 60 min) captures unmeasured heat sources (solar through windows, body heat, appliances, thermal mass) that the physics model cannot see. The bias uses exponential decay controlled by `TREND_DECAY_TAU_HOURS` (default 1.5h) so near-future predictions strongly reflect observed momentum while far-future predictions rely on physics.
  - New config variable `TREND_DECAY_TAU_HOURS` (default `1.5`, env-overridable) controls decay time constant.
  - Trend bias is clamped to ±0.05°C per step and gated on `abs(trend) > 0.01` to prevent floating-point noise.
  - Passed from both binary search optimization and trajectory verification callers in `model_wrapper.py`.
- **Binary search diagnostic logging**: Added debug logs on first iteration of binary search to show resolved `inlet_temp`, `delta_t_floor`, `indoor_temp_delta_60m`, optimization horizon, and trajectory result (steps, start→end temperatures). Helps verify the `_features` fix is working correctly in production.

### Fixed
- **Critical: Binary search `_features` NameError causing 35°C fallback**: `predict_thermal_trajectory` failed every binary search iteration with `name '_features' is not defined`, causing silent fallback to max outlet temperature (35°C). The gradual temperature control then capped/smoothed this down, masking the root cause but producing suboptimal heating decisions. Fixed by replacing bare `_features.get(...)` with `self._current_features.get(...)` using the safe `hasattr` guard pattern used elsewhere.
- **Dashboard `titlefont` crash**: Replaced deprecated Plotly `titlefont` → `title_font` in `dashboard/components/overview.py` confidence/error dual-axis chart. Plotly 6.x removed `titlefont`, causing `ValueError` on every dashboard load.
- **Noisy "Logging MAE"/"Logging RMSE" debug messages**: Removed unnecessary `logging.debug("Logging MAE")` and `logging.debug("Logging RMSE")` in `ha_client.py` that produced low-value noise every 10-minute cycle. The actual HA state updates and their results already provide sufficient logging.

### Added
- **Electricity Price-Aware Optimization**: Tibber-integrated price classification that shifts the binary search target temperature based on current electricity price relative to today's distribution.
  - `PriceOptimizer` class with percentile-based classification (CHEAP/NORMAL/EXPENSIVE) using daily price arrays from Tibber sensor.
  - CHEAP → target +0.2°C (heat more), EXPENSIVE → target −0.2°C (heat less), NORMAL → unchanged. Convergence precision stays at ±0.01°C.
  - Trajectory correction: EXPENSIVE tightens future overshoot threshold from +0.5°C to +0.2°C, preventing unnecessary heating during expensive hours.
  - Feature flag `ELECTRICITY_PRICE_ENABLED` (default: `false`) — zero behaviour change until explicitly enabled.
  - New module: `src/price_optimizer.py`.
  - New config variables: `ELECTRICITY_PRICE_ENTITY_ID`, `PRICE_CHEAP_PERCENTILE`, `PRICE_EXPENSIVE_PERCENTILE`, `PRICE_TARGET_OFFSET`, `PRICE_EXPENSIVE_OVERSHOOT`.
- **`sensor.ml_heating_features`**: New HA sensor exporting all last-run features as attributes for debugging and diagnostics.
- **`sensor.ml_heating_price_level`**: New HA sensor showing current price classification, thresholds, and target offset.
- **Enhanced `sensor.ml_heating_learning`**: Now exports ALL learnable parameters unconditionally (previously gated behind `ENABLE_HEAT_SOURCE_CHANNELS`), plus per-channel diagnostics (`ch_{name}_history_count`, `ch_{name}_last_error`).
- **29 unit tests** for price optimizer: classification, offsets, trajectory thresholds, feature flag, integration with binary search, singleton, edge cases.

### Added
- **Heat Source Channel Architecture (Phase 2-4)**: Decomposed heat-source learning with independent channels for heat pump, solar/PV, fireplace, and TV/electronics. Each channel has its own learnable parameters and prediction history, preventing cross-contamination of learned parameters.
  - `HeatSourceChannel` abstract base class with `estimate_heat_contribution()`, `estimate_decay_contribution()`, `get_learnable_parameters()`, and `apply_gradient_update()` methods. Channels self-learn via `_learn_from_recent()` triggered on each `record_learning()` call.
  - `HeatPumpChannel`: wraps existing slab model (outlet effectiveness, slab time constant, delta-T floor).
  - `SolarChannel`: forecast-aware PV heat estimation with cloud factor, solar lag, solar decay τ (0.5 h default for sun-warmed surface residual heat), and `predict_future_contribution()` with decay-smoothed evening transitions.
  - `FireplaceChannel`: exponential decay model after fireplace off (τ ~ 45 min) with room spread delay. **Learns independently** via gradient descent from prediction errors — no dependency on `adaptive_fireplace_learning.py`.
  - `TVChannel`: simple additive heat source for TV/electronics (~0.25 kW).
  - `HeatSourceChannelOrchestrator`: routes learning updates to correct channel, combines all channels for total heat prediction, proportional error attribution across active channels.
- **`ENABLE_HEAT_SOURCE_CHANNELS` config variable**: Enable/disable decomposed heat-source learning (default: `true`). Existing Phase 1 guards (fireplace, PV, pump-OFF) remain active independently.
- **Channel-isolated gradient descent (Phase 3)**: HP channel learns only from clean cycles (no fireplace, low PV); solar channel only from PV > 500 W; fireplace channel only when fireplace active.
- **Solar transition forecasting (Phase 4)**: `SolarChannel.predict_future_contribution()` uses PV forecast array to predict future solar heat per 10-min step, with exponential decay smoothing when PV drops (solar_decay_tau_hours). Enables proactive outlet temperature increase before sunset.
- **Orchestrator integration (Steps 10-11)**: `ThermalEquilibriumModel` initializes orchestrator when `ENABLE_HEAT_SOURCE_CHANNELS` is true and routes learning through it in `update_prediction_feedback()`.
- **New module**: `src/heat_source_channels.py` — 4 channel implementations + orchestrator.
- **Comprehensive scenario tests**: Evening (Step 17: PV 3000→0 W), morning (PV 0→3000 W with slab residual heat), solar decay τ, fireplace independent learning, and orchestrator integration — 36 tests total.

### Fixed
- **Test `test_learning_isolation`**: Fixed `test_hp_params_update_when_no_contamination` to provide enough prediction feedback records (≥ `RECENT_ERRORS_WINDOW`) and use pump-ON context (`delta_t=5.0`) so gradient adaptation is actually triggered.

- **UFH Slab (Estrich) Thermal Model**: First-order lag between commanded outlet temperature and effective heating temperature: `T_slab(t+Δt) = T_slab(t) + Δt/τ_slab · (T_cmd − T_slab)`. `T_slab(0)` is initialised from `inlet_temp` (Rücklauf = current slab state). This prevents the trajectory model from applying a cold outlet command instantly to the room, which caused spurious `+15°C` corrections in cycles with PV-recovery paths.
- **`slab_time_constant_hours` as learnable parameter**: New adaptive parameter (default 1.0 h, bounds 0.25–4.0 h) using the same finite-difference gradient framework as all other parameters (`_calculate_parameter_gradient`). Gradient is non-zero only when `inlet_temp ≠ outlet_cmd` (transient phases), zero at equilibrium — correct physics.
- **`solar_lag_minutes_delta` persistence fix**: `solar_lag_minutes` learning updates were accumulated in-memory but never persisted across restarts. Both `solar_lag_minutes_delta` and the new `slab_time_constant_delta` are now written to `unified_thermal_state.json` via `_save_learning_to_thermal_state`.
- **`inlet_temp` in prediction context**: `inlet_temp` (Rücklauf) is now included in the `prediction_context` dict stored in `prediction_history`, enabling the slab gradient to be reconstructed from historical records.
- **`SLAB_TIME_CONSTANT_HOURS` config variable**: Overridable via environment variable, default `1.0`.
- **Test suite** (`tests/unit/test_slab_model.py`): 6 test classes covering slab dynamics (buffering, monotonicity, backward-compat), gradient observability (non-zero at disequilibrium, zero at equilibrium), parameter update/clipping, persistence of both new delta keys, and config bounds.

### Changed
- `predict_thermal_trajectory`: new optional `inlet_temp` parameter; when provided the slab model is active; when `None` the existing behaviour is preserved (backward-compatible).
- `_calculate_parameter_gradient`: passes `inlet_temp` from prediction context to both `+ε` and `−ε` trajectory evaluations.
- `unified_thermal_state.py`: `parameter_adjustments` default dict and `set_calibrated_baseline` reset dict extended with `solar_lag_minutes_delta` and `slab_time_constant_delta`; `update_learning_state` now accepts any key (no longer silently drops unknown delta keys).
- `physics_calibration.py`: `calibrated_params` now includes `slab_time_constant_hours` (preserved from current runtime value, not re-optimised — stable-period data cannot identify slab dynamics).
- `physics_calibration.py`: `_filter_pv_only_periods()` filters periods with `indoor_temp >= PV_CALIBRATION_INDOOR_CEILING` before PV Pass 2 scipy optimization.
- `physics_calibration.py`: `filter_pv_decay_periods()` uses 6-step sliding-window crossing detection instead of single-step sharp-drop filter.
- `heat_source_channels.py`: `SolarChannel._learn_from_recent()` and `apply_gradient_update()` skip `cloud_factor_exponent` updates when `CLOUD_COVER_CORRECTION_ENABLED=false`.
- `thermal_config.py`: `pv_heat_weight` default 0.0005 → 0.0002, bounds (0.0005, 0.005) → (0.0001, 0.005).
- `influx_service.py`: Flux query entity filter now includes `LIVING_ROOM_TEMP_ENTITY_ID`.
- `config.py`: Removed duplicate `LIVING_ROOM_TEMP_ENTITY_ID` definition.

### Fixed
- **Control Stability:** Fixed "Deadbeat Control" oscillation by decoupling the control interval (30m) from the optimization horizon (4h). This prevents excessive outlet temperature spikes when correcting small deviations.
- **HP-off outlet spike (35°C)**: When heat pump is off (`delta_t < 1.0`), the binary search now simulates "HP on" using the learned `delta_t_floor` (~2.55°C) from the HP channel. Previously, all outlet candidates produced identical slab-passive trajectories → "unreachable" → outlet spiked to max 35°C pointlessly. The simulated HP-on delta_t lets candidates differentiate so the binary search converges to a sensible setpoint that tells NIBE when to start heating.
- **PV routing at sunset**: `_is_pv_active()` now uses `max(pv_power_current, pv_power_smoothed)` against the 500W threshold. This captures solar thermal lag where smoothed PV stays high after instantaneous PV drops.
- **PV smoothing window**: Shortened from 3h (18 readings) to `solar_decay_tau` (~30min, 3 readings) in `temperature_control.py`. The old 3h window included stale morning PV values in the afternoon.
- **Slab pump-on gate**: Pump-ON branch now requires `measured_delta_t >= 1.0` in addition to `outlet_temp > t_slab`. Prevents slab model from entering active heating when HP is actually off (delta_t ≈ 0 but outlet reads higher than inlet due to stale setpoint).
- **Cloud discount on PV scalar**: Applied 1h cloud forecast discount to the PV scalar in `_extract_thermal_features()` before it enters the binary search. Raw sensor spikes during brief sun breaks (e.g. 4kW) no longer cause the binary search to snap outlet to 18°C, preventing 6am–11am outlet oscillation (21.8–24.9°C).
- **Blind-contaminated PV calibration data**: Root cause fix for `pv_heat_weight` stuck at lower bound. Automated blinds close when rooms > 22.9°C → real solar heating drops 70–90% → roof PV sensor still reads ~2000W → optimizer sees high PV with flat indoor temp → pushes weight to lower bound. `_filter_pv_only_periods()` now excludes periods where `indoor_temp >= PV_CALIBRATION_INDOOR_CEILING` (default 23.0°C).
- **`pv_heat_weight` bounds**: Default lowered from 0.0005 → 0.0002, lower bound from 0.0005 → 0.0001. Previous default = lower bound prevented the optimizer from exploring below the initial value.
- **`cloud_factor_exponent` learning when disabled**: Online learning (`SolarChannel._learn_from_recent()`) and batch calibration (`calibrate_cloud_factor()`) now gated behind `CLOUD_COVER_CORRECTION_ENABLED`. Previously both ran unconditionally — when the flag was `false`, the prediction path returned 1.0 but gradients still updated the exponent, causing parameter drift without feedback.
- **`calibrate_delta_t_floor` sensitivity**: Raised minimum delta_t threshold from 0.5 → 1.0°C and minimum calibration result from 0.5 → 1.0°C. Prevents floor from converging to sub-1°C values where HP is effectively off.
- **PV decay period detection**: Replaced single-step sharp-drop filter with 6-step (30 min) sliding-window crossing detection in `filter_pv_decay_periods()`. Old method required an exact single-step PV drop below threshold, missing gradual sunset transitions. New method detects when a 6-reading window crosses from above to below the PV threshold.
- **`cloud_cover_pct` calibration default**: Changed all 4 hardcoded fallback values from 50.0 → 0.0. When cloud cover data is unavailable, assuming clear sky (0%) is physically correct — the calibration should learn the actual heating at the measured PV power, not discount it by an assumed 50% cloud cover.

### Added
- **`slab_passive_delta` sensor**: New diagnostic metric (`inlet_temp - indoor_temp`) exported to HA. Positive = slab warmer than room (passive heating available), negative = slab absorbing heat. Visible in thermal features and HA sensor attributes.
- **`_search_delta_t_floor`**: Internal variable ensuring both binary search pre-check, loop, and trajectory verification use the same (potentially simulated) delta_t value per cycle.
- **`PV_CALIBRATION_INDOOR_CEILING` config variable**: Indoor temperature ceiling (default 23.0°C) for filtering blind-contaminated PV calibration periods. When automated blinds close (rooms ≥ 22.9°C), real solar heating drops 70–90% while the roof PV sensor still reads high — this causes the optimizer to push `pv_heat_weight` to its lower bound. Periods with `indoor_temp >= ceiling` are now excluded from PV Pass 2 calibration.
- **`LIVING_ROOM_TEMP_ENTITY_ID` in InfluxDB query**: Living room temperature sensor now included in the Flux query entity filter, ensuring indoor temperature data is available for calibration.

### Technical Achievements

## [0.2.5] - 2026-04-17

### Added
- **Electricity Price-Aware Optimization**: Tibber-integrated price classification that shifts the binary search target temperature based on current electricity price relative to today's distribution.
  - `PriceOptimizer` class with percentile-based classification (CHEAP/NORMAL/EXPENSIVE) using daily price arrays from Tibber sensor.
  - CHEAP → target +0.2°C (heat more), EXPENSIVE → target −0.2°C (heat less), NORMAL → unchanged. Convergence precision stays at ±0.01°C.
  - Trajectory correction: EXPENSIVE tightens future overshoot threshold from +0.5°C to +0.2°C, preventing unnecessary heating during expensive hours.
  - Feature flag `ELECTRICITY_PRICE_ENABLED` (default: `false`) — zero behaviour change until explicitly enabled.
  - New module: `src/price_optimizer.py`.
  - New config variables: `ELECTRICITY_PRICE_ENTITY_ID`, `PRICE_CHEAP_PERCENTILE`, `PRICE_EXPENSIVE_PERCENTILE`, `PRICE_TARGET_OFFSET`, `PRICE_EXPENSIVE_OVERSHOOT`.
- **`sensor.ml_heating_features`**: New HA sensor exporting all last-run features as attributes for debugging and diagnostics.
- **`sensor.ml_heating_price_level`**: New HA sensor showing current price classification, thresholds, and target offset.
- **Enhanced `sensor.ml_heating_learning`**: Now exports ALL learnable parameters unconditionally (previously gated behind `ENABLE_HEAT_SOURCE_CHANNELS`), plus per-channel diagnostics (`ch_{name}_history_count`, `ch_{name}_last_error`).
- **29 unit tests** for price optimizer: classification, offsets, trajectory thresholds, feature flag, integration with binary search, singleton, edge cases.

### Added
- **Heat Source Channel Architecture (Phase 2-4)**: Decomposed heat-source learning with independent channels for heat pump, solar/PV, fireplace, and TV/electronics. Each channel has its own learnable parameters and prediction history, preventing cross-contamination of learned parameters.
  - `HeatSourceChannel` abstract base class with `estimate_heat_contribution()`, `estimate_decay_contribution()`, `get_learnable_parameters()`, and `apply_gradient_update()` methods. Channels self-learn via `_learn_from_recent()` triggered on each `record_learning()` call.
  - `HeatPumpChannel`: wraps existing slab model (outlet effectiveness, slab time constant, delta-T floor).
  - `SolarChannel`: forecast-aware PV heat estimation with cloud factor, solar lag, solar decay τ (0.5 h default for sun-warmed surface residual heat), and `predict_future_contribution()` with decay-smoothed evening transitions.
  - `FireplaceChannel`: exponential decay model after fireplace off (τ ~ 45 min) with room spread delay. **Learns independently** via gradient descent from prediction errors — no dependency on `adaptive_fireplace_learning.py`.
  - `TVChannel`: simple additive heat source for TV/electronics (~0.25 kW).
  - `HeatSourceChannelOrchestrator`: routes learning updates to correct channel, combines all channels for total heat prediction, proportional error attribution across active channels.
- **`ENABLE_HEAT_SOURCE_CHANNELS` config variable**: Enable/disable decomposed heat-source learning (default: `true`). Existing Phase 1 guards (fireplace, PV, pump-OFF) remain active independently.
- **Channel-isolated gradient descent (Phase 3)**: HP channel learns only from clean cycles (no fireplace, low PV); solar channel only from PV > 500 W; fireplace channel only when fireplace active.
- **Solar transition forecasting (Phase 4)**: `SolarChannel.predict_future_contribution()` uses PV forecast array to predict future solar heat per 10-min step, with exponential decay smoothing when PV drops (solar_decay_tau_hours). Enables proactive outlet temperature increase before sunset.
- **Orchestrator integration (Steps 10-11)**: `ThermalEquilibriumModel` initializes orchestrator when `ENABLE_HEAT_SOURCE_CHANNELS` is true and routes learning through it in `update_prediction_feedback()`.
- **New module**: `src/heat_source_channels.py` — 4 channel implementations + orchestrator.
- **Comprehensive scenario tests**: Evening (Step 17: PV 3000→0 W), morning (PV 0→3000 W with slab residual heat), solar decay τ, fireplace independent learning, and orchestrator integration — 36 tests total.

### Fixed
- **Test `test_learning_isolation`**: Fixed `test_hp_params_update_when_no_contamination` to provide enough prediction feedback records (≥ `RECENT_ERRORS_WINDOW`) and use pump-ON context (`delta_t=5.0`) so gradient adaptation is actually triggered.

- **UFH Slab (Estrich) Thermal Model**: First-order lag between commanded outlet temperature and effective heating temperature: `T_slab(t+Δt) = T_slab(t) + Δt/τ_slab · (T_cmd − T_slab)`. `T_slab(0)` is initialised from `inlet_temp` (Rücklauf = current slab state). This prevents the trajectory model from applying a cold outlet command instantly to the room, which caused spurious `+15°C` corrections in cycles with PV-recovery paths.
- **`slab_time_constant_hours` as learnable parameter**: New adaptive parameter (default 1.0 h, bounds 0.25–4.0 h) using the same finite-difference gradient framework as all other parameters (`_calculate_parameter_gradient`). Gradient is non-zero only when `inlet_temp ≠ outlet_cmd` (transient phases), zero at equilibrium — correct physics.
- **`solar_lag_minutes_delta` persistence fix**: `solar_lag_minutes` learning updates were accumulated in-memory but never persisted across restarts. Both `solar_lag_minutes_delta` and the new `slab_time_constant_delta` are now written to `unified_thermal_state.json` via `_save_learning_to_thermal_state`.
- **`inlet_temp` in prediction context**: `inlet_temp` (Rücklauf) is now included in the `prediction_context` dict stored in `prediction_history`, enabling the slab gradient to be reconstructed from historical records.
- **`SLAB_TIME_CONSTANT_HOURS` config variable**: Overridable via environment variable, default `1.0`.
- **Test suite** (`tests/unit/test_slab_model.py`): 6 test classes covering slab dynamics (buffering, monotonicity, backward-compat), gradient observability (non-zero at disequilibrium, zero at equilibrium), parameter update/clipping, persistence of both new delta keys, and config bounds.

### Changed
- `predict_thermal_trajectory`: new optional `inlet_temp` parameter; when provided the slab model is active; when `None` the existing behaviour is preserved (backward-compatible).
- `_calculate_parameter_gradient`: passes `inlet_temp` from prediction context to both `+ε` and `−ε` trajectory evaluations.
- `unified_thermal_state.py`: `parameter_adjustments` default dict and `set_calibrated_baseline` reset dict extended with `solar_lag_minutes_delta` and `slab_time_constant_delta`; `update_learning_state` now accepts any key (no longer silently drops unknown delta keys).
- `physics_calibration.py`: `calibrated_params` now includes `slab_time_constant_hours` (preserved from current runtime value, not re-optimised — stable-period data cannot identify slab dynamics).
- `physics_calibration.py`: `_filter_pv_only_periods()` filters periods with `indoor_temp >= PV_CALIBRATION_INDOOR_CEILING` before PV Pass 2 scipy optimization.
- `physics_calibration.py`: `filter_pv_decay_periods()` uses 6-step sliding-window crossing detection instead of single-step sharp-drop filter.
- `heat_source_channels.py`: `SolarChannel._learn_from_recent()` and `apply_gradient_update()` skip `cloud_factor_exponent` updates when `CLOUD_COVER_CORRECTION_ENABLED=false`.
- `thermal_config.py`: `pv_heat_weight` default 0.0005 → 0.0002, bounds (0.0005, 0.005) → (0.0001, 0.005).
- `influx_service.py`: Flux query entity filter now includes `LIVING_ROOM_TEMP_ENTITY_ID`.
- `config.py`: Removed duplicate `LIVING_ROOM_TEMP_ENTITY_ID` definition.

### Fixed
- **Control Stability:** Fixed "Deadbeat Control" oscillation by decoupling the control interval (30m) from the optimization horizon (4h). This prevents excessive outlet temperature spikes when correcting small deviations.
- **HP-off outlet spike (35°C)**: When heat pump is off (`delta_t < 1.0`), the binary search now simulates "HP on" using the learned `delta_t_floor` (~2.55°C) from the HP channel. Previously, all outlet candidates produced identical slab-passive trajectories → "unreachable" → outlet spiked to max 35°C pointlessly. The simulated HP-on delta_t lets candidates differentiate so the binary search converges to a sensible setpoint that tells NIBE when to start heating.
- **PV routing at sunset**: `_is_pv_active()` now uses `max(pv_power_current, pv_power_smoothed)` against the 500W threshold. This captures solar thermal lag where smoothed PV stays high after instantaneous PV drops.
- **PV smoothing window**: Shortened from 3h (18 readings) to `solar_decay_tau` (~30min, 3 readings) in `temperature_control.py`. The old 3h window included stale morning PV values in the afternoon.
- **Slab pump-on gate**: Pump-ON branch now requires `measured_delta_t >= 1.0` in addition to `outlet_temp > t_slab`. Prevents slab model from entering active heating when HP is actually off (delta_t ≈ 0 but outlet reads higher than inlet due to stale setpoint).
- **Cloud discount on PV scalar**: Applied 1h cloud forecast discount to the PV scalar in `_extract_thermal_features()` before it enters the binary search. Raw sensor spikes during brief sun breaks (e.g. 4kW) no longer cause the binary search to snap outlet to 18°C, preventing 6am–11am outlet oscillation (21.8–24.9°C).
- **Blind-contaminated PV calibration data**: Root cause fix for `pv_heat_weight` stuck at lower bound. Automated blinds close when rooms > 22.9°C → real solar heating drops 70–90% → roof PV sensor still reads ~2000W → optimizer sees high PV with flat indoor temp → pushes weight to lower bound. `_filter_pv_only_periods()` now excludes periods where `indoor_temp >= PV_CALIBRATION_INDOOR_CEILING` (default 23.0°C).
- **`pv_heat_weight` bounds**: Default lowered from 0.0005 → 0.0002, lower bound from 0.0005 → 0.0001. Previous default = lower bound prevented the optimizer from exploring below the initial value.
- **`cloud_factor_exponent` learning when disabled**: Online learning (`SolarChannel._learn_from_recent()`) and batch calibration (`calibrate_cloud_factor()`) now gated behind `CLOUD_COVER_CORRECTION_ENABLED`. Previously both ran unconditionally — when the flag was `false`, the prediction path returned 1.0 but gradients still updated the exponent, causing parameter drift without feedback.
- **`calibrate_delta_t_floor` sensitivity**: Raised minimum delta_t threshold from 0.5 → 1.0°C and minimum calibration result from 0.5 → 1.0°C. Prevents floor from converging to sub-1°C values where HP is effectively off.
- **PV decay period detection**: Replaced single-step sharp-drop filter with 6-step (30 min) sliding-window crossing detection in `filter_pv_decay_periods()`. Old method required an exact single-step PV drop below threshold, missing gradual sunset transitions. New method detects when a 6-reading window crosses from above to below the PV threshold.
- **`cloud_cover_pct` calibration default**: Changed all 4 hardcoded fallback values from 50.0 → 0.0. When cloud cover data is unavailable, assuming clear sky (0%) is physically correct — the calibration should learn the actual heating at the measured PV power, not discount it by an assumed 50% cloud cover.

### Added
- **`slab_passive_delta` sensor**: New diagnostic metric (`inlet_temp - indoor_temp`) exported to HA. Positive = slab warmer than room (passive heating available), negative = slab absorbing heat. Visible in thermal features and HA sensor attributes.
- **`_search_delta_t_floor`**: Internal variable ensuring both binary search pre-check, loop, and trajectory verification use the same (potentially simulated) delta_t value per cycle.
- **`PV_CALIBRATION_INDOOR_CEILING` config variable**: Indoor temperature ceiling (default 23.0°C) for filtering blind-contaminated PV calibration periods. When automated blinds close (rooms ≥ 22.9°C), real solar heating drops 70–90% while the roof PV sensor still reads high — this causes the optimizer to push `pv_heat_weight` to its lower bound. Periods with `indoor_temp >= ceiling` are now excluded from PV Pass 2 calibration.
- **`LIVING_ROOM_TEMP_ENTITY_ID` in InfluxDB query**: Living room temperature sensor now included in the Flux query entity filter, ensuring indoor temperature data is available for calibration.

### Technical Achievements

## [0.2.4] - 2026-04-17

### Added
- **Electricity Price-Aware Optimization**: Tibber-integrated price classification (CHEAP/NORMAL/EXPENSIVE) that shifts the binary search target temperature based on current electricity price relative to today's distribution. Feature flag `ELECTRICITY_PRICE_ENABLED` (default: `false`) — zero behaviour change until explicitly enabled.
- **`sensor.ml_heating_features`**: New HA sensor exporting all last-run features as attributes for debugging and diagnostics.
- **`sensor.ml_heating_price_level`**: New HA sensor showing current price classification, thresholds, and target offset.
- **Enhanced `sensor.ml_heating_learning`**: Now exports ALL learnable parameters unconditionally, plus per-channel diagnostics.
- **Heat Source Channel Architecture**: Decomposed heat-source learning with independent channels for heat pump, solar/PV, fireplace, and TV/electronics. Each channel has its own learnable parameters and prediction history.
- **`ENABLE_HEAT_SOURCE_CHANNELS` config variable**: Enable/disable decomposed heat-source learning (default: `true`).
- **Solar transition forecasting**: `SolarChannel.predict_future_contribution()` uses PV forecast array to predict future solar heat per 10-min step with exponential decay smoothing at sunset.
- **UFH Slab (Estrich) Thermal Model**: First-order lag between commanded outlet temperature and effective heating temperature, preventing spurious `+15°C` trajectory corrections.
- **`slab_time_constant_hours` as learnable parameter**: New adaptive parameter (default 1.0 h, bounds 0.25–4.0 h).
- **`slab_passive_delta` sensor**: New diagnostic metric (`inlet_temp - indoor_temp`) exported to HA.
- **`PV_CALIBRATION_INDOOR_CEILING` config variable**: Indoor temperature ceiling (default 23.0°C) for filtering blind-contaminated PV calibration periods.
- **`LIVING_ROOM_TEMP_ENTITY_ID` in InfluxDB query**: Living room temperature now included in entity filter for calibration.
- **New config variables**: `ELECTRICITY_PRICE_ENTITY_ID`, `PRICE_CHEAP_PERCENTILE`, `PRICE_EXPENSIVE_PERCENTILE`, `PRICE_TARGET_OFFSET`, `PRICE_EXPENSIVE_OVERSHOOT`, `SLAB_TIME_CONSTANT_HOURS`.

### Fixed
- **Control Stability**: Fixed "Deadbeat Control" oscillation by decoupling the control interval (30m) from the optimization horizon (4h).
- **HP-off outlet spike (35°C)**: Binary search now simulates "HP on" using the learned `delta_t_floor` when heat pump is off, preventing pointless 35°C setpoints.
- **PV routing at sunset**: `_is_pv_active()` now uses `max(pv_power_current, pv_power_smoothed)` to capture solar thermal lag.
- **PV smoothing window**: Shortened from 3h to `solar_decay_tau` (~30min), eliminating stale morning PV values in the afternoon.
- **Blind-contaminated PV calibration data**: Root cause fix for `pv_heat_weight` stuck at lower bound — periods with indoor temp above ceiling are now excluded from PV calibration.
- **`pv_heat_weight` bounds**: Default lowered from 0.0005 → 0.0002; lower bound from 0.0005 → 0.0001.
- **`cloud_factor_exponent` learning when disabled**: Online learning and batch calibration now correctly gated behind `CLOUD_COVER_CORRECTION_ENABLED`.
- **`solar_lag_minutes_delta` persistence**: Learning updates now persisted across restarts in `unified_thermal_state.json`.
- **`calibrate_delta_t_floor` sensitivity**: Raised minimum delta_t threshold from 0.5 → 1.0°C to prevent convergence to sub-1°C values.
- **PV decay period detection**: Replaced single-step filter with 6-step sliding-window crossing detection for gradual sunset transitions.

### Changed
- `pv_heat_weight` default 0.0005 → 0.0002, bounds lower limit 0.0005 → 0.0001.
- `predict_thermal_trajectory`: new optional `inlet_temp` parameter for slab model (backward-compatible).
- InfluxDB Flux query entity filter now includes `LIVING_ROOM_TEMP_ENTITY_ID`.

## [0.2.0] - 2026-02-10

### Added
- Initial release of ML Heating Underfloor addon
- Physics-based machine learning heating control optimized for underfloor heating
- Underfloor-specific thermal defaults (lower outlet temps, higher effectiveness)
- Complete parameter sync with .env configuration
- Cooling mode support with underfloor-specific bounds
- Heat source channel architecture for isolated learning
- Indoor trend protection to prevent parameter drift
- Full InfluxDB v2 integration with features bucket
- Solar correction and PV forecast integration
- Delta forecast calibration for local weather offsets

### Optimized for Underfloor
- CLAMP_MAX_ABS set to 35°C (protects floor covering)
- OUTLET_EFFECTIVENESS at 0.93 (large radiating surface)
- Conservative learning rates for slow thermal mass
- Extended training lookback (1800 hours) for screed slab dynamics
- Slab time constant parameter for Estrich thermal modeling
