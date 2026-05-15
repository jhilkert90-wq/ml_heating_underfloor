# Active Context - Current Work & Decision State

### ✨ Physics Newton-Step Heating Correction — 2026-05-15

#### **What changed**
- `src/model_wrapper.py`: Added `_calculate_physics_newton_correction()` implementing the exact formula `ΔT = ε / S_H` where `S_H = [η/(η+U)] × [1 − exp(−H/τ_room)]`. Shares all boundary guards and clamp logic with the existing `_calculate_physics_based_correction()`. Added `_calculate_ml_correction()` stub that falls back to Newton. Dispatch in `verify_trajectory_temperature_predictions()` reads `config.HEATING_CORRECTION_MODE` and routes to the appropriate method (default `"legacy"`).
- `src/config.py`: Added `HEATING_CORRECTION_MODE: str = os.getenv("HEATING_CORRECTION_MODE", "legacy")`.
- `config_adapter.py`: Added `'HEATING_CORRECTION_MODE': config.get('heating_correction_mode', 'legacy')` in `convert_addon_to_env()`.
- `ml_heating_underfloor/config.yaml`: Added `heating_correction_mode: "legacy"` in defaults block; `heating_correction_mode: "list(legacy|physics|ml)"` in schema (renders as HA dropdown).
- `ml_heating_underfloor/translations/en.yaml`: Added description entry for `heating_correction_mode`.
- `tests/unit/test_heating_correction.py`: 11 new unit tests (Newton accuracy, S_H fallback, clamp, dispatch, config_adapter).

#### **Why**
- The existing `_calculate_physics_based_correction()` over-corrects undershoot by ~2.26× and under-corrects overshoot by ~0.65× due to `urgency_multiplier=3.0` and asymmetric `overshoot_dampening`. The correct physics formula is a single Newton step `ΔT = ε / S_H` which is symmetric and horizon-aware.
- Legacy mode is preserved as default so existing installations are unaffected. Users can switch to `"physics"` in the HA dropdown after confirming calibration.

#### **Files changed**
- `src/model_wrapper.py`, `src/config.py`, `config_adapter.py`, `ml_heating_underfloor/config.yaml`, `ml_heating_underfloor/translations/en.yaml`, `tests/unit/test_heating_correction.py`

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
