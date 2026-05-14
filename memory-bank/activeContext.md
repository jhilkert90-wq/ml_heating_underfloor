# Active Context - Current Work & Decision State

### ✨ Feature: Cooling ML configurable calibration start date — 2026-05-14

#### **What changed**
- `src/cooling_ml_calibration.py`: `calibrate_cooling_ml()` reads `COOLING_ML_CALIBRATION_START_DATE` at the top of step 0. If non-empty and valid `DD.MM.YYYY`, computes `lookback_hours = int((now_utc - start_dt).total_seconds() / 3600)`. Falls back to default 2160 h with a warning on invalid input or future date.
- `src/config.py`: adds `COOLING_ML_CALIBRATION_START_DATE: str` (default `""`) and `_parse_cooling_start_date(s) → Optional[datetime]`.
- `ml_heating_underfloor/config.yaml`: adds `cooling_ml_calibration_start_date: ""` in options and `cooling_ml_calibration_start_date: "str?"` in schema.
- `ml_heating_underfloor/translations/en.yaml`: adds tooltip for `cooling_ml_calibration_start_date`.
- `tests/unit/test_cooling_ml_calibration.py`: adds `TestCoolingStartDate` with 6 tests.

#### **Why**
- The previous 2160 h relative lookback made it impossible to pin training to a specific summer/cooling season start. A user who wants the model trained only on data since, say, 1 June 2024 had no option. The start-date field is cooling-ML-only — physics calibration `training_lookback_hours` is unchanged.

#### **Files changed**
- `src/config.py`
- `src/cooling_ml_calibration.py`
- `ml_heating_underfloor/config.yaml`
- `ml_heating_underfloor/translations/en.yaml`
- `tests/unit/test_cooling_ml_calibration.py`
- `CHANGELOG.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

---

#### **What changed**
- `src/cooling_ml_calibration.py` Step 6 now builds the feature set from all 12 AT hindcast hours (`AT_roh_1h`–`AT_roh_12h`) and all 12 PV hindcast hours (`pv_forecast_1h`–`pv_forecast_12h`) instead of only `AT_roh_4h`.  Controlled by `COOLING_ML_AT_FORECAST_HOURS` and `COOLING_ML_PV_FORECAST_HOURS` env vars.
- `src/config.py` adds `COOLING_ML_AT_FORECAST_HOURS`, `COOLING_ML_PV_FORECAST_HOURS`, and keeps `COOLING_ML_FORECAST_HOURS` as a backward-compat alias.
- 4 new tests in `tests/unit/test_cooling_ml_calibration.py::TestForecastHourSelection` verify custom hour selection, default all-12h, and the legacy alias.

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
