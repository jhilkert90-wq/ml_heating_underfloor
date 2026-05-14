# Active Context - Current Work & Decision State

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
