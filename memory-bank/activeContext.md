# Active Context - Current Work & Decision State

### 🔀 Resolve PR Merge Conflicts (Latest Sync) — 2026-05-14

#### **What changed**
- Merged latest `origin/main` into the branch after new merge-conflict reports on the PR.
- Resolved conflict markers in `memory-bank/activeContext.md` and `memory-bank/progress.md` by preserving content from both branches.
- Accepted incoming base-branch updates for `.github/workflows/build.yaml` and `CHANGELOG.md`.

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
- Added a regression assertion in `tests/unit/test_pre_cooling_integration.py` proving that the corrected thermal PV keys reach `predict_thermal_trajectory()` as `pv_power=6000.0` and `pv_forecasts=[6000.0] * 13`.
- Updated stale cooling ML test configs in `tests/unit/test_cooling_ml.py` and `tests/unit/test_cooling_ml_calibration.py` from `PRE_COOL_LEAD_TIME_HOURS=8.0` to `3.0` so tests match production defaults.

#### **Why**
- The earlier fix corrected the test helper keys, but the integration test still did not explicitly assert that PV values were forwarded into the predictor call, leaving a regression gap.
- Several cooling ML tests still encoded the old 8-hour lead-time default, which could hide future drift between calibration tests and runtime behavior.

#### **Files changed**
- `tests/unit/test_pre_cooling_integration.py`
- `tests/unit/test_cooling_ml.py`
- `tests/unit/test_cooling_ml_calibration.py`

---

### 🐛 Fix Pre-Cooling Calibration Bugs — 2026-05-14

#### **What changed**
Five bugs in the pre-cooling calibration pipeline were identified and fixed:
1. `scikit-learn>=1.0.0` added to `requirements.txt` (was missing, causing silent AUC failure).
2. Feature key names corrected in `test_pre_cooling_integration.py::_make_features()` — used `pv_now`, `pv_forecast_{h}h`, `temp_forecast_{h}h` (matching `OverheatingPredictor`).
3. `cooling_ml_model._extract_feature()` now prefers raw electrical PV scale at inference to match training data.
4. `PRE_COOL_LEAD_TIME_HOURS` hardcoded fallback fixed from `8.0` → `3.0` in `cooling_ml_calibration.py`.
5. Observation buffer in `main.py` persisted whenever new labels are resolved (not only on successful retrain).

#### **Why**
- Bug 1 was confirmed from production logs; sklearn was absent so AUC was always `null`.
- Bugs 2–5 were latent correctness issues discovered during analysis.

#### **Files changed**
- `requirements.txt`
- `src/cooling_ml_calibration.py`
- `src/cooling_ml_model.py`
- `src/main.py`
- `tests/unit/test_pre_cooling_integration.py`

---

### 🛡️ PV Key Ownership Codified & Pre-cooling Path Regressions — 2026-05-14

#### **What changed**
- **Codified PV key ownership:**
  - Added explicit documentation and regression tests to prevent misuse of PV feature keys in pre-cooling and ML cooling paths.
  - Canonical AI MODEL NOTICE section added to `memory-bank/systemPatterns.md` with two-family key table, per-module usage map, four explicit rules, and citations.
  - Warning block added to `docs/ML_COOLING_MODEL_GUIDE.md` above Feature Engineering, instructing all contributors and AI models to use the correct PV key family.
- **Regression tests:**
  - Added `TestPVKeyContract` class (5 tests) to `tests/unit/test_overheating_predictor.py`:
    - Locks `OverheatingPredictor` to thermal keys (`pv_now`, `pv_forecast_{h}h`)
    - Locks `HLCCycle` to electrical key (`pv_now_electrical`)
    - Asserts guards and trajectory call kwargs to prevent silent regressions
- **Refactor:**
  - Moved `hlc_learner` imports to module level in `test_overheating_predictor.py` for clarity.
  - Hardened assertions in PV key contract tests per reviewer feedback.

#### **Why**
- Prevents future regressions where the wrong PV key family is used, which previously caused silent over-estimation of solar gain in thermal trajectory simulation.
- Ensures all contributors (human and AI) have a single, visible contract for PV feature key usage.

#### **Files changed**
- `memory-bank/systemPatterns.md` — canonical PV key contract section
- `docs/ML_COOLING_MODEL_GUIDE.md` — warning block above Feature Engineering
- `tests/unit/test_overheating_predictor.py` — 5 regression tests, refactor
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

### 🧪 Reviewer Follow-up — PV Contract Tests Made Strict — 2026-05-14

#### **What changed**
- Updated 2 regression tests in `tests/unit/test_overheating_predictor.py` (`TestPVKeyContract`) so they assert `predict_thermal_trajectory()` kwargs directly instead of relying only on `result["risk"]`.
- Added explicit assertions for:
  - `pv_power == pv_now` in thermal-key-only scenarios
  - `pv_forecasts` list being constructed from thermal `pv_forecast_{h}h` keys when `pv_forecast_electrical_*` keys are absent
- Re-checked markdown table formatting in `memory-bank/systemPatterns.md` and `docs/ML_COOLING_MODEL_GUIDE.md`; no malformed `||` table rows remain.

#### **Why**
- PR review correctly flagged that mocked trajectory outputs could allow false positives even if key-family wiring regressed. Asserting the call kwargs directly makes these tests true contract tests for key selection.

#### **Files changed**
- `tests/unit/test_overheating_predictor.py`
- `CHANGELOG.md`
- `memory-bank/progress.md`
- `memory-bank/activeContext.md`

---

### 📚 PV Feature Key Contract — Documentation & Regression Tests — 2026-05-14

#### **What changed**
- Added a permanent, highly-visible **`⚠️ AI MODEL NOTICE — PV Feature Key Contract`** section at the top of `memory-bank/systemPatterns.md`. This is the canonical reference for every contributor and AI model that touches PV-related code. It contains: a table of the two key families, a per-module usage map, four explicit rules, and verified source-code citations.
- Added a **`⚠️ PV Feature Key Contract`** warning block to `docs/ML_COOLING_MODEL_GUIDE.md` directly above the Feature Engineering section so it is visible to anyone modifying the cooling ML pipeline.
- Added **5 regression tests** (`TestPVKeyContract`) to `tests/unit/test_overheating_predictor.py` that lock in the correct key usage for `OverheatingPredictor` and `HLCCycle._build_cycle`.

#### **Why**
- AI models previously used `pv_now_electrical` / `pv_forecast_electrical_*` in places that should use the thermally-corrected `pv_now` / `pv_forecast_{h}h` keys, causing silent over-estimation of solar gain in the thermal trajectory simulation. The documentation and tests make the contract explicit and machin
