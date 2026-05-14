# Active Context - Current Work & Decision State

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
