# Active Context - Current Work & Decision State

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
- AI models previously used `pv_now_electrical` / `pv_forecast_electrical_*` in places that should use the thermally-corrected `pv_now` / `pv_forecast_{h}h` keys, causing silent over-estimation of solar gain in the thermal trajectory simulation. The documentation and tests make the contract explicit and machine-checkable.

#### **Files changed**
- `memory-bank/systemPatterns.md` — canonical PV key contract note (top of file)
- `docs/ML_COOLING_MODEL_GUIDE.md` — warning block above Feature Engineering
- `tests/unit/test_overheating_predictor.py` — 5 regression tests
- `CHANGELOG.md`
- `memory-bank/progress.md`
- `memory-bank/activeContext.md`

---

### 🔖 ML Heating Underfloor v0.2.30 Release Bump — 2026-05-13

#### **What changed**
- Version updated in `config.yaml` from `0.2.29` to `0.2.30`.
- No functional changes; this is a release bump to reflect recent bug fixes and ML pre-cooling enhancements already documented in the [Unreleased] changelog.

#### **Files changed**
- `ml_heating_underfloor/config.yaml`

---

### **Fix Cooling ML Calibration Import Crash — 2026-05-14**

#### **What changed**
- Fixed `import config` → `try: from . import config / except ImportError: import config` in `src/cooling_ml_calibration.py` (line 63) and `src/cooling_ml_model.py` (line 253)
- Added `joblib` and uncommented `lightgbm` in `requirements.txt`
- Created `tests/unit/test_cooling_ml_calibration.py` with 23 tests covering all calibration pipeline paths
- Extended `tests/unit/test_cooling_ml.py` with 4 import regression tests

#### **Why**
- Production log showed `calibrate_cooling_ml: missing dependency — No module named 'config'` at container startup. The bare `import config` worked only when the module was executed directly (not as a package), which is never the case in the add-on container.

#### **Files changed**
- `src/cooling_ml_calibration.py` — import fix
- `src/cooling_ml_model.py` — import fix
- `requirements.txt` — dependency fixes
- `tests/unit/test_cooling_ml_calibration.py` — 23 new tests (NEW)
-
