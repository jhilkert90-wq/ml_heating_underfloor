# Active Context - Current Work & Decision State

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
- `tests/unit/test_cooling_ml.py` — 4 new tests
- `CHANGELOG.md`

---

### 🧪 **Extended Unit Tests for ML Pre-Cooling — 2026-05-13**

#### **What changed**
- Added comprehensive unit tests for all ML pre-cooling modules:
  - **Cold start scenarios**: Verified correct behavior when model, buffer, or metadata files are missing (empty buffer, no-risk prediction, graceful calibration failure).
  - **CoolingObservationBuffer**: Edge cases for NaN/Inf in features, label resolution at horizon boundaries, buffer overflow/eviction, and JSON serialization.
  - **CoolingMLModel**: Inference with empty/malformed features, prediction exceptions, and shadow mode logging.
  - **OverheatingPredictor**: Handling of missing forecast keys, reactive cooling logic, and fallback paths.
  - **Calibration/Online Learning**: Label logic, retrain triggers, and config default consistency (checked against hardcoded calibration logic).
- Added baseline `model_metadata.json` for ML cooling calibration state.
- All new tests in `tests/unit/test_cooling_ml_extended.py` (36+ cases).

#### **Why**
- Ensures robust cold start, edge case, and online learning behavior for ML-based pre-cooling. Prevents silent failures and regression in observation buffer and model logic. Verifies config defaults match calibration code.

#### **Files changed**
- `tests/unit/test_cooling_ml_extended.py` — new test suite
- `notebooks/analysis/models/model_metadata.json` — baseline model state
- `src/cooling_ml_observation_buffer.py`, `src/cooling_ml_model.py`, `src/cooling_ml_calibration.py`, `src/config.py` — minor fixes for testability and edge cases
- `.gitignore` — ignore new data/log files

---

### Fix HP False-Active from Residual Slab Heat — 2026-05-13

#### **What changed**
- Fixed `_is_heat_pump_active()` in `src/heat_source_channels.py`: added idle-band guard so that when both `thermal_power` and `delta_t` are near zero (< 0.1), the outlet/inlet temperature fallback is suppressed. This prevents residual slab thermal mass from falsely marking HP as active.
- Added 8 regression tests covering the bug scenario (HP off + warm outlet + PV active) in both heating and cooling modes.

#### **Why**
- On sunny days, the floor slab retains heat from previous HP operation or PV solar gain. The outlet temp stays above indoor + 1.0°C even with HP off (thermal_power=0). The fallback `outlet > indoor + 1.0 and outlet > inlet + 0.5` returned True → HP appeared in `active_contributions` → mixed-source attribution split learning errors between HP and PV → HP parameters (`outlet_effectiveness`, `heat_loss_coefficient`) were contaminated by PV-induced temperature changes.

#### **Files changed**
- `src/heat_source_channels.py` — idle-band guard in `_is_heat_pump_active()`
- `tests/unit/test_cooling_bugfixes.py` — 5 new unit tests
- `tests/unit/test_heat_source_channels.py` — 3 new routing integration tests
- `CHANGELOG.md`

---

### 🧊 Pre-Cooling ML Review & Bug
