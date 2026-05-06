# Active Context - Current Work & Decision State

### 🔧 **OE Drive Filter + HLC Default Target Temp Fix — May 6, 2026**

#### **What changed**
- Removed `drive >= 3°C` gate from analytical OE — was discarding 68% of HP-only periods and biasing OE downward. Now uses `drive > 0` (outlet > indoor, HP running) matching scipy path behavior.
- OE scipy refinement now uses `ThermalEquilibriumModel.predict_equilibrium_temperature()` instead of simplified formula, matching the scipy path's objective exactly.
- `calibrate_hlc()` in `hlc_learner.py`: when `target_temp` column is unavailable, synthesises a constant column from `HLC_DEFAULT_TARGET_TEMP` (default 22.6°C) instead of skipping quality gates.
- Added `HLC_DEFAULT_TARGET_TEMP` config parameter in `config.py`.
- Updated test for new `target_temp` behavior (INFO instead of WARNING).

#### **Why**
- With HLC=0.119 (from poor regression, R²=-0.06, no target_temp quality gates), OE converged to 0.81 in both physics-direct and scipy paths. With correct HLC=0.133 (from `calculate_direct_heat_loss`), OE converges to 0.91-0.92 — both paths agree.
- The previous OE=0.95 was the **default** that was never actually overwritten by calibration — `calibrate_hlc` produced a bad HLC, which made the OE optimization plateau at a lower value, but the old code never wrote the OE result back.
- Offline comparison of both calibration paths confirms: Physics-direct OE=0.906, Scipy OE=0.919 (HP-only MAE=0.62°C, much better than previous 0.65°C).

#### **Files changed**
- `src/physics_calibration_direct.py` — drive filter, scipy refinement
- `src/hlc_learner.py` — default target_temp synthesis
- `src/config.py` — HLC_DEFAULT_TARGET_TEMP
- `tests/unit/test_hlc_learner.py` — updated test
- `test_calibration_compare.py` — new comparison script

---

### 🔧 **Physics-Direct Calibration Accuracy Fixes — May 2026**
- `filter_stable_periods()` in `src/physics_calibration.py` now resolves the output path for `stable_periods.json` dynamically from `os.path.dirname(config.UNIFIED_STATE_FILE)` instead of the hardcoded `/opt/ml_heating/` string.
- The directory for `stable_periods.json` is created if it does not exist (`os.makedirs(..., exist_ok=True)`).
- If writing the file fails (e.g., permission or disk error), the exception is caught and a warning is logged, but calibration continues.
- Added `import os` to the module-level imports in `src/physics_calibration.py`.

#### **Why**
- In some Home Assistant add-on environments the `/opt/ml_heating/` directory does not exist, causing a `FileNotFoundError` when calibration tried to write `stable_periods.json`. Using the same directory as configured via `config.UNIFIED_STATE_FILE` is the correct and consistent approach. The directory is created if it does not yet exist, and write failures are caught and logged as warnings so calibration continues.

#### **Files changed**
- `src/physics_calibration.py`

---

#### **What changed**
- **Physics-Direct calibration path added**: `src/physics_calibration_direct.py` now implements a fully analytical, sequential calibration method that estimates all thermal model parameters from first principles (no scipy dependency). This path is selectable from the dashboard and exposes all parameters for user editing in `config.yaml`.
- **Dashboard calibration selector**: Users can now choose between "Scipy Optimizer" and "Physics Direct" calibration methods when triggering model recalibration. The selection is persisted and triggers the appropriate calibration logic on restart.
- **Config option & schema**: Added `CALIBRATION_METHOD` to `src/config.py` and `config.yaml`. Updated schema to include bounds for `cloud_factor_exponent` and `solar_decay_tau_hours`.
- **Magic numbers refactored**: Calibration code now uses named constants for previously hardcoded values; improved comments for cloud exponent logic.
- **Config default alignment**: Fixed 6 mismatches between `src/config.py` defaults and `ThermalParameterConfig.DEFAULTS` (PV, fireplace, TV weights, thermal time constant, slab tau, total conductance).
- **State-file bounds validation**: Persisted calibration parameters are now validated against bounds before being accepted as fallback, preventing corrupted values from overriding config defaults.
- **Expanded test coverage**: Updated and expanded unit tests for physics-direct calibration, including TV weight and solar lag xcorr edge cases; all tests pass.

#### **Why**
- Provides a robust, transparent calibration path for environments where scipy optimization is unavailable or undesirable.
- Ensures all calibration parameters are user-editable and validated, preventing silent fallback to hardcoded values.
- Improves reliability and maintainability by aligning config defaults and enforcing bounds.

#### **Files changed**
- `src/physics_calibration_direct.py`
- `dashboard/components/control.py`
- `src/config.py`
- `ml_heating_underfloor/config.yaml`
- `.env_sample`
- `src/unified_thermal_state.py`
- `src/thermal_config.py`
- `tests/unit/test_physics_calibration_direct.py`
- `CHANGELOG.md`

---

### ✅ **Calibration parameter fallback + config.yaml exposure — May 2026**

#### **What changed**
- `src/physics_calibration_direct.py` — `calibrate_thermal_model_physics()` resolves the active `ThermalStateManager` at the top of the function (before step 0) and loads `baseline_parameters` from the persisted state file. A `_state_fallback(key)` helper returns the persisted value when it's a valid float, otherwise falls back to `ThermalParameterConfig.get_default()`. All 13 step-level fallbacks now use `_state_fallback()` instead of `ThermalParameterConfig.get_default()`. Log messages now indicate whether "persisted" or "default" was used.
- `src/unified_thermal_state.py` — `_get_default_state()` now includes `cloud_factor_exponent` and `solar_decay_tau_hours` in the `baseline_parameters` dict. `set_calibrated_baseline()` persists those two parameters if present in the input dict.
- `src/thermal_config.py` — All calibration parameters now have bounds and are validated before being accepted from state file.
- `src/config.py`, `.env_sample`, `ml_heating_underfloor/config.yaml` — All calibration parameters are now user-editable and documented.

#### **Why**
- Ensures robust fallback for all calibration parameters: calibrated value > persisted value > config.yaml value.
- Prevents corrupted or out-of-bounds values from being loaded from state.
- All parameters are now visible and editable by the user.

#### **Files changed**
- `src/physics_calibration_direct.py`
- `src/unified_thermal_state.py`
- `src/thermal_config.py`
- `src/config.py`
- `.env_sample`
- `ml_heating_underfloor/config.yaml`
- `CHANGELOG.md`
