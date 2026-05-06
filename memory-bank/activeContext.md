# Active Context - Current Work & Decision State

### 🚀 **Physics-Direct Calibration Enhancement — May 2026**

#### **What changed**
- **Outlet Effectiveness (OE) Calibration**: `_calibrate_oe_analytical()` in `src/physics_calibration_direct.py` now uses a two-stage approach: (1) analytical weighted-median OE as initial guess, (2) scipy `minimize_scalar` refinement with HLC locked, minimizing MAE against HP-only stable periods. This improves OE accuracy from ~0.72 to ~0.95 and robustness against sensor noise when temperature drive is small.
- **Solar Lag Calibration**: `_calibrate_solar_lag_xcorr()` rewritten to correlate PV with the rate of change of residuals (`d(residual)/dt`) instead of residual level, reducing slab-mass delay bias. Maximum lag reduced from 36 to 12 steps (60 min), correlation threshold increased from 0.1 to 0.3, and weighted median is used for lag estimation.
- **Thermal Time Constant Calibration**: Calibration now prioritizes transient parameter estimation using `calibrate_transient_parameters()` and `filter_transient_periods()` (heating sequences, abundant data), falling back to cooling curve analysis only if necessary.
- **Unit Labels Correction**: `ThermalParameterConfig` in `src/thermal_config.py` now correctly labels `outlet_effectiveness` and `heat_loss_coefficient` as "kW/K" instead of "dimensionless" and "1/hour".
- **Logging & Error Handling**: Added new logging and error handling for scipy optimization failures, improving traceability and robustness.
- **Documentation Improvements**: Enhanced inline documentation and comments for calibration routines for clarity.

#### **Why**
- Analytical OE formula was numerically fragile with small temperature drives, causing sensor noise to dominate. Scipy refinement is robust to per-sample noise.
- Previous solar lag calibration was biased by slab-mass smoothing, resulting in incorrect lag values. Using `d(residual)/dt` removes this effect.
- Cooling curve-based time constant calibration was rarely possible; transient calibration uses more available heating data.
- Correct unit labeling ensures physical consistency and clarity for users and developers.

#### **Files changed**
- `src/physics_calibration_direct.py`
- `src/thermal_config.py`
- `memory-bank/progress.md`
- `memory-bank/activeContext.md`

---

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
- **Magic numbers refactored**: Calibration code now uses named constants for previously hardcoded values; improved comments for cloud exponent

#### **Files changed**
- `src/physics_calibration_direct.py`
- `dashboard/components/control.py`
- `src/config.py`
- `ml_heating_underfloor/config.yaml`
- `.env_sample`
- `src/unified_thermal_state.py`
- `src/thermal_config.py`
- `tests/unit/test_physics_calibration_direct.py`
