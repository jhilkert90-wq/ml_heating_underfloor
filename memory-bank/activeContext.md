# Active Context - Current Work & Decision State

### ✅ **stable_periods.json path bug fix — May 2026**

#### **What changed**
- `filter_stable_periods()` in `src/physics_calibration.py` now resolves the output path for `stable_periods.json` dynamically from `os.path.dirname(config.UNIFIED_STATE_FILE)` instead of the hardcoded `/opt/ml_heating/` string.
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
- `src/thermal_config.py` — `ThermalParameterConfig.get_default()` now checks the `config` module (which reads from env vars / `config.yaml`) before returning the hardcoded `DEFAULTS` value. A `_CONFIG_VAR_MAP` dict maps param names to their config variable names.
- `src/config.py` — Added 7 missing config vars: `HEAT_LOSS_COEFFICIENT`, `OUTLET_EFFECTIVENESS`, `DELTA_T_FLOOR`, `FP_DECAY_TIME_CONSTANT`, `ROOM_SPREAD_DELAY_MINUTES`, `CLOUD_FACTOR_EXPONENT`, `SOLAR_DECAY_TAU_HOURS`.
- `.env_sample` — Added `CLOUD_FACTOR_EXPONENT` and `SOLAR_DECAY_TAU_HOURS`.
- `ml_heating_underfloor/config.yaml` — Added `cloud_factor_exponent` and `solar_decay_tau_hours` entries.

#### **Why**
When calibration data is insufficient for a parameter (e.g. no fireplace-active rows), the fallback chain is now:
1. Calibrated value from data
2. Last valid value from `unified_thermal_state.json` (survives restarts)
3. User-editable value from `config.yaml` / environment variable

Previously the fallback jumped directly to the hardcoded Python constant in `ThermalParameterConfig.DEFAULTS`, bypassing both the state file and the user-editable config. This meant a corrupted or missing parameter silently reverted to a value the user could not change without editing source code.

#### **Files changed**
- `src/physics_calibration_direct.py`
- `src/unified_thermal_state.py`
- `src/thermal_config.py`
- `src/config.py`
- `.env_sample`
- `ml_heating_underfloor/config.yaml`
- `CHANGELOG.md`
- `memory-bank/progress.md`
- `memory-bank/activeContext.md`

---

### ✅ **CI Workflow fixes — May 2026**

#### **What changed**
- `.github/workflows/auto-docs.yaml` — upgraded `actions/checkout@v4` → `@v6` to fix Node.js 20 deprecation warning; changed API base URL from `https://api.githubcopilot.com` to `https://models.inference.ai.azure.com` to fix "server-to-server token not supported" error; updated model comment.
- `.github/workflows/ai-code-review.yaml` — same API base URL fix; model changed from `gpt-5.4` to `gpt-4.1`; updated footer comment and heading text to reference GitHub Models instead of Copilot.

#### **Why**
The `update-docs` workflow was emitting two warnings on every push to main:
1. `actions/checkout@v4` running on Node.js 20 (deprecated, forced to Node.js 24 from June 2026).
2. `AI call failed: checking server-to-server token: bad request` — the GitHub Copilot endpoint (`api.githubcopilot.com`) does not accept the `GITHUB_TOKEN` server-to-server token issued to GitHub Actions. The GitHub Models endpoint (`models.inference.ai.azure.com`) supports it with the `models: read` permission already present in both workflows.

#### **Files changed**
- `.github/workflows/auto-docs.yaml`
- `.github/workflows/ai-code-review.yaml`

---

### ✅ **Predictive Pre-Cooling Implementation — June 2025**

#### **What changed**
- NEW `src/overheating_predictor.py` — `OverheatingPredictor` class that runs a passive thermal trajectory simulation (HP OFF, outlet=inlet) using PV + outdoor forecasts to predict future room temperature peaks.
- `src/config.py` —
