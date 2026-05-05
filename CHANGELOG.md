# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Physics-Direct calibration path** — Added a fully analytical, sequential calibration method (`Physics Direct`) that estimates all thermal model parameters from first principles, without relying on scipy optimization. This path is selectable from the dashboard and exposes all parameters for user editing in `config.yaml`.
- **Calibration method selector in dashboard** — The dashboard now allows users to choose between "Scipy Optimizer" (default) and "Physics Direct" calibration methods when triggering model recalibration.

### Changed
- **Calibration method config option** — Added `CALIBRATION_METHOD` to `src/config.py` and `config.yaml`, allowing explicit selection of calibration path via config/environment.
- **Config schema validation** — Updated `config.yaml` schema to include bounds for `cloud_factor_exponent` and `solar_decay_tau_hours`.
- **Magic numbers refactored** — Extracted magic numbers as constants in calibration code; improved comments for cloud exponent logic.

### Fixed
- **Config default mismatches** — Fixed 6 mismatches between `src/config.py` defaults and `ThermalParameterConfig.DEFAULTS` (PV, fireplace, TV weights, thermal time constant, slab tau, total conductance). Now config defaults are aligned and validated.
- **State-file bounds validation** — Persisted calibration parameters are now validated against bounds before being accepted as fallback, preventing corrupted values from overriding config defaults.
- **Test coverage** — Updated and expanded unit tests for physics-direct calibration, including TV weight and solar lag xcorr edge cases; all tests pass.

### Fixed
- **Solar lag xcorr: non-contiguous data bug** — `_calibrate_solar_lag_xcorr()` in `src/physics_calibration_direct.py` previously operated on `stable_periods` (non-adjacent 20-min windows scattered across weeks), making lag shifts meaningless. Rewrote to operate on the raw time-sorted DataFrame, computing xcorr within each **contiguous PV-active episode** and returning the modal best-lag across episodes. Added 3 new edge-case tests.
- **Slab tau step-ordering dependency** — `_calibrate_slab_tau_grid_search()` used the config default for `delta_t_floor` even though the calibrated value was available from the immediately preceding step 8. Added `delta_t_floor` parameter; `calibrate_thermal_model_physics()` now passes the step-8 calibrated value so the slab-tau estimate is not biased by an incorrect default.
- **OE docstring incorrect** — Corrected the docstring in `_calibrate_oe_analytical()`: weight description now consistently uses the term `drive = T_outlet − T_indoor` to avoid confusion with the reciprocal.
- **IQR outlier rejection in residual weight estimator** — `_residual_heat_source_weight()` now applies a 1.5-IQR fence to the collected sample distribution before taking the percentile, preventing extreme residuals from transient slab effects or sensor spikes from biasing the estimate.
- **bfill NaN gap warning** — `calibrate_thermal_model_physics()` now logs a `⚠️` warning for each key column (flow_rate, inlet_temp, target_outlet_temp) that has more than 6 consecutive NaN rows (>30 min gap) after imputation, making silent bfill contamination visible.
- **Tau calibration duration gate** — `calculate_cooling_time_constant()` in `src/physics_calibration.py` now requires HP-off blocks to span at least 2 hours before including their tau estimate in the weighted average, preventing short blocks (typical in underfloor systems) from systematically underestimating the room thermal time constant.

### Added
- **State-file fallback for all calibration parameters** — `calibrate_thermal_model_physics()` now resolves the active `ThermalStateManager` early and uses a `_state_fallback()` helper. When a calibration step cannot produce a value (e.g. no fireplace-active rows in the dataset), it uses the previously persisted value from `unified_thermal_state.json` instead of the hardcoded default, preserving any prior calibration result. If the state file also has no valid value, the `config.yaml`-editable default is used as the final fallback.
- **`cloud_factor_exponent` and `solar_decay_tau_hours` persisted to state file** — `set_calibrated_baseline()` in `unified_thermal_state.py` and `_get_default_state()` now include these two parameters so they survive restarts and are available as warm-start fallbacks on the next calibration run.

### Changed
- **PV `min_periods` raised from 5 to 15** — `calibrate_thermal_model_physics()` now requires at least 15 qualifying stable periods for PV heat-weight estimation, reducing sensitivity to occupancy-correlated noise when PV-active days are few.
- **All calibration parameters now editable in `config.yaml`** — `ThermalParameterConfig.get_default()` now reads from `config` module variables first (which are sourced from `config.yaml` / environment variables) before falling back to the hardcoded `DEFAULTS` dict. Added 7 new config vars to `src/config.py` that were previously missing: `HEAT_LOSS_COEFFICIENT`, `OUTLET_EFFECTIVENESS`, `DELTA_T_FLOOR`, `FP_DECAY_TIME_CONSTANT`, `ROOM_SPREAD_DELAY_MINUTES`, `CLOUD_FACTOR_EXPONENT`, `SOLAR_DECAY_TAU_HOURS`. Both `.env_sample` and `config.yaml` are updated with the two previously undocumented entries (`cloud_factor_exponent`, `solar_decay_tau_hours`).


### Added
- **Physics-Direct Calibration Path** (`src/physics_calibration_direct.py`): new `calibrate_thermal_model_physics()` function that estimates every thermal parameter analytically — no scipy optimizer, no MAE fitting.  Sequential decoupling derives each parameter after locking previous ones:
  1. HLC via `calibrate_hlc()` (OLS flow-meter regression)
  2. OE via per-window algebra from HP-only stable periods
  3. τ_room from log-linear OLS on HP-off cooling curves
  4. `pv_heat_weight` via residual energy balance on PV-on periods
  5. `fireplace_heat_weight` via residual energy balance on FP-on periods
  6. `tv_heat_weight` via residual energy balance on TV-on periods
  7. `solar_lag_minutes` via PV ↔ indoor-residual cross-correlation
  8. `delta_t_floor` via P25 percentile of (outlet − inlet)
  9. `slab_time_constant_hours` via 1-D grid search over [0.1, 4.0 h] (replaces scipy dependency)
  10. `fp_decay_time_constant` via existing log-linear OLS
  11. `room_spread_delay_minutes` via existing cross-correlation
  12. `cloud_factor_exponent` via log-OLS (when `CLOUD_COVER_CORRECTION_ENABLED`)
  13. `solar_decay_tau_hours` via existing log-linear OLS
- `CALIBRATION_METHOD` config variable (`"scipy"` default, or `"physics"`): selects the calibration path system-wide.
- `--calibrate-physics-direct` CLI flag to `src/main.py`: runs the physics-direct path explicitly, exits after calibration.
- Flag-file support in `src/main.py`: writing `/data/config/calibrate_physics_direct_flag` triggers the physics-direct path on next startup (same pattern as the existing HLC flag).
- Dashboard calibration method radio toggle in `dashboard/components/control.py`: "Scipy Optimizer" vs "Physics Direct" — selecting Physics Direct writes the `calibrate_physics_direct_flag`.
- 19 new unit tests in `tests/unit/test_physics_calibration_direct.py` covering all new analytical estimators.

### Changed
- `train_thermal_equilibrium_model()` in `src/physics_calibration.py` now accepts a `method` parameter (`"scipy"` or `"physics"`).  Existing callers are unaffected (default remains `"scipy"`).  When `CALIBRATION_METHOD="physics"` is set in config and `method="scipy"` is passed (the default), the config value takes precedence.
- `--calibrate-physics` CLI flag now respects `CALIBRATION_METHOD` from config rather than always running scipy.

## [0.2.0] - 2026-02-10

### Added
- **Gentle Trajectory Correction System**: Intelligent additive correction preventing outlet temperature spikes during thermal trajectory deviations
- **Enhanced Forecast Integration**: Fixed feature storage during binary search for accurate trajectory verification with real PV/temperature forecast data
- **Open Window Adaptation**: System automatically detects sudden heat loss changes and restabilizes when disturbances end
- **Comprehensive TDD Test Suite**: 11 tests for differential scaling removal with 100% pass rate
- Thermal state validator for robust physics parameter validation
- Comprehensive thermal physics test suite with 36 critical tests
- Smart temperature rounding using thermal model predictions
- Enhanced logging to show actual applied temperatures

### Changed
- **MAJOR: Trajectory Correction Algorithm**: Replaced aggressive multiplicative correction (7x factors causing outlet spikes) with gentle additive approach based on user's heat curve automation (5°C/8°C/12°C per degree)
- **MAJOR: Thermal Model Simplification**: Removed differential-based effectiveness scaling to eliminate calibration-runtime mismatch and ensure consistent model behavior
- **Correction Boundaries**: Conservative ≤0.5°C/≤1.0°C/>1.0°C thresholds instead of aggressive ≤0.3°C/>0.5°C thresholds
- **Heat Curve Alignment**: Trajectory corrections now use proven 15°C per degree shift logic, scaled for direct outlet temperature adjustment
- Simplified heat balance equation to use constant outlet effectiveness coefficient
- Enhanced test coverage for thermal physics edge cases and validation
- Updated logging format to show rounded temperatures applied to HA sensors

### Fixed
- **CRITICAL: Aggressive Trajectory Correction** - Eliminated outlet temperature doubling (0.5°C error → 65°C outlet) by replacing multiplicative with gentle additive corrections (0.5°C error → +2.5°C adjustment)
- **Feature Storage During Binary Search** - Fixed missing forecast data access during trajectory verification phases
- **CRITICAL: Thermal Physics Model Bug** - Fixed fundamental physics implementation error causing physically impossible temperature predictions (heating systems predicting cooling)
- Binary search convergence issues - system now finds optimal outlet temperatures correctly
- Energy conservation violations in thermal equilibrium calculations
- Cosmetic logging issue showing unrounded vs applied temperature values
- Test suite failures for outdoor coupling and thermal physics validation
- Heat input calculations using corrected physics formula: T_eq = (eff × outlet + loss × outdoor + external) / (eff + loss)

## [0.2.0-beta.3] - 2025-12-03

### Added
- **Unified Model Wrapper Architecture**: Consolidated enhanced_model_wrapper.py into single model_wrapper.py with EnhancedModelWrapper class
- **Persistent Thermal Learning**: Automatic state persistence across Home Assistant restarts with warm/cold start detection
- **ThermalEquilibriumModel Integration**: Physics-based thermal parameter adaptation with confidence tracking
- **Enhanced Prediction Pipeline**: Single prediction path replacing complex Heat Balance Controller (1,000+ lines removed)
- **Continuous Learning System**: Always-on parameter adaptation with learning confidence metrics
- **State Management Enhancement**: Thermal learning state persistence with automatic save/restore functionality
- **Architecture Simplification**: 70% complexity reduction while maintaining full enhanced capabilities

### Changed
- Simplified model wrapper from dual-file to single-file architecture
- Enhanced thermal predictions with simplified interface maintaining all functionality
- Improved maintainability with unified EnhancedModelWrapper class
- Streamlined import structure eliminating duplicate dependencies
- Upgraded learning persistence to survive service restarts automatically

### Removed
- enhanced_model_wrapper.py (consolidated into model_wrapper.py)
- enhanced_physics_features.py (unused dead code eliminated)
- Heat Balance Controller complexity (~1,000 lines of complex control logic)
- Duplicate functionality and redundant code paths

### Fixed
- Import dependencies updated across all test files
- Test suite validation maintained (29/29 tests passing)
- Backward compatibility preserved for all existing interfaces
- Learning state persistence across system restarts

## [0.2.0-beta.2] - 2025-12-03

### Added
- **Thermal Equilibrium Model with Adaptive Learning**: Real-time parameter adaptation with 96% accuracy
- **Enhanced Physics Features Integration**: 34 total thermal intelligence features for ±0.1°C control precision  
- **Multi-Heat-Source Physics Engine**: Complete coordination system for PV (1.5kW), fireplace (6kW), electronics (0.5kW)
- **Adaptive Fireplace Learning System**: Advanced learning from temperature differential patterns with state persistence
- **PV Forecast Integration**: 1-4 hour lookahead capability with cross-day boundary handling
- **Comprehensive Test Coverage**: 130 passed tests with excellent defensive programming patterns (3 intentionally skipped)
- **Production-Ready Integration**: Complete Home Assistant and InfluxDB integration endpoints
- **Advanced Safety Systems**: Physics-aware bounds checking and parameter stability monitoring
- **Real-Time Learning Architecture**: Gradient-based optimization with confidence-based effectiveness scaling
- **Multi-Source Heat Coordination**: Intelligent heat contribution balancing with weather effectiveness factors

### Changed
- Enhanced physics features from 19 to 34 total features with thermal momentum analysis
- Upgraded test suite to 130+ tests with comprehensive multi-heat-source validation
- Improved learning convergence to <100 iterations typical with 96% prediction accuracy
- Enhanced system efficiency bounds to 40-90% with adaptive optimization

### Fixed
- PV forecast test interference issue with datetime mocking isolation
- Thermal equilibrium model parameter bounds and gradient validation
- Adaptive fireplace learning safety bounds enforcement (1.0-5.0kW)
- Multi-heat-source physics integration with robust error handling

## [0.2.0-beta.1] - 2025-12-02

### Added
- **Enhanced Physics Features**: 15 new thermal momentum features (thermal gradients, extended lag analysis, cyclical time encoding)
- **Comprehensive Test Suite**: 18/18 enhanced feature tests passing with mathematical validation
- **Backward Compatibility**: 100% preservation of original 19 features with zero regressions
- **Performance Optimization**: <50ms feature build time with minimal memory impact
- **Advanced Feature Engineering**: P0/P1 priority thermal intelligence capabilities
- Version strategy and development workflow documentation
- Changelog standards and commit message conventions
- Professional GitHub Issues management system
- Memory bank documentation with Week 2 completion milestone
- Comprehensive technical achievement summaries and performance metrics

### Changed
- Extended physics features from 19 to 34 total thermal intelligence features
- Enhanced thermal momentum detection with multi-timeframe analysis
- Improved predictive control through delta features and cyclical encoding
- Upgraded test coverage to include comprehensive edge case validation

## [0.0.1-dev.1] - 2024-11-27

### Added
- Initial Home Assistant add-on structure and configuration
- Physics-based machine learning heating control system
- Real-time dashboard with overview, control, and performance panels
- Comprehensive configuration schema with entity validation
- InfluxDB integration for data storage and retrieval
- Multi-architecture support (amd64, arm64, armv7, armhf, i386)
- Backup and restore functionality for ML models
- Development API for external access (Jupyter notebooks)
- Advanced learning features with seasonal adaptation
- External heat source detection (PV, fireplace, TV)
- Blocking detection for DHW, defrost, and maintenance cycles
- Physics validation and safety constraints
- Professional project documentation and issue templates

### Fixed
- Home Assistant add-on discovery issue by implementing proper semantic versioning
- Add-on configuration validation and schema structure

### Security
- Secure API key authentication for development access
- InfluxDB token-based authentication
- AppArmor disabled for system-level heat pump control access

[Unreleased]: https://github.com/jhilkert90-wq/ml_heating_underfloor/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jhilkert90-wq/ml_heating_underfloor/compare/v0.2.0-beta.3...v0.2.0
[0.2.0-beta.3]: https://github.com/jhilkert90-wq/ml_heating_underfloor/compare/v0.2.0-beta.2...v0.2.0-beta.3
[0.2.0-beta.2]: https://github.com/jhilkert90-wq/ml_heating_underfloor/compare/v0.2.0-beta.1...v0.2.0-beta.2
[0.2.0-beta.1]: https://github.com/jhilkert90-wq/ml_heating_underfloor/compare/v0.0.1-dev.1...v0.2.0-beta.1
[0.0.1-dev.1]: https://github.com/jhilkert90-wq/ml_heating_underfloor/releases/tag/v0.0.1-dev.1
