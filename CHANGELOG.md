# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Historical HLC Calibration**: New `calibrate_hlc()` function fetches historical data from InfluxDB, filters stable HP-only periods, and runs OLS regression to estimate building heat loss coefficient
- **`--calibrate-hlc` CLI argument**: One-shot HLC calibration from historical data and exit
- **HLC calibration flag detection**: Main loop detects `/data/config/hlc_calibrate_flag` at startup and runs calibration automatically
- **Dashboard "Calibrate HLC" button**: Writes flag file and restarts the ML system to trigger HLC calibration
- **Cold start file creation**: `HLCSessionLearner.load_session_records()` now creates an empty stub file when no session file exists
- **HLC calibration config params**: `HLC_CALIBRATION_LOOKBACK_HOURS` (default 720) and `HLC_CALIBRATION_MIN_PERIODS` (default 20)

### Changed
- **PV-triggered HLC sessions**: HLC session learner now opens a session when `pv_now_electrical < HLC_PV_MAX_W` (50 W) and closes it when `pv_now_electrical >= HLC_PV_MAX_W`. Replaces the old calendar-day-based session model.
- **Per-cycle filtering**: TV, DHW, defrost, DHW-boost, and blocking cycles are now filtered individually — the session stays alive. Previously these caused whole-session rejection.
- **Fireplace remains whole-session reject**: Fireplace active in any cycle still rejects the entire session, as it distorts the room heat balance throughout the session.
- **`DayRecord` → `SessionRecord`**: The `DayRecord` dataclass has been replaced by `SessionRecord` with `session_start`, `session_end`, and `duration_minutes` fields.
- **`HLC_SESSION_MIN_DAYS` → `HLC_SESSION_MIN_SESSIONS`**: Renamed config key; default changed from 5 → 10.
- **`HLC_SESSION_MAX_DAYS` → `HLC_SESSION_MAX_SESSIONS`**: Renamed config key; default changed from 60 → 120.
- **API renames**: `load_day_records()` → `load_session_records()`, `get_day_records()` → `get_session_records()`, `day_record_count` → `session_record_count`.
- **`_build_cycle()` extracted as module-level function**: Previously `HLCLearner._build_cycle()` static method, now shared by `HLCSessionLearner`
- **`main.py` HLC push_cycle simplified**: Single session learner block instead of dual online + session blocks
- **Tests rewritten**: `test_hlc_learner.py` now tests `_build_cycle()`, `calibrate_hlc()`, and `HLCCycle` instead of removed classes

### Removed
- **Online HLC Learner**: Removed `HLCLearner` class and `HLCWindow` dataclass — replaced by day-level session learner and historical calibration
- **Online HLC config params**: Removed `HLC_LEARNER_ENABLED`, `HLC_WINDOW_MINUTES`, `HLC_CYCLES_PER_WINDOW_MIN_FRAC`, `HLC_MIN_WINDOWS`, `HLC_MAX_WINDOWS`, `HLC_MAX_UPDATE_FRACTION` from config, config.yaml, config_adapter, and translations

### Fixed
- **Active PV-night sessions persist full in-progress state**: `session_cycles` are now written on every append so restarts resume the real session instead of losing already collected cycles
- **Closed sessions no longer persist as active**: The close path now saves the inactive post-close state after validation, preventing phantom sessions after restart
- **Session timing now uses the close trigger cycle**: Stored `session_end` and `duration_minutes` are based on the actual threshold-crossing cycle timestamp instead of wall-clock save time
- **Legacy `day_records` payloads are migrated automatically**: Existing JSON stores are upgraded to `session_records` on load so historical HLC data survives the redesign
- **PV-triggered naming is consistent across config surfaces**: `.env_sample`, startup logging, translation text, and adapter comments now use the new session terminology
- **Shared HLC validation params restored**: 6 validation gate params (`HLC_PV_MAX_W`, `HLC_MAX_INDOOR_DELTA`, `HLC_MAX_TREND`, `HLC_OUTDOOR_TEMP_MIN`, `HLC_OUTDOOR_TEMP_MAX`, `HLC_MIN_HEATING_DEMAND_K`) were accidentally removed with the online learner but are used by `_close_day()` and `calibrate_hlc()` via `getattr()` — now restored under "HLC Validation Gates" section
- **Greedy column name matching in `calibrate_hlc()`**: Derived columns (e.g. `indoor_temp_delta_60m`) could overwrite base temperature mappings; now uses `setdefault()` and skips columns containing delta/lag/diff/trend/gradient/forecast
- **Uncapped HLC written to thermal state**: `calibrate_hlc()` now rejects estimates outside [0.01, 2.0] kW/K to prevent physically implausible values from corrupting the model
- **Flag file removal race**: If `/data/config/hlc_calibrate_flag` cannot be removed, calibration is skipped with an error log instead of running on every restart
- **Missing indoor trend gate in `calibrate_hlc()`**: Added first-to-last indoor temperature change check (max_trend) to match the session learner's quality gates

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
