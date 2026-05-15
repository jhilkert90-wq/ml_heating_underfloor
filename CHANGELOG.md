# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Heating Correction ML: Online Learning** (`HeatingCorrectionObservationBuffer`): Mirrors the pre-cooling sliding-window observation buffer pattern for the LightGBM heating regressor.
  - `src/heating_correction_ml_observation_buffer.py` — new `HeatingCorrectionObservationBuffer` class; stores heating-cycle feature snapshots, resolves regression labels `−(T_indoor[t+N] − T_target) / S_H` after `label_horizon_steps` cycles, auto-triggers retrain via `calibrate_heating_correction_ml()` when `n_labeled ≥ min_training_samples AND labeled_since_last_train ≥ retrain_trigger_k`; JSON persistence with atomic tmp→replace writes
  - Per-cycle integration in `src/main.py`: `push_pending` on every heating cycle, `resolve_labels` every cycle, auto-retrain with hot-reload (resets `EnhancedModelWrapper._heating_correction_ml_model = None` to force singleton reload)
  - Buffer collects observations regardless of `HEATING_CORRECTION_MODE` so data accumulates even before ML mode is activated
  - S_H recomputed at resolve-time from current calibrated thermal parameters (`_compute_s_h` / `_read_baseline_thermal_params`)
- **New config vars**: `HEATING_ML_OBSERVATION_BUFFER_PATH` (default: `_UNIFIED_STATE_DIR/heating_correction_ml_obs_buffer.json`), `HEATING_ML_RETRAIN_TRIGGER_K` (default `50`), `HEATING_ML_BUFFER_MAX_N` (default `500`)
- 25 new unit tests in `tests/unit/test_heating_correction_observation_buffer.py`


  - `src/heating_correction_ml_model.py` — inference class `HeatingCorrectionMLModel`; loads joblib model + metadata, exposes `predict(features, target_indoor)` and `r2_score`
  - `src/heating_correction_ml_calibration.py` — one-shot training: cold-season filter (AT < 18 °C), feature vector with AT hindcast, PV hindcast, dynamic fireplace/TV lags, regression label `−(T_future − T_target) / S_H`
  - Blended dispatch in `model_wrapper._calculate_ml_correction()`: confidence-weighted blend `w = R²` (clamped, with `HEATING_ML_BLEND_MIN_R2` minimum threshold)
  - `--calibrate-heating-correction-ml` CLI flag and `/data/config/calibrate_heating_correction_ml_flag` flag file
- **Heating ML feature expansion**:
  - PV hindcast features (`pv_forecast_1h`–`pv_forecast_Nh`) controlled by `HEATING_ML_PV_FORECAST_HOURS` (default `"1,2,3,4"`)
  - PV instantaneous and rolling features (`PV_Generate`, `pv_roll_1h`, `pv_roll_2h`) at both training and inference time; prefers `pv_now_electrical`/`pv_forecast_electrical_Xh` keys to match training scale
  - Dynamic fireplace lag windows (`fireplace_lag_1h`, `fireplace_lag_2h`, …) controlled by `HEATING_ML_FIREPLACE_LAG_HOURS` (default `"1,2"`)
  - Dynamic TV lag windows (`tv_lag_30m`, `tv_lag_1h`, …) controlled by `HEATING_ML_TV_LAG_HOURS` (default `"0.5,1"`)
  - Regex-based feature extraction in inference model handles any `fireplace_lag_Xh/m`, `tv_lag_Xh/m`, `pv_forecast_Xh`, `AT_roh_Xh` pattern without hardcoding individual names
- **New config vars**: `HEATING_ML_COLD_THRESHOLD_C`, `HEATING_ML_CALIBRATION_START_DATE`, `HEATING_ML_AT_FORECAST_HOURS`, `HEATING_ML_PV_FORECAST_HOURS`, `HEATING_ML_FIREPLACE_LAG_HOURS`, `HEATING_ML_TV_LAG_HOURS`, `HEATING_ML_CORRECTION_MODEL_PATH`, `HEATING_ML_CORRECTION_METADATA_PATH`, `HEATING_ML_MIN_TRAINING_SAMPLES`, `HEATING_ML_LABEL_HORIZON_H`, `HEATING_ML_BLEND_MIN_R2`
- `_parse_heating_start_date()` helper in `config.py` (mirrors `_parse_cooling_start_date`)
- 60 new unit tests across 3 test files (calibration, inference, blend dispatch)
- Documentation: section 7 of `docs/HEATING_CORRECTION_PHYSICS_VS_ML_ANALYSIS.md` updated with final feature list, label construction, and blend formula

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
