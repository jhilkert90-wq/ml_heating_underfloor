# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **CI: GitHub Actions workflow versions and architecture**: Updated `.github/workflows/build.yaml` to use newer versions of GitHub Actions (`actions/checkout@v4`, `docker/login-action@v3`, `docker/setup-buildx-action@v3`, `docker/build-push-action@v6`) and improved architecture handling. Native runners are now used per architecture (no QEMU required), and build cache is separated per arch for improved reliability and performance.

### Added
- **PV Feature Key Contract documentation**: Added a canonical, highly-visible `⚠️ AI MODEL NOTICE — PV Feature Key Contract` section at the top of `memory-bank/systemPatterns.md` that maps every consumer of `pv_now_electrical` / `pv_forecast_electrical_*` (electrical, raw) vs `pv_now` / `pv_forecast_{h}h` (thermal, corrected). Includes a table, explicit rules, anti-patterns, and source-code citations.
- **ML Cooling Guide PV warning**: Added a `⚠️ PV Feature Key Contract` warning block to `docs/ML_COOLING_MODEL_GUIDE.md` clarifying that `OverheatingPredictor` and ML cooling paths must use thermal keys.
- **PV key contract regression tests** (`tests/unit/test_overheating_predictor.py`): 5 new tests in `TestPVKeyContract` that:
  - Confirm `OverheatingPredictor` reads `pv_now` (thermal) for its guard check and works without `pv_now_electrical`
  - Confirm trajectory PV forecast is built from `pv_forecast_{h}h` (thermal) even when electrical keys are absent
  - Confirm `pv_now_electrical` alone (with `pv_now=0`) cannot pass the guard
  - Confirm `HLCCycle._build_cycle()` reads `pv_now_electrical` (not `pv_now`) from context
  - Confirm `_build_cycle()` defaults `pv_now_electrical` to 0.0 when the key is absent
- **Reviewer-follow-up regression hardening**: Strengthened 2 `TestPVKeyContract` tests to assert `predict_thermal_trajectory()` call kwargs (`pv_power`, `pv_forecasts`) directly, preventing false positives from mocked trajectory outputs.

### Fixed
- **Merge conflict resolution (latest sync)**: Resolved newly introduced conflicts while merging latest `origin/main` into the PR branch, reconciling `memory-bank/activeContext.md` and `memory-bank/progress.md` and preserving entries from both branches.
- **[HIGH] Cooling observation buffer durability gap before label maturity**: Pre-cooling loop now persists the observation buffer after every `push_pending()`/`resolve_labels()` cycle, so pending samples and evolving label state survive restarts even before labels mature.
- **[HIGH] Remaining PV roll scale drift in cooling ML inference**: Added raw `pv_power_history_electrical` to physics features and updated cooling ML `pv_roll_*` extraction to prefer the raw electrical history used during training.
- **Merge conflict resolution for PR branch**: Resolved conflicts against `origin/main` in `CHANGELOG.md`, `memory-bank/activeContext.md`, and `memory-bank/progress.md` by preserving entries from both branches and removing conflict markers.
- **[CRITICAL] Missing `scikit-learn` dependency**: Added `scikit-learn>=1.0.0` to `requirements.txt`; its absence caused `roc_auc_score` import to silently fail, writing `null` AUC to model metadata.
- **[HIGH] Wrong PV/forecast feature keys in pre-cooling integration tests**: `_make_features()` in `test_pre_cooling_integration.py` used `pv_now_electrical`, `pv_forecast_electrical_{h}h`, and `outdoor_forecast_{h}h` — none of which `OverheatingPredictor` reads. Fixed to `pv_now`, `pv_forecast_{h}h`, and `temp_forecast_{h}h` so the PV guard is actually exercised.
- **[HIGH] Missing PV contract assertion in pre-cooling integration tests**: Added an explicit assertion that `OverheatingPredictor` passes the corrected PV values into `predict_thermal_trajectory()` as `pv_power` and `pv_forecasts`, so PV-guard regressions are caught.
- **[MEDIUM] Training/inference PV scale mismatch for LGBM cooling model**: `_extract_feature()` in `cooling_ml_model.py` used thermally-corrected `pv_now`/`pv_forecast_{h}h` at inference while training used raw electrical watts. Now prefers `pv_now_electrical`/`pv_forecast_electrical_{h}h` with graceful fallback to corrected values.
- **[LOW] Incorrect `PRE_COOL_LEAD_TIME_HOURS` fallback in calibration**: Hardcoded default was `8.0` in `cooling_ml_calibration.py`; corrected to `3.0` to match `config.py`.
- **[LOW] Cooling observation buffer not persisted between cycles**: Buffer was only saved on successful retrain. Now saved whenever `resolve_labels()` returns newly-labeled observations, preventing data loss on restart.
- **[LOW] Stale cooling ML test defaults**: Updated cooling ML test fixtures and fake configs to use the runtime default `PRE_COOL_LEAD_TIME_HOURS=3.0` instead of the old `8.0`, keeping calibration tests aligned with production behavior.

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
