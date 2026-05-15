# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Binary search range-collapse bypass** (`model_wrapper.py`): when the binary search range collapsed to < 0.05 °C (e.g. saturating at 21 °C min outlet due to high PV forecast), the early-exit path returned immediately without running `_verify_trajectory_and_correct`. The correction layer is now called before returning in this path, matching the converged and non-converged paths.
- **`projected_indoor` linear over-estimation** (`model_wrapper.py`, both `_calculate_physics_based_correction` and `_calculate_physics_newton_correction`): the self-correction gate computed `projected_indoor = current + TRAJECTORY_STEPS × trend` (linear). At H=4 h this overestimated the room's natural travel by 2–3× compared to the actual trajectory, causing legitimate overshoot/undershoot corrections to be skipped. Replaced with the exponential-decay integral `current + trend × τ × (1 − exp(−H/τ))` using `TREND_DECAY_TAU_HOURS` (default 1.5 h), matching exactly how the trajectory model accumulates the trend bias.
- **Newton `else` branch threshold inconsistency** (`model_wrapper.py`, `_calculate_physics_newton_correction`): the internal fallback branch used `reaches_target_at > cycle_hours` (1× cycle) while the outer gate filters at `cycle_hours + tolerance_hours` (3× cycle). Aligned the Newton branch to `cycle_hours + tolerance_hours` for consistency.
- **Fragile exact float equality** (`model_wrapper.py`, `_calculate_physics_newton_correction`): replaced `if temp_error == 0.0` with `if abs(temp_error) < 1e-6` to avoid potential floating-point precision issues.
- **Newton uses `S(H)` instead of `S(t_worst)`** (`model_wrapper.py`, `_calculate_physics_newton_correction`): sensitivity was always evaluated at the full horizon H via `S_H = [η/(η+U)] × [1 − exp(−H/τ_room)]`, but ε was measured at the worst trajectory point which may occur at `t_worst < H`. Since `S(H) > S(t_worst)`, dividing by `S_H` under-corrects systematically — worst when PV drives a mid-horizon overshoot (solar gain peaks early in the day). Fixed by looking up the time-step index of the worst point and evaluating `S(t_worst)` there. The trajectory `times` array is used when present; otherwise the step size is inferred as `H / n_steps`.

### Added
- **Physics Newton-Step Heating Correction** (`_calculate_physics_newton_correction()` in `model_wrapper.py`): implements the exact formula `ΔT_outlet = ε / S_H` where `S_H = [η/(η+U)] × [1 − exp(−H/τ_room)]`. Symmetric for under- and overshoot, horizon-aware, and ~2× more accurate than the legacy formula after calibration. Shares all boundary-violation guards and clamp logic with the existing method.
- **ML Correction Stub** (`_calculate_ml_correction()` in `model_wrapper.py`): placeholder that warns and falls back to the Newton step, ready to be replaced with a LightGBM regressor once sufficient historical data is available.
- **`HEATING_CORRECTION_MODE` config variable** (`src/config.py`): selects the active correction algorithm at runtime. Accepted values: `"legacy"` (default), `"physics"`, `"ml"`.
- **Home Assistant dropdown selector** (`ml_heating_underfloor/config.yaml`): `heating_correction_mode: "list(legacy|physics|ml)"` renders as a dropdown in the HA add-on UI — no text entry required.
- **config_adapter wiring** (`config_adapter.py`): `heating_correction_mode` option is now mapped to `HEATING_CORRECTION_MODE` env var in `convert_addon_to_env()`.
- **Translation entry** (`ml_heating_underfloor/translations/en.yaml`): descriptive label and tooltip for the new dropdown.
- **11 unit tests** (`tests/unit/test_heating_correction.py`): cover Newton undershoot/overshoot accuracy (±0.01°C tolerance), degenerate-S_H fallback, clamp guard, mode dispatch for all three modes, and config_adapter mapping.

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
