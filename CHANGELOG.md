# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Runtime-replay epsilon tuning**: Finalized finite-difference epsilon calibration using sensitivity analysis and the calibration review notebook so each runtime-reachable parameter produces ΔT ≈ 0.1–0.3°C while remaining in the linear regime
  - `THERMAL_TIME_CONSTANT_EPSILON`: 2.0 → 0.2
  - `HEAT_LOSS_COEFFICIENT_EPSILON`: 0.005 → 0.008
  - `OUTLET_EFFECTIVENESS_EPSILON`: 0.05 → 0.1
  - `TV_HEAT_WEIGHT_EPSILON`: 0.05 → 0.1
  - `SLAB_TIME_CONSTANT_EPSILON`: 0.5 → 1.595
  - `SOLAR_LAG_EPSILON`: kept at 5.0 because the current runtime replay path cannot reach the target signal window (best sweep at 22.5 only yields ΔT ≈ 0.0097°C)
- **Consolidated epsilon constants**: Added `SOLAR_LAG_EPSILON` and `SLAB_TIME_CONSTANT_EPSILON` to `PhysicsConstants`; wrapper methods now reference constants instead of hardcoded values
- `_is_heat_pump_active()` signature: now reads `climate_mode` from context dict
- `_resolve_delta_t_floor()`: accepts optional `climate_mode` parameter for mode-aware behavior
- `predict_thermal_trajectory()`: accepts `climate_mode` kwarg, propagated through binary search
- `COOLING_DEFAULTS`: `pv_heat_weight` 0.0003 → 0.002, `slab_time_constant_hours` 0.8 → 3.19
- `COOLING_BOUNDS`: `slab_time_constant_hours` upper bound 2.5 → 8.0
- Cooling inlet guard replaced with full cooling cycle gate state machine
- `main.py` learning context: removed inline `heat_pump_active` calculation, uses `_is_heat_pump_active()` via `climate_mode` in context

### Added
- `target_indoor_temp_cooling_entity` now visible in the Home Assistant add-on configuration UI with label and description (`config.yaml` schema + `translations/en.yaml`)
- **Cooling cycle gate persistence**: Gate state (`running`/`recovery`) persisted in cooling JSON and restored on mode switch and add-on restart
- `cooling_cycle_gate` key in cooling operational state schema (`unified_thermal_state_cooling.py`)
- 8 new review-round regression tests in `test_cooling_bugfixes.py` (transient filter, gate persistence, delta_t default, duplicate keys, target validation)
- **Cooling cycle gate** (Bug 11): State machine with `RUNNING`/`RECOVERY` states prevents HP short-cycling in cooling mode using gradient-based transitions with existing `cooling_shutdown_margin_k` parameter
- **`TARGET_INDOOR_TEMP_COOLING_ENTITY_ID`** (Bug 5): Separate target temperature entity for cooling mode in `config.py`, `config.yaml`, `.env_sample`, and `config_adapter.py`
- **Early climate mode detection** (Bug 3): Climate mode determined before learning step so learning context uses correct mode
- Comprehensive cooling bugfix test suite (`test_cooling_bugfixes.py`) with 19 tests

### Fixed
- **Duplicate keys in learning `prediction_context`**: Removed duplicate `inlet_temp` and `delta_t` entries from the dict literal in `main.py`; Python silently used the last value
- **Transient drop filter fires incorrectly in cooling mode**: Filter now skipped when `climate_mode == "cooling"` — in cooling a temp drop is normal (HP is cooling); a door opening causes a RISE
- **Cooling cycle gate state lost on restart**: `_cooling_cycle_state` persisted to cooling JSON; restored in `__init__` and `set_climate_mode("cooling")`
- **`_search_delta_t_floor` stale/zero on binary search early exit**: Early exit now sets `_search_delta_t_floor = None`; gate falls back to the thermal model's learned delta_t floor instead of optimistic 0.0
- **Cooling target entity not validated**: `_cooling_target` from HA now wrapped in `float()` with `try/except` to reject non-numeric values ("unavailable" is already `None`-filtered by `ha_client.get_state`)
- **Test `test_cooling_binary_search_uses_cooling_bounds` used old bounds**: Assertion changed from `effective_min = COOLING_CLAMP_MIN_ABS + SHUTDOWN_MARGIN` to `COOLING_CLAMP_MIN_ABS`
- **Cooling binary search uses full outlet range**: `get_outlet_bounds("cooling")` now returns `(COOLING_CLAMP_MIN_ABS, COOLING_CLAMP_MAX_ABS)` without adding `COOLING_SHUTDOWN_MARGIN_K` — the post-search RUNNING/RECOVERY gate handles HP safety
- **Removed inlet-guard tightening from binary search bounds**: `_calculate_required_outlet_temp()` no longer clamps `outlet_max` to `inlet − MIN_COOLING_DELTA_K` before the search; the search explores the full range and the cycle gate prevents HP short-cycling after convergence
- **Previous-cycle learning context after mode changes**: Persist `last_climate_mode` and `last_target_indoor_temp`, reuse them during next-cycle online learning, and switch the wrapper to the previous cycle's mode before feedback learning runs
- **Cooling forecast demand semantics in feature building**: `build_physics_features()` now accepts climate mode plus a resolved target override, uses the cooling target consistently, and computes forecast demand as `forecast - target` in cooling mode
- **HP active detection in cooling** (Bug 1): `_is_heat_pump_active()` now mode-aware — uses `HEATING_MIN_THERMAL_POWER_KW` (0.5) for heating, `COOLING_MIN_THERMAL_POWER_KW` (-0.5) for cooling; delta_t and outlet checks inverted for cooling
- **Slab model pump_on gate** (Bug 1b): Slab pump-on detection in `thermal_equilibrium_model.py` now works for cooling (outlet < t_slab, delta_t <= -1.0)
- **HP-OFF delta_t floor substitution** (Bug 1c): In cooling mode, HP-off detected when delta_t > -1.0 (not < 1.0), simulated delta_t is negative for binary search
- **PV surplus offset inverted in cooling** (Bug 6): High PV now lowers target (more cooling) instead of raising it
- **Price offset inverted in cooling** (Bug 7): Cheap electricity now lowers target in cooling mode
- **`heating_demand_forecast` hardcoded 21°C** (Bug 8/9): Replaced with actual `target_temp_f` from sensor data
- **Cooling `pv_heat_weight` 7× too low** (Bug 4): Set to 0.002 (same as heating — building property)
- **Cooling `slab_time_constant_hours` wrong** (Bug 2): Set to 3.19h (same as heating — same slab mass)
- **HP channel never learns in cooling**: All fixes combined enable HP learning in cooling mode (previously 0 history entries after 69 cycles)

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
