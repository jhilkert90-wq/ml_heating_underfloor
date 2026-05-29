# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Cooling ML model: 23 new features** ported from heating model and newly implemented:
  - HA context features: `wind_speed`, `living_room_temp`, `fireplace_on`, `tv_on` + dynamic rolling lags (`fireplace_lag_30m/1h/2h`, `tv_lag_30m/1h`)
  - Derived physics features: `heat_loss_driving_force`, `indoor_temp_gradient`, `indoor_margin_rate`, `delta_T_indoor_lag1`, `d_inlet_temp_60min`, `is_equilibrium`, `thermal_power_rolling_1h`, `is_overshoot`, `is_hp_active`, `is_weekend`, `heat_loss_interaction`
  - Solar/shading features: `solar_thermal_proxy`, `shading_proxy`, `pv_forecast_delta`
  - Trajectory-derived features: `traj_predicted_error`, `traj_convergence_rate`, `traj_reaches_target_hours`, `traj_overshoot_magnitude`, `traj_equilibrium_gap` — vectorized analytical Newton-decay approximation at calibration, OverheatingPredictor trajectory injection at inference
- **Cooling HA entity fetch expanded**: `fetch_historical_data_for_calibration(purpose="cooling")` now fetches 11 entities (was 7) — adds wind_speed, fireplace, TV, living_room_temp
- **Trajectory injection for cooling LGBM inference**: `cycle_routes.py` injects OverheatingPredictor's trajectory result into CoolingMLModel features, enabling physics-ML bridge
- **Cooling ML Analysis Notebook** (`notebooks/analysis/09_cooling_ml_analysis.ipynb`): Full analysis of overheating classifier with 17 new derivable features, incremental pruning, regression alternative (regression wins: AUC 0.9502 vs 0.9431, F2 0.9492 vs 0.9424), Optuna HPO (AUC 0.9582, MAE 0.0827°C), threshold sensitivity analysis. Key finding: regression approach predicting `delta_indoor_8h` then thresholding at 22.93°C outperforms direct binary classification.
- **Optimized ML Training Notebook** (`notebooks/analysis/08_heating_ml_optimized.ipynb`): Residualized label architecture with 53 features, outlier filtering, incremental pruning, Optuna HPO, 5-fold TS-CV, SHAP analysis, sensor noise floor analysis, outlet temp adjustment verification
- **Residualized label as primary architecture**: `adjusted_label = -(T_future - T_current) / S_H`; at inference: `full_correction = model.predict(X) - indoor_margin / S_H`. Achieves adj R²=0.9755, recon R²=0.9042, MAE=0.1277°C
- **Outlier filtering with forward-looking label contamination**: Removes fireplace (4.4%), window-open (3.4%), PV spikes — 85,078→78,483 rows (7.8% removed). Key enabler for R²>0.90
- **Incremental PI-based pruning**: Drops features one-by-one (worst PI first, threshold PI<0.001). Removed `shortwave_radiation_wm2` and `pv_roll_1h` — 55→53 features, MAE improved 0.1304→0.1277
- **Sensor noise floor analysis**: Measured actual sensor resolutions (indoor_temp: 0.002°C, VLT/RLT: 0.02°C, AT: 0.01°C). Label quantization = 0.004°C — sensor accuracy is NOT the bottleneck
- **5 new engineered features**: `cumulative_Q_wp_4h` (142 splits), `AT_forecast_trend`, `thermal_momentum`, `indoor_accel`, `pv_cumulative_4h`
- **Outlet temperature adjustment verification**: Worked examples showing exact correction formula for undershoot/overshoot scenarios (±1.22°C per 0.6°C margin)
- **Heating ML Standalone Training Notebook** (`notebooks/analysis/07_heating_ml_standalone.ipynb`): Offline LightGBM training with HA_LOG-style output, commentable feature list for ablation, leave-one-out and group ablation analysis, `indoor_temp` dominance diagnosis with decorrelation experiments, SHAP analysis (optional), residual-based missing-feature analysis, and full diagnostic dashboard
- **Open-Meteo solar radiation enrichment** in notebook: Fetches historical `shortwave_radiation` (W/m²) from Open-Meteo archive API for 48.928°N/10.069°E, interpolates hourly data to CSV resolution (~6 min), adds `shortwave_radiation_wm2` as 52nd feature. Includes HA REST sensor configuration for live integration.
- **Residualized label experiment** (Section 11c-bis): Subtracts trivial `indoor_margin/S_H` component to isolate temperature-change perturbation; reveals `living_room_temp` as hidden dominant proxy
- **New engineered features experiment** (Section 15b): Tests `cumulative_Q_wp_4h` (112 splits), `AT_forecast_trend`, `indoor_accel`, `pv_cumulative_4h`, `thermal_momentum` — combined MAE improvement -0.0035

### Changed
- **Feature selection refined**: Removed `indoor_temp` and `living_room_temp` (redundant with `indoor_margin` in residualized framework); kept `indoor_margin` (physics input) and `is_overshoot` (HP mode signal)

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
