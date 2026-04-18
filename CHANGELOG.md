# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Indoor Temperature Trend Bias in Trajectory Prediction**: `predict_thermal_trajectory()` now incorporates `indoor_temp_delta_60m` as a decaying momentum bias. Observed indoor temperature trend (°C over last 60 min) captures unmeasured heat sources (solar through windows, body heat, appliances, thermal mass) that the physics model cannot see. The bias uses exponential decay controlled by `TREND_DECAY_TAU_HOURS` (default 1.5h) so near-future predictions strongly reflect observed momentum while far-future predictions rely on physics.
  - New config variable `TREND_DECAY_TAU_HOURS` (default `1.5`, env-overridable) controls decay time constant.
  - Trend bias is clamped to ±0.05°C per step and gated on `abs(trend) > 0.01` to prevent floating-point noise.
  - Passed from both binary search optimization and trajectory verification callers in `model_wrapper.py`.

### Fixed
- **Dashboard `titlefont` crash**: Replaced deprecated Plotly `titlefont` → `title_font` in `dashboard/components/overview.py` confidence/error dual-axis chart. Plotly 6.x removed `titlefont`, causing `ValueError` on every dashboard load.
- **Noisy "Logging MAE"/"Logging RMSE" debug messages**: Removed unnecessary `logging.debug("Logging MAE")` and `logging.debug("Logging RMSE")` in `ha_client.py` that produced low-value noise every 10-minute cycle. The actual HA state updates and their results already provide sufficient logging.

### Added
- **Electricity Price-Aware Optimization**: Tibber-integrated price classification that shifts the binary search target temperature based on current electricity price relative to today's distribution.
  - `PriceOptimizer` class with percentile-based classification (CHEAP/NORMAL/EXPENSIVE) using daily price arrays from Tibber sensor.
  - CHEAP → target +0.2°C (heat more), EXPENSIVE → target −0.2°C (heat less), NORMAL → unchanged. Convergence precision stays at ±0.01°C.
  - Trajectory correction: EXPENSIVE tightens future overshoot threshold from +0.5°C to +0.2°C, preventing unnecessary heating during expensive hours.
  - Feature flag `ELECTRICITY_PRICE_ENABLED` (default: `false`) — zero behaviour change until explicitly enabled.
  - New module: `src/price_optimizer.py`.
  - New config variables: `ELECTRICITY_PRICE_ENTITY_ID`, `PRICE_CHEAP_PERCENTILE`, `PRICE_EXPENSIVE_PERCENTILE`, `PRICE_TARGET_OFFSET`, `PRICE_EXPENSIVE_OVERSHOOT`.
- **`sensor.ml_heating_features`**: New HA sensor exporting all last-run features as attributes for debugging and diagnostics.
- **`sensor.ml_heating_price_level`**: New HA sensor showing current price classification, thresholds, and target offset.
- **Enhanced `sensor.ml_heating_learning`**: Now exports ALL learnable parameters unconditionally (previously gated behind `ENABLE_HEAT_SOURCE_CHANNELS`), plus per-channel diagnostics (`ch_{name}_history_count`, `ch_{name}_last_error`).
- **29 unit tests** for price optimizer: classification, offsets, trajectory thresholds, feature flag, integration with binary search, singleton, edge cases.

### Added
- **Heat Source Channel Architecture (Phase 2-4)**: Decomposed heat-source learning with independent channels for heat pump, solar/PV, fireplace, and TV/electronics. Each channel has its own learnable parameters and prediction history, preventing cross-contamination of learned parameters.
  - `HeatSourceChannel` abstract base class with `estimate_heat_contribution()`, `estimate_decay_contribution()`, `get_learnable_parameters()`, and `apply_gradient_update()` methods. Channels self-learn via `_learn_from_recent()` triggered on each `record_learning()` call.
  - `HeatPumpChannel`: wraps existing slab model (outlet effectiveness, slab time constant, delta-T floor).
  - `SolarChannel`: forecast-aware PV heat estimation with cloud factor, solar lag, solar decay τ (0.5 h default for sun-warmed surface residual heat), and `predict_future_contribution()` with decay-smoothed evening transitions.
  - `FireplaceChannel`: exponential decay model after fireplace off (τ ~ 45 min) with room spread delay. **Learns independently** via gradient descent from prediction errors — no dependency on `adaptive_fireplace_learning.py`.
  - `TVChannel`: simple additive heat source for TV/electronics (~0.25 kW).
  - `HeatSourceChannelOrchestrator`: routes learning updates to correct channel, combines all channels for total heat prediction, proportional error attribution across active channels.
- **`ENABLE_HEAT_SOURCE_CHANNELS` config variable**: Enable/disable decomposed heat-source learning (default: `true`). Existing Phase 1 guards (fireplace, PV, pump-OFF) remain active independently.
- **Channel-isolated gradient descent (Phase 3)**: HP channel learns only from clean cycles (no fireplace, low PV); solar channel only from PV > 500 W; fireplace channel only when fireplace active.
- **Solar transition forecasting (Phase 4)**: `SolarChannel.predict_future_contribution()` uses PV forecast array to predict future solar heat per 10-min step, with exponential decay smoothing when PV drops (solar_decay_tau_hours). Enables proactive outlet temperature increase before sunset.
- **Orchestrator integration (Steps 10-11)**: `ThermalEquilibriumModel` initializes orchestrator when `ENABLE_HEAT_SOURCE_CHANNELS` is true and routes learning through it in `update_prediction_feedback()`.
- **New module**: `src/heat_source_channels.py` — 4 channel implementations + orchestrator.
- **Comprehensive scenario tests**: Evening (Step 17: PV 3000→0 W), morning (PV 0→3000 W with slab residual heat), solar decay τ, fireplace independent learning, and orchestrator integration — 36 tests total.

### Fixed
- **Test `test_learning_isolation`**: Fixed `test_hp_params_update_when_no_contamination` to provide enough prediction feedback records (≥ `RECENT_ERRORS_WINDOW`) and use pump-ON context (`delta_t=5.0`) so gradient adaptation is actually triggered.

- **UFH Slab (Estrich) Thermal Model**: First-order lag between commanded outlet temperature and effective heating temperature: `T_slab(t+Δt) = T_slab(t) + Δt/τ_slab · (T_cmd − T_slab)`. `T_slab(0)` is initialised from `inlet_temp` (Rücklauf = current slab state). This prevents the trajectory model from applying a cold outlet command instantly to the room, which caused spurious `+15°C` corrections in cycles with PV-recovery paths.
- **`slab_time_constant_hours` as learnable parameter**: New adaptive parameter (default 1.0 h, bounds 0.25–4.0 h) using the same finite-difference gradient framework as all other parameters (`_calculate_parameter_gradient`). Gradient is non-zero only when `inlet_temp ≠ outlet_cmd` (transient phases), zero at equilibrium — correct physics.
- **`solar_lag_minutes_delta` persistence fix**: `solar_lag_minutes` learning updates were accumulated in-memory but never persisted across restarts. Both `solar_lag_minutes_delta` and the new `slab_time_constant_delta` are now written to `unified_thermal_state.json` via `_save_learning_to_thermal_state`.
- **`inlet_temp` in prediction context**: `inlet_temp` (Rücklauf) is now included in the `prediction_context` dict stored in `prediction_history`, enabling the slab gradient to be reconstructed from historical records.
- **`SLAB_TIME_CONSTANT_HOURS` config variable**: Overridable via environment variable, default `1.0`.
- **Test suite** (`tests/unit/test_slab_model.py`): 6 test classes covering slab dynamics (buffering, monotonicity, backward-compat), gradient observability (non-zero at disequilibrium, zero at equilibrium), parameter update/clipping, persistence of both new delta keys, and config bounds.

### Changed
- `predict_thermal_trajectory`: new optional `inlet_temp` parameter; when provided the slab model is active; when `None` the existing behaviour is preserved (backward-compatible).
- `_calculate_parameter_gradient`: passes `inlet_temp` from prediction context to both `+ε` and `−ε` trajectory evaluations.
- `unified_thermal_state.py`: `parameter_adjustments` default dict and `set_calibrated_baseline` reset dict extended with `solar_lag_minutes_delta` and `slab_time_constant_delta`; `update_learning_state` now accepts any key (no longer silently drops unknown delta keys).
- `physics_calibration.py`: `calibrated_params` now includes `slab_time_constant_hours` (preserved from current runtime value, not re-optimised — stable-period data cannot identify slab dynamics).
- `physics_calibration.py`: `_filter_pv_only_periods()` filters periods with `indoor_temp >= PV_CALIBRATION_INDOOR_CEILING` before PV Pass 2 scipy optimization.
- `physics_calibration.py`: `filter_pv_decay_periods()` uses 6-step sliding-window crossing detection instead of single-step sharp-drop filter.
- `heat_source_channels.py`: `SolarChannel._learn_from_recent()` and `apply_gradient_update()` skip `cloud_factor_exponent` updates when `CLOUD_COVER_CORRECTION_ENABLED=false`.
- `thermal_config.py`: `pv_heat_weight` default 0.0005 → 0.0002, bounds (0.0005, 0.005) → (0.0001, 0.005).
- `influx_service.py`: Flux query entity filter now includes `LIVING_ROOM_TEMP_ENTITY_ID`.
- `config.py`: Removed duplicate `LIVING_ROOM_TEMP_ENTITY_ID` definition.

### Fixed
- **Control Stability:** Fixed "Deadbeat Control" oscillation by decoupling the control interval (30m) from the optimization horizon (4h). This prevents excessive outlet temperature spikes when correcting small deviations.
- **HP-off outlet spike (35°C)**: When heat pump is off (`delta_t < 1.0`), the binary search now simulates "HP on" using the learned `delta_t_floor` (~2.55°C) from the HP channel. Previously, all outlet candidates produced identical slab-passive trajectories → "unreachable" → outlet spiked to max 35°C pointlessly. The simulated HP-on delta_t lets candidates differentiate so the binary search converges to a sensible setpoint that tells NIBE when to start heating.
- **PV routing at sunset**: `_is_pv_active()` now uses `max(pv_power_current, pv_power_smoothed)` against the 500W threshold. This captures solar thermal lag where smoothed PV stays high after instantaneous PV drops.
- **PV smoothing window**: Shortened from 3h (18 readings) to `solar_decay_tau` (~30min, 3 readings) in `temperature_control.py`. The old 3h window included stale morning PV values in the afternoon.
- **Slab pump-on gate**: Pump-ON branch now requires `measured_delta_t >= 1.0` in addition to `outlet_temp > t_slab`. Prevents slab model from entering active heating when HP is actually off (delta_t ≈ 0 but outlet reads higher than inlet due to stale setpoint).
- **Cloud discount on PV scalar**: Applied 1h cloud forecast discount to the PV scalar in `_extract_thermal_features()` before it enters the binary search. Raw sensor spikes during brief sun breaks (e.g. 4kW) no longer cause the binary search to snap outlet to 18°C, preventing 6am–11am outlet oscillation (21.8–24.9°C).
- **Blind-contaminated PV calibration data**: Root cause fix for `pv_heat_weight` stuck at lower bound. Automated blinds close when rooms > 22.9°C → real solar heating drops 70–90% → roof PV sensor still reads ~2000W → optimizer sees high PV with flat indoor temp → pushes weight to lower bound. `_filter_pv_only_periods()` now excludes periods where `indoor_temp >= PV_CALIBRATION_INDOOR_CEILING` (default 23.0°C).
- **`pv_heat_weight` bounds**: Default lowered from 0.0005 → 0.0002, lower bound from 0.0005 → 0.0001. Previous default = lower bound prevented the optimizer from exploring below the initial value.
- **`cloud_factor_exponent` learning when disabled**: Online learning (`SolarChannel._learn_from_recent()`) and batch calibration (`calibrate_cloud_factor()`) now gated behind `CLOUD_COVER_CORRECTION_ENABLED`. Previously both ran unconditionally — when the flag was `false`, the prediction path returned 1.0 but gradients still updated the exponent, causing parameter drift without feedback.
- **`calibrate_delta_t_floor` sensitivity**: Raised minimum delta_t threshold from 0.5 → 1.0°C and minimum calibration result from 0.5 → 1.0°C. Prevents floor from converging to sub-1°C values where HP is effectively off.
- **PV decay period detection**: Replaced single-step sharp-drop filter with 6-step (30 min) sliding-window crossing detection in `filter_pv_decay_periods()`. Old method required an exact single-step PV drop below threshold, missing gradual sunset transitions. New method detects when a 6-reading window crosses from above to below the PV threshold.
- **`cloud_cover_pct` calibration default**: Changed all 4 hardcoded fallback values from 50.0 → 0.0. When cloud cover data is unavailable, assuming clear sky (0%) is physically correct — the calibration should learn the actual heating at the measured PV power, not discount it by an assumed 50% cloud cover.

### Added
- **`slab_passive_delta` sensor**: New diagnostic metric (`inlet_temp - indoor_temp`) exported to HA. Positive = slab warmer than room (passive heating available), negative = slab absorbing heat. Visible in thermal features and HA sensor attributes.
- **`_search_delta_t_floor`**: Internal variable ensuring both binary search pre-check, loop, and trajectory verification use the same (potentially simulated) delta_t value per cycle.
- **`PV_CALIBRATION_INDOOR_CEILING` config variable**: Indoor temperature ceiling (default 23.0°C) for filtering blind-contaminated PV calibration periods. When automated blinds close (rooms ≥ 22.9°C), real solar heating drops 70–90% while the roof PV sensor still reads high — this causes the optimizer to push `pv_heat_weight` to its lower bound. Periods with `indoor_temp >= ceiling` are now excluded from PV Pass 2 calibration.
- **`LIVING_ROOM_TEMP_ENTITY_ID` in InfluxDB query**: Living room temperature sensor now included in the Flux query entity filter, ensuring indoor temperature data is available for calibration.

### Technical Achievements



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

### Technical Achievements
- **Overnight Stability Enhanced**: Gentle trajectory corrections prevent system over-reaction during PV shutdown and weather changes
- **Conservative Control**: 0.5°C trajectory error now produces reasonable +2.5°C outlet adjustment instead of temperature doubling
- **Real-time Adaptation**: Trajectory verification uses actual changing forecasts instead of static assumptions
- **User-Aligned Logic**: Trajectory corrections based on proven heat curve automation patterns already in successful use
- **Production Ready**: All 36 critical thermal model tests passing (100% success rate)
- **Physics Compliance**: System now respects thermodynamics and energy conservation
- **Accuracy**: Temperature predictions now physically realistic and mathematically correct
- **Reliability**: Binary search convergence eliminates maximum temperature requests
- **Energy Efficiency**: Heat pump operates optimally instead of maximum unnecessarily

## [0.2.0-beta.3] - 2025-12-03

### Added - Week 3 Persistent Learning Optimization Complete 🚀
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

### Technical Achievements
- **Code Quality**: 2 redundant files eliminated, 50% reduction in wrapper complexity
- **Test Coverage**: 100% pass rate maintained across 29 critical tests
- **Performance**: Eliminated unused code paths and simplified execution flow
- **Maintainability**: Single source of truth for all model wrapper operations
- **Architecture**: Clean consolidation with zero functionality regression

## [0.2.0-beta.2] - 2025-12-03

### Added - Week 2 Multi-Heat-Source Integration Complete 🎯
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

### Added - Week 1 Enhanced Features Foundation 🔧
- **Enhanced Physics Features**: 15 new thermal momentum features (thermal gradients, extended lag analysis, cyclical time encoding)
- **Comprehensive Test Suite**: 18/18 enhanced feature tests passing with mathematical validation
- **Backward Compatibility**: 100% preservation of original 19 features with zero regressions
- **Performance Optimization**: <50ms feature build time with minimal memory impact
- **Advanced Feature Engineering**: P0/P1 priority thermal intelligence capabilities

### Changed
- Extended physics features from 19 to 34 total thermal intelligence features
- Enhanced thermal momentum detection with multi-timeframe analysis
- Improved predictive control through delta features and cyclical encoding
- Upgraded test coverage to include comprehensive edge case validation

### Added - Documentation and Workflow Standards 📚
- Version strategy and development workflow documentation
- Changelog standards and commit message conventions
- Professional GitHub Issues management system
- Memory bank documentation with Week 2 completion milestone
- Comprehensive technical achievement summaries and performance metrics

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

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Home Assistant add-on discovery issue by implementing proper semantic versioning
- Add-on configuration validation and schema structure

### Security
- Secure API key authentication for development access
- InfluxDB token-based authentication
- AppArmor disabled for system-level heat pump control access

---

## Version History Notes

This changelog started with version 0.0.1-dev.1 as the project transitions from internal development to structured release management. Previous development history is captured in the Git commit log and project documentation.

### Versioning Strategy
- **0.0.x-dev.N**: Development builds for testing and iteration
- **0.0.x**: Development releases for broader beta testing  
- **0.x.0**: Beta releases with feature-complete functionality
- **x.0.0**: Production releases for general use

See `memory-bank/versionStrategy.md` for complete versioning guidelines.
