# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Cooling ML correction pipeline** — Full production cooling ML correction with calibration (`src/cooling_correction_ml_calibration.py`), inference model (`src/cooling_correction_ml_model.py`), main.py CLI/flag integration, model_wrapper dispatch, dashboard button, config.yaml, and translations
- **Residualized label architecture for heating ML** — Changed label from `-(T_future - T_target)/S_H` to `-(T_future - T_current)/S_H` with reconstruction `delta = delta - indoor_margin/S_H` at inference time
- **Forward-looking outlier filtering** — 4 filters (fireplace, window-open, PV spikes, extreme label) applied after label computation in both heating and cooling calibration
- **5 NB08-derived features** — `cumulative_Q_wp_4h`, `indoor_accel`, `AT_forecast_trend`, `pv_cumulative_4h`, `thermal_momentum` added to both heating and cooling ML pipelines
- **Cooling ML config parameters** — Full `COOLING_ML_CORRECTION_*` config block in `src/config.py` with `COOLING_CORRECTION_MODE` (default "physics")
- **Dashboard cooling ML calibration** — "Calibrate ML Cooling Correction" button, settings group, schema entries
- **Test coverage for new features** — `test_cooling_correction_ml_calibration.py`, `test_cooling_correction_ml_model.py`, `test_heating_ml_nb08_features.py` (69 new tests)

### Changed
- **Heating ML calibration** (`src/heating_correction_ml_calibration.py`) — Residualized label, forward-looking outlier filtering, 5 new features, removed `indoor_temp`/`living_room_temp` from features
- **Heating ML model** (`src/heating_correction_ml_model.py`) — Residualized reconstruction in predict(), new feature extractors, `label_type`/`s_h` metadata support
- **Model wrapper** (`src/model_wrapper.py`) — Cooling mode checks `COOLING_CORRECTION_MODE` for "ml" routing to `_calculate_cooling_ml_correction()`

### Fixed
- **Deprecated `.fillna(method="ffill")`** (`src/heating_correction_ml_calibration.py`) — Replaced with `.ffill()` for pandas 2.0+ compatibility
- **Extreme label filter dead code** (both calibration files) — Labels were clipped to ±5.0 before filtering for >8.0 (unreachable); changed filter threshold to >5.0 and moved clipping after filter
- **`COOLING_OUTLET_EFFECTIVENESS` missing from config** (`src/config.py`) — Added env-var-backed config parameter (default 0.20) so it can be overridden without code changes
- **S_H degenerate fallback** (both calibration files) — Added secondary guard: abort calibration if S_H < 0.01 even after fallback to defaults
- **Silent residualized skip** (both ML models) — Added warning log when `label_type="residualized"` but S_H ≤ 0.05 causes reconstruction to be skipped
- **Outlier filter data loss** (both calibration files) — Added early abort if outlier filters reduce dataset below 100 rows

### Added
- **NB09: Cooling ML Correction — Regression Model** (`notebooks/analysis/09_cooling_ml_correction.ipynb`) — First cooling-mode overshoot/undershoot correction model using same residualized label architecture as heating NB08. Computes regression label from binary cooling CSV: `label = -(T_future - T_target) / S_H_cooling` with S_H=0.3508. Key results: adj MAE=0.2510, R²=0.9621; recon R²=0.8530; 5-fold CV: R²=0.9654±0.014 (matches heating R²=0.9653). MAE ~2x heating (0.277 vs 0.133) due to lower S_H amplifying label magnitudes, but R² parity confirms comparable model quality. 63 features (pruning unable to drop any without >0.5% MAE regression). Optuna worse than defaults (kept pre-tuning model). Residual diagnostics show slight positive bias at high PV (>8kW) — solar gain signal not fully captured.
- **NB16: Constrained Cooling OE Calibration** (`notebooks/analysis/16_constrained_cooling_oe_calibration.ipynb`) — Calibrates OE_cooling from `cooling_training_data.csv.gz` with HLC and τ locked from heating calibration. Key results: OE_cooling≈0.20 (was 0.953, 5x too high); RMSE drops from 3.06°C (production) to 2.62°C; dual-HLC (HLC_on=0.158, HLC_off=0.032) gives additional 9.7% improvement; +2°C bias at moderate outdoor temps from unmodeled solar/internal gains, near-zero bias at AT≥28°C.

### Changed
- **Constrained cooling calibration** (`src/physics_calibration_cooling.py`) — HLC and τ are now locked from heating calibration (building/slab physics are mode-invariant). OE estimation replaced naive algebraic inversion with scipy.optimize RMSE minimisation on non-saturated cooling data. This prevents the confounded HLC/OE/τ drift that caused OE=0.953 and τ=41h in online learning.
- **Cooling parameter defaults** (`src/thermal_config.py`) — `outlet_effectiveness` default 0.90→0.20 (calibrated), `thermal_time_constant` default 3.0→4.8h (same slab), OE lower bound 0.3→0.05.
- **Test updated** (`tests/unit/test_physics_calibration_cooling.py`) — `test_tau_uses_actual_outlet_when_target_outlet_missing` → `test_tau_locked_from_heating_state` to match new constrained calibration behavior.

### Added
- **NB15 Phase D-bis: Bias-corrected cross-mode analysis** — Subtracts mode-specific a0 bias (heating=-2.098°C, cooling=+0.700°C) from residuals and re-evaluates all 5 correction methods. Key result: cross-mode RMSE drops from 2.58→1.42 (45-53% improvement across all methods), closing the gap to same-mode performance (1.42 vs 1.43). This validates the "shared LUT + per-mode offset" approach — a single solar correction LUT works for both modes once the equilibrium bias is removed. Elev×Az LUT achieves best cross-mode RMSE (1.418) — even slightly better than same-mode heating (1.432).
- **NB15: Solar Weight — Heating Training Data + Cross-Dataset Comparison** (`notebooks/analysis/15_solar_weight_heating_data_comparison.ipynb`) — Repeats NB14 analysis on heating_training_data.csv.gz (85k rows, AT<18°C). Key findings: (1) Heating training data confirms NB14: solar_w near-zero in heating mode (bias=-2.05°C), positive in cooling/passive (w_pv=0.000013-0.000019); (2) Cooling/passive subset from heating data gives even smaller weights than NB14's cooling data (0.000013 vs 0.000062) — colder outdoor temps reduce solar gain visibility; (3) West windows at low elevation again dominant (0.00496 vs 0.00085 South); (4) Cross-dataset calibration shows models don't transfer well between modes (RMSE 3.14 vs 1.43 same-mode); (5) Elev×Az LUT best for in-mode correction, single weight best for cross-mode. Implementation plan: 4 priorities from lock-solar-learning to joint recalibration. 5/6 consistency checks passed.
- **NB14: Solar Weight Calibration in Heating Mode** (`notebooks/analysis/14_solar_weight_heating_calibration.ipynb`) — Investigates pv_heat_weight oscillation (0.0003↔0.003) root cause. Key findings: (1) Solar weight is undetectable in heating mode — locked HLC/OE create -2°C equilibrium bias that masks solar signal; (2) Cooling/passive mode calibration gives w_pv=0.000062 kW/W (27x smaller than production 0.001659); (3) GHI weight drops monotonically with sun elevation (0.0099 at 5-15° → 0.0012 at 55-65°) matching Fresnel window transmission theory; (4) West windows at low elevation show 20x higher weight than South (afternoon sun horizontal penetration); (5) Hour-of-day LUT is best correction method (RMSE -0.34 vs single weight); (6) Open Meteo forecast API confirmed available for PV_forecast replacement. 6/6 consistency checks passed.
- **NB13: Physics-Based Solar Model Calibration** (`notebooks/analysis/13_physics_solar_calibration.ipynb`) — Tests whether a physics-based solar model (Erbs GHI→DNI/DHI decomposition, Kasten cloud model, Open Meteo direct DNI/DHI, astral sun positions, directional S/E/W gains, EMA thermal battery) closes the dual-HLC gap. Key results: physics solar improves RMSE by only 0.0045°C (1.3%); dual-HLC gap persists (0.022°C vs 0.021°C); HLC_on/HLC_off = 2.5x confirmed as real physical effect, not solar artifact. EMA smoothing is detrimental. Best model: erbs_total_raw (RMSE=0.332°C). Critical bug found and fixed: dt_h must be 5min (not 30min) for 5-minute training data.

### Changed
- **Phase 2 prompt updated with night-filtered HLC calibration** (`prompts/plan-cooling-calibration-nb10.prompt.md`) — Night-filtered dual-HLC calibration (HLC_on=0.327, HLC_off=0.005, ratio=65.4x) confirms dual-HLC is physical, not Beschattung artifact. HLC_off lower bound kept at 0.005. Beschattung reduces apparent heat exchange by 46% but building is genuinely well-insulated when HP fan off.

### Added
- **NB12 Section F: HLC/Beschattung Analysis** — 4 cells added to NB12 for HP-off drift analysis, visualization, night-filtered re-calibration, and interpretation. Night slope=-0.003139, Day+PV=-0.001706.
- **Analysis scripts** (`scripts/section_f_analysis.py`, `scripts/analyze_hlc_dual.py`, `scripts/analyze_hlc_beschattung.py`, `scripts/analyze_hlc_full.py`) — Standalone HLC analysis scripts for faster iteration than notebook kernel.
- **Notebook 12: Solar Parameter Replacement Analysis** (`notebooks/analysis/12_solar_parameter_replacement.ipynb`) — Determines which Open Meteo radiation variable best replaces PV_Generate/pv_forecast for cooling-mode calibration, thermal model, and ML classifier. Key results: GHI wins for thermal calibration (RMSE=0.476 vs PV=0.528); GTI has highest PV correlation (r=0.700); GHI forecast outperforms PV forecast at 1-4h horizons; PV still best for ML classifier (AUC=0.945 vs GHI=0.931). Weight conversion: ghi_weight=0.004144 kW/(W/m²), proposed bounds [0.0008, 0.040].
- **Prompt: Cooling Calibration Plan** (`prompts/plan-cooling-calibration-nb10.prompt.md`) — Implementation plan for dual-HLC, OE bounds, τ cap, and PV weight changes based on NB10 findings.
- **Prompt: Pre-Cooling Adaptation Plan** (`prompts/plan-precooling-nb11.prompt.md`) — Implementation plan for LGBM threshold, trajectory margin, RECOVERY gate, and shadow-mode based on NB11 findings.
- **Notebook 10: Cooling Thermal Calibration** (`notebooks/analysis/10_cooling_thermal_calibration.ipynb`) — Offline HLC/OE/τ calibration using `cooling_training_data.csv.gz`. Key findings: dual-HLC (HP-ON=0.146 vs HP-OFF=0.016) improves RMSE by 85.8%; cooling OE=0.19 is 4.4x lower than heating; online τ=41h inflated, scipy finds 8h.
- **Open Meteo solar radiation integration** in Notebook 10 — Fetches GHI, DNI, direct, diffuse radiation from archive API (9,504 hourly records, May 2025–May 2026). SSL bypass via `verify=False` for corporate proxy. Feature ablation: GHI (RMSE=0.751°C) slightly outperforms PV (RMSE=0.755°C); GHI correlates r=0.419 with indoor_trend vs PV's r=0.387.
- **Notebook 11: Cooling Cycle Analysis** (`notebooks/analysis/11_cooling_cycle_analysis.ipynb`) — Cycle-by-cycle analysis of 157 cooling cycles from 3 production logs. LGBM pre-cool is bimodal (std=0.23) but always triggers (min=0.417 > thr=0.345); trajectory discriminates (12% rejection); prediction MAE=0.036°C; Newton correction median=-0.9°C.
- Generator scripts for reproducible notebook creation (`scripts/create_notebook_10.py`, `scripts/create_notebook_11.py`, `scripts/create_notebook_12.py`)

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
