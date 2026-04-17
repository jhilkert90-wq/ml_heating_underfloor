# Changelog - ML Heating Underfloor

## [0.2.4] - 2026-04-17

### Added
- **Electricity Price-Aware Optimization**: Tibber-integrated price classification (CHEAP/NORMAL/EXPENSIVE) that shifts the binary search target temperature based on current electricity price relative to today's distribution. Feature flag `ELECTRICITY_PRICE_ENABLED` (default: `false`) — zero behaviour change until explicitly enabled.
- **`sensor.ml_heating_features`**: New HA sensor exporting all last-run features as attributes for debugging and diagnostics.
- **`sensor.ml_heating_price_level`**: New HA sensor showing current price classification, thresholds, and target offset.
- **Enhanced `sensor.ml_heating_learning`**: Now exports ALL learnable parameters unconditionally, plus per-channel diagnostics.
- **Heat Source Channel Architecture**: Decomposed heat-source learning with independent channels for heat pump, solar/PV, fireplace, and TV/electronics. Each channel has its own learnable parameters and prediction history.
- **`ENABLE_HEAT_SOURCE_CHANNELS` config variable**: Enable/disable decomposed heat-source learning (default: `true`).
- **Solar transition forecasting**: `SolarChannel.predict_future_contribution()` uses PV forecast array to predict future solar heat per 10-min step with exponential decay smoothing at sunset.
- **UFH Slab (Estrich) Thermal Model**: First-order lag between commanded outlet temperature and effective heating temperature, preventing spurious `+15°C` trajectory corrections.
- **`slab_time_constant_hours` as learnable parameter**: New adaptive parameter (default 1.0 h, bounds 0.25–4.0 h).
- **`slab_passive_delta` sensor**: New diagnostic metric (`inlet_temp - indoor_temp`) exported to HA.
- **`PV_CALIBRATION_INDOOR_CEILING` config variable**: Indoor temperature ceiling (default 23.0°C) for filtering blind-contaminated PV calibration periods.
- **`LIVING_ROOM_TEMP_ENTITY_ID` in InfluxDB query**: Living room temperature now included in entity filter for calibration.
- **New config variables**: `ELECTRICITY_PRICE_ENTITY_ID`, `PRICE_CHEAP_PERCENTILE`, `PRICE_EXPENSIVE_PERCENTILE`, `PRICE_TARGET_OFFSET`, `PRICE_EXPENSIVE_OVERSHOOT`, `SLAB_TIME_CONSTANT_HOURS`.

### Fixed
- **Control Stability**: Fixed "Deadbeat Control" oscillation by decoupling the control interval (30m) from the optimization horizon (4h).
- **HP-off outlet spike (35°C)**: Binary search now simulates "HP on" using the learned `delta_t_floor` when heat pump is off, preventing pointless 35°C setpoints.
- **PV routing at sunset**: `_is_pv_active()` now uses `max(pv_power_current, pv_power_smoothed)` to capture solar thermal lag.
- **PV smoothing window**: Shortened from 3h to `solar_decay_tau` (~30min), eliminating stale morning PV values in the afternoon.
- **Blind-contaminated PV calibration data**: Root cause fix for `pv_heat_weight` stuck at lower bound — periods with indoor temp above ceiling are now excluded from PV calibration.
- **`pv_heat_weight` bounds**: Default lowered from 0.0005 → 0.0002; lower bound from 0.0005 → 0.0001.
- **`cloud_factor_exponent` learning when disabled**: Online learning and batch calibration now correctly gated behind `CLOUD_COVER_CORRECTION_ENABLED`.
- **`solar_lag_minutes_delta` persistence**: Learning updates now persisted across restarts in `unified_thermal_state.json`.
- **`calibrate_delta_t_floor` sensitivity**: Raised minimum delta_t threshold from 0.5 → 1.0°C to prevent convergence to sub-1°C values.
- **PV decay period detection**: Replaced single-step filter with 6-step sliding-window crossing detection for gradual sunset transitions.

### Changed
- `pv_heat_weight` default 0.0005 → 0.0002, bounds lower limit 0.0005 → 0.0001.
- `predict_thermal_trajectory`: new optional `inlet_temp` parameter for slab model (backward-compatible).
- InfluxDB Flux query entity filter now includes `LIVING_ROOM_TEMP_ENTITY_ID`.

## [0.2.0] - 2026-02-10

### Added
- Initial release of ML Heating Underfloor addon
- Physics-based machine learning heating control optimized for underfloor heating
- Underfloor-specific thermal defaults (lower outlet temps, higher effectiveness)
- Complete parameter sync with .env configuration
- Cooling mode support with underfloor-specific bounds
- Heat source channel architecture for isolated learning
- Indoor trend protection to prevent parameter drift
- Full InfluxDB v2 integration with features bucket
- Solar correction and PV forecast integration
- Delta forecast calibration for local weather offsets

### Optimized for Underfloor
- CLAMP_MAX_ABS set to 35°C (protects floor covering)
- OUTLET_EFFECTIVENESS at 0.93 (large radiating surface)
- Conservative learning rates for slow thermal mass
- Extended training lookback (1800 hours) for screed slab dynamics
- Slab time constant parameter for Estrich thermal modeling
