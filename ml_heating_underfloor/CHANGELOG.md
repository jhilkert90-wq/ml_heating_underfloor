# Changelog - ML Heating Underfloor

## [0.2.6] - 2026-04-18

### Added
- **Indoor Temperature Trend Bias in Trajectory Prediction**: `predict_thermal_trajectory()` now incorporates `indoor_temp_delta_60m` as a decaying momentum bias. Observed indoor temperature trend (°C over last 60 min) captures unmeasured heat sources (solar through windows, body heat, appliances, thermal mass) that the physics model cannot see. The bias uses exponential decay controlled by `TREND_DECAY_TAU_HOURS` (default 1.5h) so near-future predictions strongly reflect observed momentum while far-future predictions rely on physics.
  - New config variable `TREND_DECAY_TAU_HOURS` (default `1.5`, env-overridable) controls decay time constant.
  - Trend bias is clamped to ±0.05°C per step and gated on `abs(trend) > 0.01` to prevent floating-point noise.
  - Passed from both binary search optimization and trajectory verification callers in `model_wrapper.py`.

### Fixed
- **Critical: Binary search `_features` NameError causing 35°C fallback**: `predict_thermal_trajectory` failed every binary search iteration with `name '_features' is not defined`, causing silent fallback to max outlet temperature (35°C). The gradual temperature control then capped/smoothed this down, masking the root cause but producing suboptimal heating decisions. Fixed by replacing bare `_features.get(...)` with `self._current_features.get(...)` using the safe `hasattr` guard pattern used elsewhere.
- **Dashboard `titlefont` crash**: Replaced deprecated Plotly `titlefont` → `title_font` in `dashboard/components/overview.py` confidence/error dual-axis chart. Plotly 6.x removed `titlefont`, causing `ValueError` on every dashboard load.
- **Noisy "Logging MAE"/"Logging RMSE" debug messages**: Removed unnecessary `logging.debug("Logging MAE")` and `logging.debug("Logging RMSE")` in `ha_client.py` that produced low-value noise every 10-minute cycle. The actual HA state updates and their results already provide sufficient logging.

### Added
- **Binary search diagnostic logging**: Added debug logs on first iteration of binary search to show resolved `inlet_temp`, `delta_t_floor`, `indoor_temp_delta_60m`, optimization horizon, and trajectory result (steps, start→end temperatures). Helps verify the `_features` fix is working correctly in production.

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

## [0.2.5] - 2026-04-17

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
