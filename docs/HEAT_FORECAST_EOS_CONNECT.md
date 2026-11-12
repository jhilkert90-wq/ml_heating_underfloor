# Heating Energy Forecast for EOS Connect

The add-on can publish an optional, read-only hourly forecast for the heat pump.
It is intended as an input profile for EOS Connect; it does not change the
heating controller or configure EOS Connect itself.

## Enablement and inputs

Enable `heat_forecast_enabled` to publish the forecast. Enable
`cop_calibration_enabled` separately to learn the electrical COP from energy
counters. This setting is read-only and does not change heating control; it
does not enable the forecast sensor by itself. Both options are off by default.
Configure cumulative kWh counters for space-heating electricity and heat,
domestic-hot-water electricity, standby electricity, and (where available)
defrost electricity and heat.

The dashboard setting `heat_forecast_include_dhw` includes domestic hot-water
electricity by default. Turn it off when EOS Connect manages that load and
starts hot-water preparation itself. This only changes the published forecast,
including the current hour and fireplace scenarios; it does not disable or
control actual hot-water preparation. The DHW breakdown is then zero and
`includes` omits `dhw`; counter history is still collected.

History is loaded from InfluxDB first and falls back to Home Assistant Recorder
history when InfluxDB is unavailable or has too few aligned samples. Forecast
history queries use `heat_forecast_influx_timeout_seconds` (default 60 seconds,
range 10–300). Calibration waits for at least 48 aligned heating hours by
default; failed or insufficient bootstraps retry with backoff rather than
being marked complete. While the first bootstrap runs, the forecast can be
published with defaults and reports its data source and coverage.

The heating electricity counter may include defrost electricity. The default
`cop_defrost_el_in_heating_counter: true` matches that setup. The thermal
heating counter is treated as gross heat; the defrost heat counter is used to
subtract heat consumed by defrost. `auto` compares the electricity-meter delta
and defrost counter with the interval's power integral; it needs repeated
defrost intervals to make a determination and initially assumes inclusion.
The add-on's configured defrost fractions are used until measured defrost
counters have accumulated data; measured ratios are blended in over the
configured cold-run hours (100 by default).

Energy counters are the source for calibration and historical profiles. The
optional direct thermal-power sensor (`thermal_power_source: direct`) supplies
real-time thermal power only; it is not integrated into calibration. The
electrical power sensor is used for the current-hour remainder and the
defrost-counter diagnostic, not as a normal calibration source. Missing
counters or invalid gaps are excluded; the current-hour estimate reports a
degraded status when meter coverage is incomplete.

At startup, available one-minute history for the defrost signal, electrical
power, and direct thermal power is loaded from InfluxDB, then Home Assistant
history, to estimate defrost rates, energy per event, and initial fractions.
Power-derived fractions are accepted only with at least five plausible events;
these values seed the forecast only, while COP calibration still uses
energy-counter deltas. Counter-derived defrost fractions blend in over the
configured cold compressor hours.

## Published entity

The default entity is `sensor.ml_heating_wp_load_forecast`. Its state is the
estimated total electrical energy for the current hour, in kWh. Attributes
contain hour-aligned local timestamps with UTC offsets (including DST changes)
and parallel arrays:

- `forecast`: EOS-ready `{start, value}` rows for total electrical kWh.
- `forecast_heat_th_kwh`: required space-heating thermal energy per hour.
- `forecast_heat_el_kwh`, `forecast_defrost_el_kwh`,
  `forecast_dhw_el_kwh`, `forecast_standby_el_kwh`: electrical breakdown.
- `data_quality`: weather/PV source, missing-data padding, COP extrapolation,
  active climate mode, and fireplace state for each hour.
- `calibration_source`/`calibration_hours`, `price_source_per_hour`/
  `prices_reused_hours`, defrost diagnostics, `current_hour_estimate`, and
  `degraded_reasons`.

When tomorrow's prices are not yet published, the forecast reuses today's price
for the same clock hour and labels it `reused_today`; it reclassifies prices
automatically after the next cache refresh. At restart, historical counter
values can anchor the current hour; otherwise the hour is marked partial and
estimated from available power.

The space-heating simulation uses outdoor temperature, the thermal PV gain
weight, building/slab parameters, fireplace contribution and residual heat,
and active price/PV heating offsets. If the fireplace is currently on, separate
3-hour and 4-hour scenarios are included. DHW and standby use recent energy
counter profiles or explicit configured fallback values. A missing weather
hour uses the last available temperature and is marked padded; a missing PV
hour contributes zero and is marked padded. Cooling is not forecast in this
feature.

Use the `forecast` time series as the external heat-pump load profile in EOS
Connect. The thermal-energy array is available for analysis and is not the
electrical load profile. EOS Connect's own profile/entity integration must be
configured separately.


## Forecast behaviour (since the 2026-11 overhaul)

- **Lookahead controller:** the simulation applies, per step, the smallest
  constant thermal power that keeps the room above `target - 0.1 K` for
  `heat_forecast_lookahead_hours` (default 3 h). `heat_forecast_max_ramp_kw_per_step`
  limits power changes per step. Hourly smoothing is opt-in
  (`heat_forecast_smoothing_enabled`, default off).
- **Current hour:** thermal energy of the current hour is derived from the
  electrical remainder so `forecast_heat_th_kwh` and `forecast_heat_el_kwh`
  stay consistent.
- **PV gaps:** with `heat_forecast_pv_gap_mode: daily_profile` a missing PV hour
  reuses the value from 24 h earlier; `none` contributes zero.
- **COP cap:** outside the calibrated outdoor-temperature range the COP is capped
  at the best observed COP plus `heat_forecast_cop_extrapolation_margin`
  (default 0.3, negative disables). Attribute `cop_extrapolated_hours`.
- **Accuracy:** attribute `accuracy` contains bias, MAE, MAPE, missed-start and
  false-start rate per lead time (1 h, 6 h, 24 h) over 24 h and 7 d, measured
  against the hourly counter energy.
- **Near-range controller (optional, default off):**
  `heat_forecast_near_controller_enabled` uses the live outlet setpoint for the
  first `heat_forecast_near_hours` (power = scale x outlet effectiveness x
  (outlet - room), blended out over `heat_forecast_near_blend_hours`). The
  dimensionless scale (0.5-2.0) is learned against the measured thermal power
  (`thermal_power_entity_id`). Active only in heating mode, outside DHW, with a
  snapshot younger than 30 min and either outlet - inlet >= `delta_t_floor` or
  measured thermal power >= 0.5 kW.
- **Start estimate (optional):** an external HA sensor can provide the minutes
  until the heat pump starts and its runtime (attribute `hp_start_estimate`).
  See below.

## Dashboard

All options of this feature are in the dashboard category
**"WP Load Forecast & Start Estimate"** (forecast, COP calibration, counters,
near-range controller and start-estimate options). Entity IDs are free-text
fields, switches are checkboxes. Price/cost-mode options
(`price_cost_*`, `hp_power_*`) stay in "Price & PV Optimization".

## Start-estimate sensors in Home Assistant (optional)

The add-on can read up to two template sensors you create in Home Assistant, one
for heating and one for cooling:

| Sensor | Purpose |
|---|---|
| `sensor.wp_minuten_bis_start_heizen` | minutes until the heat pump starts in heating mode |
| `sensor.wp_minuten_bis_start_kuehlen` | same for cooling mode (can be left empty) |

State contract: minutes to start (0 = running, `unknown` = no start expected).
Attributes: `status` (`idle`/`starting_soon`/`running`/`blocked`/`unknown`),
`mode`, `method` (`degree_minutes`/`hysteresis`/`learned`),
`expected_runtime_min`, `runtime_unbounded`, `confidence`, `updated_at`.

Setup in short (full YAML and background in
[HA_WP_START_ESTIMATE_GUIDE.md](HA_WP_START_ESTIMATE_GUIDE.md)):

1. Find your heat pump entities (degree minutes, actual supply temperature,
   calculated target supply temperature, compressor frequency) in
   *Developer tools -> States* (guide, section 2).
2. Create the threshold helpers `input_number` (guide, section 3).
3. Add the heating template sensor (guide, section 4); for cooling use section 5,
   or variant B for pure hysteresis heat pumps (section 5b).
4. *Developer tools -> YAML -> Check configuration*, reload template entities
   (guide, section 6).
5. Verify the values against real starts before using them (guide, section 7).
6. In the dashboard category "WP Load Forecast & Start Estimate" set
   `hp_start_estimate_enabled`, the heating/cooling entity IDs and, one by one,
   the `hp_start_estimate_use_for_*` switches (all off by default). With the
   switches off or the sensor missing, stale or of low confidence the add-on
   behaves exactly as without it.

## Known gaps and open items

See [HEAT_FORECAST_OPEN_ITEMS.md](HEAT_FORECAST_OPEN_ITEMS.md).
