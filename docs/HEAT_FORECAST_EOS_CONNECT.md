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
