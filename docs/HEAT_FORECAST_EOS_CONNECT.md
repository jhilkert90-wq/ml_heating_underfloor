# Heating Energy Forecast for EOS Connect

The add-on can publish an optional, read-only hourly forecast for the heat pump.
It is intended as an input profile for EOS Connect; it does not change the
heating controller or configure EOS Connect itself.

## Enablement and inputs

Enable `heat_forecast_enabled` to publish the forecast. Enable
`cop_calibration_enabled` separately to learn the electrical COP from energy
counters. Both options are off by default. Configure cumulative kWh counters
for space-heating electricity and heat, domestic-hot-water electricity,
standby electricity, and (where available) defrost electricity and heat.

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

At startup, available one-minute Influx history for the defrost signal,
electrical power, and direct thermal power is used once to estimate average
defrost electrical/thermal rates, energy per event, and initial defrost
fractions. These power-derived values seed the forecast only; COP calibration
still uses energy-counter deltas. Counter-derived defrost fractions blend in
over the configured cold compressor hours.

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
- `cop_curve`, defrost diagnostics, learning coverage, and current-hour
  estimate status.

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
