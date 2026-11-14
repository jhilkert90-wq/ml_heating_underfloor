# WP Load Forecast: Not Yet Implemented / Open Items

Status after the 2026-11 forecast overhaul. Everything not listed here is
implemented (see `HEAT_FORECAST_EOS_CONNECT.md`).

## Confirmed accuracy ingestion issue (2026-11-14)

The supplied `sensor.ml_heating_wp_load_forecast` snapshot has `n: 0` for every
lead/window, not a sensor-rendering problem. Its 48-slot electrical components
sum correctly: the current 0.0481 kWh is 0.0013 kWh heating plus 0.0468 kWh
standby. COP calibration and forecast accuracy are separate histories; the
reported 109.47 calibration hours do not establish add-on uptime or completed
forecast/actual pairs. Accuracy compares heating electrical/thermal energy,
not the total including standby and DHW.

The following integration defect was reproduced against unchanged code using
an isolated forecast engine and synthetic counter history:

1. `_record_interval()` replaces the previous hour's anchor when crossing the
   hour. A live reading seven minutes after the boundary creates a partial
   anchor, so `actual_from_anchors()` cannot close the previous hour.
2. `_refresh_hour_anchor()` can subsequently interpolate an exact boundary
   from history, but does not retry the accuracy pairing or retain the replaced
   previous anchor for that purpose.
3. In the reproduction, accuracy stays at `n=0` after a successful repair.
   Pairing the saved old anchor and repaired new anchor manually produces
   valid actuals of 0.5 kWh electrical / 2.0 kWh thermal; explicitly recording
   these changes `n` to 1.

There is a second limitation: `coverage: counter_anchor` accepts some
pre-boundary readings that accuracy correctly rejects outside its five-minute
alignment tolerance. During idle periods, unchanged HA counters can have old
`last_updated` timestamps. The supplied production log includes anchors whose
readings are one hour old despite that coverage label. It has 23 pending-count
messages and no closed-accuracy-hour messages. `degraded_reasons: []` does not
certify accuracy-history coverage.

**Analysis only; not fixed:** retain validated boundary data long enough to
pair adjacent hours after history repair, and expose rejection reasons.
Do not fabricate observations, widen the tolerance or replace missing actuals
with predictions. The live `heat_forecast_state.json` was not supplied, so the
exact rejection reason for each production hour remains unverified. Existing
unit tests validate pairing of manually supplied anchors, but not this
repair-to-accuracy handoff.

## Not implemented

| Item | Description | Effect while missing |
|---|---|---|
| PR 4a-5 (b) | Start time of the heat pump as a control variable in the outlet search | Search only uses the start window as a trajectory input |
| PR 4a-5 (c) | Short-run protection (minimum runtime) using the estimated runtime | No protection against very short runs |
| PR 4a-6 | Grace wait uses the start estimate | Grace wait loop has no access to HA states; behaviour unchanged |
| Cooling forecast | Start-estimate window for cooling mode in the forecast | Forecast stays heating only |
| Hysteresis duty cycle | Forecast path for `method: hysteresis` (cycling instead of one window) | Hysteresis sensors only give a single start/runtime window |
| Near range + start estimate | Near-range controller does not use `t_start`/runtime of the sensor | Near range assumes the heat pump is running at the live outlet; a future start is not modelled |
| PR 4a-7 | Start-time error metric per estimate method | No quality metric of the sensor, only `accuracy` of the energy forecast |
| `scripts/` tool | Offline evaluation of `accuracy` data | Accuracy only visible as sensor attribute |
| Replay test | Test replaying the 2026-11-12 10:01 log (outlet 25, room 22.78, outdoor 9.9, eff 0.72) | Only synthetic tests |

## Open questions

- Mapping from `sensor.ml_vorlauftemperatur` to the NIBE setpoint
  (`hp_start_estimate_required_offset_k` is only an approximation).
- Sign convention and thresholds of cooling degree minutes.
- Accuracy of the hysteresis variant.
- Why the binary search log shows `delta_t_floor` 2.24 while the forecast uses about 2.6.

## Behaviour changes to review

- `price_cost_max_overshoot_k` (default 0.3 K) limits forecast overshoot in cost
  mode and changes live control for existing users.
