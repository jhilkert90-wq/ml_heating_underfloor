# WP Load Forecast: Not Yet Implemented / Open Items

Status after the 2026-11 forecast overhaul. Everything not listed here is
implemented (see `HEAT_FORECAST_EOS_CONNECT.md`).

## Resolved accuracy ingestion issue (2026-11-15)

The supplied `sensor.ml_heating_wp_load_forecast` snapshot has `n: 0` for every
lead/window, not a sensor-rendering problem. Its 48-slot electrical components
sum correctly: the current 0.0481 kWh is 0.0013 kWh heating plus 0.0468 kWh
standby. COP calibration and forecast accuracy are separate histories; the
reported 109.47 calibration hours do not establish add-on uptime or completed
forecast/actual pairs. Accuracy compares heating electrical/thermal energy,
not the total including standby and DHW.

The following pre-fix integration defect was reproduced using an isolated
forecast engine and synthetic counter history:

1. `_record_interval()` replaced the previous hour's anchor when crossing the
   hour. A live reading seven minutes after the boundary created a partial
   anchor, so `actual_from_anchors()` could not close the previous hour.
2. `_refresh_hour_anchor()` could subsequently interpolate an exact boundary
   from history, but did not retry the accuracy pairing or retain the replaced
   previous anchor for that purpose.
3. In the reproduction, accuracy stayed at `n=0` after a successful repair.
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

**Implemented fix:** persist `previous_hour_anchor` alongside the current
anchor and retry the same accuracy handoff after history repair. Repair the
previous boundary too when stale, without replacing an already valid current
anchor with poorer history. Stale/partial history is retried on subsequent
forecast cycles; HA fallback also handles present-but-unaligned Influx data.
Repairs and completed pairs are immediately saved in forecast state. Valid live
boundary pairs retain their history-free fast path and repeated repair does
not duplicate accuracy rows.

The shared anchor validator preserves the configured five-minute tolerance,
and history interpolation still uses the existing maximum-gap rule. Invalid
anchors, stale counter timestamps and counter resets now have INFO rejection
diagnostics. No observations, predictions, COP calibration data or model
parameters are invented or reset.

**Remaining operational limits:** this is a workspace fix, not a deployment.
The history-repair step still runs after publication, so newly repaired
accuracy appears at the next normal sensor publication. Counts require a
stored prediction and a completed, valid counter hour; they can remain zero
when observations are genuinely missing. Lead-6/24 metrics need matching
earlier predictions, not just COP calibration history. The live
`heat_forecast_state.json` was not supplied, so the exact rejection reason for
each historical production hour remains unverified.

Five new regression cases cover live boundaries, repair across restart,
stale previous anchors, delayed history and aligned HA fallback; 123 focused
tests pass. An additional 412-case comparison against pre-fix validation
confirms unchanged counter acceptance and defrost normalization.

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
