# WP Load Forecast: Not Yet Implemented / Open Items

Status after the 2026-11 forecast overhaul. Everything not listed here is
implemented (see `HEAT_FORECAST_EOS_CONNECT.md`).

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
