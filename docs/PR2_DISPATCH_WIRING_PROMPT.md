# PR2 Prompt: Wire Dispatch in main.py

## Context

PR1 created all the infrastructure needed to replace the ~2000-line inline
loop body in `main.py` with clean route dispatch calls:

- `src/loop_state.py` — LoopState dataclass (cross-cycle runtime variables)
- `src/pre_dispatch.py` — Extracted pre-dispatch functions (sensor buffer,
  shadow mode, online learning, grace period, blocking check, climate mode)
- `src/cycle_state.py` — Updated with GRACE_PERIOD as 5th CycleState
- `src/cycle_routes.py` — Updated `run_idle_route` (full features + learning),
  added `run_grace_period_route`

## Task for PR2

Replace the inline loop body in `main.py` (lines ~520–2565) with:

1. **Initialization** (before loop): Replace lines ~520-609 with `LoopState()`
   initialization. Map existing locals to LoopState fields.

2. **Loop start** (each cycle): Replace lines ~610-926 with calls to:
   - `loop.increment_cycle()`
   - `update_sensor_buffer_and_thermo(loop, ha_client, all_states, influx_service)`
   - `resolve_shadow_mode_for_cycle(ha_client, all_states)`
   - `emit_network_error_state(ha_client)` (on connection failure)

3. **Online learning** (lines ~927-1336): Replace with:
   - `run_online_learning(ha_client, all_states, state, effective_shadow_mode, climate_mode, wrapper)`

4. **Grace period** (lines ~1338-1387): Replace with:
   - `is_gp = handle_grace_period(ha_client, state, state_manager, effective_shadow_mode)`

5. **State determination** (lines ~1389-1431): Replace with:
   - `heating_active, climate_mode, state_mgr, reloaded = check_and_resolve_climate_mode(...)`
   - `cycle_state = determine_cycle_state(is_blocking, heating_active, climate_mode, is_grace_period=is_gp)`

6. **Dispatch** (lines ~1433-2565): Replace entire inline routing with:
   ```python
   ctx = CycleContext(
       ha_client=ha_client,
       all_states=all_states,
       state=state,
       state_manager=state_mgr,
       wrapper=loop.wrapper,
       climate_mode=climate_mode,
       heating_obs_buffer=loop.heating_obs_buffer,
       last_indoor_temp=last_indoor_temp,
       # ... other fields
   )

   if cycle_state == CycleState.BLOCKING:
       run_blocking_route(ctx)
   elif cycle_state == CycleState.GRACE_PERIOD:
       run_grace_period_route(ctx)
   elif cycle_state == CycleState.IDLE:
       run_idle_route(ctx)
   elif cycle_state == CycleState.COOLING:
       run_cooling_route(ctx)
   else:  # HEATING
       run_heating_route(ctx)
   ```

7. **Post-dispatch**: Extract `last_indoor_temp` and any cross-cycle values
   from `ctx` back into `loop` / locals for next iteration.

## State File Routing (important!)

| CycleState     | State Manager                     | Notes                                  |
|---------------|-----------------------------------|----------------------------------------|
| HEATING       | wrapper.state_manager_heating     | `set_climate_mode("heating")`          |
| COOLING       | wrapper.state_manager_cooling     | `set_climate_mode("cooling")`          |
| IDLE          | wrapper.state_manager_heating     | `set_climate_mode("heating")` — saves to heating state |
| BLOCKING      | (preserves current)               | No mode switch                         |
| GRACE_PERIOD  | wrapper.state_manager_heating     | Preserves last_final_temp              |

## Key Constraints

- Do NOT change any business logic — only restructure control flow
- Verify 1429+ tests still pass after refactoring
- The `run_idle_route` now does full feature calculation + saves state
  (not just passive features) — so online learning has valid data next cycle
- `run_online_learning()` in pre_dispatch.py handles the heating obs buffer
  resolve/push + cooling obs buffer + HLC parameter learning
- Shadow mode comparison sums are tracked in `loop.shadow_ml_error_sum` etc.
- Keep `time.sleep(CYCLE_INTERVAL_MINUTES * 60)` at the end of the loop

## Testing Strategy

- All 1429 existing tests must pass unchanged
- Add integration test: mock HA states → run one full cycle through dispatch →
  verify correct route was called and state saved
- Add test: IDLE cycle writes to heating state file
- Add test: mode transition (heating→cooling) reloads correct state
