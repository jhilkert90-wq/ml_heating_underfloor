# Electricity Price-Aware Optimization

## Overview

The system calls the `tibber.get_prices` Home Assistant service to fetch
the electricity price forecast, caches the result in memory, and classifies
the current price relative to today's price distribution.  Both 15-minute
(quarter-hourly, 96 entries/day) and 60-minute (hourly, 24 entries/day)
resolutions are handled transparently — the percentile classification works
on whatever list size it receives.

| Price Level | Condition | Target Offset | Trajectory Overshoot |
|-------------|-----------|---------------|---------------------|
| **CHEAP** | ≤ P33 of daily prices | +0.2 °C | +0.5 °C (default) |
| **NORMAL** | between P33 and P67 | 0.0 °C | +0.5 °C (default) |
| **EXPENSIVE** | ≥ P67 of daily prices | −0.2 °C | +0.2 °C (tighter) |

The binary search convergence precision remains ±0.01 °C — only the
**target** shifts, not the tolerance band.

## How It Works

```
HA sends target_temp = 22.0 °C
    ↓
PriceOptimizer classifies current price (e.g. 0.05 EUR/kWh → CHEAP)
    ↓
target_adjusted = 22.0 + 0.2 = 22.2 °C
    ↓
Binary search: find outlet where equilibrium = 22.2 °C ± 0.01
    ↓
Outlet ~42.4 °C (slightly higher than normal ~41.8 °C)
    ↓
Trajectory correction with standard +0.5 °C future overshoot
    ↓
Learning uses ORIGINAL target (22.0 °C) — no parameter corruption
```

For **EXPENSIVE** periods:
```
target_adjusted = 22.0 − 0.2 = 21.8 °C
    ↓
Binary search: outlet ~40.9 °C (lower)
    ↓
Trajectory correction with TIGHT +0.2 °C future overshoot
    ↓
Result: heats less, stops earlier when overshooting
```

## Configuration

All variables are in `src/config.py` and overridable via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `ELECTRICITY_PRICE_ENABLED` | `false` | Master feature flag |
| `PRICE_CHEAP_PERCENTILE` | `33` | Percentile below which prices are CHEAP |
| `PRICE_EXPENSIVE_PERCENTILE` | `67` | Percentile above which prices are EXPENSIVE |
| `PRICE_TARGET_OFFSET` | `0.2` | Temperature shift magnitude (°C) |
| `PRICE_EXPENSIVE_OVERSHOOT` | `0.2` | Tighter future overshoot for EXPENSIVE (°C) |
| `PRICE_CACHE_REFRESH_MINUTES` | `60` | Cache refresh interval (minutes) |
| `FEATURES_ENTITY_ID` | `sensor.ml_heating_features` | Features debug sensor |

## Enabling

Set the environment variable:

```yaml
ELECTRICITY_PRICE_ENABLED: "true"
```

When disabled (default), the system behaves exactly as before — no code paths
are executed, no sensors are published, and all tests pass identically.

## Tibber Price Data

Prices are fetched via the `tibber.get_prices` HA service call:

```
POST /api/services/tibber/get_prices
Body:  {"start": "2026-04-14 00:00:00", "end": "2026-04-16 00:00:00"}
Params: return_response=true
```

**Response** (auto-detects first home):
```json
{
  "service_response": {
    "prices": {
      "My Home": [
        {"start_time": "2026-04-14 00:00:00+02:00", "price": 0.12},
        {"start_time": "2026-04-14 00:15:00+02:00", "price": 0.13}
      ]
    }
  }
}
```

### Cache strategy

- Refreshes every `PRICE_CACHE_REFRESH_MINUTES` (default 60)
- Refreshes on calendar day change
- After 13:00 local time, re-fetches if tomorrow's prices are not yet cached
- In-memory only — re-fetches automatically on add-on restart

### Resolution transparency

The system works identically with both 15-minute (96 entries/day) and
60-minute (24 entries/day) price data.  The current price is found by
scanning for the last entry whose `start_time ≤ now`.  Percentile
classification uses all prices for the current calendar day.

## New HA Sensors

### `sensor.ml_heating_features`
Exports all features from the last prediction cycle as attributes.
Useful for debugging what data the model received.

- **State**: Current living room temperature
- **Attributes**: All feature keys (outdoor_temp, pv_now, flow_rate, etc.)

### `sensor.ml_heating_price_level`
Exports the current price classification.

- **State**: Current price in EUR/kWh
- **Attributes**: `price_level`, `cheap_threshold`, `expensive_threshold`,
  `target_offset`

## Learning Safety

The price integration **cannot corrupt learned parameters** because:

1. The target temperature is shifted, not the outlet temperature
2. The binary search converges to a self-consistent outlet for the adjusted target
3. The actual room temperature matches the prediction for the adjusted target
4. Therefore `prediction_error ≈ 0` — no spurious learning signal
5. The original target is preserved in metadata (`target_temp_original`) for reference

## Files Modified

| File | Changes |
|------|---------|
| `src/price_optimizer.py` | PriceLevel enum, PriceOptimizer class, Tibber cache + time lookup |
| `src/config.py` | Price configuration variables incl. `PRICE_CACHE_REFRESH_MINUTES` |
| `src/model_wrapper.py` | Binary search uses adjusted target; trajectory uses price-aware overshoot |
| `src/ha_client.py` | `call_tibber_get_prices()`, `publish_last_run_features()`, `publish_price_level()` |
| `src/main.py` | Cache refresh → price data → prediction → publish sensors |
| `tests/unit/test_price_optimizer.py` | 51 tests (29 classification + 22 cache/time) |

## Future: Thermal Pre-Charging (Option F)

A planned enhancement uses the 24-hour price forecast + the slab thermal
time constant to look ahead and pre-charge the thermal mass during cheap
periods before an expensive window arrives. This is saved for a later phase.
