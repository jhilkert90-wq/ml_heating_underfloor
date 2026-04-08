# ML Heating Underfloor

Physics-based machine learning heating control system optimized for **underfloor heating** (Fußbodenheizung) with online learning.

## Overview

This addon provides ML-based heating control specifically tuned for underfloor heating systems with slow thermal response and large thermal mass (screed/slab). It uses the same core engine as ML Heating but with defaults optimized for:

- **Lower outlet temperatures** (max 35°C to protect floor coverings)
- **Higher outlet effectiveness** (0.93 — large radiating surface area)
- **Conservative learning rates** (slow thermal mass needs careful adaptation)
- **Extended training lookback** (1800 hours for screed slab dynamics)
- **Slab time constant modeling** (Estrich first-order thermal time constant)

## Features

- **ThermalEquilibriumModel** — Physics-based ML with continuous parameter adaptation
- **Underfloor-optimized defaults** — Pre-tuned for slow thermal response systems
- **Delta Forecast Calibration** — Local weather forecast calibration for enhanced prediction
- **Enhanced Model Wrapper** — Intelligent outlet temperature prediction with smart rounding
- **Adaptive Learning** — Real-time parameter optimization based on prediction feedback
- **Multi-Heat Source Integration** — PV solar, fireplace, and electronics heat coordination
- **Heat Source Channels** — Isolated learning per source (heat pump, solar, fireplace, TV)
- **Active & Shadow Modes** — Safe testing mode alongside production heat curve operation
- **Cooling Mode Support** — Automatic cooling control when heat pump reports "cool"
- **Live Performance Tracking** — Real-time confidence and accuracy monitoring
- **Comprehensive Safety** — Blocking detection, gradual temperature control, health monitoring
- **Indoor Trend Protection** — Prevents parameter drift during setpoint changes

## Installation

1. Add this repository to Home Assistant:
   ```
   https://github.com/jhilkert90-wq/ml_heating_underfloor
   ```
2. Install "ML Heating Underfloor" from the Add-on Store
3. Configure with your Home Assistant entities
4. Start the add-on

## Configuration

The add-on requires configuration of various Home Assistant entities including:

- Temperature sensors (indoor, outdoor, outlet, inlet)
- Climate control entity (heating status)
- InfluxDB connection (for historical data)
- Optional: PV forecast, fireplace detection, solar correction, etc.

### Key Underfloor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clamp_max_abs` | 35.0 | Max outlet temp — protects floor covering |
| `clamp_min_abs` | 18.0 | Min outlet temp |
| `outlet_effectiveness` | 0.93 | High due to large UFH surface area |
| `heat_loss_coefficient` | 0.15 | Building heat loss rate |
| `thermal_time_constant` | 4.0 | Screed slab thermal response (hours) |
| `slab_time_constant_hours` | 1.0 | UFH slab forced-convection time constant |
| `training_lookback_hours` | 1800 | Extended lookback for slow dynamics |
| `cycle_interval_minutes` | 10 | Learning/prediction cycle interval |

See the add-on configuration tab for detailed setup instructions.

## Dashboard

Access the ML Heating dashboard through the Home Assistant sidebar panel.

The dashboard provides:
- Real-time performance metrics
- Model prediction accuracy (MAE/RMSE)
- System learning status and confidence
- Physics validation results
- Heat source channel status

## Shadow Mode

Start with `shadow_mode: true` to observe ML recommendations without affecting your heating. The ML system will learn from your existing heat curve and publish shadow predictions for comparison. Switch to active mode once confident in the predictions.

## Support

- **Documentation**: See configuration tab for detailed setup
- **Issues**: Report bugs via GitHub repository
- **Development**: For testing latest features, install the Development channel separately
