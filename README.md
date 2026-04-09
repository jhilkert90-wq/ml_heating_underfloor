# ML Heating Control for Home Assistant

> **Warning**
> This project is an initial test and proof of concept heating controller. However, heating systems are critical infrastructure - always monitor its behavior and ensure you have safety mechanisms in place. Use at your own risk.

This project implements a **physics-based machine learning heating control system** that integrates with Home Assistant. It uses a ThermalEquilibriumModel to predict the optimal water outlet temperature for a heat pump, continuously learning from real-world results to efficiently maintain your target indoor temperature.

## Overview

Traditional heating curves are static: they map outdoor temperature to a fixed outlet temperature. This system is **dynamic and adaptive** - it learns your house's unique thermal characteristics and continuously improves its predictions based on what actually happens.

## Goal

The primary goal is to improve upon traditional heat curves by creating a **self-learning system** that:

-   **Increases Efficiency:** Minimizes energy waste by precisely matching heating output to actual need
-   **Improves Comfort:** Maintains stable indoor temperature with less overshoot/undershoot
-   **Adapts Continuously:** Learns from every cycle, adjusting to seasons, weather patterns, house changes, and occupancy
-   **Anticipates Conditions:** Uses weather and PV forecasts to proactively adjust heating
-   **Operates Safely:** Includes comprehensive blocking logic, gradual temperature changes, and health monitoring

## Key Features

### Core Capabilities

-   **Physics-Based Machine Learning:** Uses ThermalEquilibriumModel that understands thermodynamic principles while learning your house's unique characteristics from real data
-   **Online Learning:** Continuously adapts after every heating cycle - learns from what actually happened vs what was predicted
-   **Heating & Cooling Modes:** Full support for both heating and cooling with independent thermal parameters, separate state files, and mode-specific learning. Cooling mode is activated automatically when the climate entity reports "cool"
-   **Heat Source Channel Architecture:** Decomposed learning with independent channels for heat pump, solar, fireplace, and TV — prevents parameter cross-contamination when multiple heat sources are active simultaneously
-   **Live Performance Tracking:** Real-time confidence and accuracy monitoring that adapts to actual prediction performance
-   **Delta Forecast Calibration:** Advanced forecasting integration with weather and PV power predictions
-   **Seasonal Adaptation:** Automatic parameter adjustment learns seasonal variations without manual recalibration
-   **Active & Shadow Modes:** Can run in active mode (controlling heating) or shadow mode (calculating but not applying, for safe testing and comparison)
-   **State Compression:** Efficient allow-list compression of thermal state history to reduce file size while retaining all essential learning context
-   **Home Assistant Integration:** Seamless bi-directional integration - reads sensors, writes control temperatures, publishes metrics and diagnostics
-   **InfluxDB Historical Data:** Leverages your existing Home Assistant/InfluxDB setup for initial calibration and historical feature engineering

### Intelligent Control

-   **Enhanced Model Wrapper:** Intelligent prediction system that replaces complex control logic with simplified outlet temperature prediction
-   **Gentle Trajectory Correction:** Intelligent additive correction system that prevents outlet temperature spikes during thermal trajectory deviations, using proven heat curve automation logic (5°C/8°C/12°C per degree) instead of aggressive multiplicative factors
-   **Overshoot & Undershoot Gates:** Projected-temperature gates that skip corrections when indoor temperature is already moving in the right direction, preventing overcorrection oscillation
-   **Smart Rounding:** Tests both floor and ceiling temperatures, predicts outcomes, chooses the one that gets closest to target
-   **Gradual Temperature Control:** Limits maximum temperature change per cycle to protect heat pump from abrupt setpoint jumps
-   **Differential-Based Effectiveness:** Heat transfer effectiveness scales with outlet-indoor temperature differential for accurate low-differential scenarios
-   **Adaptive Learning:** Model parameters continuously adjust based on prediction feedback to improve accuracy over time
-   **Open Window Adaptation:** System automatically detects sudden heat loss changes (like opened windows) and applies gentle corrections, then restabilizes when disturbances end
-   **HP-Off Simulated Delta-T:** When the heat pump is off (delta_t < 1°C), the binary search substitutes the learned delta_t_floor (~2.55°C) to simulate "HP on" conditions, preventing false 35°C outlet spikes from the slab passive mode making all candidates identical
-   **Slab Pump Gate:** Dual-condition gate (`outlet > t_slab AND measured_delta_t >= 1.0`) ensures the slab passive heating model only activates when the heat pump is genuinely running, preventing false pump-on states

### Safety & Robustness

-   **Blocking Event Handling:** Automatically pauses and waits during DHW heating, defrosting, disinfection, and DHW boost heater operation
-   **Grace Period after Blocking:** After blocking events end, intelligently waits for outlet temperature to stabilize before resuming ML control:
    
    - **After DHW heating** (outlet hotter): Sets aggressive cool-down target and waits for outlet to cool
    - **After defrost** (outlet colder): Restores exact pre-defrost target and waits for outlet to fully recover
    
    - Intelligently determines whether to wait for cooling or warming based on measured outlet vs target
    - Enforces maximum timeout to prevent indefinite stalling
    - Prevents inefficient temperature oscillations and protects model learning quality

-   **Heating Status Check:** Skips prediction and learning when heating system is not in 'heat' or 'auto' mode
-   **Absolute Temperature Clamping:** Enforces safe minimum/maximum outlet temperatures
-   **Confidence Monitoring:** Tracks model confidence and publishes it to Home Assistant for monitoring and automation

### External Heat Sources

-   **Fireplace Mode:** When fireplace is active, uses alternative temperature sensor (e.g., average of other rooms) to prevent incorrect learning
-   **PV Solar Warming:** Learns how solar power generation affects indoor temperature
-   **Cloud-Aware PV Scalar:** PV heating contribution is discounted by 1-hour cloud forecast percentage, preventing overestimation on cloudy days
-   **Improved PV Routing:** PV active detection uses `max(current, smoothed)` power, preventing false negatives when current power briefly dips below threshold
-   **Fast PV Smoothing:** PV smoothing window uses `solar_decay_tau` (~30 min) instead of a fixed 10-reading window, enabling faster response to solar changes
-   **TV/Electronics Heat:** Can track heat contribution from electronics and appliances

### Live Performance Tracking

-   **Real-time Confidence Calculation:** Dynamic confidence based on rolling window of recent prediction accuracy
-   **Adaptive Uncertainty Bounds:** Automatically adjusts based on actual performance
-   **Live Learning Updates:** Model parameters continuously adjust based on prediction feedback
-   **Prediction Error Attribution:** Tracks actual vs predicted temperature changes for transparent performance monitoring
-   **Dynamic Performance Diagnostics:** ML state sensor includes real-time confidence that reflects current accuracy

### Monitoring & Diagnostics

-   **ML State Sensor:** Comprehensive sensor (`sensor.ml_heating_state`) with numeric state codes and detailed attributes for monitoring system health
-   **Performance Metrics:** Real-time confidence tracking that updates every cycle
-   **Shadow Mode Metrics:** When in shadow mode, tracks separate metrics for ML vs heat curve performance comparison
-   **Learning Metrics Export:** Exports learning parameters to InfluxDB every cycle for detailed analysis
-   **Learning Dashboard:** Interactive Jupyter notebook for at-a-glance learning status
-   **Detailed Logging:** Comprehensive logging of decisions, predictions, and learning progress

### Deployment

-   **Home Assistant Add-on:** Available in both stable and alpha (development) channels
-   **Systemd Service:** Production-ready systemd service configuration for reliable background operation
-   **Automatic Restart:** Configured to restart on failure with 5-minute delay
-   **Jupyter Notebooks:** Analysis notebooks included for learning dashboard, model diagnosis, performance monitoring, and validation

## Installation

### Home Assistant Add-on Installation (Recommended)

#### 1. Add Repository to Home Assistant
1. Navigate to **Settings** → **Add-ons** → **Add-on Store**
2. Click the **⋮** menu → **Repositories**
3. Add this repository URL:
   ```
   https://github.com/helgeerbe/ml_heating
   ```
4. Click **Add** and wait for repository to load

#### 2. Choose Your Channel

**For Stable/Production Use:**
- Install **"ML Heating Control"**
- Automatic updates enabled
- Production-ready configuration

**For Testing/Development:**
- Install **"ML Heating Control (Alpha)"**
- Manual updates required
- Latest features and improvements
- Enhanced debugging capabilities

#### 3. Configure the Add-on
1. Click on your chosen add-on
2. Go to **Configuration** tab
3. Configure your Home Assistant entity IDs (autocomplete available)
4. Set up InfluxDB connection details
5. Configure external heat sources if available
6. Click **Save**

#### 4. Start the Add-on
1. Go to **Info** tab
2. Click **Start**
3. Enable **Start on boot** for automatic startup
4. Monitor logs for successful initialization

### Manual/Development Installation

For advanced users, developers, or non-Home Assistant deployments:

#### Prerequisites

- Python 3.11 or higher
- Home Assistant with REST API access
- InfluxDB with Home Assistant data (for initial calibration)
- Systemd (for service deployment)

#### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/helgeerbe/ml_heating.git
cd ml_heating

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or for development (includes test dependencies)
pip install -e ".[dev]"
```

#### 2. Configure Environment

Copy the sample configuration and edit with your settings:

```bash
cp .env_sample .env
nano .env
```

#### Critical Configuration Settings

**Home Assistant Connection:**
- `HASS_URL`: Your Home Assistant URL (e.g., `http://homeassistant.local:8123`)
- `HASS_TOKEN`: Long-Lived Access Token from your HA profile

**InfluxDB Connection:**
- `INFLUX_URL`: Your InfluxDB URL with port
- `INFLUX_TOKEN`: API token with read access to HA bucket
- `INFLUX_ORG`: Your InfluxDB organization
- `INFLUX_BUCKET`: Bucket with Home Assistant data

**Core Entity IDs (MUST match your HA setup):**
- `TARGET_INDOOR_TEMP_ENTITY_ID`: Desired indoor temperature
- `INDOOR_TEMP_ENTITY_ID`: Actual indoor temperature sensor
- `ACTUAL_OUTLET_TEMP_ENTITY_ID`: Heat pump outlet temperature sensor
- `TARGET_OUTLET_TEMP_ENTITY_ID`: Entity ML writes its calculated temperature to
- `ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID`: Entity that was actually applied (for learning)
  - **Active mode:** Same as `TARGET_OUTLET_TEMP_ENTITY_ID`
  - **Shadow mode:** Different entity (e.g., from your heat curve automation)
- `HEATING_STATUS_ENTITY_ID`: Climate entity to check heating mode

**See `.env_sample` for complete configuration with detailed comments.**

#### 3. Initial Calibration (Recommended)

Calibrate the thermal model on your historical data:

```bash
source .venv/bin/activate
python3 -m src.main --calibrate-physics
```

This trains the model on historical data from InfluxDB, learning your house's unique thermal characteristics.

#### 4. Test Run

Start in foreground to verify configuration:

```bash
python3 -m src.main
```

Watch the logs to ensure:
- Connects to Home Assistant successfully
- Reads all required sensors
- Makes predictions
- No errors or warnings

Press Ctrl+C to stop.

#### 5. Deploy as Systemd Service

Create the service file:

```bash
sudo nano /etc/systemd/system/ml_heating.service
```

Add the following content (**update paths for your installation**):

```ini
[Unit]
Description=ML Heating Control Service
After=network.target home-assistant.service influxdb.service
Wants=home-assistant.service influxdb.service

[Service]
Type=simple
User=your_user
WorkingDirectory=/opt/ml_heating
ExecStart=/opt/ml_heating/.venv/bin/python3 -m src.main
Restart=on-failure
RestartSec=5m
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ml_heating.service
sudo systemctl start ml_heating.service
```

#### 6. Monitor Operation

**Check service status:**
```bash
sudo systemctl status ml_heating.service
```

**View live logs:**
```bash
sudo journalctl -u ml_heating.service -f
```

## How It Works

### Architecture Overview

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Home Assistant │◄────►│  ML Heating      │◄────►│   InfluxDB      │
│                 │      │  Controller      │      │                 │
│ - Sensors       │      │                  │      │ - Historical    │
│ - Controls      │      │ - Thermal Model  │      │   Data          │
│ - Metrics       │◄──── │ - Online Learning│      │ - Features      │
│ - Live Perf     │      │ - Live Track     │      └─────────────────┘
└─────────────────┘      │ - Optimization   │      
                         └──────────────────┘      
                                │ Real-time Performance
                                ▼ Confidence & Accuracy Updates
                         ┌──────────────────┐      
                         │ Rolling Window   │      
                         │ Error Tracking   │      
                         │ Adaptive Learning│      
                         └──────────────────┘      
```

### The Thermal Model

The controller uses **ThermalEquilibriumModel** - a physics-based machine learning approach that combines thermodynamic principles with data-driven learning:

**Core Physics Parameters (learned from data):**
- Thermal time constant: How quickly the house responds to heating changes
- Heat loss coefficient: Rate of heat loss to outdoor environment
- Outlet effectiveness: Efficiency of heat transfer from outlet to indoor air
- Slab time constant: Thermal inertia of the underfloor heating slab
- Delta-T floor: Minimum temperature difference for effective heat transfer

**External Heat Sources (automatically calibrated via channel architecture):**
- PV solar warming: Heat gain from solar power generation (with solar lag and cloud-aware discount)
- Fireplace heating rate: Contribution with exponential decay after shutdown
- TV/electronics heat: Additive heat contribution from appliances
- Each source has its own independent learning channel preventing cross-contamination

> 📖 **For a deep-dive into all thermal parameters**, see [docs/THERMAL_PARAMETER_REFERENCE.md](docs/THERMAL_PARAMETER_REFERENCE.md)

**Enhanced Learning Features:**
- Adaptive learning: Parameters adjust based on prediction accuracy
- Differential-based effectiveness: Heat transfer scales with outlet-indoor temperature differential
- Weather forecast integration: Anticipates temperature changes
- PV forecast integration: Predicts solar heat gain (cloud-discounted)
- Slab passive delta tracking: Monitors `inlet_temp - indoor_temp` as a diagnostic for passive slab heating potential

**System States & Blocking:**
- DHW heating, defrosting, disinfection, DHW boost heater
- Automatically detected and handled with appropriate wait periods

### Model Calibration & Learning

**Initial Calibration (Recommended):**
```bash
python3 -m src.main --calibrate-physics
```
- Trains model on historical data from InfluxDB
- Duration: `TRAINING_LOOKBACK_HOURS` (default: 168 hours / 7 days)
- Learns your house's unique thermal characteristics
- Calibrates external heat source effects

**Continuous Online Learning:**
After calibration, the model learns from every heating cycle:
1. Sets outlet temperature based on current prediction
2. Waits one cycle (`CYCLE_INTERVAL_MINUTES`)
3. Measures actual indoor temperature change
4. Reads what temperature was actually applied (supports shadow mode)
5. Updates model parameters based on prediction accuracy
6. Improves future predictions

This happens automatically in both **active mode** (ML controls heating) and **shadow mode** (heat curve controls, ML learns).

### Active vs Shadow Mode

The system supports two operating modes:

**Active Mode:**
- ML directly controls heating by writing to `TARGET_OUTLET_TEMP_ENTITY_ID`
- Model learns from its own decisions
- Configuration: Set both entity IDs to the same value

**Shadow Mode:**
- Heat curve controls heating (writes to one entity)
- ML calculates but doesn't apply (writes to different entity)
- Model learns from heat curve's decisions
- Use shadow mode to:
  - Safely test ML before going active
  - Compare performance quantitatively
  - Continue learning while heat curve controls
  - Decide when ML is ready to take over

**Switching Modes:**
Simply change entity IDs in `.env` and restart:
```bash
# Shadow mode (different entities)
TARGET_OUTLET_TEMP_ENTITY_ID=sensor.ml_vorlauftemperatur
ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID=sensor.hp_target_temp_circuit1

# Active mode (same entity)
TARGET_OUTLET_TEMP_ENTITY_ID=sensor.ml_vorlauftemperatur
ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID=sensor.ml_vorlauftemperatur
```

Switch to active mode when you're confident in ML performance through shadow mode observation.

### Heating & Cooling Modes

The system supports both **heating** and **cooling** operation with fully independent thermal parameters and learning:

**Heating Mode** (default):
- Optimizes outlet temperature to warm the house
- Uses heating-specific parameter defaults and bounds
- State persisted in `unified_thermal_state.json`

**Cooling Mode**:
- Activated automatically when `HEATING_STATUS_ENTITY_ID` reports "cool"
- Uses dedicated cooling parameters (faster slab response, narrower outlet range 18–24°C)
- Independent state file (`unified_thermal_state_cooling.json`) prevents learning cross-contamination
- External heat sources (solar, fireplace, TV) act as loads *against* cooling
- Separate calibration tracking and learning confidence

Mode detection is automatic — no manual switching required.

### Monitoring & Diagnostics

The system publishes a suite of detailed sensors to Home Assistant for comprehensive monitoring and diagnostics:

#### `sensor.ml_heating_learning`
- **State**: Learning confidence score.
- **Key Attributes**:
    - `thermal_time_constant`, `total_conductance`, `equilibrium_ratio`, `heat_loss_coefficient`, `outlet_effectiveness`: Learned thermal parameters.
    - `cycle_count`, `parameter_updates`: Learning progress indicators.
    - `model_health`: Overall health status of the model.
    - `is_improving`, `improvement_percentage`: Trend of model performance.

#### `sensor.ml_model_mae`
- **State**: All-time Mean Absolute Error (MAE).
- **Key Attributes**:
    - `mae_1h`, `mae_6h`, `mae_24h`: Time-windowed MAE metrics.
    - `trend_direction`: "improving", "degrading", or "stable".
    - `prediction_count`: Total number of predictions.

#### `sensor.ml_model_rmse`
- **State**: All-time Root Mean Squared Error (RMSE).
- **Key Attributes**:
    - `recent_max_error`: The maximum prediction error in the recent past.
    - `std_error`: Standard deviation of prediction errors.
    - `mean_bias`: Systematic over/under-prediction.

#### `sensor.ml_prediction_accuracy`
- **State**: Percentage of "good" control actions in the last 24 hours (prediction within ±0.2°C of actual).
- **Key Attributes**:
    - `perfect_accuracy_pct`, `tolerable_accuracy_pct`, `poor_accuracy_pct`: Breakdown of prediction accuracy over 24 hours.
    - `excellent_all_time_pct`, `good_all_time_pct`: All-time accuracy metrics.
    - `prediction_count_24h`: Number of predictions in the last 24 hours.

## Analysis & Debugging

### Jupyter Notebooks

Analysis notebooks are included in `notebooks/` with comprehensive interpretation guides:

**Development Notebooks (`notebooks/development/`):**
- `01_hybrid_learning_strategy_development.ipynb` - Learning strategy development
- `02_mae_rmse_tracking_development.ipynb` - Performance tracking development
- `03_trajectory_prediction_development.ipynb` - Trajectory prediction development
- `04_historical_calibration_development.ipynb` - Historical calibration development

**Monitoring Notebooks (`notebooks/monitoring/`):**
- `01_hybrid_learning_monitor.ipynb` - Hybrid learning monitoring
- `02_prediction_accuracy_monitor.ipynb` - Prediction accuracy monitoring
- `03_trajectory_prediction_monitor.ipynb` - Trajectory prediction monitoring

**Legacy/Archive Notebooks (`notebooks/archive/`):**
- Historical development notebooks and experiments

To use notebooks:
```bash
source .venv/bin/activate
jupyter notebook notebooks/
```

### Command Line Options

**Calibration:**
```bash
python3 -m src.main --calibrate-physics
```
- Trains model on historical data from InfluxDB
- Duration: `TRAINING_LOOKBACK_HOURS` (default: 7 days)
- Then starts normal operation

**Validation:**
```bash
python3 -m src.main --validate-physics
```
- Tests model predictions across temperature ranges
- Exits without modifying anything

**Debug Mode:**
```bash
python3 -m src.main --debug
```
- Enables verbose logging
- Includes detailed feature vectors and model decisions

**Normal Operation:**
```bash
python3 -m src.main
```
- Standard operation mode
- Recommended for production use

### Operating Modes

**Starting in Shadow Mode (Recommended):**
1. Configure different entity IDs in `.env`
2. Keep your existing heat curve automation active
3. ML calculates but doesn't control heating
4. Monitor logs and performance
5. When confident, switch to active mode

**Switching to Active Mode:**
1. Update `.env` to use same entity for both
2. Disable your heat curve automation
3. Restart ml_heating service
4. ML now controls heating directly

### File Structure
```
ml_heating/
├── src/
│   ├── main.py                         # Main control loop
│   ├── thermal_equilibrium_model.py    # ThermalEquilibriumModel (physics core)
│   ├── model_wrapper.py                # Enhanced 7-stage prediction pipeline
│   ├── physics_features.py             # Feature engineering
│   ├── physics_calibration.py          # Historical training & calibration
│   ├── heat_source_channels.py         # Multi-source channel architecture (HP, PV, FP, TV)
│   ├── heating_controller.py           # Control orchestration
│   ├── temperature_control.py          # Temperature management & PV smoothing
│   ├── ha_client.py                    # Home Assistant integration (sensor publishing)
│   ├── ha_history_service.py           # HA history data retrieval
│   ├── influx_service.py               # InfluxDB queries
│   ├── state_manager.py                # State persistence
│   ├── unified_thermal_state.py        # Unified thermal state (heating)
│   ├── unified_thermal_state_cooling.py# Unified thermal state (cooling)
│   ├── prediction_metrics.py           # Performance tracking
│   ├── prediction_context.py           # Unified prediction context
│   ├── adaptive_fireplace_learning.py  # Adaptive fireplace learning
│   ├── adaptive_learning_metrics_schema.py # Learning metrics schema
│   ├── forecast_analytics.py           # Forecast analysis
│   ├── multi_heat_source_physics.py    # Multi-source physics
│   ├── sensor_buffer.py                # Sensor data buffering
│   ├── shadow_mode.py                  # Shadow mode logic
│   ├── thermal_config.py               # Central parameter configuration
│   ├── thermal_constants.py            # Physical constants
│   ├── thermal_parameters.py           # Parameter management
│   ├── thermal_state_validator.py      # State validation
│   ├── utils_metrics.py                # Custom MAE/RMSE metrics
│   └── config.py                       # Environment configuration
├── ml_heating/                         # Stable Home Assistant Add-on
│   ├── config.yaml                     # Production configuration
│   └── README.md                       # Add-on documentation
├── ml_heating_dev/                     # Alpha Home Assistant Add-on
│   ├── config.yaml                     # Development configuration
│   └── README.md                       # Development add-on docs
├── notebooks/                          # Analysis notebooks
│   ├── development/                    # Development notebooks
│   ├── monitoring/                     # Monitoring notebooks
│   └── archive/                        # Historical notebooks
├── tests/                              # Test suite (unit/ & integration/)
├── docs/                               # Documentation
├── memory-bank/                        # Project context & knowledge base
├── .env                                # Your configuration (not in git)
├── .env_sample                         # Configuration template
├── pyproject.toml                      # Project metadata & dev dependencies
├── requirements.txt                    # Python dependencies
├── CHANGELOG.md                        # Version history
├── RELEASE_TODO.md                     # Release preparation checklist
└── README.md                           # This file
```

## Troubleshooting

### Common Issues

**State Code 3: Network Error**
- Check Home Assistant is running and accessible
- Verify `HASS_URL` and `HASS_TOKEN` in `.env`
- Check network connectivity

**State Code 4: No Data**
- Check all required entity IDs exist in Home Assistant
- Look at `missing_sensors` attribute on `sensor.ml_heating_state`
- Verify entity IDs match exactly (case-sensitive)

**State Code 6: Heating Off**
- Heating system not in 'heat' or 'auto' mode
- Check your climate entity
- Normal during summer or when heating manually disabled

**State Code 7: Model Error**
- Check `last_error` attribute for details
- May indicate corrupted model file
- Try recalibrating: `--calibrate-physics`
- Check logs for Python exceptions

**Poor Performance**
- Recalibrate model with recent data
- Increase `CYCLE_INTERVAL_MINUTES` for better learning signal
- Check if blocking events are properly detected
- Verify all external heat sources configured correctly

### Debug Mode

Enable verbose logging:
```bash
# Temporarily for testing
python3 -m src.main --debug

# Or in service, edit /etc/systemd/system/ml_heating.service
# Add to [Service] section:
Environment="DEBUG=1"
```

### Emergency: Revert to Heat Curve

If something goes wrong:

```bash
# Stop ML heating
sudo systemctl stop ml_heating.service

# Re-enable your original heat curve automation in Home Assistant

# Or, quickly switch to shadow mode in .env:
ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID=sensor.hp_target_temp_circuit1
sudo systemctl restart ml_heating.service
```

The model continues learning in shadow mode, so you can analyze what went wrong and switch back when ready.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional external heat source integrations
- Alternative prediction algorithms
- Enhanced forecasting integration
- Grafana dashboard templates
- Documentation improvements

Please open an issue first to discuss major changes.

## License

See LICENSE file for details.

## Acknowledgments

This project builds on thermodynamic principles and machine learning techniques to create a practical, production-ready heating controller. Special thanks to the Home Assistant and InfluxDB communities for their excellent integration capabilities.
