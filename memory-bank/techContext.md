# Technology Context - ML Heating Control System

## Technology Stack

### Core Technologies

**Python 3.11+**
- **Primary Language**: Chosen for rich ML ecosystem and Home Assistant integration
- **Version**: 3.11+ required (per pyproject.toml) for modern type hints and performance improvements
- **Virtual Environment**: `.venv` for isolated dependency management

**Custom Metrics Framework**
- **Lightweight Implementation**: Custom NumPy-based metrics in `utils_metrics.py`
- **Minimal Dependencies**: No external ML framework dependencies
- **Key Components**: 
  - Custom `MAE` and `RMSE` classes for performance tracking
  - Physics-based models with incremental learning patterns
  - **Adaptive Learning Modules**: Specialized classes for source attribution (Fireplace, TV, PV)
- **Why Custom**: Eliminates compilation issues and reduces container build complexity

**Home Assistant REST API**
- **Integration Pattern**: RESTful API communication via HTTP/HTTPS
- **Authentication**: Long-lived access tokens
- **Bidirectional**: Read sensors + write calculated temperatures
- **Entity Management**: Create and update custom sensors for monitoring

**InfluxDB 2.x**
- **Time-Series Database**: Historical sensor data storage
- **Query Language**: Flux queries for feature engineering
- **Data Source**: Existing Home Assistant integration via InfluxDB connector
- **Usage Patterns**:
  - Historical data retrieval for model calibration
  - Feature importance export for analysis
  - Learning metrics tracking

### Development Infrastructure

**Jupyter Notebooks**
- **Analysis Platform**: 6 specialized notebooks for monitoring and debugging
- **Interactive Analysis**: Real-time model diagnostics and visualization
- **User Interface**: Primary tool for understanding system behavior
- **Notebooks**:
  - `00_learning_dashboard.ipynb` - At-a-glance status
  - `01_physics_model_diagnosis.ipynb` - Model validation
  - `02_performance_monitoring.ipynb` - Performance tracking
  - `03_behavior_analysis.ipynb` - Heating patterns
  - `04_model_validation.ipynb` - Physics compliance
  - `05_multilag_seasonal_analysis.ipynb` - Advanced features

**Systemd Service Management**
- **Production Deployment**: Linux systemd service for 24/7 operation
- **Auto-restart**: Configured with failure recovery and restart delays
- **Logging**: Journal integration for centralized log management
- **Dependencies**: Proper service ordering with Home Assistant and InfluxDB

**Environment Configuration**
- **dotenv**: `.env` file for configuration management
- **Secrets Management**: Tokens and sensitive data isolated from code
- **Configuration Validation**: Comprehensive parameter validation and defaults

## Key Dependencies

### Core Python Packages

```python
# requirements.txt key dependencies
numpy>=1.21.0           # Numerical computing for physics calculations
pandas>=1.3.0           # Data manipulation and feature engineering
requests>=2.26.0        # HTTP client for Home Assistant API
influxdb-client>=1.24.0 # InfluxDB 2.x integration
lightgbm                # Machine learning models
python-dotenv>=0.19.0   # Environment configuration
matplotlib>=3.5.0       # Visualization in notebooks
jupyter>=1.0.0          # Notebook environment
```

### System Dependencies

**Operating System Requirements**:
- **Linux**: Primary target (systemd service support)
- **Python 3.11+**: Language runtime
- **systemd**: Service management (production deployment)
- **Network Access**: Connectivity to Home Assistant and InfluxDB

**Optional Dependencies**:
- **Git**: Version control and updates
- **Docker**: Alternative containerized deployment
- **Grafana**: Advanced visualization (future enhancement)

## Architecture Decisions

### Why Custom Lightweight Implementation vs Traditional ML

**Traditional Approach Issues**:
- **Batch Training**: Requires periodic model retraining
- **Data Management**: Complex pipeline for collecting training data
- **Concept Drift**: Poor adaptation to seasonal/house changes
- **Deployment Complexity**: Separate training and inference infrastructure
- **Build Complexity**: External ML frameworks require compilation on multiple architectures

**Custom Lightweight Benefits**:
- **Continuous Learning**: Adapts every 30-minute cycle with physics-based models
- **No External Dependencies**: Pure NumPy implementation eliminates build issues
- **Container Optimization**: Builds successfully on ARM/x64 without Rust compilers
- **Simple Deployment**: Single process handles learning and inference
- **Memory Efficiency**: Incremental updates, no large datasets
- **Maintainable**: Custom metrics implementation is transparent and modifiable

### Heat Source Channel Architecture (Phase 2-4)

**Module**: `src/heat_source_channels.py`

**Problem Solved**: The single-model gradient descent contaminates HP parameters (OE, HLC) when uncontrollable heat sources (fireplace, solar) are active — it cannot distinguish which source caused a prediction error.

**Architecture**:
```
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │  HP Channel   │   │ Solar Channel │   │  FP Channel   │   │  TV Channel   │
  │  (controlled) │   │ (forecast)   │   │ (observed)    │   │ (minor)       │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                   │                   │                   │
         └───────────┬───────┘───────────────────┘───────────────────┘
                     ▼
              HeatSourceChannelOrchestrator
              Q_total = Q_hp + Q_solar + Q_fireplace + Q_tv
```

**Channel Isolation Rules** (via `route_learning()`):
- HP learns only from clean cycles (no fireplace, PV < 500W)
- Solar learns only during daytime (PV > 500W)
- Fireplace learns only when fireplace active
- TV learns only when TV on
- When multiple external sources active, each gets learning; HP never learns

**Key Classes**:
- `HeatSourceChannel` — ABC with `estimate_heat_contribution()`, `estimate_decay_contribution()`, `get_learnable_parameters()`, `apply_gradient_update()`
- `HeatPumpChannel` — Outlet effectiveness, slab time constant, delta-T floor
- `SolarChannel` — PV weight, solar lag, cloud factor; `predict_future_contribution()` for sunset pre-heating
- `FireplaceChannel` — Heat output kW, exponential decay τ (~45 min), room spread delay
- `TVChannel` — Simple additive heat weight
- `HeatSourceChannelOrchestrator` — Routes learning, combines channels, proportional error attribution

**Config**: `ENABLE_HEAT_SOURCE_CHANNELS=true` (env var, default true)

### Slab Model & Binary Search Fixes (April 2026)

**Slab Pump Gate**: Dual condition prevents false pump-ON when HP is off:
```python
pump_on = (float(outlet_temp) > t_slab and measured_delta_t >= 1.0)
```
When HP off (measured_delta_t < 1.0), slab always enters passive branch regardless of outlet/inlet relationship.

**HP-Off Binary Search**: When delta_t < 1.0, binary search substitutes the learned `delta_t_floor` (~2.55°C) so trajectories simulate "HP running at this outlet":
```python
if _dtf < 1.0:
    _dtf = self.thermal_model._resolve_delta_t_floor(_dtf)  # ~2.55
```
Without this, all candidates produce identical passive-slab predictions → "unreachable" → 35°C spike.

**Cloud Discount on PV Scalar**: Applied in `_extract_thermal_features()` before binary search:
```python
cloud_factor = max(min_factor, 1.0 - (cloud_pct / 100.0))
pv_scalar *= cloud_factor
```
Uses 1h cloud forecast (`cloud_cover_1h`). Prevents raw PV sensor spikes from causing outlet oscillation.

**PV Routing**: `_is_pv_active()` uses `max(pv_current, pv_smoothed) > 500W` to capture solar thermal lag at sunset.

**PV Smoothing**: Window shortened from 3h (18 readings) to `solar_decay_tau` (~30min, 3 readings) to exclude stale morning values.

**Slab Passive Delta**: `inlet_temp - indoor_temp` exported as `slab_passive_delta` diagnostic. Positive = passive heating available.

### Cooling Mode Architecture (April 2026)

**Module**: `src/unified_thermal_state_cooling.py`

**Problem Solved**: Heating and cooling modes have fundamentally different thermal dynamics — sharing the same state file caused learning cross-contamination where heating-tuned parameters degraded cooling performance and vice versa.

**Architecture**:
- `CoolingThermalStateManager` — dedicated state manager with own JSON file (`unified_thermal_state_cooling.json`)
- Singleton access via `get_cooling_state_manager()`
- Mode detection: `config.get_climate_mode()` returns `"heating"` / `"cooling"` / `"off"` based on `HEATING_STATUS_ENTITY_ID`
- `config.get_outlet_bounds()` and `config.get_fallback_outlet()` return mode-appropriate values

**Cooling-Specific Parameter Differences**:
- `slab_time_constant_hours`: 0.8h (vs 3.19h heating) — cold water through warm slab exchanges heat faster
- `outlet_temp_min/max`: 18–24°C (vs 0–35°C heating) — narrow cooling band
- `thermal_time_constant`: 3.0h (vs 4.39h) — cooling response is faster
- External heat sources act as loads *against* cooling (sign reversal)

**State Isolation**: Each mode has:
- Independent learning state (cycle count, confidence, parameter deltas)
- Separate calibration tracking (date, cycles)
- Own buffer state persistence for sensor snapshots
- Shadow-mode support via `get_effective_cooling_state_file()`

### Unified Thermal State Compression

**Module**: `src/unified_thermal_state.py`

Allow-list compression slims history entries on persistence and migration:
- **Channel history** keeps: error, context, parameters, changes; drops parameters_before/after
- **Parameter history** keeps: flat snapshot, changes, gradients; drops triple-stored before/after/channel_parameter_changes
- Controlled by `PREDICTION_CONTEXT_KEYS`, `PARAMETER_HISTORY_KEYS`, `CHANNEL_HISTORY_CONTEXT_KEYS`

### Physics-Based Model Design

**Hybrid Approach**:
```python
class RealisticPhysicsModel:
    def predict_one(self, features):
        # Physics-based core (domain knowledge)
        base_heating = outlet_effect * self.base_heating_rate
        target_boost = temp_gap * self.target_influence
        weather_adjustment = base_heating * outdoor_penalty
        
        # Data-driven external sources (learned)
        pv_contribution = self._calculate_pv_lagged()
        fireplace_contribution = self._calculate_fireplace_lagged()
        
        return base_heating + target_boost + weather_adjustment + pv_contribution + fireplace_contribution
```

**Benefits**:
- **Faster Convergence**: Physics knowledge reduces required training data
- **Interpretability**: Clear understanding of model decisions
- **Bounds Checking**: Natural constraints prevent unrealistic predictions
- **Robustness**: Graceful degradation when external sources unavailable

### Multi-Lag Learning Architecture

**Ring Buffer Implementation**:
```python
# Efficient time-delay tracking
self.pv_history = []      # Last 5 cycles (0-120min)
self.fireplace_history = [] # Last 4 cycles (0-90min)  
self.tv_history = []      # Last 2 cycles (0-30min)

def _update_histories(self, pv_power, fireplace, tv):
    self.pv_history.append(pv_power)
    if len(self.pv_history) > 5:
        self.pv_history.pop(0)  # Ring buffer behavior
```

**Correlation-Based Learning**:
```python
def _learn_pv_lags(self):
    # Calculate correlations between power at different lags and temperature changes
    for lag_idx in range(1, 5):
        lag_powers = [effect['power_history'][-1-lag_idx] for effect in recent_effects]
        correlation = np.corrcoef(lag_powers, temperature_changes)[0, 1]
        # Distribute total effect by correlation strength
```

### Seasonal Adaptation via Trigonometric Functions

**Mathematical Foundation**:
```python
# Automatic seasonal variation modeling
month_radians = 2 * π * current_month / 12
seasonal_multiplier = 1.0 + (
    pv_seasonal_cos * cos(month_radians) + 
    pv_seasonal_sin * sin(month_radians)
)

# Applied to external sources
pv_effect = base_pv_effect * seasonal_multiplier
```

**Learning Approach**:
- **Summer Data Collection**: Clean signal when HVAC is off
- **Correlation Analysis**: Learn cos/sin coefficients from HVAC-off periods
- **Automatic Modulation**: ±30-50% seasonal variation without manual adjustment

### Safety and Robustness Patterns

**Multi-Layer Safety Architecture**:

**Layer 1: Configuration Bounds**
```python
CLAMP_MIN_ABS = 14.0°C   # Absolute minimum outlet temperature
CLAMP_MAX_ABS = 65.0°C   # Absolute maximum outlet temperature
```

**Layer 2: Rate Limiting**
```python
MAX_TEMP_CHANGE_PER_CYCLE = 2°C  # Gradual changes only
if abs(delta) > max_change:
    final_temp = baseline + sign(delta) * max_change
```

**Layer 3: Blocking Detection**
```python
# Pause during heat pump system operations
blocking_entities = [DHW_STATUS, DEFROST_STATUS, DISINFECTION_STATUS]
if any_blocking_active:
    skip_prediction_cycle()
    preserve_state_for_grace_period()
```

**Layer 4: Grace Period Restoration**
```python
# Intelligent recovery after blocking ends
if blocking_just_ended:
    if outlet_hotter_than_target:  # DHW scenario
        use_aggressive_cooldown_target()
    else:  # Defrost scenario  
        restore_exact_previous_target()
    wait_for_outlet_stabilization()
```

**Layer 5: Physics Validation**
```python
# Monotonic enforcement - higher outlet must predict higher indoor
def enforce_monotonic(candidates, predictions):
    # Correct any thermodynamically impossible predictions
```

## Development Patterns

### Configuration Management

**Environment-Based Configuration**:
```python
# config.py pattern
HASS_URL = os.getenv("HASS_URL", "https://home.example.com")
HASS_TOKEN = os.getenv("HASS_TOKEN", "").strip()
CYCLE_INTERVAL_MINUTES = int(os.getenv("CYCLE_INTERVAL_MINUTES", "30"))
```

**Entity ID Mapping**:
```python
# Flexible entity mapping for different HA setups
TARGET_INDOOR_TEMP_ENTITY_ID = os.getenv("TARGET_INDOOR_TEMP_ENTITY_ID", "input_number.target_temp")
INDOOR_TEMP_ENTITY_ID = os.getenv("INDOOR_TEMP_ENTITY_ID", "sensor.indoor_temp")
```

### Error Handling and Resilience

**Network Resilience**:
```python
def create_ha_client():
    try:
        return HAClient(config.HASS_URL, config.HASS_TOKEN)
    except Exception:
        logging.error("Failed to create HA client")
        publish_network_error_state()
        raise
```

**Graceful Degradation**:
```python
# Handle missing optional sensors
pv_power = ha_client.get_state(config.PV1_POWER_ENTITY_ID) or 0.0
fireplace_on = ha_client.get_state(config.FIREPLACE_STATUS_ENTITY_ID, is_binary=True) or False

# Continue operation with reduced functionality
if all_critical_sensors_available:
    perform_full_prediction()
else:
    log_missing_sensors()
    skip_cycle_safely()
```

### Monitoring and Observability

**Comprehensive State Reporting**:
```python
# ML State Sensor with rich diagnostics
attributes = {
    'confidence': round(confidence, 4),
    'mae': round(mae.get(), 4), 
    'rmse': round(rmse.get(), 4),
    'suggested_temp': round(suggested_temp, 2),
    'predicted_indoor': round(predicted_indoor, 2),
    'missing_sensors': missing_sensors,
    'blocking_reasons': blocking_reasons,
    'last_prediction_time': datetime.now().isoformat()
}
```

**Learning Metrics Export**:
```python
# Export to InfluxDB for analysis
learning_metrics = model.export_learning_metrics()
influx_service.write_feature_importances(
    learning_metrics, 
    bucket=config.INFLUX_FEATURES_BUCKET,
    measurement="learning_parameters"
)
```

## Deployment Architecture

### Production Service Configuration

**Systemd Service Unit**:
```ini
[Unit]
Description=ML Heating Control Service
After=network.target home-assistant.service influxdb.service

[Service]
Type=simple
User=ml_heating
WorkingDirectory=/opt/ml_heating
ExecStart=/opt/ml_heating/.venv/bin/python3 -m src.main
Restart=on-failure
RestartSec=5m

[Install]
WantedBy=multi-user.target
```

**File System Layout**:
```
/opt/ml_heating/
├── .venv/                 # Python virtual environment
├── src/                   # Source code modules
├── notebooks/            # Analysis notebooks  
├── memory-bank/          # Documentation and context
├── .env                  # Configuration (not in git)
├── requirements.txt      # Python dependencies
└── unified_thermal_state.json # Unified thermal state
```

### Integration Patterns

**Home Assistant Integration**:
- **REST API**: Bidirectional communication via HTTP/HTTPS
- **Entity Creation**: Dynamic sensor creation for monitoring
- **Authentication**: Long-lived access tokens
- **Error Recovery**: Graceful handling of HA restarts/updates

**InfluxDB Integration**:
- **Historical Queries**: Feature engineering from time-series data
- **Metrics Export**: Learning parameters and feature importance
- **Batch Processing**: Efficient bulk data retrieval for calibration

**Logging and Monitoring**:
- **Systemd Journal**: Centralized logging via journalctl
- **Home Assistant Sensors**: Real-time status monitoring
- **Jupyter Notebooks**: Interactive analysis and debugging

This technology stack provides a robust, maintainable, and scalable foundation for sophisticated heating control with continuous learning and comprehensive safety mechanisms.
