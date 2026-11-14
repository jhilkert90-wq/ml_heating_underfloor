# Validation Scripts

This directory contains validation and testing scripts for the ML Heating system. These scripts are used for development, debugging, and validation purposes rather than production unit tests.

## Directory Structure

```
validation/
├── README.md                          # This file
├── validate_equilibrium_scenarios.py  # Phase 4: Equilibrium equation validation
├── validate_trajectory_prediction.py  # Phase 4: Trajectory prediction validation
├── demo_multi_heat_source.py         # Multi-heat source demonstration
├── debug_physics_prediction.py       # Debug physics prediction components
├── debug_production_model.py         # Debug production model behavior
└── analyze_log_discrepancy.py        # Analyze log discrepancies
```

## Script Categories

### 🔬 Phase 4 Physics Validation (NEW)
- **`validate_equilibrium_scenarios.py`** - Validates corrected equilibrium equation physics
- **`validate_trajectory_prediction.py`** - Tests trajectory prediction with thermal dynamics

### 🧪 Scenario Testing
- **`demo_multi_heat_source.py`** - Multi-heat source demonstration and testing

### 🔍 Debug & Analysis
- **`debug_physics_prediction.py`** - Traces physics prediction components step-by-step
- **`debug_production_model.py`** - Debugs production model behavior with real parameters
- **`analyze_log_discrepancy.py`** - Analyzes discrepancies between expected and logged predictions

## Usage

### Offline trajectory-control comparison (Phase 3)

Run [compare_trajectory_control.py](compare_trajectory_control.py) **in a separate
offline process**, from the repository root:

```powershell
python -m validation.compare_trajectory_control 'C:\path\production.log' --comfort-band 0.1
# Optional override:
python -m validation.compare_trajectory_control 'C:\path\production.log' --output 'C:\path\comparison' --comfort-band 0.1
```

Without `--output`, all three files are written next to the effective
`UNIFIED_STATE_FILE` (including the existing shadow-path resolution), not next
to the input log or into a fixed session folder. For the supplied production
configuration this is `/config/ml_heating/`. The directory setting is read
without loading or changing the existing thermal state. An explicit `--output`
still takes precedence. Output files replace the previous offline run and are
ignored by Git. This is not an automatic per-cycle add-on export: files are
generated only when the offline command is run.

This tool has no live-control integration, HA client or publishing step. Model
state is isolated in a temporary directory. Its explicit `OFFLINE_PROFILE`
temporarily replaces process-local settings and restores them afterwards; never
call it inside a running add-on. It does not edit production configuration,
learn channel parameters, change cooling, or remove production stabilizers.

The comparison distinguishes the recorded production output, the current
corrected search plus FD-Newton, and a sampled one-dimensional alternative.
The alternative evaluates the **same existing simulator closure** over a 0.1 K
grid, including exact bounds and the recorded baseline. Its objective is
`sum(dt_h * (below_band_K**2 + above_band_K**2))` over four hours in ten-minute
steps, with equal under-/overshoot weights and a +/-0.1 K comfort band around the
price-adjusted target. There is no separate Newton post-correction. Equal-cost
candidates prefer the lower outlet when currently above the comfort band;
otherwise they prefer the smallest baseline change. Both methods pass through
the unchanged gradual-control, EMA and hold/hysteresis functions.

Outputs:
- `trajectory_control_comparison.log`: UTF-8 diagnostic log for the offline run,
  including existing controller messages and a completion marker.
- `trajectory_control_comparison.csv`: per-cycle raw and publishable outlets,
  separate under-/overshoot degree-hours and durations, squared violation scores,
  temperature extrema, execution timings and trajectory cross-check errors.
- `trajectory_control_summary.json`: input/script hashes, full offline profile,
  aggregate metrics, excluded cycles and reconstruction limitations.

#### Production-log comparison: 2026-11-14

The supplied log has 23 feature snapshots, all above room target; it contains no
real undershoot case. Twenty-one cycles from 19:01 to 22:27 were reconstructable.
Excluded: the initial partial 18:41 cycle, 18:51 without a same-cycle model
snapshot, and the 19:31 grace cycle without features. No previous/future cycle's
missing data is silently substituted.

GM and required/actual-supply attributes are absent. Sensor start/runtime windows
are therefore **frozen for both methods**, including known starts and runtimes.
For unknown idle starts at 19:57 and 20:07, an immediate-start/unlimited-runtime
scenario supplements the historical no-start window. These are explicit
conditional scenarios, not a reconstruction of Phase 1's candidate-dependent
start/runtime adjustments. Other assumptions include cloud cover 50%, rounded
search bounds, unchanged previous room target and zero remaining hold unless an
explicit hold line supplies the count. The profile includes assumptions where
the original configuration is not available.

| Cycle / historical window | Logged published | Corrected raw / publishable | Alternative raw / publishable |
|---|---:|---:|---:|
| 19:21 | 24.6 C | 23.4 / 24.3 C | 24.5 / 24.6 C |
| 19:47 | 25.3 C | 24.0 / 24.1 C | 25.1 / 24.6 C |
| 20:07 | 26.1 C | 24.0 / 24.5 C | 24.0 / 24.5 C |
| 21:07 | 25.3 C | 23.9 / 24.5 C | 23.9 / 24.5 C |

Across the 21 historical-window scenarios, 7 raw selections and 4 publishable
values differ; 17 publishable values agree. Neither method increases above the
recorded baseline in any of the 23 cycle/scenario rows. Mean simulated squared
comfort violation is 2.122112 versus 2.100186 K^2 h before stabilization
(1.03% lower for the alternative), and 2.139637 versus 2.134692 K^2 h after
stabilization (0.23% lower). These are means of overlapping, constant-outlet
four-hour predictions, **not measured comfort or energy savings**. Desktop
execution was approximately 14-16 ms for the current search/correction and
31-35 ms for the alternative; this is not a Pi benchmark.

The model is not monotonically warmer for every higher outlet near its pump
gate. For example, at 19:21 a constant 23.4 C outlet predicts 23.348 C after four
hours, whereas 24.5 C predicts 23.294 C with the same frozen 43-minute HP runtime
and 2.08 K loop delta. The optimizer can exploit this model response; it is not
proof that raising the real outlet during surplus improves comfort. The logged
trajectory cross-check differs by up to 0.0403 K, even when using its own logged
verification outlet rather than the post-correction output. Missing candidate
windows and rounded telemetry still prevent an exact replay.

**Conclusion:** the new objective is simpler, but this log demonstrates little
additional simulated benefit over the bugfixed controller and does not justify
a live switch or removing EMA/hold/gates. Next evidence needed: real undershoot
logs, complete decision-time HP sensor attributes/configuration, and validation
of the model response near the pump gate. Grace double-search and HA fetch
errors remain report-only and unchanged.

### Running Individual Scripts

```bash
# Test user scenario reproduction  
cd /opt/ml_heating
python validation/test_user_scenario.py

# Comprehensive model validation
python validation/test_model_validation.py

# Debug physics predictions
python validation/debug_physics_prediction.py
```

### Running from Validation Directory

```bash
cd /opt/ml_heating/validation

# Make sure to run from parent directory or adjust paths
python ../validation/test_user_scenario.py
```

## Key Validation Scripts

### 🏠 User Scenario Test
**File:** `test_user_scenario.py`

Reproduces the exact user problem scenario:
- Current: 20.4°C, Target: 21.0°C (0.6°C gap)
- Old selection: 14°C (WRONG)
- New selection: 65°C (CORRECT)

**Success Criteria:**
- Stage 1 detection (0.6°C > 0.2°C threshold)
- Maximum heating selection (≥60°C)
- CHARGING mode activation
- Significant improvement over old logic

### 📈 Model Validation
**File:** `test_model_validation.py`

Comprehensive validation using real InfluxDB data:
- Loads historical data with enhanced filtering
- Splits into 2/3 training, 1/3 testing
- Trains new model and evaluates on unseen data
- Compares with production model performance

**Validation Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Prediction accuracy distribution
- Physics behavior compliance

### 🔬 Phase 4 Equilibrium Validation (NEW)
**File:** `validate_equilibrium_scenarios.py`

Validates the corrected thermal equilibrium equation after Phase 1-3 physics fixes:
- Tests equilibrium calculation without thermal_time_constant interference
- Validates PV, fireplace, and TV heat source contributions
- Verifies energy conservation principles
- Tests multiple heat sources working additively

**Test Scenarios:**
1. No external heat sources (baseline)
2. PV solar contribution (1kW test case)
3. Fireplace active contribution
4. Multiple heat sources combined
5. Energy conservation verification

**Success Criteria:**
- Equilibrium temperatures within physical bounds (outdoor < indoor < outlet)
- External heat sources increase temperature additively
- Energy balance equation satisfied
- Physics checks pass for all scenarios

### 🔮 Phase 4 Trajectory Validation (NEW)
**File:** `validate_trajectory_prediction.py`

Tests thermal trajectory prediction functionality after physics corrections:
- Validates heating scenarios with thermal dynamics
- Tests PV solar contribution to trajectory
- Verifies cooling scenario behavior
- Tests weather forecast integration

**Test Scenarios:**
1. Heating scenario (18°C → 22°C)
2. Trajectory with PV contribution (500W)
3. Cooling scenario (25°C → 22°C)
4. Weather forecast integration

**Success Criteria:**
- Temperature increases in heating scenarios
- PV power increases final temperatures
- Weather forecasts affect trajectory calculations
- Thermal time constant influences trajectory shape

### 🔍 Physics Debug
**File:** `debug_physics_prediction.py`

Step-by-step tracing of physics predictions:
- Basic heating/cooling calculations
- External heat source contributions
- Forecast adjustments
- Multi-lag effects
- Physics bounds enforcement

## Unit Tests vs Validation Scripts

### Unit Tests (`../tests/`)
- **Purpose:** Automated testing of individual components
- **Scope:** Specific functions, classes, modules
- **Runtime:** Fast execution for CI/CD
- **Examples:** `test_heat_balance_controller.py`, `test_state_manager.py`

### Validation Scripts (`validation/`)
- **Purpose:** End-to-end validation, debugging, analysis
- **Scope:** Complete workflows, real data, scenarios
- **Runtime:** Longer execution, detailed analysis
- **Examples:** Model validation, user scenario reproduction

## Integration with Development Workflow

### Before Deployment
1. Run battery charger logic tests
2. Validate model performance with real data
3. Test physics constraint compliance
4. Debug any discrepancies

### After Issues
1. Reproduce problem with scenario tests
2. Debug predictions with analysis scripts
3. Validate fixes with comprehensive tests
4. Document results and learnings

### Model Updates
1. Run model validation with train/test split
2. Compare new vs production model
3. Test physics behavior compliance
4. Validate Heat Balance Controller logic

## Environment Requirements

- Python 3.8+
- Required packages: pandas, numpy, logging, pickle
- Access to InfluxDB (for model validation)
- ML Heating source code in `../src/`

## Contributing

When adding new validation scripts:

1. **Naming:** Use descriptive names starting with test_, debug_, or analyze_
2. **Documentation:** Include clear docstrings and purpose
3. **Structure:** Follow existing patterns for imports and organization
4. **Results:** Include clear pass/fail criteria and summary output
5. **README:** Update this README with script description

## Troubleshooting

### Import Errors
Ensure scripts run from project root or validation directory with proper path setup:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
```

### InfluxDB Connection Issues
Check InfluxDB configuration in `../src/config.py`:
- INFLUX_URL
- INFLUX_TOKEN  
- INFLUX_ORG
- INFLUX_BUCKET

### Model Loading Errors
Verify model files exist:
- `../unified_thermal_state.json` (production state)
- `../physics_model_validation.pkl` (validation model)

## Related Documentation

- [Unit Tests](../tests/) - Automated component testing
- [Source Code](../src/) - Main application code
- [Notebooks](../notebooks/) - Data analysis and exploration
- [Documentation](../docs/) - Project documentation
