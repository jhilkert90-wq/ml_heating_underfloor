# Active Context - Current Work & Decision State

### ✅ **CI Workflow fixes — May 2026**

#### **What changed**
- `.github/workflows/auto-docs.yaml` — upgraded `actions/checkout@v4` → `@v6` to fix Node.js 20 deprecation warning; changed API base URL from `https://api.githubcopilot.com` to `https://models.inference.ai.azure.com` to fix "server-to-server token not supported" error; updated model comment.
- `.github/workflows/ai-code-review.yaml` — same API base URL fix; model changed from `gpt-5.4` to `gpt-4.1`; updated footer comment and heading text to reference GitHub Models instead of Copilot.

#### **Why**
The `update-docs` workflow was emitting two warnings on every push to main:
1. `actions/checkout@v4` running on Node.js 20 (deprecated, forced to Node.js 24 from June 2026).
2. `AI call failed: checking server-to-server token: bad request` — the GitHub Copilot endpoint (`api.githubcopilot.com`) does not accept the `GITHUB_TOKEN` server-to-server token issued to GitHub Actions. The GitHub Models endpoint (`models.inference.ai.azure.com`) supports it with the `models: read` permission already present in both workflows.

#### **Files changed**
- `.github/workflows/auto-docs.yaml`
- `.github/workflows/ai-code-review.yaml`

---

### ✅ **Predictive Pre-Cooling Implementation — June 2025**

#### **What changed**
- NEW `src/overheating_predictor.py` — `OverheatingPredictor` class that runs a passive thermal trajectory simulation (HP OFF, outlet=inlet) using PV + outdoor forecasts to predict future room temperature peaks.
- `src/config.py` — 7 new `PRE_COOL_*` parameters (enabled, trigger margin, horizon, lead time, target offset, min PV, min outdoor).
- `src/main.py` — Integrated pre-cool check before `simplified_outlet_prediction()`. When `should_cool_now` and room ≤ target, shifts `target_indoor_temp` down by `PRE_COOL_TARGET_OFFSET_K` to make binary search start the HP proactively. Added pre-cool state to HA sensor attributes and persisted to unified thermal state.
- `ml_heating_underfloor/config.yaml` + `translations/en.yaml` — Config UI options and schema.
- NEW `tests/unit/test_overheating_predictor.py` (27 tests) + `tests/unit/test_pre_cooling_integration.py` (9 tests).

#### **Why**
Underfloor cooling starts too late when rooms are already overheated. Due to thermal inertia (~0.8h slab tau), active cooldown is nearly impossible once the room exceeds the cooling target. This predictive approach uses the existing physics model to look ahead and start cooling before overheating occurs.

#### **Key design decisions**
- Target-shift method: no changes to binary search algorithm needed — shifting target down makes existing logic find the correct cooling outlet (~20°C).
- Cooling mode only: safety gate prevents pre-cooling in heating/idle modes.
- Guard thresholds: both PV AND outdoor must be below minimums to block (either one being high is enough to allow pre-cooling).
- Reactive fallback: if room is already above target, pre-cooling fires regardless of forecast guards.

### ✅ **Cooling test helper cleanup — May 2026**

#### **What changed**
- `tests/unit/test_heat_source_channels.py` — simplified `make_context()` to accept override kwargs instead of a long explicit parameter list, while preserving the same derived defaults for HP-active `delta_t` and `thermal_power`.

#### **Why**
The follow-up review on the cooling regression tests called out that the helper signature had grown too large and was becoming harder to read. Converting it to an override-based helper keeps the test setup compact and makes future cooling routing assertions easier to extend without continually expanding the helper signature.

#### **Files changed**
- `tests/unit/test_heat_source_channels.py`
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

### ✅ **Cooling follow-up review fixes — May 2026**

#### **What changed**
- `src/heat_source_channels.py` — cooling `delta_t_floor` learning now stores `abs(delta_t)` for cooling samples, keeping the learned parameter as a positive magnitude.
- `src/temperature_control.py` — carries `climate_mode` into both active and shadow `prediction_context` payloads.
- Added regression tests for cooling HP+PV routing, PV decay co-routing, positive `delta_t_floor` learning, `climate_mode` propagation, and the RUNNING→RECOVERY gate branches.

#### **Files changed**
- `src/heat_source_channels.py`
- `src/temperature_control.py`
- `tests/unit/test_heat_source_channels.py`
- `tests/unit/test_temperature_control.py`
- `tests/unit/test_cooling_mode.py`
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`

---

### ✅ **Cooling gate: use existing HP detection — May 2026**

#### **What changed**
- Unified HP detection for cooling cycle gates by reusing `_is_heat_pump_active()` from `heat_source_channels.py` instead of a bespoke `delta_t < threshold` check.

#### **Files changed**
- `src/model_wrapper.py`
- `src/heat_source_channels.py`
- `src/temperature_control.py`
- `CHANGELOG.md`, `memory-bank/progress.md`, `memory-bank/activeContext.md`
