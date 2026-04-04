"""
Solar transition scenario tests for Plan Steps 15-17.

Step 17 — Evening scenario: PV drops from 3000 W → 0 W over ~2 h.
  Verify: required outlet increases proactively before sunset.
  Verify: indoor temp stays close to target through the transition.

Morning scenario: PV ramps from 0 → 3000 W over ~2 h.
  Verify: outlet can be reduced as PV provides heat.
  Verify: no undershoot before sun is confirmed up.
  Verify: slab residual heat is accounted for (warm slab from night
          heating continues to emit heat even after heat pump stops).

Solar decay τ: sun-warmed surfaces emit residual heat after PV drops,
  smoothing the evening transition.

Fireplace independent learning: FireplaceChannel learns by its own
  gradient mechanism, no dependency on adaptive_fireplace_learning.
"""

import pytest

from src.heat_source_channels import (
    FireplaceChannel,
    HeatSourceChannelOrchestrator,
    SolarChannel,
    _get_min_records_for_learning,
)
from src.thermal_equilibrium_model import ThermalEquilibriumModel
from src import config


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_context(
    pv_power=0,
    pv_forecast=None,
    fireplace_on=0,
    tv_on=0,
    outlet_temp=40.0,
    current_indoor=21.0,
    outdoor_temp=5.0,
    inlet_temp=30.0,
    avg_cloud_cover=0.0,
):
    """Build a prediction context dict for channel / model tests."""
    return {
        "outlet_temp": outlet_temp,
        "current_indoor": current_indoor,
        "outdoor_temp": outdoor_temp,
        "pv_power": pv_power,
        "pv_forecast": pv_forecast,
        "fireplace_on": fireplace_on,
        "tv_on": tv_on,
        "avg_cloud_cover": avg_cloud_cover,
        "inlet_temp": inlet_temp,
        "delta_t": 5.0 if outlet_temp > inlet_temp else 0.0,
    }


@pytest.fixture()
def realistic_model():
    """ThermalEquilibriumModel with deterministic, physically meaningful
    parameters (overrides conftest TDD values for scenario realism)."""
    model = ThermalEquilibriumModel()
    model.thermal_time_constant = 4.0
    model.heat_loss_coefficient = 0.4
    model.outlet_effectiveness = 0.5
    model.slab_time_constant_hours = 1.0
    model.external_source_weights = {
        "pv": 0.002,
        "fireplace": 5.0,
        "tv": 0.2,
    }
    model.safety_margin = 0.5
    model.momentum_decay_rate = 0.1
    return model


# ===================================================================
#  Original regression test
# ===================================================================

def test_solar_transition_channel_learns():
    """PV channel learns when PV > threshold, not after sunset."""
    orch = HeatSourceChannelOrchestrator()
    ctx = _make_context(pv_power=1000, pv_forecast=[1000] * 12 + [0] * 12)
    orch.route_learning(1.0, ctx)
    assert len(orch.channels["pv"].history) == 1
    # After sunset — PV below threshold → PV channel must NOT learn again
    ctx2 = _make_context(pv_power=0, pv_forecast=[0] * 24)
    orch.route_learning(1.0, ctx2)
    assert len(orch.channels["pv"].history) == 1


# ===================================================================
#  Step 17 — Evening Scenario
# ===================================================================

class TestEveningScenario:
    """Sunny afternoon (PV = 3000 W) → evening (PV drops to 0 over 2 h)."""

    def test_solar_channel_predicts_declining_contribution(self):
        """PV channel's own forecast must decline as PV forecast drops."""
        ch = SolarChannel()
        pv_forecast = [3000, 2000, 1000, 0]  # hourly
        ctx = _make_context(pv_power=3000, pv_forecast=pv_forecast)
        future = ch.predict_future_contribution(horizon_hours=4, context=ctx)

        assert future[0] > future[-1], (
            f"Solar should decline: first={future[0]:.4f}, "
            f"last={future[-1]:.4f}"
        )
        # Last value should be well below peak (decay residual only)
        assert future[-1] < future[0] * 0.25, (
            f"After sunset, contribution should be <25% of peak: "
            f"peak={future[0]:.4f}, last={future[-1]:.4f}"
        )

    def test_higher_outlet_needed_without_pv(self, realistic_model):
        """Without PV, equilibrium is lower → higher outlet needed."""
        model = realistic_model
        outdoor = 5.0
        target = 21.0

        eq_pv = model.predict_equilibrium_temperature(
            outlet_temp=40.0, outdoor_temp=outdoor,
            current_indoor=target, pv_power=3000,
        )
        eq_no_pv = model.predict_equilibrium_temperature(
            outlet_temp=40.0, outdoor_temp=outdoor,
            current_indoor=target, pv_power=0,
        )
        assert eq_pv > eq_no_pv, (
            f"PV should raise equilibrium: with={eq_pv:.2f} "
            f"vs without={eq_no_pv:.2f}"
        )

    def test_trajectory_drops_when_pv_fades(self, realistic_model):
        """Fading PV with constant outlet gives lower final indoor temp
        than constant PV — proving PV fade affects trajectory."""
        model = realistic_model
        kw = dict(
            current_indoor=21.0, target_indoor=21.0,
            outlet_temp=30.0, outdoor_temp=5.0,
            time_horizon_hours=4, time_step_minutes=60,
            pv_power=3000,
        )
        traj_fade = model.predict_thermal_trajectory(
            pv_forecasts=[3000, 2000, 1000, 0], **kw
        )
        traj_const = model.predict_thermal_trajectory(
            pv_forecasts=[3000, 3000, 3000, 3000], **kw
        )
        assert traj_fade["trajectory"][-1] < traj_const["trajectory"][-1], (
            f"Fading PV should give lower indoor than constant PV: "
            f"fade={traj_fade['trajectory'][-1]:.2f}, "
            f"const={traj_const['trajectory'][-1]:.2f}"
        )

    def test_higher_outlet_compensates_pv_fade(self, realistic_model):
        """Higher outlet keeps indoor warmer through sunset transition."""
        model = realistic_model
        kw = dict(
            current_indoor=21.0, target_indoor=21.0, outdoor_temp=5.0,
            time_horizon_hours=4, time_step_minutes=60,
            pv_forecasts=[3000, 2000, 1000, 0], pv_power=3000,
        )
        low = model.predict_thermal_trajectory(outlet_temp=35.0, **kw)
        high = model.predict_thermal_trajectory(outlet_temp=42.0, **kw)
        assert high["trajectory"][-1] > low["trajectory"][-1], (
            f"Higher outlet should compensate for PV fade"
        )

    def test_evening_orchestrator_monotone_decline(self):
        """Orchestrator future heat must decline monotonically (clear sky,
        dropping PV forecast)."""
        orch = HeatSourceChannelOrchestrator()
        ctx = _make_context(
            pv_power=3000,
            pv_forecast=[3000, 2000, 500, 0],
        )
        future = orch.predict_future_heat(horizon_hours=4, context=ctx)
        mid = len(future) // 2
        assert future[0] > future[mid] > future[-1], (
            "Solar contribution should decline monotonically"
        )


# ===================================================================
#  Morning Scenario (with slab consideration)
# ===================================================================

class TestMorningScenario:
    """PV ramps from 0 → 3000 W over ~2 h.

    Key physics:
    - The slab (Estrich) stores heat from overnight HP operation.
    - Even after HP reduces / stops, warm slab continues emitting heat
      (``inlet_temp`` high → ``t_slab`` high → passive radiation).
    - Combined slab residual + PV must not cause overshoot.
    - Before sun is confirmed up, the system must not undershoot.
    """

    def test_solar_channel_predicts_rising_contribution(self):
        """PV channel's own forecast must rise as PV forecast increases."""
        ch = SolarChannel()
        ctx = _make_context(pv_power=0, pv_forecast=[0, 500, 1500, 3000])
        future = ch.predict_future_contribution(horizon_hours=4, context=ctx)

        assert future[-1] > future[0], (
            f"Solar should rise: first={future[0]:.4f}, "
            f"last={future[-1]:.4f}"
        )
        # First hour (PV=0) should have minimal PV contribution
        for val in future[:6]:
            assert val < 0.1, (
                f"Before sunrise, PV contribution should be ~0: {val:.4f}"
            )

    def test_overshoot_risk_high_pv_plus_high_outlet(self, realistic_model):
        """High PV + night-level outlet → equilibrium above target
        (overshoot risk)."""
        model = realistic_model
        eq = model.predict_equilibrium_temperature(
            outlet_temp=42.0, outdoor_temp=10.0,
            current_indoor=21.0, pv_power=4000,
        )
        assert eq > 21.0, (
            f"High PV + high outlet should overshoot: eq={eq:.2f}"
        )

    def test_reduced_outlet_closer_to_target(self, realistic_model):
        """Reducing outlet when PV is high brings equilibrium closer to
        target (overshoot prevention)."""
        model = realistic_model
        eq_high = model.predict_equilibrium_temperature(
            outlet_temp=42.0, outdoor_temp=10.0,
            current_indoor=21.0, pv_power=4000,
        )
        eq_low = model.predict_equilibrium_temperature(
            outlet_temp=30.0, outdoor_temp=10.0,
            current_indoor=21.0, pv_power=4000,
        )
        target = 21.0
        assert abs(eq_low - target) < abs(eq_high - target), (
            f"Reduced outlet should be closer to target: "
            f"low={eq_low:.2f}, high={eq_high:.2f}"
        )

    def test_no_undershoot_before_sunrise(self, realistic_model):
        """With night-level outlet and PV forecast showing sun in 2 h,
        indoor must NOT drop below target in the first 2 hours."""
        model = realistic_model
        traj = model.predict_thermal_trajectory(
            current_indoor=21.0, target_indoor=21.0,
            outlet_temp=42.0,  # night heating
            outdoor_temp=2.0,  # cold morning
            time_horizon_hours=4, time_step_minutes=60,
            pv_forecasts=[0, 0, 1000, 3000],
            pv_power=0,  # currently dark
        )
        # First 2 hours: indoor must stay near target (full heating, no PV)
        for temp in traj["trajectory"][:2]:
            assert temp >= 21.0 - 0.5, (
                f"No undershoot before sunrise: {temp:.2f}°C"
            )

    def test_slab_continues_heating_after_hp_reduces(self, realistic_model):
        """When HP outlet is reduced below slab (pump-off mode), the warm
        slab continues emitting heat to the room.

        Scenario: slab is warm (inlet = 30 °C) from overnight heating.
        HP outlet reduced to 25 °C (below slab → pump-off mode).
        Indoor should stay above outdoor because slab radiates.
        """
        model = realistic_model
        traj = model.predict_thermal_trajectory(
            current_indoor=21.0, target_indoor=21.0,
            outlet_temp=25.0,  # below inlet → pump-off mode
            outdoor_temp=5.0,
            time_horizon_hours=2, time_step_minutes=30,
            inlet_temp=30.0,  # warm slab from overnight
            pv_power=0,
        )
        # Slab at 30 °C radiates toward indoor; indoor should stay well
        # above outdoor (5 °C) for at least the first hour.
        first_hour = traj["trajectory"][:2]  # 2 × 30 min steps
        for temp in first_hour:
            assert temp > 15.0, (
                f"Warm slab should keep indoor above 15 °C: {temp:.2f}"
            )

    def test_slab_plus_pv_overshoot_vs_reduced_outlet(self, realistic_model):
        """Warm slab + rising PV with night-level outlet → overshoot.
        Reduced outlet → less overshoot."""
        model = realistic_model
        kw = dict(
            current_indoor=21.0, target_indoor=21.0, outdoor_temp=5.0,
            time_horizon_hours=4, time_step_minutes=60,
            pv_forecasts=[0, 800, 2000, 3000],
            pv_power=0, inlet_temp=30.0,
            delta_t_floor=2.0,  # HP is on
        )
        high = model.predict_thermal_trajectory(outlet_temp=42.0, **kw)
        low = model.predict_thermal_trajectory(outlet_temp=35.0, **kw)
        target = 21.0
        # Higher outlet → more overshoot
        assert abs(low["trajectory"][-1] - target) < abs(
            high["trajectory"][-1] - target
        ), "Reduced outlet should give less overshoot with slab + PV"

    def test_morning_orchestrator_rising_forecast(self):
        """Orchestrator future heat must rise as PV forecast increases."""
        orch = HeatSourceChannelOrchestrator()
        ctx = _make_context(pv_power=0, pv_forecast=[0, 1000, 2000, 3000])
        future = orch.predict_future_heat(horizon_hours=4, context=ctx)
        mid = len(future) // 2
        assert future[-1] > future[mid] > future[0], (
            "Solar contribution should rise with PV forecast"
        )


# ===================================================================
#  Solar Decay τ
# ===================================================================

class TestSolarDecay:
    """SolarChannel should model residual heat from sun-warmed surfaces
    after PV drops (exponential decay with ``solar_decay_tau_hours``)."""

    def test_decay_contribution_after_pv_drops(self):
        ch = SolarChannel()
        ctx = {"last_pv_power": 3000, "avg_cloud_cover": 0.0}
        # Immediately after PV drops to 0
        decay_0 = ch.estimate_decay_contribution(0.0, ctx)
        assert decay_0 == 0.0, "No decay at time=0"

        decay_10min = ch.estimate_decay_contribution(1 / 6, ctx)
        assert decay_10min > 0, "Should have residual heat after 10 min"

        decay_2h = ch.estimate_decay_contribution(2.0, ctx)
        assert decay_2h < decay_10min, "Decay should diminish over time"

    def test_decay_zero_without_prior_pv(self):
        ch = SolarChannel()
        ctx = {"last_pv_power": 0}
        assert ch.estimate_decay_contribution(0.5, ctx) == 0.0

    def test_predict_future_smoothes_evening_drop(self):
        """When PV forecast drops, predict_future_contribution should
        show a smooth (decay-smoothed) transition rather than a step."""
        ch = SolarChannel()
        ctx = _make_context(
            pv_power=3000,
            pv_forecast=[3000, 0, 0, 0],  # PV drops after hour 0
        )
        future = ch.predict_future_contribution(horizon_hours=4, context=ctx)

        # Hour 0 steps (0-5) are high
        assert future[0] > 0
        # Transition steps (6-11) should NOT jump to zero immediately
        # because of decay smoothing
        step_6 = future[6]  # first step of hour 1 (PV = 0 in forecast)
        # With decay τ = 0.5 h, step 6 is 10 min after drop → decayed
        assert step_6 > 0, (
            f"Decay smoothing should give residual heat: step_6={step_6:.4f}"
        )
        # But it should be less than peak
        assert step_6 < future[0], "Decayed value should be < peak"

    def test_predict_future_no_lag_on_increase(self):
        """PV increase should be immediate (no smoothing delay)."""
        ch = SolarChannel()
        ctx = _make_context(
            pv_power=0,
            pv_forecast=[0, 3000, 3000, 3000],
        )
        future = ch.predict_future_contribution(horizon_hours=4, context=ctx)
        # Hour 0 is 0, hour 1 jumps to full — no lag
        assert future[0] == 0.0
        # Step 6 = first step of hour 1 → should be full
        expected = 3000 * ch.pv_heat_weight  # cloud=0 → factor=1
        assert future[6] == pytest.approx(expected, rel=0.01)

    def test_solar_decay_tau_is_learnable(self):
        ch = SolarChannel()
        params = ch.get_learnable_parameters()
        assert "solar_decay_tau_hours" in params
        original = ch.solar_decay_tau_hours
        ch.apply_gradient_update(
            {"solar_decay_tau_hours": 1.0}, learning_rate=0.01
        )
        assert ch.solar_decay_tau_hours != original


# ===================================================================
#  Fireplace Independent Learning
# ===================================================================

class TestFireplaceIndependentLearning:
    """FireplaceChannel must learn via its own gradient mechanism —
    no dependency on adaptive_fireplace_learning.py."""

    def test_fireplace_learns_from_positive_error(self):
        """Positive error (actual > predicted) → fireplace provides more
        heat → fp_heat_output_kw should increase."""
        ch = FireplaceChannel()
        original_kw = ch.fp_heat_output_kw
        ctx = _make_context(fireplace_on=1)
        # Feed enough positive-error observations
        for _ in range(_get_min_records_for_learning()):
            ch.record_learning(error=1.0, context=ctx)
        assert ch.fp_heat_output_kw > original_kw, (
            f"Positive error should increase fp_heat_output_kw: "
            f"before={original_kw}, after={ch.fp_heat_output_kw}"
        )

    def test_fireplace_learns_from_negative_error(self):
        """Negative error (actual < predicted) → overestimated FP heat
        → fp_heat_output_kw should decrease."""
        ch = FireplaceChannel()
        original_kw = ch.fp_heat_output_kw
        ctx = _make_context(fireplace_on=1)
        for _ in range(_get_min_records_for_learning()):
            ch.record_learning(error=-1.0, context=ctx)
        assert ch.fp_heat_output_kw < original_kw, (
            f"Negative error should decrease fp_heat_output_kw: "
            f"before={original_kw}, after={ch.fp_heat_output_kw}"
        )

    def test_fireplace_no_learning_below_dead_zone(self):
        """Errors below dead zone should not trigger parameter updates."""
        ch = FireplaceChannel()
        original_kw = ch.fp_heat_output_kw
        ctx = _make_context(fireplace_on=1)
        for _ in range(_get_min_records_for_learning()):
            ch.record_learning(error=0.005, context=ctx)
        assert ch.fp_heat_output_kw == original_kw, (
            "Below dead zone: fp_heat_output_kw should not change"
        )

    def test_fireplace_no_adaptive_fireplace_dependency(self):
        """FireplaceChannel must not import or use
        adaptive_fireplace_learning."""
        import inspect
        import src.heat_source_channels as mod

        source = inspect.getsource(mod)
        assert "adaptive_fireplace_learning" not in source, (
            "heat_source_channels.py must not reference "
            "adaptive_fireplace_learning"
        )

    def test_fireplace_learning_via_orchestrator(self):
        """Orchestrator routes error to fireplace → fireplace self-learns."""
        orch = HeatSourceChannelOrchestrator()
        fp = orch.channels["fireplace"]
        original_kw = fp.fp_heat_output_kw
        ctx = _make_context(fireplace_on=1)
        for _ in range(_get_min_records_for_learning()):
            orch.route_learning(error=1.5, context=ctx)
        assert fp.fp_heat_output_kw > original_kw, (
            "Fireplace channel should self-learn through orchestrator"
        )

    def test_fireplace_params_bounded(self):
        """fp_heat_output_kw must stay within [0.5, 15.0]."""
        ch = FireplaceChannel()
        ctx = _make_context(fireplace_on=1)
        # Push hard with many large positive errors
        for _ in range(100):
            ch.record_learning(error=10.0, context=ctx)
        assert ch.fp_heat_output_kw <= 15.0
        # Push hard negative
        for _ in range(100):
            ch.record_learning(error=-10.0, context=ctx)
        assert ch.fp_heat_output_kw >= 0.5


# ===================================================================
#  Orchestrator Integration
# ===================================================================

class TestOrchestratorIntegration:
    """Orchestrator + ThermalEquilibriumModel work together."""

    def test_model_has_orchestrator(self):
        """When ENABLE_HEAT_SOURCE_CHANNELS is True, the model should
        have an orchestrator attribute."""
        model = ThermalEquilibriumModel()
        if config.ENABLE_HEAT_SOURCE_CHANNELS:
            assert model.orchestrator is not None

    def test_learning_routes_through_orchestrator(self):
        """update_prediction_feedback routes errors through the
        orchestrator."""
        model = ThermalEquilibriumModel()
        if model.orchestrator is None:
            pytest.skip("Orchestrator not enabled")

        ctx = _make_context(fireplace_on=1)
        model.update_prediction_feedback(
            predicted_temp=20.0, actual_temp=21.0,
            prediction_context=ctx,
        )
        fp_ch = model.orchestrator.channels["fireplace"]
        assert len(fp_ch.history) >= 1, (
            "Fireplace channel should have received learning record"
        )

    def test_orchestrator_future_forecast_consistency(self):
        """Constant PV forecast → steady heat contribution."""
        orch = HeatSourceChannelOrchestrator()
        ctx = _make_context(
            pv_power=2000,
            pv_forecast=[2000, 2000, 2000, 2000],
        )
        future = orch.predict_future_heat(horizon_hours=4, context=ctx)
        variation = max(future) - min(future)
        assert variation < 1.0, (
            f"Constant PV forecast should give steady heat: "
            f"variation={variation:.4f}"
        )

    def test_orchestrator_learning_routing_no_cross_contamination(self, monkeypatch):
        """Multiple scenarios must not cross-contaminate channels."""
        monkeypatch.setattr(config, "ENABLE_MIXED_SOURCE_ATTRIBUTION", False)
        orch = HeatSourceChannelOrchestrator()
        # PV active → only PV learns
        orch.route_learning(0.5, _make_context(pv_power=2000))
        assert len(orch.channels["pv"].history) == 1
        assert len(orch.channels["heat_pump"].history) == 0
        # Dark → only HP learns
        orch.route_learning(0.3, _make_context(pv_power=0))
        assert len(orch.channels["heat_pump"].history) == 1
        # FP + PV → both learn, HP unchanged
        orch.route_learning(
            0.4, _make_context(pv_power=2000, fireplace_on=1)
        )
        assert len(orch.channels["pv"].history) == 2
        assert len(orch.channels["fireplace"].history) == 1
        assert len(orch.channels["heat_pump"].history) == 1

    def test_error_attribution_proportional(self):
        """Error is attributed proportionally to active channels."""
        orch = HeatSourceChannelOrchestrator()
        ctx = _make_context(
            pv_power=2000, fireplace_on=1,
            outlet_temp=40.0, current_indoor=21.0,
        )
        attributed = orch.attribute_error(1.0, ctx)
        # Both PV and FP should get attribution
        assert "pv" in attributed or "fireplace" in attributed
        total = sum(attributed.values())
        assert total == pytest.approx(1.0, abs=0.01), (
            f"Total attribution should equal error: {total:.4f}"
        )
