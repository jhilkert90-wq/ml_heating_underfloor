import pytest

from src import config
from src import model_wrapper
from src import thermal_equilibrium_model
from src import unified_thermal_state
from src.heat_source_channels import _get_min_records_for_learning
from src.model_wrapper import get_enhanced_model_wrapper


def make_learning_context(
    fireplace_on=0,
    pv_power=0.0,
    heat_pump_active=False,
):
    return {
        "outlet_temp": 40.0 if heat_pump_active else 21.0,
        "current_indoor": 20.0,
        "outdoor_temp": 5.0,
        "target_temp": 21.0,
        "pv_power": pv_power,
        "fireplace_on": fireplace_on,
        "tv_on": 0.0,
        "avg_other_rooms_temp": 20.0,
        "living_room_temp": 24.0 if fireplace_on else 20.0,
        "avg_cloud_cover": 50.0,
        "inlet_temp": 30.0 if heat_pump_active else 20.0,
        "delta_t": 5.0 if heat_pump_active else 0.0,
        "thermal_power": 2.0 if heat_pump_active else 0.0,
        "heat_pump_active": heat_pump_active,
    }


@pytest.fixture
def isolated_channel_state(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ENABLE_HEAT_SOURCE_CHANNELS", True)
    monkeypatch.setattr(config, "ENABLE_MIXED_SOURCE_ATTRIBUTION", False)
    monkeypatch.setattr(
        model_wrapper.EnhancedModelWrapper,
        "_export_metrics_to_influxdb",
        lambda self: None,
    )
    monkeypatch.setattr(
        model_wrapper.EnhancedModelWrapper,
        "export_metrics_to_ha",
        lambda self: None,
    )

    state_manager = unified_thermal_state.ThermalStateManager(
        state_file=str(tmp_path / "thermal_state.json")
    )
    unified_thermal_state._thermal_state_manager = state_manager
    model_wrapper._enhanced_model_wrapper_instance = None

    yield state_manager

    model_wrapper._enhanced_model_wrapper_instance = None
    unified_thermal_state._thermal_state_manager = None


@pytest.mark.parametrize(
    ("scenario", "context_kwargs", "expected_channels"),
    [
        ("none", {}, {"heat_pump"}),
        ("heat_pump", {"heat_pump_active": True}, {"heat_pump"}),
        ("pv", {"pv_power": 1500.0}, {"pv"}),
        ("fireplace", {"fireplace_on": 1}, {"fireplace"}),
        (
            "fireplace_pv",
            {"fireplace_on": 1, "pv_power": 1500.0},
            {"fireplace", "pv"},
        ),
        (
            "fireplace_heat_pump",
            {"fireplace_on": 1, "heat_pump_active": True},
            {"fireplace"},
        ),
        (
            "heat_pump_pv",
            {"heat_pump_active": True, "pv_power": 1500.0},
            {"pv"},
        ),
        (
            "all_three",
            {
                "fireplace_on": 1,
                "heat_pump_active": True,
                "pv_power": 1500.0,
            },
            {"fireplace", "pv"},
        ),
    ],
)
def test_channel_mode_persists_expected_learning_routes(
    isolated_channel_state,
    scenario,
    context_kwargs,
    expected_channels,
):
    """Persisted channel state should match full feedback routing."""
    model = thermal_equilibrium_model.ThermalEquilibriumModel()

    assert model.orchestrator is not None

    context = make_learning_context(**context_kwargs)
    n = _get_min_records_for_learning() + 1
    for _ in range(n):
        model.update_prediction_feedback(
            predicted_temp=20.0,
            actual_temp=21.0,
            prediction_context=context,
        )

    persisted_state = isolated_channel_state.get_heat_source_channel_state()

    for channel_name in ("heat_pump", "pv", "fireplace", "tv"):
        expected_history = n if channel_name in expected_channels else 0
        assert persisted_state[channel_name]["history_count"] == expected_history, (
            f"Scenario {scenario} routed unexpected history into "
            f"{channel_name}"
        )
        assert len(persisted_state[channel_name].get("history", [])) == expected_history, (
            f"Scenario {scenario} persisted unexpected history payload for "
            f"{channel_name}"
        )


def test_wrapper_channel_workflow_persists_fireplace_learning_across_restart(
    isolated_channel_state,
):
    """Channel mode should learn, persist, reload, and reuse fireplace state."""
    wrapper = get_enhanced_model_wrapper()
    assert wrapper.adaptive_fireplace is None

    wrapper.cycle_count = 2
    isolated_channel_state.update_learning_state(cycle_count=2)
    wrapper._current_features = {
        "living_room_temp": 24.0,
        "other_rooms_temp": 20.0,
    }

    baseline_prediction = wrapper.predict_indoor_temp(
        outlet_temp=21.0,
        outdoor_temp=5.0,
        current_indoor=20.0,
        fireplace_on=1,
        pv_power=0.0,
        tv_on=0.0,
    )

    fireplace_channel = wrapper.thermal_model.orchestrator.channels[
        "fireplace"
    ]
    initial_fireplace_power = fireplace_channel.fp_heat_output_kw

    learning_context = make_learning_context(fireplace_on=1)
    n = _get_min_records_for_learning() + 1
    for _ in range(n):
        wrapper.learn_from_prediction_feedback(
            predicted_temp=20.0,
            actual_temp=25.0,
            prediction_context=learning_context,
        )

    learned_fireplace_power = fireplace_channel.fp_heat_output_kw
    persisted_state = isolated_channel_state.get_heat_source_channel_state()

    assert learned_fireplace_power > initial_fireplace_power
    assert persisted_state["fireplace"]["parameters"][
        "fp_heat_output_kw"
    ] == pytest.approx(learned_fireplace_power)
    assert len(persisted_state["fireplace"].get("history", [])) == n

    model_wrapper._enhanced_model_wrapper_instance = None
    reloaded_wrapper = get_enhanced_model_wrapper()
    reloaded_wrapper._current_features = {
        "living_room_temp": 24.0,
        "other_rooms_temp": 20.0,
    }

    restored_fireplace_power = (
        reloaded_wrapper.thermal_model.orchestrator.channels[
            "fireplace"
        ].fp_heat_output_kw
    )
    restored_fireplace_history = reloaded_wrapper.thermal_model.orchestrator.channels[
        "fireplace"
    ].history
    reloaded_prediction = reloaded_wrapper.predict_indoor_temp(
        outlet_temp=21.0,
        outdoor_temp=5.0,
        current_indoor=20.0,
        fireplace_on=1,
        pv_power=0.0,
        tv_on=0.0,
    )

    assert reloaded_wrapper.adaptive_fireplace is None
    assert restored_fireplace_power == pytest.approx(learned_fireplace_power)
    assert len(restored_fireplace_history) == n
    assert reloaded_prediction > baseline_prediction