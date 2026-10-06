import pytest

from src.building_capacity import (
    resolve_capacities,
    room_capacity_kwh_per_k,
    slab_capacity_kwh_per_k,
)

PARAMS = {
    "thermal_time_constant": 4.75,
    "heat_loss_coefficient": 0.104,
    "outlet_effectiveness": 0.99,
    "slab_time_constant_hours": 4.03,
}


def _resolve(mode, params=PARAMS, flow=15.0, learned=None):
    return resolve_capacities(
        mode,
        params,
        manual_room=6.0,
        manual_slab=3.0,
        median_flow_l_min=flow,
        specific_heat_kj_per_kg_k=4.186,
        learned_slab=learned,
    )


def test_room_capacity_includes_the_effectiveness_conductance():
    assert room_capacity_kwh_per_k(PARAMS) == pytest.approx(4.75 * (0.104 + 0.99))


def test_room_capacity_without_effectiveness_falls_back_to_hlc_only():
    params = {"thermal_time_constant": 4.0, "heat_loss_coefficient": 0.1}

    assert room_capacity_kwh_per_k(params) == pytest.approx(0.4)


@pytest.mark.parametrize("missing", ["thermal_time_constant", "heat_loss_coefficient"])
def test_room_capacity_needs_tau_and_hlc(missing):
    params = {key: value for key, value in PARAMS.items() if key != missing}

    assert room_capacity_kwh_per_k(params) is None


def test_slab_capacity_from_flow_and_time_constant():
    assert slab_capacity_kwh_per_k(PARAMS, 15.0, 4.186) == pytest.approx(
        15.0 / 60.0 * 4.186 * 4.03
    )
    assert slab_capacity_kwh_per_k(PARAMS, None, 4.186) is None
    assert slab_capacity_kwh_per_k(PARAMS, 0.0, 4.186) is None


def test_model_mode_derives_both_capacities():
    result = _resolve("model")

    assert result["room"] == pytest.approx(5.1965, abs=1e-3)
    assert result["slab"] == pytest.approx(4.2175, abs=1e-3)
    assert result["source"] == "room:model,slab:model"


def test_manual_mode_uses_the_configured_values():
    assert _resolve("manual") == {"room": 6.0, "slab": 3.0, "source": "manual"}


def test_model_mode_falls_back_to_manual_values():
    params = {"thermal_time_constant": 4.0}

    result = _resolve("model", params=params, flow=None)

    assert (result["room"], result["slab"]) == (6.0, 3.0)
    assert result["source"] == "room:manual_fallback,slab:manual_fallback"


def test_learned_mode_prefers_the_learned_slab_capacity():
    result = _resolve("learned", learned=7.5)

    assert result["slab"] == 7.5
    assert result["source"] == "room:model,slab:learned"


def test_learned_mode_falls_back_to_the_model_slab_capacity():
    result = _resolve("learned", learned=None)

    assert result["source"] == "room:model,slab:model"
