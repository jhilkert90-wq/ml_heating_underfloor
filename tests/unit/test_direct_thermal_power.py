import pytest

from src import config
from src.physics_features import calculate_thermodynamic_metrics


def test_direct_thermal_power_is_used_without_changing_spread_default(monkeypatch):
    kwargs = {
        "outlet_temp": 35.0,
        "inlet_temp": 30.0,
        "flow_rate": 10.0,
        "power_consumption": 1000.0,
    }

    monkeypatch.setattr(config, "THERMAL_POWER_SOURCE", "spread")
    spread = calculate_thermodynamic_metrics(**kwargs)
    assert spread["thermal_power_source"] == "spread"
    assert spread["thermal_power_kw"] == pytest.approx(
        10.0 / 60.0 * config.SPECIFIC_HEAT_CAPACITY * 5.0
    )

    monkeypatch.setattr(config, "THERMAL_POWER_SOURCE", "direct")
    direct = calculate_thermodynamic_metrics(
        **kwargs, thermal_power_direct_w=2500.0
    )
    assert direct["thermal_power_source"] == "direct"
    assert direct["thermal_power_kw"] == pytest.approx(2.5)
    assert direct["cop_realtime"] == pytest.approx(2.5)


def test_direct_negative_heat_is_clamped_only_during_normal_heating(monkeypatch):
    monkeypatch.setattr(config, "THERMAL_POWER_SOURCE", "direct")
    args = {
        "outlet_temp": 35.0,
        "inlet_temp": 30.0,
        "flow_rate": 10.0,
        "power_consumption": 1000.0,
        "thermal_power_direct_w": -500.0,
    }
    normal_heating = calculate_thermodynamic_metrics(**args)
    defrost = calculate_thermodynamic_metrics(**args, defrosting=True)
    cooling = calculate_thermodynamic_metrics(**args, climate_mode="cooling")
    assert normal_heating["thermal_power_kw"] == 0.0
    assert defrost["thermal_power_kw"] == pytest.approx(-0.5)
    assert cooling["thermal_power_kw"] == pytest.approx(-0.5)
