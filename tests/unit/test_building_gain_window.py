import math
from types import SimpleNamespace

import pytest

from src.building_gain_window import (
    BalanceSettings,
    BuildingGainWindow,
    external_gain_kw,
)

HLC = 0.1
STEP = 600


def _sample(t, **overrides):
    sample = {
        "t": t,
        "outdoor": 12.0,
        "indoor": 22.0,
        "pv": 0.0,
        "fireplace": 0.0,
        "tv": 0.0,
        "thermal_power_kw": 0.4,
        "return_temp": 30.0,
        "outlet_temp": 33.0,
        "flow": 10.0,
        "mode": "heating",
        "dhw_active": False,
        "g_heating": 0.5,
        "g_cooling": 0.1,
    }
    sample.update(overrides)
    return sample


def _fill(window, hours, start=0, step=STEP, **overrides):
    last = start
    for index in range(int(hours * 3600 / step) + 1):
        last = start + index * step
        window.add_sample(_sample(last, **overrides))
    return last + step


def _no_storage(**kwargs):
    return BalanceSettings(storage_enabled=False, **kwargs)


# --- modelled estimate ----------------------------------------------------


def test_gain_is_unavailable_below_minimum_coverage():
    window = BuildingGainWindow()
    now = _fill(window, 5)

    assert window.gain_estimate("heating", now, HLC) is None


def test_gain_is_provisional_then_full():
    provisional = BuildingGainWindow()
    now = _fill(provisional, 7)
    estimate = provisional.gain_estimate("heating", now, HLC)

    assert estimate["status"] == "provisional"
    assert estimate["gain_kw"] == pytest.approx(0.5)
    assert estimate["uncertainty_k"] == pytest.approx(0.0)

    full = BuildingGainWindow()
    now = _fill(full, 24)
    assert full.gain_estimate("cooling", now, HLC)["status"] == "full"
    assert full.gain_estimate("cooling", now, HLC)["gain_kw"] == pytest.approx(0.1)


def test_gap_lowers_coverage_without_clearing_the_window():
    window = BuildingGainWindow()
    _fill(window, 3, g_heating=1.0)
    now = _fill(window, 3, start=18000, g_heating=0.0)

    estimate = window.gain_estimate("heating", now, HLC)

    assert estimate["window_minutes"] == 400
    assert estimate["gain_kw"] == pytest.approx(12600 / 24000)


def test_samples_without_modelled_gain_do_not_count_as_covered():
    window = BuildingGainWindow()
    now = _fill(window, 7, g_heating=None)

    assert window.gain_estimate("heating", now, HLC) is None
    assert window.gain_estimate("cooling", now, HLC) is not None


def test_window_length_and_provisional_minimum_follow_settings():
    settings = BalanceSettings(window_hours=12, min_provisional_hours=2)
    window = BuildingGainWindow(settings=settings)
    now = _fill(window, 3)

    assert window.gain_estimate("heating", now, HLC)["status"] == "provisional"

    now = _fill(window, 12, start=now)
    assert window.gain_estimate("heating", now, HLC)["status"] == "full"


def test_window_survives_restart(tmp_path):
    path = str(tmp_path / "window.json")
    first = BuildingGainWindow(path=path)
    now = _fill(first, 7)

    restored = BuildingGainWindow(path=path)

    assert restored.gain_estimate("heating", now, HLC)["gain_kw"] == pytest.approx(
        0.5
    )


def test_missing_directory_is_not_created(tmp_path):
    path = tmp_path / "missing" / "window.json"
    window = BuildingGainWindow(path=str(path))

    window.add_sample(_sample(0))

    assert not path.parent.exists()


def test_older_or_duplicate_timestamps_are_ignored():
    window = BuildingGainWindow()
    window.add_sample(_sample(600))
    window.add_sample(_sample(600))
    window.add_sample(_sample(300))

    assert len(window._samples) == 1


def test_median_flow_ignores_pump_off_and_dhw_samples():
    window = BuildingGainWindow()
    for index, (flow, dhw) in enumerate(
        [(0.0, False), (12.0, False), (20.0, False), (30.0, True)]
    ):
        window.add_sample(_sample(index * STEP, flow=flow, dhw_active=dhw))

    assert window.median_flow(4 * STEP) == pytest.approx(16.0)


# --- measured estimate ----------------------------------------------------


def test_measured_gain_balances_hp_heat_and_losses():
    window = BuildingGainWindow(settings=_no_storage())
    now = _fill(window, 7)

    estimate = window.measured_estimate("heating", now, HLC, None, None)

    # 0.1 * (22 - 12) - 0.4 kW heat pump
    assert estimate["gain_kw"] == pytest.approx(0.6)
    assert window.measured_estimate("cooling", now, HLC, None, None) is None


def test_defrost_counts_as_negative_heat_from_the_slab():
    window = BuildingGainWindow(settings=_no_storage())
    now = _fill(window, 7, thermal_power_kw=-0.5)

    estimate = window.measured_estimate("heating", now, HLC, None, None)

    assert estimate["gain_kw"] == pytest.approx(1.5)


def _dhw_window(settings):
    window = BuildingGainWindow(settings=settings)
    for index in range(49):
        window.add_sample(
            _sample(
                index * STEP,
                thermal_power_kw=1.0,
                dhw_active=18 <= index < 30,
            )
        )
    return window


def test_dhw_intervals_send_no_heat_to_the_floor_by_default():
    window = _dhw_window(_no_storage())

    estimate = window.measured_estimate("heating", 49 * STEP, HLC, None, None)

    # 12 of 48 intervals have P = 0 (gain 1.0), the rest 1.0 - 1.0 = 0.
    assert estimate["gain_kw"] == pytest.approx(0.25)
    assert estimate["window_minutes"] == 480


def test_dhw_intervals_can_be_excluded():
    window = _dhw_window(_no_storage(dhw_intervals="exclude"))

    estimate = window.measured_estimate("heating", 49 * STEP, HLC, None, None)

    assert estimate["gain_kw"] == pytest.approx(0.0)
    assert estimate["window_minutes"] == 360


def test_measured_requires_capacities_when_storage_is_enabled():
    window = BuildingGainWindow()
    now = _fill(window, 7)

    assert window.measured_estimate("heating", now, HLC, None, 4.0) is None
    assert window.measured_estimate("heating", now, HLC, 5.0, 4.0) is not None


def test_dhw_holds_the_last_return_temperature_for_the_storage_term():
    window = BuildingGainWindow()
    for index in range(49):
        dhw = 18 <= index < 30
        window.add_sample(
            _sample(
                index * STEP,
                thermal_power_kw=0.0,
                dhw_active=dhw,
                return_temp=55.0 if dhw else 30.0,
            )
        )

    estimate = window.measured_estimate("heating", 49 * STEP, HLC, 5.0, 4.0)

    # The DHW loop temperature never reaches the slab storage term.
    assert estimate["gain_kw"] == pytest.approx(HLC * 10.0)


def _simulate_two_node(hours, power_of, gain_of, outlet_offset=0.0):
    """Explicit-Euler room + slab model; exact closure for the estimator."""
    c_room, c_slab, oe, hlc, outdoor = 5.0, 4.0, 1.0, 0.1, 5.0
    room, slab = 21.0, 25.0
    dt_h = STEP / 3600.0
    samples, gains = [], []
    for index in range(int(hours * 3600 / STEP) + 1):
        power, gain = power_of(index), gain_of(index)
        samples.append(
            _sample(
                index * STEP,
                outdoor=outdoor,
                indoor=room,
                return_temp=slab,
                outlet_temp=slab + outlet_offset,
                thermal_power_kw=power,
                g_heating=gain,
            )
        )
        gains.append(gain)
        flux = oe * (slab + outlet_offset / 2.0 - room)
        room += dt_h / c_room * (flux - hlc * (room - outdoor) + gain)
        slab += dt_h / c_slab * (power - flux)
    return samples, gains


def test_measured_gain_recovers_true_gain_despite_daytime_preheating():
    def power(index):
        return 3.0 if 36 <= index < 84 else 0.0  # 6 h-14 h preheat, night off

    def gain(index):
        return 0.4 + 0.3 * math.sin(index / 23.0)

    samples, gains = _simulate_two_node(24, power, gain)
    window = BuildingGainWindow()
    for sample in samples:
        window.add_sample(sample)
    # Flux uses the slab temperature, so make that explicit for the helper.
    estimate = window.measured_estimate(
        "heating", samples[-1]["t"] + STEP, 0.1, 5.0, 4.0
    )

    assert estimate["status"] == "full"
    assert estimate["gain_kw"] == pytest.approx(
        sum(gains[:-1]) / len(gains[:-1]), abs=1e-9
    )


def test_ignoring_storage_biases_the_gain_after_preheating():
    def power(index):
        return 3.0 if 36 <= index < 84 else 0.0

    samples, gains = _simulate_two_node(24, power, lambda index: 0.5)
    window = BuildingGainWindow(settings=_no_storage())
    for sample in samples:
        window.add_sample(sample)

    estimate = window.measured_estimate(
        "heating", samples[-1]["t"] + STEP, 0.1, None, None
    )

    assert abs(estimate["gain_kw"] - 0.5) > 0.005


def test_learned_slab_capacity_recovers_the_simulated_capacity():
    def power(index):
        return 2.0 + 2.0 * math.sin(index / 7.0)

    samples, _ = _simulate_two_node(8, power, lambda index: 0.3, outlet_offset=3.0)
    window = BuildingGainWindow()
    for sample in samples:
        window.add_sample(sample)

    capacity = window.learned_slab_capacity(
        "heating", samples[-1]["t"] + STEP, 1.0
    )

    assert capacity == pytest.approx(4.0, abs=1e-6)


def test_learned_slab_capacity_needs_enough_excitation():
    window = BuildingGainWindow()
    now = _fill(window, 8)

    assert window.learned_slab_capacity("heating", now, 1.0) is None
    assert window.learned_slab_capacity("heating", now, None) is None


# --- settings and helpers -------------------------------------------------


def test_settings_default_values():
    settings = BalanceSettings.from_config(SimpleNamespace())

    assert settings == BalanceSettings()
    assert settings.method == "measured_with_fallback"
    assert settings.dhw_intervals == "zero_floor_power"


def test_settings_clamp_values_and_reject_unknown_choices():
    cfg = SimpleNamespace(
        BUILDING_BALANCE_METHOD="bogus",
        BUILDING_BALANCE_WINDOW_HOURS=1000,
        BUILDING_BALANCE_MIN_PROVISIONAL_HOURS=0,
        BUILDING_BALANCE_CAPACITY_MODE="manual",
        BUILDING_BALANCE_STORAGE_ENABLED=False,
        BUILDING_BALANCE_HVAC_OFF_MIN_SAMPLES="7",
        BUILDING_BALANCE_ROOM_CAPACITY_KWH_PER_K="nan",
    )

    settings = BalanceSettings.from_config(cfg)

    assert settings.method == "measured_with_fallback"
    assert settings.window_hours == 72
    assert settings.min_provisional_hours == 1
    assert settings.capacity_mode == "manual"
    assert settings.storage_enabled is False
    assert settings.hvac_off_min_samples == 7
    assert settings.room_capacity_kwh_per_k == 5.0


class _Model:
    heat_loss_coefficient = 0.2

    def predict_equilibrium_temperature(self, **kwargs):
        assert kwargs["thermal_power"] == 0.0
        return kwargs["outdoor_temp"] + 0.001 * kwargs["pv_power"][-1] / 0.2


def test_external_gain_kw_uses_model_energy_branch():
    assert external_gain_kw(_Model(), [0.0, 500.0], 5.0, 0.0, 0.0) == pytest.approx(
        0.5
    )
