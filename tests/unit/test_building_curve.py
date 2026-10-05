import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src import building_curve as bc


PARAMS = {
    "heat_loss_coefficient": 0.2,
    "outlet_effectiveness": 0.5,
    "pv_heat_weight": 0.002,
}


def test_base_outlet_ignores_external_and_is_raw():
    # (21*0.7 - 0.2*(-20)) / 0.5 = 37.4
    assert bc.base_outlet_temp(-20, 21, 0.2, 0.5) == pytest.approx(37.4)
    # raw: no clamping, may exceed outdoor bounds
    assert bc.base_outlet_temp(40, 21, 0.2, 0.5) < 21


def test_log_base_outlet_value_matches_temperature_balance_formula():
    assert bc.base_outlet_temp(8.7, 22.6, 0.104642, 0.985403) == pytest.approx(
        24.1, abs=0.05
    )


def test_base_outlet_no_effectiveness():
    assert bc.base_outlet_temp(0, 21, 0.2, 0.0) is None


def test_load_kw_heating_and_cooling():
    assert bc.building_load_kw(1, 21, 0.2) == pytest.approx(4.0)
    assert bc.building_load_kw(21, 21, 0.2) == 0.0
    assert bc.building_load_kw(31, 24, 0.2, "cooling") == pytest.approx(1.4)


def test_grid_ranges():
    h = bc.outdoor_grid("heating")
    c = bc.outdoor_grid("cooling")
    assert (h[0], h[-1]) == (-20.0, 20.0)
    assert (c[0], c[-1]) == (10.0, 40.0)


def test_polynomial_fit_exact_for_linear_curve():
    x = bc.outdoor_grid("heating")
    fit = bc.fit_polynomial(x, 0.2 * (21 - x), 4)
    assert len(fit["coeffs"]) == 5
    assert fit["rmse"] < 1e-6


def test_compute_curves_contains_all_coefficients():
    out = bc.compute_curves(
        {"heating": PARAMS, "cooling": PARAMS},
        {"heating": 21.0, "cooling": 24.0},
        degree=4,
    )
    for key in ("outlet", "kw"):
        for mode in ("heating", "cooling"):
            assert len(out[key][f"{mode}_poly_coeffs"]) == 5
            for p in range(5):
                assert f"{mode}_coef_x{p}" in out[key]
    assert len(out["kw"]["heating_values"]) == 41
    assert len(out["kw"]["cooling_values"]) == 31
    assert len(json.dumps(out["kw"])) < bc.MAX_ATTRIBUTE_BYTES


def test_degree_clamped():
    x = np.arange(0, 10.0)
    assert bc.fit_polynomial(x, x, 9)["degree"] == 4
    assert bc.fit_polynomial(x, x, 0)["degree"] == 1


def test_shrink_drops_points_when_too_large():
    attrs = {"heating_values": ["x" * 20000], "a": 1}
    slim = bc._shrink_to_limit(attrs)
    assert "heating_values" not in slim and slim["a"] == 1


def _model():
    model = MagicMock()
    model._get_current_export_parameters.return_value = dict(PARAMS)
    model.external_source_weights = {"pv": 0.002}
    return model


def test_publisher_publishes_two_sensors_then_throttles():
    ha = MagicMock()
    ha.set_state.return_value = True
    pub = bc.BuildingCurvePublisher()
    targets = {"heating": 21.0, "cooling": 24.0}
    models = {"heating": _model(), "cooling": _model()}
    assert pub.publish(ha, models, targets, "heating", 5.0, now=1000.0)
    assert ha.set_state.call_count == 2
    ids = [c.args[0] for c in ha.set_state.call_args_list]
    assert bc.BASE_OUTLET_ENTITY_ID in ids
    assert bc.BUILDING_KW_ENTITY_ID in ids
    attrs = ha.set_state.call_args_list[0].args[2]
    assert attrs["param_heat_loss_coefficient"] == 0.2
    assert attrs["param_weight_pv"] == 0.002
    ha.reset_mock()
    assert not pub.publish(ha, models, targets, "heating", 5.1, now=1100.0)
    assert ha.set_state.call_count == 0
    # outdoor change, then hourly refresh
    assert pub.publish(ha, models, targets, "heating", 8.0, now=1200.0)
    assert pub.publish(ha, models, targets, "heating", 8.0, now=5000.0)


def test_publisher_retries_after_a_sensor_write_fails():
    ha = MagicMock()
    ha.set_state.side_effect = [True, False]
    pub = bc.BuildingCurvePublisher()
    models = {"heating": _model(), "cooling": _model()}
    targets = {"heating": 21.0, "cooling": 24.0}

    assert not pub.publish(ha, models, targets, "heating", 5.0, now=1000.0)
    ha.set_state.side_effect = None
    ha.set_state.return_value = True
    assert pub.publish(ha, models, targets, "heating", 5.0, now=1001.0)


def test_publisher_skips_missing_outdoor_temperature():
    ha = MagicMock()
    pub = bc.BuildingCurvePublisher()
    models = {"heating": _model(), "cooling": _model()}

    assert not pub.publish(
        ha,
        models,
        {"heating": 21.0},
        "heating",
        None,
        now=1000.0,
    )
    ha.set_state.assert_not_called()


def test_publisher_exports_balance_estimate_after_stable_idle_samples():
    ha = MagicMock()
    ha.set_state.return_value = True
    pub = bc.BuildingCurvePublisher()
    models = {"heating": _model(), "cooling": _model()}
    targets = {"heating": 21.0, "cooling": 24.0}
    observation = {
        "mode": "heating",
        "indoor_temp": 22.0,
        "outdoor_temp": 10.0,
        "indoor_temp_delta_60m": 0.1,
        "thermal_power_kw": 0.0,
        "flow_rate": 0.0,
    }

    for now in (1000.0, 4600.0, 8200.0):
        assert pub.publish(
            ha,
            models,
            targets,
            "heating",
            10.0,
            now=now,
            balance_observation=observation,
        )

    attributes = ha.set_state.call_args_list[-1].args[2]
    assert attributes["heating_balance_outdoor_temp"] == pytest.approx(9.0)
    assert attributes["heating_balance_gain_kw"] == pytest.approx(2.4)
    assert attributes["heating_balance_estimate_method"] == (
        "stable_hvac_off_observations"
    )
    assert attributes["cooling_balance_outdoor_temp"] is None


def test_publisher_uses_both_mode_parameters_and_signs_both():
    ha = MagicMock()
    ha.set_state.return_value = True
    pub = bc.BuildingCurvePublisher()
    models = {"heating": _model(), "cooling": _model()}
    targets = {"heating": 21.0, "cooling": 24.0}

    assert pub.publish(ha, models, targets, "heating", 5.0, now=1000.0)
    attributes = ha.set_state.call_args_list[0].args[2]
    assert attributes["param_heating_heat_loss_coefficient"] == 0.2
    assert attributes["param_cooling_heat_loss_coefficient"] == 0.2
    ha.reset_mock()
    assert not pub.publish(ha, models, targets, "heating", 5.0, now=1001.0)

    models["cooling"]._get_current_export_parameters.return_value[
        "heat_loss_coefficient"
    ] = 0.4
    assert pub.publish(ha, models, targets, "heating", 5.0, now=1002.0)


def test_compute_curves_uses_matching_mode_parameters():
    cooling_params = {**PARAMS, "heat_loss_coefficient": 0.4}
    curves = bc.compute_curves(
        {"heating": PARAMS, "cooling": cooling_params},
        {"heating": 21.0, "cooling": 24.0},
    )

    assert curves["kw"]["heating_values"][0] == pytest.approx(8.2)
    assert curves["kw"]["cooling_values"][0] == pytest.approx(-5.6)


def test_balance_estimate_uses_observed_non_hvac_gains():
    curves = bc.compute_curves(
        {"heating": PARAMS, "cooling": {**PARAMS, "heat_loss_coefficient": 0.4}},
        {"heating": 21.0, "cooling": 24.0},
        balance_estimates={
            "heating": {
                "gain_kw": 0.6,
                "sample_count": 4,
                "window_minutes": 180,
                "uncertainty_k": 0.0,
            },
            "cooling": {
                "gain_kw": 0.8,
                "sample_count": 5,
                "window_minutes": 240,
                "uncertainty_k": 0.0,
            },
        },
    )

    assert curves["kw"]["heating_balance_outdoor_temp"] == pytest.approx(18.0)
    assert curves["kw"]["cooling_balance_outdoor_temp"] == pytest.approx(22.0)
    assert curves["kw"]["heating_no_gains_zero_load_outdoor_temp"] == 21.0
    assert curves["kw"]["cooling_no_gains_zero_load_outdoor_temp"] == 24.0
    assert curves["kw"]["heating_balance_sample_count"] == 4
    assert curves["kw"]["cooling_balance_window_minutes"] == 240
    assert curves["kw"]["heating_balance_uncertainty_k"] == 0.0


@pytest.mark.parametrize(
    "hlc,gains",
    [(0, 0.5), (-1, 0.5), (0.2, -0.1), (float("nan"), 0.5), (0.2, float("inf"))],
)
def test_balance_estimate_rejects_invalid_inputs(hlc, gains):
    assert bc.balance_outdoor_temp(21.0, hlc, gains) is None


def test_balance_estimate_is_unavailable_without_samples():
    curves = bc.compute_curves(
        {"heating": PARAMS, "cooling": PARAMS},
        {"heating": 21.0, "cooling": 24.0},
    )

    assert curves["kw"]["heating_balance_outdoor_temp"] is None
    assert curves["kw"]["heating_balance_estimate_status"] == "unavailable"
    assert curves["kw"]["heating_no_gains_zero_load_outdoor_temp"] == 21.0


def test_balance_estimator_requires_stable_hvac_off_observations():
    estimator = bc.BalancePointEstimator()
    common = {
        "mode": "heating",
        "indoor_temp": 22.0,
        "outdoor_temp": 10.0,
        "heat_loss_coefficient": 0.2,
        "indoor_temp_delta_60m": 0.1,
        "thermal_power_kw": 0.0,
        "flow_rate": 0.0,
    }

    assert estimator.observe(now=0, **common) is None
    assert estimator.observe(
        now=3600, **{**common, "thermal_power_kw": 0.5}
    ) is None
    assert estimator.observe(
        now=3600, **{**common, "indoor_temp_delta_60m": 0.5}
    ) is None
    assert estimator.observe(now=5400, **common) is None
    estimate = estimator.observe(now=7200, **common)

    assert estimate is not None
    estimate = estimator.get_estimate("heating", now=7200)
    assert estimate["gain_kw"] == pytest.approx(2.4)
    assert bc.balance_outdoor_temp(
        22.0, 0.2, estimate["gain_kw"]
    ) == pytest.approx(10.0)
    assert estimate["sample_count"] == 3
    assert estimate["window_minutes"] == 120
    assert estimate["uncertainty_k"] == pytest.approx(0.0)


def test_balance_estimator_uses_mode_specific_heat_loss_coefficient():
    estimator = bc.BalancePointEstimator()
    for now in (0, 3600, 7200):
        estimator.observe(
            mode="cooling",
            indoor_temp=25.0,
            outdoor_temp=15.0,
            heat_loss_coefficient=0.4,
            indoor_temp_delta_60m=0.0,
            thermal_power_kw=0.0,
            flow_rate=0.0,
            now=now,
        )

    estimate = estimator.get_estimate("cooling", now=7200)
    assert estimate["gain_kw"] == pytest.approx(4.0)
    assert bc.balance_outdoor_temp(
        24.0, 0.4, estimate["gain_kw"]
    ) == pytest.approx(14.0)


def test_balance_estimator_rejects_variable_gain_conditions():
    estimator = bc.BalancePointEstimator()
    for now, indoor_temp in ((0, 18.0), (3600, 22.0), (7200, 26.0)):
        estimator.observe(
            mode="heating",
            indoor_temp=indoor_temp,
            outdoor_temp=10.0,
            heat_loss_coefficient=0.2,
            indoor_temp_delta_60m=0.0,
            thermal_power_kw=0.0,
            flow_rate=0.0,
            now=now,
        )

    assert estimator.get_estimate(
        "heating", now=7200, heat_loss_coefficient=0.2
    ) is None
