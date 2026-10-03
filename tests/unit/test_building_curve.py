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
        PARAMS, {"heating": 21.0, "cooling": 24.0}, degree=4
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
    pub = bc.BuildingCurvePublisher()
    targets = {"heating": 21.0, "cooling": 24.0}
    assert pub.publish(ha, _model(), targets, "heating", 5.0, now=1000.0)
    assert ha.set_state.call_count == 2
    ids = [c.args[0] for c in ha.set_state.call_args_list]
    assert bc.BASE_OUTLET_ENTITY_ID in ids
    assert bc.BUILDING_KW_ENTITY_ID in ids
    attrs = ha.set_state.call_args_list[0].args[2]
    assert attrs["param_heat_loss_coefficient"] == 0.2
    assert attrs["param_weight_pv"] == 0.002
    ha.reset_mock()
    assert not pub.publish(ha, _model(), targets, "heating", 5.1, now=1100.0)
    assert ha.set_state.call_count == 0
    # outdoor change, then hourly refresh
    assert pub.publish(ha, _model(), targets, "heating", 8.0, now=1200.0)
    assert pub.publish(ha, _model(), targets, "heating", 8.0, now=5000.0)
