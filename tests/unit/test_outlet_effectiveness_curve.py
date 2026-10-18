import pytest

from src.outlet_effectiveness_curve import (
    calibrate_curve,
    interpolate_curve,
    normalize_curve,
    update_curve_at_temperature,
)
from src.thermal_config import ThermalParameterConfig


def test_curve_interpolates_and_falls_back_to_scalar():
    curve = [
        {"outdoor_temp": 0, "outlet_effectiveness": 0.5},
        {"outdoor_temp": 10, "outlet_effectiveness": 1.5},
    ]

    assert interpolate_curve(curve, 5, 0.9) == pytest.approx(1.0)
    assert interpolate_curve(curve, -10, 0.9) == pytest.approx(0.5)
    assert interpolate_curve([], 5, 0.9) == pytest.approx(0.9)
    assert normalize_curve([{"outdoor_temp": "bad"}]) == []


def test_curve_calibration_shrinks_sparse_bins_to_scalar():
    periods = [
        {
            "outdoor_temp": 0.0,
            "indoor_temp": 20.0,
            "outlet_temp": 30.0,
            "effective_temp": 30.0,
        }
        for _ in range(20)
    ]

    curve = calibrate_curve(
        periods, 0.2, 0.8,
        ThermalParameterConfig.get_bounds("outlet_effectiveness"),
    )

    zero_bin = next(point for point in curve if point["outdoor_temp"] == 0)
    empty_bin = next(point for point in curve if point["outdoor_temp"] == -10)
    assert zero_bin["sample_count"] == 20
    assert 0.5 < zero_bin["outlet_effectiveness"] < 0.8
    assert empty_bin["sample_count"] == 0
    assert empty_bin["outlet_effectiveness"] == pytest.approx(0.8)


def test_curve_update_is_local_and_bounded():
    curve = [
        {"outdoor_temp": temp, "outlet_effectiveness": 0.5}
        for temp in (-5, 0, 5, 10)
    ]
    updated = update_curve_at_temperature(
        curve, 5, 10.0, (0.1, 1.0)
    )

    assert updated[2]["outlet_effectiveness"] == pytest.approx(1.0)
    assert updated[1]["outlet_effectiveness"] == pytest.approx(1.0)
    assert updated[0]["outlet_effectiveness"] == pytest.approx(0.5)
