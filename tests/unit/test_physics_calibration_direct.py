"""
Tests for the physics-direct calibration path (physics_calibration_direct).
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers to build realistic synthetic data
# ---------------------------------------------------------------------------

def _make_stable_periods(n=60, oe=0.95, hlc=0.12):
    """Generate synthetic stable-period dicts consistent with the equilibrium model."""
    rng = np.random.default_rng(42)
    periods = []
    for i in range(n):
        t_out = rng.uniform(-5, 10)
        t_outlet = rng.uniform(30, 45)
        t_in = (oe * t_outlet + hlc * t_out) / (oe + hlc) + rng.normal(0, 0.05)
        periods.append({
            "indoor_temp": float(t_in),
            "outdoor_temp": float(t_out),
            "outlet_temp": float(t_outlet),
            "effective_temp": float(t_outlet),
            "pv_power": 0.0,
            "fireplace_on": 0,
            "tv_on": 0,
            "thermal_power_kw": 1.0,
            "minutes_since_defrost": 999.0,
            "inlet_temp": float(t_outlet) - 3.0,
        })
    return periods


def _make_pv_periods(n=30, oe=0.95, hlc=0.12, pv_w=0.0020):
    """Generate synthetic PV-on period dicts."""
    rng = np.random.default_rng(7)
    periods = []
    for _ in range(n):
        t_out = rng.uniform(-5, 10)
        t_outlet = rng.uniform(30, 45)
        pv = rng.uniform(200, 2000)
        p_pv = pv * pv_w
        denom = oe + hlc
        t_in = (oe * t_outlet + hlc * t_out + p_pv) / denom + rng.normal(0, 0.05)
        periods.append({
            "indoor_temp": float(t_in),
            "outdoor_temp": float(t_out),
            "effective_temp": float(t_outlet),
            "outlet_temp": float(t_outlet),
            "pv_power": float(pv),
            "fireplace_on": 0,
            "tv_on": 0,
            "thermal_power_kw": 1.0,
        })
    return periods


def _make_fp_periods(n=20, oe=0.95, hlc=0.12, fp_kw=0.38):
    """Generate synthetic fireplace-on period dicts."""
    rng = np.random.default_rng(99)
    periods = []
    for _ in range(n):
        t_out = rng.uniform(-5, 10)
        t_outlet = rng.uniform(30, 45)
        denom = oe + hlc
        t_in = (oe * t_outlet + hlc * t_out + fp_kw) / denom + rng.normal(0, 0.05)
        periods.append({
            "indoor_temp": float(t_in),
            "outdoor_temp": float(t_out),
            "effective_temp": float(t_outlet),
            "outlet_temp": float(t_outlet),
            "pv_power": 0.0,
            "fireplace_on": 1,
            "tv_on": 0,
            "thermal_power_kw": 1.0,
        })
    return periods


# ---------------------------------------------------------------------------
# Tests for _calibrate_oe_analytical
# ---------------------------------------------------------------------------

class TestAnalyticalOE:
    """Unit tests for the analytical OE estimator."""

    def test_recovers_known_oe(self):
        """Analytical OE should recover the ground truth OE within 5 %."""
        from src.physics_calibration_direct import _calibrate_oe_analytical

        true_oe = 0.95
        true_hlc = 0.12
        periods = _make_stable_periods(n=80, oe=true_oe, hlc=true_hlc)

        with patch("src.physics_calibration_direct.config") as mc:
            mc.HEATING_MIN_THERMAL_POWER_KW = 0.5
            mc.DEFROST_RECOVERY_GRACE_MINUTES = 45
            result = _calibrate_oe_analytical(periods, hlc=true_hlc)

        assert result is not None
        assert abs(result - true_oe) / true_oe < 0.05, (
            f"OE estimate {result:.4f} deviates more than 5 % from true {true_oe}"
        )

    def test_returns_none_for_zero_hlc(self):
        from src.physics_calibration_direct import _calibrate_oe_analytical
        periods = _make_stable_periods(n=20)
        result = _calibrate_oe_analytical(periods, hlc=0.0)
        assert result is None

    def test_returns_none_for_insufficient_periods(self):
        from src.physics_calibration_direct import _calibrate_oe_analytical
        # Only 3 periods — below the minimum of 10
        periods = _make_stable_periods(n=3)
        with patch("src.physics_calibration_direct.config") as mc:
            mc.HEATING_MIN_THERMAL_POWER_KW = 0.5
            mc.DEFROST_RECOVERY_GRACE_MINUTES = 45
            result = _calibrate_oe_analytical(periods, hlc=0.12)
        assert result is None

    def test_respects_bounds(self):
        """Result must always be within ThermalParameterConfig bounds."""
        from src.physics_calibration_direct import _calibrate_oe_analytical
        from src.thermal_config import ThermalParameterConfig

        periods = _make_stable_periods(n=80)
        with patch("src.physics_calibration_direct.config") as mc:
            mc.HEATING_MIN_THERMAL_POWER_KW = 0.5
            mc.DEFROST_RECOVERY_GRACE_MINUTES = 45
            result = _calibrate_oe_analytical(periods, hlc=0.12)

        if result is not None:
            lo, hi = ThermalParameterConfig.get_bounds("outlet_effectiveness")
            assert lo <= result <= hi


# ---------------------------------------------------------------------------
# Tests for _residual_heat_source_weight
# ---------------------------------------------------------------------------

class TestResidualHeatSourceWeight:
    """Unit tests for the residual heat-source weight estimator."""

    def test_pv_weight_recovery(self):
        """PV weight should be recovered within 10 % from synthetic data."""
        from src.physics_calibration_direct import _residual_heat_source_weight

        true_oe = 0.95
        true_hlc = 0.12
        true_pv_w = 0.0020

        periods = _make_pv_periods(n=60, oe=true_oe, hlc=true_hlc, pv_w=true_pv_w)
        result = _residual_heat_source_weight(
            periods, "pv", true_hlc, true_oe, min_periods=5
        )
        assert result is not None
        assert abs(result - true_pv_w) / true_pv_w < 0.10, (
            f"PV weight estimate {result:.6f} deviates more than 10 % "
            f"from true {true_pv_w}"
        )

    def test_fireplace_weight_recovery(self):
        """Fireplace weight should be recovered within 10 % from synthetic data."""
        from src.physics_calibration_direct import _residual_heat_source_weight

        true_oe = 0.95
        true_hlc = 0.12
        true_fp = 0.38

        periods = _make_fp_periods(n=30, oe=true_oe, hlc=true_hlc, fp_kw=true_fp)
        result = _residual_heat_source_weight(
            periods, "fp", true_hlc, true_oe, min_periods=5
        )
        assert result is not None
        assert abs(result - true_fp) / true_fp < 0.10, (
            f"FP weight estimate {result:.4f} deviates more than 10 % "
            f"from true {true_fp}"
        )

    def test_returns_none_for_zero_denom(self):
        from src.physics_calibration_direct import _residual_heat_source_weight
        periods = _make_pv_periods(n=20)
        result = _residual_heat_source_weight(periods, "pv", 0.0, 0.0)
        assert result is None

    def test_returns_none_for_too_few_periods(self):
        from src.physics_calibration_direct import _residual_heat_source_weight
        periods = _make_pv_periods(n=3)
        result = _residual_heat_source_weight(
            periods, "pv", 0.12, 0.95, min_periods=5
        )
        assert result is None


# ---------------------------------------------------------------------------
# Tests for _calibrate_solar_lag_xcorr
# ---------------------------------------------------------------------------

class TestSolarLagXcorr:
    """Unit tests for the solar lag cross-correlation estimator."""

    def test_returns_float_for_valid_data(self):
        """Should return a float lag value for data with meaningful PV signal."""
        from src.physics_calibration_direct import _calibrate_solar_lag_xcorr

        rng = np.random.default_rng(1234)
        n = 100
        periods = []
        for i in range(n):
            periods.append({
                "indoor_temp": 20.0 + rng.normal(0, 0.3),
                "outdoor_temp": 5.0,
                "effective_temp": 38.0,
                "outlet_temp": 38.0,
                "pv_power": float(max(0, 500 * np.sin(i / 10) + rng.normal(0, 50))),
                "timestamp": None,
            })
        result = _calibrate_solar_lag_xcorr(periods, hlc=0.12, oe=0.95)
        # May return None or a valid float in [0, 180]
        if result is not None:
            assert 0 <= result <= 180

    def test_returns_none_for_insufficient_periods(self):
        from src.physics_calibration_direct import _calibrate_solar_lag_xcorr
        periods = [{"indoor_temp": 20, "outdoor_temp": 5, "effective_temp": 38,
                    "outlet_temp": 38, "pv_power": 500}]
        result = _calibrate_solar_lag_xcorr(periods, hlc=0.12, oe=0.95)
        assert result is None

    def test_returns_none_for_zero_denom(self):
        from src.physics_calibration_direct import _calibrate_solar_lag_xcorr
        rng = np.random.default_rng(0)
        periods = [{"indoor_temp": float(20+rng.normal()), "outdoor_temp": 5.0,
                    "effective_temp": 38.0, "outlet_temp": 38.0,
                    "pv_power": 100.0} for _ in range(50)]
        result = _calibrate_solar_lag_xcorr(periods, hlc=0.0, oe=0.0)
        assert result is None


# ---------------------------------------------------------------------------
# Tests for _calibrate_slab_tau_grid_search
# ---------------------------------------------------------------------------

class TestSlabTauGridSearch:
    """Unit tests for the 1-D grid search slab time constant estimator."""

    def _make_slab_df(self, tau_true=1.0, n_events=3, steps_per_event=20):
        """Build a DataFrame simulating HP startup inlet approach."""
        rows = []
        base_t = datetime(2024, 1, 1)
        step_h = 5 / 60.0  # 5-min steps

        for ev in range(n_events):
            # 5 off-periods before HP starts
            for k in range(5):
                t = base_t + timedelta(hours=ev * 6 + k * step_h)
                rows.append({
                    "_time": t,
                    "hp_outlet_temp": 25.0,
                    "hp_inlet_temp": 25.0,
                    "hp_current_flow_rate": 0.0,
                })
            # HP-on periods: inlet exponentially approaches outlet - dtf
            outlet = 40.0
            dtf = 2.3
            target = outlet - dtf
            inlet = 25.0
            for k in range(steps_per_event):
                t = base_t + timedelta(hours=ev * 6 + (5 + k) * step_h)
                alpha = min(1.0, step_h / tau_true)
                inlet = inlet + alpha * (target - inlet)
                rows.append({
                    "_time": t,
                    "hp_outlet_temp": outlet,
                    "hp_inlet_temp": inlet,
                    "hp_current_flow_rate": 8.0,
                })

        return pd.DataFrame(rows)

    def test_recovers_known_tau(self):
        """Grid search should recover tau_true = 1.0 h within 0.15 h."""
        from src.physics_calibration_direct import _calibrate_slab_tau_grid_search

        tau_true = 1.0
        df = self._make_slab_df(tau_true=tau_true, n_events=5, steps_per_event=25)

        with patch("src.physics_calibration_direct.config") as mc:
            mc.INLET_TEMP_ENTITY_ID = "sensor.hp_inlet_temp"
            mc.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID = "sensor.hp_outlet_temp"
            mc.FLOW_RATE_ENTITY_ID = "sensor.hp_current_flow_rate"
            mc.SPECIFIC_HEAT_CAPACITY = 4.186
            mc.HEATING_MIN_THERMAL_POWER_KW = 0.5

            result = _calibrate_slab_tau_grid_search(df)

        assert result is not None
        assert abs(result - tau_true) <= 0.15, (
            f"Grid search tau {result:.3f}h deviates more than 0.15h from true {tau_true}h"
        )

    def test_returns_none_for_empty_df(self):
        from src.physics_calibration_direct import _calibrate_slab_tau_grid_search
        result = _calibrate_slab_tau_grid_search(None)
        assert result is None

        result = _calibrate_slab_tau_grid_search(pd.DataFrame())
        assert result is None

    def test_returns_none_missing_columns(self):
        from src.physics_calibration_direct import _calibrate_slab_tau_grid_search
        df = pd.DataFrame({"_time": [datetime(2024, 1, 1)]})
        with patch("src.physics_calibration_direct.config") as mc:
            mc.INLET_TEMP_ENTITY_ID = "sensor.hp_inlet_temp"
            mc.ACTUAL_TARGET_OUTLET_TEMP_ENTITY_ID = "sensor.hp_outlet_temp"
            mc.FLOW_RATE_ENTITY_ID = "sensor.hp_current_flow_rate"
            mc.SPECIFIC_HEAT_CAPACITY = 4.186
            mc.HEATING_MIN_THERMAL_POWER_KW = 0.5
            result = _calibrate_slab_tau_grid_search(df)
        assert result is None


# ---------------------------------------------------------------------------
# Tests for _filter_fp_only_periods / _filter_tv_only_periods
# ---------------------------------------------------------------------------

class TestFilterHelpers:
    """Unit tests for the new FP/TV filter helpers."""

    def test_fp_filter_selects_fp_on(self):
        from src.physics_calibration_direct import _filter_fp_only_periods
        periods = [
            {"fireplace_on": 1, "pv_power": 0, "tv_on": 0,
             "thermal_power_kw": 1.0},
            {"fireplace_on": 0, "pv_power": 0, "tv_on": 0,
             "thermal_power_kw": 1.0},
            {"fireplace_on": 1, "pv_power": 500, "tv_on": 0,
             "thermal_power_kw": 1.0},  # PV too high
        ]
        with patch("src.physics_calibration_direct.config") as mc:
            mc.HEATING_MIN_THERMAL_POWER_KW = 0.5
            result = _filter_fp_only_periods(periods)
        assert len(result) == 1
        assert result[0]["fireplace_on"] == 1

    def test_tv_filter_selects_tv_on(self):
        from src.physics_calibration_direct import _filter_tv_only_periods
        periods = [
            {"tv_on": 1, "pv_power": 0, "fireplace_on": 0,
             "thermal_power_kw": 1.0},
            {"tv_on": 0, "pv_power": 0, "fireplace_on": 0,
             "thermal_power_kw": 1.0},
        ]
        with patch("src.physics_calibration_direct.config") as mc:
            mc.HEATING_MIN_THERMAL_POWER_KW = 0.5
            result = _filter_tv_only_periods(periods)
        assert len(result) == 1
        assert result[0]["tv_on"] == 1


# ---------------------------------------------------------------------------
# Tests for train_thermal_equilibrium_model method dispatch
# ---------------------------------------------------------------------------

class TestTrainDispatch:
    """Verify that train_thermal_equilibrium_model routes to the correct path."""

    def test_scipy_path_called_by_default(self):
        """Default method should invoke the existing scipy optimization."""
        with patch("src.physics_calibration.config") as mc, \
             patch("src.physics_calibration.fetch_historical_data_for_calibration") as mock_fetch, \
             patch("src.physics_calibration.filter_stable_periods") as mock_filter, \
             patch("src.physics_calibration.optimize_thermal_parameters") as mock_opt, \
             patch("src.physics_calibration.backup_existing_calibration"):
            mc.CALIBRATION_METHOD = "scipy"
            mc.TRAINING_LOOKBACK_HOURS = 168
            mock_fetch.return_value = pd.DataFrame({"_time": [1] * 100})
            mock_filter.return_value = [{}] * 60
            mock_opt.return_value = None  # triggers early return from scipy path

            from src.physics_calibration import train_thermal_equilibrium_model
            train_thermal_equilibrium_model(method="scipy")

            mock_opt.assert_called_once()

    def test_physics_path_called_when_requested(self):
        """Method='physics' should delegate to calibrate_thermal_model_physics."""
        mock_model = MagicMock()
        with patch(
            "src.physics_calibration.calibrate_thermal_model_physics",
            return_value=mock_model,
            create=True,
        ) as mock_phys, \
        patch("src.physics_calibration.config") as mc:
            mc.CALIBRATION_METHOD = "scipy"  # config default is scipy

            # Patch the import inside train_thermal_equilibrium_model
            with patch.dict("sys.modules", {
                "src.physics_calibration_direct": MagicMock(
                    calibrate_thermal_model_physics=mock_phys
                ),
            }):
                from src.physics_calibration import train_thermal_equilibrium_model
                result = train_thermal_equilibrium_model(method="physics")

        # The physics function should have been called
        mock_phys.assert_called_once()

    def test_config_method_overrides_default(self):
        """When CALIBRATION_METHOD='physics' in config, calling with
        method='scipy' (default) should redirect to the physics path."""
        from src import physics_calibration as pc_mod

        mock_phys_fn = MagicMock(return_value=MagicMock())

        # Temporarily swap the import inside the function by patching sys.modules
        import sys
        fake_direct = MagicMock()
        fake_direct.calibrate_thermal_model_physics = mock_phys_fn
        original = sys.modules.get("src.physics_calibration_direct")
        sys.modules["src.physics_calibration_direct"] = fake_direct
        try:
            with patch.object(pc_mod, "config") as mc:
                mc.CALIBRATION_METHOD = "physics"
                pc_mod.train_thermal_equilibrium_model(method="scipy")
        finally:
            if original is None:
                sys.modules.pop("src.physics_calibration_direct", None)
            else:
                sys.modules["src.physics_calibration_direct"] = original

        mock_phys_fn.assert_called_once()
