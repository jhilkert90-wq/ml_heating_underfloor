
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src import physics_calibration


@pytest.fixture(autouse=True)
def mock_config():
    with patch('src.physics_calibration.config') as mock_config:
        mock_config.INDOOR_TEMP_ENTITY_ID = "sensor.indoor_temp"
        mock_config.ACTUAL_OUTLET_TEMP_ENTITY_ID = "sensor.outlet_temp"
        mock_config.OUTDOOR_TEMP_ENTITY_ID = "sensor.outdoor_temp"
        mock_config.PV_POWER_ENTITY_ID = "sensor.pv_power"
        mock_config.TV_STATUS_ENTITY_ID = "sensor.tv_on"
        mock_config.FIREPLACE_STATUS_ENTITY_ID = "sensor.fireplace_on"
        mock_config.DHW_STATUS_ENTITY_ID = "sensor.dhw_heating"
        mock_config.DEFROST_STATUS_ENTITY_ID = "sensor.defrosting"
        mock_config.DISINFECTION_STATUS_ENTITY_ID = "sensor.disinfection"
        mock_config.DHW_BOOST_HEATER_STATUS_ENTITY_ID = "sensor.boost_heater"
        mock_config.INLET_TEMP_ENTITY_ID = "sensor.inlet_temp"
        mock_config.FLOW_RATE_ENTITY_ID = "sensor.flow_rate"
        mock_config.POWER_CONSUMPTION_ENTITY_ID = "sensor.power_consumption"
        mock_config.GRACE_PERIOD_MAX_MINUTES = 30
        mock_config.DEFROST_RECOVERY_GRACE_MINUTES = 45
        mock_config.TRAINING_LOOKBACK_HOURS = 24
        mock_config.PV_CALIBRATION_INDOOR_CEILING = 23.0
        mock_config.CLOUD_COVER_CORRECTION_ENABLED = False
        mock_config.LIVING_ROOM_TEMP_ENTITY_ID = "sensor.living_room_temp"
        yield mock_config


def test_check_blocking_states():
    df = pd.DataFrame({
        'dhw_heating': [0, 1, 0],
        'defrosting': [0, 0, 0],
        'disinfection': [0, 0, 1],
        'boost_heater': [0, 0, 0]
    })
    blocking, reasons = physics_calibration.check_blocking_states(
        df, 'dhw_heating', 'defrosting', 'disinfection', 'boost_heater'
    )
    assert blocking
    assert 'dhw_heating' in reasons
    assert 'disinfection' in reasons

    df_no_blocking = pd.DataFrame({
        'dhw_heating': [0, 0, 0],
    })
    blocking, reasons = physics_calibration.check_blocking_states(
        df_no_blocking, 'dhw_heating', 'defrosting', 'disinfection',
        'boost_heater'
    )
    assert not blocking
    assert not reasons


@pytest.fixture
def sample_df():
    base_time = pd.to_datetime('2023-01-01')
    time_series = pd.to_datetime(
        [base_time + pd.Timedelta(minutes=i * 5) for i in range(100)]
    )
    data = {
        '_time': time_series,
        'indoor_temp': 21.0,
        'outlet_temp': 45.0,
        'outdoor_temp': 5.0,
        'pv_power': 0.0,
        'fireplace_on': 0.0,
        'tv_on': 0.0,
        'dhw_heating': 0,
        'defrosting': 0,
        'disinfection': 0,
        'boost_heater': 0,
        'inlet_temp': 40.0,
        'flow_rate': 1000.0,
        'power_consumption': 500.0,
        'thermal_power_kw': 2.0,
    }
    df = pd.DataFrame(data)
    return df


def test_filter_stable_periods_stable_data(sample_df):
    with patch('src.physics_calibration.log_filtering_stats'), \
         patch('src.physics_calibration.json'), \
         patch('builtins.open'):
        stable_periods = physics_calibration.filter_stable_periods(sample_df)
        assert len(stable_periods) > 0
        for period in stable_periods:
            assert 'stability_score' in period


def test_filter_stable_periods_unstable_temp(sample_df):
    sample_df['indoor_temp'] = np.sin(np.arange(100)) * 5 + 20
    with patch('src.physics_calibration.log_filtering_stats'), \
         patch('src.physics_calibration.json'), \
         patch('builtins.open'):
        stable_periods = physics_calibration.filter_stable_periods(sample_df)
        assert len(stable_periods) == 0


def test_filter_stable_periods_with_blocking(sample_df):
    sample_df.loc[40:60, 'dhw_heating'] = 1
    with patch('src.physics_calibration.log_filtering_stats'), \
         patch('src.physics_calibration.json'), \
         patch('builtins.open'):
        stable_periods = physics_calibration.filter_stable_periods(sample_df)
        assert len(stable_periods) > 0
        assert len(stable_periods) < 83


def test_filter_stable_periods_low_flow(sample_df):
    sample_df['flow_rate'] = 5.0  # Below 10.0 threshold
    with patch('src.physics_calibration.log_filtering_stats'), \
         patch('src.physics_calibration.json'), \
         patch('builtins.open'):
        stable_periods = physics_calibration.filter_stable_periods(sample_df)
        # Should be filtered out because flow rate is too low
        assert len(stable_periods) == 0


@pytest.fixture
def stable_periods_fixture():
    return [
        {
            'indoor_temp': 21.0, 'outlet_temp': 40.0, 'outdoor_temp': 5.0,
            'pv_power': 0.0, 'fireplace_on': 0.0, 'tv_on': 0.0
        },
        {
            'indoor_temp': 22.0, 'outlet_temp': 45.0, 'outdoor_temp': 10.0,
            'pv_power': 500.0, 'fireplace_on': 0.0, 'tv_on': 1.0
        }
    ]


def test_calculate_mae_for_params(stable_periods_fixture):
    param_names = [
        'thermal_time_constant',
        'heat_loss_coefficient',
        'outlet_effectiveness'
    ]
    params = [20.0, 0.5, 0.8]
    current_params = {
        'thermal_time_constant': 20.0,
        'heat_loss_coefficient': 0.5,
        'outlet_effectiveness': 0.8,
        'pv_heat_weight': 0.001,
        'fireplace_heat_weight': 2.0,
        'tv_heat_weight': 0.2,
        'solar_lag_minutes': 30.0
    }

    with patch('src.physics_calibration.ThermalEquilibriumModel') as mock_cls:
        mock_model_instance = mock_cls.return_value
        mock_model_instance.predict_equilibrium_temperature.return_value = 21.5

        mae = physics_calibration.calculate_mae_for_params(
            params, param_names, stable_periods_fixture, current_params
        )
        assert mae == pytest.approx(0.5)
        mock_model_instance.sync_heat_source_channels_from_model_state.assert_called_once_with()
        assert (
            mock_model_instance.predict_equilibrium_temperature.call_count == 1
        )


def test_calculate_direct_heat_loss():
    """Test direct heat loss calculation from thermal power."""
    stable_periods = [
        {
            'indoor_temp': 21.0,
            'outdoor_temp': 1.0,
            'thermal_power_kw': 2.0,  # 2kW for 20K delta -> U=0.1
            'pv_power': 0,
            'fireplace_on': 0,
            'tv_on': 0
        },
        {
            'indoor_temp': 21.0,
            'outdoor_temp': 1.0,
            'thermal_power_kw': 4.0,  # 4kW for 20K delta -> U=0.2
            'pv_power': 0,
            'fireplace_on': 0,
            'tv_on': 0
        },
        {
            'indoor_temp': 21.0,
            'outdoor_temp': 1.0,
            'thermal_power_kw': None,  # Should be ignored
            'pv_power': 0,
            'fireplace_on': 0,
            'tv_on': 0
        },
        {
            'indoor_temp': 21.0,
            'outdoor_temp': 1.0,
            'thermal_power_kw': 2.0,
            'pv_power': 500,  # Should be ignored (high PV)
            'fireplace_on': 0,
            'tv_on': 0
        }
    ]

    u_val = physics_calibration.calculate_direct_heat_loss(stable_periods)
    
    # Expected: Average of 0.1 and 0.2 = 0.15
    assert u_val == pytest.approx(0.15)


def test_build_optimization_params():
    current_params = {
        'thermal_time_constant': 10,
        'heat_loss_coefficient': 0.1,
        'outlet_effectiveness': 0.5,
        'pv_heat_weight': 0.001,
        'fireplace_heat_weight': 2.0,
        'tv_heat_weight': 0.5,
        'solar_lag_minutes': 30.0
    }
    excluded_params = []

    with patch(
        'src.physics_calibration.ThermalParameterConfig'
    ) as mock_config:
        mock_config.get_bounds.return_value = (0, 1)

        # Default: returns HLC + OE
        names, values, bounds = physics_calibration.build_optimization_params(
            current_params, excluded_params
        )

        assert 'pv_heat_weight' not in names
        assert 'fireplace_heat_weight' not in names
        assert 'tv_heat_weight' not in names
        assert 'solar_lag_minutes' not in names
        assert len(names) == 2
        assert names[0] == 'heat_loss_coefficient'
        assert names[1] == 'outlet_effectiveness'

        # Test with heat loss center (constrained bounds)
        names, values, bounds = physics_calibration.build_optimization_params(
            current_params, excluded_params, heat_loss_center=0.2
        )
        
        hl_idx = names.index('heat_loss_coefficient')
        hl_bounds = bounds[hl_idx]
        assert hl_bounds[0] == pytest.approx(0.18)
        assert hl_bounds[1] == pytest.approx(0.22)

        # Test fix_hlc=True: only OE returned, HLC excluded
        names, values, bounds = physics_calibration.build_optimization_params(
            current_params, excluded_params,
            heat_loss_center=0.15, fix_hlc=True,
        )
        assert len(names) == 1
        assert names[0] == 'outlet_effectiveness'
        assert 'heat_loss_coefficient' not in names

        # fix_hlc without heat_loss_center falls back to including HLC
        names, values, bounds = physics_calibration.build_optimization_params(
            current_params, excluded_params, fix_hlc=True,
        )
        assert len(names) == 2
        assert 'heat_loss_coefficient' in names


@pytest.fixture
def mock_scipy_minimize():
    with patch('src.physics_calibration.minimize') as mock_minimize:
        yield mock_minimize


@patch('src.physics_calibration.ThermalParameterConfig')
def test_optimize_thermal_parameters(
    mock_thermal_config, stable_periods_fixture, mock_scipy_minimize
):
    # _run_optimization_pass calls minimize.  With only 2 fixture periods
    # and 1 PV-only period (< MIN_PV_PERIODS=5), only Pass 1 (HLC+OE)
    # executes.  minimize is called once.
    mock_scipy_minimize.return_value = MagicMock(
        success=True, x=np.array([0.18, 0.65]), fun=0.123
    )
    mock_thermal_config.get_default.return_value = 0.5
    mock_thermal_config.get_bounds.return_value = (0, 1)

    with patch('src.physics_calibration.debug_thermal_predictions'):
        result = physics_calibration.optimize_thermal_parameters(
            stable_periods_fixture
        )

        mock_scipy_minimize.assert_called_once()
        assert result is not None
        assert result['optimization_success'] is True
        assert result['mae'] == 0.123
        # HLC and OE come from Pass 1 result
        assert result['heat_loss_coefficient'] == pytest.approx(0.18)
        assert result['outlet_effectiveness'] == pytest.approx(0.65)


def test_optimize_thermal_parameters_scipy_disabled(stable_periods_fixture):
    with patch('src.physics_calibration.minimize', None):
        result = physics_calibration.optimize_thermal_parameters(
            stable_periods_fixture
        )
        assert result is None


# ===================================================================
# Tests for isolated-pass optimization split
# ===================================================================


def test_filter_hp_only_periods():
    """HP-only: PV<100, FP=0, TV=0, thermal_power_kw>=0.5, defrost grace, outlet>inlet."""
    periods = [
        # 0: valid HP-only period
        {'pv_power': 0, 'fireplace_on': 0, 'tv_on': 0, 'indoor_temp': 21,
         'thermal_power_kw': 1.5, 'effective_temp': 28.0, 'inlet_temp': 25.0,
         'minutes_since_defrost': 120},
        # 1: valid HP-only period (PV just below threshold)
        {'pv_power': 50, 'fireplace_on': 0, 'tv_on': 0, 'indoor_temp': 21,
         'thermal_power_kw': 0.8, 'effective_temp': 27.0, 'inlet_temp': 24.0,
         'minutes_since_defrost': 60},
        # 2: PV too high (==100 is excluded)
        {'pv_power': 100, 'fireplace_on': 0, 'tv_on': 0, 'indoor_temp': 21,
         'thermal_power_kw': 1.0, 'effective_temp': 28.0, 'inlet_temp': 25.0,
         'minutes_since_defrost': 120},
        # 3: PV too high
        {'pv_power': 500, 'fireplace_on': 0, 'tv_on': 0, 'indoor_temp': 22,
         'thermal_power_kw': 1.2, 'effective_temp': 28.0, 'inlet_temp': 25.0,
         'minutes_since_defrost': 120},
        # 4: fireplace on
        {'pv_power': 0, 'fireplace_on': 1, 'tv_on': 0, 'indoor_temp': 22,
         'thermal_power_kw': 1.0, 'effective_temp': 28.0, 'inlet_temp': 25.0,
         'minutes_since_defrost': 120},
        # 5: TV on
        {'pv_power': 0, 'fireplace_on': 0, 'tv_on': 1, 'indoor_temp': 22,
         'thermal_power_kw': 1.0, 'effective_temp': 28.0, 'inlet_temp': 25.0,
         'minutes_since_defrost': 120},
        # 6: HP off (standby, low thermal power)
        {'pv_power': 0, 'fireplace_on': 0, 'tv_on': 0, 'indoor_temp': 21,
         'thermal_power_kw': 0.1, 'effective_temp': 23.0, 'inlet_temp': 22.5,
         'minutes_since_defrost': 120},
        # 7: post-defrost slab recovery (within 45-min grace)
        {'pv_power': 0, 'fireplace_on': 0, 'tv_on': 0, 'indoor_temp': 21,
         'thermal_power_kw': 1.5, 'effective_temp': 28.0, 'inlet_temp': 25.0,
         'minutes_since_defrost': 20},
        # 8: outlet <= inlet (active defrost / cooling)
        {'pv_power': 0, 'fireplace_on': 0, 'tv_on': 0, 'indoor_temp': 21,
         'thermal_power_kw': 0.8, 'effective_temp': 23.0, 'inlet_temp': 24.0,
         'minutes_since_defrost': 120},
    ]
    result = physics_calibration._filter_hp_only_periods(periods)
    # Only first two qualify: all criteria met, defrost grace elapsed, outlet > inlet
    assert len(result) == 2
    assert all(p['pv_power'] < 100 for p in result)
    assert all(p['fireplace_on'] == 0 for p in result)
    assert all(p['tv_on'] == 0 for p in result)
    assert all(p['thermal_power_kw'] >= 0.5 for p in result)
    assert all(p['minutes_since_defrost'] >= 45 for p in result)
    assert all(p['effective_temp'] > p['inlet_temp'] for p in result)


def test_filter_pv_only_periods():
    """PV-only: PV>100, FP=0, TV=0 (no ceiling without hlc/oe)."""
    periods = [
        {'pv_power': 0, 'fireplace_on': 0, 'tv_on': 0, 'indoor_temp': 21},
        {'pv_power': 500, 'fireplace_on': 0, 'tv_on': 0, 'indoor_temp': 22},
        {'pv_power': 200, 'fireplace_on': 1, 'tv_on': 0, 'indoor_temp': 22},
        {'pv_power': 300, 'fireplace_on': 0, 'tv_on': 1, 'indoor_temp': 22},
        {'pv_power': 1000, 'fireplace_on': 0, 'tv_on': 0, 'indoor_temp': 22.5},
    ]
    result = physics_calibration._filter_pv_only_periods(periods)
    assert len(result) == 2
    assert result[0]['pv_power'] == 500
    assert result[1]['pv_power'] == 1000


def test_build_pv_params():
    """_build_pv_params returns only pv_heat_weight (solar_lag frozen)."""
    current_params = {
        'pv_heat_weight': 0.002,
        'solar_lag_minutes': 45.0,
    }
    names, values, bounds = physics_calibration._build_pv_params(current_params)
    assert names == ['pv_heat_weight']
    assert values == [0.002]
    assert bounds[0] == (0.0001, 0.005)


def test_calculate_mae_for_params_with_frozen_params(stable_periods_fixture):
    """frozen_params override both param_dict and current_params."""
    param_names = ['heat_loss_coefficient', 'outlet_effectiveness']
    params = [0.15, 0.7]
    current_params = {
        'thermal_time_constant': 20.0,
        'heat_loss_coefficient': 0.5,
        'outlet_effectiveness': 0.8,
        'pv_heat_weight': 0.001,
        'fireplace_heat_weight': 2.0,
        'tv_heat_weight': 0.2,
        'solar_lag_minutes': 30.0,
    }
    # Freeze PV/FP/TV to 0 — simulating Pass 1
    frozen = {
        'pv_heat_weight': 0.0,
        'fireplace_heat_weight': 0.0,
        'tv_heat_weight': 0.0,
        'solar_lag_minutes': 0.0,
    }

    with patch('src.physics_calibration.ThermalEquilibriumModel') as mock_cls:
        mock_inst = mock_cls.return_value
        mock_inst.predict_equilibrium_temperature.return_value = 21.5

        physics_calibration.calculate_mae_for_params(
            params, param_names, stable_periods_fixture, current_params,
            frozen_params=frozen,
        )

        # Verify weights were set to frozen values (0), not current_params
        assert mock_inst.external_source_weights.__setitem__.call_args_list[0] == \
            (('pv', 0.0),)
        assert mock_inst.external_source_weights.__setitem__.call_args_list[1] == \
            (('fireplace', 0.0),)
        assert mock_inst.external_source_weights.__setitem__.call_args_list[2] == \
            (('tv', 0.0),)


@patch('src.physics_calibration.ThermalParameterConfig')
def test_pass1_hlc_oe_isolation(mock_thermal_config, mock_scipy_minimize):
    """Pass 1 fixes HLC from energy balance and optimises OE only."""
    mock_thermal_config.get_default.return_value = 0.5
    mock_thermal_config.get_bounds.return_value = (0, 1)

    # Pass 1 is OE-only (1 param) when direct HLC is available
    mock_scipy_minimize.return_value = MagicMock(
        success=True, x=np.array([0.7]), fun=0.1
    )

    # 15 HP-only periods with thermal_power_kw so direct HLC can be computed.
    # U = thermal_power / (indoor - outdoor) = 2.0 / (21 - 5) = 0.125
    hp_periods = [
        {'indoor_temp': 21, 'outlet_temp': 40, 'outdoor_temp': 5,
         'pv_power': 0, 'fireplace_on': 0, 'tv_on': 0,
         'thermal_power_kw': 2.0}
    ] * 15

    with patch('src.physics_calibration.debug_thermal_predictions'):
        result = physics_calibration.optimize_thermal_parameters(hp_periods)

    assert result is not None
    # HLC is fixed from energy balance (direct_u), not from optimizer
    assert result['heat_loss_coefficient'] == pytest.approx(0.125, abs=0.01)
    assert result['outlet_effectiveness'] == pytest.approx(0.7)
    # minimize called once (Pass 1 OE-only, no Pass 2 — no PV data)
    assert mock_scipy_minimize.call_count == 1


@patch('src.physics_calibration.ThermalParameterConfig')
def test_pass2_pv_isolation(mock_thermal_config, mock_scipy_minimize):
    """Pass 2 runs when enough PV-only periods exist; only optimizes pv_weight (solar_lag frozen)."""
    mock_thermal_config.get_default.return_value = 0.5
    mock_thermal_config.get_bounds.return_value = (0, 1)

    # Pass 1 is OE-only (1 param) when direct HLC is available
    pass1_result = MagicMock(
        success=True, x=np.array([0.6]), fun=0.12
    )
    pass2_result = MagicMock(
        success=True, x=np.array([3.0]), fun=0.09  # scaled: 3.0 × 0.001 = 0.003
    )
    mock_scipy_minimize.side_effect = [pass1_result, pass2_result]

    # HP periods with thermal_power_kw for direct HLC.
    # U = 2.0 / (21 - 5) = 0.125
    # PV-only periods need indoor > HP eq + 0.1 to pass residual filter.
    # With HLC=0.125, OE=0.6: hp_eq = (0.6*30 + 0.125*8) / (0.6+0.125) = 26.21
    # indoor=28 > 26.31 → passes.
    periods = (
        # 10 HP-only
        [{'indoor_temp': 21, 'outlet_temp': 40, 'outdoor_temp': 5,
          'pv_power': 0, 'fireplace_on': 0, 'tv_on': 0,
          'thermal_power_kw': 2.0}] * 10 +
        # 6 PV-only with indoor well above HP eq (blinds open)
        [{'indoor_temp': 28, 'outlet_temp': 30, 'outdoor_temp': 8,
          'pv_power': 800, 'fireplace_on': 0, 'tv_on': 0}] * 6
    )

    with patch('src.physics_calibration.debug_thermal_predictions'):
        result = physics_calibration.optimize_thermal_parameters(periods)

    assert result is not None
    # HLC fixed from energy balance
    assert result['heat_loss_coefficient'] == pytest.approx(0.125, abs=0.01)
    assert result['outlet_effectiveness'] == pytest.approx(0.6)
    # PV weight from Pass 2
    assert result['pv_heat_weight'] == pytest.approx(0.003)
    # solar_lag frozen at default (not optimized)
    assert result['solar_lag_minutes'] == 0.5  # default from mock
    # minimize called twice (Pass 1 + Pass 2)
    assert mock_scipy_minimize.call_count == 2


@patch('src.physics_calibration.ThermalParameterConfig')
def test_pass2_skipped_when_insufficient_pv_data(
    mock_thermal_config, mock_scipy_minimize
):
    """Pass 2 is skipped when < 5 PV-only periods — defaults used."""
    mock_thermal_config.get_default.return_value = 0.5
    mock_thermal_config.get_bounds.return_value = (0, 1)

    # Pass 1 is OE-only (1 param) when direct HLC is available
    mock_scipy_minimize.return_value = MagicMock(
        success=True, x=np.array([0.65]), fun=0.11
    )

    periods = (
        [{'indoor_temp': 21, 'outlet_temp': 40, 'outdoor_temp': 5,
          'pv_power': 0, 'fireplace_on': 0, 'tv_on': 0,
          'thermal_power_kw': 2.0}] * 12 +
        # Only 3 PV-only (< 5)
        [{'indoor_temp': 22, 'outlet_temp': 42, 'outdoor_temp': 8,
          'pv_power': 500, 'fireplace_on': 0, 'tv_on': 0}] * 3
    )

    with patch('src.physics_calibration.debug_thermal_predictions'):
        result = physics_calibration.optimize_thermal_parameters(periods)

    assert result is not None
    # PV weight should be default since Pass 2 was skipped
    assert result['pv_heat_weight'] == 0.5  # default from mock
    assert result['solar_lag_minutes'] == 0.5  # default from mock
    # Only 1 minimize call (Pass 1 only)
    assert mock_scipy_minimize.call_count == 1


def test_tv_weight_not_optimised():
    """tv_heat_weight is never included in any optimization pass."""
    current_params = {
        'pv_heat_weight': 0.001,
    }
    # Check _build_pv_params doesn't include TV
    names, _, _ = physics_calibration._build_pv_params(current_params)
    assert 'tv_heat_weight' not in names
    # solar_lag should not be optimized either
    assert 'solar_lag_minutes' not in names

    # Check build_optimization_params doesn't include TV
    full_params = {
        'thermal_time_constant': 10, 'heat_loss_coefficient': 0.1,
        'outlet_effectiveness': 0.5, 'pv_heat_weight': 0.001,
        'fireplace_heat_weight': 2.0, 'tv_heat_weight': 0.5,
        'solar_lag_minutes': 30.0,
    }
    with patch('src.physics_calibration.ThermalParameterConfig') as mc:
        mc.get_bounds.return_value = (0, 1)
        names, _, _ = physics_calibration.build_optimization_params(
            full_params, []
        )
    assert 'tv_heat_weight' not in names


def test_fp_weight_not_in_scipy():
    """fireplace_heat_weight is never included in any optimization pass."""
    current_params = {
        'pv_heat_weight': 0.001,
    }
    names, _, _ = physics_calibration._build_pv_params(current_params)
    assert 'fireplace_heat_weight' not in names

    full_params = {
        'thermal_time_constant': 10, 'heat_loss_coefficient': 0.1,
        'outlet_effectiveness': 0.5, 'pv_heat_weight': 0.001,
        'fireplace_heat_weight': 2.0, 'tv_heat_weight': 0.5,
        'solar_lag_minutes': 30.0,
    }
    with patch('src.physics_calibration.ThermalParameterConfig') as mc:
        mc.get_bounds.return_value = (0, 1)
        names, _, _ = physics_calibration.build_optimization_params(
            full_params, []
        )
    assert 'fireplace_heat_weight' not in names


@patch('src.physics_calibration.calculate_cooling_time_constant', return_value=(None, 0.0))
@patch('src.physics_calibration.filter_transient_periods', return_value=[])
@patch('src.physics_calibration.optimize_thermal_parameters')
@patch('src.physics_calibration.filter_stable_periods')
@patch('src.physics_calibration.fetch_historical_data_for_calibration')
@patch('src.physics_calibration.backup_existing_calibration')
@patch('src.physics_calibration.get_thermal_state_manager')
@patch('src.physics_calibration.ThermalEquilibriumModel')
def test_train_thermal_equilibrium_model_syncs_channels_when_enabled(
    mock_model_cls,
    mock_get_state_manager,
    mock_backup,
    mock_fetch,
    mock_filter_stable,
    mock_optimize,
    mock_filter_transient,
    mock_cooling_tau,
):
    mock_fetch.return_value = pd.DataFrame({'_time': [pd.Timestamp('2023-01-01')]})
    mock_filter_stable.return_value = [
        {'indoor_temp': 21.0, 'outlet_temp': 40.0, 'outdoor_temp': 5.0}
        for _ in range(60)
    ]
    mock_optimize.return_value = {
        'optimization_success': True,
        'mae': 0.12,
        'thermal_time_constant': 5.5,
        'heat_loss_coefficient': 0.18,
        'outlet_effectiveness': 0.64,
        'pv_heat_weight': 0.0012,
        'fireplace_heat_weight': 4.6,
        'tv_heat_weight': 0.28,
        'solar_lag_minutes': 55.0,
    }

    temp_model = MagicMock()
    temp_model.external_source_weights = {'pv': 0.0, 'fireplace': 0.0, 'tv': 0.0}
    temp_model.slab_time_constant_hours = 1.4

    final_model = MagicMock()
    final_model.external_source_weights = {'pv': 0.0, 'fireplace': 0.0, 'tv': 0.0}
    final_model.thermal_time_constant = 6.0
    final_model.heat_loss_coefficient = 0.15
    final_model.outlet_effectiveness = 0.55
    final_model.learning_confidence = 3.0
    final_model.slab_time_constant_hours = 1.4

    mock_model_cls.side_effect = [temp_model, final_model]
    mock_state_manager = MagicMock()
    mock_get_state_manager.return_value = mock_state_manager

    result = physics_calibration.train_thermal_equilibrium_model()

    assert result is final_model
    temp_model.sync_heat_source_channels_from_model_state.assert_called_once_with()
    assert final_model.sync_heat_source_channels_from_model_state.call_count >= 1
    assert final_model.sync_heat_source_channels_from_model_state.call_args_list[-1].kwargs == {'persist': True}
    mock_state_manager.set_calibrated_baseline.assert_called_once()
    mock_state_manager.update_learning_state.assert_called_once_with(
        learning_confidence=3.0
    )


# ===================================================================
# Phase: PV calibration fixes — blind filter, bounds, cloud guard
# ===================================================================


def test_filter_pv_only_periods_residual_excludes_blind_closed():
    """Periods where indoor ≤ HP equilibrium are excluded (blinds closed)."""
    # HLC=0.15, OE=0.7: hp_eq = (0.7*30 + 0.15*5)/(0.7+0.15) = 25.59
    periods = [
        {'pv_power': 2000, 'fireplace_on': 0, 'tv_on': 0,
         'indoor_temp': 25.0, 'effective_temp': 30.0, 'outdoor_temp': 5.0},  # below eq
        {'pv_power': 2000, 'fireplace_on': 0, 'tv_on': 0,
         'indoor_temp': 25.5, 'effective_temp': 30.0, 'outdoor_temp': 5.0},  # ~eq
        {'pv_power': 2000, 'fireplace_on': 0, 'tv_on': 0,
         'indoor_temp': 26.5, 'effective_temp': 30.0, 'outdoor_temp': 5.0},  # above eq
    ]
    result = physics_calibration._filter_pv_only_periods(periods, hlc=0.15, oe=0.7)
    # Only the last period (26.5 > 25.59 + 0.1) should pass
    assert len(result) == 1
    assert result[0]['indoor_temp'] == 26.5


def test_filter_pv_only_periods_no_hlc_keeps_all():
    """Without hlc/oe, all PV-only periods are kept (no residual filter)."""
    periods = [
        {'pv_power': 500, 'fireplace_on': 0, 'tv_on': 0, 'indoor_temp': 20.0},
        {'pv_power': 800, 'fireplace_on': 0, 'tv_on': 0, 'indoor_temp': 21.5},
        {'pv_power': 1200, 'fireplace_on': 0, 'tv_on': 0, 'indoor_temp': 24.0},
    ]
    result = physics_calibration._filter_pv_only_periods(periods)
    assert len(result) == 3


def test_pv_heat_weight_new_bounds():
    """ThermalParameterConfig bounds for pv_heat_weight updated to (0.0001, 0.005)."""
    from src.thermal_config import ThermalParameterConfig
    lo, hi = ThermalParameterConfig.get_bounds('pv_heat_weight')
    assert lo == 0.0001
    assert hi == 0.005
    # Default is the calibrated baseline value
    assert ThermalParameterConfig.get_default('pv_heat_weight') == 0.0020704649305198215


def test_cloud_exponent_not_learned_when_disabled():
    """cloud_factor_exponent should not change when CLOUD_COVER_CORRECTION_ENABLED=false."""
    from src.heat_source_channels import SolarChannel
    from src import config

    original = getattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False)
    config.CLOUD_COVER_CORRECTION_ENABLED = False
    try:
        solar = SolarChannel()
        initial = solar.cloud_factor_exponent
        # Simulate learning with cloud context — should NOT update cloud_factor_exponent
        for _ in range(15):
            solar.record_learning(1.5, {
                "pv_power": 2000,
                "avg_cloud_cover": 80.0,
                "pv_power_history": [500.0, 1000.0],
            })
        assert solar.cloud_factor_exponent == pytest.approx(initial)
    finally:
        config.CLOUD_COVER_CORRECTION_ENABLED = original


def test_cloud_exponent_learned_when_enabled():
    """cloud_factor_exponent should change when CLOUD_COVER_CORRECTION_ENABLED=true."""
    from src.heat_source_channels import SolarChannel
    from src import config

    original = getattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False)
    config.CLOUD_COVER_CORRECTION_ENABLED = True
    try:
        solar = SolarChannel()
        initial = solar.cloud_factor_exponent
        for _ in range(15):
            solar.record_learning(1.5, {
                "pv_power": 2000,
                "avg_cloud_cover": 80.0,
                "pv_power_history": [500.0, 1000.0],
            })
        assert solar.cloud_factor_exponent != pytest.approx(initial)
    finally:
        config.CLOUD_COVER_CORRECTION_ENABLED = original


def test_filter_pv_decay_sliding_window():
    """Gradual PV decline across multiple steps should be detected."""
    import math
    # Use mock entity IDs (from mock_config fixture):
    # PV_POWER_ENTITY_ID = "sensor.pv_power" → pv_col = "pv_power"
    # INDOOR_TEMP_ENTITY_ID = "sensor.indoor_temp" → indoor_col = "indoor_temp"
    # OUTDOOR_TEMP_ENTITY_ID = "sensor.outdoor_temp" → outdoor_col = "outdoor_temp"
    pv_col = "pv_power"
    indoor_col = "indoor_temp"
    outdoor_col = "outdoor_temp"

    rows = []
    # 10 steps of high PV
    for _ in range(10):
        rows.append({pv_col: 2000, indoor_col: 22.0, outdoor_col: 10.0})
    # Gradual decline over 6 steps (30 min)
    for pv in [1500, 1000, 600, 300, 80, 20]:
        rows.append({pv_col: pv, indoor_col: 22.0, outdoor_col: 10.0})
    # Post-drop: indoor excess decaying
    for k in range(30):
        excess = 2.0 * math.exp(-k * 5 / 30.0)
        rows.append({pv_col: 10, indoor_col: 10.0 + excess, outdoor_col: 10.0})

    times = pd.date_range("2024-01-01", periods=len(rows), freq="5min")
    df = pd.DataFrame(rows)
    df["_time"] = times

    periods = physics_calibration.filter_pv_decay_periods(df)
    assert len(periods) >= 1, "Gradual sunset should be detected by sliding window"


def test_delta_t_floor_rejects_low_dt():
    """delta_t <= 1.0 periods are excluded from calibration."""
    periods = []
    # All with dt=0.8 (below new threshold of 1.0)
    for _ in range(20):
        periods.append({"outlet_temp": 30.0, "inlet_temp": 29.2, "thermal_power_kw": 2.0})
    result = physics_calibration.calibrate_delta_t_floor(periods)
    assert result is None  # All excluded by dt > 1.0 filter

