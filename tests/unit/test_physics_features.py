
import pytest
import logging
import pandas as pd
from unittest.mock import MagicMock, patch
from src.physics_features import build_physics_features
from src import config


@pytest.fixture
def mock_ha_client():
    client = MagicMock()
    client.get_all_states.return_value = {}
    # Updated side_effect to include new sensors
    # Order: Indoor, LivingRoom, Outdoor, Outlet, Target, Inlet, Flow, Power,
    # DHW, Disinfection, Boost, Defrost, PV, SolarCorrection, WindSpeed, Fireplace, TV
    client.get_state.side_effect = [
        20.0,  # Indoor
        20.0,  # Living Room
        5.0,   # Outdoor
        40.0,  # Outlet
        21.0,  # Target
        35.0,  # Inlet (New)
        1000.0, # Flow (New)
        1500.0, # Power (New)
        True,  # DHW
        False, # Disinfection
        False, # Boost
        True,  # Defrost
        500.0, # PV
        0.0,   # Solar Correction
        3.5,   # Wind Speed
        True,  # Fireplace
        False  # TV
    ]
    client.get_calibrated_hourly_forecast.return_value = [6.0, 7.0, 8.0, 9.0]
    return client


@pytest.fixture
def mock_influx_service():
    service = MagicMock()
    service.fetch_outlet_history.return_value = [35.0, 36.0, 37.0, 38.0, 39.0, 40.0]
    service.fetch_indoor_history.return_value = [19.0, 19.2, 19.4, 19.6, 19.8, 20.0]
    service.fetch_pv_history.return_value = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    service.fetch_inlet_history.return_value = [34.0, 34.2, 34.4, 34.6, 34.8, 35.0, 35.0]
    return service


def test_build_physics_features_success(mock_ha_client, mock_influx_service):
    """Test successful build of physics features."""
    features_df, _ = build_physics_features(mock_ha_client, mock_influx_service)
    assert isinstance(features_df, pd.DataFrame)
    
    # Verify column count: dynamic forecast keys scale with TRAJECTORY_STEPS (default 4).
    # Previously 58 columns assumed 6 forecast slots; with TRAJECTORY_STEPS=4 → 52 columns.
    # Base includes prior +1 for pv_now_electrical.
    expected_cols_without_raw_history = 58 - 3 * (6 - config.TRAJECTORY_STEPS) + 1
    # Add +1 for the newly added pv_power_history_electrical column.
    # Add +3 for ML correction features: wind_speed, is_weekend, indoor_margin_rate.
    # Add +2 for slab thermal state features: d_inlet_temp_60min, is_equilibrium.
    expected_cols = expected_cols_without_raw_history + 1 + 3 + 2  # 3 groups: temp, pv, cloud_cover
    assert len(features_df.columns) == expected_cols
    
    # Verify original features
    assert features_df['indoor_temp_lag_30m'][0] == 19.6
    assert features_df['pv_power_history_electrical'][0] == [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    assert features_df['pv_power_history_electrical'][0] != features_df['pv_power_history'][0]
    
    # Verify new thermodynamic features
    assert features_df['inlet_temp'][0] == 35.0
    assert features_df['flow_rate'][0] == 1000.0
    assert features_df['power_consumption'][0] == 1500.0
    
    # Verify derived calculations
    # Delta T = Outlet (40) - Inlet (35) = 5
    assert features_df['delta_t'][0] == 5.0
    
    # Thermal Power = (Flow/60) * SpecificHeat * DeltaT
    # (1000/60) * 4.186 * 5 = 16.66 * 4.186 * 5 = 348.83 kW approx
    expected_power = (1000.0 / 60.0) * config.SPECIFIC_HEAT_CAPACITY * 5.0
    assert abs(features_df['thermal_power_kw'][0] - expected_power) < 0.01
    
    # COP = Thermal Power / Electrical Power (kW)
    # Electrical = 1500W = 1.5kW
    # COP = 5.81 / 1.5 = 3.87 approx
    expected_cop = expected_power / 1.5
    assert abs(features_df['cop_realtime'][0] - expected_cop) < 0.01

    # Verify slab thermal state features
    # Default HISTORY_STEP_MINUTES=10 → steps_per_hour=6.
    # inlet_lag_history[-6] = 34.2, inlet_temp_f = 35.0 → d_inlet_temp_60min = 0.8
    assert abs(features_df['d_inlet_temp_60min'][0] - 0.8) < 0.01
    # |0.8| >= 0.3 → is_equilibrium = 0.0
    assert features_df['is_equilibrium'][0] == 0.0


def test_build_physics_features_uses_history_step_minutes_for_60min_delta(
    mock_ha_client, mock_influx_service, monkeypatch
):
    """d_inlet_temp_60min should track 60min based on HISTORY_STEP_MINUTES."""
    monkeypatch.setattr(config, "HISTORY_STEP_MINUTES", 15)
    mock_influx_service.fetch_inlet_history.return_value = [34.0, 34.5, 35.0, 35.5, 36.0]

    features_df, _ = build_physics_features(mock_ha_client, mock_influx_service)

    assert features_df is not None
    mock_influx_service.fetch_inlet_history.assert_called_once_with(5)
    # steps_per_hour=4 so value at [-4] is 34.5, inlet is 35.0 => 0.5 K
    assert features_df["d_inlet_temp_60min"][0] == pytest.approx(0.5)
    assert features_df["is_equilibrium"][0] == 0.0


def test_build_physics_features_inlet_default_history_forces_neutral_trend(
    mock_ha_client, mock_influx_service
):
    """All-default fallback inlet history should not create a fake trend."""
    mock_influx_service.fetch_inlet_history.return_value = [30.0] * 7

    features_df, _ = build_physics_features(mock_ha_client, mock_influx_service)

    assert features_df is not None
    assert features_df["d_inlet_temp_60min"][0] == 0.0
    assert features_df["is_equilibrium"][0] == 1.0


def test_build_physics_features_cooling_demand_uses_forecast_above_target(
    mock_ha_client, mock_influx_service
):
    """Cooling demand should increase when forecast exceeds target."""
    mock_ha_client.get_calibrated_hourly_forecast.return_value = [24.0] * 4

    features_df, _ = build_physics_features(
        mock_ha_client,
        mock_influx_service,
        climate_mode="cooling",
    )

    assert features_df is not None
    assert features_df["target_temp"][0] == 21.0
    assert features_df["heating_demand_forecast"][0] == pytest.approx(0.3)
    assert features_df["combined_forecast_thermal_load"][0] == pytest.approx(0.3)


def test_build_physics_features_heating_demand_keeps_heating_sign(
    mock_ha_client, mock_influx_service
):
    """Heating mode should retain target-minus-forecast demand semantics."""
    mock_ha_client.get_calibrated_hourly_forecast.return_value = [18.0] * 4

    features_df, _ = build_physics_features(
        mock_ha_client,
        mock_influx_service,
        climate_mode="heating",
    )

    assert features_df is not None
    assert features_df["heating_demand_forecast"][0] == pytest.approx(0.3)
    assert features_df["combined_forecast_thermal_load"][0] == pytest.approx(0.3)


def test_build_physics_features_missing_data(mock_ha_client, mock_influx_service):
    """Test feature building with missing critical data."""
    # Fail on first critical sensor (Indoor Temp)
    mock_ha_client.get_state.side_effect = [
        None, # Indoor (Missing)
        20.0, # Living Room
        5.0, 40.0, 21.0, # Criticals
        35.0, 1000.0, 1500.0 # Thermodynamics
    ]
    features_df, _ = build_physics_features(mock_ha_client, mock_influx_service)
    assert features_df is None


def test_build_physics_features_insufficient_history(mock_ha_client, mock_influx_service):
    """Test feature building with insufficient history."""
    mock_influx_service.fetch_indoor_history.return_value = [19.8, 20.0]
    features_df, _ = build_physics_features(mock_ha_client, mock_influx_service)
    assert features_df is None


class TestCloudCoverGate:
    """Tests for cloud cover gating behind CLOUD_COVER_CORRECTION_ENABLED."""

    def test_cloud_cover_not_fetched_when_disabled(self, mock_ha_client, mock_influx_service, monkeypatch):
        """When CLOUD_COVER_CORRECTION_ENABLED=False, get_hourly_cloud_cover must NOT be called."""
        monkeypatch.setattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False)
        mock_ha_client.get_hourly_cloud_cover = MagicMock(return_value=[30.0] * config.TRAJECTORY_STEPS)

        features_df, _ = build_physics_features(mock_ha_client, mock_influx_service)
        assert features_df is not None
        mock_ha_client.get_hourly_cloud_cover.assert_not_called()

        # Cloud cover columns should all be 0.0 (clear sky default)
        for h in range(1, config.TRAJECTORY_STEPS + 1):
            assert features_df[f'cloud_cover_forecast_{h}h'][0] == 0.0

    def test_cloud_cover_fetched_when_enabled(self, mock_ha_client, mock_influx_service, monkeypatch):
        """When CLOUD_COVER_CORRECTION_ENABLED=True, get_hourly_cloud_cover IS called."""
        monkeypatch.setattr(config, "CLOUD_COVER_CORRECTION_ENABLED", True)
        n = config.TRAJECTORY_STEPS
        cloud_values = [float((h + 1) * 10) for h in range(n)]
        mock_ha_client.get_hourly_cloud_cover = MagicMock(return_value=cloud_values)

        features_df, _ = build_physics_features(mock_ha_client, mock_influx_service)
        assert features_df is not None
        mock_ha_client.get_hourly_cloud_cover.assert_called_once()

        assert features_df['cloud_cover_forecast_1h'][0] == 10.0
        assert features_df[f'cloud_cover_forecast_{n}h'][0] == float(n * 10)

    def test_no_cloud_cover_log_when_disabled(self, mock_ha_client, mock_influx_service, monkeypatch, caplog):
        """No ☁️ cloud cover log line when feature is disabled."""
        monkeypatch.setattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False)
        mock_ha_client.get_hourly_cloud_cover = MagicMock(return_value=[30.0] * 6)

        with caplog.at_level(logging.DEBUG):
            build_physics_features(mock_ha_client, mock_influx_service)

        cloud_logs = [r for r in caplog.records if "☁️" in r.message]
        assert len(cloud_logs) == 0, f"Expected no ☁️ logs but found: {cloud_logs}"

    def test_cloud_cover_log_emitted_when_enabled(self, mock_ha_client, mock_influx_service, monkeypatch, caplog):
        """☁️ cloud cover log line IS emitted when feature is enabled."""
        monkeypatch.setattr(config, "CLOUD_COVER_CORRECTION_ENABLED", True)
        mock_ha_client.get_hourly_cloud_cover = MagicMock(
            return_value=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        )

        with caplog.at_level(logging.DEBUG):
            build_physics_features(mock_ha_client, mock_influx_service)

        cloud_logs = [r for r in caplog.records if "☁️" in r.message]
        assert len(cloud_logs) >= 1, "Expected ☁️ log when cloud cover is enabled"


class TestExtendedForecastHorizon:
    """Tests for the extended forecast horizon when PV_TRAJ_FORECAST_MODE_ENABLED=True."""

    def _make_ha_client(self, traj_steps: int, max_steps: int):
        """Return a mock ha_client that returns max_steps forecasts."""
        client = MagicMock()
        client.get_all_states.return_value = {}
        client.get_state.side_effect = [
            20.0,    # Indoor
            20.0,    # Living Room
            5.0,     # Outdoor
            40.0,    # Outlet
            21.0,    # Target
            35.0,    # Inlet
            1000.0,  # Flow
            1500.0,  # Power
            True,    # DHW
            False,   # Disinfection
            False,   # Boost
            True,    # Defrost
            500.0,   # PV
            0.0,     # Solar Correction
            3.5,     # Wind Speed
            True,    # Fireplace
            False,   # TV
        ]
        # Return max_steps temperature forecasts
        client.get_calibrated_hourly_forecast.return_value = [6.0] * max_steps
        client.get_hourly_forecast.return_value = [6.0] * max_steps
        return client

    def test_extended_pv_forecast_keys_present_when_forecast_mode_enabled(
        self, mock_influx_service, monkeypatch
    ):
        """When PV_TRAJ_FORECAST_MODE_ENABLED=True, features_dict must contain
        pv_forecast_1h … pv_forecast_{PV_TRAJ_MAX_STEPS}h even if TRAJECTORY_STEPS < MAX_STEPS."""
        traj_steps = 4
        max_steps = 12
        monkeypatch.setattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", True)
        monkeypatch.setattr(config, "PV_TRAJ_MAX_STEPS", max_steps)
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", traj_steps)
        monkeypatch.setattr(config, "PV_FORECAST_ENTITY_ID", "")  # skip watts parsing

        ha_client = self._make_ha_client(traj_steps, max_steps)
        features_df, _ = build_physics_features(ha_client, mock_influx_service)

        assert features_df is not None
        # All 12 pv forecast keys must be present (thermally-corrected)
        for h in range(1, max_steps + 1):
            key = f"pv_forecast_{h}h"
            assert key in features_df.columns, f"Missing key: {key}"
        # All 12 electrical pv forecast keys must be present (raw, for trajectory algorithm)
        for h in range(1, max_steps + 1):
            key = f"pv_forecast_electrical_{h}h"
            assert key in features_df.columns, f"Missing key: {key}"
        # All 12 temp forecast keys must be present
        for h in range(1, max_steps + 1):
            key = f"temp_forecast_{h}h"
            assert key in features_df.columns, f"Missing key: {key}"

    def test_no_extra_keys_when_forecast_mode_disabled(
        self, mock_ha_client, mock_influx_service, monkeypatch
    ):
        """When PV_TRAJ_FORECAST_MODE_ENABLED=False, only TRAJECTORY_STEPS keys present."""
        traj_steps = 4
        max_steps = 12
        monkeypatch.setattr(config, "PV_TRAJ_FORECAST_MODE_ENABLED", False)
        monkeypatch.setattr(config, "PV_TRAJ_MAX_STEPS", max_steps)
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", traj_steps)

        features_df, _ = build_physics_features(mock_ha_client, mock_influx_service)
        assert features_df is not None

        # Keys beyond traj_steps should NOT be present
        assert f"pv_forecast_{traj_steps}h" in features_df.columns
        assert f"pv_forecast_{traj_steps + 1}h" not in features_df.columns
        # Electrical forecast keys should NOT be present when forecast mode is disabled
        assert "pv_forecast_electrical_1h" not in features_df.columns
