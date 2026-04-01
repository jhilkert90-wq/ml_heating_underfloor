
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, timezone

from src.influx_service import (
    InfluxService,
    get_influx_service,
    reset_influx_service,
)


@pytest.fixture
def mock_config():
    """Fixture to mock the config module."""
    with patch('src.influx_service.config') as mock_config:
        mock_config.INFLUX_URL = "http://localhost:8086"
        mock_config.INFLUX_TOKEN = "test-token"
        mock_config.INFLUX_ORG = "test-org"
        mock_config.INFLUX_BUCKET = "test-bucket"
        mock_config.INFLUX_FEATURES_BUCKET = "test-features"
        mock_config.SHADOW_MODE = False
        mock_config.HISTORY_STEP_MINUTES = 5
        mock_config.ACTUAL_OUTLET_TEMP_ENTITY_ID = "sensor.outlet_temp"
        mock_config.INDOOR_TEMP_ENTITY_ID = "sensor.indoor_temp"
        yield mock_config


@pytest.fixture
def influx_service(mock_config):
    """Returns an initialized InfluxService instance."""
    service = InfluxService(
        url="http://localhost:8086", token="test-token", org="test-org"
    )
    yield service
    service.close()


def test_get_pv_forecast_success(influx_service):
    """Test successful retrieval of PV forecast."""
    mock_df = pd.DataFrame({
        "_time": pd.to_datetime(
            ["2023-01-01T13:00:00Z", "2023-01-01T14:00:00Z"]
        ),
        "total": [100.0, 200.0]
    })
    influx_service.query_api.query_data_frame = MagicMock(return_value=mock_df)

    mock_now = datetime(2023, 1, 1, 12, 1, 0, tzinfo=timezone.utc)
    with patch('src.influx_service.datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_now
        result = influx_service.get_pv_forecast()
        assert len(result) == 4
        assert result[0] == 100.0
        assert result[1] == 200.0


def test_fetch_history_success(influx_service):
    """Test successful fetching of historical data."""
    mock_df = pd.DataFrame({"value": [20.0, 21.0, 22.0]})
    influx_service.query_api.query_data_frame = MagicMock(return_value=mock_df)
    result = influx_service.fetch_history("sensor.test", 5, 19.0)
    assert len(result) == 5
    assert result == [20.0, 21.0, 22.0, 22.0, 22.0]


def test_write_metrics(influx_service):
    """Test writing various metrics to InfluxDB."""
    influx_service.write_api.write = MagicMock()

    # Test write_feature_importances
    influx_service.write_feature_importances({"feat1": 0.5})
    influx_service.write_api.write.assert_called()

    # Test write_prediction_metrics
    influx_service.write_prediction_metrics({"1h": {"mae": 0.1}})
    influx_service.write_api.write.assert_called()

    # Test write_thermal_learning_metrics
    mock_model = MagicMock()
    mock_model.get_adaptive_learning_metrics.return_value = {
        "current_parameters": {}
    }
    influx_service.write_thermal_learning_metrics(mock_model)
    influx_service.write_api.write.assert_called()

    # Test write_thermodynamic_metrics (New)
    thermo_metrics = {
        "cop_realtime": 3.5,
        "thermal_power_kw": 5.0,
        "delta_t": 5.0,
        "flow_rate": 1000.0,
        "inlet_temp": 35.0
    }
    influx_service.write_thermodynamic_metrics(thermo_metrics)
    influx_service.write_api.write.assert_called()


def test_singleton_logic(mock_config):
    """Test the singleton creation and reset logic."""
    reset_influx_service()
    instance1 = get_influx_service()
    instance2 = get_influx_service()
    assert instance1 is instance2
    
    # Reset should close instance1
    reset_influx_service()
    
    instance3 = get_influx_service()
    assert instance1 is not instance3
    
    # Clean up final instance
    reset_influx_service()


def test_write_thermal_learning_metrics_includes_channel_fields(influx_service):
    """Channel-aware thermal exports should include the new channel fields."""
    influx_service.write_api.write = MagicMock()

    mock_model = MagicMock()
    mock_model.thermal_time_constant = 5.5
    mock_model.heat_loss_coefficient = 0.22
    mock_model.outlet_effectiveness = 0.88
    mock_model.pv_heat_weight = 0.0032
    mock_model.fireplace_heat_weight = 6.4
    mock_model.tv_heat_weight = 0.46
    mock_model.solar_lag_minutes = 75.0
    mock_model.slab_time_constant_hours = 1.7
    mock_model.base_outlet_effectiveness = 0.8
    mock_model.get_adaptive_learning_metrics.return_value = {
        "learning_confidence": 3.6,
        "current_learning_rate": 0.01,
        "parameter_updates": 8,
        "thermal_time_constant_stability": 0.2,
        "heat_loss_coefficient_stability": 0.03,
        "outlet_effectiveness_stability": 0.04,
        "heat_source_channels_enabled": True,
        "current_parameters": {
            "thermal_time_constant": 5.5,
            "heat_loss_coefficient": 0.22,
            "outlet_effectiveness": 0.88,
            "pv_heat_weight": 0.0032,
            "fireplace_heat_weight": 6.4,
            "tv_heat_weight": 0.46,
            "solar_lag_minutes": 75.0,
            "slab_time_constant_hours": 1.7,
            "delta_t_floor": 3.4,
            "cloud_factor_exponent": 1.4,
            "solar_decay_tau_hours": 0.9,
            "fp_heat_output_kw": 6.4,
            "fp_decay_time_constant": 1.1,
            "room_spread_delay_minutes": 42.0,
        },
    }

    influx_service.write_thermal_learning_metrics(mock_model)

    record = influx_service.write_api.write.call_args.kwargs["record"]
    fields = record._fields

    assert fields["fireplace_heat_weight"] == pytest.approx(6.4)
    assert fields["heat_source_channels_enabled"] is True
    assert fields["delta_t_floor"] == pytest.approx(3.4)
    assert fields["cloud_factor_exponent"] == pytest.approx(1.4)
    assert fields["solar_decay_tau_hours"] == pytest.approx(0.9)
    assert fields["fp_heat_output_kw"] == pytest.approx(6.4)
    assert fields["fp_decay_time_constant"] == pytest.approx(1.1)
    assert fields["room_spread_delay_minutes"] == pytest.approx(42.0)


def test_generated_metrics_use_shadow_features_bucket(influx_service, mock_config):
    """Configured features bucket should gain the shadow suffix in shadow deployment."""
    influx_service.write_api.write = MagicMock()
    mock_config.SHADOW_MODE = True
    mock_config.INFLUX_FEATURES_BUCKET = "custom_features"

    influx_service.write_feature_importances({"feat1": 0.5})

    assert influx_service.write_api.write.call_args.kwargs["bucket"] == (
        "custom_features_shadow"
    )


def test_explicit_generated_metrics_bucket_is_shadow_suffixed(
    influx_service, mock_config
):
    """Explicit generated-metrics buckets should also be shadow-isolated."""
    influx_service.write_api.write = MagicMock()
    mock_config.SHADOW_MODE = True

    influx_service.write_thermodynamic_metrics(
        {"cop_realtime": 3.2, "thermal_power_kw": 4.8},
        bucket="manual_features",
    )

    assert influx_service.write_api.write.call_args.kwargs["bucket"] == (
        "manual_features_shadow"
    )
