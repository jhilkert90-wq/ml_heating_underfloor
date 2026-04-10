
import pytest
from unittest.mock import patch, MagicMock
from src.ha_client import get_sensor_attributes, create_ha_client


@pytest.fixture
def mock_config():
    """Fixture to mock the config module."""
    with patch('src.ha_client.config') as mock_config:
        mock_config.HASS_URL = "http://localhost:8123"
        mock_config.HASS_TOKEN = "test-token"
        mock_config.SHADOW_MODE = False
        mock_config.TARGET_OUTLET_TEMP_ENTITY_ID = "sensor.ml_vorlauftemperatur"
        mock_config.MAE_ENTITY_ID = "sensor.ml_model_mae"
        mock_config.RMSE_ENTITY_ID = "sensor.ml_model_rmse"
        yield mock_config


@pytest.fixture
def ha_client(mock_config):
    """Returns an initialized HAClient instance."""
    return create_ha_client()


@patch('src.ha_client.requests')
def test_get_all_states_success(mock_requests, ha_client):
    """Test successful retrieval of all states."""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"entity_id": "sensor.temp1", "state": "20"},
        {"entity_id": "sensor.temp2", "state": "30"},
    ]
    mock_requests.get.return_value = mock_response

    states = ha_client.get_all_states()
    assert len(states) == 2
    assert states["sensor.temp1"]["state"] == "20"


@patch('src.ha_client.requests')
def test_get_state_from_cache(mock_requests, ha_client):
    """Test retrieving state from cache."""
    states_cache = {"sensor.temp1": {"state": "25"}}
    state = ha_client.get_state("sensor.temp1", states_cache)
    assert state == 25.0


@patch('src.ha_client.requests')
def test_set_state_success(mock_requests, ha_client):
    """Test successfully setting a state."""
    ha_client.set_state("sensor.test", 22.5)
    mock_requests.post.assert_called_once()


@patch('src.ha_client.requests')
def test_set_state_suffixes_shadow_output_entity(mock_requests, ha_client, mock_config):
    """Shadow deployment should publish to suffixed HA output entities."""
    mock_config.SHADOW_MODE = True

    ha_client.set_state("sensor.ml_model_mae", 22.5)

    called_url = mock_requests.post.call_args.args[0]
    assert called_url.endswith("/api/states/sensor.ml_model_mae_shadow")


def test_get_calibrated_hourly_forecast(ha_client):
    """Test forecast calibration logic."""
    with patch.object(
        ha_client, 'get_hourly_forecast', return_value=[10, 12, 14, 11]
    ):
        calibrated = ha_client.get_calibrated_hourly_forecast(11.5)
        assert calibrated == [11.5, 13.5, 15.5, 12.5, 12.5, 12.5]


@patch('src.ha_client.requests')
def test_log_feature_importance(mock_requests, ha_client):
    """Test logging feature importance."""
    ha_client.log_feature_importance({"feat1": 0.8, "feat2": 0.2})
    mock_requests.post.assert_called()


@patch('src.ha_client.requests')
def test_log_model_metrics(mock_requests, ha_client):
    """Test logging MAE and RMSE metrics."""
    ha_client.log_model_metrics(0.1, 0.2)
    assert mock_requests.post.call_count == 2


@patch('src.ha_client.HAClient.set_state')
def test_log_adaptive_learning_metrics_uses_configured_metric_entity_ids(
    mock_set_state, ha_client, mock_config
):
    """Configured MAE/RMSE entity ids from env must be honored."""
    mock_config.MAE_ENTITY_ID = "sensor.custom_mae"
    mock_config.RMSE_ENTITY_ID = "sensor.custom_rmse"

    ha_client.log_adaptive_learning_metrics(
        {
            "learning_confidence": 0.95,
            "mae_all_time": 0.1,
            "rmse_all_time": 0.2,
        }
    )

    called_entity_ids = [call.args[0] for call in mock_set_state.call_args_list]
    assert "sensor.custom_mae" in called_entity_ids
    assert "sensor.custom_rmse" in called_entity_ids


@patch('src.ha_client.HAClient.set_state')
def test_log_adaptive_learning_metrics(mock_set_state, ha_client):
    """Test logging of adaptive learning metrics."""
    metrics = {
        "learning_confidence": 0.95,
        "mae_all_time": 0.1,
        "rmse_all_time": 0.2,
        "fireplace_heat_weight": 6.4,
        "heat_source_channels_enabled": True,
        "delta_t_floor": 3.2,
        "cloud_factor_exponent": 1.3,
        "solar_decay_tau_hours": 0.8,
        "fp_heat_output_kw": 6.4,
        "fp_decay_time_constant": 1.0,
        "room_spread_delay_minutes": 38.0,
    }
    ha_client.log_adaptive_learning_metrics(metrics)

    learning_attributes = mock_set_state.call_args_list[0].args[2]

    assert mock_set_state.call_count == 4
    assert learning_attributes["fireplace_heat_weight"] == pytest.approx(6.4)
    assert learning_attributes["heat_source_channels_enabled"] is True
    assert learning_attributes["delta_t_floor"] == pytest.approx(3.2)
    assert learning_attributes["cloud_factor_exponent"] == pytest.approx(1.3)
    assert learning_attributes["solar_decay_tau_hours"] == pytest.approx(0.8)
    assert learning_attributes["fp_heat_output_kw"] == pytest.approx(6.4)
    assert learning_attributes["fp_decay_time_constant"] == pytest.approx(1.0)
    assert learning_attributes["room_spread_delay_minutes"] == pytest.approx(38.0)


@patch('src.ha_client.HAClient.set_state')
def test_log_adaptive_learning_metrics_omits_channel_only_attrs_when_disabled(
    mock_set_state, ha_client
):
    """Channel-only HA attributes should only appear in channel mode."""
    metrics = {
        "learning_confidence": 0.95,
        "mae_all_time": 0.1,
        "rmse_all_time": 0.2,
        "fireplace_heat_weight": 6.4,
        "heat_source_channels_enabled": False,
    }

    ha_client.log_adaptive_learning_metrics(metrics)

    learning_attributes = mock_set_state.call_args_list[0].args[2]
    assert learning_attributes["fireplace_heat_weight"] == pytest.approx(6.4)
    assert "delta_t_floor" not in learning_attributes
    assert "cloud_factor_exponent" not in learning_attributes
    assert "solar_decay_tau_hours" not in learning_attributes


def test_get_sensor_attributes():
    """Test the retrieval of sensor attributes."""
    attrs = get_sensor_attributes("sensor.ml_model_mae")
    assert attrs["friendly_name"] == "ML Model MAE"


def test_get_sensor_attributes_shadow_variant_has_distinct_metadata(mock_config):
    """Shadow sensor metadata should keep base semantics but unique identity."""
    mock_config.SHADOW_MODE = True
    mock_config.MAE_ENTITY_ID = "sensor.custom_mae"

    attrs = get_sensor_attributes("sensor.custom_mae")

    assert attrs["friendly_name"] == "ML Model MAE Shadow"
    assert attrs["unique_id"] == "ml_heating_model_mae_shadow"


@patch('src.ha_client.HAClient.set_state')
def test_log_feature_importance_uses_shadow_entity_in_shadow_deployment(
    mock_set_state, ha_client, mock_config
):
    """Hard-coded HA output sensors should also be shadow-suffixed."""
    mock_config.SHADOW_MODE = True

    ha_client.log_feature_importance({"feat1": 0.8, "feat2": 0.2})

    assert mock_set_state.call_args.args[0] == "sensor.ml_feature_importance_shadow"


@patch('src.ha_client.requests')
def test_set_state_sanitizes_numpy_types(mock_requests, ha_client):
    """Numpy types in attributes must be converted to native Python types."""
    import numpy as np

    attributes = {
        "final_temp": np.float64(21.0),
        "confidence": np.float64(5.0),
        "count": np.int64(42),
        "is_active": np.bool_(True),
        "nested": {"value": np.float64(3.14)},
        "list_vals": [np.float64(1.0), np.float64(2.0)],
        "plain_string": "hello",
    }

    ha_client.set_state("sensor.test", np.float64(22.5), attributes)

    call_kwargs = mock_requests.post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

    # All numpy types should be converted to plain Python
    attrs = payload["attributes"]
    assert type(attrs["final_temp"]) is float
    assert type(attrs["confidence"]) is float
    assert type(attrs["count"]) is int
    assert type(attrs["is_active"]) is bool
    assert type(attrs["nested"]["value"]) is float
    assert all(type(v) is float for v in attrs["list_vals"])
    assert attrs["plain_string"] == "hello"
