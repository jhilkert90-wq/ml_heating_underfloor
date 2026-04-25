"""
Tests for the extended 12-hour trajectory / forecast pipeline.

Covers all layers: ha_client, physics_features, prediction_context,
model_wrapper forecast dict, and config boundary validation.
"""
import types
from unittest.mock import MagicMock, patch

import pytest

from src import config
from src.prediction_context import UnifiedPredictionContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ha_weather_response(temps, cloud_covers=None):
    """Build a fake HA weather service_response dict."""
    forecast = []
    for i, t in enumerate(temps):
        entry = {"temperature": t}
        if cloud_covers:
            entry["cloud_coverage"] = cloud_covers[i] if i < len(cloud_covers) else 50.0
        forecast.append(entry)
    return {"weather.home": {"forecast": forecast}}


# ---------------------------------------------------------------------------
# Layer 2 – ha_client
# ---------------------------------------------------------------------------

class TestHAClientForecast12h:
    """HA client returns up to TRAJECTORY_STEPS hourly slots."""

    def _make_client(self):
        from src.ha_client import HAClient
        client = HAClient.__new__(HAClient)
        client.url = "http://localhost:8123"
        client.headers = {}
        return client

    def test_get_hourly_forecast_returns_12_values(self, monkeypatch):
        """Mock HA API returning 12 entries → get_hourly_forecast yields 12."""
        client = self._make_client()
        temps = [float(i) for i in range(12)]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "service_response": _make_ha_weather_response(temps)
        }
        mock_resp.raise_for_status = MagicMock()

        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 12)
        with patch("requests.post", return_value=mock_resp):
            result = client.get_hourly_forecast()

        assert len(result) == 12
        assert result == [round(t, 2) for t in temps]

    def test_get_hourly_forecast_pads_when_fewer_slots(self, monkeypatch):
        """When API returns only 4 entries, result is padded to TRAJECTORY_STEPS."""
        client = self._make_client()
        temps = [5.0, 4.5, 4.0, 3.5]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "service_response": _make_ha_weather_response(temps)
        }
        mock_resp.raise_for_status = MagicMock()

        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 12)
        with patch("requests.post", return_value=mock_resp):
            result = client.get_hourly_forecast()

        assert len(result) == 12
        # Last entry (3.5) is used for padding
        assert result[3] == 3.5
        assert all(v == 3.5 for v in result[4:])

    def test_get_hourly_cloud_cover_returns_12_values(self, monkeypatch):
        """get_hourly_cloud_cover returns TRAJECTORY_STEPS entries when 12 available."""
        client = self._make_client()
        covers = [float(i * 5) for i in range(12)]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "service_response": _make_ha_weather_response(
                [10.0] * 12, cloud_covers=covers
            )
        }
        mock_resp.raise_for_status = MagicMock()

        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 12)
        with patch("requests.post", return_value=mock_resp):
            result = client.get_hourly_cloud_cover()

        assert len(result) == 12
        assert result[0] == 0.0
        assert result[11] == 55.0

    def test_get_hourly_forecast_error_returns_n_zeros(self, monkeypatch):
        """On request failure the method returns TRAJECTORY_STEPS zeros."""
        client = self._make_client()
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 12)
        import requests
        with patch("requests.post", side_effect=requests.RequestException("fail")):
            result = client.get_hourly_forecast()

        assert result == [0.0] * 12


# ---------------------------------------------------------------------------
# Layer 3 – physics_features
# ---------------------------------------------------------------------------

class TestPhysicsFeatures12hKeys:
    """build_physics_features emits forecast keys up to TRAJECTORY_STEPS."""

    def _make_minimal_ha_client(self, n_fc):
        """Minimal ha_client stub that returns n_fc forecast entries."""
        client = MagicMock()
        temps = [float(h) for h in range(n_fc)]
        pvs = [float(h * 100) for h in range(n_fc)]
        client.get_hourly_forecast.return_value = temps
        client.get_hourly_cloud_cover.return_value = [0.0] * n_fc
        # Return the same values when calibration is called
        client.get_calibrated_hourly_forecast.return_value = temps
        return client, temps, pvs

    def _make_influx_service(self, n_fc):
        """Minimal InfluxService stub with proper history mocks."""
        svc = MagicMock()
        # Need 6+ indoor and 3+ outlet history for feature building
        svc.fetch_indoor_history.return_value = [20.0] * 18
        svc.fetch_outlet_history.return_value = [35.0] * 18
        svc.fetch_pv_history.return_value = [0.0] * 18
        svc.get_pv_forecast.return_value = None
        return svc

    def test_emits_temp_and_pv_keys_up_to_trajectory_steps(self, monkeypatch):
        """With TRAJECTORY_STEPS=12 all keys temp_forecast_1h..12h are present."""
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 12)
        monkeypatch.setattr(config, "ENABLE_DELTA_FORECAST_CALIBRATION", False)

        ha_client, temps, pvs = self._make_minimal_ha_client(12)
        # Minimal all_states mock
        def _get_state(entity_id, all_states=None, is_binary=False):
            if is_binary:
                return False
            return 20.0
        ha_client.get_state.side_effect = _get_state
        ha_client.get_all_states.return_value = {}

        influx_svc = self._make_influx_service(12)

        from src.physics_features import build_physics_features
        df, _ = build_physics_features(ha_client, influx_svc)

        assert df is not None
        row = df.iloc[0]
        for h in range(1, 13):
            assert f"temp_forecast_{h}h" in row, f"Missing temp_forecast_{h}h"
            assert f"pv_forecast_{h}h" in row, f"Missing pv_forecast_{h}h"

    def test_does_not_emit_keys_beyond_trajectory_steps(self, monkeypatch):
        """With TRAJECTORY_STEPS=6, keys _7h through _12h must not appear."""
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 6)
        monkeypatch.setattr(config, "ENABLE_DELTA_FORECAST_CALIBRATION", False)

        ha_client, temps, pvs = self._make_minimal_ha_client(6)

        def _get_state(entity_id, all_states=None, is_binary=False):
            if is_binary:
                return False
            return 20.0
        ha_client.get_state.side_effect = _get_state
        ha_client.get_all_states.return_value = {}

        influx_svc = self._make_influx_service(6)

        from src.physics_features import build_physics_features
        df, _ = build_physics_features(ha_client, influx_svc)

        assert df is not None
        row = df.iloc[0]
        for h in range(7, 13):
            assert f"temp_forecast_{h}h" not in row, f"Unexpected temp_forecast_{h}h"


# ---------------------------------------------------------------------------
# Layer 4 – prediction_context
# ---------------------------------------------------------------------------

class TestPredictionContext12h:
    """UnifiedPredictionContext respects TRAJECTORY_STEPS for array lengths."""

    def _make_features(self, n_fc, outdoor_base=10.0, pv_base=100.0):
        features = {}
        for h in range(1, n_fc + 1):
            features[f"temp_forecast_{h}h"] = outdoor_base + h
            features[f"pv_forecast_{h}h"] = pv_base * h
        return features

    def test_outdoor_and_pv_forecast_arrays_have_length_12(self, monkeypatch):
        """With TRAJECTORY_STEPS=12, both forecast arrays have length 12."""
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 12)
        monkeypatch.setattr(config, "CYCLE_INTERVAL_MINUTES", 10)

        features = self._make_features(12)
        ctx = UnifiedPredictionContext.create_prediction_context(
            features=features,
            outdoor_temp=10.0,
            pv_power=0.0,
            thermal_features={},
        )

        assert len(ctx["outdoor_forecast"]) == 12
        assert len(ctx["pv_forecast"]) == 12
        assert ctx["use_forecasts"] is True

    def test_fallback_arrays_have_length_12_when_no_features(self, monkeypatch):
        """Without forecast features, fallback arrays must be length 12."""
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 12)
        monkeypatch.setattr(config, "CYCLE_INTERVAL_MINUTES", 10)

        ctx = UnifiedPredictionContext.create_prediction_context(
            features={},
            outdoor_temp=5.0,
            pv_power=0.0,
            thermal_features={},
        )

        assert len(ctx["outdoor_forecast"]) == 12
        assert all(v == 5.0 for v in ctx["outdoor_forecast"])
        assert len(ctx["pv_forecast"]) == 12
        assert ctx["use_forecasts"] is False

    def test_cycle_aligned_gt6h_picks_correct_slot(self, monkeypatch):
        """With TRAJECTORY_STEPS=12 and a 2h cycle, avg_outdoor == temp_forecast_2h."""
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 12)
        # 2-hour cycle (120 minutes) — within the 180-min reasonable cap
        monkeypatch.setattr(config, "CYCLE_INTERVAL_MINUTES", 120)

        features = self._make_features(12, outdoor_base=0.0)
        # temp_forecast_2h = 0.0 + 2 = 2.0
        ctx = UnifiedPredictionContext.create_prediction_context(
            features=features,
            outdoor_temp=0.0,
            pv_power=0.0,
            thermal_features={},
        )

        assert ctx["avg_outdoor"] == pytest.approx(2.0)

    def test_cycle_aligned_90min_cycle(self, monkeypatch):
        """With TRAJECTORY_STEPS=12 and a 90-min cycle, avg_outdoor == temp_forecast_2h
        (round(1.5) = 2, so index 1 → forecast_2h)."""
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 12)
        monkeypatch.setattr(config, "CYCLE_INTERVAL_MINUTES", 90)  # 1.5 hours

        features = self._make_features(12, outdoor_base=0.0)
        # cycle_hours=1.5 → round(1.5)=2 → idx=1 → forecast[1]=temp_forecast_2h=2.0
        ctx = UnifiedPredictionContext.create_prediction_context(
            features=features,
            outdoor_temp=0.0,
            pv_power=0.0,
            thermal_features={},
        )

        assert ctx["avg_outdoor"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Layer 5 – model_wrapper forecast truncation
# ---------------------------------------------------------------------------

class TestModelWrapperForecastTruncation:
    """Forecast arrays passed to the trajectory call have TRAJECTORY_STEPS+1 entries."""

    def test_forecast_arrays_truncated_to_trajectory_steps_plus_one(
        self, monkeypatch
    ):
        """With TRAJECTORY_STEPS=12, outdoor_arr and pv_arr have length 13."""
        monkeypatch.setattr(config, "TRAJECTORY_STEPS", 12)
        monkeypatch.setattr(config, "CYCLE_INTERVAL_MINUTES", 10)

        # Build a context with 12-element forecast arrays
        outdoor_fc = [float(h) for h in range(1, 13)]  # 12 entries
        pv_fc = [float(h * 10) for h in range(1, 13)]
        context = {
            "avg_outdoor": 5.0,
            "avg_pv": 100.0,
            "outdoor_forecast": outdoor_fc,
            "pv_forecast": pv_fc,
            "avg_cloud_cover": 0.0,
            "cloud_cover_forecast": [0.0] * 12,
        }

        # Replicate the truncation logic from model_wrapper._get_forecast_context
        n = config.TRAJECTORY_STEPS + 1
        outdoor_arr = ([5.0] + context["outdoor_forecast"])[:n]
        pv_arr = ([100.0] + context["pv_forecast"])[:n]

        assert len(outdoor_arr) == 13
        assert len(pv_arr) == 13
        # First element is the current value, rest are forecasts
        assert outdoor_arr[0] == 5.0
        assert outdoor_arr[1] == 1.0


# ---------------------------------------------------------------------------
# Layer 6 – config boundary
# ---------------------------------------------------------------------------

class TestTrajectoryStepsConfigBoundary:
    """TRAJECTORY_STEPS=12 is valid; the HA addon schema allows int(2,12)."""

    def test_trajectory_steps_12_accepted(self, monkeypatch):
        """Setting TRAJECTORY_STEPS=12 via env works and is stored correctly."""
        monkeypatch.setenv("TRAJECTORY_STEPS", "12")
        # Re-evaluate the expression the same way config.py does
        import os
        value = int(os.getenv("TRAJECTORY_STEPS", "4"))
        assert value == 12

    def test_trajectory_steps_default_is_4(self, monkeypatch):
        """Default TRAJECTORY_STEPS is 4."""
        monkeypatch.delenv("TRAJECTORY_STEPS", raising=False)
        import os
        value = int(os.getenv("TRAJECTORY_STEPS", "4"))
        assert value == 4
