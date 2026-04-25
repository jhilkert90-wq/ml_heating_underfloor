"""
This module provides a client for interacting with the Home Assistant API.

It abstracts the details of making HTTP requests to Home Assistant for
fetching sensor states, setting sensor values, and calling services. This
centralizes all communication with Home Assistant, making the rest of the
application cleaner and easier to manage.
"""
import logging
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import requests

# Support both package-relative and direct import for notebooks
try:
    from . import config  # Package-relative import
    from .shadow_mode import (
        get_base_output_entity_id,
        get_shadow_output_entity_id,
    )
except ImportError:
    import config  # Direct import fallback for notebooks
    from shadow_mode import (  # type: ignore
        get_base_output_entity_id,
        get_shadow_output_entity_id,
    )


def _sanitize_for_json(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization.

    Recursively walks dicts/lists and converts numpy scalars (float64,
    int64, bool_, etc.) to their plain-Python equivalents so that
    ``json.dumps`` and Home Assistant can handle them without surprises.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        sanitized = [_sanitize_for_json(v) for v in obj]
        return type(obj)(sanitized) if isinstance(obj, tuple) else sanitized
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class HAClient:
    """A client for interacting with the Home Assistant API."""

    def __init__(self, url: str, token: str):
        """
        Initializes the Home Assistant client.

        Args:
            url: The base URL of the Home Assistant instance
            (e.g., http://homeassistant.local:8123).
            token: A long-lived access token for authentication.
        """
        self.url = url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def get_all_states(self) -> Dict[str, Any]:
        """
        Retrieves a snapshot of all entity states from Home Assistant.

        This method is highly efficient as it fetches all states in a single
        API call, which is much faster than querying entities one by one. The
        result is cached by the caller for subsequent `get_state` calls within
        the same update cycle.

        Returns:
            A dictionary where keys are entity_ids and values are the
            corresponding state objects from Home Assistant. Returns an
            empty dictionary if the request fails.
        """
        url = f"{self.url}/api/states"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            # Create a dictionary mapping entity_id to its state object for quick lookups.
            return {entity["entity_id"]: entity for entity in resp.json()}
        except requests.RequestException as exc:
            warnings.warn(f"HA request error for all states: {exc}")
            return {}

    def get_state(
        self,
        entity_id: str,
        states_cache: Optional[Dict[str, Any]] = None,
        is_binary: bool = False,
    ) -> Optional[Any]:
        """
        Retrieves the state of a specific Home Assistant entity.

        This method prioritizes using a provided `states_cache` (from
        `get_all_states`) for efficiency. If no cache is given, it makes a
        direct API call as a fallback. It handles type conversion for
        numerical and binary sensors.

        Args:
            entity_id: The full ID of the entity (e.g., 'sensor.temperature').
            states_cache: A dictionary of all states, as returned by `get_all_states`.
            is_binary: If True, treats the state as binary ('on'/'off').

        Returns:
            The processed state of the entity (float, bool, or string),
            or None if the entity is not found or its state is invalid.
        """
        if states_cache is None:
            # Fallback to individual request if no cache is provided.
            url = f"{self.url}/api/states/{entity_id}"
            try:
                resp = requests.get(url, headers=self.headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                warnings.warn(f"HA request error {entity_id}: {exc}")
                return None
        else:
            # Use the provided cache.
            data = states_cache.get(entity_id)

        if data is None:
            return None

        state = data.get("state")
        if state in (None, "unknown", "unavailable"):
            return None

        if is_binary:
            return state == "on"

        try:
            return float(state)
        except (TypeError, ValueError):
            return state

    def call_tibber_get_prices(
        self,
        start: str,
        end: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Call the tibber.get_prices HA service and return price entries.

        Args:
            start: ISO-format start time (e.g. '2026-04-14 00:00:00').
            end:   ISO-format end time.

        Returns:
            List of ``{"start_time": str, "price": float}`` dicts for the
            first (auto-detected) Tibber home, or *None* on failure.
        """
        svc_url = f"{self.url}/api/services/tibber/get_prices"
        body = {"start": start, "end": end}

        try:
            resp = requests.post(
                svc_url,
                headers=self.headers,
                json=body,
                timeout=15,
                params={"return_response": "true"},
            )
            resp.raise_for_status()
            svc_resp = resp.json().get("service_response", {})
        except Exception as exc:
            logging.warning("tibber.get_prices service call failed: %s", exc)
            return None

        prices_by_home = svc_resp.get("prices", svc_resp)
        if not isinstance(prices_by_home, dict) or not prices_by_home:
            logging.warning(
                "tibber.get_prices returned unexpected format: %s",
                type(prices_by_home),
            )
            return None

        # Auto-detect: take the first home's price list
        first_home_prices = next(iter(prices_by_home.values()))
        if not isinstance(first_home_prices, list):
            logging.warning(
                "tibber.get_prices: first home value is not a list"
            )
            return None

        return first_home_prices

    # --- Legacy sensor-based price reading (deprecated) -----------------

    def get_electricity_price(
        self,
        states_cache: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Read current electricity price from HA sensor state.

        .. deprecated::
            Use :meth:`call_tibber_get_prices` via the PriceOptimizer cache
            instead.  This method is kept as a fallback.

        Returns a dict with:
          current_price: float (EUR/kWh)
          today:         list of hourly prices
          tomorrow:      list of hourly prices (may be empty before ~13:00)

        Returns None if the sensor is missing or unavailable.
        """
        entity_id = config.ELECTRICITY_PRICE_ENTITY_ID
        if states_cache:
            data = states_cache.get(entity_id)
        else:
            url = f"{self.url}/api/states/{entity_id}"
            try:
                resp = requests.get(url, headers=self.headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                logging.warning("Failed to read electricity price: %s", exc)
                return None

        if data is None:
            return None

        state = data.get("state")
        if state in (None, "unknown", "unavailable"):
            return None

        try:
            current_price = float(state)
        except (TypeError, ValueError):
            logging.warning(
                "Electricity price state is not numeric: %s", state
            )
            return None

        attrs = data.get("attributes", {})
        today = attrs.get("today", [])
        tomorrow = attrs.get("tomorrow", [])

        # Tibber stores prices as list of dicts with 'total' key, or plain floats
        def _extract_prices(raw: Any) -> List[float]:
            if not isinstance(raw, list):
                return []
            result = []
            for item in raw:
                if isinstance(item, dict):
                    val = item.get("total", item.get("value"))
                    if val is not None:
                        try:
                            result.append(float(val))
                        except (TypeError, ValueError):
                            pass
                else:
                    try:
                        result.append(float(item))
                    except (TypeError, ValueError):
                        pass
            return result

        return {
            "current_price": current_price,
            "today": _extract_prices(today),
            "tomorrow": _extract_prices(tomorrow),
        }

    def set_state(
        self,
        entity_id: str,
        value: float,
        attributes: Optional[Dict[str, Any]] = None,
        round_digits: Optional[int] = 1,
    ) -> None:
        """
        Creates or updates the state of a sensor entity in Home Assistant.

        This is the primary method for publishing the model's outputs (like
        the target temperature or performance metrics) back to Home
        Assistant, making them visible and usable in the HA frontend and
        automations.

        Args:
            entity_id: The ID of the sensor to create/update.
            value: The main state value for the sensor.
            attributes: A dictionary of additional attributes for the sensor.
            round_digits: The number of decimal places to round the state
            value to.
        """
        effective_entity_id = get_shadow_output_entity_id(
            entity_id,
            shadow_deployment=getattr(config, "SHADOW_MODE", False),
        )
        url = f"{self.url}/api/states/{effective_entity_id}"

        # Round the value only if round_digits is specified.
        if round_digits is not None:
            state_value = round(value, round_digits)
        else:
            state_value = value

        # Convert numpy types to native Python for JSON serialization
        state_value = _sanitize_for_json(state_value)
        clean_attributes = _sanitize_for_json(attributes or {})

        # The state must be sent as a string.
        payload = {
            "state": (
                f"{state_value:.{round_digits}f}"
                if isinstance(state_value, float) and round_digits is not None
                else str(state_value)
            ),
            "attributes": clean_attributes,
        }
        try:
            logging.debug(
                "Setting HA state for %s: %s",
                effective_entity_id,
                payload,
            )
            requests.post(url, headers=self.headers, json=payload, timeout=10)
        except requests.RequestException as exc:
            warnings.warn(
                f"HA state set failed for {effective_entity_id}: {exc}"
            )

    def get_hourly_forecast(self) -> List[float]:
        """
        Retrieves the hourly weather forecast via the Home Assistant service.
        Supports up to TRAJECTORY_STEPS-hour forecasts and pads if needed.
        Returns:
            A list of forecasted temperatures for the next TRAJECTORY_STEPS hours,
            or a default list of zeros if the call fails.
        """
        n = config.TRAJECTORY_STEPS
        svc_url = f"{self.url}/api/services/weather/get_forecasts"
        body = {"entity_id": ["weather.home"], "type": "hourly"}

        try:
            resp = requests.post(
                svc_url,
                headers=self.headers,
                json=body,
                timeout=10,
                params={"return_response": "true"},
            )
            resp.raise_for_status()
            data = resp.json()["service_response"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            logging.debug(
                "Weather forecast request failed: %s", exc
            )
            return [0.0] * n

        try:
            forecast_list = data["weather.home"]["forecast"]
        except (KeyError, TypeError):
            return [0.0] * n

        # Extract the temperature from the first n forecast entries.
        result = []
        for entry in forecast_list[:n]:
            temp = entry.get("temperature") if isinstance(entry, dict) else None
            result.append(
                round(temp, 2) if isinstance(temp, (int, float)) else 0.0
            )
        # Pad to n if less
        while len(result) < n:
            result.append(result[-1] if result else 0.0)
        return result

    def get_hourly_cloud_cover(self) -> List[float]:
        """
        Retrieves the hourly cloud cover forecast from weather.home.
        Reuses the same forecast API call as get_hourly_forecast().
        Returns:
            A list of cloud cover percentages (0-100) for the next TRAJECTORY_STEPS hours.
            Defaults to 50% if data is unavailable.
        """
        n = config.TRAJECTORY_STEPS
        svc_url = f"{self.url}/api/services/weather/get_forecasts"
        body = {"entity_id": ["weather.home"], "type": "hourly"}

        try:
            resp = requests.post(
                svc_url,
                headers=self.headers,
                json=body,
                timeout=10,
                params={"return_response": "true"},
            )
            resp.raise_for_status()
            data = resp.json()["service_response"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            logging.debug(
                "Cloud cover forecast request failed: %s", exc
            )
            return [50.0] * n

        try:
            forecast_list = data["weather.home"]["forecast"]
        except (KeyError, TypeError):
            return [50.0] * n

        # Extract cloud cover from the first n forecast entries
        result = []
        for entry in forecast_list[:n]:
            cloud_cover = entry.get("cloud_coverage") if isinstance(entry, dict) else None
            # Cloud cover is typically 0-100%, default to 50% if missing
            result.append(
                float(cloud_cover) if isinstance(cloud_cover, (int, float)) else 50.0
            )
        
        # Pad to n if less
        while len(result) < n:
            result.append(result[-1] if result else 50.0)

        avg = sum(result) / len(result)
        logging.debug(
            "☁️ Cloud cover forecast fetched: %s (avg=%.1f%%, min=%.0f%%, max=%.0f%%)",
            [f"{v:.0f}%" for v in result],
            avg,
            min(result),
            max(result),
        )
        return result

    def get_calibrated_hourly_forecast(
        self, 
        current_outdoor_temp: float, 
        enable_delta_calibration: bool = True
    ) -> List[float]:
        """
        Retrieves weather forecasts calibrated to local measurements.
        Supports up to TRAJECTORY_STEPS-hour forecasts and pads if needed.
        Args:
            current_outdoor_temp: Actual measured outdoor temperature
            enable_delta_calibration: Whether to apply delta calibration
        Returns:
            List of calibrated temperature forecasts for next TRAJECTORY_STEPS hours
        """
        n = config.TRAJECTORY_STEPS
        # Get raw absolute forecasts (up to TRAJECTORY_STEPS hours)
        raw_forecasts = self.get_hourly_forecast()
        # If delta calibration is disabled, return raw forecasts
        if not enable_delta_calibration:
            logging.debug("Delta calibration disabled, using raw forecasts")
            return raw_forecasts
        # Validate inputs
        if not raw_forecasts or raw_forecasts[0] == 0.0:
            logging.warning(
                "Invalid raw forecast data, skipping delta calibration"
            )
            return raw_forecasts
        if current_outdoor_temp is None or abs(current_outdoor_temp) > 60:
            logging.warning(
                f"Invalid outdoor temperature {current_outdoor_temp}°C, "
                f"skipping delta calibration"
            )
            return raw_forecasts
        # Calculate local temperature offset
        forecast_current_temp = raw_forecasts[0]
        temperature_offset = current_outdoor_temp - forecast_current_temp
        # Apply offset to all forecast hours
        calibrated_forecasts = []
        for raw_temp in raw_forecasts:
            calibrated_temp = raw_temp + temperature_offset
            calibrated_forecasts.append(round(calibrated_temp, 2))
        # Pad to n if less
        while len(calibrated_forecasts) < n:
            calibrated_forecasts.append(calibrated_forecasts[-1] if calibrated_forecasts else current_outdoor_temp)
        logging.debug(
            f"Delta calibration applied: offset={temperature_offset:+.2f}°C"
        )
        logging.debug(f"Raw forecasts: {raw_forecasts}")
        logging.debug(f"Calibrated forecasts: {calibrated_forecasts}")
        return calibrated_forecasts

    def log_feature_importance(self, importances: Dict[str, float]):
        """
        Publishes the model's feature importances to a Home Assistant sensor.

        This creates a sensor (`sensor.ml_feature_importance`) where the
        state is the number of features, and an attribute `top_features`
        lists the most influential features and their importance scores. This
        is useful for monitoring and understanding the model's behavior.

        Args:
            importances: A dictionary mapping feature names to importance
            scores.
        """
        if not importances:
            return

        # Sort features by importance (descending).
        sorted_importances = sorted(
            importances.items(), key=lambda item: item[1], reverse=True
        )

        feature_importance_entity_id = get_shadow_output_entity_id(
            "sensor.ml_feature_importance",
            shadow_deployment=getattr(config, "SHADOW_MODE", False),
        )
        attributes = get_sensor_attributes(feature_importance_entity_id)
        # Store the top 10 features and their importance percentage.
        attributes["top_features"] = {
            f: round(v * 100, 2) for f, v in sorted_importances[:10]
        }
        attributes["last_updated"] = datetime.now(timezone.utc).isoformat()

        logging.debug("Logging feature importance")
        # The state of the sensor is the total number of features.
        self.set_state(
            feature_importance_entity_id,
            len(sorted_importances),
            attributes,
            round_digits=None,
        )

    def log_model_metrics(self, mae: float, rmse: float) -> None:
        """
        Publishes key model performance metrics to dedicated HA sensors.

        This creates sensors for Mean Absolute Error (MAE) and Root Mean
        Squared Error (RMSE), allowing for real-time tracking of the model's
        performance from within Home Assistant.
        
        Note: Model confidence is now provided via 
        sensor.ml_heating_learning.state to avoid redundancy.

        Args:
            mae: The current Mean Absolute Error.
            rmse: The current Root Mean Squared Error.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        mae_entity_id = get_shadow_output_entity_id(
            config.MAE_ENTITY_ID,
            shadow_deployment=getattr(config, "SHADOW_MODE", False),
        )
        rmse_entity_id = get_shadow_output_entity_id(
            config.RMSE_ENTITY_ID,
            shadow_deployment=getattr(config, "SHADOW_MODE", False),
        )

        # Log Mean Absolute Error (MAE)
        attributes_mae = get_sensor_attributes(mae_entity_id)
        attributes_mae["last_updated"] = now_utc
        self.set_state(
            mae_entity_id,
            mae,
            attributes_mae,
            round_digits=4,
        )

        # Log Root Mean Squared Error (RMSE)
        attributes_rmse = get_sensor_attributes(rmse_entity_id)
        attributes_rmse["last_updated"] = now_utc
        self.set_state(
            rmse_entity_id,
            rmse,
            attributes_rmse,
            round_digits=4,
        )

    def log_adaptive_learning_metrics(self, learning_metrics: Dict[str, Any]) -> None:
        """
        Publishes adaptive learning metrics to Home Assistant sensors.
        
        REFACTORED: Implements clean sensor schema to eliminate redundancy:
        - ml_heating_learning: Learning confidence and thermal parameters only
        - ml_model_mae: Enhanced with time-windowed attributes
        - ml_model_rmse: Enhanced with error distribution attributes
        - ml_prediction_accuracy: 24h control quality (no redundant MAE/RMSE)

        Args:
            learning_metrics: Dictionary containing all learning metrics
                from the EnhancedModelWrapper
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        learning_entity_id = get_shadow_output_entity_id(
            "sensor.ml_heating_learning",
            shadow_deployment=getattr(config, "SHADOW_MODE", False),
        )
        mae_entity_id = get_shadow_output_entity_id(
            config.MAE_ENTITY_ID,
            shadow_deployment=getattr(config, "SHADOW_MODE", False),
        )
        rmse_entity_id = get_shadow_output_entity_id(
            config.RMSE_ENTITY_ID,
            shadow_deployment=getattr(config, "SHADOW_MODE", False),
        )
        prediction_accuracy_entity_id = get_shadow_output_entity_id(
            "sensor.ml_prediction_accuracy",
            shadow_deployment=getattr(config, "SHADOW_MODE", False),
        )

        # 1. ML Heating Learning sensor: Confidence + thermal parameters
        attributes_learning = get_sensor_attributes(learning_entity_id)
        attributes_learning.update({
            # Learned thermal parameters exported in all modes
            "thermal_time_constant": learning_metrics.get("thermal_time_constant", 6.0),
            "heat_loss_coefficient": learning_metrics.get("heat_loss_coefficient", 0.146),
            "outlet_effectiveness": learning_metrics.get("outlet_effectiveness", 0.936),
            "pv_heat_weight": learning_metrics.get("pv_heat_weight", 0.001),
            "fireplace_heat_weight": learning_metrics.get("fireplace_heat_weight", 0.0),
            "tv_heat_weight": learning_metrics.get("tv_heat_weight", 0.1),
            "solar_lag_minutes": learning_metrics.get("solar_lag_minutes", 0.0),
            "slab_time_constant_hours": learning_metrics.get("slab_time_constant_hours", 2.0),
            "heat_source_channels_enabled": learning_metrics.get("heat_source_channels_enabled", False),
            
            # Legacy/derived parameters (for backward compatibility)
            "total_conductance": learning_metrics.get("total_conductance", 0.3),
            "equilibrium_ratio": learning_metrics.get("equilibrium_ratio", 0.5),
            
            # Learning progress & health
            "cycle_count": learning_metrics.get("cycle_count", 0),
            "parameter_updates": learning_metrics.get("parameter_updates", 0),
            "model_health": learning_metrics.get("model_health", "unknown"),
            "learning_progress": min(1.0, learning_metrics.get("cycle_count", 0) / 100.0),
            "is_improving": learning_metrics.get("is_improving", False),
            "improvement_percentage": learning_metrics.get("improvement_percentage", 0.0),
            "total_predictions": learning_metrics.get("total_predictions", 0),
            "learning_confidence": learning_metrics.get("learning_confidence", 1.0),
            "last_updated": now_utc
        })

        if learning_metrics.get("heat_source_channels_enabled", False):
            attributes_learning.update({
                "delta_t_floor": learning_metrics.get("delta_t_floor", 0.0),
                "cloud_factor_exponent": learning_metrics.get("cloud_factor_exponent", 1.0),
                "solar_decay_tau_hours": learning_metrics.get("solar_decay_tau_hours", 0.0),
                "fp_heat_output_kw": learning_metrics.get("fp_heat_output_kw", 0.0),
                "fp_decay_time_constant": learning_metrics.get("fp_decay_time_constant", 0.0),
                "room_spread_delay_minutes": learning_metrics.get("room_spread_delay_minutes", 0.0),
            })

            # Per-channel diagnostics (history count, last error)
            for ch_name in ("heat_pump", "pv", "fireplace", "tv"):
                prefix = f"ch_{ch_name}"
                attributes_learning[f"{prefix}_history_count"] = learning_metrics.get(
                    f"{prefix}_history_count", 0
                )
                attributes_learning[f"{prefix}_last_error"] = learning_metrics.get(
                    f"{prefix}_last_error", 0.0
                )

        # State is the learning confidence score (no redundant attribute)
        learning_confidence = learning_metrics.get("learning_confidence", 0.0)
        self.set_state(
            learning_entity_id,
            learning_confidence,
            attributes_learning,
            round_digits=3,
        )

        # 2. Enhanced MAE sensor: All-time MAE + time-windowed breakdowns
        attributes_mae = get_sensor_attributes(mae_entity_id)
        attributes_mae.update({
            "mae_1h": learning_metrics.get("mae_1h", 0.0),
            "mae_6h": learning_metrics.get("mae_6h", 0.0), 
            "mae_24h": learning_metrics.get("mae_24h", 0.0),
            "trend_direction": self._get_mae_trend(learning_metrics),
            "prediction_count": learning_metrics.get("total_predictions", 0),
            "last_updated": now_utc
        })

        # State is all-time MAE
        mae_all_time = learning_metrics.get("mae_all_time", 0.0)
        self.set_state(
            mae_entity_id,
            mae_all_time,
            attributes_mae,
            round_digits=4,
        )

        # 3. Enhanced RMSE sensor: All-time RMSE + error distribution
        attributes_rmse = get_sensor_attributes(rmse_entity_id)
        attributes_rmse.update({
            "recent_max_error": learning_metrics.get("recent_max_error", 0.0),
            "std_error": self._calculate_std_error(learning_metrics),
            "mean_bias": self._calculate_mean_bias(learning_metrics),
            "prediction_count": learning_metrics.get("total_predictions", 0),
            "last_updated": now_utc
        })

        # State is all-time RMSE
        rmse_all_time = learning_metrics.get("rmse_all_time", 0.0)
        self.set_state(
            rmse_entity_id,
            rmse_all_time,
            attributes_rmse,
            round_digits=4,
        )

        # 4. Clean prediction accuracy sensor: 24h control quality only
        attributes_accuracy = get_sensor_attributes(
            prediction_accuracy_entity_id
        )
        attributes_accuracy.update({
            "perfect_accuracy_pct": learning_metrics.get("perfect_accuracy_pct", 0.0),
            "tolerable_accuracy_pct": learning_metrics.get("tolerable_accuracy_pct", 0.0),
            "poor_accuracy_pct": learning_metrics.get("poor_accuracy_pct", 0.0),
            "prediction_count_24h": learning_metrics.get("prediction_count_24h", 0),
            "excellent_all_time_pct": learning_metrics.get("excellent_accuracy_pct", 0.0),
            "good_all_time_pct": learning_metrics.get("good_accuracy_pct", 0.0),
            "last_updated": now_utc
        })

        # State is the percentage of good control (24h window, ±0.2°C)
        good_control_24h = learning_metrics.get("good_control_pct", 0.0)
        self.set_state(
            prediction_accuracy_entity_id,
            good_control_24h,
            attributes_accuracy,
            round_digits=1,
        )

        logging.debug("Logged refactored sensor metrics to HA")

    def _get_mae_trend(self, metrics: Dict[str, Any]) -> str:
        """Determine MAE trend direction."""
        improvement_pct = metrics.get("improvement_percentage", 0.0)
        if improvement_pct > 5.0:
            return "improving"
        elif improvement_pct < -5.0:
            return "degrading"
        else:
            return "stable"

    def _calculate_std_error(self, metrics: Dict[str, Any]) -> float:
        """Calculate standard deviation of errors (placeholder)."""
        # This would need access to individual prediction errors
        # For now, estimate from MAE vs RMSE relationship
        mae = metrics.get("mae_all_time", 0.0)
        rmse = metrics.get("rmse_all_time", 0.0)
        if rmse > mae:
            return round((rmse**2 - mae**2)**0.5, 4)
        return 0.0

    def _calculate_mean_bias(self, metrics: Dict[str, Any]) -> float:
        """Calculate mean bias (systematic over/under-prediction)."""
        # This would need access to individual prediction errors with sign
        # For now, return 0.0 as placeholder
        return 0.0

    def publish_last_run_features(
        self, features_dict: Dict[str, Any]
    ) -> None:
        """Publish all last-run features to sensor.ml_heating_features.

        State is current indoor temperature; all feature keys become
        attributes so they are visible in HA Developer Tools.
        """
        entity_id = get_shadow_output_entity_id(
            config.FEATURES_ENTITY_ID,
            shadow_deployment=getattr(config, "SHADOW_MODE", False),
        )
        attrs = get_sensor_attributes(entity_id)
        now_utc = datetime.now(timezone.utc).isoformat()

        # Copy all features as attributes
        for key, value in features_dict.items():
            attrs[key] = _sanitize_for_json(value)
        attrs["last_updated"] = now_utc

        # State = indoor temp (most useful primary value)
        state_value = features_dict.get(
            "living_room_temp",
            features_dict.get("indoor_temp_lag_30m", 0.0),
        )
        self.set_state(entity_id, state_value, attrs, round_digits=2)
        logging.debug("Published %d features to %s", len(features_dict), entity_id)

    def publish_price_level(
        self, price_info: Dict[str, object]
    ) -> None:
        """Publish electricity price classification to HA."""
        entity_id = get_shadow_output_entity_id(
            "sensor.ml_heating_price_level",
            shadow_deployment=getattr(config, "SHADOW_MODE", False),
        )
        attrs = get_sensor_attributes(entity_id)
        attrs.update({
            "price_eur_kwh": price_info.get("price_eur_kwh"),
            "price_level": price_info.get("price_level", "normal"),
            "cheap_threshold": price_info.get("price_cheap_threshold"),
            "expensive_threshold": price_info.get("price_expensive_threshold"),
            "target_offset": price_info.get("price_target_offset", 0.0),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        })
        # State is the price level string
        self.set_state(
            entity_id,
            price_info.get("price_eur_kwh", 0.0),
            attrs,
            round_digits=4,
        )

    def get_history_bulk(
        self,
        entity_ids: List[str],
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> Optional[List[List[Dict[str, Any]]]]:
        """
        Fetch historical state data for multiple entities from HA REST API.

        Calls ``/api/history/period/{start}`` with ``filter_entity_id``,
        ``minimal_response`` and ``no_attributes`` for efficiency.

        Args:
            entity_ids: List of full entity IDs to fetch.
            start_time: Start of the history window (timezone-aware).
            end_time: Optional end of the window (defaults to now).

        Returns:
            A list of entity history arrays (one per entity), or None on
            failure.  Each inner list contains dicts with at least
            ``state`` and ``last_changed`` keys.
        """
        start_iso = start_time.isoformat()
        url = f"{self.url}/api/history/period/{start_iso}"

        params: Dict[str, str] = {
            "filter_entity_id": ",".join(entity_ids),
            "minimal_response": "",
            "no_attributes": "",
        }
        if end_time is not None:
            params["end_time"] = end_time.isoformat()

        try:
            resp = requests.get(
                url, headers=self.headers, params=params, timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                logging.warning("HA history: unexpected response type %s", type(data))
                return None
            return data
        except requests.Timeout:
            logging.warning("HA history request timed out for %d entities", len(entity_ids))
            return None
        except requests.RequestException as exc:
            logging.warning("HA history request failed: %s", exc)
            return None


def get_sensor_attributes(entity_id: str) -> Dict[str, Any]:
    """
    Provides a standardized set of attributes for the sensors created by this script.

    This function acts as a central repository for sensor metadata like
    `friendly_name`, `unit_of_measurement`, `device_class`, and `icon`.
    This ensures that all sensors created by the application have a
    consistent and user-friendly appearance in the Home Assistant frontend.

    Args:
        entity_id: The ID of the sensor for which to get attributes.

    Returns:
        A dictionary of attributes for that sensor.
    """
    effective_entity_id = get_shadow_output_entity_id(
        entity_id,
        shadow_deployment=getattr(config, "SHADOW_MODE", False),
    )
    base_entity_id = get_base_output_entity_id(effective_entity_id)

    base_attributes = {
        "state_class": "measurement",
    }
    sensor_specific_attributes = {
        config.TARGET_OUTLET_TEMP_ENTITY_ID: {
            "unique_id": "ml_heating_target_outlet_temp",
            "friendly_name": "ML Target Outlet Temp",
            "unit_of_measurement": "°C",
            "device_class": "temperature",
            "icon": "mdi:thermometer-water",
        },
        "sensor.ml_predicted_indoor_temp": {
            "unique_id": "ml_heating_predicted_indoor_temp",
            "friendly_name": "ML Predicted Indoor Temp",
            "unit_of_measurement": "°C",
            "device_class": "temperature",
            "icon": "mdi:home-thermometer-outline",
        },
        "sensor.ml_model_confidence": {
            "unique_id": "ml_heating_model_confidence",
            "friendly_name": "ML Model Confidence",
            "unit_of_measurement": "std dev",
            "icon": "mdi:chart-line",
        },
        config.MAE_ENTITY_ID: {
            "unique_id": "ml_heating_model_mae",
            "friendly_name": "ML Model MAE",
            "unit_of_measurement": "°C",
            "icon": "mdi:chart-line",
        },
        config.RMSE_ENTITY_ID: {
            "unique_id": "ml_heating_model_rmse",
            "friendly_name": "ML Model RMSE",
            "unit_of_measurement": "°C",
            "icon": "mdi:chart-line",
        },
        "sensor.ml_feature_importance": {
            "unique_id": "ml_heating_feature_importance",
            "friendly_name": "ML Feature Importance",
            "unit_of_measurement": "%",
            "icon": "mdi:format-list-numbered",
        },
        "sensor.ml_heating_state": {
            "unique_id": "ml_heating_state",
            "friendly_name": "ML Heating State",
            "unit_of_measurement": "state",
            "icon": "mdi:information-outline",
        },
        "sensor.ml_heating_learning": {
            "unique_id": "ml_heating_learning",
            "friendly_name": "ML Adaptive Learning Metrics",
            "unit_of_measurement": "metrics",
            "icon": "mdi:brain",
        },
        "sensor.ml_prediction_accuracy": {
            "unique_id": "ml_prediction_accuracy", 
            "friendly_name": "ML Prediction Accuracy",
            "unit_of_measurement": "%",
            "icon": "mdi:target",
        },
        "sensor.ml_heating_cop_realtime": {
            "unique_id": "ml_heating_cop_realtime",
            "friendly_name": "ML Heating COP (Realtime)",
            "unit_of_measurement": "COP",
            "icon": "mdi:heat-pump",
            "device_class": "power_factor",
        },
        "sensor.ml_heating_thermal_power": {
            "unique_id": "ml_heating_thermal_power",
            "friendly_name": "ML Heating Thermal Power",
            "unit_of_measurement": "kW",
            "icon": "mdi:flash",
            "device_class": "power",
        },
        "sensor.ml_heating_features": {
            "unique_id": "ml_heating_features",
            "friendly_name": "ML Heating Features",
            "unit_of_measurement": "°C",
            "device_class": "temperature",
            "icon": "mdi:feature-search-outline",
        },
        "sensor.ml_heating_price_level": {
            "unique_id": "ml_heating_price_level",
            "friendly_name": "ML Heating Price Level",
            "icon": "mdi:currency-eur",
        },
    }

    def _default_unique_id(sensor_id: str) -> str:
        return sensor_id.replace(".", "_")

    def _default_friendly_name(sensor_id: str) -> str:
        return sensor_id.split(".", 1)[-1].replace("_", " ").title()

    attributes = base_attributes.copy()
    attributes.update(sensor_specific_attributes.get(base_entity_id, {}))

    if "unique_id" not in attributes:
        attributes["unique_id"] = _default_unique_id(base_entity_id)
    if "friendly_name" not in attributes:
        attributes["friendly_name"] = _default_friendly_name(base_entity_id)

    if effective_entity_id != base_entity_id:
        if not attributes["unique_id"].endswith("_shadow"):
            attributes["unique_id"] = f"{attributes['unique_id']}_shadow"
        if not attributes["friendly_name"].endswith(" Shadow"):
            attributes["friendly_name"] = (
                f"{attributes['friendly_name']} Shadow"
            )

    return attributes


def create_ha_client():
    """
    Factory function to create an instance of the HAClient.

    It simplifies the creation of a client by reading the required URL and
    token directly from the application's configuration module.
    """
    return HAClient(config.HASS_URL, config.HASS_TOKEN)
