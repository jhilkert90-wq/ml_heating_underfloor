"""
Heat Source Channel Architecture for Decomposed Learning.

This module implements Phase 2-4 of the decomposed heat-source learning plan.
Each heat source (heat pump, solar/PV, fireplace, TV/electronics) gets its own
independent learning channel with isolated parameters and history, preventing
cross-contamination of learned parameters.

Architecture:
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │  HP Channel   │   │ Solar Channel │   │  FP Channel   │   │  TV Channel   │
  │  (controlled) │   │ (forecast)   │   │ (observed)    │   │ (minor)       │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                   │                   │                   │
         └───────────┬───────┘───────────────────┘───────────────────┘
                     ▼
              Q_total = Q_hp + Q_solar + Q_fireplace + Q_tv
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

try:
    from . import config
except ImportError:
    import config

try:
    from .thermal_parameters import thermal_params
except ImportError:
    from thermal_parameters import thermal_params

try:
    from .thermal_config import ThermalParameterConfig
except ImportError:
    from thermal_config import ThermalParameterConfig

try:
    from .thermal_constants import PhysicsConstants
except ImportError:
    from thermal_constants import PhysicsConstants

logger = logging.getLogger(__name__)


def _get_min_records_for_learning() -> int:
    """Channel window aligned with the main model's RECENT_ERRORS_WINDOW."""
    return getattr(config, "RECENT_ERRORS_WINDOW", 10)


def _get_channel_learning_rate() -> float:
    """Channel base learning rate aligned with the main model's config."""
    return getattr(config, "ADAPTIVE_LEARNING_RATE", 0.01)


def _get_learning_dead_zone() -> float:
    """Dead-zone threshold aligned with PhysicsConstants."""
    return PhysicsConstants.LEARNING_DEAD_ZONE


def _get_parameter_default(param_name: str, fallback: float) -> float:
    """Read channel startup defaults from the unified parameter manager."""
    try:
        return float(thermal_params.get(param_name))
    except (KeyError, TypeError, ValueError):
        return fallback


def _clip_to_parameter_bounds(
    param_name: str, value: float, fallback_bounds
) -> float:
    """Clamp channel parameters to the canonical runtime bounds."""
    try:
        lower, upper = ThermalParameterConfig.get_bounds(param_name)
    except KeyError:
        lower, upper = fallback_bounds
    return max(lower, min(upper, value))


def _to_float(value, default: float = 0.0) -> float:
    """Convert possibly scalar-like values to float."""
    try:
        if value is None:
            return default
        if isinstance(value, list):
            value = value[-1] if value else default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_fireplace_active(context: Dict) -> bool:
    return _to_float(context.get("fireplace_on", 0.0)) > 0.0


def _is_tv_active(context: Dict) -> bool:
    return _to_float(context.get("tv_on", 0.0)) > 0.0


def _get_pv_power(context: Dict) -> float:
    if "pv_power_current" in context:
        return _to_float(
            context.get("pv_power_current", 0.0),
            _to_float(context.get("pv_power", 0.0)),
        )
    return _to_float(context.get("pv_power", 0.0))


def _is_pv_active(context: Dict) -> bool:
    """PV is 'active' when either the current or smoothed power exceeds the
    threshold.  Using the smoothed value captures the solar thermal lag:
    sun-warmed surfaces continue heating the room after PV drops below
    the threshold (e.g. late afternoon), so PV should still own learning."""
    pv_current = _get_pv_power(context)
    pv_smoothed = _to_float(context.get("pv_power", 0.0))
    return max(pv_current, pv_smoothed) > SolarChannel.PV_LEARNING_THRESHOLD


def _is_heat_pump_active(context: Dict) -> bool:
    explicit_state = context.get("heat_pump_active")
    if explicit_state is not None:
        return bool(explicit_state)

    thermal_power = _to_float(context.get("thermal_power", 0.0))
    if thermal_power > 0.05:
        return True

    delta_t = _to_float(context.get("delta_t", 0.0))
    if delta_t > 0.5:
        return True

    outlet = _to_float(context.get("outlet_temp", 0.0))
    indoor = _to_float(context.get("current_indoor", 0.0))
    inlet = _to_float(context.get("inlet_temp", indoor))
    return outlet > indoor + 1.0 and outlet > inlet + 0.5


def _average_recent_error(history: List[Dict]) -> Optional[float]:
    min_records = _get_min_records_for_learning()
    if len(history) < min_records:
        return None
    recent = history[-min_records:]
    return sum(r["error"] for r in recent) / len(recent)


def _average_recent_raw_error(history: List[Dict]) -> Optional[float]:
    """Average *raw* prediction error for dead-zone gating.

    In mixed-source attribution the per-channel ``error`` is only a
    fraction of the total prediction error.  The dead zone should gate
    on the total signal strength (raw error), not the attributed slice,
    so that small-share channels (e.g. PV during HP+PV overlap) are not
    permanently silenced.
    """
    min_records = _get_min_records_for_learning()
    if len(history) < min_records:
        return None
    recent = history[-min_records:]
    values: List[float] = []
    for r in recent:
        ctx = r.get("context", {})
        raw = ctx.get("raw_prediction_error")
        if raw is not None:
            values.append(float(raw))
        else:
            values.append(r["error"])
    return sum(values) / len(values) if values else None


def _snapshot_parameter_changes(
    before: Dict[str, float], after: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """Return structured before/after/delta snapshots for changed parameters."""
    changes: Dict[str, Dict[str, float]] = {}
    for name in sorted(set(before) | set(after)):
        old_value = before.get(name)
        new_value = after.get(name)
        if old_value is None or new_value is None:
            continue

        try:
            old_float = float(old_value)
            new_float = float(new_value)
        except (TypeError, ValueError):
            continue

        delta = new_float - old_float
        if abs(delta) <= 1e-12:
            continue

        changes[name] = {
            "before": old_float,
            "after": new_float,
            "delta": delta,
        }
    return changes


def _format_logged_value(value: float) -> str:
    """Format logged numeric values compactly while preserving small deltas."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _get_channel_log_prefix(channel_name: str) -> str:
    prefixes = {
        "heat_pump": "♨️ heat_pump",
        "pv": "☀️ pv",
        "fireplace": "🔥 fireplace",
        "tv": "📺 tv",
    }
    return prefixes.get(channel_name, channel_name)


class HeatSourceChannel(ABC):
    """
    Abstract base class for a heat source learning channel.

    Each channel independently tracks its own prediction history and
    learnable parameters, preventing cross-contamination between sources.
    Channels learn from their own history via ``_learn_from_recent()``,
    which is called automatically after each ``record_learning()`` call
    once enough observations have accumulated.
    """

    def __init__(self, name: str):
        self.name = name
        self.history: List[Dict] = []
        self._max_history = 200

    @abstractmethod
    def estimate_heat_contribution(self, context: Dict) -> float:
        """Estimate current heat contribution in kW."""

    @abstractmethod
    def estimate_decay_contribution(
        self, time_since_off: float, context: Dict
    ) -> float:
        """Estimate residual heat contribution after source turned off (kW)."""

    @abstractmethod
    def get_learnable_parameters(self) -> Dict[str, float]:
        """Return current learnable parameters."""

    def get_state_parameters(self) -> Dict[str, float]:
        """Return the full persisted/exported channel parameter snapshot."""
        return self.get_learnable_parameters()

    @abstractmethod
    def apply_gradient_update(
        self, gradients: Dict[str, float], learning_rate: float
    ) -> None:
        """Apply gradient-based parameter update."""

    def record_learning(self, error: float, context: Dict) -> Optional[Dict[str, Any]]:
        """Record a learning observation and return a structured update event."""
        parameters_before = self.get_state_parameters().copy()
        record = {
            "error": error,
            "context": context.copy(),
            "parameters": parameters_before,
            "parameters_before": parameters_before.copy(),
        }
        self.history.append(record)
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]
        # Trigger independent gradient learning
        self._learn_from_recent()
        parameters_after = self.get_state_parameters().copy()
        changes = _snapshot_parameter_changes(parameters_before, parameters_after)
        record["parameters_after"] = parameters_after
        record["changes"] = changes

        if not changes:
            return None

        return {
            "record_type": "channel_update",
            "channel": self.name,
            "error": error,
            "learning_rate": _get_channel_learning_rate(),
            "parameters_before": parameters_before,
            "parameters_after": parameters_after,
            "changes": changes,
        }

    def _learn_from_recent(self) -> None:
        """Compute gradients from recent observations and update parameters.

        Default is a no-op; subclasses override for channel-specific
        gradient logic.
        """

    def predict_future_contribution(
        self, horizon_hours: float, context: Dict
    ) -> List[float]:
        """Predict future heat contribution per 10-min step.

        Default implementation returns constant current contribution.
        Subclasses (e.g. SolarChannel) override for forecast-aware behaviour.
        """
        steps = max(1, int(horizon_hours * 6))
        current = self.estimate_heat_contribution(context)
        return [current] * steps


class HeatPumpChannel(HeatSourceChannel):
    """
    Heat pump channel — the primary controllable heat source.

    Parameters learned only when fireplace=OFF and PV < 100 W (clean signal).
    Wraps the existing slab model (Estrich) from ThermalEquilibriumModel.
    """

    def __init__(self):
        super().__init__("heat_pump")
        self.thermal_time_constant = _get_parameter_default(
            "thermal_time_constant", 4.0
        )
        self.heat_loss_coefficient = _get_parameter_default(
            "heat_loss_coefficient", 0.15
        )
        self.outlet_effectiveness = _get_parameter_default(
            "outlet_effectiveness", 0.17
        )
        self.slab_time_constant_hours = _get_parameter_default(
            "slab_time_constant_hours", 1.0
        )
        self.delta_t_floor = _get_parameter_default("delta_t_floor", 2.0)

    def estimate_heat_contribution(self, context: Dict) -> float:
        if not _is_heat_pump_active(context):
            return 0.0
        outlet = context.get("outlet_temp", 40.0)
        indoor = context.get("current_indoor", 21.0)
        return self.outlet_effectiveness * max(0.0, outlet - indoor)

    def estimate_decay_contribution(
        self, time_since_off: float, context: Dict
    ) -> float:
        indoor = context.get("current_indoor", 21.0)
        inlet = context.get("inlet_temp", indoor)
        if time_since_off <= 0:
            return 0.0
        alpha = 1.0 - math.exp(
            -time_since_off / self.slab_time_constant_hours
        )
        slab_temp = inlet + alpha * (indoor - inlet)
        return max(0.0, self.outlet_effectiveness * (slab_temp - indoor))

    def get_learnable_parameters(self) -> Dict[str, float]:
        return {
            "thermal_time_constant": self.thermal_time_constant,
            "heat_loss_coefficient": self.heat_loss_coefficient,
            "outlet_effectiveness": self.outlet_effectiveness,
            "slab_time_constant_hours": self.slab_time_constant_hours,
            "delta_t_floor": self.delta_t_floor,
        }

    def apply_gradient_update(
        self, gradients: Dict[str, float], learning_rate: float
    ) -> None:
        if "thermal_time_constant" in gradients:
            delta = gradients["thermal_time_constant"] * learning_rate
            self.thermal_time_constant = _clip_to_parameter_bounds(
                "thermal_time_constant",
                self.thermal_time_constant + max(-0.2, min(0.2, delta)),
                (3.0, 100.0),
            )
        if "heat_loss_coefficient" in gradients:
            delta = gradients["heat_loss_coefficient"] * learning_rate
            self.heat_loss_coefficient = _clip_to_parameter_bounds(
                "heat_loss_coefficient",
                self.heat_loss_coefficient + max(-0.01, min(0.01, delta)),
                (0.01, 1.2),
            )
        if "outlet_effectiveness" in gradients:
            delta = gradients["outlet_effectiveness"] * learning_rate
            self.outlet_effectiveness = _clip_to_parameter_bounds(
                "outlet_effectiveness",
                self.outlet_effectiveness + max(-0.005, min(0.005, delta)),
                (0.3, 2.0),
            )
        if "slab_time_constant_hours" in gradients:
            delta = gradients["slab_time_constant_hours"] * learning_rate
            self.slab_time_constant_hours = _clip_to_parameter_bounds(
                "slab_time_constant_hours",
                self.slab_time_constant_hours + max(-0.05, min(0.05, delta)),
                (0.5, 3.0),
            )
        if "delta_t_floor" in gradients:
            delta = gradients["delta_t_floor"] * learning_rate
            self.delta_t_floor = _clip_to_parameter_bounds(
                "delta_t_floor",
                self.delta_t_floor + max(-0.2, min(0.2, delta)),
                (0.0, 10.0),
            )

    def _learn_from_recent(self) -> None:
        min_records = _get_min_records_for_learning()
        recent = self.history[-min_records:]
        if any(_is_fireplace_active(record.get("context", {})) for record in recent):
            logger.debug(
                "♨️ HeatPumpChannel skipping self-learning: fireplace-contaminated history"
            )
            return

        avg_error = _average_recent_error(self.history)
        raw_avg = _average_recent_raw_error(self.history)
        if avg_error is None or raw_avg is None or abs(raw_avg) < _get_learning_dead_zone():
            return

        delta_t_samples = [
            _to_float(record.get("context", {}).get("delta_t", 0.0))
            for record in recent
            if _to_float(record.get("context", {}).get("delta_t", 0.0)) > 0.5
        ]
        avg_delta_t = (
            sum(delta_t_samples) / len(delta_t_samples)
            if delta_t_samples
            else self.delta_t_floor
        )

        gradients: Dict[str, float] = {
            "thermal_time_constant": -avg_error * 0.5,
            "heat_loss_coefficient": -avg_error * 0.1,
            "outlet_effectiveness": avg_error,
            "slab_time_constant_hours": avg_error * 0.2,
            "delta_t_floor": avg_delta_t - self.delta_t_floor,
        }
        self.apply_gradient_update(gradients, _get_channel_learning_rate())
        logger.debug(
            "♨️ HeatPumpChannel self-learned: avg_error=%.3f, "
            "tau=%.2fh, hlc=%.3f, outlet_effectiveness=%.3f, "
            "slab_tau=%.2fh, delta_t_floor=%.2f",
            avg_error,
            self.thermal_time_constant,
            self.heat_loss_coefficient,
            self.outlet_effectiveness,
            self.slab_time_constant_hours,
            self.delta_t_floor,
        )


class SolarChannel(HeatSourceChannel):
    """
    Solar/PV channel — uncontrollable, forecast-aware.

    Parameters learned only from daytime cycles (PV > threshold).
    Decay: small τ (default 0.5 h ≈ 30 min) models residual heat from
    sun-warmed surfaces (floors, walls near windows) after PV drops.
    """

    # PV must exceed this threshold for the channel to learn.
    PV_LEARNING_THRESHOLD = float(
        getattr(config, "PV_LEARNING_THRESHOLD", 50.0)
    )  # Watts

    def __init__(self):
        super().__init__("pv")
        self.pv_heat_weight = _get_parameter_default("pv_heat_weight", 0.002)
        self.solar_lag_minutes = _get_parameter_default(
            "solar_lag_minutes", 45.0
        )
        self.cloud_factor_exponent = _get_parameter_default(
            "cloud_factor_exponent", 1.0
        )
        # Thermal mass of sun-warmed surfaces (floors, walls near windows).
        # After PV drops, these surfaces continue emitting heat with this τ.
        self.solar_decay_tau_hours = _get_parameter_default(
            "solar_decay_tau_hours", 0.5
        )

    def estimate_heat_contribution(self, context: Dict) -> float:
        pv_power = _get_pv_power(context)
        cloud = context.get("avg_cloud_cover", 50.0)
        cloud_factor = self._cloud_factor(cloud)
        return float(pv_power) * self.pv_heat_weight * cloud_factor

    def _cloud_factor(self, cloud_cover_pct: float) -> float:
        """Cloud adjustment aligned with the prediction model."""
        if not getattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False):
            return 1.0
        cloud_pct = max(0.0, min(100.0, cloud_cover_pct))
        min_factor = float(
            getattr(config, "CLOUD_CORRECTION_MIN_FACTOR", 0.1)
        )
        min_factor = max(0.0, min(1.0, min_factor))
        return max(
            min_factor,
            1.0 - (cloud_pct / 100.0) ** self.cloud_factor_exponent,
        )

    def estimate_decay_contribution(
        self, time_since_off: float, context: Dict
    ) -> float:
        """Residual heat from sun-warmed surfaces after PV drops."""
        if time_since_off <= 0 or self.solar_decay_tau_hours <= 0:
            return 0.0
        last_pv = context.get("last_pv_power", 0)
        if not last_pv or float(last_pv) <= 0:
            return 0.0
        peak_heat = float(last_pv) * self.pv_heat_weight
        return peak_heat * math.exp(
            -time_since_off / self.solar_decay_tau_hours
        )

    def get_learnable_parameters(self) -> Dict[str, float]:
        return {
            "pv_heat_weight": self.pv_heat_weight,
            "solar_lag_minutes": self.solar_lag_minutes,
            "cloud_factor_exponent": self.cloud_factor_exponent,
            "solar_decay_tau_hours": self.solar_decay_tau_hours,
        }

    def apply_gradient_update(
        self, gradients: Dict[str, float], learning_rate: float
    ) -> None:
        if "pv_heat_weight" in gradients:
            delta = gradients["pv_heat_weight"] * learning_rate
            self.pv_heat_weight = _clip_to_parameter_bounds(
                "pv_heat_weight",
                self.pv_heat_weight + max(-0.0002, min(0.0002, delta)),
                (0.0001, 0.005),
            )
        if "solar_lag_minutes" in gradients:
            delta = gradients["solar_lag_minutes"] * learning_rate
            self.solar_lag_minutes = _clip_to_parameter_bounds(
                "solar_lag_minutes",
                self.solar_lag_minutes + max(-5.0, min(5.0, delta)),
                (0.0, 180.0),
            )
        if "cloud_factor_exponent" in gradients:
            delta = gradients["cloud_factor_exponent"] * learning_rate
            self.cloud_factor_exponent = _clip_to_parameter_bounds(
                "cloud_factor_exponent",
                self.cloud_factor_exponent + max(-0.05, min(0.05, delta)),
                (0.1, 3.0),
            )
        if "solar_decay_tau_hours" in gradients:
            delta = gradients["solar_decay_tau_hours"] * learning_rate
            self.solar_decay_tau_hours = _clip_to_parameter_bounds(
                "solar_decay_tau_hours",
                self.solar_decay_tau_hours + max(-0.05, min(0.05, delta)),
                (0.0, 3.0),
            )

    def predict_future_contribution(
        self, horizon_hours: float, context: Dict
    ) -> List[float]:
        """Use PV forecast to predict future solar heat, smoothed by decay τ.

        When PV increases, heat rises immediately (sunshine warms fast).
        When PV decreases, residual heat from sun-warmed surfaces decays
        exponentially with ``solar_decay_tau_hours``.
        """
        pv_forecast = context.get("pv_forecast")
        cloud = context.get("avg_cloud_cover", 50.0)
        cloud_factor = self._cloud_factor(cloud)
        steps = max(1, int(horizon_hours * 6))
        step_hours = 1.0 / 6.0  # 10 minutes

        if not pv_forecast:
            current = self.estimate_heat_contribution(context)
            return [current] * steps

        # Compute raw heat per step from forecast
        raw_heat: List[float] = []
        for i in range(steps):
            hour_idx = min(i // 6, len(pv_forecast) - 1)
            pv_val = float(pv_forecast[hour_idx])
            raw_heat.append(pv_val * self.pv_heat_weight * cloud_factor)

        # Apply exponential decay smoothing when PV drops.
        # Rising PV → immediate heat increase (no lag on increase).
        # Falling PV → smooth decay from sun-warmed thermal mass.
        if self.solar_decay_tau_hours > 0:
            decay_factor = math.exp(
                -step_hours / self.solar_decay_tau_hours
            )
        else:
            decay_factor = 0.0

        smoothed: List[float] = []
        prev = raw_heat[0]
        for direct in raw_heat:
            if direct >= prev:
                smoothed.append(direct)
            else:
                decayed = prev * decay_factor
                smoothed.append(max(direct, decayed))
            prev = smoothed[-1]
        return smoothed

    def _learn_from_recent(self) -> None:
        avg_error = _average_recent_error(self.history)
        raw_avg = _average_recent_raw_error(self.history)
        if avg_error is None or raw_avg is None or abs(raw_avg) < _get_learning_dead_zone():
            return

        min_records = _get_min_records_for_learning()
        recent = self.history[-min_records:]
        cloud_samples = [
            _to_float(record.get("context", {}).get("avg_cloud_cover", 0.0))
            for record in recent
        ]
        avg_cloud_cover = (
            sum(cloud_samples) / len(cloud_samples)
            if cloud_samples
            else 0.0
        )

        rising_pv_samples = []
        for record in recent:
            context = record.get("context", {})
            pv_now = _get_pv_power(context)
            pv_history = context.get("pv_power_history")
            if not isinstance(pv_history, list) or len(pv_history) < 2:
                continue
            trailing_average = sum(float(value) for value in pv_history) / len(
                pv_history
            )
            rising_pv_samples.append(max(0.0, pv_now - trailing_average))

        avg_rising_pv = (
            sum(rising_pv_samples) / len(rising_pv_samples)
            if rising_pv_samples
            else 0.0
        )

        gradients: Dict[str, float] = {
            "pv_heat_weight": avg_error * 0.1,
            "solar_decay_tau_hours": avg_error * 0.05,
        }
        if (avg_cloud_cover > 0.0
                and getattr(config, "CLOUD_COVER_CORRECTION_ENABLED", False)):
            gradients["cloud_factor_exponent"] = avg_error * (
                avg_cloud_cover / 100.0
            )
        if avg_rising_pv > 0.0:
            gradients["solar_lag_minutes"] = -avg_error * (
                avg_rising_pv / max(1.0, SolarChannel.PV_LEARNING_THRESHOLD)
            )
        self.apply_gradient_update(gradients, _get_channel_learning_rate())
        logger.debug(
            "☀️ SolarChannel self-learned: avg_error=%.3f, "
            "pv_heat_weight=%.5f, solar_lag=%.1fmin, cloud_exp=%.2f, "
            "solar_decay_tau=%.2fh",
            avg_error,
            self.pv_heat_weight,
            self.solar_lag_minutes,
            self.cloud_factor_exponent,
            self.solar_decay_tau_hours,
        )


class FireplaceChannel(HeatSourceChannel):
    """
    Fireplace channel — uncontrollable, observed via binary sensor.

    Learns independently from prediction errors during fireplace-active
    periods via its own gradient-based parameter updates (no external
    learning module dependency).

    Decay: exponential after fireplace turned off (τ ~ 30-60 min).
    Room spread: living room heat distributes to house average with delay.
    """

    def __init__(self):
        super().__init__("fireplace")
        self.fp_heat_output_kw = _get_parameter_default(
            "fp_heat_output_kw", 5.0
        )
        self.fp_decay_time_constant = _get_parameter_default(
            "fp_decay_time_constant", 0.75
        )
        self.room_spread_delay_minutes = _get_parameter_default(
            "room_spread_delay_minutes", 30.0
        )

    def estimate_heat_contribution(self, context: Dict) -> float:
        fireplace_on = context.get("fireplace_on", 0)
        if not fireplace_on:
            return 0.0
        return self.fp_heat_output_kw

    def estimate_decay_contribution(
        self, time_since_off: float, context: Dict
    ) -> float:
        if time_since_off <= 0:
            return 0.0
        return self.fp_heat_output_kw * math.exp(
            -time_since_off / self.fp_decay_time_constant
        )

    def get_learnable_parameters(self) -> Dict[str, float]:
        return {
            "fp_heat_output_kw": self.fp_heat_output_kw,
        }

    def get_state_parameters(self) -> Dict[str, float]:
        return {
            "fp_heat_output_kw": self.fp_heat_output_kw,
            "fp_decay_time_constant": self.fp_decay_time_constant,
            "room_spread_delay_minutes": self.room_spread_delay_minutes,
        }

    def apply_gradient_update(
        self, gradients: Dict[str, float], learning_rate: float
    ) -> None:
        if "fp_heat_output_kw" in gradients:
            delta = gradients["fp_heat_output_kw"] * learning_rate
            self.fp_heat_output_kw += max(-0.5, min(0.5, delta))
            self.fp_heat_output_kw = max(
                0.5, min(15.0, self.fp_heat_output_kw)
            )
        if "fp_decay_time_constant" in gradients:
            delta = gradients["fp_decay_time_constant"] * learning_rate
            self.fp_decay_time_constant += max(-0.1, min(0.1, delta))
            self.fp_decay_time_constant = max(
                0.1, min(2.0, self.fp_decay_time_constant)
            )

    def _learn_from_recent(self) -> None:
        """Independent gradient learning from fireplace-active observations.

        Computes a simple gradient for ``fp_heat_output_kw`` from the
        average recent prediction error:
        - positive error (actual > predicted) → fireplace provides more
          heat than modelled → increase ``fp_heat_output_kw``
        - negative error (actual < predicted) → decrease
        """
        min_records = _get_min_records_for_learning()
        if len(self.history) < min_records:
            return

        recent = self.history[-min_records:]
        avg_error = sum(r["error"] for r in recent) / len(recent)
        raw_avg = _average_recent_raw_error(self.history)

        if raw_avg is None or abs(raw_avg) < _get_learning_dead_zone():
            return

        # Positive error = actual hotter → underestimated FP heat
        gradients: Dict[str, float] = {
            "fp_heat_output_kw": avg_error,
        }
        self.apply_gradient_update(gradients, _get_channel_learning_rate())
        logger.debug(
            "🔥 FireplaceChannel self-learned: avg_error=%.3f, "
            "fp_heat_output_kw=%.2f",
            avg_error,
            self.fp_heat_output_kw,
        )


class TVChannel(HeatSourceChannel):
    """
    TV/Electronics channel — minor heat source (~0.25 kW).

    Simple additive term; not worth its own complex model.
    """

    def __init__(self):
        super().__init__("tv")
        self.tv_heat_weight = _get_parameter_default("tv_heat_weight", 0.2)

    def estimate_heat_contribution(self, context: Dict) -> float:
        tv_on = context.get("tv_on", 0)
        if not tv_on:
            return 0.0
        return self.tv_heat_weight

    def estimate_decay_contribution(
        self, time_since_off: float, context: Dict
    ) -> float:
        return 0.0

    def get_learnable_parameters(self) -> Dict[str, float]:
        return {"tv_heat_weight": self.tv_heat_weight}

    def apply_gradient_update(
        self, gradients: Dict[str, float], learning_rate: float
    ) -> None:
        if "tv_heat_weight" in gradients:
            delta = gradients["tv_heat_weight"] * learning_rate
            self.tv_heat_weight += max(-0.05, min(0.05, delta))
            self.tv_heat_weight = max(0.0, min(1.0, self.tv_heat_weight))

    def _learn_from_recent(self) -> None:
        avg_error = _average_recent_error(self.history)
        raw_avg = _average_recent_raw_error(self.history)
        if avg_error is None or raw_avg is None or abs(raw_avg) < _get_learning_dead_zone():
            return

        gradients: Dict[str, float] = {"tv_heat_weight": avg_error * 0.1}
        self.apply_gradient_update(gradients, _get_channel_learning_rate())
        logger.debug(
            "📺 TVChannel self-learned: avg_error=%.3f, tv_heat_weight=%.3f",
            avg_error,
            self.tv_heat_weight,
        )


class HeatSourceChannelOrchestrator:
    """
    Orchestrates all heat source channels.

    Combines contributions from all channels for total heat prediction and
    routes learning updates to the correct channel based on which sources
    are currently active, preventing cross-contamination.
    """

    def __init__(self):
        self.channels: Dict[str, HeatSourceChannel] = {
            "heat_pump": HeatPumpChannel(),
            "pv": SolarChannel(),
            "fireplace": FireplaceChannel(),
            "tv": TVChannel(),
        }
        # Track fireplace state transitions for post-off decay routing.
        # When FP turns off, we keep routing learning to the FP channel
        # for a decay window (3×τ) to prevent residual FP heat from being
        # misattributed to the HP channel.
        self._fireplace_off_cycle_count: int = 0
        self._fireplace_was_on: bool = False
        # Track PV state transitions for post-off decay routing.
        # When PV drops below threshold, we keep routing learning to the
        # PV channel for a decay window (thermal_time_constant × multiplier)
        # to prevent residual room heat from PV being misattributed to HP.
        self._pv_off_cycle_count: int = 0
        self._pv_was_active: bool = False

    def sync_from_model_parameters(self, parameters: Dict[str, float]) -> None:
        """Seed channel parameters from the current thermal-model state."""
        heat_pump = self.channels["heat_pump"]
        assert isinstance(heat_pump, HeatPumpChannel)
        heat_pump.thermal_time_constant = parameters.get(
            "thermal_time_constant", heat_pump.thermal_time_constant
        )
        heat_pump.heat_loss_coefficient = parameters.get(
            "heat_loss_coefficient", heat_pump.heat_loss_coefficient
        )
        heat_pump.outlet_effectiveness = parameters.get(
            "outlet_effectiveness", heat_pump.outlet_effectiveness
        )
        heat_pump.slab_time_constant_hours = parameters.get(
            "slab_time_constant_hours", heat_pump.slab_time_constant_hours
        )
        heat_pump.delta_t_floor = parameters.get(
            "delta_t_floor", heat_pump.delta_t_floor
        )

        solar = self.channels["pv"]
        assert isinstance(solar, SolarChannel)
        solar.pv_heat_weight = parameters.get(
            "pv_heat_weight", solar.pv_heat_weight
        )
        solar.solar_lag_minutes = parameters.get(
            "solar_lag_minutes", solar.solar_lag_minutes
        )

        fireplace = self.channels["fireplace"]
        assert isinstance(fireplace, FireplaceChannel)
        fireplace.fp_heat_output_kw = parameters.get(
            "fireplace_heat_weight", fireplace.fp_heat_output_kw
        )
        fireplace.fp_decay_time_constant = parameters.get(
            "fp_decay_time_constant", fireplace.fp_decay_time_constant
        )
        fireplace.room_spread_delay_minutes = parameters.get(
            "room_spread_delay_minutes", fireplace.room_spread_delay_minutes
        )

        tv = self.channels["tv"]
        assert isinstance(tv, TVChannel)
        tv.tv_heat_weight = parameters.get(
            "tv_heat_weight", tv.tv_heat_weight
        )

        # Channel-specific params that may be persisted from batch calibration
        heat_pump.delta_t_floor = parameters.get(
            "delta_t_floor", heat_pump.delta_t_floor
        )
        solar.cloud_factor_exponent = parameters.get(
            "cloud_factor_exponent", solar.cloud_factor_exponent
        )
        solar.solar_decay_tau_hours = parameters.get(
            "solar_decay_tau_hours", solar.solar_decay_tau_hours
        )

    def export_model_parameters(self) -> Dict[str, float]:
        """Return channel parameters in thermal-model shape."""
        heat_pump = self.channels["heat_pump"]
        solar = self.channels["pv"]
        fireplace = self.channels["fireplace"]
        tv = self.channels["tv"]
        return {
            "thermal_time_constant": heat_pump.thermal_time_constant,
            "heat_loss_coefficient": heat_pump.heat_loss_coefficient,
            "outlet_effectiveness": heat_pump.outlet_effectiveness,
            "slab_time_constant_hours": heat_pump.slab_time_constant_hours,
            "delta_t_floor": heat_pump.delta_t_floor,
            "pv_heat_weight": solar.pv_heat_weight,
            "solar_lag_minutes": solar.solar_lag_minutes,
            "cloud_factor_exponent": solar.cloud_factor_exponent,
            "solar_decay_tau_hours": solar.solar_decay_tau_hours,
            "fireplace_heat_weight": fireplace.fp_heat_output_kw,
            "fp_decay_time_constant": fireplace.fp_decay_time_constant,
            "room_spread_delay_minutes": fireplace.room_spread_delay_minutes,
            "tv_heat_weight": tv.tv_heat_weight,
        }

    def total_heat(self, context: Dict) -> float:
        """Calculate total heat contribution from all active sources."""
        return sum(
            ch.estimate_heat_contribution(context)
            for ch in self.channels.values()
        )

    def _get_active_contributions(self, context: Dict) -> Dict[str, float]:
        """Return positive heat contributions for currently active channels."""
        contributions: Dict[str, float] = {}
        for name, channel in self.channels.items():
            contribution = channel.estimate_heat_contribution(context)
            if contribution > 0:
                contributions[name] = contribution
        return contributions

    def _build_learning_context(
        self,
        context: Dict,
        attributed_error: float,
        active_contributions: Dict[str, float],
        attribution_applied: bool,
        heat_pump_frozen_by_fireplace: bool,
        heat_pump_frozen_by_pv_decay: bool = False,
    ) -> Dict:
        """Build the context used by per-channel learning updates.

        `heat_pump_frozen_by_pv_decay` is intentionally optional so existing
        callers that do not compute or track PV-decay freeze state continue to
        behave as before. In those paths, omitting the argument means "no
        heat-pump freeze due to PV decay" and the stored learning-context flag
        defaults to ``False``.
        """
        learning_context = context.copy()
        learning_context["raw_prediction_error"] = context.get(
            "raw_prediction_error", attributed_error
        )
        learning_context["attributed_error"] = attributed_error
        learning_context["attribution_applied"] = attribution_applied
        learning_context["active_contributions"] = active_contributions.copy()
        learning_context[
            "heat_pump_frozen_by_fireplace"
        ] = heat_pump_frozen_by_fireplace
        learning_context[
            "heat_pump_frozen_by_pv_decay"
        ] = heat_pump_frozen_by_pv_decay
        return learning_context

    def _record_channel_learning(
        self,
        channel_name: str,
        error: float,
        context: Dict,
        active_contributions: Dict[str, float],
        attribution_applied: bool,
        heat_pump_frozen_by_fireplace: bool,
        update_events: List[Dict[str, Any]],
    ) -> None:
        """Record learning and collect a structured update event when params changed."""
        update_event = self.channels[channel_name].record_learning(error, context)
        if update_event is None:
            logger.debug(
                "%s received observation (error=%+.4f) — no parameter delta",
                _get_channel_log_prefix(channel_name),
                error,
            )
            return

        update_event.update(
            {
                "raw_prediction_error": context.get(
                    "raw_prediction_error", error
                ),
                "attributed_error": error,
                "attribution_applied": attribution_applied,
                "active_contributions": active_contributions.copy(),
                "heat_pump_frozen_by_fireplace": heat_pump_frozen_by_fireplace,
            }
        )
        update_events.append(update_event)

    def _log_channel_parameter_updates(
        self, update_events: List[Dict[str, Any]]
    ) -> None:
        """Emit one info log line per active channel with real parameter deltas."""
        for event in update_events:
            changes = event.get("changes", {})
            if not changes:
                continue

            formatted_changes = []
            for param_name, change in changes.items():
                formatted_changes.append(
                    (
                        f"{param_name}: "
                        f"{_format_logged_value(change['before'])}→"
                        f"{_format_logged_value(change['after'])} "
                        f"(Δ{change['delta']:+.6f})"
                    )
                )

            logger.info(
                "%s parameter update: attributed_error=%+.6f | %s",
                _get_channel_log_prefix(event.get("channel", "")),
                float(event.get("attributed_error", 0.0)),
                " | ".join(formatted_changes),
            )

    def route_learning(self, error: float, context: Dict) -> List[Dict[str, Any]]:
        """
        Route a learning observation to the appropriate channel.

        Isolation rules (Phase 3 — channel-isolated gradient descent):
        - Fireplace ON  → only fireplace channel learns
        - PV > threshold → only PV channel learns
        - TV ON          → only TV channel learns
        - Otherwise      → only heat pump channel learns

        When multiple external sources are active, each active external
        source channel gets the learning record.  HP never learns when
        any external source is active (to keep OE/HLC clean).

        Each channel's ``record_learning`` triggers its own independent
        ``_learn_from_recent()`` gradient update.
        """
        update_events: List[Dict[str, Any]] = []
        fireplace_on = _is_fireplace_active(context)
        pv_active = _is_pv_active(context)
        tv_on = _is_tv_active(context)
        heat_pump_active = _is_heat_pump_active(context)

        # --- Fireplace post-off decay window ---
        # Track FP state transitions.  When FP turns off, continue routing
        # learning to the FP channel for a decay window of 3×τ cycles so
        # that residual FP heat is not misattributed to HP.
        fp_channel = self.channels.get("fireplace")
        cycle_min = getattr(config, "CYCLE_INTERVAL_MINUTES", 10)
        fp_tau_h = getattr(fp_channel, "fp_decay_time_constant", 0.75) if fp_channel else 0.75
        fp_decay_window_cycles = max(1, int((fp_tau_h * 3 * 60) / cycle_min))

        if fireplace_on:
            self._fireplace_was_on = True
            self._fireplace_off_cycle_count = 0
        elif self._fireplace_was_on:
            self._fireplace_off_cycle_count += 1
            if self._fireplace_off_cycle_count > fp_decay_window_cycles:
                self._fireplace_was_on = False
                self._fireplace_off_cycle_count = 0

        fp_in_decay = (
            not fireplace_on
            and self._fireplace_was_on
            and self._fireplace_off_cycle_count <= fp_decay_window_cycles
        )

        # --- PV post-off decay window ---
        # When PV drops below threshold, continue routing learning to the
        # PV channel for a decay window of thermal_time_constant × multiplier
        # so that residual room heat from PV is not misattributed to HP.
        hp_channel = self.channels.get("heat_pump")
        ttc = getattr(hp_channel, "thermal_time_constant", 4.0) if hp_channel else 4.0
        pv_decay_multiplier = getattr(config, "PV_ROOM_DECAY_MULTIPLIER", 2.0)
        pv_decay_window_cycles = max(1, int((ttc * pv_decay_multiplier * 60) / cycle_min))

        if pv_active:
            self._pv_was_active = True
            self._pv_off_cycle_count = 0
        elif self._pv_was_active:
            self._pv_off_cycle_count += 1
            if self._pv_off_cycle_count > pv_decay_window_cycles:
                self._pv_was_active = False
                self._pv_off_cycle_count = 0

        pv_in_decay = (
            not pv_active
            and self._pv_was_active
            and self._pv_off_cycle_count <= pv_decay_window_cycles
        )

        # --- Early decay cancellation ---
        # If indoor temp has returned to near target, residual heat from
        # the external source has dissipated — cancel decay early so HP
        # can resume learning safely.
        current_indoor = _to_float(context.get("current_indoor", 0.0))
        target_temp = _to_float(context.get("target_temp", 22.0))
        decay_cancel_margin = getattr(config, "DECAY_CANCEL_MARGIN", 0.1)

        if fp_in_decay and current_indoor <= target_temp + decay_cancel_margin:
            logger.debug(
                "FP decay cancelled early: indoor %.2f <= target %.2f + %.1f",
                current_indoor, target_temp, decay_cancel_margin,
            )
            fp_in_decay = False
            self._fireplace_was_on = False
            self._fireplace_off_cycle_count = 0

        if pv_in_decay and current_indoor <= target_temp + decay_cancel_margin:
            logger.debug(
                "PV decay cancelled early: indoor %.2f <= target %.2f + %.1f",
                current_indoor, target_temp, decay_cancel_margin,
            )
            pv_in_decay = False
            self._pv_was_active = False
            self._pv_off_cycle_count = 0

        # When FP or PV is in decay window, treat it as active for routing
        any_external_active = fireplace_on or fp_in_decay or pv_active or pv_in_decay or tv_on

        raw_context = context.copy()
        raw_context["raw_prediction_error"] = error

        if not any_external_active:
            logger.debug(
                "Defaulting channel learning to heat_pump: "
                "pv_power_current=%.3f, pv_power=%.3f, thermal_power=%.3f, "
                "delta_t=%.3f, heat_pump_active=%s",
                _to_float(raw_context.get("pv_power_current", 0.0)),
                _to_float(raw_context.get("pv_power", 0.0)),
                _to_float(raw_context.get("thermal_power", 0.0)),
                _to_float(raw_context.get("delta_t", 0.0)),
                heat_pump_active,
            )
            active_contributions = {"heat_pump": 1.0}
            learning_context = self._build_learning_context(
                raw_context,
                attributed_error=error,
                active_contributions=active_contributions,
                attribution_applied=False,
                heat_pump_frozen_by_fireplace=False,
            )
            self._record_channel_learning(
                "heat_pump",
                error,
                learning_context,
                active_contributions,
                False,
                False,
                update_events,
            )
            self._log_channel_parameter_updates(update_events)
            self._log_routing_summary(
                ["heat_pump"], error, raw_context, update_events,
            )
            return update_events

        legacy_active_contributions = self._get_active_contributions(raw_context)
        if not getattr(config, "ENABLE_MIXED_SOURCE_ATTRIBUTION", False):
            if fireplace_on or fp_in_decay:
                self._record_channel_learning(
                    "fireplace",
                    error,
                    raw_context,
                    legacy_active_contributions,
                    False,
                    False,
                    update_events,
                )
            if pv_active or pv_in_decay:
                self._record_channel_learning(
                    "pv",
                    error,
                    raw_context,
                    legacy_active_contributions,
                    False,
                    False,
                    update_events,
                )
            if tv_on:
                self._record_channel_learning(
                    "tv",
                    error,
                    raw_context,
                    legacy_active_contributions,
                    False,
                    False,
                    update_events,
                )
            routed = (
                (["fireplace"] if (fireplace_on or fp_in_decay) else [])
                + (["pv"] if (pv_active or pv_in_decay) else [])
                + (["tv"] if tv_on else [])
            )
            self._log_channel_parameter_updates(update_events)
            self._log_routing_summary(
                routed, error, raw_context, update_events,
            )
            return update_events

        active_contributions = self._get_active_contributions(raw_context)
        heat_pump_frozen_by_fireplace = False
        heat_pump_frozen_by_pv_decay = False
        if (fireplace_on or fp_in_decay) and "heat_pump" in active_contributions:
            heat_pump_frozen_by_fireplace = True
            active_contributions.pop("heat_pump", None)
            logger.debug(
                "HP frozen during FP/FP-decay (indoor=%.2f, target=%.2f)",
                current_indoor, target_temp,
            )
        if pv_in_decay and "heat_pump" in active_contributions:
            heat_pump_frozen_by_pv_decay = True
            active_contributions.pop("heat_pump", None)
            logger.debug(
                "HP frozen during PV decay cycle %d/%d (indoor=%.2f, target=%.2f)",
                self._pv_off_cycle_count, pv_decay_window_cycles,
                current_indoor, target_temp,
            )
        # If FP is in post-off decay, ensure it appears in active_contributions
        if fp_in_decay and "fireplace" not in active_contributions:
            active_contributions["fireplace"] = 0.5  # partial weight during decay
        # If PV is in post-off decay, ensure it appears in active_contributions
        if pv_in_decay and "pv" not in active_contributions:
            active_contributions["pv"] = 0.5  # partial weight during decay

        if not active_contributions:
            if heat_pump_active and not fireplace_on and not fp_in_decay and not pv_in_decay:
                fallback_contributions = {"heat_pump": 1.0}
                learning_context = self._build_learning_context(
                    raw_context,
                    attributed_error=error,
                    active_contributions=fallback_contributions,
                    attribution_applied=False,
                    heat_pump_frozen_by_fireplace=False,
                )
                self._record_channel_learning(
                    "heat_pump",
                    error,
                    learning_context,
                    fallback_contributions,
                    False,
                    False,
                    update_events,
                )
            else:
                logger.debug(
                    "Skipping channel learning: no attributable active heat source "
                    "(pv_power_current=%.3f, pv_power=%.3f, thermal_power=%.3f, "
                    "delta_t=%.3f, heat_pump_active=%s)",
                    _to_float(raw_context.get("pv_power_current", 0.0)),
                    _to_float(raw_context.get("pv_power", 0.0)),
                    _to_float(raw_context.get("thermal_power", 0.0)),
                    _to_float(raw_context.get("delta_t", 0.0)),
                    heat_pump_active,
                )
            self._log_channel_parameter_updates(update_events)
            self._log_routing_summary(
                ["heat_pump"] if (heat_pump_active and not fireplace_on) else [],
                error, raw_context, update_events,
            )
            return update_events

        total_heat = sum(active_contributions.values())
        attributed_errors = {
            name: error * (contribution / total_heat)
            for name, contribution in active_contributions.items()
        }

        attribution_applied = len(attributed_errors) > 1
        for channel_name, attributed_error in attributed_errors.items():
            learning_context = self._build_learning_context(
                raw_context,
                attributed_error=attributed_error,
                active_contributions=active_contributions,
                attribution_applied=attribution_applied,
                heat_pump_frozen_by_fireplace=heat_pump_frozen_by_fireplace,
                heat_pump_frozen_by_pv_decay=heat_pump_frozen_by_pv_decay,
            )
            self._record_channel_learning(
                channel_name,
                attributed_error,
                learning_context,
                active_contributions,
                attribution_applied,
                heat_pump_frozen_by_fireplace,
                update_events,
            )

        self._log_channel_parameter_updates(update_events)
        self._log_routing_summary(
            list(attributed_errors.keys()), error, raw_context, update_events,
        )
        return update_events

    def _log_routing_summary(
        self,
        routed_channels: List[str],
        error: float,
        context: Dict,
        update_events: List[Dict[str, Any]],
    ) -> None:
        """Emit one INFO line per route_learning() call showing chosen channels."""
        changed = [e.get("channel", "") for e in update_events if e.get("changes")]
        pv_current = _to_float(context.get("pv_power_current", 0.0))
        pv_smoothed = _to_float(context.get("pv_power", 0.0))
        labels = []
        for ch in routed_channels:
            tag = _get_channel_log_prefix(ch)
            labels.append(f"{tag}{'*' if ch in changed else ''}")
        logger.info(
            "Channel routing → [%s] | error=%+.4f | "
            "pv_current=%.0fW pv_smoothed=%.0fW | "
            "params_changed=%s",
            ", ".join(labels) if labels else "none",
            error,
            pv_current,
            pv_smoothed,
            ",".join(changed) if changed else "none",
        )

    def predict_future_heat(
        self, horizon_hours: float, context: Dict
    ) -> List[float]:
        """
        Predict total future heat contribution per 10-min step.

        Each channel provides its own forecast; the orchestrator sums them.
        """
        steps = max(1, int(horizon_hours * 6))
        total = [0.0] * steps
        for ch in self.channels.values():
            contribution = ch.predict_future_contribution(
                horizon_hours, context
            )
            for i in range(min(steps, len(contribution))):
                total[i] += contribution[i]
        return total

    def get_all_parameters(self) -> Dict[str, Dict[str, float]]:
        """Return full persisted/exported parameters from all channels."""
        return {
            name: ch.get_state_parameters()
            for name, ch in self.channels.items()
        }

    def get_channel_state(self) -> Dict:
        """Return serialisable state for persistence."""
        from datetime import datetime

        now_iso = datetime.now().isoformat()
        result = {
            name: {
                "parameters": ch.get_state_parameters(),
                "history_count": len(ch.history),
                "history": ch.history[-ch._max_history:],
                "last_saved": now_iso,
            }
            for name, ch in self.channels.items()
        }
        # Persist orchestrator-level decay tracking so it survives restarts
        result["_orchestrator"] = {
            "fireplace_was_on": self._fireplace_was_on,
            "fireplace_off_cycle_count": self._fireplace_off_cycle_count,
            "pv_was_active": self._pv_was_active,
            "pv_off_cycle_count": self._pv_off_cycle_count,
        }
        return result

    # Parameters whose authority is baseline + delta in unified_thermal_state.
    # load_channel_state must NOT overwrite these from channel snapshots.
    _BASELINE_MANAGED_PARAMS = frozenset({
        "thermal_time_constant",
        "heat_loss_coefficient",
        "outlet_effectiveness",
        "slab_time_constant_hours",
        "delta_t_floor",
        "fp_decay_time_constant",
        "room_spread_delay_minutes",
    })

    def load_channel_state(
        self, state: Dict, baseline_calibration_date: Optional[str] = None,
    ) -> None:
        """Restore channel state from persisted data.

        Parameters listed in ``_BASELINE_MANAGED_PARAMS`` are only
        skipped when a **newer calibration** has been applied since the
        channel state was last saved.  This preserves online learning
        across normal restarts while still letting a fresh calibration
        take precedence.
        """
        for name, ch_state in state.items():
            if name == "_orchestrator":
                continue
            if name not in self.channels:
                continue

            ch = self.channels[name]

            # Decide whether calibration is newer than channel snapshot.
            # If so, baseline wins for managed params; otherwise channel
            # learning is restored.
            skip_managed = False
            if baseline_calibration_date:
                last_saved = ch_state.get("last_saved")
                if not last_saved:
                    # Legacy state without timestamp — baseline wins.
                    skip_managed = True
                else:
                    skip_managed = baseline_calibration_date > last_saved

            params = ch_state.get("parameters", {})
            for key, value in params.items():
                if skip_managed and key in self._BASELINE_MANAGED_PARAMS:
                    continue
                if hasattr(ch, key):
                    setattr(ch, key, value)

            restored_history = ch_state.get("history")
            if isinstance(restored_history, list):
                ch.history = [
                    record for record in restored_history if isinstance(record, dict)
                ][-ch._max_history:]

        # Restore orchestrator-level decay tracking
        orch_state = state.get("_orchestrator", {})
        self._fireplace_was_on = orch_state.get("fireplace_was_on", False)
        self._fireplace_off_cycle_count = orch_state.get("fireplace_off_cycle_count", 0)
        self._pv_was_active = orch_state.get("pv_was_active", False)
        self._pv_off_cycle_count = orch_state.get("pv_off_cycle_count", 0)

    def attribute_error(
        self, error: float, context: Dict
    ) -> Dict[str, float]:
        """
        Proportional error attribution across active channels (Phase 3, Step 13).

        Each active channel gets a share of the error proportional to its
        current heat estimate, so gradient updates are scaled correctly.
        """
        contributions = self._get_active_contributions(context)

        total_q = sum(contributions.values())
        if total_q <= 0:
            return {}

        return {
            name: error * (q / total_q)
            for name, q in contributions.items()
        }
