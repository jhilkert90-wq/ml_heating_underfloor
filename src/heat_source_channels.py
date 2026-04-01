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
from typing import Dict, List, Optional

try:
    from .thermal_parameters import thermal_params
except ImportError:
    from thermal_parameters import thermal_params

logger = logging.getLogger(__name__)

# Minimum number of learning records before a channel adapts parameters
_MIN_RECORDS_FOR_LEARNING = 5
# Default channel learning rate (conservative to avoid over-correction)
_CHANNEL_LEARNING_RATE = 0.01
# Dead zone: skip learning when average error is below sensor noise
_LEARNING_DEAD_ZONE = 0.05  # °C


def _get_parameter_default(param_name: str, fallback: float) -> float:
    """Read channel startup defaults from the unified parameter manager."""
    try:
        return float(thermal_params.get(param_name))
    except (KeyError, TypeError, ValueError):
        return fallback


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
    return _to_float(context.get("pv_power", 0.0))


def _is_pv_active(context: Dict) -> bool:
    return _get_pv_power(context) > SolarChannel.PV_LEARNING_THRESHOLD


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
    if len(history) < _MIN_RECORDS_FOR_LEARNING:
        return None
    recent = history[-_MIN_RECORDS_FOR_LEARNING:]
    return sum(r["error"] for r in recent) / len(recent)


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

    @abstractmethod
    def apply_gradient_update(
        self, gradients: Dict[str, float], learning_rate: float
    ) -> None:
        """Apply gradient-based parameter update."""

    def record_learning(self, error: float, context: Dict) -> None:
        """Record a learning observation and trigger self-learning."""
        record = {
            "error": error,
            "context": context.copy(),
            "parameters": self.get_learnable_parameters(),
        }
        self.history.append(record)
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]
        # Trigger independent gradient learning
        self._learn_from_recent()

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
            "outlet_effectiveness": self.outlet_effectiveness,
            "slab_time_constant_hours": self.slab_time_constant_hours,
            "delta_t_floor": self.delta_t_floor,
        }

    def apply_gradient_update(
        self, gradients: Dict[str, float], learning_rate: float
    ) -> None:
        if "outlet_effectiveness" in gradients:
            delta = gradients["outlet_effectiveness"] * learning_rate
            self.outlet_effectiveness += max(-0.005, min(0.005, delta))
            self.outlet_effectiveness = max(
                0.01, min(1.0, self.outlet_effectiveness)
            )
        if "slab_time_constant_hours" in gradients:
            delta = gradients["slab_time_constant_hours"] * learning_rate
            self.slab_time_constant_hours += max(-0.1, min(0.1, delta))
            self.slab_time_constant_hours = max(
                0.3, min(10.0, self.slab_time_constant_hours)
            )

    def _learn_from_recent(self) -> None:
        avg_error = _average_recent_error(self.history)
        if avg_error is None or abs(avg_error) < _LEARNING_DEAD_ZONE:
            return

        gradients: Dict[str, float] = {
            "outlet_effectiveness": avg_error,
            "slab_time_constant_hours": avg_error * 0.2,
        }
        self.apply_gradient_update(gradients, _CHANNEL_LEARNING_RATE)
        logger.debug(
            "♨️ HeatPumpChannel self-learned: avg_error=%.3f, "
            "outlet_effectiveness=%.3f, slab_tau=%.2fh",
            avg_error,
            self.outlet_effectiveness,
            self.slab_time_constant_hours,
        )


class SolarChannel(HeatSourceChannel):
    """
    Solar/PV channel — uncontrollable, forecast-aware.

    Parameters learned only from daytime cycles (PV > threshold).
    Decay: small τ (default 0.5 h ≈ 30 min) models residual heat from
    sun-warmed surfaces (floors, walls near windows) after PV drops.
    """

    # PV must exceed this threshold for the channel to learn
    PV_LEARNING_THRESHOLD = 500.0  # Watts

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
        pv_power = context.get("pv_power", 0)
        if isinstance(pv_power, list):
            pv_power = pv_power[-1] if pv_power else 0
        cloud = context.get("avg_cloud_cover", 50.0)
        cloud_factor = max(
            0.1, 1.0 - (cloud / 100.0) ** self.cloud_factor_exponent
        )
        return float(pv_power) * self.pv_heat_weight * cloud_factor

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
            self.pv_heat_weight += max(-0.0002, min(0.0002, delta))
            self.pv_heat_weight = max(0.0, min(0.01, self.pv_heat_weight))
        if "solar_lag_minutes" in gradients:
            delta = gradients["solar_lag_minutes"] * learning_rate
            self.solar_lag_minutes += max(-5.0, min(5.0, delta))
            self.solar_lag_minutes = max(0.0, min(120.0, self.solar_lag_minutes))
        if "solar_decay_tau_hours" in gradients:
            delta = gradients["solar_decay_tau_hours"] * learning_rate
            self.solar_decay_tau_hours += max(-0.05, min(0.05, delta))
            self.solar_decay_tau_hours = max(
                0.0, min(2.0, self.solar_decay_tau_hours)
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
        cloud_factor = max(
            0.1, 1.0 - (cloud / 100.0) ** self.cloud_factor_exponent
        )
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
        if avg_error is None or abs(avg_error) < _LEARNING_DEAD_ZONE:
            return

        gradients: Dict[str, float] = {
            "pv_heat_weight": avg_error * 0.1,
            "solar_decay_tau_hours": avg_error * 0.05,
        }
        self.apply_gradient_update(gradients, _CHANNEL_LEARNING_RATE)
        logger.debug(
            "☀️ SolarChannel self-learned: avg_error=%.3f, "
            "pv_heat_weight=%.5f, solar_decay_tau=%.2fh",
            avg_error,
            self.pv_heat_weight,
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
        if len(self.history) < _MIN_RECORDS_FOR_LEARNING:
            return

        recent = self.history[-_MIN_RECORDS_FOR_LEARNING:]
        avg_error = sum(r["error"] for r in recent) / len(recent)

        if abs(avg_error) < _LEARNING_DEAD_ZONE:
            return

        # Positive error = actual hotter → underestimated FP heat
        gradients: Dict[str, float] = {
            "fp_heat_output_kw": avg_error,
        }
        self.apply_gradient_update(gradients, _CHANNEL_LEARNING_RATE)
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
        if avg_error is None or abs(avg_error) < _LEARNING_DEAD_ZONE:
            return

        gradients: Dict[str, float] = {"tv_heat_weight": avg_error * 0.1}
        self.apply_gradient_update(gradients, _CHANNEL_LEARNING_RATE)
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

    def sync_from_model_parameters(self, parameters: Dict[str, float]) -> None:
        """Seed channel parameters from the current thermal-model state."""
        heat_pump = self.channels["heat_pump"]
        assert isinstance(heat_pump, HeatPumpChannel)
        heat_pump.outlet_effectiveness = parameters.get(
            "outlet_effectiveness", heat_pump.outlet_effectiveness
        )
        heat_pump.slab_time_constant_hours = parameters.get(
            "slab_time_constant_hours", heat_pump.slab_time_constant_hours
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

        tv = self.channels["tv"]
        assert isinstance(tv, TVChannel)
        tv.tv_heat_weight = parameters.get(
            "tv_heat_weight", tv.tv_heat_weight
        )

    def export_model_parameters(self) -> Dict[str, float]:
        """Return channel parameters in thermal-model shape."""
        heat_pump = self.channels["heat_pump"]
        solar = self.channels["pv"]
        fireplace = self.channels["fireplace"]
        tv = self.channels["tv"]
        return {
            "outlet_effectiveness": heat_pump.outlet_effectiveness,
            "slab_time_constant_hours": heat_pump.slab_time_constant_hours,
            "pv_heat_weight": solar.pv_heat_weight,
            "solar_lag_minutes": solar.solar_lag_minutes,
            "fireplace_heat_weight": fireplace.fp_heat_output_kw,
            "tv_heat_weight": tv.tv_heat_weight,
        }

    def total_heat(self, context: Dict) -> float:
        """Calculate total heat contribution from all active sources."""
        return sum(
            ch.estimate_heat_contribution(context)
            for ch in self.channels.values()
        )

    def route_learning(self, error: float, context: Dict) -> None:
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
        fireplace_on = _is_fireplace_active(context)
        pv_active = _is_pv_active(context)
        tv_on = _is_tv_active(context)
        heat_pump_active = _is_heat_pump_active(context)

        any_external_active = fireplace_on or pv_active or tv_on

        if any_external_active:
            # Route to each active external channel
            if fireplace_on:
                self.channels["fireplace"].record_learning(error, context)
            if pv_active:
                self.channels["pv"].record_learning(error, context)
            if tv_on:
                self.channels["tv"].record_learning(error, context)
        elif heat_pump_active:
            # Clean signal — only heat pump learns
            self.channels["heat_pump"].record_learning(error, context)
        else:
            logger.debug(
                "Skipping channel learning: no active heat source in context"
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
        """Return learnable parameters from all channels."""
        return {
            name: ch.get_learnable_parameters()
            for name, ch in self.channels.items()
        }

    def get_channel_state(self) -> Dict:
        """Return serialisable state for persistence."""
        return {
            name: {
                "parameters": ch.get_learnable_parameters(),
                "history_count": len(ch.history),
            }
            for name, ch in self.channels.items()
        }

    def load_channel_state(self, state: Dict) -> None:
        """Restore channel parameters from persisted state."""
        for name, ch_state in state.items():
            if name in self.channels and "parameters" in ch_state:
                params = ch_state["parameters"]
                ch = self.channels[name]
                for key, value in params.items():
                    if hasattr(ch, key):
                        setattr(ch, key, value)

    def attribute_error(
        self, error: float, context: Dict
    ) -> Dict[str, float]:
        """
        Proportional error attribution across active channels (Phase 3, Step 13).

        Each active channel gets a share of the error proportional to its
        current heat estimate, so gradient updates are scaled correctly.
        """
        contributions: Dict[str, float] = {}
        for name, ch in self.channels.items():
            q = ch.estimate_heat_contribution(context)
            if q > 0:
                contributions[name] = q

        total_q = sum(contributions.values())
        if total_q <= 0:
            return {}

        return {
            name: error * (q / total_q)
            for name, q in contributions.items()
        }
