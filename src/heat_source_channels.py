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

logger = logging.getLogger(__name__)


class HeatSourceChannel(ABC):
    """
    Abstract base class for a heat source learning channel.

    Each channel independently tracks its own prediction history and
    learnable parameters, preventing cross-contamination between sources.
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
        """Record a learning observation for this channel."""
        record = {
            "error": error,
            "context": context.copy(),
            "parameters": self.get_learnable_parameters(),
        }
        self.history.append(record)
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]

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
        self.outlet_effectiveness = 0.17
        self.slab_time_constant_hours = 1.0
        self.delta_t_floor = 2.0

    def estimate_heat_contribution(self, context: Dict) -> float:
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
            self.outlet_effectiveness = max(0.01, min(1.0, self.outlet_effectiveness))
        if "slab_time_constant_hours" in gradients:
            delta = gradients["slab_time_constant_hours"] * learning_rate
            self.slab_time_constant_hours += max(-0.1, min(0.1, delta))
            self.slab_time_constant_hours = max(
                0.3, min(10.0, self.slab_time_constant_hours)
            )


class SolarChannel(HeatSourceChannel):
    """
    Solar/PV channel — uncontrollable, forecast-aware.

    Parameters learned only from daytime cycles (PV > threshold).
    Decay: none (immediate drop when sun sets).
    """

    # PV must exceed this threshold for the channel to learn
    PV_LEARNING_THRESHOLD = 500.0  # Watts

    def __init__(self):
        super().__init__("pv")
        self.pv_heat_weight = 0.002
        self.solar_lag_minutes = 45.0
        self.cloud_factor_exponent = 1.0

    def estimate_heat_contribution(self, context: Dict) -> float:
        pv_power = context.get("pv_power", 0)
        if isinstance(pv_power, list):
            pv_power = pv_power[-1] if pv_power else 0
        cloud = context.get("avg_cloud_cover", 50.0)
        cloud_factor = max(0.1, 1.0 - (cloud / 100.0) ** self.cloud_factor_exponent)
        return float(pv_power) * self.pv_heat_weight * cloud_factor

    def estimate_decay_contribution(
        self, time_since_off: float, context: Dict
    ) -> float:
        # Solar has no thermal mass — immediate drop
        return 0.0

    def get_learnable_parameters(self) -> Dict[str, float]:
        return {
            "pv_heat_weight": self.pv_heat_weight,
            "solar_lag_minutes": self.solar_lag_minutes,
            "cloud_factor_exponent": self.cloud_factor_exponent,
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

    def predict_future_contribution(
        self, horizon_hours: float, context: Dict
    ) -> List[float]:
        """Use PV forecast to predict future solar heat contribution."""
        pv_forecast = context.get("pv_forecast")
        cloud = context.get("avg_cloud_cover", 50.0)
        cloud_factor = max(0.1, 1.0 - (cloud / 100.0) ** self.cloud_factor_exponent)
        steps = max(1, int(horizon_hours * 6))

        if not pv_forecast:
            current = self.estimate_heat_contribution(context)
            return [current] * steps

        result = []
        for i in range(steps):
            # pv_forecast is assumed hourly; map 10-min steps to hourly index
            hour_idx = min(i // 6, len(pv_forecast) - 1)
            pv_val = float(pv_forecast[hour_idx])
            result.append(pv_val * self.pv_heat_weight * cloud_factor)
        return result


class FireplaceChannel(HeatSourceChannel):
    """
    Fireplace channel — uncontrollable, observed via binary sensor.

    Decay: exponential after fireplace turned off (τ ~ 30-60 min).
    Room spread: living room heat distributes to house average with delay.
    """

    def __init__(self):
        super().__init__("fireplace")
        self.fp_heat_output_kw = 5.0
        self.fp_decay_time_constant = 0.75  # hours (~45 min)
        self.room_spread_delay_minutes = 30.0

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
            self.fp_heat_output_kw = max(0.5, min(15.0, self.fp_heat_output_kw))
        if "fp_decay_time_constant" in gradients:
            delta = gradients["fp_decay_time_constant"] * learning_rate
            self.fp_decay_time_constant += max(-0.1, min(0.1, delta))
            self.fp_decay_time_constant = max(
                0.1, min(2.0, self.fp_decay_time_constant)
            )


class TVChannel(HeatSourceChannel):
    """
    TV/Electronics channel — minor heat source (~0.25 kW).

    Simple additive term; not worth its own complex model.
    """

    def __init__(self):
        super().__init__("tv")
        self.tv_heat_weight = 0.2  # kW effective contribution

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

    def total_heat(self, context: Dict) -> float:
        """Calculate total heat contribution from all active sources."""
        return sum(ch.estimate_heat_contribution(context) for ch in self.channels.values())

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
        """
        fp_on_val = context.get("fireplace_on", 0)
        fireplace_on = bool(fp_on_val) and fp_on_val > 0
        pv_power = context.get("pv_power", 0)
        if isinstance(pv_power, list):
            pv_power = pv_power[-1] if pv_power else 0
        pv_active = float(pv_power) > SolarChannel.PV_LEARNING_THRESHOLD
        tv_on_val = context.get("tv_on", 0)
        tv_on = bool(tv_on_val) and tv_on_val > 0

        any_external_active = fireplace_on or pv_active or tv_on

        if any_external_active:
            # Route to each active external channel
            if fireplace_on:
                self.channels["fireplace"].record_learning(error, context)
            if pv_active:
                self.channels["pv"].record_learning(error, context)
            if tv_on:
                self.channels["tv"].record_learning(error, context)
        else:
            # Clean signal — only heat pump learns
            self.channels["heat_pump"].record_learning(error, context)

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
            contribution = ch.predict_future_contribution(horizon_hours, context)
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
