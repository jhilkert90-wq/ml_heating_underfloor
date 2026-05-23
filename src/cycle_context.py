"""
Cycle context dataclass — carries shared state between cycle steps.

A single CycleContext instance is created at the start of each cycle and
passed to every step function and route handler.  This eliminates the need
for dozens of local variables scattered across a 2000+ line function.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CycleContext:
    """Shared mutable context for a single control-loop iteration."""

    # --- Cycle metadata ---
    cycle_number: int = 0
    cycle_start_time: float = field(default_factory=time.time)
    cycle_start_datetime: datetime = field(default_factory=datetime.now)

    # --- External clients ---
    ha_client: Any = None
    influx_service: Any = None
    all_states: dict | None = None

    # --- State management ---
    state: dict = field(default_factory=dict)
    state_manager: Any = None  # active UnifiedThermalState manager
    wrapper: Any = None  # EnhancedModelWrapper singleton

    # --- Mode flags ---
    climate_mode: str = "heating"  # "heating" or "cooling"
    is_blocking: bool = False
    blocking_reasons: list = field(default_factory=list)
    effective_shadow_mode: bool = False
    shadow_mode: Any = None  # ShadowMode resolved object
    ml_heating_enabled: bool = True

    # --- Sensor data ---
    sensor_data: dict = field(default_factory=dict)
    target_indoor_temp: float | None = None
    actual_indoor: float | None = None
    actual_outlet_temp: float | None = None
    avg_other_rooms_temp: float | None = None
    fireplace_on: bool = False
    outdoor_temp: float | None = None
    owm_temp: float | None = None
    prediction_indoor_temp: float | None = None

    # --- Features ---
    features: Any = None  # DataFrame or dict
    features_dict: dict = field(default_factory=dict)
    outlet_history: Any = None

    # --- Prediction results ---
    suggested_temp: float | None = None
    final_temp: float | None = None
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
    predicted_indoor: float | None = None

    # --- Hold / smoothing state ---
    new_hold_cycles: int = 0

    # --- Pre-cooling state ---
    pre_cool_active: bool = False
    pre_cool_result: dict | None = None

    # --- Buffers ---
    sensor_buffer: Any = None
    cooling_ml_model: Any = None
    cooling_obs_buffer: Any = None
    cooling_ml_model_type: str = "trajectory"
    heating_obs_buffer: Any = None

    # --- Learning ---
    last_indoor_temp: float | None = None
    last_run_features: dict | None = None

    # --- Thermodynamic metrics flag ---
    thermodynamic_metrics_written_in_sensor_update: bool = False

    # --- Price data ---
    price_data: Any = None

    # --- Blocking entities ---
    blocking_entities: list = field(default_factory=list)

    @property
    def logger(self):
        return logging.getLogger(__name__)
