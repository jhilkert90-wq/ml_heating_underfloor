"""
Cross-cycle persistent state for the main control loop.

LoopState holds all variables that persist between cycles but are not part
of the persisted operational state (saved to disk).  These are runtime-only
variables that live for the lifetime of the process.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class LoopState:
    """Runtime variables that persist across control-loop iterations.

    These are NOT saved to disk — they only live in the process memory.
    On restart, they reset to their defaults.
    """

    # --- Cycle counters ---
    cycle_number: int = 0
    last_cycle_end_time: float | None = None

    # --- Shadow mode comparison (legacy) ---
    shadow_ml_error_sum: float = 0.0
    shadow_hc_error_sum: float = 0.0
    shadow_comparison_count: int = 0

    # --- Cooling ML model + observation buffer ---
    cooling_ml_model: Any = None
    cooling_obs_buffer: Any = None
    cooling_ml_model_type: str = "trajectory"

    # --- Heating correction ML observation buffer ---
    heating_obs_buffer: Any = None

    # --- Sensor buffer ---
    sensor_buffer: Any = None

    # --- InfluxDB service ---
    influx_service: Any = None

    # --- Model wrapper (singleton reference) ---
    wrapper: Any = None

    # --- One-time flags ---
    sensor_validation_done: bool = False

    # --- Blocking entities (config-derived, constant per run) ---
    blocking_entities: list = field(default_factory=list)

    def increment_cycle(self) -> tuple[int, float, datetime]:
        """Advance to the next cycle. Returns (cycle_number, start_time, start_datetime)."""
        self.cycle_number += 1
        start_time = time.time()
        start_datetime = datetime.now()
        return self.cycle_number, start_time, start_datetime
