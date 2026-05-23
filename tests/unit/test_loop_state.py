"""Tests for src.loop_state.LoopState."""
import time
from datetime import datetime

from src.loop_state import LoopState


class TestLoopState:
    """Basic LoopState dataclass behavior."""

    def test_default_initialization(self):
        ls = LoopState()
        assert ls.cycle_number == 0
        assert ls.last_cycle_end_time is None
        assert ls.shadow_ml_error_sum == 0.0
        assert ls.shadow_hc_error_sum == 0.0
        assert ls.shadow_comparison_count == 0
        assert ls.cooling_ml_model is None
        assert ls.heating_obs_buffer is None
        assert ls.sensor_buffer is None
        assert ls.influx_service is None
        assert ls.wrapper is None
        assert ls.sensor_validation_done is False
        assert ls.blocking_entities == []

    def test_increment_cycle(self):
        ls = LoopState()
        before = time.time()
        cycle_num, start_time, start_dt = ls.increment_cycle()
        after = time.time()

        assert cycle_num == 1
        assert ls.cycle_number == 1
        assert before <= start_time <= after
        assert isinstance(start_dt, datetime)

    def test_increment_cycle_successive(self):
        ls = LoopState()
        ls.increment_cycle()
        ls.increment_cycle()
        cycle_num, _, _ = ls.increment_cycle()
        assert cycle_num == 3
        assert ls.cycle_number == 3

    def test_blocking_entities_isolation(self):
        """Ensure default list is not shared between instances."""
        ls1 = LoopState()
        ls2 = LoopState()
        ls1.blocking_entities.append("sensor.dhw")
        assert ls2.blocking_entities == []
