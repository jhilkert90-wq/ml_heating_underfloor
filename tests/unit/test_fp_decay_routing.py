"""Tests for Phase 4: FP post-off learning routing (decay window)."""

import pytest

from src.heat_source_channels import HeatSourceChannelOrchestrator


def _context(fireplace_on=0, pv_power=0, tv_on=0,
             thermal_power=2.0, delta_t=3.0):
    return {
        "fireplace_on": fireplace_on,
        "pv_power": pv_power,
        "pv_power_current": pv_power,
        "tv_on": tv_on,
        "thermal_power": thermal_power,
        "delta_t": delta_t,
        "current_indoor": 20.0,
        "outdoor_temp": 5.0,
        "outlet_temp": 30.0,
    }


class TestFpDecayRouting:
    """Verify that learning routes to FP during post-off decay window.

    We check routing by inspecting channel ``history`` lengths —
    ``record_learning`` always appends to history even when no parameter
    change occurs.
    """

    def setup_method(self):
        self.orch = HeatSourceChannelOrchestrator()

    def _history_len(self, name):
        return len(self.orch.channels[name].history)

    def test_fp_on_routes_to_fireplace(self):
        """FP on -> learning goes to fireplace channel."""
        self.orch.route_learning(0.5, _context(fireplace_on=1))
        assert self._history_len("fireplace") == 1
        assert self._history_len("heat_pump") == 0

    def test_fp_off_immediately_routes_to_fireplace(self):
        """FP was on last cycle, now off -> still routes to fireplace (decay)."""
        self.orch.route_learning(0.5, _context(fireplace_on=1))
        self.orch.route_learning(0.3, _context(fireplace_on=0))
        assert self._history_len("fireplace") == 2
        assert self._history_len("heat_pump") == 0

    def test_fp_decay_window_routes_to_fireplace(self):
        """FP off for 2 cycles after being on -> still routes to fireplace."""
        self.orch.route_learning(0.5, _context(fireplace_on=1))
        self.orch.route_learning(0.3, _context(fireplace_on=0))
        self.orch.route_learning(0.2, _context(fireplace_on=0))
        assert self._history_len("fireplace") == 3
        assert self._history_len("heat_pump") == 0

    def test_fp_decay_window_expires(self):
        """After decay window (3*tau cycles), learning returns to HP."""
        self.orch.route_learning(0.5, _context(fireplace_on=1))

        fp_ch = self.orch.channels["fireplace"]
        tau_h = fp_ch.fp_decay_time_constant
        from src import config
        cycle_min = config.CYCLE_INTERVAL_MINUTES
        decay_cycles = max(1, int((tau_h * 3 * 60) / cycle_min))

        for _ in range(decay_cycles + 2):
            self.orch.route_learning(0.1, _context(fireplace_on=0))

        # Now should route to HP (one more call)
        hp_before = self._history_len("heat_pump")
        self.orch.route_learning(0.1, _context(fireplace_on=0))
        assert self._history_len("heat_pump") > hp_before

    def test_never_on_routes_to_hp(self):
        """FP never on -> goes straight to HP."""
        self.orch.route_learning(0.5, _context(fireplace_on=0))
        assert self._history_len("heat_pump") == 1
        assert self._history_len("fireplace") == 0

    def test_fp_decay_with_pv_active(self):
        """FP in decay + PV active -> both get learning, HP does not."""
        self.orch.route_learning(0.5, _context(fireplace_on=1))
        self.orch.route_learning(
            0.3, _context(fireplace_on=0, pv_power=3000)
        )
        assert self._history_len("fireplace") == 2
        assert self._history_len("pv") == 1
        assert self._history_len("heat_pump") == 0


class TestDecayWindowTracking:
    """Unit tests for the decay window state machine."""

    def test_initial_state(self):
        orch = HeatSourceChannelOrchestrator()
        assert orch._fireplace_was_on is False
        assert orch._fireplace_off_cycle_count == 0

    def test_fp_on_sets_flag(self):
        orch = HeatSourceChannelOrchestrator()
        orch.route_learning(0.5, _context(fireplace_on=1))
        assert orch._fireplace_was_on is True
        assert orch._fireplace_off_cycle_count == 0

    def test_fp_off_increments_counter(self):
        orch = HeatSourceChannelOrchestrator()
        orch.route_learning(0.5, _context(fireplace_on=1))
        orch.route_learning(0.3, _context(fireplace_on=0))
        assert orch._fireplace_was_on is True
        assert orch._fireplace_off_cycle_count == 1

    def test_fp_back_on_resets_counter(self):
        orch = HeatSourceChannelOrchestrator()
        orch.route_learning(0.5, _context(fireplace_on=1))
        orch.route_learning(0.3, _context(fireplace_on=0))
        orch.route_learning(0.3, _context(fireplace_on=0))
        # FP comes back on
        orch.route_learning(0.5, _context(fireplace_on=1))
        assert orch._fireplace_off_cycle_count == 0
