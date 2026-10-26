"""
Electricity price-aware optimization for underfloor heating.

Classifies the current electricity price as CHEAP / NORMAL / EXPENSIVE
relative to today's price distribution, and provides a binary-search
target offset so the thermal model heats slightly more during cheap
periods and slightly less during expensive ones.

Prices are fetched via the ``tibber.get_prices`` HA service call and
cached in-memory.  If the service returns 15-minute resolution data it
is automatically merged to 60-minute (hourly) entries by averaging the
prices within each clock hour.
"""
import logging
from collections import deque
from datetime import date, datetime, timedelta, timezone, tzinfo
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from . import config
except ImportError:
    import config  # type: ignore

logger = logging.getLogger(__name__)


class PriceLevel(str, Enum):
    CHEAP = "cheap"
    NORMAL = "normal"
    EXPENSIVE = "expensive"


class PriceOptimizer:
    """Classify electricity prices and compute heating target offsets."""

    def __init__(
        self,
        cheap_percentile: float = None,
        expensive_percentile: float = None,
        target_offset: float = None,
        expensive_overshoot: float = None,
    ):
        self.cheap_percentile = (
            cheap_percentile
            if cheap_percentile is not None
            else getattr(config, "PRICE_CHEAP_PERCENTILE", 33.0)
        )
        self.expensive_percentile = (
            expensive_percentile
            if expensive_percentile is not None
            else getattr(config, "PRICE_EXPENSIVE_PERCENTILE", 67.0)
        )
        self.target_offset = (
            target_offset
            if target_offset is not None
            else getattr(config, "PRICE_TARGET_OFFSET", 0.2)
        )
        self.expensive_overshoot = (
            expensive_overshoot
            if expensive_overshoot is not None
            else getattr(config, "PRICE_EXPENSIVE_OVERSHOOT", 0.2)
        )
        # Cache the last classification for HA export
        self._last_price: Optional[float] = None
        self._last_level: PriceLevel = PriceLevel.NORMAL
        self._last_thresholds: Tuple[float, float] = (0.0, 0.0)

        # --- Tibber service price cache ---
        # Each entry is (start_time: datetime, price: float), sorted by time.
        self._price_entries: List[Tuple[datetime, float]] = []
        self._cache_time: Optional[datetime] = None
        self._cache_date: Optional[date] = None
        self._has_tomorrow: bool = False
        # Timezone extracted from Tibber price timestamps (e.g. MESZ +02:00)
        self._price_tz: Optional[tzinfo] = None

    # ------------------------------------------------------------------
    # Tibber price cache
    # ------------------------------------------------------------------

    def refresh_prices_if_needed(self, ha_client: object) -> None:
        """Fetch prices from the ``tibber.get_prices`` service if stale.

        Refresh triggers:
        1. Cache is empty.
        2. Calendar day changed since last fetch.
        3. Cache older than ``PRICE_CACHE_REFRESH_MINUTES``.
        4. After 13:00 local time and tomorrow's prices not yet cached.

        Args:
            ha_client: An ``HAClient`` instance with
                :meth:`call_tibber_get_prices`.
        """
        now = datetime.now(timezone.utc)
        now_local = self._to_local(now)
        refresh_minutes = getattr(
            config, "PRICE_CACHE_REFRESH_MINUTES", 60
        )

        needs_refresh = False
        if not self._price_entries:
            needs_refresh = True
        elif self._cache_date != now_local.date():
            needs_refresh = True
        elif (
            self._cache_time is not None
            and (now - self._cache_time).total_seconds() > refresh_minutes * 60
        ):
            needs_refresh = True
        elif (
            now_local.hour >= 13
            and not self._has_tomorrow
        ):
            needs_refresh = True

        if not needs_refresh:
            return

        # Request today 00:00 → day_after_tomorrow 00:00 (up to 48 h)
        today_start = now_local.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = today_start + timedelta(days=2)
        start_str = today_start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end.strftime("%Y-%m-%d %H:%M:%S")

        try:
            raw_entries = ha_client.call_tibber_get_prices(start_str, end_str)
        except Exception as exc:
            logger.warning("Tibber price refresh failed: %s", exc)
            return

        if not raw_entries:
            logger.debug("Tibber returned no price entries")
            return

        self._parse_price_entries(raw_entries, now_local)

    def _parse_price_entries(
        self,
        raw_entries: List[Dict],
        now_local: datetime,
    ) -> None:
        """Parse raw Tibber entries into ``_price_entries`` and update cache
        metadata.

        If the raw data has 15-minute resolution it is automatically
        merged to 60-minute (hourly) entries by averaging the prices
        within each clock hour.
        """
        entries: List[Tuple[datetime, float]] = []
        for item in raw_entries:
            try:
                ts = datetime.fromisoformat(str(item["start_time"]))
                price = float(item["price"])
                entries.append((ts, price))
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("Skipping invalid price entry %s: %s", item, exc)

        if not entries:
            return

        entries.sort(key=lambda e: e[0])
        entries = self._merge_to_hourly_if_needed(entries)
        self._price_entries = entries
        self._price_tz = entries[0][0].tzinfo

        now_utc = datetime.now(timezone.utc)
        self._cache_time = now_utc
        self._cache_date = now_local.date()

        tomorrow = now_local.date() + timedelta(days=1)
        self._has_tomorrow = any(
            self._to_local(ts).date() == tomorrow for ts, _ in entries
        )

        logger.info(
            "Price cache updated: %d entries, tomorrow=%s",
            len(entries),
            self._has_tomorrow,
        )

    # ------------------------------------------------------------------
    # 15-min → 60-min merge
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_to_hourly_if_needed(
        entries: List[Tuple[datetime, float]],
    ) -> List[Tuple[datetime, float]]:
        """Merge 15-minute price entries to hourly by averaging.

        Detection: if the first two entries are exactly 15 minutes apart
        the data is treated as quarter-hourly.  Entries are grouped by
        their clock hour (truncated to HH:00) and the mean price of each
        group becomes the hourly price.

        If the data is already hourly (or any other resolution) the
        entries are returned unchanged.
        """
        if len(entries) < 2:
            return entries
        delta = entries[1][0] - entries[0][0]
        if delta != timedelta(minutes=15):
            return entries

        # Group by the start of each clock hour
        hourly: Dict[datetime, List[float]] = {}
        for ts, price in entries:
            hour_start = ts.replace(minute=0, second=0, microsecond=0)
            hourly.setdefault(hour_start, []).append(price)

        merged = [
            (hour_start, sum(prices) / len(prices))
            for hour_start, prices in sorted(hourly.items())
        ]
        logger.info(
            "Merged %d quarter-hourly entries → %d hourly entries",
            len(entries),
            len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # Current price and today's prices from cache
    # ------------------------------------------------------------------

    def get_current_price(
        self, now: Optional[datetime] = None
    ) -> Optional[float]:
        """Return the price that is active at *now*.

        Finds the last entry whose ``start_time ≤ now``.
        """
        if not self._price_entries:
            return None
        if now is None:
            now = datetime.now(timezone.utc)
        # Ensure timezone-aware comparison
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        best: Optional[float] = None
        for ts, price in self._price_entries:
            ts_cmp = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            if ts_cmp <= now:
                best = price
            else:
                break  # entries are sorted
        return best

    def get_today_prices(
        self, now: Optional[datetime] = None
    ) -> List[float]:
        """Return all prices for today's calendar day.

        Used for percentile-based classification.  Returns 24 hourly entries.
        Quarter-hourly input is merged to hourly during parsing.
        """
        if not self._price_entries:
            return []
        if now is None:
            now = datetime.now(timezone.utc)
        now_local = self._to_local(now)
        today = now_local.date()

        return [
            price
            for ts, price in self._price_entries
            if self._to_local(ts).date() == today
        ]

    def get_price_data_for_features(self) -> Optional[Dict[str, object]]:
        """Build the ``_electricity_price`` dict consumed by model_wrapper.

        Returns the same format as the deprecated
        ``HAClient.get_electricity_price()`` so that downstream code
        (``calculate_optimal_outlet_temp``) needs no changes.

        Returns ``None`` when the cache is empty or the current price
        cannot be determined.
        """
        current = self.get_current_price()
        if current is None:
            return None
        today = self.get_today_prices()
        if not today:
            return None
        return {"current_price": current, "today": today}

    # ------------------------------------------------------------------
    # Cost-aware (PV / feed-in) target offset
    # ------------------------------------------------------------------

    def get_hourly_grid_prices(
        self, hours: int, now: Optional[datetime] = None
    ) -> List[float]:
        """Return grid prices for the next *hours* clock hours.

        Hour 0 is the current clock hour.  The list is truncated where the
        cached price data ends (e.g. tomorrow's prices not yet published).
        """
        if not self._price_entries or hours <= 0:
            return []
        if now is None:
            now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        last_ts = self._price_entries[-1][0]
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        data_end = last_ts + timedelta(hours=1)
        hour_start = self._to_local(now).replace(
            minute=0, second=0, microsecond=0
        )
        prices: List[float] = []
        for h in range(hours):
            at = hour_start + timedelta(hours=h)
            if at >= data_end:
                break
            price = self.get_current_price(at)
            if price is None:
                break
            prices.append(price)
        return prices

    def compute_cost_target_offset(
        self,
        pv_hourly_w: List[float],
        hp_power_kw: float,
        now: Optional[datetime] = None,
    ) -> Optional[Dict[str, float]]:
        """Compute a heating-direction target offset from effective prices.

        The effective price of each hour blends the grid price with the
        feed-in tariff according to the share ``f`` of the heat pump demand
        that PV can cover::

            p_eff = (1 - f) * p_grid + f * min(p_grid, feed_in)

        The current hour is compared with the mean effective price of the
        look-ahead window.  Cheaper than average → positive offset (pre-heat,
        up to ``PRICE_BAND_UP_C``); more expensive → negative offset (up to
        ``PRICE_BAND_DOWN_C``).  Scaling is linear between the mean and the
        window extreme.  *pv_hourly_w* is the PV forecast [W] per hour with
        index 0 = current hour; the window is cut to the hours for which a PV
        forecast exists (all hours assume no PV if no forecast is given).

        Returns ``None`` when no prices are available; otherwise a dict with
        ``offset`` and diagnostics.
        """
        lookahead = max(
            1, int(getattr(config, "PRICE_COST_LOOKAHEAD_HOURS", 24))
        )
        grid = self.get_hourly_grid_prices(lookahead, now)
        if not grid:
            return None
        price_hours = len(grid)
        if len(pv_hourly_w) > 1:
            grid = grid[: len(pv_hourly_w)]
        feed_in = float(getattr(config, "FEED_IN_TARIFF_EUR_KWH", 0.079))
        demand_w = max(1.0, float(hp_power_kw) * 1000.0)
        eff: List[float] = []
        for h, p_grid in enumerate(grid):
            pv_w = pv_hourly_w[h] if h < len(pv_hourly_w) else 0.0
            f = min(1.0, max(0.0, float(pv_w)) / demand_w)
            eff.append((1.0 - f) * p_grid + f * min(p_grid, feed_in))

        p_now = eff[0]
        p_ref = sum(eff) / len(eff)
        p_min, p_max = min(eff), max(eff)
        min_spread = float(
            getattr(config, "PRICE_COST_MIN_SPREAD_EUR_KWH", 0.02)
        )
        band_up = float(getattr(config, "PRICE_BAND_UP_C", 0.5))
        band_down = float(getattr(config, "PRICE_BAND_DOWN_C", 0.2))

        offset = 0.0
        if (p_max - p_min) >= min_spread:
            if p_now < p_ref and p_ref - p_min > 1e-9:
                offset = band_up * (p_ref - p_now) / (p_ref - p_min)
            elif p_now > p_ref and p_max - p_ref > 1e-9:
                offset = -band_down * (p_now - p_ref) / (p_max - p_ref)
        return {
            "offset": offset,
            "p_eff_now": p_now,
            "p_eff_mean": p_ref,
            "p_eff_min": p_min,
            "p_eff_max": p_max,
            "hp_power_kw": float(hp_power_kw),
            "hours": float(len(eff)),
            "price_hours_available": float(price_hours),
            "pv_hours_available": float(len(pv_hourly_w)),
            "pv_share_now": min(
                1.0,
                max(0.0, float(pv_hourly_w[0] if pv_hourly_w else 0.0))
                / demand_w,
            ),
            "p_grid_now": grid[0],
        }

    # ------------------------------------------------------------------
    # Timezone helper
    # ------------------------------------------------------------------

    def _to_local(self, dt: datetime) -> datetime:
        """Convert *dt* to the price timezone (from Tibber entries).

        Falls back to system local time when no prices have been parsed yet.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if self._price_tz is not None:
            return dt.astimezone(self._price_tz)
        return dt.astimezone()

    # ------------------------------------------------------------------
    # Core classification
    # ------------------------------------------------------------------

    def classify_price(
        self,
        current_price: float,
        daily_prices: List[float],
    ) -> PriceLevel:
        """Classify *current_price* relative to today's price distribution.

        Args:
            current_price: Current electricity price in EUR/kWh.
            daily_prices: List of 24 hourly prices for today.

        Returns:
            PriceLevel.CHEAP, NORMAL, or EXPENSIVE.
        """
        if not daily_prices or len(daily_prices) < 2:
            logger.debug(
                "Price classification: insufficient daily prices (%d), "
                "defaulting to NORMAL",
                len(daily_prices or []),
            )
            self._last_price = current_price
            self._last_level = PriceLevel.NORMAL
            return PriceLevel.NORMAL

        prices = np.array(daily_prices, dtype=float)
        cheap_threshold = float(np.percentile(prices, self.cheap_percentile))
        expensive_threshold = float(
            np.percentile(prices, self.expensive_percentile)
        )

        self._last_price = current_price
        self._last_thresholds = (cheap_threshold, expensive_threshold)

        if current_price <= cheap_threshold:
            level = PriceLevel.CHEAP
        elif current_price >= expensive_threshold:
            level = PriceLevel.EXPENSIVE
        else:
            level = PriceLevel.NORMAL

        self._last_level = level
        logger.debug(
            "Price %.4f EUR/kWh → %s (cheap<%.4f, expensive>%.4f)",
            current_price,
            level.value,
            cheap_threshold,
            expensive_threshold,
        )
        return level

    # ------------------------------------------------------------------
    # Target offset for binary search
    # ------------------------------------------------------------------

    def get_target_offset(self, price_level: PriceLevel) -> float:
        """Return the temperature offset to add to the binary search target.

        CHEAP    → +offset  (e.g. +0.2°C → search targets 22.2°C)
        NORMAL   →  0.0
        EXPENSIVE→ -offset  (e.g. -0.2°C → search targets 21.8°C)
        """
        if price_level == PriceLevel.CHEAP:
            return self.target_offset
        elif price_level == PriceLevel.EXPENSIVE:
            return -self.target_offset
        return 0.0

    # ------------------------------------------------------------------
    # Trajectory correction thresholds
    # ------------------------------------------------------------------

    def get_trajectory_thresholds(
        self,
        price_level: PriceLevel,
        target_indoor: float,
    ) -> Dict[str, float]:
        """Return overshoot thresholds for trajectory correction.

        Returns a dict with:
          immediate_overshoot: threshold for next-cycle overshoot detection
          future_overshoot:    threshold for multi-hour trajectory overshoot

        CHEAP:     immediate = target+0.1 (unchanged), future = target+0.5 (unchanged)
        NORMAL:    immediate = target+0.1,              future = target+0.5
        EXPENSIVE: immediate = target+0.1,              future = target+expensive_overshoot
        """
        immediate = target_indoor + 0.1  # always tight for next cycle

        if price_level == PriceLevel.EXPENSIVE:
            future = target_indoor + self.expensive_overshoot
        else:
            future = target_indoor + 0.5  # existing default

        return {
            "immediate_overshoot": immediate,
            "future_overshoot": future,
        }

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def get_price_info(self) -> Dict[str, object]:
        """Return last classification result for logging / HA export."""
        return {
            "price_eur_kwh": self._last_price,
            "price_level": self._last_level.value,
            "price_cheap_threshold": self._last_thresholds[0],
            "price_expensive_threshold": self._last_thresholds[1],
            "price_target_offset": self.get_target_offset(self._last_level),
        }


# ------------------------------------------------------------------
# Expected heat pump electrical power (space heating/cooling only)
# ------------------------------------------------------------------


class HPPowerTracker:
    """Rolling mean of the heat pump's electrical power while it is
    actively heating (or cooling) the building.

    Samples are rejected while hot water (DHW, disinfection, boost heater)
    is running, during defrost and its recovery grace period, and when the
    heat pump is not delivering thermal power in the active mode.
    """

    def __init__(self) -> None:
        self._samples: deque = deque()

    def record(
        self,
        features: Dict[str, object],
        climate_mode: str = "heating",
        now: Optional[datetime] = None,
    ) -> bool:
        """Record a sample if it qualifies. Returns True if accepted."""
        try:
            power_w = float(features.get("power_consumption", 0.0) or 0.0)
            thermal_kw = float(features.get("thermal_power_kw", 0.0) or 0.0)
            if (
                features.get("dhw_heating")
                or features.get("dhw_disinfection")
                or features.get("dhw_boost_heater")
                or features.get("defrosting")
            ):
                logger.debug("HP power sample skipped: DHW/defrost active")
                return False
            since_defrost = features.get("minutes_since_defrost")
            grace = float(getattr(config, "DEFROST_RECOVERY_GRACE_MINUTES", 45))
            if since_defrost is not None and float(since_defrost) < grace:
                return False
        except (TypeError, ValueError):
            return False

        if power_w < float(getattr(config, "HP_POWER_MIN_ON_W", 300.0)):
            return False
        if climate_mode == "cooling":
            if thermal_kw > float(config.COOLING_MIN_THERMAL_POWER_KW):
                return False
        elif thermal_kw < float(config.HEATING_MIN_THERMAL_POWER_KW):
            return False

        now = now or datetime.now(timezone.utc)
        self._samples.append((now, power_w))
        self._prune(now)
        return True

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def _prune(self, now: datetime) -> None:
        window = timedelta(
            hours=float(getattr(config, "HP_POWER_WINDOW_HOURS", 6.0))
        )
        while self._samples and now - self._samples[0][0] > window:
            self._samples.popleft()

    def expected_power_kw(self, now: Optional[datetime] = None) -> float:
        """Mean qualifying power [kW]; configured fallback without samples."""
        now = now or datetime.now(timezone.utc)
        self._prune(now)
        if not self._samples:
            return float(getattr(config, "HP_EXPECTED_POWER_FALLBACK_KW", 1.5))
        return sum(p for _, p in self._samples) / len(self._samples) / 1000.0


_hp_tracker: Optional[HPPowerTracker] = None


def get_hp_power_tracker() -> HPPowerTracker:
    """Return (or create) the module-level HPPowerTracker."""
    global _hp_tracker
    if _hp_tracker is None:
        _hp_tracker = HPPowerTracker()
    return _hp_tracker


# ------------------------------------------------------------------
# Module-level singleton (lazy)
# ------------------------------------------------------------------

_optimizer: Optional[PriceOptimizer] = None


def get_price_optimizer() -> PriceOptimizer:
    """Return (or create) the module-level PriceOptimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PriceOptimizer()
    return _optimizer
