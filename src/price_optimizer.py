"""
Electricity price-aware optimization for underfloor heating.

Classifies the current electricity price as CHEAP / NORMAL / EXPENSIVE
relative to today's price distribution, and provides a binary-search
target offset so the thermal model heats slightly more during cheap
periods and slightly less during expensive ones.

Prices are fetched via the ``tibber.get_prices`` HA service call and
cached in-memory.  Both 15-minute and 60-minute price resolutions are
handled transparently — percentile calculations always use every entry
for the current calendar day.
"""
import logging
from datetime import date, datetime, timedelta, timezone
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
        metadata."""
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
        self._price_entries = entries

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
    # Current price and today's prices from cache
    # ------------------------------------------------------------------

    def get_current_price(
        self, now: Optional[datetime] = None
    ) -> Optional[float]:
        """Return the price that is active at *now*.

        Finds the last entry whose ``start_time ≤ now``.  Works for both
        15-minute and 60-minute resolutions.
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

        Used for percentile-based classification.  Returns 24 entries for
        hourly data or 96 for 15-minute data — the classifier handles both.
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
    # Timezone helper
    # ------------------------------------------------------------------

    @staticmethod
    def _to_local(dt: datetime) -> datetime:
        """Convert *dt* to the local timezone (``astimezone(None)``)."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
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
# Module-level singleton (lazy)
# ------------------------------------------------------------------

_optimizer: Optional[PriceOptimizer] = None


def get_price_optimizer() -> PriceOptimizer:
    """Return (or create) the module-level PriceOptimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PriceOptimizer()
    return _optimizer
