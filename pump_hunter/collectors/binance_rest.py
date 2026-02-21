"""
Binance Futures REST collector.
Polls data not available via WebSocket: OI, funding rate, L/S ratios, OI history.
Respects rate limits (1200 weight/min).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from typing import Any, Dict, List, Optional

import aiohttp
import structlog

from pump_hunter.config.settings import Settings
from pump_hunter.storage.database import Database
from pump_hunter.storage.redis_store import RedisStore

logger = structlog.get_logger(__name__)

BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_TESTNET = "https://testnet.binancefuture.com"


class RateLimiter:
    """Token bucket rate limiter for Binance API (1200 weight/min)."""

    def __init__(self, max_weight: int = 1100, window_seconds: int = 60):
        self.max_weight = max_weight
        self.window = window_seconds
        self._used_weight = 0
        self._window_start = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, weight: int = 1) -> None:
        async with self._lock:
            now = time.time()
            if now - self._window_start >= self.window:
                self._used_weight = 0
                self._window_start = now

            if self._used_weight + weight > self.max_weight:
                wait = self.window - (now - self._window_start) + 0.5
                logger.warning("rate_limit_wait", wait_seconds=round(wait, 1))
                await asyncio.sleep(wait)
                self._used_weight = 0
                self._window_start = time.time()

            self._used_weight += weight


class BinanceREST:
    """
    REST API collector for Binance Futures.
    Polls OI, funding, L/S ratios on configurable intervals.
    """

    def __init__(self, settings: Settings, db: Database, redis: RedisStore):
        self.settings = settings
        self.db = db
        self.redis = redis
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = RateLimiter()
        self._running = False
        self._tasks: List[asyncio.Task] = []

    @property
    def base_url(self) -> str:
        return BINANCE_TESTNET if self.settings.exchanges.use_testnet else BINANCE_FAPI

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self, symbols: List[str]) -> None:
        """Start periodic polling tasks."""
        self._session = aiohttp.ClientSession()
        self._running = True
        self._symbols = symbols

        # OI polling — every 60s
        self._tasks.append(asyncio.create_task(
            self._poll_loop("open_interest", self._fetch_all_oi, 60)
        ))

        # funding rate — every 60s
        self._tasks.append(asyncio.create_task(
            self._poll_loop("funding_rate", self._fetch_all_funding, 60)
        ))

        # L/S ratio — every 5 min
        self._tasks.append(asyncio.create_task(
            self._poll_loop("long_short_ratio", self._fetch_all_ls_ratio, 300)
        ))

        logger.info("rest_collector_started", symbols=len(symbols))

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._session:
            await self._session.close()
        logger.info("rest_collector_stopped")

    def update_symbols(self, symbols: List[str]) -> None:
        """Update the symbol list (called after refresh)."""
        self._symbols = symbols

    # ------------------------------------------------------------------
    # generic poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(
        self, name: str, fetch_fn, interval_seconds: int
    ) -> None:
        """Generic polling loop with error handling."""
        while self._running:
            try:
                await fetch_fn()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("poll_error", name=name, error=str(e))
            await asyncio.sleep(interval_seconds)

    # ------------------------------------------------------------------
    # API calls — with rate limiting
    # ------------------------------------------------------------------

    async def _api_get(self, path: str, params: dict = None, weight: int = 1) -> Any:
        """Make a GET request to Binance API with rate limiting."""
        await self._rate_limiter.acquire(weight)
        url = f"{self.base_url}{path}"
        async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 429:
                # rate limited — wait and retry
                retry_after = int(resp.headers.get("Retry-After", 60))
                logger.warning("api_rate_limited", retry_after=retry_after)
                await asyncio.sleep(retry_after)
                return await self._api_get(path, params, weight)
            resp.raise_for_status()
            return await resp.json()

    # ------------------------------------------------------------------
    # Open Interest
    # ------------------------------------------------------------------

    async def _fetch_all_oi(self) -> None:
        """Fetch open interest for all symbols."""
        batch_size = 20
        for i in range(0, len(self._symbols), batch_size):
            batch = self._symbols[i : i + batch_size]
            tasks = [self._fetch_oi(sym) for sym in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            # small delay between batches
            await asyncio.sleep(0.2)

    async def _fetch_oi(self, symbol: str) -> None:
        """Fetch open interest for a single symbol."""
        try:
            data = await self._api_get(
                "/fapi/v1/openInterest",
                {"symbol": symbol},
                weight=1,
            )
            oi = float(data.get("openInterest", 0))
            price_data = await self.redis.get_price(symbol)
            price = price_data.get("price", 0) if price_data else 0
            oi_value = oi * price

            # get historical OI for change calculation
            cached = await self.redis.get_open_interest(symbol)
            old_oi = cached.get("oi", oi) if cached else oi
            change_pct = ((oi_value - old_oi) / old_oi * 100) if old_oi > 0 else 0

            await self.redis.set_open_interest(symbol, oi_value, change_pct)

        except Exception as e:
            if "429" not in str(e):
                logger.debug("oi_fetch_error", symbol=symbol, error=str(e))

    # ------------------------------------------------------------------
    # Funding Rate
    # ------------------------------------------------------------------

    async def _fetch_all_funding(self) -> None:
        """Fetch current funding rates."""
        try:
            # premium index gives funding for all symbols at once (weight 1)
            data = await self._api_get("/fapi/v1/premiumIndex", weight=10)
            for item in data:
                symbol = item.get("symbol", "")
                if symbol in self._symbols:
                    rate = float(item.get("lastFundingRate", 0) or 0)
                    await self.redis.set_funding(symbol, rate)
        except Exception as e:
            logger.error("funding_fetch_error", error=str(e))

    # ------------------------------------------------------------------
    # Long/Short Ratio
    # ------------------------------------------------------------------

    async def _fetch_all_ls_ratio(self) -> None:
        """Fetch global L/S account ratio for all symbols."""
        batch_size = 10
        for i in range(0, len(self._symbols), batch_size):
            batch = self._symbols[i : i + batch_size]
            tasks = [self._fetch_ls_ratio(sym) for sym in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0.5)

    async def _fetch_ls_ratio(self, symbol: str) -> None:
        """Fetch L/S ratio for a single symbol."""
        try:
            data = await self._api_get(
                "/futures/data/globalLongShortAccountRatio",
                {"symbol": symbol, "period": "5m", "limit": 1},
                weight=1,
            )
            if data and len(data) > 0:
                ratio = float(data[0].get("longShortRatio", 1.0))
                await self.redis.set_long_short(symbol, ratio)
        except Exception as e:
            # many coins don't support this endpoint — silently ignore
            pass

    async def _fetch_top_trader_ls(self, symbol: str) -> Optional[float]:
        """Fetch top trader L/S ratio (secondary data point)."""
        try:
            data = await self._api_get(
                "/futures/data/topLongShortAccountRatio",
                {"symbol": symbol, "period": "5m", "limit": 1},
                weight=1,
            )
            if data and len(data) > 0:
                return float(data[0].get("longShortRatio", 1.0))
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Bootstrap — historical data seeding
    # ------------------------------------------------------------------

    async def bootstrap_symbol(self, symbol: str, symbol_id: int) -> None:
        """Fetch historical data for a newly discovered symbol."""
        logger.info("bootstrap_start", symbol=symbol)

        try:
            # historical klines
            klines = await self._fetch_klines(symbol, limit=self.settings.general.bootstrap_candles)
            if klines:
                from pump_hunter.storage.timeseries import TimeseriesStore
                # will be passed externally
                logger.debug("bootstrap_klines", symbol=symbol, count=len(klines))

            # historical OI
            oi_hist = await self._fetch_oi_history(symbol, limit=self.settings.general.bootstrap_oi_points)
            if oi_hist:
                snapshots = []
                for point in oi_hist:
                    snapshots.append({
                        "symbol_id": symbol_id,
                        "exchange": "binance",
                        "timestamp": dt.datetime.utcfromtimestamp(point["timestamp"] / 1000),
                        "open_interest_value": float(point.get("sumOpenInterestValue", 0)),
                        "open_interest": float(point.get("sumOpenInterest", 0)),
                    })
                await self.db.insert_snapshots(snapshots)
                logger.debug("bootstrap_oi", symbol=symbol, count=len(snapshots))

            logger.info("bootstrap_done", symbol=symbol)
            return klines

        except Exception as e:
            logger.error("bootstrap_error", symbol=symbol, error=str(e))
            return None

    async def _fetch_klines(
        self, symbol: str, interval: str = "1h", limit: int = 500
    ) -> Optional[list]:
        """Fetch historical klines."""
        try:
            data = await self._api_get(
                "/fapi/v1/klines",
                {"symbol": symbol, "interval": interval, "limit": limit},
                weight=5,
            )
            return data
        except Exception as e:
            logger.debug("klines_fetch_error", symbol=symbol, error=str(e))
            return None

    async def _fetch_oi_history(
        self, symbol: str, period: str = "1h", limit: int = 200
    ) -> Optional[list]:
        """Fetch historical OI (may not work on all CCXT versions)."""
        try:
            data = await self._api_get(
                "/futures/data/openInterestHist",
                {"symbol": symbol, "period": period, "limit": limit},
                weight=1,
            )
            return data
        except Exception as e:
            logger.debug("oi_hist_fetch_error", symbol=symbol, error=str(e))
            return None

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "polling_tasks": len(self._tasks),
            "symbols": len(self._symbols) if hasattr(self, "_symbols") else 0,
            "running": self._running,
        }
