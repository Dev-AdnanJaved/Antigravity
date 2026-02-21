"""
Redis-based real-time cache and pub/sub messaging.
Stores hot market data with TTL, leaderboards, and inter-component communication.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Dict, List, Optional

import orjson
import redis.asyncio as aioredis
import structlog

from pump_hunter.config.settings import Settings

logger = structlog.get_logger(__name__)


class RedisStore:
    """Async Redis wrapper for hot data caching and pub/sub."""

    # key prefixes
    PFX_PRICE = "ph:price:"
    PFX_TICKER = "ph:ticker:"
    PFX_OI = "ph:oi:"
    PFX_FUNDING = "ph:funding:"
    PFX_DEPTH = "ph:depth:"
    PFX_LS = "ph:ls:"
    PFX_WHALE = "ph:whale:"
    PFX_LIQ = "ph:liq:"
    PFX_STATE = "ph:state:"

    # pub/sub channels
    CH_PRICE_UPDATE = "ph:ch:price"
    CH_PUMP_ALERT = "ph:ch:pump"
    CH_PREDICTION = "ph:ch:prediction"

    # sorted sets
    SS_PRICE_CHANGE = "ph:ss:price_change"
    SS_VOLUME_CHANGE = "ph:ss:volume_change"
    SS_OI_CHANGE = "ph:ss:oi_change"
    SS_SCORES = "ph:ss:scores"

    DEFAULT_TTL = 120  # seconds

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pool: Optional[aioredis.Redis] = None
        self._pubsub: Optional[aioredis.client.PubSub] = None
        self._sub_tasks: List[asyncio.Task] = []

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to Redis."""
        cfg = self.settings.redis
        self._pool = aioredis.Redis(
            host=cfg.host,
            port=cfg.port,
            db=cfg.db,
            password=cfg.password or None,
            decode_responses=False,  # we use orjson
        )
        # ping to verify
        await self._pool.ping()
        logger.info("redis_connected", host=cfg.host, port=cfg.port)

    async def disconnect(self) -> None:
        """Close Redis connection."""
        for task in self._sub_tasks:
            task.cancel()
        if self._pubsub:
            await self._pubsub.close()
        if self._pool:
            await self._pool.close()
        logger.info("redis_disconnected")

    # ------------------------------------------------------------------
    # basic key/value with TTL
    # ------------------------------------------------------------------

    async def set_json(self, key: str, data: Any, ttl: int = DEFAULT_TTL) -> None:
        """Store JSON-serializable data with TTL."""
        await self._pool.set(key, orjson.dumps(data), ex=ttl)

    async def get_json(self, key: str) -> Optional[Any]:
        """Retrieve JSON data."""
        raw = await self._pool.get(key)
        if raw is None:
            return None
        return orjson.loads(raw)

    async def delete(self, key: str) -> None:
        """Delete a key."""
        await self._pool.delete(key)

    # ------------------------------------------------------------------
    # market data helpers
    # ------------------------------------------------------------------

    async def set_price(self, symbol: str, price: float, change_pct: float = 0) -> None:
        """Cache latest price and update price change leaderboard."""
        await self._pool.set(
            f"{self.PFX_PRICE}{symbol}",
            orjson.dumps({"price": price, "change_pct": change_pct}),
            ex=self.DEFAULT_TTL,
        )
        # update sorted set
        await self._pool.zadd(self.SS_PRICE_CHANGE, {symbol: change_pct})

    async def get_price(self, symbol: str) -> Optional[dict]:
        """Get cached price."""
        return await self.get_json(f"{self.PFX_PRICE}{symbol}")

    async def set_ticker(self, symbol: str, ticker: dict) -> None:
        """Cache full ticker data."""
        await self.set_json(f"{self.PFX_TICKER}{symbol}", ticker)

    async def get_ticker(self, symbol: str) -> Optional[dict]:
        return await self.get_json(f"{self.PFX_TICKER}{symbol}")

    async def set_open_interest(self, symbol: str, oi: float, change_pct: float = 0) -> None:
        """Cache OI and update OI change leaderboard."""
        await self.set_json(f"{self.PFX_OI}{symbol}", {"oi": oi, "change_pct": change_pct})
        await self._pool.zadd(self.SS_OI_CHANGE, {symbol: change_pct})

    async def get_open_interest(self, symbol: str) -> Optional[dict]:
        return await self.get_json(f"{self.PFX_OI}{symbol}")

    async def set_funding(self, symbol: str, rate: float) -> None:
        await self.set_json(f"{self.PFX_FUNDING}{symbol}", {"rate": rate})

    async def get_funding(self, symbol: str) -> Optional[dict]:
        return await self.get_json(f"{self.PFX_FUNDING}{symbol}")

    async def set_depth(self, symbol: str, bids_usd: float, asks_usd: float) -> None:
        await self.set_json(f"{self.PFX_DEPTH}{symbol}", {
            "bids_usd": bids_usd,
            "asks_usd": asks_usd,
            "imbalance": bids_usd / asks_usd if asks_usd > 0 else 0,
        })

    async def get_depth(self, symbol: str) -> Optional[dict]:
        return await self.get_json(f"{self.PFX_DEPTH}{symbol}")

    async def set_long_short(self, symbol: str, ratio: float) -> None:
        await self.set_json(f"{self.PFX_LS}{symbol}", {"ratio": ratio})

    async def get_long_short(self, symbol: str) -> Optional[dict]:
        return await self.get_json(f"{self.PFX_LS}{symbol}")

    async def set_score(self, symbol: str, score: float) -> None:
        """Update composite score leaderboard."""
        await self._pool.zadd(self.SS_SCORES, {symbol: score})

    async def set_market_state(self, symbol: str, state: dict) -> None:
        """Cache full aggregated market state for a symbol."""
        await self.set_json(f"{self.PFX_STATE}{symbol}", state, ttl=300)

    async def get_market_state(self, symbol: str) -> Optional[dict]:
        return await self.get_json(f"{self.PFX_STATE}{symbol}")

    # ------------------------------------------------------------------
    # leaderboards (sorted sets)
    # ------------------------------------------------------------------

    async def get_top_movers(self, key: str = SS_PRICE_CHANGE, count: int = 20) -> List[tuple]:
        """Get top N movers from a sorted set (highest first)."""
        results = await self._pool.zrevrange(key, 0, count - 1, withscores=True)
        return [(name.decode() if isinstance(name, bytes) else name, score) for name, score in results]

    async def get_top_scores(self, count: int = 20) -> List[tuple]:
        """Get top N symbols by composite score."""
        return await self.get_top_movers(self.SS_SCORES, count)

    # ------------------------------------------------------------------
    # pub/sub
    # ------------------------------------------------------------------

    async def publish(self, channel: str, data: Any) -> None:
        """Publish data to a channel."""
        await self._pool.publish(channel, orjson.dumps(data))

    async def subscribe(
        self,
        channel: str,
        callback: Callable[[Any], Coroutine],
    ) -> None:
        """Subscribe to a channel and handle messages with callback."""
        if self._pubsub is None:
            self._pubsub = self._pool.pubsub()

        await self._pubsub.subscribe(channel)

        async def _listener():
            try:
                async for message in self._pubsub.listen():
                    if message["type"] == "message":
                        data = orjson.loads(message["data"])
                        await callback(data)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("pubsub_error", channel=channel, error=str(e))

        task = asyncio.create_task(_listener())
        self._sub_tasks.append(task)

    async def publish_pump_alert(self, alert_data: dict) -> None:
        """Publish pump alert for consumers."""
        await self.publish(self.CH_PUMP_ALERT, alert_data)

    async def publish_prediction(self, prediction_data: dict) -> None:
        """Publish ML prediction."""
        await self.publish(self.CH_PREDICTION, prediction_data)

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------

    async def flush_all(self) -> None:
        """Clear all pump_hunter keys (for testing/reset)."""
        async for key in self._pool.scan_iter(match="ph:*"):
            await self._pool.delete(key)
        logger.warning("redis_flushed")

    async def health_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            await self._pool.ping()
            return True
        except Exception:
            return False
