"""
Market state — central per-symbol state with rolling windows.
Receives updates from WebSocket and REST collectors, provides snapshot views.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import structlog

from pump_hunter.config.settings import Settings
from pump_hunter.storage.redis_store import RedisStore

logger = structlog.get_logger(__name__)


class SymbolState:
    """Real-time state for a single symbol."""

    __slots__ = (
        "symbol",
        "price", "price_open_24h", "price_high_24h", "price_low_24h",
        "volume_24h", "change_pct_24h",
        "open_interest", "oi_value", "oi_change_pct",
        "funding_rate", "next_funding_time",
        "long_short_ratio", "top_trader_ls",
        "bid_depth_usd", "ask_depth_usd", "depth_imbalance",
        "best_bid", "best_ask", "spread_pct",
        "whale_buy_vol", "whale_sell_vol", "whale_count",
        "liq_buy_vol", "liq_sell_vol", "liq_count",
        "last_update", "cross_exchange",
        "_price_history", "_volume_history", "_oi_history",
    )

    def __init__(self, symbol: str):
        self.symbol = symbol

        # live values
        self.price: float = 0.0
        self.price_open_24h: float = 0.0
        self.price_high_24h: float = 0.0
        self.price_low_24h: float = 0.0
        self.volume_24h: float = 0.0
        self.change_pct_24h: float = 0.0

        self.open_interest: float = 0.0
        self.oi_value: float = 0.0
        self.oi_change_pct: float = 0.0

        self.funding_rate: float = 0.0
        self.next_funding_time: Optional[dt.datetime] = None

        self.long_short_ratio: float = 1.0
        self.top_trader_ls: float = 1.0

        self.bid_depth_usd: float = 0.0
        self.ask_depth_usd: float = 0.0
        self.depth_imbalance: float = 1.0
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.spread_pct: float = 0.0

        self.whale_buy_vol: float = 0.0
        self.whale_sell_vol: float = 0.0
        self.whale_count: int = 0

        self.liq_buy_vol: float = 0.0
        self.liq_sell_vol: float = 0.0
        self.liq_count: int = 0

        self.last_update: float = 0.0
        self.cross_exchange: Dict[str, dict] = {}

        # rolling history (tuples of (timestamp, value))
        self._price_history: Deque[Tuple[float, float]] = deque(maxlen=1440)  # ~24h of 1m
        self._volume_history: Deque[Tuple[float, float]] = deque(maxlen=1440)
        self._oi_history: Deque[Tuple[float, float]] = deque(maxlen=1440)

    def update_ticker(
        self,
        price: float,
        open_24h: float = 0,
        high_24h: float = 0,
        low_24h: float = 0,
        volume_24h: float = 0,
        change_pct: float = 0,
    ) -> None:
        """Update from miniTicker data."""
        self.price = price
        self.price_open_24h = open_24h or self.price_open_24h
        self.price_high_24h = high_24h or self.price_high_24h
        self.price_low_24h = low_24h or self.price_low_24h
        self.volume_24h = volume_24h or self.volume_24h
        self.change_pct_24h = change_pct
        self.last_update = time.time()

        self._price_history.append((self.last_update, price))

    def update_oi(self, oi_value: float, change_pct: float = 0) -> None:
        """Update open interest."""
        self.oi_value = oi_value
        self.oi_change_pct = change_pct
        self._oi_history.append((time.time(), oi_value))

    def update_funding(self, rate: float) -> None:
        """Update funding rate."""
        self.funding_rate = rate

    def update_depth(self, bid_usd: float, ask_usd: float) -> None:
        """Update orderbook depth."""
        self.bid_depth_usd = bid_usd
        self.ask_depth_usd = ask_usd
        self.depth_imbalance = bid_usd / ask_usd if ask_usd > 0 else 0

    def update_whale(self, buy_vol: float, sell_vol: float, count: int) -> None:
        """Update whale activity."""
        self.whale_buy_vol = buy_vol
        self.whale_sell_vol = sell_vol
        self.whale_count = count

    def update_liquidations(self, buy_vol: float, sell_vol: float, count: int) -> None:
        """Update liquidation data."""
        self.liq_buy_vol = buy_vol
        self.liq_sell_vol = sell_vol
        self.liq_count = count

    def update_long_short(self, ratio: float) -> None:
        self.long_short_ratio = ratio

    def get_price_change(self, minutes: int = 60) -> float:
        """Get price change % over last N minutes."""
        if not self._price_history:
            return 0.0
        cutoff = time.time() - (minutes * 60)
        old_prices = [v for t, v in self._price_history if t <= cutoff]
        if not old_prices:
            # use oldest available
            old_price = self._price_history[0][1]
        else:
            old_price = old_prices[-1]
        if old_price <= 0:
            return 0.0
        return (self.price - old_price) / old_price * 100

    def get_oi_change(self, minutes: int = 60) -> float:
        """Get OI change % over last N minutes."""
        if not self._oi_history:
            return 0.0
        cutoff = time.time() - (minutes * 60)
        old_oi = [v for t, v in self._oi_history if t <= cutoff]
        if not old_oi:
            old_val = self._oi_history[0][1]
        else:
            old_val = old_oi[-1]
        if old_val <= 0:
            return 0.0
        return (self.oi_value - old_val) / old_val * 100

    def to_snapshot_dict(self, symbol_id: int) -> dict:
        """Convert to a snapshot dict for database storage."""
        return {
            "symbol_id": symbol_id,
            "exchange": "binance",
            "timestamp": dt.datetime.utcnow(),
            "price": self.price,
            "price_high": self.price_high_24h,
            "price_low": self.price_low_24h,
            "price_open": self.price_open_24h,
            "price_close": self.price,
            "volume_quote": self.volume_24h,
            "volume_24h_quote": self.volume_24h,
            "open_interest": self.open_interest,
            "open_interest_value": self.oi_value,
            "funding_rate": self.funding_rate,
            "long_short_ratio": self.long_short_ratio,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread_pct": self.spread_pct,
            "bid_depth_usd": self.bid_depth_usd,
            "ask_depth_usd": self.ask_depth_usd,
            "depth_imbalance": self.depth_imbalance,
            "liq_buy_volume": self.liq_buy_vol,
            "liq_sell_volume": self.liq_sell_vol,
            "liq_count": self.liq_count,
            "whale_buy_volume": self.whale_buy_vol,
            "whale_sell_volume": self.whale_sell_vol,
            "whale_trade_count": self.whale_count,
        }

    def to_dict(self) -> dict:
        """Convert to a simple dict for Redis/display."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "change_pct_24h": self.change_pct_24h,
            "volume_24h": self.volume_24h,
            "oi_value": self.oi_value,
            "oi_change_pct": self.oi_change_pct,
            "funding_rate": self.funding_rate,
            "long_short_ratio": self.long_short_ratio,
            "depth_imbalance": self.depth_imbalance,
            "bid_depth_usd": self.bid_depth_usd,
            "ask_depth_usd": self.ask_depth_usd,
            "whale_buy_vol": self.whale_buy_vol,
            "whale_sell_vol": self.whale_sell_vol,
            "liq_buy_vol": self.liq_buy_vol,
            "liq_sell_vol": self.liq_sell_vol,
            "last_update": self.last_update,
        }


class MarketStateManager:
    """Manages SymbolState for all tracked symbols."""

    def __init__(self, settings: Settings, redis: RedisStore):
        self.settings = settings
        self.redis = redis
        self._states: Dict[str, SymbolState] = {}

    def get_state(self, symbol: str) -> SymbolState:
        """Get or create state for a symbol."""
        if symbol not in self._states:
            self._states[symbol] = SymbolState(symbol)
        return self._states[symbol]

    def get_all_states(self) -> Dict[str, SymbolState]:
        """Get all symbol states."""
        return self._states

    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol's state."""
        self._states.pop(symbol, None)

    async def sync_from_redis(self, symbol: str) -> None:
        """Pull latest data from Redis into symbol state."""
        state = self.get_state(symbol)

        price_data = await self.redis.get_price(symbol)
        if price_data:
            state.price = price_data.get("price", state.price)
            state.change_pct_24h = price_data.get("change_pct", state.change_pct_24h)
            if state.price > 0:
                state.last_update = time.time()
                state._price_history.append((state.last_update, state.price))

        oi_data = await self.redis.get_open_interest(symbol)
        if oi_data:
            state.oi_value = oi_data.get("oi", state.oi_value)
            state.oi_change_pct = oi_data.get("change_pct", state.oi_change_pct)

        funding = await self.redis.get_funding(symbol)
        if funding:
            state.funding_rate = funding.get("rate", state.funding_rate)

        depth = await self.redis.get_depth(symbol)
        if depth:
            state.bid_depth_usd = depth.get("bids_usd", state.bid_depth_usd)
            state.ask_depth_usd = depth.get("asks_usd", state.ask_depth_usd)
            state.depth_imbalance = depth.get("imbalance", state.depth_imbalance)

        ls = await self.redis.get_long_short(symbol)
        if ls:
            state.long_short_ratio = ls.get("ratio", state.long_short_ratio)

        whale = await self.redis.get_json(f"ph:whale:{symbol}")
        if whale:
            state.update_whale(
                whale.get("buy_volume", 0),
                whale.get("sell_volume", 0),
                whale.get("count", 0),
            )

        liq = await self.redis.get_json(f"ph:liq:{symbol}")
        if liq:
            state.update_liquidations(
                liq.get("buy_volume", 0),
                liq.get("sell_volume", 0),
                liq.get("count", 0),
            )

    async def sync_all(self, symbols: List[str]) -> None:
        """Sync all symbols from Redis."""
        tasks = [self.sync_from_redis(sym) for sym in symbols]
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_stats(self) -> dict:
        """Return market state statistics."""
        active = [s for s in self._states.values() if s.last_update > 0]
        return {
            "total_symbols": len(self._states),
            "active_symbols": len(active),
            "avg_price_change": np.mean([s.change_pct_24h for s in active]) if active else 0,
        }
