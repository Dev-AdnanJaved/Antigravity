"""
Symbol manager — discovers and maintains list of all USDT perpetual futures.
Refreshes every 6 hours, filters by config criteria, stores in DB.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from typing import Dict, List, Optional, Set

import aiohttp
import structlog

from pump_hunter.config.settings import Settings
from pump_hunter.storage.database import Database
from pump_hunter.storage.redis_store import RedisStore

logger = structlog.get_logger(__name__)

BINANCE_FAPI_BASE = "https://fapi.binance.com"
BINANCE_TESTNET_BASE = "https://testnet.binancefuture.com"


class SymbolManager:
    """Discovers and manages all tradeable USDT perpetual futures symbols."""

    def __init__(self, settings: Settings, db: Database, redis: RedisStore):
        self.settings = settings
        self.db = db
        self.redis = redis
        self._symbols: Dict[str, dict] = {}  # symbol -> metadata
        self._last_refresh: float = 0
        self._refresh_interval = 6 * 3600  # 6 hours
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def base_url(self) -> str:
        if self.settings.exchanges.use_testnet:
            return BINANCE_TESTNET_BASE
        return BINANCE_FAPI_BASE

    @property
    def active_symbols(self) -> List[str]:
        """Get list of active symbol names."""
        return list(self._symbols.keys())

    @property
    def symbol_count(self) -> int:
        return len(self._symbols)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initial discovery of all futures symbols."""
        self._session = aiohttp.ClientSession()
        await self.refresh()

    async def stop(self) -> None:
        """Cleanup."""
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    async def refresh(self) -> None:
        """Fetch all USDT perpetual symbols from Binance and apply filters."""
        now = time.time()
        if now - self._last_refresh < self._refresh_interval and self._symbols:
            return

        logger.info("symbol_refresh_start")

        try:
            exchange_info = await self._fetch_exchange_info()
            symbols = self._parse_symbols(exchange_info)
            tickers = await self._fetch_24hr_tickers()
            ticker_map = {t["symbol"]: t for t in tickers}

            filtered = self._apply_filters(symbols, ticker_map)

            # detect additions/removals
            old_set = set(self._symbols.keys())
            new_set = set(filtered.keys())
            added = new_set - old_set
            removed = old_set - new_set

            if added:
                logger.info("symbols_added", count=len(added), symbols=sorted(added)[:10])
            if removed:
                logger.info("symbols_removed", count=len(removed), symbols=sorted(removed)[:10])

            # update DB
            for sym, meta in filtered.items():
                await self.db.upsert_symbol(
                    symbol=sym,
                    base_asset=meta["base_asset"],
                    quote_asset=meta["quote_asset"],
                    exchange="binance",
                    listed_exchanges=meta.get("listed_exchanges", {}),
                )

            if removed:
                await self.db.deactivate_symbols(list(removed))

            self._symbols = filtered
            self._last_refresh = now

            logger.info(
                "symbol_refresh_done",
                total=len(filtered),
                added=len(added),
                removed=len(removed),
            )

        except Exception as e:
            logger.error("symbol_refresh_error", error=str(e))
            if not self._symbols:
                raise

    async def _fetch_exchange_info(self) -> dict:
        """Fetch /fapi/v1/exchangeInfo."""
        url = f"{self.base_url}/fapi/v1/exchangeInfo"
        async with self._session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _fetch_24hr_tickers(self) -> list:
        """Fetch /fapi/v1/ticker/24hr for volume filtering."""
        url = f"{self.base_url}/fapi/v1/ticker/24hr"
        async with self._session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()

    def _parse_symbols(self, exchange_info: dict) -> Dict[str, dict]:
        """Parse exchange info into symbol metadata."""
        result = {}
        for s in exchange_info.get("symbols", []):
            if (
                s.get("contractType") == "PERPETUAL"
                and s.get("quoteAsset") == "USDT"
                and s.get("status") == "TRADING"
            ):
                sym = s["symbol"]
                result[sym] = {
                    "base_asset": s.get("baseAsset", ""),
                    "quote_asset": s.get("quoteAsset", "USDT"),
                    "price_precision": s.get("pricePrecision", 8),
                    "quantity_precision": s.get("quantityPrecision", 8),
                    "tick_size": self._get_tick_size(s),
                    "listed_exchanges": {"binance": True},
                }
        return result

    def _get_tick_size(self, symbol_info: dict) -> float:
        """Extract tick size from filters."""
        for f in symbol_info.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                return float(f.get("tickSize", 0.01))
        return 0.01

    def _apply_filters(
        self, symbols: Dict[str, dict], tickers: Dict[str, dict]
    ) -> Dict[str, dict]:
        """Apply config-based filters (volume, exclusions, etc.)."""
        cfg = self.settings.filters
        filtered = {}

        for sym, meta in symbols.items():
            # exclusion list
            if sym in cfg.excluded_symbols:
                continue

            # only_symbols filter (if set)
            if cfg.only_symbols and sym not in cfg.only_symbols:
                continue

            # volume filter
            ticker = tickers.get(sym, {})
            vol_24h = float(ticker.get("quoteVolume", 0) or 0)
            if vol_24h < cfg.min_volume_24h_usd:
                continue

            # store volume metadata
            meta["volume_24h"] = vol_24h
            meta["price"] = float(ticker.get("lastPrice", 0) or 0)
            meta["price_change_pct"] = float(ticker.get("priceChangePercent", 0) or 0)

            filtered[sym] = meta

        return filtered

    def get_symbol_meta(self, symbol: str) -> Optional[dict]:
        """Get metadata for a symbol."""
        return self._symbols.get(symbol)

    def get_ws_streams(self) -> List[str]:
        """Generate WebSocket stream names for all active symbols."""
        streams = []
        for sym in self._symbols:
            s = sym.lower()
            streams.extend([
                f"{s}@miniTicker",
                f"{s}@kline_1m",
                f"{s}@depth20@100ms",
                f"{s}@aggTrade",
            ])
        # add global liquidation stream
        streams.append("!forceOrder@arr")
        return streams

    def get_ws_stream_batches(self) -> List[List[str]]:
        """Split streams into batches for multiple WS connections."""
        all_streams = self.get_ws_streams()
        batch_size = self.settings.websocket.max_streams_per_connection
        return [
            all_streams[i : i + batch_size]
            for i in range(0, len(all_streams), batch_size)
        ]
