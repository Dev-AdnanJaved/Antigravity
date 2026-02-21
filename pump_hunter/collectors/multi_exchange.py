"""
Multi-exchange data collector via CCXT.
Collects ticker, OI, funding, and orderbook from secondary exchanges (Bybit, OKX, Bitget).
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any, Dict, List, Optional, Set

import ccxt.async_support as ccxt
import structlog

from pump_hunter.config.settings import Settings
from pump_hunter.storage.database import Database
from pump_hunter.storage.redis_store import RedisStore

logger = structlog.get_logger(__name__)

# supported exchanges and their CCXT class names
EXCHANGE_MAP = {
    "bybit": "bybit",
    "okx": "okx",
    "bitget": "bitget",
}


class MultiExchangeCollector:
    """
    CCXT-based collector for cross-exchange data.
    Runs periodic scans of secondary exchanges for matching symbols.
    """

    def __init__(self, settings: Settings, db: Database, redis: RedisStore):
        self.settings = settings
        self.db = db
        self.redis = redis
        self._exchanges: Dict[str, ccxt.Exchange] = {}
        self._exchange_symbols: Dict[str, Set[str]] = {}  # exchange -> set of symbols
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self, binance_symbols: List[str]) -> None:
        """Initialize exchanges and start periodic collection."""
        self._binance_symbols = set(binance_symbols)
        self._running = True

        for ex_name in self.settings.exchanges.secondary:
            if ex_name not in EXCHANGE_MAP:
                logger.warning("unsupported_exchange", exchange=ex_name)
                continue
            try:
                exchange = await self._create_exchange(ex_name)
                self._exchanges[ex_name] = exchange
                # discover matching symbols
                await self._discover_symbols(ex_name, exchange)
                logger.info(
                    "exchange_initialized",
                    exchange=ex_name,
                    matching_symbols=len(self._exchange_symbols.get(ex_name, set())),
                )
            except Exception as e:
                logger.error("exchange_init_error", exchange=ex_name, error=str(e))

        if self._exchanges:
            self._task = asyncio.create_task(self._poll_loop())
            logger.info("multi_exchange_started", exchanges=list(self._exchanges.keys()))

    async def stop(self) -> None:
        """Close all exchange connections."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        for name, ex in self._exchanges.items():
            try:
                await ex.close()
            except Exception:
                pass
        logger.info("multi_exchange_stopped")

    def update_symbols(self, binance_symbols: List[str]) -> None:
        """Update the list of symbols to track."""
        self._binance_symbols = set(binance_symbols)

    # ------------------------------------------------------------------
    # exchange setup
    # ------------------------------------------------------------------

    async def _create_exchange(self, name: str) -> ccxt.Exchange:
        """Create a CCXT exchange instance."""
        cls_name = EXCHANGE_MAP[name]
        cls = getattr(ccxt, cls_name)

        config = {
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }

        # add API keys if configured
        api_key = self.settings.api_keys.get(name)
        if api_key and api_key.key:
            config["apiKey"] = api_key.key
            config["secret"] = api_key.secret
            if api_key.passphrase:
                config["password"] = api_key.passphrase

        exchange = cls(config)
        await exchange.load_markets()
        return exchange

    async def _discover_symbols(self, name: str, exchange: ccxt.Exchange) -> None:
        """Find which Binance symbols also exist on this exchange."""
        matching = set()
        for market_id, market in exchange.markets.items():
            if (
                market.get("type") == "swap"
                and market.get("quote") == "USDT"
                and market.get("active", True)
                and market.get("linear", True)
            ):
                # normalize to Binance format (e.g., BTC/USDT:USDT -> BTCUSDT)
                base = market.get("base", "")
                binance_sym = f"{base}USDT"
                if binance_sym in self._binance_symbols:
                    matching.add(binance_sym)

        self._exchange_symbols[name] = matching

    # ------------------------------------------------------------------
    # polling
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Periodically collect data from all secondary exchanges."""
        interval = self.settings.general.scan_interval_seconds

        while self._running:
            try:
                for ex_name, exchange in self._exchanges.items():
                    symbols = self._exchange_symbols.get(ex_name, set())
                    if not symbols:
                        continue

                    await self._collect_exchange(ex_name, exchange, list(symbols))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("multi_exchange_poll_error", error=str(e))

            await asyncio.sleep(interval)

    async def _collect_exchange(
        self, name: str, exchange: ccxt.Exchange, symbols: List[str]
    ) -> None:
        """Collect data from a single exchange for all matching symbols."""
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            tasks = [self._collect_symbol(name, exchange, sym) for sym in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            errors = sum(1 for r in results if isinstance(r, Exception))
            if errors > 0:
                logger.debug(
                    "exchange_batch_errors",
                    exchange=name,
                    errors=errors,
                    batch_size=len(batch),
                )

    async def _collect_symbol(
        self, ex_name: str, exchange: ccxt.Exchange, binance_symbol: str
    ) -> None:
        """Collect all available data for a symbol on an exchange."""
        try:
            # convert symbol format
            base = binance_symbol.replace("USDT", "")
            ccxt_symbol = f"{base}/USDT:USDT"

            if ccxt_symbol not in exchange.markets:
                return

            data = {}

            # ticker
            try:
                ticker = await exchange.fetch_ticker(ccxt_symbol)
                data["ticker"] = {
                    "price": ticker.get("last", 0),
                    "volume_24h": ticker.get("quoteVolume", 0),
                    "change_pct": ticker.get("percentage", 0),
                    "high": ticker.get("high", 0),
                    "low": ticker.get("low", 0),
                }
            except Exception:
                pass

            # open interest
            try:
                if hasattr(exchange, "fetch_open_interest"):
                    oi = await exchange.fetch_open_interest(ccxt_symbol)
                    data["open_interest"] = oi.get("openInterestValue", 0) or oi.get("openInterestAmount", 0)
            except Exception:
                pass

            # funding rate
            try:
                fr = await exchange.fetch_funding_rate(ccxt_symbol)
                data["funding_rate"] = fr.get("fundingRate", 0)
            except Exception:
                pass

            # store cross-exchange data in Redis
            if data:
                key = f"ph:xex:{ex_name}:{binance_symbol}"
                await self.redis.set_json(key, data, ttl=300)

        except Exception as e:
            logger.debug("xex_collect_error", exchange=ex_name, symbol=binance_symbol, error=str(e))

    # ------------------------------------------------------------------
    # data access
    # ------------------------------------------------------------------

    async def get_cross_exchange_data(self, symbol: str) -> Dict[str, dict]:
        """Get collected data from all secondary exchanges for a symbol."""
        result = {}
        for ex_name in self._exchanges:
            key = f"ph:xex:{ex_name}:{symbol}"
            data = await self.redis.get_json(key)
            if data:
                result[ex_name] = data
        return result

    async def get_cross_exchange_volumes(self, symbol: str) -> Dict[str, float]:
        """Get 24h volumes across all exchanges for a symbol."""
        volumes = {}

        # binance from main ticker cache
        bticker = await self.redis.get_ticker(symbol)
        if bticker:
            volumes["binance"] = bticker.get("volume_24h", 0)

        # secondary exchanges
        for ex_name in self._exchanges:
            key = f"ph:xex:{ex_name}:{symbol}"
            data = await self.redis.get_json(key)
            if data and "ticker" in data:
                volumes[ex_name] = data["ticker"].get("volume_24h", 0)

        return volumes

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "exchanges": list(self._exchanges.keys()),
            "matching_symbols": {
                name: len(syms) for name, syms in self._exchange_symbols.items()
            },
            "running": self._running,
        }
