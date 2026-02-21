"""
Binance Futures WebSocket collector.
Connects to combined streams for real-time price, kline, depth, aggTrade, and liquidation data.
Auto-reconnects with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

import structlog
import websockets
from websockets.exceptions import ConnectionClosed

from pump_hunter.config.settings import Settings
from pump_hunter.storage.redis_store import RedisStore
from pump_hunter.storage.timeseries import TimeseriesStore

logger = structlog.get_logger(__name__)

BINANCE_WS_BASE = "wss://fstream.binance.com"
BINANCE_WS_TESTNET = "wss://stream.binancefuture.com"


class BinanceWebSocket:
    """
    Manages multiple WebSocket connections to Binance Futures combined streams.
    Distributes incoming data to handlers and caches in Redis + TimeseriesStore.
    """

    def __init__(
        self,
        settings: Settings,
        redis: RedisStore,
        timeseries: TimeseriesStore,
    ):
        self.settings = settings
        self.redis = redis
        self.timeseries = timeseries

        self._connections: List[asyncio.Task] = []
        self._running = False
        self._last_message_time: Dict[int, float] = {}

        # external handlers
        self._on_ticker: Optional[Callable] = None
        self._on_kline: Optional[Callable] = None
        self._on_depth: Optional[Callable] = None
        self._on_agg_trade: Optional[Callable] = None
        self._on_liquidation: Optional[Callable] = None

        # stats
        self._msg_count = 0
        self._error_count = 0
        self._reconnect_count = 0

    @property
    def base_url(self) -> str:
        if self.settings.exchanges.use_testnet:
            return BINANCE_WS_TESTNET
        return BINANCE_WS_BASE

    # ------------------------------------------------------------------
    # handlers registration
    # ------------------------------------------------------------------

    def on_ticker(self, handler: Callable) -> None:
        self._on_ticker = handler

    def on_kline(self, handler: Callable) -> None:
        self._on_kline = handler

    def on_depth(self, handler: Callable) -> None:
        self._on_depth = handler

    def on_agg_trade(self, handler: Callable) -> None:
        self._on_agg_trade = handler

    def on_liquidation(self, handler: Callable) -> None:
        self._on_liquidation = handler

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self, stream_batches: List[List[str]]) -> None:
        """Start WebSocket connections for each stream batch."""
        self._running = True

        for i, batch in enumerate(stream_batches):
            task = asyncio.create_task(
                self._run_connection(i, batch),
                name=f"ws-conn-{i}",
            )
            self._connections.append(task)
            self._last_message_time[i] = time.time()

        logger.info(
            "websocket_started",
            connections=len(stream_batches),
            total_streams=sum(len(b) for b in stream_batches),
        )

        # start stale checker
        asyncio.create_task(self._stale_checker())

    async def stop(self) -> None:
        """Stop all WebSocket connections."""
        self._running = False
        for task in self._connections:
            task.cancel()
        await asyncio.gather(*self._connections, return_exceptions=True)
        self._connections.clear()
        logger.info("websocket_stopped")

    # ------------------------------------------------------------------
    # connection management
    # ------------------------------------------------------------------

    async def _run_connection(self, conn_id: int, streams: List[str]) -> None:
        """Run a single WebSocket connection with auto-reconnect."""
        delay = self.settings.websocket.reconnect_delay_seconds
        max_delay = 60

        while self._running:
            try:
                url = f"{self.base_url}/stream?streams={'/'.join(streams)}"
                logger.info("ws_connecting", conn_id=conn_id, streams=len(streams))

                async with websockets.connect(
                    url,
                    ping_interval=self.settings.websocket.ping_interval_seconds,
                    ping_timeout=30,
                    max_size=10 * 1024 * 1024,  # 10MB
                    close_timeout=5,
                ) as ws:
                    delay = self.settings.websocket.reconnect_delay_seconds  # reset
                    logger.info("ws_connected", conn_id=conn_id)

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        self._last_message_time[conn_id] = time.time()
                        self._msg_count += 1

                        try:
                            msg = json.loads(raw_msg)
                            await self._dispatch(msg)
                        except Exception as e:
                            self._error_count += 1
                            if self._error_count % 100 == 1:
                                logger.warning("ws_parse_error", error=str(e))

            except ConnectionClosed as e:
                self._reconnect_count += 1
                logger.warning(
                    "ws_disconnected",
                    conn_id=conn_id,
                    code=e.code,
                    reason=str(e.reason)[:100],
                    reconnect_in=delay,
                )
            except Exception as e:
                self._reconnect_count += 1
                logger.error("ws_error", conn_id=conn_id, error=str(e), reconnect_in=delay)

            if self._running:
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

    async def _stale_checker(self) -> None:
        """Check for stale connections and log warnings."""
        timeout = self.settings.websocket.stale_timeout_seconds
        while self._running:
            await asyncio.sleep(30)
            now = time.time()
            for conn_id, last_time in self._last_message_time.items():
                if now - last_time > timeout:
                    logger.warning(
                        "ws_stale_connection",
                        conn_id=conn_id,
                        seconds_since_last=int(now - last_time),
                    )

    # ------------------------------------------------------------------
    # message dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, msg: dict) -> None:
        """Route incoming WebSocket message to the appropriate handler."""
        stream = msg.get("stream", "")
        data = msg.get("data", {})

        if not stream or not data:
            return

        event_type = data.get("e", "")

        try:
            if "miniTicker" in stream or event_type == "24hrMiniTicker":
                await self._handle_mini_ticker(data)

            elif "kline" in stream or event_type == "kline":
                await self._handle_kline(data)

            elif "depth" in stream or event_type == "depthUpdate":
                await self._handle_depth(data, stream)

            elif "aggTrade" in stream or event_type == "aggTrade":
                await self._handle_agg_trade(data)

            elif "forceOrder" in stream or event_type == "forceOrder":
                await self._handle_liquidation(data)

        except Exception as e:
            self._error_count += 1
            if self._error_count % 500 == 1:
                logger.warning("ws_handler_error", event=event_type, error=str(e))

    # ------------------------------------------------------------------
    # handlers
    # ------------------------------------------------------------------

    async def _handle_mini_ticker(self, data: dict) -> None:
        """Process 24hr mini ticker update."""
        symbol = data.get("s", "")
        if not symbol:
            return

        price = float(data.get("c", 0))  # close price
        open_price = float(data.get("o", 0))
        high = float(data.get("h", 0))
        low = float(data.get("l", 0))
        volume = float(data.get("q", 0))  # quote volume 24h

        change_pct = ((price - open_price) / open_price * 100) if open_price else 0

        # update Redis cache
        await self.redis.set_price(symbol, price, change_pct)
        await self.redis.set_ticker(symbol, {
            "symbol": symbol,
            "price": price,
            "open": open_price,
            "high": high,
            "low": low,
            "volume_24h": volume,
            "change_pct": change_pct,
            "timestamp": data.get("E", 0),
        })

        if self._on_ticker:
            await self._on_ticker(symbol, data)

    async def _handle_kline(self, data: dict) -> None:
        """Process kline/candlestick update."""
        k = data.get("k", {})
        if not k:
            return

        symbol = k.get("s", "")
        is_closed = k.get("x", False)

        if is_closed and symbol:
            import datetime as dt
            ts = dt.datetime.utcfromtimestamp(k["t"] / 1000)
            self.timeseries.append_candle(
                symbol=symbol,
                exchange="binance",
                timestamp=ts,
                open_=float(k.get("o", 0)),
                high=float(k.get("h", 0)),
                low=float(k.get("l", 0)),
                close=float(k.get("c", 0)),
                volume_base=float(k.get("v", 0)),
                volume_quote=float(k.get("q", 0)),
            )

        if self._on_kline:
            await self._on_kline(symbol, k, is_closed)

    async def _handle_depth(self, data: dict, stream: str) -> None:
        """Process orderbook depth snapshot."""
        # extract symbol from stream name (e.g., btcusdt@depth20@100ms)
        parts = stream.split("@")
        symbol = parts[0].upper() if parts else ""
        if not symbol:
            return

        bids = data.get("b", data.get("bids", []))
        asks = data.get("a", data.get("asks", []))

        # calculate depth in USD
        bids_usd = sum(float(b[0]) * float(b[1]) for b in bids) if bids else 0
        asks_usd = sum(float(a[0]) * float(a[1]) for a in asks) if asks else 0

        await self.redis.set_depth(symbol, bids_usd, asks_usd)

        if self._on_depth:
            await self._on_depth(symbol, bids, asks, bids_usd, asks_usd)

    async def _handle_agg_trade(self, data: dict) -> None:
        """Process aggregate trade — detect whale trades."""
        symbol = data.get("s", "")
        price = float(data.get("p", 0))
        qty = float(data.get("q", 0))
        is_buyer_maker = data.get("m", False)  # True = sell, False = buy

        trade_value = price * qty

        # whale threshold: $50K+ USD
        if trade_value >= 50_000:
            key = f"ph:whale:{symbol}"
            existing = await self.redis.get_json(key) or {
                "buy_volume": 0, "sell_volume": 0, "count": 0,
            }

            if is_buyer_maker:
                existing["sell_volume"] += trade_value
            else:
                existing["buy_volume"] += trade_value
            existing["count"] += 1

            await self.redis.set_json(key, existing, ttl=300)

        if self._on_agg_trade:
            await self._on_agg_trade(symbol, price, qty, trade_value, is_buyer_maker)

    async def _handle_liquidation(self, data: dict) -> None:
        """Process forced liquidation order."""
        order = data.get("o", data)
        symbol = order.get("s", "")
        side = order.get("S", "")  # BUY = short liq, SELL = long liq
        price = float(order.get("p", 0))
        qty = float(order.get("q", 0))
        liq_value = price * qty

        key = f"ph:liq:{symbol}"
        existing = await self.redis.get_json(key) or {
            "buy_volume": 0, "sell_volume": 0, "count": 0,
        }

        if side == "BUY":  # short liquidation
            existing["buy_volume"] += liq_value
        else:  # long liquidation
            existing["sell_volume"] += liq_value
        existing["count"] += 1

        await self.redis.set_json(key, existing, ttl=300)

        if self._on_liquidation:
            await self._on_liquidation(symbol, side, price, qty, liq_value)

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "connections": len(self._connections),
            "messages_total": self._msg_count,
            "errors_total": self._error_count,
            "reconnects_total": self._reconnect_count,
            "running": self._running,
        }
