"""
In-memory ring buffers for recent OHLCV data.
Numpy-backed for fast signal computation, with pandas DataFrame views.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class CandleBuffer:
    """Ring buffer for OHLCV candles of a single symbol on a single exchange."""

    __slots__ = ("symbol", "exchange", "max_size", "_data", "_timestamps", "_count", "_head")

    # columns: open, high, low, close, volume_base, volume_quote
    COLS = 6

    def __init__(self, symbol: str, exchange: str, max_size: int = 500):
        self.symbol = symbol
        self.exchange = exchange
        self.max_size = max_size
        self._data = np.zeros((max_size, self.COLS), dtype=np.float64)
        self._timestamps = np.zeros(max_size, dtype="datetime64[ms]")
        self._count = 0
        self._head = 0

    @property
    def length(self) -> int:
        return min(self._count, self.max_size)

    def append(
        self,
        timestamp: dt.datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume_base: float,
        volume_quote: float = 0.0,
    ) -> None:
        """Add a new candle to the buffer."""
        idx = self._head % self.max_size
        self._data[idx] = [open_, high, low, close, volume_base, volume_quote]
        self._timestamps[idx] = np.datetime64(timestamp, "ms")
        self._head += 1
        self._count += 1

    def append_raw(self, timestamp_ms: int, ohlcv: list) -> None:
        """Add candle from raw Binance format [open, high, low, close, volume]."""
        if len(ohlcv) < 5:
            return
        ts = dt.datetime.utcfromtimestamp(timestamp_ms / 1000)
        o, h, l, c, v = float(ohlcv[0]), float(ohlcv[1]), float(ohlcv[2]), float(ohlcv[3]), float(ohlcv[4])
        vq = v * c  # approximate quote volume
        self.append(ts, o, h, l, c, v, vq)

    def get_array(self, last_n: int | None = None) -> np.ndarray:
        """Get ordered numpy array of candles (oldest first)."""
        length = self.length
        if length == 0:
            return np.empty((0, self.COLS))

        if self._count <= self.max_size:
            arr = self._data[:length]
        else:
            start = self._head % self.max_size
            arr = np.concatenate([self._data[start:], self._data[:start]])

        if last_n and last_n < len(arr):
            arr = arr[-last_n:]
        return arr

    def get_timestamps(self, last_n: int | None = None) -> np.ndarray:
        """Get ordered timestamp array."""
        length = self.length
        if length == 0:
            return np.empty(0, dtype="datetime64[ms]")

        if self._count <= self.max_size:
            arr = self._timestamps[:length]
        else:
            start = self._head % self.max_size
            arr = np.concatenate([self._timestamps[start:], self._timestamps[:start]])

        if last_n and last_n < len(arr):
            arr = arr[-last_n:]
        return arr

    def to_dataframe(self, last_n: int | None = None) -> pd.DataFrame:
        """Convert to pandas DataFrame with column names."""
        arr = self.get_array(last_n)
        ts = self.get_timestamps(last_n)
        df = pd.DataFrame(
            arr,
            columns=["open", "high", "low", "close", "volume_base", "volume_quote"],
        )
        if len(ts) > 0:
            df.index = pd.DatetimeIndex(ts, name="timestamp")
        return df

    @property
    def latest_price(self) -> float:
        if self.length == 0:
            return 0.0
        idx = (self._head - 1) % self.max_size
        return float(self._data[idx, 3])  # close

    @property
    def latest_volume(self) -> float:
        if self.length == 0:
            return 0.0
        idx = (self._head - 1) % self.max_size
        return float(self._data[idx, 5])  # volume_quote


class TimeseriesStore:
    """
    Manages CandleBuffers for all active symbols across exchanges.
    Provides fast access for signal computation.
    """

    def __init__(self, max_candles: int = 500):
        self.max_candles = max_candles
        # (symbol, exchange) -> CandleBuffer
        self._buffers: Dict[Tuple[str, str], CandleBuffer] = {}

    def get_buffer(self, symbol: str, exchange: str = "binance") -> CandleBuffer:
        """Get or create a candle buffer for a symbol/exchange pair."""
        key = (symbol, exchange)
        if key not in self._buffers:
            self._buffers[key] = CandleBuffer(symbol, exchange, self.max_candles)
        return self._buffers[key]

    def append_candle(
        self,
        symbol: str,
        exchange: str,
        timestamp: dt.datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume_base: float,
        volume_quote: float = 0.0,
    ) -> None:
        """Append a candle to the appropriate buffer."""
        buf = self.get_buffer(symbol, exchange)
        buf.append(timestamp, open_, high, low, close, volume_base, volume_quote)

    def get_dataframe(
        self, symbol: str, exchange: str = "binance", last_n: int | None = None
    ) -> pd.DataFrame:
        """Get a DataFrame of candles for a symbol."""
        buf = self.get_buffer(symbol, exchange)
        return buf.to_dataframe(last_n)

    def get_closes(
        self, symbol: str, exchange: str = "binance", last_n: int | None = None
    ) -> np.ndarray:
        """Get close prices as numpy array (for fast signal computation)."""
        buf = self.get_buffer(symbol, exchange)
        arr = buf.get_array(last_n)
        if len(arr) == 0:
            return np.empty(0)
        return arr[:, 3]

    def get_volumes(
        self, symbol: str, exchange: str = "binance", last_n: int | None = None
    ) -> np.ndarray:
        """Get quote volumes as numpy array."""
        buf = self.get_buffer(symbol, exchange)
        arr = buf.get_array(last_n)
        if len(arr) == 0:
            return np.empty(0)
        return arr[:, 5]

    def get_latest_price(self, symbol: str, exchange: str = "binance") -> float:
        """Quick access to latest close price."""
        key = (symbol, exchange)
        if key not in self._buffers:
            return 0.0
        return self._buffers[key].latest_price

    def get_all_symbols(self) -> List[str]:
        """Get all tracked symbols."""
        return list(set(k[0] for k in self._buffers.keys()))

    def remove_symbol(self, symbol: str) -> None:
        """Remove all buffers for a symbol."""
        keys = [k for k in self._buffers if k[0] == symbol]
        for k in keys:
            del self._buffers[k]

    def get_stats(self) -> dict:
        """Return buffer statistics."""
        total_buffers = len(self._buffers)
        total_candles = sum(b.length for b in self._buffers.values())
        return {
            "total_buffers": total_buffers,
            "total_candles": total_candles,
            "symbols_tracked": len(self.get_all_symbols()),
        }
