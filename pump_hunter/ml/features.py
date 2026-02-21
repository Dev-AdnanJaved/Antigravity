"""
ML features — 50+ features extracted from market state for pump prediction.
Organized by category: price, volume, OI, orderbook, funding, cross-exchange, temporal.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

from pump_hunter.storage.timeseries import TimeseriesStore

logger = structlog.get_logger(__name__)


class MLFeatureExtractor:
    """Extracts ML features from market state and candle history."""

    def __init__(self, timeseries: TimeseriesStore, window_minutes: int = 60):
        self.ts = timeseries
        self.window = window_minutes

    def extract(self, symbol: str, state_dict: dict) -> Dict[str, float]:
        """Extract all features for a symbol at current time."""
        features = {}

        df = self.ts.get_dataframe(symbol, "binance", last_n=200)
        if len(df) < 20:
            return features

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        volumes = df["volume_quote"].values

        # === PRICE FEATURES ===
        features.update(self._price_features(closes))

        # === VOLUME FEATURES ===
        features.update(self._volume_features(volumes, closes))

        # === VOLATILITY FEATURES ===
        features.update(self._volatility_features(closes, highs, lows))

        # === OI / FUNDING / ORDERBOOK from state ===
        features.update(self._state_features(state_dict))

        # === TEMPORAL FEATURES ===
        features.update(self._temporal_features())

        return features

    def _price_features(self, closes: np.ndarray) -> dict:
        """Price-based features: returns, RSI, momentum."""
        f = {}
        if len(closes) < 2:
            return f

        # returns at various windows
        for period in [1, 5, 15, 30, 60]:
            if len(closes) > period:
                ret = (closes[-1] - closes[-period]) / closes[-period] * 100
                f[f"return_{period}m"] = ret

        # RSI (14-period)
        if len(closes) > 15:
            deltas = np.diff(closes[-15:])
            gains = np.maximum(deltas, 0)
            losses = np.abs(np.minimum(deltas, 0))
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.0001
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            f["rsi_14"] = 100 - (100 / (1 + rs))

        # rate of change of rate of change (acceleration)
        if len(closes) > 10:
            roc5 = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0
            roc10 = (closes[-5] - closes[-10]) / closes[-10] if closes[-10] > 0 else 0
            f["price_acceleration"] = roc5 - roc10

        # distance from high/low
        if len(closes) > 20:
            high_20 = np.max(closes[-20:])
            low_20 = np.min(closes[-20:])
            rng = high_20 - low_20
            if rng > 0:
                f["pct_from_20_high"] = (closes[-1] - high_20) / high_20 * 100
                f["pct_from_20_low"] = (closes[-1] - low_20) / low_20 * 100
                f["position_in_range"] = (closes[-1] - low_20) / rng

        return f

    def _volume_features(self, volumes: np.ndarray, closes: np.ndarray) -> dict:
        """Volume-based features."""
        f = {}
        if len(volumes) < 20:
            return f

        recent_vol = np.mean(volumes[-5:])
        avg_vol = np.mean(volumes[-60:]) if len(volumes) >= 60 else np.mean(volumes)

        f["vol_ratio_5_60"] = recent_vol / avg_vol if avg_vol > 0 else 1.0
        f["vol_change_5m"] = (
            (np.sum(volumes[-5:]) - np.sum(volumes[-10:-5])) / np.sum(volumes[-10:-5]) * 100
            if np.sum(volumes[-10:-5]) > 0 else 0
        )

        # VWAP deviation
        if len(volumes) > 20 and len(closes) > 20:
            vwap = np.sum(closes[-20:] * volumes[-20:]) / np.sum(volumes[-20:]) if np.sum(volumes[-20:]) > 0 else closes[-1]
            f["vwap_deviation_pct"] = (closes[-1] - vwap) / vwap * 100 if vwap > 0 else 0

        # volume profile — concentration
        if len(volumes) > 50:
            vol_std = np.std(volumes[-50:])
            vol_mean = np.mean(volumes[-50:])
            f["vol_coefficient_var"] = vol_std / vol_mean if vol_mean > 0 else 0

        return f

    def _volatility_features(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> dict:
        """Volatility features: ATR, BB width, Parkinson."""
        f = {}
        if len(closes) < 20:
            return f

        # Bollinger Band width
        mean = np.mean(closes[-20:])
        std = np.std(closes[-20:])
        f["bb_width"] = (2 * std) / mean * 100 if mean > 0 else 0

        # ATR (14-period)
        if len(closes) > 15:
            tr = np.maximum(
                highs[-14:] - lows[-14:],
                np.maximum(
                    np.abs(highs[-14:] - closes[-15:-1]),
                    np.abs(lows[-14:] - closes[-15:-1]),
                ),
            )
            f["atr_14"] = float(np.mean(tr))
            f["atr_pct"] = f["atr_14"] / closes[-1] * 100 if closes[-1] > 0 else 0

        # Parkinson volatility (uses high-low)
        if len(highs) > 20 and len(lows) > 20:
            hl_ratio = np.log(highs[-20:] / lows[-20:])
            f["parkinson_vol"] = float(np.sqrt(np.mean(hl_ratio ** 2) / (4 * np.log(2))))

        # realized volatility
        if len(closes) > 20:
            log_returns = np.diff(np.log(closes[-21:]))
            f["realized_vol_20"] = float(np.std(log_returns) * np.sqrt(1440))  # annualize

        return f

    def _state_features(self, state: dict) -> dict:
        """Features from current market state."""
        f = {}
        f["oi_value"] = state.get("oi_value", 0)
        f["oi_change_pct"] = state.get("oi_change_pct", 0)
        f["funding_rate"] = state.get("funding_rate", 0) * 10000  # basis points
        f["long_short_ratio"] = state.get("long_short_ratio", 1.0)
        f["depth_imbalance"] = state.get("depth_imbalance", 1.0)
        f["bid_depth_usd"] = state.get("bid_depth_usd", 0)
        f["ask_depth_usd"] = state.get("ask_depth_usd", 0)
        f["whale_buy_vol"] = state.get("whale_buy_vol", 0)
        f["whale_sell_vol"] = state.get("whale_sell_vol", 0)
        f["whale_net"] = state.get("whale_buy_vol", 0) - state.get("whale_sell_vol", 0)
        f["liq_buy_vol"] = state.get("liq_buy_vol", 0)
        f["liq_sell_vol"] = state.get("liq_sell_vol", 0)
        f["volume_24h"] = state.get("volume_24h", 0)
        f["spread_pct"] = state.get("spread_pct", 0)
        return f

    def _temporal_features(self) -> dict:
        """Time-based features."""
        now = dt.datetime.utcnow()
        f = {}
        f["hour_of_day"] = now.hour
        f["day_of_week"] = now.weekday()
        f["is_weekend"] = 1 if now.weekday() >= 5 else 0
        # crypto tends to pump in specific hours
        f["is_asia_session"] = 1 if 0 <= now.hour < 8 else 0
        f["is_europe_session"] = 1 if 8 <= now.hour < 16 else 0
        f["is_us_session"] = 1 if 16 <= now.hour < 24 else 0
        return f


# feature name list for consistent ordering
FEATURE_NAMES = [
    "return_1m", "return_5m", "return_15m", "return_30m", "return_60m",
    "rsi_14", "price_acceleration", "pct_from_20_high", "pct_from_20_low", "position_in_range",
    "vol_ratio_5_60", "vol_change_5m", "vwap_deviation_pct", "vol_coefficient_var",
    "bb_width", "atr_14", "atr_pct", "parkinson_vol", "realized_vol_20",
    "oi_value", "oi_change_pct", "funding_rate", "long_short_ratio",
    "depth_imbalance", "bid_depth_usd", "ask_depth_usd",
    "whale_buy_vol", "whale_sell_vol", "whale_net",
    "liq_buy_vol", "liq_sell_vol",
    "volume_24h", "spread_pct",
    "hour_of_day", "day_of_week", "is_weekend",
    "is_asia_session", "is_europe_session", "is_us_session",
]
