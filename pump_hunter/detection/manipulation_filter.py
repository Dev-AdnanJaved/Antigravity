"""
Manipulation filter — detects adversarial patterns and adjusts scores downward.

Sub-filters:
    1. Orderbook Spoof Detection    (depth cancel rate)
    2. Fake Breakout Detection      (price revert after level break)
    3. Pump-and-Dump Pattern        (spike w/o OI support)
    4. Thin Book Trap               (tiny depth + huge volume)
    5. Coordinated Activity         (multi-coin simultaneous pump)
    6. Reversal Probability         (RSI + exhaustion)
"""

from __future__ import annotations

import datetime as dt
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from pump_hunter.config.settings import Settings
from pump_hunter.core.market_state import SymbolState
from pump_hunter.storage.redis_store import RedisStore
from pump_hunter.storage.timeseries import TimeseriesStore

logger = structlog.get_logger(__name__)


class ManipulationFilter:
    """Detects adversarial market patterns and returns penalty multipliers."""

    def __init__(
        self,
        settings: Settings,
        redis: RedisStore,
        timeseries: TimeseriesStore,
    ):
        self.cfg = settings.manipulation
        self.redis = redis
        self.ts = timeseries

        # depth snapshot history for spoof detection: {symbol: [(time, bids, asks)]}
        self._depth_history: Dict[str, list] = defaultdict(list)
        self._max_depth_snapshots = 60  # keep last 60 snapshots

        # track recent alerts for coordinated detection
        self._recent_alerts: List[Tuple[float, str]] = []  # (time, symbol)

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        state: SymbolState,
        signals: Dict[str, Optional[float]],
        base_score: float,
    ) -> Tuple[float, Dict[str, dict]]:
        """
        Evaluate all manipulation filters.

        Returns:
            (adjusted_score, filter_results)
        """
        if not self.cfg.enabled:
            return base_score, {}

        multiplier = 1.0
        results = {}

        # 1. Spoof detection
        try:
            spoof = self._check_spoof(symbol, state)
            if spoof["detected"]:
                multiplier *= 0.5
                results["spoof"] = spoof
        except Exception:
            pass

        # 2. Fake breakout
        try:
            fake_bo = self._check_fake_breakout(symbol)
            if fake_bo["detected"]:
                multiplier *= 0.4
                results["fake_breakout"] = fake_bo
        except Exception:
            pass

        # 3. Pump-and-dump pattern
        try:
            pnd = self._check_pump_dump(symbol, state, signals)
            if pnd["detected"]:
                multiplier *= 0.3
                results["pump_dump"] = pnd
        except Exception:
            pass

        # 4. Thin book trap
        try:
            thin = self._check_thin_book(state)
            if thin["detected"]:
                multiplier *= 0.5
                results["thin_book"] = thin
        except Exception:
            pass

        # 5. Coordinated activity
        try:
            coord = self._check_coordinated(symbol, base_score)
            if coord["detected"]:
                multiplier *= 0.6
                results["coordinated"] = coord
        except Exception:
            pass

        # 6. Reversal probability
        try:
            reversal = self._check_reversal(symbol, state)
            if reversal["probability"] > 0.3:
                multiplier *= (1.0 - reversal["probability"])
                results["reversal"] = reversal
        except Exception:
            pass

        adjusted = base_score * max(multiplier, 0.1)  # floor at 10% of original

        if results:
            logger.info(
                "manipulation_filters_triggered",
                symbol=symbol,
                original_score=round(base_score, 1),
                adjusted_score=round(adjusted, 1),
                multiplier=round(multiplier, 3),
                filters=list(results.keys()),
            )

        return adjusted, results

    # ------------------------------------------------------------------
    # 1. orderbook spoof detection
    # ------------------------------------------------------------------

    def record_depth_snapshot(
        self, symbol: str, bid_depth_usd: float, ask_depth_usd: float
    ) -> None:
        """Record a depth snapshot for spoof analysis. Call every tick."""
        history = self._depth_history[symbol]
        history.append((time.time(), bid_depth_usd, ask_depth_usd))
        if len(history) > self._max_depth_snapshots:
            self._depth_history[symbol] = history[-self._max_depth_snapshots:]

    def _check_spoof(self, symbol: str, state: SymbolState) -> dict:
        """Detect orderbook spoofing via rapid depth changes."""
        history = self._depth_history.get(symbol, [])
        window = self.cfg.spoof_check_window_seconds
        threshold = self.cfg.spoof_cancel_rate_threshold

        if len(history) < 5:
            return {"detected": False}

        now = time.time()
        recent = [h for h in history if now - h[0] <= window]

        if len(recent) < 3:
            return {"detected": False}

        # check bid-side volatility (spoofing usually on bids)
        bid_values = [h[1] for h in recent]
        if not bid_values or max(bid_values) == 0:
            return {"detected": False}

        # cancel rate: how much depth disappears between snapshots
        changes = []
        for i in range(1, len(bid_values)):
            if bid_values[i - 1] > 0:
                change_rate = abs(bid_values[i] - bid_values[i - 1]) / bid_values[i - 1]
                changes.append(change_rate)

        if not changes:
            return {"detected": False}

        avg_change = float(np.mean(changes))
        max_change = float(np.max(changes))

        detected = avg_change > threshold or max_change > 0.9

        return {
            "detected": detected,
            "avg_change_rate": round(avg_change, 3),
            "max_change_rate": round(max_change, 3),
            "snapshots_checked": len(recent),
        }

    # ------------------------------------------------------------------
    # 2. fake breakout detection
    # ------------------------------------------------------------------

    def _check_fake_breakout(self, symbol: str) -> dict:
        """Detect price breaking a level then reverting within N candles."""
        df = self.ts.get_dataframe(symbol, "binance", last_n=50)
        revert_candles = self.cfg.fake_breakout_revert_candles
        revert_pct = self.cfg.fake_breakout_revert_pct / 100

        if len(df) < revert_candles + 20:
            return {"detected": False}

        closes = df["close"].values
        highs = df["high"].values

        # find recent 20-candle high (resistance)
        lookback_end = -(revert_candles + 1)
        resistance = float(np.max(highs[:lookback_end][-20:]))

        if resistance <= 0:
            return {"detected": False}

        # check if price broke above resistance then came back
        recent = closes[-(revert_candles + 1):]
        broke_above = any(c > resistance * 1.001 for c in recent[:-1])
        current_below = recent[-1] < resistance

        # how much of the breakout was retraced
        if broke_above and current_below:
            peak = float(np.max(recent[:-1]))
            breakout_size = peak - resistance
            retrace_size = peak - recent[-1]

            if breakout_size > 0:
                retrace_ratio = retrace_size / breakout_size
                detected = retrace_ratio >= revert_pct

                return {
                    "detected": detected,
                    "resistance": round(resistance, 4),
                    "peak": round(peak, 4),
                    "current": round(float(recent[-1]), 4),
                    "retrace_pct": round(retrace_ratio * 100, 1),
                }

        return {"detected": False}

    # ------------------------------------------------------------------
    # 3. pump-and-dump pattern
    # ------------------------------------------------------------------

    def _check_pump_dump(
        self, symbol: str, state: SymbolState, signals: Dict[str, Optional[float]]
    ) -> dict:
        """
        Detect >20% price spike with disproportionately low OI increase.
        Real pumps have OI expanding; pump-and-dumps don't.
        """
        df = self.ts.get_dataframe(symbol, "binance", last_n=60)
        oi_threshold = self.cfg.pump_dump_oi_threshold

        if len(df) < 20:
            return {"detected": False}

        closes = df["close"].values

        # check for >20% spike in last 60 candles
        min_price = float(np.min(closes))
        max_price = float(np.max(closes))
        spike_pct = (max_price - min_price) / min_price * 100 if min_price > 0 else 0

        if spike_pct < 20:
            return {"detected": False}

        # check OI change
        oi_change = state.oi_change_pct if hasattr(state, "oi_change_pct") else 0
        oi_pct = abs(oi_change) / 100  # normalize

        # pump-dump: big price move but OI didn't grow proportionally
        oi_ratio = oi_pct / (spike_pct / 100) if spike_pct > 0 else 1.0

        detected = oi_ratio < oi_threshold

        return {
            "detected": detected,
            "spike_pct": round(spike_pct, 1),
            "oi_change_pct": round(oi_change, 1),
            "oi_ratio": round(oi_ratio, 3),
            "reason": "price spike without OI expansion" if detected else "",
        }

    # ------------------------------------------------------------------
    # 4. thin book trap
    # ------------------------------------------------------------------

    def _check_thin_book(self, state: SymbolState) -> dict:
        """Detect when orderbook is too thin for the volume being traded."""
        ask_depth = state.ask_depth_usd if hasattr(state, "ask_depth_usd") else 0
        volume_24h = state.volume_24h if hasattr(state, "volume_24h") else 0
        threshold = self.cfg.thin_book_threshold_usd

        if ask_depth <= 0:
            return {"detected": False}

        detected = ask_depth < threshold and volume_24h > threshold * 10

        return {
            "detected": detected,
            "ask_depth_usd": round(ask_depth, 0),
            "volume_24h": round(volume_24h, 0),
            "threshold": threshold,
            "reason": "thin orderbook with high volume" if detected else "",
        }

    # ------------------------------------------------------------------
    # 5. coordinated activity
    # ------------------------------------------------------------------

    def _check_coordinated(self, symbol: str, score: float) -> dict:
        """
        Detect if multiple low-cap coins are pumping simultaneously,
        which suggests coordinated group activity.
        """
        now = time.time()
        window = self.cfg.coordinated_window_minutes * 60
        min_coins = self.cfg.coordinated_min_coins

        # record this symbol's alert
        if score >= 60:
            self._recent_alerts.append((now, symbol))

        # prune old alerts
        self._recent_alerts = [
            (t, s) for t, s in self._recent_alerts if now - t <= window
        ]

        # count unique symbols alerting in window
        recent_symbols = set(s for t, s in self._recent_alerts if now - t <= window)
        recent_symbols.discard(symbol)  # exclude current

        # exclude large-cap (BTC, ETH, etc.) — they don't count
        large_caps = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"}
        small_cap_alerts = recent_symbols - large_caps

        detected = len(small_cap_alerts) >= min_coins

        return {
            "detected": detected,
            "concurrent_alerts": len(small_cap_alerts),
            "symbols": list(small_cap_alerts)[:5] if detected else [],
            "window_minutes": self.cfg.coordinated_window_minutes,
        }

    # ------------------------------------------------------------------
    # 6. reversal probability
    # ------------------------------------------------------------------

    def _check_reversal(self, symbol: str, state: SymbolState) -> dict:
        """
        Estimate probability of reversal using:
        - RSI overbought
        - Volume exhaustion (declining volume on up-move)
        - Distance from VWAP
        """
        df = self.ts.get_dataframe(symbol, "binance", last_n=50)
        if len(df) < 20:
            return {"detected": False, "probability": 0.0}

        closes = df["close"].values
        volumes = df["volume_quote"].values
        rsi_threshold = self.cfg.reversal_rsi_threshold
        vol_exhaust_pct = self.cfg.reversal_volume_exhaustion_pct / 100

        probability = 0.0

        # RSI check
        if len(closes) > 15:
            deltas = np.diff(closes[-15:])
            gains = np.maximum(deltas, 0)
            losses = np.abs(np.minimum(deltas, 0))
            avg_gain = float(np.mean(gains))
            avg_loss = float(np.mean(losses)) + 0.0001
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            if rsi >= rsi_threshold:
                probability += 0.35
            elif rsi >= 70:
                probability += 0.15

        # volume exhaustion: recent volume declining while price rising
        if len(closes) > 10 and len(volumes) > 10:
            price_up = closes[-1] > closes[-10]
            vol_declining = np.mean(volumes[-5:]) < np.mean(volumes[-10:-5]) * vol_exhaust_pct

            if price_up and vol_declining:
                probability += 0.3

        # extreme distance from VWAP
        if len(volumes) > 20 and len(closes) > 20:
            vwap = np.sum(closes[-20:] * volumes[-20:]) / np.sum(volumes[-20:]) if np.sum(volumes[-20:]) > 0 else closes[-1]
            deviation = (closes[-1] - vwap) / vwap * 100 if vwap > 0 else 0

            if deviation > 5:
                probability += 0.2
            elif deviation > 3:
                probability += 0.1

        # bearish divergence: higher price but lower RSI (simplified)
        if len(closes) > 30:
            price_higher = closes[-1] > closes[-15]
            # compare recent momentum vs prior momentum
            recent_mom = float(np.mean(np.diff(closes[-5:])))
            prior_mom = float(np.mean(np.diff(closes[-15:-10])))
            momentum_declining = recent_mom < prior_mom * 0.5

            if price_higher and momentum_declining:
                probability += 0.15

        probability = min(probability, 0.95)

        return {
            "detected": probability > 0.3,
            "probability": round(probability, 3),
        }
