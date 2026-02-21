"""
Market regime engine — classifies current market conditions to adjust scoring.

Regimes:
    RISK_ON           Trending up, low vol   → pumps succeed → score ×1.2
    RISK_OFF          Drawdown, high vol     → pumps fail    → score ×0.6
    VOLATILITY_EXP    BTC breakout/expansion → uncertain     → score ×0.8
    LOW_LIQUIDITY     Weekends/holidays       → thin books    → score ×0.7
    NEUTRAL           Normal conditions       → no adjustment → score ×1.0

Uses BTC as the market-wide anchor.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from enum import Enum
from typing import Dict, Optional

import numpy as np
import structlog

from pump_hunter.config.settings import Settings
from pump_hunter.storage.redis_store import RedisStore
from pump_hunter.storage.timeseries import TimeseriesStore

logger = structlog.get_logger(__name__)


class MarketRegime(str, Enum):
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    NEUTRAL = "NEUTRAL"


class RegimeEngine:
    """Real-time market regime classifier using BTC as anchor."""

    def __init__(
        self,
        settings: Settings,
        redis: RedisStore,
        timeseries: TimeseriesStore,
    ):
        self.settings = settings
        self.cfg = settings.regime
        self.redis = redis
        self.ts = timeseries
        self._btc = self.cfg.btc_symbol

        self._regime = MarketRegime.NEUTRAL
        self._confidence = 0.0
        self._last_update = 0.0
        self._regime_history: list = []

        # multiplier map
        self._multipliers = {
            MarketRegime.RISK_ON: self.cfg.risk_on_multiplier,
            MarketRegime.RISK_OFF: self.cfg.risk_off_multiplier,
            MarketRegime.NEUTRAL: self.cfg.neutral_multiplier,
            MarketRegime.VOLATILITY_EXPANSION: self.cfg.volatility_expansion_multiplier,
            MarketRegime.LOW_LIQUIDITY: self.cfg.low_liquidity_multiplier,
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def current_regime(self) -> MarketRegime:
        return self._regime

    @property
    def confidence(self) -> float:
        return self._confidence

    def get_modifier(self) -> float:
        """Return the score multiplier for current regime."""
        if not self.cfg.enabled:
            return 1.0
        return self._multipliers.get(self._regime, 1.0)

    def get_regime_info(self) -> dict:
        """Return full regime state for logging/alerts."""
        return {
            "regime": self._regime.value,
            "confidence": round(self._confidence, 2),
            "modifier": self.get_modifier(),
            "last_update": self._last_update,
        }

    # ------------------------------------------------------------------
    # update cycle
    # ------------------------------------------------------------------

    async def update(self) -> MarketRegime:
        """Re-classify regime. Call periodically (every 5 min)."""
        now = time.time()
        if now - self._last_update < self.cfg.regime_update_interval_seconds:
            return self._regime

        try:
            btc_metrics = await self._compute_btc_metrics()
            temporal = self._compute_temporal_features()
            funding = await self._get_funding_index()

            regime, confidence = self._classify(btc_metrics, temporal, funding)
            prev = self._regime
            self._regime = regime
            self._confidence = confidence
            self._last_update = now

            self._regime_history.append({
                "time": now, "regime": regime.value, "confidence": confidence,
            })
            # keep last 288 entries (~24h at 5-min intervals)
            if len(self._regime_history) > 288:
                self._regime_history = self._regime_history[-288:]

            if regime != prev:
                logger.info(
                    "regime_change",
                    from_regime=prev.value,
                    to_regime=regime.value,
                    confidence=f"{confidence:.0%}",
                    modifier=self.get_modifier(),
                )

            # cache in Redis for other components
            await self.redis.redis.set(
                "ph:regime:current", regime.value, ex=600,
            )

        except Exception as e:
            logger.debug("regime_update_error", error=str(e))

        return self._regime

    # ------------------------------------------------------------------
    # BTC metrics
    # ------------------------------------------------------------------

    async def _compute_btc_metrics(self) -> dict:
        """Compute BTC trend, volatility, and momentum."""
        df = self.ts.get_dataframe(self._btc, "binance", last_n=200)
        if len(df) < 20:
            return {}

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        lookback = min(self.cfg.btc_trend_lookback_hours * 60, len(closes))
        vol_lookback = min(self.cfg.btc_volatility_lookback, len(closes))

        # trend: return over lookback
        ret = (closes[-1] - closes[-lookback]) / closes[-lookback] * 100 if lookback > 0 else 0

        # ATR-based volatility
        atr = 0.0
        if len(closes) > 15:
            tr = np.maximum(
                highs[-14:] - lows[-14:],
                np.maximum(
                    np.abs(highs[-14:] - closes[-15:-1]),
                    np.abs(lows[-14:] - closes[-15:-1]),
                ),
            )
            atr = float(np.mean(tr))

        atr_pct = atr / closes[-1] * 100 if closes[-1] > 0 else 0

        # realized volatility (log returns std)
        if len(closes) > vol_lookback:
            log_rets = np.diff(np.log(closes[-vol_lookback - 1:]))
            realized_vol = float(np.std(log_rets) * np.sqrt(1440))
        else:
            realized_vol = 0.0

        # short-term momentum (5m return vs 1h return)
        ret_5m = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) > 5 else 0
        ret_1h = (closes[-1] - closes[-60]) / closes[-60] * 100 if len(closes) > 60 else 0

        # drawdown from recent high
        if len(closes) > 20:
            recent_high = float(np.max(closes[-20:]))
            drawdown = (closes[-1] - recent_high) / recent_high * 100
        else:
            drawdown = 0.0

        return {
            "return": ret,
            "atr_pct": atr_pct,
            "realized_vol": realized_vol,
            "ret_5m": ret_5m,
            "ret_1h": ret_1h,
            "drawdown": drawdown,
        }

    # ------------------------------------------------------------------
    # temporal features
    # ------------------------------------------------------------------

    def _compute_temporal_features(self) -> dict:
        """Weekend + session detection."""
        now = dt.datetime.utcnow()
        return {
            "hour": now.hour,
            "day_of_week": now.weekday(),
            "is_weekend": now.weekday() >= 5,
            "is_low_liquidity_hours": now.hour in (3, 4, 5, 6),  # 3-6 UTC = Asia late night
        }

    # ------------------------------------------------------------------
    # funding index
    # ------------------------------------------------------------------

    async def _get_funding_index(self) -> float:
        """Average funding rate across top coins from Redis."""
        try:
            top_symbols = [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
                "MATICUSDT", "LTCUSDT", "NEARUSDT", "ATOMUSDT", "OPUSDT",
                "ARBUSDT", "APTUSDT", "INJUSDT", "SUIUSDT", "TIAUSDT",
            ]
            rates = []
            for sym in top_symbols:
                data = await self.redis.get_market_data(sym)
                if data and "funding_rate" in data:
                    rates.append(float(data["funding_rate"]))

            return float(np.mean(rates)) if rates else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # classification logic
    # ------------------------------------------------------------------

    def _classify(
        self, btc: dict, temporal: dict, funding_index: float
    ) -> tuple:
        """Classify into regime with confidence score."""
        if not btc:
            return MarketRegime.NEUTRAL, 0.3

        ret = btc.get("return", 0)
        atr_pct = btc.get("atr_pct", 0)
        realized_vol = btc.get("realized_vol", 0)
        drawdown = btc.get("drawdown", 0)
        is_weekend = temporal.get("is_weekend", False)
        is_low_hours = temporal.get("is_low_liquidity_hours", False)

        scores = {
            MarketRegime.RISK_ON: 0.0,
            MarketRegime.RISK_OFF: 0.0,
            MarketRegime.VOLATILITY_EXPANSION: 0.0,
            MarketRegime.LOW_LIQUIDITY: 0.0,
            MarketRegime.NEUTRAL: 0.2,  # base bias toward neutral
        }

        # ── RISK_ON signals ──
        if ret > 2.0:
            scores[MarketRegime.RISK_ON] += 0.3
        if ret > 5.0:
            scores[MarketRegime.RISK_ON] += 0.2
        if atr_pct < 1.5:
            scores[MarketRegime.RISK_ON] += 0.15
        if funding_index > 0.0005:
            scores[MarketRegime.RISK_ON] += 0.1  # positive funding = bullish bias
        if drawdown > -2.0:
            scores[MarketRegime.RISK_ON] += 0.1

        # ── RISK_OFF signals ──
        if ret < -3.0:
            scores[MarketRegime.RISK_OFF] += 0.3
        if ret < -5.0:
            scores[MarketRegime.RISK_OFF] += 0.2
        if drawdown < -5.0:
            scores[MarketRegime.RISK_OFF] += 0.25
        if atr_pct > 3.0:
            scores[MarketRegime.RISK_OFF] += 0.15
        if funding_index < -0.001:
            scores[MarketRegime.RISK_OFF] += 0.1

        # ── VOLATILITY_EXPANSION ──
        if atr_pct > 2.5:
            scores[MarketRegime.VOLATILITY_EXPANSION] += 0.3
        if realized_vol > 0.6:
            scores[MarketRegime.VOLATILITY_EXPANSION] += 0.25
        if abs(ret) > 4.0:
            scores[MarketRegime.VOLATILITY_EXPANSION] += 0.2

        # ── LOW_LIQUIDITY ──
        if is_weekend:
            scores[MarketRegime.LOW_LIQUIDITY] += 0.4
        if is_low_hours:
            scores[MarketRegime.LOW_LIQUIDITY] += 0.3

        # find winner
        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.0

        return best_regime, min(confidence, 1.0)
