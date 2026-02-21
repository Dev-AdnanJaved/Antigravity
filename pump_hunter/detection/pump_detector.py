"""
Pump detector — scoring, classification, event detection, and smart levels.
Combines weighted signals with interaction bonuses and penalties.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from pump_hunter.config.settings import Settings
from pump_hunter.core.market_state import SymbolState
from pump_hunter.storage.timeseries import TimeseriesStore

logger = structlog.get_logger(__name__)


class PumpDetector:
    """
    Scoring engine: weighted signal sum + interaction bonuses - penalties.
    Classification: CRITICAL, HIGH_ALERT, WATCHLIST, MONITOR.
    Smart levels: entry/stop/TP using ATR + orderbook.
    """

    def __init__(self, settings: Settings, timeseries: TimeseriesStore):
        self.settings = settings
        self.ts = timeseries
        self.weights = settings.detection.score_weights
        self.bonuses = settings.detection.interaction_bonuses
        self.classification = settings.detection.classification
        self.levels_cfg = settings.levels

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def score(
        self,
        signals: Dict[str, Optional[float]],
        state: SymbolState,
        prev_score: Optional[float] = None,
        prev_class: Optional[str] = None,
    ) -> dict:
        """
        Compute composite score from signal features.
        Returns dict with score, classification, bonuses, penalties, events, levels.
        """
        # base score = weighted sum (skip None signals)
        base_score = 0.0
        total_weight = 0.0
        active_signals = {}

        for name, weight in self.weights.items():
            value = signals.get(name)
            if value is not None:
                base_score += value * weight
                total_weight += weight
                active_signals[name] = value

        # normalize if some signals are missing
        if total_weight > 0 and total_weight < 0.99:
            base_score = base_score / total_weight

        # interaction bonuses
        bonuses_applied = {}

        # squeeze_setup: negative funding + OI surge + depth imbalance
        if (
            signals.get("funding_rate", 0) and signals["funding_rate"] >= 40
            and signals.get("oi_surge", 0) and signals["oi_surge"] >= 40
        ):
            bonuses_applied["squeeze_setup"] = self.bonuses["squeeze_setup"]

        # cascade_setup: high liquidation leverage + negative funding + low L/S
        if (
            signals.get("liquidation_leverage", 0) and signals["liquidation_leverage"] >= 50
            and signals.get("long_short_ratio", 0) and signals["long_short_ratio"] >= 30
        ):
            bonuses_applied["cascade_setup"] = self.bonuses["cascade_setup"]

        # accumulation_setup: volume-price decouple + OI surge + whale activity
        if (
            signals.get("volume_price_decouple", 0) and signals["volume_price_decouple"] >= 30
            and signals.get("oi_surge", 0) and signals["oi_surge"] >= 30
        ):
            bonuses_applied["accumulation_setup"] = self.bonuses["accumulation_setup"]

        # apply bonuses
        bonus_total = sum(bonuses_applied.values())
        score_with_bonus = base_score * (1 + bonus_total)

        # penalties
        penalties_applied = {}
        penalty_cfg = self.settings.detection.price_extended_penalty

        # price extended penalty
        price_change_7d = state.get_price_change(minutes=penalty_cfg.lookback_days * 1440)
        if price_change_7d > penalty_cfg.threshold_pct:
            penalties_applied["price_extended"] = penalty_cfg.penalty_pct / 100
            score_with_bonus *= (1 - penalties_applied["price_extended"])

        final_score = max(0, min(100, score_with_bonus))

        # classify
        classification = self._classify(final_score)

        # event detection
        events = []
        if prev_score is not None:
            score_jump = final_score - prev_score
            if score_jump >= 20:
                events.append("SCORE_JUMP")
            if prev_class and classification != prev_class:
                # check if upgrade
                class_order = ["MONITOR", "WATCHLIST", "HIGH_ALERT", "CRITICAL"]
                if (
                    prev_class in class_order
                    and classification in class_order
                    and class_order.index(classification) > class_order.index(prev_class)
                ):
                    events.append("UPGRADE")

        # detect ignition (multiple signals firing simultaneously)
        high_signals = sum(1 for v in active_signals.values() if v >= 60)
        if high_signals >= 4:
            events.append("IGNITION")

        # compute levels for CRITICAL/HIGH_ALERT
        levels = {}
        if classification in ("CRITICAL", "HIGH_ALERT"):
            levels = self._compute_levels(state, classification, final_score)

        return {
            "composite_score": round(final_score, 1),
            "classification": classification,
            "bonuses_applied": bonuses_applied,
            "penalties_applied": penalties_applied,
            "events": events,
            "active_signals": active_signals,
            "levels": levels,
        }

    def _classify(self, score: float) -> str:
        """Classify score into tier."""
        c = self.classification
        if score >= c.critical:
            return "CRITICAL"
        elif score >= c.high_alert:
            return "HIGH_ALERT"
        elif score >= c.watchlist:
            return "WATCHLIST"
        elif score >= c.monitor:
            return "MONITOR"
        return "LOW"

    # ------------------------------------------------------------------
    # smart levels
    # ------------------------------------------------------------------

    def _compute_levels(
        self, state: SymbolState, classification: str, score: float
    ) -> dict:
        """
        Compute entry/stop/TP using ATR, orderbook data.
        """
        symbol = state.symbol
        price = state.price
        if price <= 0:
            return {}

        # compute ATR from candle history
        closes = self.ts.get_closes(symbol, "binance", last_n=50)
        df = self.ts.get_dataframe(symbol, "binance", last_n=50)

        if len(df) < self.levels_cfg.atr_period:
            # fallback ATR from 24h range
            if state.price_high_24h > 0 and state.price_low_24h > 0:
                atr = (state.price_high_24h - state.price_low_24h) / 14
            else:
                atr = price * 0.03  # 3% default
        else:
            highs = df["high"].values
            lows = df["low"].values
            closes_arr = df["close"].values

            # true range
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes_arr[:-1]),
                    np.abs(lows[1:] - closes_arr[:-1]),
                ),
            )
            atr = float(np.mean(tr[-self.levels_cfg.atr_period :]))

        if atr <= 0:
            atr = price * 0.03

        # stop loss — ATR-based, clamped
        stop_pct = (atr / price) * 200  # 2× ATR as stop
        stop_pct = max(self.levels_cfg.stop_loss_min_pct, min(self.levels_cfg.stop_loss_max_pct, stop_pct))

        # adjust by depth if available
        if state.bid_depth_usd > 0 and state.ask_depth_usd > 0:
            # if strong bid support, tighten stop
            if state.depth_imbalance > 2.0:
                stop_pct *= 0.85

        stop_loss = price * (1 - stop_pct / 100)

        # entry zone
        if classification == "CRITICAL":
            entry = price  # market entry
        else:
            entry = price * 0.995  # slight pullback

        # take profits — ATR multiples
        tps = []
        for mult in self.levels_cfg.tp_multipliers:
            tp = price + atr * mult

            # score multiplier: higher score = more ambitious TPs
            score_mult = 1 + (score - 60) / 100  # 60 score = 1.0×, 80 = 1.2×
            tp = price + (tp - price) * score_mult

            tps.append(round(tp, 8))

        return {
            "entry_price": round(entry, 8),
            "stop_loss": round(stop_loss, 8),
            "stop_pct": round(stop_pct, 2),
            "tp1": tps[0] if len(tps) > 0 else None,
            "tp2": tps[1] if len(tps) > 1 else None,
            "tp3": tps[2] if len(tps) > 2 else None,
            "atr": round(atr, 8),
            "risk_reward": round((tps[0] - entry) / (entry - stop_loss), 2) if stop_loss < entry else 0,
        }

    # ------------------------------------------------------------------
    # pump detection (ATR-adaptive or static)
    # ------------------------------------------------------------------

    def _get_atr_adaptive_threshold(self, symbol: str) -> Optional[float]:
        """
        Compute dynamic pump threshold from ATR percentile.

        High ATR (>80th pct) → raise threshold to 2× ATR (filter noise)
        Low ATR (<20th pct) → lower threshold to 0.5× ATR (catch early)
        Middle → interpolate linearly
        """
        cfg = self.settings.detection

        df = self.ts.get_dataframe(symbol, "binance", last_n=200)
        if len(df) < 30:
            return None  # fall back to static

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # true range series
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )

        if len(tr) < 20:
            return None

        # 14-period ATR
        atr_period = self.levels_cfg.atr_period
        atr = float(np.mean(tr[-atr_period:]))
        current_price = closes[-1]

        if current_price <= 0 or atr <= 0:
            return None

        atr_pct = atr / current_price * 100  # ATR as % of price

        # percentile of current ATR vs historical ATR values
        atr_series = []
        for i in range(atr_period, len(tr)):
            atr_series.append(float(np.mean(tr[i - atr_period: i])))

        if len(atr_series) < 10:
            return None

        current_atr_val = atr_series[-1]
        percentile = sum(1 for a in atr_series if a <= current_atr_val) / len(atr_series) * 100

        # scale threshold based on percentile
        high_pct = cfg.atr_adaptive_high_percentile
        low_pct = cfg.atr_adaptive_low_percentile

        if percentile >= high_pct:
            # high volatility: need bigger move to qualify as pump
            threshold = atr_pct * cfg.atr_adaptive_high_mult
        elif percentile <= low_pct:
            # low volatility: smaller move = significant
            threshold = atr_pct * cfg.atr_adaptive_low_mult
        else:
            # linear interpolation
            t = (percentile - low_pct) / (high_pct - low_pct)
            low_thresh = atr_pct * cfg.atr_adaptive_low_mult
            high_thresh = atr_pct * cfg.atr_adaptive_high_mult
            threshold = low_thresh + t * (high_thresh - low_thresh)

        # clamp to reasonable range
        threshold = max(1.0, min(threshold, cfg.pump_max_pct * 0.5))

        return threshold

    def detect_pump(self, state: SymbolState) -> Optional[dict]:
        """
        Detect if a pump is happening.

        Supports two modes:
            - "static": fixed pump_threshold_pct (original behavior)
            - "atr_adaptive": threshold scales with ATR percentile
        """
        cfg = self.settings.detection

        # determine threshold
        if cfg.pump_threshold_mode == "atr_adaptive":
            adaptive = self._get_atr_adaptive_threshold(state.symbol)
            threshold = adaptive if adaptive is not None else cfg.pump_threshold_pct
            threshold_source = "atr_adaptive" if adaptive is not None else "static_fallback"
        else:
            threshold = cfg.pump_threshold_pct
            threshold_source = "static"

        # check various windows
        for minutes in [5, 15, 30, 60]:
            change = state.get_price_change(minutes=minutes)

            if threshold <= change <= cfg.pump_max_pct:
                pump_type = "spike" if minutes <= 15 else "gradual"
                return {
                    "symbol": state.symbol,
                    "price": state.price,
                    "change_pct": round(change, 2),
                    "window_minutes": minutes,
                    "pump_type": pump_type,
                    "pump_threshold": round(threshold, 2),
                    "threshold_source": threshold_source,
                    "timestamp": dt.datetime.utcnow(),
                }

        return None
