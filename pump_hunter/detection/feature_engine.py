"""
Feature engine — computes 12 signal features for pump detection.
Each signal normalized to 0-100. NULL if data unreliable.
Fixes all known issues from previous project (OI always 100, liq always 100, etc).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from pump_hunter.config.settings import Settings
from pump_hunter.core.market_state import SymbolState
from pump_hunter.storage.database import Database
from pump_hunter.storage.redis_store import RedisStore
from pump_hunter.storage.timeseries import TimeseriesStore

logger = structlog.get_logger(__name__)


def piecewise_lerp(value: float, breakpoints: List[Tuple[float, float]]) -> float:
    """
    Piecewise linear interpolation for signal normalization.
    breakpoints: list of (input_value, output_score) tuples, sorted by input_value.
    Returns score clamped to [0, 100].
    """
    if not breakpoints:
        return 0.0
    if value <= breakpoints[0][0]:
        return breakpoints[0][1]
    if value >= breakpoints[-1][0]:
        return breakpoints[-1][1]

    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0) if x1 != x0 else 0
            return y0 + t * (y1 - y0)

    return breakpoints[-1][1]


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division."""
    return a / b if b != 0 else default


class FeatureEngine:
    """Computes all signal features for a symbol."""

    def __init__(
        self,
        settings: Settings,
        db: Database,
        redis: RedisStore,
        timeseries: TimeseriesStore,
    ):
        self.settings = settings
        self.db = db
        self.redis = redis
        self.ts = timeseries
        self.filters = settings.filters

    async def compute_all(
        self, symbol: str, state: SymbolState, aggregated: dict
    ) -> Dict[str, Optional[float]]:
        """
        Compute all signal features for a symbol.
        Returns dict of signal_name -> score (0-100 or None if unreliable).
        """
        signals = {}

        signals["oi_surge"] = await self._oi_surge(symbol, state, aggregated)
        signals["funding_rate"] = self._funding_rate(state)
        signals["liquidation_leverage"] = self._liquidation_leverage(state, aggregated)
        signals["cross_exchange_volume"] = self._cross_exchange_volume(aggregated)
        signals["depth_imbalance"] = self._depth_imbalance(state)
        signals["volume_price_decouple"] = self._volume_price_decouple(symbol)
        signals["volatility_compression"] = self._volatility_compression(symbol)
        signals["long_short_ratio"] = self._long_short_ratio(state)
        signals["futures_spot_divergence"] = self._futures_spot_divergence(symbol)
        signals["whale_activity"] = self._whale_activity(state)

        return signals

    # ------------------------------------------------------------------
    # 1. OI Surge (16%) — Per-exchange comparison, $100K floor, ±200% cap
    # ------------------------------------------------------------------

    async def _oi_surge(
        self, symbol: str, state: SymbolState, aggregated: dict
    ) -> Optional[float]:
        """
        Detect stealth OI buildup.
        FIXES: per-exchange comparison, absolute OI floor, capped %.
        """
        # data quality gate: need minimum OI
        if state.oi_value < self.filters.min_open_interest_usd:
            return None

        # use per-exchange OI change from state rolling history
        oi_change = state.get_oi_change(minutes=60)  # 1h change

        # cap to ±200% to avoid absurd values (12M% from previous project)
        oi_change = max(-200, min(200, oi_change))

        # compare OI change vs price change — divergence is the signal
        price_change = state.get_price_change(minutes=60)
        divergence = abs(oi_change) - abs(price_change)

        # OI rising while price flat = accumulation
        if oi_change > 0 and abs(price_change) < 5:
            # strong accumulation signal
            effective = oi_change * 1.5
        elif oi_change > 0:
            effective = divergence
        else:
            effective = 0

        score = piecewise_lerp(effective, [
            (0, 0),
            (5, 15),
            (15, 35),
            (30, 55),
            (50, 75),
            (80, 90),
            (120, 100),
        ])

        return round(score, 1)

    # ------------------------------------------------------------------
    # 2. Funding Rate (15%) — magnitude + persistence
    # ------------------------------------------------------------------

    def _funding_rate(self, state: SymbolState) -> Optional[float]:
        """
        Negative funding = crowded shorts = squeeze fuel.
        Score based on magnitude (55%) + how negative (persistence proxy).
        """
        rate = state.funding_rate

        if rate == 0:
            return 0.0

        # we care about negative funding (shorts paying longs)
        # very negative = very crowded shorts
        magnitude = abs(rate)

        if rate < 0:
            # negative funding — squeeze potential
            mag_score = piecewise_lerp(magnitude * 10000, [  # convert to basis points
                (0, 0),
                (3, 15),     # -0.03% = mild
                (8, 35),     # -0.08% = moderate
                (15, 55),    # -0.15% = significant
                (30, 75),    # -0.30% = strong
                (50, 90),    # -0.50% = extreme
                (100, 100),  # -1.0%+ = rare extreme
            ])
        else:
            # positive funding is less interesting but still signals sentiment
            mag_score = piecewise_lerp(magnitude * 10000, [
                (0, 0),
                (5, 5),
                (15, 15),
                (30, 25),
                (50, 35),
            ])

        return round(mag_score, 1)

    # ------------------------------------------------------------------
    # 3. Liquidation Leverage (13%) — with minimum depth check, 50× cap
    # ------------------------------------------------------------------

    def _liquidation_leverage(
        self, state: SymbolState, aggregated: dict
    ) -> Optional[float]:
        """
        Cascade potential: estimated short liquidation volume vs ask resistance.
        FIXES: $10K min ask depth, 50× ratio cap, use actual L/S when available.
        """
        # data quality: need meaningful orderbook depth
        if state.ask_depth_usd < self.filters.min_orderbook_depth_usd:
            return None

        if state.oi_value < self.filters.min_open_interest_usd:
            return None

        # estimate short-side OI
        ls_ratio = state.long_short_ratio
        if ls_ratio > 0 and ls_ratio != 1.0:
            short_fraction = 1.0 / (1.0 + ls_ratio)
        else:
            short_fraction = 0.5  # assume 50/50 when no data

        short_oi = state.oi_value * short_fraction

        # estimate liquidable volume within 15% move
        # assume average leverage 10-20× → liquidation at ~5-10% move
        avg_leverage = 12  # reasonable estimate
        liq_fraction = min(0.15 * avg_leverage / 100, 0.8)  # = min(0.18, 0.8) = 0.18
        liq_volume = short_oi * liq_fraction

        # ratio of liq volume to ask resistance
        ratio = safe_divide(liq_volume, state.ask_depth_usd, 0)

        # CAP at 50× (previous project had 57,000× which is meaningless)
        ratio = min(ratio, 50)

        score = piecewise_lerp(ratio, [
            (0, 0),
            (1, 10),
            (3, 25),
            (6, 45),
            (10, 65),
            (20, 80),
            (35, 90),
            (50, 100),
        ])

        return round(score, 1)

    # ------------------------------------------------------------------
    # 4. Cross-Exchange Volume (12%) — max/median ratio
    # ------------------------------------------------------------------

    def _cross_exchange_volume(self, aggregated: dict) -> Optional[float]:
        """
        Detect accumulation on one venue.
        Score based on max volume / median volume across exchanges.
        """
        vol_by_ex = aggregated.get("volume_by_exchange", {})
        if len(vol_by_ex) < 2:
            return None  # need multiple exchanges to compare

        volumes = [v for v in vol_by_ex.values() if v > 0]
        if len(volumes) < 2:
            return None

        ratio = aggregated.get("xex_volume_ratio", 1.0)

        score = piecewise_lerp(ratio, [
            (1, 0),
            (1.5, 10),
            (2, 25),
            (3, 45),
            (5, 65),
            (8, 80),
            (15, 95),
            (25, 100),
        ])

        return round(score, 1)

    # ------------------------------------------------------------------
    # 5. Depth Imbalance (11%) — bid/ask value ratio
    # ------------------------------------------------------------------

    def _depth_imbalance(self, state: SymbolState) -> Optional[float]:
        """
        Orderbook imbalance — high bid/ask ratio = buying pressure.
        """
        if state.bid_depth_usd < 1000 or state.ask_depth_usd < 1000:
            return None

        ratio = state.depth_imbalance  # bid_usd / ask_usd

        # >1 means more bids than asks (bullish pressure)
        score = piecewise_lerp(ratio, [
            (0.3, 0),    # heavy ask side
            (0.7, 10),
            (1.0, 25),   # balanced
            (1.3, 40),
            (1.6, 55),
            (2.0, 70),
            (2.5, 82),
            (3.0, 90),
            (4.0, 100),
        ])

        return round(score, 1)

    # ------------------------------------------------------------------
    # 6. Volume-Price Decouple (9%) — volume up + price flat
    # ------------------------------------------------------------------

    def _volume_price_decouple(self, symbol: str) -> Optional[float]:
        """
        Hidden buying: volume increasing while price stays flat.
        FIXES: use quote volume, relaxed price dampener (20% threshold).
        """
        volumes = self.ts.get_volumes(symbol, "binance", last_n=48)  # 48 1m candles
        closes = self.ts.get_closes(symbol, "binance", last_n=48)

        if len(volumes) < 24:
            return None

        recent_vol = np.sum(volumes[-12:])  # last 12 candles
        prev_vol = np.sum(volumes[-24:-12])  # previous 12 candles

        if prev_vol <= 0:
            return None

        vol_change_pct = (recent_vol - prev_vol) / prev_vol * 100

        # price change over same period
        if len(closes) >= 24 and closes[-24] > 0:
            price_change = abs((closes[-1] - closes[-24]) / closes[-24] * 100)
        else:
            price_change = 0

        # dampener: if price moved a lot, this isn't "hidden" buying
        # 20% threshold instead of 12% (previous was too aggressive)
        dampener = max(0.15, 1.0 - price_change / 20.0)

        effective = max(0, vol_change_pct * dampener)

        score = piecewise_lerp(effective, [
            (0, 0),
            (15, 10),    # +15% volume, low price move
            (30, 25),
            (50, 45),
            (80, 65),
            (120, 80),
            (200, 95),
            (300, 100),
        ])

        return round(score, 1)

    # ------------------------------------------------------------------
    # 7. Volatility Compression (8%) — Bollinger Band width percentile
    # ------------------------------------------------------------------

    def _volatility_compression(self, symbol: str) -> Optional[float]:
        """
        Coiled spring: Bollinger Band width at historical low = breakout imminent.
        """
        closes = self.ts.get_closes(symbol, "binance", last_n=100)
        if len(closes) < 50:
            return None

        # compute BB width
        window = 20
        widths = []
        for i in range(window, len(closes)):
            segment = closes[i - window : i]
            mean = np.mean(segment)
            std = np.std(segment)
            if mean > 0:
                bb_width = (2 * std) / mean * 100  # as percentage
                widths.append(bb_width)

        if len(widths) < 10:
            return None

        current_width = widths[-1]
        # percentile: how current width ranks vs recent history
        percentile = sum(1 for w in widths if w >= current_width) / len(widths) * 100

        # high percentile = current width is smaller than most (compressed)
        score = piecewise_lerp(percentile, [
            (0, 0),      # current width is the widest (no compression)
            (30, 10),
            (50, 25),
            (70, 45),
            (80, 60),
            (90, 80),
            (95, 90),
            (99, 100),   # extreme compression
        ])

        return round(score, 1)

    # ------------------------------------------------------------------
    # 8. Long/Short Ratio (6%) — short dominance
    # ------------------------------------------------------------------

    def _long_short_ratio(self, state: SymbolState) -> Optional[float]:
        """
        L/S ratio < 1.0 = more shorts = potential squeeze fuel.
        FIXES: when L/S data unavailable, estimate from funding rate.
        """
        ratio = state.long_short_ratio

        # if no real data (default 1.0), try funding rate estimate
        if ratio == 1.0 and state.funding_rate != 0:
            # negative funding suggests more shorts
            if state.funding_rate < -0.0001:
                ratio = 0.8  # estimate
            elif state.funding_rate < -0.0005:
                ratio = 0.6  # strongly short
            elif state.funding_rate > 0.0005:
                ratio = 1.4  # strongly long
            else:
                return 0.0  # can't determine

        if ratio <= 0:
            return None

        # lower ratio = more shorts = higher score
        score = piecewise_lerp(ratio, [
            (0.3, 100),  # extreme short dominance
            (0.5, 85),
            (0.7, 65),
            (0.8, 50),
            (0.9, 35),
            (1.0, 20),   # balanced
            (1.2, 10),
            (1.5, 0),    # long dominance — no squeeze
        ])

        return round(score, 1)

    # ------------------------------------------------------------------
    # 9. Futures/Spot Divergence (5%) — volume comparison
    # ------------------------------------------------------------------

    def _futures_spot_divergence(self, symbol: str) -> Optional[float]:
        """
        Compare current volume vs historical average (apples-to-apples).
        FIXES: use same source for both current and historical.
        """
        volumes = self.ts.get_volumes(symbol, "binance", last_n=200)
        if len(volumes) < 50:
            return None

        # current: last 24 candles total
        recent = np.sum(volumes[-24:]) if len(volumes) >= 24 else np.sum(volumes)

        # historical: average of prior 24-candle windows
        prior = volumes[:-24] if len(volumes) > 24 else volumes
        if len(prior) < 24:
            return None

        # compute rolling 24-period sums
        window_sums = []
        for i in range(24, len(prior) + 1):
            window_sums.append(np.sum(prior[i - 24 : i]))

        if not window_sums:
            return None

        avg_prior = np.mean(window_sums)
        if avg_prior <= 0:
            return None

        ratio = recent / avg_prior

        # minimum volume floor
        if recent < self.filters.min_volume_24h_usd * 0.01:  # scale down for candle volume
            return None

        score = piecewise_lerp(ratio, [
            (0.5, 0),
            (1.0, 10),
            (1.5, 25),
            (2.0, 40),
            (3.0, 60),
            (5.0, 80),
            (8.0, 95),
            (10, 100),
        ])

        return round(score, 1)

    # ------------------------------------------------------------------
    # 10. Whale Activity (5%) — large trade detection
    # ------------------------------------------------------------------

    def _whale_activity(self, state: SymbolState) -> Optional[float]:
        """
        Score based on whale trade volume (from aggTrade stream, filtered $50K+).
        """
        total_whale = state.whale_buy_vol + state.whale_sell_vol
        if total_whale <= 0:
            return 0.0

        whale_count = state.whale_count

        # buy dominance
        buy_ratio = safe_divide(state.whale_buy_vol, total_whale, 0.5)

        # score based on total whale activity and buy dominance
        activity_score = piecewise_lerp(total_whale, [
            (0, 0),
            (100_000, 15),
            (500_000, 35),
            (1_000_000, 55),
            (5_000_000, 75),
            (10_000_000, 90),
            (50_000_000, 100),
        ])

        # boost if whales are buying, dampen if selling
        direction_mult = piecewise_lerp(buy_ratio, [
            (0, 0.3),     # all sells
            (0.3, 0.6),
            (0.5, 1.0),   # neutral
            (0.7, 1.3),
            (1.0, 1.5),   # all buys
        ])

        score = min(100, activity_score * direction_mult)
        return round(score, 1)
