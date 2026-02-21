"""
Capital allocator — portfolio-level position sizing and selection.

When multiple coins trigger simultaneously:
    1. Score-weighted allocation (higher score → larger fraction)
    2. Liquidity-adjusted sizing (cap at 1% of 24h volume)
    3. Volatility normalization (inverse ATR sizing)
    4. Correlation filter (sector limits)
    5. Drawdown scaling (reduce after losses)
    6. Max concurrent position enforcement
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from pump_hunter.config.settings import Settings
from pump_hunter.core.market_state import MarketStateManager, SymbolState
from pump_hunter.storage.timeseries import TimeseriesStore

logger = structlog.get_logger(__name__)

# rough sector classification for correlation filter
SECTOR_MAP = {
    "BTCUSDT": "btc",
    "ETHUSDT": "eth",
    "BNBUSDT": "exchange",
    "SOLUSDT": "l1",
    "AVAXUSDT": "l1",
    "DOTUSDT": "l1",
    "NEARUSDT": "l1",
    "ATOMUSDT": "l1",
    "APTUSDT": "l1",
    "SUIUSDT": "l1",
    "ARBUSDT": "l2",
    "OPUSDT": "l2",
    "MATICUSDT": "l2",
    "LINKUSDT": "oracle",
    "DOGEUSDT": "meme",
    "SHIBUSDT": "meme",
    "PEPEUSDT": "meme",
    "FLOKIUSDT": "meme",
    "BONKUSDT": "meme",
    "WIFUSDT": "meme",
    "XRPUSDT": "payment",
    "ADAUSDT": "l1",
    "LTCUSDT": "payment",
    "INJUSDT": "defi",
    "TIAUSDT": "modular",
    "AAVEUSDT": "defi",
    "UNIUSDT": "defi",
    "MKRUSDT": "defi",
}


class CapitalAllocator:
    """Portfolio-level position sizing and concurrent position management."""

    def __init__(
        self,
        settings: Settings,
        market_state: MarketStateManager,
        timeseries: TimeseriesStore,
    ):
        self.cfg = settings.allocation
        self.exec_cfg = settings.execution
        self.market_state = market_state
        self.ts = timeseries

        # active positions
        self._positions: Dict[str, dict] = {}  # symbol → position info
        self._equity = self.exec_cfg.default_trade_size_usd * 10  # starting "virtual" equity
        self._peak_equity = self._equity
        self._total_pnl = 0.0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        return len(self._positions)

    @property
    def active_symbols(self) -> List[str]:
        return list(self._positions.keys())

    def allocate(
        self,
        candidates: List[dict],
    ) -> List[dict]:
        """
        Given a list of candidate alerts (sorted by score desc), decide:
        - Which to accept
        - How much to allocate to each
        - Which to reject and why

        Each candidate should have: symbol, score, classification, state, levels
        """
        available_slots = self.cfg.max_concurrent_positions - self.active_count

        if available_slots <= 0:
            return [{
                **c,
                "allocated": False,
                "reject_reason": "max_positions_reached",
            } for c in candidates]

        # filter by min score
        eligible = [
            c for c in candidates
            if c.get("score", 0) >= self.cfg.min_score_for_allocation
        ]

        if not eligible:
            return [{**c, "allocated": False, "reject_reason": "below_min_score"} for c in candidates]

        # apply correlation filter
        eligible = self._apply_correlation_filter(eligible)

        # sort by adjusted score (score × liquidity factor)
        for c in eligible:
            c["_adj_score"] = self._compute_adjusted_score(c)

        eligible.sort(key=lambda x: x["_adj_score"], reverse=True)

        # allocate to top N
        results = []
        allocated_count = 0

        for c in eligible:
            if allocated_count >= available_slots:
                results.append({**c, "allocated": False, "reject_reason": "slots_filled"})
                continue

            size = self._compute_size(c, allocated_count, len(eligible))
            if size < 100:  # minimum $100
                results.append({**c, "allocated": False, "reject_reason": "size_too_small"})
                continue

            results.append({
                **c,
                "allocated": True,
                "position_size_usd": round(size, 2),
                "pct_of_equity": round(size / self._equity * 100, 1) if self._equity > 0 else 0,
            })

            # track position
            self._positions[c["symbol"]] = {
                "size_usd": size,
                "entry_time": time.time(),
                "score": c.get("score", 0),
            }

            allocated_count += 1

        # add rejected candidates not in eligible
        allocated_symbols = {r["symbol"] for r in results}
        for c in candidates:
            if c["symbol"] not in allocated_symbols:
                results.append({**c, "allocated": False, "reject_reason": "filtered_out"})

        logger.info(
            "allocation_complete",
            candidates=len(candidates),
            allocated=allocated_count,
            active_positions=self.active_count,
        )

        return results

    def close_position(self, symbol: str, pnl_usd: float) -> None:
        """Record position close and update equity."""
        if symbol in self._positions:
            del self._positions[symbol]

        self._total_pnl += pnl_usd
        self._equity += pnl_usd
        self._peak_equity = max(self._peak_equity, self._equity)

    def get_stats(self) -> dict:
        """Return allocation statistics."""
        return {
            "active_positions": self.active_count,
            "active_symbols": self.active_symbols,
            "equity": round(self._equity, 2),
            "peak_equity": round(self._peak_equity, 2),
            "total_pnl": round(self._total_pnl, 2),
            "drawdown_pct": round(
                (self._peak_equity - self._equity) / self._peak_equity * 100, 1
            ) if self._peak_equity > 0 else 0,
        }

    # ------------------------------------------------------------------
    # correlation filter
    # ------------------------------------------------------------------

    def _apply_correlation_filter(self, candidates: List[dict]) -> List[dict]:
        """Limit exposure to same sector."""
        sector_limit = self.cfg.correlation_sector_limit

        # count existing sector exposure
        sector_counts: Dict[str, int] = {}
        for sym in self._positions:
            sector = SECTOR_MAP.get(sym, "other")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        filtered = []
        for c in candidates:
            symbol = c["symbol"]
            sector = SECTOR_MAP.get(symbol, "other")
            current = sector_counts.get(sector, 0)

            if current >= sector_limit:
                c["_sector_filtered"] = True
                c["reject_reason"] = f"sector_limit ({sector})"
            else:
                filtered.append(c)
                sector_counts[sector] = current + 1

        return filtered

    # ------------------------------------------------------------------
    # scoring adjustments
    # ------------------------------------------------------------------

    def _compute_adjusted_score(self, candidate: dict) -> float:
        """Adjust score by liquidity and volatility factors."""
        base_score = candidate.get("score", 0)

        # liquidity bonus: higher volume = more reliable signal
        state = candidate.get("state")
        if state and hasattr(state, "volume_24h") and state.volume_24h > 0:
            vol_millions = state.volume_24h / 1_000_000
            liquidity_factor = min(1.0 + np.log10(max(vol_millions, 0.1)) * 0.1, 1.5)
        else:
            liquidity_factor = 1.0

        return base_score * liquidity_factor

    # ------------------------------------------------------------------
    # position sizing
    # ------------------------------------------------------------------

    def _compute_size(
        self, candidate: dict, current_allocated: int, total_eligible: int
    ) -> float:
        """Compute position size considering all constraints."""
        base = self.exec_cfg.default_trade_size_usd

        # 1. score-weighted: higher score gets proportionally more
        score = candidate.get("score", 50)
        score_weight = score / 100  # 0-1 range

        # 2. max single position as % of equity
        max_by_equity = self._equity * (self.cfg.max_single_position_pct / 100)

        # 3. liquidity cap
        state = candidate.get("state")
        volume_24h = state.volume_24h if state and hasattr(state, "volume_24h") else 0
        max_by_liquidity = (
            volume_24h * (self.cfg.max_position_pct_of_volume / 100)
            if volume_24h > 0
            else float("inf")
        )

        # 4. volatility normalization (inverse ATR)
        vol_factor = 1.0
        if self.cfg.volatility_normalize:
            vol_factor = self._compute_volatility_factor(candidate.get("symbol", ""))

        # 5. drawdown scaling
        drawdown_factor = 1.0
        if self._peak_equity > 0:
            dd_pct = (self._peak_equity - self._equity) / self._peak_equity
            if dd_pct > 0.02:  # start scaling down after 2% drawdown
                drawdown_factor = max(self.cfg.drawdown_scale_factor, 1.0 - dd_pct)

        # combine
        size = base * score_weight * vol_factor * drawdown_factor
        size = min(size, max_by_equity, max_by_liquidity)

        return max(size, 0)

    def _compute_volatility_factor(self, symbol: str) -> float:
        """
        Inverse volatility sizing: volatile coins get smaller positions.
        Normalized so that average volatility → factor = 1.0.
        """
        df = self.ts.get_dataframe(symbol, "binance", last_n=30)
        if len(df) < 10:
            return 1.0

        closes = df["close"].values
        if len(closes) < 10:
            return 1.0

        # realized volatility (annualized from 1-min candles)
        log_rets = np.diff(np.log(closes))
        vol = float(np.std(log_rets))

        if vol <= 0:
            return 1.0

        # target vol: 0.02 (≈2% daily) → factor = 1.0
        target_vol = 0.002  # per-candle (1 min)
        factor = target_vol / vol

        # clamp between 0.3 and 2.0
        return float(np.clip(factor, 0.3, 2.0))
