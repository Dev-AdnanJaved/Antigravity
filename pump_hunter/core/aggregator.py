"""
Aggregator — combines data from multiple exchanges, normalizes,
and detects cross-exchange discrepancies.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import structlog

from pump_hunter.collectors.multi_exchange import MultiExchangeCollector
from pump_hunter.core.market_state import MarketStateManager, SymbolState
from pump_hunter.storage.redis_store import RedisStore

logger = structlog.get_logger(__name__)


def _sanitize_for_json(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_sanitize_for_json(x) for x in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class Aggregator:
    """
    Combines single-exchange and cross-exchange data into unified metrics.
    Computes aggregate OI, weighted funding, combined depth, cross-exchange volume ratios.
    """

    def __init__(
        self,
        market_state: MarketStateManager,
        multi_exchange: Optional[MultiExchangeCollector],
        redis: RedisStore,
    ):
        self.market_state = market_state
        self.multi_exchange = multi_exchange
        self.redis = redis

    async def aggregate(self, symbol: str) -> dict:
        """
        Build an aggregated data dict for a symbol, combining Binance + secondary exchanges.
        Returns a dict suitable for feature computation.
        """
        state = self.market_state.get_state(symbol)
        result = state.to_dict()

        # cross-exchange data
        xex_data = {}
        if self.multi_exchange:
            xex_data = await self.multi_exchange.get_cross_exchange_data(symbol)

        result["cross_exchange"] = xex_data

        # aggregate OI across exchanges
        total_oi = state.oi_value
        oi_by_exchange = {"binance": state.oi_value}
        for ex, data in xex_data.items():
            ex_oi = data.get("open_interest", 0)
            if ex_oi:
                total_oi += ex_oi
                oi_by_exchange[ex] = ex_oi
        result["total_oi"] = total_oi
        result["oi_by_exchange"] = oi_by_exchange

        # aggregate volume
        total_vol = state.volume_24h
        vol_by_exchange = {"binance": state.volume_24h}
        for ex, data in xex_data.items():
            ticker = data.get("ticker", {})
            ex_vol = ticker.get("volume_24h", 0)
            if ex_vol:
                total_vol += ex_vol
                vol_by_exchange[ex] = ex_vol
        result["total_volume"] = total_vol
        result["volume_by_exchange"] = vol_by_exchange

        # cross-exchange volume ratio (max/median)
        if len(vol_by_exchange) > 1:
            volumes = list(vol_by_exchange.values())
            max_vol = max(volumes)
            median_vol = float(np.median(volumes))
            result["xex_volume_ratio"] = max_vol / median_vol if median_vol > 0 else 1.0
        else:
            result["xex_volume_ratio"] = 1.0

        # weighted avg funding
        funding_rates = [state.funding_rate]
        for ex, data in xex_data.items():
            fr = data.get("funding_rate")
            if fr is not None:
                funding_rates.append(fr)
        result["avg_funding_rate"] = float(np.mean(funding_rates)) if funding_rates else 0

        # price divergence across exchanges
        prices = {"binance": state.price}
        for ex, data in xex_data.items():
            ticker = data.get("ticker", {})
            ex_price = ticker.get("price", 0)
            if ex_price:
                prices[ex] = ex_price
        if len(prices) > 1:
            price_vals = list(prices.values())
            result["price_spread_pct"] = (
                (max(price_vals) - min(price_vals)) / min(price_vals) * 100
                if min(price_vals) > 0 else 0
            )
        else:
            result["price_spread_pct"] = 0

        result["prices_by_exchange"] = prices

        # sanitize numpy types before Redis caching (prevents orjson errors)
        result = _sanitize_for_json(result)

        # cache aggregated state in Redis
        await self.redis.set_market_state(symbol, result)

        return result

    async def aggregate_all(self, symbols: List[str]) -> Dict[str, dict]:
        """Aggregate data for all symbols."""
        results = {}
        for sym in symbols:
            try:
                results[sym] = await self.aggregate(sym)
            except Exception as e:
                logger.debug("aggregate_error", symbol=sym, error=str(e))
        return results
