"""
Pump recorder — captures full pre/during/post pump data for ML training.
Records market state at high granularity around pump events.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from typing import Dict, List, Optional, Set

import structlog

from pump_hunter.config.settings import Settings
from pump_hunter.core.market_state import MarketStateManager, SymbolState
from pump_hunter.storage.database import Database
from pump_hunter.storage.redis_store import RedisStore
from pump_hunter.storage.timeseries import TimeseriesStore

logger = structlog.get_logger(__name__)


class ActivePump:
    """Tracks an active pump event being recorded."""

    def __init__(self, pump_event_id: int, symbol: str, start_price: float):
        self.pump_event_id = pump_event_id
        self.symbol = symbol
        self.start_price = start_price
        self.start_time = time.time()
        self.peak_price = start_price
        self.peak_time = time.time()
        self.phase = "during"  # pre, during, post
        self.datapoints_count = 0
        self.post_pump_start: Optional[float] = None

    @property
    def current_change_pct(self) -> float:
        return (self.peak_price - self.start_price) / self.start_price * 100 if self.start_price else 0


class PumpRecorder:
    """
    Records full pump events with pre/during/post data capture.
    Pre-pump: 60 min at 1-minute granularity.
    During-pump: continuous capture every scan.
    Post-pump: 120 min of post-pump behavior.
    """

    def __init__(
        self,
        settings: Settings,
        db: Database,
        redis: RedisStore,
        market_state: MarketStateManager,
        timeseries: TimeseriesStore,
    ):
        self.settings = settings
        self.db = db
        self.redis = redis
        self.market_state = market_state
        self.timeseries = timeseries
        self._active_pumps: Dict[str, ActivePump] = {}
        self._cooldown: Dict[str, float] = {}  # symbol -> last pump time
        self._cooldown_minutes = 30  # don't re-record same symbol within 30 min

    @property
    def active_symbols(self) -> Set[str]:
        return set(self._active_pumps.keys())

    # ------------------------------------------------------------------
    # pump lifecycle
    # ------------------------------------------------------------------

    async def start_recording(
        self,
        symbol: str,
        pump_info: dict,
        signal_scores: Dict[str, Optional[float]],
        score_result: dict,
    ) -> Optional[int]:
        """
        Start recording a new pump event.
        Creates DB record, captures pre-pump data, starts monitoring.
        """
        # cooldown check
        now = time.time()
        if symbol in self._cooldown:
            if now - self._cooldown[symbol] < self._cooldown_minutes * 60:
                return None

        if symbol in self._active_pumps:
            return None  # already recording

        # create pump event in DB
        state = self.market_state.get_state(symbol)
        db_symbols = await self.db.get_active_symbols()
        symbol_id = None
        for s in db_symbols:
            if s.symbol == symbol:
                symbol_id = s.id
                break

        if not symbol_id:
            logger.warning("pump_record_no_symbol_id", symbol=symbol)
            return None

        event_id = await self.db.create_pump_event({
            "symbol_id": symbol_id,
            "start_time": pump_info.get("timestamp", dt.datetime.utcnow()),
            "start_price": pump_info["price"],
            "pump_type": pump_info.get("pump_type", "unknown"),
            "pre_pump_score": score_result.get("composite_score"),
            "pre_pump_signals": {k: v for k, v in signal_scores.items() if v is not None},
            "status": "active",
        })

        self._active_pumps[symbol] = ActivePump(event_id, symbol, pump_info["price"])

        # capture pre-pump data from timeseries history
        await self._capture_pre_pump(symbol, event_id, symbol_id)

        logger.info(
            "pump_recording_started",
            symbol=symbol,
            event_id=event_id,
            start_price=pump_info["price"],
            change_pct=pump_info.get("change_pct", 0),
        )

        return event_id

    async def update(self, symbol: str, signal_scores: dict) -> None:
        """
        Update an active pump recording with current data.
        Transitions from during -> post when price drops from peak.
        """
        if symbol not in self._active_pumps:
            return

        active = self._active_pumps[symbol]
        state = self.market_state.get_state(symbol)

        # update peak
        if state.price > active.peak_price:
            active.peak_price = state.price
            active.peak_time = time.time()

        # phase transition: during -> post
        if active.phase == "during":
            retrace = (active.peak_price - state.price) / active.peak_price * 100
            if retrace > 5:  # 5% retrace from peak = pump ending
                active.phase = "post"
                active.post_pump_start = time.time()

                await self.db.update_pump_event(active.pump_event_id, {
                    "peak_price": active.peak_price,
                    "peak_time": dt.datetime.utcfromtimestamp(active.peak_time),
                    "pump_pct": active.current_change_pct,
                    "duration_minutes": int((active.peak_time - active.start_time) / 60),
                })

        # capture datapoint
        db_symbols = await self.db.get_active_symbols()
        symbol_id = None
        for s in db_symbols:
            if s.symbol == symbol:
                symbol_id = s.id
                break

        if symbol_id:
            await self.db.insert_pump_datapoints([{
                "pump_event_id": active.pump_event_id,
                "timestamp": dt.datetime.utcnow(),
                "phase": active.phase,
                "price": state.price,
                "volume_1m": state.volume_24h / 1440,  # approximate
                "open_interest": state.oi_value,
                "funding_rate": state.funding_rate,
                "bid_depth": state.bid_depth_usd,
                "ask_depth": state.ask_depth_usd,
                "long_short_ratio": state.long_short_ratio,
                "liq_volume": state.liq_buy_vol + state.liq_sell_vol,
                "whale_volume": state.whale_buy_vol + state.whale_sell_vol,
                "signal_scores": {k: v for k, v in signal_scores.items() if v is not None},
            }])
            active.datapoints_count += 1

        # check if post-pump capture window is done
        if active.phase == "post" and active.post_pump_start:
            elapsed = (time.time() - active.post_pump_start) / 60
            if elapsed >= self.settings.detection.post_pump_capture_minutes:
                await self._finalize_pump(symbol)

    async def _capture_pre_pump(self, symbol: str, event_id: int, symbol_id: int) -> None:
        """Capture pre-pump data from timeseries history."""
        df = self.timeseries.get_dataframe(
            symbol, "binance",
            last_n=self.settings.detection.pre_pump_capture_minutes,
        )

        if len(df) == 0:
            return

        datapoints = []
        for idx, row in df.iterrows():
            datapoints.append({
                "pump_event_id": event_id,
                "timestamp": idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else dt.datetime.utcnow(),
                "phase": "pre",
                "price": float(row["close"]),
                "volume_1m": float(row["volume_quote"]),
                "open_interest": None,
                "funding_rate": None,
                "bid_depth": None,
                "ask_depth": None,
                "long_short_ratio": None,
                "liq_volume": None,
                "whale_volume": None,
                "signal_scores": None,
            })

        if datapoints:
            await self.db.insert_pump_datapoints(datapoints)
            logger.info("pre_pump_captured", symbol=symbol, datapoints=len(datapoints))

    async def _finalize_pump(self, symbol: str) -> None:
        """Finalize and close a pump recording."""
        active = self._active_pumps.pop(symbol, None)
        if not active:
            return

        state = self.market_state.get_state(symbol)
        retrace_pct = (active.peak_price - state.price) / active.peak_price * 100

        # extract ML features
        features = await self._extract_features(symbol, active)

        await self.db.update_pump_event(active.pump_event_id, {
            "end_time": dt.datetime.utcnow(),
            "end_price": state.price,
            "retrace_pct": round(retrace_pct, 2),
            "status": "completed",
            "features": features,
        })

        self._cooldown[symbol] = time.time()

        logger.info(
            "pump_recording_completed",
            symbol=symbol,
            event_id=active.pump_event_id,
            pump_pct=round(active.current_change_pct, 2),
            retrace_pct=round(retrace_pct, 2),
            datapoints=active.datapoints_count,
        )

    async def _extract_features(self, symbol: str, active: ActivePump) -> dict:
        """Extract ML-ready features from a completed pump event."""
        state = self.market_state.get_state(symbol)
        df = self.timeseries.get_dataframe(symbol, "binance", last_n=120)

        features = {
            "pump_magnitude_pct": active.current_change_pct,
            "pump_duration_min": (active.peak_time - active.start_time) / 60,
            "retrace_pct": (active.peak_price - state.price) / active.peak_price * 100,
            "volume_24h_at_start": state.volume_24h,
            "oi_at_start": state.oi_value,
            "funding_at_start": state.funding_rate,
            "ls_ratio_at_start": state.long_short_ratio,
            "depth_imbalance_at_start": state.depth_imbalance,
            "whale_buy_vol": state.whale_buy_vol,
            "whale_sell_vol": state.whale_sell_vol,
            "liq_vol": state.liq_buy_vol + state.liq_sell_vol,
            "datapoints_captured": active.datapoints_count,
        }

        # add candle-based features if available
        if len(df) > 20:
            closes = df["close"].values
            volumes = df["volume_quote"].values
            features["volatility_20"] = float(np.std(closes[-20:]) / np.mean(closes[-20:]) * 100) if np.mean(closes[-20:]) > 0 else 0
            features["avg_volume_20"] = float(np.mean(volumes[-20:]))
            features["volume_spike"] = float(volumes[-1] / np.mean(volumes[-20:])) if np.mean(volumes[-20:]) > 0 else 0

        return features

    async def force_close_all(self) -> None:
        """Force close all active pump recordings (for shutdown)."""
        symbols = list(self._active_pumps.keys())
        for sym in symbols:
            await self._finalize_pump(sym)

    def get_stats(self) -> dict:
        return {
            "active_recordings": len(self._active_pumps),
            "symbols": list(self._active_pumps.keys()),
            "cooldowns": len(self._cooldown),
        }
