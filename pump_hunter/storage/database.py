"""
Async PostgreSQL database manager using SQLAlchemy + asyncpg.
Handles connections, migrations, batch inserts, and data retention.
"""

from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional, Sequence

import structlog
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from pump_hunter.config.settings import Settings
from pump_hunter.storage.models import (
    Base,
    Prediction,
    PumpDatapoint,
    PumpEvent,
    SignalScore,
    Snapshot,
    Symbol,
    TradeAlert,
)

logger = structlog.get_logger(__name__)


class Database:
    """Async PostgreSQL database wrapper."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create engine and ensure tables exist."""
        db = self.settings.database
        self._engine = create_async_engine(
            db.dsn,
            pool_size=db.pool_max,
            pool_pre_ping=True,
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

        # create tables
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("database_connected", host=db.host, db=db.name)

    async def disconnect(self) -> None:
        """Close the engine."""
        if self._engine:
            await self._engine.dispose()
            logger.info("database_disconnected")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide an async session context."""
        assert self._session_factory is not None, "Database not connected"
        async with self._session_factory() as sess:
            try:
                yield sess
                await sess.commit()
            except Exception:
                await sess.rollback()
                raise

    # ------------------------------------------------------------------
    # symbols
    # ------------------------------------------------------------------

    async def upsert_symbol(
        self,
        symbol: str,
        base_asset: str,
        quote_asset: str = "USDT",
        exchange: str = "binance",
        listed_exchanges: dict | None = None,
    ) -> int:
        """Insert or update a symbol, return its id."""
        async with self.session() as sess:
            result = await sess.execute(
                select(Symbol).where(Symbol.symbol == symbol)
            )
            row = result.scalar_one_or_none()
            if row:
                row.is_active = True
                row.last_updated = dt.datetime.utcnow()
                if listed_exchanges:
                    row.listed_exchanges = listed_exchanges
                await sess.flush()
                return row.id
            else:
                new = Symbol(
                    symbol=symbol,
                    base_asset=base_asset,
                    quote_asset=quote_asset,
                    exchange=exchange,
                    listed_exchanges=listed_exchanges or {},
                )
                sess.add(new)
                await sess.flush()
                return new.id

    async def get_active_symbols(self) -> List[Symbol]:
        """Return all active symbols."""
        async with self.session() as sess:
            result = await sess.execute(
                select(Symbol).where(Symbol.is_active == True)
            )
            return list(result.scalars().all())

    async def deactivate_symbols(self, symbols: List[str]) -> None:
        """Mark symbols as inactive."""
        async with self.session() as sess:
            for sym in symbols:
                result = await sess.execute(
                    select(Symbol).where(Symbol.symbol == sym)
                )
                row = result.scalar_one_or_none()
                if row:
                    row.is_active = False

    # ------------------------------------------------------------------
    # snapshots
    # ------------------------------------------------------------------

    async def insert_snapshots(self, snapshots: List[dict]) -> None:
        """Batch insert snapshot records."""
        if not snapshots:
            return
        async with self.session() as sess:
            objs = [Snapshot(**s) for s in snapshots]
            sess.add_all(objs)

    async def get_snapshots(
        self,
        symbol_id: int,
        exchange: str | None = None,
        hours_back: int = 72,
        limit: int = 500,
    ) -> List[Snapshot]:
        """Get recent snapshots for a symbol."""
        cutoff = dt.datetime.utcnow() - dt.timedelta(hours=hours_back)
        async with self.session() as sess:
            q = (
                select(Snapshot)
                .where(Snapshot.symbol_id == symbol_id)
                .where(Snapshot.timestamp >= cutoff)
                .order_by(Snapshot.timestamp.desc())
                .limit(limit)
            )
            if exchange:
                q = q.where(Snapshot.exchange == exchange)
            result = await sess.execute(q)
            return list(result.scalars().all())

    # ------------------------------------------------------------------
    # signal scores
    # ------------------------------------------------------------------

    async def insert_signal_score(self, score_data: dict) -> int:
        """Insert a signal score record."""
        async with self.session() as sess:
            obj = SignalScore(**score_data)
            sess.add(obj)
            await sess.flush()
            return obj.id

    async def get_signal_history(
        self, symbol_id: int, hours_back: int = 24, limit: int = 100
    ) -> List[SignalScore]:
        """Get recent signal scores for a symbol."""
        cutoff = dt.datetime.utcnow() - dt.timedelta(hours=hours_back)
        async with self.session() as sess:
            result = await sess.execute(
                select(SignalScore)
                .where(SignalScore.symbol_id == symbol_id)
                .where(SignalScore.timestamp >= cutoff)
                .order_by(SignalScore.timestamp.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    # ------------------------------------------------------------------
    # pump events
    # ------------------------------------------------------------------

    async def create_pump_event(self, event_data: dict) -> int:
        """Create a new pump event record."""
        async with self.session() as sess:
            obj = PumpEvent(**event_data)
            sess.add(obj)
            await sess.flush()
            logger.info("pump_event_created", id=obj.id, symbol_id=event_data.get("symbol_id"))
            return obj.id

    async def update_pump_event(self, event_id: int, updates: dict) -> None:
        """Update an existing pump event."""
        async with self.session() as sess:
            result = await sess.execute(
                select(PumpEvent).where(PumpEvent.id == event_id)
            )
            row = result.scalar_one_or_none()
            if row:
                for k, v in updates.items():
                    setattr(row, k, v)

    async def insert_pump_datapoints(self, datapoints: List[dict]) -> None:
        """Batch insert pump datapoints."""
        if not datapoints:
            return
        async with self.session() as sess:
            objs = [PumpDatapoint(**d) for d in datapoints]
            sess.add_all(objs)

    async def get_pump_events(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> List[PumpEvent]:
        """Get pump events, optionally filtered by status."""
        async with self.session() as sess:
            q = select(PumpEvent).order_by(PumpEvent.start_time.desc()).limit(limit)
            if status:
                q = q.where(PumpEvent.status == status)
            result = await sess.execute(q)
            return list(result.scalars().all())

    async def get_completed_pumps_for_training(self) -> List[PumpEvent]:
        """Get all completed pumps with features for ML training."""
        async with self.session() as sess:
            result = await sess.execute(
                select(PumpEvent)
                .where(PumpEvent.status == "completed")
                .where(PumpEvent.features.isnot(None))
                .order_by(PumpEvent.start_time)
            )
            return list(result.scalars().all())

    # ------------------------------------------------------------------
    # predictions
    # ------------------------------------------------------------------

    async def insert_prediction(self, pred_data: dict) -> int:
        """Insert an ML prediction."""
        async with self.session() as sess:
            obj = Prediction(**pred_data)
            sess.add(obj)
            await sess.flush()
            return obj.id

    # ------------------------------------------------------------------
    # trade alerts
    # ------------------------------------------------------------------

    async def insert_alert(self, alert_data: dict) -> int:
        """Insert a trade alert."""
        async with self.session() as sess:
            obj = TradeAlert(**alert_data)
            sess.add(obj)
            await sess.flush()
            return obj.id

    async def get_recent_alert(
        self, symbol_id: int, seconds_back: int = 900
    ) -> Optional[TradeAlert]:
        """Check if we sent an alert for this symbol recently (rate limiting)."""
        cutoff = dt.datetime.utcnow() - dt.timedelta(seconds=seconds_back)
        async with self.session() as sess:
            result = await sess.execute(
                select(TradeAlert)
                .where(TradeAlert.symbol_id == symbol_id)
                .where(TradeAlert.timestamp >= cutoff)
                .order_by(TradeAlert.timestamp.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # maintenance
    # ------------------------------------------------------------------

    async def prune_old_data(self, days: int = 30) -> int:
        """Delete snapshots older than N days. Returns count deleted."""
        cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)
        async with self.session() as sess:
            result = await sess.execute(
                delete(Snapshot).where(Snapshot.timestamp < cutoff)
            )
            count = result.rowcount
            logger.info("data_pruned", rows_deleted=count, cutoff_days=days)
            return count

    async def get_stats(self) -> dict:
        """Return database statistics."""
        async with self.session() as sess:
            symbols = await sess.execute(
                select(text("count(*)")).select_from(Symbol)
            )
            snapshots = await sess.execute(
                select(text("count(*)")).select_from(Snapshot)
            )
            pumps = await sess.execute(
                select(text("count(*)")).select_from(PumpEvent)
            )
            signals = await sess.execute(
                select(text("count(*)")).select_from(SignalScore)
            )
            return {
                "symbols": symbols.scalar() or 0,
                "snapshots": snapshots.scalar() or 0,
                "pump_events": pumps.scalar() or 0,
                "signal_scores": signals.scalar() or 0,
            }
