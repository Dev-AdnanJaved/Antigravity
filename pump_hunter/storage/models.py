"""
SQLAlchemy ORM models for PostgreSQL storage.
Tables: symbols, snapshots, signal_scores, pump_events, pump_datapoints,
        predictions, trade_alerts.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

class Symbol(Base):
    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    base_asset: Mapped[str] = mapped_column(String(30), nullable=False)
    quote_asset: Mapped[str] = mapped_column(String(10), nullable=False, default="USDT")
    exchange: Mapped[str] = mapped_column(String(20), nullable=False, default="binance")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    listed_exchanges: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    first_seen: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=dt.datetime.utcnow
    )
    last_updated: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow
    )

    # relationships
    snapshots = relationship("Snapshot", back_populates="symbol_rel", lazy="dynamic")
    signal_scores = relationship("SignalScore", back_populates="symbol_rel", lazy="dynamic")
    pump_events = relationship("PumpEvent", back_populates="symbol_rel", lazy="dynamic")


# ---------------------------------------------------------------------------
# Snapshots — point-in-time market data per symbol per exchange
# ---------------------------------------------------------------------------

class Snapshot(Base):
    __tablename__ = "snapshots"
    __table_args__ = (
        Index("ix_snapshots_symbol_ts", "symbol_id", "timestamp"),
        Index("ix_snapshots_exchange_ts", "exchange", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False
    )
    exchange: Mapped[str] = mapped_column(String(20), nullable=False)
    timestamp: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=dt.datetime.utcnow
    )

    # price data
    price: Mapped[Optional[float]] = mapped_column(Float)
    price_high: Mapped[Optional[float]] = mapped_column(Float)
    price_low: Mapped[Optional[float]] = mapped_column(Float)
    price_open: Mapped[Optional[float]] = mapped_column(Float)
    price_close: Mapped[Optional[float]] = mapped_column(Float)

    # volume
    volume_base: Mapped[Optional[float]] = mapped_column(Float)
    volume_quote: Mapped[Optional[float]] = mapped_column(Float)
    volume_24h_quote: Mapped[Optional[float]] = mapped_column(Float)
    buy_volume_ratio: Mapped[Optional[float]] = mapped_column(Float)

    # derivatives data
    open_interest: Mapped[Optional[float]] = mapped_column(Float)
    open_interest_value: Mapped[Optional[float]] = mapped_column(Float)
    funding_rate: Mapped[Optional[float]] = mapped_column(Float)
    next_funding_time: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    long_short_ratio: Mapped[Optional[float]] = mapped_column(Float)
    top_trader_ls_ratio: Mapped[Optional[float]] = mapped_column(Float)

    # orderbook
    best_bid: Mapped[Optional[float]] = mapped_column(Float)
    best_ask: Mapped[Optional[float]] = mapped_column(Float)
    spread_pct: Mapped[Optional[float]] = mapped_column(Float)
    bid_depth_usd: Mapped[Optional[float]] = mapped_column(Float)
    ask_depth_usd: Mapped[Optional[float]] = mapped_column(Float)
    depth_imbalance: Mapped[Optional[float]] = mapped_column(Float)

    # liquidations (from forceOrder stream)
    liq_buy_volume: Mapped[Optional[float]] = mapped_column(Float, default=0)
    liq_sell_volume: Mapped[Optional[float]] = mapped_column(Float, default=0)
    liq_count: Mapped[Optional[int]] = mapped_column(Integer, default=0)

    # whale trades (from aggTrade stream, filtered by size)
    whale_buy_volume: Mapped[Optional[float]] = mapped_column(Float, default=0)
    whale_sell_volume: Mapped[Optional[float]] = mapped_column(Float, default=0)
    whale_trade_count: Mapped[Optional[int]] = mapped_column(Integer, default=0)

    # raw extras
    extra: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    symbol_rel = relationship("Symbol", back_populates="snapshots")


# ---------------------------------------------------------------------------
# Signal scores — computed once per scan cycle
# ---------------------------------------------------------------------------

class SignalScore(Base):
    __tablename__ = "signal_scores"
    __table_args__ = (
        Index("ix_signals_symbol_ts", "symbol_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=dt.datetime.utcnow
    )

    # individual signals (0-100 each, NULL if data unavailable)
    oi_surge: Mapped[Optional[float]] = mapped_column(Float)
    funding_rate: Mapped[Optional[float]] = mapped_column(Float)
    liquidation_leverage: Mapped[Optional[float]] = mapped_column(Float)
    cross_exchange_volume: Mapped[Optional[float]] = mapped_column(Float)
    depth_imbalance: Mapped[Optional[float]] = mapped_column(Float)
    volume_price_decouple: Mapped[Optional[float]] = mapped_column(Float)
    volatility_compression: Mapped[Optional[float]] = mapped_column(Float)
    long_short_ratio: Mapped[Optional[float]] = mapped_column(Float)
    futures_spot_divergence: Mapped[Optional[float]] = mapped_column(Float)
    whale_activity: Mapped[Optional[float]] = mapped_column(Float)

    # composite
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    bonuses_applied: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    penalties_applied: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    classification: Mapped[Optional[str]] = mapped_column(String(20))

    # event flags
    is_score_jump: Mapped[bool] = mapped_column(Boolean, default=False)
    is_upgrade: Mapped[bool] = mapped_column(Boolean, default=False)
    is_ignition: Mapped[bool] = mapped_column(Boolean, default=False)

    # smart levels (only for CRITICAL/HIGH_ALERT)
    entry_price: Mapped[Optional[float]] = mapped_column(Float)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float)
    tp1: Mapped[Optional[float]] = mapped_column(Float)
    tp2: Mapped[Optional[float]] = mapped_column(Float)
    tp3: Mapped[Optional[float]] = mapped_column(Float)

    symbol_rel = relationship("Symbol", back_populates="signal_scores")


# ---------------------------------------------------------------------------
# Pump events — detected pump occurrences
# ---------------------------------------------------------------------------

class PumpEvent(Base):
    __tablename__ = "pump_events"
    __table_args__ = (
        Index("ix_pumps_symbol_start", "symbol_id", "start_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False
    )

    # timing
    start_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    peak_time: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))

    # price movement
    start_price: Mapped[float] = mapped_column(Float, nullable=False)
    peak_price: Mapped[Optional[float]] = mapped_column(Float)
    end_price: Mapped[Optional[float]] = mapped_column(Float)
    pump_pct: Mapped[Optional[float]] = mapped_column(Float)
    retrace_pct: Mapped[Optional[float]] = mapped_column(Float)

    # classification
    pump_type: Mapped[Optional[str]] = mapped_column(String(30))  # gradual, spike, squeeze
    duration_minutes: Mapped[Optional[int]] = mapped_column(Integer)

    # pre-pump signal scores
    pre_pump_score: Mapped[Optional[float]] = mapped_column(Float)
    pre_pump_signals: Mapped[Optional[dict]] = mapped_column(JSONB)

    # all extracted ML features
    features: Mapped[Optional[dict]] = mapped_column(JSONB)

    # status
    status: Mapped[str] = mapped_column(String(20), default="active")  # active, completed, false_alarm
    notes: Mapped[Optional[str]] = mapped_column(Text)

    symbol_rel = relationship("Symbol", back_populates="pump_events")
    datapoints = relationship("PumpDatapoint", back_populates="pump_event", lazy="dynamic")


# ---------------------------------------------------------------------------
# Pump datapoints — granular data within pump events
# ---------------------------------------------------------------------------

class PumpDatapoint(Base):
    __tablename__ = "pump_datapoints"
    __table_args__ = (
        Index("ix_pdp_event_ts", "pump_event_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    pump_event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("pump_events.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    phase: Mapped[str] = mapped_column(String(15), nullable=False)  # pre, during, post

    # market data at this point
    price: Mapped[float] = mapped_column(Float, nullable=False)
    volume_1m: Mapped[Optional[float]] = mapped_column(Float)
    open_interest: Mapped[Optional[float]] = mapped_column(Float)
    funding_rate: Mapped[Optional[float]] = mapped_column(Float)
    bid_depth: Mapped[Optional[float]] = mapped_column(Float)
    ask_depth: Mapped[Optional[float]] = mapped_column(Float)
    long_short_ratio: Mapped[Optional[float]] = mapped_column(Float)
    liq_volume: Mapped[Optional[float]] = mapped_column(Float)
    whale_volume: Mapped[Optional[float]] = mapped_column(Float)

    # signal scores at this point
    signal_scores: Mapped[Optional[dict]] = mapped_column(JSONB)

    pump_event = relationship("PumpEvent", back_populates="datapoints")


# ---------------------------------------------------------------------------
# ML Predictions
# ---------------------------------------------------------------------------

class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_pred_symbol_ts", "symbol_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=dt.datetime.utcnow
    )
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    features_used: Mapped[Optional[dict]] = mapped_column(JSONB)

    # outcome tracking
    outcome: Mapped[Optional[str]] = mapped_column(String(20))  # pump, no_pump, pending
    actual_move_pct: Mapped[Optional[float]] = mapped_column(Float)


# ---------------------------------------------------------------------------
# Trade alerts — sent notifications
# ---------------------------------------------------------------------------

class TradeAlert(Base):
    __tablename__ = "trade_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=dt.datetime.utcnow
    )
    alert_type: Mapped[str] = mapped_column(String(30), nullable=False)  # pump, prediction, upgrade
    classification: Mapped[str] = mapped_column(String(20), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    delivered: Mapped[bool] = mapped_column(Boolean, default=False)
    delivery_channel: Mapped[Optional[str]] = mapped_column(String(20))
