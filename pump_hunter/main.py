"""
Pump Hunter — main entry point and orchestrator.
Wires up all components, runs the scan loop, handles graceful shutdown.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import signal
import sys
import time
from typing import Dict, List, Optional

# Ensure the parent directory is on sys.path so that
# `from pump_hunter.xxx import ...` works when running `python main.py`
# from inside the pump_hunter directory.
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import structlog

# configure structlog first
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    context_class=dict,
)

logger = structlog.get_logger("pump_hunter")


async def run():
    """Main async entry point."""
    from pump_hunter.config.settings import load_settings, Settings
    from pump_hunter.storage.database import Database
    from pump_hunter.storage.redis_store import RedisStore
    from pump_hunter.storage.timeseries import TimeseriesStore
    from pump_hunter.collectors.symbol_manager import SymbolManager
    from pump_hunter.collectors.binance_ws import BinanceWebSocket
    from pump_hunter.collectors.binance_rest import BinanceREST
    from pump_hunter.collectors.multi_exchange import MultiExchangeCollector
    from pump_hunter.core.market_state import MarketStateManager
    from pump_hunter.core.aggregator import Aggregator
    from pump_hunter.core.regime_engine import RegimeEngine
    from pump_hunter.core.execution_simulator import ExecutionSimulator
    from pump_hunter.core.capital_allocator import CapitalAllocator
    from pump_hunter.core.paper_trader import PaperTrader
    from pump_hunter.detection.feature_engine import FeatureEngine
    from pump_hunter.detection.pump_detector import PumpDetector
    from pump_hunter.detection.pump_recorder import PumpRecorder
    from pump_hunter.detection.manipulation_filter import ManipulationFilter
    from pump_hunter.detection.survival_analysis import SurvivalAnalysis
    from pump_hunter.alerts.notifier import Notifier
    from pump_hunter.ml.predictor import Predictor
    from pump_hunter.ml.feature_drift import FeatureDriftMonitor

    # ── load config ──────────────────────────────────────────────
    settings = load_settings()
    logger.info(
        "config_loaded",
        exchange=settings.exchanges.primary,
        scan_interval=settings.general.scan_interval_seconds,
        ml_enabled=settings.ml.enabled,
    )

    # ── initialize components ────────────────────────────────────
    db = Database(settings)
    redis = RedisStore(settings)
    timeseries = TimeseriesStore(
        max_candles=settings.general.max_candles_per_symbol,
        exchanges=["binance"] + settings.exchanges.secondary,
    )

    # collectors
    symbol_mgr = SymbolManager(settings, db, redis)
    binance_ws = BinanceWebSocket(settings, redis, timeseries)
    binance_rest = BinanceREST(settings, db, redis)
    multi_exchange = MultiExchangeCollector(settings, db, redis)

    # core
    market_state = MarketStateManager(settings, redis)
    aggregator = Aggregator(market_state, multi_exchange, redis)

    # detection
    feature_engine = FeatureEngine(settings, db, redis, timeseries)
    pump_detector = PumpDetector(settings, timeseries)
    pump_recorder = PumpRecorder(settings, db, redis, market_state, timeseries)

    # alerts
    notifier = Notifier(settings)

    # advanced layers
    regime_engine = RegimeEngine(settings, redis, timeseries)
    manipulation_filter = ManipulationFilter(settings, redis, timeseries)
    execution_sim = ExecutionSimulator(settings, timeseries)
    survival = SurvivalAnalysis(settings, db)
    capital_alloc = CapitalAllocator(settings, market_state, timeseries)

    # feature drift monitor
    drift_monitor = FeatureDriftMonitor(settings) if settings.drift.enabled else None

    # paper trader (mandatory phase before live trading)
    paper_trader = (
        PaperTrader(settings, timeseries)
        if settings.execution.paper_trading
        else None
    )

    # ML
    predictor = Predictor(settings, db, market_state, timeseries) if settings.ml.enabled else None

    # ── state ────────────────────────────────────────────────────
    running = True
    scan_count = 0
    start_time = time.time()
    prev_scores: Dict[str, float] = {}
    prev_classes: Dict[str, str] = {}

    # ── graceful shutdown ────────────────────────────────────────
    def handle_signal(*_):
        nonlocal running
        running = False
        logger.info("shutdown_signal_received")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── start all services ───────────────────────────────────────
    try:
        logger.info("starting_services")

        await db.start()
        await redis.start()
        await notifier.start()

        # register notifier callbacks
        async def get_status():
            return {
                "running": running,
                "symbols": symbol_mgr.symbol_count,
                "last_scan": f"{scan_count} scans total",
                "snapshots": (await db.get_stats()).get("snapshots", 0),
                "active_pumps": len(pump_recorder.active_symbols),
                "ws_messages": binance_ws.get_stats().get("messages_total", 0),
                "uptime": str(dt.timedelta(seconds=int(time.time() - start_time))),
                "regime": regime_engine.get_regime_info(),
                "kill_switch": execution_sim.is_killed,
                "positions": capital_alloc.get_stats(),
            }

        async def get_watchlist():
            items = []
            for sym, score in prev_scores.items():
                cls = prev_classes.get(sym, "LOW")
                if cls in ("CRITICAL", "HIGH_ALERT", "WATCHLIST"):
                    items.append({"symbol": sym, "score": score, "classification": cls})
            items.sort(key=lambda x: x["score"], reverse=True)
            return items[:20]

        notifier.set_callbacks(
            get_status=get_status,
            get_watchlist=get_watchlist,
        )

        # discover symbols
        await symbol_mgr.start()
        symbols = symbol_mgr.active_symbols
        logger.info("symbols_discovered", count=len(symbols))

        if not symbols:
            logger.error("no_symbols_found")
            return

        # start REST poller
        await binance_rest.start(symbols)

        # start multi-exchange collector
        if settings.exchanges.secondary:
            await multi_exchange.start(symbols)

        # start WebSocket streams
        stream_batches = symbol_mgr.get_ws_stream_batches()
        await binance_ws.start(stream_batches)

        # load ML model if enabled
        if predictor and settings.ml.enabled:
            if predictor.load_model():
                logger.info("ml_model_loaded")
            else:
                logger.info("ml_no_model_available")

        # let WS data flow in
        logger.info("warming_up", seconds=10)
        await asyncio.sleep(10)

        # ── main scan loop ───────────────────────────────────────
        logger.info("scan_loop_started", interval=settings.general.scan_interval_seconds)

        while running:
            scan_start = time.time()
            scan_count += 1

            try:
                # refresh symbols periodically
                await symbol_mgr.refresh()
                symbols = symbol_mgr.active_symbols
                binance_rest.update_symbols(symbols)

                # sync market state from Redis
                await market_state.sync_all(symbols)

                # update regime (every 5 min internally)
                await regime_engine.update()
                regime_modifier = regime_engine.get_modifier()

                # update survival analysis (every 6h internally)
                await survival.update()

                # collect alert candidates for capital allocation
                alert_candidates = []

                # process each symbol
                alert_count = 0
                for symbol in symbols:
                    try:
                        state = market_state.get_state(symbol)
                        if state.price <= 0 or state.last_update <= 0:
                            continue

                        # record depth for spoof detection
                        manipulation_filter.record_depth_snapshot(
                            symbol,
                            state.bid_depth_usd if hasattr(state, 'bid_depth_usd') else 0,
                            state.ask_depth_usd if hasattr(state, 'ask_depth_usd') else 0,
                        )

                        # aggregate cross-exchange data
                        aggregated = await aggregator.aggregate(symbol)

                        # compute signal features
                        signals = await feature_engine.compute_all(symbol, state, aggregated)

                        # score
                        score_result = pump_detector.score(
                            signals, state,
                            prev_score=prev_scores.get(symbol),
                            prev_class=prev_classes.get(symbol),
                        )

                        composite_score = score_result["composite_score"]

                        # --- REGIME ADJUSTMENT ---
                        composite_score *= regime_modifier
                        score_result["composite_score"] = composite_score
                        score_result["regime"] = regime_engine.get_regime_info()

                        # --- MANIPULATION FILTER ---
                        composite_score, manip_results = manipulation_filter.evaluate(
                            symbol, state, signals, composite_score,
                        )
                        score_result["composite_score"] = composite_score
                        if manip_results:
                            score_result["manipulation_flags"] = manip_results

                        # re-classify after adjustments
                        classification = pump_detector._classify(composite_score)
                        score_result["classification"] = classification

                        # update tracking
                        prev_scores[symbol] = composite_score
                        prev_classes[symbol] = classification

                        # store signal score in DB
                        db_symbols = await db.get_active_symbols()
                        symbol_id = next(
                            (s.id for s in db_symbols if s.symbol == symbol), None
                        )
                        if symbol_id:
                            await db.insert_signal_score({
                                "symbol_id": symbol_id,
                                "composite_score": composite_score,
                                "classification": classification,
                                "scores": {k: v for k, v in signals.items() if v is not None},
                                "bonuses": score_result.get("bonuses_applied", {}),
                                "penalties": score_result.get("penalties_applied", {}),
                            })

                        # check for pump event
                        pump_info = pump_detector.detect_pump(state)
                        if pump_info:
                            await pump_recorder.start_recording(
                                symbol, pump_info, signals, score_result
                            )

                        # update existing pump recordings
                        if symbol in pump_recorder.active_symbols:
                            await pump_recorder.update(symbol, signals)

                        # collect alert candidates
                        if classification in ("CRITICAL", "HIGH_ALERT"):
                            alert_candidates.append({
                                "symbol": symbol,
                                "score": composite_score,
                                "classification": classification,
                                "state": state,
                                "levels": score_result.get("levels", {}),
                                "signals": signals,
                                "score_result": score_result,
                                "pump_info": pump_info,
                            })

                        # update leaderboard
                        await redis.update_leaderboard("ph:leaderboard:score", symbol, composite_score)

                        # record feature values for drift monitoring
                        if drift_monitor:
                            drift_monitor.record_features(signals)

                    except Exception as e:
                        logger.debug("symbol_scan_error", symbol=symbol, error=str(e))

                # --- TAIL RISK CHECK (before execution) ---
                btc_state = market_state.get_state("BTCUSDT")
                tail_risks = execution_sim.check_tail_risk(btc_state=btc_state)
                if tail_risks:
                    logger.warning("tail_risk_triggered", risks=list(tail_risks.keys()))

                # --- CAPITAL ALLOCATION ---
                if alert_candidates:
                    allocations = capital_alloc.allocate(alert_candidates)
                    for alloc in allocations:
                        if not alloc.get("allocated"):
                            continue

                        sym = alloc["symbol"]

                        # --- EXECUTION EVALUATION ---
                        exec_plan = execution_sim.evaluate(
                            sym, alloc["state"], alloc["score"],
                            alloc["classification"], alloc["levels"],
                        )

                        if not exec_plan.get("accept"):
                            logger.info("trade_rejected", symbol=sym,
                                        reason=exec_plan.get("reject_reason"))
                            continue

                        # --- PAPER TRADE ---
                        if paper_trader:
                            paper_result = paper_trader.open_position(
                                sym, alloc["state"], exec_plan,
                                alloc["score_result"],
                            )
                            exec_plan["paper_trade"] = paper_result
                            # apply funding size penalty from tail risk
                            if execution_sim.funding_size_penalty < 1.0:
                                exec_plan["position_size_usd"] = (
                                    exec_plan.get("position_size_usd", 0)
                                    * execution_sim.funding_size_penalty
                                )

                        # enrich score result with advanced data
                        sr = alloc["score_result"]
                        sr["execution"] = exec_plan
                        sr["survival"] = survival.get_survival_info()
                        sr["allocation"] = {
                            "position_size_usd": alloc.get("position_size_usd"),
                            "pct_of_equity": alloc.get("pct_of_equity"),
                        }

                        sent = await notifier.send_pump_alert(
                            sym, sr, alloc["signals"], alloc.get("pump_info"),
                        )
                        if sent:
                            alert_count += 1

                # ML predictions (every nth scan)
                if predictor and predictor.is_ready and scan_count % 5 == 0:
                    predictions = await predictor.predict_all(symbols)
                    for pred in predictions[:3]:  # top 3
                        await notifier.send_prediction_alert(
                            pred["symbol"],
                            pred["probability"],
                            pred["model_version"],
                            pred["features"],
                        )

                # update paper positions against live prices
                if paper_trader and paper_trader.active_count > 0:
                    closed = paper_trader.update_positions(
                        lambda s: market_state.get_state(s)
                    )
                    for result in closed:
                        execution_sim.record_result(
                            result["pnl_usd"], result["pnl_pct"]
                        )

                # drift monitoring (hourly internally)
                if drift_monitor and scan_count % 10 == 0:
                    regime_name = regime_engine.current_regime.value
                    drift_results = drift_monitor.compute_psi_all(regime_name)
                    critical_drift = [
                        f for f, r in drift_results.items()
                        if r.get("status") == "significant_drift"
                    ]
                    if critical_drift:
                        logger.warning("feature_drift_detected", features=critical_drift)

                # periodic data pruning
                if scan_count % 100 == 0:
                    await db.prune_old_data(days=settings.general.data_retention_days)

                scan_duration = time.time() - scan_start

                # log scan summary
                if scan_count % 10 == 0 or alert_count > 0:
                    top = sorted(prev_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                    logger.info(
                        "scan_complete",
                        scan=scan_count,
                        symbols=len(symbols),
                        alerts=alert_count,
                        duration=f"{scan_duration:.1f}s",
                        regime=regime_engine.current_regime.value,
                        top_scores={k: round(v, 1) for k, v in top},
                        active_pumps=len(pump_recorder.active_symbols),
                        positions=capital_alloc.active_count,
                        paper_positions=paper_trader.active_count if paper_trader else 0,
                        kill_switch=execution_sim.is_killed,
                    )

            except Exception as e:
                logger.error("scan_error", scan=scan_count, error=str(e))

            # sleep until next scan
            elapsed = time.time() - scan_start
            sleep_time = max(1, settings.general.scan_interval_seconds - elapsed)
            await asyncio.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        # ── shutdown ─────────────────────────────────────────────
        logger.info("shutting_down")
        await pump_recorder.force_close_all()
        await binance_ws.stop()
        await binance_rest.stop()
        await multi_exchange.stop()
        await notifier.stop()
        await redis.stop()
        await db.stop()
        logger.info(
            "shutdown_complete",
            total_scans=scan_count,
            uptime=str(dt.timedelta(seconds=int(time.time() - start_time))),
        )


def main():
    """Sync entry point."""
    print(
        "\n"
        "╔═══════════════════════════════════════════╗\n"
        "║    🔍 PUMP HUNTER v1.0                    ║\n"
        "║    Market Intelligence System              ║\n"
        "║    Binance Futures Pump Detection           ║\n"
        "╚═══════════════════════════════════════════╝\n"
    )

    try:
        import uvloop
        uvloop.install()
        logger.info("uvloop_installed")
    except ImportError:
        logger.info("uvloop_not_available_using_default_loop")

    asyncio.run(run())


if __name__ == "__main__":
    main()
