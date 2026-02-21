"""
Paper trader — simulates trade execution against real-time data.

Mandatory phase before live trading. Tracks:
    - Virtual fills with realistic slippage
    - Live-vs-backtest gap metrics
    - Full trade journal (JSON-serializable)
    - Performance metrics: Sharpe, Sortino, max drawdown, win rate
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import structlog

from pump_hunter.config.settings import Settings
from pump_hunter.core.market_state import SymbolState
from pump_hunter.storage.timeseries import TimeseriesStore

logger = structlog.get_logger(__name__)


class PaperTrader:
    """
    Paper trading engine — no real orders, full execution simulation.
    """

    def __init__(
        self,
        settings: Settings,
        timeseries: TimeseriesStore,
        journal_path: str = "data/paper_journal.jsonl",
    ):
        self.cfg = settings.execution
        self.ts = timeseries
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

        # open positions: symbol → position dict
        self._positions: Dict[str, dict] = {}
        # closed trades
        self._closed: List[dict] = []
        # equity curve
        self._equity_curve: List[float] = []
        self._starting_equity = float(self.cfg.default_trade_size_usd) * 10  # 10× base
        self._equity = self._starting_equity
        self._peak_equity = self._equity
        self._max_drawdown_pct = 0.0

        # gap tracking: expected fill vs actual market state at fill time
        self._gap_metrics: List[dict] = []

    # ------------------------------------------------------------------
    # position management
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        return len(self._positions)

    def open_position(
        self,
        symbol: str,
        state: SymbolState,
        exec_plan: dict,
        score_result: dict,
    ) -> dict:
        """
        Open a paper position. Records expected vs simulated fill.
        """
        if symbol in self._positions:
            return {"opened": False, "reason": "already_open"}

        expected_entry = exec_plan.get("entry_price", state.price)
        # simulate realistic fill: add jitter based on volatility
        fill_slippage = self._simulate_fill_slippage(symbol, state)
        actual_fill = expected_entry * (1 + fill_slippage / 100)

        position = {
            "symbol": symbol,
            "entry_price": actual_fill,
            "expected_entry": expected_entry,
            "fill_slippage_pct": round(fill_slippage, 4),
            "size_usd": exec_plan.get("position_size_usd", self.cfg.default_trade_size_usd),
            "stop_loss": exec_plan.get("stop_loss", actual_fill * 0.97),
            "take_profits": exec_plan.get("take_profits", []),
            "open_time": time.time(),
            "score": score_result.get("composite_score", 0),
            "classification": score_result.get("classification", ""),
            "peak_price": actual_fill,
            "trough_price": actual_fill,
            "partial_exits": [],
        }

        self._positions[symbol] = position

        # log gap metric
        gap = {
            "symbol": symbol,
            "expected_fill": expected_entry,
            "actual_fill": actual_fill,
            "gap_pct": round((actual_fill - expected_entry) / expected_entry * 100, 4),
            "time": time.time(),
        }
        self._gap_metrics.append(gap)

        self._write_journal_entry("OPEN", position)

        logger.info(
            "paper_position_opened",
            symbol=symbol,
            entry=round(actual_fill, 6),
            size=round(position["size_usd"], 2),
            slippage=f"{fill_slippage:.3f}%",
        )

        return {"opened": True, "position": position}

    def update_positions(self, market_state_getter) -> List[dict]:
        """
        Update all open positions against current market data.
        Triggers stops and TPs. Returns list of closed position results.
        """
        closed_this_round = []

        for symbol in list(self._positions.keys()):
            pos = self._positions[symbol]
            state = market_state_getter(symbol)
            if state is None or state.price <= 0:
                continue

            current_price = state.price
            pos["peak_price"] = max(pos["peak_price"], current_price)
            pos["trough_price"] = min(pos["trough_price"], current_price)

            # check stop loss
            if current_price <= pos["stop_loss"]:
                result = self._close_position(symbol, current_price, "stop_loss")
                closed_this_round.append(result)
                continue

            # check take profits (partial exits)
            remaining_tps = [tp for tp in pos["take_profits"]
                             if tp not in [pe.get("tp") for pe in pos["partial_exits"]]]

            for tp in remaining_tps:
                tp_price = tp.get("price", tp) if isinstance(tp, dict) else tp
                if current_price >= tp_price:
                    # partial exit: 33% per TP level
                    partial_size = pos["size_usd"] * 0.33
                    pnl = partial_size * (current_price - pos["entry_price"]) / pos["entry_price"]

                    pos["partial_exits"].append({
                        "tp": tp,
                        "price": current_price,
                        "size_usd": partial_size,
                        "pnl_usd": round(pnl, 2),
                        "time": time.time(),
                    })

                    pos["size_usd"] -= partial_size
                    self._equity += pnl

                    logger.info(
                        "paper_partial_exit",
                        symbol=symbol,
                        tp_price=tp_price,
                        pnl=round(pnl, 2),
                    )

            # if position fully exited through partials
            if pos["size_usd"] < 10:  # < $10 remaining
                result = self._close_position(symbol, current_price, "all_tps_hit")
                closed_this_round.append(result)

        return closed_this_round

    def _close_position(self, symbol: str, exit_price: float, reason: str) -> dict:
        """Close a paper position and record results."""
        pos = self._positions.pop(symbol)

        # P&L on remaining size
        remaining_pnl = pos["size_usd"] * (exit_price - pos["entry_price"]) / pos["entry_price"]
        partial_pnl = sum(pe["pnl_usd"] for pe in pos["partial_exits"])
        total_pnl = remaining_pnl + partial_pnl

        self._equity += remaining_pnl
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = (self._peak_equity - self._equity) / self._peak_equity * 100
        self._max_drawdown_pct = max(self._max_drawdown_pct, drawdown)
        self._equity_curve.append(self._equity)

        result = {
            "symbol": symbol,
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "reason": reason,
            "pnl_usd": round(total_pnl, 2),
            "pnl_pct": round(total_pnl / (pos["size_usd"] + sum(pe["size_usd"] for pe in pos["partial_exits"])) * 100, 2),
            "hold_time_min": round((time.time() - pos["open_time"]) / 60, 1),
            "peak_price": pos["peak_price"],
            "trough_price": pos["trough_price"],
            "max_favorable_pct": round((pos["peak_price"] - pos["entry_price"]) / pos["entry_price"] * 100, 2),
            "max_adverse_pct": round((pos["entry_price"] - pos["trough_price"]) / pos["entry_price"] * 100, 2),
            "partial_exits": len(pos["partial_exits"]),
            "score_at_entry": pos["score"],
            "classification_at_entry": pos["classification"],
        }

        self._closed.append(result)
        self._write_journal_entry("CLOSE", result)

        logger.info(
            "paper_position_closed",
            symbol=symbol,
            pnl=f"${total_pnl:.2f}",
            reason=reason,
            hold=f"{result['hold_time_min']}m",
        )

        return result

    # ------------------------------------------------------------------
    # fill simulation
    # ------------------------------------------------------------------

    def _simulate_fill_slippage(self, symbol: str, state: SymbolState) -> float:
        """
        Simulate realistic fill slippage based on:
        - Orderbook depth
        - Recent volatility
        - Trade size relative to liquidity
        """
        trade_size = self.cfg.default_trade_size_usd
        ask_depth = state.ask_depth_usd if hasattr(state, "ask_depth_usd") else 0

        # base slippage from depth
        if ask_depth > 0:
            depth_slip = (trade_size / ask_depth) * 0.5 * 100
        else:
            depth_slip = 0.3  # 0.3% default

        # volatility component: use recent returns to add realistic jitter
        closes = self.ts.get_closes(symbol, "binance", last_n=20)
        if len(closes) >= 5:
            returns = np.abs(np.diff(closes) / closes[:-1])
            vol_jitter = float(np.std(returns)) * 100 * np.random.uniform(0, 1)
        else:
            vol_jitter = 0.05

        # always adverse for longs (price moves up while filling)
        total_slip = depth_slip + vol_jitter
        return min(total_slip, 2.0)  # cap at 2%

    # ------------------------------------------------------------------
    # performance metrics
    # ------------------------------------------------------------------

    def get_performance(self) -> dict:
        """Compute comprehensive paper trading performance metrics."""
        if not self._closed:
            return {
                "trades": 0,
                "equity": round(self._equity, 2),
                "pnl": 0,
                "status": "no_trades_yet",
            }

        pnls = [t["pnl_usd"] for t in self._closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        pnl_pcts = [t["pnl_pct"] for t in self._closed]

        # Sharpe ratio (annualized, assuming 1 trade per day)
        if len(pnl_pcts) >= 2:
            mean_ret = np.mean(pnl_pcts)
            std_ret = np.std(pnl_pcts)
            sharpe = (mean_ret / std_ret * math.sqrt(365)) if std_ret > 0 else 0
        else:
            sharpe = 0

        # Sortino ratio (only downside deviation)
        downside = [r for r in pnl_pcts if r < 0]
        if downside and len(pnl_pcts) >= 2:
            downside_std = float(np.std(downside))
            sortino = (float(np.mean(pnl_pcts)) / downside_std * math.sqrt(365)) if downside_std > 0 else 0
        else:
            sortino = 0

        # profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # avg hold time
        hold_times = [t["hold_time_min"] for t in self._closed]

        return {
            "trades": len(self._closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(self._closed) * 100, 1),
            "total_pnl_usd": round(sum(pnls), 2),
            "avg_pnl_usd": round(float(np.mean(pnls)), 2),
            "best_trade_usd": round(max(pnls), 2),
            "worst_trade_usd": round(min(pnls), 2),
            "equity": round(self._equity, 2),
            "return_pct": round((self._equity - self._starting_equity) / self._starting_equity * 100, 2),
            "max_drawdown_pct": round(self._max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
            "avg_hold_min": round(float(np.mean(hold_times)), 1) if hold_times else 0,
            "active_positions": len(self._positions),
        }

    def get_gap_metrics(self) -> dict:
        """Compute live-vs-backtest fill gap statistics."""
        if not self._gap_metrics:
            return {"n_trades": 0}

        gaps = [g["gap_pct"] for g in self._gap_metrics]

        return {
            "n_trades": len(gaps),
            "mean_gap_pct": round(float(np.mean(gaps)), 4),
            "median_gap_pct": round(float(np.median(gaps)), 4),
            "max_gap_pct": round(max(gaps), 4),
            "std_gap_pct": round(float(np.std(gaps)), 4),
            "positive_gap_pct": round(sum(1 for g in gaps if g > 0) / len(gaps) * 100, 1),
        }

    # ------------------------------------------------------------------
    # journal
    # ------------------------------------------------------------------

    def _write_journal_entry(self, action: str, data: dict) -> None:
        """Append a journal entry to the JSONL file."""
        entry = {
            "action": action,
            "timestamp": time.time(),
            "data": self._serialize(data),
        }
        try:
            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.debug("journal_write_error", error=str(e))

    @staticmethod
    def _serialize(obj):
        """Make a dict JSON-serializable."""
        if isinstance(obj, dict):
            return {k: PaperTrader._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [PaperTrader._serialize(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
