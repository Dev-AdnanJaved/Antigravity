"""
Execution simulator — models realistic trade execution for each alert.

Components:
    - Entry mode selection (market / limit / pullback / adaptive)
    - Slippage estimation from orderbook depth
    - Latency simulation (configurable delay added to entry price)
    - Position sizing via fractional Kelly criterion
    - Stop management with trailing schedule
    - Kill switch: auto-pause after consecutive losses or drawdown
"""

from __future__ import annotations

import datetime as dt
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from pump_hunter.config.settings import Settings
from pump_hunter.core.market_state import SymbolState
from pump_hunter.storage.timeseries import TimeseriesStore

logger = structlog.get_logger(__name__)


class ExecutionSimulator:
    """Simulates trade execution to produce realistic P&L estimates."""

    def __init__(self, settings: Settings, timeseries: TimeseriesStore):
        self.cfg = settings.execution
        self.level_cfg = settings.levels
        self.ts = timeseries

        # kill switch state
        self._consecutive_losses = 0
        self._peak_equity = 0.0
        self._current_equity = 0.0
        self._killed = False
        self._kill_time = 0.0
        self._kill_reason = ""

        # trade journal
        self._trades: List[dict] = []
        self._win_rate = 0.5  # initial assumption

        # tail risk state
        self._api_results: List[bool] = []  # True=success, False=error
        self._liq_events: List[Tuple[float, str]] = []  # (timestamp, direction)
        self._symbol_kills: Dict[str, float] = {}  # symbol → kill time
        self._funding_penalty_active = False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def is_killed(self) -> bool:
        """Check if global kill switch is active."""
        if not self._killed:
            return False
        # check cooldown
        elapsed = time.time() - self._kill_time
        if elapsed > self.cfg.kill_switch_cooldown_minutes * 60:
            self._killed = False
            self._consecutive_losses = 0
            self._kill_reason = ""
            logger.info("kill_switch_deactivated", cooldown_elapsed=True)
            return False
        return True

    def is_symbol_killed(self, symbol: str) -> bool:
        """Check if a specific symbol is killed (mark dislocation, liq cascade)."""
        kill_time = self._symbol_kills.get(symbol)
        if kill_time is None:
            return False
        if time.time() - kill_time > 15 * 60:  # 15 min cooldown per symbol
            del self._symbol_kills[symbol]
            return False
        return True

    # ------------------------------------------------------------------
    # tail risk: event recording
    # ------------------------------------------------------------------

    def record_api_result(self, success: bool) -> None:
        """Record an API call result for error rate monitoring."""
        self._api_results.append(success)
        window = self.cfg.tail_risk_api_error_window
        if len(self._api_results) > window * 2:
            self._api_results = self._api_results[-window:]

    def record_liquidation_event(self, symbol: str, direction: str) -> None:
        """Record a forced liquidation event for cascade detection."""
        now = time.time()
        self._liq_events.append((now, direction))
        # trim old events
        cutoff = now - self.cfg.tail_risk_liq_cascade_window_sec
        self._liq_events = [(t, d) for t, d in self._liq_events if t >= cutoff]

    # ------------------------------------------------------------------
    # tail risk: check triggers
    # ------------------------------------------------------------------

    def check_tail_risk(
        self,
        btc_state=None,
        current_symbol: Optional[str] = None,
        mark_price: Optional[float] = None,
        last_price: Optional[float] = None,
        current_funding: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Check all tail risk conditions. Returns dict of triggered risks.
        Called before each evaluation cycle.
        """
        triggered = {}

        # 1. BTC flash crash
        if btc_state is not None:
            btc_change = btc_state.get_price_change(
                minutes=self.cfg.tail_risk_btc_crash_window_min
            )
            if btc_change <= -self.cfg.tail_risk_btc_crash_pct:
                triggered["btc_crash"] = {
                    "change_pct": round(btc_change, 2),
                    "threshold": -self.cfg.tail_risk_btc_crash_pct,
                }
                self._activate_kill("btc_crash", f"BTC dropped {btc_change:.1f}% in {self.cfg.tail_risk_btc_crash_window_min}m")

        # 2. Mark price dislocation (per symbol)
        if mark_price and last_price and last_price > 0 and current_symbol:
            dislocation = abs(mark_price - last_price) / last_price * 100
            if dislocation >= self.cfg.tail_risk_mark_dislocation_pct:
                triggered["mark_dislocation"] = {
                    "symbol": current_symbol,
                    "gap_pct": round(dislocation, 2),
                    "threshold": self.cfg.tail_risk_mark_dislocation_pct,
                }
                self._symbol_kills[current_symbol] = time.time()
                logger.warning(
                    "tail_risk_mark_dislocation",
                    symbol=current_symbol,
                    gap_pct=round(dislocation, 2),
                )

        # 3. API error rate
        window = self.cfg.tail_risk_api_error_window
        recent_api = self._api_results[-window:]
        if len(recent_api) >= window // 2:
            error_rate = sum(1 for r in recent_api if not r) / len(recent_api)
            if error_rate >= self.cfg.tail_risk_api_error_threshold:
                triggered["api_errors"] = {
                    "error_rate": round(error_rate, 2),
                    "threshold": self.cfg.tail_risk_api_error_threshold,
                    "recent_calls": len(recent_api),
                }
                self._activate_kill("api_errors", f"API error rate {error_rate:.0%}")

        # 4. Funding spike
        if current_funding is not None:
            funding_abs = abs(current_funding) * 100  # convert to %
            if funding_abs >= self.cfg.tail_risk_funding_spike_pct:
                triggered["funding_spike"] = {
                    "funding_pct": round(funding_abs, 3),
                    "threshold": self.cfg.tail_risk_funding_spike_pct,
                    "action": "reduce_size_50pct",
                }
                self._funding_penalty_active = True
                logger.warning("tail_risk_funding_spike", funding=round(funding_abs, 3))
            else:
                self._funding_penalty_active = False

        # 5. Liquidation cascade
        now = time.time()
        cutoff = now - self.cfg.tail_risk_liq_cascade_window_sec
        recent_liqs = [(t, d) for t, d in self._liq_events if t >= cutoff]
        if len(recent_liqs) >= self.cfg.tail_risk_liq_cascade_count:
            # check if same direction (correlated)
            directions = [d for _, d in recent_liqs]
            dominant = max(set(directions), key=directions.count)
            if directions.count(dominant) >= self.cfg.tail_risk_liq_cascade_count:
                triggered["liq_cascade"] = {
                    "count": len(recent_liqs),
                    "direction": dominant,
                    "window_sec": self.cfg.tail_risk_liq_cascade_window_sec,
                }
                self._activate_kill(
                    "liq_cascade",
                    f"{len(recent_liqs)} {dominant} liquidations in {self.cfg.tail_risk_liq_cascade_window_sec}s",
                )

        return triggered

    def _activate_kill(self, reason: str, detail: str) -> None:
        """Activate global kill switch from tail risk."""
        if not self._killed:
            self._killed = True
            self._kill_time = time.time()
            self._kill_reason = reason
            logger.warning(
                "tail_risk_kill_switch",
                reason=reason,
                detail=detail,
                cooldown_min=self.cfg.kill_switch_cooldown_minutes,
            )

    @property
    def funding_size_penalty(self) -> float:
        """Returns position size multiplier (0.5 if funding spike, 1.0 otherwise)."""
        return 0.5 if self._funding_penalty_active else 1.0

    def evaluate(
        self,
        symbol: str,
        state: SymbolState,
        score: float,
        classification: str,
        levels: dict,
    ) -> dict:
        """
        Evaluate an alert and produce execution plan.

        Returns dict with:
            - accept: bool
            - reject_reason: str
            - entry_mode: str
            - entry_price: float
            - slippage_pct: float
            - position_size_usd: float
            - stop_loss: float
            - take_profits: list
            - estimated_pnl: dict
        """
        if not self.cfg.enabled:
            return {"accept": True, "entry_mode": "disabled"}

        if self.is_killed:
            return {
                "accept": False,
                "reject_reason": "kill_switch_active",
                "cooldown_remaining_min": round(
                    (self.cfg.kill_switch_cooldown_minutes * 60 - (time.time() - self._kill_time)) / 60, 1
                ),
            }

        price = state.price
        if price <= 0:
            return {"accept": False, "reject_reason": "invalid_price"}

        # 1. select entry mode
        entry_mode = self._select_entry_mode(symbol, state, classification)

        # 2. estimate slippage
        slippage_pct = self._estimate_slippage(state)
        if slippage_pct > self.cfg.max_slippage_pct:
            return {
                "accept": False,
                "reject_reason": f"slippage_too_high ({slippage_pct:.2f}%)",
                "slippage_pct": slippage_pct,
            }

        # 3. compute entry price with latency
        entry_price = self._apply_latency(symbol, price)
        entry_price *= (1 + slippage_pct / 100)  # add slippage

        # 4. compute position size
        stop_loss = levels.get("stop_loss", price * 0.97)
        position_size = self._compute_position_size(
            entry_price, stop_loss, state.volume_24h if hasattr(state, "volume_24h") else 0,
        )

        # 5. estimated P&L
        take_profits = levels.get("take_profits", [])
        pnl_estimate = self._estimate_pnl(entry_price, stop_loss, take_profits, position_size)

        return {
            "accept": True,
            "entry_mode": entry_mode,
            "entry_price": round(entry_price, 6),
            "slippage_pct": round(slippage_pct, 3),
            "position_size_usd": round(position_size, 2),
            "stop_loss": round(stop_loss, 6),
            "take_profits": take_profits,
            "estimated_pnl": pnl_estimate,
            "kill_switch_status": {
                "consecutive_losses": self._consecutive_losses,
                "remaining_before_kill": self.cfg.kill_switch_consecutive_losses - self._consecutive_losses,
            },
        }

    # ------------------------------------------------------------------
    # entry mode selection
    # ------------------------------------------------------------------

    def _select_entry_mode(
        self, symbol: str, state: SymbolState, classification: str
    ) -> str:
        """Select entry mode based on conditions."""
        if self.cfg.entry_mode != "adaptive":
            return self.cfg.entry_mode

        # adaptive logic:
        # - CRITICAL → market order (don't miss it)
        # - HIGH_ALERT + thin book → limit order (protect from slippage)
        # - HIGH_ALERT + deep book → pullback (better entry)
        ask_depth = state.ask_depth_usd if hasattr(state, "ask_depth_usd") else 0

        if classification == "CRITICAL":
            return "market"
        elif ask_depth < 10_000:
            return "limit"
        else:
            return "pullback"

    # ------------------------------------------------------------------
    # slippage estimation
    # ------------------------------------------------------------------

    def _estimate_slippage(self, state: SymbolState) -> float:
        """Estimate slippage based on orderbook depth vs trade size."""
        trade_size = self.cfg.default_trade_size_usd
        ask_depth = state.ask_depth_usd if hasattr(state, "ask_depth_usd") else 0

        if ask_depth <= 0:
            return 1.0  # assume 1% if no depth data

        # slippage = trade_size / depth_available * impact_factor
        depth_ratio = trade_size / ask_depth
        impact_factor = 0.5  # empirical: you rarely eat the full book

        slippage = depth_ratio * impact_factor * 100  # in %
        return min(slippage, 5.0)  # cap at 5%

    # ------------------------------------------------------------------
    # latency simulation
    # ------------------------------------------------------------------

    def _apply_latency(self, symbol: str, current_price: float) -> float:
        """Simulate latency: price moves adversely during delay."""
        latency_ms = self.cfg.latency_ms
        if latency_ms <= 0:
            return current_price

        df = self.ts.get_dataframe(symbol, "binance", last_n=20)
        if len(df) < 5:
            # assume 0.1% adverse move per 100ms
            return current_price * (1 + 0.001 * latency_ms / 100)

        # use recent volatility to estimate price movement during latency
        closes = df["close"].values
        if len(closes) > 1:
            returns_per_candle = np.abs(np.diff(closes) / closes[:-1])
            avg_move_per_min = float(np.mean(returns_per_candle))
            # scale to latency period (candle = 1 min = 60_000ms)
            expected_move = avg_move_per_min * (latency_ms / 60_000)
            # adverse direction for long entry
            return current_price * (1 + expected_move)

        return current_price

    # ------------------------------------------------------------------
    # position sizing (fractional Kelly)
    # ------------------------------------------------------------------

    def _compute_position_size(
        self, entry: float, stop: float, volume_24h: float
    ) -> float:
        """
        Position size using fractional Kelly criterion + liquidity cap.

        Kelly formula: f* = (p * b - q) / b
          p = win probability
          b = avg win / avg loss ratio
          q = 1 - p
        """
        base_size = self.cfg.default_trade_size_usd

        # Kelly sizing
        p = self._win_rate
        q = 1.0 - p
        b = 2.0  # assume avg winner is 2× avg loser (conservative)

        kelly_full = (p * b - q) / b if b > 0 else 0.01
        kelly_full = max(kelly_full, 0.01)  # minimum 1%
        kelly_fraction = kelly_full * self.cfg.kelly_fraction

        kelly_size = base_size * kelly_fraction / 0.25  # normalize around default fraction

        # liquidity cap: max % of 24h volume
        if volume_24h > 0:
            liquidity_cap = volume_24h * (self.cfg.max_slippage_pct / 100)
            kelly_size = min(kelly_size, liquidity_cap)

        # risk-based cap: don't risk more than stop distance allows
        if entry > 0 and stop > 0:
            risk_per_unit = abs(entry - stop) / entry
            if risk_per_unit > 0:
                max_risk_size = base_size * 0.02 / risk_per_unit  # 2% max risk
                kelly_size = min(kelly_size, max_risk_size)

        return max(kelly_size, 100)  # minimum $100

    # ------------------------------------------------------------------
    # P&L estimation
    # ------------------------------------------------------------------

    def _estimate_pnl(
        self,
        entry: float,
        stop: float,
        take_profits: list,
        size_usd: float,
    ) -> dict:
        """Estimate best/worst/expected P&L."""
        if entry <= 0:
            return {}

        risk_pct = abs(entry - stop) / entry * 100
        risk_usd = size_usd * risk_pct / 100

        tp_pnls = []
        for tp in take_profits:
            tp_val = tp.get("price", tp) if isinstance(tp, dict) else tp
            if tp_val > entry:
                reward_pct = (tp_val - entry) / entry * 100
                tp_pnls.append(size_usd * reward_pct / 100)

        best_case = tp_pnls[-1] if tp_pnls else size_usd * 0.1
        worst_case = -risk_usd

        # expected value = p(win) × avg_win - p(loss) × loss
        avg_win = float(np.mean(tp_pnls)) if tp_pnls else best_case * 0.5
        expected = self._win_rate * avg_win - (1 - self._win_rate) * risk_usd

        return {
            "best_case_usd": round(best_case, 2),
            "worst_case_usd": round(worst_case, 2),
            "expected_value_usd": round(expected, 2),
            "risk_reward_ratio": round(best_case / risk_usd, 2) if risk_usd > 0 else 0,
            "risk_usd": round(risk_usd, 2),
            "risk_pct": round(risk_pct, 2),
        }

    # ------------------------------------------------------------------
    # trade result tracking (for Kelly update + kill switch)
    # ------------------------------------------------------------------

    def record_result(self, pnl_usd: float, pnl_pct: float) -> None:
        """Record a trade result to update win rate and kill switch."""
        self._trades.append({
            "time": time.time(),
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
        })

        # update win rate (rolling last 50 trades)
        recent = self._trades[-50:]
        wins = sum(1 for t in recent if t["pnl_usd"] > 0)
        self._win_rate = wins / len(recent) if recent else 0.5

        # kill switch: consecutive losses
        if pnl_usd < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        if self._consecutive_losses >= self.cfg.kill_switch_consecutive_losses:
            self._killed = True
            self._kill_time = time.time()
            logger.warning(
                "kill_switch_activated",
                reason="consecutive_losses",
                losses=self._consecutive_losses,
                cooldown_min=self.cfg.kill_switch_cooldown_minutes,
            )

        # kill switch: drawdown
        self._current_equity += pnl_usd
        self._peak_equity = max(self._peak_equity, self._current_equity)
        if self._peak_equity > 0:
            drawdown_pct = (self._peak_equity - self._current_equity) / self._peak_equity * 100
            if drawdown_pct >= self.cfg.kill_switch_drawdown_pct:
                self._killed = True
                self._kill_time = time.time()
                logger.warning(
                    "kill_switch_activated",
                    reason="drawdown",
                    drawdown_pct=round(drawdown_pct, 1),
                    cooldown_min=self.cfg.kill_switch_cooldown_minutes,
                )

    def get_stats(self) -> dict:
        """Return execution stats."""
        recent = self._trades[-50:]
        if not recent:
            return {"trades": 0, "win_rate": 0.5}

        wins = [t for t in recent if t["pnl_usd"] > 0]
        losses = [t for t in recent if t["pnl_usd"] <= 0]

        return {
            "trades": len(self._trades),
            "recent_trades": len(recent),
            "win_rate": round(self._win_rate, 3),
            "avg_win": round(float(np.mean([t["pnl_usd"] for t in wins])), 2) if wins else 0,
            "avg_loss": round(float(np.mean([t["pnl_usd"] for t in losses])), 2) if losses else 0,
            "consecutive_losses": self._consecutive_losses,
            "kill_switch_active": self._killed,
        }
