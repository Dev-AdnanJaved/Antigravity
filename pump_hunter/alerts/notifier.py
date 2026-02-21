"""
Notifier — Telegram bot with commands and rich alerts, plus console output.
Rate limited, async, with full signal breakdown in alerts.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from typing import Any, Dict, List, Optional

import structlog
from telegram import Bot, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from pump_hunter.config.settings import Settings

logger = structlog.get_logger(__name__)


class Notifier:
    """
    Telegram bot + console alert system.
    Sends pump alerts, prediction alerts, and handles interactive commands.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._app: Optional[Application] = None
        self._bot: Optional[Bot] = None
        self._last_alerts: Dict[str, float] = {}  # symbol -> timestamp
        self._running = False

        # external callbacks for commands
        self._on_scan = None
        self._get_watchlist = None
        self._get_status = None
        self._get_pump_detail = None

    def set_callbacks(
        self,
        on_scan=None,
        get_watchlist=None,
        get_status=None,
        get_pump_detail=None,
    ):
        """Set callback functions for interactive commands."""
        self._on_scan = on_scan
        self._get_watchlist = get_watchlist
        self._get_status = get_status
        self._get_pump_detail = get_pump_detail

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize and start the Telegram bot."""
        cfg = self.settings.telegram
        if not cfg.enabled or not cfg.bot_token:
            logger.info("telegram_disabled")
            return

        try:
            self._app = (
                Application.builder()
                .token(cfg.bot_token)
                .build()
            )

            # register commands
            if cfg.commands_enabled:
                self._app.add_handler(CommandHandler("status", self._cmd_status))
                self._app.add_handler(CommandHandler("scan", self._cmd_scan))
                self._app.add_handler(CommandHandler("watchlist", self._cmd_watchlist))
                self._app.add_handler(CommandHandler("pump", self._cmd_pump))
                self._app.add_handler(CommandHandler("config", self._cmd_config))
                self._app.add_handler(CommandHandler("alerts", self._cmd_alerts))
                self._app.add_handler(CommandHandler("help", self._cmd_help))

            await self._app.initialize()
            await self._app.start()
            if self._app.updater:
                await self._app.updater.start_polling(drop_pending_updates=True)

            self._bot = self._app.bot
            self._running = True

            logger.info("telegram_started")

            # send startup message
            await self._send(
                "🚀 *Pump Hunter started*\n"
                f"📊 Scan interval: {self.settings.general.scan_interval_seconds}s\n"
                f"🔔 Min alert score: {cfg.alert_min_score}\n"
                "Use /help for commands"
            )
        except Exception as e:
            logger.error("telegram_start_error", error=str(e))

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False
        if self._app:
            try:
                await self._send("🔴 *Pump Hunter stopping...*")
                if self._app.updater and self._app.updater.running:
                    await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                logger.debug("telegram_stop_error", error=str(e))
            logger.info("telegram_stopped")

    # ------------------------------------------------------------------
    # send messages
    # ------------------------------------------------------------------

    async def _send(self, text: str, parse_mode: str = "Markdown") -> None:
        """Send a message to the configured chat."""
        if not self._bot or not self.settings.telegram.chat_id:
            return
        try:
            await self._bot.send_message(
                chat_id=self.settings.telegram.chat_id,
                text=text,
                parse_mode=parse_mode,
            )
        except Exception as e:
            logger.error("telegram_send_error", error=str(e))

    # ------------------------------------------------------------------
    # pump alert
    # ------------------------------------------------------------------

    async def send_pump_alert(
        self,
        symbol: str,
        score_result: dict,
        signals: Dict[str, Optional[float]],
        pump_info: Optional[dict] = None,
    ) -> bool:
        """
        Send a pump detection alert via Telegram + console.
        Returns True if sent (not rate limited).
        """
        # rate limit check
        now = time.time()
        rate_limit = self.settings.telegram.rate_limit_seconds
        if symbol in self._last_alerts:
            if now - self._last_alerts[symbol] < rate_limit:
                return False

        score = score_result.get("composite_score", 0)
        if score < self.settings.telegram.alert_min_score:
            return False

        classification = score_result.get("classification", "UNKNOWN")
        events = score_result.get("events", [])
        bonuses = score_result.get("bonuses_applied", {})
        levels = score_result.get("levels", {})

        # build signal bars
        signal_lines = []
        for name, value in sorted(signals.items(), key=lambda x: x[1] or 0, reverse=True):
            if value is not None:
                bar = self._make_bar(value)
                weight = self.settings.detection.score_weights.get(name, 0) * 100
                signal_lines.append(f"  {name}: {bar} {value:.0f} ({weight:.0f}%)")

        signal_text = "\n".join(signal_lines)

        # emoji by classification
        emoji = {"CRITICAL": "🔴", "HIGH_ALERT": "🟠", "WATCHLIST": "🟡", "MONITOR": "🔵"}.get(classification, "⚪")

        # build message
        msg_parts = [
            f"{emoji} *{classification}* — `{symbol}`",
            f"📊 Score: *{score:.1f}*",
        ]

        if pump_info:
            msg_parts.append(f"📈 Move: +{pump_info.get('change_pct', 0):.1f}% in {pump_info.get('window_minutes', 0)}m")

        if events:
            msg_parts.append(f"⚡ Events: {', '.join(events)}")

        if bonuses:
            bonus_text = ", ".join(f"{k} (+{v*100:.0f}%)" for k, v in bonuses.items())
            msg_parts.append(f"🎯 Bonuses: {bonus_text}")

        msg_parts.append(f"\n*Signals:*\n{signal_text}")

        if levels:
            msg_parts.append(
                f"\n*Levels:*\n"
                f"  Entry: `{levels.get('entry_price', 'N/A')}`\n"
                f"  Stop : `{levels.get('stop_loss', 'N/A')}` ({levels.get('stop_pct', 0):.1f}%)\n"
                f"  TP1  : `{levels.get('tp1', 'N/A')}`\n"
                f"  TP2  : `{levels.get('tp2', 'N/A')}`\n"
                f"  TP3  : `{levels.get('tp3', 'N/A')}`\n"
                f"  R:R  : {levels.get('risk_reward', 'N/A')}"
            )

        msg_parts.append(f"\n⏰ {dt.datetime.utcnow().strftime('%H:%M:%S UTC')}")

        full_msg = "\n".join(msg_parts)

        # send telegram
        if self.settings.telegram.enabled and self.settings.telegram.alert_on_pump:
            await self._send(full_msg)

        # console output
        self._console_alert(symbol, score, classification, signals, events, levels)

        self._last_alerts[symbol] = now
        return True

    async def send_prediction_alert(
        self,
        symbol: str,
        probability: float,
        model_version: str,
        features: dict,
    ) -> None:
        """Send ML prediction alert."""
        if not self.settings.telegram.alert_on_prediction:
            return

        msg = (
            f"🤖 *ML PREDICTION* — `{symbol}`\n"
            f"📊 Probability: *{probability*100:.1f}%*\n"
            f"🧠 Model: {model_version}\n"
            f"⏰ {dt.datetime.utcnow().strftime('%H:%M:%S UTC')}"
        )
        await self._send(msg)

    # ------------------------------------------------------------------
    # console output
    # ------------------------------------------------------------------

    def _console_alert(
        self,
        symbol: str,
        score: float,
        classification: str,
        signals: dict,
        events: list,
        levels: dict,
    ) -> None:
        """Print formatted console alert."""
        logger.info(
            "pump_alert",
            symbol=symbol,
            score=round(score, 1),
            classification=classification,
            events=events,
            top_signals={k: round(v, 1) for k, v in sorted(
                ((k, v) for k, v in signals.items() if v is not None),
                key=lambda x: x[1], reverse=True
            )[:5]},
            entry=levels.get("entry_price"),
            stop=levels.get("stop_loss"),
            tp1=levels.get("tp1"),
        )

    # ------------------------------------------------------------------
    # bot commands
    # ------------------------------------------------------------------

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        if self._get_status:
            status = await self._get_status()
            msg = (
                "📊 *Pump Hunter Status*\n\n"
                f"🟢 Running: {status.get('running', False)}\n"
                f"📈 Symbols: {status.get('symbols', 0)}\n"
                f"🔄 Last scan: {status.get('last_scan', 'Never')}\n"
                f"📦 Snapshots: {status.get('snapshots', 0)}\n"
                f"🎯 Active pumps: {status.get('active_pumps', 0)}\n"
                f"📡 WS messages: {status.get('ws_messages', 0)}\n"
                f"⏰ Uptime: {status.get('uptime', 'N/A')}"
            )
        else:
            msg = "📊 Status callback not configured"
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /scan command — force immediate scan."""
        await update.message.reply_text("🔄 Forcing scan...")
        if self._on_scan:
            await self._on_scan()
            await update.message.reply_text("✅ Scan complete")

    async def _cmd_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /watchlist command."""
        if self._get_watchlist:
            watchlist = await self._get_watchlist()
            if not watchlist:
                await update.message.reply_text("📋 No active watchlist items")
                return

            lines = ["📋 *Current Watchlist*\n"]
            for item in watchlist[:20]:
                emoji = {"CRITICAL": "🔴", "HIGH_ALERT": "🟠", "WATCHLIST": "🟡"}.get(
                    item.get("classification", ""), "⚪"
                )
                lines.append(
                    f"{emoji} `{item['symbol']}` — {item['score']:.1f} ({item['classification']})"
                )
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        else:
            await update.message.reply_text("📋 Watchlist not available")

    async def _cmd_pump(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /pump <symbol> command."""
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /pump BTCUSDT")
            return

        symbol = args[0].upper()
        if self._get_pump_detail:
            detail = await self._get_pump_detail(symbol)
            if detail:
                await update.message.reply_text(detail, parse_mode="Markdown")
            else:
                await update.message.reply_text(f"No data for {symbol}")
        else:
            await update.message.reply_text("Pump detail not available")

    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /config command — show config (masked secrets)."""
        cfg = self.settings
        msg = (
            "⚙️ *Configuration*\n\n"
            f"Scan interval: {cfg.general.scan_interval_seconds}s\n"
            f"Primary exchange: {cfg.exchanges.primary}\n"
            f"Secondary: {', '.join(cfg.exchanges.secondary)}\n"
            f"Min volume: ${cfg.filters.min_volume_24h_usd:,.0f}\n"
            f"Min OI: ${cfg.filters.min_open_interest_usd:,.0f}\n"
            f"Pump threshold: {cfg.detection.pump_threshold_pct}%\n"
            f"Alert min score: {cfg.telegram.alert_min_score}\n"
            f"ML enabled: {cfg.ml.enabled}"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_alerts(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /alerts on|off command."""
        args = context.args
        if not args:
            status = "🔔 ON" if self.settings.telegram.alert_on_pump else "🔕 OFF"
            await update.message.reply_text(f"Alerts are currently {status}")
            return

        if args[0].lower() == "on":
            self.settings.telegram.alert_on_pump = True
            await update.message.reply_text("🔔 Alerts turned ON")
        elif args[0].lower() == "off":
            self.settings.telegram.alert_on_pump = False
            await update.message.reply_text("🔕 Alerts turned OFF")

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        msg = (
            "🤖 *Pump Hunter Commands*\n\n"
            "/status — System status & stats\n"
            "/scan — Force immediate scan\n"
            "/watchlist — Current watchlist with scores\n"
            "/pump <SYM> — Detailed analysis for a coin\n"
            "/config — Show current configuration\n"
            "/alerts on|off — Toggle pump alerts\n"
            "/help — This message"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_bar(value: float, width: int = 10) -> str:
        """Create a text bar visualization."""
        filled = int(value / 100 * width)
        return "█" * filled + "░" * (width - filled)
