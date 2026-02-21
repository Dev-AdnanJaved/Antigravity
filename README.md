# 🔍 Pump Hunter v1.0

**Real-time Binance Futures pump detection & market intelligence system.**

Monitors 200+ futures pairs across 4 exchanges, scores them on 10 microstructure signals, and alerts via Telegram — with regime awareness, manipulation filtering, paper trading, and adaptive risk controls.

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start (5 minutes)](#quick-start)
- [Configuration Guide](#configuration-guide)
- [Running the Bot](#running-the-bot)
- [Architecture Overview](#architecture-overview)
- [Feature Reference](#feature-reference)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Runtime |
| **Docker** | 20+ | PostgreSQL & Redis |
| **Docker Compose** | v2+ | Container orchestration |
| **Binance Account** | — | API access (read-only is sufficient) |
| **Telegram Bot** | — | Alert delivery |

---

## Quick Start

### Step 1: Clone & Install Dependencies

```bash
# navigate to project directory
cd "Antigravity - Bot"

# create virtual environment
python -m venv venv

# activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

### Step 2: Start Database & Redis

```bash
docker-compose up -d
```

This starts:
- **PostgreSQL 16** on port `5432` (stores signal history, pump recordings, ML data)
- **Redis 7** on port `6379` (real-time state cache, leaderboards, pub/sub)

Verify they're running:
```bash
docker-compose ps
```
Both should show `healthy` status.

### Step 3: Configure API Keys & Telegram

Open `config.json` and fill in:

```jsonc
{
  "api_keys": {
    "binance": {
      "key": "YOUR_BINANCE_API_KEY",       // ← required
      "secret": "YOUR_BINANCE_SECRET"       // ← required
    },
    "bybit": {
      "key": "",    // optional — for cross-exchange signals
      "secret": ""
    },
    "okx": {
      "key": "",    // optional
      "secret": "",
      "passphrase": ""
    }
  },
  "telegram": {
    "enabled": true,
    "bot_token": "YOUR_BOT_TOKEN",   // ← from @BotFather
    "chat_id": "YOUR_CHAT_ID"        // ← your Telegram user/group ID
  }
}
```

> **How to get Telegram credentials:**
> 1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token
> 2. Message [@userinfobot](https://t.me/userinfobot) → copy your chat ID

### Step 4: Run the Bot

```bash
python -m pump_hunter.main
```

You should see:
```
╔═══════════════════════════════════════════╗
║    🔍 PUMP HUNTER v1.0                    ║
║    Market Intelligence System              ║
║    Binance Futures Pump Detection           ║
╚═══════════════════════════════════════════╝
```

The bot will warm up for ~10 seconds, then begin scanning every 60 seconds.

---

## Configuration Guide

All configuration lives in **`config.json`**. Below is every section explained.

### General Settings

```json
"general": {
  "scan_interval_seconds": 60,      // how often to scan all symbols
  "bootstrap_on_start": true,       // load historical candles on startup
  "bootstrap_candles": 500,         // candles to fetch per symbol at start
  "log_level": "INFO",              // DEBUG, INFO, WARNING, ERROR
  "data_retention_days": 30         // auto-prune old DB records
}
```

### Exchange Configuration

```json
"exchanges": {
  "primary": "binance",            // main data source
  "secondary": ["bybit", "okx", "bitget"],  // cross-exchange volume signals
  "use_testnet": false             // set true for paper testing on testnet
}
```

> **Tip:** Secondary exchanges are optional. The bot works with Binance alone, but cross-exchange signals improve accuracy.

### Database & Redis

```json
"database": {
  "host": "localhost",
  "port": 5432,
  "name": "pump_hunter",
  "user": "pump_hunter",
  "password": "pump_hunter_pass"   // must match docker-compose.yml
},
"redis": {
  "host": "localhost",
  "port": 6379,
  "db": 0,
  "password": ""                   // leave empty for default Redis
}
```

These match the `docker-compose.yml` defaults. Only change if you use external/cloud databases.

### Symbol Filters

```json
"filters": {
  "min_volume_24h_usd": 500000,      // skip low-volume coins
  "min_open_interest_usd": 100000,   // skip low-OI coins
  "min_market_cap_usd": 0,           // 0 = no filter
  "min_orderbook_depth_usd": 10000,  // thin books get filtered
  "excluded_symbols": ["USDCUSDT"],  // blacklisted symbols
  "only_symbols": []                 // empty = scan all (or specify ["BTCUSDT", "ETHUSDT"])
}
```

### Telegram Alerts

```json
"telegram": {
  "enabled": true,
  "bot_token": "...",
  "chat_id": "...",
  "alert_on_pump": true,           // send pump detection alerts
  "alert_on_prediction": true,     // send ML prediction alerts
  "alert_min_score": 60,           // minimum composite score to alert
  "rate_limit_seconds": 900        // 15min cooldown per symbol
}
```

### Detection Engine

```json
"detection": {
  "pump_threshold_pct": 10.0,       // static fallback threshold
  "pump_max_pct": 60.0,             // ignore moves > 60% (likely data error)
  "pump_threshold_mode": "atr_adaptive",  // "static" or "atr_adaptive"
  "atr_adaptive_high_percentile": 80.0,   // high-vol ATR cutoff
  "atr_adaptive_low_percentile": 20.0,    // low-vol ATR cutoff
  "atr_adaptive_high_mult": 2.0,          // threshold = 2× ATR in high vol
  "atr_adaptive_low_mult": 0.5            // threshold = 0.5× ATR in low vol
}
```

**ATR-Adaptive mode** (recommended): Automatically adjusts pump thresholds based on current volatility. In calm markets, it catches smaller moves early. In volatile markets, it filters out noise.

### Execution & Paper Trading

```json
"execution": {
  "enabled": true,
  "paper_trading": true,              // ⚠️ START WITH THIS ON
  "default_trade_size_usd": 1000,
  "max_slippage_pct": 0.5,
  "kelly_fraction": 0.25,
  "kill_switch_consecutive_losses": 3,
  "kill_switch_drawdown_pct": 5.0,
  "kill_switch_cooldown_minutes": 60,
  "entry_mode": "adaptive",          // market, limit, pullback, adaptive

  // Tail Risk Kill Switch
  "tail_risk_btc_crash_pct": 3.0,              // BTC drops 3% in 5min → kill all
  "tail_risk_btc_crash_window_min": 5,
  "tail_risk_mark_dislocation_pct": 2.0,       // mark vs last price gap > 2% → kill symbol
  "tail_risk_api_error_threshold": 0.5,        // > 50% API errors → kill all
  "tail_risk_api_error_window": 100,
  "tail_risk_funding_spike_pct": 0.3,          // |funding| > 0.3% → reduce size 50%
  "tail_risk_liq_cascade_count": 3,            // 3+ same-direction liquidations → kill
  "tail_risk_liq_cascade_window_sec": 60
}
```

> ⚠️ **IMPORTANT:** Always start with `"paper_trading": true`. This simulates trades against live data without placing real orders. Review the paper journal at `data/paper_journal.jsonl` before going live.

### Regime Detection

```json
"regime": {
  "enabled": true,
  "btc_symbol": "BTCUSDT",
  "risk_on_multiplier": 1.2,       // boost scores in bullish regimes
  "risk_off_multiplier": 0.6,      // suppress in bearish regimes
  "neutral_multiplier": 1.0,
  "volatility_expansion_multiplier": 0.8,
  "low_liquidity_multiplier": 0.7
}
```

### Manipulation Filter

```json
"manipulation": {
  "enabled": true,
  "spoof_cancel_rate_threshold": 0.8,   // >80% cancel rate = spoofing
  "fake_breakout_revert_pct": 80.0,     // breakout retraces 80% = fake
  "thin_book_threshold_usd": 5000,      // books < $5k = dangerous
  "coordinated_min_coins": 3            // 3+ coins moving together = coordinated
}
```

### Feature Drift Monitoring

```json
"drift": {
  "enabled": true,
  "update_interval_minutes": 60,          // check drift hourly
  "psi_moderate_threshold": 0.1,          // PSI > 0.1 = warning
  "psi_significant_threshold": 0.25,      // PSI > 0.25 = retrain needed
  "psi_bins": 10,
  "auc_decay_threshold": 0.05,            // AUC drops > 5% = model degrading
  "regime_stratified": true               // separate baselines per regime
}
```

### ML Predictor (Optional)

```json
"ml": {
  "enabled": false,            // enable after collecting training data
  "model_type": "xgboost",
  "min_training_samples": 50,
  "retrain_interval_hours": 24,
  "prediction_threshold": 0.7,
  "model_path": "models/"
}
```

> ML requires labeled pump data collected over time. Leave disabled initially and enable after the system has recorded enough pump events.

---

## Running the Bot

### Standard Run
```bash
python -m pump_hunter.main
```

### Run in Background (Linux/Server)
```bash
nohup python -m pump_hunter.main > logs/pump_hunter.log 2>&1 &
```

### Run with Docker (Full Stack)
If you prefer everything containerized, you can run:
```bash
# start databases
docker-compose up -d

# run bot in a screen/tmux session
screen -S pump_hunter
python -m pump_hunter.main
# Ctrl+A, D to detach
```

### Monitoring

- **Logs:** Structured JSON logs via `structlog` (stdout)
- **Paper Journal:** `data/paper_journal.jsonl` — every simulated trade
- **Telegram:** Real-time alerts for CRITICAL and HIGH_ALERT scores

---

## Architecture Overview

```
pump_hunter/
├── main.py                     # Orchestrator — scan loop, component wiring
├── config/
│   └── settings.py             # Pydantic config models
├── collectors/
│   ├── binance_ws.py           # WebSocket streams (trades, OI, liq)
│   ├── binance_rest.py         # REST API (funding, OI history)
│   ├── multi_exchange.py       # Cross-exchange data (Bybit, OKX, Bitget)
│   └── symbol_manager.py       # Active symbol discovery & filtering
├── core/
│   ├── market_state.py         # Real-time state per symbol
│   ├── aggregator.py           # Multi-source data fusion
│   ├── regime_engine.py        # Market regime classification (5 states)
│   ├── execution_simulator.py  # Trade simulation + kill switch
│   ├── capital_allocator.py    # Portfolio-level position management
│   └── paper_trader.py         # Paper trading engine
├── detection/
│   ├── feature_engine.py       # 10 microstructure signals (0-100)
│   ├── pump_detector.py        # Scoring, classification, ATR-adaptive
│   ├── pump_recorder.py        # Record pump lifecycle for ML training
│   ├── manipulation_filter.py  # 6 adversarial pattern detectors
│   └── survival_analysis.py    # Historical pump statistics
├── ml/
│   ├── predictor.py            # XGBoost/LightGBM pump predictor
│   ├── feature_drift.py        # PSI-based drift monitoring
│   └── walk_forward.py         # Rolling-window backtesting
├── storage/
│   ├── database.py             # PostgreSQL async (asyncpg)
│   ├── redis_store.py          # Redis state cache
│   └── timeseries.py           # In-memory candle store
├── alerts/
│   └── notifier.py             # Telegram bot integration
└── train.py                    # ML model training script
```

### Signal Flow

```
Exchange WebSocket → Market State → Feature Engine (10 signals)
    → Regime Modifier → Manipulation Filter → Score & Classify
    → Capital Allocator → Execution Simulator → Paper Trade or Alert
    → Telegram Notification
```

---

## Feature Reference

### 10 Microstructure Signals

| Signal | Weight | What it measures |
|--------|--------|-----------------|
| OI Surge | 16% | Open interest spike vs 24h baseline |
| Funding Rate | 15% | Extreme funding = crowded positioning |
| Liquidation Leverage | 13% | Liquidation cascade potential |
| Cross-Exchange Volume | 12% | Unusual volume vs other exchanges |
| Depth Imbalance | 11% | Bid/ask orderbook asymmetry |
| Volume-Price Decouple | 9% | Volume spike without price follow |
| Volatility Compression | 8% | Bollinger squeeze = breakout incoming |
| Long/Short Ratio | 6% | Retail sentiment skew |
| Futures-Spot Divergence | 5% | Basis widening/narrowing |
| Whale Activity | 5% | Large order detection |

### Classification Thresholds

| Class | Score | Action |
|-------|-------|--------|
| **CRITICAL** | ≥ 78 | Immediate alert, market entry |
| **HIGH_ALERT** | ≥ 62 | Alert, slight pullback entry |
| **WATCHLIST** | ≥ 48 | Monitor closely |
| **MONITOR** | ≥ 33 | Log only |
| **LOW** | < 33 | Ignore |

---

## Troubleshooting

### Database Connection Error
```
Error: could not connect to server: Connection refused
```
**Fix:** Ensure Docker containers are running:
```bash
docker-compose up -d
docker-compose ps   # both should say "healthy"
```

### Telegram Not Sending
1. Verify `bot_token` and `chat_id` are correct in `config.json`
2. Send `/start` to your bot first in Telegram
3. Check `rate_limit_seconds` — alerts have a 15min per-symbol cooldown

### Binance API Errors
```
Error: fapiPublicGetOpenInterestHist not found
```
**Fix:** This is a CCXT v4+ compatibility issue. The code already handles this with the updated endpoint. Run:
```bash
pip install --upgrade ccxt
```

### High Memory Usage
Reduce the number of tracked symbols:
```json
"filters": {
  "min_volume_24h_usd": 2000000    // raise to track fewer symbols
}
```
Or reduce candle history:
```json
"general": {
  "bootstrap_candles": 200         // default 500, reduce if memory tight
}
```

### Kill Switch Activated
If you see `kill_switch_activated` in logs, the bot has paused trading due to:
- 3 consecutive losses, or
- 5% drawdown, or
- A tail risk event (BTC crash, API errors, etc.)

It auto-resets after `kill_switch_cooldown_minutes` (default: 60 min). You can also restart the bot to clear it.

### Paper Trading Journal
Review simulated trades:
```bash
# Windows
type data\paper_journal.jsonl

# Linux
cat data/paper_journal.jsonl | python -m json.tool
```

---

## Going Live Checklist

Before switching from paper to live trading:

- [ ] Run paper trading for **at least 7 days**
- [ ] Review `data/paper_journal.jsonl` — check win rate, Sharpe, drawdown
- [ ] Verify gap metrics (expected vs actual fills are realistic)
- [ ] Confirm no feature drift warnings in logs
- [ ] Test kill switch triggers manually
- [ ] Set `"paper_trading": false` in `config.json`
- [ ] Start with **small position sizes** (`"default_trade_size_usd": 100`)
- [ ] Monitor Telegram alerts closely for the first 48 hours

---

## License

This project is proprietary. All rights reserved.
