"""
Configuration management — loads config.json with Pydantic validation.
Auto-creates config.json with defaults if missing.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Resolve config path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = _PROJECT_ROOT / "config.json"


# ---------------------------------------------------------------------------
# Pydantic sub-models
# ---------------------------------------------------------------------------

class GeneralConfig(BaseModel):
    scan_interval_seconds: int = Field(60, ge=10, le=3600)
    bootstrap_on_start: bool = True
    bootstrap_candles: int = Field(500, ge=50, le=2000)
    bootstrap_oi_points: int = Field(200, ge=20, le=1000)
    log_level: str = "INFO"
    timezone: str = "UTC"
    data_retention_days: int = Field(30, ge=1, le=365)


class ExchangeConfig(BaseModel):
    primary: str = "binance"
    secondary: List[str] = Field(default_factory=lambda: ["bybit", "okx", "bitget"])
    use_testnet: bool = False


class APIKeyEntry(BaseModel):
    key: str = ""
    secret: str = ""
    passphrase: Optional[str] = None


class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = Field(5432, ge=1, le=65535)
    name: str = "pump_hunter"
    user: str = "pump_hunter"
    password: str = "pump_hunter_pass"
    pool_min: int = Field(5, ge=1, le=50)
    pool_max: int = Field(20, ge=1, le=100)

    @property
    def dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def sync_dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = Field(6379, ge=1, le=65535)
    db: int = Field(0, ge=0, le=15)
    password: str = ""

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class TelegramConfig(BaseModel):
    enabled: bool = True
    bot_token: str = ""
    chat_id: str = ""
    alert_on_pump: bool = True
    alert_on_prediction: bool = True
    alert_min_score: float = Field(60.0, ge=0, le=100)
    commands_enabled: bool = True
    rate_limit_seconds: int = Field(900, ge=60, le=7200)


class FiltersConfig(BaseModel):
    min_volume_24h_usd: float = Field(500_000, ge=0)
    min_open_interest_usd: float = Field(100_000, ge=0)
    min_market_cap_usd: float = Field(0, ge=0)
    min_orderbook_depth_usd: float = Field(10_000, ge=0)
    excluded_symbols: List[str] = Field(default_factory=lambda: ["USDCUSDT"])
    only_symbols: List[str] = Field(default_factory=list)


class PriceExtendedPenalty(BaseModel):
    threshold_pct: float = Field(15.0, ge=1, le=100)
    lookback_days: int = Field(7, ge=1, le=30)
    penalty_pct: float = Field(40.0, ge=0, le=100)


class ClassificationConfig(BaseModel):
    critical: float = Field(78, ge=0, le=100)
    high_alert: float = Field(62, ge=0, le=100)
    watchlist: float = Field(48, ge=0, le=100)
    monitor: float = Field(33, ge=0, le=100)


class DetectionConfig(BaseModel):
    pump_threshold_pct: float = Field(10.0, ge=1, le=100)
    pump_max_pct: float = Field(60.0, ge=10, le=500)
    pump_window_minutes: int = Field(60, ge=5, le=1440)
    pre_pump_capture_minutes: int = Field(60, ge=5, le=720)
    post_pump_capture_minutes: int = Field(120, ge=5, le=1440)
    pump_threshold_mode: str = "atr_adaptive"  # "static" or "atr_adaptive"
    atr_adaptive_high_percentile: float = Field(80.0, ge=50, le=99)
    atr_adaptive_low_percentile: float = Field(20.0, ge=1, le=50)
    atr_adaptive_high_mult: float = Field(2.0, ge=1.0, le=5.0)
    atr_adaptive_low_mult: float = Field(0.5, ge=0.1, le=1.0)
    score_weights: Dict[str, float] = Field(default_factory=lambda: {
        "oi_surge": 0.16,
        "funding_rate": 0.15,
        "liquidation_leverage": 0.13,
        "cross_exchange_volume": 0.12,
        "depth_imbalance": 0.11,
        "volume_price_decouple": 0.09,
        "volatility_compression": 0.08,
        "long_short_ratio": 0.06,
        "futures_spot_divergence": 0.05,
        "whale_activity": 0.05,
    })
    interaction_bonuses: Dict[str, float] = Field(default_factory=lambda: {
        "squeeze_setup": 0.25,
        "cascade_setup": 0.30,
        "accumulation_setup": 0.20,
    })
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    price_extended_penalty: PriceExtendedPenalty = Field(
        default_factory=PriceExtendedPenalty
    )

    @field_validator("score_weights")
    @classmethod
    def weights_must_sum_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"score_weights must sum to 1.0, got {total:.4f}")
        return v


class LevelsConfig(BaseModel):
    stop_loss_min_pct: float = Field(2.5, ge=0.5, le=50)
    stop_loss_max_pct: float = Field(15.0, ge=1, le=50)
    atr_period: int = Field(14, ge=5, le=50)
    tp_multipliers: List[float] = Field(default_factory=lambda: [3.0, 5.5, 9.0])
    trail_schedule: Dict[str, int] = Field(default_factory=lambda: {
        "5": 0, "10": 5, "15": 10, "25": 18, "40": 30, "60": 45,
    })


class MLConfig(BaseModel):
    enabled: bool = False
    model_type: str = "xgboost"
    min_training_samples: int = Field(50, ge=10, le=10000)
    retrain_interval_hours: int = Field(24, ge=1, le=720)
    prediction_threshold: float = Field(0.7, ge=0.1, le=0.99)
    feature_window_minutes: int = Field(60, ge=5, le=1440)
    model_path: str = "models/"


class WebSocketConfig(BaseModel):
    max_streams_per_connection: int = Field(200, ge=10, le=1024)
    reconnect_delay_seconds: int = Field(5, ge=1, le=120)
    ping_interval_seconds: int = Field(20, ge=5, le=120)
    stale_timeout_seconds: int = Field(60, ge=10, le=600)


class RegimeConfig(BaseModel):
    enabled: bool = True
    btc_symbol: str = "BTCUSDT"
    risk_on_multiplier: float = Field(1.2, ge=0.5, le=2.0)
    risk_off_multiplier: float = Field(0.6, ge=0.1, le=1.0)
    neutral_multiplier: float = Field(1.0, ge=0.5, le=1.5)
    volatility_expansion_multiplier: float = Field(0.8, ge=0.1, le=1.5)
    low_liquidity_multiplier: float = Field(0.7, ge=0.1, le=1.0)
    btc_trend_lookback_hours: int = Field(4, ge=1, le=24)
    btc_volatility_lookback: int = Field(20, ge=5, le=100)
    regime_update_interval_seconds: int = Field(300, ge=60, le=3600)


class ManipulationConfig(BaseModel):
    enabled: bool = True
    spoof_cancel_rate_threshold: float = Field(0.8, ge=0.5, le=1.0)
    spoof_check_window_seconds: int = Field(5, ge=2, le=30)
    fake_breakout_revert_candles: int = Field(3, ge=1, le=10)
    fake_breakout_revert_pct: float = Field(80.0, ge=50, le=100)
    pump_dump_oi_threshold: float = Field(0.5, ge=0.1, le=1.0)
    thin_book_threshold_usd: float = Field(5000, ge=1000, le=50000)
    coordinated_window_minutes: int = Field(5, ge=1, le=30)
    coordinated_min_coins: int = Field(3, ge=2, le=10)
    reversal_rsi_threshold: float = Field(80.0, ge=60, le=95)
    reversal_volume_exhaustion_pct: float = Field(50.0, ge=20, le=90)


class ExecutionConfig(BaseModel):
    enabled: bool = True
    paper_trading: bool = True  # paper mode by default; disable for live
    default_trade_size_usd: float = Field(1000, ge=100, le=100000)
    max_slippage_pct: float = Field(0.5, ge=0.01, le=5.0)
    latency_ms: int = Field(200, ge=0, le=5000)
    kelly_fraction: float = Field(0.25, ge=0.05, le=1.0)
    kill_switch_consecutive_losses: int = Field(3, ge=1, le=20)
    kill_switch_drawdown_pct: float = Field(5.0, ge=1.0, le=50.0)
    kill_switch_cooldown_minutes: int = Field(60, ge=5, le=1440)
    entry_mode: str = "adaptive"  # market, limit, pullback, adaptive
    pullback_wait_pct: float = Field(0.5, ge=0.1, le=3.0)
    pullback_timeout_seconds: int = Field(30, ge=5, le=300)
    # tail risk kill switch
    tail_risk_btc_crash_pct: float = Field(3.0, ge=1.0, le=20.0)
    tail_risk_btc_crash_window_min: int = Field(5, ge=1, le=60)
    tail_risk_mark_dislocation_pct: float = Field(2.0, ge=0.5, le=10.0)
    tail_risk_api_error_threshold: float = Field(0.5, ge=0.1, le=1.0)
    tail_risk_api_error_window: int = Field(100, ge=10, le=1000)
    tail_risk_funding_spike_pct: float = Field(0.3, ge=0.05, le=2.0)
    tail_risk_liq_cascade_count: int = Field(3, ge=1, le=20)
    tail_risk_liq_cascade_window_sec: int = Field(60, ge=10, le=600)


class SurvivalConfig(BaseModel):
    min_events_for_analysis: int = Field(20, ge=5, le=1000)
    retrace_thresholds: List[float] = Field(default_factory=lambda: [25.0, 50.0, 75.0])
    optimal_entry_delay_minutes: float = Field(2.0, ge=0, le=30)
    peak_time_bins_minutes: List[float] = Field(
        default_factory=lambda: [5, 15, 30, 60, 120, 240]
    )
    update_interval_hours: int = Field(6, ge=1, le=168)


class AllocationConfig(BaseModel):
    max_concurrent_positions: int = Field(3, ge=1, le=20)
    max_position_pct_of_volume: float = Field(1.0, ge=0.01, le=10.0)
    max_single_position_pct: float = Field(33.0, ge=5, le=100)
    min_score_for_allocation: float = Field(60.0, ge=0, le=100)
    volatility_normalize: bool = True
    correlation_sector_limit: int = Field(2, ge=1, le=10)
    drawdown_scale_factor: float = Field(0.5, ge=0.1, le=1.0)


class DriftConfig(BaseModel):
    enabled: bool = True
    update_interval_minutes: int = Field(60, ge=10, le=1440)
    psi_moderate_threshold: float = Field(0.1, ge=0.01, le=1.0)
    psi_significant_threshold: float = Field(0.25, ge=0.05, le=2.0)
    psi_bins: int = Field(10, ge=5, le=50)
    auc_decay_threshold: float = Field(0.05, ge=0.01, le=0.5)
    baseline_lookback_days: int = Field(30, ge=7, le=180)
    regime_stratified: bool = True


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    exchanges: ExchangeConfig = Field(default_factory=ExchangeConfig)
    api_keys: Dict[str, APIKeyEntry] = Field(default_factory=dict)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    levels: LevelsConfig = Field(default_factory=LevelsConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    manipulation: ManipulationConfig = Field(default_factory=ManipulationConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    survival: SurvivalConfig = Field(default_factory=SurvivalConfig)
    allocation: AllocationConfig = Field(default_factory=AllocationConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)

    # ----- helpers ---------------------------------------------------------

    @classmethod
    def _default_dict(cls) -> Dict[str, Any]:
        """Return a clean default config dict."""
        return cls().model_dump()

    @classmethod
    def ensure_config_file(cls, path: Path | None = None) -> Path:
        """Create config.json with defaults if it doesn't exist."""
        p = path or CONFIG_PATH
        if not p.exists():
            logger.warning("config_missing", path=str(p))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(cls._default_dict(), indent=2),
                encoding="utf-8",
            )
            logger.info("config_created", path=str(p))
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        """Load and validate config.json."""
        p = cls.ensure_config_file(path)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("config_parse_error", path=str(p), error=str(exc))
            sys.exit(1)

        try:
            settings = cls.model_validate(raw)
        except Exception as exc:
            logger.error("config_validation_error", error=str(exc))
            sys.exit(1)

        logger.info(
            "config_loaded",
            path=str(p),
            scan_interval=settings.general.scan_interval_seconds,
            primary_exchange=settings.exchanges.primary,
            telegram_enabled=settings.telegram.enabled,
        )
        return settings

    def save(self, path: Path | None = None) -> None:
        """Persist current settings back to config.json."""
        p = path or CONFIG_PATH
        p.write_text(
            json.dumps(self.model_dump(), indent=2),
            encoding="utf-8",
        )
        logger.info("config_saved", path=str(p))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_settings: Settings | None = None


def get_settings(path: Path | None = None) -> Settings:
    """Get or create the global Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings.load(path)
    return _settings


def reload_settings(path: Path | None = None) -> Settings:
    """Force reload of the config file."""
    global _settings
    _settings = Settings.load(path)
    return _settings
