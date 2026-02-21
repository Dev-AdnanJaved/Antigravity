"""
ML predictor — runs trained model on live market state.
Outputs pump probability per symbol, triggers alerts above threshold.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Dict, List, Optional

import structlog

from pump_hunter.config.settings import Settings
from pump_hunter.core.market_state import MarketStateManager
from pump_hunter.ml.features import MLFeatureExtractor
from pump_hunter.ml.model import PumpModel
from pump_hunter.storage.database import Database
from pump_hunter.storage.timeseries import TimeseriesStore

logger = structlog.get_logger(__name__)


class Predictor:
    """Runs trained ML model on live market state for pump prediction."""

    def __init__(
        self,
        settings: Settings,
        db: Database,
        market_state: MarketStateManager,
        timeseries: TimeseriesStore,
    ):
        self.settings = settings
        self.db = db
        self.market_state = market_state
        self.ts = timeseries
        self.feature_extractor = MLFeatureExtractor(
            timeseries, settings.ml.feature_window_minutes
        )
        self._model: Optional[PumpModel] = None
        self._threshold = settings.ml.prediction_threshold

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """Load the latest trained model."""
        model_path = PumpModel.find_latest_model(self.settings.ml.model_path)
        if model_path is None:
            logger.warning("no_model_found", model_dir=self.settings.ml.model_path)
            return False

        self._model = PumpModel(model_path)
        return self._model.load()

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._model.is_loaded

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------

    async def predict_all(self, symbols: List[str]) -> List[dict]:
        """
        Run predictions on all symbols.
        Returns list of predictions above threshold, sorted by probability desc.
        """
        if not self.is_ready:
            return []

        predictions = []

        for symbol in symbols:
            state = self.market_state.get_state(symbol)
            if state.price <= 0:
                continue

            features = self.feature_extractor.extract(symbol, state.to_dict())
            if not features:
                continue

            probability = self._model.predict_proba(features)

            if probability >= self._threshold:
                pred = {
                    "symbol": symbol,
                    "probability": probability,
                    "model_version": self._model.version,
                    "features": features,
                    "timestamp": dt.datetime.utcnow(),
                }
                predictions.append(pred)

                # store in DB
                db_symbols = await self.db.get_active_symbols()
                symbol_id = next(
                    (s.id for s in db_symbols if s.symbol == symbol), None
                )
                if symbol_id:
                    await self.db.insert_prediction({
                        "symbol_id": symbol_id,
                        "model_version": self._model.version,
                        "probability": probability,
                        "features_used": features,
                        "outcome": "pending",
                    })

                logger.info(
                    "prediction_alert",
                    symbol=symbol,
                    probability=f"{probability:.2%}",
                    model=self._model.version,
                )

        # sort by probability descending
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        return predictions

    async def predict_single(self, symbol: str) -> Optional[dict]:
        """Run prediction on a single symbol."""
        if not self.is_ready:
            return None

        state = self.market_state.get_state(symbol)
        features = self.feature_extractor.extract(symbol, state.to_dict())
        if not features:
            return None

        probability = self._model.predict_proba(features)
        return {
            "symbol": symbol,
            "probability": probability,
            "model_version": self._model.version,
        }
