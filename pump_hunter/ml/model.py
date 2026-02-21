"""
ML model wrapper — loads saved models, provides consistent predict interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import structlog

from pump_hunter.ml.features import FEATURE_NAMES

logger = structlog.get_logger(__name__)


class PumpModel:
    """Wrapper for trained pump prediction models."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._model = None
        self._version = ""
        self._metadata: dict = {}

    def load(self) -> bool:
        """Load the model from disk."""
        try:
            self._model = joblib.load(self.model_path)
            self._version = self.model_path.stem.replace("pump_model_", "")

            # load metadata if available
            meta_path = self.model_path.with_suffix(".json")
            if meta_path.exists():
                self._metadata = json.loads(meta_path.read_text())

            logger.info("model_loaded", path=str(self.model_path), version=self._version)
            return True
        except Exception as e:
            logger.error("model_load_error", path=str(self.model_path), error=str(e))
            return False

    @property
    def version(self) -> str:
        return self._version

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        Predict pump probability from feature dict.
        Returns probability between 0 and 1.
        """
        if not self.is_loaded:
            return 0.0

        # ensure feature ordering matches training
        feature_values = [features.get(name, 0) for name in FEATURE_NAMES]
        X = pd.DataFrame([feature_values], columns=FEATURE_NAMES)

        try:
            proba = self._model.predict_proba(X)[0, 1]
            return float(proba)
        except Exception as e:
            logger.error("predict_error", error=str(e))
            return 0.0

    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        """Predict probabilities for multiple symbols."""
        if not self.is_loaded or not features_list:
            return [0.0] * len(features_list)

        rows = []
        for features in features_list:
            rows.append([features.get(name, 0) for name in FEATURE_NAMES])

        X = pd.DataFrame(rows, columns=FEATURE_NAMES)

        try:
            probas = self._model.predict_proba(X)[:, 1]
            return probas.tolist()
        except Exception as e:
            logger.error("batch_predict_error", error=str(e))
            return [0.0] * len(features_list)

    @staticmethod
    def find_latest_model(model_dir: str = "models/") -> Optional[Path]:
        """Find the most recent model file."""
        model_path = Path(model_dir)
        if not model_path.exists():
            return None

        models = sorted(model_path.glob("pump_model_*.joblib"), reverse=True)
        return models[0] if models else None
