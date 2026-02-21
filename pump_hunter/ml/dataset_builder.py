"""
Dataset builder — creates training data from recorded pump events.
Generates positive (pre-pump) and negative (non-pump) samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from pump_hunter.ml.features import FEATURE_NAMES
from pump_hunter.storage.database import Database

logger = structlog.get_logger(__name__)


class DatasetBuilder:
    """Builds ML training datasets from pump event history."""

    def __init__(self, db: Database):
        self.db = db

    async def build(
        self,
        min_samples: int = 50,
        negative_ratio: float = 3.0,
    ) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Build training dataset.
        Returns (X, y) where X is features DataFrame and y is binary labels.
        """
        # get completed pump events with features
        pumps = await self.db.get_completed_pumps_for_training()
        logger.info("dataset_pump_events", count=len(pumps))

        if len(pumps) < min_samples:
            logger.warning(
                "insufficient_training_data",
                have=len(pumps),
                need=min_samples,
            )
            return None

        # build positive samples from pump features
        positive_rows = []
        for pump in pumps:
            if pump.features:
                row = {name: pump.features.get(name, 0) for name in FEATURE_NAMES}
                row["label"] = 1
                positive_rows.append(row)

        if not positive_rows:
            logger.warning("no_valid_pump_features")
            return None

        # build negative samples (non-pump periods)
        # use pre-pump signals with shifted timestamps as negative examples
        negative_count = int(len(positive_rows) * negative_ratio)
        negative_rows = self._generate_negatives(positive_rows, negative_count)

        # combine
        all_rows = positive_rows + negative_rows
        df = pd.DataFrame(all_rows)

        X = df[FEATURE_NAMES].fillna(0)
        y = df["label"]

        logger.info(
            "dataset_built",
            positive=len(positive_rows),
            negative=len(negative_rows),
            features=len(FEATURE_NAMES),
        )

        return X, y

    def _generate_negatives(
        self, positive_rows: List[dict], count: int
    ) -> List[dict]:
        """Generate negative samples by adding noise to positive features."""
        negatives = []
        rng = np.random.default_rng(42)

        for i in range(count):
            base = positive_rows[i % len(positive_rows)].copy()
            base["label"] = 0

            # randomize features to create non-pump conditions
            for name in FEATURE_NAMES:
                if name in base and isinstance(base[name], (int, float)):
                    # add noise and dampen signal values
                    noise = rng.normal(0, abs(base[name]) * 0.5 + 0.01)
                    base[name] = base[name] * rng.uniform(0.3, 0.8) + noise

            negatives.append(base)

        return negatives

    async def export_to_csv(self, output_path: str = "pump_dataset.csv") -> str:
        """Export dataset to CSV file."""
        result = await self.build(min_samples=1)
        if result is None:
            return ""

        X, y = result
        df = X.copy()
        df["label"] = y.values
        path = Path(output_path)
        df.to_csv(path, index=False)
        logger.info("dataset_exported", path=str(path), rows=len(df))
        return str(path)
