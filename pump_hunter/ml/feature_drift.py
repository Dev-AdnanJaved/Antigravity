"""
Feature drift monitor — detects distribution shift using Population Stability Index.

Tracks each feature for:
    - PSI vs training baseline (per-feature monthly)
    - Predictive decay via rolling AUC
    - Regime-stratified baselines (separate PSI per regime)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from pump_hunter.config.settings import Settings

logger = structlog.get_logger(__name__)


class FeatureDriftMonitor:
    """
    Monitors feature distributions for drift using PSI.

    PSI thresholds:
        < 0.10  →  stable (no action)
        0.10–0.25  →  moderate drift (warning)
        > 0.25  →  significant drift (alert, consider retraining)
    """

    def __init__(self, settings: Settings):
        self.cfg = settings.drift
        self.n_bins = self.cfg.psi_bins

        # baselines: feature_name → bin_proportions (array of length n_bins)
        self._baselines: Dict[str, np.ndarray] = {}
        # regime-stratified baselines: (feature_name, regime) → bin_proportions
        self._regime_baselines: Dict[Tuple[str, str], np.ndarray] = {}
        # bin edges per feature: feature_name → edges array
        self._bin_edges: Dict[str, np.ndarray] = {}

        # rolling feature values for live PSI: feature_name → list of recent values
        self._live_buffer: Dict[str, List[float]] = {}
        self._buffer_max = 2000  # keep last 2000 values

        # rolling prediction tracking for AUC decay
        self._predictions: List[Tuple[float, int]] = []  # (prob, actual_label)

        # results cache
        self._last_psi: Dict[str, float] = {}
        self._last_update = 0.0
        self._drift_alerts: List[dict] = []

    # ------------------------------------------------------------------
    # baseline management
    # ------------------------------------------------------------------

    def set_baseline(
        self,
        feature_name: str,
        values: np.ndarray,
        regime: Optional[str] = None,
    ) -> None:
        """
        Build baseline histogram from training data.
        Call once per feature after model training.
        """
        values = values[~np.isnan(values)]
        if len(values) < 50:
            logger.warning("baseline_too_small", feature=feature_name, n=len(values))
            return

        # compute bin edges from training data
        edges = np.histogram_bin_edges(values, bins=self.n_bins)
        proportions = self._compute_proportions(values, edges)

        if regime:
            self._regime_baselines[(feature_name, regime)] = proportions
        else:
            self._baselines[feature_name] = proportions
            self._bin_edges[feature_name] = edges

        logger.info(
            "baseline_set",
            feature=feature_name,
            regime=regime,
            n_values=len(values),
            bins=self.n_bins,
        )

    def set_baselines_from_dataframe(self, df, regime_col: Optional[str] = None) -> None:
        """
        Build baselines from a pandas DataFrame of training features.
        Each numeric column becomes a feature baseline.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            values = df[col].dropna().values
            self.set_baseline(col, values)

            # regime-stratified baselines
            if regime_col and regime_col in df.columns and self.cfg.regime_stratified:
                for regime in df[regime_col].unique():
                    mask = df[regime_col] == regime
                    regime_values = df.loc[mask, col].dropna().values
                    if len(regime_values) >= 30:
                        self.set_baseline(col, regime_values, regime=str(regime))

    # ------------------------------------------------------------------
    # live tracking
    # ------------------------------------------------------------------

    def record_features(self, signals: Dict[str, Optional[float]]) -> None:
        """Record live feature values for drift computation."""
        for name, value in signals.items():
            if value is None:
                continue
            if name not in self._live_buffer:
                self._live_buffer[name] = []
            buf = self._live_buffer[name]
            buf.append(float(value))
            # trim buffer
            if len(buf) > self._buffer_max:
                self._live_buffer[name] = buf[-self._buffer_max:]

    def record_prediction(self, probability: float, actual_label: int) -> None:
        """Record a prediction + outcome for AUC decay tracking."""
        self._predictions.append((probability, actual_label))
        if len(self._predictions) > 5000:
            self._predictions = self._predictions[-5000:]

    # ------------------------------------------------------------------
    # PSI computation
    # ------------------------------------------------------------------

    def compute_psi_all(self, current_regime: Optional[str] = None) -> Dict[str, dict]:
        """
        Compute PSI for all tracked features.

        Returns dict of feature_name → {
            psi: float,
            status: "stable" | "moderate_drift" | "significant_drift",
            n_samples: int,
        }
        """
        now = time.time()
        if now - self._last_update < self.cfg.update_interval_minutes * 60:
            return {k: {"psi": v, "status": self._classify_psi(v)} for k, v in self._last_psi.items()}

        results = {}
        self._drift_alerts.clear()

        for feature_name, baseline in self._baselines.items():
            live_values = self._live_buffer.get(feature_name, [])
            if len(live_values) < 100:
                continue

            edges = self._bin_edges.get(feature_name)
            if edges is None:
                continue

            live_arr = np.array(live_values)
            live_proportions = self._compute_proportions(live_arr, edges)

            psi = self._psi(baseline, live_proportions)
            status = self._classify_psi(psi)

            # regime-stratified PSI (if available and current regime known)
            regime_psi = None
            if current_regime and self.cfg.regime_stratified:
                regime_baseline = self._regime_baselines.get((feature_name, current_regime))
                if regime_baseline is not None:
                    regime_psi = self._psi(regime_baseline, live_proportions)

            self._last_psi[feature_name] = psi

            result = {
                "psi": round(psi, 4),
                "status": status,
                "n_samples": len(live_values),
            }
            if regime_psi is not None:
                result["regime_psi"] = round(regime_psi, 4)
                result["regime_status"] = self._classify_psi(regime_psi)

            results[feature_name] = result

            # generate alerts for significant drift
            if status == "significant_drift":
                self._drift_alerts.append({
                    "feature": feature_name,
                    "psi": round(psi, 4),
                    "status": status,
                    "regime": current_regime,
                    "recommendation": "retrain_model",
                })

        self._last_update = now
        return results

    # ------------------------------------------------------------------
    # AUC decay
    # ------------------------------------------------------------------

    def compute_auc_decay(self) -> Optional[dict]:
        """
        Compute rolling AUC from recent predictions.
        Compare recent window to older window to detect predictive decay.
        """
        if len(self._predictions) < 200:
            return None

        preds = self._predictions
        mid = len(preds) // 2
        older = preds[:mid]
        recent = preds[mid:]

        auc_older = self._approximate_auc(older)
        auc_recent = self._approximate_auc(recent)

        if auc_older is None or auc_recent is None:
            return None

        decay = auc_older - auc_recent

        return {
            "auc_older": round(auc_older, 4),
            "auc_recent": round(auc_recent, 4),
            "decay": round(decay, 4),
            "decaying": decay > self.cfg.auc_decay_threshold,
            "n_older": len(older),
            "n_recent": len(recent),
        }

    # ------------------------------------------------------------------
    # summary / alerts
    # ------------------------------------------------------------------

    def get_drift_summary(self) -> dict:
        """Return current drift status summary."""
        drifting = [f for f, psi in self._last_psi.items()
                    if psi >= self.cfg.psi_moderate_threshold]
        critical = [f for f, psi in self._last_psi.items()
                    if psi >= self.cfg.psi_significant_threshold]

        return {
            "features_tracked": len(self._last_psi),
            "features_drifting": drifting,
            "features_critical": critical,
            "alerts": self._drift_alerts,
            "auc_decay": self.compute_auc_decay(),
        }

    def get_alerts(self) -> List[dict]:
        """Return pending drift alerts."""
        return list(self._drift_alerts)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Compute bin proportions for a set of values."""
        counts, _ = np.histogram(values, bins=edges)
        proportions = counts / len(values)
        # replace zeros with small epsilon to avoid log(0)
        proportions = np.where(proportions == 0, 1e-6, proportions)
        return proportions

    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
        """
        Population Stability Index.
        PSI = Σ (actual_i - expected_i) × ln(actual_i / expected_i)
        """
        # ensure no zeros
        expected = np.where(expected == 0, 1e-6, expected)
        actual = np.where(actual == 0, 1e-6, actual)

        psi = np.sum((actual - expected) * np.log(actual / expected))
        return float(max(0, psi))

    def _classify_psi(self, psi: float) -> str:
        """Classify PSI value."""
        if psi >= self.cfg.psi_significant_threshold:
            return "significant_drift"
        elif psi >= self.cfg.psi_moderate_threshold:
            return "moderate_drift"
        return "stable"

    @staticmethod
    def _approximate_auc(predictions: List[Tuple[float, int]]) -> Optional[float]:
        """
        Approximate AUC using Mann-Whitney U statistic.
        Avoids sklearn dependency for this single metric.
        """
        positives = [p for p, l in predictions if l == 1]
        negatives = [p for p, l in predictions if l == 0]

        if len(positives) < 5 or len(negatives) < 5:
            return None

        # Mann-Whitney: P(positive > negative)
        concordant = 0
        total = len(positives) * len(negatives)

        # sample if too many comparisons (>100K)
        if total > 100_000:
            rng = np.random.default_rng(42)
            pos_sample = rng.choice(positives, size=min(500, len(positives)), replace=False)
            neg_sample = rng.choice(negatives, size=min(500, len(negatives)), replace=False)
        else:
            pos_sample = positives
            neg_sample = negatives

        for p in pos_sample:
            for n in neg_sample:
                if p > n:
                    concordant += 1
                elif p == n:
                    concordant += 0.5

        auc = concordant / (len(pos_sample) * len(neg_sample))
        return float(auc)
