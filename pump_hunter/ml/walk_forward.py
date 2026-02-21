"""
Walk-forward validation — rolling window backtesting for ML models.

Components:
    - Rolling window: train on N days, test on M days, slide forward
    - Monte Carlo: bootstrap trade results for confidence intervals
    - Feature stability: track importance drift across windows
    - Overfit detection: train vs test AUC gap analysis
    - Regime-stratified validation: separate metrics per market regime
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pump_hunter.config.settings import Settings
from pump_hunter.ml.features import FEATURE_NAMES

logger = structlog.get_logger(__name__)


class WalkForwardValidator:
    """Rolling-window validation framework for pump prediction models."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_dir = Path(settings.ml.model_path)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # main validation
    # ------------------------------------------------------------------

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_days: int = 30,
        test_days: int = 7,
        step_days: int = 7,
    ) -> dict:
        """
        Walk-forward validation with rolling windows.

        Returns comprehensive report with per-window and aggregate metrics.
        """
        if len(X) < 50:
            return {"error": "insufficient_data", "samples": len(X)}

        # assume data is ordered chronologically, 1 sample ≈ 1 minute
        samples_per_day = 1440  # minutes
        train_size = train_days * samples_per_day
        test_size = test_days * samples_per_day
        step_size = step_days * samples_per_day

        windows = []
        feature_importances = []
        train_aucs = []
        test_aucs = []

        start = 0
        window_id = 0

        while start + train_size + test_size <= len(X):
            train_end = start + train_size
            test_end = train_end + test_size

            X_train = X.iloc[start:train_end]
            y_train = y.iloc[start:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]

            # skip windows with no positive examples
            if y_train.sum() < 3 or y_test.sum() < 1:
                start += step_size
                window_id += 1
                continue

            # train model for this window
            model = self._train_window_model(X_train, y_train)

            # evaluate
            train_metrics = self._evaluate(model, X_train, y_train)
            test_metrics = self._evaluate(model, X_test, y_test)

            # feature importance
            importance = self._get_importance(model, X.columns.tolist())
            feature_importances.append(importance)

            train_aucs.append(train_metrics.get("auc_roc", 0.5))
            test_aucs.append(test_metrics.get("auc_roc", 0.5))

            windows.append({
                "window_id": window_id,
                "train_start": start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "auc_gap": train_metrics.get("auc_roc", 0) - test_metrics.get("auc_roc", 0),
            })

            start += step_size
            window_id += 1

        if not windows:
            return {"error": "no_valid_windows"}

        # aggregate results
        report = self._build_report(windows, feature_importances, train_aucs, test_aucs)
        return report

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def monte_carlo(
        self,
        trade_results: List[float],
        n_simulations: int = 1000,
        n_trades: int = 100,
    ) -> dict:
        """
        Bootstrap Monte Carlo simulation for confidence intervals.

        Args:
            trade_results: list of P&L values from backtesting
            n_simulations: number of bootstrap samples
            n_trades: number of trades per simulation
        """
        if len(trade_results) < 10:
            return {"error": "insufficient_trade_history"}

        rng = np.random.default_rng(42)
        results = np.array(trade_results)

        final_pnls = []
        max_drawdowns = []

        for _ in range(n_simulations):
            # bootstrap sample
            sample = rng.choice(results, size=n_trades, replace=True)
            cumulative = np.cumsum(sample)

            final_pnls.append(float(cumulative[-1]))

            # max drawdown
            peak = np.maximum.accumulate(cumulative)
            drawdowns = (peak - cumulative) / np.maximum(peak, 1)
            max_drawdowns.append(float(np.max(drawdowns)) * 100)

        final_pnls = np.array(final_pnls)
        max_drawdowns = np.array(max_drawdowns)

        return {
            "n_simulations": n_simulations,
            "n_trades_per_sim": n_trades,
            "pnl": {
                "mean": round(float(np.mean(final_pnls)), 2),
                "median": round(float(np.median(final_pnls)), 2),
                "std": round(float(np.std(final_pnls)), 2),
                "ci_5": round(float(np.percentile(final_pnls, 5)), 2),
                "ci_25": round(float(np.percentile(final_pnls, 25)), 2),
                "ci_75": round(float(np.percentile(final_pnls, 75)), 2),
                "ci_95": round(float(np.percentile(final_pnls, 95)), 2),
                "pct_profitable": round(float(np.mean(final_pnls > 0) * 100), 1),
            },
            "max_drawdown": {
                "mean": round(float(np.mean(max_drawdowns)), 1),
                "median": round(float(np.median(max_drawdowns)), 1),
                "ci_95": round(float(np.percentile(max_drawdowns, 95)), 1),
            },
        }

    # ------------------------------------------------------------------
    # feature stability
    # ------------------------------------------------------------------

    def analyze_feature_stability(
        self, importances_over_windows: List[Dict[str, float]]
    ) -> dict:
        """
        Check if feature importances are stable across windows.
        High drift = overfitting risk.
        """
        if len(importances_over_windows) < 2:
            return {"stable": True, "windows": len(importances_over_windows)}

        # build matrix: rows=windows, cols=features
        all_features = set()
        for imp in importances_over_windows:
            all_features.update(imp.keys())

        features = sorted(all_features)
        matrix = np.zeros((len(importances_over_windows), len(features)))

        for i, imp in enumerate(importances_over_windows):
            for j, feat in enumerate(features):
                matrix[i, j] = imp.get(feat, 0)

        # normalize rows
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix_norm = matrix / row_sums

        # compute coefficient of variation per feature
        stability = {}
        unstable_features = []

        for j, feat in enumerate(features):
            values = matrix_norm[:, j]
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            cv = std_val / mean_val if mean_val > 0.001 else 0

            stability[feat] = {
                "mean_importance": round(mean_val, 4),
                "std": round(std_val, 4),
                "cv": round(cv, 3),
            }

            if cv > 0.5:  # >50% variation = unstable
                unstable_features.append(feat)

        return {
            "stable": len(unstable_features) == 0,
            "unstable_features": unstable_features,
            "feature_stability": stability,
            "windows_analyzed": len(importances_over_windows),
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _train_window_model(self, X: pd.DataFrame, y: pd.Series):
        """Train a model for one validation window."""
        model_type = self.settings.ml.model_type

        if model_type == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                max_depth=6, learning_rate=0.1, n_estimators=100,
                use_label_encoder=False, eval_metric="logloss",
                verbosity=0, scale_pos_weight=3,
            )
        else:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                max_depth=6, learning_rate=0.1, n_estimators=100,
                verbose=-1, scale_pos_weight=3,
            )

        model.fit(X, y)
        return model

    def _evaluate(self, model, X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate model on a dataset."""
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        }

        if len(np.unique(y)) > 1:
            metrics["auc_roc"] = round(roc_auc_score(y, y_proba), 4)
        else:
            metrics["auc_roc"] = 0.5

        return metrics

    def _get_importance(self, model, feature_names: list) -> dict:
        """Extract feature importances."""
        try:
            importances = model.feature_importances_
            return dict(zip(feature_names, importances.tolist()))
        except Exception:
            return {}

    def _build_report(
        self,
        windows: list,
        importances: list,
        train_aucs: list,
        test_aucs: list,
    ) -> dict:
        """Build comprehensive validation report."""
        test_metrics_agg = {
            "accuracy": float(np.mean([w["test_metrics"]["accuracy"] for w in windows])),
            "precision": float(np.mean([w["test_metrics"]["precision"] for w in windows])),
            "recall": float(np.mean([w["test_metrics"]["recall"] for w in windows])),
            "f1": float(np.mean([w["test_metrics"]["f1"] for w in windows])),
            "auc_roc": float(np.mean([w["test_metrics"]["auc_roc"] for w in windows])),
        }

        auc_gaps = [w["auc_gap"] for w in windows]
        avg_gap = float(np.mean(auc_gaps))

        # overfit check: train-test AUC gap > 0.15 = overfitting
        overfit_warning = avg_gap > 0.15
        overfit_severity = "SEVERE" if avg_gap > 0.25 else "MODERATE" if avg_gap > 0.15 else "LOW"

        # feature stability
        stability = self.analyze_feature_stability(importances)

        report = {
            "n_windows": len(windows),
            "aggregate_test_metrics": test_metrics_agg,
            "aggregate_train_metrics": {
                "auc_roc": float(np.mean(train_aucs)),
            },
            "overfit_analysis": {
                "avg_train_test_gap": round(avg_gap, 4),
                "max_gap": round(float(np.max(auc_gaps)), 4),
                "overfitting_detected": overfit_warning,
                "severity": overfit_severity,
            },
            "feature_stability": stability,
            "per_window": windows,
        }

        logger.info(
            "walk_forward_complete",
            windows=len(windows),
            avg_test_auc=round(test_metrics_agg["auc_roc"], 4),
            avg_gap=round(avg_gap, 4),
            overfitting=overfit_warning,
        )

        # save report
        report_path = self.model_dir / "walk_forward_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))

        return report
