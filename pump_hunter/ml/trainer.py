"""
ML trainer — XGBoost/LightGBM with Optuna hyperparameter optimization.
Time-series-aware splits, feature importance, model versioning.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import structlog
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from pump_hunter.config.settings import Settings
from pump_hunter.ml.features import FEATURE_NAMES

logger = structlog.get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Trainer:
    """Trains pump prediction models with hyperparameter optimization."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_dir = Path(settings.ml.model_path)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        optimize: bool = True,
        n_trials: int = 50,
    ) -> Tuple[object, dict]:
        """
        Train a model on the provided dataset.
        Returns (model, metrics_dict).
        """
        model_type = self.settings.ml.model_type

        # time-series split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(
            "training_start",
            model_type=model_type,
            train_size=len(X_train),
            test_size=len(X_test),
            positive_ratio=f"{y_train.mean():.2%}",
        )

        if optimize:
            best_params = self._optimize(X_train, y_train, model_type, n_trials)
        else:
            best_params = self._default_params(model_type)

        # train final model
        model = self._create_model(model_type, best_params)
        model.fit(X_train, y_train)

        # evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "model_type": model_type,
            "params": best_params,
        }

        logger.info("training_complete", **{k: v for k, v in metrics.items() if k != "params"})
        logger.info("classification_report", report=classification_report(y_test, y_pred, zero_division=0))

        # feature importance
        importance = self._get_feature_importance(model, X.columns.tolist())
        metrics["feature_importance"] = importance

        # save model
        version = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_path = self._save_model(model, version, metrics)
        metrics["model_path"] = str(model_path)
        metrics["version"] = version

        return model, metrics

    def _optimize(
        self, X: pd.DataFrame, y: pd.Series, model_type: str, n_trials: int
    ) -> dict:
        """Optuna hyperparameter optimization."""
        def objective(trial):
            if model_type == "xgboost":
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
                }
            else:  # lightgbm
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
                }

            model = self._create_model(model_type, params)
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for train_idx, val_idx in tscv.split(X):
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                proba = model.predict_proba(X.iloc[val_idx])[:, 1]
                if len(np.unique(y.iloc[val_idx])) > 1:
                    scores.append(roc_auc_score(y.iloc[val_idx], proba))
                else:
                    scores.append(0.5)
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        logger.info("optimization_complete", best_auc=study.best_value, best_params=study.best_params)
        return study.best_params

    def _default_params(self, model_type: str) -> dict:
        """Default parameters if not optimizing."""
        if model_type == "xgboost":
            return {
                "max_depth": 6, "learning_rate": 0.1, "n_estimators": 200,
                "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8,
                "scale_pos_weight": 3,
            }
        else:
            return {
                "max_depth": 6, "learning_rate": 0.1, "n_estimators": 200,
                "num_leaves": 31, "min_child_samples": 20,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "scale_pos_weight": 3,
            }

    def _create_model(self, model_type: str, params: dict):
        """Create a model instance."""
        if model_type == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                **params,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
        else:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**params, verbose=-1)

    def _get_feature_importance(self, model, feature_names: list) -> dict:
        """Get feature importance from trained model."""
        try:
            importances = model.feature_importances_
            importance_dict = dict(zip(feature_names, importances.tolist()))
            sorted_imp = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            # log top 10
            top10 = list(sorted_imp.items())[:10]
            logger.info("top_features", features=top10)
            return sorted_imp
        except Exception:
            return {}

    def _save_model(self, model, version: str, metrics: dict) -> Path:
        """Save model and metadata."""
        model_path = self.model_dir / f"pump_model_{version}.joblib"
        meta_path = self.model_dir / f"pump_model_{version}.json"

        joblib.dump(model, model_path)

        # save metrics (excluding non-serializable items)
        meta = {k: v for k, v in metrics.items() if k not in ("model",)}
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

        logger.info("model_saved", path=str(model_path))
        return model_path
