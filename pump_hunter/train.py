"""
Pump Hunter — ML training entry point.
Builds dataset from recorded pump events, trains model, saves with versioning.
"""

from __future__ import annotations

import asyncio
import sys

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    context_class=dict,
)

logger = structlog.get_logger("trainer")


async def train():
    """Train pump prediction model."""
    from pump_hunter.config.settings import load_settings
    from pump_hunter.storage.database import Database
    from pump_hunter.ml.dataset_builder import DatasetBuilder
    from pump_hunter.ml.trainer import Trainer

    settings = load_settings()

    # connect to DB
    db = Database(settings)
    await db.start()

    try:
        # build dataset
        builder = DatasetBuilder(db)
        result = builder.build()

        if asyncio.iscoroutine(result):
            result = await result

        if result is None:
            logger.error("insufficient_data_for_training")
            logger.info(
                "💡 Run the scanner first to collect pump events.\n"
                "   Minimum 50 recorded pump events needed."
            )
            return

        X, y = result
        logger.info("dataset_ready", samples=len(X), positive=int(y.sum()), features=X.shape[1])

        # export dataset to CSV
        csv_path = await builder.export_to_csv("pump_dataset.csv")
        if csv_path:
            logger.info("dataset_exported", path=csv_path)

        # train model
        trainer = Trainer(settings)
        model, metrics = trainer.train(
            X, y,
            optimize=True,
            n_trials=settings.ml.optuna_trials,
        )

        logger.info(
            "🎉 Training complete!",
            accuracy=f"{metrics['accuracy']:.2%}",
            precision=f"{metrics['precision']:.2%}",
            recall=f"{metrics['recall']:.2%}",
            f1=f"{metrics['f1']:.2%}",
            auc=f"{metrics['auc_roc']:.4f}",
            model_path=metrics.get("model_path", ""),
        )

    finally:
        await db.stop()


def main():
    print(
        "\n"
        "╔═══════════════════════════════════════════╗\n"
        "║    🧠 PUMP HUNTER — Model Training        ║\n"
        "╚═══════════════════════════════════════════╝\n"
    )

    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass

    asyncio.run(train())


if __name__ == "__main__":
    main()
