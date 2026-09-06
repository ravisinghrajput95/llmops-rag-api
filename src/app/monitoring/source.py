"""Read recorded production queries back out of MLflow.

Kept apart from `drift.py` so the scoring logic stays pure functions over
plain records -- testable without standing up a tracking backend, which is
most of why it is testable at all.
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.monitoring.drift import QueryRecord

logger = logging.getLogger(__name__)


def load_recent_queries(settings: Settings, limit: int = 500) -> list[QueryRecord]:
    """Most recent `/query` runs, newest first.

    Runs recorded before the `refused` metric existed are skipped rather than
    defaulted. Treating a missing metric as "not refused" would read a window
    of old traffic as a perfect zero refusal rate, which is the most flattering
    possible answer and the least true one.
    """
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(settings.mlflow_experiment)
    if experiment is None:
        return []

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.stage = 'query'",
        order_by=["attributes.start_time DESC"],
        max_results=limit,
    )

    records: list[QueryRecord] = []
    skipped = 0
    for run in runs:
        metrics, tags, params = run.data.metrics, run.data.tags, run.data.params
        if "refused" not in metrics:
            skipped += 1
            continue
        records.append(
            QueryRecord(
                refused=bool(metrics["refused"]),
                grounded=tags.get("grounded") == "true",
                top_similarity=float(metrics.get("top_similarity", 0.0)),
                retrieved_chunks=float(metrics.get("retrieved_chunks", 0.0)),
                cost_usd=float(metrics.get("estimated_cost_usd", 0.0)),
                latency_ms=float(metrics.get("latency_ms", 0.0)),
                prompt_version=params.get("prompt_version", ""),
            )
        )
    if skipped:
        logger.info(
            "skipped runs predating the refused metric",
            extra={"skipped": skipped, "used": len(records)},
        )
    return records
