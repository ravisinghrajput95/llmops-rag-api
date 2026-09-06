"""Read recorded production queries back out of MLflow.

Kept apart from `drift.py` so the scoring logic stays pure functions over
plain records -- testable without standing up a tracking backend, which is
most of why it is testable at all.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from app.config import Settings
from app.monitoring.drift import QueryRecord
from app.persistence import SnapshotStore, list_snapshot_objects

logger = logging.getLogger(__name__)


def load_recent_queries(settings: Settings, limit: int = 500) -> list[QueryRecord]:
    """Most recent `/query` runs from the local tracking store, newest first."""
    return _read_tracking_db(settings.mlflow_tracking_uri, settings, limit)


def load_snapshot_queries(settings: Settings, limit: int = 500) -> list[QueryRecord]:
    """Merge every instance's snapshot shard into one window.

    Production writes one object per process, so reassembling them is the
    reader's job. Doing it here rather than on write is the whole reason two
    instances can no longer overwrite each other's runs.

    Missing or unreadable shards are skipped: a partial window is a worse
    measurement, but no window at all is not a better one.
    """
    names = list_snapshot_objects(
        settings.gcs_bucket,
        settings.mlflow_snapshot_prefix,
        limit=settings.mlflow_snapshot_max_shards,
    )
    if not names:
        return []

    records: list[QueryRecord] = []
    with tempfile.TemporaryDirectory() as workspace:
        for index, name in enumerate(names):
            target = Path(workspace) / str(index)
            result = SnapshotStore(settings.gcs_bucket, name).restore(str(target))
            if not result.ok:
                continue
            db = next(target.glob("*.db"), None)
            if db is None:
                logger.warning("shard holds no database", extra={"object": name})
                continue
            records.extend(_read_tracking_db(f"sqlite:///{db}", settings, limit))

    logger.info(
        "merged tracking shards",
        extra={"shards": len(names), "runs": len(records)},
    )
    # Newest first across all shards, so `limit` means the same thing it does
    # for a single store.
    records.sort(key=lambda r: r.started_at, reverse=True)
    return records[:limit]


def _read_tracking_db(uri: str, settings: Settings, limit: int) -> list[QueryRecord]:
    """Read `/query` runs out of one MLflow backend.

    Runs recorded before the `refused` metric existed are skipped rather than
    defaulted. Treating a missing metric as "not refused" would read a window
    of old traffic as a perfect zero refusal rate, which is the most flattering
    possible answer and the least true one.
    """
    import mlflow

    mlflow.set_tracking_uri(uri)
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
                started_at=float(run.info.start_time or 0),
            )
        )
    if skipped:
        logger.info(
            "skipped runs predating the refused metric",
            extra={"skipped": skipped, "used": len(records)},
        )
    return records
