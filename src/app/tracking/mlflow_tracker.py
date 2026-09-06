"""MLflow tracking for LLM calls.

Design rule: **tracking must never break a request.** Every public method here
is wrapped so that an MLflow failure (locked SQLite, unreachable GCS bucket,
expired credentials) degrades to a warning log and the user still gets their
answer. An observability layer that can take down the service it observes is
worse than no observability layer.

Backend store : SQLite  (MLFLOW_TRACKING_URI)
Artifact store: GCS     (MLFLOW_ARTIFACT_LOCATION, set at experiment creation)
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.config import Settings
from app.logging_config import ensure_structured_logging
from app.persistence import SnapshotStore

logger = logging.getLogger(__name__)


@dataclass
class RunPayload:
    """Everything we want to record about one LLM interaction."""

    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    # name -> text content, written to the artifact store (GCS in prod).
    artifacts: dict[str, str] = field(default_factory=dict)


class MLflowTracker:
    def __init__(self, settings: Settings, snapshots: SnapshotStore | None = None) -> None:
        self.enabled = settings.mlflow_enabled
        self._settings = settings
        self._experiment_id: str | None = None
        self._lock = threading.Lock()
        self._mlflow: Any = None
        # The tracker owns its own durability: nothing else knows when the
        # backing file has changed. It writes its own shard and never reads
        # anyone else's -- see `mlflow_snapshot_prefix` for why sharding rather
        # than a shared object.
        self._snapshots = snapshots or SnapshotStore("", "", enabled=False)
        self._db_dir: Path | None = None
        self._runs_since_snapshot = 0

        if not self.enabled:
            logger.info("mlflow tracking disabled")
            return

        try:
            self._init_backend()
        except Exception:
            logger.warning(
                "mlflow init failed; tracking disabled for this process",
                exc_info=True,
            )
            self.enabled = False

    def _init_backend(self) -> None:
        import mlflow

        self._mlflow = mlflow
        uri = self._settings.mlflow_tracking_uri

        # A SQLite file can't be created inside a directory that doesn't exist.
        if uri.startswith("sqlite:///"):
            db_path = Path(uri.replace("sqlite:///", "", 1))
            if db_path.parent and str(db_path.parent) not in ("", "."):
                db_path.parent.mkdir(parents=True, exist_ok=True)
                self._db_dir = db_path.parent

        mlflow.set_tracking_uri(uri)

        # Creating the schema emits ~30 alembic migration lines through
        # MLflow's own logging config, which we cannot pre-empt because it is
        # installed during this very call. Blanket-suppress INFO and below for
        # the duration: those lines are unstructured, cost Cloud Logging
        # ingest, and say nothing useful on a cold start.
        logging.disable(logging.INFO)
        try:
            experiment = mlflow.get_experiment_by_name(self._settings.mlflow_experiment)
            if experiment is None:
                # artifact_location is immutable after creation, so it can only
                # be applied here, on first run against a fresh backend.
                artifact_location = self._settings.mlflow_artifact_location or None
                self._experiment_id = mlflow.create_experiment(
                    self._settings.mlflow_experiment, artifact_location=artifact_location
                )
            else:
                self._experiment_id = experiment.experiment_id
        finally:
            logging.disable(logging.NOTSET)

        # MLflow reconfigured the root logger while building its schema above.
        ensure_structured_logging()

        logger.info(
            "mlflow tracking ready",
            extra={
                "tracking_uri": uri,
                "experiment": self._settings.mlflow_experiment,
                "experiment_id": self._experiment_id,
                "artifact_location": self._settings.mlflow_artifact_location or "local",
            },
        )

    @contextmanager
    def _guard(self) -> Iterator[None]:
        try:
            yield
        except Exception:
            logger.warning("mlflow logging failed; request unaffected", exc_info=True)

    def log_run(self, run_name: str, payload: RunPayload) -> str | None:
        """Write one MLflow run. Returns the run id, or None on any failure."""
        if not self.enabled or self._mlflow is None:
            return None

        run_id: str | None = None
        with self._guard():
            # SQLite tolerates concurrent readers but not concurrent writers;
            # serialising here avoids "database is locked" under Cloud Run's
            # default concurrency of 80 requests per instance.
            with self._lock:
                with self._mlflow.start_run(
                    experiment_id=self._experiment_id, run_name=run_name
                ) as run:
                    run_id = run.info.run_id
                    if payload.tags:
                        self._mlflow.set_tags(payload.tags)
                    if payload.params:
                        self._mlflow.log_params(_truncate_params(payload.params))
                    if payload.metrics:
                        self._mlflow.log_metrics(payload.metrics)
                    if payload.artifacts and self._settings.mlflow_log_artifacts:
                        for name, content in payload.artifacts.items():
                            self._mlflow.log_text(content, name)
        # Guard against MLflow reconfiguring logging on any later code path.
        ensure_structured_logging()

        if run_id is not None:
            self._runs_since_snapshot += 1
            every = self._settings.mlflow_snapshot_every
            if every > 0 and self._runs_since_snapshot >= every:
                self.flush()
        return run_id

    def flush(self) -> None:
        """Push the tracking DB to durable storage. Safe to call at any time.

        Called on a run-count threshold and again at shutdown, so an instance
        that is being drained saves what the threshold has not yet covered.
        Cloud Run's SIGTERM grace period makes that best-effort, not a
        guarantee -- hence the threshold as well.
        """
        if not self._snapshots.enabled or self._db_dir is None:
            return
        self._runs_since_snapshot = 0
        with self._guard():
            with self._lock:
                self._snapshots.save(str(self._db_dir))

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tracking_uri": self._settings.mlflow_tracking_uri if self.enabled else None,
            "experiment": self._settings.mlflow_experiment if self.enabled else None,
        }


def _truncate_params(params: dict[str, Any]) -> dict[str, Any]:
    """MLflow rejects param values over 6000 chars; artifacts hold the full text."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        text = str(value)
        out[key] = text if len(text) <= 500 else text[:497] + "..."
    return out


def resolve_artifact_location(settings: Settings) -> str:
    """Validate MLFLOW_ARTIFACT_LOCATION, warning about the common mistakes."""
    location = settings.mlflow_artifact_location.strip()
    if not location:
        return ""
    parsed = urlparse(location)
    if parsed.scheme not in ("gs", "file", ""):
        logger.warning(
            "unsupported artifact scheme; falling back to local files",
            extra={"artifact_location": location},
        )
        return ""
    if parsed.scheme == "gs" and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        # On Cloud Run the metadata server supplies ADC, so this is expected
        # there; it is only worth flagging for local development.
        logger.debug("using ambient application default credentials for GCS")
    return location
