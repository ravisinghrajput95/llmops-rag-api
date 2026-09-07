"""Durable state for a service whose filesystem is not durable.

The problem this solves, stated plainly: Cloud Run gives each instance a
private in-memory tmpfs. Documents ingested by one instance are invisible to
another, and everything vanishes when the service scales to zero after a few
idle minutes. Until now that was a documented limitation -- ingest, then query,
on a warm instance, and hope.

The fix is the one the README costed out as "free-ish": snapshot the Chroma
directory to the GCS bucket that already exists for MLflow artifacts, and
restore it on cold start. GCS is Always Free to 5 GB in US regions, so this
adds durability without adding a bill. Cloud SQL with pgvector would be the
"correct" answer and costs ~₹700/month, which is 2.6x the entire remaining
budget this project was built against.

What this is honest about:

  * **Concurrent writes are detected, not silently lost.** `save` takes an
    `expected_generation` and GCS refuses the upload if the object moved since
    the caller last read it. `RAGPipeline._save_snapshot` then replays its own
    chunks onto the newer snapshot and retries, so two instances ingesting at
    once end up with both sets of documents rather than whichever landed last.
    What this does *not* fix is the losing instance's own memory: its local
    store still lacks the other's documents until a cold start restores the
    merged file. Rebuilding a live Chroma client mid-request is a bigger risk
    than that staleness.
  * **Snapshots are whole-directory.** Chroma's SQLite file plus its HNSW index
    must move together or the collection is corrupt, so there is no useful
    incremental version.
  * **It is never in the query path.** Restore happens once at startup; upload
    happens after ingest. A query never touches GCS, so retrieval latency is
    unchanged.
  * **Failures are non-fatal.** A GCS outage degrades this to the previous
    ephemeral behaviour rather than taking the service down.
"""

from __future__ import annotations

import io
import logging
import shutil
import tarfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotResult:
    ok: bool
    detail: str
    bytes_transferred: int = 0
    duration_ms: float = 0.0
    # The GCS object generation this call read or wrote. Pass it back to the
    # next `save` to say "only write if nobody else has since".
    generation: int = 0
    # Someone else wrote between our read and our write. Not a failure: the
    # caller is expected to merge onto their version and try again.
    conflict: bool = False


class SnapshotStore:
    """Tar a local directory to a single GCS object, and back again.

    Construction never touches the network; the client is created lazily so
    that importing this module (and running the test suite) works with no GCP
    credentials present.
    """

    def __init__(self, bucket: str, object_name: str, enabled: bool = True) -> None:
        self._bucket_name = bucket
        self._object_name = object_name
        self._enabled = bool(enabled and bucket)
        self._client = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def uri(self) -> str:
        return f"gs://{self._bucket_name}/{self._object_name}" if self._enabled else ""

    def _blob(self):
        """Resolve the GCS blob, creating the client on first use."""
        if self._client is None:
            from google.cloud import storage  # imported lazily: see class docstring

            self._client = storage.Client()
        return self._client.bucket(self._bucket_name).blob(self._object_name)

    # -- save --------------------------------------------------------------
    def save(self, source_dir: str, expected_generation: int | None = None) -> SnapshotResult:
        """Upload `source_dir` as a gzipped tar. Never raises.

        With `expected_generation`, the upload is conditional: it succeeds only
        if the object still has that generation, and otherwise reports
        `conflict` instead of overwriting. 0 means "only if it does not exist
        yet". Without it the write is unconditional, which is correct only when
        one process owns the object -- the MLflow shards, not this one.
        """
        if not self._enabled:
            return SnapshotResult(ok=False, detail="disabled")

        source = Path(source_dir)
        if not source.exists():
            return SnapshotResult(ok=False, detail="source directory does not exist")

        started = time.perf_counter()
        try:
            # Built in memory: these snapshots are a few MB at demo scale, and
            # writing a temp file would consume the same tmpfs (and so the same
            # 512Mi memory limit) we are trying to work around.
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                archive.add(source, arcname=".")
            payload = buffer.getvalue()

            with self._lock:
                blob = self._blob()
                if expected_generation is None:
                    blob.upload_from_string(payload, content_type="application/gzip")
                else:
                    blob.upload_from_string(
                        payload,
                        content_type="application/gzip",
                        if_generation_match=expected_generation,
                    )
                written = int(getattr(blob, "generation", 0) or 0)

            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "snapshot uploaded",
                extra={
                    "uri": self.uri,
                    "bytes": len(payload),
                    "duration_ms": duration_ms,
                },
            )
            return SnapshotResult(
                ok=True,
                detail="uploaded",
                bytes_transferred=len(payload),
                duration_ms=duration_ms,
                generation=written,
            )
        except Exception as exc:
            if _is_precondition_failure(exc):
                # Expected under concurrency, not an error. Reported so the
                # caller can rebase its own writes onto the newer snapshot;
                # silently overwriting is what used to lose them.
                logger.info(
                    "snapshot changed underneath us; caller should merge and retry",
                    extra={"uri": self.uri, "expected_generation": expected_generation},
                )
                return SnapshotResult(ok=False, conflict=True, detail="generation conflict")
            # Losing a snapshot degrades durability; it must never fail the
            # ingest request that triggered it.
            logger.warning(
                "snapshot upload failed; continuing without durability",
                extra={"uri": self.uri, "error": str(exc)},
            )
            return SnapshotResult(ok=False, detail=f"upload failed: {exc}")

    # -- restore -----------------------------------------------------------
    def restore(self, target_dir: str) -> SnapshotResult:
        """Download and unpack into `target_dir`. Never raises.

        A missing object is the normal first-boot case, not an error.
        """
        if not self._enabled:
            return SnapshotResult(ok=False, detail="disabled")

        started = time.perf_counter()
        try:
            blob = self._blob()
            if not blob.exists():
                logger.info("no snapshot to restore", extra={"uri": self.uri})
                return SnapshotResult(ok=False, detail="no snapshot found")

            payload = blob.download_as_bytes()
            target = Path(target_dir)
            # Replace rather than merge: a half-old, half-new Chroma directory
            # is worse than either one alone.
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)

            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                _safe_extract(archive, target)

            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "snapshot restored",
                extra={
                    "uri": self.uri,
                    "bytes": len(payload),
                    "generation": getattr(blob, "generation", None),
                    "duration_ms": duration_ms,
                },
            )
            return SnapshotResult(
                ok=True,
                detail="restored",
                bytes_transferred=len(payload),
                duration_ms=duration_ms,
                generation=int(getattr(blob, "generation", 0) or 0),
            )
        except Exception as exc:
            logger.warning(
                "snapshot restore failed; starting with empty state",
                extra={"uri": self.uri, "error": str(exc)},
            )
            return SnapshotResult(ok=False, detail=f"restore failed: {exc}")


def _is_precondition_failure(exc: Exception) -> bool:
    """Did GCS reject the write because the object moved under us?

    Matched structurally rather than by importing google.api_core, which the
    lazy-client design keeps out of import time.
    """
    return getattr(exc, "code", None) == 412 or type(exc).__name__ == "PreconditionFailed"


def list_snapshot_objects(bucket: str, prefix: str, limit: int = 50) -> list[str]:
    """Object names under `prefix`, most recently written first. Never raises.

    Used to read a sharded snapshot back: one writer per object means no writer
    ever overwrites another, and the reader is what puts them back together.
    """
    if not bucket:
        return []
    try:
        from google.cloud import storage

        client = storage.Client()
        blobs = list(client.list_blobs(bucket, prefix=prefix, max_results=limit * 4))
        blobs.sort(key=lambda b: b.updated or 0, reverse=True)
        return [b.name for b in blobs[:limit]]
    except Exception as exc:
        logger.warning(
            "could not list snapshots",
            extra={"bucket": bucket, "prefix": prefix, "error": str(exc)},
        )
        return []


def _safe_extract(archive: tarfile.TarFile, target: Path) -> None:
    """Extract, refusing any member that would escape `target`.

    The archive is written by this same service, so this is defence in depth
    rather than a live threat -- but a tar extraction that trusts its input is
    exactly how path traversal bugs get shipped, and the check is three lines.
    """
    resolved_target = target.resolve()
    for member in archive.getmembers():
        destination = (resolved_target / member.name).resolve()
        if not destination.is_relative_to(resolved_target):
            raise ValueError(f"refusing unsafe tar member: {member.name}")
    # filter="data" is passed explicitly rather than left to the default: the
    # default is changing to this in Python 3.14 and warns until then, and a
    # security-relevant extraction should not quietly change behaviour under
    # an interpreter upgrade. It refuses absolute paths, links escaping the
    # destination and special files -- all belt-and-braces over the loop above,
    # which a snapshot of our own Chroma directory never trips.
    archive.extractall(target, filter="data")
