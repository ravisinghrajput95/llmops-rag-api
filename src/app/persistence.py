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

  * **Last write wins.** Two instances ingesting at once will clobber each
    other's snapshot. Bounding that properly needs GCS generation preconditions
    and a retry loop, which is real complexity for a demo that ingests rarely
    and runs at max_instances=2. The generation of the restored snapshot is
    logged, so a lost write is at least diagnosable.
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
    def save(self, source_dir: str) -> SnapshotResult:
        """Upload `source_dir` as a gzipped tar. Never raises."""
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
                self._blob().upload_from_string(payload, content_type="application/gzip")

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
            )
        except Exception as exc:
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
            )
        except Exception as exc:
            logger.warning(
                "snapshot restore failed; starting with empty state",
                extra={"uri": self.uri, "error": str(exc)},
            )
            return SnapshotResult(ok=False, detail=f"restore failed: {exc}")


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
    archive.extractall(target)
