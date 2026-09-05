"""Tests for GCS snapshot/restore.

No network and no credentials: the GCS blob is replaced with an in-memory
fake, so these run in CI exactly as they run locally. What is being tested is
this module's behaviour -- round-tripping, replacement semantics, failure
containment and tar safety -- not the correctness of google-cloud-storage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.persistence import SnapshotStore, _safe_extract


class FakeBlob:
    """Minimal stand-in for a google.cloud.storage Blob."""

    def __init__(self) -> None:
        self.data: bytes | None = None
        self.generation = 1
        self.upload_calls = 0

    def exists(self) -> bool:
        return self.data is not None

    def upload_from_string(self, payload: bytes, content_type: str = "") -> None:
        self.data = payload
        self.upload_calls += 1
        self.generation += 1

    def download_as_bytes(self) -> bytes:
        if self.data is None:
            raise FileNotFoundError("no such object")
        return self.data


@pytest.fixture
def snapshot_store(monkeypatch: pytest.MonkeyPatch) -> tuple[SnapshotStore, FakeBlob]:
    snapshots = SnapshotStore(bucket="test-bucket", object_name="snapshots/chroma.tar.gz")
    blob = FakeBlob()
    monkeypatch.setattr(snapshots, "_blob", lambda: blob)
    return snapshots, blob


def _make_tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "chroma.sqlite3").write_text("pretend sqlite content")
    nested = root / "index"
    nested.mkdir(exist_ok=True)
    (nested / "hnsw.bin").write_text("pretend hnsw index")


class TestDisabledStore:
    def test_no_bucket_disables_the_store(self) -> None:
        snapshots = SnapshotStore(bucket="", object_name="x")
        assert snapshots.enabled is False
        assert snapshots.uri == ""
        assert snapshots.save("/tmp").ok is False
        assert snapshots.restore("/tmp").ok is False

    def test_explicitly_disabled_store_is_inert(self) -> None:
        snapshots = SnapshotStore(bucket="b", object_name="o", enabled=False)
        assert snapshots.enabled is False


class TestRoundTrip:
    def test_uri_is_reported(self, snapshot_store) -> None:
        snapshots, _ = snapshot_store
        assert snapshots.uri == "gs://test-bucket/snapshots/chroma.tar.gz"

    def test_save_then_restore_reproduces_the_tree(
        self, snapshot_store, tmp_path: Path
    ) -> None:
        snapshots, _ = snapshot_store
        source = tmp_path / "chroma"
        _make_tree(source)

        assert snapshots.save(str(source)).ok is True

        target = tmp_path / "restored"
        result = snapshots.restore(str(target))

        assert result.ok is True
        assert (target / "chroma.sqlite3").read_text() == "pretend sqlite content"
        assert (target / "index" / "hnsw.bin").read_text() == "pretend hnsw index"

    def test_restore_replaces_rather_than_merges(self, snapshot_store, tmp_path: Path) -> None:
        """A half-old, half-new Chroma directory is worse than either alone."""
        snapshots, _ = snapshot_store
        source = tmp_path / "chroma"
        _make_tree(source)
        snapshots.save(str(source))

        target = tmp_path / "target"
        target.mkdir()
        (target / "stale.txt").write_text("should not survive")

        snapshots.restore(str(target))

        assert not (target / "stale.txt").exists()
        assert (target / "chroma.sqlite3").exists()

    def test_save_reports_bytes_transferred(self, snapshot_store, tmp_path: Path) -> None:
        snapshots, _ = snapshot_store
        source = tmp_path / "chroma"
        _make_tree(source)
        result = snapshots.save(str(source))
        assert result.bytes_transferred > 0
        assert result.duration_ms >= 0


class TestFailureContainment:
    def test_missing_source_directory_is_not_fatal(
        self, snapshot_store, tmp_path: Path
    ) -> None:
        snapshots, _ = snapshot_store
        result = snapshots.save(str(tmp_path / "does-not-exist"))
        assert result.ok is False
        assert "does not exist" in result.detail

    def test_first_boot_with_no_snapshot_is_not_an_error(
        self, snapshot_store, tmp_path: Path
    ) -> None:
        snapshots, _ = snapshot_store
        result = snapshots.restore(str(tmp_path / "target"))
        assert result.ok is False
        assert result.detail == "no snapshot found"

    def test_upload_failure_does_not_raise(self, snapshot_store, tmp_path: Path) -> None:
        """A GCS outage must degrade durability, never fail the ingest."""
        snapshots, blob = snapshot_store
        source = tmp_path / "chroma"
        _make_tree(source)

        def boom(*args, **kwargs):
            raise RuntimeError("GCS is having a day")

        blob.upload_from_string = boom
        result = snapshots.save(str(source))

        assert result.ok is False
        assert "upload failed" in result.detail

    def test_restore_failure_leaves_service_startable(
        self, snapshot_store, tmp_path: Path
    ) -> None:
        snapshots, blob = snapshot_store
        blob.data = b"this is not a valid gzip tarball"
        result = snapshots.restore(str(tmp_path / "target"))
        assert result.ok is False
        assert "restore failed" in result.detail


class TestTarSafety:
    def test_path_traversal_member_is_refused(self, tmp_path: Path) -> None:
        """Defence in depth: we write these archives ourselves, but a tar
        extraction that trusts its input is how traversal bugs ship."""
        import io
        import tarfile

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            payload = b"owned"
            info = tarfile.TarInfo(name="../escaped.txt")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

        target = tmp_path / "target"
        target.mkdir()
        buffer.seek(0)
        with tarfile.open(fileobj=buffer, mode="r:gz") as archive:
            with pytest.raises(ValueError, match="unsafe tar member"):
                _safe_extract(archive, target)

        assert not (tmp_path / "escaped.txt").exists()


class TestIngestIntegration:
    """The wiring that matters: ingest must persist, query must not."""

    def test_ingest_triggers_a_snapshot(self, client, pipeline, monkeypatch) -> None:
        snapshots = SnapshotStore(bucket="test-bucket", object_name="chroma.tar.gz")
        blob = FakeBlob()
        monkeypatch.setattr(snapshots, "_blob", lambda: blob)
        pipeline._snapshots = snapshots

        response = client.post(
            "/ingest", json={"documents": [{"text": "Cloud Run scales to zero."}]}
        )

        assert response.status_code == 201
        assert blob.upload_calls == 1

    def test_query_never_touches_gcs(self, client, pipeline, monkeypatch) -> None:
        """Retrieval latency must not depend on a GCS round trip."""
        snapshots = SnapshotStore(bucket="test-bucket", object_name="chroma.tar.gz")
        blob = FakeBlob()
        monkeypatch.setattr(snapshots, "_blob", lambda: blob)
        pipeline._snapshots = snapshots

        client.post("/query", json={"question": "does this hit gcs?"})

        assert blob.upload_calls == 0

    def test_ingest_with_no_chunks_skips_the_upload(self, pipeline, monkeypatch) -> None:
        """Snapshotting an unchanged store would burn a GCS write per no-op."""
        snapshots = SnapshotStore(bucket="test-bucket", object_name="chroma.tar.gz")
        blob = FakeBlob()
        monkeypatch.setattr(snapshots, "_blob", lambda: blob)
        pipeline._snapshots = snapshots

        pipeline.ingest([("   ", None, {})])

        assert blob.upload_calls == 0

    def test_ready_reports_persistence_state(self, client, pipeline, monkeypatch) -> None:
        snapshots = SnapshotStore(bucket="test-bucket", object_name="chroma.tar.gz")
        monkeypatch.setattr(snapshots, "_blob", lambda: FakeBlob())
        pipeline._snapshots = snapshots

        persistence = client.get("/ready").json()["persistence"]

        assert persistence["enabled"] is True
        assert persistence["uri"] == "gs://test-bucket/chroma.tar.gz"
