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


class FakePreconditionFailed(Exception):
    """Stands in for google.api_core.exceptions.PreconditionFailed (HTTP 412)."""

    code = 412


class FakeBlob:
    """Minimal stand-in for a google.cloud.storage Blob.

    Enforces `if_generation_match`, because that precondition is the whole
    mechanism stopping two instances from overwriting each other -- a double
    that accepted the argument and ignored it would make the tests pass
    whether or not the feature worked.
    """

    def __init__(self) -> None:
        self.data: bytes | None = None
        self.generation = 0
        self.upload_calls = 0

    def exists(self) -> bool:
        return self.data is not None

    @property
    def _live_generation(self) -> int:
        # GCS treats "does not exist" as generation 0 for preconditions.
        return self.generation if self.data is not None else 0

    def upload_from_string(
        self,
        payload: bytes,
        content_type: str = "",
        if_generation_match: int | None = None,
    ) -> None:
        if if_generation_match is not None and if_generation_match != self._live_generation:
            raise FakePreconditionFailed("generation mismatch")
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


class TestConcurrentIngest:
    """Two instances ingesting at once must not discard each other's documents.

    The store is a single shared object, and it used to be written
    unconditionally: whoever uploaded last won, and the other instance's
    documents were gone with no error anywhere. Documents are the one thing in
    that bucket that cannot be regenerated.
    """

    def _pipeline(self, tmp_path, name, blob, monkeypatch):
        from app.config import Settings
        from app.rag.pipeline import RAGPipeline
        from app.rag.vectorstore import ChromaVectorStore
        from app.tracking.mlflow_tracker import MLflowTracker
        from tests.conftest import FakeChatClient, FakeEmbeddingClient

        settings = Settings(
            openai_api_key="x",
            chroma_dir=str(tmp_path / name / "chroma"),
            chroma_collection="shared",
            mlflow_enabled=False,
            min_similarity=0.0,
        )
        store = SnapshotStore(bucket="b", object_name="snapshots/chroma.tar.gz")
        monkeypatch.setattr(store, "_blob", lambda: blob)
        return RAGPipeline(
            settings=settings,
            store=ChromaVectorStore(settings.chroma_dir, settings.chroma_collection),
            embedding_client=FakeEmbeddingClient(),
            chat_client=FakeChatClient(),
            tracker=MLflowTracker(settings),
            snapshots=store,
        )

    def _docs_in_snapshot(self, blob, tmp_path, monkeypatch):
        """Restore the shared object and list the doc ids actually in it."""
        from app.rag.vectorstore import ChromaVectorStore

        reader = SnapshotStore(bucket="b", object_name="snapshots/chroma.tar.gz")
        monkeypatch.setattr(reader, "_blob", lambda: blob)
        target = tmp_path / "verify"
        assert reader.restore(str(target)).ok
        store = ChromaVectorStore(str(target), "shared")
        hits = store.search([1.0] + [0.0] * 63, top_k=50, min_similarity=-1.0)
        return {h.doc_id for h in hits}, store.count()

    def test_a_second_instance_does_not_erase_the_first(self, tmp_path, monkeypatch) -> None:
        blob = FakeBlob()
        a = self._pipeline(tmp_path, "a", blob, monkeypatch)
        b = self._pipeline(tmp_path, "b", blob, monkeypatch)

        # Both started before either wrote, so both believe generation 0.
        a.ingest([("Cloud Run scales to zero when idle.", "from-a", {})])
        b.ingest([("Chroma is an embedded vector database.", "from-b", {})])

        docs, count = self._docs_in_snapshot(blob, tmp_path, monkeypatch)
        assert docs == {"from-a", "from-b"}, f"a document was discarded: {docs}"
        assert count == 2

    def test_the_loser_of_the_race_retries_rather_than_failing(
        self, tmp_path, monkeypatch
    ) -> None:
        """The second writer should upload twice: once refused, once merged."""
        blob = FakeBlob()
        a = self._pipeline(tmp_path, "a", blob, monkeypatch)
        b = self._pipeline(tmp_path, "b", blob, monkeypatch)

        a.ingest([("Cloud Run scales to zero.", "from-a", {})])
        uploads_after_a = blob.upload_calls
        b.ingest([("Chroma is embedded.", "from-b", {})])

        assert blob.upload_calls == uploads_after_a + 1, "the merged write landed"

    def test_three_way_contention_still_keeps_every_document(
        self, tmp_path, monkeypatch
    ) -> None:
        blob = FakeBlob()
        pipelines = [
            self._pipeline(tmp_path, name, blob, monkeypatch) for name in ("a", "b", "c")
        ]
        for index, pipeline in enumerate(pipelines):
            pipeline.ingest([(f"Document number {index}.", f"doc-{index}", {})])

        docs, _ = self._docs_in_snapshot(blob, tmp_path, monkeypatch)
        assert docs == {"doc-0", "doc-1", "doc-2"}

    def test_re_ingesting_the_same_document_does_not_duplicate_it(
        self, tmp_path, monkeypatch
    ) -> None:
        """Merging replays chunks by id, so convergence depends on the upsert."""
        blob = FakeBlob()
        a = self._pipeline(tmp_path, "a", blob, monkeypatch)
        b = self._pipeline(tmp_path, "b", blob, monkeypatch)

        a.ingest([("The same text entirely.", "shared-doc", {})])
        b.ingest([("The same text entirely.", "shared-doc", {})])

        docs, count = self._docs_in_snapshot(blob, tmp_path, monkeypatch)
        assert docs == {"shared-doc"}
        assert count == 1, "the replay must upsert, not append"

    def test_a_single_instance_still_writes_once(self, tmp_path, monkeypatch) -> None:
        """No merge round-trip in the common uncontended case."""
        blob = FakeBlob()
        a = self._pipeline(tmp_path, "a", blob, monkeypatch)

        a.ingest([("One document.", "only", {})])
        a.ingest([("Another document.", "second", {})])

        assert blob.upload_calls == 2, "one upload per ingest when uncontended"
