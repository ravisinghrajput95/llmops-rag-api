"""MLflow tracking, including the guarantee that it can never break a request."""

from __future__ import annotations

import mlflow
import pytest

from app.config import Settings
from app.rag.pipeline import RAGPipeline
from app.tracking.mlflow_tracker import MLflowTracker, RunPayload, _truncate_params
from tests.conftest import SAMPLE_DOCS, FakeChatClient, FakeEmbeddingClient


@pytest.fixture
def tracking_settings(tmp_path) -> Settings:
    return Settings(
        openai_api_key="test-key-not-real",
        chroma_dir=str(tmp_path / "chroma"),
        chroma_collection="tracking-test",
        mlflow_enabled=True,
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        mlflow_experiment="test-experiment",
        top_k=3,
    )


@pytest.fixture
def tracked_pipeline(tracking_settings, tmp_path) -> RAGPipeline:
    from app.rag.vectorstore import ChromaVectorStore

    return RAGPipeline(
        settings=tracking_settings,
        store=ChromaVectorStore(
            tracking_settings.chroma_dir, tracking_settings.chroma_collection
        ),
        embedding_client=FakeEmbeddingClient(),
        chat_client=FakeChatClient(),
        tracker=MLflowTracker(tracking_settings),
    )


def _get_run(tracking_settings: Settings, run_id: str):
    mlflow.set_tracking_uri(tracking_settings.mlflow_tracking_uri)
    return mlflow.MlflowClient().get_run(run_id)


def test_query_creates_an_mlflow_run(tracked_pipeline, tracking_settings):
    tracked_pipeline.ingest([(SAMPLE_DOCS[0]["text"], "cloudrun", {})])
    result = tracked_pipeline.query("Does Cloud Run scale to zero?")

    assert result.mlflow_run_id is not None
    run = _get_run(tracking_settings, result.mlflow_run_id)
    assert run.info.status == "FINISHED"


def test_run_records_latency_tokens_and_cost(tracked_pipeline, tracking_settings):
    tracked_pipeline.ingest([(SAMPLE_DOCS[0]["text"], "cloudrun", {})])
    result = tracked_pipeline.query("What is Cloud Run?")

    metrics = _get_run(tracking_settings, result.mlflow_run_id).data.metrics
    for expected in (
        "latency_ms",
        "retrieval_ms",
        "generation_ms",
        "prompt_tokens",
        "completion_tokens",
        "embedding_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "estimated_cost_inr",
        "retrieved_chunks",
        "top_similarity",
    ):
        assert expected in metrics, f"missing metric: {expected}"

    assert metrics["total_tokens"] == (
        metrics["prompt_tokens"] + metrics["completion_tokens"] + metrics["embedding_tokens"]
    )
    assert metrics["estimated_cost_usd"] > 0


def test_run_records_prompt_and_response_artifacts(tracked_pipeline, tracking_settings):
    tracked_pipeline.ingest([(SAMPLE_DOCS[0]["text"], "cloudrun", {})])
    result = tracked_pipeline.query("What is Cloud Run?")

    mlflow.set_tracking_uri(tracking_settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    artifacts = {a.path for a in client.list_artifacts(result.mlflow_run_id)}

    assert {"question.txt", "answer.txt", "context.txt"} <= artifacts


def test_run_records_model_params_and_tags(tracked_pipeline, tracking_settings):
    tracked_pipeline.ingest([(SAMPLE_DOCS[0]["text"], "cloudrun", {})])
    result = tracked_pipeline.query("What is Cloud Run?")

    run = _get_run(tracking_settings, result.mlflow_run_id)
    assert run.data.params["chat_model"] == "gpt-4o-mini"
    assert run.data.params["top_k"] == "3"
    assert run.data.tags["endpoint"] == "/query"
    assert run.data.tags["grounded"] == "true"


def test_run_records_which_prompt_version_produced_it(tracked_pipeline, tracking_settings):
    """Without this, a prompt rewrite is invisible when two runs are compared
    months apart -- accuracy moves and nothing says why."""
    from app.rag.prompts import PROMPTS

    tracked_pipeline.ingest([(SAMPLE_DOCS[0]["text"], "cloudrun", {})])
    result = tracked_pipeline.query("What is Cloud Run?")

    params = _get_run(tracking_settings, result.mlflow_run_id).data.params
    assert params["prompt_version"] == PROMPTS.tracked_version
    assert params["prompt_fingerprint"] == PROMPTS.fingerprint


def test_ungrounded_query_is_tagged_as_such(tracked_pipeline, tracking_settings):
    result = tracked_pipeline.query("Nothing has been ingested yet.")

    run = _get_run(tracking_settings, result.mlflow_run_id)
    assert run.data.tags["grounded"] == "false"


def test_ingest_creates_its_own_run(tracked_pipeline, tracking_settings):
    tracked_pipeline.ingest([(SAMPLE_DOCS[0]["text"], "cloudrun", {})])

    mlflow.set_tracking_uri(tracking_settings.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(tracking_settings.mlflow_experiment)
    runs = mlflow.MlflowClient().search_runs([experiment.experiment_id])

    assert any(run.data.tags.get("endpoint") == "/ingest" for run in runs)


def test_disabled_tracker_is_a_no_op(tmp_path):
    tracker = MLflowTracker(
        Settings(mlflow_enabled=False, mlflow_tracking_uri=f"sqlite:///{tmp_path / 'x.db'}")
    )

    assert tracker.enabled is False
    assert tracker.log_run("query", RunPayload(metrics={"a": 1.0})) is None


def test_unreachable_backend_disables_tracking_instead_of_raising():
    tracker = MLflowTracker(
        Settings(mlflow_enabled=True, mlflow_tracking_uri="sqlite:////nope/does/not/exist.db")
    )

    assert tracker.enabled is False
    assert tracker.log_run("query", RunPayload()) is None


def test_query_still_succeeds_when_tracking_fails(tracking_settings, tmp_path):
    """The core fail-open guarantee: observability must not take down the API."""
    from app.rag.vectorstore import ChromaVectorStore

    class ExplodingTracker(MLflowTracker):
        def __init__(self) -> None:
            self.enabled = True
            self._settings = tracking_settings
            self._mlflow = object()  # non-None so log_run proceeds

        def log_run(self, run_name, payload):  # type: ignore[override]
            with self._guard():
                raise RuntimeError("mlflow backend is on fire")
            return None

    pipeline = RAGPipeline(
        settings=tracking_settings,
        store=ChromaVectorStore(str(tmp_path / "chroma2"), "failopen"),
        embedding_client=FakeEmbeddingClient(),
        chat_client=FakeChatClient(answer="Still answered."),
        tracker=ExplodingTracker(),
    )
    pipeline.ingest([(SAMPLE_DOCS[0]["text"], "cloudrun", {})])
    result = pipeline.query("What is Cloud Run?")

    assert result.answer == "Still answered."
    assert result.mlflow_run_id is None


def test_oversized_param_values_are_truncated_for_mlflow():
    truncated = _truncate_params({"question": "x" * 900})

    assert len(truncated["question"]) == 500
    assert truncated["question"].endswith("...")


class TestTrackingDurability:
    """The tracking DB has to outlive the instance, or drift is blind.

    MLflow artifacts already reach GCS; the run *metrics and tags* the monitor
    reads did not, because they live in a SQLite file on Cloud Run's tmpfs.
    """

    def _tracker(self, settings, store):
        from app.tracking.mlflow_tracker import MLflowTracker

        return MLflowTracker(settings, snapshots=store)

    def test_runs_are_snapshotted_once_the_threshold_is_reached(
        self, tmp_path, monkeypatch
    ) -> None:
        from app.persistence import SnapshotStore

        settings = Settings(
            openai_api_key="x",
            mlflow_enabled=True,
            mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow' / 'm.db'}",
            mlflow_experiment="durability",
            mlflow_snapshot_every=3,
        )
        saved: list[str] = []
        store = SnapshotStore("bucket", "snapshots/mlflow.tar.gz", enabled=True)
        monkeypatch.setattr(store, "save", lambda source: saved.append(source))
        tracker = self._tracker(settings, store)

        for _ in range(3):
            tracker.log_run("query", RunPayload(metrics={"refused": 0.0}))

        assert len(saved) == 1, "one upload per threshold, not one per run"
        assert saved[0].endswith("mlflow")

    def test_an_instance_never_reads_another_instances_shard(
        self, tmp_path, monkeypatch
    ) -> None:
        """Each process owns one object and starts empty.

        This used to restore a single shared snapshot, which is what made two
        instances overwrite each other's runs: both loaded the same file, both
        appended to it, and whichever wrote last won. Owning a shard removes
        the conflict instead of arbitrating it -- the reader merges.
        """
        from app.persistence import SnapshotStore

        settings = Settings(
            openai_api_key="x",
            mlflow_enabled=True,
            mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow' / 'm.db'}",
            mlflow_experiment="no-restore",
        )
        events: list[str] = []
        store = SnapshotStore("bucket", "snapshots/mlflow/a.tar.gz", enabled=True)
        monkeypatch.setattr(store, "restore", lambda target: events.append("restore"))
        monkeypatch.setattr(store, "save", lambda source: events.append("save"))

        tracker = self._tracker(settings, store)
        tracker.log_run("query", RunPayload(metrics={"refused": 1.0}))
        tracker.flush()

        assert "restore" not in events
        assert events == ["save"]

    def test_flush_uploads_whatever_the_threshold_has_not(self, tmp_path, monkeypatch) -> None:
        from app.persistence import SnapshotStore

        settings = Settings(
            openai_api_key="x",
            mlflow_enabled=True,
            mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow' / 'm.db'}",
            mlflow_experiment="flush",
            mlflow_snapshot_every=100,
        )
        saved: list[str] = []
        store = SnapshotStore("bucket", "snapshots/mlflow.tar.gz", enabled=True)
        monkeypatch.setattr(store, "save", lambda source: saved.append(source))
        tracker = self._tracker(settings, store)
        tracker.log_run("query", RunPayload(metrics={"refused": 0.0}))
        assert saved == []

        tracker.flush()

        assert len(saved) == 1

    def test_a_failing_snapshot_never_breaks_a_request(self, tmp_path, monkeypatch) -> None:
        from app.persistence import SnapshotStore

        settings = Settings(
            openai_api_key="x",
            mlflow_enabled=True,
            mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow' / 'm.db'}",
            mlflow_experiment="failopen",
            mlflow_snapshot_every=1,
        )

        def explode(source):
            raise RuntimeError("GCS is down")

        store = SnapshotStore("bucket", "snapshots/mlflow.tar.gz", enabled=True)
        monkeypatch.setattr(store, "save", explode)
        tracker = self._tracker(settings, store)

        assert tracker.log_run("query", RunPayload(metrics={"refused": 0.0})) is not None

    def test_snapshotting_is_off_without_a_bucket(self, tmp_path) -> None:
        """The local default must stay GCP-free."""
        settings = Settings(
            openai_api_key="x",
            mlflow_enabled=True,
            mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow' / 'm.db'}",
            mlflow_experiment="nobucket",
        )
        tracker = self._tracker(settings, None)

        tracker.flush()  # must not raise
        assert tracker.log_run("query", RunPayload()) is not None
