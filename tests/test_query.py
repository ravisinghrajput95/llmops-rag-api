"""Query endpoint: retrieval, grounding, token/cost accounting."""

from __future__ import annotations

import pytest

from tests.conftest import SAMPLE_DOCS


@pytest.fixture
def ingested_client(client):
    client.post("/ingest", json={"documents": SAMPLE_DOCS})
    return client


def test_query_returns_grounded_answer_with_sources(ingested_client, chat_client):
    chat_client.answer = "It scales to zero when idle. [1]"

    response = ingested_client.post(
        "/query", json={"question": "Does Cloud Run scale to zero?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "It scales to zero when idle. [1]"
    assert body["model"] == "gpt-4o-mini"
    assert len(body["sources"]) >= 1
    assert all(0.0 <= source["similarity"] <= 1.0 for source in body["sources"])


def test_query_retrieves_the_relevant_document(ingested_client):
    body = ingested_client.post(
        "/query", json={"question": "Which vector database is embedded and needs no server?"}
    ).json()

    # The Chroma document shares vocabulary with the question, so it must rank
    # above the Cloud Run one.
    assert body["sources"][0]["doc_id"] == "chroma"


def test_query_prompt_contains_numbered_context(ingested_client, chat_client):
    ingested_client.post("/query", json={"question": "What is Cloud Run?"})

    prompt = chat_client.last_user_prompt
    assert "[1]" in prompt
    assert "Question: What is Cloud Run?" in prompt
    assert "Cloud Run is a serverless container platform" in prompt


def test_query_reports_token_usage_and_cost(ingested_client):
    body = ingested_client.post("/query", json={"question": "What is Cloud Run?"}).json()

    usage = body["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["embedding_tokens"] > 0
    assert body["estimated_cost_usd"] > 0
    assert body["estimated_cost_inr"] == pytest.approx(
        body["estimated_cost_usd"] * 88.0, rel=1e-3
    )


def test_query_reports_latency_split_by_stage(ingested_client):
    body = ingested_client.post("/query", json={"question": "What is Cloud Run?"}).json()

    assert body["retrieval_ms"] >= 0
    assert body["generation_ms"] >= 0
    # Total wall time must cover both stages, allowing for rounding.
    assert body["latency_ms"] + 1e-6 >= body["retrieval_ms"] + body["generation_ms"] - 1.0


def test_query_honours_top_k(ingested_client):
    body = ingested_client.post("/query", json={"question": "serverless", "top_k": 1}).json()

    assert len(body["sources"]) == 1


def test_query_on_empty_collection_skips_the_llm_call(client, chat_client):
    """No context means no useful answer -- so we must not pay for a completion."""
    response = client.post("/query", json={"question": "Anything in here?"})

    assert response.status_code == 200
    body = response.json()
    assert "I don't know" in body["answer"]
    assert body["sources"] == []
    assert body["usage"]["prompt_tokens"] == 0
    assert body["usage"]["completion_tokens"] == 0
    assert chat_client.calls == []  # the expensive call never happened


def test_query_rejects_empty_question(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_query_rejects_missing_question(client):
    assert client.post("/query", json={}).status_code == 422


def test_query_rejects_out_of_range_top_k(client):
    assert client.post("/query", json={"question": "x", "top_k": 0}).status_code == 422
    assert client.post("/query", json={"question": "x", "top_k": 99}).status_code == 422


def test_upstream_llm_failure_surfaces_as_error(ingested_client, chat_client):
    chat_client.error = RuntimeError("openai upstream exploded")

    with pytest.raises(RuntimeError):
        ingested_client.post("/query", json={"question": "What is Cloud Run?"})


def test_openai_failure_returns_502_not_500(ingested_client, chat_client):
    """A provider outage is a bad gateway, not an internal error.

    502 tells the caller the service itself is healthy and the request is
    worth retrying; a bare 500 buries that.
    """
    from openai import APIConnectionError

    chat_client.error = APIConnectionError(request=None)

    response = ingested_client.post("/query", json={"question": "What is Cloud Run?"})

    assert response.status_code == 502
    body = response.json()
    assert body["error_type"] == "APIConnectionError"
    assert "Upstream LLM provider" in body["detail"]


def test_similarity_floor_prevents_paying_for_irrelevant_context(tmp_path):
    """The cost control that matters in production.

    A non-empty collection always returns *something* from a vector search, so
    the empty-collection check alone is not enough: an unrelated question still
    retrieves a junk chunk and pays for a completion. The similarity floor is
    what actually prevents that spend. Verified against the live service, where
    an unrelated question scored 0.10 while a relevant one scored 0.62.
    """
    from app.config import Settings
    from app.rag.pipeline import RAGPipeline
    from app.rag.vectorstore import ChromaVectorStore
    from app.tracking.mlflow_tracker import MLflowTracker
    from tests.conftest import FakeChatClient, FakeEmbeddingClient

    settings = Settings(
        openai_api_key="test-key-not-real",
        chroma_dir=str(tmp_path / "floor"),
        chroma_collection="floor-test",
        mlflow_enabled=False,
        min_similarity=0.9,  # only near-identical text may reach the model
    )
    chat = FakeChatClient()
    pipeline = RAGPipeline(
        settings=settings,
        store=ChromaVectorStore(settings.chroma_dir, settings.chroma_collection),
        embedding_client=FakeEmbeddingClient(),
        chat_client=chat,
        tracker=MLflowTracker(settings),
    )
    pipeline.ingest([("Cloud Run scales to zero when idle.", "gcp", {})])

    result = pipeline.query("zebra giraffe rhinoceros safari wildlife")

    assert "I don't know" in result.answer
    assert result.sources == []
    assert result.completion_tokens == 0
    assert chat.calls == [], "the paid completion call must not happen"
