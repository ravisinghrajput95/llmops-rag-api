"""Ingest endpoint: chunking, embedding, storage and cost reporting."""

from __future__ import annotations

import io

from tests.conftest import SAMPLE_DOCS


def test_ingest_documents_returns_counts_and_cost(client, embedding_client):
    response = client.post("/ingest", json={"documents": SAMPLE_DOCS})

    assert response.status_code == 201
    body = response.json()
    assert body["ingested_documents"] == 2
    assert body["ingested_chunks"] >= 2
    assert body["collection_size"] == body["ingested_chunks"]
    assert body["document_ids"] == ["cloudrun", "chroma"]
    assert body["embedding_tokens"] > 0
    assert body["estimated_cost_usd"] > 0
    # ~88 INR to the dollar, so the rupee figure must be the larger one.
    assert body["estimated_cost_inr"] > body["estimated_cost_usd"]
    assert body["latency_ms"] >= 0

    # One batched embedding call, not one call per chunk -- batching is what
    # keeps ingest cheap and fast.
    assert len(embedding_client.calls) == 1


def test_ingest_generates_stable_id_from_content(client):
    payload = {"documents": [{"text": "Repeatable content for hashing."}]}
    first = client.post("/ingest", json=payload).json()["document_ids"][0]
    second = client.post("/ingest", json=payload).json()["document_ids"][0]

    assert first.startswith("doc-")
    assert first == second


def test_reingesting_same_document_upserts_instead_of_duplicating(client):
    payload = {"documents": [SAMPLE_DOCS[0]]}
    first = client.post("/ingest", json=payload).json()
    second = client.post("/ingest", json=payload).json()

    assert second["collection_size"] == first["collection_size"]


def test_long_document_is_split_into_multiple_chunks(client):
    long_text = "\n\n".join(f"Paragraph {i} about serverless compute." for i in range(60))
    body = client.post("/ingest", json={"documents": [{"text": long_text}]}).json()

    assert body["ingested_chunks"] > 1


def test_ingest_rejects_empty_document_list(client):
    assert client.post("/ingest", json={"documents": []}).status_code == 422


def test_ingest_rejects_blank_text(client):
    response = client.post("/ingest", json={"documents": [{"text": ""}]})
    assert response.status_code == 422


def test_ingest_file_accepts_markdown(client):
    file = ("notes.md", io.BytesIO(b"# Notes\n\nCloud Run scales to zero."), "text/markdown")
    response = client.post("/ingest/file", files={"file": file})

    assert response.status_code == 201
    assert response.json()["ingested_chunks"] >= 1


def test_ingest_file_rejects_unsupported_extension(client):
    file = ("data.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")
    response = client.post("/ingest/file", files={"file": file})

    assert response.status_code == 415


def test_ingest_file_rejects_non_utf8_payload(client):
    file = ("bad.txt", io.BytesIO(b"\xff\xfe\x00binary"), "text/plain")
    response = client.post("/ingest/file", files={"file": file})

    assert response.status_code == 400


def test_ingest_file_rejects_empty_file(client):
    file = ("empty.txt", io.BytesIO(b"   \n  "), "text/plain")
    assert client.post("/ingest/file", files={"file": file}).status_code == 400
