"""Health, readiness and the OpenAPI contract."""

from __future__ import annotations


def test_health_is_ok_and_does_no_io(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "llmops-rag-api"
    assert body["version"]


def test_ready_reports_empty_collection(client):
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["collection_size"] == 0
    assert body["openai_configured"] is True


def test_ready_reflects_ingested_documents(client):
    client.post("/ingest", json={"documents": [{"text": "Cloud Run scales to zero."}]})
    assert client.get("/ready").json()["collection_size"] == 1


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/query" in schema["paths"]
    assert "/ingest" in schema["paths"]
