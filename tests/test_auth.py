"""Optional API-key guard on the endpoints that spend money."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.dependencies import get_pipeline
from app.main import app


@pytest.fixture
def secured_client(pipeline, settings: Settings):
    """Client with APP_API_KEY enforced."""
    secured = settings.model_copy(update={"app_api_key": "s3cret"})

    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_settings] = lambda: secured
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_query_without_key_is_rejected(secured_client):
    response = secured_client.post("/query", json={"question": "hi"})

    assert response.status_code == 401


def test_query_with_wrong_key_is_rejected(secured_client):
    response = secured_client.post(
        "/query", json={"question": "hi"}, headers={"X-API-Key": "wrong"}
    )

    assert response.status_code == 401


def test_query_with_correct_key_is_allowed(secured_client):
    response = secured_client.post(
        "/query", json={"question": "hi"}, headers={"X-API-Key": "s3cret"}
    )

    assert response.status_code == 200


def test_ingest_is_protected_too(secured_client):
    unauthenticated = secured_client.post("/ingest", json={"documents": [{"text": "x"}]})
    authenticated = secured_client.post(
        "/ingest", json={"documents": [{"text": "x"}]}, headers={"X-API-Key": "s3cret"}
    )

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 201


def test_health_stays_open_for_uptime_checks(secured_client):
    """Probes must not need the secret, or they cannot report on the service."""
    assert secured_client.get("/health").status_code == 200


def test_auth_is_disabled_when_no_key_is_configured(client):
    """Local development stays frictionless with APP_API_KEY unset."""
    assert client.post("/query", json={"question": "hi"}).status_code == 200
