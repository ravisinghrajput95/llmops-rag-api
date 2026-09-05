"""End-to-end coverage for the two 429 paths.

The suite disables both limiters globally (see conftest), because they hold
process-wide state that would otherwise make every test depend on how many
requests ran before it. These tests re-enable them deliberately and put them
back afterwards.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.rate_limit import RateLimiter
from app.tracking.spend_guard import BudgetExceededError, SpendGuard


@pytest.fixture
def throttled_client(client: TestClient):
    """Swap in a limiter that allows exactly two metered requests."""
    original = main_module._rate_limiter
    main_module._rate_limiter = RateLimiter(requests_per_minute=2)
    try:
        yield client
    finally:
        main_module._rate_limiter = original


def test_rate_limit_returns_429_with_retry_after(throttled_client: TestClient) -> None:
    payloads = {"question": "what is in the docs?"}
    assert throttled_client.post("/query", json=payloads).status_code == 200
    assert throttled_client.post("/query", json=payloads).status_code == 200

    response = throttled_client.post("/query", json=payloads)
    assert response.status_code == 429
    assert response.json()["detail"] == "Rate limit exceeded."
    assert response.json()["retry_after_seconds"] > 0
    # Clients need this header to back off correctly.
    assert int(response.headers["Retry-After"]) >= 1


def test_health_is_never_rate_limited(throttled_client: TestClient) -> None:
    """Startup probes and uptime checks must not be throttled -- a 429 on
    /health would make Cloud Run consider the revision unhealthy."""
    for _ in range(10):
        assert throttled_client.get("/health").status_code == 200


def test_budget_exhaustion_returns_429(client: TestClient, pipeline) -> None:
    """A drained ceiling surfaces as an actionable 429, not a 500."""
    pipeline._spend = SpendGuard(budget_usd=0.001)
    pipeline._spend.record(0.001)

    response = client.post("/query", json={"question": "anything at all"})

    assert response.status_code == 429
    body = response.json()
    assert "Daily spend ceiling reached" in body["detail"]
    assert body["budget_usd"] == 0.001
    assert body["resets_in_seconds"] > 0


def test_budget_blocks_ingest_before_embedding(client: TestClient, pipeline) -> None:
    """The check must happen before the embedding call, not after -- embedding
    a large upload is the most expensive thing the service does."""
    pipeline._spend = SpendGuard(budget_usd=0.001)
    pipeline._spend.record(0.001)

    response = client.post("/ingest", json={"documents": [{"text": "some text here"}]})

    assert response.status_code == 429


def test_ready_reports_remaining_budget(client: TestClient, pipeline) -> None:
    pipeline._spend = SpendGuard(budget_usd=1.0)
    pipeline._spend.record(0.25)

    spend = client.get("/ready").json()["spend"]

    assert spend["enabled"] is True
    assert spend["spent_usd"] == pytest.approx(0.25)
    assert spend["remaining_usd"] == pytest.approx(0.75)


def test_query_succeeds_while_budget_remains(client: TestClient, pipeline) -> None:
    pipeline._spend = SpendGuard(budget_usd=10.0)
    assert client.post("/query", json={"question": "what is here?"}).status_code == 200
    # The successful call must have been recorded against the ceiling.
    assert pipeline._spend.snapshot().calls >= 1


def test_budget_error_message_is_informative() -> None:
    error = BudgetExceededError(spent_usd=0.5, budget_usd=0.25, resets_in_seconds=100)
    assert "0.2500" in str(error)
    assert "100s" in str(error)
