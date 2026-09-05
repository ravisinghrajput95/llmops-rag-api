"""Test fixtures.

Two rules this suite enforces by construction:
  1. No network calls. The OpenAI chat and embedding clients are replaced with
     deterministic fakes, so `pytest` costs nothing and works offline/in CI.
  2. No shared state. Each test gets its own Chroma directory and MLflow DB.

The fake embedder is a hashed bag-of-words, not random noise, so documents that
share vocabulary genuinely score higher -- retrieval assertions are meaningful
rather than tautological.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from pathlib import Path

import pytest

# Settings are read (and cached) at import time, so the environment has to be
# in place before `app.*` is imported anywhere.
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="llmops-tests-"))
os.environ.update(
    {
        "OPENAI_API_KEY": "test-key-not-real",
        "CHROMA_DIR": str(_TMP_ROOT / "chroma"),
        "MLFLOW_ENABLED": "false",
        "MLFLOW_TRACKING_URI": f"sqlite:///{_TMP_ROOT / 'mlflow.db'}",
        "APP_API_KEY": "",
        "LOG_LEVEL": "WARNING",
        "GCP_PROJECT_ID": "test-project",
        # The spend ceiling and rate limiter hold process-wide state, so left
        # on they would couple every test to how many requests ran before it.
        # Both are covered directly in test_spend_guard.py, and exercised
        # end-to-end in test_limits.py against purpose-built instances.
        "DAILY_BUDGET_USD": "0",
        "RATE_LIMIT_PER_MINUTE": "0",
    }
)

from app.config import Settings, get_settings  # noqa: E402
from app.dependencies import get_pipeline  # noqa: E402
from app.llm.openai_client import ChatResult, EmbeddingResult  # noqa: E402
from app.rag.pipeline import RAGPipeline  # noqa: E402
from app.rag.vectorstore import ChromaVectorStore  # noqa: E402
from app.tracking.mlflow_tracker import MLflowTracker  # noqa: E402

_WORD = re.compile(r"[a-z0-9]+")
EMBED_DIM = 64


def _fake_vector(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic hashed bag-of-words embedding, L2-normalised."""
    vector = [0.0] * dim
    for word in _WORD.findall(text.lower()):
        index = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        # Chroma rejects all-zero vectors under cosine distance.
        return [1.0] + [0.0] * (dim - 1)
    return [value / norm for value in vector]


class FakeEmbeddingClient:
    """Stands in for OpenAI embeddings; records every call for assertions."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> EmbeddingResult:
        self.calls.append(list(texts))
        if not texts:
            return EmbeddingResult(vectors=[], model=self.model, tokens=0)
        return EmbeddingResult(
            vectors=[_fake_vector(text) for text in texts],
            model=self.model,
            # Rough stand-in for tokenisation: ~4 chars per token.
            tokens=sum(max(1, len(text) // 4) for text in texts),
        )


class FakeChatClient:
    """Stands in for OpenAI chat completions."""

    def __init__(self, answer: str = "Cloud Run scales to zero. [1]") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []
        self.error: Exception | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> ChatResult:
        self.calls.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        return ChatResult(
            text=self.answer,
            model="gpt-4o-mini",
            prompt_tokens=max(1, len(user_prompt) // 4),
            completion_tokens=max(1, len(self.answer) // 4),
        )

    @property
    def last_user_prompt(self) -> str:
        return self.calls[-1][1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key="test-key-not-real",
        chroma_dir=str(tmp_path / "chroma"),
        chroma_collection="test-documents",
        mlflow_enabled=False,
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        chunk_size=400,
        chunk_overlap=60,
        top_k=3,
        # Pinned off so retrieval tests are unaffected by the production
        # similarity floor, which is calibrated for real OpenAI embeddings
        # rather than this suite's hashed bag-of-words fake.
        min_similarity=0.0,
        usd_to_inr=88.0,
    )


@pytest.fixture
def store(settings: Settings) -> ChromaVectorStore:
    return ChromaVectorStore(settings.chroma_dir, settings.chroma_collection)


@pytest.fixture
def embedding_client() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
def chat_client() -> FakeChatClient:
    return FakeChatClient()


@pytest.fixture
def tracker(settings: Settings) -> MLflowTracker:
    return MLflowTracker(settings)


@pytest.fixture
def pipeline(
    settings: Settings,
    store: ChromaVectorStore,
    embedding_client: FakeEmbeddingClient,
    chat_client: FakeChatClient,
    tracker: MLflowTracker,
) -> RAGPipeline:
    return RAGPipeline(
        settings=settings,
        store=store,
        embedding_client=embedding_client,
        chat_client=chat_client,
        tracker=tracker,
    )


@pytest.fixture
def client(pipeline: RAGPipeline):
    """TestClient with the network-backed pipeline swapped for the fake one."""
    from fastapi.testclient import TestClient

    from app.main import app

    app.dependency_overrides[get_pipeline] = lambda: pipeline
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    get_settings.cache_clear()


SAMPLE_DOCS = [
    {
        "doc_id": "cloudrun",
        "text": (
            "Cloud Run is a serverless container platform on Google Cloud. "
            "It scales to zero when idle, so an unused service costs nothing. "
            "The free tier includes two million requests per month."
        ),
        "metadata": {"source": "gcp-docs", "topic": "compute"},
    },
    {
        "doc_id": "chroma",
        "text": (
            "Chroma is an embedded vector database written in Python. "
            "It persists collections to a local directory and requires no server, "
            "which keeps a retrieval stack free of managed database charges."
        ),
        "metadata": {"source": "chroma-docs", "topic": "vectors"},
    },
]
