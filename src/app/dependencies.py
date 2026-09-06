"""Wiring: build the pipeline once at startup, expose it to routes.

Kept separate from main.py so tests can override `get_pipeline` without
importing anything that touches the network.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from functools import lru_cache

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import Settings, get_settings
from app.llm.openai_client import (
    OpenAIChatClient,
    OpenAIEmbeddingClient,
    build_openai_client,
)
from app.persistence import SnapshotStore
from app.rag.pipeline import RAGPipeline
from app.rag.vectorstore import ChromaVectorStore
from app.tracking.mlflow_tracker import MLflowTracker
from app.tracking.spend_guard import SpendGuard

logger = logging.getLogger(__name__)


@lru_cache
def shard_id() -> str:
    """A name for this process's snapshot object, stable for its lifetime.

    Random rather than derived from anything Cloud Run exposes: the unit that
    must not collide is the writing *process*, and no revision or service name
    is unique per instance. The revision is prefixed anyway, so a shard can be
    attributed to the deploy that produced it when reading them back.
    """
    revision = re.sub(r"[^a-zA-Z0-9_-]", "", os.getenv("K_REVISION", "local"))[:40]
    return f"{revision or 'local'}-{uuid.uuid4().hex[:8]}"


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Construct the real, network-backed pipeline. Called once on startup."""
    snapshots = SnapshotStore(
        bucket=settings.gcs_bucket,
        object_name=settings.chroma_snapshot_object,
        enabled=settings.persistence_enabled,
    )
    # Restore BEFORE Chroma opens the directory: PersistentClient reads its
    # SQLite file and HNSW index at construction, so a restore afterwards
    # would be invisible until the next cold start.
    if snapshots.enabled:
        snapshots.restore(settings.chroma_dir)

    store = ChromaVectorStore(
        persist_dir=settings.chroma_dir, collection_name=settings.chroma_collection
    )
    openai_client = build_openai_client(settings)
    return RAGPipeline(
        settings=settings,
        store=store,
        embedding_client=OpenAIEmbeddingClient(openai_client, settings),
        chat_client=OpenAIChatClient(openai_client, settings),
        tracker=MLflowTracker(
            settings,
            # One object per process, under its own prefix. Two instances
            # writing a shared object would overwrite each other's runs
            # wholesale, and this is written far more often than the Chroma
            # snapshot that pattern was chosen for.
            snapshots=SnapshotStore(
                bucket=settings.gcs_bucket,
                object_name=f"{settings.mlflow_snapshot_prefix}{shard_id()}.tar.gz",
                enabled=settings.persistence_enabled,
            ),
        ),
        spend_guard=SpendGuard(budget_usd=settings.daily_budget_usd),
        snapshots=snapshots,
    )


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        # Startup failed (almost always a missing OPENAI_API_KEY). Fail with a
        # clear 503 rather than an opaque AttributeError.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "RAG pipeline is not initialised. Check that OPENAI_API_KEY is set; "
                "see startup logs for details."
            ),
        )
    return pipeline


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """Optional shared-secret guard for the endpoints that spend money.

    Disabled when APP_API_KEY is empty, so local development stays frictionless.
    """
    if not settings.auth_enabled:
        return
    if x_api_key != settings.app_api_key:
        logger.warning("rejected request with invalid api key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key"
        )
