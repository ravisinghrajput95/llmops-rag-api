"""Chroma-backed vector store.

Embedded mode only: Chroma runs in-process and writes to a local directory.
No server, no managed vector database, no bill.

IMPORTANT on Cloud Run: the persist directory lives on the instance's
filesystem, which is an in-memory tmpfs scoped to a single container instance.
Data written by one instance is invisible to the next and is lost on scale-to-
zero. That is fine for a demo (ingest, then query, on a warm instance) and is
called out in the README. Durable storage would mean Cloud SQL / a hosted
vector DB, both of which cost money.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Must be set before chromadb is imported: its telemetry client is built at
# import time, and in 0.6.x the posthog shim logs a spurious error on every
# start. We want no telemetry and no noise in Cloud Logging either way.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings

from app.rag.chunking import Chunk

logger = logging.getLogger(__name__)

# chroma 0.6.x ships a posthog shim whose signature no longer matches, so it
# logs "Failed to send telemetry event" on every call even with telemetry
# disabled. Nothing actionable, and it would pollute Cloud Logging.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    similarity: float
    metadata: dict[str, str]


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "documents",
    ) -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # Cosine is the right metric for OpenAI embeddings and gives us a
            # bounded distance in [0, 2] that maps cleanly onto a similarity.
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_name = collection_name
        logger.info(
            "vector store ready",
            extra={"persist_dir": persist_dir, "collection": collection_name},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"embedding count ({len(embeddings)}) != chunk count ({len(chunks)})"
            )
        # upsert (not add) so re-ingesting the same document updates it in
        # place instead of raising on duplicate ids.
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)

    def search(
        self,
        embedding: list[float],
        top_k: int = 4,
        min_similarity: float = 0.0,
        min_similarity_ratio: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Nearest chunks, filtered by an absolute floor and a relative one.

        The two filters answer different questions. `min_similarity` asks
        "does this query match the corpus at all?", which is the same question
        for every query and so is fairly asked in absolute terms. The ratio
        asks "given what this query matched, is this chunk one of the good
        ones?" -- and that cannot be absolute, because similarity is not
        calibrated across queries: 0.45 is a weak result for one question and
        the best available for another. Rescaling by the top hit is what makes
        the second question answerable at all.
        """
        size = self.count()
        if size == 0:
            return []

        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, size),
            include=["documents", "metadatas", "distances"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        # Chroma returns nearest-first, so the first distance is the best hit
        # and sets the bar the rest are measured against.
        threshold = min_similarity
        if min_similarity_ratio > 0 and distances:
            threshold = max(threshold, min_similarity_ratio * (1.0 - float(distances[0])))

        hits: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            metadata = {str(k): str(v) for k, v in (metadata or {}).items()}
            similarity = 1.0 - float(distance)
            if similarity < threshold:
                continue
            hits.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    doc_id=metadata.get("doc_id", ""),
                    text=text or "",
                    similarity=round(similarity, 4),
                    metadata=metadata,
                )
            )
        return hits

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop every vector. Used by tests and the /admin reset path."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
