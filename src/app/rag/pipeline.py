"""The RAG pipeline: ingest -> embed -> store, and question -> retrieve -> answer.

Every LLM interaction that leaves this module has been measured (latency split
by stage, tokens, estimated cost) and handed to the tracker.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.llm.openai_client import ChatClient, EmbeddingClient
from app.persistence import SnapshotStore
from app.rag.chunking import Chunk, chunk_document, content_hash
from app.rag.vectorstore import ChromaVectorStore, RetrievedChunk
from app.tracking.cost import estimate_cost_usd, usd_to_inr
from app.tracking.mlflow_tracker import MLflowTracker, RunPayload
from app.tracking.spend_guard import SpendGuard

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about a private document "
    "collection. Use ONLY the numbered context passages provided. If the answer "
    "is not contained in them, reply exactly: \"I don't know based on the "
    'provided documents." Cite the passages you used as [1], [2] and so on. '
    "Be concise."
)

USER_PROMPT_TEMPLATE = """Context passages:
{context}

Question: {question}

Answer (cite passages as [n]):"""

NO_CONTEXT_ANSWER = (
    "I don't know based on the provided documents. "
    "Nothing has been ingested yet, or nothing matched this question."
)


@dataclass
class IngestResult:
    document_ids: list[str]
    chunk_count: int
    collection_size: int
    embedding_tokens: int
    cost_usd: float
    cost_inr: float
    latency_ms: float


@dataclass
class QueryResult:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    cost_usd: float = 0.0
    cost_inr: float = 0.0
    mlflow_run_id: str | None = None


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(chunks, start=1))


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        store: ChromaVectorStore,
        embedding_client: EmbeddingClient,
        chat_client: ChatClient,
        tracker: MLflowTracker,
        spend_guard: SpendGuard | None = None,
        snapshots: SnapshotStore | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embeddings = embedding_client
        self._chat = chat_client
        self._tracker = tracker
        # Default to a disabled guard so every existing caller (and every
        # test) keeps working without knowing budgets exist.
        self._spend = spend_guard or SpendGuard(budget_usd=0.0)
        # A disabled store by default, so nothing touches GCS unless asked.
        self._snapshots = snapshots or SnapshotStore(bucket="", object_name="", enabled=False)

    def collection_size(self) -> int:
        return self._store.count()

    def tracker_info(self) -> dict:
        return self._tracker.describe()

    def persistence_info(self) -> dict:
        return {"enabled": self._snapshots.enabled, "uri": self._snapshots.uri}

    def spend_info(self) -> dict:
        snap = self._spend.snapshot()
        return {
            "enabled": snap.enabled,
            "spent_usd": snap.spent_usd,
            "budget_usd": snap.budget_usd,
            "remaining_usd": snap.remaining_usd,
            "calls": snap.calls,
            "window_resets_in_seconds": snap.window_resets_in_seconds,
        }

    # -- ingest ------------------------------------------------------------
    def ingest(self, documents: list[tuple[str, str | None, dict[str, str]]]) -> IngestResult:
        """documents: list of (text, doc_id_or_None, metadata)."""
        started = time.perf_counter()
        settings = self._settings
        # Embedding a large upload is the single most expensive thing this
        # service does, so the ceiling is checked before any of it happens.
        self._spend.check()

        all_chunks: list[Chunk] = []
        doc_ids: list[str] = []
        for text, doc_id, metadata in documents:
            if len(text) > settings.max_ingest_chars:
                raise ValueError(
                    f"document exceeds max_ingest_chars ({settings.max_ingest_chars})"
                )
            resolved_id = doc_id or f"doc-{content_hash(text)}"
            chunks = chunk_document(
                text=text,
                doc_id=resolved_id,
                metadata=metadata,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            if not chunks:
                continue
            all_chunks.extend(chunks)
            doc_ids.append(resolved_id)

        embedding_tokens = 0
        if all_chunks:
            result = self._embeddings.embed([c.text for c in all_chunks])
            embedding_tokens = result.tokens
            self._store.add(all_chunks, result.vectors)

        cost_usd = estimate_cost_usd(settings.embedding_model, prompt_tokens=embedding_tokens)
        self._spend.record(cost_usd)
        latency_ms = (time.perf_counter() - started) * 1000

        outcome = IngestResult(
            document_ids=doc_ids,
            chunk_count=len(all_chunks),
            collection_size=self._store.count(),
            embedding_tokens=embedding_tokens,
            cost_usd=cost_usd,
            cost_inr=usd_to_inr(cost_usd, settings.usd_to_inr),
            latency_ms=round(latency_ms, 2),
        )

        logger.info(
            "ingest complete",
            extra={
                "documents": len(doc_ids),
                "chunks": outcome.chunk_count,
                "collection_size": outcome.collection_size,
                "embedding_tokens": embedding_tokens,
                "cost_usd": cost_usd,
                "latency_ms": outcome.latency_ms,
            },
        )

        # Persist only when something changed. Snapshotting an unchanged
        # store would burn a GCS write per no-op request.
        if all_chunks and self._settings.snapshot_on_ingest and self._snapshots.enabled:
            self._snapshots.save(self._settings.chroma_dir)

        self._tracker.log_run(
            run_name="ingest",
            payload=RunPayload(
                params={
                    "embedding_model": settings.embedding_model,
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                    "documents": len(doc_ids),
                },
                metrics={
                    "chunks": float(outcome.chunk_count),
                    "collection_size": float(outcome.collection_size),
                    "embedding_tokens": float(embedding_tokens),
                    "cost_usd": cost_usd,
                    "latency_ms": outcome.latency_ms,
                },
                tags={"endpoint": "/ingest", "stage": "ingest"},
            ),
        )
        return outcome

    # -- retrieval ---------------------------------------------------------
    def retrieve(
        self, question: str, top_k: int | None = None
    ) -> tuple[list[RetrievedChunk], int, float]:
        """Embed a question and fetch matching chunks. No LLM call.

        Public because retrieval is worth measuring on its own: a parameter
        sweep over chunk size, retrieval depth and similarity floor needs
        thousands of retrievals and none of the generations, which is the
        difference between a sweep costing a fraction of a cent and costing
        real money.

        Returns (hits, embedding_tokens, elapsed_ms).
        """
        started = time.perf_counter()
        k = top_k or self._settings.top_k
        embedded = self._embeddings.embed([question])
        query_vector = embedded.vectors[0] if embedded.vectors else []
        hits = (
            self._store.search(
                query_vector, top_k=k, min_similarity=self._settings.min_similarity
            )
            if query_vector
            else []
        )
        return hits, embedded.tokens, (time.perf_counter() - started) * 1000

    # -- query -------------------------------------------------------------
    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        started = time.perf_counter()
        settings = self._settings
        k = top_k or settings.top_k
        self._spend.check()

        # 1. Embed the question and retrieve.
        hits, embedding_tokens, retrieval_ms = self.retrieve(question, k)

        # 2. Short-circuit when nothing matched: skip the LLM call entirely.
        #    No context means no useful answer, so paying for tokens is waste.
        if not hits:
            outcome = QueryResult(
                answer=NO_CONTEXT_ANSWER,
                sources=[],
                model=settings.chat_model,
                embedding_tokens=embedding_tokens,
                retrieval_ms=round(retrieval_ms, 2),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                cost_usd=estimate_cost_usd(
                    settings.embedding_model, prompt_tokens=embedding_tokens
                ),
            )
            outcome.cost_inr = usd_to_inr(outcome.cost_usd, settings.usd_to_inr)
            self._spend.record(outcome.cost_usd)
            logger.info(
                "query returned no context",
                extra={"question_chars": len(question), "top_k": k},
            )
            outcome.mlflow_run_id = self._log_query_run(question, outcome, k, hit=False)
            return outcome

        # 3. Generate.
        context = build_context(hits)
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

        generation_started = time.perf_counter()
        completion = self._chat.complete(SYSTEM_PROMPT, user_prompt)
        generation_ms = (time.perf_counter() - generation_started) * 1000

        chat_cost = estimate_cost_usd(
            completion.model, completion.prompt_tokens, completion.completion_tokens
        )
        embed_cost = estimate_cost_usd(
            settings.embedding_model, prompt_tokens=embedding_tokens
        )
        total_cost = round(chat_cost + embed_cost, 8)
        self._spend.record(total_cost)

        outcome = QueryResult(
            answer=completion.text,
            sources=hits,
            model=completion.model,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            embedding_tokens=embedding_tokens,
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            cost_usd=total_cost,
            cost_inr=usd_to_inr(total_cost, settings.usd_to_inr),
        )

        logger.info(
            "query complete",
            extra={
                "model": outcome.model,
                "top_k": k,
                "sources": len(hits),
                "top_similarity": hits[0].similarity,
                "prompt_tokens": outcome.prompt_tokens,
                "completion_tokens": outcome.completion_tokens,
                "cost_usd": outcome.cost_usd,
                "latency_ms": outcome.latency_ms,
                "retrieval_ms": outcome.retrieval_ms,
                "generation_ms": outcome.generation_ms,
            },
        )

        outcome.mlflow_run_id = self._log_query_run(question, outcome, k, hit=True)
        return outcome

    def _log_query_run(
        self, question: str, outcome: QueryResult, top_k: int, hit: bool
    ) -> str | None:
        settings = self._settings
        payload = RunPayload(
            params={
                "chat_model": outcome.model or settings.chat_model,
                "embedding_model": settings.embedding_model,
                "top_k": top_k,
                "temperature": settings.temperature,
                "max_output_tokens": settings.max_output_tokens,
                "question": question,
            },
            metrics={
                "latency_ms": outcome.latency_ms,
                "retrieval_ms": outcome.retrieval_ms,
                "generation_ms": outcome.generation_ms,
                "prompt_tokens": float(outcome.prompt_tokens),
                "completion_tokens": float(outcome.completion_tokens),
                "embedding_tokens": float(outcome.embedding_tokens),
                "total_tokens": float(
                    outcome.prompt_tokens
                    + outcome.completion_tokens
                    + outcome.embedding_tokens
                ),
                "estimated_cost_usd": outcome.cost_usd,
                "estimated_cost_inr": outcome.cost_inr,
                "retrieved_chunks": float(len(outcome.sources)),
                "top_similarity": outcome.sources[0].similarity if outcome.sources else 0.0,
            },
            tags={
                "endpoint": "/query",
                "stage": "query",
                "grounded": str(hit).lower(),
            },
            artifacts={
                "question.txt": question,
                "answer.txt": outcome.answer,
                "context.txt": build_context(outcome.sources) or "(no context retrieved)",
            },
        )
        return self._tracker.log_run(run_name="query", payload=payload)
