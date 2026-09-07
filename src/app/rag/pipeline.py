"""The RAG pipeline: ingest -> embed -> store, and question -> retrieve -> answer.

Every LLM interaction that leaves this module has been measured (latency split
by stage, tokens, estimated cost) and handed to the tracker.
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.llm.openai_client import ChatClient, EmbeddingClient
from app.persistence import SnapshotStore
from app.rag.chunking import Chunk, chunk_document, content_hash
from app.rag.prompts import PROMPTS, REFUSAL_MARKER, PromptSet
from app.rag.vectorstore import ChromaVectorStore, RetrievedChunk
from app.tracking.cost import estimate_cost_usd, usd_to_inr
from app.tracking.mlflow_tracker import MLflowTracker, RunPayload
from app.tracking.spend_guard import SpendGuard

logger = logging.getLogger(__name__)

# How many times an ingest will rebase its chunks onto a newer snapshot before
# giving up. Contention needs two instances ingesting in the same few seconds,
# which at max-instances=2 and a demo's ingest rate is already unlikely; a
# third round is deep in the tail.
SNAPSHOT_MERGE_ATTEMPTS = 3


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
    # Whether the answer declined to answer -- either because retrieval found
    # nothing, or because the model read the context and judged it insufficient.
    # Recorded because on live traffic it is the only quality signal available:
    # there are no labels, and the eval measured the model refusing 66 of 67
    # out-of-corpus questions, which makes this a calibrated estimator of how
    # much traffic the corpus cannot answer. See monitoring/drift.py.
    refused: bool = False
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
        prompts: PromptSet | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embeddings = embedding_client
        self._chat = chat_client
        self._tracker = tracker
        # Injectable so that a future eval can run two prompt versions against
        # the same corpus and compare them; the default is what ships.
        self._prompts = prompts or PROMPTS
        # Default to a disabled guard so every existing caller (and every
        # test) keeps working without knowing budgets exist.
        self._spend = spend_guard or SpendGuard(budget_usd=0.0)
        # A disabled store by default, so nothing touches GCS unless asked.
        self._snapshots = snapshots or SnapshotStore(bucket="", object_name="", enabled=False)
        # The generation this instance last read or wrote. 0 means "we have
        # seen no snapshot", which as a precondition asks GCS to create the
        # object only if it does not exist -- so the very first two instances
        # to ingest race safely too.
        self._snapshot_generation = 0

    def collection_size(self) -> int:
        return self._store.count()

    def tracker_info(self) -> dict:
        return self._tracker.describe()

    def persistence_info(self) -> dict:
        return {"enabled": self._snapshots.enabled, "uri": self._snapshots.uri}

    def with_prompts(self, prompts: PromptSet) -> RAGPipeline:
        """A pipeline identical to this one but answering with other prompts.

        Shares the vector store and the clients deliberately: comparing two
        prompt versions is only meaningful if everything else -- the corpus,
        the chunking, the retrieval config -- is held fixed. Re-ingesting for
        the second arm would also pay the embedding cost twice for a variable
        that does not affect retrieval at all.
        """
        return RAGPipeline(
            settings=self._settings,
            store=self._store,
            embedding_client=self._embeddings,
            chat_client=self._chat,
            tracker=self._tracker,
            spend_guard=self._spend,
            snapshots=self._snapshots,
            prompts=prompts,
        )

    def flush_tracking(self) -> None:
        """Persist tracking state before the instance goes away."""
        self._tracker.flush()

    def prompt_info(self) -> dict:
        return self._prompts.describe()

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
            self._save_snapshot(all_chunks, result.vectors)

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

    def adopt_snapshot_generation(self, generation: int) -> None:
        """Record the generation restored at startup, so the first ingest from
        this instance writes against what it actually read."""
        self._snapshot_generation = generation

    def _save_snapshot(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Upload the store, rebasing onto a concurrent writer if there is one.

        A single shared object with last-write-wins silently discarded the
        other instance's documents, and documents are the one thing in that
        bucket nobody can regenerate. This uploads conditionally instead: if
        another instance wrote since we last synced, GCS refuses, and we replay
        *our* chunks on top of *their* snapshot rather than over it.

        Replay is safe because the store upserts on stable chunk ids, so a
        document ingested by both instances converges instead of duplicating.

        The honest limit: this makes the *snapshot* complete, not this
        instance's memory. Our local store still lacks the other instance's
        documents until a cold start restores the merged file. Fixing that
        would mean rebuilding the live Chroma client mid-request, which is a
        much larger risk than the staleness it removes.
        """
        settings = self._settings
        outcome = self._snapshots.save(
            settings.chroma_dir, expected_generation=self._snapshot_generation
        )

        attempts = 0
        while outcome.conflict and attempts < SNAPSHOT_MERGE_ATTEMPTS:
            attempts += 1
            with tempfile.TemporaryDirectory() as workspace:
                theirs = self._snapshots.restore(workspace)
                if not theirs.ok:
                    break
                merged = ChromaVectorStore(workspace, settings.chroma_collection)
                merged.add(chunks, vectors)
                outcome = self._snapshots.save(
                    workspace, expected_generation=theirs.generation
                )

        if outcome.ok:
            self._snapshot_generation = outcome.generation
            if attempts:
                logger.info(
                    "snapshot merged with a concurrent ingest",
                    extra={"attempts": attempts, "chunks_replayed": len(chunks)},
                )
        elif outcome.conflict:
            # Never silently: losing an ingest is the failure this exists to
            # prevent, so a give-up is loud even though it does not fail the
            # request.
            logger.warning(
                "snapshot still contended after retries; this ingest is not durable",
                extra={"attempts": attempts, "chunks": len(chunks)},
            )

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
                query_vector,
                top_k=k,
                min_similarity=self._settings.min_similarity,
                min_similarity_ratio=self._settings.min_similarity_ratio,
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
                answer=self._prompts["no_context_answer"].text,
                sources=[],
                refused=True,
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
        user_prompt = self._prompts.render("answer_user", context=context, question=question)

        generation_started = time.perf_counter()
        completion = self._chat.complete(self._prompts["answer_system"].text, user_prompt)
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
            refused=REFUSAL_MARKER in (completion.text or "").lower(),
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
                "refused": outcome.refused,
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
                # The fingerprint, not the text: this is what makes a prompt
                # change visible when two runs are compared months apart.
                "prompt_version": self._prompts.tracked_version,
                "prompt_fingerprint": self._prompts.fingerprint,
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
                "refused": float(outcome.refused),
            },
            tags={
                "endpoint": "/query",
                "stage": "query",
                "grounded": str(hit).lower(),
                "refused": str(outcome.refused).lower(),
            },
            artifacts={
                "question.txt": question,
                "answer.txt": outcome.answer,
                "context.txt": build_context(outcome.sources) or "(no context retrieved)",
            },
        )
        return self._tracker.log_run(run_name="query", payload=payload)
