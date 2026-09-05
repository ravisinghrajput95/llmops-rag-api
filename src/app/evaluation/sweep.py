"""Parameter sweeps over retrieval configuration, logged to MLflow.

Until this existed, MLflow was being used as a logger rather than an
experiment tracker: every call was recorded, but nothing ever compared two
configurations to decide which was better. Choosing chunk_size=800 and top_k=4
was a guess dressed up as a default.

**Why this is nearly free.** A sweep varies retrieval parameters -- chunk size,
overlap, retrieval depth, similarity floor -- and every one of those can be
scored without generating a single answer. Embedding a 22-document corpus costs
around $0.0001; the generation that would follow costs 50x that. So the whole
grid runs for a fraction of a cent, and the expensive step is deliberately
excluded because it would not inform the choice being made.

What this cannot tell you: whether a configuration produces *better answers*.
Retrieval hit rate says the right passage was fetched, not that the model used
it well. Use the best few configurations from a sweep as candidates, then run
the full `make eval` on those.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.evaluation.dataset import GoldenCase
from app.evaluation.metrics import _matches_expected_doc
from app.tracking.cost import estimate_cost_usd
from app.tracking.mlflow_tracker import RunPayload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepPoint:
    """One configuration in the grid."""

    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_similarity: float

    def as_params(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
        }

    def label(self) -> str:
        return (
            f"cs={self.chunk_size} ov={self.chunk_overlap} "
            f"k={self.top_k} floor={self.min_similarity}"
        )


@dataclass
class SweepResult:
    point: SweepPoint
    chunks: int
    retrieval_hit_rate: float
    # Fraction of out-of-corpus questions that retrieved nothing. The floor is
    # what should reject these, so this measures the floor directly rather
    # than measuring the model's willingness to refuse.
    empty_on_out_of_corpus: float
    # Fraction of grounded cases whose retrieved context actually contains a
    # fact the answer needs. This exists because doc-level hit rate does not:
    # a configuration can retrieve the right *document* via some other chunk
    # while the passage holding the answer never reaches the prompt. Measured
    # by keyword presence in the retrieved text -- crude, free, and unlike hit
    # rate it tracks whether generation can succeed at all.
    answerable_rate: float
    mean_retrieval_ms: float
    # Chunks fed to the prompt per grounded question. This is the cost driver:
    # every retrieved chunk becomes input tokens on every single query.
    mean_chunks_returned: float
    embedding_tokens: int
    embedding_cost_usd: float
    failures: list[str] = field(default_factory=list)

    @property
    def score(self) -> tuple[float, float, float, float]:
        """Ranking key: answerable first, then hit rate, floor, fewer chunks.

        answerable_rate leads deliberately. Ranking on doc-level hit rate alone
        picked a configuration that scored a perfect 1.000 on it and then lost
        7 points of end-to-end accuracy to false refusals, because the passage
        carrying each answer had been filtered out while its document stayed
        represented. Hit rate could not see that; this ordering can.

        Fewer chunks breaks the final tie: two configurations that retrieve the
        answer equally often are not equally good if one sends more context to
        the model on every query for the rest of its life.
        """
        return (
            self.answerable_rate,
            self.retrieval_hit_rate,
            self.empty_on_out_of_corpus,
            -self.mean_chunks_returned,
        )


def build_grid(
    chunk_sizes: list[int],
    overlaps: list[int],
    top_ks: list[int],
    floors: list[float],
) -> list[SweepPoint]:
    """Cartesian product, dropping combinations that make no sense.

    An overlap at or above the chunk size would repeat an entire chunk into the
    next one, which is not a configuration anybody wants to evaluate.
    """
    grid = []
    for size, overlap, k, floor in itertools.product(chunk_sizes, overlaps, top_ks, floors):
        if overlap >= size:
            continue
        grid.append(SweepPoint(size, overlap, k, floor))
    return grid


def evaluate_point(
    point: SweepPoint,
    cases: list[GoldenCase],
    build_pipeline_for,
    corpus_dir: str,
    embedding_model: str = "text-embedding-3-small",
) -> SweepResult:
    """Score one configuration. `build_pipeline_for(settings)` supplies a
    pipeline whose vector store is empty and whose settings match the point."""
    from app.evaluation.runner import ingest_corpus

    pipeline = build_pipeline_for(point)
    ingested = ingest_corpus(pipeline, corpus_dir)

    grounded = [c for c in cases if not c.is_refusal_case]
    refusals = [c for c in cases if c.is_refusal_case]

    hits = 0
    answerable = 0
    failures: list[str] = []
    returned: list[int] = []
    latencies: list[float] = []
    question_tokens = 0

    for case in grounded:
        found, tokens, elapsed = pipeline.retrieve(case.question, point.top_k)
        question_tokens += tokens
        latencies.append(elapsed)
        returned.append(len(found))
        docs = list(dict.fromkeys(c.doc_id for c in found))
        if case.expected_doc and _matches_expected_doc(case.expected_doc, docs):
            hits += 1
        else:
            failures.append(f"{case.id}: expected {case.expected_doc}, got {docs}")

        # Did the answer's supporting text actually reach the prompt?
        context = " ".join(c.text for c in found).lower()
        if not case.expected_keywords or any(
            kw.lower() in context for kw in case.expected_keywords
        ):
            answerable += 1
        else:
            failures.append(f"{case.id}: doc retrieved but no supporting passage in context")

    empty = 0
    for case in refusals:
        found, tokens, elapsed = pipeline.retrieve(case.question, point.top_k)
        question_tokens += tokens
        latencies.append(elapsed)
        if not found:
            empty += 1

    # Corpus embedding dominates -- it is ~40x the question tokens -- so
    # counting only the questions would understate the sweep's cost badly.
    total_tokens = question_tokens + ingested.embedding_tokens
    return SweepResult(
        point=point,
        chunks=ingested.chunk_count,
        retrieval_hit_rate=round(hits / len(grounded), 4) if grounded else 1.0,
        answerable_rate=round(answerable / len(grounded), 4) if grounded else 1.0,
        empty_on_out_of_corpus=round(empty / len(refusals), 4) if refusals else 1.0,
        mean_retrieval_ms=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        mean_chunks_returned=round(sum(returned) / len(returned), 2) if returned else 0.0,
        embedding_tokens=total_tokens,
        embedding_cost_usd=estimate_cost_usd(embedding_model, prompt_tokens=total_tokens),
        failures=failures,
    )


def log_sweep_point(tracker, result: SweepResult, rank: int | None = None) -> None:
    """One MLflow run per configuration, so the grid is comparable in the UI."""
    if tracker is None:
        return
    tracker.log_run(
        run_name=f"sweep-{result.point.label()}",
        payload=RunPayload(
            params=result.point.as_params() | {"chunks": result.chunks},
            metrics={
                "retrieval_hit_rate": result.retrieval_hit_rate,
                "answerable_rate": result.answerable_rate,
                "empty_on_out_of_corpus": result.empty_on_out_of_corpus,
                "mean_retrieval_ms": result.mean_retrieval_ms,
                "mean_chunks_returned": result.mean_chunks_returned,
                "embedding_tokens": float(result.embedding_tokens),
                "embedding_cost_usd": result.embedding_cost_usd,
                "rank": float(rank) if rank is not None else 0.0,
            },
            tags={"stage": "sweep"},
            artifacts={"failures.txt": "\n".join(result.failures) or "(none)"},
        ),
    )


def run_sweep(
    grid: list[SweepPoint],
    cases: list[GoldenCase],
    build_pipeline_for,
    corpus_dir: str,
    tracker=None,
    embedding_model: str = "text-embedding-3-small",
) -> list[SweepResult]:
    """Evaluate every point, best first."""
    results: list[SweepResult] = []
    for index, point in enumerate(grid, start=1):
        started = time.perf_counter()
        result = evaluate_point(point, cases, build_pipeline_for, corpus_dir, embedding_model)
        results.append(result)
        logger.info(
            "sweep point complete",
            extra={
                "point": point.label(),
                "progress": f"{index}/{len(grid)}",
                "hit_rate": result.retrieval_hit_rate,
                "seconds": round(time.perf_counter() - started, 1),
            },
        )

    results.sort(key=lambda r: r.score, reverse=True)
    for rank, result in enumerate(results, start=1):
        log_sweep_point(tracker, result, rank)
    return results


def format_sweep(results: list[SweepResult], baseline: Settings | None = None) -> str:
    """A comparison table. The point of a sweep is the comparison, not the run."""
    lines = [
        "",
        "Retrieval parameter sweep",
        "=" * 104,
        f"{'rank':<5}{'chunk':<7}{'ovlp':<6}{'k':<4}{'floor':<7}"
        f"{'chunks':<8}{'answerable':<12}{'hit rate':<10}{'floor ok':<10}"
        f"{'ctx/query':<11}{'ms':<7}",
        "-" * 104,
    ]
    for rank, r in enumerate(results, start=1):
        p = r.point
        marker = " *" if rank == 1 else "  "
        lines.append(
            f"{rank:<5}{p.chunk_size:<7}{p.chunk_overlap:<6}{p.top_k:<4}"
            f"{p.min_similarity:<7}{r.chunks:<8}{r.answerable_rate:<12.1%}"
            f"{r.retrieval_hit_rate:<10.1%}"
            f"{r.empty_on_out_of_corpus:<10.0%}{r.mean_chunks_returned:<11.2f}"
            f"{r.mean_retrieval_ms:<7.1f}{marker}"
        )
    lines.append("-" * 104)

    total = sum(r.embedding_cost_usd for r in results)
    lines.append(f"{len(results)} configurations, total embedding cost ${total:.6f}")

    if results:
        best = results[0]
        lines.append("")
        lines.append(f"Best: {best.point.label()}")
        lines.append(
            f"  answerable {best.answerable_rate:.1%} | "
            f"hit rate {best.retrieval_hit_rate:.1%} | "
            f"out-of-corpus rejected {best.empty_on_out_of_corpus:.0%} | "
            f"{best.mean_chunks_returned:.2f} chunks per query"
        )
        if baseline is not None:
            current = SweepPoint(
                baseline.chunk_size,
                baseline.chunk_overlap,
                baseline.top_k,
                baseline.min_similarity,
            )
            if current == best.point:
                lines.append("  This is the configuration already in use.")
            else:
                lines.append(f"  Currently configured: {current.label()}")
        if best.failures:
            lines.append(f"  Still failing ({len(best.failures)}):")
            lines.extend(f"    {f}" for f in best.failures[:8])
    lines.append("")
    return "\n".join(lines)
