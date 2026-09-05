"""Per-case scoring.

Deliberately no LLM judge. A judge would cost money on every eval run, add
non-determinism to a CI gate, and -- using the same model family that produced
the answer -- inherit that model's blind spots. These metrics are cheap,
deterministic, and measure the things this pipeline can actually get wrong:
retrieving from the wrong document, inventing an answer the corpus does not
support, or failing to refuse a question it cannot answer.

The honest limitation is that keyword matching cannot detect a fluent answer
that is subtly wrong in a way none of the keywords capture. It is a regression
gate, not a correctness proof.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.evaluation.dataset import GoldenCase

# The exact sentence the system prompt tells the model to use when the context
# does not answer the question. Matching the phrase rather than the whole
# sentence keeps this robust to trailing punctuation and added detail.
REFUSAL_MARKER = "i don't know based on the provided documents"

_CITATION = re.compile(r"\[\d+\]")


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    question: str
    answer: str
    # Did the expected source document appear among the retrieved chunks?
    retrieval_hit: bool
    # Did the answer contain at least one expected keyword?
    keyword_hit: bool
    # Did the model refuse (say "I don't know")?
    refused: bool
    # Did the model do the right thing overall for this case?
    correct: bool
    # Does the answer cite its sources as the prompt requires?
    cites_sources: bool
    latency_ms: float
    cost_usd: float
    retrieved_docs: list[str]
    failure_reason: str = ""


def _doc_ids(sources) -> list[str]:
    """Source doc ids, de-duplicated, order preserved."""
    seen: list[str] = []
    for source in sources:
        doc_id = getattr(source, "doc_id", "")
        if doc_id and doc_id not in seen:
            seen.append(doc_id)
    return seen


def _matches_expected_doc(expected: str, retrieved: list[str]) -> bool:
    """Exact match on doc id.

    This was substring matching, which is actively wrong once the corpus holds
    near-neighbour documents: expecting "cloud-run" would be satisfied by
    retrieving "cloud-run-scaling", silently crediting a hit for fetching the
    wrong document. Since discriminating between adjacent topics is exactly
    what a retrieval metric is for, that failure mode inflates the one number
    it is meant to police.

    `ingest_corpus` sets each doc id to its filename stem, so exact equality
    is what a golden case's `expected_doc` already means.
    """
    return expected in retrieved


def score_case(case: GoldenCase, result) -> CaseScore:
    """Score one query result against its golden expectation."""
    answer = (result.answer or "").strip()
    lowered = answer.lower()
    refused = REFUSAL_MARKER in lowered
    retrieved = _doc_ids(result.sources)

    retrieval_hit = (
        _matches_expected_doc(case.expected_doc, retrieved) if case.expected_doc else False
    )
    keyword_hit = (
        any(keyword.lower() in lowered for keyword in case.expected_keywords)
        if case.expected_keywords
        else True
    )
    # Only meaningful when an answer was actually generated.
    cites_sources = bool(_CITATION.search(answer)) if not refused else False

    failure_reason = ""
    if case.is_refusal_case:
        # The only correct behaviour is an explicit refusal. Answering anyway
        # is the worst failure this system has: a confident hallucination.
        correct = refused
        if not correct:
            failure_reason = "answered a question the corpus cannot support"
    else:
        correct = retrieval_hit and keyword_hit and not refused
        if refused:
            failure_reason = "refused a question the corpus does answer"
        elif not retrieval_hit:
            failure_reason = f"expected doc {case.expected_doc!r}, retrieved {retrieved}"
        elif not keyword_hit:
            failure_reason = f"answer missing all of {case.expected_keywords}"

    return CaseScore(
        case_id=case.id,
        question=case.question,
        answer=answer,
        retrieval_hit=retrieval_hit,
        keyword_hit=keyword_hit,
        refused=refused,
        correct=correct,
        cites_sources=cites_sources,
        latency_ms=float(getattr(result, "latency_ms", 0.0)),
        cost_usd=float(getattr(result, "cost_usd", 0.0)),
        retrieved_docs=retrieved,
        failure_reason=failure_reason,
    )


@dataclass(frozen=True)
class EvalSummary:
    total: int
    correct: int
    accuracy: float
    # Retrieval quality, measured only over cases that expect a document.
    retrieval_hit_rate: float
    # Refusal quality, measured only over out-of-corpus cases. This is the
    # metric that catches hallucination regressions.
    refusal_accuracy: float
    citation_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    scores: list[CaseScore]

    @property
    def failures(self) -> list[CaseScore]:
        return [score for score in self.scores if not score.correct]


def _rate(hits: int, total: int) -> float:
    # An empty denominator scores 1.0: a suite with no refusal cases has not
    # failed refusal, and a threshold should not fire on a metric with no data.
    return 1.0 if total == 0 else round(hits / total, 4)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return round(ordered[index], 2)


def summarise(cases: list[GoldenCase], scores: list[CaseScore]) -> EvalSummary:
    grounded_cases = {c.id for c in cases if not c.is_refusal_case}
    refusal_cases = {c.id for c in cases if c.is_refusal_case}

    retrieval_scored = [s for s in scores if s.case_id in grounded_cases]
    refusal_scored = [s for s in scores if s.case_id in refusal_cases]
    latencies = [s.latency_ms for s in scores]

    return EvalSummary(
        total=len(scores),
        correct=sum(1 for s in scores if s.correct),
        accuracy=_rate(sum(1 for s in scores if s.correct), len(scores)),
        retrieval_hit_rate=_rate(
            sum(1 for s in retrieval_scored if s.retrieval_hit), len(retrieval_scored)
        ),
        refusal_accuracy=_rate(
            sum(1 for s in refusal_scored if s.refused), len(refusal_scored)
        ),
        citation_rate=_rate(
            sum(1 for s in retrieval_scored if s.cites_sources), len(retrieval_scored)
        ),
        mean_latency_ms=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        p95_latency_ms=_percentile(latencies, 0.95),
        total_cost_usd=round(sum(s.cost_usd for s in scores), 8),
        scores=scores,
    )
