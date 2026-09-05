"""Unit tests for the eval harness, plus the regression gate itself.

The gate at the bottom runs the real golden set against the real retrieval
path, using the deterministic fake embedder from conftest. That makes it free
and offline, so it can run on every push -- and it genuinely catches the
regressions that matter here: a chunking change that breaks retrieval, or a
similarity floor that starts refusing answerable questions.

What it cannot catch is generation quality, because the fake chat client does
not generate. `scripts/run_eval.py` covers that against real OpenAI, and costs
about $0.002 a run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.dataset import GoldenCase, load_cases
from app.evaluation.metrics import REFUSAL_MARKER, score_case, summarise
from app.evaluation.runner import Thresholds, ingest_corpus, load_golden_set, run_evaluation

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "evals" / "golden.jsonl"
CORPUS_DIR = Path(__file__).resolve().parent.parent / "evals" / "corpus"


class FakeSource:
    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id


class FakeResult:
    def __init__(self, answer: str, docs: list[str], latency_ms=10.0, cost_usd=0.0001):
        self.answer = answer
        self.sources = [FakeSource(d) for d in docs]
        self.latency_ms = latency_ms
        self.cost_usd = cost_usd


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class TestDataset:
    def test_golden_set_loads(self) -> None:
        cases = load_cases(GOLDEN_PATH)
        assert len(cases) >= 10
        assert all(case.question for case in cases)

    def test_golden_set_has_refusal_cases(self) -> None:
        """Without these the suite cannot detect hallucination regressions."""
        cases = load_cases(GOLDEN_PATH)
        assert sum(1 for c in cases if c.is_refusal_case) >= 3

    def test_duplicate_ids_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "dupes.jsonl"
        path.write_text('{"id": "a", "question": "q1"}\n{"id": "a", "question": "q2"}\n')
        with pytest.raises(ValueError, match="duplicate case id"):
            load_cases(path)

    def test_blank_lines_and_comments_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "commented.jsonl"
        path.write_text('# a comment\n\n{"id": "a", "question": "q"}\n')
        assert len(load_cases(path)) == 1

    def test_missing_question_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "a", "question": "  "}\n')
        with pytest.raises(ValueError, match="has no question"):
            load_cases(path)

    def test_invalid_json_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.jsonl"
        path.write_text('{"id": "a", "question": "q"}\nnot json at all\n')
        with pytest.raises(ValueError, match="invalid JSON"):
            load_cases(path)

    def test_empty_golden_set_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("\n# nothing here\n")
        with pytest.raises(ValueError, match="golden set is empty"):
            load_cases(path)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
class TestScoring:
    def test_grounded_case_passes_when_doc_and_keyword_match(self) -> None:
        case = GoldenCase(
            id="c", question="q", expected_doc="storage", expected_keywords=["tmpfs"]
        )
        score = score_case(case, FakeResult("It is a tmpfs [1].", ["storage"]))
        assert score.correct is True
        assert score.retrieval_hit and score.keyword_hit and score.cites_sources

    def test_wrong_document_fails_and_says_so(self) -> None:
        case = GoldenCase(
            id="c", question="q", expected_doc="storage", expected_keywords=["tmpfs"]
        )
        score = score_case(case, FakeResult("A tmpfs [1].", ["mlflow"]))
        assert score.correct is False
        assert "expected doc" in score.failure_reason

    def test_missing_keyword_fails(self) -> None:
        case = GoldenCase(
            id="c", question="q", expected_doc="storage", expected_keywords=["tmpfs"]
        )
        score = score_case(case, FakeResult("Something unrelated [1].", ["storage"]))
        assert score.correct is False
        assert "missing all of" in score.failure_reason

    def test_keywords_are_alternatives_not_a_conjunction(self) -> None:
        """ "2 million" and "two million" are both right; failing one would
        measure phrasing rather than correctness."""
        case = GoldenCase(
            id="c",
            question="q",
            expected_doc="cloud-run",
            expected_keywords=["two million", "2 million"],
        )
        score = score_case(case, FakeResult("About 2 million requests [1].", ["cloud-run"]))
        assert score.correct is True

    def test_refusing_an_answerable_question_fails(self) -> None:
        case = GoldenCase(
            id="c", question="q", expected_doc="storage", expected_keywords=["tmpfs"]
        )
        score = score_case(case, FakeResult(REFUSAL_MARKER.title(), ["storage"]))
        assert score.correct is False
        assert "refused a question the corpus does answer" in score.failure_reason

    def test_refusal_case_passes_only_when_refused(self) -> None:
        case = GoldenCase(id="c", question="q", must_be_grounded=False)
        assert score_case(case, FakeResult(REFUSAL_MARKER.capitalize(), [])).correct is True

    def test_answering_an_out_of_corpus_question_fails(self) -> None:
        """The worst failure available: a confident hallucination."""
        case = GoldenCase(id="c", question="q", must_be_grounded=False)
        score = score_case(case, FakeResult("The weather is sunny.", []))
        assert score.correct is False
        assert "cannot support" in score.failure_reason


# --------------------------------------------------------------------------
# Aggregation and thresholds
# --------------------------------------------------------------------------
class TestSummary:
    def _cases(self):
        return [
            GoldenCase(id="g1", question="q", expected_doc="a", expected_keywords=["x"]),
            GoldenCase(id="g2", question="q", expected_doc="b", expected_keywords=["y"]),
            GoldenCase(id="r1", question="q", must_be_grounded=False),
        ]

    def test_rates_are_scoped_to_the_right_cases(self) -> None:
        cases = self._cases()
        scores = [
            score_case(cases[0], FakeResult("x [1]", ["a"])),
            score_case(cases[1], FakeResult("nope", ["zzz"])),
            score_case(cases[2], FakeResult(REFUSAL_MARKER, [])),
        ]
        summary = summarise(cases, scores)
        # Retrieval is measured over the two grounded cases only.
        assert summary.retrieval_hit_rate == 0.5
        # Refusal is measured over the one refusal case only.
        assert summary.refusal_accuracy == 1.0
        assert summary.accuracy == pytest.approx(2 / 3, abs=0.01)

    def test_empty_denominator_scores_one(self) -> None:
        """A suite with no refusal cases has not failed refusal."""
        cases = [GoldenCase(id="g", question="q", expected_doc="a", expected_keywords=["x"])]
        summary = summarise(cases, [score_case(cases[0], FakeResult("x [1]", ["a"]))])
        assert summary.refusal_accuracy == 1.0

    def test_thresholds_report_each_breach(self) -> None:
        cases = self._cases()
        scores = [
            score_case(cases[0], FakeResult("nope", ["zzz"])),
            score_case(cases[1], FakeResult("nope", ["zzz"])),
            score_case(cases[2], FakeResult("I will answer anyway.", [])),
        ]
        breaches = Thresholds().check(summarise(cases, scores))
        assert any("accuracy" in b for b in breaches)
        assert any("refusal_accuracy" in b for b in breaches)

    def test_passing_run_reports_no_breaches(self) -> None:
        cases = self._cases()
        scores = [
            score_case(cases[0], FakeResult("x [1]", ["a"])),
            score_case(cases[1], FakeResult("y [1]", ["b"])),
            score_case(cases[2], FakeResult(REFUSAL_MARKER, [])),
        ]
        assert Thresholds().check(summarise(cases, scores)) == []


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------
# The floor the fake embedder is held to. It is NOT a quality target.
#
# conftest's embedder is a hashed bag-of-words with no semantic content. Under
# this suite's settings (chunk_size=400, top_k=3 -> 85 chunks, so ~3.5% of the
# corpus per query) it scores a measured, deterministic 56.7% -- verified
# identical across repeated runs. That is its ceiling, not a retrieval defect:
# it cannot resolve "Why does chunking use overlap?" to a passage sharing few
# literal tokens with it. text-embedding-3-small scores 100% on these same
# cases.
#
# So the floor sits well below that baseline and exists to catch a retrieval
# *collapse* -- a broken chunker, a mis-wired store, an inverted similarity
# comparison. Real retrieval quality is measured by `make eval` against the
# real embedder and gated at 0.85 there.
FAKE_EMBEDDER_RETRIEVAL_FLOOR = 0.45


class TestRegressionGate:
    """Runs the real golden set through the real retrieval path.

    Generation is faked, so only retrieval is asserted -- and only as a
    tripwire; see FAKE_EMBEDDER_RETRIEVAL_FLOOR for why the number is low.
    Chunking, embedding and similarity-floor changes all surface here.
    """

    def test_corpus_ingests(self, pipeline) -> None:
        assert ingest_corpus(pipeline, CORPUS_DIR) > 0

    def test_corpus_is_larger_than_retrieval_depth(self, pipeline, settings) -> None:
        """The guard that keeps the retrieval metric meaningful.

        Retrieving k chunks from a corpus of roughly k makes a perfect hit rate
        arithmetically inevitable and the metric worthless -- which is exactly
        what the original 5-chunk corpus did. Fail loudly if the corpus ever
        shrinks back to that.
        """
        chunks = ingest_corpus(pipeline, CORPUS_DIR)
        assert chunks >= settings.top_k * 5, (
            f"corpus is {chunks} chunks against top_k={settings.top_k}; "
            "retrieval hit rate stops discriminating when the corpus is not "
            "substantially larger than the retrieval depth"
        )

    def test_retrieval_has_not_collapsed(self, pipeline) -> None:
        ingest_corpus(pipeline, CORPUS_DIR)
        cases = load_golden_set(GOLDEN_PATH)

        summary, _ = run_evaluation(pipeline, cases, log_to_mlflow=False)

        grounded = {c.id for c in cases if not c.is_refusal_case}
        assert summary.total == len(cases)
        assert summary.retrieval_hit_rate >= FAKE_EMBEDDER_RETRIEVAL_FLOOR, (
            f"retrieval collapsed to {summary.retrieval_hit_rate:.1%}: "
            + "; ".join(
                f"{s.case_id}: {s.failure_reason}"
                for s in summary.failures
                if s.case_id in grounded
            )
        )

    # Deliberately NOT tested here: refusal accuracy. The same bag-of-words
    # limitation means an out-of-corpus question scores far too highly against
    # unrelated text, so asserting refusal here would measure the fake rather
    # than the system. scripts/run_eval.py measures it against the real model,
    # where the similarity floor and the refusal instruction both apply.
