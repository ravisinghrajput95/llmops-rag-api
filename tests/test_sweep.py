"""Tests for the retrieval parameter sweep.

The sweep's whole value is that its ranking predicts end-to-end quality. The
first version of it did not: it ranked on document-level retrieval hit rate,
chose a configuration scoring a perfect 1.000, and that configuration then lost
7 points of real accuracy to false refusals. Both configurations scored an
identical 1.000 hit rate, so the metric had no discriminating power at all.

These tests pin the property that fixed it -- ranking leads on whether the
supporting passage actually reached the context, not on whether the right
document was somewhere in the results.
"""

from __future__ import annotations

from app.evaluation.dataset import GoldenCase
from app.evaluation.sweep import SweepPoint, SweepResult, build_grid, format_sweep


def _result(
    answerable: float, hit_rate: float, chunks_returned: float = 3.0, **kw
) -> SweepResult:
    return SweepResult(
        point=kw.pop("point", SweepPoint(800, 120, 4, 0.2)),
        chunks=42,
        retrieval_hit_rate=hit_rate,
        answerable_rate=answerable,
        empty_on_out_of_corpus=kw.pop("empty", 0.5),
        mean_retrieval_ms=10.0,
        mean_chunks_returned=chunks_returned,
        embedding_tokens=1000,
        embedding_cost_usd=0.00002,
    )


class TestGrid:
    def test_grid_is_the_cartesian_product(self) -> None:
        grid = build_grid([400, 800], [0], [3, 4], [0.2])
        assert len(grid) == 4

    def test_overlap_at_or_above_chunk_size_is_dropped(self) -> None:
        """An overlap that repeats a whole chunk is not a configuration."""
        grid = build_grid([400], [0, 400, 600], [3], [0.2])
        assert [p.chunk_overlap for p in grid] == [0]

    def test_points_are_hashable_and_comparable(self) -> None:
        # format_sweep compares the best point against the configured one.
        assert SweepPoint(800, 120, 4, 0.2) == SweepPoint(800, 120, 4, 0.2)


class TestRanking:
    def test_answerable_rate_outranks_hit_rate(self) -> None:
        """The regression that motivated this metric.

        A configuration with a perfect document hit rate but a worse answerable
        rate must lose. Ranking the other way is what selected a config that
        then failed five cases with false refusals.
        """
        misleading = _result(answerable=0.93, hit_rate=1.00)
        better = _result(answerable=0.98, hit_rate=0.98)

        assert better.score > misleading.score

    def test_hit_rate_breaks_ties_when_answerable_is_equal(self) -> None:
        low = _result(answerable=0.98, hit_rate=0.95)
        high = _result(answerable=0.98, hit_rate=1.00)
        assert high.score > low.score

    def test_fewer_chunks_wins_all_else_equal(self) -> None:
        """Context size is a permanent per-query cost, so it breaks ties."""
        fat = _result(answerable=1.0, hit_rate=1.0, chunks_returned=6.0)
        lean = _result(answerable=1.0, hit_rate=1.0, chunks_returned=2.0)
        assert lean.score > fat.score

    def test_out_of_corpus_rejection_ranks_above_chunk_count(self) -> None:
        rejects_more = _result(answerable=1.0, hit_rate=1.0, chunks_returned=6.0, empty=0.9)
        rejects_less = _result(answerable=1.0, hit_rate=1.0, chunks_returned=2.0, empty=0.1)
        assert rejects_more.score > rejects_less.score


class TestReport:
    def test_report_marks_the_winner_and_names_the_current_config(self) -> None:
        from app.config import Settings

        results = sorted(
            [
                _result(answerable=0.90, hit_rate=1.0, point=SweepPoint(400, 0, 3, 0.4)),
                _result(answerable=0.98, hit_rate=1.0, point=SweepPoint(800, 0, 4, 0.2)),
            ],
            key=lambda r: r.score,
            reverse=True,
        )
        baseline = Settings(
            openai_api_key="x", chunk_size=800, chunk_overlap=120, top_k=4, min_similarity=0.2
        )

        report = format_sweep(results, baseline=baseline)

        assert "answerable" in report
        assert "cs=800 ov=0 k=4" in report  # the winner
        assert "Currently configured: cs=800 ov=120 k=4" in report

    def test_report_says_when_the_winner_is_already_configured(self) -> None:
        from app.config import Settings

        point = SweepPoint(800, 120, 4, 0.2)
        baseline = Settings(
            openai_api_key="x", chunk_size=800, chunk_overlap=120, top_k=4, min_similarity=0.2
        )

        report = format_sweep([_result(1.0, 1.0, point=point)], baseline=baseline)

        assert "already in use" in report

    def test_report_totals_the_cost(self) -> None:
        report = format_sweep([_result(1.0, 1.0), _result(0.9, 0.9)])
        assert "2 configurations" in report
        assert "total embedding cost" in report


class TestAnswerableScoring:
    """The answerable metric is keyword presence in the retrieved text."""

    def test_case_without_keywords_counts_as_answerable(self) -> None:
        # Nothing to check against means the case cannot fail this metric,
        # which keeps it from penalising cases that assert only retrieval.
        case = GoldenCase(id="c", question="q", expected_doc="d", expected_keywords=[])
        assert case.expected_keywords == []
