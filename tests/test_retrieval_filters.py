"""The two retrieval filters: the absolute floor and the relative ratio.

These build vectors with exactly known cosine similarities rather than going
through the fake embedder. The fake is a hashed bag of words, so asserting a
similarity threshold against it would measure the fake and not the filter --
the trap CLAUDE.md calls out. A vector at a chosen angle has the similarity it
has, under any embedding model or none.
"""

from __future__ import annotations

import math
import tempfile

import pytest

from app.rag.chunking import Chunk
from app.rag.vectorstore import ChromaVectorStore

DIM = 8


def at_similarity(score: float) -> list[float]:
    """A unit vector whose cosine similarity to QUERY is exactly `score`."""
    return [score, math.sqrt(max(0.0, 1.0 - score * score))] + [0.0] * (DIM - 2)


QUERY = [1.0] + [0.0] * (DIM - 1)


@pytest.fixture
def store_with():
    """Build a store holding one chunk per requested similarity."""

    def build(scores: list[float]) -> ChromaVectorStore:
        store = ChromaVectorStore(tempfile.mkdtemp(), "filters")
        chunks = [
            Chunk(
                chunk_id=f"c{i}",
                doc_id=f"doc{i}",
                text=f"chunk {i}",
                index=i,
                metadata={"doc_id": f"doc{i}"},
            )
            for i, _ in enumerate(scores)
        ]
        store.add(chunks, [at_similarity(s) for s in scores])
        return store

    return build


def sims(hits) -> list[float]:
    return [round(h.similarity, 2) for h in hits]


# -- the absolute floor -----------------------------------------------------
def test_absolute_floor_drops_chunks_below_it(store_with):
    store = store_with([0.80, 0.50, 0.20])

    assert sims(store.search(QUERY, top_k=4, min_similarity=0.4)) == [0.80, 0.50]


def test_query_below_the_floor_retrieves_nothing(store_with):
    """The floor doubles as a query-level gate: when the best chunk cannot
    clear it, nothing does, and the caller skips the paid completion."""
    store = store_with([0.25, 0.20, 0.10])

    assert store.search(QUERY, top_k=4, min_similarity=0.28) == []


# -- the relative ratio -----------------------------------------------------
def test_ratio_drops_a_weak_chunk_that_clears_the_absolute_floor(store_with):
    """The point of the ratio. 0.45 is above any floor an answerable question
    can tolerate, but next to a 0.80 hit it is padding -- and it costs input
    tokens on every query that retrieves it.
    """
    store = store_with([0.80, 0.75, 0.45])

    assert sims(store.search(QUERY, top_k=4, min_similarity=0.28)) == [0.80, 0.75, 0.45]
    assert sims(
        store.search(QUERY, top_k=4, min_similarity=0.28, min_similarity_ratio=0.60)
    ) == [0.80, 0.75]


def test_ratio_rescales_per_query_rather_than_using_one_cut_point(store_with):
    """Why a ratio and not a higher floor: the same 0.45 chunk is padding
    beside a 0.80 hit and the best evidence available beside a 0.50 one. No
    absolute threshold can treat those two cases differently; this does.
    """
    strong = store_with([0.80, 0.45])
    weak = store_with([0.50, 0.45])
    kwargs = {"top_k": 4, "min_similarity": 0.28, "min_similarity_ratio": 0.60}

    assert sims(strong.search(QUERY, **kwargs)) == [0.80]
    assert sims(weak.search(QUERY, **kwargs)) == [0.50, 0.45]


def test_ratio_never_drops_the_best_chunk(store_with):
    """A ratio of the top score cannot exclude the top score, at any setting.
    Retrieval degrades to one chunk, never to none."""
    store = store_with([0.62, 0.61, 0.60])

    hits = store.search(QUERY, top_k=4, min_similarity=0.0, min_similarity_ratio=0.99)

    assert sims(hits) == [0.62]


def test_ratio_of_zero_disables_it(store_with):
    store = store_with([0.80, 0.30])

    assert sims(
        store.search(QUERY, top_k=4, min_similarity=0.28, min_similarity_ratio=0.0)
    ) == [
        0.80,
        0.30,
    ]


def test_the_stricter_of_the_two_filters_wins(store_with):
    """Absolute and relative are both floors, so the higher one binds."""
    store = store_with([0.90, 0.40])

    # ratio bar is 0.60 * 0.90 = 0.54, above the 0.28 absolute floor.
    assert sims(
        store.search(QUERY, top_k=4, min_similarity=0.28, min_similarity_ratio=0.60)
    ) == [0.90]
    # absolute floor of 0.80 is above the ratio bar of 0.45.
    assert sims(
        store.search(QUERY, top_k=4, min_similarity=0.80, min_similarity_ratio=0.50)
    ) == [0.90]


# -- the shipped defaults ---------------------------------------------------
def test_shipped_defaults_are_the_calibrated_pair():
    """These two numbers were measured together on real embeddings against 60
    answerable and 68 out-of-corpus questions. Changing one without re-running
    that calibration is what this test is here to make someone think about.
    """
    from app.config import Settings

    settings = Settings(openai_api_key="test-key-not-real")

    assert settings.min_similarity == 0.28
    assert settings.min_similarity_ratio == 0.60
