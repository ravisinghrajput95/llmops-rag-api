"""Cost estimation -- the numbers the whole project exists to report."""

from __future__ import annotations

import pytest

from app.tracking.cost import PRICING, estimate_cost_usd, price_for, usd_to_inr


def test_known_model_prices_match_the_published_rate_card():
    # gpt-4o-mini: $0.15 per 1M input, $0.60 per 1M output.
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert estimate_cost_usd("gpt-4o-mini", 0, 1_000_000) == pytest.approx(0.60)


def test_embedding_model_has_no_output_charge():
    assert PRICING["text-embedding-3-small"].output_per_1m == 0.0
    assert estimate_cost_usd("text-embedding-3-small", 0, 5_000) == 0.0


def test_typical_rag_query_costs_a_fraction_of_a_rupee():
    """Guards the core cost claim in the README."""
    cost = estimate_cost_usd("gpt-4o-mini", prompt_tokens=1_200, completion_tokens=250)

    assert usd_to_inr(cost, 88.0) < 0.10


def test_dated_model_snapshots_resolve_to_base_pricing():
    assert price_for("gpt-4o-mini-2024-07-18") == PRICING["gpt-4o-mini"]


def test_unknown_model_falls_back_without_raising():
    assert estimate_cost_usd("some-future-model", 1_000, 100) > 0


def test_zero_tokens_cost_nothing():
    assert estimate_cost_usd("gpt-4o-mini", 0, 0) == 0.0


def test_sub_cent_costs_are_not_rounded_away():
    """A single query is ~$0.0002; rounding to cents would report zero."""
    assert estimate_cost_usd("gpt-4o-mini", 500, 100) > 0
