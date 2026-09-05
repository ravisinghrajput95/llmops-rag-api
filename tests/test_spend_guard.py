"""Tests for the daily spend ceiling and the per-client rate limiter."""

from __future__ import annotations

import pytest

from app.rate_limit import RateLimiter
from app.tracking.spend_guard import BudgetExceededError, SpendGuard


class TestSpendGuard:
    def test_disabled_guard_never_blocks(self) -> None:
        guard = SpendGuard(budget_usd=0.0)
        assert guard.enabled is False
        guard.record(1_000.0)
        guard.check()  # must not raise

    def test_allows_spend_below_budget(self) -> None:
        guard = SpendGuard(budget_usd=1.0)
        guard.record(0.4)
        guard.check()
        assert guard.snapshot().remaining_usd == pytest.approx(0.6)

    def test_blocks_once_budget_reached(self) -> None:
        guard = SpendGuard(budget_usd=0.10)
        guard.record(0.10)
        with pytest.raises(BudgetExceededError) as excinfo:
            guard.check()
        assert excinfo.value.budget_usd == 0.10
        assert excinfo.value.spent_usd == pytest.approx(0.10)

    def test_error_reports_reset_window(self) -> None:
        guard = SpendGuard(budget_usd=0.01, window_seconds=3600)
        guard.record(0.02)
        with pytest.raises(BudgetExceededError) as excinfo:
            guard.check()
        # Nothing has elapsed, so the whole window remains.
        assert 3500 <= excinfo.value.resets_in_seconds <= 3600

    def test_window_rolls_and_clears_spend(self) -> None:
        # A zero-length window rolls on the very next call, which exercises the
        # reset path without making the test sleep.
        guard = SpendGuard(budget_usd=0.01, window_seconds=0)
        guard.record(5.0)
        guard.check()  # window rolled, so the ceiling is clear again
        assert guard.snapshot().spent_usd == 0.0

    def test_snapshot_reports_calls_and_totals(self) -> None:
        guard = SpendGuard(budget_usd=1.0)
        guard.record(0.1)
        guard.record(0.2)
        snap = guard.snapshot()
        assert snap.calls == 2
        assert snap.spent_usd == pytest.approx(0.3)
        assert snap.enabled is True

    def test_zero_and_negative_costs_are_ignored(self) -> None:
        guard = SpendGuard(budget_usd=1.0)
        guard.record(0.0)
        guard.record(-5.0)
        assert guard.snapshot().calls == 0


class TestRateLimiter:
    def test_disabled_limiter_always_allows(self) -> None:
        limiter = RateLimiter(requests_per_minute=0)
        assert limiter.enabled is False
        for _ in range(100):
            assert limiter.allow("someone")[0] is True

    def test_allows_up_to_the_burst_then_blocks(self) -> None:
        limiter = RateLimiter(requests_per_minute=3)
        assert [limiter.allow("a")[0] for _ in range(3)] == [True, True, True]
        allowed, retry_after = limiter.allow("a")
        assert allowed is False
        assert retry_after > 0

    def test_clients_are_isolated(self) -> None:
        limiter = RateLimiter(requests_per_minute=1)
        assert limiter.allow("a")[0] is True
        assert limiter.allow("a")[0] is False
        # A different key has its own untouched bucket.
        assert limiter.allow("b")[0] is True

    def test_evicts_when_key_count_exceeds_cap(self) -> None:
        limiter = RateLimiter(requests_per_minute=5, max_keys=8)
        for i in range(40):
            limiter.allow(f"client-{i}")
        assert len(limiter._buckets) <= 8 + 1
