"""A hard ceiling on what this service can spend at OpenAI in a day.

Why this exists: GCP teardown does not stop the OpenAI bill. The Cloud Run
service is public (`allUsers` has run.invoker), it holds a real API key, and
the only thing between a stranger and your balance is the shared secret in
APP_API_KEY. If that key leaks -- pasted in a demo, committed by accident,
shoulder-surfed at a meetup -- the blast radius should be one day's budget,
not the whole card.

The guard is deliberately *pre-flight*: it refuses a call whose cost it cannot
yet know, based on spend already recorded. That means the ceiling can be
overshot by at most one request, which is the correct trade-off -- the
alternative is calling OpenAI to find out what calling OpenAI would cost.

Scope and honesty about it: state is per-process, held in memory. With
max_instances=2 the effective ceiling is up to 2x DAILY_BUDGET_USD, and it
resets when Cloud Run scales to zero. A shared counter would need Redis or
Firestore -- neither is free at this budget, and both would add a network hop
to every request. For a demo whose whole point is spending nothing, an
approximate in-process cap that fails closed is worth far more than a perfect
distributed one that costs money to run.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_DAY_SECONDS = 86_400


class BudgetExceededError(RuntimeError):
    """Raised when the daily spend ceiling has been reached.

    Carries the numbers so the API layer can render an actionable 429 instead
    of a bare "too many requests".
    """

    def __init__(self, spent_usd: float, budget_usd: float, resets_in_seconds: float) -> None:
        self.spent_usd = round(spent_usd, 6)
        self.budget_usd = budget_usd
        self.resets_in_seconds = int(max(0.0, resets_in_seconds))
        super().__init__(
            f"daily budget of ${budget_usd:.4f} exhausted "
            f"(spent ${self.spent_usd:.4f}); resets in {self.resets_in_seconds}s"
        )


@dataclass(frozen=True)
class SpendSnapshot:
    enabled: bool
    spent_usd: float
    budget_usd: float
    remaining_usd: float
    calls: int
    window_resets_in_seconds: int


class SpendGuard:
    """Rolling 24h spend ceiling, enforced in-process.

    A fixed window rather than a sliding one: it is one comparison and one
    counter, it is trivially explainable in a README, and the failure mode
    (a full budget available immediately after a reset) is acceptable for a
    ceiling whose job is catching runaway loops, not smoothing traffic.
    """

    def __init__(self, budget_usd: float, window_seconds: int = _DAY_SECONDS) -> None:
        self._budget = max(0.0, budget_usd)
        self._window = window_seconds
        self._lock = threading.Lock()
        self._spent = 0.0
        self._calls = 0
        self._window_started = time.monotonic()

    @property
    def enabled(self) -> bool:
        # 0 disables the guard entirely, which is the right default for local
        # development and for tests that must not care about budgets.
        return self._budget > 0

    def _roll_window_locked(self) -> None:
        elapsed = time.monotonic() - self._window_started
        if elapsed >= self._window:
            self._spent = 0.0
            self._calls = 0
            self._window_started = time.monotonic()
            logger.info("spend window reset", extra={"budget_usd": self._budget})

    def check(self) -> None:
        """Raise BudgetExceededError if the ceiling is already reached.

        Call this *before* spending money.
        """
        if not self.enabled:
            return
        with self._lock:
            self._roll_window_locked()
            if self._spent >= self._budget:
                resets_in = self._window - (time.monotonic() - self._window_started)
                raise BudgetExceededError(self._spent, self._budget, resets_in)

    def record(self, cost_usd: float) -> None:
        """Add an actual (estimated) cost to the running total."""
        if not self.enabled or cost_usd <= 0:
            return
        with self._lock:
            self._roll_window_locked()
            self._spent += cost_usd
            self._calls += 1
            # Warn once the budget is most of the way gone, so the operator
            # sees it in Cloud Logging before requests start failing.
            if self._spent >= self._budget * 0.8:
                logger.warning(
                    "daily spend ceiling nearly reached",
                    extra={
                        "spent_usd": round(self._spent, 6),
                        "budget_usd": self._budget,
                        "calls": self._calls,
                    },
                )

    def snapshot(self) -> SpendSnapshot:
        with self._lock:
            self._roll_window_locked()
            resets_in = self._window - (time.monotonic() - self._window_started)
            return SpendSnapshot(
                enabled=self.enabled,
                spent_usd=round(self._spent, 6),
                budget_usd=self._budget,
                remaining_usd=round(max(0.0, self._budget - self._spent), 6),
                calls=self._calls,
                window_resets_in_seconds=int(max(0.0, resets_in)),
            )
