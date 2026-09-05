"""Per-client request throttling.

The spend guard bounds the *day*; this bounds the *minute*. They solve
different problems: a runaway client loop would exhaust the daily budget in
seconds without a rate limit, and the operator would find out from a 429 storm
rather than from a gradual drawdown they could react to.

Same honest caveat as SpendGuard: state is per-process, so with
max_instances=2 the effective allowance is up to 2x the configured rate. That
is fine for a demo. A correct distributed limiter needs shared state, which
costs money this project deliberately does not spend.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Bucket:
    tokens: float
    updated: float = field(default_factory=time.monotonic)


class RateLimiter:
    """Token bucket, keyed by client identity.

    Burst equals the per-minute rate, so a client may spend its whole minute's
    allowance at once and then refill steadily. That suits a demo being poked
    by hand far better than a strict evenly-spaced limiter would.
    """

    def __init__(self, requests_per_minute: int, max_keys: int = 4096) -> None:
        self._rate = max(0, requests_per_minute)
        self._burst = float(self._rate)
        self._refill_per_second = self._rate / 60.0
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        # Bounds memory: an attacker rotating client keys must not be able to
        # grow this dict without limit. Eviction is crude (drop the oldest
        # half) because precision here buys nothing.
        self._max_keys = max_keys

    @property
    def enabled(self) -> bool:
        return self._rate > 0

    def _evict_locked(self) -> None:
        if len(self._buckets) <= self._max_keys:
            return
        ordered = sorted(self._buckets.items(), key=lambda kv: kv[1].updated)
        for key, _ in ordered[: len(ordered) // 2]:
            del self._buckets[key]

    def allow(self, key: str) -> tuple[bool, float]:
        """Consume one token for `key`.

        Returns (allowed, retry_after_seconds). retry_after is 0 when allowed.
        """
        if not self.enabled:
            return True, 0.0

        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                self._evict_locked()
                self._buckets[key] = _Bucket(tokens=self._burst - 1.0, updated=now)
                return True, 0.0

            elapsed = now - bucket.updated
            bucket.tokens = min(self._burst, bucket.tokens + elapsed * self._refill_per_second)
            bucket.updated = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0

            # How long until one whole token is available again.
            deficit = 1.0 - bucket.tokens
            retry_after = (
                deficit / self._refill_per_second if self._refill_per_second else 60.0
            )
            return False, round(retry_after, 2)
