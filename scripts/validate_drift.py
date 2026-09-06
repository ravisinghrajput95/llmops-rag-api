#!/usr/bin/env python
"""Does the drift check cry wolf, and does it catch real drift?

    python scripts/validate_drift.py

Free and offline: it resamples `evals/signals.json`, which holds the measured
retrieval signal and the real gpt-4o-mini refusal outcome for all 128 golden
questions. A drift detector nobody has characterised is worse than none -- it
either fires constantly and gets ignored, or never fires and gets trusted.

The false-positive figures are optimistic by construction: the null windows are
resampled from the same answerable cases the baseline is built from, so they
are more alike than real traffic would be. Read them as a lower bound.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.monitoring.drift import (  # noqa: E402
    MIN_WINDOW,
    Baseline,
    QueryRecord,
    compare,
    diagnose,
    summarise,
)


def load_populations(path: Path):
    data = json.loads(path.read_text())
    groups: dict[str, list[QueryRecord]] = {"answerable": [], "near": [], "offtopic": []}
    for row in data["cases"]:
        record = QueryRecord(
            refused=row["refused"],
            grounded=row["grounded"],
            top_similarity=row["top_similarity"],
            retrieved_chunks=float(row["retrieved_chunks"]),
            prompt_version="v1",
        )
        tier = row["tier"]
        key = tier if tier in ("answerable", "near") else "offtopic"
        groups[key].append(record)
    return groups, data


def make_baseline(answerable: list[QueryRecord]) -> Baseline:
    n = len(answerable)
    sims = [r.top_similarity for r in answerable]
    return Baseline(
        n=n,
        refusal_rate=sum(r.refused for r in answerable) / n,
        ungrounded_rate=sum(not r.grounded for r in answerable) / n,
        mean_top_similarity=sum(sims) / n,
        mean_chunks=sum(r.retrieved_chunks for r in answerable) / n,
        similarities=sims,
        prompt_version="v1",
        source="validation fixture",
    )


def draw(rng, answerable, contaminant, fraction, n):
    k = int(round(fraction * n))
    return [rng.choice(contaminant) for _ in range(k)] + [
        rng.choice(answerable) for _ in range(n - k)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", default="evals/signals.json")
    parser.add_argument("--trials", type=int, default=2000)
    args = parser.parse_args()

    groups, data = load_populations(Path(args.signals))
    answerable, near, offtopic = groups["answerable"], groups["near"], groups["offtopic"]
    base = make_baseline(answerable)

    print(
        f"\npopulations   {len(answerable)} answerable, {len(near)} near-miss, "
        f"{len(offtopic)} off-topic"
    )
    print(
        f"baseline      refusal {base.refusal_rate:.1%}, mean top score "
        f"{base.mean_top_similarity:.3f}, {base.mean_chunks:.2f} chunks/query"
    )

    print("\nFALSE POSITIVES -- windows of purely answerable traffic")
    print(f"{'window':>8}{'any finding':>14}{'alert':>9}")
    for n in [20, 30, 50, 100, 200]:
        rng = random.Random(0)
        fired = alerts = 0
        for _ in range(args.trials):
            findings = [
                f
                for f in compare(base, summarise(draw(rng, answerable, near, 0.0, n)))
                if f.signal != "window"
            ]
            fired += bool(findings)
            alerts += any(f.severity == "alert" for f in findings)
        note = "  (below MIN_WINDOW: not scored)" if n < MIN_WINDOW else ""
        print(f"{n:>8}{fired / args.trials:>13.1%}{alerts / args.trials:>9.1%}{note}")

    trials = max(1, args.trials // 4)
    for label, contaminant in (("NEAR-MISS", near), ("OFF-TOPIC", offtopic)):
        print(f"\nDETECTION -- {label} contamination, window of 100")
        print(f"{'mix':>6}{'detected':>11}{'via refusal':>14}{'via similarity':>17}")
        for fraction in [0.05, 0.10, 0.20, 0.30, 0.50]:
            rng = random.Random(1)
            detected = by_refusal = by_similarity = 0
            for _ in range(trials):
                findings = compare(
                    base, summarise(draw(rng, answerable, contaminant, fraction, 100))
                )
                signals = {f.signal for f in findings}
                detected += bool(signals - {"window"})
                by_refusal += "refusal_rate" in signals
                by_similarity += "top_similarity" in signals
            print(
                f"{fraction:>6.0%}{detected / trials:>11.1%}"
                f"{by_refusal / trials:>14.1%}{by_similarity / trials:>17.1%}"
            )

    print("\nDIAGNOSIS -- is the named cause the right one? (window 100, 30% mix)")
    for label, contaminant in (("near-miss", near), ("off-topic", offtopic)):
        rng = random.Random(2)
        counts: dict[str, int] = {}
        for _ in range(200):
            window = summarise(draw(rng, answerable, contaminant, 0.30, 100))
            head = diagnose(base, window, compare(base, window)).split(":")[0]
            counts[head] = counts.get(head, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        print(f"  {label:<10} " + "; ".join(f"{k} {v / 200:.0%}" for k, v in ranked[:2]))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
