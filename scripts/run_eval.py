#!/usr/bin/env python
"""Run the RAG evaluation against real OpenAI.

    OPENAI_API_KEY=sk-... python scripts/run_eval.py

This SPENDS MONEY -- roughly $0.002 for the default 15-case set on
gpt-4o-mini. The CI gate in tests/test_evaluation.py runs the same harness
against deterministic fakes for free; this exists to measure what the real
model actually does. Exits non-zero when a threshold is breached, so it can
gate a release if you want it to.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.config import get_settings  # noqa: E402
from app.dependencies import build_pipeline  # noqa: E402
from app.evaluation.runner import (  # noqa: E402
    Thresholds,
    format_report,
    ingest_corpus,
    load_golden_set,
    run_evaluation,
    write_baseline,
)
from app.logging_config import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default="evals/golden.jsonl")
    parser.add_argument("--corpus", default="evals/corpus")
    parser.add_argument("--accuracy", type=float, default=0.80)
    parser.add_argument("--retrieval", type=float, default=0.85)
    parser.add_argument("--refusal", type=float, default=0.98)
    parser.add_argument(
        "--no-mlflow", action="store_true", help="skip logging the run to MLflow"
    )
    parser.add_argument("--baseline", default="evals/baseline.json")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="do not refresh the monitoring baseline from this run",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(level="WARNING", service_name=settings.service_name)

    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set; this script needs a real key.", file=sys.stderr)
        return 2

    pipeline = build_pipeline(settings)
    cases = load_golden_set(args.golden)

    print(f"Ingesting {args.corpus} ...")
    ingested = ingest_corpus(pipeline, args.corpus)
    print(f"  {ingested.chunk_count} chunks indexed")
    prompts = pipeline.prompt_info()
    print(f"Running {len(cases)} cases against {settings.chat_model} ...")
    print(f"  prompts {prompts['version']} ({prompts['fingerprint']})")

    summary, breaches = run_evaluation(
        pipeline,
        cases,
        thresholds=Thresholds(
            accuracy=args.accuracy,
            retrieval_hit_rate=args.retrieval,
            refusal_accuracy=args.refusal,
        ),
        log_to_mlflow=not args.no_mlflow,
    )
    print(format_report(summary, breaches))

    # Refreshed by default: the baseline describes the configuration that was
    # just measured, and a stale one silently compares production against a
    # setup that no longer exists. Git is where you see it change.
    if not args.no_baseline:
        baseline = write_baseline(args.baseline, cases, summary, settings, prompts)
        print(
            f"Monitoring baseline written to {args.baseline} "
            f"({baseline.n} answerable cases, refusal {baseline.refusal_rate:.1%})"
        )

    return 1 if breaches else 0


if __name__ == "__main__":
    logging.captureWarnings(True)
    raise SystemExit(main())
