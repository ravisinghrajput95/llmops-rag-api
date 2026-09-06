#!/usr/bin/env python
"""Run the golden set under two prompt versions and compare them.

    python scripts/compare_prompts.py --variant evals/prompt_variants/v2-precision
    python scripts/compare_prompts.py --variant <dir> --ids case-a,case-b

SPENDS MONEY -- roughly twice a `make eval`, because it generates every answer
twice. `--ids` and `--limit` exist to make a targeted comparison cheap when you
are chasing specific failures.

Versioning a prompt made a change *visible* in eval history. This is what makes
one *decidable*: both arms run against the same store, the same corpus and the
same retrieval config, so the only variable is the wording. Retrieval is
identical by construction, which is why the arms share a pipeline rather than
re-ingesting.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.config import get_settings  # noqa: E402
from app.dependencies import build_pipeline  # noqa: E402
from app.evaluation.metrics import EvalSummary  # noqa: E402
from app.evaluation.runner import (  # noqa: E402
    ingest_corpus,
    load_golden_set,
    run_evaluation,
)
from app.logging_config import configure_logging  # noqa: E402
from app.rag.prompts import PROMPTS, PromptSet, load_templates  # noqa: E402


def compare_table(results: list[tuple[str, EvalSummary]]) -> str:
    rows = [
        ("accuracy", lambda s: f"{s.accuracy:.1%}"),
        ("retrieval hit rate", lambda s: f"{s.retrieval_hit_rate:.1%}"),
        ("refusal accuracy", lambda s: f"{s.refusal_accuracy:.1%}"),
        ("citation rate", lambda s: f"{s.citation_rate:.1%}"),
        ("mean latency", lambda s: f"{s.mean_latency_ms:.0f} ms"),
        ("total cost", lambda s: f"${s.total_cost_usd:.6f}"),
    ]
    width = max(len(name) for name, _ in results) + 2
    lines = [
        "",
        "Prompt comparison",
        "=" * (22 + width * len(results)),
        f"{'':22}" + "".join(f"{name:<{width}}" for name, _ in results),
        "-" * (22 + width * len(results)),
    ]
    for label, render in rows:
        lines.append(
            f"  {label:<20}" + "".join(f"{render(summary):<{width}}" for _, summary in results)
        )
    return "\n".join(lines)


def flips(results: list[tuple[str, EvalSummary]]) -> str:
    """Per-case changes. The aggregate can stay flat while cases trade places,
    and a prompt that fixes two failures by causing two others is not progress.
    """
    base = results[0][1]
    candidate = results[-1][1]
    by_id = {score.case_id: score for score in base.scores}
    fixed, broken = [], []
    for score in candidate.scores:
        before = by_id.get(score.case_id)
        if before is None or before.correct == score.correct:
            continue
        (fixed if score.correct else broken).append(score)

    lines = ["", f"Changed cases ({len(fixed)} fixed, {len(broken)} broken):"]
    if not fixed and not broken:
        lines.append("  none -- every case scored the same under both")
    for score in fixed:
        lines.append(f"  FIXED   {score.case_id}")
        lines.append(f"          now: {score.answer[:110]}")
    for score in broken:
        lines.append(f"  BROKEN  {score.case_id}: {score.failure_reason}")
        lines.append(f"          now: {score.answer[:110]}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="directory of .txt templates")
    parser.add_argument("--label", default="", help="name for the variant arm")
    parser.add_argument("--golden", default="evals/golden.jsonl")
    parser.add_argument("--corpus", default="evals/corpus")
    parser.add_argument("--ids", default="", help="comma-separated case ids to run")
    parser.add_argument("--limit", type=int, default=0, help="first N cases only")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(level="WARNING", service_name=settings.service_name)
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set; this script needs a real key.", file=sys.stderr)
        return 2

    cases = load_golden_set(args.golden)
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        cases = [c for c in cases if c.id in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    variant = PromptSet(
        version=args.label or Path(args.variant).name,
        templates=load_templates(args.variant),
        locked=True,
    )

    pipeline = build_pipeline(settings)
    print(f"Ingesting {args.corpus} ...")
    print(f"  {ingest_corpus(pipeline, args.corpus).chunk_count} chunks indexed")
    print(f"Comparing {len(cases)} cases x 2 prompt versions on {settings.chat_model}")
    print(f"  A: {PROMPTS.tracked_version} ({PROMPTS.fingerprint})")
    print(f"  B: {variant.tracked_version} ({variant.fingerprint})")

    results: list[tuple[str, EvalSummary]] = []
    for prompts in (PROMPTS, variant):
        arm = pipeline.with_prompts(prompts)
        summary, _ = run_evaluation(arm, cases, log_to_mlflow=True)
        results.append((prompts.tracked_version, summary))

    print(compare_table(results))
    print(flips(results))
    print()
    return 0


if __name__ == "__main__":
    logging.captureWarnings(True)
    raise SystemExit(main())
