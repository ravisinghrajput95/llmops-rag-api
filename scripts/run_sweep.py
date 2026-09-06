#!/usr/bin/env python
"""Sweep retrieval parameters and compare the results in MLflow.

    OPENAI_API_KEY=sk-... python scripts/run_sweep.py

Costs a fraction of a cent: the sweep scores retrieval only, so it embeds the
corpus once per configuration and never generates an answer. A 12-point grid
runs for roughly $0.002. Use the winning configuration as a candidate, then
confirm it with `make eval`, which does pay for generation.

  --offline   use a deterministic fake embedder (free, no API key). Useful to
              see the mechanics; the rankings are not meaningful, because a
              bag-of-words embedder has no semantics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.config import Settings, get_settings  # noqa: E402
from app.evaluation.runner import load_golden_set  # noqa: E402
from app.evaluation.sweep import build_grid, format_sweep, run_sweep  # noqa: E402
from app.llm.openai_client import (  # noqa: E402
    OpenAIChatClient,
    OpenAIEmbeddingClient,
    build_openai_client,
)
from app.logging_config import configure_logging  # noqa: E402
from app.rag.pipeline import RAGPipeline  # noqa: E402
from app.rag.vectorstore import ChromaVectorStore  # noqa: E402
from app.tracking.mlflow_tracker import MLflowTracker  # noqa: E402


def _ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default="evals/golden.jsonl")
    parser.add_argument("--corpus", default="evals/corpus")
    parser.add_argument("--chunk-sizes", default="400,800,1200")
    parser.add_argument("--overlaps", default="0,120")
    parser.add_argument("--top-k", default="3,4,6")
    parser.add_argument("--floors", default="0.2")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    base = get_settings()
    configure_logging(level="WARNING", service_name=base.service_name)

    if not args.offline and not base.openai_api_key:
        print(
            "OPENAI_API_KEY is not set; use --offline to see the mechanics.", file=sys.stderr
        )
        return 2

    import tempfile

    def build_pipeline_for(point):
        # A fresh store per configuration: chunk size changes what gets
        # indexed, so reusing one would mix chunkings from different points.
        settings = base.model_copy(
            update={
                "chunk_size": point.chunk_size,
                "chunk_overlap": point.chunk_overlap,
                "top_k": point.top_k,
                "min_similarity": point.min_similarity,
                "chroma_dir": tempfile.mkdtemp(prefix="sweep-"),
                "mlflow_enabled": False,  # per-ingest runs would bury the sweep runs
                "gcs_bucket": "",
                "daily_budget_usd": 0.0,
            }
        )
        store = ChromaVectorStore(settings.chroma_dir, "sweep")
        if args.offline:
            embed, chat = _fakes()
        else:
            client = build_openai_client(settings)
            embed = OpenAIEmbeddingClient(client, settings)
            chat = OpenAIChatClient(client, settings)
        return RAGPipeline(settings, store, embed, chat, MLflowTracker(settings))

    grid = build_grid(
        _ints(args.chunk_sizes), _ints(args.overlaps), _ints(args.top_k), _floats(args.floors)
    )
    cases = load_golden_set(args.golden)
    print(f"Sweeping {len(grid)} configurations over {len(cases)} cases ...")

    tracker = None
    if not args.no_mlflow:
        tracker = MLflowTracker(base)

    results = run_sweep(grid, cases, build_pipeline_for, args.corpus, tracker=tracker)
    print(format_sweep(results, baseline=base))
    return 0


def _fakes():
    """Deterministic offline stand-ins, mirroring the test suite's fakes."""
    import hashlib
    import math
    import re

    from app.llm.openai_client import ChatResult, EmbeddingResult

    word = re.compile(r"[a-z0-9]+")

    def vector(text: str, dim: int = 64) -> list[float]:
        v = [0.0] * dim
        for w in word.findall(text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else [1.0] + [0.0] * (dim - 1)

    class E:
        def embed(self, texts):
            return EmbeddingResult(vectors=[vector(t) for t in texts], tokens=len(texts))

    class C:
        def complete(self, system, user):
            return ChatResult(
                text="offline", model="fake", prompt_tokens=0, completion_tokens=0
            )

    return E(), C()


if __name__ == "__main__":
    raise SystemExit(main())
