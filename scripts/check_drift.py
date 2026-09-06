#!/usr/bin/env python
"""Compare recent production traffic against the quality baseline.

    python scripts/check_drift.py            # last 500 /query runs
    python scripts/check_drift.py --limit 100

Free: it reads MLflow, calls no model and spends nothing. Exits non-zero when
anything is at ALERT, so it can be wired to a schedule.

The baseline comes from `make eval` (`evals/baseline.json`). It describes what
the signals look like when every question is answerable, so a healthy window
should sit near it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.config import get_settings  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.monitoring.drift import Baseline, compare, format_report, summarise  # noqa: E402
from app.monitoring.source import (  # noqa: E402
    load_recent_queries,
    load_snapshot_queries,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="evals/baseline.json")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--source",
        choices=("auto", "local", "snapshots"),
        default="auto",
        help="auto reads merged GCS shards when GCS_BUCKET is set, else local",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(level="WARNING", service_name=settings.service_name)

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(
            f"No baseline at {baseline_path}. Run `make eval` to record one.",
            file=sys.stderr,
        )
        return 2

    baseline = Baseline.load(baseline_path)
    # Production writes one shard per instance, so a local read would see only
    # whatever this machine recorded -- which on a laptop is nothing.
    use_snapshots = args.source == "snapshots" or (
        args.source == "auto" and bool(settings.gcs_bucket)
    )
    if use_snapshots:
        records = load_snapshot_queries(settings, limit=args.limit)
    else:
        records = load_recent_queries(settings, limit=args.limit)
    if not records:
        print(
            "No /query runs recorded yet (or none since the refused metric "
            "was added). Nothing to compare.",
            file=sys.stderr,
        )
        return 0

    window = summarise(records)
    findings = compare(baseline, window)
    print(format_report(baseline, window, findings))
    return 1 if any(f.severity == "alert" for f in findings) else 0


if __name__ == "__main__":
    logging.captureWarnings(True)
    raise SystemExit(main())
