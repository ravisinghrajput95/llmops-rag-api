#!/usr/bin/env python
"""Re-pin the prompt lock after editing a template.

    python scripts/lock_prompts.py --version v2

Costs nothing and touches nothing but `src/app/rag/prompts.lock.json`. Run it
when `make test` tells you the templates have drifted from the lock -- and
give the change a new version, because that string is how a run in MLflow six
weeks from now says which words produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.rag.prompts import (  # noqa: E402
    LOCK_PATH,
    PromptSet,
    build_lock,
    load_templates,
)


def _current_version() -> str:
    try:
        return str(json.loads(LOCK_PATH.read_text()).get("version", "")) or "(none)"
    except (OSError, json.JSONDecodeError):
        return "(none)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="version to record, e.g. v2. Bump it whenever the text changes.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift without writing (what the test does).",
    )
    args = parser.parse_args()

    templates = load_templates()
    prompt_set = PromptSet(version=args.version, templates=templates)
    lock = build_lock(prompt_set, args.version)

    if args.check:
        on_disk = json.loads(LOCK_PATH.read_text())
        if on_disk == lock:
            print(f"prompt lock is current ({args.version})")
            return 0
        print("prompt lock is stale:", json.dumps(lock, indent=2))
        return 1

    previous = _current_version()
    LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"prompt lock updated: {previous} -> {args.version} ({lock['fingerprint']})")
    for name, digest in lock["templates"].items():
        print(f"  {name:<20} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
