"""The golden dataset: questions with known-correct behaviour.

JSONL rather than a Python module so that non-engineers can add cases, and so
that a diff on the dataset is readable in review. Every field is optional
except the question, because a case that only asserts "this must be refused"
is as valuable as one that asserts an exact fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    # Which corpus document should be retrieved. None means "none should be" --
    # the out-of-corpus refusal cases.
    expected_doc: str | None = None
    # Substrings that a correct answer should contain. Matched case-insensitively
    # and treated as alternatives, not a conjunction: a model may legitimately
    # write "2 million" or "two million", and failing it for picking one would
    # measure phrasing rather than correctness.
    expected_keywords: list[str] = field(default_factory=list)
    # Whether the corpus can answer this at all. False means the correct
    # behaviour is an explicit "I don't know".
    must_be_grounded: bool = True
    notes: str = ""

    @property
    def is_refusal_case(self) -> bool:
        return not self.must_be_grounded


def load_cases(path: str | Path) -> list[GoldenCase]:
    """Read a JSONL golden set. Blank lines and # comments are skipped."""
    cases: list[GoldenCase] = []
    seen: set[str] = set()

    for line_number, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

        case_id = record.get("id") or f"case-{line_number}"
        if case_id in seen:
            # Duplicate ids would silently collide in the per-case report and
            # make a regression impossible to attribute.
            raise ValueError(f"{path}:{line_number}: duplicate case id {case_id!r}")
        seen.add(case_id)

        question = record.get("question", "").strip()
        if not question:
            raise ValueError(f"{path}:{line_number}: case {case_id!r} has no question")

        cases.append(
            GoldenCase(
                id=case_id,
                question=question,
                expected_doc=record.get("expected_doc"),
                expected_keywords=list(record.get("expected_keywords") or []),
                must_be_grounded=bool(record.get("must_be_grounded", True)),
                notes=record.get("notes", ""),
            )
        )

    if not cases:
        raise ValueError(f"{path}: golden set is empty")
    return cases
