"""Prompt versioning: the lock, the render contract, and what gets tracked.

The gate here is `test_lock_matches_the_template_files`. Editing a prompt
without re-pinning the lock fails the build, which is the whole mechanism:
it converts a silent prompt change into a reviewable one.
"""

from __future__ import annotations

import json

import pytest

from app.evaluation.dataset import GoldenCase
from app.evaluation.metrics import REFUSAL_MARKER
from app.evaluation.runner import run_evaluation
from app.rag.prompts import (
    LOCK_PATH,
    PROMPTS,
    PromptError,
    PromptSet,
    PromptTemplate,
    build_lock,
    fingerprint,
    load_prompt_set,
    load_templates,
)


def _write_set(tmp_path, texts: dict[str, str], lock: dict | None):
    template_dir = tmp_path / "prompt_templates"
    template_dir.mkdir()
    for name, text in texts.items():
        (template_dir / f"{name}.txt").write_text(text)
    lock_path = tmp_path / "prompts.lock.json"
    if lock is not None:
        lock_path.write_text(json.dumps(lock))
    return template_dir, lock_path


VALID = {
    # Must promise the refusal sentence: prompts.py rejects a set that does not.
    "answer_system": (
        "Answer only from the passages. If they do not contain the answer, reply "
        'exactly: "I don\'t know based on the provided documents."'
    ),
    "answer_user": "Context:\n$context\n\nQuestion: $question",
    "no_context_answer": "I don't know based on the provided documents.",
}


# -- the lock ---------------------------------------------------------------
def test_lock_matches_the_template_files():
    """Fails when a template was edited without `make prompts-lock VERSION=...`.

    A hand-maintained version number goes stale silently; this is what stops
    two different prompts both claiming to be v1 in eval history.
    """
    lock = json.loads(LOCK_PATH.read_text())
    expected = build_lock(PROMPTS, lock["version"])
    assert lock == expected, (
        "prompt templates have drifted from the lock. If the change is "
        "intended, run: make prompts-lock VERSION=<next version>"
    )
    assert PROMPTS.locked
    assert PROMPTS.tracked_version == lock["version"]


def test_edited_template_is_dirty_and_refingerprinted(tmp_path):
    lock = build_lock(PromptSet(version="v1", templates=load_templates()), "v1")
    template_dir, lock_path = _write_set(tmp_path, VALID, lock)
    (template_dir / "answer_user.txt").write_text("Passages:\n$context\n\nAsk: $question")

    edited = load_prompt_set(template_dir, lock_path)

    assert edited.locked is False
    assert edited.tracked_version == "v1-dirty"
    assert edited.fingerprint != PROMPTS.fingerprint


def test_missing_lock_degrades_to_unversioned(tmp_path):
    """A missing lock must not stop the service -- it would leave no way to
    run the script that writes one."""
    template_dir, lock_path = _write_set(tmp_path, VALID, lock=None)

    prompts = load_prompt_set(template_dir, lock_path)

    assert prompts.tracked_version == "unversioned-dirty"


def test_set_fingerprint_changes_when_any_member_changes(tmp_path):
    lock = build_lock(PromptSet(version="v1", templates=load_templates()), "v1")
    template_dir, lock_path = _write_set(tmp_path, VALID, lock)
    before = load_prompt_set(template_dir, lock_path).fingerprint

    (template_dir / "no_context_answer.txt").write_text(
        "I don't know based on the provided documents. Nothing matched."
    )

    assert load_prompt_set(template_dir, lock_path).fingerprint != before


# -- loading and rendering --------------------------------------------------
def test_trailing_newline_does_not_change_the_prompt(tmp_path):
    texts = dict(VALID, answer_system=VALID["answer_system"] + "\n\n")
    template_dir, lock_path = _write_set(tmp_path, texts, lock=None)

    loaded = load_prompt_set(template_dir, lock_path)

    assert loaded["answer_system"].text == VALID["answer_system"]
    assert loaded["answer_system"].fingerprint == fingerprint(VALID["answer_system"])


def test_renamed_placeholder_fails_at_load(tmp_path):
    texts = dict(VALID, answer_user="Context:\n$context\n\nQuery: $query")
    template_dir, lock_path = _write_set(tmp_path, texts, lock=None)

    with pytest.raises(PromptError, match="placeholders"):
        load_prompt_set(template_dir, lock_path)


def test_bare_dollar_sign_fails_at_load(tmp_path):
    """A literal `$` is a ValueError at substitution time, which on the request
    path is a 500 on every query. Catch it at import instead."""
    texts = dict(
        VALID, answer_system=VALID["answer_system"] + " Budgets are capped at $0.25 a day."
    )
    template_dir, lock_path = _write_set(tmp_path, texts, lock=None)

    with pytest.raises(PromptError, match="dollar"):
        load_prompt_set(template_dir, lock_path)


def test_rewording_the_refusal_sentence_fails_at_load(tmp_path):
    """The contract that scoring and production monitoring both depend on.

    Rewording the refusal does not break answering -- it breaks *detection*.
    Refusal accuracy and the production refusal rate would both silently read
    zero, and both would look like a model regression rather than a prompt edit.
    """
    texts = dict(
        VALID,
        answer_system="Answer only from the passages. Otherwise say you cannot help.",
    )
    template_dir, lock_path = _write_set(tmp_path, texts, lock=None)

    with pytest.raises(PromptError, match="refusal sentence"):
        load_prompt_set(template_dir, lock_path)


def test_empty_template_fails_at_load(tmp_path):
    template_dir, lock_path = _write_set(tmp_path, dict(VALID, answer_system=""), None)

    with pytest.raises(PromptError, match="empty"):
        load_prompt_set(template_dir, lock_path)


def test_missing_template_file_is_fatal(tmp_path):
    texts = {k: v for k, v in VALID.items() if k != "answer_user"}
    template_dir, lock_path = _write_set(tmp_path, texts, lock=None)

    with pytest.raises(PromptError, match="cannot read"):
        load_prompt_set(template_dir, lock_path)


def test_render_substitutes_and_reports_missing_values():
    rendered = PROMPTS.render("answer_user", context="[1] Cloud Run", question="Where?")
    assert "[1] Cloud Run" in rendered
    assert "Where?" in rendered

    with pytest.raises(PromptError, match="question"):
        PROMPTS.render("answer_user", context="[1] Cloud Run")


def test_unknown_prompt_name_is_reported():
    with pytest.raises(PromptError, match="unknown prompt"):
        PROMPTS["summarise"]


def test_braces_survive_substitution(tmp_path):
    """The reason this uses string.Template rather than str.format: a prompt
    picks up a JSON example sooner or later, and `{` would be a KeyError."""
    template = PromptTemplate(
        name="t", text='Reply as {"answer": "..."} using $context', placeholders=("context",)
    )

    assert template.render(context="[1] x") == 'Reply as {"answer": "..."} using [1] x'


# -- the cross-file contract ------------------------------------------------
def test_refusal_marker_matches_both_places_it_is_promised():
    """`REFUSAL_MARKER` scores every refusal case. If a prompt edit rewords the
    refusal sentence, refusal accuracy silently reads 0% and the eval blames
    the model. Nothing else in the suite checks these three files agree.
    """
    assert REFUSAL_MARKER in PROMPTS["answer_system"].text.lower()
    assert REFUSAL_MARKER in PROMPTS["no_context_answer"].text.lower()


# -- what reaches MLflow ----------------------------------------------------
class _CapturingTracker:
    """Records payloads instead of writing them, so the eval run can be
    inspected without standing up an MLflow backend."""

    def __init__(self) -> None:
        self.payloads: list = []

    def log_run(self, run_name: str, payload) -> str:
        self.payloads.append((run_name, payload))
        return "run-id"


def test_eval_run_stores_the_prompt_text_itself(pipeline):
    """The eval run is the one place the words are kept. Query runs record the
    fingerprint only, so this is what you open when accuracy moved."""
    tracker = _CapturingTracker()
    pipeline._tracker = tracker
    pipeline.ingest([("Cloud Run scales to zero when idle.", "cloud-run", {})])
    cases = [
        GoldenCase(
            id="c1",
            question="What does Cloud Run do when idle?",
            expected_doc="cloud-run",
        )
    ]

    run_evaluation(pipeline, cases, log_to_mlflow=True)

    _, payload = tracker.payloads[-1]
    assert payload.params["prompt_version"] == PROMPTS.tracked_version
    assert payload.params["prompt_fingerprint"] == PROMPTS.fingerprint
    assert payload.artifacts["prompts/answer_system.txt"] == PROMPTS["answer_system"].text
    assert set(PROMPTS.as_artifacts()) <= set(payload.artifacts)


def test_ready_reports_the_prompt_version(client):
    body = client.get("/ready").json()

    assert body["prompts"]["version"] == PROMPTS.tracked_version
    assert body["prompts"]["fingerprint"] == PROMPTS.fingerprint
    assert body["prompts"]["locked"] is True
