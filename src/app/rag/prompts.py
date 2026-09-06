"""Versioned prompts, so a prompt change is visible in eval history.

Until this existed the prompts were string constants inside the pipeline,
which made them invisible to everything MLflow recorded. Two evaluations three
weeks apart could differ by four points of accuracy with nothing in the
tracking data to say the prompt had been rewritten in between. A prompt is a
parameter of this system in exactly the way `chunk_size` is, and it was the
only one not being logged.

Three pieces make a change impossible to miss:

* **The text lives in files** (`prompt_templates/*.txt`), so a prompt edit is a
  readable diff rather than a change buried in escaped Python string
  concatenation.
* **Every template is content-addressed.** The fingerprint is a hash of the
  text, so unlike a hand-maintained version number it cannot be forgotten.
  That hash is what actually ties a recorded run to the words that produced it.
* **A lock file pins the declared version to those hashes.** Editing a template
  without bumping the version fails `tests/test_prompts.py`. That is the point:
  it makes a prompt change deliberate and reviewable rather than silent.

A stale lock does not stop the service -- it is an editorial mistake, not an
outage -- but the version reported to MLflow is suffixed `-dirty`, so no run is
ever attributed to a clean version it did not actually use.

Two deliberate limitations. The lock proves the text changed, not that anyone
re-measured it: only `make eval` can say whether the new wording is better.
And this versions the prompt, not the model -- the same prompt against a new
`gpt-4o-mini` snapshot is a different system, and `chat_model` is the param
that records that.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from string import Template

logger = logging.getLogger(__name__)

# The sentence the system prompt dictates for a refusal, and the opening of
# the canned no-context reply. It lives here because it is a property of the
# prompt contract, not of any one consumer: the pipeline uses it to record
# whether a request was refused, and the eval uses it to score refusal
# accuracy. Both were previously reading their own copy of the same string.
# Matched as a phrase rather than a whole sentence, so trailing punctuation or
# added detail does not break detection.
REFUSAL_MARKER = "i don't know based on the provided documents"

TEMPLATE_DIR = Path(__file__).parent / "prompt_templates"
LOCK_PATH = Path(__file__).parent / "prompts.lock.json"

# name -> the placeholders its text must contain. Declared here rather than
# inferred from the file so that renaming `$question` to `$query` fails at
# import time instead of on the first request that needs it.
TEMPLATE_PLACEHOLDERS: dict[str, tuple[str, ...]] = {
    "answer_system": (),
    "answer_user": ("context", "question"),
    # Not a prompt but the canned reply used when retrieval returns nothing.
    # It is versioned alongside the prompts because it is bound by the same
    # contract: `metrics.REFUSAL_MARKER` has to match both this sentence and
    # the one the system prompt dictates, or refusal accuracy silently reads 0.
    "no_context_answer": (),
}


def fingerprint(text: str) -> str:
    """Short content hash.

    12 hex characters is 48 bits: unambiguous at the scale of one repository's
    prompt history, and short enough to read in an MLflow params table.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class PromptError(RuntimeError):
    """A prompt file is missing or malformed.

    Unlike tracking, snapshots and cost estimation, this does not fail open.
    There is no degraded mode for a service that cannot build its prompt, and
    a malformed template surfaces at import -- in CI, before deploy -- rather
    than as a KeyError on a user's request.
    """


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    text: str
    placeholders: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.text)

    def render(self, **values: str) -> str:
        missing = sorted(set(self.placeholders) - set(values))
        if missing:
            raise PromptError(f"prompt {self.name!r} is missing values for {missing}")
        return Template(self.text).substitute(**values)


@dataclass(frozen=True)
class PromptSet:
    """The prompts one build of the service answers with, as a unit.

    Versioned as a set rather than per template because they are only
    meaningful together: the system prompt dictates a refusal sentence that
    `no_context_answer` has to match, so a run is produced by the combination,
    not by any one file.
    """

    version: str
    templates: dict[str, PromptTemplate]
    # False when the files on disk no longer match the lock.
    locked: bool = True

    @property
    def fingerprint(self) -> str:
        joined = "\n".join(
            f"{name}:{template.fingerprint}"
            for name, template in sorted(self.templates.items())
        )
        return fingerprint(joined)

    @property
    def tracked_version(self) -> str:
        """The version string recorded against a run.

        Borrowed from `git describe --dirty`: an edited working copy must never
        claim to be the released version, because the whole purpose here is
        that eval history can be trusted.
        """
        return self.version if self.locked else f"{self.version}-dirty"

    def __getitem__(self, name: str) -> PromptTemplate:
        try:
            return self.templates[name]
        except KeyError:
            raise PromptError(f"unknown prompt {name!r}") from None

    def render(self, name: str, **values: str) -> str:
        return self[name].render(**values)

    def describe(self) -> dict:
        return {
            "version": self.tracked_version,
            "fingerprint": self.fingerprint,
            "locked": self.locked,
            "templates": {
                name: template.fingerprint for name, template in self.templates.items()
            },
        }

    def as_artifacts(self) -> dict[str, str]:
        """The full prompt text, for MLflow to store beside an eval run.

        Query runs get the fingerprint only. Artifacts are the one thing that
        writes to GCS on the request path, and duplicating three unchanging
        files onto every query would be a per-request cost for text that is
        already recoverable -- from git, or from the eval run that measured it.
        """
        return {
            f"prompts/{name}.txt": template.text for name, template in self.templates.items()
        }


def load_templates(template_dir: Path = TEMPLATE_DIR) -> dict[str, PromptTemplate]:
    """Read and validate the template files. Separate from the lock so that
    `scripts/lock_prompts.py` can write the very lock this module requires."""
    templates: dict[str, PromptTemplate] = {}
    for name, placeholders in TEMPLATE_PLACEHOLDERS.items():
        path = template_dir / f"{name}.txt"
        try:
            # Stripped so a file can end with the newline every editor adds
            # without that newline becoming part of the prompt -- and, more to
            # the point, without it changing the fingerprint.
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise PromptError(f"cannot read prompt template {path}: {exc}") from exc
        _validate(name, text, placeholders)
        templates[name] = PromptTemplate(name=name, text=text, placeholders=placeholders)
    return templates


def load_prompt_set(
    template_dir: Path = TEMPLATE_DIR, lock_path: Path = LOCK_PATH
) -> PromptSet:
    templates = load_templates(template_dir)
    version, locked = _read_lock(lock_path, templates)
    return PromptSet(version=version, templates=templates, locked=locked)


# Templates whose text must contain REFUSAL_MARKER. Enforced at load because
# a prompt edit that rewords the refusal sentence does not break answering --
# it breaks *detection*, silently. Refusal accuracy would read 0%, production
# refusal rate would read 0%, and both would look like a model regression.
MUST_PROMISE_REFUSAL = ("answer_system", "no_context_answer")


def _validate(name: str, text: str, placeholders: tuple[str, ...]) -> None:
    if not text:
        raise PromptError(f"prompt template {name!r} is empty")
    if name in MUST_PROMISE_REFUSAL and REFUSAL_MARKER not in text.lower():
        raise PromptError(
            f"prompt template {name!r} no longer contains the refusal sentence "
            f"{REFUSAL_MARKER!r}; refusal detection would silently read zero"
        )
    template = Template(text)
    if not template.is_valid():
        # A bare `$` -- in a price, say -- is a ValueError at substitution
        # time, which on the request path means a 500 for every query.
        raise PromptError(
            f"prompt template {name!r} has a malformed placeholder; "
            "write a literal dollar sign as $$"
        )
    found = set(template.get_identifiers())
    if found != set(placeholders):
        raise PromptError(
            f"prompt template {name!r} declares placeholders {sorted(placeholders)} "
            f"but its text uses {sorted(found)}"
        )


UNVERSIONED = "unversioned"


def _read_lock(lock_path: Path, templates: dict[str, PromptTemplate]) -> tuple[str, bool]:
    """Return (declared version, whether the files still match the lock).

    A missing or unreadable lock is not fatal, unlike a missing template. The
    lock is a review discipline rather than a runtime dependency, and making it
    fatal would leave no way to run the script that regenerates it. The run is
    still recorded honestly: it is tagged `unversioned-dirty`, and
    `tests/test_prompts.py` is what stops that reaching a release.
    """
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "prompt lock unreadable; runs will be tagged unversioned",
            extra={"lock_path": str(lock_path), "error": str(exc)},
        )
        return UNVERSIONED, False

    version = str(lock.get("version", "")).strip() or UNVERSIONED
    locked_hashes = lock.get("templates", {})
    drifted = sorted(
        name
        for name, template in templates.items()
        if locked_hashes.get(name) != template.fingerprint
    )
    if drifted:
        logger.warning(
            "prompt templates differ from the lock; runs will be tagged -dirty",
            extra={"drifted": drifted, "locked_version": version},
        )
    return version, not drifted


def build_lock(prompt_set: PromptSet, version: str) -> dict:
    return {
        "version": version,
        "fingerprint": prompt_set.fingerprint,
        "templates": {
            name: template.fingerprint
            for name, template in sorted(prompt_set.templates.items())
        },
    }


def write_lock(prompt_set: PromptSet, version: str, lock_path: Path = LOCK_PATH) -> dict:
    lock = build_lock(prompt_set, version)
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


PROMPTS = load_prompt_set()
