"""Production drift detection.

The scoring is pure functions over plain records, so none of this needs an
MLflow backend, a model, or a network. `scripts/validate_drift.py` characterises
the detector statistically against real measured signals; these tests pin the
behaviour each of those characteristics depends on.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from app.monitoring.drift import (
    MIN_WINDOW,
    Z_ALERT,
    Baseline,
    QueryRecord,
    compare,
    diagnose,
    ks_statistic,
    summarise,
    two_proportion_z,
)

BASELINE = Baseline(
    n=61,
    refusal_rate=0.033,
    ungrounded_rate=0.0,
    mean_top_similarity=0.536,
    mean_chunks=2.87,
    similarities=[0.5 + (i % 20) / 100 for i in range(61)],
    prompt_version="v1",
    source="test",
)


def records(n, *, refused=False, grounded=True, shift=0.0, chunks=2.9, version="v1"):
    """A window shaped like the baseline, offset by `shift`.

    The similarities have to be spread rather than constant: a window where
    every query scores identically is itself a distribution change, and the KS
    test is right to say so.
    """
    return [
        QueryRecord(
            refused=refused,
            grounded=grounded,
            top_similarity=(0.5 + (i % 20) / 100 + shift) if grounded else 0.0,
            retrieved_chunks=chunks,
            prompt_version=version,
        )
        for i in range(n)
    ]


# -- window guard -----------------------------------------------------------
def test_a_short_window_is_not_scored():
    """A rate difference over a handful of queries means nothing, and a monitor
    that fires on it teaches people to ignore it."""
    window = summarise(records(MIN_WINDOW - 1, refused=True))

    findings = compare(BASELINE, window)

    assert [f.signal for f in findings] == ["window"]
    assert "Not enough traffic" in diagnose(BASELINE, window, findings)


# -- refusal ----------------------------------------------------------------
def test_a_wave_of_refusals_is_flagged():
    window = summarise(records(50, refused=True))

    signals = {f.signal for f in compare(BASELINE, window)}

    assert "refusal_rate" in signals


def test_refusing_less_than_baseline_is_not_drift():
    """One-sided on purpose. Traffic the corpus covers better than the baseline
    is good news, and a monitor that reports it as drift is noise.

    The gap here is deliberately large enough that a two-sided test would fire
    on it -- otherwise this passes whether or not the code is one-sided.
    """
    chatty = replace(BASELINE, refusal_rate=0.30, n=200)
    window = summarise(records(200, refused=False))

    assert two_proportion_z(window.refusal_rate, window.n, 0.30, 200) < -Z_ALERT
    assert "refusal_rate" not in {f.signal for f in compare(chatty, window)}


# -- what separates a content gap from off-topic traffic --------------------
def test_refusals_without_rejections_read_as_a_content_gap():
    """The near-miss signature: retrieval works normally, scores look normal,
    and the model refuses anyway because the fact is not in the document."""
    window = summarise(records(100, refused=True, grounded=True))

    findings = compare(BASELINE, window)
    signals = {f.signal for f in findings}

    assert "refusal_rate" in signals
    assert "ungrounded_rate" not in signals
    assert "content gap" in diagnose(BASELINE, window, findings)


def test_refusals_with_rejections_read_as_off_topic_traffic():
    """The floor rejects subjects the corpus does not cover, and never rejects
    near-misses -- which is what makes this distinguishable at all."""
    window = summarise(records(100, refused=True, grounded=False, chunks=0.0))

    findings = compare(BASELINE, window)
    signals = {f.signal for f in findings}

    assert "ungrounded_rate" in signals
    assert "off-topic" in diagnose(BASELINE, window, findings)


def test_fewer_chunks_with_refusals_reads_as_a_retrieval_regression():
    window = summarise(records(100, refused=True, grounded=True, chunks=1.2))

    findings = compare(BASELINE, window)

    assert "retrieved_chunks" in {f.signal for f in findings}
    assert "retrieval regression" in diagnose(BASELINE, window, findings)


def test_healthy_traffic_produces_no_findings():
    window = summarise(records(100, refused=False, chunks=2.9))

    findings = compare(BASELINE, window)

    assert findings == []
    assert "No drift" in diagnose(BASELINE, window, findings)


# -- prompt provenance ------------------------------------------------------
def test_a_prompt_the_baseline_never_measured_is_reported():
    """Comparing traffic to a baseline captured under a different prompt is
    comparing two things, not one."""
    window = summarise(records(50, version="v2"))

    findings = compare(BASELINE, window)

    assert "prompt_version" in {f.signal for f in findings}


# -- statistics -------------------------------------------------------------
def test_ks_statistic_matches_a_hand_computed_case():
    # ECDFs separate by 0.5 at the midpoint of two disjoint, equal-sized sets.
    assert ks_statistic([1.0, 2.0], [1.0, 3.0]) == pytest.approx(0.5)
    assert ks_statistic([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)
    assert ks_statistic([], [1.0]) == 0.0


def test_two_proportion_z_is_signed_and_scales_with_sample_size():
    small = two_proportion_z(0.5, 30, 0.1, 30)
    large = two_proportion_z(0.5, 300, 0.1, 300)

    assert 0 < small < large, "the same gap is stronger evidence in a bigger sample"
    assert two_proportion_z(0.1, 30, 0.5, 30) < 0
    assert two_proportion_z(0.5, 0, 0.1, 30) == 0.0


# -- the baseline file ------------------------------------------------------
def test_baseline_round_trips(tmp_path):
    path = tmp_path / "baseline.json"
    BASELINE.save(path)

    loaded = Baseline.load(path)

    assert loaded == BASELINE


def test_baseline_ignores_fields_it_does_not_know(tmp_path):
    """So a newer baseline file does not crash an older checkout."""
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({**BASELINE.__dict__, "invented_later": 1}))

    assert Baseline.load(path).n == 61


def test_baseline_needs_answerable_cases():
    class Settings:
        min_similarity = 0.28
        min_similarity_ratio = 0.6
        chat_model = "gpt-4o-mini"
        embedding_model = "text-embedding-3-small"

    with pytest.raises(ValueError, match="zero answerable"):
        Baseline.from_eval([], Settings(), {})
