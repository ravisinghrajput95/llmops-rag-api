"""Quality monitoring for live traffic, which has no labels.

The eval measures accuracy against a golden set. Production cannot: nobody
labels a real question, so there is no accuracy number to watch. What can be
watched is whether the *distribution* of what the service sees and does has
moved away from the conditions under which quality was last measured.

**The signal that makes this work.** `refused` is recorded on every query, and
the eval established what it means: the model refused 66 of 67 out-of-corpus
questions and only 2 of 61 answerable ones. A refusal is therefore a fairly
well calibrated statement that the corpus could not answer that question, and
the refusal rate over a window estimates how much of the traffic the corpus
cannot serve. That is a quality measurement derived from an unlabelled stream.

**Two signals, because one cannot tell you why.** Refusal rate rising says
something is wrong; it does not say what. Pairing it with the similarity
distribution does, and the reason comes straight out of the floor research
(README, "Why the floor stops here"): near-miss questions score *identically*
to answerable ones. So

    refusal up,   similarity unchanged  -> traffic is asking things the corpus
                                           does not cover, on topics it does
    refusal up,   similarity down       -> traffic has moved off-topic entirely
    refusal up,   chunks/query down     -> retrieval or its config regressed
    refusal flat, similarity down       -> the corpus or the embedding changed

The first is a content gap, the third is a bug, and they need opposite
responses. Reporting only "drift detected" would leave that undistinguished.

**What this cannot do.** It detects change, not badness. Traffic legitimately
moving to new topics looks exactly like traffic degrading, because without
labels those *are* the same observation. A finding here is a prompt to go and
look at the questions, not a verdict. It also cannot see a fluent, confident,
wrong answer -- the model states it with the same confidence and the same
similarity scores as a right one, and nothing in an unlabelled stream
distinguishes them.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Below this many queries a window is reported as insufficient rather than
# scored. Measured, not guessed: at n=30 the false-positive rate of the whole
# check sits near the nominal 5%, and below n=20 it climbs while the power to
# detect real drift collapses. See scripts/validate_drift.py.
MIN_WINDOW = 30

# Two-sided normal critical values, for the proportion test.
Z_WARN = 1.960  # alpha 0.05
Z_ALERT = 2.576  # alpha 0.01

# Kolmogorov-Smirnov coefficients for the same two levels.
KS_WARN = 1.358
KS_ALERT = 1.628


@dataclass(frozen=True)
class QueryRecord:
    """One production query, as recorded by the tracker."""

    refused: bool
    grounded: bool
    top_similarity: float
    retrieved_chunks: float
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    prompt_version: str = ""
    # Epoch millis. Only used to order runs merged from different shards, so
    # that `limit` still means "the most recent N" across all of them.
    started_at: float = 0.0


@dataclass
class WindowStats:
    n: int
    refusal_rate: float
    ungrounded_rate: float
    mean_top_similarity: float
    mean_chunks: float
    mean_cost_usd: float
    p95_latency_ms: float
    similarities: list[float] = field(default_factory=list)
    prompt_versions: list[str] = field(default_factory=list)


@dataclass
class Baseline:
    """Reference conditions, captured from an eval run over answerable cases.

    Only the grounded cases are used. The golden set is deliberately about half
    out-of-corpus, so its overall refusal rate is a property of the test set
    and says nothing about healthy traffic. The answerable half is the right
    reference: it describes what the signals look like when every question is
    one the corpus can serve.
    """

    n: int
    refusal_rate: float
    ungrounded_rate: float
    mean_top_similarity: float
    mean_chunks: float
    similarities: list[float]
    prompt_version: str = ""
    prompt_fingerprint: str = ""
    min_similarity: float = 0.0
    min_similarity_ratio: float = 0.0
    chat_model: str = ""
    embedding_model: str = ""
    source: str = ""

    @classmethod
    def from_eval(cls, scores, settings, prompts, source: str = "") -> Baseline:
        """Build the reference from the answerable cases of an eval run.

        `scores` must already be filtered to grounded cases -- passing the
        whole golden set would bake its ~50% out-of-corpus share into the
        reference refusal rate and make every healthy window look better than
        baseline.
        """
        n = len(scores)
        if n == 0:
            raise ValueError("cannot build a baseline from zero answerable cases")
        sims = [float(s.top_similarity) for s in scores]
        return cls(
            n=n,
            refusal_rate=sum(bool(s.refused) for s in scores) / n,
            ungrounded_rate=sum(1 for s in scores if not s.retrieved_chunks) / n,
            mean_top_similarity=sum(sims) / n,
            mean_chunks=sum(int(s.retrieved_chunks) for s in scores) / n,
            similarities=[round(v, 4) for v in sims],
            prompt_version=prompts.get("version", ""),
            prompt_fingerprint=prompts.get("fingerprint", ""),
            min_similarity=settings.min_similarity,
            min_similarity_ratio=settings.min_similarity_ratio,
            chat_model=settings.chat_model,
            embedding_model=settings.embedding_model,
            source=source,
        )

    @classmethod
    def load(cls, path: str | Path) -> Baseline:
        data = json.loads(Path(path).read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


@dataclass(frozen=True)
class Finding:
    signal: str
    severity: str  # "warn" | "alert"
    detail: str


def summarise(records: list[QueryRecord]) -> WindowStats:
    n = len(records)
    if n == 0:
        return WindowStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sims = [r.top_similarity for r in records]
    latencies = sorted(r.latency_ms for r in records)
    return WindowStats(
        n=n,
        refusal_rate=sum(r.refused for r in records) / n,
        ungrounded_rate=sum(not r.grounded for r in records) / n,
        mean_top_similarity=sum(sims) / n,
        mean_chunks=sum(r.retrieved_chunks for r in records) / n,
        mean_cost_usd=sum(r.cost_usd for r in records) / n,
        p95_latency_ms=latencies[min(n - 1, int(round(0.95 * (n - 1))))],
        similarities=sims,
        prompt_versions=sorted({r.prompt_version for r in records if r.prompt_version}),
    )


def two_proportion_z(rate_a: float, n_a: int, rate_b: float, n_b: int) -> float:
    """Standard two-proportion z. Zero when neither sample has any variance."""
    if n_a == 0 or n_b == 0:
        return 0.0
    pooled = (rate_a * n_a + rate_b * n_b) / (n_a + n_b)
    denom = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if denom == 0:
        return 0.0
    return (rate_a - rate_b) / denom


def ks_statistic(a: list[float], b: list[float]) -> float:
    """Two-sample Kolmogorov-Smirnov D: the largest gap between two ECDFs."""
    if not a or not b:
        return 0.0
    sa, sb = sorted(a), sorted(b)
    i = j = 0
    d = 0.0
    while i < len(sa) and j < len(sb):
        value = min(sa[i], sb[j])
        while i < len(sa) and sa[i] <= value:
            i += 1
        while j < len(sb) and sb[j] <= value:
            j += 1
        d = max(d, abs(i / len(sa) - j / len(sb)))
    return d


def ks_threshold(n: int, m: int, coefficient: float) -> float:
    if n == 0 or m == 0:
        return math.inf
    return coefficient * math.sqrt((n + m) / (n * m))


def compare(baseline: Baseline, window: WindowStats) -> list[Finding]:
    """Score a production window against the reference. Empty == no drift."""
    if window.n < MIN_WINDOW:
        return [
            Finding(
                "window",
                "warn",
                f"only {window.n} queries; {MIN_WINDOW} needed before a rate "
                "difference means anything. Not scored.",
            )
        ]

    findings: list[Finding] = []

    # Refusal is one-sided on purpose. Refusing *less* than the reference is not
    # a regression -- it means the traffic is well covered by the corpus.
    z = two_proportion_z(window.refusal_rate, window.n, baseline.refusal_rate, baseline.n)
    if z >= Z_WARN:
        findings.append(
            Finding(
                "refusal_rate",
                "alert" if z >= Z_ALERT else "warn",
                f"{window.refusal_rate:.1%} of queries refused against a "
                f"{baseline.refusal_rate:.1%} baseline (z={z:.2f})",
            )
        )

    # The floor rejects topically unrelated questions outright and never
    # rejects near-misses, so this is what separates "users left the subject"
    # from "the corpus has a hole". It carries the distinction the similarity
    # test was too under-powered to make; see scripts/validate_drift.py.
    zu = two_proportion_z(
        window.ungrounded_rate, window.n, baseline.ungrounded_rate, baseline.n
    )
    if zu >= Z_WARN:
        findings.append(
            Finding(
                "ungrounded_rate",
                "alert" if zu >= Z_ALERT else "warn",
                f"{window.ungrounded_rate:.1%} of queries retrieved nothing at all "
                f"against a {baseline.ungrounded_rate:.1%} baseline (z={zu:.2f})",
            )
        )

    d = ks_statistic(window.similarities, baseline.similarities)
    warn_at = ks_threshold(len(window.similarities), len(baseline.similarities), KS_WARN)
    alert_at = ks_threshold(len(window.similarities), len(baseline.similarities), KS_ALERT)
    if d >= warn_at:
        findings.append(
            Finding(
                "top_similarity",
                "alert" if d >= alert_at else "warn",
                f"retrieval score distribution has moved (KS D={d:.3f} against "
                f"a {warn_at:.3f} threshold); mean "
                f"{window.mean_top_similarity:.3f} against "
                f"{baseline.mean_top_similarity:.3f}",
            )
        )

    # A drop in chunks per query means the filters are keeping less than they
    # were calibrated to. That is a config or corpus change, not traffic.
    if baseline.mean_chunks > 0:
        shortfall = 1 - (window.mean_chunks / baseline.mean_chunks)
        if shortfall >= 0.25:
            findings.append(
                Finding(
                    "retrieved_chunks",
                    "alert" if shortfall >= 0.40 else "warn",
                    f"{window.mean_chunks:.2f} chunks per query against a "
                    f"{baseline.mean_chunks:.2f} baseline ({shortfall:.0%} fewer)",
                )
            )

    # Not drift, but the thing you most want to know when reading a drift
    # report: whether the prompt changed underneath the comparison.
    drifted_prompt = [v for v in window.prompt_versions if v and v != baseline.prompt_version]
    if drifted_prompt:
        findings.append(
            Finding(
                "prompt_version",
                "warn",
                f"window served {drifted_prompt} but the baseline was measured "
                f"on {baseline.prompt_version!r}; quality was never measured "
                "for what is running",
            )
        )
    return findings


def diagnose(baseline: Baseline, window: WindowStats, findings: list[Finding]) -> str:
    """Name the likeliest cause. The point of collecting two signals.

    Near-miss questions score like answerable ones, so refusals rising with a
    *steady* similarity distribution is the signature of a content gap, while
    the same rise with falling scores is traffic leaving the subject area.
    Those need opposite responses -- write documents, or check the retriever --
    and the pair distinguishes them where either alone cannot.
    """
    signals = {f.signal for f in findings}
    if not signals:
        return "No drift. Traffic resembles the conditions quality was measured under."
    if "window" in signals:
        return "Not enough traffic to say anything yet."

    refusing = "refusal_rate" in signals
    ungrounded = "ungrounded_rate" in signals
    scores_moved = "top_similarity" in signals
    fewer_chunks = "retrieved_chunks" in signals
    scores_fell = window.mean_top_similarity < baseline.mean_top_similarity

    # Ordered by how specific the evidence is, not by severity. Retrieval
    # returning less is checked first because it is the one cause that is a bug
    # rather than a fact about the traffic.
    if refusing and fewer_chunks and not ungrounded:
        return (
            "Likely a retrieval regression: fewer chunks are surviving the "
            "filters and refusals have risen with them. Check min_similarity, "
            "min_similarity_ratio and that the corpus is still ingested."
        )
    if refusing and ungrounded:
        return (
            "Likely traffic drift off-topic: questions are being rejected by "
            "the similarity floor before the model sees them, which is what it "
            "does to subjects the corpus does not cover at all. The corpus may "
            "not be the right one for what users are now asking."
        )
    if refusing and scores_moved and scores_fell:
        return (
            "Likely traffic drift off-topic: questions are scoring lower "
            "against the corpus and being refused. The corpus may simply not "
            "be the right one for what users are now asking."
        )
    if refusing and not scores_moved:
        return (
            "Likely a content gap: questions still look like corpus topics -- "
            "they retrieve normally and score normally -- and are being refused "
            "anyway. That is the near-miss signature: the right document comes "
            "back and does not contain the fact. Read the refused questions; the "
            "fix is a document, not a threshold."
        )
    if scores_moved and not refusing:
        return (
            "Retrieval scores moved without more refusals. Usually the corpus "
            "changed, or the embedding model did -- in which case "
            "min_similarity needs recalibrating, since it is model-dependent."
        )
    return "Drift detected; see the findings above."


def format_report(baseline: Baseline, window: WindowStats, findings: list[Finding]) -> str:
    lines = [
        "",
        "Production drift check",
        "=" * 66,
        f"  window             {window.n} queries",
        f"  baseline           {baseline.n} answerable eval cases ({baseline.source})",
        "",
        f"{'':21}{'window':>12}{'baseline':>12}",
        f"  refusal rate       {window.refusal_rate:>11.1%}" f"{baseline.refusal_rate:>12.1%}",
        f"  ungrounded rate    {window.ungrounded_rate:>11.1%}"
        f"{baseline.ungrounded_rate:>12.1%}",
        f"  mean top score     {window.mean_top_similarity:>11.3f}"
        f"{baseline.mean_top_similarity:>12.3f}",
        f"  chunks per query   {window.mean_chunks:>11.2f}{baseline.mean_chunks:>12.2f}",
        f"  mean cost/query    ${window.mean_cost_usd:.6f}",
        f"  p95 latency        {window.p95_latency_ms:.0f} ms",
        "",
    ]
    if findings:
        lines.append(f"Findings ({len(findings)}):")
        for finding in findings:
            lines.append(
                f"  [{finding.severity.upper():<5}] {finding.signal}: {finding.detail}"
            )
    else:
        lines.append("No findings.")
    lines += ["", diagnose(baseline, window, findings), ""]
    return "\n".join(lines)
