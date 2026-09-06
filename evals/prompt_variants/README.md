# Prompt variants

Candidate prompt sets for `scripts/compare_prompts.py`. Kept after the fact so
that a rejected idea stays rejected rather than being rediscovered and re-paid
for. Each was measured against the shipped prompt over the full 128-case golden
set on `gpt-4o-mini`, both arms sharing one store and one retrieval config.

Run-to-run variance on this set is about two cases, so an accuracy difference
of one case decides nothing. Read the per-case flips, not the aggregate.

| variant | system prompt | outcome |
|---|---|---|
| `v2-precision` | 951 chars | **Rejected.** Told the model to answer the supported part and name the unsupported part. It named it *using the refusal sentence*, so a partially-correct answer scored as a refusal — the opposite of the intent. Broke `artifacts-and-free-storage`. |
| `v3-precision` | 1029 chars | **Rejected on cost.** Fixed the backfire above and reached 98.4% accuracy and 98.4% citations, but cost **+30%** per query for an accuracy difference inside the noise band. Most of its length was the two rules that did not work. |
| `v4-minimal` | 439 chars | **Shipped as prompt v2.** Keeps only the rule that repeatedly worked — do simple comparison or arithmetic rather than declining — for **+5.7%** cost. Fixed `archive-minimum-duration` in every run it appeared in, and took citation rate to 100%. |

What none of them fixed is `near-cr-maxconc`: asked for Cloud Run's maximum
concurrency, the model reports the documented *default* of eighty as a maximum.
Both an explicit "a default is not a limit" instruction (v2, v3) and the
shipped prompt fail it, and it is intermittent — the shipped prompt refused it
correctly in one run out of five. It is a reading error on the one passage that
should have been retrieved, which is the same wall the similarity floor hits;
see "Why the floor stops here" in the README.
