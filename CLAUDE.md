# CLAUDE.md

Notes for working in this repo. Covers the things that are costly to
rediscover — not the structure, which the README already lays out.

## What this project is

A cost-aware RAG API: FastAPI + Chroma + OpenAI on Cloud Run, with MLflow
tracking every call's latency, tokens and estimated cost. It was built against
a hard constraint — a small, expiring GCP trial credit balance — and that
constraint is the reason behind most of the design decisions below. Preserve
it. "This would be easier with Cloud SQL" is true and beside the point.

## Commands

```bash
make test     # pytest, no network, no spend
make lint     # ruff check + format --check (line-length 95)
make fmt      # auto-fix
make run      # uvicorn on :8080
make eval     # SPENDS ~$0.002 against real OpenAI -- never run unprompted
make prompts-lock VERSION=v2   # re-pin the prompt lock after editing a prompt
make drift    # compare recent traffic to the eval baseline (free, reads MLflow)
make compare-prompts VARIANT=evals/prompt_variants/<name>   # SPENDS ~2x eval
```

Tests need no API key and no GCP credentials. If a change makes them require
either, the change is wrong.

## The two bills

GCP and OpenAI are separate, and this trips people up constantly:

- **GCP** — every resource here is Always Free at demo scale. `terraform
  destroy` stops all of it.
- **OpenAI** — billed separately. `terraform destroy` stops **none** of it.
  The only bound is `DAILY_BUDGET_USD`, enforced in-process by `SpendGuard`.

Never suggest that tearing down GCP resources protects the OpenAI balance.

## Traps that have already cost time

**WIF pool ids are burned for 30 days.** `terraform destroy` soft-deletes the
workload identity pool, and the id cannot be reused until it expires. Reusing
one makes `terraform apply` fail *partway through* — the pool, its provider and
the impersonation binding fail while all ~24 other resources succeed. It
presents as a mysterious partial deploy, not a name collision. Bump
`wif_pool_id` in `terraform.tfvars` after any destroy. Check what is still
burned with:

```bash
gcloud iam workload-identity-pools list --location=global --show-deleted
```

**The fake embedder is not a small model — it is a bag of words.**
`tests/conftest.py` uses a hashed bag-of-words embedding. It is deterministic
and good enough that retrieval assertions are meaningful, but it scores
"weather in Mumbai" at **0.57** against unrelated text on common-word overlap
alone. Never assert refusal behaviour or semantic similarity thresholds
against it — that measures the fake, not the system. Refusal accuracy is only
measurable in `make eval`.

**Cloud Run is amd64 only.** A plain `docker build` on Apple Silicon produces
an arm64 image that fails at startup with an exec format error. The Makefile
pins `--platform linux/amd64`; keep it.

**The similarity floor cannot reject near-misses, and this is settled.**
A question on a topic the corpus covers whose specific fact it lacks ("Cloud
Run's maximum request timeout", against docs that discuss Cloud Run
concurrency but never state a maximum) scores *identically* to an answerable
question -- AUC 0.518 against 0.500 chance, measured on real embeddings over
61 answerable and 67 out-of-corpus questions. Every distributional
alternative (margin, ratio, z-score, doc concentration) scored worse than
plain top-1. The retriever is not wrong in these cases: it fetches exactly the
document a person would. Similarity measures topical relevance; refusal needs
factual sufficiency. Do not re-litigate this with a cleverer threshold -- the
model does that job, and README "Why the floor stops here" has the numbers.
The floor is tuned for what it *can* do: rejecting topically unrelated
questions, where it separates at AUC 0.927.

**Rate limiter and spend guard are per-process.** With `max-instances=2` the
effective ceilings are up to 2x configured, and they reset on scale-to-zero.
This is deliberate — a shared counter needs Redis or Firestore, neither of
which is free. Do not "fix" it into a distributed limiter without a cost
conversation.

## Conventions that matter here

- **Comments explain *why*, never *what*.** The existing code is consistent
  about this; match it. A comment restating the line below it is a regression.
- **Cost and limitations are documented honestly**, including in module
  docstrings. When you add something with a real limitation, say so plainly
  rather than omitting it. See `persistence.py` for the tone.
- **Failures degrade, they do not cascade.** MLflow logging, GCS snapshots and
  tracker calls all fail open: they log and continue rather than failing the
  request that triggered them.
- Settings are env-driven via `pydantic-settings`, cached with `lru_cache`.
  `conftest.py` sets env vars *before* importing `app.*`, because settings are
  read at import time.

## Testing

- `pytest` is the CI gate and includes a RAG quality gate
  (`tests/test_evaluation.py`) that runs the golden set through the real
  retrieval path with fakes. A chunking or similarity-floor regression fails
  the build here.
- Global limiters are disabled in `conftest.py` (`DAILY_BUDGET_USD=0`,
  `RATE_LIMIT_PER_MINUTE=0`) because they hold process-wide state that would
  otherwise couple every test to how many ran before it. `tests/test_limits.py`
  re-enables them deliberately against purpose-built instances.
- `/ingest` returns **201**, not 200.
- **Editing a prompt fails the suite until you re-lock it.** Prompt text lives
  in `src/app/rag/prompt_templates/*.txt`, hashed into
  `src/app/rag/prompts.lock.json`. `tests/test_prompts.py` compares the two, so
  any edit fails with a message pointing at `make prompts-lock VERSION=<next>`.
  That is the feature: it forces a prompt change to be deliberate and land with
  a new version, because that version string is how an MLflow run months later
  says which words produced it. Bump the version rather than re-pinning the old
  one — re-pinning without bumping makes two different prompts both claim v1.
  Prompts substitute with `string.Template` (`$context`), not `str.format`, and
  their placeholders are declared in `TEMPLATE_PLACEHOLDERS`; changing one
  without the other fails at import, by design.

## Monitoring

- `evals/baseline.json` is the reference for `make drift`, and `make eval`
  rewrites it every run. It is built from the **answerable cases only** — the
  golden set is about half out-of-corpus by construction, so its overall
  refusal rate is a property of the test set, not of healthy traffic. Passing
  the whole set would make every real window look better than baseline.
- `evals/signals.json` is a committed fixture: the measured retrieval signal
  and the real refusal outcome for all 128 golden questions. It exists so
  `scripts/validate_drift.py` can characterise the detector for free. Regenerate
  it when the corpus, the retrieval config or the embedding model changes.
- **The MLflow DB is snapshotted to GCS, and it has to be.** It lives on Cloud
  Run's tmpfs, so without this the run metadata `make drift` reads dies on every
  scale-to-zero and the monitor is blind to production while looking healthy.
- **It is sharded per process, and must stay that way.** Each instance writes
  `snapshots/mlflow/<revision>-<id>.tar.gz` and never restores anyone else's;
  the reader merges. Do not "simplify" this back to one shared object like the
  Chroma snapshot — that pattern is safe there only because ingest is rare. This
  is written every `MLFLOW_SNAPSHOT_EVERY` runs by every instance, so a shared
  object means instances overwriting each other's runs wholesale at
  `max-instances=2`. Uploads are batched because GCS allows 5,000 free class A
  operations a month and a write per query would spend them.
- **Bucket lifecycle rules are prefix-scoped, and must stay that way.** An
  unscoped age rule covers `snapshots/chroma.tar.gz` too, so a service left idle
  longer than the retention window silently loses every ingested document — the
  one object in that bucket that cannot be regenerated. Artifacts and MLflow
  shards expire; the Chroma snapshot does not.
- Runs recorded before the `refused` metric existed are skipped, not defaulted.
  Defaulting them to "not refused" would read a window of old traffic as a
  perfect zero refusal rate.

## Changing a prompt

Write the candidate into `evals/prompt_variants/<name>/` and run
`make compare-prompts VARIANT=...` rather than editing the shipped templates and
running `make eval`. Both arms then share one store and one retrieval config, so
the only variable is the wording.

**Run-to-run variance on the golden set is about two cases** at
`temperature=0.2`. A one-case difference decides nothing; read the per-case
FIXED/BROKEN list, not the aggregate. Two of three candidates tried so far were
rejected — one because telling the model to name the part it could not answer
made it name that part *with the refusal sentence*, scoring a good answer as a
refusal. `evals/prompt_variants/README.md` records what each one measured, so a
rejected idea stays rejected instead of being re-paid for.

Longer prompts cost real money on every query forever: the rejected v3 was
+30%, the shipped v2 is +5.7%. Weigh that against the measured gain.

## Deploying

Terraform in `terraform/` is the source of truth; `scripts/bootstrap.sh` is a
gcloud-only equivalent kept for readability. After `terraform apply`, the
GitHub secrets must match the new outputs — especially
`GCP_WORKLOAD_IDENTITY_PROVIDER`, which changes whenever `wif_pool_id` does.
A stale value fails the deploy job at the auth step while lint/test pass.
