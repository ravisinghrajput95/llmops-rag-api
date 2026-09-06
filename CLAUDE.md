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

## Deploying

Terraform in `terraform/` is the source of truth; `scripts/bootstrap.sh` is a
gcloud-only equivalent kept for readability. After `terraform apply`, the
GitHub secrets must match the new outputs — especially
`GCP_WORKLOAD_IDENTITY_PROVIDER`, which changes whenever `wif_pool_id` does.
A stale value fails the deploy job at the auth step while lint/test pass.
